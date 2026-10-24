"""Single-change PPO update: rehearse existing matching-value labels in actor loss."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiments.full_matching_pretrain_v17 import listwise_loss, subset
from maintenance_capacity_coordination.same_tick_policy_v15 import ppo_update as original_ppo_update


class LabelCycle:
    """Shuffle without replacement, cycling through a case's fixed training rows."""

    def __init__(self, count: int, seed: int):
        if count <= 0:
            raise ValueError('empty label bank')
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.position = 0

    def take(self, size: int) -> np.ndarray:
        if size <= 0:
            raise ValueError('empty label minibatch')
        pieces = []
        remaining = size
        while remaining:
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.count)
                self.position = 0
            take = min(remaining, len(self.order) - self.position)
            pieces.append(self.order[self.position:self.position + take])
            self.position += take
            remaining -= take
        return np.concatenate(pieces)


def ppo_update(policy, optimizer, transitions, bootstrap, labels, cycle: LabelCycle,
               *, weight: float = .5, label_batch_size: int = 32,
               epochs: int = 4, minibatch: int = 64, gamma: float = .99,
               gae_lambda: float = .95, clip: float = .2, entropy_coef: float = .01,
               on_minibatch=None):
    """Original canonical-sequence PPO plus a fixed listwise actor term.

    At weight zero, dispatch to the unchanged original implementation. A
    separate NumPy generator supplies label indices and never consumes Torch
    sampling randomness.
    """
    if weight < 0 or not np.isfinite(weight):
        raise ValueError('invalid rehearsal weight')
    if weight == 0:
        return original_ppo_update(policy, optimizer, transitions, bootstrap,
                                   epochs=epochs, minibatch=minibatch, gamma=gamma,
                                   gae_lambda=gae_lambda, clip=clip, entropy_coef=entropy_coef)
    if len(transitions) == 0:
        raise ValueError('empty PPO rollout')
    if labels['targets'].shape[0] != cycle.count:
        raise ValueError('label cycle and tensor bank disagree')
    count = len(transitions)
    advantages = np.zeros(count, dtype=np.float32)
    next_value, advantage = float(bootstrap), 0.
    for i in range(count - 1, -1, -1):
        item = transitions[i]
        continuation = 0. if item.done else 1.
        delta = item.reward + gamma * next_value * continuation - item.value
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[i] = advantage
        next_value = item.value
    returns = advantages + np.asarray([t.value for t in transitions], dtype=np.float32)
    pre = torch.as_tensor(np.stack([t.pre_obs for t in transitions]), dtype=torch.float32)
    machine = torch.as_tensor(np.stack([t.machine_obs for t in transitions]), dtype=torch.float32)
    worker = torch.as_tensor(np.stack([t.worker_obs for t in transitions]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([t.edge_mask for t in transitions]), dtype=torch.bool)
    sequences = torch.as_tensor(np.stack([t.sequence for t in transitions]), dtype=torch.long)
    old_logprobs = torch.as_tensor([t.old_logprob for t in transitions], dtype=torch.float32)
    standardized = torch.as_tensor((advantages - advantages.mean()) /
                                   (advantages.std() + 1e-8), dtype=torch.float32)
    returns_t = torch.as_tensor(returns)
    rng = np.random.default_rng(int(torch.randint(0, 2**31, ()).item()))
    losses, ppo_losses, rehearsal_losses, ratio_deviations = [], [], [], []
    label_candidate_paths = 0
    label_token_positions = 0
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(count), max(1, int(np.ceil(count / minibatch)))):
            idx = torch.as_tensor(indices, dtype=torch.long)
            logprob, entropy = policy.logprob_entropy(machine[idx], worker[idx], mask[idx], sequences[idx])
            ratio = torch.exp(logprob - old_logprobs[idx])
            surrogate = torch.minimum(ratio * standardized[idx],
                                      ratio.clamp(1 - clip, 1 + clip) * standardized[idx])
            value = policy.value(pre[idx])
            base_loss = (-surrogate.mean() + .5 * (value - returns_t[idx]).square().mean()
                         - entropy_coef * entropy.mean())
            label_ids = cycle.take(label_batch_size)
            label_batch = subset(labels, torch.as_tensor(label_ids, dtype=torch.long))
            matching_loss = listwise_loss(policy, label_batch)
            # Preserve the small rehearsal scalar beside a large critic loss;
            # casting the sum does not change either component's gradient.
            loss = base_loss.double() + weight * matching_loss.double()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite v19 PPO/rehearsal loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            ppo_losses.append(float(base_loss.detach()))
            rehearsal_losses.append(float(matching_loss.detach()))
            ratio_deviations.append(float((ratio.detach() - 1).abs().mean()))
            label_candidate_paths += int(label_batch['candidate_valid'].sum())
            label_token_positions += int(((label_batch['sequences'] >= 0) &
                                          label_batch['candidate_valid'][..., None]).sum())
            if on_minibatch is not None:
                on_minibatch({'label_row_indices': label_ids.tolist(),
                              'ppo_loss': float(base_loss.detach()),
                              'rehearsal_loss': float(matching_loss.detach()),
                              'combined_loss': float(loss.detach())})
    return {'ppo_loss_mean': float(np.mean(ppo_losses)),
            'combined_loss_mean': float(np.mean(losses)),
            'rehearsal_loss_mean': float(np.mean(rehearsal_losses)),
            'sequence_ratio_abs_deviation_mean': float(np.mean(ratio_deviations)),
            'return_mean': float(np.mean(returns)),
            'advantage_mean': float(np.mean(advantages)),
            'rehearsal_minibatches': len(rehearsal_losses),
            'rehearsal_candidate_paths': label_candidate_paths,
            'rehearsal_token_positions': label_token_positions}
