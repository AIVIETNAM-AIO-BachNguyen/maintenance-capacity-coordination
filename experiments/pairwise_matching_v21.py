"""Reliability-weighted pairwise matching supervision on immutable v20 labels."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss, subset, tensorize as base_tensorize
from experiments.matching_value_rehearsal_v19 import LabelCycle
from maintenance_capacity_coordination.same_tick_policy_v15 import ppo_update as original_ppo_update


def tensorize(records):
    data = base_tensorize(records)
    n, width = data['candidate_valid'].shape
    targets = np.zeros((n, width, width), dtype=np.float32)
    weights = np.zeros((n, width, width), dtype=np.float32)
    valid = np.zeros((n, width, width), dtype=bool)
    scale = np.zeros(n, dtype=np.float32)
    for row_index, row in enumerate(records):
        count = len(row['matching_indices'])
        if count < 2 or len(row['branches']) != count:
            raise ValueError('malformed pairwise candidate row')
        tape_costs = []
        for index, branch in enumerate(row['branches']):
            if branch['matching_index'] != row['matching_indices'][index]:
                raise ValueError('branch/candidate order mismatch')
            h = np.asarray(branch['H_costs'], dtype=np.float64)
            tail = np.asarray(branch['tail_costs'], dtype=np.float64)
            if len(h) != 16 or len(tail) != 16 or not np.isfinite(h + tail).all():
                raise ValueError('pairwise targets need 16 finite paired tapes')
            tape_costs.append(h + tail)
        tape_costs = np.stack(tape_costs)
        for i in range(count):
            for j in range(i + 1, count):
                difference = tape_costs[i] - tape_costs[j]
                delta = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(16))
                targets[row_index, i, j] = 1 / (1 + math.exp(delta / 20))
                weights[row_index, i, j] = abs(delta) / (abs(delta) + 2 * se + 1e-8)
                valid[row_index, i, j] = True
        if weights[row_index].sum() == 0:
            weights[row_index][valid[row_index]] = 1
        scale[row_index] = math.log(count) / math.log(2)
    data.update(pair_targets=torch.from_numpy(targets), pair_weights=torch.from_numpy(weights),
                pair_valid=torch.from_numpy(valid), pair_scale=torch.from_numpy(scale))
    return data


def pairwise_loss_from_scores(scores, data):
    row, i, j = data['pair_valid'].nonzero(as_tuple=True)
    logit = scores[row, i] - scores[row, j]
    losses = F.binary_cross_entropy_with_logits(logit, data['pair_targets'][row, i, j], reduction='none')
    weighted = torch.zeros(scores.shape[0], dtype=scores.dtype, device=scores.device)
    denominator = torch.zeros_like(weighted)
    weights = data['pair_weights'][row, i, j]
    weighted.index_add_(0, row, losses * weights)
    denominator.index_add_(0, row, weights)
    if not torch.all(denominator > 0):
        raise ValueError('pairwise row has no positive weight')
    return ((weighted / denominator) * data['pair_scale']).mean()


def pairwise_loss(policy, data):
    return pairwise_loss_from_scores(candidate_logits(policy, data), data)


def pretrain(policy, records, seed, *, objective='pairwise', epochs=96, batch_size=32, on_epoch=None):
    if objective not in ('pairwise', 'listwise'):
        raise ValueError(objective)
    data = tensorize(records)
    loss_fn = pairwise_loss if objective == 'pairwise' else listwise_loss
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    n = len(records)
    updates = 0
    for epoch in range(1, epochs + 1):
        weighted = 0.
        for indices in np.array_split(rng.permutation(n), max(1, int(np.ceil(n / batch_size)))):
            batch = subset(data, torch.as_tensor(indices, dtype=torch.long))
            loss = loss_fn(policy, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite v21 matching pretraining loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
            optimizer.step()
            weighted += float(loss.detach()) * len(indices)
            updates += 1
        if on_epoch is not None:
            on_epoch({'epoch': epoch, 'mean_supervised_loss': weighted / n,
                      'cumulative_updates': updates, 'objective': objective})
    return {'pretraining_records': n, 'pretraining_epochs': epochs,
            'pretraining_updates': updates, 'supervised_objective': objective}


def ppo_update(policy, optimizer, transitions, bootstrap, labels, cycle: LabelCycle,
               *, weight: float = .5, label_batch_size: int = 32,
               epochs: int = 4, minibatch: int = 64, gamma: float = .99,
               gae_lambda: float = .95, clip: float = .2, entropy_coef: float = .01,
               on_minibatch=None, objective='pairwise'):
    """Original canonical-sequence PPO plus one fixed supervised actor term.

    At weight zero, dispatch to the unchanged original implementation. A
    separate NumPy generator supplies label indices and never consumes Torch
    sampling randomness.
    """
    if objective not in ('pairwise', 'listwise'):
        raise ValueError(objective)
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
    loss_fn = pairwise_loss if objective == 'pairwise' else listwise_loss
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
            matching_loss = loss_fn(policy, label_batch)
            # Preserve the small rehearsal scalar beside a large critic loss;
            # casting the sum does not change either component's gradient.
            loss = base_loss.double() + weight * matching_loss.double()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite v21 PPO/rehearsal loss')
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
