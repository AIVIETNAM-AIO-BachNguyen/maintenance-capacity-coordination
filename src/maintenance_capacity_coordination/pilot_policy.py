"""Exact two-stage masked policy and ordinary PPO update for pilot v1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from maintenance_capacity_coordination.coordination_v1 import GLOBAL_DIM, LOCAL_DIM


def _mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))


class PilotPolicy(nn.Module):
    def __init__(self, arm, matching_actions=21):
        super().__init__()
        if arm not in ("B", "C", "D", "E"):
            raise ValueError(arm)
        self.arm = arm
        self.machine = None if arm == "D" else _mlp(LOCAL_DIM + (GLOBAL_DIM if arm == "E" else 0), 1)
        self.coordinator = _mlp(GLOBAL_DIM, matching_actions)
        self.critic = _mlp(GLOBAL_DIM, 1)

    def value(self, pre_obs):
        return self.critic(pre_obs).squeeze(-1)

    def distributions(self, machine_obs, eligible, coordinator_obs, matching_mask):
        request_dist = None
        if self.machine is not None:
            logits = self.machine(machine_obs).squeeze(-1)
            request_dist = Bernoulli(logits=logits)
        coordinator_logits = self.coordinator(coordinator_obs)
        coordinator_logits = coordinator_logits.masked_fill(~matching_mask, -1e9)
        return request_dist, Categorical(logits=coordinator_logits)

    def sample_requests(self, machine_obs, eligible, deterministic=False):
        if self.arm == "D":
            return eligible.to(torch.float32)
        logits = self.machine(machine_obs).squeeze(-1)
        bits = (logits >= 0).float() if deterministic else Bernoulli(logits=logits).sample()
        return bits * eligible.float()

    def sample_matching(self, coordinator_obs, matching_mask, deterministic=False):
        logits = self.coordinator(coordinator_obs).masked_fill(~matching_mask, -1e9)
        return logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()

    def logprob_entropy(self, machine_obs, eligible, request_bits, coordinator_obs,
                        matching_mask, matching_index):
        request_dist, matching_dist = self.distributions(machine_obs, eligible, coordinator_obs, matching_mask)
        logprob = matching_dist.log_prob(matching_index)
        entropy = matching_dist.entropy()
        if request_dist is not None:
            logprob = logprob + (request_dist.log_prob(request_bits) * eligible.float()).sum(dim=-1)
            entropy = entropy + (request_dist.entropy() * eligible.float()).sum(dim=-1)
        return logprob, entropy

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class Transition:
    pre_obs: np.ndarray
    machine_obs: np.ndarray
    eligible: np.ndarray
    request_bits: np.ndarray
    coordinator_obs: np.ndarray
    matching_mask: np.ndarray
    matching_index: int
    old_logprob: float
    value: float
    reward: float
    done: bool


def ppo_update(policy, optimizer, transitions, bootstrap, *, epochs=4, minibatch=64,
               gamma=.99, gae_lambda=.95, clip=.2, entropy_coef=.01):
    """The numerator and denominator include both sampled action stages."""
    count = len(transitions)
    advantages = np.zeros(count, dtype=np.float32)
    next_value = float(bootstrap)
    advantage = 0.
    for i in range(count - 1, -1, -1):
        item = transitions[i]
        continuation = 0. if item.done else 1.
        delta = item.reward + gamma * next_value * continuation - item.value
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[i] = advantage
        next_value = item.value
    returns = advantages + np.asarray([item.value for item in transitions], dtype=np.float32)

    def batch_array(field, dtype):
        return torch.as_tensor(np.stack([getattr(item, field) for item in transitions]), dtype=dtype)

    pre_obs = batch_array("pre_obs", torch.float32)
    if policy.arm == "D":
        machine_obs = None
    else:
        machine_obs = batch_array("machine_obs", torch.float32)
    eligible = batch_array("eligible", torch.bool)
    request_bits = batch_array("request_bits", torch.float32)
    coordinator_obs = batch_array("coordinator_obs", torch.float32)
    matching_mask = batch_array("matching_mask", torch.bool)
    actions = torch.as_tensor([item.matching_index for item in transitions], dtype=torch.long)
    old_logprobs = torch.as_tensor([item.old_logprob for item in transitions], dtype=torch.float32)
    advantage_tensor = torch.as_tensor(advantages)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
    return_tensor = torch.as_tensor(returns)
    rng = np.random.default_rng(int(torch.randint(0, 2**31, ()).item()))
    losses = []
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(count), max(1, int(np.ceil(count / minibatch)))):
            idx = torch.as_tensor(indices, dtype=torch.long)
            logprobs, entropy = policy.logprob_entropy(
                machine_obs[idx] if machine_obs is not None else None,
                eligible[idx], request_bits[idx], coordinator_obs[idx],
                matching_mask[idx], actions[idx],
            )
            ratio = torch.exp(logprobs - old_logprobs[idx])
            surrogate = torch.minimum(ratio * advantage_tensor[idx],
                                      ratio.clamp(1 - clip, 1 + clip) * advantage_tensor[idx])
            value = policy.value(pre_obs[idx])
            loss = -surrogate.mean() + .5 * (value - return_tensor[idx]).square().mean() - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {"ppo_loss_mean": float(np.mean(losses)), "return_mean": float(np.mean(returns)),
            "advantage_mean": float(np.mean(advantages))}
