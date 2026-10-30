"""Structured machine-demand/technician-commitment actor and matched PPO controls.

The actor's only stochastic action is a masked complete matching. Machine
embeddings are learned features, not calibrated economic values or actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue
from maintenance_capacity_coordination.coordination_v1 import GLOBAL_DIM, LOCAL_DIM

ACTIONS = catalogue(4, 2)
WORKER_DIM = 4
DIRECT_DIM = 4 * LOCAL_DIM + 2 * WORKER_DIM


def worker_features(config, state):
    p = state.physical
    rows = []
    for j in range(config.technicians):
        assigned = p.assigned[j]
        rows.append([float(p.remaining[j] == 0), p.remaining[j] / 12,
                     (assigned + 1) / (config.machines + 1),
                     float(assigned >= 0 and p.failed[assigned])])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (2, WORKER_DIM):
        raise RuntimeError("v3 worker shape changed")
    return result


def actor_inputs(config, admitted, tick):
    from maintenance_capacity_coordination.coordination_v1 import requester_features
    machine = requester_features(config, admitted, tick, "C")
    worker = worker_features(config, admitted)
    if machine.shape != (4, LOCAL_DIM):
        raise RuntimeError("v3 machine shape changed")
    return machine, worker


class StructuredActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.machine = nn.Sequential(nn.Linear(LOCAL_DIM, 32), nn.Tanh(),
                                     nn.Linear(32, 16), nn.Tanh())
        self.worker = nn.Sequential(nn.Linear(WORKER_DIM, 16), nn.Tanh(),
                                    nn.Linear(16, 16), nn.Tanh())
        self.edge = nn.Sequential(nn.Linear(34, 72), nn.Tanh(),
                                  nn.Linear(72, 32), nn.Tanh())
        self.action_score = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        machine_idx = np.zeros((len(ACTIONS), 2), dtype=np.int64)
        worker_idx = np.zeros((len(ACTIONS), 2), dtype=np.int64)
        active = np.zeros((len(ACTIONS), 2), dtype=bool)
        for a, action in enumerate(ACTIONS):
            for slot, (m, j) in enumerate(action):
                machine_idx[a, slot], worker_idx[a, slot], active[a, slot] = m, j, True
        self.register_buffer("machine_idx", torch.as_tensor(machine_idx), persistent=False)
        self.register_buffer("worker_idx", torch.as_tensor(worker_idx), persistent=False)
        self.register_buffer("active", torch.as_tensor(active), persistent=False)

    def forward(self, machine, worker):
        single = machine.ndim == 2
        if single:
            machine, worker = machine.unsqueeze(0), worker.unsqueeze(0)
        m = self.machine(machine)
        w = self.worker(worker)
        m_edge = m[:, self.machine_idx]
        w_edge = w[:, self.worker_idx]
        service = machine[:, self.machine_idx, 14 + self.worker_idx]
        restoration = machine[:, self.machine_idx, 16 + self.worker_idx]
        edge = self.edge(torch.cat((m_edge, w_edge, service.unsqueeze(-1),
                                    restoration.unsqueeze(-1)), dim=-1))
        selected = (edge * self.active[None, :, :, None]).sum(dim=2)
        pooled_m = m.mean(dim=1)[:, None, :].expand(-1, len(ACTIONS), -1)
        pooled_w = w.mean(dim=1)[:, None, :].expand(-1, len(ACTIONS), -1)
        logits = self.action_score(torch.cat((selected, pooled_m, pooled_w), dim=-1)).squeeze(-1)
        return logits.squeeze(0) if single else logits


class DirectActor(nn.Module):
    def __init__(self, width=52):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIRECT_DIM, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, len(ACTIONS)))

    def forward(self, machine, worker):
        single = machine.ndim == 2
        if single:
            machine, worker = machine.unsqueeze(0), worker.unsqueeze(0)
        flat = torch.cat((machine.flatten(start_dim=1), worker.flatten(start_dim=1)), dim=-1)
        logits = self.net(flat)
        return logits.squeeze(0) if single else logits


class MatchingPolicy(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ("S", "R", "G3", "D3"):
            raise ValueError(arm)
        self.arm = arm
        self.actor = StructuredActor() if arm in ("S", "R") else DirectActor()
        self.critic = nn.Sequential(nn.Linear(GLOBAL_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def actor_view(self, machine, worker):
        if self.arm != "R":
            return machine, worker
        machine, worker = machine.clone(), worker.clone()
        machine[..., 7] = 0
        machine[..., 20] = 0
        worker[..., 1] = 0
        return machine, worker

    def logits(self, machine, worker, mask):
        machine, worker = self.actor_view(machine, worker)
        return self.actor(machine, worker).masked_fill(~mask, -1e9)

    def distribution(self, machine, worker, mask):
        return Categorical(logits=self.logits(machine, worker, mask))

    def sample(self, machine, worker, mask, deterministic=False):
        logits = self.logits(machine, worker, mask)
        return logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()

    def logprob_entropy(self, machine, worker, mask, action):
        dist = self.distribution(machine, worker, mask)
        return dist.log_prob(action), dist.entropy()

    def value(self, pre_obs):
        return self.critic(pre_obs).squeeze(-1)

    @property
    def actor_parameters(self):
        return sum(p.numel() for p in self.actor.parameters())

    @property
    def critic_parameters(self):
        return sum(p.numel() for p in self.critic.parameters())


@dataclass
class Transition:
    pre_obs: np.ndarray
    machine_obs: np.ndarray
    worker_obs: np.ndarray
    matching_mask: np.ndarray
    matching_index: int
    old_logprob: float
    value: float
    reward: float
    done: bool


def ppo_update(policy, optimizer, transitions, bootstrap, *, epochs=4, minibatch=64,
               gamma=.99, gae_lambda=.95, clip=.2, entropy_coef=.01):
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
    returns = advantages + np.asarray([t.value for t in transitions], dtype=np.float32)
    pre = torch.as_tensor(np.stack([t.pre_obs for t in transitions]), dtype=torch.float32)
    machine = torch.as_tensor(np.stack([t.machine_obs for t in transitions]), dtype=torch.float32)
    worker = torch.as_tensor(np.stack([t.worker_obs for t in transitions]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([t.matching_mask for t in transitions]), dtype=torch.bool)
    actions = torch.as_tensor([t.matching_index for t in transitions], dtype=torch.long)
    old_logprobs = torch.as_tensor([t.old_logprob for t in transitions], dtype=torch.float32)
    advantage_tensor = torch.as_tensor(advantages)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
    return_tensor = torch.as_tensor(returns)
    rng = np.random.default_rng(int(torch.randint(0, 2**31, ()).item()))
    losses = []
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(count), max(1, int(np.ceil(count / minibatch)))):
            idx = torch.as_tensor(indices, dtype=torch.long)
            logprobs, entropy = policy.logprob_entropy(machine[idx], worker[idx], mask[idx], actions[idx])
            ratio = torch.exp(logprobs - old_logprobs[idx])
            surrogate = torch.minimum(ratio * advantage_tensor[idx],
                                      ratio.clamp(1 - clip, 1 + clip) * advantage_tensor[idx])
            value = policy.value(pre[idx])
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
