"""N5/K3 structured and flat full-action PPO controls on present-state inputs."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue
from maintenance_capacity_coordination.scaled_coordination_v11 import actor_inputs_scaled, global_dim, local_dim
from maintenance_capacity_coordination.structured_policy_v3 import Transition, ppo_update

N_MACHINE, N_WORKER = 5, 3
LOCAL_DIM, GLOBAL_DIM, WORKER_DIM = local_dim(N_WORKER), global_dim(N_MACHINE, N_WORKER), 4
ACTIONS = catalogue(N_MACHINE, N_WORKER)
DIRECT_DIM = N_MACHINE * LOCAL_DIM + N_WORKER * WORKER_DIM

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
        machine_idx = np.zeros((len(ACTIONS), N_WORKER), dtype=np.int64)
        worker_idx = np.zeros((len(ACTIONS), N_WORKER), dtype=np.int64)
        active = np.zeros((len(ACTIONS), N_WORKER), dtype=bool)
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
        restoration = machine[:, self.machine_idx, 14 + N_WORKER + self.worker_idx]
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
        if arm not in ("S11", "G11", "Q11"):
            raise ValueError(arm)
        self.arm = arm
        self.actor = StructuredActor() if arm == "S11" else DirectActor()
        self.critic = nn.Sequential(nn.Linear(GLOBAL_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def logits(self, machine, worker, mask):
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



def choose(policy, env, deterministic=False):
    """One masked complete-matching action; Q11 uses always-request admission."""
    pre = env.pre_observation()
    submissions = env.fixed_submissions("D" if policy.arm == "Q11" else "threshold")
    admitted = env.admitted(submissions)
    machine, worker = actor_inputs_scaled(env.config, admitted, env.tick)
    mask = env.matching_mask(submissions)
    if not mask.any() or not mask[0]:
        raise RuntimeError("WAIT must remain legal")
    with torch.no_grad():
        m, w = torch.from_numpy(machine), torch.from_numpy(worker)
        mask_t = torch.from_numpy(mask)
        index = policy.sample(m, w, mask_t, deterministic)
        logprob, _ = policy.logprob_entropy(m, w, mask_t, index)
        value = policy.value(torch.from_numpy(pre))
    return {"pre_obs": pre, "machine_obs": machine, "worker_obs": worker,
            "matching_mask": mask, "matching_index": int(index),
            "old_logprob": float(logprob), "value": float(value),
            "submissions": submissions}
