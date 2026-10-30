"""Present-state matching score actors for the prelocked v6 interaction pilot."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue
from maintenance_capacity_coordination.coordination_v1 import LOCAL_DIM
from maintenance_capacity_coordination.structured_policy_v3 import WORKER_DIM

ACTIONS = catalogue(4, 2)
ARMS = ("J6", "A6", "G6")
TARGET_SCALE = 20.0


class EdgeScoreActor(nn.Module):
    """J6 has an explicit two-edge interaction; A6 is strictly additive."""

    def __init__(self, joint: bool):
        super().__init__()
        width = 32 if joint else 90
        self.joint = joint
        self.machine = nn.Sequential(nn.Linear(LOCAL_DIM, 32), nn.Tanh(),
                                     nn.Linear(32, 16), nn.Tanh())
        self.worker = nn.Sequential(nn.Linear(WORKER_DIM, 16), nn.Tanh(),
                                    nn.Linear(16, 16), nn.Tanh())
        self.edge_hidden = nn.Sequential(nn.Linear(34, width), nn.Tanh())
        self.edge_output = nn.Linear(width, 1)
        if joint:
            self.pair = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        machine_idx = np.zeros((len(ACTIONS), 2), dtype=np.int64)
        worker_idx = np.zeros((len(ACTIONS), 2), dtype=np.int64)
        active = np.zeros((len(ACTIONS), 2), dtype=bool)
        for ai, action in enumerate(ACTIONS):
            for slot, (m, j) in enumerate(action):
                machine_idx[ai, slot], worker_idx[ai, slot], active[ai, slot] = m, j, True
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
        duration = machine[:, self.machine_idx, 14 + self.worker_idx]
        restoration = machine[:, self.machine_idx, 16 + self.worker_idx]
        edge_input = torch.cat((m_edge, w_edge, duration.unsqueeze(-1),
                                restoration.unsqueeze(-1)), dim=-1)
        hidden = self.edge_hidden(edge_input)
        score = (self.edge_output(hidden).squeeze(-1)
                 * self.active[None, :, :]).sum(dim=-1)
        if self.joint:
            pair_score = self.pair(torch.cat((hidden[:, :, 0, :], hidden[:, :, 1, :]),
                                             dim=-1)).squeeze(-1)
            score = score + pair_score * self.active[None, :, :].all(dim=-1)
        return score.squeeze(0) if single else score


class FlatScoreActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * LOCAL_DIM + 2 * WORKER_DIM, 33), nn.Tanh(),
                                 nn.Linear(33, 33), nn.Tanh(),
                                 nn.Linear(33, len(ACTIONS)))

    def forward(self, machine, worker):
        single = machine.ndim == 2
        if single:
            machine, worker = machine.unsqueeze(0), worker.unsqueeze(0)
        flat = torch.cat((machine.flatten(start_dim=1), worker.flatten(start_dim=1)), dim=-1)
        raw = self.net(flat)
        score = raw - raw[:, :1]
        return score.squeeze(0) if single else score


def actor_for(arm):
    if arm == "J6":
        return EdgeScoreActor(joint=True)
    if arm == "A6":
        return EdgeScoreActor(joint=False)
    if arm == "G6":
        return FlatScoreActor()
    raise ValueError(arm)


def parameter_count(actor):
    return sum(p.numel() for p in actor.parameters() if p.requires_grad)


def teacher_targets(root_scores, mask):
    """F2 forecast benefit relative to WAIT, divided by fixed scale."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(ACTIONS),) or not mask[0]:
        raise ValueError("invalid legal mask")
    target = np.zeros(len(ACTIONS), dtype=np.float32)
    if int(mask.sum()) == 1:
        if root_scores:
            raise ValueError("one-action F2 must skip forecast")
        return target
    if len(root_scores) != int(mask.sum()):
        raise ValueError("incomplete F2 root scores")
    observed = set()
    cost_by_index = {}
    for row in root_scores:
        action = tuple(tuple(edge) for edge in row["matching"])
        index = ACTIONS.index(action)
        if index in observed or not mask[index]:
            raise ValueError("duplicate or illegal F2 root")
        observed.add(index)
        cost_by_index[index] = float(row["expected_cost"])
    if observed != set(np.flatnonzero(mask)):
        raise ValueError("F2 root coverage mismatch")
    wait = cost_by_index[0]
    for index in observed:
        target[index] = (wait - cost_by_index[index]) / TARGET_SCALE
    return target


def masked_score_mse(predicted, target, mask):
    mask = mask.to(dtype=torch.bool)
    per_state = ((predicted - target).square() * mask).sum(dim=-1) / mask.sum(dim=-1)
    return per_state.mean()


def pair_interaction_errors(predicted, root_scores):
    """Absolute finite-forecast cost interaction errors for all legal pairs."""
    values = {tuple(tuple(edge) for edge in r["matching"]): float(r["expected_cost"])
              for r in root_scores}
    if not values:
        return []
    errors = []
    for action, cost in values.items():
        if len(action) != 2:
            continue
        single_a, single_b = (action[0],), (action[1],)
        true = cost - values[single_a] - values[single_b] + values[()]
        pair_idx = ACTIONS.index(action)
        a_idx = ACTIONS.index(single_a)
        b_idx = ACTIONS.index(single_b)
        estimated = -TARGET_SCALE * (float(predicted[pair_idx]) - float(predicted[a_idx])
                                     - float(predicted[b_idx]) + float(predicted[0]))
        errors.append(abs(estimated - true))
    return errors
