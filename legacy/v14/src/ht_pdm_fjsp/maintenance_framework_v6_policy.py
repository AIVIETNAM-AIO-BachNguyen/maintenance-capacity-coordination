"""Matched-size shared versus position-specific edge heads for N4/K2 matching."""
from __future__ import annotations

import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

VARIANTS = ("shared_edge", "unshared_edge")
MACHINES = 4
TECHNICIANS = 2
OBSERVATION_DIM = 56
ACTIONS = catalogue(MACHINES, TECHNICIANS)


def action_incidence():
    result = torch.zeros(len(ACTIONS), MACHINES * TECHNICIANS)
    for index, matching in enumerate(ACTIONS):
        for machine, worker in matching:
            result[index, machine * TECHNICIANS + worker] = 1
    return result


class EdgeSharingActor(nn.Module):
    """Factorized matching score with either tied or untied edge MLP weights."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.machine_encoder = nn.Sequential(nn.Linear(8, 24), nn.Tanh())
        # Worker identity/assigned-machine index is excluded from actor.
        self.worker_encoder = nn.Sequential(nn.Linear(2, 12), nn.Tanh())
        if variant == "shared_edge":
            self.edge_network = nn.Sequential(
                nn.Linear(40, 32), nn.Tanh(), nn.Linear(32, 1),
            )
        else:
            # 8 × (40→4→1) = 1,352 head parameters, versus 1,345 tied.
            self.edge_networks = nn.ModuleList(
                nn.Sequential(nn.Linear(40, 4), nn.Tanh(), nn.Linear(4, 1))
                for _ in range(MACHINES * TECHNICIANS)
            )
        self.register_buffer("incidence", action_incidence())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
            raise ValueError(f"expected [batch,{OBSERVATION_DIM}] observation")
        batch = observation.shape[0]
        global_features = observation[:, :2]
        machine = observation[:, 2:34].reshape(batch, MACHINES, 8)
        worker = observation[:, 34:40].reshape(batch, TECHNICIANS, 3)
        edge = observation[:, 40:56].reshape(batch, MACHINES, TECHNICIANS, 2)
        machine_embed = self.machine_encoder(machine)
        worker_embed = self.worker_encoder(worker[..., :2])
        m = machine_embed[:, :, None, :].expand(batch, MACHINES, TECHNICIANS, -1)
        w = worker_embed[:, None, :, :].expand(batch, MACHINES, TECHNICIANS, -1)
        g = global_features[:, None, None, :].expand(batch, MACHINES, TECHNICIANS, -1)
        inputs = torch.cat((m, w, edge, g), dim=-1).reshape(batch, MACHINES * TECHNICIANS, 40)
        if self.variant == "shared_edge":
            scores = self.edge_network(inputs).squeeze(-1)
        else:
            scores = torch.cat(
                [network(inputs[:, index, :]) for index, network in enumerate(self.edge_networks)],
                dim=1,
            )
        return scores @ self.incidence.T


class EdgeSharingPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, variant: str = "shared_edge", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kwargs.setdefault("net_arch", dict(pi=[], vf=[64, 64]))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = EdgeSharingActor(self.variant)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )
