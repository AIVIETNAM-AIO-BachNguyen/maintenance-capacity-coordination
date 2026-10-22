"""Capacity-matched identity control for V14 technician opportunity cost."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

ACTIONS = catalogue()


def action_tensors():
    incidence = torch.zeros((len(ACTIONS), 4, 2), dtype=torch.float32)
    for index, matching in enumerate(ACTIONS):
        for machine, technician in matching:
            incidence[index, machine, technician] = 1.
    served = incidence.sum(dim=2).clamp(max=1)
    return incidence, 1. - served


class OpportunityCostActor(nn.Module):
    """Current-information scoring of legal matchings via global unserved demand."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in ('competition', 'count_calibrated'):
            raise ValueError(mode)
        self.mode = mode
        self.machine_encoder = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.worker_encoder = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
        self.edge_head = nn.Sequential(nn.Linear(52, 32), nn.Tanh(), nn.Linear(32, 1))
        self.demand_head = nn.Sequential(nn.Linear(52, 16), nn.Tanh(), nn.Linear(16, 1))
        self.wait_head = nn.Sequential(nn.Linear(26, 16), nn.Tanh(), nn.Linear(16, 1))
        incidence, unserved = action_tensors()
        self.register_buffer('incidence', incidence)
        self.register_buffer('unserved', unserved)

    def components(self, observation: torch.Tensor):
        if observation.ndim != 2 or observation.shape[1] != 56:
            raise ValueError('expected [batch,56] current observation')
        batch = observation.shape[0]
        clock = observation[:, :2]
        machine = self.machine_encoder(observation[:, 2:34].reshape(batch, 4, 8))
        worker = self.worker_encoder(observation[:, 34:40].reshape(batch, 2, 3))
        edges = observation[:, 40:56].reshape(batch, 4, 2, 2)
        global_machine = machine.mean(dim=1)
        global_worker = worker.mean(dim=1)
        local = torch.cat((
            machine[:, :, None, :].expand(-1, -1, 2, -1),
            worker[:, None, :, :].expand(-1, 4, -1, -1),
            edges,
            global_machine[:, None, None, :].expand(-1, 4, 2, -1),
            global_worker[:, None, None, :].expand(-1, 4, 2, -1),
            clock[:, None, None, :].expand(-1, 4, 2, -1),
        ), dim=-1)
        edge_utility = self.edge_head(local).squeeze(-1)
        demand = F.softplus(self.demand_head(local).squeeze(-1))
        wait = self.wait_head(torch.cat((global_machine, global_worker, clock), dim=-1)).squeeze(-1)
        utility = torch.einsum('bmj,amj->ba', edge_utility, self.incidence)
        duration = edges[..., 1].clamp(min=0)
        if self.mode == 'competition':
            occupancy = torch.einsum('bmj,amj->baj', duration, self.incidence)
            exposed = torch.einsum('bmj,am->baj', demand, self.unserved)
            penalty = (occupancy * exposed).sum(dim=2)
        else:
            # Keep the number, but erase the identity, of other served machines.
            cardinality = (1. - self.unserved).sum(dim=1)
            factor = (4. - cardinality) / 3.
            other_demand = demand.sum(dim=1, keepdim=True) - demand
            penalty = torch.einsum('bmj,amj->ba', duration * other_demand, self.incidence)
            penalty = penalty * factor[None, :]
        logits = utility - penalty
        wait_index = ACTIONS.index(())
        logits[:, wait_index] = wait
        return logits, penalty, demand

    def forward(self, observation: torch.Tensor):
        return self.components(observation)[0]
