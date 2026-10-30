"""One-pass dynamic bipartite encoder for the unchanged canonical matching policy.

Compatibility messages include busy technicians and unrequested machines. Only the
v15 physical mask determines what may be assigned now; graph updates are planning
computations inside a tick and never advance the simulator.
"""
from __future__ import annotations

import torch
from torch import nn

from maintenance_capacity_coordination.same_tick_policy_v15 import (
    ContextActor, EDGES, N_MACHINE, N_WORKER, STOP, SequencePolicy,
)


class DynamicBipartiteActor(ContextActor):
    """Graph mode passes compatible-edge messages; input mode pools all entities."""

    def __init__(self, mode: str):
        if mode not in ("graph", "input"):
            raise ValueError(mode)
        super().__init__(direct=False, canonical=True)
        self.mode = mode
        self.context = nn.Sequential(nn.Linear(129, 48), nn.Tanh(),
                                     nn.Linear(48, 32), nn.Tanh())
        if mode == "graph":
            # Machine -> technician -> machine is one bipartite pass. Sum, rather
            # than normalized attention, retains competing-demand multiplicity.
            self.to_worker = nn.Sequential(nn.Linear(38, 32), nn.Tanh(),
                                           nn.Linear(32, 16), nn.Tanh())
            self.update_worker = nn.Sequential(nn.Linear(34, 32), nn.Tanh(),
                                               nn.Linear(32, 16), nn.Tanh())
            self.to_machine = nn.Sequential(nn.Linear(37, 32), nn.Tanh(),
                                            nn.Linear(32, 16), nn.Tanh())
            self.update_machine = nn.Sequential(nn.Linear(33, 32), nn.Tanh(),
                                                nn.Linear(32, 16), nn.Tanh())

    def encode(self, machine: torch.Tensor, worker: torch.Tensor):
        # The old actor returns fixed edge encodings here. Retain the full raw
        # present-state tensors so graph mode can refresh after each token.
        return machine, worker

    def _node_edge_embeddings(self, machine, worker, initial_mask, used_m, used_w, selected):
        batch = machine.shape[0]
        base_m = self.machine_encoder(machine)
        base_w = self.worker_encoder(worker)
        duration = machine[:, :, 14:14 + N_WORKER]
        restoration = machine[:, :, 14 + N_WORKER:14 + 2 * N_WORKER]
        compatible = duration > 0
        if self.mode == "graph":
            selected_matrix = selected.reshape(batch, N_MACHINE, N_WORKER).to(duration.dtype)
            provisional_eta = (selected_matrix * duration).sum(dim=1)
            eta = torch.where(used_w, provisional_eta, worker[:, :, 1])
            mb = base_m[:, :, None, :].expand(-1, -1, N_WORKER, -1)
            wb = base_w[:, None, :, :].expand(-1, N_MACHINE, -1, -1)
            sm = used_m[:, :, None, None].expand(-1, -1, N_WORKER, -1).to(duration.dtype)
            sw = used_w[:, None, :, None].expand(-1, N_MACHINE, -1, -1).to(duration.dtype)
            eb = eta[:, None, :, None].expand(-1, N_MACHINE, -1, -1)
            ready = initial_mask.reshape(batch, N_MACHINE, N_WORKER)[..., None].to(duration.dtype)
            attributes = torch.stack((duration, restoration), -1)
            to_worker = self.to_worker(torch.cat((mb, wb, attributes, sm, sw, eb, ready), -1))
            incoming_w = (to_worker * compatible[..., None]).sum(dim=1)
            dynamic_w = self.update_worker(torch.cat(
                (base_w, incoming_w, used_w[..., None].to(duration.dtype), eta[..., None]), -1))
            dw = dynamic_w[:, None, :, :].expand(-1, N_MACHINE, -1, -1)
            to_machine = self.to_machine(torch.cat((mb, dw, attributes, sm, eb, ready), -1))
            incoming_m = (to_machine * compatible[..., None]).sum(dim=2)
            dynamic_m = self.update_machine(torch.cat(
                (base_m, incoming_m, used_m[..., None].to(duration.dtype)), -1))
        else:
            dynamic_m, dynamic_w = base_m, base_w
        m, w = self.machine_id, self.worker_id
        edge_input = torch.cat((dynamic_m[:, m], dynamic_w[:, w],
                                duration[:, m, w, None], restoration[:, m, w, None]), -1)
        edges = self.edge_encoder(edge_input)
        all_entity = torch.cat((dynamic_m.mean(dim=1), dynamic_w.mean(dim=1)), -1)
        return edges, all_entity

    def logits(self, encoded, initial_mask, used_m, used_w, selected, *, frozen_context=None):
        machine, worker = encoded
        batch = machine.shape[0]
        feasible = initial_mask & ~used_m[:, self.machine_id] & ~used_w[:, self.worker_id]
        index = torch.arange(STOP, device=feasible.device)
        last = torch.where(selected, index[None, :], -1).amax(1)
        feasible = feasible & (index[None, :] > last[:, None])
        if frozen_context is None:
            edges, all_entity = self._node_edge_embeddings(
                machine, worker, initial_mask, used_m, used_w, selected)
            selected_sum = (edges * selected.to(edges.dtype)[..., None]).sum(1)
            cur_mean, cur_max, cur_count, after_mean, after_max, after_count = (
                self.residual_summaries(edges, feasible))
            candidate_input = torch.cat((selected_sum[:, None, :].expand(batch, STOP, -1),
                                         after_mean, after_max, after_count[..., None],
                                         all_entity[:, None, :].expand(batch, STOP, -1)), -1)
            stop_input = torch.cat((selected_sum, cur_mean, cur_max, cur_count[:, None],
                                    all_entity), -1)
            candidate_context = self.context(candidate_input)
            stop_context = self.context(stop_input)
        else:
            edges, candidate_context, stop_context = frozen_context
        edge_logits = self.edge_score(torch.cat((edges, candidate_context), -1)).squeeze(-1)
        stop_logit = self.stop_score(stop_context)
        logits = torch.cat((edge_logits, stop_logit), 1)
        mask = torch.cat((feasible, torch.ones((batch, 1), dtype=torch.bool,
                                                device=feasible.device)), 1)
        return logits.masked_fill(~mask, -1e9), feasible, (edges, candidate_context, stop_context)


class DynamicSequencePolicy(SequencePolicy):
    """Reuse the verified canonical sampler/PPO; D16 uses its frozen-root path."""

    def __init__(self, arm: str):
        if arm not in ("C16", "D16", "I16", "P16"):
            raise ValueError(arm)
        super().__init__("D15" if arm == "D16" else "C15")
        self.v16_arm = arm
        self.actor = DynamicBipartiteActor("input" if arm == "I16" else "graph")
