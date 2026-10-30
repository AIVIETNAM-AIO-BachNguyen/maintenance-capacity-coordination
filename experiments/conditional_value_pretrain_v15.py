"""Listwise auxiliary training on simulator-only conditional second-edge costs."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, STOP, SequencePolicy

PRETRAIN_ARMS = ("C15", "D15", "B15", "S15", "G15")
EPOCHS = 64
BATCH = 64


def tensorize(records):
    if not records:
        raise ValueError("empty conditional value records")
    width = max(len(r["second_edges"]) for r in records)
    n = len(records)
    seconds = np.zeros((n, width), dtype=np.int64)
    matching = np.zeros((n, width), dtype=np.int64)
    targets = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    for i, row in enumerate(records):
        size = len(row["second_edges"])
        if size < 2 or len(row["matching_indices"]) != size or len(row["target_probabilities"]) != size:
            raise ValueError("malformed conditional label row")
        seconds[i, :size] = row["second_edges"]
        matching[i, :size] = row["matching_indices"]
        targets[i, :size] = row["target_probabilities"]
        valid[i, :size] = True
        if not np.isclose(targets[i].sum(), 1., atol=1e-6):
            raise ValueError("unnormalized target")
    return {"machine": torch.as_tensor(np.asarray([r["machine_obs"] for r in records]), dtype=torch.float32),
            "worker": torch.as_tensor(np.asarray([r["worker_obs"] for r in records]), dtype=torch.float32),
            "edge_mask": torch.as_tensor(np.asarray([r["edge_mask"] for r in records]), dtype=torch.bool),
            "first": torch.as_tensor([r["first_edge"] for r in records], dtype=torch.long),
            "seconds": torch.from_numpy(seconds), "matching": torch.from_numpy(matching),
            "targets": torch.from_numpy(targets), "candidate_valid": torch.from_numpy(valid)}


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def candidate_logits(policy, data):
    machine, worker, valid = data["machine"], data["worker"], data["candidate_valid"]
    if isinstance(policy, SequencePolicy):
        actor = policy.actor
        encoded = actor.encode(machine, worker)
        count = machine.shape[0]
        used_m = torch.zeros((count, 5), dtype=torch.bool)
        used_w = torch.zeros((count, 3), dtype=torch.bool)
        selected = torch.zeros((count, STOP), dtype=torch.bool)
        first = data["first"]
        row = torch.arange(count)
        if not bool(data["edge_mask"][row, first].all()):
            raise ValueError("infeasible first edge in label")
        frozen = None
        if policy.arm == "D15":
            _, _, frozen = actor.logits(encoded, data["edge_mask"], used_m, used_w, selected)
        selected[row, first] = True
        used_m[row, actor.machine_id[first]] = True
        used_w[row, actor.worker_id[first]] = True
        logits, feasible, _ = actor.logits(encoded, data["edge_mask"], used_m, used_w, selected,
                                           frozen_context=frozen)
        if not bool(feasible.gather(1, data["seconds"])[valid].all()):
            raise ValueError("infeasible second edge in label")
        scores = logits[:, :STOP].gather(1, data["seconds"])
    else:
        logits = policy.actor(machine, worker)
        scores = logits.gather(1, data["matching"])
    return scores.masked_fill(~valid, -1e9)


def listwise_loss(policy, data):
    scores = candidate_logits(policy, data)
    return -(data["targets"] * F.log_softmax(scores, dim=1)).sum(1).mean()


def pretrain(policy, records, seed, *, epochs=EPOCHS, batch_size=BATCH, on_epoch=None):
    if policy.arm not in PRETRAIN_ARMS:
        raise ValueError("arm does not receive conditional value pretraining")
    data = tensorize(records)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    n = len(records)
    total_updates = 0
    for epoch in range(1, epochs + 1):
        weighted_loss = 0.
        for indices in np.array_split(rng.permutation(n), max(1, int(np.ceil(n / batch_size)))):
            batch = subset(data, torch.as_tensor(indices, dtype=torch.long))
            loss = listwise_loss(policy, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite v15 pretraining loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(indices)
            total_updates += 1
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "mean_listwise_loss": weighted_loss / n,
                      "cumulative_updates": total_updates})
    return {"pretraining_records": n, "pretraining_epochs": epochs,
            "pretraining_updates": total_updates}
