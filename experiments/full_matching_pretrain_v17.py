"""Listwise training on simulator-valued complete physical matchings.

The candidate-subset softmax is a ranking surrogate. Canonical paths have
one-to-one support with physical matchings, so each path log probability is
the exact matching log probability under a sequential policy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from maintenance_capacity_coordination.same_tick_policy_v15 import MAX_TOKENS, SequencePolicy

EPOCHS = 32
BATCH = 32
PRETRAIN_ARMS = ("C17", "D17", "I17", "R17", "B17", "S17", "G17")


def tensorize(records):
    if not records:
        raise ValueError("empty v17 label records")
    width = max(len(r["matching_indices"]) for r in records)
    n = len(records)
    matching = np.zeros((n, width), dtype=np.int64)
    sequences = np.full((n, width, MAX_TOKENS), -1, dtype=np.int64)
    targets = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    for i, row in enumerate(records):
        size = len(row["matching_indices"])
        if not 2 <= size <= 8 or len(row["sequences"]) != size or len(row["target_probabilities"]) != size:
            raise ValueError("malformed v17 complete-matching label row")
        matching[i, :size] = row["matching_indices"]
        targets[i, :size] = row["target_probabilities"]
        valid[i, :size] = True
        for j, tokens in enumerate(row["sequences"]):
            if not 0 <= len(tokens) <= MAX_TOKENS:
                raise ValueError("invalid canonical token length")
            sequences[i, j, :len(tokens)] = tokens
        if not np.isclose(targets[i].sum(), 1., atol=1e-6):
            raise ValueError("unnormalized target")
    return {
        "machine": torch.as_tensor(np.asarray([r["machine_obs"] for r in records]), dtype=torch.float32),
        "worker": torch.as_tensor(np.asarray([r["worker_obs"] for r in records]), dtype=torch.float32),
        "edge_mask": torch.as_tensor(np.asarray([r["edge_mask"] for r in records]), dtype=torch.bool),
        "matching": torch.from_numpy(matching), "sequences": torch.from_numpy(sequences),
        "targets": torch.from_numpy(targets), "candidate_valid": torch.from_numpy(valid),
    }


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def candidate_logits(policy, data):
    valid = data["candidate_valid"]
    if isinstance(policy, SequencePolicy):
        ri, ci = valid.nonzero(as_tuple=True)
        logprob, _ = policy.logprob_entropy(
            data["machine"][ri], data["worker"][ri], data["edge_mask"][ri],
            data["sequences"][ri, ci])
        scores = torch.full(valid.shape, -1e9, dtype=logprob.dtype, device=logprob.device)
        scores = scores.index_put((ri, ci), logprob)
    else:
        logits = policy.actor(data["machine"], data["worker"])
        scores = logits.gather(1, data["matching"])
    return scores.masked_fill(~valid, -1e9)


def listwise_loss(policy, data):
    scores = candidate_logits(policy, data)
    return -(data["targets"] * F.log_softmax(scores, dim=1)).sum(1).mean()


def pretrain(policy, records, seed, *, epochs=EPOCHS, batch_size=BATCH, on_epoch=None):
    data = tensorize(records)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    n = len(records)
    updates = 0
    for epoch in range(1, epochs + 1):
        weighted = 0.
        for indices in np.array_split(rng.permutation(n), max(1, int(np.ceil(n / batch_size)))):
            batch = subset(data, torch.as_tensor(indices, dtype=torch.long))
            loss = listwise_loss(policy, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite v17 matching pretraining loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
            optimizer.step()
            weighted += float(loss.detach()) * len(indices)
            updates += 1
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "mean_listwise_loss": weighted / n,
                      "cumulative_updates": updates})
    return {"pretraining_records": n, "pretraining_epochs": epochs,
            "pretraining_updates": updates}
