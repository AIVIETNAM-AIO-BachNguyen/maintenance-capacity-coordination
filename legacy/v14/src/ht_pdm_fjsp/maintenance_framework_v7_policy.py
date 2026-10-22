"""Exact shortest-duration technician allocator within each legal machine subset.

The learned actor still chooses which machine subset to service. This filter only
resolves technician assignment for a fixed subset from current known durations.
"""
from __future__ import annotations

import numpy as np


def canonical_mask(config, actions, legal_mask):
    """Keep only minimum-total-duration legal matchings per serviced-machine set.

    Ties remain legal; the empty matching (WAIT) remains a policy decision.
    No physical event tape or future realization is consulted.
    """
    legal = np.asarray(legal_mask, dtype=bool)
    if legal.shape != (len(actions),) or not legal.any():
        raise ValueError("invalid legal action mask")
    best: dict[tuple[int, ...], int] = {}
    for index, matching in enumerate(actions):
        if not legal[index]:
            continue
        machines = tuple(sorted(machine for machine, _ in matching))
        duration = sum(config.service_time[machine][worker] for machine, worker in matching)
        best[machines] = min(best.get(machines, duration), duration)
    result = np.zeros_like(legal)
    for index, matching in enumerate(actions):
        if not legal[index]:
            continue
        machines = tuple(sorted(machine for machine, _ in matching))
        duration = sum(config.service_time[machine][worker] for machine, worker in matching)
        result[index] = duration == best[machines]
    if not result.any() or not np.all(~result | legal):
        raise RuntimeError("canonical assignment filter broke feasibility")
    return result
