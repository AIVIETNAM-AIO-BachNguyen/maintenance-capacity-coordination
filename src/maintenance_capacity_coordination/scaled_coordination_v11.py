"""Present-state N/K adapter over unchanged request and physical transitions."""
from __future__ import annotations

import numpy as np

from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL, catalogue
from ht_pdm_fjsp.maintenance_requests import RequestEnv
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv


def global_dim(machines: int, technicians: int) -> int:
    return 2 + 8 * machines + 3 * technicians + 2 * machines * technicians


def local_dim(technicians: int) -> int:
    return 15 + 3 * technicians


def vector_observation_scaled(config, admitted, tick):
    """Generalize the inherited 56-vector contract without adding future data."""
    p = admitted.physical
    values = [tick / HORIZON, (HORIZON - tick) / HORIZON]
    for m, request in enumerate(admitted.requests):
        worker = next((j for j, assigned in enumerate(p.assigned) if assigned == m), -1)
        values += [p.ages[m] / config.max_age, float(p.failed[m]),
                   float(request is not None and request.started_at is None),
                   float(request is not None and request.kind == "corrective"),
                   min(tick - request.created_at, 64) / 64 if request else 0.,
                   float(worker >= 0), p.remaining[worker] / 12 if worker >= 0 else 0.,
                   p.pending_wait[m] / 64]
    for j in range(config.technicians):
        values += [float(p.remaining[j] == 0), p.remaining[j] / 12,
                   (p.assigned[j] + 1) / (config.machines + 1)]
    for m in range(config.machines):
        for j in range(config.technicians):
            values += [float(config.service_time[m][j] > 0),
                       config.service_time[m][j] / 12]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (global_dim(config.machines, config.technicians),):
        raise RuntimeError("scaled global observation dimension changed")
    return result


def machine_features_scaled(config, state, tick):
    """Generalize inherited full-capacity requester features for every machine."""
    p = state.physical
    rows = []
    for m, request in enumerate(state.requests):
        worker = next((j for j, assigned in enumerate(p.assigned) if assigned == m), -1)
        local = [
            tick / HORIZON, p.ages[m] / config.max_age, float(p.failed[m]),
            float(request is not None),
            float(request is not None and request.kind == "corrective"),
            min(tick - request.created_at, 64) / 64 if request else 0.,
            float(worker >= 0), p.remaining[worker] / 12 if worker >= 0 else 0.,
            config.failure_age / config.max_age, config.failure_probability,
            config.preventive_cost / 20, config.corrective_cost / 20,
            config.unavailable_cost / 20, config.failure_cost / 20,
            *(config.service_time[m][j] / 12 for j in range(config.technicians)),
            *(config.restoration[m][j] for j in range(config.technicians)),
            *(float(p.remaining[j] == 0 and config.service_time[m][j] > 0)
              for j in range(config.technicians)),
            min((p.remaining[j] for j in range(config.technicians)
                 if config.service_time[m][j] > 0), default=0) / 12,
        ]
        rows.append(local)
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (config.machines, local_dim(config.technicians)):
        raise RuntimeError("scaled machine observation dimension changed")
    return result


def worker_features_scaled(config, state):
    p = state.physical
    rows = []
    for j in range(config.technicians):
        assigned = p.assigned[j]
        rows.append([float(p.remaining[j] == 0), p.remaining[j] / 12,
                     (assigned + 1) / (config.machines + 1),
                     float(assigned >= 0 and p.failed[assigned])])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (config.technicians, 4):
        raise RuntimeError("scaled worker observation dimension changed")
    return result


def actor_inputs_scaled(config, admitted, tick):
    return machine_features_scaled(config, admitted, tick), worker_features_scaled(config, admitted)


class ScaledCoordinationEnv(CoordinationEnv):
    """N/K matching interface; inherits exact H/tail transition and accounting."""

    def __init__(self, config):
        config.validate()
        if config.horizon < HORIZON + TAIL:
            raise ValueError("scaled contract requires complete H48 + tail16")
        self.config = config
        self.actions = catalogue(config.machines, config.technicians)
        self.inner = RequestEnv(config)
        self.tick = 0
        self.episode_return = 0.
        self.legal_service_waits = 0
        self.request_creation_capacity = []
        self.cutoff = None
        self.cutoff_queue = None

    def pre_observation(self):
        return vector_observation_scaled(self.config, self.inner.state, self.tick)

    def coordinator_observation(self, submissions):
        return vector_observation_scaled(self.config, self.admitted(submissions), self.tick)

    def machine_observation(self, arm):
        if arm not in ("C", "threshold"):
            raise ValueError(arm)
        return machine_features_scaled(self.config, self.inner.state, self.tick)
