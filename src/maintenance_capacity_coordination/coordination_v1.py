"""Two-stage demand/capacity adapter over the unchanged request simulator.

The environment owns event tapes; policies receive only arrays made from the
present state. The matching action is conditional on admitted requests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    HORIZON, TAIL, VERSION as BASE_VERSION, catalogue, vector_observation,
)
from ht_pdm_fjsp.maintenance_request_policies import request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import RequestEnv, legal_matchings, queue_snapshot, submit_requests

VERSION = "coordination_pilot_v1"
ARMS = ("B", "C", "D", "E")
LOCAL_DIM = 21
GLOBAL_DIM = 56


def eligible_requests(config, state):
    """Machines allowed to submit a new PM at this tick."""
    occupied = set(state.physical.assigned)
    return np.asarray([
        r is None and not state.physical.failed[m] and m not in occupied
        for m, r in enumerate(state.requests)
    ], dtype=bool)


def requester_features(config, state, tick, arm):
    """Shared per-machine actor inputs; B has no capacity feedback.

    E receives an additional full-state vector. Neither this function nor
    vector_observation can access RequestEnv._events.
    """
    if arm not in ("B", "C", "E"):
        raise ValueError(arm)
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
        ]
        capacity = [
            *(float(p.remaining[j] == 0 and config.service_time[m][j] > 0)
              for j in range(config.technicians)),
            min((p.remaining[j] for j in range(config.technicians)
                 if config.service_time[m][j] > 0), default=0) / 12,
        ]
        if arm == "B":
            capacity = [0.] * len(capacity)
        row = local + capacity
        if len(row) != LOCAL_DIM:
            raise RuntimeError("local observation dimension changed")
        rows.append(row)
    result = np.asarray(rows, dtype=np.float32)
    if arm == "E":
        global_state = vector_observation(config, state, tick)
        result = np.concatenate((result, np.broadcast_to(global_state, (config.machines, GLOBAL_DIM))), axis=1)
    return result


@dataclass(frozen=True)
class Decision:
    submissions: tuple[int, ...]
    matching: tuple[tuple[int, int], ...]
    matching_index: int


class CoordinationEnv:
    """H=48 decisions and the inherited deterministic 16-tick tail rule."""

    def __init__(self, config):
        if (config.machines, config.technicians) != (4, 2):
            raise ValueError("pilot decoder is restricted to N4/K2")
        self.config = config
        self.actions = catalogue(4, 2)
        self.inner = RequestEnv(config)
        self.tick = 0
        self.episode_return = 0.
        self.legal_service_waits = 0
        self.request_creation_capacity = []
        self.cutoff = None
        self.cutoff_queue = None

    def reset(self, seed):
        self.inner.reset(role_seed(int(seed), BASE_VERSION + ":physical"))
        self.tick = 0
        self.episode_return = 0.
        self.legal_service_waits = 0
        self.request_creation_capacity = []
        self.cutoff = None
        self.cutoff_queue = None
        return self.pre_observation()

    def pre_observation(self):
        return vector_observation(self.config, self.inner.state, self.tick)

    def request_mask(self):
        return eligible_requests(self.config, self.inner.state)

    def machine_observation(self, arm):
        return requester_features(self.config, self.inner.state, self.tick, arm)

    def fixed_submissions(self, arm):
        if arm == "D":
            return tuple(np.flatnonzero(self.request_mask()).tolist())
        if arm == "threshold":
            return request_action(self.config, self.inner.state, self.tick, "threshold")
        raise ValueError(arm)

    def admitted(self, submissions):
        return submit_requests(self.config, self.inner.state, self.tick, submissions)

    def coordinator_observation(self, submissions):
        return vector_observation(self.config, self.admitted(submissions), self.tick)

    def matching_mask(self, submissions):
        legal = set(legal_matchings(self.config, self.admitted(submissions)))
        return np.asarray([action in legal for action in self.actions], dtype=bool)

    def step(self, submissions, matching_index):
        if self.tick >= HORIZON:
            raise RuntimeError("episode finished")
        submissions = tuple(int(m) for m in submissions)
        allowed = set(np.flatnonzero(self.request_mask()).tolist())
        if len(set(submissions)) != len(submissions) or any(m not in allowed for m in submissions):
            raise ValueError("illegal PM submission")
        mask = self.matching_mask(submissions)
        index = int(matching_index)
        if not 0 <= index < len(self.actions) or not mask[index]:
            raise ValueError("masked/infeasible matching")
        matching = self.actions[index]
        if index == 0 and mask.sum() > 1:
            self.legal_service_waits += 1
        for m in submissions:
            free_compatible = sum(
                self.inner.state.physical.remaining[j] == 0 and self.config.service_time[m][j] > 0
                for j in range(self.config.technicians)
            )
            self.request_creation_capacity.append(free_compatible)
        _, reward, _, _ = self.inner.step(submissions, matching)
        self.episode_return += reward
        self.tick += 1
        done = self.tick == HORIZON
        if done:
            self.cutoff = dict(self.inner.metrics)
            self.cutoff_queue = queue_snapshot(self.inner.state, HORIZON)
            for _ in range(TAIL):
                sub = request_action(self.config, self.inner.state, self.inner.time, "threshold")
                admitted = submit_requests(self.config, self.inner.state, self.inner.time, sub)
                tail_matching = risk_matching(self.config, admitted, self.inner.time)
                _, tail_reward, _, _ = self.inner.step(sub, tail_matching)
                reward += tail_reward
                self.episode_return += tail_reward
            if abs(self.episode_return + self.inner.metrics["objective"]) > 1e-8:
                raise RuntimeError("H+tail return/cost mismatch")
        return reward, done

    def metrics(self):
        if self.cutoff is None:
            raise RuntimeError("episode unfinished")
        end = self.inner.metrics
        return {
            **end,
            "H_cost": self.cutoff["objective"],
            "tail_cost": end["objective"] - self.cutoff["objective"],
            "H_availability": 1 - self.cutoff["unavailable_ticks"] / (self.config.machines * HORIZON),
            "H_failures": self.cutoff["failures"],
            "legal_service_waits": self.legal_service_waits,
            "request_creation_free_compatible_mean":
                float(np.mean(self.request_creation_capacity)) if self.request_creation_capacity else None,
            "episode_return": self.episode_return,
            **{"H_" + key: value for key, value in self.cutoff_queue.items()},
            **{"end_" + key: value for key, value in queue_snapshot(self.inner.state, self.inner.time).items()},
        }
