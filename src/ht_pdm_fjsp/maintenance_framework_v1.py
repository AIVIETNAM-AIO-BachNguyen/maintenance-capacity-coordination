"""Current-information controller contract for persistent maintenance requests."""
from __future__ import annotations

from dataclasses import asdict
from typing import Protocol
import time

import gymnasium as gym
import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_request_policies import choose, family_configs, request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import RequestEnv, legal_matchings, queue_snapshot, submit_requests

VERSION = "maintenance_framework_v1"
CASES = ("reciprocal_scale2", "interchangeable_scale1", "reciprocal_scale3",
         "specialist_generalist_scale2")
HORIZON = 48
TAIL = 16


def configs():
    family = family_configs("full")
    return {name: family[name] for name in CASES}


def catalogue(machines=4, technicians=2):
    """All partial matchings, including empty WAIT, independent of state."""
    actions = set()

    def visit(j, used, pairs):
        if j == technicians:
            actions.add(tuple(sorted(pairs)))
            return
        visit(j + 1, used, pairs)
        for m in range(machines):
            if m not in used:
                visit(j + 1, used | {m}, (*pairs, (m, j)))

    visit(0, set(), ())
    return tuple(sorted(actions))


class Controller(Protocol):
    def choose(self, config, state, tick, *, case: str, seed: int):
        """Return a matching using only the present state/configuration."""


class RuleController:
    def __init__(self, name: str, check_timeout=lambda: None):
        if name not in ("fcfs", "risk_matching", "rollout"):
            raise ValueError(name)
        self.name = name
        self.forecast_transitions = 0
        self.check_timeout = check_timeout

    def choose(self, config, state, tick, *, case: str, seed: int):
        forecast = role_seed(62300000, f"{VERSION}:forecast:{case}:{seed}:{tick}")
        submissions, matching, diagnostic = choose(
            config, state, tick, "threshold", self.name,
            policy_horizon=HORIZON, forecast_seed=forecast,
            scenarios=16, forecast_horizon=6, check_timeout=self.check_timeout,
        )
        self.forecast_transitions += diagnostic["forecast_transitions"]
        return submissions, matching


class LearnedController:
    def __init__(self, model):
        self.model = model
        self.forecast_transitions = 0

    def choose(self, config, state, tick, *, case: str, seed: int):
        submissions = request_action(config, state, tick, "threshold")
        admitted = submit_requests(config, state, tick, submissions)
        actions = catalogue(config.machines, config.technicians)
        mask = np.asarray([a in legal_matchings(config, admitted) for a in actions], dtype=bool)
        obs = vector_observation(config, admitted, tick)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        index = int(action)
        if not mask[index]:
            raise RuntimeError("model selected masked action")
        return submissions, actions[index]


def vector_observation(config, admitted, tick):
    """Fixed 56-dimensional vector from current state; never an event tape."""
    p = admitted.physical
    values = [tick / HORIZON, (HORIZON - tick) / HORIZON]
    for m, request in enumerate(admitted.requests):
        worker = next((j for j, assigned in enumerate(p.assigned) if assigned == m), -1)
        values += [p.ages[m] / config.max_age, float(p.failed[m]),
                   float(request is not None and request.started_at is None),
                   float(request is not None and request.kind == "corrective"),
                   min(tick - request.created_at, 64) / 64 if request else 0.,
                   float(worker >= 0),
                   (p.remaining[worker] / 12 if worker >= 0 else 0.),
                   p.pending_wait[m] / 64]
    for j in range(config.technicians):
        values += [float(p.remaining[j] == 0), p.remaining[j] / 12,
                   (p.assigned[j] + 1) / (config.machines + 1)]
    for m in range(config.machines):
        for j in range(config.technicians):
            values += [float(config.service_time[m][j] > 0),
                       config.service_time[m][j] / 12]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (56,):
        raise RuntimeError(f"observation shape {result.shape}")
    return result


class FrameworkEnv(gym.Env):
    """H48 Gym adapter; final step charges the shared 16-tick tail exactly once."""
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or configs()["reciprocal_scale2"]
        self.actions = catalogue(self.config.machines, self.config.technicians)
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (56,), dtype=np.float32)
        self.inner = RequestEnv(self.config)
        self.episode_return = 0.
        self.seed_raw = 0
        self.cutoff = None
        self.cutoff_queue = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed_raw = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.inner.reset(role_seed(self.seed_raw, VERSION + ":physical"))
        self.episode_return = 0.
        self.cutoff = None
        self.cutoff_queue = None
        return self._observation(), {}

    def _admitted(self):
        submissions = request_action(self.config, self.inner.state, self.inner.time, "threshold")
        return submissions, submit_requests(self.config, self.inner.state, self.inner.time, submissions)

    def _observation(self):
        _, state = self._admitted()
        return vector_observation(self.config, state, self.inner.time)

    def action_masks(self):
        _, state = self._admitted()
        legal = set(legal_matchings(self.config, state))
        return np.asarray([action in legal for action in self.actions], dtype=bool)

    def step(self, action):
        if self.inner.time >= HORIZON:
            raise RuntimeError("episode finished")
        index = int(action)
        mask = self.action_masks()
        if index < 0 or index >= len(self.actions) or not mask[index]:
            raise ValueError("masked/infeasible action")
        submissions, _ = self._admitted()
        _, reward, _, info = self.inner.step(submissions, self.actions[index])
        self.episode_return += reward
        terminated = self.inner.time == HORIZON
        if terminated:
            self.cutoff = dict(self.inner.metrics)
            self.cutoff_queue = queue_snapshot(self.inner.state, HORIZON)
            for _ in range(TAIL):
                sub = request_action(self.config, self.inner.state, self.inner.time, "threshold")
                admitted = submit_requests(self.config, self.inner.state, self.inner.time, sub)
                matching = risk_matching(self.config, admitted, self.inner.time)
                _, tail_reward, _, _ = self.inner.step(sub, matching)
                reward += tail_reward
                self.episode_return += tail_reward
            if abs(self.episode_return + self.inner.metrics["objective"]) > 1e-9:
                raise RuntimeError("episode return differs from H+tail cost")
            info = {**info, "episode_metrics": self.episode_metrics()}
        return self._observation() if not terminated else np.zeros(56, np.float32), reward, terminated, False, info

    def episode_metrics(self):
        metrics = dict(self.inner.metrics)
        cutoff = self.cutoff
        return {**metrics, "H_cost": cutoff["objective"],
                "tail_cost": metrics["objective"] - cutoff["objective"],
                "H_availability": 1 - cutoff["unavailable_ticks"] / (self.config.machines * HORIZON),
                "H_failures": cutoff["failures"],
                **{"H_" + k: v for k, v in self.cutoff_queue.items()},
                **{"end_" + k: v for k, v in queue_snapshot(self.inner.state, self.inner.time).items()}}


def evaluate_episode(config, case, seed, controller):
    """Same physical tape and tail as Gym training, with controller diagnostics."""
    env = FrameworkEnv(config)
    env.reset(seed=seed)
    start = time.perf_counter()
    selection_seconds = 0.
    for tick in range(HORIZON):
        selection_start = time.perf_counter()
        submissions, matching = controller.choose(config, env.inner.state, tick, case=case, seed=seed)
        selection_seconds += time.perf_counter() - selection_start
        expected = request_action(config, env.inner.state, tick, "threshold")
        if submissions != expected:
            raise RuntimeError("controller violated fixed requester")
        admitted = submit_requests(config, env.inner.state, tick, submissions)
        if matching not in legal_matchings(config, admitted):
            raise RuntimeError("controller selected illegal matching")
        index = env.actions.index(matching)
        _, _, terminated, _, _ = env.step(index)
    assert terminated
    return {"case": case, "seed": seed, "controller": getattr(controller, "name", "learner"),
            "physical_seed": role_seed(seed, VERSION + ":physical"),
            "selection_seconds": selection_seconds,
            "episode_wall_seconds": time.perf_counter() - start,
            "forecast_transitions": controller.forecast_transitions,
            "episode_return": env.episode_return, **env.episode_metrics()}
