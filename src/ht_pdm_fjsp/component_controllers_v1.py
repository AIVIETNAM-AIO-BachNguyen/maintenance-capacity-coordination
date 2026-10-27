"""Full-information JCST/1 engineering controllers; no training or checkpoint loading."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

from .component_service_v1 import Action, Config, State, hazard, legal_actions, transition

OBSERVATION_MANIFEST = {
    "version": "JCST/1.1",
    "current": ["tick", "all component ages", "all failure flags",
                "all technician targets", "all job kinds", "all remaining times"],
    "public_model": ["Weibull scale and shape", "duration and compatibility",
                     "cost coefficients", "controlled horizon", "tail controller"],
    "future_realizations": False,
    "controller_model_access": {
        "wait": False, "corrective": False, "periodic": False, "threshold": False,
        "risk": "public one-step hazard and duration", "rollout": "public transition and forecast seeds",
    },
}


@dataclass(frozen=True)
class Observation:
    config: Config
    state: State
    actions: tuple[Action, ...]
    horizon: int
    tail: int


@dataclass(frozen=True)
class Decision:
    action: Action
    root_actions_evaluated: int = 0
    forecast_scenarios: int = 0
    simulated_transitions: int = 0
    selection_seconds: float = 0.0
    wall_seconds: float = 0.0
    forecast_records: tuple[dict, ...] = ()


def observe(config: Config, state: State, horizon: int, tail: int) -> Observation:
    if type(horizon) is not int or horizon < 1 or type(tail) is not int or tail < 0:
        raise ValueError("horizon/tail")
    return Observation(config, state, legal_actions(config, state), horizon, tail)


def _finish(action: Action, start: float, roots: int = 0, scenarios: int = 0,
            transitions: int = 0, forecast_records: tuple[dict, ...] = ()) -> Decision:
    elapsed = time.perf_counter() - start
    return Decision(action, roots, scenarios, transitions, elapsed, elapsed, forecast_records)


class Wait:
    name = "wait"
    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        return _finish((), start)


class Corrective:
    name = "corrective"
    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        c = obs.config
        failed = obs.state.components
        candidates = (a for a in obs.actions if all(failed[m][k].failed for m, k, _ in a))
        # More corrective starts, then shorter total occupancy, then stable index tie.
        action = min(candidates, key=lambda a: (-len(a), sum(c.duration[m][k][j] for m,k,j in a), a))
        return _finish(action, start)


class Threshold:
    name = "threshold"
    def __init__(self, age: int = 2):
        if type(age) is not int or age < 0:
            raise ValueError("threshold age")
        self.age = age

    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        c = obs.config
        items = obs.state.components
        candidates = (a for a in obs.actions if all(items[m][k].failed or items[m][k].age >= self.age
                                                  for m,k,_ in a))
        action = min(candidates, key=lambda a: (-sum(10 if items[m][k].failed else 1 for m,k,_ in a),
                                                 sum(c.duration[m][k][j] for m,k,j in a), a))
        return _finish(action, start)



class Periodic:
    name = "periodic"
    def __init__(self, period: int = 6):
        if type(period) is not int or period < 1:
            raise ValueError("positive period")
        self.period = period

    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        c = obs.config
        items = obs.state.components
        # Calendar inspection: healthy component c on m is due when its
        # staggered phase is reached. Failed components are always eligible.
        due = lambda m,k: ((obs.state.tick - (m + k)) % self.period == 0)
        candidates = (a for a in obs.actions
                      if all(items[m][k].failed or due(m,k) for m,k,_ in a))
        action = min(candidates, key=lambda a: (
            -sum(10 if items[m][k].failed else 1 for m,k,_ in a),
            sum(c.duration[m][k][j] for m,k,j in a), a))
        return _finish(action, start)


def _edge_value(obs: Observation, edge: tuple[int,int,int]) -> float:
    m, k, j = edge
    c = obs.config
    item = obs.state.components[m][k]
    duration = c.duration[m][k][j]
    # Public, deterministic heuristic. Duration discounts repair value and commitment.
    if item.failed:
        benefit = c.unavailable_cost * (obs.horizon + obs.tail - obs.state.tick + 1)
        benefit += c.failure_cost
        start_cost = c.corrective_cost
    else:
        risk = hazard(c.components[m][k], item.age)
        benefit = risk * (c.failure_cost + c.unavailable_cost * max(1, duration))
        start_cost = c.preventive_cost
    return (benefit - start_cost - c.unavailable_cost * duration) / duration


class RiskMatching:
    name = "risk"
    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        # Exact root enumeration under this additive heuristic, not horizon optimization.
        values = {edge: _edge_value(obs, edge) for action in obs.actions for edge in action}
        action = min(obs.actions, key=lambda a: (-sum(values[e] for e in a), a))
        return _finish(action, start, roots=len(obs.actions))


def forecast_seed(base: int, scenario: int) -> int:
    digest = hashlib.sha256(f"JCST/1:forecast:{base}:{scenario}".encode()).digest()
    # Physical campaign seeds are restricted below 2**64. Forecast streams
    # occupy a disjoint integer namespace while retaining hash-based keys.
    return (1 << 64) + int.from_bytes(digest[:8], "big")


class BoundedRollout:
    name = "rollout"
    def __init__(self, scenarios: int = 2, depth: int = 2,
                 max_transitions: int = 10000):
        if any(type(x) is not int or x < 1 for x in (scenarios, depth, max_transitions)):
            raise ValueError("positive rollout budget")
        self.scenarios = scenarios
        self.depth = depth
        self.max_transitions = max_transitions

    def choose(self, obs: Observation) -> Decision:
        start = time.perf_counter()
        roots = len(obs.actions)
        steps = min(self.depth, obs.horizon + obs.tail - obs.state.tick)
        needed = roots * self.scenarios * steps
        if needed > self.max_transitions:
            raise ValueError("rollout transition cap exceeded")
        risk = RiskMatching()
        best: Action = ()
        best_cost = math.inf
        used = 0
        forecast_records = []
        for action in obs.actions:
            total = 0.0
            for scenario in range(self.scenarios):
                seed = forecast_seed(obs.state.tick * 1000003 + 7919, scenario)
                state = obs.state
                chosen = action
                branch_cost = 0.0
                branch_steps = 0
                for _ in range(steps):
                    result = transition(obs.config, state, chosen, seed)
                    branch_cost += result.cost
                    state = result.state
                    used += 1
                    branch_steps += 1
                    if state.tick < obs.horizon + obs.tail:
                        chosen = risk.choose(observe(obs.config, state, obs.horizon, obs.tail)).action
                total += branch_cost
                forecast_records.append(dict(root=[list(edge) for edge in action],
                                             scenario=scenario, seed=seed,
                                             cost=branch_cost, transitions=branch_steps))
            if (total, action) < (best_cost, best):
                best_cost, best = total, action
        return _finish(best, start, roots, self.scenarios, used,
                       tuple(forecast_records))
