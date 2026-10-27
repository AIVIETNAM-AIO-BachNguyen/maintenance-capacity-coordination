"""JCST/1 revision 1.1: independent, pure component service physics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

VERSION = "JCST/1.1"
Action = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ComponentConfig:
    scale: float
    shape: float

    def validate(self) -> None:
        if any(not math.isfinite(x) or x <= 0 for x in (self.scale, self.shape)):
            raise ValueError("positive finite Weibull parameters required")


@dataclass(frozen=True)
class ComponentState:
    age: int = 0
    failed: bool = False


@dataclass(frozen=True)
class ServiceJob:
    machine: int
    component: int
    technician: int
    kind: str
    remaining: int


@dataclass(frozen=True)
class Config:
    components: tuple[tuple[ComponentConfig, ...], ...]
    duration: tuple[tuple[tuple[int, ...], ...], ...]
    preventive_cost: float = 1.0
    corrective_cost: float = 2.0
    unavailable_cost: float = 6.0
    failure_cost: float = 15.0
    version: str = VERSION

    @property
    def machines(self) -> int:
        return len(self.components)

    @property
    def technicians(self) -> int:
        return len(self.duration[0][0]) if self.duration and self.duration[0] else 0

    def validate(self) -> None:
        if self.version != VERSION or self.machines < 1 or self.technicians < 1:
            raise ValueError("invalid version or dimensions")
        if len(self.duration) != self.machines:
            raise ValueError("duration machine dimension")
        for comps, durations in zip(self.components, self.duration):
            if not comps or len(comps) != len(durations):
                raise ValueError("component dimension")
            for comp, row in zip(comps, durations):
                comp.validate()
                if len(row) != self.technicians or any(type(d) is not int or d < 0 for d in row):
                    raise ValueError("duration row")
                if not any(row):
                    raise ValueError("component without qualified technician")
        if any(not math.isfinite(x) or x < 0 for x in (
            self.preventive_cost, self.corrective_cost, self.unavailable_cost, self.failure_cost
        )):
            raise ValueError("cost coefficients")


@dataclass(frozen=True)
class State:
    tick: int
    components: tuple[tuple[ComponentState, ...], ...]
    jobs: tuple[ServiceJob | None, ...]


@dataclass(frozen=True)
class Outcome:
    state: State
    cost: float
    breakdown: dict[str, float]
    events: tuple[dict, ...]

    @property
    def reward(self) -> float:
        return -self.cost


def initial_state(config: Config) -> State:
    config.validate()
    return State(0, tuple(tuple(ComponentState() for _ in row) for row in config.components),
                 tuple(None for _ in range(config.technicians)))


def validate_state(config: Config, state: State) -> None:
    config.validate()
    if type(state.tick) is not int or state.tick < 0 or len(state.components) != config.machines:
        raise ValueError("state tick or machine dimension")
    if len(state.jobs) != config.technicians:
        raise ValueError("state technician dimension")
    for items, specs in zip(state.components, config.components):
        if len(items) != len(specs):
            raise ValueError("state component dimension")
        for item in items:
            if type(item.age) is not int or item.age < 0 or type(item.failed) is not bool:
                raise ValueError("component state")
    seen = set()
    for j, job in enumerate(state.jobs):
        if job is None:
            continue
        m, c = job.machine, job.component
        if (type(m) is not int or type(c) is not int or job.technician != j
                or not 0 <= m < config.machines or not 0 <= c < len(config.components[m])
                or m in seen or config.duration[m][c][j] == 0
                or type(job.remaining) is not int
                or not 1 <= job.remaining <= config.duration[m][c][j]
                or job.kind not in ("PM", "CM")):
            raise ValueError("invalid service job")
        if job.kind == "CM" and not state.components[m][c].failed:
            raise ValueError("CM target unexpectedly healthy")
        if job.kind == "PM" and state.components[m][c].failed:
            raise ValueError("PM target unexpectedly failed")
        seen.add(m)


def machine_modes(state: State) -> tuple[str, ...]:
    servicing = {job.machine for job in state.jobs if job is not None}
    return tuple("service" if m in servicing else
                 "breakdown" if any(c.failed for c in row) else "working"
                 for m, row in enumerate(state.components))


def legal_actions(config: Config, state: State) -> tuple[Action, ...]:
    validate_state(config, state)
    busy_m = {job.machine for job in state.jobs if job is not None}
    free_j = tuple(j for j, job in enumerate(state.jobs) if job is None)
    result: list[Action] = []
    def visit(m: int, used: set[int], action: Action) -> None:
        if m == config.machines:
            result.append(action)
            return
        visit(m + 1, used, action)
        if m in busy_m:
            return
        for c, row in enumerate(config.duration[m]):
            for j in free_j:
                if j not in used and row[j] > 0:
                    visit(m + 1, used | {j}, action + ((m, c, j),))
    visit(0, set(), ())
    return tuple(sorted(result, key=lambda a: (len(a), a)))


def canonical_action(config: Config, state: State, action) -> Action:
    validate_state(config, state)
    try:
        triples = tuple(tuple(x) for x in action)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid action shape") from exc
    seen_m: set[int] = set()
    seen_j: set[int] = set()
    busy_m = {job.machine for job in state.jobs if job is not None}
    for triple in triples:
        if len(triple) != 3 or any(type(v) is not int for v in triple):
            raise ValueError("action triple")
        m, c, j = triple
        if (not 0 <= m < config.machines or not 0 <= c < len(config.components[m])
                or not 0 <= j < config.technicians or m in busy_m
                or state.jobs[j] is not None or config.duration[m][c][j] == 0
                or m in seen_m or j in seen_j):
            raise ValueError("illegal complete action")
        seen_m.add(m)
        seen_j.add(j)
    return tuple(sorted(triples))


def hazard(spec: ComponentConfig, age: int) -> float:
    spec.validate()
    if type(age) is not int or age < 0:
        raise ValueError("age")
    delta = ((age + 1) / spec.scale) ** spec.shape - (age / spec.scale) ** spec.shape
    return -math.expm1(-delta)


def keyed_uniform(seed: int, tick: int, machine: int, component: int) -> float:
    key = f"JCST/1|{seed}|{tick}|{machine}|{component}".encode()
    n = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return (n + 0.5) / (2 ** 64)


def transition(config: Config, state: State, action: Action, seed: int,
               uniforms: Mapping[tuple[int, int], float] | None = None) -> Outcome:
    action = canonical_action(config, state, action)
    if type(seed) is not int:
        raise ValueError("integer seed required")
    if uniforms is not None:
        for key, value in uniforms.items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or any(type(x) is not int for x in key)
                    or not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError("invalid uniform override")
    comps = [list(row) for row in state.components]
    jobs = list(state.jobs)
    events: list[dict] = []
    pm = cm = 0
    for m, c, j in action:
        kind = "CM" if state.components[m][c].failed else "PM"
        jobs[j] = ServiceJob(m, c, j, kind, config.duration[m][c][j])
        pm += kind == "PM"
        cm += kind == "CM"
        events.append(dict(type="start", tick=state.tick, machine=m, component=c,
                           technician=j, kind=kind, duration=jobs[j].remaining))
    servicing = {job.machine for job in jobs if job is not None}
    producing = tuple(m for m, row in enumerate(state.components)
                      if m not in servicing and not any(c.failed for c in row))
    unavailable = config.machines - len(producing)
    failures = 0
    for m in producing:
        for c, before in enumerate(state.components[m]):
            u = (uniforms[(m, c)] if uniforms is not None and (m, c) in uniforms
                 else keyed_uniform(seed, state.tick, m, c))
            failed = u < hazard(config.components[m][c], before.age)
            comps[m][c] = ComponentState(before.age + 1, failed)
            if failed:
                failures += 1
                events.append(dict(type="failure", tick=state.tick + 1,
                                   machine=m, component=c, uniform=u))
    for j, job in enumerate(jobs):
        if job is None:
            continue
        events.append(dict(type="occupancy", tick=state.tick, machine=job.machine,
                           component=job.component, technician=j, remaining=job.remaining))
        left = job.remaining - 1
        if left:
            jobs[j] = ServiceJob(job.machine, job.component, j, job.kind, left)
        else:
            comps[job.machine][job.component] = ComponentState()
            jobs[j] = None
            events.append(dict(type="completion", tick=state.tick + 1,
                               machine=job.machine, component=job.component,
                               technician=j, kind=job.kind))
    breakdown = dict(
        preventive=config.preventive_cost * pm,
        corrective=config.corrective_cost * cm,
        unavailable=config.unavailable_cost * unavailable,
        failure=config.failure_cost * failures,
    )
    next_state = State(state.tick + 1, tuple(tuple(row) for row in comps), tuple(jobs))
    validate_state(config, next_state)
    return Outcome(next_state, sum(breakdown.values()), breakdown, tuple(events))


def config_json(config: Config) -> str:
    config.validate()
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def state_json(config: Config, state: State) -> str:
    validate_state(config, state)
    return json.dumps(asdict(state), sort_keys=True, separators=(",", ":"))


def config_from_json(raw: str) -> Config:
    data = json.loads(raw)
    config = Config(
        tuple(tuple(ComponentConfig(**c) for c in row) for row in data["components"]),
        tuple(tuple(tuple(row) for row in machine) for machine in data["duration"]),
        data["preventive_cost"], data["corrective_cost"], data["unavailable_cost"],
        data["failure_cost"], data["version"])
    config.validate()
    return config


def state_from_json(config: Config, raw: str) -> State:
    data = json.loads(raw)
    state = State(data["tick"],
                  tuple(tuple(ComponentState(**c) for c in row) for row in data["components"]),
                  tuple(ServiceJob(**j) if j is not None else None for j in data["jobs"]))
    validate_state(config, state)
    return state


def engineering_config() -> Config:
    return Config(
        tuple((ComponentConfig(6, 5), ComponentConfig(8, 5)) for _ in range(3)),
        tuple(((2, 8), (8, 2)) for _ in range(3)),
    )
