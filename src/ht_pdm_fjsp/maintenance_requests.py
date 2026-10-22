"""Persistent shared requests layered on the unchanged completion-time physics.

A request is not a reservation. Queued PM machines continue operating; only a
service start occupies a machine and technician. All times are tick boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from ht_pdm_fjsp import maintenance_dispatch as physical

VERSION = "shared_maintenance_requests_v1"


@dataclass(frozen=True)
class MaintenanceRequest:
    machine: int
    sequence: int
    created_at: int
    kind: Literal["preventive", "corrective"]
    started_at: int | None = None
    technician: int | None = None
    upgraded_at: int | None = None


@dataclass(frozen=True)
class RequestState:
    physical: physical.DispatchState
    requests: tuple[MaintenanceRequest | None, ...]
    generations: tuple[int, ...]


def initial_state(config, initial=None):
    """Accept an unoccupied physical initial state; auto-create failed requests."""
    if initial is None:
        initial = physical.DispatchEnv(config).state
    physical.validate_state(config, initial)
    if any(m >= 0 for m in initial.assigned):
        raise ValueError("initial service requires explicit request metadata")
    requests = tuple(
        MaintenanceRequest(m, 1, 0, "corrective") if failed else None
        for m, failed in enumerate(initial.failed)
    )
    result = RequestState(initial, requests, tuple(int(r is not None) for r in requests))
    validate_state(config, result, 0)
    return result


def validate_state(config, state, time):
    physical.validate_state(config, state.physical)
    if type(time) is not int or not 0 <= time <= config.horizon:
        raise ValueError("invalid request clock")
    if len(state.requests) != config.machines or len(state.generations) != config.machines:
        raise ValueError("request state dimensions")
    workers = {m: j for j, m in enumerate(state.physical.assigned) if m >= 0}
    for m, (request, generation) in enumerate(zip(state.requests, state.generations)):
        if type(generation) is not int or generation < 0:
            raise ValueError("invalid request generation")
        if request is None:
            if m in workers or state.physical.failed[m]:
                raise ValueError("failed or occupied machine without request")
            continue
        if (type(request.machine) is not int or request.machine != m
                or type(request.sequence) is not int or request.sequence != generation
                or generation < 1 or type(request.created_at) is not int
                or not 0 <= request.created_at <= time
                or request.kind not in ("preventive", "corrective")):
            raise ValueError("invalid request identity or creation")
        if (request.kind == "corrective") != bool(state.physical.failed[m]):
            raise ValueError("request kind inconsistent with machine")
        if request.upgraded_at is not None and (
                type(request.upgraded_at) is not int or request.kind != "corrective"
                or not request.created_at <= request.upgraded_at <= time):
            raise ValueError("invalid request upgrade")
        if request.started_at is None:
            if request.technician is not None or m in workers:
                raise ValueError("queued request has assigned resource")
        elif (type(request.started_at) is not int
              or not request.created_at <= request.started_at <= time
              or type(request.technician) is not int
              or workers.get(m) != request.technician):
            raise ValueError("service metadata inconsistent with occupancy")


def submit_requests(config, state, time, submissions):
    """Create PM requests before dispatch, with no service commitment."""
    submissions = tuple(submissions)
    if any(type(m) is not int or not 0 <= m < config.machines for m in submissions):
        raise ValueError("invalid submitted machine")
    if len(set(submissions)) != len(submissions):
        raise ValueError("duplicate submission")
    requests, generations = list(state.requests), list(state.generations)
    for m in sorted(submissions):
        if requests[m] is not None:
            raise ValueError("machine already has an active request")
        if state.physical.failed[m] or m in state.physical.assigned:
            raise ValueError("PM requires an operating unassigned machine")
        generations[m] += 1
        requests[m] = MaintenanceRequest(m, generations[m], time, "preventive")
    return RequestState(state.physical, tuple(requests), tuple(generations))


def legal_matchings(config, state):
    """All partial matchings on the shared queue, including doing nothing."""
    edges = physical.eligible_edges(config, state.physical)
    queued = tuple(m for m, r in enumerate(state.requests)
                   if r is not None and r.started_at is None)
    result = []

    def visit(worker, used, pairs):
        if worker == config.technicians:
            result.append(tuple(sorted(pairs)))
            return
        visit(worker + 1, used, pairs)
        for m in queued:
            if m not in used and edges[m][worker]:
                visit(worker + 1, used | {m}, (*pairs, (m, worker)))

    visit(0, set(), ())
    return tuple(sorted(result))


def transition(config, state, time, submissions, matching, events):
    """Atomic request admission, dispatch, physical interval and lifecycle update."""
    validate_state(config, state, time)
    if time >= config.horizon:
        raise RuntimeError("episode finished")
    matching = tuple(tuple(pair) for pair in matching)
    if any(len(pair) != 2 or any(type(i) is not int for i in pair) for pair in matching):
        raise ValueError("integer machine/technician pairs required")
    admitted = submit_requests(config, state, time, submissions)
    physical.validate_matching(config, state.physical, matching)
    requests, generations = list(admitted.requests), list(admitted.generations)
    lifecycle = [dict(event="created", at=time, **asdict(r))
                 for old, r in zip(state.requests, admitted.requests)
                 if old is None and r is not None]
    waits_started = []
    for m, j in sorted(matching):
        request = requests[m]
        if request is None or request.started_at is not None:
            raise ValueError("service start requires a queued request")
        request = replace(request, started_at=time, technician=j)
        requests[m] = request
        waits_started.append(time - request.created_at)
        lifecycle.append(dict(event="started", at=time, **asdict(request)))
    queued = [r for r in requests if r is not None and r.started_at is None]
    next_physical, cost, info = physical.transition(config, state.physical, matching, events)
    completed_waits = []
    for m, request in enumerate(requests):
        if request is not None and request.started_at is not None:
            if m not in next_physical.assigned:
                completed_waits.append(request.started_at - request.created_at)
                lifecycle.append(dict(event="completed", at=time + 1, **asdict(request)))
                requests[m] = None
        elif next_physical.failed[m]:
            if request is None:
                generations[m] += 1
                request = MaintenanceRequest(m, generations[m], time + 1, "corrective")
                lifecycle.append(dict(event="created", at=time + 1, **asdict(request)))
            elif request.kind == "preventive":
                request = replace(request, kind="corrective", upgraded_at=time + 1)
                lifecycle.append(dict(event="upgraded", at=time + 1, **asdict(request)))
            requests[m] = request
    following = RequestState(next_physical, tuple(requests), tuple(generations))
    validate_state(config, following, time + 1)
    ages = [time + 1 - r.created_at for r in requests
            if r is not None and r.started_at is None]
    info.update(
        requests_created=sum(e["event"] == "created" for e in lifecycle),
        requests_upgraded=sum(e["event"] == "upgraded" for e in lifecycle),
        service_starts=len(matching),
        requests_completed=len(completed_waits),
        completed_wait_total=sum(completed_waits),
        queued_pm_ticks=sum(r.kind == "preventive" for r in queued),
        queued_cm_ticks=sum(r.kind == "corrective" for r in queued),
        request_wait_ticks=len(queued),
        max_wait_age=max([0, *ages, *waits_started]),
        lifecycle=lifecycle,
    )
    return following, cost, info


def observation(config, state, time):
    """Only configuration/current state; no realized future events."""
    return dict(time=time, state=asdict(state), physical=physical.observation(config, state.physical, time))


def queue_snapshot(state, time):
    queued = [r for r in state.requests if r is not None and r.started_at is None]
    return dict(
        pending=len(queued),
        pending_pm=sum(r.kind == "preventive" for r in queued),
        pending_cm=sum(r.kind == "corrective" for r in queued),
        in_service=sum(r is not None and r.started_at is not None for r in state.requests),
        pending_wait_age_sum=sum(time - r.created_at for r in queued),
        oldest_pending_age=max([0, *(time - r.created_at for r in queued)]),
    )


def state_from_dict(data):
    return RequestState(
        physical.DispatchState(**{k: tuple(v) for k, v in data["physical"].items()}),
        tuple(MaintenanceRequest(**r) if r is not None else None for r in data["requests"]),
        tuple(data["generations"]),
    )


class RequestEnv:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.reset(0)

    def reset(self, seed, initial=None):
        base = physical.DispatchEnv(self.config)
        base.reset(seed, initial)
        self.state = initial_state(self.config, base.state)
        self._events = base.events
        self.time = 0
        self.metrics = dict(base.metrics)
        self.metrics.update({key: 0 for key in (
            "requests_upgraded", "service_starts", "requests_completed", "completed_wait_total",
            "queued_pm_ticks", "queued_cm_ticks", "request_wait_ticks", "max_wait_age")})
        self.metrics["requests_created"] = sum(r is not None for r in self.state.requests)
        return observation(self.config, self.state, self.time)

    def step(self, submissions=(), matching=()):
        following, cost, info = transition(self.config, self.state, self.time,
                                         submissions, matching, self._events[self.time]
                                         if self.time < self.config.horizon else ())
        self.state = following
        self.time += 1
        for key, value in info.items():
            if key == "lifecycle":
                continue
            if key == "max_wait_age":
                self.metrics[key] = max(self.metrics[key], value)
            else:
                self.metrics[key] += value
        if abs(self.metrics["objective"] - sum(self.metrics[k] for k in
                ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-8:
            raise RuntimeError("economic cost reconciliation")
        active = sum(r is not None for r in self.state.requests)
        if self.metrics["requests_created"] - self.metrics["requests_completed"] != active:
            raise RuntimeError("request conservation")
        in_service = sum(r is not None and r.started_at is not None for r in self.state.requests)
        if (self.metrics["service_starts"] - self.metrics["requests_completed"] != in_service
                or self.metrics["requests_completed"] != self.metrics["completed_jobs"]
                or self.metrics["queued_cm_ticks"] != self.metrics["failed_waiting_ticks"]):
            raise RuntimeError("service and waiting reconciliation")
        return observation(self.config, self.state, self.time), -cost, self.time == self.config.horizon, info
