"""Disclosed nonlearning request/dispatch baselines with current-state inputs."""
import random

from ht_pdm_fjsp.maintenance_dispatch import DispatchConfig
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests, transition

REQUESTERS = ("threshold", "resource_aware")
DISPATCHERS = ("fcfs", "risk_matching", "rollout")


def family_configs(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    patterns = dict(
        interchangeable=((2, 2),) * 4,
        reciprocal=((1, 3), (1, 3), (3, 1), (3, 1)),
        specialist_generalist=((1, 0), (1, 3), (1, 2), (1, 4)),
    )
    horizon = 20 if profile == "smoke" else 64
    return {
        f"{name}_scale{scale}": DispatchConfig(
            4, 2, horizon, 4, .45,
            tuple(tuple(d * scale for d in row) for row in matrix),
            ((.75, .75),) * 4, unavailable_cost=6., failure_cost=15.,
        )
        for name, matrix in patterns.items()
        for scale in ((2,) if profile == "smoke" else (1, 2, 3))
    }


def request_action(config, state, time, requester):
    if requester not in REQUESTERS:
        raise ValueError("unknown requester")
    result = []
    for m, request in enumerate(state.requests):
        if request is not None:
            continue
        threshold = config.failure_age - 1
        if requester == "resource_aware":
            earliest = min(state.physical.remaining[j] for j in range(config.technicians)
                           if config.service_time[m][j] > 0)
            threshold = max(1, threshold - earliest)
        if state.physical.ages[m] >= threshold:
            result.append(m)
    return tuple(result)


def fcfs(config, state, time):
    available = {j for j, r in enumerate(state.physical.remaining) if r == 0}
    queued = sorted((r for r in state.requests if r is not None and r.started_at is None),
                    key=lambda r: (r.created_at, r.machine))
    pairs = []
    for request in queued:
        m = request.machine
        candidates = [j for j in available if config.service_time[m][j] > 0]
        if candidates:
            j = min(candidates, key=lambda j: (config.service_time[m][j], j))
            pairs.append((m, j))
            available.remove(j)
    return tuple(sorted(pairs))


def risk_matching(config, state, time):
    def score(action):
        total = 0.
        for m, j in action:
            p = state.physical
            hazard = config.failure_probability if not p.failed[m] and p.ages[m] + 1 >= config.failure_age else 0.
            wait = time - state.requests[m].created_at
            urgency = 100 * p.failed[m] + 10 * hazard + p.ages[m] / config.max_age + .1 * min(wait, 20)
            total += (urgency + config.restoration[m][j]) / config.service_time[m][j]
        return total
    return max(legal_matchings(config, state), key=score)


def choose(config, state, time, requester, dispatcher, *, policy_horizon,
           forecast_seed, scenarios, forecast_horizon, check_timeout=lambda: None):
    """No environment object/future tape reaches a controller.

    Rollout improves only the current matching on this requester's admitted
    pool. It is a bounded-window known-model reference, not an optimal oracle
    or a joint optimizer over all possible future request policies.
    """
    if dispatcher not in DISPATCHERS:
        raise ValueError("unknown dispatcher")
    submissions = request_action(config, state, time, requester)
    admitted = submit_requests(config, state, time, submissions)
    if dispatcher == "fcfs":
        return submissions, fcfs(config, admitted, time), dict(forecast_transitions=0, root_scores=[])
    baseline = risk_matching(config, admitted, time)
    actions = legal_matchings(config, admitted)
    if dispatcher == "risk_matching" or len(actions) == 1:
        return submissions, baseline, dict(forecast_transitions=0, root_scores=[])
    depth = min(forecast_horizon, config.horizon - time)
    rng = random.Random(forecast_seed)
    bank = tuple(tuple(tuple(rng.random() < config.failure_probability
                             for _ in range(config.machines)) for _ in range(depth))
                 for _ in range(scenarios))
    scores = []
    # Put baseline first and retain it on numerical ties.
    roots = (baseline, *(a for a in actions if a != baseline))
    best, best_score = baseline, float("inf")
    for root in roots:
        check_timeout()
        total = 0.
        for tape in bank:
            simulated = state
            for offset, events in enumerate(tape):
                t = time + offset
                if offset == 0:
                    submit, action = submissions, root
                else:
                    rule = requester if t < policy_horizon else "threshold"
                    submit = request_action(config, simulated, t, rule)
                    pending = submit_requests(config, simulated, t, submit)
                    action = risk_matching(config, pending, t)
                simulated, cost, _ = transition(config, simulated, t, submit, action, events)
                total += cost
        value = total / scenarios
        scores.append(dict(matching=root, expected_cost=value))
        if value < best_score - 1e-10:
            best, best_score = root, value
    return submissions, best, dict(forecast_transitions=len(roots) * scenarios * depth,
                                   root_scores=scores)
