from dataclasses import asdict, replace
import itertools

import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchConfig, DispatchState
from ht_pdm_fjsp.maintenance_requests import (
    RequestEnv, initial_state, legal_matchings, observation, queue_snapshot,
    state_from_dict, submit_requests, transition, validate_state,
)
from ht_pdm_fjsp.maintenance_request_policies import (
    choose, family_configs, fcfs, request_action, risk_matching,
)


def config(n=1, k=1, **kwargs):
    return replace(DispatchConfig(n, k, 8, 3, 1., ((3,) * k,) * n,
                                  ((.5,) * k,) * n), **kwargs)


def physical(ages=(2,), failed=(False,), remaining=(0,), assigned=(-1,)):
    return DispatchState(ages, failed, (0,) * len(ages), remaining, assigned)


def test_pending_pm_operates_upgrades_preserves_timestamp_then_restores_at_completion():
    cfg = config()
    state = initial_state(cfg, physical())
    state, cost, info = transition(cfg, state, 0, (0,), (), (False,))
    assert cost == 0 and state.physical.ages == (3,)
    assert state.requests[0].created_at == 0
    assert info["queued_pm_ticks"] == 1 and info["max_wait_age"] == 1
    state, cost, info = transition(cfg, state, 1, (), (), (True,))
    assert cost == 15 and state.physical.ages == (4,)
    request = state.requests[0]
    assert (request.kind, request.created_at, request.upgraded_at) == ("corrective", 0, 2)
    assert info["requests_created"] == 0 and info["requests_upgraded"] == 1
    state, cost, info = transition(cfg, state, 2, (), ((0, 0),), (True,))
    assert cost == 8 and state.physical.ages == (4,)
    assert state.physical.failed == (True,) and state.physical.remaining == (2,)
    assert info["request_wait_ticks"] == 0 and info["max_wait_age"] == 2
    state, cost, _ = transition(cfg, state, 3, (), (), (True,))
    assert cost == 6 and state.physical.ages == (4,)
    state, cost, info = transition(cfg, state, 4, (), (), (True,))
    assert cost == 6 and state.physical.ages == (2,)
    assert state.requests == (None,) and state.physical.failed == (False,)
    assert info["completed_wait_total"] == 2
    assert info["requests_completed"] == info["completed_jobs"] == 1


def test_failure_without_pm_auto_creates_cm_next_boundary():
    cfg = config()
    state, cost, info = transition(cfg, initial_state(cfg, physical()), 0, (), (), (True,))
    assert cost == 15
    assert state.requests[0].created_at == 1 and state.requests[0].kind == "corrective"
    assert info["request_wait_ticks"] == 0 and info["max_wait_age"] == 0


def test_shared_queue_can_use_other_eligible_worker_after_wait():
    cfg = config(n=2, k=2, service_time=((3, 1), (2, 0)))
    state = initial_state(cfg, physical((1, 1), (False, False), (0, 0), (-1, -1)))
    state, _, _ = transition(cfg, state, 0, (0, 1), ((1, 0),), (False, False))
    assert state.requests[0].technician is None
    assert ((0, 1),) in legal_matchings(cfg, state)
    state, _, info = transition(cfg, state, 1, (), ((0, 1),), (False, False))
    assert state.requests[0] is None and state.requests[1] is None
    assert info["requests_completed"] == 2


def test_fcfs_matching_does_not_bind_unserved_request_to_a_worker():
    cfg = config(n=2, k=2, service_time=((1, 4), (1, 0)))
    state = initial_state(cfg, physical((1, 1), (False, False), (0, 0), (-1, -1)))
    state = submit_requests(cfg, state, 0, (0, 1))
    assert fcfs(cfg, state, 0) == ((0, 0),)
    assert ((0, 1), (1, 0)) in legal_matchings(cfg, state)


@pytest.mark.parametrize("submissions,matching", [
    ((0, 0), ()), ((-1,), ()), ((True,), ()), ((), ((0, 0),)),
    ((0,), ((0, 0), (0, 0))), ((0,), ((0, True),)),
])
def test_invalid_actions_leave_environment_unchanged(submissions, matching):
    env = RequestEnv(config())
    env.reset(2, physical())
    before, metrics = env.state, dict(env.metrics)
    with pytest.raises(ValueError):
        env.step(submissions, matching)
    assert env.state == before and env.time == 0 and env.metrics == metrics


def test_no_duplicate_request_while_queued_or_servicing():
    cfg = config()
    state = initial_state(cfg, physical())
    queued, _, _ = transition(cfg, state, 0, (0,), (), (False,))
    busy, _, _ = transition(cfg, state, 0, (0,), ((0, 0),), (True,))
    for existing in (queued, busy):
        with pytest.raises(ValueError, match="active request"):
            transition(cfg, existing, 1, (0,), (), (False,))


def test_new_generation_after_completed_request_and_no_premature_restore():
    cfg = config(service_time=((1,),), restoration=((1.,),))
    state = initial_state(cfg, physical())
    state, cost, info = transition(cfg, state, 0, (0,), ((0, 0),), (True,))
    assert cost == 7 and state.physical.ages == (0,) and info["failures"] == 0
    assert state.requests == (None,) and state.generations == (1,)
    state, _, _ = transition(cfg, state, 1, (0,), (), (False,))
    assert state.requests[0].sequence == 2 and state.requests[0].created_at == 1


def test_unfinished_requests_are_in_wait_exposure_and_terminal_snapshot():
    env = RequestEnv(config(horizon=2, failure_probability=0.))
    env.reset(2, physical())
    env.step((0,), ())
    _, reward, done, _ = env.step()
    assert reward == 0 and done
    assert env.metrics["request_wait_ticks"] == 2
    assert env.metrics["requests_completed"] == 0
    assert queue_snapshot(env.state, env.time) == dict(pending=1, pending_pm=1,
        pending_cm=0, in_service=0, pending_wait_age_sum=2, oldest_pending_age=2)
    with pytest.raises(RuntimeError, match="finished"):
        env.step()


def test_initial_failed_request_and_state_validation():
    cfg = config()
    state = initial_state(cfg, physical(failed=(True,)))
    assert state.requests[0].kind == "corrective"
    with pytest.raises(ValueError, match="without request"):
        validate_state(cfg, replace(state, requests=(None,)), 0)
    assert state_from_dict(asdict(state)) == state


def test_request_observation_contains_no_realized_future_shocks():
    cfg = config()
    env = RequestEnv(cfg)
    obs = env.reset(3)
    assert set(obs) == {"time", "state", "physical"}
    env2 = RequestEnv(cfg)
    env2.reset(3)
    assert env._events == env2._events and env.state == env2.state


def test_resource_aware_rule_submits_earlier_when_only_eligible_worker_busy():
    cfg = config(n=2, k=1, failure_age=4)
    initial = initial_state(cfg, physical((2, 2), (False, False)))
    busy, _, _ = transition(cfg, initial, 0, (0,), ((0, 0),), (False, False))
    busy = replace(busy, physical=replace(busy.physical, ages=(2, 1)))
    assert request_action(cfg, busy, 1, "threshold") == ()
    assert request_action(cfg, busy, 1, "resource_aware") == (1,)


def test_rollout_is_deterministic_legal_and_does_not_read_real_tape():
    cfg = config(n=2, k=2, service_time=((1, 3), (2, 1)))
    state = initial_state(cfg, physical((2, 2), (False, False), (0, 0), (-1, -1)))
    kwargs = dict(policy_horizon=5, forecast_seed=123, scenarios=3, forecast_horizon=3)
    actual = choose(cfg, state, 0, "threshold", "rollout", **kwargs)
    assert actual == choose(cfg, state, 0, "threshold", "rollout", **kwargs)
    sub, matching, info = actual
    assert matching in legal_matchings(cfg, submit_requests(cfg, state, 0, sub))
    assert info["forecast_transitions"] > 0
    assert len(info["root_scores"]) == len(legal_matchings(cfg, submit_requests(cfg, state, 0, sub)))


def test_all_legal_matchings_match_exhaustive_machine_proposals():
    cfg = config(n=3, k=2, service_time=((1, 0), (1, 2), (2, 1)))
    state = initial_state(cfg, physical((2,) * 3, (False,) * 3, (0, 0), (-1, -1)))
    state = submit_requests(cfg, state, 0, (0, 1, 2))
    expected = set()
    for proposal in itertools.product(range(3), repeat=3):
        pairs = tuple((m, j - 1) for m, j in enumerate(proposal) if j)
        workers = [j for m, j in pairs]
        if len(workers) == len(set(workers)) and all(cfg.service_time[m][j] > 0 for m, j in pairs):
            expected.add(pairs)
    assert set(legal_matchings(cfg, state)) == expected


def test_randomized_episode_conserves_requests_and_accounts_all_costs():
    cfg = family_configs("smoke")["specialist_generalist_scale2"]
    for seed in range(8):
        env = RequestEnv(cfg)
        env.reset(seed)
        total_reward = 0
        while env.time < cfg.horizon:
            sub = request_action(cfg, env.state, env.time, "resource_aware")
            matching = risk_matching(cfg, submit_requests(cfg, env.state, env.time, sub), env.time)
            _, reward, _, _ = env.step(sub, matching)
            total_reward += reward
            assert env.metrics["requests_created"] - env.metrics["requests_completed"] == sum(r is not None for r in env.state.requests)
            assert env.metrics["request_wait_ticks"] == env.metrics["queued_pm_ticks"] + env.metrics["queued_cm_ticks"]
        assert total_reward == -env.metrics["objective"]
