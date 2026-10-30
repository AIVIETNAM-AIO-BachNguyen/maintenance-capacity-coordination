"""Engineering checks for the prelocked v2 candidate-action diagnostic."""
from __future__ import annotations

from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import VERSION as BASE_VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose, request_action
from ht_pdm_fjsp.maintenance_requests import RequestEnv, initial_state, transition
from experiments.commitment_headroom_v2 import (
    assert_seed_separation, compare_state, expanded_options, forecast_seed,
    holdout_seed, scenario_bank,
)


def test_extra_candidate_serves_through_real_request_and_wait_is_legal():
    config = configs()["reciprocal_scale2"]
    env = RequestEnv(config)
    physical = DispatchState((0, 0, 0, 0), (False,) * 4, (0,) * 4,
                             (0, 0), (-1, -1))
    env.reset(role_seed(73100031, BASE_VERSION + ":physical"), initial=physical)
    state = env.state
    threshold, ordinary, expanded, submissions_for = expanded_options(config, state, 0)
    assert threshold == () and ordinary == {()}
    assert () in expanded and len(expanded) > 1
    matching = next(action for action in sorted(expanded) if len(action) == 1 and action[0][0] == 0)
    submissions, early = submissions_for(matching)
    assert early == (0,) and submissions == (0,)
    env.step(submissions, matching)
    assert env.metrics["service_starts"] == 1
    assert env.metrics["requests_created"] == 1
    assert env.metrics["objective"] == sum(env.metrics[k] for k in
                                            ("maintenance_cost", "unavailability_cost", "failure_cost"))


def test_threshold_only_root_exactly_matches_old_transition_and_no_duplicate():
    config = configs()["reciprocal_scale2"]
    seed = role_seed(73100032, BASE_VERSION + ":physical")
    old, new = RequestEnv(config), RequestEnv(config)
    old.reset(seed)
    new.reset(seed)
    for tick in range(4):
        threshold, ordinary, expanded, submissions_for = expanded_options(config, new.state, tick)
        assert threshold == request_action(config, old.state, tick, "threshold")
        matching = sorted(ordinary)[-1]
        selected_submissions, early = submissions_for(matching)
        assert selected_submissions == threshold and early == ()
        assert matching in expanded
        old.step(threshold, matching)
        new.step(selected_submissions, matching)
        assert old.state == new.state and old.metrics == new.metrics


def test_failed_machine_still_gets_cm_without_pm_submission():
    config = configs()["reciprocal_scale2"]
    physical = DispatchState((3, 0, 0, 0), (False,) * 4, (0,) * 4,
                             (0, 0), (-1, -1))
    state = initial_state(config, physical)
    following, _, _ = transition(config, state, 0, (), (), (True, False, False, False))
    assert following.physical.failed[0]
    assert following.requests[0].kind == "corrective"
    assert () in expanded_options(config, following, 1)[2]


def test_forecast_banks_are_reproducible_separate_and_no_future_tape_input():
    config = configs()["reciprocal_scale2"]
    case, seed, tick = "reciprocal_scale2", 73170000, 0
    selection = forecast_seed(case, seed, tick)
    holdout = holdout_seed(case, seed, tick)
    assert selection != holdout
    assert scenario_bank(config, selection, 6, 16) == scenario_bank(config, selection, 6, 16)
    assert scenario_bank(config, selection, 6, 16) != scenario_bank(config, holdout, 6, 16)
    assert_seed_separation((case,), (seed,), (tick,))
    env = RequestEnv(config)
    env.reset(role_seed(seed, BASE_VERSION + ":physical"))
    baseline = choose(config, env.state, tick, "threshold", "rollout",
                      policy_horizon=48, forecast_seed=selection,
                      scenarios=16, forecast_horizon=6)
    row = compare_state(config, env.state, tick, case, seed, baseline[:2], baseline[2])
    assert row["selection_forecast_seed"] == selection
    assert row["holdout_forecast_seed"] == holdout
    assert row["expanded_selection_cost"] <= row["baseline_selection_cost"] + 1e-8
    assert row["ordinary_root_count"] + row["extra_root_count"] > 1
