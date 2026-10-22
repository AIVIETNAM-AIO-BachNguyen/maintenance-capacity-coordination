"""Mechanism and protocol checks; synthetic rows are not research evidence."""
import math

import pytest

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import RequestEnv, submit_requests
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from experiments.pair_commitment_interaction_v5 import (
    CASES, HORIZON, SEEDS, additive_decision, analyze, compare_state,
    forecast_seed, holdout_seed, scenario_bank, seed_registry_check,
)


def row(action, cost):
    return {"matching": action, "expected_cost": cost}


def test_negative_joint_interaction_changes_complete_matching_choice():
    roots = [row((), 100.), row(((0, 0),), 94.), row(((1, 1),), 95.),
             row(((0, 0), (1, 1)), 70.)]
    choice, interactions = additive_decision(roots)
    assert choice == ((0, 0), (1, 1))
    assert interactions == [-19.]


def test_positive_joint_interaction_and_ties_use_root_order():
    roots = [row(((0, 0),), 90.), row((), 100.), row(((1, 1),), 90.),
             row(((0, 0), (1, 1)), 110.)]
    choice, interactions = additive_decision(roots)
    assert choice == ((0, 0), (1, 1))  # additive 80, despite true forecast 110
    assert interactions == [30.]
    tie = [row(((1, 1),), 90.), row((), 100.), row(((0, 0),), 90.)]
    assert additive_decision(tie)[0] == ((1, 1),)


def test_pair_requires_both_singleton_roots():
    with pytest.raises(RuntimeError, match="singleton"):
        additive_decision([row((), 100.), row(((0, 0),), 90.),
                           row(((0, 0), (1, 1)), 80.)])


def test_seed_roles_and_independent_common_bank():
    assert seed_registry_check()
    case, raw, tick = CASES[0], SEEDS[0], 9
    assert forecast_seed(case, raw, tick) != holdout_seed(case, raw, tick)
    assert holdout_seed(case, raw, tick) != role_seed(raw, "maintenance_framework_v1:physical")
    cfg = configs()[case]
    assert scenario_bank(cfg, holdout_seed(case, raw, tick)) == scenario_bank(
        cfg, holdout_seed(case, raw, tick))
    assert scenario_bank(cfg, holdout_seed(case, raw, tick)) != scenario_bank(
        cfg, forecast_seed(case, raw, tick))


def test_single_legal_action_never_uses_holdout_or_future_tape():
    cfg = configs()[CASES[0]]
    env = RequestEnv(cfg)
    env.reset(role_seed(SEEDS[0], "maintenance_framework_v1:physical"),
              initial=DispatchState((0,) * 4, (False,) * 4, (0,) * 4,
                                    (0, 0), (-1, -1)))
    state = env.state
    submissions = request_action(cfg, state, 0, "threshold")
    admitted = submit_requests(cfg, state, 0, submissions)
    assert submissions == () and all(r is None for r in admitted.requests)
    result = compare_state(cfg, state, 0, CASES[0], SEEDS[0], (), (),
                           {"root_scores": [], "forecast_transitions": 0})
    assert result["legal_root_count"] == 1
    assert result["holdout_forecast_transitions"] == 0
    assert result["holdout_bank_sha256"] is None
    assert result["holdout_difference"] == 0.


def test_analysis_keeps_shared_seed_clusters_and_conditional_denominator():
    rows = []
    for case in CASES:
        for seed in SEEDS:
            for tick in range(HORIZON):
                eligible = tick < 6 if case == CASES[0] else tick == 0
                difference = 3. if eligible else 0.
                rows.append({"case": case, "raw_seed": seed, "tick": tick,
                             "pair_eligible": eligible, "holdout_difference": difference,
                             "additive_matching": ((0, 0),),
                             "f2_matching": ((1, 1),) if eligible else ((0, 0),),
                             "pair_interactions": [-4.] if eligible else []})
    result = analyze(rows)
    reciprocal = result["cases"][CASES[0]]
    assert reciprocal["pair_frequency"] == 6 / HORIZON
    assert reciprocal["conditional_mean"] == 3.
    assert math.isclose(reciprocal["all_state_mean"], 3 * 6 / HORIZON)
    assert reciprocal["physical_cluster_95"] == [3., 3.]
    assert result["go"]
