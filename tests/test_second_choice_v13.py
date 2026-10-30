"""V13 engineering contracts for fixed-state second-choice value comparisons."""
import time

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v12 import STOP, SequencePolicy
from experiments.scaled_n5k3_v12_config import CASES, configs
from experiments import same_tick_pilot_v12 as pilot
from experiments import second_choice_value_v13 as study


def test_second_choices_and_bank_separation():
    mask = np.zeros(STOP, dtype=bool)
    mask[[0, 4, 5, 7]] = True
    first, seconds = study.second_choices({"tokens": (0,), "edge_mask": mask})
    assert first == 0 and seconds == (4, 5, 7)
    assert study.second_choices({"tokens": (STOP,), "edge_mask": mask}) == (None, ())
    study.seed_registry_check()
    seeds = [study.bank_seed(75711000, CASES[0], pilot.TRAIN_ROOTS[0], 12, slot)
             for slot in range(8)]
    assert len(set(seeds)) == 8


def test_branch_same_state_and_tape_replays_exactly():
    cfg = configs()[CASES[0]]
    env = ScaledCoordinationEnv(cfg)
    env.reset(study.SMOKE_SEEDS[0])
    policy = SequencePolicy("C12")
    generator = DispatchEnv(cfg)
    generator.reset(study.bank_seed(study.SMOKE_SEEDS[0], CASES[0], pilot.TRAIN_ROOTS[0], 8, 0))
    state_before, tick_before = env.inner.state, env.tick
    first = study.continuation_cost(env, policy, 0, env.fixed_submissions("threshold"),
                                    generator.events, time.monotonic() + 30)
    second = study.continuation_cost(env, policy, 0, env.fixed_submissions("threshold"),
                                     generator.events, time.monotonic() + 30)
    assert first == second
    assert abs(first[0] - first[1] - first[2]) < 1e-8
    assert env.inner.state == state_before and env.tick == tick_before


def test_locked_signal_gate_uses_independent_bank_validation():
    rows = [{"case": case, "root": root, "raw_seed": seed,
             "validation_gap": 12., "bank_a_spread": 14., "bank_b_spread": 13.}
            for case in CASES for root in pilot.TRAIN_ROOTS for seed in study.SCIENCE_SEEDS]
    coverage = [{}] * len(rows)
    summary = study.summary_and_gate(rows, coverage)
    assert summary["go"]
    assert summary["bootstrap_draws_defined"] == 5000
    for row in rows:
        if row["case"] == CASES[1]:
            row["validation_gap"] = 0.
    failed = study.summary_and_gate(rows, coverage)
    assert not failed["go"]
    assert not failed["conditions"]["both_case_validation_means_at_least_10"]
