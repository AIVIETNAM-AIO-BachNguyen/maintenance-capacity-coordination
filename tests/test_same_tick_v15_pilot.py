"""V15 pilot identity, same-weight ablation and gate wiring checks."""
import torch

from experiments import audit_same_tick_v15 as audit
from experiments import same_tick_pilot_v15 as pilot
from experiments.scaled_n5k3_v12_config import CASES, configs, evaluate_controls
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v15 import SequencePolicy


def test_pilot_seed_registry_and_all_arm_inputs():
    pilot.seed_registry_check()
    assert pilot.ARMS == ("C15", "D15", "B15", "S15", "G15", "P15", "Q15")
    for case in CASES:
        env = ScaledCoordinationEnv(configs()[case])
        env.reset(pilot.SMOKE_SEEDS[0])
        for arm in pilot.ARMS:
            choice = pilot.choose(pilot.new_policy(arm), env, False)
            assert env.matching_mask(choice["submissions"])[choice["matching_index"]]
            if arm == "Q15":
                assert choice["submissions"] == env.fixed_submissions("D")
            else:
                assert choice["submissions"] == env.fixed_submissions("threshold")


def test_same_weight_frozen_context_evaluation_identity(tmp_path):
    case = CASES[0]
    active = SequencePolicy("C15")
    frozen = SequencePolicy("D15")
    frozen.load_state_dict(active.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(active.state_dict().values(), frozen.state_dict().values()))
    rows_c = pilot.evaluate_policy(active, "C15", case, (pilot.SMOKE_SEEDS[0],),
                                   trace_path=tmp_path / "active.jsonl")
    rows_f = pilot.evaluate_policy(frozen, "CF15", case, (pilot.SMOKE_SEEDS[0],),
                                   trace_path=tmp_path / "frozen.jsonl")
    assert len(rows_c) == len(rows_f) == 1
    assert rows_c[0]["physical_seed"] == rows_f[0]["physical_seed"]
    assert sum(1 for _ in (tmp_path / "active.jsonl").open()) == 48
    assert sum(1 for _ in (tmp_path / "frozen.jsonl").open()) == 48
    audit.episode_accounting(rows_c[0]); audit.episode_accounting(rows_f[0])


def test_locked_gate_requires_both_trained_and_same_weight_feedback():
    arms = ("C15", "D15", "B15", "P15", "S15", "G15", "Q15", "CF15")
    costs = {"C15": 100., "D15": 130., "B15": 120., "S15": 120.,
             "G15": 120., "P15": 120., "Q15": 120., "CF15": 115.}
    rows = [{"seed": seed, "case": case, "arm": arm,
             "checkpoint": f"{root}:2048", "objective": costs[arm]}
            for seed in pilot.HELDOUT_SEEDS for case in CASES
            for root in pilot.TRAIN_ROOTS for arm in arms]
    assert audit.gate(rows, "heldout")["go"]
    for row in rows:
        if row["arm"] == "CF15":
            row["objective"] = 100.
    decision = audit.gate(rows, "heldout")
    assert not decision["go"]
    assert not decision["conditions"]["C15_minus_CF15_at_most_minus_10"]
    assert not decision["conditions"]["C15_minus_CF15_both_cases_negative"]
    for row in rows:
        if row["arm"] == "CF15":
            row["objective"] = costs["CF15"]
        if row["arm"] == "P15":
            row["objective"] = 95.
    no_pretrain_gain = audit.gate(rows, "heldout")
    assert not no_pretrain_gain["go"]
    assert not no_pretrain_gain["conditions"]["C15_minus_P15_at_most_minus_10"]
    assert not no_pretrain_gain["conditions"]["C15_minus_P15_both_cases_negative"]


def test_control_wrapper_covers_both_cases_and_accounting(tmp_path):
    rows = evaluate_controls(tmp_path, (pilot.SMOKE_SEEDS[0],), panel="smoke")
    assert len(rows) == 4
    assert {(r["case"], r["arm"]) for r in rows} == {
        (case, "F_" + control) for case in CASES for control in ("risk_matching", "rollout")}
    for row in rows:
        audit.episode_accounting(row)
        assert row["invalid_assignments"] == 0
