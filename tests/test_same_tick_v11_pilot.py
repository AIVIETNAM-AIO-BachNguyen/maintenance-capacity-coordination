"""V11 pilot identity, same-weight ablation and gate wiring checks."""
import torch

from experiments import audit_same_tick_v11 as audit
from experiments import same_tick_pilot_v11 as pilot
from experiments.scaled_n5k3_v11_config import CASES, configs
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v11 import SequencePolicy


def test_pilot_seed_registry_and_all_arm_inputs():
    pilot.seed_registry_check()
    assert pilot.ARMS == ("C11", "D11", "B11", "S11", "G11", "Q11")
    for case in CASES:
        env = ScaledCoordinationEnv(configs()[case])
        env.reset(pilot.SMOKE_SEEDS[0])
        for arm in pilot.ARMS:
            choice = pilot.choose(pilot.new_policy(arm), env, False)
            assert env.matching_mask(choice["submissions"])[choice["matching_index"]]
            if arm == "Q11":
                assert choice["submissions"] == env.fixed_submissions("D")
            else:
                assert choice["submissions"] == env.fixed_submissions("threshold")


def test_same_weight_frozen_context_evaluation_identity(tmp_path):
    case = CASES[0]
    active = SequencePolicy("C11")
    frozen = SequencePolicy("D11")
    frozen.load_state_dict(active.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(active.state_dict().values(), frozen.state_dict().values()))
    rows_c = pilot.evaluate_policy(active, "C11", case, (pilot.SMOKE_SEEDS[0],),
                                   trace_path=tmp_path / "active.jsonl")
    rows_f = pilot.evaluate_policy(frozen, "CF11", case, (pilot.SMOKE_SEEDS[0],),
                                   trace_path=tmp_path / "frozen.jsonl")
    assert len(rows_c) == len(rows_f) == 1
    assert rows_c[0]["physical_seed"] == rows_f[0]["physical_seed"]
    assert sum(1 for _ in (tmp_path / "active.jsonl").open()) == 48
    assert sum(1 for _ in (tmp_path / "frozen.jsonl").open()) == 48
    audit.episode_accounting(rows_c[0]); audit.episode_accounting(rows_f[0])


def test_locked_gate_requires_both_trained_and_same_weight_feedback():
    arms = ("C11", "D11", "B11", "S11", "G11", "Q11", "CF11")
    costs = {"C11": 100., "D11": 130., "B11": 120., "S11": 120.,
             "G11": 120., "Q11": 120., "CF11": 115.}
    rows = [{"seed": seed, "case": case, "arm": arm,
             "checkpoint": f"{root}:2048", "objective": costs[arm]}
            for seed in pilot.HELDOUT_SEEDS for case in CASES
            for root in pilot.TRAIN_ROOTS for arm in arms]
    assert audit.gate(rows, "heldout")["go"]
    for row in rows:
        if row["arm"] == "CF11":
            row["objective"] = 100.
    decision = audit.gate(rows, "heldout")
    assert not decision["go"]
    assert not decision["conditions"]["C11_minus_CF11_at_most_minus_10"]
    assert not decision["conditions"]["C11_minus_CF11_both_cases_negative"]
