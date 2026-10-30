"""Pilot gate, label provenance and all-arm physical action contracts."""
import torch

from experiments import audit_same_tick_v16 as audit
from experiments import same_tick_pilot_v16 as pilot
from experiments.scaled_n5k3_v12_config import CASES, configs
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv


def test_locked_label_bank_and_all_arm_actions():
    pilot.seed_registry_check()
    assert pilot.sha256(pilot.LABEL_RUN / "manifest.json") == pilot.LABEL_MANIFEST_SHA256
    assert pilot.sha256(pilot.LABEL_RUN / "audit.json") == pilot.LABEL_AUDIT_SHA256
    assert pilot.sha256(pilot.LABEL_RUN / "labels.jsonl") == pilot.LABELS_SHA256
    assert pilot.ARMS == ("C16", "D16", "I16", "R16", "B16", "S16", "G16", "P16", "Q16")
    for case in CASES:
        env = ScaledCoordinationEnv(configs()[case])
        env.reset(pilot.SMOKE_SEEDS[0])
        for arm in pilot.ARMS:
            decision = pilot.choose(pilot.new_policy(arm), env, True)
            assert env.matching_mask(decision["submissions"])[decision["matching_index"]]
            assert decision["submissions"] == env.fixed_submissions("D" if arm == "Q16" else "threshold")


def test_cf16_same_weights_and_heldout_gate_requires_relation_and_feedback():
    c, frozen = DynamicSequencePolicy("C16"), DynamicSequencePolicy("D16")
    frozen.load_state_dict(c.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(c.state_dict().values(), frozen.state_dict().values()))
    costs = {"C16": 100., "D16": 130., "CF16": 120., "I16": 120., "R16": 120.,
             "B16": 120., "S16": 120., "G16": 120., "P16": 120., "Q16": 120.}
    rows = [{"seed": seed, "case": case, "arm": arm,
             "checkpoint": f"{root}:2048", "objective": costs[arm]}
            for seed in pilot.HELDOUT_SEEDS for case in CASES
            for root in pilot.TRAIN_ROOTS for arm in costs]
    assert audit.gate(rows, "heldout")["go"]
    for comparator, condition in (("D16", "C16_minus_D16_at_most_minus_20"),
                                  ("CF16", "C16_minus_CF16_at_most_minus_10"),
                                  ("I16", "C16_minus_I16_at_most_minus_10"),
                                  ("R16", "C16_minus_R16_at_most_minus_10")):
        changed = [dict(row, objective=100.) if row["arm"] == comparator else row for row in rows]
        result = audit.gate(changed, "heldout")
        assert not result["go"]
        assert not result["conditions"][condition]


def test_gate_cluster_is_raw_seed_across_cases_and_roots():
    seeds = pilot.HELDOUT_SEEDS
    rows = [{"seed": seed, "case": case, "arm": arm,
             "checkpoint": f"{root}:2048",
             "objective": 100. if arm == "C16" else 130.}
            for seed in seeds for case in CASES for root in pilot.TRAIN_ROOTS
            for arm in ("C16", "D16")]
    contrast = audit.clustered_contrast(rows, "C16", "D16", seeds, pilot.BOOTSTRAP_SEED)
    assert contrast["overall"] == -30.
    assert contrast["physical_cluster_95_interval"] == [-30., -30.]
    assert set(contrast["training_root_means"]) == {str(root) for root in pilot.TRAIN_ROOTS}
