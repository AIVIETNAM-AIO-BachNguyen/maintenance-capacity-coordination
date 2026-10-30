"""Independent, read-only audit and fixed-protocol summary of pilot v1 TEST.

Writes audit and analysis beside the completed raw TEST files. It never edits
episode data, checkpoints, the runner, or the locked protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION


CASES = ("reciprocal_scale2", "specialist_generalist_scale2")
ARMS = ("B", "C", "D", "E")
TRAIN_SEEDS = (73110000, 73110001, 73110002)
HISTORIC_SEEDS = (72600000, 72600001, 72600002, 72600003, 72600004)
TEST_SEEDS = tuple(range(73140000, 73140032))
BOOTSTRAP_SEED = 73150000
BOOTSTRAP_DRAWS = 5000


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def audit_and_analyze(training_run: Path, replication_run: Path, test_run: Path):
    training_run = training_run.resolve()
    replication_run = replication_run.resolve()
    test_run = test_run.resolve()
    train = load_json(training_run / "manifest.json")
    replication = load_json(replication_run / "manifest.json")
    test = load_json(test_run / "manifest.json")
    assert (train["kind"], replication["kind"], test["kind"]) == ("train", "replicate", "test")
    assert all(m["status"] == "COMPLETED" for m in (train, replication, test))
    assert train["protocol_sha256"] == replication["protocol_sha256"] == test["protocol_sha256"]
    assert train["source_sha256"] == replication["source_sha256"] == test["source_sha256"]
    assert train["config_sha256"] == replication["config_sha256"] == test["config_sha256"]
    assert test["training_run"] == str(training_run)
    assert replication["training_run"] == str(training_run)
    assert load_json(replication_run / "test_go_decision.json")["go"] is True
    assert load_json(replication_run / "audit.json")["status"] == "PASS"
    assert load_json(training_run / "audit.json")["status"] == "PASS"
    assert set(TEST_SEEDS).isdisjoint(train["dev_seeds"])
    assert set(TEST_SEEDS).isdisjoint(train["replication_seeds"])
    assert tuple(test["sealed_test_seeds"]) == TEST_SEEDS
    for name, digest in test["output_sha256"].items():
        assert sha256(test_run / name) == digest, name

    learned = load_jsonl(test_run / "test_episodes.jsonl")
    controls = load_jsonl(test_run / "test_controls.jsonl")
    assert len(learned) == test["learned_episodes"] == 768
    assert len(controls) == test["control_episodes"] == 448

    expected_learned = {(case, arm, seed, physical)
                        for case in CASES for arm in ARMS
                        for seed in TRAIN_SEEDS for physical in TEST_SEEDS}
    expected_controls = {(case, arm, historic, physical)
                         for case in CASES for physical in TEST_SEEDS
                         for arm, historic in
                         (("F_risk_matching", None), ("F_rollout", None),
                          *(("A", seed) for seed in HISTORIC_SEEDS))}
    learned_keys = []
    control_keys = []
    checkpoint_sha256 = {}
    for case in CASES:
        for arm in ARMS:
            for seed in TRAIN_SEEDS:
                trial = training_run / f"{case}_{arm}_{seed}"
                selection = load_json(trial / "selection.json")
                checkpoint = trial / selection["checkpoint"]
                assert checkpoint.exists()
                checkpoint_sha256[f"{case}:{arm}:{seed}:{selection['transition']}"] = sha256(checkpoint)

    for row in learned + controls:
        assert row["case"] in CASES and row["seed"] in TEST_SEEDS
        assert row["physical_seed"] == role_seed(row["seed"], BASE_VERSION + ":physical")
        assert row["invalid_assignments"] == 0
        assert abs(row["objective"] - row["H_cost"] - row["tail_cost"]) < 1e-8
        assert abs(row["episode_return"] + row["objective"]) < 1e-8
        assert abs(row["objective"] - row["failure_cost"] - row["maintenance_cost"]
                   - row["unavailability_cost"]) < 1e-8
        assert row["H_availability"] >= 0 and row["H_availability"] <= 1
        assert row["service_starts"] >= 0 and row["failures"] >= 0
        if row in learned:
            arm = row["arm"]
            seed, transition = map(int, row["checkpoint"].split(":"))
            assert arm in ARMS and seed in TRAIN_SEEDS
            assert f"{row['case']}:{arm}:{seed}:{transition}" in checkpoint_sha256
            learned_keys.append((row["case"], arm, seed, row["seed"]))
        else:
            assert row["panel"] == "test"
            historic = row.get("historical_train_seed")
            control_keys.append((row["case"], row["arm"], historic, row["seed"]))
    assert set(learned_keys) == expected_learned and max(Counter(learned_keys).values()) == 1
    assert set(control_keys) == expected_controls and max(Counter(control_keys).values()) == 1

    audit = {
        "status": "PASS", "protocol_sha256": test["protocol_sha256"],
        "source_config_protocol_match": True, "replication_gate_and_prior_audits": "PASS",
        "raw_output_sha256_match": True, "panel_coverage": "PASS",
        "physical_seed_pairing": "PASS", "selected_checkpoint_pairing": "PASS",
        "feasibility_and_H_tail_cost": "PASS", "seed_roles": "DISJOINT",
        "learned_episodes": len(learned), "control_episodes": len(controls),
        "selected_checkpoint_sha256": checkpoint_sha256,
    }
    write_json(test_run / "audit.json", audit)

    keyed = {(r["case"], r["arm"], int(r["checkpoint"].split(":")[0]), r["seed"]): r
             for r in learned}
    comparisons = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for other in ("B", "D", "E"):
        differences = np.asarray([[[keyed[case, "C", training_seed, physical]["objective"]
                                    - keyed[case, other, training_seed, physical]["objective"]
                                    for physical in TEST_SEEDS]
                                   for case in CASES] for training_seed in TRAIN_SEEDS], dtype=float)
        draws = np.empty(BOOTSTRAP_DRAWS)
        for index in range(BOOTSTRAP_DRAWS):
            sampled = rng.integers(0, len(TEST_SEEDS),
                                   size=(len(TRAIN_SEEDS), len(CASES), len(TEST_SEEDS)))
            draws[index] = np.take_along_axis(differences, sampled, axis=2).mean()
        comparisons[other] = {
            "mean": float(differences.mean()),
            "by_case": {case: float(differences[:, ci, :].mean()) for ci, case in enumerate(CASES)},
            "by_training_seed": {str(seed): float(differences[si].mean())
                                 for si, seed in enumerate(TRAIN_SEEDS)},
            "physical_seed_bootstrap_95": np.quantile(draws, (0.025, 0.975)).tolist(),
        }
    arms = {}
    for case in CASES:
        arms[case] = {}
        for arm in ARMS:
            selected = [r for r in learned if r["case"] == case and r["arm"] == arm]
            arms[case][arm] = {metric: float(np.mean([r[metric] for r in selected]))
                               for metric in ("objective", "H_availability", "H_failures",
                                              "service_starts", "legal_service_waits")}
        for arm in ("A", "F_risk_matching", "F_rollout"):
            selected = [r for r in controls if r["case"] == case and r["arm"] == arm]
            arms[case][arm] = {metric: float(np.mean([r[metric] for r in selected]))
                               for metric in ("objective", "H_availability", "H_failures",
                                              "service_starts")}
    analysis = {
        "status": "AUDITED", "test_run": str(test_run),
        "protocol_sha256": test["protocol_sha256"],
        "estimand": "C minus B/D/E H+tail cost; equal case and training seed weights",
        "bootstrap_draws": BOOTSTRAP_DRAWS, "analysis_seed": BOOTSTRAP_SEED,
        "training_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
        "comparisons": comparisons, "arm_means_by_case": arms,
        "uncertainty_scope": "Conditional physical-seed bootstrap; only three training seeds.",
    }
    write_json(test_run / "analysis.json", analysis)
    return audit, analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--replication-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path, required=True)
    args = parser.parse_args()
    audit, analysis = audit_and_analyze(args.training_run, args.replication_run, args.test_run)
    print(json.dumps({"audit": audit["status"], "comparisons": analysis["comparisons"],
                      "arm_means_by_case": analysis["arm_means_by_case"]}, indent=2))


if __name__ == "__main__":
    main()
