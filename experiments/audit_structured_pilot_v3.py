"""Independent artifact audit and prelocked paired analysis for structured pilot v3."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import configs
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.structured_pilot_v3 import (
    ARMS, CASES, DEV_SEEDS, HELDOUT_SEEDS, MILESTONES, REPLICATION_DEV_SEEDS,
    REPLICATION_EVAL_SEEDS, REPLICATION_TRAIN_SEEDS, ROOT, TRAIN_SEEDS,
    TRIAL_TIMEOUT, PROTOCOL, seed_registry_check, sha256, write_json,
)

DRAWS = 5000
ANALYSIS_SEED = 73187000
CONTROLS = ("A", "F_risk_matching", "F_rollout")
HISTORIC_SEEDS = (72600000, 72600001, 72600002, 72600003, 72600004)


def load(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_manifest(manifest, run):
    assert manifest["status"] == "COMPLETED"
    assert manifest["protocol_sha256"] == sha256(PROTOCOL)
    cfgs = configs()
    for case in CASES:
        digest = hashlib.sha256(json.dumps(asdict(cfgs[case]), sort_keys=True).encode()).hexdigest()
        assert manifest["config_sha256"][case] == digest
    for name, digest in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in manifest["output_sha256"].items():
        assert sha256(run / name) == digest, name
    seed_registry_check()


def check_cost(row):
    assert row["invalid_assignments"] == 0
    assert abs(row["episode_return"] + row["objective"]) < 1e-8
    assert abs(row["objective"] - row["H_cost"] - row["tail_cost"]) < 1e-8
    assert abs(row["objective"] - row["maintenance_cost"] - row["failure_cost"]
               - row["unavailability_cost"]) < 1e-8
    assert 0 <= row["H_availability"] <= 1


def audit_training(run):
    run = run.resolve()
    manifest = load(run / "manifest.json")
    if manifest["kind"] not in ("smoke", "train", "replication_train"):
        raise RuntimeError("training audit requires smoke/train/replication_train")
    audit_manifest(manifest, run)
    smoke = manifest["kind"] == "smoke"
    seeds = (73100041,) if smoke else (TRAIN_SEEDS if manifest["kind"] == "train" else REPLICATION_TRAIN_SEEDS)
    dev_seeds = (73100042,) if smoke else (DEV_SEEDS if manifest["kind"] == "train" else REPLICATION_DEV_SEEDS)
    milestones = (256,) if smoke else MILESTONES
    expected = {(case, arm, seed) for case in CASES for arm in ARMS for seed in seeds}
    trials = {(t["case"], t["arm"], t["train_seed"]): t for t in manifest["trials"]}
    assert set(trials) == expected and len(manifest["trials"]) == len(expected)
    actor_params = {}
    complete_episodes = 0
    for key, summary in trials.items():
        case, arm, seed = key
        trial = run / f"{case}_{arm}_{seed}"
        item = load(trial / "manifest.json")
        assert item == summary and item["status"] == "COMPLETED"
        assert item["actual_H_transitions"] == item["requested_H_transitions"] == (256 if smoke else 8192)
        assert item["forecast_transitions"] == 0 and item["elapsed_seconds"] < TRIAL_TIMEOUT
        actor_params[arm] = item["actor_parameters"]
        rows = read_jsonl(trial / "dev_episodes.jsonl")
        assert len(rows) == len(dev_seeds) * len(milestones)
        coverage = Counter((int(r["checkpoint"]), r["seed"]) for r in rows)
        assert set(coverage) == {(milestone, physical) for milestone in milestones for physical in dev_seeds}
        assert max(coverage.values()) == 1
        for row in rows:
            assert (row["case"], row["arm"]) == (case, arm)
            assert row["physical_seed"] == role_seed(row["seed"], BASE_VERSION + ":physical")
            check_cost(row)
        checkpoints = []
        for milestone in milestones:
            checkpoint = trial / f"checkpoint_{milestone}.pt"
            assert checkpoint.exists() and checkpoint.stat().st_size > 0
            mean = np.mean([r["objective"] for r in rows if r["checkpoint"] == str(milestone)])
            checkpoints.append((float(mean), milestone, checkpoint.name))
        best = min(checkpoints)
        selection = load(trial / "selection.json")
        assert selection["checkpoint"] == best[2] and selection["transition"] == best[1]
        assert abs(selection["mean_H_tail_cost"] - best[0]) < 1e-8
        assert item["selected_checkpoint"] == best[2]
        training_rows = read_jsonl(trial / "training_episodes.jsonl")
        assert len(training_rows) == item["actual_H_transitions"] // 48
        for row in training_rows:
            check_cost(row)
        complete_episodes += len(training_rows)
    assert actor_params["S"] == actor_params["R"]
    assert abs(actor_params["S"] - actor_params["G3"]) / actor_params["S"] < .10
    audit = {"status": "PASS", "kind": manifest["kind"], "trials": len(trials),
             "H_training_decisions": sum(t["actual_H_transitions"] for t in trials.values()),
             "complete_training_episodes": complete_episodes,
             "tail_physical_transitions": complete_episodes * 16,
             "DEV_episodes": len(trials) * len(milestones) * len(dev_seeds),
             "selection_and_source_config_protocol_hashes": "PASS",
             "legality_H_tail_cost_seed_roles": "PASS",
             "actor_parameters": actor_params}
    write_json(run / "audit.json", audit)
    return audit


def audit_evaluation(run, training_run):
    run, training_run = run.resolve(), training_run.resolve()
    manifest = load(run / "manifest.json")
    train = load(training_run / "manifest.json")
    if manifest["kind"] not in ("heldout", "replication_eval"):
        raise RuntimeError("evaluation audit requires heldout/replication_eval")
    audit_manifest(manifest, run)
    assert load(training_run / "audit.json")["status"] == "PASS"
    assert manifest["training_run"] == str(training_run)
    assert train["protocol_sha256"] == manifest["protocol_sha256"]
    assert train["source_sha256"] == manifest["source_sha256"]
    assert train["config_sha256"] == manifest["config_sha256"]
    panel = manifest["kind"]
    seeds = HELDOUT_SEEDS if panel == "heldout" else REPLICATION_EVAL_SEEDS
    train_seeds = TRAIN_SEEDS if panel == "heldout" else REPLICATION_TRAIN_SEEDS
    rows = read_jsonl(run / f"{panel}_episodes.jsonl")
    controls = read_jsonl(run / f"{panel}_controls.jsonl")
    expected = {(case, arm, train_seed, physical) for case in CASES for arm in ARMS
                for train_seed in train_seeds for physical in seeds}
    keyed = {}
    for row in rows:
        seed, milestone = map(int, row["checkpoint"].split(":"))
        key = row["case"], row["arm"], seed, row["seed"]
        assert key not in keyed
        selection = load(training_run / f"{row['case']}_{row['arm']}_{seed}" / "selection.json")
        assert selection["transition"] == milestone
        assert row["physical_seed"] == role_seed(row["seed"], BASE_VERSION + ":physical")
        check_cost(row)
        keyed[key] = row
    assert set(keyed) == expected and len(rows) == manifest["learned_episodes"] == len(expected)
    expected_controls = {(case, arm, historical, physical)
                         for case in CASES for physical in seeds
                         for arm, historical in (("F_risk_matching", None), ("F_rollout", None),
                                                 *(("A", seed) for seed in HISTORIC_SEEDS))}
    control_keys = Counter((r["case"], r["arm"], r.get("historical_train_seed"), r["seed"])
                           for r in controls)
    assert set(control_keys) == expected_controls and max(control_keys.values()) == 1
    assert len(controls) == manifest["control_episodes"] == len(expected_controls)
    for row in controls:
        assert row["physical_seed"] == role_seed(row["seed"], BASE_VERSION + ":physical")
        check_cost(row)
    audit = {"status": "PASS", "kind": panel,
             "learned_episodes": len(rows), "control_episodes": len(controls),
             "source_config_protocol_and_checkpoint_selection": "PASS",
             "paired_physical_seed_coverage": "PASS", "legality_and_H_tail_cost": "PASS"}
    write_json(run / "audit.json", audit)
    rng = np.random.default_rng(ANALYSIS_SEED)
    comparisons = {}
    for other in ("G3", "R"):
        difference = np.asarray([[[keyed[case, "S", train_seed, physical]["objective"]
                                   - keyed[case, other, train_seed, physical]["objective"]
                                   for physical in seeds] for train_seed in train_seeds]
                                 for case in CASES], dtype=float)
        draws = np.empty(DRAWS)
        for i in range(DRAWS):
            indices = rng.integers(0, len(seeds), size=len(seeds))
            draws[i] = difference[:, :, indices].mean()
        comparisons[other] = {
            "mean": float(difference.mean()),
            "case_mean": {case: float(difference[ci].mean()) for ci, case in enumerate(CASES)},
            "training_seed_mean": {str(seed): float(difference[:, si, :].mean())
                                   for si, seed in enumerate(train_seeds)},
            "paired_raw_seed_cluster_95": np.quantile(draws, (.025, .975)).tolist(),
        }
    costs = {case: {arm: float(np.mean([r["objective"] for r in rows
                                     if r["case"] == case and r["arm"] == arm]))
                    for arm in ARMS} for case in CASES}
    for case in CASES:
        for arm in CONTROLS:
            costs[case][arm] = float(np.mean([r["objective"] for r in controls
                                             if r["case"] == case and r["arm"] == arm]))
    primary = comparisons["G3"]
    go = bool(primary["mean"] <= -30 and
              all(value < 0 for value in primary["case_mean"].values()) and
              sum(value < 0 for value in primary["training_seed_mean"].values()) >= 2 and
              primary["paired_raw_seed_cluster_95"][1] < 0)
    analysis = {"status": "AUDITED", "kind": panel,
                "estimand": "S minus matched-information/same-action G3 or remaining-time ablation R H+tail cost",
                "comparisons": comparisons, "costs_by_case": costs,
                "bootstrap_draws": DRAWS, "analysis_seed": ANALYSIS_SEED,
                "physical_cluster": "one raw seed across arms, training seeds and cases",
                "training_seed_scope": "three fixed trained models per arm/case; variability shown separately",
                "replication_go": go if panel == "heldout" else None}
    write_json(run / "analysis.json", analysis)
    if panel == "heldout":
        write_json(run / "replication_go_decision.json",
                   {"go": go, "reason": "prelocked S-G3 matched-action gate",
                    "protocol_sha256": manifest["protocol_sha256"]})
    return audit, analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    kind = load(args.run / "manifest.json")["kind"]
    if kind in ("smoke", "train", "replication_train"):
        result = audit_training(args.run)
        print(json.dumps(result, indent=2))
    else:
        if args.training_run is None:
            parser.error("--training-run required")
        audit, analysis = audit_evaluation(args.run, args.training_run)
        print(json.dumps({"audit": audit, "analysis": analysis}, indent=2))


if __name__ == "__main__":
    main()
