"""Audit the replication panel and apply the prelocked TEST gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from maintenance_capacity_coordination.pilot import (
    ARMS, CASES, REPLICATION_SEEDS, TRAIN_SEEDS, write_json,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def analyze(training_run, replication_run):
    training_run, replication_run = Path(training_run), Path(replication_run)
    train_manifest = json.loads((training_run / "manifest.json").read_text())
    replicate_manifest = json.loads((replication_run / "manifest.json").read_text())
    if (train_manifest["status"], train_manifest["kind"],
        replicate_manifest["status"], replicate_manifest["kind"]) != (
            "COMPLETED", "train", "COMPLETED", "replicate"):
        raise RuntimeError("incomplete training or replication run")
    if train_manifest["protocol_sha256"] != replicate_manifest["protocol_sha256"]:
        raise RuntimeError("protocol hash mismatch")
    if train_manifest["source_sha256"] != replicate_manifest["source_sha256"]:
        raise RuntimeError("source hash mismatch")
    rows = read_jsonl(replication_run / "replication_episodes.jsonl")
    keyed = {}
    for row in rows:
        train_seed = int(str(row["checkpoint"]).split(":", 1)[0])
        key = row["case"], row["arm"], train_seed, row["seed"]
        if key in keyed or row["invalid_assignments"] != 0:
            raise RuntimeError("duplicate or infeasible replication episode")
        if abs(row["objective"] - row["H_cost"] - row["tail_cost"]) > 1e-8:
            raise RuntimeError("H+tail mismatch")
        if abs(row["episode_return"] + row["objective"]) > 1e-8:
            raise RuntimeError("return/cost mismatch")
        keyed[key] = row["objective"]
    expected = {(case, arm, training_seed, seed)
                for case in CASES for arm in ARMS
                for training_seed in TRAIN_SEEDS for seed in REPLICATION_SEEDS}
    if set(keyed) != expected:
        raise RuntimeError(f"panel coverage mismatch: missing={len(expected - set(keyed))}, extra={len(set(keyed) - expected)}")
    rng = np.random.default_rng(73150000)
    comparisons = {}
    for other in ("B", "D", "E"):
        differences = np.asarray([[
            [keyed[case, "C", training_seed, seed] - keyed[case, other, training_seed, seed]
             for seed in REPLICATION_SEEDS]
            for case in CASES] for training_seed in TRAIN_SEEDS], dtype=float)
        by_train = differences.mean(axis=(1, 2)).tolist()
        draws = np.empty(5000)
        for index in range(len(draws)):
            sampled = rng.integers(0, len(REPLICATION_SEEDS),
                                   size=(len(TRAIN_SEEDS), len(CASES), len(REPLICATION_SEEDS)))
            draws[index] = np.take_along_axis(differences, sampled, axis=2).mean()
        interval = np.quantile(draws, (.025, .975)).tolist()
        comparisons[other] = {
            "estimand": "C minus " + other + " H+tail cost, equal case and training seed weights",
            "mean": float(differences.mean()),
            "by_case": {case: float(differences[:, case_index, :].mean())
                        for case_index, case in enumerate(CASES)},
            "by_training_seed": {str(seed): value for seed, value in zip(TRAIN_SEEDS, by_train)},
            "physical_seed_bootstrap_95": interval,
            "test_gate_pass": bool(all(value < 0 for value in by_train) and interval[1] < 0),
        }
    go = any(item["test_gate_pass"] for item in comparisons.values())
    summary = {
        "status": "AUDITED", "replication_run": str(replication_run.resolve()),
        "training_run": str(training_run.resolve()), "protocol_sha256": train_manifest["protocol_sha256"],
        "source_sha256_match": True, "episode_count": len(rows),
        "training_seeds": TRAIN_SEEDS, "physical_seeds": REPLICATION_SEEDS,
        "bootstrap_draws": 5000, "analysis_seed": 73150000,
        "comparisons": comparisons, "go": go,
        "scope": "Conditional physical-seed uncertainty; three training seeds are displayed individually."
    }
    write_json(replication_run / "analysis.json", summary)
    write_json(replication_run / "test_go_decision.json",
               {"go": go, "reason": "prelocked replication cost gate", "analysis_file": "analysis.json",
                "protocol_sha256": train_manifest["protocol_sha256"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--replication-run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.training_run, args.replication_run)
    print(json.dumps({"go": result["go"], "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
