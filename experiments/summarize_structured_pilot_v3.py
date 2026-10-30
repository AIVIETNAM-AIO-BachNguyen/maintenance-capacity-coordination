"""Descriptive secondary measures for the audited structured pilot v3 panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from maintenance_capacity_coordination.structured_pilot_v3 import ARMS, CASES, MILESTONES, TRAIN_SEEDS

METRICS = ("objective", "H_availability", "H_failures", "queued_pm_ticks",
           "queued_cm_ticks", "H_pending", "end_pending", "service_starts")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(training_run: Path, heldout_run: Path):
    if read_json(training_run / "audit.json")["status"] != "PASS":
        raise RuntimeError("training audit required")
    if read_json(heldout_run / "audit.json")["status"] != "PASS":
        raise RuntimeError("heldout audit required")
    rows = read_jsonl(heldout_run / "heldout_episodes.jsonl")
    controls = read_jsonl(heldout_run / "heldout_controls.jsonl")
    result = {"status": "DESCRIPTIVE", "cases": {}, "learning": {},
              "forecast_transitions": {}, "S_R_cost_equal_fraction": {}}
    for case in CASES:
        result["cases"][case] = {}
        for arm in (*ARMS, "A", "F_risk_matching", "F_rollout"):
            panel = rows if arm in ARMS else controls
            selected = [r for r in panel if r["case"] == case and r["arm"] == arm]
            result["cases"][case][arm] = {metric: float(np.mean([r[metric] for r in selected]))
                                          for metric in METRICS}
            if arm in ARMS:
                result["cases"][case][arm]["legal_service_waits"] = float(np.mean(
                    [r["legal_service_waits"] for r in selected]))
            result["cases"][case][arm]["selection_seconds_per_episode"] = float(np.mean(
                [r["selection_seconds"] for r in selected]))
        rollout = [r for r in controls if r["case"] == case and r["arm"] == "F_rollout"]
        result["forecast_transitions"][case] = {
            "F_rollout_total": sum(r["forecast_transitions"] for r in rollout),
            "F_rollout_mean_per_episode": float(np.mean([r["forecast_transitions"] for r in rollout])),
            "learned": 0,
        }
        keyed = {(r["arm"], r["checkpoint"].split(":")[0], r["seed"]): r
                 for r in rows if r["case"] == case and r["arm"] in ("S", "R")}
        equal = sum(keyed["S", str(seed), physical]["objective"] ==
                    keyed["R", str(seed), physical]["objective"]
                    for seed in TRAIN_SEEDS for physical in range(73182000, 73182032))
        result["S_R_cost_equal_fraction"][case] = equal / (len(TRAIN_SEEDS) * 32)
    for arm in ARMS:
        dev = {}
        trials = []
        for milestone in MILESTONES:
            values = []
            for case in CASES:
                for seed in TRAIN_SEEDS:
                    trial = training_run / f"{case}_{arm}_{seed}"
                    summary = read_json(trial / "dev_milestones.json")
                    values.append(next(x["mean_H_tail_cost"] for x in summary
                                       if x["transition"] == milestone))
                    if milestone == MILESTONES[0]:
                        trials.append(read_json(trial / "manifest.json"))
            dev[str(milestone)] = float(np.mean(values))
        x = np.asarray(MILESTONES, dtype=float)
        y = np.asarray([dev[str(m)] for m in MILESTONES])
        result["learning"][arm] = {
            "DEV_mean_by_milestone": dev,
            "DEV_trapezoidal_area_cost_units": float(np.trapezoid(y, x) / (x[-1] - x[0])),
            "mean_trial_wall_seconds_including_DEV": float(np.mean([t["elapsed_seconds"] for t in trials])),
            "actor_parameters": trials[0]["actor_parameters"],
            "critic_parameters": trials[0]["critic_parameters"],
            "H_training_decisions_per_trial": trials[0]["actual_H_transitions"],
        }
    (heldout_run / "secondary_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--heldout-run", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.training_run, args.heldout_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
