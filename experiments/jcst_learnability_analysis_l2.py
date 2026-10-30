"""Predeclared crossed-root/physical-seed L2 confirmation estimators."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from jcst_learnability_l2 import ARMS, CASES, ROOT, physical_seed, sha

BOOTSTRAP_SEED = 1 << 47
BOOTSTRAP_REPS = 4000
FAMILY = 18
Q = (.05 / (2 * FAMILY), 1 - .05 / (2 * FAMILY))


def read_lines(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def raw_arrays(run: Path, manifest: dict, selection: dict) -> tuple[dict, dict]:
    bank = "confirmation" if manifest["stage"] == "confirmation" else "rep_eval"
    roots = range(5) if manifest["stage"] == "confirmation" else range(5, 10)
    found = {}
    action_variation = {}
    for info in manifest["completed_evaluations"]:
        case, arm, root = info["case"], info["arm"], info["root"]
        checkpoint, mode = info["checkpoint"], info["mode"]
        if root not in roots or mode != selection["selected"][case][arm]["mode"]:
            raise ValueError("evaluation selection")
        repetitions = 1 if mode == "greedy" else 4
        costs = np.full((128, repetitions), np.nan)
        for row in read_lines(ROOT / info["file"]):
            i, rep = row["physical_index"], row["action_repetition"]
            if row["seed"] != physical_seed(bank, case, 0, i) or np.isfinite(costs[i, rep]):
                raise ValueError("raw physical/action seed or duplicate")
            costs[i, rep] = row["total_cost"]
        if not np.isfinite(costs).all():
            raise ValueError("incomplete raw learned array")
        key = (case, arm, checkpoint, root)
        if key in found:
            raise ValueError("duplicate model array")
        found[key] = costs.mean(axis=1)
        action_variation[key] = float(np.mean(np.std(costs, axis=1, ddof=0)))
    arrays = {}
    for case in CASES:
        for arm in ARMS:
            selected_cp = int(selection["selected"][case][arm]["checkpoint"])
            for checkpoint in {24000, selected_cp}:
                arrays[(case, arm, checkpoint)] = np.stack(
                    [found[(case, arm, checkpoint, root)] for root in roots])
    for case in CASES:
        for name in ("threshold5", "risk", "corrective", "wait"):
            path = run / "controls" / bank / case / name / "episodes.jsonl"
            costs = np.full(128, np.nan)
            for i, row in enumerate(read_lines(path)):
                if i >= 128 or row["seed"] != physical_seed(bank, case, 0, i):
                    raise ValueError("control physical seed")
                costs[i] = row["total_cost"]
            if not np.isfinite(costs).all():
                raise ValueError("control incompleteness")
            arrays[(case, name)] = costs
    return arrays, action_variation


def interval(left: np.ndarray, right: np.ndarray, *, paired_roots: bool,
             bootstrap_seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(bootstrap_seed)
    nroot, nphysical = left.shape
    if nroot != 5 or nphysical != 128 or right.shape not in ((5, 128), (128,)):
        raise ValueError("crossed array shape")
    draws = np.empty(BOOTSTRAP_REPS)
    for b in range(BOOTSTRAP_REPS):
        physical = rng.integers(0, nphysical, nphysical)
        root_left = rng.integers(0, nroot, nroot)
        left_mean = left[np.ix_(root_left, physical)].mean()
        if right.ndim == 1:
            right_mean = right[physical].mean()
        else:
            root_right = root_left if paired_roots else rng.integers(0, nroot, nroot)
            right_mean = right[np.ix_(root_right, physical)].mean()
        draws[b] = left_mean - right_mean
    return (float(left.mean() - right.mean()), float(np.quantile(draws, Q[0])),
            float(np.quantile(draws, Q[1])))


def analyze(arrays: dict, selection: dict, stage: str) -> dict:
    contrasts = []
    useful = {}
    representation = {}
    mechanism = {}
    competitive = {}
    index = 0
    for case in CASES:
        useful[case] = {}
        for arm in ARMS:
            cp = int(selection["selected"][case][arm]["checkpoint"])
            selected = arrays[(case, arm, cp)]
            low = arrays[(case, arm, 24000)]
            root_gain = (low - selected).mean(axis=1)
            for comparator, other, margin, paired in (
                ("24k", low, -10, True), ("wait", arrays[(case, "wait")], -20, False)):
                mean, lower, upper = interval(selected, other, paired_roots=paired,
                                               bootstrap_seed=BOOTSTRAP_SEED + index)
                contrasts.append({"case": case, "contrast": f"{arm}-{comparator}",
                                  "mean": mean, "adjusted_lower": lower,
                                  "adjusted_upper": upper, "margin": margin,
                                  "checkpoint": cp, "mode": selection["selected"][case][arm]["mode"],
                                  "paired_roots": paired})
                index += 1
            low_row, wait_row = contrasts[-2:]
            useful[case][arm] = bool(low_row["mean"] <= -10 and low_row["adjusted_upper"] < 0
                                     and wait_row["mean"] <= -20 and wait_row["adjusted_upper"] < 0
                                     and int(np.sum(root_gain >= 10)) >= 4)
        s_cp = int(selection["selected"][case]["S"]["checkpoint"])
        s = arrays[(case, "S", s_cp)]
        for comparator, other, margin in (
            ("F", arrays[(case, "F", int(selection["selected"][case]["F"]["checkpoint"]))], -10),
            ("A", arrays[(case, "A", int(selection["selected"][case]["A"]["checkpoint"]))], -10),
            ("threshold5", arrays[(case, "threshold5")], 10)):
            mean, lower, upper = interval(s, other, paired_roots=False,
                                           bootstrap_seed=BOOTSTRAP_SEED + index)
            contrasts.append({"case": case, "contrast": f"S-{comparator}",
                              "mean": mean, "adjusted_lower": lower,
                              "adjusted_upper": upper, "margin": margin,
                              "checkpoint": s_cp,
                              "mode": selection["selected"][case]["S"]["mode"],
                              "paired_roots": False})
            if comparator == "F":
                representation[case] = bool(mean <= -10 and upper < 0)
            elif comparator == "A":
                mechanism[case] = bool(mean <= -10 and upper < 0)
            else:
                competitive[case] = bool(mean <= 10 and upper <= 10)
            index += 1
    if index != FAMILY:
        raise AssertionError("primary family size")
    return {"status": "PENDING_INDEPENDENT_STATS_AUDIT", "stage": stage,
            "primary_family": FAMILY, "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED, "quantiles": Q,
            "contrasts": contrasts, "useful_learning": useful,
            "representation_supported": representation,
            "mechanism_supported": mechanism,
            "threshold5_competitive": competitive,
            "replication_eligible": bool(stage == "confirmation" and
                                         any(useful[case][arm] for case in CASES for arm in ARMS))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--physical-audit", type=Path, required=True)
    args = parser.parse_args()
    run, train, selection_file, audit_file = (path.resolve() for path in
                                             (args.run_dir, args.train_dir, args.selection,
                                              args.physical_audit))
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads(selection_file.read_text())
    physical_audit = json.loads(audit_file.read_text())
    if (manifest["status"] != "COMPLETE" or physical_audit["status"] != "PASS" or
            physical_audit["source_manifest_hash"] != sha(run / "manifest.json") or
            selection["source_manifest_hash"] != sha(train / "manifest.json")):
        raise ValueError("source/physical audit not complete")
    arrays, variation = raw_arrays(run, manifest, selection)
    result = analyze(arrays, selection, manifest["stage"])
    result.update(source_manifest_sha256=sha(run / "manifest.json"),
                  train_manifest_sha256=sha(train / "manifest.json"),
                  selection_sha256=sha(selection_file),
                  physical_audit_sha256=sha(audit_file),
                  analysis_code_sha256=sha(Path(__file__)),
                  action_sampling_within_seed_sd={"|".join(map(str, key)): value
                                                 for key, value in variation.items()})
    output = run / "analysis_preliminary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
