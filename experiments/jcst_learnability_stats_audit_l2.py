"""Independent raw-row estimator, bootstrap and gate arithmetic audit for L2."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from jcst_learnability_l2 import ARMS, CASES, ROOT, physical_seed, sha

REPS = 4000
FAMILY = 18
LOW = .05 / (2 * FAMILY)
HIGH = 1 - LOW
BOOTSTRAP_SEED = 1 << 47


def records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def rebuild(run: Path, manifest: dict, selection: dict) -> dict:
    bank = "confirmation" if manifest["stage"] == "confirmation" else "rep_eval"
    roots = range(5) if bank == "confirmation" else range(5, 10)
    data = {}
    for meta in manifest["completed_evaluations"]:
        case, arm, root, cp, mode = (meta[key] for key in ("case", "arm", "root", "checkpoint", "mode"))
        if root not in roots or mode != selection["selected"][case][arm]["mode"]:
            raise ValueError("learned panel metadata")
        reps = 1 if mode == "greedy" else 4
        by_physical = defaultdict(dict)
        for row in records(ROOT / meta["file"]):
            i, repetition = row["physical_index"], row["action_repetition"]
            if not 0 <= i < 128 or not 0 <= repetition < reps:
                raise ValueError("physical/action index")
            if row["seed"] != physical_seed(bank, case, 0, i) or repetition in by_physical[i]:
                raise ValueError("physical seed or duplicate action repetition")
            by_physical[i][repetition] = row["total_cost"]
        if len(by_physical) != 128 or any(len(group) != reps for group in by_physical.values()):
            raise ValueError("learned row completeness")
        vector = np.asarray([np.mean([by_physical[i][r] for r in range(reps)])
                             for i in range(128)], dtype=float)
        key = (case, arm, cp, root)
        if key in data:
            raise ValueError("duplicate root/checkpoint")
        data[key] = vector
    for case in CASES:
        for name in ("threshold5", "risk", "corrective", "wait"):
            costs = []
            for i, row in enumerate(records(run / "controls" / bank / case / name / "episodes.jsonl")):
                if i >= 128 or row["seed"] != physical_seed(bank, case, 0, i):
                    raise ValueError("control pairing")
                costs.append(row["total_cost"])
            if len(costs) != 128:
                raise ValueError("control panel incomplete")
            data[(case, name)] = np.asarray(costs, dtype=float)
    for case in CASES:
        for arm in ARMS:
            for cp in {24000, int(selection["selected"][case][arm]["checkpoint"])}:
                if any((case, arm, cp, root) not in data for root in roots):
                    raise ValueError("root/checkpoint panel incomplete")
    return data


def mean_and_interval(left: np.ndarray, right: np.ndarray, paired: bool,
                      seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(REPS):
        physical_indices = rng.integers(0, 128, size=128)
        left_roots = rng.integers(0, 5, size=5)
        left_sample = float(np.mean(left[left_roots[:, None], physical_indices[None, :]]))
        if right.ndim == 1:
            right_sample = float(np.mean(right[physical_indices]))
        else:
            right_roots = left_roots if paired else rng.integers(0, 5, size=5)
            right_sample = float(np.mean(right[right_roots[:, None], physical_indices[None, :]]))
        draws.append(left_sample - right_sample)
    return (float(np.mean(left) - np.mean(right)),
            float(np.percentile(draws, LOW * 100)),
            float(np.percentile(draws, HIGH * 100)))


def verify(preliminary: dict, selection: dict, data: dict, stage: str) -> dict:
    if preliminary["primary_family"] != FAMILY or len(preliminary["contrasts"]) != FAMILY:
        raise ValueError("primary family")
    useful = {case: {} for case in CASES}
    representation = {}
    mechanism = {}
    competitive = {}
    for index, row in enumerate(preliminary["contrasts"]):
        case = row["case"]
        left_name, right_name = row["contrast"].split("-")
        left_cp = int(selection["selected"][case][left_name]["checkpoint"])
        roots = range(5) if stage == "confirmation" else range(5, 10)
        left = np.stack([data[(case, left_name, left_cp, root)] for root in roots])
        if right_name in ARMS:
            cp = int(selection["selected"][case][right_name]["checkpoint"])
            right = np.stack([data[(case, right_name, cp, root)] for root in roots])
        elif right_name == "24k":
            right = np.stack([data[(case, left_name, 24000, root)] for root in roots])
        else:
            right = data[(case, right_name)]
        paired = right_name == "24k"
        mean, lower, upper = mean_and_interval(left, right, paired,
                                                BOOTSTRAP_SEED + index)
        if (row["paired_roots"] != paired or row["checkpoint"] != left_cp or
                not np.allclose([row["mean"], row["adjusted_lower"], row["adjusted_upper"]],
                                [mean, lower, upper], rtol=0, atol=1e-8)):
            raise ValueError(f"contrast/interval mismatch: {case} {row['contrast']}")
        if right_name == "F":
            representation[case] = bool(mean <= -10 and upper < 0)
        elif right_name == "A":
            mechanism[case] = bool(mean <= -10 and upper < 0)
        elif right_name == "threshold5":
            competitive[case] = bool(mean <= 10 and upper <= 10)
    lookup = {(row["case"], row["contrast"]): row for row in preliminary["contrasts"]}
    for case in CASES:
        for arm in ARMS:
            cp = int(selection["selected"][case][arm]["checkpoint"])
            roots = range(5) if stage == "confirmation" else range(5, 10)
            count = 0
            for root in roots:
                selected = data[(case, arm, cp, root)]
                low = data[(case, arm, 24000, root)]
                count += float(np.mean(low - selected)) >= 10
            low = lookup[(case, f"{arm}-24k")]
            wait = lookup[(case, f"{arm}-wait")]
            useful[case][arm] = bool(low["mean"] <= -10 and low["adjusted_upper"] < 0
                                     and wait["mean"] <= -20 and wait["adjusted_upper"] < 0
                                     and count >= 4)
    expected_replication = bool(stage == "confirmation" and
                                any(useful[case][arm] for case in CASES for arm in ARMS))
    for name, calculated in (("useful_learning", useful),
                             ("representation_supported", representation),
                             ("mechanism_supported", mechanism),
                             ("threshold5_competitive", competitive)):
        if preliminary[name] != calculated:
            raise ValueError(f"gate arithmetic: {name}")
    if preliminary["replication_eligible"] != expected_replication:
        raise ValueError("replication eligibility arithmetic")
    return {"status": "PASS", "contrasts_verified": FAMILY,
            "bootstrap_reps_verified": REPS,
            "replication_eligible": expected_replication}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    run, selection_file = args.run_dir.resolve(), args.selection.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads(selection_file.read_text())
    preliminary_file = run / "analysis_preliminary.json"
    preliminary = json.loads(preliminary_file.read_text())
    if (manifest["status"] != "COMPLETE" or preliminary["source_manifest_sha256"] != sha(run / "manifest.json")
            or preliminary["selection_sha256"] != sha(selection_file)):
        raise ValueError("analysis source provenance")
    output = ROOT / "artifacts/jcst_learnability" / ("stats-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "run_dir": str(run.relative_to(ROOT)),
              "run_manifest_sha256": sha(run / "manifest.json"),
              "preliminary_analysis_sha256": sha(preliminary_file),
              "audit_code_sha256": sha(Path(__file__))}
    try:
        data = rebuild(run, manifest, selection)
        report.update(verify(preliminary, selection, data, manifest["stage"]))
        final = dict(preliminary)
        final.update(status="AUDIT_PASS", stats_audit_file=str((output / "report.json").relative_to(ROOT)),
                     stats_audit_code_sha256=sha(Path(__file__)))
        final_path = run / "analysis.json"
        with final_path.open("x") as stream:
            json.dump(final, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException as exc:
        report.update(status="FAIL", failure=repr(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
