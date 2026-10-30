"""Independent raw-row estimator, bootstrap and gate arithmetic audit for L3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "artifacts/jcst_matched_architecture"
CASES = ("base", "generalist")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cube(run: Path, case: str, arm: str) -> np.ndarray:
    cube = np.zeros((10, 128))
    for root in range(10):
        file = run / "evaluations/eval" / case / arm / str(root) / "checkpoint-96000-greedy.jsonl"
        values = []
        for index, line in enumerate(file.open()):
            row = json.loads(line)
            if (row["case"], row["arm"], row["root"], row["checkpoint"], row["mode"],
                row["physical_index"], row["action_repetition"]) != (
                    case, arm, root, 96000, "greedy", index, 0):
                raise ValueError("raw primary row provenance")
            if not np.isclose(row["H_cost"] + row["tail_cost"], row["total_cost"]):
                raise ValueError("raw H/tail sum")
            values.append(row["total_cost"])
        if len(values) != 128:
            raise ValueError("raw primary row count")
        cube[root, :] = values
    return cube


def recompute(s: np.ndarray, comparator: np.ndarray, rng: np.random.Generator) -> tuple[float, list[float]]:
    values = []
    for _ in range(4000):
        physical = rng.choice(128, size=128, replace=True)
        sroots = rng.choice(10, size=10, replace=True)
        croots = rng.choice(10, size=10, replace=True)
        values.append(float(s[sroots][:, physical].mean() - comparator[croots][:, physical].mean()))
    return float(s.mean() - comparator.mean()), [float(x) for x in np.percentile(values, [0.625, 99.375])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("analysis_report", type=Path)
    args = parser.parse_args()
    run, report_path = args.run_dir.resolve(), args.analysis_report.resolve()
    out = OUTROOT / ("stats-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    audit = {"status": "RUNNING", "source_run": str(run.relative_to(ROOT)),
             "analysis_report": str(report_path.relative_to(ROOT)),
             "analysis_sha256": digest(report_path),
             "auditor_sha256": digest(Path(__file__))}
    try:
        report = json.loads(report_path.read_text())
        if (report["status"] != "REPLAY_AUDITED_PENDING_STATS_AUDIT" or
                report["source_manifest_sha256"] != digest(run / "manifest.json")):
            raise ValueError("analysis/source provenance")
        rng = np.random.default_rng(20261027)
        verdicts = {"S_minus_F": [], "S_minus_A": []}
        for case in CASES:
            s = read_cube(run, case, "S")
            for arm in ("F", "A"):
                other = read_cube(run, case, arm)
                actual = report["primary"][case][f"S_minus_{arm}"]
                mean, bounds = recompute(s, other, rng)
                if not np.isclose(mean, actual["mean"], rtol=0, atol=1e-9):
                    raise ValueError("primary mean")
                if not np.allclose(bounds, actual["adjusted_98_75_interval"], rtol=0, atol=1e-8):
                    raise ValueError("primary bootstrap")
                if not np.allclose(s.mean(axis=1), actual["S_root_means"], rtol=0, atol=1e-9):
                    raise ValueError("S root means")
                if not np.allclose(other.mean(axis=1), actual["comparator_root_means"], rtol=0, atol=1e-9):
                    raise ValueError("comparator root means")
                gate = mean <= -10 and bounds[1] < 0
                if gate != actual["practical_and_sign_gate"]:
                    raise ValueError("per-case gate")
                verdicts[f"S_minus_{arm}"].append(gate)
        for case in CASES:
            for cp, mode in ((24000, "greedy"), (96000, "greedy"),
                             (384000, "greedy"), (384000, "stochastic")):
                repetitions = 4 if mode == "stochastic" else 1
                for arm in ("S", "F", "A"):
                    root_means = []
                    for root in range(10):
                        file = (run / "evaluations/eval" / case / arm / str(root) /
                                f"checkpoint-{cp}-{mode}.jsonl")
                        costs = []
                        for row_index, line in enumerate(file.open()):
                            row = json.loads(line)
                            if (row["physical_index"], row["action_repetition"]) != divmod(row_index, repetitions):
                                raise ValueError("secondary row order")
                            costs.append(row["total_cost"])
                        if len(costs) != 128 * repetitions:
                            raise ValueError("secondary row count")
                        root_means.append(float(np.mean(costs)))
                    claimed = report["secondary"][case][f"{cp}-{mode}"][arm]
                    if not np.allclose(root_means, claimed["root_means"], rtol=0, atol=1e-9):
                        raise ValueError("secondary root means")
                    if not np.isclose(np.mean(root_means), claimed["mean_cost"], rtol=0, atol=1e-9):
                        raise ValueError("secondary panel mean")
        audit["checked_secondary_arm_case_checkpoint_modes"] = 24
        expected = {"structured_sample_efficiency_two_cases": all(verdicts["S_minus_F"]),
                    "cross_message_mechanism_two_cases": all(verdicts["S_minus_A"])}
        if report["claims"] != expected:
            raise ValueError("two-case gate")
        audit["status"] = "PASS"
        audit["checked_primary_contrasts"] = 4
        audit["claims"] = expected
    except BaseException as exc:
        audit["status"] = "FAIL"
        audit["failure"] = repr(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(out)


if __name__ == "__main__":
    main()
