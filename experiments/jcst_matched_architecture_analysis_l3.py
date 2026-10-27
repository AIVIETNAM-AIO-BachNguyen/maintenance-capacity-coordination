"""Prelocked primary and secondary H+tail analysis of the L3 S/F/A panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/jcst_matched_architecture"
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
COMBOS = ((24000, "greedy"), (96000, "greedy"),
          (384000, "greedy"), (384000, "stochastic"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(run: Path, case: str, arm: str, checkpoint: int, mode: str) -> tuple[np.ndarray, dict]:
    cube = np.full((10, 128), np.nan)
    extra = {"H_cost": [], "tail_cost": [], "pm_starts": [], "cm_starts": [],
             "failures": [], "unavailable_machine_intervals": [],
             "wait_H": [], "forced_wait": [], "discretionary_wait": []}
    reps = 1 if mode == "greedy" else 4
    for root in range(10):
        path = (run / "evaluations/eval" / case / arm / str(root) /
                f"checkpoint-{checkpoint}-{mode}.jsonl")
        values = np.empty((128, reps))
        with path.open() as stream:
            for n, line in enumerate(stream):
                i, rep = divmod(n, reps)
                if i >= 128:
                    raise ValueError(f"extra row {path}")
                row = json.loads(line)
                if (row["physical_index"], row["action_repetition"]) != (i, rep):
                    raise ValueError("row order")
                values[i, rep] = row["total_cost"]
                for name in extra:
                    extra[name].append(row[name] if name in row else row["metrics"][name])
        if n + 1 != 128 * reps:
            raise ValueError(f"short file {path}")
        cube[root] = values.mean(axis=1)
    return cube, {name: float(np.mean(values)) for name, values in extra.items()}


def primary(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> dict:
    observed = float(left.mean() - right.mean())
    draws = np.empty(4000)
    for b in range(4000):
        physical = rng.integers(0, 128, 128)
        sroots = rng.integers(0, 10, 10)
        croots = rng.integers(0, 10, 10)
        draws[b] = left[np.ix_(sroots, physical)].mean() - right[np.ix_(croots, physical)].mean()
    lo, hi = (float(x) for x in np.quantile(draws, [0.00625, 0.99375]))
    return {"mean": observed, "adjusted_98_75_interval": [lo, hi],
            "practical_and_sign_gate": observed <= -10 and hi < 0,
            "S_root_means": [float(x) for x in left.mean(axis=1)],
            "comparator_root_means": [float(x) for x in right.mean(axis=1)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    run, audit_path = args.run_dir.resolve(), args.audit.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest["status"] != "COMPLETE" or manifest["stage"] != "collect" or
            audit["status"] != "PASS" or
            audit["source_manifest_sha256"] != sha(run / "manifest.json")):
        raise ValueError("collection or independent replay not qualified")
    out = ARTIFACTS / ("analysis-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_run": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "replay_audit": str(audit_path.relative_to(ROOT)),
              "replay_audit_sha256": sha(audit_path),
              "analysis_source_sha256": sha(Path(__file__)),
              "primary_family": "S-F and S-A at 96k greedy separately in base and generalist; four Bonferroni-adjusted 2-sided intervals",
              "primary": {}, "secondary": {}}
    try:
        rng = np.random.default_rng(20261027)
        for case in CASES:
            report["primary"][case] = {}
            report["secondary"][case] = {}
            for cp, mode in COMBOS:
                cubes = {}
                descriptive = {}
                for arm in ARMS:
                    cubes[arm], descriptive[arm] = panel(run, case, arm, cp, mode)
                key = f"{cp}-{mode}"
                report["secondary"][case][key] = {
                    arm: {"mean_cost": float(cubes[arm].mean()),
                          "root_means": [float(x) for x in cubes[arm].mean(axis=1)],
                          **descriptive[arm]} for arm in ARMS}
                if (cp, mode) == (96000, "greedy"):
                    for arm in ("F", "A"):
                        report["primary"][case][f"S_minus_{arm}"] = primary(
                            cubes["S"], cubes[arm], rng)
        report["claims"] = {
            "structured_sample_efficiency_two_cases": all(
                report["primary"][case]["S_minus_F"]["practical_and_sign_gate"] for case in CASES),
            "cross_message_mechanism_two_cases": all(
                report["primary"][case]["S_minus_A"]["practical_and_sign_gate"] for case in CASES),
        }
        report["status"] = "REPLAY_AUDITED_PENDING_STATS_AUDIT"
    except BaseException as exc:
        report["status"] = "INCOMPLETE"
        report["failure"] = repr(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(out)


if __name__ == "__main__":
    main()
