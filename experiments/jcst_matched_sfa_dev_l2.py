"""Exploratory fixed-checkpoint, fixed-mode S/F/A contrasts on audited L2 DEV rows.

This is post-DEV analysis and cannot confirm a representation or mechanism claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/jcst_learnability/train_dev-20261026T020729.745460Z"
CHECKPOINTS = (24000, 96000, 384000, 1536000)
MODES = ("greedy", "stochastic")
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
ROOTS = range(5)
PHYSICAL = range(16)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(run: Path, case: str, arm: str, checkpoint: int, mode: str) -> np.ndarray:
    panel = np.full((5, 16), np.nan)
    for root in ROOTS:
        path = (run / "evaluations/dev" / case / arm / str(root) /
                f"checkpoint-{checkpoint}-{mode}.jsonl")
        by_physical: dict[int, list[float]] = {i: [] for i in PHYSICAL}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert (row["case"], row["arm"], row["root"], row["checkpoint"],
                        row["mode"], row["bank"], row["terminal_status"]) == (
                            case, arm, root, checkpoint, mode, "dev", "COMPLETE")
                index = int(row["physical_index"])
                assert index in by_physical
                by_physical[index].append(float(row["total_cost"]))
        repeats = 1 if mode == "greedy" else 4
        assert all(len(by_physical[i]) == repeats for i in PHYSICAL), path
        panel[root] = [np.mean(by_physical[i]) for i in PHYSICAL]
    assert np.isfinite(panel).all()
    return panel


def contrast(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> dict:
    # Roots are independent between arms; physical seeds are paired across arms.
    observed = float(left.mean() - right.mean())
    draws = np.empty(4000)
    for index in range(len(draws)):
        physical = rng.integers(0, 16, 16)
        roots_left = rng.integers(0, 5, 5)
        roots_right = rng.integers(0, 5, 5)
        draws[index] = (left[np.ix_(roots_left, physical)].mean() -
                        right[np.ix_(roots_right, physical)].mean())
    return {"mean": observed,
            "descriptive_unadjusted_95_interval": [float(v) for v in np.quantile(draws, [.025, .975])],
            "S_root_means": [float(v) for v in left.mean(axis=1)],
            "comparator_root_means": [float(v) for v in right.mean(axis=1)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "artifacts/jcst_learnability" / f"matched-dev-analysis-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "EXPLORATORY_DEV", "source_run": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "analysis_source_sha256": sha(Path(__file__)),
              "design": "same checkpoint and mode, independent arm roots, paired physical seeds; four stochastic action draws nested within each seed",
              "interval_note": "descriptive unadjusted bootstrap; DEV inspected before this analysis; not confirmation",
              "results": {}}
    rng = np.random.default_rng(20261027)
    for case in CASES:
        report["results"][case] = {}
        for checkpoint in CHECKPOINTS:
            report["results"][case][str(checkpoint)] = {}
            for mode in MODES:
                panels = {arm: load_panel(run, case, arm, checkpoint, mode) for arm in ARMS}
                report["results"][case][str(checkpoint)][mode] = {
                    f"S_minus_{arm}": contrast(panels["S"], panels[arm], rng)
                    for arm in ("F", "A")}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(out / "report.json")


if __name__ == "__main__":
    main()
