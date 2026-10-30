"""Independent raw-row SE1 area, bootstrap, gate and target arithmetic audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
PAIRS = (("S", "F"), ("A", "F"), ("S", "A"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify(run: Path, analysis_path: Path) -> dict:
    report = json.loads(analysis_path.read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if report["status"] != "ANALYZED" or report["source_manifest_sha256"] != sha(run / "manifest.json"):
        raise ValueError("analysis source identity")
    smoke = manifest["smoke"]
    checkpoints = (1200, 2400) if smoke else (24000, 48000, 72000, 96000, 144000, 192000, 288000, 384000)
    roots = 2 if smoke else 10
    physical_count = 3 if smoke else 128
    draws = 100 if smoke else 10000
    x = np.log(np.asarray(checkpoints, dtype=float))
    cube = {}
    for case in CASES:
        cube[case] = {}
        for arm in ARMS:
            values = np.empty((roots, len(checkpoints), physical_count))
            for root in range(roots):
                for j, cp in enumerate(checkpoints):
                    path = (run / "evaluations" / ("engineering" if smoke else "eval") / case / arm /
                            str(root) / f"checkpoint-{cp}-greedy.jsonl")
                    raw = list(rows(path))
                    if len(raw) != physical_count:
                        raise ValueError("raw greedy coverage")
                    values[root, j] = [row["total_cost"] for row in raw]
            cube[case][arm] = values
    rng = np.random.default_rng(20261028)
    for case in CASES:
        observed = {arm: float(np.mean(np.trapezoid(cube[case][arm].mean(axis=2), x, axis=1) /
                                       (x[-1] - x[0]))) for arm in ARMS}
        for arm in ARMS:
            if not np.isclose(observed[arm], report["primary"][case]["arm_area_means"][arm], atol=1e-9):
                raise ValueError("arm area mean")
        distributions = {f"{a}-{b}": np.empty(draws) for a, b in PAIRS}
        for draw in range(draws):
            indices = rng.integers(0, physical_count, physical_count)
            sampled = {}
            for arm in ARMS:
                root_indices = rng.integers(0, roots, roots)
                seed_means = np.take(cube[case][arm], indices, axis=2).mean(axis=2)
                root_areas = np.trapezoid(seed_means, x, axis=1) / (x[-1] - x[0])
                sampled[arm] = float(np.mean(root_areas[root_indices]))
            for a, b in PAIRS:
                distributions[f"{a}-{b}"][draw] = sampled[a] - sampled[b]
        for a, b in PAIRS:
            key = f"{a}-{b}"
            entry = report["primary"][case]["contrasts"][key]
            observed_diff = observed[a] - observed[b]
            interval = np.quantile(distributions[key], [.05 / 12, 1 - .05 / 12])
            gate = bool(observed_diff <= -10 and interval[1] < 0)
            if (not np.isclose(observed_diff, entry["mean_difference"], atol=1e-9) or
                not np.allclose(interval, entry["adjusted_interval"], atol=1e-8) or
                gate != entry["practical_pass"]):
                raise ValueError(f"area/bootstrap/gate: {case}/{key}")
    target = json.loads((run / "target-lock.json").read_text())
    for case in CASES:
        raw = [x["total_cost"] for x in rows(run / "calibration" / case / "episodes.jsonl")]
        if len(raw) != physical_count or not np.isclose(.95 * np.mean(raw), target["cases"][case]["target_cost"]):
            raise ValueError("calibration target")
        threshold = target["cases"][case]["target_cost"]
        for arm in ARMS:
            documented = report["time_to_quality"][case]["arms"][arm]
            events = []
            restricted = []
            for root in range(roots):
                means = cube[case][arm][root].mean(axis=1)
                first = next((j for j in range(len(checkpoints) - 1)
                              if means[j] <= threshold and means[j + 1] <= threshold), None)
                observed_event = first is not None
                end = len(checkpoints) - 1 if first is None else first
                events.append(observed_event)
                restricted.append(checkpoints[end])
                row = documented["roots"][root]
                if row["attained"] != observed_event or row["restricted_samples"] != checkpoints[end]:
                    raise ValueError("target root arithmetic")
            if (documented["attained_roots"] != sum(events) or
                not np.isclose(documented["restricted_mean_samples"], np.mean(restricted))):
                raise ValueError("target aggregate arithmetic")
    root_table = ROOT / report["root_checkpoint_csv"]
    with open(root_table, encoding="utf-8") as stream:
        count = sum(1 for _ in stream) - 1
    if count != len(CASES) * len(ARMS) * roots * len(checkpoints):
        raise ValueError("root checkpoint table length")
    if not report["plots"] or any(not (ROOT / p).is_file() for p in report["plots"]):
        raise ValueError("standalone plot missing")
    return {"cases": len(CASES), "arms": len(ARMS), "roots_per_arm_case": roots,
            "checkpoints": len(checkpoints), "bootstrap_draws": draws,
            "primary_contrasts": 6, "root_checkpoint_rows": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("analysis_report", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / ("stats-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "run": str(run.relative_to(ROOT)),
              "run_manifest_sha256": sha(run / "manifest.json"),
              "analysis_report_sha256": sha(args.analysis_report.resolve()),
              "auditor_sha256": sha(Path(__file__))}
    try:
        report.update(verify(run, args.analysis_report.resolve()))
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["failure"] = repr(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(out)


if __name__ == "__main__":
    main()
