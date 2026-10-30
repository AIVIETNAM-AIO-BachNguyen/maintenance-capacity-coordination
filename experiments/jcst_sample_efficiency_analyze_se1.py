"""SE1 locked learning-curve, target, compute and plot analysis after replay PASS."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ht_pdm_fjsp.component_controllers_v1 import Threshold, observe
from ht_pdm_fjsp.component_framework_v1 import advance
from ht_pdm_fjsp.component_service_v1 import initial_state
from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import action_universe, encode, tensors
from jcst_learnability_l2 import load_policy

ROOT = Path(__file__).resolve().parents[1]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CONTRASTS = (("S", "F"), ("A", "F"), ("S", "A"))
CHECKPOINTS = (24000, 48000, 72000, 96000, 144000, 192000, 288000, 384000)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def log_weights(checkpoints: tuple[int, ...]) -> np.ndarray:
    x = np.log(np.asarray(checkpoints, dtype=float))
    width = x[-1] - x[0]
    weights = np.empty(len(x), dtype=float)
    weights[0] = (x[1] - x[0]) / (2 * width)
    weights[-1] = (x[-1] - x[-2]) / (2 * width)
    weights[1:-1] = (x[2:] - x[:-2]) / (2 * width)
    assert np.isclose(weights.sum(), 1)
    return weights


def load_panel(run: Path, smoke: bool):
    checkpoints = (1200, 2400) if smoke else CHECKPOINTS
    roots = range(2) if smoke else range(10)
    count = 3 if smoke else 128
    cube = {case: {arm: np.empty((len(roots), len(checkpoints), count), dtype=float)
                   for arm in ARMS} for case in CASES}
    hcube = {case: {arm: np.empty_like(cube[case][arm]) for arm in ARMS} for case in CASES}
    tailcube = {case: {arm: np.empty_like(cube[case][arm]) for arm in ARMS} for case in CASES}
    for case in CASES:
        for arm in ARMS:
            for root in roots:
                for j, cp in enumerate(checkpoints):
                    path = run / "evaluations" / ("engineering" if smoke else "eval") / case / arm / str(root) / f"checkpoint-{cp}-greedy.jsonl"
                    data = list(rows(path))
                    if len(data) != count or [x["physical_index"] for x in data] != list(range(count)):
                        raise ValueError("greedy panel incomplete")
                    cube[case][arm][root, j] = [x["total_cost"] for x in data]
                    hcube[case][arm][root, j] = [x["H_cost"] for x in data]
                    tailcube[case][arm][root, j] = [x["tail_cost"] for x in data]
    return checkpoints, cube, hcube, tailcube


def crossed_bootstrap(cube, checkpoints, draws: int) -> dict:
    weights = log_weights(checkpoints)
    rng = np.random.default_rng(20261028)
    family_alpha = .05 / 6
    result = {}
    for case in CASES:
        data = cube[case]
        count = data["S"].shape[2]
        nroots = data["S"].shape[0]
        observed = {arm: float(np.mean(data[arm].mean(axis=2) @ weights)) for arm in ARMS}
        boot = {f"{a}-{b}": np.empty(draws) for a, b in CONTRASTS}
        for draw in range(draws):
            physical = rng.integers(0, count, count)
            sample_weight = np.bincount(physical, minlength=count) / count
            arm_means = {}
            for arm in ARMS:
                root_indices = rng.integers(0, nroots, nroots)
                root_areas = np.einsum("rcp,c,p->r", data[arm], weights, sample_weight)
                arm_means[arm] = float(np.mean(root_areas[root_indices]))
            for a, b in CONTRASTS:
                boot[f"{a}-{b}"][draw] = arm_means[a] - arm_means[b]
        result[case] = {"arm_area_means": observed, "contrasts": {}}
        for a, b in CONTRASTS:
            name = f"{a}-{b}"
            difference = observed[a] - observed[b]
            lower, upper = np.quantile(boot[name], [family_alpha / 2, 1 - family_alpha / 2])
            result[case]["contrasts"][name] = {
                "mean_difference": difference, "adjusted_interval": [float(lower), float(upper)],
                "practical_pass": bool(difference <= -10 and upper < 0)}
    return result


def target_report(run: Path, cube, checkpoints, smoke: bool) -> dict:
    target_lock = json.loads((run / "target-lock.json").read_text())
    rng = np.random.default_rng(2026102801)
    result = {}
    for case in CASES:
        calibration = np.asarray([x["total_cost"] for x in rows(run / "calibration" / case / "episodes.jsonl")])
        means = np.mean(calibration[rng.integers(0, len(calibration), (100 if smoke else 10000, len(calibration)))], axis=1)
        target = target_lock["cases"][case]["target_cost"]
        result[case] = {"threshold5_calibration_mean": float(calibration.mean()),
                        "threshold5_calibration_interval": [float(x) for x in np.quantile(means, [.025, .975])],
                        "target_interval_conditional": [float(.95 * x) for x in np.quantile(means, [.025, .975])],
                        "target_cost": target, "arms": {}}
        for arm in ARMS:
            episodes = cube[case][arm].mean(axis=2)
            records = []
            for root, costs in enumerate(episodes):
                first = next((j for j in range(len(checkpoints) - 1)
                              if costs[j] <= target and costs[j + 1] <= target), None)
                end = len(checkpoints) - 1 if first is None else first
                elapsed = json.loads((run / "trials" / case / arm / str(root) /
                                      f"checkpoint-{checkpoints[end]}.json").read_text())["elapsed_training_seconds"]
                records.append({"root": root, "attained": first is not None,
                    "first_checkpoint": None if first is None else checkpoints[first],
                    "confirmed_checkpoint": None if first is None else checkpoints[first + 1],
                    "restricted_samples": checkpoints[end], "restricted_training_seconds": elapsed})
            result[case]["arms"][arm] = {
                "roots": records,
                "attainment_rate_by_budget": {str(cp): sum(r["attained"] and r["first_checkpoint"] <= cp
                                                 for r in records) / len(records) for cp in checkpoints},
                "restricted_mean_samples": float(np.mean([r["restricted_samples"] for r in records])),
                "restricted_mean_training_seconds": float(np.mean([r["restricted_training_seconds"] for r in records])),
                "attained_roots": sum(r["attained"] for r in records),
                "censored_roots": sum(not r["attained"] for r in records),
                "median_samples": _km_median(records, "restricted_samples"),
                "median_training_seconds": _km_median(records, "restricted_training_seconds")}
    return result


def secondary_stochastic(run: Path, smoke: bool) -> list[dict]:
    bank = "engineering" if smoke else "eval"
    count = 3 if smoke else 128
    checkpoints = (2400,) if smoke else (96000, 384000)
    output = []
    for case in CASES:
        for arm in ARMS:
            for root in range(2 if smoke else 10):
                for cp in checkpoints:
                    path = run / "evaluations" / bank / case / arm / str(root) / f"checkpoint-{cp}-stochastic.jsonl"
                    data = list(rows(path))
                    if len(data) != count * 4:
                        raise ValueError("stochastic panel incomplete")
                    costs = np.asarray([x["total_cost"] for x in data]).reshape(count, 4)
                    output.append({"case": case, "arm": arm, "root": root, "checkpoint": cp,
                        "mean_cost": float(costs.mean()),
                        "mean_within_physical_seed_sd": float(costs.std(axis=1, ddof=1).mean()),
                        "physical_seeds": count, "action_repetitions_per_seed": 4})
    return output


def greedy_diagnostics(run: Path, checkpoints: tuple[int, ...], smoke: bool) -> list[dict]:
    bank = "engineering" if smoke else "eval"
    output = []
    for case in CASES:
        for arm in ARMS:
            for cp in checkpoints:
                records = []
                for root in range(2 if smoke else 10):
                    records.extend(rows(run / "evaluations" / bank / case / arm / str(root) /
                                        f"checkpoint-{cp}-greedy.jsonl"))
                row = {"case": case, "arm": arm, "checkpoint": cp}
                for name in ("preventive", "corrective", "unavailable", "failure"):
                    row[f"mean_{name}_cost"] = float(np.mean([record["components"][name] for record in records]))
                for name in ("pm_starts", "cm_starts", "failures", "unavailable_machine_intervals",
                             "wait_H", "forced_wait", "discretionary_wait"):
                    row[f"mean_{name}"] = float(np.mean([record["metrics"][name] for record in records]))
                output.append(row)
    return output


def compute_summary(run: Path, checkpoints: tuple[int, ...], smoke: bool) -> dict:
    result = {}
    for case in CASES:
        result[case] = {}
        for arm in ARMS:
            metas = [json.loads((run / "trials" / case / arm / str(root) /
                                 f"checkpoint-{checkpoints[-1]}.json").read_text())
                     for root in range(2 if smoke else 10)]
            result[case][arm] = {
                "mean_root_training_elapsed_seconds": float(np.mean([x["elapsed_training_seconds"] for x in metas])),
                "sum_root_training_cpu_seconds": float(sum(x["cpu_training_seconds"] for x in metas)),
                "mean_phase_seconds": {key: float(np.mean([x["phase_seconds"][key] for x in metas]))
                                       for key in metas[0]["phase_seconds"]},
            }
    return result


def _km_median(records: list[dict], field: str) -> float | None:
    event_times = sorted({r[field] for r in records if r["attained"]})
    survival = 1.0
    for t in event_times:
        at_risk = sum(r[field] >= t for r in records)
        events = sum(r["attained"] and r[field] == t for r in records)
        survival *= 1 - events / at_risk
        if survival <= .5:
            return float(t)
    return None


def inference_benchmark(run: Path, checkpoints: tuple[int, ...], smoke: bool) -> dict:
    report = {}
    for case in CASES:
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        state = initial_state(cfg)
        states = []
        physical_seed = 2**54 + 3 * 10**13 + CASES.index(case) * 10**12 + 7777
        controller = Threshold(5)
        for tick in range(32):
            states.append(state)
            action = controller.choose(observe(cfg, state, 48, 16)).action
            state = advance(cfg, state, action, physical_seed, 48, 16).state
        encoded = [encode(cfg, state, universe, 48, 16) for state in states]
        inputs = [tensors([item]) for item in encoded]
        report[case] = {}
        for arm in ARMS:
            latencies = []
            for root in range(2 if smoke else 10):
                _, _, policy = load_policy(run / "trials" / case / arm / str(root), checkpoints[-1])
                with torch.no_grad():
                    for _ in range(3):
                        for values in inputs:
                            policy.distribution(*values).probs
                    start = time.perf_counter()
                    for _ in range(10 if smoke else 20):
                        for values in inputs:
                            policy.distribution(*values).probs
                    elapsed = time.perf_counter() - start
                latencies.append(elapsed / (len(inputs) * (10 if smoke else 20)) * 1e6)
            report[case][arm] = {"per_root_microseconds": latencies,
                                 "mean_microseconds": float(np.mean(latencies)),
                                 "states": len(states), "warmup_repetitions": 3,
                                 "measured_repetitions": 10 if smoke else 20}
    return report


def write_csv(path: Path, data: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def make_plots(out: Path, run: Path, cube, checkpoints, smoke: bool) -> None:
    colors = {"S": "#2563eb", "F": "#dc2626", "A": "#059669"}
    for case in CASES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in ARMS:
            means = cube[case][arm].mean(axis=2)
            for trajectory in means:
                ax.plot(checkpoints, trajectory, color=colors[arm], alpha=.14, linewidth=.8)
            ax.plot(checkpoints, means.mean(axis=0), color=colors[arm], label=arm, linewidth=2)
        ax.set_xscale("log")
        ax.set_xticks(checkpoints, [f"{x//1000}k" for x in checkpoints])
        ax.set_xlabel("Controlled simulator decisions")
        ax.set_ylabel("H+tail cost")
        ax.set_title(f"SE1 learning curve: {case}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"learning_curve_samples_{case}.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in ARMS:
            for root in range(2 if smoke else 10):
                times = [json.loads((run / "trials" / case / arm / str(root) /
                                     f"checkpoint-{cp}.json").read_text())["elapsed_training_seconds"] for cp in checkpoints]
                ax.plot(times, cube[case][arm][root].mean(axis=1), color=colors[arm], alpha=.25)
            ax.plot([], [], color=colors[arm], label=arm)
        ax.set_xlabel("Measured root training elapsed (seconds)")
        ax.set_ylabel("H+tail cost")
        ax.set_title(f"SE1 cost versus training time: {case}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"learning_curve_time_{case}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run_dir.resolve()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "PASS" or audit["source_manifest_sha256"] != sha(run / "manifest.json"):
        raise ValueError("independent replay PASS required")
    smoke = json.loads((run / "manifest.json").read_text())["smoke"]
    out = run / ("analysis-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    checkpoints, cube, hcube, tailcube = load_panel(run, smoke)
    weights = log_weights(checkpoints)
    primary = crossed_bootstrap(cube, checkpoints, 100 if smoke else 10000)
    targets = target_report(run, cube, checkpoints, smoke)
    inference = inference_benchmark(run, checkpoints, smoke)
    stochastic = secondary_stochastic(run, smoke)
    diagnostics = greedy_diagnostics(run, checkpoints, smoke)
    compute = compute_summary(run, checkpoints, smoke)
    root_rows = []
    for case in CASES:
        for arm in ARMS:
            for root in range(2 if smoke else 10):
                for j, cp in enumerate(checkpoints):
                    meta = json.loads((run / "trials" / case / arm / str(root) /
                                       f"checkpoint-{cp}.json").read_text())
                    root_rows.append({"case": case, "arm": arm, "root": root, "checkpoint": cp,
                        "mean_cost": float(cube[case][arm][root, j].mean()),
                        "mean_H_cost": float(hcube[case][arm][root, j].mean()),
                        "mean_tail_cost": float(tailcube[case][arm][root, j].mean()),
                        "training_seconds": meta["elapsed_training_seconds"],
                        "cpu_training_seconds": meta["cpu_training_seconds"],
                        "root_area": float(cube[case][arm][root].mean(axis=1) @ weights)})
    write_csv(out / "root_checkpoint.csv", root_rows)
    write_csv(out / "stochastic_checkpoint.csv", stochastic)
    write_csv(out / "greedy_diagnostics.csv", diagnostics)
    make_plots(out, run, cube, checkpoints, smoke)
    control_means = {case: {name: float(np.mean([r["total_cost"] for r in rows(
        run / "controls" / ("engineering" if smoke else "eval") / case / name / "episodes.jsonl")]))
        for name in ("threshold5", "risk", "corrective", "wait")} for case in CASES}
    report = {"status": "ANALYZED", "smoke": smoke,
        "source_manifest_sha256": sha(run / "manifest.json"),
        "audit_report_sha256": sha(args.audit),
        "checkpoints": checkpoints, "area_weights": weights.tolist(),
        "bootstrap_draws": 100 if smoke else 10000,
        "primary": primary, "time_to_quality": targets,
        "controls": control_means, "inference": inference, "compute": compute,
        "root_checkpoint_csv": str((out / "root_checkpoint.csv").relative_to(ROOT)),
        "stochastic_checkpoint_csv": str((out / "stochastic_checkpoint.csv").relative_to(ROOT)),
        "greedy_diagnostics_csv": str((out / "greedy_diagnostics.csv").relative_to(ROOT)),
        "plots": [str(p.relative_to(ROOT)) for p in sorted(out.glob("*.png"))]}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
