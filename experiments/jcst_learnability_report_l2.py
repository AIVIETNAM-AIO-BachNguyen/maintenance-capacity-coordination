"""Standalone L2 TRAIN/DEV curves, policy diagnostics and paired mode tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

from jcst_learnability_l2 import ARMS, CASES, CHECKPOINTS, ROOT, case_config, sha


def records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    return float(np.mean(values))


def eval_summary(meta: dict) -> tuple[dict, dict]:
    rows = list(records(ROOT / meta["file"]))
    repetitions = 1 if meta["mode"] == "greedy" else 4
    if len(rows) != meta["physical_seeds"] * repetitions:
        raise ValueError("evaluation row count")
    config, _ = case_config(meta["case"])
    per_physical = {}
    all_diag = []
    durations = []
    for row in rows:
        i = row["physical_index"]
        per_physical.setdefault(i, []).append(row["total_cost"])
        all_diag.extend(row["diagnostics"])
        for tick in row["trace"][:48]:
            for m, c, technician in tick["action"]:
                durations.append(config.duration[m][c][technician])
    if any(len(values) != repetitions for values in per_physical.values()):
        raise ValueError("action repetitions")
    costs = np.asarray([mean(per_physical[i]) for i in range(meta["physical_seeds"])])
    metric_names = ("pm_starts", "cm_starts", "failures", "unavailable_machine_intervals",
                    "occupied_technician_intervals", "wait_H", "forced_wait", "discretionary_wait",
                    "wait_failed_free", "wait_all_healthy", "wait_all_busy",
                    "unfinished_jobs", "failed_components_at_cutoff")
    result = {key: meta[key] for key in ("case", "arm", "root", "checkpoint", "mode")}
    result.update(physical_seeds=meta["physical_seeds"],
                  action_repetitions=repetitions,
                  total_cost=mean([row["total_cost"] for row in rows]),
                  H_cost=mean([row["H_cost"] for row in rows]),
                  tail_cost=mean([row["tail_cost"] for row in rows]),
                  preventive=mean([row["components"]["preventive"] for row in rows]),
                  corrective=mean([row["components"]["corrective"] for row in rows]),
                  unavailable=mean([row["components"]["unavailable"] for row in rows]),
                  failure=mean([row["components"]["failure"] for row in rows]))
    result.update({key: mean([row["metrics"][key] for row in rows]) for key in metric_names})
    for key in ("entropy", "wait_probability", "chosen_probability", "greedy_probability",
                "legal_count", "near_tie", "exact_tie"):
        result[key] = mean([d[key] for d in all_diag])
    multi_action = [d for d in all_diag if d["legal_count"] > 1]
    for key in ("logit_gap", "probability_gap"):
        result[key] = mean([d[key] for d in multi_action])
    result["normalized_entropy"] = mean([d["entropy"] / np.log(d["legal_count"])
                                          for d in multi_action])
    result["selected_service_duration"] = mean(durations) if durations else 0.0
    result["selection_seconds"] = mean([row["selection_seconds"] for row in rows])
    return result, {"costs": costs}


def collect_train(run: Path, manifest: dict) -> tuple[list[dict], list[dict]]:
    windows = []
    updates = []
    for info in manifest["completed_trials"]:
        case, arm, root = info["case"], info["arm"], info["root"]
        trial = run / "trials" / case / arm / str(root)
        checkpoints = [json.loads((trial / f"checkpoint-{cp}.json").read_text())
                       for cp in CHECKPOINTS]
        costs = [row["total_cost"] for row in records(trial / "train.jsonl")]
        if len(costs) != CHECKPOINTS[-1] // 48:
            raise ValueError("incomplete TRAIN costs")
        previous = 0
        for cp, cp_info in zip(CHECKPOINTS, checkpoints):
            end = cp // 48
            windows.append({"case": case, "arm": arm, "root": root,
                            "checkpoint": cp, "window_start_episode": previous,
                            "window_end_episode": end,
                            "mean_stochastic_train_cost": mean(costs[previous:end]),
                            "cumulative_train_cost": mean(costs[:end]),
                            "elapsed_training_seconds": cp_info["elapsed_training_seconds"]})
            previous = end
        for row in records(trial / "updates.jsonl"):
            item = {"case": case, "arm": arm, "root": root, **row}
            updates.append(item)
    return windows, updates


def paired_decoding(cube: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    summaries = []
    rng = np.random.default_rng((1 << 46) + 2026)
    for case in CASES:
        for arm in ARMS:
            for cp in CHECKPOINTS:
                differences = np.zeros((5, 16))
                for root in range(5):
                    greedy = cube[(case, arm, root, cp, "greedy")]["costs"]
                    stochastic = cube[(case, arm, root, cp, "stochastic")]["costs"]
                    differences[root] = stochastic - greedy
                    for physical_index in range(16):
                        rows.append({"case": case, "arm": arm, "root": root,
                                     "checkpoint": cp, "physical_index": physical_index,
                                     "greedy_cost": greedy[physical_index],
                                     "stochastic_mean_cost": stochastic[physical_index],
                                     "stochastic_minus_greedy": differences[root, physical_index]})
                draws = []
                for _ in range(2000):
                    roots = rng.integers(0, 5, 5)
                    physical = rng.integers(0, 16, 16)
                    draws.append(float(differences[np.ix_(roots, physical)].mean()))
                summaries.append({"case": case, "arm": arm, "checkpoint": cp,
                                  "mean_stochastic_minus_greedy": float(differences.mean()),
                                  "descriptive_95_lower": float(np.quantile(draws, .025)),
                                  "descriptive_95_upper": float(np.quantile(draws, .975)),
                                  "roots": 5, "physical_seeds": 16, "action_repetitions": 4})
    return rows, summaries


def plots(output: Path, curves: list[dict], roots: list[dict], updates: list[dict], controls: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"S": "#225ea8", "F": "#e6550d", "A": "#31a354"}
    for axis_name, xkey in (("samples", "checkpoint"),
                            ("compute", "mean_elapsed_training_seconds")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        for ax, case in zip(axes, CASES):
            for arm in ARMS:
                for mode, style in (("greedy", "-"), ("stochastic", "--")):
                    subset = sorted((row for row in curves if row["case"] == case and
                                     row["arm"] == arm and row["mode"] == mode),
                                    key=lambda row: row["checkpoint"])
                    ax.plot([row[xkey] for row in subset], [row["DEV_mean_cost"] for row in subset],
                            style, color=palette[arm], marker="o", markersize=3,
                            label=f"{arm} {mode}")
            threshold = next(row["mean_cost"] for row in controls if row["case"] == case and row["arm"] == "threshold5")
            wait = next(row["mean_cost"] for row in controls if row["case"] == case and row["arm"] == "wait")
            ax.axhline(threshold, color="black", linestyle=":", label="threshold5")
            ax.axhline(wait, color="gray", linestyle=":", label="WAIT")
            ax.set_title(case)
            ax.set_xlabel("Controlled decisions per root" if axis_name == "samples" else "Mean train seconds per root")
            ax.set_ylabel("DEV H+tail cost (lower better)")
            ax.grid(alpha=.2)
        axes[1].legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"learning_by_{axis_name}.png", dpi=180)
        fig.savefig(output / f"learning_by_{axis_name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, case in enumerate(CASES):
        for arm in ARMS:
            subset = sorted((row for row in curves if row["case"] == case and row["arm"] == arm and row["mode"] == "greedy"),
                            key=lambda row: row["checkpoint"])
            axes[0, column].plot([row["checkpoint"] for row in subset],
                                 [row["wait_probability"] for row in subset],
                                 marker="o", color=palette[arm], label=arm)
            axes[1, column].plot([row["checkpoint"] for row in subset],
                                 [row["discretionary_wait"] for row in subset],
                                 marker="o", color=palette[arm], label=arm)
        axes[0, column].set_title(case)
        axes[0, column].set_ylabel("Mean WAIT probability")
        axes[1, column].set_ylabel("Discretionary WAIT / episode")
        axes[1, column].set_xlabel("Controlled decisions per root")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "wait_diagnostics.png", dpi=180)
    fig.savefig(output / "wait_diagnostics.pdf")
    plt.close(fig)

    telemetry = [row for row in updates if row["sampled_telemetry"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, case in zip(axes, CASES):
        for arm in ARMS:
            subset = [row for row in telemetry if row["case"] == case and row["arm"] == arm]
            by_sample = {}
            for row in subset:
                by_sample.setdefault(row["samples"], []).append(row)
            points = sorted(by_sample)
            ax.plot(points, [mean([row["actor_gradient_post"] for row in by_sample[s]]) for s in points],
                    color=palette[arm], label=f"{arm} actor")
            ax.plot(points, [mean([row["critic_gradient_post"] for row in by_sample[s]]) for s in points],
                    color=palette[arm], linestyle="--", label=f"{arm} critic")
        ax.set_title(case)
        ax.set_xlabel("Controlled decisions per root")
        ax.set_ylabel("Post-clip gradient norm")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "optimization_gradients.png", dpi=180)
    fig.savefig(output / "optimization_gradients.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    run = args.train_dir.resolve()
    selection_file = args.selection.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads(selection_file.read_text())
    if (manifest["status"] != "COMPLETE" or selection["status"] != "AUDIT_PASS" or
            selection["source_manifest_hash"] != sha(run / "manifest.json")):
        raise ValueError("TRAIN/DEV audit/selection incomplete")
    output = ROOT / "artifacts/jcst_learnability" / ("report-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_train_dir": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "selection_sha256": sha(selection_file), "report_code_sha256": sha(Path(__file__))}
    try:
        root_rows = []
        cube = {}
        for meta in manifest["completed_evaluations"]:
            result, raw = eval_summary(meta)
            root_rows.append(result)
            cube[(meta["case"], meta["arm"], meta["root"], meta["checkpoint"], meta["mode"])] = raw
        if len(root_rows) != 240:
            raise ValueError("incomplete evaluation panel")
        controls = manifest["completed_controls"]
        if len(controls) != 8:
            raise ValueError("incomplete controls")
        windows, updates = collect_train(run, manifest)
        paired_rows, paired_summary = paired_decoding(cube)
        paired_arms = []
        for case in CASES:
            threshold_path = run / "controls" / "dev" / case / "threshold5" / "episodes.jsonl"
            threshold_costs = [row["total_cost"] for row in records(threshold_path)]
            if len(threshold_costs) != 16:
                raise ValueError("threshold physical pairing")
            for cp in CHECKPOINTS:
                for mode in ("greedy", "stochastic"):
                    for physical_index in range(16):
                        cost = {arm: mean([cube[(case, arm, root, cp, mode)]["costs"][physical_index]
                                           for root in range(5)]) for arm in ARMS}
                        paired_arms.append({"case": case, "checkpoint": cp, "mode": mode,
                                            "physical_index": physical_index,
                                            "S_mean_over_roots": cost["S"],
                                            "F_mean_over_roots": cost["F"],
                                            "A_mean_over_roots": cost["A"],
                                            "threshold5_cost": threshold_costs[physical_index],
                                            "S_minus_F": cost["S"] - cost["F"],
                                            "S_minus_A": cost["S"] - cost["A"],
                                            "S_minus_threshold5": cost["S"] - threshold_costs[physical_index]})
        curves = []
        for case in CASES:
            for arm in ARMS:
                for cp in CHECKPOINTS:
                    for mode in ("greedy", "stochastic"):
                        subset = [row for row in root_rows if row["case"] == case and row["arm"] == arm
                                  and row["checkpoint"] == cp and row["mode"] == mode]
                        timing = [row["elapsed_training_seconds"] for row in windows if row["case"] == case
                                  and row["arm"] == arm and row["checkpoint"] == cp]
                        row = {"case": case, "arm": arm, "checkpoint": cp, "mode": mode,
                               "DEV_mean_cost": mean([item["total_cost"] for item in subset]),
                               "DEV_root_sd": float(np.std([item["total_cost"] for item in subset], ddof=1)),
                               "mean_elapsed_training_seconds": mean(timing)}
                        for key in ("H_cost", "tail_cost", "pm_starts", "cm_starts", "failures",
                                    "unavailable_machine_intervals", "wait_H", "forced_wait",
                                    "discretionary_wait", "wait_failed_free", "wait_all_healthy",
                                    "wait_all_busy", "entropy", "wait_probability", "greedy_probability",
                                    "chosen_probability", "logit_gap", "probability_gap", "near_tie",
                                    "exact_tie", "legal_count", "normalized_entropy",
                                    "selected_service_duration"):
                            row[key] = mean([item[key] for item in subset])
                        curves.append(row)
        csv_file(output / "learning_curves.csv", curves)
        csv_file(output / "per_training_root.csv", root_rows)
        csv_file(output / "training_cost_windows.csv", windows)
        csv_file(output / "ppo_updates.csv", updates)
        csv_file(output / "paired_decoding_physical.csv", paired_rows)
        csv_file(output / "paired_decoding_summary.csv", paired_summary)
        csv_file(output / "paired_arm_physical.csv", paired_arms)
        csv_file(output / "controls.csv", controls)
        plots(output, curves, root_rows, updates, controls)
        report.update(status="COMPLETE", root_rows=len(root_rows),
                      paired_physical_rows=len(paired_rows),
                      training_window_rows=len(windows), update_rows=len(updates))
    except BaseException as exc:
        report.update(status="INCOMPLETE", failure=repr(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
