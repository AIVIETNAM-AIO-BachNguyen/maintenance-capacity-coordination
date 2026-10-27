"""Descriptive L1 tables and standalone plots from already audited run directories."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CONTROLS = ("threshold5", "risk", "corrective", "rollout2", "wait")
CHECKPOINTS = (6000, 12000, 24000)


def load(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf8") as handle:
        for line in handle:
            yield json.loads(line)


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError("empty table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def checked(path: Path):
    manifest = load(path / "manifest.json")
    audit = load(path / "independent_audit.json")
    if manifest["status"] != "COMPLETE" or audit["status"] != "PASS":
        raise ValueError(f"unaudited/incomplete input: {path}")
    if audit["source_manifest_hash"] != digest(path / "manifest.json"):
        raise ValueError("audit/manifest hash")
    return manifest


def flatten_episode(row: dict, case: str, arm: str, root: int | None):
    return {
        "case": case, "arm": arm, "training_root": "" if root is None else root,
        "physical_seed": row["seed"], "checkpoint_samples": row.get("checkpoint", ""),
        "total_cost": row["total_cost"], "H_cost": row["H_cost"],
        "tail_cost": row["tail_cost"],
        **{f"cost_{name}": row["components"][name]
           for name in ("preventive", "corrective", "unavailable", "failure")},
        **row["metrics"],
        "selection_seconds": row["selection_seconds"],
        "episode_seconds": row["episode_seconds"],
        "forecast_transitions": row["forecast_transitions"],
    }


def sample_data(run_dir: Path, bank: str, selection: dict | None):
    records = []
    for case in CASES:
        for arm in ARMS:
            checkpoint = selection["selected"][case][arm]
            for root in range(5):
                path = run_dir / "evaluations" / bank / case / arm / str(root) / "episodes.jsonl.gz"
                for row in rows(path):
                    records.append(flatten_episode(row, case, arm, root))
        for arm in CONTROLS:
            path = run_dir / "controls" / bank / case / arm / "episodes.jsonl.gz"
            for row in rows(path):
                records.append(flatten_episode(row, case, arm, None))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path)
    args = parser.parse_args()
    train = (ROOT / args.train_dir).resolve()
    confirmation = (ROOT / args.confirmation_dir).resolve()
    train_manifest = checked(train)
    confirm_manifest = checked(confirmation)
    selection = load(train / "selection.json")
    analysis = load(confirmation / "analysis.json")
    if selection["status"] != "AUDIT_PASS" or analysis["status"] != "AUDIT_PASS":
        raise ValueError("selection/analysis not audited")
    if selection["source_manifest_hash"] != digest(train / "manifest.json"):
        raise ValueError("selection source hash")
    if confirm_manifest["train_dir"] != str(train.relative_to(ROOT)):
        raise ValueError("confirmation train source")
    rep = None
    if args.replication_dir:
        rep = (ROOT / args.replication_dir).resolve()
        checked(rep)
    output = ROOT / "artifacts" / "jcst_framework" / (
        "report-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    all_rows = sample_data(confirmation, "confirmation", selection)
    write_csv(output / "confirmation_episodes.csv", all_rows)
    groups = {}
    for row in all_rows:
        groups.setdefault((row["case"], row["arm"], row["training_root"]), []).append(row)
    numeric = [name for name in all_rows[0] if name in (
        "total_cost", "H_cost", "tail_cost", "cost_preventive", "cost_corrective",
        "cost_unavailable", "cost_failure", "pm_starts", "cm_starts", "failures",
        "completions", "unavailable_machine_intervals", "occupied_technician_intervals",
        "wait_H", "unfinished_jobs", "failed_components_at_cutoff",
        "selection_seconds", "episode_seconds", "forecast_transitions")]
    per_root = []
    for key, items in sorted(groups.items()):
        per_root.append({"case": key[0], "arm": key[1], "training_root": key[2],
                         "episodes": len(items),
                         **{name: statistics.mean(float(x[name]) for x in items)
                            for name in numeric}})
    write_csv(output / "per_training_seed.csv", per_root)
    summary = []
    for case in CASES:
        for arm in ARMS + CONTROLS:
            items = [r for r in all_rows if r["case"] == case and r["arm"] == arm]
            root_costs = [r["total_cost"] for r in per_root
                          if r["case"] == case and r["arm"] == arm]
            summary.append({"case": case, "arm": arm, "episodes": len(items),
                            "independent_training_roots": len(root_costs),
                            "training_root_sd": statistics.stdev(root_costs) if len(root_costs) > 1 else "",
                            **{name: statistics.mean(float(x[name]) for x in items)
                               for name in numeric}})
    write_csv(output / "summary.csv", summary)
    paired = []
    for case in CASES:
        physical = sorted({r["physical_seed"] for r in all_rows if r["case"] == case})
        for physical_seed in physical:
            per = [r for r in all_rows if r["case"] == case
                   and r["physical_seed"] == physical_seed]
            values = {}
            for arm in ARMS + CONTROLS:
                matches = [r["total_cost"] for r in per if r["arm"] == arm]
                if len(matches) != (5 if arm in ARMS else 1):
                    raise ValueError("paired physical panel incomplete")
                values[arm] = statistics.mean(matches)
            paired.append({"case": case, "physical_seed": physical_seed,
                           **values, "S_minus_F": values["S"] - values["F"],
                           "S_minus_A": values["S"] - values["A"],
                           "S_minus_threshold5": values["S"] - values["threshold5"]})
    write_csv(output / "paired_physical.csv", paired)
    learning = []
    dev_means = selection["means"]
    for case in CASES:
        for arm in ARMS:
            for cp in CHECKPOINTS:
                elapsed = []
                for root in range(5):
                    trial = train / "trials" / case / arm / str(root)
                    elapsed.append((trial / f"checkpoint-{cp}.pt").stat().st_mtime -
                                   (trial / "model.json").stat().st_mtime)
                learning.append({"case": case, "arm": arm, "H_decisions": cp,
                                 "DEV_mean_cost": dev_means[case][arm][str(cp)],
                                 "mean_elapsed_training_seconds": statistics.mean(elapsed),
                                 "elapsed_measure": "filesystem checkpoint minus model-file mtime"})
    write_csv(output / "learning_curves.csv", learning)
    train_time = []
    for trial in train_manifest["completed_trials"]:
        train_time.append({"case": trial["case"], "arm": trial["arm"],
                           "training_root": trial["root"], "H_decisions": trial["samples"],
                           "training_seconds": trial["train_seconds"],
                           "actor_parameters": trial["parameter_counts"]["actor"],
                           "critic_parameters": trial["parameter_counts"]["critic"]})
    write_csv(output / "training_compute.csv", train_time)
    for xfield, xlabel, filename in (
        ("H_decisions", "Controlled simulator decisions", "learning_by_samples"),
        ("mean_elapsed_training_seconds", "Approximate training elapsed seconds", "learning_by_compute"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for axis, case in zip(axes, CASES):
            for arm, color in (("S", "#1b9e77"), ("F", "#d95f02"), ("A", "#7570b3")):
                points = [r for r in learning if r["case"] == case and r["arm"] == arm]
                axis.plot([r[xfield] for r in points], [r["DEV_mean_cost"] for r in points],
                          marker="o", label=arm, color=color)
            axis.set_title(case)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("DEV H+tail cost")
            axis.grid(alpha=.25)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / f"{filename}.pdf")
        fig.savefig(output / f"{filename}.png", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, case in zip(axes, CASES):
        chosen = [r for r in summary if r["case"] == case]
        axis.bar([r["arm"] for r in chosen], [r["total_cost"] for r in chosen],
                 color=["#1b9e77", "#d95f02", "#7570b3"] + ["#666666"] * 5)
        axis.set_title(case)
        axis.set_ylabel("Confirmation H+tail cost")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "confirmation_cost.pdf")
    fig.savefig(output / "confirmation_cost.png", dpi=200)
    plt.close(fig)
    rep_statement = "Replication was not opened under the locked confirmation gate."
    if rep is not None:
        rep_statement = f"Replication artifact: {rep.relative_to(ROOT)}."
    lines = [
        "# JCST L1 audited framework campaign",
        "",
        f"TRAIN/DEV: `{train.relative_to(ROOT)}`.",
        f"Confirmation: `{confirmation.relative_to(ROOT)}`.",
        f"Both source runs and their independent replay audits passed. {rep_statement}",
        "",
        "The primary contrasts use the auditor's crossed training-root/physical-seed bootstrap",
        "with six Bonferroni-adjusted two-sided intervals. Negative cost difference favors S.",
        "",
        "| Case | Contrast | Mean | Adjusted interval | Locked margin | Pass |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in analysis["rows"]:
        lines.append(f"| {row['case']} | {row['contrast']} | {row['mean']:.2f} | "
                     f"[{row['adjusted_lo']:.2f}, {row['adjusted_hi']:.2f}] | "
                     f"{row['practical_margin']:+.0f} | {row['pass']} |")
    lines.extend(["", f"Independent retraining eligible: **{analysis['replication_eligible']}**.",
                  "",
                  "Tables retain each training-root mean, paired physical seed and cost component.",
                  "Learning-by-compute uses checkpoint/model file modification times as an",
                  "approximate elapsed measure, not CPU profiling or matched-wall-time evidence.",
                  "No learner inference uses forecast transitions; R2 has separate model access.",
                  "All costs are uncalibrated simulator units. C1 and historical TEST are not",
                  "training, DEV or confirmation data for L1."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    provenance = {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "script_hash": digest(Path(__file__)),
                  "train_manifest_hash": digest(train / "manifest.json"),
                  "train_audit_hash": digest(train / "independent_audit.json"),
                  "selection_hash": digest(train / "selection.json"),
                  "confirmation_manifest_hash": digest(confirmation / "manifest.json"),
                  "confirmation_audit_hash": digest(confirmation / "independent_audit.json"),
                  "confirmation_analysis_hash": digest(confirmation / "analysis.json"),
                  "replication_dir": str(rep.relative_to(ROOT)) if rep else None}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
