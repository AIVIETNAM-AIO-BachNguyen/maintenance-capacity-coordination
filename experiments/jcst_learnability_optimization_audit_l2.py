"""Independent structural/finite audit and summary of L2 PPO update telemetry."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jcst_learnability_l2 import ARMS, CASES, CHECKPOINTS, ROOT, sha

FULL_FIELDS = ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction",
               "actor_gradient_pre", "critic_gradient_pre", "actor_gradient_post",
               "critic_gradient_post", "actor_update", "critic_update",
               "actor_relative_update", "critic_relative_update",
               "advantage_std", "advantage_mean", "return_mean", "return_std",
               "critic_prediction_mse", "explained_variance")


def updates(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def audit(run: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE" or manifest["stage"] != "train_dev":
        raise ValueError("TRAIN/DEV not complete")
    expected = {(case, arm, root) for case in CASES for arm in ARMS for root in range(5)}
    completed = {(x["case"], x["arm"], x["root"]) for x in manifest["completed_trials"]}
    if completed != expected or len(manifest["completed_trials"]) != len(expected):
        raise ValueError("training root panel incomplete")
    summaries = []
    for case, arm, root in tqdm(sorted(expected), desc="audit PPO telemetry"):
        trial = run / "trials" / case / arm / str(root)
        samples = list(updates(trial / "updates.jsonl"))
        if len(samples) != 1280:
            raise ValueError(f"update count {case}/{arm}/{root}")
        diagnostics = []
        for index, row in enumerate(samples, start=1):
            if row["update"] != index or row["samples"] != index * 1200:
                raise ValueError("PPO update sample schedule")
            sampled = index == 1 or index % 20 == 0
            if row["sampled_telemetry"] != sampled:
                raise ValueError("PPO telemetry sampling schedule")
            if not math.isfinite(row["loss"]) or not math.isfinite(row["advantage_std"]):
                raise ValueError("nonfinite core PPO telemetry")
            if sampled:
                if any(name not in row or not math.isfinite(row[name]) for name in FULL_FIELDS):
                    raise ValueError("missing/nonfinite PPO diagnostic")
                combined_post = math.hypot(row["actor_gradient_post"], row["critic_gradient_post"])
                if combined_post > .5001:
                    raise ValueError("global gradient clip exceeded")
                if not 0 <= row["clip_fraction"] <= 1 or row["entropy"] < 0:
                    raise ValueError("PPO metric range")
                diagnostics.append(row)
        for checkpoint in CHECKPOINTS:
            if checkpoint not in {row["samples"] for row in diagnostics}:
                raise ValueError("missing checkpoint-adjacent diagnostic")
        if len(diagnostics) != 65:  # first update plus multiples of 20 through 1280
            raise ValueError("sampled diagnostic count")
        summaries.append({
            "case": case, "arm": arm, "root": root,
            "updates": len(samples), "sampled_updates": len(diagnostics),
            "mean_actor_gradient_pre": float(np.mean([x["actor_gradient_pre"] for x in diagnostics])),
            "mean_actor_gradient_post": float(np.mean([x["actor_gradient_post"] for x in diagnostics])),
            "mean_critic_gradient_pre": float(np.mean([x["critic_gradient_pre"] for x in diagnostics])),
            "mean_critic_gradient_post": float(np.mean([x["critic_gradient_post"] for x in diagnostics])),
            "mean_actor_relative_update": float(np.mean([x["actor_relative_update"] for x in diagnostics])),
            "mean_critic_relative_update": float(np.mean([x["critic_relative_update"] for x in diagnostics])),
            "fraction_actor_update_below_1e-9": float(np.mean([x["actor_update"] < 1e-9 for x in diagnostics])),
            "fraction_critic_update_below_1e-9": float(np.mean([x["critic_update"] < 1e-9 for x in diagnostics])),
            "fraction_global_clip_active": float(np.mean([
                math.hypot(x["actor_gradient_pre"], x["critic_gradient_pre"]) > .5
                for x in diagnostics])),
            "mean_entropy": float(np.mean([x["entropy"] for x in diagnostics])),
            "mean_approx_kl": float(np.mean([x["approx_kl"] for x in diagnostics])),
            "mean_clip_fraction": float(np.mean([x["clip_fraction"] for x in diagnostics])),
            "mean_explained_variance": float(np.mean([x["explained_variance"] for x in diagnostics])),
            "mean_critic_prediction_mse": float(np.mean([x["critic_prediction_mse"] for x in diagnostics])),
        })
    return summaries, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    args = parser.parse_args()
    run = args.train_dir.resolve()
    output = ROOT / "artifacts/jcst_learnability" / ("optimization-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_train_dir": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "audit_code_sha256": sha(Path(__file__))}
    try:
        summaries, manifest = audit(run)
        with (output / "per_root.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        report.update(status="PASS", roots=len(summaries),
                      updates_scanned=sum(x["updates"] for x in summaries),
                      sampled_telemetry_rows=sum(x["sampled_updates"] for x in summaries),
                      any_zero_actor_update=any(x["fraction_actor_update_below_1e-9"] > 0 for x in summaries),
                      any_zero_critic_update=any(x["fraction_critic_update_below_1e-9"] > 0 for x in summaries))
    except BaseException as exc:
        report.update(status="FAIL", failure=repr(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
