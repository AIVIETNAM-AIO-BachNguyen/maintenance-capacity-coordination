"""Freeze eligible L2 DEV selection and independent analysis before confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from jcst_learnability_l2 import ROOT, sha

FROZEN_LATER_CODE = (
    "experiments/jcst_learnability_confirmation_audit_l2.py",
    "experiments/jcst_learnability_analysis_l2.py",
    "experiments/jcst_learnability_stats_audit_l2.py",
    "experiments/jcst_learnability_report_l2.py",
    "experiments/jcst_learnability_optimization_audit_l2.py",
    "experiments/jcst_learnability_inference_benchmark_l2.py",
)


def checked_artifact(path: Path, *, status: str, source_hash: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != status or data.get("source_manifest_sha256", data.get("source_manifest_hash")) != source_hash:
        raise ValueError(f"required artifact incomplete or wrong source: {path}")
    return data


def build_lock(train: Path, audit: Path, selection_file: Path,
               report: Path, optimization_audit: Path,
               inference_benchmark: Path) -> dict:
    manifest = json.loads((train / "manifest.json").read_text())
    source_hash = sha(train / "manifest.json")
    if (manifest["status"] != "COMPLETE" or manifest["stage"] != "train_dev"
            or len(manifest["completed_trials"]) != 30
            or len(manifest["completed_evaluations"]) != 240
            or len(manifest["completed_controls"]) != 8):
        raise ValueError("full TRAIN/DEV panel incomplete")
    checked_artifact(audit, status="PASS", source_hash=source_hash)
    selected = checked_artifact(selection_file, status="AUDIT_PASS", source_hash=source_hash)
    if not selected.get("dev_eligible"):
        raise ValueError("prelocked DEV eligibility gate did not open")
    checked_artifact(report, status="COMPLETE", source_hash=source_hash)
    checked_artifact(optimization_audit, status="PASS", source_hash=source_hash)
    checked_artifact(inference_benchmark, status="COMPLETE", source_hash=source_hash)
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"scientific source hash changed: {name}")
    if sha(ROOT / "docs/jcst_learnability_protocol_l2.md") != manifest["protocol_hash"]:
        raise ValueError("prelocked protocol changed")
    started = datetime.fromisoformat(manifest["started_utc"])
    elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
    if elapsed_hours >= 72:
        raise TimeoutError("72-hour campaign cap exhausted before confirmation")
    return {
        "status": "PRECONFIRMATION_LOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_eligible": True,
        "eligible_arm_cases": selected["eligible_arm_cases"],
        "selected": selected["selected"],
        "train_dir": str(train.relative_to(ROOT)),
        "train_manifest_sha256": source_hash,
        "selection_file": str(selection_file.relative_to(ROOT)),
        "selection_sha256": sha(selection_file),
        "train_audit_file": str(audit.relative_to(ROOT)),
        "train_audit_sha256": sha(audit),
        "train_report_file": str(report.relative_to(ROOT)),
        "train_report_sha256": sha(report),
        "optimization_audit_file": str(optimization_audit.relative_to(ROOT)),
        "optimization_audit_sha256": sha(optimization_audit),
        "inference_benchmark_file": str(inference_benchmark.relative_to(ROOT)),
        "inference_benchmark_sha256": sha(inference_benchmark),
        "protocol_sha256": manifest["protocol_hash"],
        "source_sha256": manifest["source_hashes"],
        "stage3_runner_sha256": sha(ROOT / "experiments/jcst_learnability_stage3_l2.py"),
        "analysis_and_audit_sha256": {name: sha(ROOT / name) for name in FROZEN_LATER_CODE},
        "interpretation_amendment_sha256": sha(ROOT / "docs/jcst_learnability_interpretation_amendment_l2_20261026.md"),
        "confirmation_bank": "confirmation",
        "confirmation_physical_seeds_per_case": 128,
        "stochastic_action_seeds_per_physical": 4,
        "campaign_elapsed_hours_at_lock": elapsed_hours,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--optimization-audit", type=Path, required=True)
    parser.add_argument("--inference-benchmark", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in (args.train_dir, args.train_audit,
                                         args.selection, args.report,
                                         args.optimization_audit,
                                         args.inference_benchmark)]
    frozen = build_lock(*paths)
    output = ROOT / "artifacts/jcst_learnability" / ("preconfirmation-lock-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    with output.open("x") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
