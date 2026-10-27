"""L2 confirmation/replication launcher; keeps the frozen TRAIN/DEV runner intact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import torch

import jcst_learnability_l2 as base


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(base.ROOT))


def load_lock(lock_file: Path, train: Path, selection_file: Path) -> tuple[dict, dict]:
    lock = json.loads(lock_file.read_text())
    train_manifest = base.source_manifest_check(train)
    selection = json.loads(selection_file.read_text())
    if lock["status"] != "PRECONFIRMATION_LOCKED" or not lock["dev_eligible"]:
        raise ValueError("preconfirmation gate not locked/eligible")
    if (lock["train_manifest_sha256"] != base.sha(train / "manifest.json") or
            lock["selection_sha256"] != base.sha(selection_file) or
            selection["status"] != "AUDIT_PASS" or not selection["dev_eligible"]):
        raise ValueError("selection/source provenance changed")
    if lock["stage3_runner_sha256"] != base.sha(Path(__file__)):
        raise ValueError("confirmation launcher changed")
    for name, digest in lock["analysis_and_audit_sha256"].items():
        if base.sha(base.ROOT / name) != digest:
            raise ValueError(f"analysis/audit changed: {name}")
    return train_manifest, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("confirmation", "replication"), required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preconfirmation-lock", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stage = args.stage
    train = args.train_dir.resolve()
    selection_file = args.selection.resolve()
    lock_file = args.preconfirmation_lock.resolve()
    train_manifest, selection = load_lock(lock_file, train, selection_file)
    confirmation = None
    if stage == "replication":
        if not args.confirmation_dir:
            parser.error("replication requires --confirmation-dir")
        confirmation = args.confirmation_dir.resolve()
        analysis = json.loads((confirmation / "analysis.json").read_text())
        if (analysis["status"] != "AUDIT_PASS" or not analysis["replication_eligible"] or
                analysis["train_manifest_sha256"] != base.sha(train / "manifest.json") or
                analysis["selection_sha256"] != base.sha(selection_file)):
            raise ValueError("audited confirmation gate not eligible")
    out = base.new_dir(stage)
    normalized = argparse.Namespace(train_dir=relative(train),
                                    confirmation_dir=relative(confirmation) if confirmation else None)
    manifest = base.manifest(stage, out, normalized)
    manifest.update(selection_file=relative(selection_file),
                    selection_sha256=base.sha(selection_file),
                    preconfirmation_lock=relative(lock_file),
                    preconfirmation_lock_sha256=base.sha(lock_file),
                    stage3_runner_sha256=base.sha(Path(__file__)))
    base.atomic_json(out / "manifest.json", manifest)
    started = time.monotonic()
    campaign_started = datetime.fromisoformat(train_manifest["started_utc"]).timestamp()
    try:
        bank = "confirmation" if stage == "confirmation" else "rep_eval"
        roots = range(5) if stage == "confirmation" else range(5, 10)
        model_source = train if stage == "confirmation" else out
        if stage == "replication":
            for case in base.CASES:
                for arm in base.ARMS:
                    checkpoint = int(selection["selected"][case][arm]["checkpoint"])
                    checkpoints = (24000, checkpoint) if checkpoint > 24000 else (24000,)
                    for root in roots:
                        trial = base.train_trial(out, stage, case, arm, root, checkpoints,
                                                 started, campaign_started)
                        manifest["completed_trials"].append(trial)
                        base.atomic_json(out / "manifest.json", manifest)
        for case in base.CASES:
            for arm in base.ARMS:
                chosen = selection["selected"][case][arm]
                for root in roots:
                    for checkpoint in sorted({24000, int(chosen["checkpoint"])}):
                        result = base.eval_model(out, model_source, bank, case, arm, root,
                                                 checkpoint, chosen["mode"], 128, 4,
                                                 started, campaign_started)
                        manifest["completed_evaluations"].append(result)
                        base.atomic_json(out / "manifest.json", manifest)
        manifest["completed_controls"] = base.eval_controls(out, bank, 128,
                                                              started, campaign_started)
        manifest["status"] = "COMPLETE"
    except BaseException as exc:
        manifest["status"] = "INCOMPLETE"
        manifest["failure"] = repr(exc)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        base.atomic_json(out / "manifest.json", manifest)
        print(out)


if __name__ == "__main__":
    main()
