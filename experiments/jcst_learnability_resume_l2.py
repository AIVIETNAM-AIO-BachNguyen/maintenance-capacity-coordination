"""Integrity-checked continuation of an interrupted L2 train/DEV collection.

The original scientific collector and its recorded source hash stay unchanged.
Only complete, independently audited roots are retained. An interrupted root is
preserved under partials/ and trained anew from its original root seed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import torch

import jcst_learnability_l2 as run
from jcst_learnability_trial_audit_l2 import audit_trial


def expected_keys() -> list[tuple[str, str, int]]:
    return [(case, arm, root) for case in run.CASES for arm in run.ARMS for root in range(5)]


def verify(run_dir: Path, data: dict) -> list[tuple[str, str, int]]:
    if data["schema"] != "JCST-learnability-L2" or data["stage"] != "train_dev":
        raise ValueError("not an L2 train/DEV run")
    if data["status"] != "RUNNING" or data["completed_evaluations"] or data["completed_controls"]:
        raise ValueError("only interrupted training before DEV can use this recovery")
    if data["protocol_hash"] != run.sha(run.PROTOCOL):
        raise ValueError("protocol hash changed")
    if data["contract_hash"] != run.sha(run.ROOT / "docs/joint_component_service_contract_v1.md"):
        raise ValueError("contract hash changed")
    for name, digest in data["source_hashes"].items():
        if digest != run.sha(run.ROOT / name):
            raise ValueError(f"scientific source changed: {name}")
    expected = expected_keys()
    actual = [(x["case"], x["arm"], x["root"]) for x in data["completed_trials"]]
    if actual != expected[:len(actual)] or len(actual) >= len(expected):
        raise ValueError("completed roots are not a strict, unique prefix")
    for case, arm, root in actual:
        result = audit_trial(run_dir, case, arm, root)
        if result["status"] != "PASS":
            raise ValueError(f"completed trial audit failed: {case}/{arm}/{root}")
    partial = run_dir / "trials" / actual_next_path(expected[len(actual)])
    known = {run_dir / "trials" / actual_next_path(key) for key in actual}
    observed = {path.parent for path in (run_dir / "trials").rglob("model.json")}
    if observed - known - {partial}:
        raise ValueError("unexpected trial directory; refusing possible duplicate records")
    if not partial.is_dir():
        raise ValueError("expected interrupted trial directory missing")
    if any((run_dir / "evaluations").rglob("*")) if (run_dir / "evaluations").exists() else False:
        raise ValueError("unexpected evaluation records")
    return expected[len(actual):]


def actual_next_path(key: tuple[str, str, int]) -> Path:
    case, arm, root = key
    return Path(case) / arm / str(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reason", default="interrupted worker; preserve partial and restart this root")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(run.ARTIFACTS.resolve()):
        raise ValueError("run is outside L2 artifact root")
    manifest_path = run_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    remaining = verify(run_dir, data)
    old_hash = run.sha(manifest_path)
    partial = run_dir / "trials" / actual_next_path(remaining[0])
    preserved = run_dir / "partials" / actual_next_path(remaining[0])
    if preserved.exists():
        preserved = preserved.with_name(preserved.name + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if preserved.exists():
        raise FileExistsError(preserved)
    preserved.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(preserved)
    data.setdefault("recovery_events", []).append({
        "utc": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        "previous_manifest_sha256": old_hash,
        "recovery_source_sha256": run.sha(Path(__file__)),
        "preserved_partial": str(preserved.relative_to(run.ROOT)),
        "independently_audited_complete_roots": len(data["completed_trials"]),
        "remaining_roots": len(remaining),
    })
    run.atomic_json(manifest_path, data)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    campaign_started = datetime.fromisoformat(data["started_utc"]).timestamp()
    try:
        for case, arm, root in remaining:
            result = run.train_trial(run_dir, "train_dev", case, arm, root,
                                     run.CHECKPOINTS, started, campaign_started)
            data["completed_trials"].append(result)
            run.atomic_json(manifest_path, data)
        for case in run.CASES:
            for arm in run.ARMS:
                for checkpoint in run.CHECKPOINTS:
                    for root in range(5):
                        for mode in ("greedy", "stochastic"):
                            result = run.eval_model(run_dir, run_dir, "dev", case, arm,
                                                    root, checkpoint, mode, 16, 4,
                                                    started, campaign_started)
                            data["completed_evaluations"].append(result)
                            run.atomic_json(manifest_path, data)
        data["completed_controls"] = run.eval_controls(run_dir, "dev", 16,
                                                        started, campaign_started)
        data["status"] = "COMPLETE"
    except BaseException as exc:
        data["status"] = "INCOMPLETE"
        data["failure"] = repr(exc)
        raise
    finally:
        data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        data["elapsed_seconds"] = time.time() - campaign_started
        data["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        run.atomic_json(manifest_path, data)
        print(run_dir)


if __name__ == "__main__":
    main()
