"""Fresh-root matched S/F/A sample-efficiency experiment on unchanged JCST/1.1.

Reuses the frozen L2 collector's training/evaluation routines with explicit L3 seed
and resource functions injected into that module in this process only. No L2 file
or running L2 process is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch

import jcst_learnability_l2 as base
from ht_pdm_fjsp.component_service_v1 import config_json
from ht_pdm_fjsp.jcst_characterization_v1 import case_config

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/jcst_matched_architecture_protocol_l3.md"
ARTIFACTS = ROOT / "artifacts/jcst_matched_architecture"
L2_REPLICATION = ROOT / "artifacts/jcst_learnability/replication-20261027T091525.639861Z/manifest.json"
PARALLEL_AMENDMENT = ROOT / "docs/jcst_matched_architecture_parallel_amendment_l3_20261027.md"
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CHECKPOINTS = (24000, 96000, 384000)
TRACE_EPISODES = {0, 1, 2, 3, 499, 1999, 7997, 7998, 7999}
SOURCES = (
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/component_framework_v1.py",
    "src/ht_pdm_fjsp/jcst_characterization_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_l2.py",
    "experiments/jcst_learner_campaign_v1.py",
    "experiments/jcst_learnability_l2.py",
    "experiments/jcst_matched_architecture_l3.py",
)


def physical_seed(bank: str, case: str, root: int, episode: int) -> int:
    if case not in CASES:
        raise ValueError("case")
    ci = CASES.index(case)
    if bank == "train":
        if not 0 <= root < 10 or not 0 <= episode < 8000:
            raise ValueError("TRAIN seed index")
        # The caller's arm is supplied by a process-scoped context variable.
        arm = _training_arm[0]
        if arm not in ARMS:
            raise ValueError("TRAIN arm unset")
        return (2**52 + 10**13 + ci * 10**12 + ARMS.index(arm) * 10**11
                + root * 10**8 + episode)
    if bank in ("eval", "engineering"):
        if not 0 <= episode < 128:
            raise ValueError("evaluation seed index")
        offset = 2 * 10**13 if bank == "eval" else 3 * 10**13
        return 2**52 + offset + ci * 10**12 + episode
    raise ValueError("bank")


_training_arm = [None]


def torch_seed(bank: str, case: str, arm: str, root: int) -> int:
    if bank not in ("train", "engineering") or case not in CASES or arm not in ARMS or not 0 <= root < 10:
        raise ValueError("torch seed role")
    offset = 10**13 if bank == "train" else 3 * 10**13
    return (2**51 + offset + CASES.index(case) * 10**12
            + ARMS.index(arm) * 10**11 + root * 10**8)


def action_seed(bank: str, case: str, arm: str, root: int, checkpoint: int,
                physical_index: int, repetition: int) -> int:
    if bank != "eval" or case not in CASES or arm not in ARMS or not 0 <= root < 10 or not 0 <= physical_index < 128 or not 0 <= repetition < 4:
        raise ValueError("action seed role")
    key = f"JCST-L3:{case}:{arm}:{root}:{checkpoint}:{physical_index}:{repetition}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def check_caps(out: Path, started: float, campaign_started: float) -> None:
    if time.monotonic() - started > 30 * 3600:
        raise TimeoutError("L3 30-hour cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024 * 1024:
        raise MemoryError("L3 8-GiB process RAM cap")
    if sum(p.stat().st_size for p in ARTIFACTS.rglob("*") if p.is_file()) > 5 * 1024**3:
        raise OSError("L3 5-GiB artifact cap")


def configure_base() -> None:
    # Explicit process-local adapter for the frozen, already audited L2 routines.
    base.physical_seed = physical_seed
    base.torch_seed = torch_seed
    base.action_seed = action_seed
    base.check_caps = check_caps
    base.TRACE_EPISODES = TRACE_EPISODES
    base.ARTIFACTS = ARTIFACTS


def new_dir(stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ARTIFACTS / f"{stage}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(stage: str, out: Path, closure: dict | None = None, parallel_override: dict | None = None) -> dict:
    return {
        "schema": "JCST-matched-architecture-L3", "stage": stage, "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(out.relative_to(ROOT)), "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "platform": platform.platform(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "tqdm")},
        "source_hashes": {name: base.sha(ROOT / name) for name in SOURCES},
        "protocol_hash": base.sha(PROTOCOL),
        "contract_hash": base.sha(ROOT / "docs/joint_component_service_contract_v1.md"),
        "config_hashes": {case: hashlib.sha256(config_json(case_config(case)[0]).encode()).hexdigest()
                          for case in CASES},
        "parameters": {"H": 48, "B": 16, "cases": CASES, "arms": ARMS,
                       "checkpoints": CHECKPOINTS, "roots": list(range(10)),
                       "evaluation_physical_seeds": 128,
                       "stochastic_repetitions": 4,
                       "primary": {"checkpoint": 96000, "mode": "greedy",
                                   "family_size": 4, "practical_margin": -10},
                       "ppo": {"adam_lr": .0003, "epochs": 4, "minibatch": 128,
                               "clip": .2, "entropy_coefficient": .01,
                               "gradient_clip": .5, "episodes_per_update": 25,
                               "reward_divisor": 100, "gamma": 1, "gae_lambda": .95},
                       "threads": 1, "memory_cap_kib": 8 * 1024 * 1024,
                       "disk_cap_bytes": 5 * 1024**3, "hours_cap": 30},
        "seed_roles": {"physical_base": 2**52, "torch_base": 2**51,
                       "train_bank": 1, "eval_bank": 2, "engineering_bank": 3},
        "l2_replication_manifest": str(L2_REPLICATION.relative_to(ROOT)),
        "l2_closure": closure,
        "parallel_l2_override": parallel_override,
        "completed_trials": [], "completed_evaluations": [],
        "completed_controls": [], "failure": None,
    }


def require_l2_closed(replay_path: Path, stats_path: Path, findings_path: Path) -> dict:
    data = json.loads(L2_REPLICATION.read_text())
    if data["status"] == "RUNNING":
        raise RuntimeError("L2 replication still active; L3 scientific worker must wait")
    if data["status"] != "COMPLETE":
        raise RuntimeError(f"L2 replication terminal status {data['status']}; resolve L2 first")
    replay = json.loads(replay_path.read_text())
    stats = json.loads(stats_path.read_text())
    identity = str(L2_REPLICATION.parent.relative_to(ROOT))
    digest = base.sha(L2_REPLICATION)
    if (replay.get("status") != "PASS" or replay.get("stage") != "replication" or
            replay.get("source_run_dir") != identity or
            replay.get("source_manifest_hash") != digest):
        raise RuntimeError("L2 independent replication replay not PASS for this run")
    if (stats.get("status") != "PASS" or stats.get("run_dir") != identity or
            stats.get("run_manifest_sha256") != digest):
        raise RuntimeError("L2 independent replication statistics not PASS for this run")
    if not findings_path.is_file() or identity not in findings_path.read_text():
        raise RuntimeError("L2 replication findings must identify the completed run")
    return {"replay_report": str(replay_path.relative_to(ROOT)),
            "replay_sha256": base.sha(replay_path),
            "stats_report": str(stats_path.relative_to(ROOT)),
            "stats_sha256": base.sha(stats_path),
            "findings": str(findings_path.relative_to(ROOT)),
            "findings_sha256": base.sha(findings_path)}


def require_parallel_l2_override() -> dict:
    data = json.loads(L2_REPLICATION.read_text())
    if data["status"] != "RUNNING" or data.get("stage") != "replication" or data.get("failure") is not None:
        raise RuntimeError("parallel override requires a healthy RUNNING L2 replication")
    if subprocess.run(["tmux", "has-session", "-t", "jcst-l2-rep-20261027"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("L2 tmux session is absent")
    if not PARALLEL_AMENDMENT.is_file():
        raise RuntimeError("pre-outcome parallel amendment missing")
    return {"amendment": str(PARALLEL_AMENDMENT.relative_to(ROOT)),
            "amendment_sha256": base.sha(PARALLEL_AMENDMENT),
            "l2_manifest": str(L2_REPLICATION.relative_to(ROOT)),
            "l2_manifest_sha256_at_launch": base.sha(L2_REPLICATION),
            "l2_status_at_launch": "RUNNING",
            "host_logical_cpus": 16,
            "concurrent_scientific_cpu_workers_cap": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "collect"), required=True)
    parser.add_argument("--l2-replay-audit", type=Path)
    parser.add_argument("--l2-stats-audit", type=Path)
    parser.add_argument("--l2-findings", type=Path)
    parser.add_argument("--parallel-l2-override", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    closure = parallel_override = None
    if args.stage == "collect":
        if args.parallel_l2_override:
            if any((args.l2_replay_audit, args.l2_stats_audit, args.l2_findings)):
                parser.error("parallel override cannot use L2 closure inputs")
            parallel_override = require_parallel_l2_override()
        else:
            if not all((args.l2_replay_audit, args.l2_stats_audit, args.l2_findings)):
                parser.error("collect requires both PASSed L2 audits and L2 replication findings")
            closure = require_l2_closed(args.l2_replay_audit.resolve(),
                                        args.l2_stats_audit.resolve(),
                                        args.l2_findings.resolve())
    configure_base()
    out = new_dir(args.stage)
    data = manifest(args.stage, out, closure, parallel_override)
    base.atomic_json(out / "manifest.json", data)
    started = time.monotonic()
    try:
        roots = range(1) if args.stage == "smoke" else range(10)
        checkpoints = (48, 96) if args.stage == "smoke" else CHECKPOINTS
        for case in CASES:
            for arm in ARMS:
                for root in roots:
                    _training_arm[0] = arm
                    trial = base.train_trial(out, "smoke" if args.stage == "smoke" else "train_dev",
                                             case, arm, root, checkpoints, started, time.time())
                    _training_arm[0] = None
                    data["completed_trials"].append(trial)
                    base.atomic_json(out / "manifest.json", data)
        bank = "engineering" if args.stage == "smoke" else "eval"
        count = 2 if args.stage == "smoke" else 128
        combinations = ((96, "greedy"),) if args.stage == "smoke" else (
            (24000, "greedy"), (96000, "greedy"),
            (384000, "greedy"), (384000, "stochastic"))
        for case in CASES:
            for arm in ARMS:
                for root in roots:
                    for checkpoint, mode in combinations:
                        result = base.eval_model(out, out, bank, case, arm, root,
                                                 checkpoint, mode, count, 4,
                                                 started, time.time())
                        data["completed_evaluations"].append(result)
                        base.atomic_json(out / "manifest.json", data)
        data["completed_controls"] = base.eval_controls(out, bank, count,
                                                        started, time.time(), smoke=False)
        data["status"] = "COMPLETE"
    except BaseException as exc:
        data["status"] = "INCOMPLETE"
        data["failure"] = repr(exc)
        raise
    finally:
        data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        data["elapsed_seconds"] = time.monotonic() - started
        data["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        base.atomic_json(out / "manifest.json", data)
        print(out)


if __name__ == "__main__":
    main()
