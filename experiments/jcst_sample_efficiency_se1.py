"""SE1 collector: frozen JCST/1.1 PPO with independent seeds and phase clocks.

This module only wraps calls in the L1/L2 implementation; it does not alter their
learning operations. Importing it does not start an experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import jcst_learnability_l2 as base
import jcst_learner_campaign_v1 as campaign
from ht_pdm_fjsp.component_controllers_v1 import Threshold
from ht_pdm_fjsp.component_service_v1 import config_json
from ht_pdm_fjsp.jcst_characterization_v1 import case_config

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/jcst_sample_efficiency"
PROTOCOL = ROOT / "docs/jcst_sample_efficiency_protocol_se1.md"
DESIGN_LOCK = ROOT / "docs/jcst_sample_efficiency_design_lock_se1_20261028_v2.json"
AMENDMENT = ROOT / "docs/jcst_sample_efficiency_parallel_amendment_se1_20261028.md"
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CHECKPOINTS = (24000, 48000, 72000, 96000, 144000, 192000, 288000, 384000)
TRACE_EPISODES = {0, 1, 2, 3, 499, 999, 1499, 1999, 2999, 3999, 5999, 7997, 7998, 7999}
SOURCE_FILES = (
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/component_framework_v1.py",
    "src/ht_pdm_fjsp/jcst_characterization_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_l2.py",
    "experiments/jcst_learner_campaign_v1.py",
    "experiments/jcst_learnability_l2.py",
    "experiments/jcst_sample_efficiency_se1.py",
    "experiments/jcst_sample_efficiency_audit_se1.py",
    "experiments/jcst_sample_efficiency_analyze_se1.py",
    "experiments/jcst_sample_efficiency_stats_audit_se1.py",
    "experiments/jcst_sample_efficiency_pipeline_se1.py",
)
H, B = 48, 16
_arm_context = [None]
_phase = {key: 0.0 for key in ("catalogue", "encoding_mask", "inference", "simulator")}
_installed = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_seed(bank: str, case: str, root: int, episode: int) -> int:
    ci = CASES.index(case)
    if bank == "train":
        arm = _arm_context[0]
        if arm not in ARMS or not 0 <= root < 10 or not 0 <= episode < 8000:
            raise ValueError("TRAIN seed role/index")
        return 2**54 + 10**13 + ci * 10**12 + ARMS.index(arm) * 10**11 + root * 10**8 + episode
    if bank == "engineering" and _arm_context[0] in ARMS:
        if not 0 <= root < 10 or not 0 <= episode < 8000:
            raise ValueError("engineering TRAIN seed index")
        return (2**54 + 3 * 10**13 + ci * 10**12
                + ARMS.index(_arm_context[0]) * 10**11 + root * 10**8 + episode)
    if bank in ("eval", "engineering", "calibration"):
        if not 0 <= episode < (128 if bank != "engineering" else 10000):
            raise ValueError("evaluation seed index")
        offset = {"eval": 2, "engineering": 3, "calibration": 4}[bank] * 10**13
        return 2**54 + offset + ci * 10**12 + episode
    raise ValueError(bank)


def torch_seed(bank: str, case: str, arm: str, root: int) -> int:
    if bank not in ("train", "engineering") or arm not in ARMS or not 0 <= root < 10:
        raise ValueError("Torch seed role/index")
    return (2**53 + (1 if bank == "train" else 3) * 10**13 + CASES.index(case) * 10**12
            + ARMS.index(arm) * 10**11 + root * 10**8)


def action_seed(bank: str, case: str, arm: str, root: int, checkpoint: int,
                physical_index: int, repetition: int) -> int:
    if bank not in ("eval", "engineering") or case not in CASES or arm not in ARMS:
        raise ValueError("action seed role")
    key = f"JCST-SE1:{case}:{arm}:{root}:{checkpoint}:{physical_index}:{repetition}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def configure_base() -> None:
    base.physical_seed = physical_seed
    base.torch_seed = torch_seed
    base.action_seed = action_seed
    base.ARTIFACTS = ARTIFACTS


def _clocked(name, fn):
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _phase[name] += time.perf_counter() - started
    return wrapped


def install_phase_clocks() -> None:
    global _installed
    if _installed:
        return
    for module, attr, name in (
        (campaign, "observe", "catalogue"),
        (campaign, "encode", "encoding_mask"),
        (campaign, "choose", "inference"),
        (campaign, "advance", "simulator"),
        (base, "observe", "catalogue"),
        (base, "encode", "encoding_mask"),
        (base, "policy_diagnostic", "inference"),
        (base, "advance", "simulator"),
    ):
        setattr(module, attr, _clocked(name, getattr(module, attr)))
    _installed = True


def phase_snapshot() -> dict[str, float]:
    return dict(_phase)


def phase_delta(before: dict[str, float]) -> dict[str, float]:
    return {key: _phase[key] - before[key] for key in _phase}


def cap_check(run: Path, deadline: float, *, disk: bool = False) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("SE1 elapsed cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024 * 1024:
        raise MemoryError("SE1 process RSS cap")
    if disk and sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 20 * 1024**3:
        raise OSError("SE1 artifact disk cap")


def train_trial(run: Path, case: str, arm: str, root: int, checkpoints: tuple[int, ...],
                deadline: float, smoke: bool) -> dict:
    cfg, _ = case_config(case)
    universe = base.action_universe(cfg)
    bank = "engineering" if smoke else "train"
    _arm_context[0] = arm
    policy = base.policy_for(arm, cfg, universe, bank, case, root)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    trial = run / "trials" / case / arm / str(root)
    trial.mkdir(parents=True, exist_ok=False)
    base.atomic_json(trial / "model.json", {"case": case, "arm": arm, "root": root,
        "parameters": policy.parameter_counts(),
        "universe": [[list(edge) for edge in action] for action in universe],
        "config": json.loads(config_json(cfg))})
    episodes = samples = updates_done = 0
    batch = []
    cost_sum = update_seconds = 0.0
    start = time.perf_counter()
    cpu_start = time.process_time()
    phase_start = phase_snapshot()
    trace_indices = set(range(checkpoints[-1] // H)) if smoke else TRACE_EPISODES
    with open(trial / "train.jsonl", "x", encoding="utf-8") as stream, \
         open(trial / "updates.jsonl", "x", encoding="utf-8") as update_stream, \
         tqdm(total=checkpoints[-1] // H, desc=f"SE1 {case}/{arm}/{root} TRAIN", leave=False) as progress:
        for checkpoint in checkpoints:
            while samples < checkpoint:
                seed = physical_seed(bank, case, root, episodes)
                record, sequence = base.l1_episode(cfg, universe, seed, None, train=True, policy=policy)
                row = base.compact_train(record, case=case, arm=arm, root=root,
                                         episode_index=episodes, trace=episodes in trace_indices)
                row["bank"] = bank
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                cost_sum += record["total_cost"]
                batch.extend(sequence)
                episodes += 1
                samples += H
                progress.update(1)
                if len(batch) >= 25 * H:
                    updates_done += 1
                    sampled = updates_done == 1 or updates_done % 20 == 0
                    began = time.perf_counter()
                    metrics = (base.ppo_update_diagnostic(policy, optimizer, batch) if sampled else
                               base.ppo_update(policy, optimizer, batch))
                    update_seconds += time.perf_counter() - began
                    update_stream.write(json.dumps({"samples": samples, "update": updates_done,
                        "sampled_telemetry": sampled, **metrics}) + "\n")
                    batch.clear()
                    stream.flush()
                    update_stream.flush()
                    cap_check(run, deadline, disk=updates_done % 20 == 0)
                    base.atomic_json(run / "heartbeat.json", {"case": case, "arm": arm,
                        "root": root, "samples": samples, "updates": updates_done,
                        "time_utc": datetime.now(timezone.utc).isoformat()})
            if batch:
                raise AssertionError("checkpoint is not a PPO update boundary")
            checkpoint_path = trial / f"checkpoint-{checkpoint}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            elapsed = time.perf_counter() - start
            phase = phase_delta(phase_start)
            phase["ppo_updates"] = update_seconds
            phase["checkpoint_logging_overhead"] = max(0.0, elapsed - sum(phase.values()))
            base.atomic_json(trial / f"checkpoint-{checkpoint}.json", {
                "samples": samples, "episodes": episodes, "updates": updates_done,
                "mean_train_episode_cost_to_date": cost_sum / episodes,
                "elapsed_training_seconds": elapsed,
                "cpu_training_seconds": time.process_time() - cpu_start,
                "phase_seconds": phase,
                "weights_sha256": sha(checkpoint_path)})
    _arm_context[0] = None
    return {"case": case, "arm": arm, "root": root, "samples": samples,
        "episodes": episodes, "updates": updates_done,
        "training_seconds": time.perf_counter() - start,
        "cpu_training_seconds": time.process_time() - cpu_start,
        "parameters": policy.parameter_counts()}


def calibrate(run: Path, count: int, deadline: float, smoke: bool) -> dict:
    results = {}
    for case in CASES:
        cfg, _ = case_config(case)
        universe = base.action_universe(cfg)
        out = run / "calibration" / case
        out.mkdir(parents=True, exist_ok=False)
        costs = []
        with open(out / "episodes.jsonl", "x", encoding="utf-8") as stream:
            for i in tqdm(range(count), desc=f"SE1 calibration {case}", leave=False):
                seed = physical_seed("engineering" if smoke else "calibration", case, 0, i)
                record = base.evaluated_episode(cfg, universe, seed, policy=None, controller=Threshold(5))
                record.update(case=case, arm="threshold5", bank="engineering" if smoke else "calibration",
                              physical_index=i)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                costs.append(record["total_cost"])
                cap_check(run, deadline)
        target = .95 * float(np.mean(costs))
        results[case] = {"mean_threshold5_cost": float(np.mean(costs)), "target_cost": target,
                         "physical_seeds": count, "rows_sha256": sha(out / "episodes.jsonl")}
    base.atomic_json(run / "target-lock.json", {"status": "TARGET_LOCKED", "smoke": smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": results})
    return results


def run_panel(run: Path, *, smoke: bool, deadline: float) -> dict:
    configure_base()
    install_phase_clocks()
    roots = range(2) if smoke else range(10)
    checkpoints = (1200, 2400) if smoke else CHECKPOINTS
    physical_count = 3 if smoke else 128
    target = calibrate(run, physical_count if smoke else 128, deadline, smoke)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["target_lock_sha256"] = sha(run / "target-lock.json")
    manifest["calibration"] = target
    base.atomic_json(run / "manifest.json", manifest)
    for case in CASES:
        for arm in ARMS:
            for root in roots:
                trial = train_trial(run, case, arm, root, checkpoints, deadline, smoke)
                manifest["completed_trials"].append(trial)
                base.atomic_json(run / "manifest.json", manifest)
    bank = "engineering" if smoke else "eval"
    combinations = (tuple((cp, "greedy") for cp in checkpoints)
                    + tuple((cp, "stochastic") for cp in ((checkpoints[-1],) if smoke else (96000, 384000))))
    for case in CASES:
        for arm in ARMS:
            for root in roots:
                for checkpoint, mode in combinations:
                    before = phase_snapshot()
                    began = time.perf_counter()
                    # base.eval_model checks caps through its process-local hook.
                    result = base.eval_model(run, run, bank, case, arm, root,
                                             checkpoint, mode, physical_count, 4,
                                             time.monotonic(), time.time())
                    result["elapsed_seconds"] = time.perf_counter() - began
                    result["phase_seconds"] = phase_delta(before)
                    manifest["completed_evaluations"].append(result)
                    base.atomic_json(run / "manifest.json", manifest)
                    cap_check(run, deadline, disk=True)
    # The inherited control collector hardcodes its own case loop but uses our
    # evaluation seeds. Its raw episodes are independently audited below.
    manifest["completed_controls"] = base.eval_controls(run, bank, physical_count,
                                                           time.monotonic(), time.time())
    base.atomic_json(run / "manifest.json", manifest)
    return manifest


def parallel_snapshot() -> dict:
    result = {}
    for label, script, manifest in (
        ("L2", "jcst_learnability_stage3_l2.py", ROOT / "artifacts/jcst_learnability/replication-20261027T091525.639861Z/manifest.json"),
        ("L3", "jcst_matched_architecture_l3.py", ROOT / "artifacts/jcst_matched_architecture/collect-20261027T094717.495781Z/manifest.json"),
    ):
        lines = subprocess.run(["pgrep", "-af", script], text=True, capture_output=True).stdout.splitlines()
        pids = [int(line.split(None, 1)[0]) for line in lines
                if ".venv/bin/python" in line and script in line]
        result[label] = {"pids": pids, "manifest": str(manifest.relative_to(ROOT)),
                         "manifest_status": json.loads(manifest.read_text())["status"] if manifest.exists() else "MISSING"}
    return result


def make_manifest(run: Path, smoke: bool) -> dict:
    design = json.loads(DESIGN_LOCK.read_text())
    if design["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("SE1 design-lock protocol hash changed")
    return {"schema": "JCST-SE1", "status": "RUNNING", "smoke": smoke,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run.relative_to(ROOT)), "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "platform": platform.platform(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "tqdm")},
        "source_hashes": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "protocol_hash": sha(PROTOCOL), "design_lock_hash": sha(DESIGN_LOCK),
        "scheduling_amendment_hash": sha(AMENDMENT),
        "prelaunch_lock_hash": sha(Path(os.environ["SE1_PRELAUNCH_LOCK"])) if not smoke else None,
        "parallel_at_launch": parallel_snapshot(),
        "contract_hash": sha(ROOT / "docs/joint_component_service_contract_v1.md"),
        "config_hashes": {case: hashlib.sha256(config_json(case_config(case)[0]).encode()).hexdigest()
                          for case in CASES},
        "parameters": {"H": H, "B": B, "cases": CASES, "arms": ARMS,
            "checkpoints": (1200, 2400) if smoke else CHECKPOINTS,
            "roots": list(range(2) if smoke else range(10)),
            "evaluation_physical_seeds": 3 if smoke else 128,
            "stochastic_repetitions": 4, "torch_threads": 1,
            "process_memory_cap_kib": 8 * 1024 * 1024,
            "artifact_disk_cap_bytes": 20 * 1024**3, "campaign_hours_cap": 72},
        "seed_roles": {"physical_base": 2**54, "torch_base": 2**53,
            "train_bank": 1, "eval_bank": 2, "engineering_bank": 3,
            "calibration_bank": 4, "action_namespace": "JCST-SE1"},
        "completed_trials": [], "completed_evaluations": [],
        "completed_controls": [], "failure": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not args.smoke and "SE1_PRELAUNCH_LOCK" not in os.environ:
        parser.error("scientific SE1 collection requires the prelaunch-locked pipeline")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run_dir.resolve() if args.run_dir else ARTIFACTS / (
        "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    run.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(run, args.smoke)
    base.atomic_json(run / "manifest.json", manifest)
    deadline = float(os.environ["SE1_DEADLINE_MONOTONIC"]) if "SE1_DEADLINE_MONOTONIC" in os.environ else time.monotonic() + (3600 if args.smoke else 72 * 3600)
    base.check_caps = lambda out, started, campaign_started: cap_check(run, deadline)
    try:
        run_panel(run, smoke=args.smoke, deadline=deadline)
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["status"] = "COLLECTED"
    except BaseException as exc:
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["status"] = "INCOMPLETE"
        manifest["failure"] = repr(exc)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        base.atomic_json(run / "manifest.json", manifest)
        print(run)


if __name__ == "__main__":
    main()
