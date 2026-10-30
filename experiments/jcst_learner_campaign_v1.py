"""Prelocked JCST/1.1 S/F/A PPO campaign. Importing this module starts no run."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from ht_pdm_fjsp.component_service_v1 import config_json, initial_state
from ht_pdm_fjsp.component_controllers_v1 import (
    Corrective, BoundedRollout, RiskMatching, Threshold, Wait, observe,
)
from ht_pdm_fjsp.component_framework_v1 import advance
from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import (
    Policy, Transition, action_universe, choose, encode, ppo_update,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/jcst_framework_protocol_v1.md"
SOURCES = [
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/component_framework_v1.py",
    "src/ht_pdm_fjsp/jcst_characterization_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_v1.py",
    "experiments/jcst_learner_campaign_v1.py",
    "experiments/jcst_learner_audit_v1.py",
]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
ROOTS = range(5)
CHECKPOINTS = (6000, 12000, 24000)
H, B = 48, 16
REWARD_SCALE = 100.0
BASE_SEED = 1 << 42
BANK = {"engineering": 0, "train": 1, "dev": 2, "confirmation": 3,
        "rep_train": 4, "rep_eval": 5, "engineering_eval": 6}
MAX_HOURS = {"smoke": 0.5, "train_dev": 48, "confirmation": 12,
             "replication": 12}
DISK_CAP = 30 * 1024**3


def seed(role: str, case: str, root: int = 0, episode: int = 0) -> int:
    if role not in BANK or case not in CASES or not 0 <= root < 5 or not 0 <= episode < 10000:
        raise ValueError("seed role/index")
    return BASE_SEED + BANK[role] * 100_000_000 + CASES.index(case) * 10_000_000 + root * 100_000 + episode


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def provenance(stage: str, output: Path, args) -> dict:
    return {
        "schema": "JCST-framework-v1", "status": "RUNNING", "stage": stage,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(output.relative_to(ROOT)), "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "platform": platform.platform(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "tqdm", "gymnasium", "pytest")},
        "source_hashes": {name: sha(ROOT / name) for name in SOURCES},
        "protocol_hash": sha(PROTOCOL),
        "contract_hash": sha(ROOT / "docs/joint_component_service_contract_v1.md"),
        "config_hashes": {name: hashlib.sha256(config_json(case_config(name)[0]).encode()).hexdigest()
                          for name in CASES},
        "parameters": {"cases": CASES, "arms": ARMS, "roots": list(ROOTS),
                       "checkpoint_samples": CHECKPOINTS, "H": H, "B": B,
                       "gamma": 1.0, "gae_lambda": .95, "reward_divisor": REWARD_SCALE,
                       "optimizer": "Adam", "learning_rate": 3e-4, "epochs": 4,
                       "minibatch": 128, "clip": .2, "entropy_coefficient": .01,
                       "episodes_per_update": 25, "worker_count": 1,
                       "torch_threads": 1, "disk_cap_bytes": DISK_CAP},
        "seed_roles": {"base": BASE_SEED, "banks": BANK, "formula": "base + bank*100000000 + case*10000000 + root*100000 + episode"},
        "train_dir": args.train_dir, "confirmation_dir": args.confirmation_dir,
        "completed_trials": [], "completed_episodes": 0, "failure": None,
    }


def new_dir(stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ROOT / "artifacts" / "jcst_framework" / f"{stage}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_limits(run_dir: Path, started: float, stage: str) -> None:
    if time.monotonic() - started > MAX_HOURS[stage] * 3600:
        raise TimeoutError(f"{stage} wall timeout")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024 * 1024:
        raise RuntimeError("8 GiB process memory cap exceeded")
    if sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) > DISK_CAP:
        raise RuntimeError("disk cap exceeded")


def policy_seed(role: str, case: str, root: int) -> int:
    return seed(role, case, root) % (2**31 - 1)


def make_policy(arm: str, config, universe, role: str, case: str, root: int) -> Policy:
    torch.manual_seed(policy_seed(role, case, root))
    np.random.seed(policy_seed(role, case, root))
    return Policy(arm, config, universe)


def append_episode(path: Path, value: dict) -> None:
    # One complete JSON row per physical episode. Each row includes every H
    # action and cost, plus all fixed-tail actions/costs.
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def episode(config, universe, physical_seed: int, controller, *, train: bool,
            policy: Policy | None = None):
    state = initial_state(config)
    trajectory = []
    trace = []
    hcost = tailcost = 0.0
    parts = dict(preventive=0.0, corrective=0.0, unavailable=0.0, failure=0.0)
    metrics = dict(pm_starts=0, cm_starts=0, failures=0, completions=0,
                   unavailable_machine_intervals=0, occupied_technician_intervals=0,
                   wait_H=0)
    selection_seconds = 0.0
    forecast_transitions = 0
    start = time.perf_counter()
    for tick in range(H):
        obs = observe(config, state, H, B)
        before = state
        if policy is None:
            began = time.perf_counter()
            decision = controller.choose(obs)
            selection_seconds += time.perf_counter() - began
            action = decision.action
            forecast_transitions += decision.simulated_transitions
            index = universe.index(action)
            lp = value = 0.0
            encoded = None
        else:
            encoded = encode(config, state, universe, H, B)
            began = time.perf_counter()
            index, lp, value = choose(policy, encoded, deterministic=not train)
            selection_seconds += time.perf_counter() - began
            action = universe[index]
            if action not in obs.actions or not encoded.mask[index]:
                raise AssertionError("learner action outside legal catalogue")
        step = advance(config, state, action, physical_seed, H, B)
        metrics["wait_H"] += int(not action)
        if train:
            assert encoded is not None
            trajectory.append(Transition(encoded, index, lp, value,
                                         step.reward / REWARD_SCALE, tick == H - 1))
        for row in step.records:
            trace.append({"phase": row["phase"], "tick": row["tick"],
                          "action": row["action"], "cost": row["cost"],
                          "breakdown": row["breakdown"],
                          "catalogue_size": len(row["catalogue"]),
                          "index": index if row["phase"] == "H" else None})
            for key in parts:
                parts[key] += row["breakdown"][key]
            metrics["unavailable_machine_intervals"] += round(row["breakdown"]["unavailable"] / config.unavailable_cost)
            for event in row["events"]:
                if event["type"] == "start":
                    metrics["pm_starts" if event["kind"] == "PM" else "cm_starts"] += 1
                elif event["type"] == "failure":
                    metrics["failures"] += 1
                elif event["type"] == "completion":
                    metrics["completions"] += 1
                elif event["type"] == "occupancy":
                    metrics["occupied_technician_intervals"] += 1
        hcost += step.controlled_cost
        tailcost += step.tail_cost
        state = step.state
    metrics["unfinished_jobs"] = sum(job is not None for job in state.jobs)
    metrics["failed_components_at_cutoff"] = sum(item.failed for row in state.components for item in row)
    result = {
        "seed": physical_seed, "H_cost": hcost, "tail_cost": tailcost,
        "total_cost": hcost + tailcost, "components": parts, "metrics": metrics,
        "trace": trace, "final_state": asdict(state),
        "policy_decisions": H, "simulator_ticks": H + B, "ppo_samples": H if train else 0,
        "forecast_transitions": forecast_transitions,
        "selection_seconds": selection_seconds,
        "episode_seconds": time.perf_counter() - start,
        "terminal_status": "COMPLETE",
    }
    return result, trajectory


def train_trial(run_dir: Path, case: str, arm: str, root: int,
                *, smoke: bool, started: float, stage: str):
    config, _ = case_config(case)
    universe = action_universe(config)
    role = "engineering" if smoke else ("rep_train" if stage == "replication" else "train")
    policy = make_policy(arm, config, universe, role, case, root)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    trial_dir = run_dir / "trials" / case / arm / str(root)
    trial_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(trial_dir / "model.json", {"case": case, "arm": arm, "root": root,
                                          "parameter_counts": policy.parameter_counts(),
                                          "action_universe": [[list(x) for x in a] for a in universe],
                                          "config": json.loads(config_json(config))})
    samples = 0
    episodes = 0
    checkpoints = (48, 96) if smoke else CHECKPOINTS
    target = checkpoints[-1]
    batch = []
    train_start = time.perf_counter()
    for checkpoint in tqdm(checkpoints, desc=f"{case}/{arm}/{root} train", leave=False):
        while samples < checkpoint:
            episode_seed = seed(role, case, root, episodes)
            record, sequence = episode(config, universe, episode_seed, None,
                                       train=True, policy=policy)
            record.update({"case": case, "arm": arm, "root": root, "bank": role})
            append_episode(trial_dir / "train.jsonl.gz", record)
            batch.extend(sequence)
            episodes += 1
            samples += H
            if len(batch) >= (H if smoke else 25 * H):
                stats = ppo_update(policy, optimizer, batch)
                batch.clear()
                with open(trial_dir / "updates.jsonl", "a") as handle:
                    handle.write(json.dumps({"samples": samples, **stats}) + "\n")
            check_limits(run_dir, started, stage)
        # Checkpoints are aligned to complete episodes and PPO update boundaries.
        if batch:
            stats = ppo_update(policy, optimizer, batch)
            batch.clear()
            with open(trial_dir / "updates.jsonl", "a") as handle:
                handle.write(json.dumps({"samples": samples, **stats}) + "\n")
        torch.save(policy.state_dict(), trial_dir / f"checkpoint-{checkpoint}.pt")
    return {"case": case, "arm": arm, "root": root, "samples": samples,
            "episodes": episodes, "train_seconds": time.perf_counter() - train_start,
            "parameter_counts": policy.parameter_counts()}


def load_policy(trial_dir: Path, checkpoint: int):
    meta = json.loads((trial_dir / "model.json").read_text())
    config, _ = case_config(meta["case"])
    universe = action_universe(config)
    if [[list(x) for x in a] for a in universe] != meta["action_universe"]:
        raise ValueError("action universe mismatch")
    policy = Policy(meta["arm"], config, universe)
    policy.load_state_dict(torch.load(trial_dir / f"checkpoint-{checkpoint}.pt",
                                      map_location="cpu", weights_only=True))
    policy.eval()
    return config, universe, policy


def eval_trial(run_dir: Path, train_dir: Path, case: str, arm: str, root: int,
               checkpoint: int, bank: str, count: int, started: float, stage: str):
    trial_dir = train_dir / "trials" / case / arm / str(root)
    config, universe, policy = load_policy(trial_dir, checkpoint)
    output = run_dir / "evaluations" / bank / case / arm / str(root)
    output.mkdir(parents=True, exist_ok=bank == "dev")
    costs = []
    for i in tqdm(range(count), desc=f"{bank} {case}/{arm}/{root}", leave=False):
        record, _ = episode(config, universe, seed(bank, case, 0, i), None,
                            train=False, policy=policy)
        record.update({"case": case, "arm": arm, "root": root,
                       "checkpoint": checkpoint, "bank": bank})
        filename = f"episodes-{checkpoint}.jsonl.gz" if bank == "dev" else "episodes.jsonl.gz"
        append_episode(output / filename, record)
        costs.append(record["total_cost"])
        check_limits(run_dir, started, stage)
    result = {"case": case, "arm": arm, "root": root, "checkpoint": checkpoint,
              "bank": bank, "seeds": count, "mean_cost": float(np.mean(costs)),
              "checkpoint_hash": sha(trial_dir / f"checkpoint-{checkpoint}.pt")}
    summary_name = f"summary-{checkpoint}.json" if bank == "dev" else "summary.json"
    atomic_json(output / summary_name, result)
    return result


def eval_controls(run_dir: Path, bank: str, count: int, started: float, stage: str,
                  *, smoke: bool = False):
    for case in CASES[:1] if smoke else CASES:
        config, _ = case_config(case)
        universe = action_universe(config)
        controls = {"threshold5": Threshold(5), "risk": RiskMatching(),
                    "corrective": Corrective(), "rollout2": BoundedRollout(2, 2, 20000),
                    "wait": Wait()}
        for name, controller in controls.items():
            out = run_dir / "controls" / bank / case / name
            out.mkdir(parents=True, exist_ok=False)
            costs = []
            forecasts = []
            for i in tqdm(range(count), desc=f"{bank} {case}/{name}", leave=False):
                record, _ = episode(config, universe, seed(bank, case, 0, i), controller,
                                    train=False)
                record.update({"case": case, "arm": name, "bank": bank})
                append_episode(out / "episodes.jsonl.gz", record)
                costs.append(record["total_cost"])
                forecasts.append(record["forecast_transitions"])
                check_limits(run_dir, started, stage)
            atomic_json(out / "summary.json",
                        {"case": case, "arm": name, "bank": bank, "seeds": count,
                         "mean_cost": float(np.mean(costs)),
                         "mean_forecast_transitions": float(np.mean(forecasts))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(MAX_HOURS), required=True)
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--confirmation-dir", type=str)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stage = args.stage
    if stage == "confirmation" and not args.train_dir:
        parser.error("confirmation requires --train-dir")
    if stage == "replication" and not args.confirmation_dir:
        parser.error("replication requires --confirmation-dir")
    if stage == "replication":
        confirmation = ROOT / args.confirmation_dir
        report = json.loads((confirmation / "analysis.json").read_text())
        if not report["replication_eligible"] or report["status"] != "AUDIT_PASS":
            parser.error("confirmation gate is not audit-passed or eligible")
    run_dir = new_dir(stage)
    manifest = provenance(stage, run_dir, args)
    atomic_json(run_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        if stage in ("smoke", "train_dev", "replication"):
            for case in CASES:
                for arm in ARMS:
                    for root in range(1) if stage == "smoke" else ROOTS:
                        info = train_trial(run_dir, case, arm, root, smoke=stage == "smoke",
                                           started=started, stage=stage)
                        manifest["completed_trials"].append(info)
                        manifest["completed_episodes"] += info["episodes"]
                        atomic_json(run_dir / "manifest.json", manifest)
            if stage == "smoke":
                for case in CASES:
                    for arm in ARMS:
                        eval_trial(run_dir, run_dir, case, arm, 0, 96, "engineering_eval", 2,
                                   started, stage)
                eval_controls(run_dir, "engineering_eval", 1, started, stage)
            else:
                bank = "dev" if stage == "train_dev" else "rep_eval"
                count = 16 if stage == "train_dev" else 128
                if stage == "replication":
                    previous = ROOT / args.confirmation_dir
                    previous_manifest = json.loads((previous / "manifest.json").read_text())
                    selection = json.loads(((ROOT / previous_manifest["train_dir"]) / "selection.json").read_text())
                checkpoints = CHECKPOINTS if stage == "train_dev" else None
                for case in CASES:
                    for arm in ARMS:
                        for checkpoint in (checkpoints if checkpoints is not None else
                                           (selection["selected"][case][arm],)):
                            for root in ROOTS:
                                eval_trial(run_dir, run_dir, case, arm, root, checkpoint,
                                           bank, count, started, stage)
                if stage in ("train_dev", "replication"):
                    eval_controls(run_dir, bank, count, started, stage)
        elif stage == "confirmation":
            train_dir = ROOT / args.train_dir
            selection = json.loads((train_dir / "selection.json").read_text())
            if selection["status"] != "AUDIT_PASS":
                raise ValueError("train/DEV selection not audit-passed")
            for case in CASES:
                for arm in ARMS:
                    checkpoint = selection["selected"][case][arm]
                    for root in ROOTS:
                        eval_trial(run_dir, train_dir, case, arm, root, checkpoint,
                                   "confirmation", 128, started, stage)
            eval_controls(run_dir, "confirmation", 128, started, stage)
        manifest["status"] = "COMPLETE"
    except BaseException as exc:
        manifest["status"] = "INCOMPLETE"
        manifest["failure"] = repr(exc)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic_json(run_dir / "manifest.json", manifest)
        print(run_dir)


if __name__ == "__main__":
    main()
