"""Prelocked L2 budget ladder and stochastic/greedy JCST evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

import numpy as np
import torch
from tqdm import tqdm

from ht_pdm_fjsp.component_controllers_v1 import Corrective, RiskMatching, Threshold, Wait, observe
from ht_pdm_fjsp.component_framework_v1 import advance
from ht_pdm_fjsp.component_service_v1 import config_json, initial_state
from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import Policy, action_universe, choose, encode, ppo_update
from ht_pdm_fjsp.jcst_learner_l2 import policy_diagnostic, ppo_update_diagnostic
from jcst_learner_campaign_v1 import episode as l1_episode

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/jcst_learnability_protocol_l2.md"
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CHECKPOINTS = (24000, 96000, 384000, 1536000)
H, B = 48, 16
BASE_SEED = 1 << 50
TORCH_BASE = 1 << 49
ACTION_BASE = 1 << 48
BANK = {"engineering": 1, "train": 2, "dev": 3, "confirmation": 4,
        "intervention1_train": 5, "intervention1_dev": 6,
        "intervention2_train": 7, "intervention2_dev": 8,
        "rep_train": 9, "rep_eval": 10}
TRACE_EPISODES = {0, 1, 2, 3, 499, 1999, 7999, 31997, 31998, 31999}
SOURCES = (
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/component_framework_v1.py",
    "src/ht_pdm_fjsp/jcst_characterization_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_v1.py",
    "src/ht_pdm_fjsp/jcst_learner_l2.py",
    "experiments/jcst_learner_campaign_v1.py",
    "experiments/jcst_learnability_l2.py",
    "experiments/jcst_learnability_audit_l2.py",
)
ARTIFACTS = ROOT / "artifacts/jcst_learnability"
DISK_CAP = 15 * 1024**3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_seed(bank: str, case: str, root: int, episode: int) -> int:
    if bank not in BANK or case not in CASES or not 0 <= root < 10 or not 0 <= episode < 1_000_000:
        raise ValueError("physical seed role/index")
    return BASE_SEED + BANK[bank] * 10**12 + CASES.index(case) * 10**10 + root * 10**7 + episode


def torch_seed(bank: str, case: str, arm: str, root: int) -> int:
    return TORCH_BASE + BANK[bank] * 10**12 + CASES.index(case) * 10**10 + ARMS.index(arm) * 10**9 + root * 10**7


def action_seed(bank: str, case: str, arm: str, root: int, checkpoint: int,
                physical_index: int, repetition: int) -> int:
    fields = f"{ACTION_BASE}:{bank}:{case}:{arm}:{root}:{checkpoint}:{physical_index}:{repetition}"
    return int.from_bytes(hashlib.blake2b(fields.encode(), digest_size=8).digest(), "big")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def new_dir(stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ARTIFACTS / f"{stage}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(stage: str, out: Path, args) -> dict:
    return {
        "schema": "JCST-learnability-L2", "stage": stage, "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(out.relative_to(ROOT)), "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "platform": platform.platform(), "python": sys.version,
        "packages": {key: importlib.metadata.version(key) for key in ("numpy", "torch", "tqdm")},
        "source_hashes": {name: sha(ROOT / name) for name in SOURCES},
        "protocol_hash": sha(PROTOCOL),
        "contract_hash": sha(ROOT / "docs/joint_component_service_contract_v1.md"),
        "config_hashes": {case: hashlib.sha256(config_json(case_config(case)[0]).encode()).hexdigest()
                          for case in CASES},
        "parameters": {"H": H, "B": B, "cases": CASES, "arms": ARMS,
                       "checkpoints": CHECKPOINTS, "roots": [0, 1, 2, 3, 4],
                       "gamma": 1, "gae_lambda": .95, "reward_divisor": 100,
                       "adam_lr": .0003, "epochs": 4, "minibatch": 128,
                       "clip": .2, "entropy_coefficient": .01, "gradient_clip": .5,
                       "episodes_per_update": 25, "dev_physical_seeds": 16,
                       "stochastic_action_repetitions": 4,
                       "threads": 1, "memory_cap_kib": 8 * 1024 * 1024,
                       "disk_cap_bytes": DISK_CAP},
        "seed_roles": {"physical_base": BASE_SEED, "torch_base": TORCH_BASE,
                       "action_base": ACTION_BASE, "banks": BANK},
        "source_train_dir": args.train_dir, "completed_trials": [],
        "completed_evaluations": [], "completed_controls": [], "failure": None,
    }


def check_caps(out: Path, started: float, campaign_started: float) -> None:
    if time.monotonic() - started > 72 * 3600 or time.time() - campaign_started > 72 * 3600:
        raise TimeoutError("L2 72-hour scientific wall cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024 * 1024:
        raise MemoryError("L2 8 GiB process cap")
    if sum(path.stat().st_size for path in ARTIFACTS.rglob("*") if path.is_file()) > DISK_CAP:
        raise OSError("L2 15 GiB artifact cap")


def policy_for(arm: str, config, universe, bank: str, case: str, root: int) -> Policy:
    torch.manual_seed(torch_seed(bank, case, arm, root))
    np.random.seed(torch_seed(bank, case, arm, root) % (2**32 - 1))
    return Policy(arm, config, universe)


def compact_train(record: dict, *, case: str, arm: str, root: int,
                  episode_index: int, trace: bool) -> dict:
    keep = ("seed", "H_cost", "tail_cost", "total_cost", "components", "metrics",
            "policy_decisions", "simulator_ticks", "ppo_samples", "terminal_status")
    row = {key: record[key] for key in keep}
    row.update(case=case, arm=arm, root=root, episode_index=episode_index)
    if trace:
        row["trace"] = record["trace"]
        row["final_state"] = record["final_state"]
    return row


def train_trial(out: Path, stage: str, case: str, arm: str, root: int,
                checkpoints: tuple[int, ...], started: float, campaign_started: float) -> dict:
    config, _ = case_config(case)
    universe = action_universe(config)
    bank = "engineering" if stage == "smoke" else "rep_train" if stage == "replication" else "train"
    policy = policy_for(arm, config, universe, bank, case, root)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    trial = out / "trials" / case / arm / str(root)
    trial.mkdir(parents=True, exist_ok=False)
    atomic_json(trial / "model.json", {"case": case, "arm": arm, "root": root,
                                       "parameters": policy.parameter_counts(),
                                       "universe": [[list(edge) for edge in action] for action in universe],
                                       "config": json.loads(config_json(config))})
    episodes = samples = update_index = 0
    batch = []
    cost_sum = 0.0
    start = time.perf_counter()
    with open(trial / "train.jsonl", "w", encoding="utf-8") as stream, \
         open(trial / "updates.jsonl", "w", encoding="utf-8") as updates, \
         tqdm(total=checkpoints[-1] // H, desc=f"{case}/{arm}/{root} TRAIN", leave=False) as progress:
        for checkpoint in checkpoints:
            while samples < checkpoint:
                episode_index = episodes
                seed = physical_seed(bank, case, root, episode_index)
                record, sequence = l1_episode(config, universe, seed, None, train=True, policy=policy)
                row = compact_train(record, case=case, arm=arm, root=root,
                                    episode_index=episode_index,
                                    trace=stage == "smoke" or episode_index in TRACE_EPISODES)
                row["bank"] = bank
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                cost_sum += record["total_cost"]
                batch.extend(sequence)
                episodes += 1
                samples += H
                progress.update(1)
                if len(batch) >= (H if stage == "smoke" else 25 * H):
                    update_index += 1
                    sampled = update_index == 1 or update_index % 20 == 0
                    metrics = (ppo_update_diagnostic(policy, optimizer, batch) if sampled else
                               ppo_update(policy, optimizer, batch))
                    updates.write(json.dumps({"samples": samples, "update": update_index,
                                              "sampled_telemetry": sampled, **metrics}) + "\n")
                    batch.clear()
                    stream.flush()
                    updates.flush()
                    check_caps(out, started, campaign_started)
            if batch:
                raise AssertionError("checkpoint must end at PPO update boundary")
            torch.save(policy.state_dict(), trial / f"checkpoint-{checkpoint}.pt")
            atomic_json(trial / f"checkpoint-{checkpoint}.json",
                        {"samples": samples, "episodes": episodes, "updates": update_index,
                         "mean_train_episode_cost_to_date": cost_sum / episodes,
                         "elapsed_training_seconds": time.perf_counter() - start,
                         "weights_sha256": sha(trial / f"checkpoint-{checkpoint}.pt")})
    return {"case": case, "arm": arm, "root": root, "samples": samples,
            "episodes": episodes, "updates": update_index,
            "training_seconds": time.perf_counter() - start,
            "parameters": policy.parameter_counts()}


def load_policy(trial: Path, checkpoint: int):
    meta = json.loads((trial / "model.json").read_text())
    config, _ = case_config(meta["case"])
    universe = action_universe(config)
    if meta["universe"] != [[list(edge) for edge in action] for action in universe]:
        raise ValueError("action universe changed")
    policy = Policy(meta["arm"], config, universe)
    policy.load_state_dict(torch.load(trial / f"checkpoint-{checkpoint}.pt", weights_only=True,
                                      map_location="cpu"))
    policy.eval()
    return config, universe, policy


def evaluated_episode(config, universe, seed: int, *, policy: Policy | None,
                      controller=None, generator: torch.Generator | None = None):
    state = initial_state(config)
    trace = []
    diagnostics = []
    hcost = tailcost = 0.0
    pieces = dict(preventive=0.0, corrective=0.0, unavailable=0.0, failure=0.0)
    metrics = dict(pm_starts=0, cm_starts=0, failures=0, completions=0,
                   unavailable_machine_intervals=0, occupied_technician_intervals=0,
                   wait_H=0, forced_wait=0, discretionary_wait=0,
                   wait_failed_free=0, wait_all_healthy=0, wait_all_busy=0)
    select_seconds = 0.0
    forecast = 0
    for tick in range(H):
        obs = observe(config, state, H, B)
        legal_count = len(obs.actions)
        start = time.perf_counter()
        if policy is None:
            decision = controller.choose(obs)
            action = decision.action
            index = universe.index(action)
            forecast += decision.simulated_transitions
        else:
            encoded = encode(config, state, universe, H, B)
            diagnostic = policy_diagnostic(policy, encoded,
                                           deterministic=generator is None,
                                           generator=generator)
            index = diagnostic["index"]
            action = universe[index]
            if action not in obs.actions or not encoded.mask[index]:
                raise AssertionError("evaluation action illegal")
            diagnostic.update(tick=tick, action=index,
                              near_tie=diagnostic["logit_gap"] <= .01,
                              exact_tie=diagnostic["logit_gap"] <= 1e-6)
            diagnostics.append(diagnostic)
        select_seconds += time.perf_counter() - start
        if not action:
            metrics["wait_H"] += 1
            if any(any(state.components[m][c].failed for m, c, _ in candidate)
                   for candidate in obs.actions if candidate):
                metrics["wait_failed_free"] += 1
            if all(not item.failed for row in state.components for item in row):
                metrics["wait_all_healthy"] += 1
            if all(job is not None for job in state.jobs):
                metrics["wait_all_busy"] += 1
            if legal_count == 1:
                metrics["forced_wait"] += 1
            else:
                metrics["discretionary_wait"] += 1
        step = advance(config, state, action, seed, H, B)
        for item in step.records:
            trace.append({"phase": item["phase"], "tick": item["tick"],
                          "action": item["action"], "cost": item["cost"],
                          "breakdown": item["breakdown"],
                          "catalogue_size": len(item["catalogue"]),
                          "index": index if item["phase"] == "H" else None})
            for key in pieces:
                pieces[key] += item["breakdown"][key]
            metrics["unavailable_machine_intervals"] += round(item["breakdown"]["unavailable"] / config.unavailable_cost)
            for event in item["events"]:
                kind = event["type"]
                if kind == "start":
                    metrics["pm_starts" if event["kind"] == "PM" else "cm_starts"] += 1
                elif kind == "failure":
                    metrics["failures"] += 1
                elif kind == "completion":
                    metrics["completions"] += 1
                elif kind == "occupancy":
                    metrics["occupied_technician_intervals"] += 1
        hcost += step.controlled_cost
        tailcost += step.tail_cost
        state = step.state
    metrics["unfinished_jobs"] = sum(job is not None for job in state.jobs)
    metrics["failed_components_at_cutoff"] = sum(item.failed for row in state.components for item in row)
    return {"seed": seed, "H_cost": hcost, "tail_cost": tailcost,
            "total_cost": hcost + tailcost, "components": pieces, "metrics": metrics,
            "trace": trace, "diagnostics": diagnostics, "final_state": asdict(state),
            "policy_decisions": H, "simulator_ticks": H + B,
            "ppo_samples": 0, "forecast_transitions": forecast,
            "selection_seconds": select_seconds, "terminal_status": "COMPLETE"}


def eval_model(out: Path, source: Path, bank: str, case: str, arm: str, root: int,
               checkpoint: int, mode: str, count: int, repetitions: int,
               started: float, campaign_started: float) -> dict:
    trial = source / "trials" / case / arm / str(root)
    config, universe, policy = load_policy(trial, checkpoint)
    dest = out / "evaluations" / bank / case / arm / str(root)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f"checkpoint-{checkpoint}-{mode}.jsonl"
    if output.exists():
        raise FileExistsError(output)
    costs = []
    checkpoint_hash = sha(trial / f"checkpoint-{checkpoint}.pt")
    with open(output, "x", encoding="utf-8") as stream:
        for i in tqdm(range(count), desc=f"{bank} {case}/{arm}/{root}/{checkpoint}/{mode}", leave=False):
            for rep in range(1 if mode == "greedy" else repetitions):
                generator = None
                if mode == "stochastic":
                    generator = torch.Generator().manual_seed(action_seed(bank, case, arm, root,
                                                                           checkpoint, i, rep))
                record = evaluated_episode(config, universe, physical_seed(bank, case, 0, i),
                                           policy=policy, generator=generator)
                record.update(case=case, arm=arm, root=root, checkpoint=checkpoint,
                              mode=mode, physical_index=i, action_repetition=rep, bank=bank,
                              checkpoint_sha256=checkpoint_hash)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                costs.append(record["total_cost"])
            stream.flush()
            check_caps(out, started, campaign_started)
    return {"case": case, "arm": arm, "root": root, "checkpoint": checkpoint,
            "mode": mode, "bank": bank, "physical_seeds": count,
            "action_repetitions": 1 if mode == "greedy" else repetitions,
            "mean_cost": float(np.mean(costs)), "file": str(output.relative_to(ROOT))}


def eval_controls(out: Path, bank: str, count: int, started: float,
                  campaign_started: float, *, smoke: bool = False) -> list[dict]:
    results = []
    for case in CASES[:1] if smoke else CASES:
        config, _ = case_config(case)
        universe = action_universe(config)
        for name, controller in (("threshold5", Threshold(5)), ("risk", RiskMatching()),
                                 ("corrective", Corrective()), ("wait", Wait())):
            dest = out / "controls" / bank / case / name
            dest.mkdir(parents=True, exist_ok=False)
            costs = []
            with open(dest / "episodes.jsonl", "x", encoding="utf-8") as stream:
                for i in tqdm(range(count), desc=f"{bank} {case}/{name}", leave=False):
                    record = evaluated_episode(config, universe, physical_seed(bank, case, 0, i),
                                               policy=None, controller=controller)
                    record.update(case=case, arm=name, bank=bank)
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    costs.append(record["total_cost"])
                    check_caps(out, started, campaign_started)
            results.append({"case": case, "arm": name, "bank": bank,
                            "physical_seeds": count, "mean_cost": float(np.mean(costs))})
    return results


def source_manifest_check(source: Path) -> dict:
    source_data = json.loads((source / "manifest.json").read_text())
    if source_data["schema"] != "JCST-learnability-L2" or source_data["status"] != "COMPLETE":
        raise ValueError("source training incomplete")
    if source_data["protocol_hash"] != sha(PROTOCOL):
        raise ValueError("protocol changed")
    for name, digest in source_data["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
    return source_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "train_dev", "confirmation", "replication"), required=True)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--confirmation-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stage = args.stage
    if stage in ("confirmation", "replication") and not args.train_dir:
        parser.error("confirmation/replication require --train-dir")
    if stage in ("confirmation", "replication") and not args.selection:
        parser.error("confirmation/replication require --selection")
    source = ROOT / args.train_dir if args.train_dir else None
    selection = json.loads((ROOT / args.selection).read_text()) if args.selection else None
    if source is not None:
        source_data = source_manifest_check(source)
        if selection.get("status") != "AUDIT_PASS" or selection.get("source_manifest_hash") != sha(source / "manifest.json"):
            raise ValueError("selection not audit-frozen")
    if stage == "replication":
        if not args.confirmation_dir:
            parser.error("replication requires --confirmation-dir")
        confirmation = json.loads(((ROOT / args.confirmation_dir) / "analysis.json").read_text())
        if confirmation.get("status") != "AUDIT_PASS" or not confirmation.get("replication_eligible"):
            raise ValueError("confirmation not eligible for replication")
    out = new_dir(stage)
    data = manifest(stage, out, args)
    atomic_json(out / "manifest.json", data)
    started = time.monotonic()
    campaign_started = (datetime.fromisoformat(source_data["started_utc"]).timestamp()
                        if source is not None else time.time())
    try:
        if stage in ("smoke", "train_dev"):
            checkpoints = (48, 96) if stage == "smoke" else CHECKPOINTS
            for case in CASES:
                for arm in ARMS:
                    for root in (range(1) if stage == "smoke" else range(5)):
                        result = train_trial(out, stage, case, arm, root, checkpoints,
                                             started, campaign_started)
                        data["completed_trials"].append(result)
                        atomic_json(out / "manifest.json", data)
            bank = "engineering" if stage == "smoke" else "dev"
            count = 2 if stage == "smoke" else 16
            modes = ("greedy",) if stage == "smoke" else ("greedy", "stochastic")
            for case in CASES:
                for arm in ARMS:
                    for checkpoint in checkpoints:
                        for root in (range(1) if stage == "smoke" else range(5)):
                            for mode in modes:
                                result = eval_model(out, out, bank, case, arm, root, checkpoint,
                                                    mode, count, 4, started, campaign_started)
                                data["completed_evaluations"].append(result)
                                atomic_json(out / "manifest.json", data)
            data["completed_controls"] = eval_controls(out, bank, 1 if stage == "smoke" else 16,
                                                       started, campaign_started, smoke=stage == "smoke")
        else:
            bank = "confirmation" if stage == "confirmation" else "rep_eval"
            count = 128
            roots = range(5) if stage == "confirmation" else range(5, 10)
            train_source = source if stage == "confirmation" else out
            if stage == "replication":
                for case in CASES:
                    for arm in ARMS:
                        for root in roots:
                            checkpoint = int(selection["selected"][case][arm]["checkpoint"])
                            replicate_checkpoints = (24000, checkpoint) if checkpoint > 24000 else (24000,)
                            result = train_trial(out, "replication", case, arm, root,
                                                 replicate_checkpoints, started, campaign_started)
                            data["completed_trials"].append(result)
                            atomic_json(out / "manifest.json", data)
            for case in CASES:
                for arm in ARMS:
                    chosen = selection["selected"][case][arm]
                    for root in roots:
                        for checkpoint in sorted({24000, int(chosen["checkpoint"])}):
                            result = eval_model(out, train_source, bank, case, arm, root,
                                                checkpoint, chosen["mode"], count, 4,
                                                started, campaign_started)
                            data["completed_evaluations"].append(result)
                            atomic_json(out / "manifest.json", data)
            data["completed_controls"] = eval_controls(out, bank, count, started, campaign_started)
        data["status"] = "COMPLETE"
    except BaseException as exc:
        data["status"] = "INCOMPLETE"
        data["failure"] = repr(exc)
        raise
    finally:
        data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        data["elapsed_seconds"] = time.monotonic() - started
        data["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic_json(out / "manifest.json", data)
        print(out)


if __name__ == "__main__":
    main()
