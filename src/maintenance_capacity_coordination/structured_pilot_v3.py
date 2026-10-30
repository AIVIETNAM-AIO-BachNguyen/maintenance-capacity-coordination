"""Prelocked structured demand/technician-commitment PPO pilot runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import configs
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.pilot import CASES, evaluate_controls
from maintenance_capacity_coordination.structured_policy_v3 import (
    MatchingPolicy, Transition, actor_inputs, ppo_update,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/structured_demand_commitment_pilot_v3.md"
ARMS = ("S", "R", "G3", "D3")
TRAIN_SEEDS = (73180000, 73180001, 73180002)
DEV_SEEDS = tuple(range(73181000, 73181016))
HELDOUT_SEEDS = tuple(range(73182000, 73182032))
REPLICATION_TRAIN_SEEDS = (73183000, 73183001, 73183002)
REPLICATION_DEV_SEEDS = tuple(range(73184000, 73184016))
REPLICATION_EVAL_SEEDS = tuple(range(73185000, 73185032))
SEALED_TEST_SEEDS = tuple(range(73186000, 73186032))
MILESTONES = (2048, 4096, 8192)
TRIAL_TIMEOUT = 7200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def seed_registry_check():
    prior = (set(range(73110000, 73110003)) | set(range(73120000, 73120016))
             | set(range(73130000, 73130016)) | set(range(73140000, 73140032))
             | set(range(73160000, 73160032)) | set(range(73170000, 73170016)))
    groups = (TRAIN_SEEDS, DEV_SEEDS, HELDOUT_SEEDS, REPLICATION_TRAIN_SEEDS,
              REPLICATION_DEV_SEEDS, REPLICATION_EVAL_SEEDS, SEALED_TEST_SEEDS)
    raw = [seed for group in groups for seed in group]
    if len(set(raw)) != len(raw) or set(raw) & prior:
        raise RuntimeError("raw seed roles overlap")
    old_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior}
    for seed in range(73110000, 73110003):
        for case in CASES:
            for episode in range(200):
                reset = role_seed(seed, f"pilot_train:{case}:{episode}")
                old_physical.add(role_seed(reset, BASE_VERSION + ":physical"))
    new_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    if len(new_physical) != len(raw) or new_physical & old_physical:
        raise RuntimeError("derived physical seed roles overlap")
    derived_train = set()
    for group in (TRAIN_SEEDS, REPLICATION_TRAIN_SEEDS):
        for seed in group:
            for case in CASES:
                for episode in range(200):
                    reset = role_seed(seed, f"structured_train_v3:{case}:{episode}")
                    physical = role_seed(reset, BASE_VERSION + ":physical")
                    if physical in old_physical or physical in new_physical or physical in derived_train:
                        raise RuntimeError("training episode physical seed overlap")
                    derived_train.add(physical)


def create_run(output_root, kind):
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ("structured_pilot_v3_" + kind + "_"
                          + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    path.mkdir(exist_ok=False)
    cfg = configs()
    manifest = {
        "status": "RUNNING", "kind": kind, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": source_hashes(),
        "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                          for case in CASES},
        "cases": CASES, "arms": ARMS,
        "train_seeds": TRAIN_SEEDS, "dev_seeds": DEV_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "replication_train_seeds": REPLICATION_TRAIN_SEEDS,
        "replication_dev_seeds": REPLICATION_DEV_SEEDS,
        "replication_eval_seeds": REPLICATION_EVAL_SEEDS,
        "sealed_test_seeds": SEALED_TEST_SEEDS,
        "milestones": MILESTONES, "trial_timeout_seconds": TRIAL_TIMEOUT,
        "forecast_transitions": 0,
    }
    write_json(path / "manifest.json", manifest)
    return path, manifest


def choose(policy, env, deterministic=False):
    pre = env.pre_observation()
    submissions = env.fixed_submissions("D" if policy.arm == "D3" else "threshold")
    admitted = env.admitted(submissions)
    machine, worker = actor_inputs(env.config, admitted, env.tick)
    mask = env.matching_mask(submissions)
    if not mask.any() or not mask[0]:
        raise RuntimeError("WAIT must be legal")
    with torch.no_grad():
        m = torch.from_numpy(machine)
        w = torch.from_numpy(worker)
        mask_t = torch.from_numpy(mask)
        index = policy.sample(m, w, mask_t, deterministic)
        logprob, _ = policy.logprob_entropy(m, w, mask_t, index)
        value = policy.value(torch.from_numpy(pre))
    return {"pre_obs": pre, "machine_obs": machine, "worker_obs": worker,
            "matching_mask": mask, "matching_index": int(index),
            "old_logprob": float(logprob), "value": float(value),
            "submissions": submissions}


def load_policy(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = MatchingPolicy(payload["arm"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy


def evaluate_policy(policy, case, seeds, path=None, *, label=None, deadline=None):
    env = CoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f"Evaluate v3 {case} {policy.arm} {label}", leave=False):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("trial timeout during evaluation")
        env.reset(seed)
        selection_seconds = 0.0
        for _ in range(48):
            started = time.monotonic()
            decision = choose(policy, env, deterministic=True)
            selection_seconds += time.monotonic() - started
            _, done = env.step(decision["submissions"], decision["matching_index"])
        assert done
        row = {"case": case, "arm": policy.arm, "seed": seed,
               "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
               "checkpoint": label, "selection_seconds": selection_seconds,
               **env.metrics()}
        if (row["invalid_assignments"] != 0 or abs(row["episode_return"] + row["objective"]) > 1e-8
                or abs(row["objective"] - row["H_cost"] - row["tail_cost"]) > 1e-8):
            raise RuntimeError("evaluation feasibility/accounting gate")
        rows.append(row)
        if path is not None:
            append_jsonl(path, row)
    return rows


def train_trial(run, case, arm, seed, *, steps=8192, eval_seeds=DEV_SEEDS):
    trial = run / f"{case}_{arm}_{seed}"
    trial.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    policy = MatchingPolicy(arm)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    env = CoordinationEnv(configs()[case])
    episode_index = 0
    env.reset(role_seed(seed, f"structured_train_v3:{case}:{episode_index}"))
    started = time.monotonic()
    deadline = started + TRIAL_TIMEOUT
    record = {"status": "RUNNING", "case": case, "arm": arm, "train_seed": seed,
              "requested_H_transitions": steps, "actual_H_transitions": 0,
              "actor_parameters": policy.actor_parameters,
              "critic_parameters": policy.critic_parameters,
              "forecast_transitions": 0,
              "ppo": {"n_steps": 256, "minibatch": 64, "epochs": 4, "learning_rate": 3e-4,
                      "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .01}}
    write_json(trial / "manifest.json", record)
    dev_means = []
    try:
        with tqdm(total=steps, desc=f"Train v3 {case} {arm} {seed}") as bar:
            for milestone in ((steps,) if steps < MILESTONES[0] else MILESTONES):
                if milestone > steps:
                    break
                while record["actual_H_transitions"] < milestone:
                    trajectory = []
                    for _ in range(min(256, milestone - record["actual_H_transitions"])):
                        if time.monotonic() > deadline:
                            raise TimeoutError("two-hour v3 trial timeout")
                        decision = choose(policy, env, deterministic=False)
                        reward, done = env.step(decision["submissions"], decision["matching_index"])
                        trajectory.append(Transition(
                            decision["pre_obs"], decision["machine_obs"], decision["worker_obs"],
                            decision["matching_mask"], decision["matching_index"],
                            decision["old_logprob"], decision["value"], reward, done))
                        record["actual_H_transitions"] += 1
                        bar.update(1)
                        if done:
                            append_jsonl(trial / "training_episodes.jsonl",
                                         {"episode": episode_index,
                                          "transition": record["actual_H_transitions"],
                                          **env.metrics()})
                            episode_index += 1
                            env.reset(role_seed(seed, f"structured_train_v3:{case}:{episode_index}"))
                    with torch.no_grad():
                        bootstrap = float(policy.value(torch.from_numpy(env.pre_observation())))
                    update = ppo_update(policy, optimizer, trajectory, bootstrap)
                    append_jsonl(trial / "updates.jsonl",
                                 {"transition": record["actual_H_transitions"], **update})
                    write_json(trial / "manifest.json", record)
                checkpoint = trial / f"checkpoint_{milestone}.pt"
                torch.save({"arm": arm, "case": case, "train_seed": seed,
                            "transition": milestone, "state_dict": policy.state_dict()}, checkpoint)
                restored = load_policy(checkpoint)
                rows = evaluate_policy(restored, case, eval_seeds,
                                       trial / "dev_episodes.jsonl", label=str(milestone), deadline=deadline)
                mean = float(np.mean([row["objective"] for row in rows]))
                dev_means.append({"transition": milestone, "mean_H_tail_cost": mean,
                                  "checkpoint": checkpoint.name})
                write_json(trial / "dev_milestones.json", dev_means)
        selected = min(dev_means, key=lambda item: (item["mean_H_tail_cost"], item["transition"]))
        write_json(trial / "selection.json", selected)
        record.update(status="COMPLETED", selected_checkpoint=selected["checkpoint"],
                      dev_mean=selected["mean_H_tail_cost"])
    except Exception as error:
        record.update(status="FAILED", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(trial / "manifest.json", record)
    return record


def evaluate_selected(run, training_run, seeds, panel):
    original = json.loads((training_run / "manifest.json").read_text())
    if original["status"] != "COMPLETED" or original["kind"] not in ("train", "replication_train"):
        raise RuntimeError("training run is not complete")
    current = json.loads((run / "manifest.json").read_text())
    if (original["protocol_sha256"] != current["protocol_sha256"]
            or original["source_sha256"] != current["source_sha256"]
            or original["config_sha256"] != current["config_sha256"]):
        raise RuntimeError("source/protocol/config changed since training")
    train_seeds = TRAIN_SEEDS if original["kind"] == "train" else REPLICATION_TRAIN_SEEDS
    count = 0
    for case in CASES:
        for arm in ARMS:
            for seed in train_seeds:
                trial = training_run / f"{case}_{arm}_{seed}"
                selection = json.loads((trial / "selection.json").read_text())
                policy = load_policy(trial / selection["checkpoint"])
                count += len(evaluate_policy(policy, case, seeds, run / f"{panel}_episodes.jsonl",
                                             label=f"{seed}:{selection['transition']}"))
    controls = evaluate_controls(run, seeds, panel=panel)
    return count, len(controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "heldout", "replication_train", "replication_eval"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--pilot-heldout-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_registry_check()
    if args.mode in ("heldout", "replication_eval") and args.training_run is None:
        parser.error("--training-run required")
    run, manifest = create_run(args.output_root, args.mode)
    print(run.resolve(), flush=True)
    started = time.monotonic()
    try:
        if args.mode in ("smoke", "train", "replication_train"):
            trials = []
            cases = CASES if args.mode != "smoke" else CASES
            seeds = TRAIN_SEEDS if args.mode == "train" else REPLICATION_TRAIN_SEEDS
            if args.mode == "smoke":
                seeds = (73100041,)
            eval_seeds = (73100042,) if args.mode == "smoke" else (
                DEV_SEEDS if args.mode == "train" else REPLICATION_DEV_SEEDS)
            steps = 256 if args.mode == "smoke" else 8192
            if args.mode == "replication_train":
                if args.pilot_heldout_run is None:
                    raise RuntimeError("independent replication requires --pilot-heldout-run")
                gate = json.loads((args.pilot_heldout_run / "replication_go_decision.json").read_text())
                if gate.get("go") is not True or gate.get("protocol_sha256") != manifest["protocol_sha256"]:
                    raise RuntimeError("pilot gate did not qualify independent training replication")
            for case in cases:
                for arm in ARMS:
                    for seed in seeds:
                        trials.append(train_trial(run, case, arm, seed, steps=steps, eval_seeds=eval_seeds))
                        manifest["trials"] = trials
                        write_json(run / "manifest.json", manifest)
        else:
            training_run = args.training_run.resolve()
            panel = "heldout" if args.mode == "heldout" else "replication_eval"
            expected_kind = "train" if args.mode == "heldout" else "replication_train"
            original = json.loads((training_run / "manifest.json").read_text())
            if original["kind"] != expected_kind:
                raise RuntimeError("wrong training role for evaluation")
            if args.mode == "heldout" and not (training_run / "audit.json").exists():
                raise RuntimeError("training audit required before held-out evaluation")
            if args.mode == "replication_eval" and not (training_run / "audit.json").exists():
                raise RuntimeError("replication training audit required before evaluation")
            seeds = HELDOUT_SEEDS if args.mode == "heldout" else REPLICATION_EVAL_SEEDS
            manifest["training_run"] = str(training_run)
            learned, controls = evaluate_selected(run, training_run, seeds, panel)
            manifest["learned_episodes"] = learned
            manifest["control_episodes"] = controls
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {str(item.relative_to(run)): sha256(item) for item in run.rglob("*")
                                      if item.is_file() and item.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
