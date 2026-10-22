"""Prelocked pilot runner. Each invocation creates a fresh artifact directory."""
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
from ht_pdm_fjsp.maintenance_framework_v1 import configs, evaluate_episode, RuleController
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import ARMS, BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.pilot_policy import PilotPolicy, Transition, ppo_update

CASES = ("reciprocal_scale2", "specialist_generalist_scale2")
TRAIN_SEEDS = (73110000, 73110001, 73110002)
DEV_SEEDS = tuple(range(73120000, 73120016))
REPLICATION_SEEDS = tuple(range(73130000, 73130016))
TEST_SEEDS = tuple(range(73140000, 73140032))
MILESTONES = (2048, 4096, 8192)
TRIAL_TIMEOUT = 7200
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/coordination_pilot_v1.md"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def new_run(output_root, kind):
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / (f"pilot_v1_{kind}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    path.mkdir(parents=True, exist_ok=False)
    source = sorted((ROOT / "src").rglob("*.py"))
    configs_by_case = configs()
    manifest = {
        "status": "RUNNING", "kind": kind, "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": {str(item.relative_to(ROOT)): sha256(item) for item in source},
        "config_sha256": {case: hashlib.sha256(json.dumps(asdict(configs_by_case[case]),
                                                     sort_keys=True).encode()).hexdigest() for case in CASES},
        "cases": CASES, "arms": ARMS, "train_seeds": TRAIN_SEEDS,
        "dev_seeds": DEV_SEEDS, "replication_seeds": REPLICATION_SEEDS,
        "sealed_test_seeds": TEST_SEEDS,
        "milestones": MILESTONES, "trial_timeout_seconds": TRIAL_TIMEOUT,
        "forecast_transitions": 0,
    }
    write_json(path / "manifest.json", manifest)
    return path, manifest


def _tensor(array, dtype=None):
    return torch.as_tensor(array, dtype=dtype)


def choose(policy, env, deterministic):
    """Return recorded observations and exact sampled joint log probability."""
    pre = env.pre_observation()
    eligible = env.request_mask()
    machine_obs = None if policy.arm == "D" else env.machine_observation(policy.arm)
    with torch.no_grad():
        bits = policy.sample_requests(_tensor(machine_obs) if machine_obs is not None else None,
                                      _tensor(eligible), deterministic)
        request_bits = bits.cpu().numpy().astype(np.float32)
        submissions = tuple(np.flatnonzero(request_bits).tolist())
        coordinator_obs = env.coordinator_observation(submissions)
        matching_mask = env.matching_mask(submissions)
        if not matching_mask.any():
            raise RuntimeError("no legal matching, including WAIT")
        index = policy.sample_matching(_tensor(coordinator_obs), _tensor(matching_mask), deterministic)
        logprob, _ = policy.logprob_entropy(
            _tensor(machine_obs) if machine_obs is not None else None,
            _tensor(eligible), bits, _tensor(coordinator_obs), _tensor(matching_mask), index,
        )
        value = policy.value(_tensor(pre))
    return {
        "pre_obs": pre, "machine_obs": machine_obs, "eligible": eligible,
        "request_bits": request_bits, "submissions": submissions,
        "coordinator_obs": coordinator_obs, "matching_mask": matching_mask,
        "matching_index": int(index), "old_logprob": float(logprob), "value": float(value),
    }


def evaluate_policy(policy, case, seeds, episode_path=None, *, label=None, deadline=None):
    env = CoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f"Evaluate {case} {label or policy.arm}", leave=False):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("trial timeout during evaluation")
        env.reset(seed)
        for _ in range(48):
            decision = choose(policy, env, deterministic=True)
            _, done = env.step(decision["submissions"], decision["matching_index"])
        assert done
        row = {"case": case, "arm": policy.arm, "seed": seed,
               "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
               "checkpoint": label, **env.metrics()}
        if row["invalid_assignments"] != 0 or abs(row["episode_return"] + row["objective"]) > 1e-8:
            raise RuntimeError("evaluation feasibility/accounting gate")
        rows.append(row)
        if episode_path is not None:
            append_jsonl(episode_path, row)
    return rows


def train_trial(run, case, arm, seed, *, steps=8192, eval_seeds=DEV_SEEDS):
    trial = run / f"{case}_{arm}_{seed}"
    trial.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    policy = PilotPolicy(arm)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    env = CoordinationEnv(configs()[case])
    episode_index = 0
    env.reset(role_seed(seed, f"pilot_train:{case}:{episode_index}"))
    started = time.monotonic()
    deadline = started + TRIAL_TIMEOUT
    record = {"status": "RUNNING", "case": case, "arm": arm, "train_seed": seed,
              "requested_transitions": steps, "actual_transitions": 0,
              "parameter_count": policy.parameter_count, "forecast_transitions": 0,
              "ppo": {"n_steps": 256, "minibatch": 64, "epochs": 4, "learning_rate": 3e-4,
                      "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .01}}
    write_json(trial / "manifest.json", record)
    dev_means = []
    try:
        with tqdm(total=steps, desc=f"Train {case} {arm} {seed}") as bar:
            for milestone in ((steps,) if steps < MILESTONES[0] else MILESTONES):
                if milestone > steps:
                    break
                while record["actual_transitions"] < milestone:
                    trajectory = []
                    for _ in range(min(256, milestone - record["actual_transitions"])):
                        if time.monotonic() > deadline:
                            raise TimeoutError("two-hour trial timeout")
                        decision = choose(policy, env, deterministic=False)
                        reward, done = env.step(decision["submissions"], decision["matching_index"])
                        trajectory.append(Transition(
                            decision["pre_obs"], decision["machine_obs"], decision["eligible"],
                            decision["request_bits"], decision["coordinator_obs"],
                            decision["matching_mask"], decision["matching_index"],
                            decision["old_logprob"], decision["value"], reward, done,
                        ))
                        record["actual_transitions"] += 1
                        bar.update(1)
                        if done:
                            append_jsonl(trial / "training_episodes.jsonl",
                                         {"episode": episode_index, "transition": record["actual_transitions"],
                                          **env.metrics()})
                            episode_index += 1
                            env.reset(role_seed(seed, f"pilot_train:{case}:{episode_index}"))
                    with torch.no_grad():
                        bootstrap = float(policy.value(_tensor(env.pre_observation())))
                    update = ppo_update(policy, optimizer, trajectory, bootstrap)
                    append_jsonl(trial / "updates.jsonl",
                                 {"transition": record["actual_transitions"], **update})
                    write_json(trial / "manifest.json", record)
                checkpoint = trial / f"checkpoint_{milestone}.pt"
                torch.save({"arm": arm, "case": case, "train_seed": seed,
                            "transition": milestone, "state_dict": policy.state_dict()}, checkpoint)
                restored = load_policy(checkpoint)
                rows = evaluate_policy(restored, case, eval_seeds, trial / "dev_episodes.jsonl",
                                       label=str(milestone), deadline=deadline)
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


def load_policy(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = PilotPolicy(payload["arm"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy


def evaluate_controls(run, seeds, *, panel):
    config_by_case = configs()
    rows = []
    for case in CASES:
        for name in ("risk_matching", "rollout"):
            for seed in tqdm(seeds, desc=f"Control {case} {name}"):
                controller = RuleController(name)
                row = evaluate_episode(config_by_case[case], case, seed, controller)
                rows.append({"panel": panel, "arm": "F_" + name, **row})
                append_jsonl(run / f"{panel}_controls.jsonl", rows[-1])
        for train_seed in (72600000, 72600001, 72600002, 72600003, 72600004):
            checkpoint = ROOT / f"baselines/v14/checkpoints/count_calibrated/{train_seed}"
            candidates = list(checkpoint.glob("*.pt"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one selected V14 checkpoint in {checkpoint}")
            for seed in tqdm(seeds, desc=f"Control {case} V14 {train_seed}"):
                controller = V14Controller.from_checkpoint(candidates[0])
                row = evaluate_episode(config_by_case[case], case, seed, controller)
                rows.append({"panel": panel, "arm": "A", "historical_train_seed": train_seed, **row})
                append_jsonl(run / f"{panel}_controls.jsonl", rows[-1])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "replicate", "test"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--replication-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode in ("replicate", "test") and args.training_run is None:
        parser.error("--training-run is required")
    run, manifest = new_run(args.output_root, args.mode)
    started = time.monotonic()
    print(run.resolve(), flush=True)
    try:
        if args.mode == "smoke":
            trials = []
            for case in CASES:
                for arm in ARMS:
                    trials.append(train_trial(run, case, arm, 73100009, steps=256, eval_seeds=(73100010,)))
                    manifest["trials"] = trials
                    write_json(run / "manifest.json", manifest)
        elif args.mode == "train":
            trials = []
            for case in CASES:
                for arm in ARMS:
                    for seed in TRAIN_SEEDS:
                        trials.append(train_trial(run, case, arm, seed))
                        manifest["trials"] = trials
                        write_json(run / "manifest.json", manifest)
            manifest["dev_controls"] = len(evaluate_controls(run, DEV_SEEDS, panel="dev"))
        else:
            training_run = args.training_run.resolve()
            original = json.loads((training_run / "manifest.json").read_text())
            if original["status"] != "COMPLETED" or original["kind"] != "train":
                raise RuntimeError("training run is not complete")
            if original["protocol_sha256"] != manifest["protocol_sha256"] or original["source_sha256"] != manifest["source_sha256"]:
                raise RuntimeError("source/protocol changed since training")
            if args.mode == "test":
                if args.replication_run is None:
                    raise RuntimeError("sealed TEST requires --replication-run")
                replication_run = args.replication_run.resolve()
                gate = replication_run / "test_go_decision.json"
                replication_manifest = json.loads((replication_run / "manifest.json").read_text())
                if (replication_manifest["status"] != "COMPLETED"
                        or replication_manifest["kind"] != "replicate"
                        or replication_manifest["training_run"] != str(training_run)
                        or not gate.exists()
                        or json.loads(gate.read_text()).get("go") is not True
                        or json.loads(gate.read_text()).get("protocol_sha256") != manifest["protocol_sha256"]):
                    raise RuntimeError("sealed TEST requires recorded positive replication gate")
            panel = "replication" if args.mode == "replicate" else "test"
            seeds = REPLICATION_SEEDS if panel == "replication" else TEST_SEEDS
            manifest["training_run"] = str(training_run)
            count = 0
            for case in CASES:
                for arm in ARMS:
                    for seed in TRAIN_SEEDS:
                        trial = training_run / f"{case}_{arm}_{seed}"
                        selection = json.loads((trial / "selection.json").read_text())
                        policy = load_policy(trial / selection["checkpoint"])
                        rows = evaluate_policy(policy, case, seeds, run / f"{panel}_episodes.jsonl",
                                               label=f"{seed}:{selection['transition']}")
                        count += len(rows)
            manifest["learned_episodes"] = count
            manifest["control_episodes"] = len(evaluate_controls(run, seeds, panel=panel))
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
