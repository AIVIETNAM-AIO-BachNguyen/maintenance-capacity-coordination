"""Prelocked v14 scaled N5/K3 same-tick PPO pilot; no TEST execution mode."""
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
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs, evaluate_controls
from maintenance_capacity_coordination.same_tick_policy_v14 import (
    SequencePolicy, SequenceTransition, choose as sequence_choose, ppo_update as sequence_ppo,
)
from maintenance_capacity_coordination.scaled_matching_policy_v14 import MatchingPolicy, Transition, choose as matching_choose, ppo_update as matching_ppo

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/canonical_matching_v14_prelock.md"
ARMS = ("C14", "D14", "B14", "U14", "S14", "G14", "Q14")
TRAIN_ROOTS = (76710000, 76710001, 76710002)
DEV_SEEDS = tuple(range(76711000, 76711016))
HELDOUT_SEEDS = tuple(range(76712000, 76712032))
BOOTSTRAP_SEED = 76713000
REPL_TRAIN_ROOTS = (76720000, 76720001, 76720002)
REPL_DEV_SEEDS = tuple(range(76721000, 76721016))
REPL_EVAL_SEEDS = tuple(range(76722000, 76722032))
REPL_BOOTSTRAP_SEED = 76723000
SEALED_TEST = tuple(range(76730000, 76730032))
SMOKE_ROOT = 76700000
SMOKE_SEEDS = (76700001, 76700002)
MILESTONES = (2048, 4096, 8192)
TRIAL_TIMEOUT = 7200


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


def seed_registry_check():
    groups = ((SMOKE_ROOT,), SMOKE_SEEDS, TRAIN_ROOTS, DEV_SEEDS, HELDOUT_SEEDS,
              (BOOTSTRAP_SEED,), REPL_TRAIN_ROOTS, REPL_DEV_SEEDS, REPL_EVAL_SEEDS,
              (REPL_BOOTSTRAP_SEED,), SEALED_TEST)
    raw = [seed for group in groups for seed in group]
    if len(raw) != len(set(raw)) or not all(76700000 <= seed < 76800000 for seed in raw):
        raise RuntimeError("v14 raw seed roles overlap")
    eval_raw = [*SMOKE_SEEDS, *DEV_SEEDS, *HELDOUT_SEEDS, *REPL_DEV_SEEDS, *REPL_EVAL_SEEDS, *SEALED_TEST]
    derived = {role_seed(seed, BASE_VERSION + ":physical") for seed in eval_raw}
    if len(derived) != len(eval_raw):
        raise RuntimeError("v14 derived evaluation physical seeds overlap")
    # Prior raw roles are below 76700000; include their whole reserved bands.
    prior_raw = (*range(72600000, 72600005), *range(73100000, 73800000),
                 *range(74700000, 75800000))
    prior_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior_raw}
    from experiments.second_choice_value_v13 import bank_seed
    prior_banks = {bank_seed(raw, case, root, tick, slot)
                   for raw in (*range(75700001, 75700003), *range(75711000, 75711016))
                   for case in CASES for root in (74710000, 74710001, 74710002)
                   for tick in range(8, 40) for slot in range(8)}
    nested_v12 = {role_seed(role_seed(root, f"same_tick_v12:{case}:{episode}"),
                            BASE_VERSION + ":physical")
                  for root in (74700000, 74710000, 74710001, 74710002,
                               74720000, 74720001, 74720002)
                  for case in CASES for episode in range(200)}
    prior_physical |= prior_banks | nested_v12
    if derived & prior_physical:
        raise RuntimeError("v14/prior derived physical tape overlap")
    for root in (SMOKE_ROOT, *TRAIN_ROOTS, *REPL_TRAIN_ROOTS):
        for case in CASES:
            for episode in range(200):
                reset = role_seed(root, f"same_tick_v14:{case}:{episode}")
                physical = role_seed(reset, BASE_VERSION + ":physical")
                if physical in derived or physical in prior_physical:
                    raise RuntimeError("v14 training/evaluation/prior physical tape overlap")
                derived.add(physical)


def new_run(output_root, kind):
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / (f"same_tick_v14_{kind}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(exist_ok=False)
    source = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "experiments/same_tick_pilot_v14.py", ROOT / "experiments/scaled_n5k3_v12_config.py", ROOT / "pyproject.toml"]
    cfg = configs()
    manifest = {
        "status": "RUNNING", "kind": kind, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
        "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                          for case in CASES},
        "cases": CASES, "config_seeds": CONFIG_SEEDS, "arms": ARMS, "train_roots": TRAIN_ROOTS, "dev_seeds": DEV_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS, "bootstrap_seed": BOOTSTRAP_SEED,
        "replication_train_roots": REPL_TRAIN_ROOTS, "replication_dev_seeds": REPL_DEV_SEEDS,
        "replication_eval_seeds": REPL_EVAL_SEEDS, "replication_bootstrap_seed": REPL_BOOTSTRAP_SEED,
        "sealed_test_seeds": SEALED_TEST, "smoke_root": SMOKE_ROOT, "smoke_seeds": SMOKE_SEEDS,
        "milestones": MILESTONES, "trial_timeout_seconds": TRIAL_TIMEOUT,
    }
    write_json(run / "manifest.json", manifest)
    append_jsonl(run / "run.log", {"event": "started", "utc": manifest["started_utc"], "kind": kind})
    return run, manifest


def new_policy(arm):
    return SequencePolicy(arm) if arm in ("B14", "C14", "D14", "U14") else MatchingPolicy(arm)


def load_policy(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = new_policy(payload["arm"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy


def choose(policy, env, deterministic):
    return sequence_choose(policy, env, deterministic) if isinstance(policy, SequencePolicy) else matching_choose(policy, env, deterministic)


def evaluate_policy(policy, arm, case, seeds, path=None, *, label=None, deadline=None, trace_path=None):
    env = ScaledCoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f"v14 eval {case} {arm} {label}", leave=False):
        if deadline and time.monotonic() > deadline:
            raise TimeoutError("trial timeout during DEV")
        env.reset(seed)
        selection_seconds, decoder_passes, entropy_sum, cardinality_sum = 0., 0, 0., 0
        for tick in range(48):
            started = time.monotonic()
            decision = choose(policy, env, True)
            selection_seconds += time.monotonic() - started
            reward, done = env.step(decision["submissions"], decision["matching_index"])
            decoder_passes += decision.get("decoder_forward_passes", 1)
            entropy_sum += decision.get("sequence_entropy", 0.)
            cardinality_sum += len(env.actions[decision["matching_index"]])
            if trace_path is not None:
                append_jsonl(trace_path, {"case": case, "arm": arm, "seed": seed, "tick": tick,
                                          "submissions": decision["submissions"],
                                          "matching_index": decision["matching_index"],
                                          "matching": env.actions[decision["matching_index"]],
                                          "sequence": decision.get("tokens"),
                                          "logprob": decision["old_logprob"], "reward": reward,
                                          "decoder_forward_passes": decision.get("decoder_forward_passes", 1)})
        assert done
        row = {"case": case, "arm": arm, "seed": seed,
               "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
               "checkpoint": label, "selection_seconds": selection_seconds,
               "decoder_forward_passes": decoder_passes,
               "sequence_entropy_sum": entropy_sum, "matching_cardinality_sum": cardinality_sum,
               **env.metrics()}
        if (row["invalid_assignments"] != 0 or abs(row["episode_return"] + row["objective"]) > 1e-8
                or abs(row["objective"] - row["H_cost"] - row["tail_cost"]) > 1e-8):
            raise RuntimeError("v14 evaluation feasibility/accounting gate")
        rows.append(row)
        if path is not None:
            append_jsonl(path, row)
    return rows


def train_trial(run, case, arm, root, *, steps=8192, eval_seeds=DEV_SEEDS):
    trial = run / f"{case}_{arm}_{root}"
    trial.mkdir(exist_ok=False)
    torch.manual_seed(root)
    np.random.seed(root % 2**32)
    policy = new_policy(arm)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    env = ScaledCoordinationEnv(configs()[case])
    episode = 0
    env.reset(role_seed(root, f"same_tick_v14:{case}:{episode}"))
    started = time.monotonic()
    deadline = started + TRIAL_TIMEOUT
    record = {"status": "RUNNING", "case": case, "arm": arm, "train_root": root,
              "requested_H_decisions": steps, "actual_H_decisions": 0,
              "decoder_forward_passes": 0, "forecast_transitions": 0,
              "actor_parameters": policy.actor_parameters, "critic_parameters": policy.critic_parameters,
              "ppo": {"n_steps": 256, "minibatch": 64, "epochs": 4, "learning_rate": 3e-4,
                      "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .01}}
    write_json(trial / "manifest.json", record)
    dev_means = []
    try:
        with tqdm(total=steps, desc=f"v14 train {case} {arm} {root}") as bar:
            for milestone in ((steps,) if steps < MILESTONES[0] else MILESTONES):
                if milestone > steps:
                    break
                while record["actual_H_decisions"] < milestone:
                    trajectory = []
                    for _ in range(min(256, milestone - record["actual_H_decisions"])):
                        if time.monotonic() > deadline:
                            raise TimeoutError("two-hour v14 trial timeout")
                        tick = env.tick
                        decision = choose(policy, env, False)
                        reward, done = env.step(decision["submissions"], decision["matching_index"])
                        append_jsonl(trial / "training_decisions.jsonl", {
                            "case": case, "arm": arm, "train_root": root, "episode": episode,
                            "tick": tick, "reset_seed": role_seed(root, f"same_tick_v14:{case}:{episode}"),
                            "submissions": decision["submissions"],
                            "matching_index": decision["matching_index"],
                            "sequence": decision.get("tokens"), "logprob": decision["old_logprob"],
                            "reward": reward, "done": done,
                            "decoder_forward_passes": decision.get("decoder_forward_passes", 1)})
                        if isinstance(policy, SequencePolicy):
                            trajectory.append(SequenceTransition(
                                decision["pre_obs"], decision["machine_obs"], decision["worker_obs"],
                                decision["edge_mask"], decision["sequence"], decision["old_logprob"],
                                decision["value"], reward, done))
                        else:
                            trajectory.append(Transition(
                                decision["pre_obs"], decision["machine_obs"], decision["worker_obs"],
                                decision["matching_mask"], decision["matching_index"],
                                decision["old_logprob"], decision["value"], reward, done))
                        record["decoder_forward_passes"] += decision.get("decoder_forward_passes", 1)
                        record["actual_H_decisions"] += 1
                        bar.update(1)
                        if done:
                            append_jsonl(trial / "training_episodes.jsonl",
                                         {"episode": episode, "transition": record["actual_H_decisions"],
                                          **env.metrics()})
                            episode += 1
                            env.reset(role_seed(root, f"same_tick_v14:{case}:{episode}"))
                    with torch.no_grad():
                        bootstrap = float(policy.value(torch.from_numpy(env.pre_observation())))
                    update = sequence_ppo(policy, optimizer, trajectory, bootstrap) if isinstance(policy, SequencePolicy) else matching_ppo(policy, optimizer, trajectory, bootstrap)
                    append_jsonl(trial / "updates.jsonl", {"transition": record["actual_H_decisions"], **update})
                    write_json(trial / "manifest.json", record)
                checkpoint = trial / f"checkpoint_{milestone}.pt"
                torch.save({"arm": arm, "case": case, "train_root": root,
                            "transition": milestone, "state_dict": policy.state_dict()}, checkpoint)
                restored = load_policy(checkpoint)
                rows = evaluate_policy(restored, arm, case, eval_seeds, trial / "dev_episodes.jsonl",
                                       label=str(milestone), deadline=deadline)
                dev_means.append({"transition": milestone,
                                  "mean_H_tail_cost": float(np.mean([r["objective"] for r in rows])),
                                  "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint)})
                write_json(trial / "dev_milestones.json", dev_means)
        selected = min(dev_means, key=lambda r: (r["mean_H_tail_cost"], r["transition"]))
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
    current = json.loads((run / "manifest.json").read_text())
    expected = "train" if panel == "heldout" else "replication_train"
    if original["status"] != "COMPLETED" or original["kind"] != expected:
        raise RuntimeError("wrong or incomplete training run")
    audit_path = training_run / "audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text()).get("status") != "PASS":
        raise RuntimeError("passing training audit required")
    for key in ("protocol_sha256", "source_sha256", "config_sha256"):
        if original[key] != current[key]:
            raise RuntimeError(f"{key} changed since training")
    roots = TRAIN_ROOTS if panel == "heldout" else REPL_TRAIN_ROOTS
    count = 0
    for case in CASES:
        for arm in ARMS:
            for root in roots:
                trial = training_run / f"{case}_{arm}_{root}"
                selected = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selected["checkpoint"]
                if sha256(checkpoint) != selected["checkpoint_sha256"]:
                    raise RuntimeError("selected checkpoint hash changed")
                policy = load_policy(checkpoint)
                count += len(evaluate_policy(policy, arm, case, seeds,
                                             run / f"{panel}_episodes.jsonl",
                                             label=f"{root}:{selected['transition']}",
                                             trace_path=run / f"{panel}_decisions.jsonl"))
                if arm == "C14":
                    frozen = SequencePolicy("D14")
                    frozen.load_state_dict(policy.state_dict(), strict=True)
                    frozen.eval()
                    count += len(evaluate_policy(frozen, "CF14", case, seeds,
                                                 run / f"{panel}_episodes.jsonl",
                                                 label=f"{root}:{selected['transition']}",
                                                 trace_path=run / f"{panel}_decisions.jsonl"))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "heldout", "replication_train", "replication_eval"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--pilot-heldout-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_registry_check()
    run, manifest = new_run(args.output_root, args.mode)
    print(run.resolve(), flush=True)
    started = time.monotonic()
    try:
        if args.mode in ("smoke", "train", "replication_train"):
            if args.mode == "replication_train":
                if args.pilot_heldout_run is None:
                    raise RuntimeError("audited pilot held-out run required for replication")
                heldout = args.pilot_heldout_run.resolve()
                heldout_manifest = json.loads((heldout / "manifest.json").read_text())
                heldout_audit = json.loads((heldout / "audit.json").read_text())
                decision_path = heldout / "replication_go_decision.json"
                decision = json.loads(decision_path.read_text())
                if (heldout_manifest.get("kind") != "heldout" or heldout_manifest.get("status") != "COMPLETED"
                        or heldout_audit.get("status") != "PASS"
                        or heldout_audit.get("decision_sha256") != sha256(decision_path)
                        or decision.get("go") is not True
                        or decision.get("protocol_sha256") != manifest["protocol_sha256"]
                        or heldout_manifest.get("source_sha256") != manifest["source_sha256"]
                        or heldout_manifest.get("config_sha256") != manifest["config_sha256"]):
                    raise RuntimeError("audited v14 replication gate did not pass")
            roots = (SMOKE_ROOT,) if args.mode == "smoke" else (TRAIN_ROOTS if args.mode == "train" else REPL_TRAIN_ROOTS)
            cases = CASES if args.mode != "smoke" else (CASES[0],)
            seeds = SMOKE_SEEDS if args.mode == "smoke" else (DEV_SEEDS if args.mode == "train" else REPL_DEV_SEEDS)
            steps = 256 if args.mode == "smoke" else 8192
            trials = []
            for case in cases:
                for arm in ARMS:
                    for root in roots:
                        trials.append(train_trial(run, case, arm, root, steps=steps, eval_seeds=seeds))
                        append_jsonl(run / "run.log", {"event": "trial_completed", "utc": datetime.now(timezone.utc).isoformat(),
                                                       "case": case, "arm": arm, "train_root": root})
                        manifest["trials"] = trials
                        write_json(run / "manifest.json", manifest)
        else:
            if args.training_run is None:
                raise RuntimeError("--training-run required")
            panel = "heldout" if args.mode == "heldout" else "replication_eval"
            seeds = HELDOUT_SEEDS if panel == "heldout" else REPL_EVAL_SEEDS
            manifest["training_run"] = str(args.training_run.resolve())
            manifest["learned_episodes"] = evaluate_selected(run, args.training_run.resolve(), seeds, panel)
            controls = evaluate_controls(run, seeds, panel=panel)
            manifest["control_episodes"] = len(controls)
        manifest["status"] = "COMPLETED"
        append_jsonl(run / "run.log", {"event": "completed", "utc": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        append_jsonl(run / "run.log", {"event": "failed", "utc": datetime.now(timezone.utc).isoformat(),
                                       "error": repr(error)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
