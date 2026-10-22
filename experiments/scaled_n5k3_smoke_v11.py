"""Bounded N5/K3 engineering smoke and independent episode replay; no pilot outcomes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.scaled_n5k3_v11_config import CASES, configs
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v11 import (
    SequencePolicy, SequenceTransition, choose as choose_sequence, ppo_update as sequence_ppo,
)
from maintenance_capacity_coordination.scaled_matching_policy_v11 import (
    MatchingPolicy, Transition, choose as choose_matching, ppo_update as matching_ppo,
)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("C11", "D11", "B11", "S11", "G11", "Q11")
TRAIN_ROOT = 73700000
EVAL_SEED = 73700001


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def new_dir(kind):
    path = ROOT / "artifacts" / ("same_tick_v11_" + kind + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    path.mkdir(exist_ok=False)
    return path


def choose(policy, env, deterministic):
    return choose_sequence(policy, env, deterministic) if isinstance(policy, SequencePolicy) else choose_matching(policy, env, deterministic)


def new_policy(arm):
    return SequencePolicy(arm) if arm in ("C11", "D11", "B11") else MatchingPolicy(arm)


def evaluate(policy, case, seed, trace_path, arm):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(seed)
    for tick in range(HORIZON):
        d = choose(policy, env, True)
        reward, done = env.step(d["submissions"], d["matching_index"])
        append(trace_path, {"case": case, "arm": arm, "seed": seed, "tick": tick,
                            "submissions": d["submissions"], "matching_index": d["matching_index"],
                            "reward": reward, "done": done})
    assert done
    return env.metrics()


def run():
    torch.set_num_threads(1)
    path = new_dir("smoke")
    cfg = configs()
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "experiments/scaled_n5k3_v11_config.py"]
    manifest = {"status": "RUNNING", "kind": "engineering_smoke",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "torch": torch.__version__,
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
                "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest() for case in CASES},
                "cases": CASES, "arms": ARMS, "train_root": TRAIN_ROOT, "eval_seed": EVAL_SEED,
                "trials": []}
    write(path / "manifest.json", manifest)
    started = time.monotonic()
    print(path.resolve(), flush=True)
    try:
        for case in CASES:
            for arm in ARMS:
                torch.manual_seed(TRAIN_ROOT)
                policy = new_policy(arm)
                optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
                env = ScaledCoordinationEnv(cfg[case])
                reset_seed = role_seed(TRAIN_ROOT, f"v11_smoke:{case}:{arm}")
                env.reset(reset_seed)
                trajectory = []
                for _ in tqdm(range(64), desc=f"v11 smoke {case} {arm}", leave=False):
                    d = choose(policy, env, False)
                    reward, done = env.step(d["submissions"], d["matching_index"])
                    if isinstance(policy, SequencePolicy):
                        trajectory.append(SequenceTransition(
                            d["pre_obs"], d["machine_obs"], d["worker_obs"],
                            d["edge_mask"], d["sequence"], d["old_logprob"],
                            d["value"], reward, done))
                    else:
                        trajectory.append(Transition(
                            d["pre_obs"], d["machine_obs"], d["worker_obs"],
                            d["matching_mask"], d["matching_index"], d["old_logprob"],
                            d["value"], reward, done))
                    if done:
                        env.reset(role_seed(reset_seed, "next"))
                with torch.no_grad():
                    bootstrap = float(policy.value(torch.from_numpy(env.pre_observation())))
                update = (sequence_ppo if isinstance(policy, SequencePolicy) else matching_ppo)(policy, optimizer, trajectory, bootstrap)
                checkpoint = path / f"{case}_{arm}.pt"
                torch.save({"arm": arm, "state_dict": policy.state_dict()}, checkpoint)
                restored = new_policy(arm)
                restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
                restored.eval()
                metrics = evaluate(restored, case, EVAL_SEED, path / "decisions.jsonl", arm)
                append(path / "episodes.jsonl", {"case": case, "arm": arm, "seed": EVAL_SEED,
                                                    "checkpoint_sha256": sha(checkpoint), **metrics})
                manifest["trials"].append({"case": case, "arm": arm, "update": update,
                                           "actor_parameters": policy.actor_parameters,
                                           "critic_parameters": policy.critic_parameters})
                write(path / "manifest.json", manifest)
            for name in ("risk_matching", "rollout"):
                controller = RuleController(name)
                env = ScaledCoordinationEnv(cfg[case])
                env.reset(EVAL_SEED)
                for tick in range(HORIZON):
                    submissions, matching = controller.choose(cfg[case], env.inner.state, tick,
                                                              case=case, seed=EVAL_SEED)
                    index = env.actions.index(matching)
                    reward, done = env.step(submissions, index)
                    append(path / "decisions.jsonl", {"case": case, "arm": "F_" + name,
                                                        "seed": EVAL_SEED, "tick": tick,
                                                        "submissions": submissions,
                                                        "matching_index": index, "reward": reward,
                                                        "done": done})
                assert done
                append(path / "episodes.jsonl", {"case": case, "arm": "F_" + name,
                                                    "seed": EVAL_SEED, "forecast_transitions": controller.forecast_transitions,
                                                    **env.metrics()})
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {p.name: sha(p) for p in path.iterdir() if p.is_file() and p.name != "manifest.json"}
        write(path / "manifest.json", manifest)
    return path


def audit(path):
    review = new_dir("smoke_audit")
    result = {"status": "RUNNING", "source_run": str(path.resolve()),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write(review / "audit.json", result)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["status"] == "COMPLETED"
        for name, expected in manifest["source_sha256"].items():
            assert sha(ROOT / name) == expected, name
        cfg = configs()
        for case, expected in manifest["config_sha256"].items():
            assert hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest() == expected
        for name, expected in manifest["output_sha256"].items():
            assert sha(path / name) == expected
        rows = [json.loads(x) for x in (path / "decisions.jsonl").read_text().splitlines()]
        episodes = [json.loads(x) for x in (path / "episodes.jsonl").read_text().splitlines()]
        assert len(rows) == len(episodes) * HORIZON == len(CASES) * (len(ARMS) + 2) * HORIZON
        cursor = 0
        for episode in episodes:
            case, arm, seed = episode["case"], episode["arm"], episode["seed"]
            env = ScaledCoordinationEnv(cfg[case]); env.reset(seed)
            if arm in ARMS:
                checkpoint = path / f"{case}_{arm}.pt"
                assert sha(checkpoint) == episode["checkpoint_sha256"]
                policy = new_policy(arm)
                policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
                policy.eval()
            else:
                controller = RuleController(arm.removeprefix("F_"))
            for tick in range(HORIZON):
                row = rows[cursor]; cursor += 1
                assert (row["case"], row["arm"], row["seed"], row["tick"]) == (case, arm, seed, tick)
                if arm in ARMS:
                    decision = choose(policy, env, True)
                    submissions, index = decision["submissions"], decision["matching_index"]
                else:
                    submissions, matching = controller.choose(cfg[case], env.inner.state, tick,
                                                              case=case, seed=seed)
                    index = env.actions.index(matching)
                assert list(submissions) == row["submissions"] and index == row["matching_index"]
                reward, done = env.step(submissions, index)
                assert abs(reward - row["reward"]) < 1e-7 and done == row["done"]
            assert abs(env.metrics()["objective"] - episode["objective"]) < 1e-7
            assert abs(env.metrics()["episode_return"] + episode["objective"]) < 1e-7
            assert episode["invalid_assignments"] == 0
            if arm not in ARMS:
                assert controller.forecast_transitions == episode["forecast_transitions"]
        assert cursor == len(rows)
        result.update(status="PASS", episodes=len(episodes), decisions=len(rows),
                      source_manifest_sha256=sha(path / "manifest.json"))
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(review / "audit.json", result)
    print(review.resolve(), flush=True)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "audit"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        run()
    else:
        if args.run is None:
            parser.error("--run required")
        audit(args.run)


if __name__ == "__main__":
    main()
