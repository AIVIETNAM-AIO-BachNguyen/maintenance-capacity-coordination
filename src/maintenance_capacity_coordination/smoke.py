"""Bounded macOS/CPU engineering validation, not a research benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import (
    FrameworkEnv, LearnedController, RuleController, configs, evaluate_episode,
)
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from maintenance_capacity_coordination.baselines import V14Controller, make_ppo


def write_json(destination, value):
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    run = args.output_root / ("migration_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_files = sorted((root / "src").rglob("*.py"))
    source_files += [root / "docs/migration_validation_protocol.md", root / "pyproject.toml"]
    manifest = {
        "status": "RUNNING", "kind": "engineering_smoke", "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "training_seed": 62500001, "evaluation_seed": 62500002,
        "requested_training_steps": 64, "expected_evaluation_episodes": 16,
        "source_hashes": {str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest()
                          for source in source_files},
    }
    for label, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                           ("git_status", ["git", "status", "--porcelain"])):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        manifest[label] = result.stdout.strip() if result.returncode == 0 else None
    write_json(run / "manifest.json", manifest)

    def check_deadline():
        if time.monotonic() - started > 120:
            raise TimeoutError("120-second engineering budget exceeded")

    class DeadlineCallback(BaseCallback):
        def _on_step(self):
            check_deadline()
            return True

    try:
        configuration = configs()
        env = FrameworkEnv(configuration["reciprocal_scale2"])
        model = make_ppo(env, 62500001)
        model.learn(total_timesteps=64, progress_bar=True, callback=DeadlineCallback())
        model.save(run / "ppo")
        reloaded = MaskablePPO.load(run / "ppo.zip", device="cpu")
        observation, _ = env.reset(seed=62500002)
        for _ in range(48):
            check_deadline()
            mask = env.action_masks()
            before, _ = model.predict(observation, action_masks=mask, deterministic=True)
            after, _ = reloaded.predict(observation, action_masks=mask, deterministic=True)
            if int(before) != int(after):
                raise RuntimeError("PPO checkpoint action mismatch")
            observation, _, done, _, _ = env.step(before)
            if done:
                break

        torch.manual_seed(62500000)
        actor = OpportunityCostActor("count_calibrated")
        torch.save({"metadata": {"arm": actor.mode, "untrained": True},
                    "state_dict": actor.state_dict()}, run / "v14_initialized.pt")
        restored_actor = V14Controller.from_checkpoint(run / "v14_initialized.pt").actor
        for config in configuration.values():
            check_deadline()
            actor_env = FrameworkEnv(config)
            obs, _ = actor_env.reset(seed=62500002)
            with torch.no_grad():
                tensor = torch.from_numpy(obs).reshape(1, 56)
                if not torch.equal(actor(tensor), restored_actor(tensor)):
                    raise RuntimeError("V14 checkpoint logit mismatch")
            actor_env.close()

        episodes = []
        tasks = [(case, name) for case in configuration for name in ("fcfs", "risk_matching", "rollout", "ppo")]
        for case, name in tqdm(tasks, desc="Engineering baseline evaluation"):
            check_deadline()
            controller = LearnedController(reloaded) if name == "ppo" else RuleController(name, check_deadline)
            episode = evaluate_episode(configuration[case], case, 62500002, controller)
            episode["controller"] = name
            if episode["invalid_assignments"] != 0 or not np.isclose(episode["episode_return"], -episode["objective"]):
                raise RuntimeError("episode feasibility/accounting failure")
            episodes.append(episode)
            write_json(run / "episodes.json", episodes)
        env.close()
        manifest.update(status="COMPLETED", actual_training_steps=model.num_timesteps,
                        evaluation_episodes=len(episodes), checkpoint_roundtrip="PASS")
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["output_hashes"] = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                     for item in run.iterdir() if item.is_file() and item.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)
        print(run.resolve())


if __name__ == "__main__":
    main()
