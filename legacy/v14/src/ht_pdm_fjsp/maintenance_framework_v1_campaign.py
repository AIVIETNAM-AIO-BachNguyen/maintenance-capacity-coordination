"""Bounded CPU campaign for maintenance coordination framework v1."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES, FrameworkEnv, LearnedController, RuleController, VERSION, configs,
    evaluate_episode,
)
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/maintenance_framework_v1_protocol_vi.md"
REGISTRY = ROOT / "configs/maintenance_framework_v1_seed_registry.json"
SOURCES = (
    ROOT / "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    ROOT / "src/ht_pdm_fjsp/maintenance_framework_v1_campaign.py",
    ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
    ROOT / "src/ht_pdm_fjsp/maintenance_requests.py",
    ROOT / "src/ht_pdm_fjsp/maintenance_request_policies.py",
    PROTOCOL, REGISTRY, ROOT / "pyproject.toml", ROOT / "uv.lock",
)
FULL_STEPS = (50176, 100352, 149504, 199680)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_registry(profile):
    registry = json.loads(REGISTRY.read_text())
    historical = set()
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        if path != REGISTRY:
            historical |= registered_integers(json.loads(path.read_text()))
    def panel(name):
        value = registry[name]
        return list(range(value["start"], value["start"] + value["count"]))
    groups = [panel(key) for key in ("training", "development", "sealed_test", "smoke")]
    groups.extend([[registry["forecast_seed"]], [registry["bootstrap_seed"]]])
    flat = [item for group in groups for item in group]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise RuntimeError("seed overlap with historical registry or another split")
    return registry, groups


def git_info():
    def command(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    changed = command("diff", "--name-only", "HEAD", "--", "src", "configs", "pyproject.toml", "uv.lock", "docs/maintenance_framework_v1_protocol_vi.md")
    untracked = command("ls-files", "--others", "--exclude-standard", "--", "src", "configs", "docs/maintenance_framework_v1_protocol_vi.md")
    return dict(commit=command("rev-parse", "HEAD"), branch=command("branch", "--show-current"),
                dirty=bool(command("status", "--porcelain")), blocking_paths=sorted(path for path in set((changed + "\n" + untracked).splitlines()) if path))


class ProgressCallback(BaseCallback):
    def __init__(self, out, train_seed, steps, deadline):
        super().__init__(verbose=0)
        self.out, self.train_seed, self.steps, self.deadline = out, train_seed, steps, deadline
        self.last = 0
        self.next_index = 0
        self.bar = tqdm(total=steps[-1], desc=f"PPO {train_seed}", unit="step")
        self.progress_file = out / "training_progress.csv"
        self.checkpoint_dir = out / "checkpoints" / str(train_seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        current = self.num_timesteps
        if current - self.last >= 256 or current == self.steps[-1]:
            self.bar.update(current - self.last)
            self.last = current
            if time.monotonic() >= self.deadline:
                raise TimeoutError("campaign six-hour deadline during training")
        if self.next_index < len(self.steps) and current == self.steps[self.next_index]:
            path = self.checkpoint_dir / f"{current}.zip"
            self.model.save(path)
            with self.progress_file.open("a", newline="") as f:
                csv.writer(f).writerow([self.train_seed, current, str(path.relative_to(self.out)), time.time()])
            self.next_index += 1
        return True

    def _on_training_end(self):
        self.bar.close()


def model_new(env, seed):
    return MaskablePPO(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0, n_steps=1024,
        batch_size=256, n_epochs=5, gamma=.99, gae_lambda=.95,
        clip_range=.2, learning_rate=3e-4, ent_coef=.01,
        vf_coef=.5, max_grad_norm=.5, policy_kwargs={"net_arch": [64, 64]},
    )


def add_row(path, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check_checkpoint(path, config, seed):
    model = MaskablePPO.load(path, device="cpu")
    env = FrameworkEnv(config)
    obs, _ = env.reset(seed=seed)
    mask = env.action_masks()
    first = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    again = int(MaskablePPO.load(path, device="cpu").predict(obs, action_masks=mask, deterministic=True)[0])
    if first != again or not mask[first]:
        raise RuntimeError("checkpoint reload/mask audit")
    return model


def summarize(rows, bootstrap_seed):
    result = {}
    rng = np.random.default_rng(bootstrap_seed)
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        by_controller = {}
        for controller in ("fcfs", "risk_matching", "rollout", "learner"):
            panel = sorted((r for r in case_rows if r["controller"] == controller), key=lambda r: int(r["seed"]))
            by_controller[controller] = panel
            metrics = ("objective", "H_availability", "H_failures", "queued_cm_ticks",
                       "queued_pm_ticks", "H_pending", "end_pending", "H_in_service",
                       "end_in_service", "maintenance_cost", "unavailability_cost",
                       "failure_cost", "tail_cost", "selection_seconds", "episode_wall_seconds",
                       "forecast_transitions")
            result[f"{case}/{controller}"] = {
                "count": len(panel), "means": {key: float(np.mean([float(r[key]) for r in panel])) for key in metrics}
            }
        learner = by_controller["learner"]
        for baseline in ("fcfs", "risk_matching", "rollout"):
            other = by_controller[baseline]
            if [int(r["seed"]) for r in learner] != [int(r["seed"]) for r in other]:
                raise RuntimeError("unpaired seed panel")
            delta = np.array([float(a["objective"]) - float(b["objective"]) for a, b in zip(learner, other)])
            draws = rng.integers(0, len(delta), size=(2000, len(delta)))
            lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            result[f"{case}/learner_minus_{baseline}"] = {
                "paired_mean": float(delta.mean()), "bootstrap_ci95": [float(lo), float(hi)],
                "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum()),
            }
    return result


def run(profile, output_dir):
    if profile not in ("smoke", "full"):
        raise ValueError(profile)
    out = Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("refusing to overwrite run")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.monotonic()
    deadline = start + (21600 if profile == "full" else 1200)
    registry, groups = seed_registry(profile)
    train_seeds, dev_seeds, test_seeds, smoke_seeds = groups[:4]
    git = git_info()
    if profile == "full" and git["blocking_paths"]:
        raise RuntimeError(f"full run requires committed inputs: {git['blocking_paths']}")
    cfgs = configs()
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}
    manifest = {
        "status": "RUNNING", "phase": "preflight", "profile": profile,
        "protocol": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git, "seed_registry": registry, "source_hashes": hashes,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "device": "cpu"},
        "deadline_seconds": 21600 if profile == "full" else 1200,
    }
    write_json(out / "manifest.json", manifest)
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(REGISTRY, out / "seed_registry.json")
    write_json(out / "resolved_config.json", {
        "cases": {name: asdict(c) for name, c in cfgs.items()},
        "steps": list(FULL_STEPS), "hyperparameters": {
            "n_steps": 1024, "batch_size": 256, "n_epochs": 5, "gamma": .99,
            "gae_lambda": .95, "clip_range": .2, "learning_rate": .0003,
            "ent_coef": .01, "vf_coef": .5, "max_grad_norm": .5, "net_arch": [64, 64]},
    })
    def deadline_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("campaign six-hour deadline")

    try:
        manifest["phase"] = "training"
        write_json(out / "manifest.json", manifest)
        steps = FULL_STEPS if profile == "full" else (1024,)
        seeds = train_seeds if profile == "full" else smoke_seeds[:1]
        with (out / "training_progress.csv").open("x", newline="") as f:
            csv.writer(f).writerow(["train_seed", "steps", "checkpoint", "epoch_seconds"])
        paths = []
        for seed in seeds:
            deadline_check()
            env = FrameworkEnv(cfgs["reciprocal_scale2"])
            env.reset(seed=seed)
            model = model_new(env, seed)
            cb = ProgressCallback(out, seed, steps, deadline)
            model.learn(total_timesteps=steps[-1], callback=cb, progress_bar=False)
            for step in steps:
                path = out / "checkpoints" / str(seed) / f"{step}.zip"
                if not path.exists():
                    raise RuntimeError(f"checkpoint missing: {path}")
                paths.append((seed, step, path))
            env.close()
        manifest["phase"] = "development"
        write_json(out / "manifest.json", manifest)
        development = []
        panel = dev_seeds if profile == "full" else smoke_seeds
        with tqdm(total=len(paths) * len(panel), desc="Development", unit="episode") as bar:
            for train_seed, step, path in paths:
                model = check_checkpoint(path, cfgs["reciprocal_scale2"], panel[0])
                for seed in panel:
                    deadline_check()
                    row = evaluate_episode(cfgs["reciprocal_scale2"], "reciprocal_scale2", seed,
                                           LearnedController(model))
                    row.update(train_seed=train_seed, checkpoint_step=step,
                               checkpoint=str(path.relative_to(out)))
                    development.append(row)
                    add_row(out / "development.partial.csv", row)
                    bar.update(1)
        shutil.copy2(out / "development.partial.csv", out / "development.csv")
        scores = []
        for train_seed, step, path in paths:
            selected_rows = [r for r in development if r["train_seed"] == train_seed and r["checkpoint_step"] == step]
            scores.append((float(np.mean([r["objective"] for r in selected_rows])), train_seed, step, path))
        score, train_seed, step, selected_path = min(scores)
        selection = {"train_seed": train_seed, "steps": step,
                     "checkpoint": str(selected_path.relative_to(out)),
                     "development_mean_cost": score,
                     "candidate_scores": [
                         {"mean_cost": s, "train_seed": t, "steps": n, "checkpoint": str(p.relative_to(out))}
                         for s, t, n, p in scores],
                     "selection_rule": "lowest anchor development mean; tie smaller train seed then earlier step"}
        write_json(out / "selection.json", selection)
        manifest["phase"] = "sealed_evaluation" if profile == "full" else "smoke_evaluation"
        manifest["selected_checkpoint"] = selection["checkpoint"]
        write_json(out / "manifest.json", manifest)
        model = check_checkpoint(selected_path, cfgs["reciprocal_scale2"], panel[0])
        eval_seeds = test_seeds if profile == "full" else smoke_seeds
        rows = []
        grid = [(case, controller, seed) for case in CASES
                for controller in ("fcfs", "risk_matching", "rollout", "learner")
                for seed in eval_seeds]
        with tqdm(grid, desc="Sealed evaluation" if profile == "full" else "Smoke evaluation",
                  unit="episode") as bar:
            for case, controller_name, seed in bar:
                deadline_check()
                controller = (LearnedController(model) if controller_name == "learner"
                              else RuleController(controller_name, deadline_check))
                row = evaluate_episode(cfgs[case], case, seed, controller)
                if abs(row["episode_return"] + row["objective"]) > 1e-9:
                    raise RuntimeError("episode reward mismatch")
                if abs(row["objective"] - sum(row[k] for k in
                       ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-9:
                    raise RuntimeError("cost component mismatch")
                rows.append(row)
                add_row(out / "episodes.partial.csv", row)
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        expected = len(CASES) * 4 * len(eval_seeds)
        if len(rows) != expected or any(int(r["invalid_assignments"]) != 0 for r in rows):
            raise RuntimeError("incomplete grid or invalid assignments")
        if hashes != {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}:
            raise RuntimeError("source/config changed during campaign")
        summary = summarize(rows, registry["bootstrap_seed"])
        write_json(out / "summary.json", summary)
        write_json(out / "audit.json", {
            "status": "PASS", "episodes": len(rows), "development_episodes": len(development),
            "zero_invalid_assignments": True, "episode_return_equals_negative_cost": True,
            "cost_components_reconcile": True, "source_hashes_stable": True,
            "checkpoint_reload": True, "sealed_panel_opened_after_selection": profile == "full"})
        manifest.update(status="COMPLETED", phase="done", elapsed_seconds=time.monotonic() - start,
                        counts={"episodes": len(rows), "development": len(development),
                                "training_steps": len(seeds) * steps[-1]},
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        artifact_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
        return out
    except (TimeoutError, KeyboardInterrupt) as exc:
        manifest.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - start)
        write_json(out / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - start)
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = run(args.profile, args.output_dir)
    print(json.dumps({"status": "COMPLETED", "output_dir": str(out)}))


if __name__ == "__main__":
    main()
