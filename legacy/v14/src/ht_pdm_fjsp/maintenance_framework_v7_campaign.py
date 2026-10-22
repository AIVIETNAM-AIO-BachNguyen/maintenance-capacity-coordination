"""Prelocked scratch-RL test of machine-subset/technician assignment decomposition."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES, HORIZON, FrameworkEnv, LearnedController, catalogue, configs,
    evaluate_episode, vector_observation,
)
from ht_pdm_fjsp.maintenance_framework_v1_campaign import (
    ProgressCallback, add_row, check_checkpoint, git_info, model_new, sha, write_json,
)
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingPolicy
from ht_pdm_fjsp.maintenance_framework_v7_policy import canonical_mask
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import choose, request_action
from ht_pdm_fjsp.maintenance_requests import RequestEnv

ROOT = Path(__file__).resolve().parents[2]
VERSION = "maintenance_framework_v7_canonical_assignment"
PROTOCOL = ROOT / "research_result/framework_v7_protocol_vi.md"
PROPOSAL = ROOT / "research_result/method_v7_proposal_vi.md"
REGISTRY = ROOT / "configs/maintenance_framework_v7_seed_registry.json"
LEARNED = ("flat_free", "flat_canonical", "shared_free", "shared_canonical")
RULES = ("fcfs", "risk_matching", "rollout")
CONTROLLERS = (*RULES, *LEARNED)
PAIRS = (("shared_canonical", "shared_free"),
         ("flat_canonical", "flat_free"),
         ("shared_canonical", "flat_canonical"),
         ("shared_canonical", "risk_matching"),
         ("shared_canonical", "rollout"),
         ("shared_canonical", "fcfs"))
SOURCES = tuple(ROOT / path for path in (
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v1_campaign.py",
    "src/ht_pdm_fjsp/maintenance_framework_v6_policy.py",
    "src/ht_pdm_fjsp/maintenance_framework_v7_policy.py",
    "src/ht_pdm_fjsp/maintenance_framework_v7_campaign.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "research_result/method_v7_proposal_vi.md",
    "research_result/framework_v7_protocol_vi.md",
    "configs/maintenance_framework_v7_seed_registry.json",
    "pyproject.toml", "uv.lock",
))
METRICS = ("objective", "H_availability", "H_failures", "queued_cm_ticks",
           "queued_pm_ticks", "H_pending", "end_pending", "H_in_service",
           "end_in_service", "service_starts", "maintenance_cost", "unavailability_cost",
           "failure_cost", "tail_cost", "selection_seconds",
           "episode_wall_seconds", "forecast_transitions")


class PairedMixtureEnv(FrameworkEnv):
    def __init__(self, training_seed: int, *, canonical: bool):
        self.canonical = canonical
        self.cases = configs()
        self.case_rng = np.random.default_rng(role_seed(training_seed, VERSION + ":case"))
        self.physical_rng = np.random.default_rng(role_seed(training_seed, VERSION + ":physical_sequence"))
        self.episode_tape: list[tuple[str, int]] = []
        super().__init__(self.cases[CASES[0]])

    def reset(self, *, seed=None, options=None):
        case = CASES[int(self.case_rng.integers(len(CASES)))]
        raw_seed = int(self.physical_rng.integers(0, 2**31))
        self.config = self.cases[case]
        self.inner = RequestEnv(self.config)
        observation, info = super().reset(seed=raw_seed, options=options)
        self.episode_tape.append((case, raw_seed))
        return observation, info

    def action_masks(self):
        legal = super().action_masks()
        return canonical_mask(self.config, self.actions, legal) if self.canonical else legal


class V7LearnedController(LearnedController):
    def __init__(self, model, *, canonical: bool):
        super().__init__(model)
        self.canonical = canonical

    def choose(self, config, state, tick, *, case: str, seed: int):
        submissions = request_action(config, state, tick, "threshold")
        admitted = submit_requests(config, state, tick, submissions)
        actions = catalogue(config.machines, config.technicians)
        legal = set(legal_matchings(config, admitted))
        mask = np.asarray([action in legal for action in actions], dtype=bool)
        if self.canonical:
            mask = canonical_mask(config, actions, mask)
        observation = vector_observation(config, admitted, tick)
        action, _ = self.model.predict(observation, action_masks=mask, deterministic=True)
        index = int(action)
        if not mask[index]:
            raise RuntimeError("model selected masked action")
        return submissions, actions[index]


class V7RuleController:
    def __init__(self, name, forecast_seed, check_timeout=lambda: None):
        if name not in RULES:
            raise ValueError(name)
        self.name = name
        self.forecast_seed = forecast_seed
        self.forecast_transitions = 0
        self.check_timeout = check_timeout

    def choose(self, config, state, tick, *, case, seed):
        forecast = role_seed(self.forecast_seed, f"{VERSION}:forecast:{case}:{seed}:{tick}")
        submissions, matching, diagnostic = choose(
            config, state, tick, "threshold", self.name,
            policy_horizon=HORIZON, forecast_seed=forecast,
            scenarios=16, forecast_horizon=6, check_timeout=self.check_timeout,
        )
        self.forecast_transitions += diagnostic["forecast_transitions"]
        return submissions, matching


def seed_registry():
    registry = json.loads(REGISTRY.read_text())
    if registry["campaign_version"] != VERSION:
        raise RuntimeError("seed registry campaign version mismatch")
    historical = set()
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        if path != REGISTRY:
            historical |= registered_integers(json.loads(path.read_text()))
    def panel(name):
        spec = registry[name]
        return list(range(spec["start"], spec["start"] + spec["count"]))
    groups = [panel(key) for key in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [value for group in groups for value in group]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise RuntimeError("v7 seed overlap with historical registry or split")
    return registry, groups


def source_info():
    info = git_info()
    relative = [str(path.relative_to(ROOT)) for path in (PROPOSAL, PROTOCOL, REGISTRY)]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *relative], cwd=ROOT, text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *relative],
        cwd=ROOT, text=True,
    ).splitlines()
    info["blocking_paths"] = sorted(set(info["blocking_paths"] + changed + untracked))
    return info


def model_for_arm(arm, env, seed):
    if arm.startswith("flat_"):
        return model_new(env, seed)
    return MaskablePPO(
        EdgeSharingPolicy, env, policy_kwargs={"variant": "shared_edge"},
        seed=seed, device="cpu", verbose=0, n_steps=1024,
        batch_size=256, n_epochs=5, gamma=.99, gae_lambda=.95,
        clip_range=.2, learning_rate=3e-4, ent_coef=.01,
        vf_coef=.5, max_grad_norm=.5,
    )


def policy_digest(model, keys=None):
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name, tensor in sorted(state.items()):
        if keys is not None and not any(name.startswith(key) for key in keys):
            continue
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def hierarchical(values, bootstrap_seed, key, draws=20000):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("expected training × physical matrix")
    training_count, physical_count = matrix.shape
    rng = np.random.default_rng(role_seed(bootstrap_seed, f"{VERSION}:{key}"))
    training_indices = rng.integers(0, training_count, size=(draws, training_count))
    physical_indices = rng.integers(0, physical_count, size=(draws, physical_count))
    resampled = matrix[training_indices[:, :, None], physical_indices[:, None, :]].mean(axis=(1, 2))
    return np.quantile(resampled, [.025, .975]).tolist()


def summarize(rows, train_seeds, physical_seeds, bootstrap_seed):
    result = {}
    cells = {(row["case"], row["controller"], int(row["train_seed"]) if row["train_seed"] != "" else None,
              int(row["seed"])): row for row in rows}
    for case in CASES:
        for controller in CONTROLLERS:
            keys = ([(case, controller, train, seed) for train in train_seeds for seed in physical_seeds]
                    if controller in LEARNED else
                    [(case, controller, None, seed) for seed in physical_seeds])
            panel = [cells[key] for key in keys]
            result[f"{case}/{controller}"] = {
                "count": len(panel),
                "means": {metric: float(np.mean([float(row[metric]) for row in panel]))
                          for metric in METRICS},
            }
            if controller in LEARNED:
                result[f"{case}/{controller}"]["train_seed_mean_cost"] = {
                    str(train): float(np.mean([float(cells[(case, controller, train, seed)]["objective"])
                                               for seed in physical_seeds]))
                    for train in train_seeds
                }
        for subject, comparator in PAIRS:
            matrix = np.asarray([
                [float(cells[(case, subject, train, seed)]["objective"]) -
                 float(cells[(case, comparator, train if comparator in LEARNED else None, seed)]["objective"])
                 for seed in physical_seeds]
                for train in train_seeds
            ])
            key = f"{case}/{subject}_minus_{comparator}"
            result[key] = {
                "paired_mean": float(matrix.mean()),
                "ci95_descriptive": hierarchical(matrix, bootstrap_seed, key),
                "train_seed_means": {str(train): float(matrix[i].mean())
                                     for i, train in enumerate(train_seeds)},
            }
    for subject, comparator in PAIRS:
        matrix = np.asarray([
            [np.mean([
                float(cells[(case, subject, train, seed)]["objective"]) -
                float(cells[(case, comparator, train if comparator in LEARNED else None, seed)]["objective"])
                for case in CASES]) for seed in physical_seeds]
            for train in train_seeds
        ])
        key = f"pooled_{subject}_minus_{comparator}"
        ci = hierarchical(matrix, bootstrap_seed, key)
        primary = subject == "shared_canonical" and comparator == "shared_free"
        result[key] = {
            "paired_mean": float(matrix.mean()), "hierarchical_ci95": ci,
            "confirmed": bool(matrix.mean() < 0 and ci[1] < 0) if primary else None,
            "training_seed_count": len(train_seeds), "physical_seed_count": len(physical_seeds),
            "cases_per_cell": len(CASES),
            "wins": int((matrix < 0).sum()), "ties": int((matrix == 0).sum()),
            "losses": int((matrix > 0).sum()),
            "train_seed_means": {str(train): float(matrix[i].mean())
                                 for i, train in enumerate(train_seeds)},
        }
    return result


def run(profile, output_dir):
    if profile not in ("smoke", "full"):
        raise ValueError(profile)
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("refusing existing run directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + (14400 if profile == "full" else 1200)
    registry, groups = seed_registry()
    training_seeds, dev_seeds, test_seeds, smoke_seeds = groups[:4]
    git = source_info()
    if profile == "full" and git["blocking_paths"]:
        raise RuntimeError(f"full run requires committed inputs: {git['blocking_paths']}")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}
    all_configs = configs()
    manifest = {
        "status": "RUNNING", "phase": "preflight", "profile": profile,
        "protocol": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git, "seed_registry": registry, "source_hashes": source_hashes,
        "invocation": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "device": "cpu"},
        "deadline_seconds": 14400 if profile == "full" else 1200,
    }
    write_json(out / "manifest.json", manifest)
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(PROPOSAL, out / "method_proposal.md")
    shutil.copy2(REGISTRY, out / "seed_registry.json")
    steps = (50176, 100352) if profile == "full" else (1024,)
    write_json(out / "resolved_config.json", {
        "cases": {name: asdict(config) for name, config in all_configs.items()},
        "train_case_distribution": "uniform over four fixed cases",
        "steps": list(steps), "arms": list(LEARNED),
        "allocator": "min sum known service duration within each legal machine subset; ties retained",
        "ppo": {"n_steps": 1024, "batch_size": 256, "n_epochs": 5,
                "gamma": .99, "gae_lambda": .95, "clip_range": .2,
                "learning_rate": .0003, "ent_coef": .01, "vf_coef": .5,
                "max_grad_norm": .5},
    })
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("v7 four-hour campaign deadline")
    try:
        train = training_seeds if profile == "full" else smoke_seeds[:1]
        dev = dev_seeds if profile == "full" else smoke_seeds
        evaluation = test_seeds if profile == "full" else smoke_seeds
        manifest["phase"] = "training"
        write_json(out / "manifest.json", manifest)
        candidates = {arm: {} for arm in LEARNED}
        training_audit = {}
        training_tapes = {}
        model_parameters = {}
        for training_seed in train:
            check_deadline()
            reference_tape = None
            reference_common_start = None
            reference_initial_by_family = {}
            training_tapes[str(training_seed)] = {}
            training_audit[str(training_seed)] = {}
            for arm in LEARNED:
                arm_dir = out / "arms" / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                progress = arm_dir / "training_progress.csv"
                if not progress.exists():
                    with progress.open("x", newline="") as stream:
                        csv.writer(stream).writerow(["train_seed", "steps", "checkpoint", "epoch_seconds"])
                env = PairedMixtureEnv(training_seed, canonical=arm.endswith("_canonical"))
                model = model_for_arm(arm, env, training_seed)
                model_parameters[arm] = sum(param.numel() for param in model.policy.parameters())
                initial_hash = policy_digest(model)
                family = arm.split("_", 1)[0]
                if family not in reference_initial_by_family:
                    reference_initial_by_family[family] = initial_hash
                elif initial_hash != reference_initial_by_family[family]:
                    raise RuntimeError("paired policy starts differ within architecture")
                common_start = policy_digest(model, ("action_net.machine_encoder", "action_net.worker_encoder",
                                                     "mlp_extractor.value_net", "value_net")) if family == "shared" else None
                if common_start is not None:
                    if reference_common_start is None:
                        reference_common_start = common_start
                    elif common_start != reference_common_start:
                        raise RuntimeError("structured encoder/critic starts differ")
                start_arm = time.perf_counter()
                model.learn(total_timesteps=steps[-1],
                            callback=ProgressCallback(arm_dir, training_seed, steps, deadline),
                            progress_bar=False)
                elapsed = time.perf_counter() - start_arm
                tape = list(env.episode_tape)
                if reference_tape is None:
                    reference_tape = tape
                elif tape != reference_tape:
                    raise RuntimeError("training case/physical tapes differ across arms")
                training_tapes[str(training_seed)][arm] = tape
                training_audit[str(training_seed)][arm] = {
                    "initial_policy_sha256": initial_hash,
                    "common_start_sha256": common_start,
                    "training_episode_count": len(tape),
                    "training_tape_sha256": hashlib.sha256(
                        json.dumps(tape, separators=(",", ":")).encode()).hexdigest(),
                    "wall_seconds": elapsed,
                }
                candidates[arm][training_seed] = []
                for step in steps:
                    path = arm_dir / "checkpoints" / str(training_seed) / f"{step}.zip"
                    if not path.is_file():
                        raise RuntimeError(f"missing checkpoint {path}")
                    candidates[arm][training_seed].append((step, path))
                write_json(out / "training_tapes.json", training_tapes)
                write_json(out / "training_audit.json", training_audit)
                env.close()
                del model
        manifest["phase"] = "development"
        write_json(out / "manifest.json", manifest)
        development = []
        total_dev = len(LEARNED) * len(train) * len(steps) * len(CASES) * len(dev)
        with tqdm(total=total_dev, desc="Development", unit="episode") as bar:
            for arm in LEARNED:
                for training_seed in train:
                    for step, path in candidates[arm][training_seed]:
                        model = check_checkpoint(path, all_configs[CASES[0]], dev[0])
                        for case in CASES:
                            for seed in dev:
                                check_deadline()
                                controller = V7LearnedController(model, canonical=arm.endswith("_canonical"))
                                controller.name = arm
                                row = evaluate_episode(all_configs[case], case, seed, controller)
                                row.update(train_seed=training_seed, checkpoint_step=step,
                                           checkpoint=str(path.relative_to(out)))
                                development.append(row)
                                add_row(out / "development.partial.csv", row)
                                bar.update(1)
        shutil.copy2(out / "development.partial.csv", out / "development.csv")
        selections = {arm: {} for arm in LEARNED}
        for arm in LEARNED:
            for training_seed in train:
                scores = []
                for step, path in candidates[arm][training_seed]:
                    cell = [float(row["objective"]) for row in development
                            if row["controller"] == arm and row["train_seed"] == training_seed
                            and row["checkpoint_step"] == step]
                    if len(cell) != len(CASES) * len(dev):
                        raise RuntimeError("incomplete development candidate")
                    scores.append((float(np.mean(cell)), step, path))
                value, step, path = min(scores)
                selections[arm][str(training_seed)] = {
                    "steps": step, "checkpoint": str(path.relative_to(out)),
                    "checkpoint_sha256": sha(path), "development_mean_cost": value,
                    "candidate_scores": [
                        {"mean_cost": score, "steps": candidate_step,
                         "checkpoint": str(candidate_path.relative_to(out))}
                        for score, candidate_step, candidate_path in scores
                    ],
                }
        write_json(out / "selection.json", selections)
        manifest["phase"] = "sealed_evaluation" if profile == "full" else "smoke_evaluation"
        manifest["selection_sha256"] = sha(out / "selection.json")
        write_json(out / "manifest.json", manifest)
        models = {arm: {training_seed: check_checkpoint(
            out / selections[arm][str(training_seed)]["checkpoint"],
            all_configs[CASES[0]], dev[0]) for training_seed in train}
            for arm in LEARNED}
        episodes = []
        grid = [(case, controller, training_seed if controller in LEARNED else None, seed)
                for case in CASES for controller in CONTROLLERS
                for training_seed in (train if controller in LEARNED else [None])
                for seed in evaluation]
        with tqdm(grid, desc="Sealed evaluation" if profile == "full" else "Smoke evaluation",
                  unit="episode") as bar:
            for case, controller_name, training_seed, seed in bar:
                check_deadline()
                if controller_name in LEARNED:
                    controller = V7LearnedController(models[controller_name][training_seed],
                                                     canonical=controller_name.endswith("_canonical"))
                    controller.name = controller_name
                    selected = selections[controller_name][str(training_seed)]
                    checkpoint_step = selected["steps"]
                    checkpoint = selected["checkpoint"]
                else:
                    controller = V7RuleController(
                        controller_name, registry["forecast_seed"], check_deadline,
                    )
                    checkpoint_step = ""
                    checkpoint = ""
                row = evaluate_episode(all_configs[case], case, seed, controller)
                row.update(train_seed=training_seed if training_seed is not None else "",
                           checkpoint_step=checkpoint_step, checkpoint=checkpoint)
                if abs(row["episode_return"] + row["objective"]) > 1e-9:
                    raise RuntimeError("evaluation reward/cost mismatch")
                if abs(row["objective"] - sum(row[key] for key in
                       ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-9:
                    raise RuntimeError("evaluation cost components mismatch")
                if row["requests_created"] - row["requests_completed"] != (
                        row["end_pending"] + row["end_in_service"]):
                    raise RuntimeError("evaluation request conservation mismatch")
                episodes.append(row)
                add_row(out / "episodes.partial.csv", row)
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        expected = len(CASES) * len(evaluation) * (len(RULES) + len(LEARNED) * len(train))
        if len(episodes) != expected or any(row["invalid_assignments"] != 0 for row in episodes):
            raise RuntimeError("incomplete evaluation or invalid assignment")
        if source_hashes != {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}:
            raise RuntimeError("source changed during campaign")
        if any(sha(out / item["checkpoint"]) != item["checkpoint_sha256"]
               for arm in LEARNED for item in selections[arm].values()):
            raise RuntimeError("selected checkpoint changed")
        summary = summarize(episodes, train, evaluation, registry["bootstrap_seed"])
        write_json(out / "summary.json", summary)
        write_json(out / "audit.json", {
            "status": "PASS", "training_tapes_equal": True,
            "structured_common_start_equal": True, "paired_full_policy_start_equal": True,
            "checkpoint_reload": True,
            "source_checkpoint_hashes_stable": True,
            "development_episodes": len(development),
            "evaluation_episodes": len(episodes),
            "reward_components_request_reconciliation": True,
            "zero_invalid_assignments": True,
            "selection_before_sealed_panel": profile == "full",
        })
        manifest.update(
            status="COMPLETED", phase="done",
            elapsed_seconds=time.monotonic() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts={"ppo_training_steps": len(train) * len(LEARNED) * steps[-1],
                    "ppo_checkpoints": len(train) * len(LEARNED) * len(steps),
                    "development": len(development), "evaluation": len(episodes)},
            policy_parameters=model_parameters,
            artifact_hashes={path.name: sha(path) for path in out.iterdir()
                             if path.is_file() and path.name != "manifest.json"},
        )
        write_json(out / "manifest.json", manifest)
        return out
    except (TimeoutError, KeyboardInterrupt) as exc:
        manifest.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run(args.profile, args.output_dir)
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output)}))


if __name__ == "__main__":
    main()
