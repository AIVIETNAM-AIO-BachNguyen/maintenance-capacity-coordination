"""Prelocked v6 full-root-score joint matching pilot. No TEST entry point."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose as rollout_choose
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.joint_score_v6 import (
    ACTIONS, ARMS, actor_for, masked_score_mse, pair_interaction_errors,
    parameter_count, teacher_targets,
)
from maintenance_capacity_coordination.pilot import (
    CASES, evaluate_controls, evaluate_policy, load_policy,
)
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs
from experiments.threshold_control_v1 import evaluate_g

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/joint_demand_capacity_pilot_v6.md"
TRAIN_ROOTS = (73210000, 73210001, 73210002)
DEV_SEEDS = tuple(range(73211000, 73211016))
HELDOUT_SEEDS = tuple(range(73212000, 73212032))
REPLICATION_ROOTS = (73213000, 73213001, 73213002)
REPLICATION_DEV = tuple(range(73214000, 73214016))
REPLICATION_EVAL = tuple(range(73215000, 73215032))
SEALED_TEST = tuple(range(73216000, 73216032))
SMOKE_ROOT = 73217000
SMOKE_DEV = (73217001,)
ANALYSIS_SEED = 73218000
TEACHER_EPISODES = 16
EPOCHS = 64
MILESTONES = (16, 32, 64)
BATCH = 128
TRIAL_TIMEOUT = 1800
PANEL_TIMEOUT = 7200


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__),
        ROOT / "experiments/threshold_control_v1.py"]
    audit = ROOT / "experiments/audit_joint_score_v6.py"
    if audit.exists():
        files.append(audit)
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def config_hashes():
    return {case: digest(asdict(configs()[case])) for case in CASES}


def training_raw(root, case, episode):
    return role_seed(root, f"joint_score_v6:{case}:{episode}")


def forecast_seed(case, raw, tick):
    return role_seed(62300000, f"{VERSION}:forecast:{case}:{raw}:{tick}")


def seed_registry_check():
    old_raw = (set(range(73110000, 73110003)) | set(range(73120000, 73120016))
               | set(range(73130000, 73130016)) | set(range(73140000, 73140032))
               | set(range(73160000, 73160032)) | set(range(73170000, 73170016))
               | set(range(73180000, 73180003)) | set(range(73181000, 73181016))
               | set(range(73182000, 73182032)) | set(range(73183000, 73183003))
               | set(range(73184000, 73184016)) | set(range(73185000, 73185032))
               | set(range(73186000, 73186032)) | set(range(73190000, 73190003))
               | set(range(73191000, 73191016)) | set(range(73192000, 73192032))
               | set(range(73193000, 73193003)) | set(range(73194000, 73194016))
               | set(range(73195000, 73195032)) | set(range(73196000, 73196032))
               | set(range(73200000, 73200016)) | set(range(73203000, 73203002)))
    roles = (TRAIN_ROOTS, DEV_SEEDS, HELDOUT_SEEDS, REPLICATION_ROOTS,
             REPLICATION_DEV, REPLICATION_EVAL, SEALED_TEST, (SMOKE_ROOT,), SMOKE_DEV)
    raw = [seed for role in roles for seed in role]
    if len(raw) != len(set(raw)) or set(raw) & old_raw:
        raise RuntimeError("v6 raw seed overlap")
    old_episodes = set()
    for root in range(73110000, 73110003):
        for case in CASES:
            old_episodes.update(role_seed(root, f"pilot_train:{case}:{ep}")
                                for ep in range(200))
    for root in (*range(73180000, 73180003), *range(73183000, 73183003)):
        for case in CASES:
            old_episodes.update(role_seed(root, f"structured_train_v3:{case}:{ep}")
                                for ep in range(200))
    for root in (*range(73190000, 73190003), *range(73193000, 73193003)):
        for case in CASES:
            old_episodes.update(role_seed(root, f"forecast_distill_v4:{case}:{ep}")
                                for ep in range(16))
    current_episodes = [training_raw(root, case, ep)
                        for root in (*TRAIN_ROOTS, *REPLICATION_ROOTS, SMOKE_ROOT)
                        for case in CASES for ep in range(TEACHER_EPISODES)]
    if len(current_episodes) != len(set(current_episodes)) or set(current_episodes) & (old_raw | old_episodes | set(raw)):
        raise RuntimeError("v6 teacher raw seed overlap")
    current_physical = {role_seed(seed, VERSION + ":physical") for seed in (*raw, *current_episodes)}
    old_physical = {role_seed(seed, VERSION + ":physical") for seed in old_raw | old_episodes}
    if len(current_physical) != len(raw) + len(current_episodes) or current_physical & old_physical:
        raise RuntimeError("v6 physical role overlap")
    all_current_raw = (*raw, *current_episodes)
    current_forecast = {forecast_seed(case, seed, tick)
                        for case in CASES for seed in all_current_raw for tick in range(HORIZON)}
    old_forecast = {forecast_seed(case, seed, tick)
                    for case in CASES for seed in old_raw | old_episodes
                    for tick in range(HORIZON)}
    old_holdouts = {role_seed(73171000, f"commitment_headroom_v2:holdout:{case}:{seed}:{tick}")
                    for case in CASES for seed in range(73170000, 73170016)
                    for tick in range(0, HORIZON, 6)}
    old_holdouts.update(role_seed(73201000, f"pair_commitment_v5:holdout:{case}:{seed}:{tick}")
                        for case in CASES for seed in (*range(73200000, 73200016), 73203000, 73203001)
                        for tick in range(HORIZON))
    expected = len(CASES) * len(all_current_raw) * HORIZON
    if (len(current_forecast) != expected
            or current_forecast & (old_forecast | old_holdouts | old_physical | current_physical)):
        raise RuntimeError("v6 forecast role overlap")
    return True


def new_run(output_root, kind):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("joint_score_v6_" + kind + "_"
                   + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    path.mkdir(exist_ok=False)
    manifest = {"status": "RUNNING", "kind": kind,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": source_hashes(),
        "config_sha256": config_hashes(), "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "cases": CASES, "arms": ARMS,
        "training_roots": TRAIN_ROOTS, "dev_seeds": DEV_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS, "replication_roots": REPLICATION_ROOTS,
        "replication_dev": REPLICATION_DEV, "replication_eval": REPLICATION_EVAL,
        "sealed_test_reserved": SEALED_TEST,
        "teacher_episodes_per_case_root": TEACHER_EPISODES,
        "epochs": EPOCHS, "milestones": MILESTONES, "batch": BATCH,
        "teacher_forecast_transitions": 0, "teacher_H_transitions": 0,
        "teacher_tail_transitions": 0, "student_H_transitions": 0,
        "student_tail_transitions": 0, "control_forecast_transitions": 0}
    write_json(path / "manifest.json", manifest)
    return path, manifest


def finish_run(path, manifest, started):
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_sha256"] = {str(p.relative_to(path)): sha256(p) for p in path.rglob("*")
                                 if p.is_file() and p.name != "manifest.json"}
    write_json(path / "manifest.json", manifest)


def collect_teacher(run, case, root, episodes=TEACHER_EPISODES):
    dataset = run / f"teacher_{case}_{root}"
    dataset.mkdir(exist_ok=False)
    cfg = configs()[case]
    env = CoordinationEnv(cfg)
    states = dataset / "states.jsonl"
    episodes_file = dataset / "episodes.jsonl"
    record = {"status": "RUNNING", "case": case, "root": root,
              "episodes_requested": episodes, "episodes_completed": 0,
              "rows": 0, "legal_targets": 0, "forecast_transitions": 0,
              "H_transitions": 0, "tail_transitions": 0}
    write_json(dataset / "manifest.json", record)
    started = time.monotonic()
    def check_timeout():
        if time.monotonic() - started > TRIAL_TIMEOUT:
            raise TimeoutError("v6 teacher timeout")
    try:
        for episode in tqdm(range(episodes), desc=f"V6 teacher {case} {root}"):
            raw = training_raw(root, case, episode)
            env.reset(raw)
            for tick in range(HORIZON):
                check_timeout()
                state = env.inner.state
                submissions, matching, diag = rollout_choose(
                    cfg, state, tick, "threshold", "rollout",
                    policy_horizon=HORIZON, forecast_seed=forecast_seed(case, raw, tick),
                    scenarios=16, forecast_horizon=6, check_timeout=check_timeout)
                if submissions != env.fixed_submissions("threshold"):
                    raise RuntimeError("v6 threshold contract mismatch")
                admitted = env.admitted(submissions)
                machine, worker = actor_inputs(cfg, admitted, tick)
                mask = env.matching_mask(submissions)
                index = ACTIONS.index(matching)
                target = teacher_targets(diag["root_scores"], mask)
                if not mask[index] or abs(target[0]) > 1e-7:
                    raise RuntimeError("v6 teacher target mismatch")
                row = {"case": case, "root": root, "episode": episode,
                       "tick": tick, "raw_seed": raw,
                       "physical_seed": role_seed(raw, VERSION + ":physical"),
                       "forecast_seed": forecast_seed(case, raw, tick),
                       "state_sha256": hashlib.sha256(repr(state).encode()).hexdigest(),
                       "submissions": submissions, "matching": matching,
                       "matching_index": index, "mask": mask.tolist(),
                       "machine": machine.tolist(), "worker": worker.tolist(),
                       "root_scores": diag["root_scores"],
                       "root_scores_sha256": digest(diag["root_scores"]),
                       "target": target.tolist(), "forecast_transitions": diag["forecast_transitions"]}
                append_jsonl(states, row)
                record["rows"] += 1
                record["legal_targets"] += int(mask.sum())
                record["forecast_transitions"] += diag["forecast_transitions"]
                _, done = env.step(submissions, index)
                record["H_transitions"] += 1
            assert done
            record["tail_transitions"] += 16
            append_jsonl(episodes_file, {"case": case, "root": root, "episode": episode,
                "raw_seed": raw, "physical_seed": role_seed(raw, VERSION + ":physical"),
                **env.metrics()})
            record["episodes_completed"] += 1
            write_json(dataset / "manifest.json", record)
        record["status"] = "COMPLETED"
    except Exception as error:
        record.update(status="FAILED", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["states_sha256"] = sha256(states) if states.exists() else None
        record["episodes_sha256"] = sha256(episodes_file) if episodes_file.exists() else None
        write_json(dataset / "manifest.json", record)
    return dataset, record


def new_student(arm, root):
    torch.manual_seed(root)
    return actor_for(arm)


def save_student(path, actor, case, arm, root, epoch, dataset_hash):
    torch.save({"case": case, "arm": arm, "root": root, "epoch": epoch,
                "dataset_sha256": dataset_hash, "actor_state_dict": actor.state_dict()}, path)


def load_student(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actor = actor_for(payload["arm"])
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    return actor, payload


def evaluate_student(actor, case, seeds, output=None, *, arm, root, checkpoint):
    env = CoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f"V6 {arm} {case} {checkpoint}", leave=False):
        env.reset(seed)
        selection_seconds = 0.
        for _ in range(HORIZON):
            submissions = env.fixed_submissions("threshold")
            admitted = env.admitted(submissions)
            machine, worker = actor_inputs(env.config, admitted, env.tick)
            mask = env.matching_mask(submissions)
            started = time.perf_counter()
            with torch.no_grad():
                score = actor(torch.from_numpy(machine), torch.from_numpy(worker))
                action = int(score.masked_fill(~torch.from_numpy(mask), -1e9).argmax())
            selection_seconds += time.perf_counter() - started
            _, done = env.step(submissions, action)
        assert done
        row = {"case": case, "arm": arm, "root": root, "seed": seed,
               "physical_seed": role_seed(seed, VERSION + ":physical"),
               "checkpoint": checkpoint, "selection_seconds": selection_seconds,
               "forecast_transitions": 0, **env.metrics()}
        if (row["invalid_assignments"] or
                abs(row["objective"] - row["H_cost"] - row["tail_cost"]) > 1e-8 or
                abs(row["episode_return"] + row["objective"]) > 1e-8):
            raise RuntimeError("v6 student legality/accounting failure")
        rows.append(row)
        if output is not None:
            append_jsonl(output, row)
    return rows


def train_student(run, dataset, case, arm, root, epochs=EPOCHS, dev_seeds=DEV_SEEDS):
    trial = run / f"student_{case}_{arm}_{root}"
    trial.mkdir(exist_ok=False)
    rows = read_jsonl(dataset / "states.jsonl")
    if len(rows) != 48 * (2 if epochs == 1 else TEACHER_EPISODES):
        raise RuntimeError("v6 teacher coverage")
    machine = torch.tensor(np.asarray([r["machine"] for r in rows]), dtype=torch.float32)
    worker = torch.tensor(np.asarray([r["worker"] for r in rows]), dtype=torch.float32)
    mask = torch.tensor(np.asarray([r["mask"] for r in rows]), dtype=torch.bool)
    target = torch.tensor(np.asarray([r["target"] for r in rows]), dtype=torch.float32)
    actor = new_student(arm, root)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4, weight_decay=0)
    rng = np.random.default_rng(root)
    dataset_hash = sha256(dataset / "states.jsonl")
    record = {"status": "RUNNING", "case": case, "arm": arm, "root": root,
              "dataset_sha256": dataset_hash, "examples": len(rows),
              "legal_targets": int(mask.sum()), "actor_parameters": parameter_count(actor),
              "epochs_requested": epochs, "epochs_completed": 0, "gradient_updates": 0,
              "dev_episodes": 0, "H_transitions": 0, "tail_transitions": 0,
              "optimizer": {"type": "Adam", "lr": 3e-4, "weight_decay": 0,
                            "batch": BATCH, "loss": "mean legal-root squared error per state"}}
    write_json(trial / "manifest.json", record)
    started = time.monotonic()
    milestones = (1,) if epochs == 1 else MILESTONES
    dev_means = []
    try:
        for epoch in tqdm(range(1, epochs + 1), desc=f"V6 train {case} {arm} {root}"):
            if time.monotonic() - started > TRIAL_TIMEOUT:
                raise TimeoutError("v6 student timeout")
            actor.train()
            order = rng.permutation(len(rows))
            loss_total = 0.
            for start in range(0, len(rows), BATCH):
                idx = torch.as_tensor(order[start:start + BATCH], dtype=torch.long)
                predicted = actor(machine[idx], worker[idx])
                loss = masked_score_mse(predicted, target[idx], mask[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(idx)
                record["gradient_updates"] += 1
            append_jsonl(trial / "epochs.jsonl", {"epoch": epoch,
                         "mean_scaled_root_mse": loss_total / len(rows)})
            record["epochs_completed"] = epoch
            if epoch in milestones:
                checkpoint = trial / f"checkpoint_epoch_{epoch}.pt"
                save_student(checkpoint, actor, case, arm, root, epoch, dataset_hash)
                restored, _ = load_student(checkpoint)
                dev = evaluate_student(restored, case, dev_seeds, trial / "dev_episodes.jsonl",
                                       arm=arm, root=root, checkpoint=checkpoint.name)
                record["dev_episodes"] += len(dev)
                record["H_transitions"] += HORIZON * len(dev)
                record["tail_transitions"] += 16 * len(dev)
                mean = float(np.mean([r["objective"] for r in dev]))
                dev_means.append({"epoch": epoch, "checkpoint": checkpoint.name,
                                  "mean_H_tail_cost": mean,
                                  "checkpoint_sha256": sha256(checkpoint)})
                write_json(trial / "dev_milestones.json", dev_means)
            write_json(trial / "manifest.json", record)
        selected = min(dev_means, key=lambda row: (row["mean_H_tail_cost"], row["epoch"]))
        write_json(trial / "selection.json", selected)
        record.update(status="COMPLETED", selected_checkpoint=selected["checkpoint"],
                      selected_checkpoint_sha256=selected["checkpoint_sha256"])
    except Exception as error:
        record.update(status="FAILED", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(trial / "manifest.json", record)
    return trial, record


def _selected_checkpoint(training_run, case, arm, root):
    trial = training_run / f"student_{case}_{arm}_{root}"
    selection = json.loads((trial / "selection.json").read_text())
    checkpoint = trial / selection["checkpoint"]
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("v6 selected checkpoint hash mismatch")
    actor, payload = load_student(checkpoint)
    if (payload["case"], payload["arm"], payload["root"], payload["epoch"]) != (
            case, arm, root, selection["epoch"]):
        raise RuntimeError("v6 selected checkpoint identity mismatch")
    if payload["dataset_sha256"] != sha256(training_run / f"teacher_{case}_{root}" / "states.jsonl"):
        raise RuntimeError("v6 selected dataset hash mismatch")
    return actor, selection


def evaluate_contextual_controls(run, seeds):
    g_train = ROOT / "artifacts/threshold_control_v1_train_20261022T063058659512Z"
    d_train = ROOT / "artifacts/pilot_v1_train_20261022T061152746250Z"
    rows = []
    hashes = {}
    for case in CASES:
        for root in (73110000, 73110001, 73110002):
            for arm, source in (("G", g_train), ("D", d_train)):
                trial = source / f"{case}_{arm}_{root}"
                selection = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selection["checkpoint"]
                hashes[f"{case}:{arm}:{root}"] = sha256(checkpoint)
                policy = load_policy(checkpoint)
                if arm == "G":
                    evaluated = evaluate_g(policy, case, seeds, None,
                                           label=f"{root}:{selection['transition']}")
                else:
                    evaluated = evaluate_policy(policy, case, seeds, None,
                                                label=f"{root}:{selection['transition']}")
                for row in evaluated:
                    full = {"contextual_train_root": root,
                            "checkpoint_sha256": hashes[f"{case}:{arm}:{root}"], **row}
                    append_jsonl(run / "heldout_contextual.jsonl", full)
                    rows.append(full)
    return rows, hashes


def evaluate_interaction_fidelity(run, training_run, seeds, roots, cases=CASES):
    output = run / "heldout_interactions.jsonl"
    total_forecasts = 0
    for case in cases:
        cfg = configs()[case]
        actors = {(arm, root): _selected_checkpoint(training_run, case, arm, root)[0]
                  for root in roots for arm in ARMS}
        env = CoordinationEnv(cfg)
        for seed in tqdm(seeds, desc=f"V6 interaction {case}"):
            env.reset(seed)
            for tick in range(HORIZON):
                state = env.inner.state
                submissions, matching, diag = rollout_choose(
                    cfg, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                    forecast_seed=forecast_seed(case, seed, tick),
                    scenarios=16, forecast_horizon=6)
                admitted = env.admitted(submissions)
                machine, worker = actor_inputs(cfg, admitted, tick)
                mask = env.matching_mask(submissions)
                if len(diag["root_scores"]) != (int(mask.sum()) if mask.sum() > 1 else 0):
                    raise RuntimeError("v6 heldout F2 score coverage")
                errors = {}
                for root in roots:
                    errors[str(root)] = {}
                    for arm in ARMS:
                        with torch.no_grad():
                            predicted = actors[arm, root](torch.from_numpy(machine),
                                                           torch.from_numpy(worker))
                        errors[str(root)][arm] = pair_interaction_errors(
                            predicted, diag["root_scores"])
                append_jsonl(output, {"case": case, "seed": seed, "tick": tick,
                    "physical_seed": role_seed(seed, VERSION + ":physical"),
                    "forecast_seed": forecast_seed(case, seed, tick),
                    "state_sha256": hashlib.sha256(repr(state).encode()).hexdigest(),
                    "root_scores_sha256": digest(diag["root_scores"]),
                    "legal_root_count": int(mask.sum()),
                    "pair_root_count": sum(len(a) == 2 for a in ACTIONS if a in
                                           {tuple(tuple(edge) for edge in r["matching"])
                                            for r in diag["root_scores"]}),
                    "forecast_transitions": diag["forecast_transitions"],
                    "errors": errors})
                total_forecasts += diag["forecast_transitions"]
                env.step(submissions, ACTIONS.index(matching))
    return total_forecasts


def _comparison(learned, other, seeds, roots):
    per_seed = np.asarray([np.mean([
        learned[case, "J6", root, seed]["objective"]
        - learned[case, other, root, seed]["objective"]
        for case in CASES for root in roots]) for seed in seeds], dtype=float)
    rng = np.random.default_rng(ANALYSIS_SEED)
    draw_indices = rng.integers(0, len(seeds), size=(5000, len(seeds)))
    draws = per_seed[draw_indices].mean(axis=1)
    return {"overall": float(per_seed.mean()),
            "by_case": {case: float(np.mean([
                learned[case, "J6", root, seed]["objective"]
                - learned[case, other, root, seed]["objective"]
                for root in roots for seed in seeds])) for case in CASES},
            "by_training_root": {str(root): float(np.mean([
                learned[case, "J6", root, seed]["objective"]
                - learned[case, other, root, seed]["objective"]
                for case in CASES for seed in seeds])) for root in roots},
            "physical_cluster_ci95": np.quantile(draws, (.025, .975)).tolist()}


def analyze_panel(run, seeds, roots):
    students = read_jsonl(run / "heldout_students.jsonl")
    controls = read_jsonl(run / "heldout_controls.jsonl")
    contextual = read_jsonl(run / "heldout_contextual.jsonl")
    interactions = read_jsonl(run / "heldout_interactions.jsonl")
    learned = {(r["case"], r["arm"], r["root"], r["seed"]): r for r in students}
    if len(learned) != len(CASES) * len(ARMS) * len(roots) * len(seeds):
        raise RuntimeError("v6 student panel incomplete")
    if len(controls) != len(CASES) * len(seeds) * 7:
        raise RuntimeError("v6 F/V14 control panel incomplete")
    if len(contextual) != len(CASES) * len(seeds) * 2 * 3:
        raise RuntimeError("v6 G/D control panel incomplete")
    if len(interactions) != len(CASES) * len(seeds) * HORIZON:
        raise RuntimeError("v6 interaction panel incomplete")
    comparisons = {other: _comparison(learned, other, seeds, roots) for other in ("A6", "G6")}
    errors = {}
    for case in CASES:
        errors[case] = {}
        for arm in ARMS:
            values = [value for row in interactions if row["case"] == case
                      for root in roots for value in row["errors"][str(root)][arm]]
            per_seed = [(sum(v for row in interactions if row["case"] == case
                             and row["seed"] == seed for root in roots
                             for v in row["errors"][str(root)][arm]),
                         sum(len(row["errors"][str(root)][arm]) for row in interactions
                             if row["case"] == case and row["seed"] == seed
                             for root in roots)) for seed in seeds]
            rng = np.random.default_rng(ANALYSIS_SEED)
            sampled = rng.integers(0, len(seeds), size=(5000, len(seeds)))
            pair_array = np.asarray(per_seed, dtype=float)
            numerator = pair_array[sampled, 0].sum(axis=1)
            denominator = pair_array[sampled, 1].sum(axis=1)
            interval = np.quantile(numerator[denominator > 0] / denominator[denominator > 0],
                                   (.025, .975)).tolist()
            errors[case][arm] = {"pair_roots": len(values),
                                 "mean_absolute_cost_error": float(np.mean(values)),
                                 "physical_episode_cluster_ci95": interval}
    reciprocal = errors[CASES[0]]
    a_error = reciprocal["A6"]["mean_absolute_cost_error"]
    ratio = reciprocal["J6"]["mean_absolute_cost_error"] / a_error if a_error > 0 else None
    conditions = {}
    for other in ("A6", "G6"):
        c = comparisons[other]
        conditions[f"J6_minus_{other}_overall_le_-10"] = c["overall"] <= -10
        conditions[f"J6_minus_{other}_reciprocal_le_-20"] = c["by_case"][CASES[0]] <= -20
        conditions[f"J6_minus_{other}_specialist_le_5"] = c["by_case"][CASES[1]] <= 5
        conditions[f"J6_minus_{other}_two_roots_negative"] = sum(
            value < 0 for value in c["by_training_root"].values()) >= 2
        conditions[f"J6_minus_{other}_ci_upper_lt_0"] = c["physical_cluster_ci95"][1] < 0
    conditions["reciprocal_pair_mae_ratio_le_0.8"] = ratio is not None and ratio <= .8
    control_means = {case: {arm: float(np.mean([r["objective"] for r in controls
                                              if r["case"] == case and r["arm"] == arm]))
                            for arm in ("A", "F_risk_matching", "F_rollout")}
                     for case in CASES}
    for case in CASES:
        control_means[case].update({arm: float(np.mean([r["objective"] for r in contextual
                                                       if r["case"] == case and r["arm"] == arm]))
                                    for arm in ("G", "D")})
    secondary_fields = ("H_availability", "H_failures", "queued_pm_ticks",
                        "queued_cm_ticks", "H_pending", "H_pending_pm", "H_pending_cm",
                        "end_pending", "service_starts", "legal_service_waits")
    secondary = {}
    for case in CASES:
        secondary[case] = {}
        for arm in (*ARMS, "A", "F_risk_matching", "F_rollout", "G", "D"):
            source = students if arm in ARMS else (contextual if arm in ("G", "D") else controls)
            selected = [r for r in source if r["case"] == case and r["arm"] == arm]
            if not selected:
                raise RuntimeError(f"missing v6 secondary rows {case} {arm}")
            secondary[case][arm] = {field: float(np.mean([r[field] for r in selected]))
                                    for field in secondary_fields}
    result = {"comparisons_J6_minus": comparisons,
              "pair_interaction_absolute_errors": errors,
              "reciprocal_J6_over_A6_pair_error_ratio": ratio,
              "mean_H_tail_cost_by_case": {case: {arm: float(np.mean([
                  learned[case, arm, root, seed]["objective"] for root in roots for seed in seeds]))
                  for arm in ARMS} for case in CASES},
              "control_mean_H_tail_cost_by_case": control_means,
              "secondary_mean_by_case_arm": secondary,
              "mean_selection_seconds": {arm: float(np.mean([
                  r["selection_seconds"] for r in students if r["arm"] == arm])) for arm in ARMS},
              "gate_conditions": conditions, "go": all(conditions.values()),
              "protocol_sha256": sha256(PROTOCOL), "analysis_seed": ANALYSIS_SEED}
    write_json(run / "analysis.json", result)
    write_json(run / "replication_go_decision.json",
               {"go": result["go"], "gate_conditions": conditions,
                "protocol_sha256": result["protocol_sha256"]})
    return result


def _require_pilot_go(pilot_run):
    if pilot_run is None:
        raise RuntimeError("v6 independent training replication requires a pilot artifact")
    manifest = json.loads((Path(pilot_run) / "manifest.json").read_text())
    decision = json.loads((Path(pilot_run) / "replication_go_decision.json").read_text())
    if (manifest["kind"] != "heldout" or manifest["status"] != "COMPLETED"
            or manifest.get("audit_status") != "PASS" or not decision["go"]
            or decision["protocol_sha256"] != sha256(PROTOCOL)):
        raise RuntimeError("v6 independent training replication gate is closed")


def run_training(output_root, *, smoke=False, replication=False, pilot_run=None):
    seed_registry_check()
    if replication:
        _require_pilot_go(pilot_run)
    kind = "smoke" if smoke else ("replication_train" if replication else "train")
    run, manifest = new_run(output_root, kind)
    started = time.monotonic()
    roots = (SMOKE_ROOT,) if smoke else (REPLICATION_ROOTS if replication else TRAIN_ROOTS)
    cases = CASES[:1] if smoke else CASES
    episodes = 2 if smoke else TEACHER_EPISODES
    dev = SMOKE_DEV if smoke else (REPLICATION_DEV if replication else DEV_SEEDS)
    try:
        for case in cases:
            for root in roots:
                if time.monotonic() - started > PANEL_TIMEOUT:
                    raise TimeoutError("v6 training panel timeout")
                dataset, record = collect_teacher(run, case, root, episodes=episodes)
                manifest["teacher_forecast_transitions"] += record["forecast_transitions"]
                manifest["teacher_H_transitions"] += record["H_transitions"]
                manifest["teacher_tail_transitions"] += record["tail_transitions"]
                write_json(run / "manifest.json", manifest)
                for arm in ARMS:
                    _, trial = train_student(run, dataset, case, arm, root,
                                             epochs=1 if smoke else EPOCHS, dev_seeds=dev)
                    manifest["student_H_transitions"] += trial["H_transitions"]
                    manifest["student_tail_transitions"] += trial["tail_transitions"]
                    write_json(run / "manifest.json", manifest)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finish_run(run, manifest, started)
    return run


def run_evaluation(output_root, training_run, *, replication=False):
    seed_registry_check()
    kind = "replication_eval" if replication else "heldout"
    run, manifest = new_run(output_root, kind)
    started = time.monotonic()
    training_run = Path(training_run)
    seeds = REPLICATION_EVAL if replication else HELDOUT_SEEDS
    roots = REPLICATION_ROOTS if replication else TRAIN_ROOTS
    try:
        training_manifest = json.loads((training_run / "manifest.json").read_text())
        if (training_manifest["status"] != "COMPLETED"
                or training_manifest.get("audit_status") != "PASS"
                or training_manifest["kind"] != ("replication_train" if replication else "train")
                or training_manifest["protocol_sha256"] != manifest["protocol_sha256"]
                or training_manifest["source_sha256"] != manifest["source_sha256"]
                or training_manifest["config_sha256"] != manifest["config_sha256"]):
            raise RuntimeError("v6 training source/protocol/config/audit mismatch")
        manifest["training_run"] = str(training_run.resolve())
        manifest["training_manifest_sha256"] = sha256(training_run / "manifest.json")
        for case in CASES:
            for root in roots:
                for arm in ARMS:
                    if time.monotonic() - started > PANEL_TIMEOUT:
                        raise TimeoutError("v6 evaluation panel timeout")
                    actor, selection = _selected_checkpoint(training_run, case, arm, root)
                    rows = evaluate_student(actor, case, seeds, run / "heldout_students.jsonl",
                                            arm=arm, root=root, checkpoint=selection["checkpoint"])
                    manifest["student_H_transitions"] += HORIZON * len(rows)
                    manifest["student_tail_transitions"] += 16 * len(rows)
                    write_json(run / "manifest.json", manifest)
        control_rows = evaluate_controls(run, seeds, panel="heldout")
        manifest["v14_checkpoint_sha256"] = {}
        for historical_seed in (72600000, 72600001, 72600002, 72600003, 72600004):
            checkpoint_dir = ROOT / f"baselines/v14/checkpoints/count_calibrated/{historical_seed}"
            candidates = list(checkpoint_dir.glob("*.pt"))
            if len(candidates) != 1:
                raise RuntimeError("v6 V14 checkpoint identity mismatch")
            manifest["v14_checkpoint_sha256"][str(historical_seed)] = sha256(candidates[0])
        contextual, hashes = evaluate_contextual_controls(run, seeds)
        manifest["contextual_checkpoint_sha256"] = hashes
        manifest["control_forecast_transitions"] = sum(r["forecast_transitions"] for r in control_rows)
        manifest["control_H_transitions"] = HORIZON * (len(control_rows) + len(contextual))
        manifest["control_tail_transitions"] = 16 * (len(control_rows) + len(contextual))
        write_json(run / "manifest.json", manifest)
        manifest["interaction_forecast_transitions"] = evaluate_interaction_fidelity(
            run, training_run, seeds, roots)
        analyze_panel(run, seeds, roots)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finish_run(run, manifest, started)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "heldout", "replication_train",
                                          "replication_eval"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode in ("smoke", "train", "replication_train"):
        path = run_training(args.output_root, smoke=args.mode == "smoke",
                            replication=args.mode == "replication_train",
                            pilot_run=args.pilot_run)
    else:
        if args.training_run is None:
            parser.error("--training-run is required")
        path = run_evaluation(args.output_root, args.training_run,
                              replication=args.mode == "replication_eval")
    print(path.resolve())


if __name__ == "__main__":
    main()
