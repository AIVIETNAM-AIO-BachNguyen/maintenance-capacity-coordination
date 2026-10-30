"""Prelocked v4 F2-teacher amortization pilot. No TEST entry point."""
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
from torch.nn import functional as F
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose as rollout_choose
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.pilot import CASES, evaluate_controls
from maintenance_capacity_coordination.structured_policy_v3 import MatchingPolicy, actor_inputs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/forecast_demand_commitment_pilot_v4.md"
ARMS = ("S4", "R4", "G4")
BACKING_ARM = {"S4": "S", "R4": "R", "G4": "G3"}
TRAIN_ROOTS = (73190000, 73190001, 73190002)
DEV_SEEDS = tuple(range(73191000, 73191016))
HELDOUT_SEEDS = tuple(range(73192000, 73192032))
REPLICATION_ROOTS = (73193000, 73193001, 73193002)
REPLICATION_DEV = tuple(range(73194000, 73194016))
REPLICATION_EVAL = tuple(range(73195000, 73195032))
SEALED_TEST = tuple(range(73196000, 73196032))
EPOCHS = 32
MILESTONES = (8, 16, 32)
BATCH = 128
TEACHER_EPISODES = 16
TRAIN_TIMEOUT = 1800
TEACHER_TIMEOUT = 1800
ANALYSIS_SEED = 73198000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    audit = ROOT / "experiments/audit_forecast_distillation_v4.py"
    if audit.exists():
        files.append(audit)
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


def config_hashes():
    cfg = configs()
    return {case: digest(asdict(cfg[case])) for case in CASES}


def training_raw(root, case, episode):
    return role_seed(root, f"forecast_distill_v4:{case}:{episode}")


def forecast_seed(case, raw, tick):
    return role_seed(62300000, f"{VERSION}:forecast:{case}:{raw}:{tick}")


def seed_registry_check():
    old_raw = (set(range(73110000, 73110003)) | set(range(73120000, 73120016))
               | set(range(73130000, 73130016)) | set(range(73140000, 73140032))
               | set(range(73160000, 73160032)) | set(range(73170000, 73170016))
               | set(range(73180000, 73180003)) | set(range(73181000, 73181016))
               | set(range(73182000, 73182032)) | set(range(73183000, 73183003))
               | set(range(73184000, 73184016)) | set(range(73185000, 73185032))
               | set(range(73186000, 73186032)))
    roles = (TRAIN_ROOTS, DEV_SEEDS, HELDOUT_SEEDS, REPLICATION_ROOTS,
             REPLICATION_DEV, REPLICATION_EVAL, SEALED_TEST)
    raw = [seed for role in roles for seed in role]
    if len(raw) != len(set(raw)) or set(raw) & old_raw:
        raise RuntimeError("v4 raw seed role overlap")
    current_episode_raw = [training_raw(root, case, ep)
                           for root in (*TRAIN_ROOTS, *REPLICATION_ROOTS)
                           for case in CASES for ep in range(TEACHER_EPISODES)]
    if len(current_episode_raw) != len(set(current_episode_raw)):
        raise RuntimeError("v4 training episode raw seed overlap")
    physical_raw = [*raw, *current_episode_raw]
    physical = [role_seed(seed, VERSION + ":physical") for seed in physical_raw]
    old_episode_raw = set()
    for root in range(73110000, 73110003):
        for case in CASES:
            for ep in range(200):
                old_episode_raw.add(role_seed(root, f"pilot_train:{case}:{ep}"))
    for root in range(73180000, 73180003):
        for case in CASES:
            for ep in range(200):
                old_episode_raw.add(role_seed(root, f"structured_train_v3:{case}:{ep}"))
    old_physical = {role_seed(seed, VERSION + ":physical")
                    for seed in old_raw | old_episode_raw}
    if len(physical) != len(set(physical)) or set(physical) & old_physical:
        raise RuntimeError("v4 derived physical seed role overlap")
    # Check both teacher and reserved evaluation forecast roles against all known
    # prior F2 and v2 holdout forecast seeds, including unopened reserved roles.
    current_forecasts = [forecast_seed(case, seed, tick)
                         for case in CASES
                         for seed in (*current_episode_raw, *DEV_SEEDS, *HELDOUT_SEEDS,
                                      *REPLICATION_DEV, *REPLICATION_EVAL, *SEALED_TEST)
                         for tick in range(HORIZON)]
    old_forecasts = {forecast_seed(case, seed, tick)
                     for case in CASES for seed in old_raw | old_episode_raw
                     for tick in range(HORIZON)}
    old_forecasts.update(role_seed(
        73171000, f"commitment_headroom_v2:holdout:{case}:{seed}:{tick}")
        for case in CASES for seed in range(73170000, 73170016)
        for tick in range(0, HORIZON, 6))
    if (len(current_forecasts) != len(set(current_forecasts))
            or set(current_forecasts) & (set(physical) | old_physical | old_forecasts)):
        raise RuntimeError("v4 derived forecast seed role overlap")
    return True


def create_run(output_root, kind):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ("forecast_distillation_v4_" + kind + "_"
                          + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    path.mkdir(exist_ok=False)
    manifest = {
        "status": "RUNNING", "kind": kind, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": source_hashes(),
        "config_sha256": config_hashes(), "cases": CASES, "arms": ARMS,
        "training_roots": TRAIN_ROOTS, "dev_seeds": DEV_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "replication_roots": REPLICATION_ROOTS,
        "replication_dev_seeds": REPLICATION_DEV,
        "replication_eval_seeds": REPLICATION_EVAL,
        "sealed_test_seeds_reserved": SEALED_TEST,
        "teacher_episodes_per_case_root": TEACHER_EPISODES,
        "epochs": EPOCHS, "milestones": MILESTONES, "batch": BATCH,
        "teacher_forecast_transitions": 0, "control_forecast_transitions": 0,
        "teacher_H_transitions": 0, "teacher_tail_transitions": 0,
        "student_H_transitions": 0, "student_tail_transitions": 0,
    }
    write_json(path / "manifest.json", manifest)
    return path, manifest


def finalize_run(path, manifest, started):
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_sha256"] = {str(p.relative_to(path)): sha256(p) for p in path.rglob("*")
                                 if p.is_file() and p.name != "manifest.json"}
    write_json(path / "manifest.json", manifest)


def collect_teacher(run, case, root, *, episodes=TEACHER_EPISODES):
    dataset = run / f"teacher_{case}_{root}"
    dataset.mkdir(exist_ok=False)
    rows_file = dataset / "states.jsonl"
    episodes_file = dataset / "episodes.jsonl"
    cfg = configs()[case]
    env = CoordinationEnv(cfg)
    started = time.monotonic()
    deadline = started + TEACHER_TIMEOUT
    record = {"status": "RUNNING", "case": case, "root": root,
              "episodes_requested": episodes, "episodes_completed": 0,
              "rows": 0, "forecast_transitions": 0,
              "H_transitions": 0, "tail_transitions": 0}
    write_json(dataset / "manifest.json", record)

    def check_timeout():
        if time.monotonic() > deadline:
            raise TimeoutError("v4 teacher dataset timeout")

    try:
        for episode in tqdm(range(episodes), desc=f"Teacher {case} {root}"):
            raw = training_raw(root, case, episode)
            env.reset(raw)
            for tick in range(HORIZON):
                check_timeout()
                assert env.tick == tick
                state = env.inner.state
                fseed = forecast_seed(case, raw, tick)
                submissions, matching, diag = rollout_choose(
                    cfg, state, tick, "threshold", "rollout",
                    policy_horizon=HORIZON, forecast_seed=fseed,
                    scenarios=16, forecast_horizon=6, check_timeout=check_timeout)
                if submissions != env.fixed_submissions("threshold"):
                    raise RuntimeError("teacher request contract changed")
                admitted = env.admitted(submissions)
                machine, worker = actor_inputs(cfg, admitted, tick)
                mask = env.matching_mask(submissions)
                index = env.actions.index(matching)
                if not mask[index] or not mask[0]:
                    raise RuntimeError("illegal teacher matching")
                scores = diag["root_scores"]
                if len(scores) != (int(mask.sum()) if mask.sum() > 1 else 0):
                    raise RuntimeError("teacher root-score coverage")
                row = {"case": case, "root": root, "episode": episode, "tick": tick,
                       "raw_seed": raw, "physical_seed": role_seed(raw, VERSION + ":physical"),
                       "forecast_seed": fseed, "state_sha256": hashlib.sha256(repr(state).encode()).hexdigest(),
                       "submissions": submissions, "matching_index": index,
                       "matching": matching, "mask": mask.tolist(),
                       "machine": machine.tolist(), "worker": worker.tolist(),
                       "root_scores": scores, "scores_sha256": digest(scores),
                       "forecast_transitions": diag["forecast_transitions"]}
                append_jsonl(rows_file, row)
                record["rows"] += 1
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
        record["states_sha256"] = sha256(rows_file) if rows_file.exists() else None
        record["episodes_sha256"] = sha256(episodes_file) if episodes_file.exists() else None
        write_json(dataset / "manifest.json", record)
    return dataset, record


def new_student(arm, root):
    torch.manual_seed(root)
    return MatchingPolicy(BACKING_ARM[arm])


def save_student(path, policy, case, arm, root, epoch, dataset_hash):
    torch.save({"case": case, "arm": arm, "root": root, "epoch": epoch,
                "dataset_sha256": dataset_hash, "actor_state_dict": policy.actor.state_dict()}, path)


def load_student(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = MatchingPolicy(BACKING_ARM[payload["arm"]])
    policy.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    policy.eval()
    return policy, payload


def evaluate_student(policy, case, seeds, output=None, *, arm, root, checkpoint):
    env = CoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f"Student {case} {arm} {checkpoint}", leave=False):
        env.reset(seed)
        selection_seconds = 0.0
        for _ in range(HORIZON):
            submissions = env.fixed_submissions("threshold")
            admitted = env.admitted(submissions)
            machine, worker = actor_inputs(env.config, admitted, env.tick)
            mask = env.matching_mask(submissions)
            started = time.perf_counter()
            with torch.no_grad():
                action = int(policy.sample(torch.from_numpy(machine), torch.from_numpy(worker),
                                           torch.from_numpy(mask), deterministic=True))
            selection_seconds += time.perf_counter() - started
            _, done = env.step(submissions, action)
        assert done
        row = {"case": case, "arm": arm, "root": root, "seed": seed,
               "physical_seed": role_seed(seed, VERSION + ":physical"),
               "checkpoint": checkpoint, "selection_seconds": selection_seconds,
               "forecast_transitions": 0, **env.metrics()}
        if (row["invalid_assignments"] or abs(row["objective"] - row["H_cost"] - row["tail_cost"]) > 1e-8
                or abs(row["episode_return"] + row["objective"]) > 1e-8):
            raise RuntimeError("v4 student accounting/legality failure")
        rows.append(row)
        if output is not None:
            append_jsonl(output, row)
    return rows


def train_student(run, dataset, case, arm, root, *, epochs=EPOCHS, dev_seeds=DEV_SEEDS):
    trial = run / f"student_{case}_{arm}_{root}"
    trial.mkdir(exist_ok=False)
    rows = read_jsonl(dataset / "states.jsonl")
    if len(rows) != 48 * (2 if epochs == 1 else TEACHER_EPISODES):
        raise RuntimeError("teacher example coverage")
    machine = torch.tensor(np.asarray([row["machine"] for row in rows]), dtype=torch.float32)
    worker = torch.tensor(np.asarray([row["worker"] for row in rows]), dtype=torch.float32)
    mask = torch.tensor(np.asarray([row["mask"] for row in rows]), dtype=torch.bool)
    labels = torch.tensor([row["matching_index"] for row in rows], dtype=torch.long)
    if not torch.all(mask[torch.arange(len(rows)), labels]):
        raise RuntimeError("masked teacher label")
    policy = new_student(arm, root)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4, weight_decay=0)
    rng = np.random.default_rng(root)
    dataset_hash = sha256(dataset / "states.jsonl")
    record = {"status": "RUNNING", "case": case, "arm": arm, "root": root,
              "dataset_sha256": dataset_hash, "examples": len(rows),
              "actor_parameters": policy.actor_parameters,
              "epochs_requested": epochs, "epochs_completed": 0, "gradient_updates": 0,
              "dev_episodes": 0, "H_transitions": 0, "tail_transitions": 0,
              "optimizer": {"type": "Adam", "lr": 3e-4, "weight_decay": 0, "batch": BATCH}}
    write_json(trial / "manifest.json", record)
    milestones = (1,) if epochs == 1 else MILESTONES
    started = time.monotonic()
    deadline = started + TRAIN_TIMEOUT
    dev_means = []
    try:
        for epoch in tqdm(range(1, epochs + 1), desc=f"Train {case} {arm} {root}"):
            if time.monotonic() > deadline:
                raise TimeoutError("v4 student trial timeout")
            permutation = rng.permutation(len(rows))
            policy.train()
            total_loss = 0.0
            for start in range(0, len(rows), BATCH):
                idx = torch.as_tensor(permutation[start:start + BATCH], dtype=torch.long)
                logits = policy.logits(machine[idx], worker[idx], mask[idx])
                loss = F.cross_entropy(logits, labels[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(idx)
                record["gradient_updates"] += 1
            append_jsonl(trial / "epochs.jsonl", {"epoch": epoch,
                                                 "mean_teacher_nll": total_loss / len(rows)})
            record["epochs_completed"] = epoch
            if epoch in milestones:
                checkpoint = trial / f"checkpoint_epoch_{epoch}.pt"
                save_student(checkpoint, policy, case, arm, root, epoch, dataset_hash)
                restored, _ = load_student(checkpoint)
                dev = evaluate_student(restored, case, dev_seeds, trial / "dev_episodes.jsonl",
                                       arm=arm, root=root, checkpoint=checkpoint.name)
                record["dev_episodes"] += len(dev)
                record["H_transitions"] += 48 * len(dev)
                record["tail_transitions"] += 16 * len(dev)
                mean = float(np.mean([row["objective"] for row in dev]))
                dev_means.append({"epoch": epoch, "checkpoint": checkpoint.name,
                                  "mean_H_tail_cost": mean, "checkpoint_sha256": sha256(checkpoint)})
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


def _paired_comparison(learned, controls, other, seeds, roots):
    values = []
    for seed in seeds:
        per_seed = []
        for case in CASES:
            for root in roots:
                s = learned[(case, "S4", root, seed)]
                ref = (controls[(case, "F_rollout", seed)] if other == "F2"
                       else learned[(case, other, root, seed)])
                per_seed.append(s["objective"] - ref["objective"])
        values.append(float(np.mean(per_seed)))
    rng = np.random.default_rng(ANALYSIS_SEED)
    array = np.asarray(values)
    draws = np.asarray([np.mean(array[rng.integers(0, len(array), len(array))])
                        for _ in range(5000)])
    by_case = {case: float(np.mean([
        learned[(case, "S4", root, seed)]["objective"]
        - (controls[(case, "F_rollout", seed)]["objective"] if other == "F2"
           else learned[(case, other, root, seed)]["objective"])
        for root in roots for seed in seeds])) for case in CASES}
    by_root = {str(root): float(np.mean([
        learned[(case, "S4", root, seed)]["objective"]
        - (controls[(case, "F_rollout", seed)]["objective"] if other == "F2"
           else learned[(case, other, root, seed)]["objective"])
        for case in CASES for seed in seeds])) for root in roots}
    return {"overall": float(array.mean()), "by_case": by_case, "by_training_root": by_root,
            "physical_cluster_ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def analyze_panel(run, seeds, roots):
    learned_rows = read_jsonl(run / "heldout_students.jsonl")
    control_rows = read_jsonl(run / "heldout_controls.jsonl")
    learned = {(r["case"], r["arm"], r["root"], r["seed"]): r for r in learned_rows}
    controls = {(r["case"], r["arm"], r["seed"]): r for r in control_rows
                if r["arm"] in ("F_rollout", "F_risk_matching")}
    expected_learned = len(CASES) * len(ARMS) * len(roots) * len(seeds)
    if len(learned) != expected_learned:
        raise RuntimeError("v4 learned panel incomplete")
    if len(controls) != len(CASES) * 2 * len(seeds):
        raise RuntimeError("v4 control panel incomplete")
    comparisons = {other: _paired_comparison(learned, controls, other, seeds, roots)
                   for other in ("G4", "R4", "F2")}
    s_time = float(np.mean([r["selection_seconds"] for r in learned_rows if r["arm"] == "S4"]))
    f_time = float(np.mean([r["selection_seconds"] for r in control_rows if r["arm"] == "F_rollout"]))
    speed_ratio = s_time / f_time
    sg, sr, sf = (comparisons[k] for k in ("G4", "R4", "F2"))
    go = (sg["overall"] <= -30 and all(v < 0 for v in sg["by_case"].values())
          and sum(v < 0 for v in sg["by_training_root"].values()) >= 2
          and sg["physical_cluster_ci95"][1] < 0
          and sr["overall"] <= -10 and all(v < 0 for v in sr["by_case"].values())
          and sum(v < 0 for v in sr["by_training_root"].values()) >= 2
          and sr["physical_cluster_ci95"][1] < 0
          and sf["overall"] <= 20 and all(v <= 30 for v in sf["by_case"].values())
          and speed_ratio <= .2)
    summary = {"comparisons_S4_minus": comparisons,
               "mean_selection_seconds_S4": s_time, "mean_selection_seconds_F2": f_time,
               "selection_time_ratio_S4_over_F2": speed_ratio,
               "gate": {"go": bool(go), "protocol_sha256": sha256(PROTOCOL)}}
    write_json(run / "analysis.json", summary)
    write_json(run / "replication_go_decision.json", summary["gate"])
    return summary


def run_training(output_root, *, smoke=False, replication=False):
    seed_registry_check()
    kind = "smoke" if smoke else ("replication_train" if replication else "train")
    run, manifest = create_run(output_root, kind)
    started = time.monotonic()
    cases = CASES[:1] if smoke else CASES
    roots = (73100099,) if smoke else (REPLICATION_ROOTS if replication else TRAIN_ROOTS)
    episodes = 2 if smoke else TEACHER_EPISODES
    dev_seeds = (73100098,) if smoke else (REPLICATION_DEV if replication else DEV_SEEDS)
    try:
        for case in cases:
            for root in roots:
                dataset, teacher = collect_teacher(run, case, root, episodes=episodes)
                manifest["teacher_forecast_transitions"] += teacher["forecast_transitions"]
                manifest["teacher_H_transitions"] += teacher["H_transitions"]
                manifest["teacher_tail_transitions"] += teacher["tail_transitions"]
                write_json(run / "manifest.json", manifest)
                for arm in ARMS:
                    _, trial = train_student(run, dataset, case, arm, root,
                                             epochs=1 if smoke else EPOCHS, dev_seeds=dev_seeds)
                    manifest["student_H_transitions"] += trial["H_transitions"]
                    manifest["student_tail_transitions"] += trial["tail_transitions"]
                    write_json(run / "manifest.json", manifest)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finalize_run(run, manifest, started)
    return run


def run_evaluation(output_root, training_run, *, replication=False):
    seed_registry_check()
    kind = "replication_eval" if replication else "heldout"
    run, manifest = create_run(output_root, kind)
    started = time.monotonic()
    training_run = Path(training_run)
    seeds = REPLICATION_EVAL if replication else HELDOUT_SEEDS
    roots = REPLICATION_ROOTS if replication else TRAIN_ROOTS
    try:
        train_manifest = json.loads((training_run / "manifest.json").read_text())
        if (train_manifest["status"] != "COMPLETED"
                or train_manifest.get("audit_status") != "PASS"
                or train_manifest["kind"] != ("replication_train" if replication else "train")
                or train_manifest["protocol_sha256"] != manifest["protocol_sha256"]
                or train_manifest["source_sha256"] != manifest["source_sha256"]
                or train_manifest["config_sha256"] != manifest["config_sha256"]):
            raise RuntimeError("v4 training source/protocol/config mismatch")
        manifest["training_run"] = str(training_run.resolve())
        manifest["training_manifest_sha256"] = sha256(training_run / "manifest.json")
        for case in CASES:
            for root in roots:
                for arm in ARMS:
                    trial = training_run / f"student_{case}_{arm}_{root}"
                    selection = json.loads((trial / "selection.json").read_text())
                    checkpoint = trial / selection["checkpoint"]
                    if sha256(checkpoint) != selection["checkpoint_sha256"]:
                        raise RuntimeError("selected checkpoint hash mismatch")
                    policy, payload = load_student(checkpoint)
                    if (payload["case"], payload["arm"], payload["root"]) != (case, arm, root):
                        raise RuntimeError("selected checkpoint identity mismatch")
                    rows = evaluate_student(policy, case, seeds, run / "heldout_students.jsonl",
                                            arm=arm, root=root, checkpoint=selection["checkpoint"])
                    manifest["student_H_transitions"] += 48 * len(rows)
                    manifest["student_tail_transitions"] += 16 * len(rows)
                    write_json(run / "manifest.json", manifest)
        control_rows = evaluate_controls(run, seeds, panel="heldout")
        manifest["control_forecast_transitions"] = sum(r["forecast_transitions"] for r in control_rows)
        manifest["control_H_transitions"] = 48 * len(control_rows)
        manifest["control_tail_transitions"] = 16 * len(control_rows)
        analyze_panel(run, seeds, roots)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finalize_run(run, manifest, started)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "heldout", "replication_train", "replication_eval"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode in ("smoke", "train", "replication_train"):
        path = run_training(args.output_root, smoke=args.mode == "smoke",
                            replication=args.mode == "replication_train")
    else:
        if args.training_run is None:
            parser.error("--training-run is required")
        path = run_evaluation(args.output_root, args.training_run,
                              replication=args.mode == "replication_eval")
    print(path.resolve())


if __name__ == "__main__":
    main()
