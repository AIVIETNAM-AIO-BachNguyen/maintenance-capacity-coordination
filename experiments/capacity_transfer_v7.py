"""Prelocked v7 N4/K2 service-duration transfer of frozen v3 PPO policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL, RuleController
from ht_pdm_fjsp.maintenance_request_policies import family_configs
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.pilot import choose as choose_v1, load_policy as load_v1
from maintenance_capacity_coordination.structured_pilot_v3 import (
    choose as choose_v3, load_policy as load_v3, train_trial as train_v3_trial,
)
from maintenance_capacity_coordination.structured_policy_v3 import ACTIONS
from experiments.threshold_control_v1 import choose_g

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/capacity_transfer_check_v7.md"
V3_TRAIN = ROOT / "artifacts/structured_pilot_v3_train_20261022T072700626808Z"
G_TRAIN = ROOT / "artifacts/threshold_control_v1_train_20261022T063058659512Z"
D_TRAIN = ROOT / "artifacts/pilot_v1_train_20261022T061152746250Z"
PAIRS = (("reciprocal_scale2", "reciprocal_scale1"),
         ("specialist_generalist_scale2", "specialist_generalist_scale3"))
ARMS = ("S", "R", "G3")
ROOTS = (73180000, 73180001, 73180002)
HIST_ROOTS = (73110000, 73110001, 73110002)
V14_ROOTS = (72600000, 72600001, 72600002, 72600003, 72600004)
SMOKE_SEEDS = (73300000, 73300001)
PILOT_SEEDS = tuple(range(73301000, 73301032))
BOOTSTRAP_SEED = 73302000
REPL_TRAIN_ROOTS = (73310000, 73310001, 73310002)
REPL_DEV_SEEDS = tuple(range(73311000, 73311016))
REPL_EVAL_SEEDS = tuple(range(73312000, 73312032))
SEALED_TEST = tuple(range(73313000, 73313032))
EVAL_TIMEOUT = 7200
SOURCE_FILES = (
    "experiments/capacity_transfer_v7.py", "experiments/audit_capacity_transfer_v7.py",
    "experiments/threshold_control_v1.py",
    "src/maintenance_capacity_coordination/coordination_v1.py",
    "src/maintenance_capacity_coordination/structured_policy_v3.py",
    "src/maintenance_capacity_coordination/structured_pilot_v3.py",
    "src/maintenance_capacity_coordination/pilot.py",
    "src/maintenance_capacity_coordination/pilot_policy.py",
    "src/maintenance_capacity_coordination/baselines.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v14_policy.py",
)


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
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    return {rel: sha256(ROOT / rel) for rel in SOURCE_FILES}


def config_by_name():
    family = family_configs("full")
    return {case: family[case] for pair in PAIRS for case in pair}


def config_hashes():
    return {name: digest(asdict(cfg)) for name, cfg in config_by_name().items()}


def check_seed_roles(scan_artifacts=True):
    groups = (SMOKE_SEEDS, PILOT_SEEDS, (BOOTSTRAP_SEED,), REPL_TRAIN_ROOTS,
              REPL_DEV_SEEDS, REPL_EVAL_SEEDS, SEALED_TEST)
    raw = [seed for group in groups for seed in group]
    if len(set(raw)) != len(raw) or any(seed < 73300000 for seed in raw):
        raise RuntimeError("v7 raw seed role collision")
    derived = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    if len(derived) != len(raw):
        raise RuntimeError("v7 derived physical seed collision")
    for root in REPL_TRAIN_ROOTS:
        for case, _ in PAIRS:
            for episode in range(200):
                reset = role_seed(root, f"structured_train_v3:{case}:{episode}")
                value = role_seed(reset, BASE_VERSION + ":physical")
                if value in derived:
                    raise RuntimeError("v7 training/physical seed collision")
                derived.add(value)
    forecast = {role_seed(62300000, f"maintenance_framework_v1:forecast:{case}:{seed}:{tick}")
                for seed in raw for case in config_by_name() for tick in range(HORIZON)}
    if len(forecast) != len(raw) * len(config_by_name()) * HORIZON or forecast & derived:
        raise RuntimeError("v7 forecast/physical seed collision")
    if scan_artifacts:
        pattern = re.compile(r'"physical_seed"\s*:\s*(\d+)')
        prior = set()
        for path in (ROOT / "artifacts").rglob("*.jsonl"):
            if "capacity_transfer_v7_" in str(path):
                continue
            with path.open() as stream:
                for line in stream:
                    match = pattern.search(line)
                    if match:
                        prior.add(int(match.group(1)))
        if prior & (derived | forecast):
            raise RuntimeError("v7 derived seeds overlap prior raw artifacts")
    return True


def checkpoint_identity(training_run, case, arm, root):
    trial = Path(training_run) / f"{case}_{arm}_{root}"
    record = json.loads((trial / "manifest.json").read_text())
    selected = json.loads((trial / "selection.json").read_text())
    checkpoint = trial / selected["checkpoint"]
    if (record["status"] != "COMPLETED" or record["selected_checkpoint"] != checkpoint.name
            or selected["transition"] not in (2048, 4096, 8192)):
        raise RuntimeError("unqualified v3 checkpoint")
    return {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint),
            "selection_sha256": sha256(trial / "selection.json"),
            "trial_manifest_sha256": sha256(trial / "manifest.json"),
            "transition": selected["transition"]}


def historical_identity(training_run, case, arm, root):
    trial = Path(training_run) / f"{case}_{arm}_{root}"
    selected = json.loads((trial / "selection.json").read_text())
    checkpoint = trial / selected["checkpoint"]
    return {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint),
            "selection_sha256": sha256(trial / "selection.json"),
            "transition": selected["transition"]}


def v14_identity(root):
    directory = ROOT / f"baselines/v14/checkpoints/count_calibrated/{root}"
    candidates = list(directory.glob("*.pt"))
    if len(candidates) != 1:
        raise RuntimeError("V14 checkpoint identity")
    return {"path": str(candidates[0].resolve()), "sha256": sha256(candidates[0])}


def freeze_checkpoints(training_run, roots):
    identities = {}
    for source, _ in PAIRS:
        for root in roots:
            for arm in ARMS:
                identities[f"v3:{source}:{arm}:{root}"] = checkpoint_identity(
                    training_run, source, arm, root)
        for root in ROOTS:
            identities[f"v3:{source}:D3:{root}"] = checkpoint_identity(
                V3_TRAIN, source, "D3", root)
        for root in HIST_ROOTS:
            identities[f"G:{source}:{root}"] = historical_identity(G_TRAIN, source, "G", root)
            identities[f"D:{source}:{root}"] = historical_identity(D_TRAIN, source, "D", root)
    for root in V14_ROOTS:
        identities[f"A:{root}"] = v14_identity(root)
    return identities


def verify_parent(training_run, replication):
    training_run = Path(training_run)
    manifest = json.loads((training_run / "manifest.json").read_text())
    audit = json.loads((training_run / "audit.json").read_text())
    if manifest["status"] != "COMPLETED" or audit["status"] != "PASS":
        raise RuntimeError("parent training audit failed")
    if replication:
        if (manifest.get("kind") != "replication_train"
                or manifest["protocol_sha256"] != sha256(PROTOCOL)
                or manifest["source_sha256"] != source_hashes()
                or manifest["config_sha256"] != config_hashes()):
            raise RuntimeError("v7 replication training source/protocol mismatch")
    else:
        if manifest.get("kind") != "train":
            raise RuntimeError("v3 source training role mismatch")
        for rel in ("src/maintenance_capacity_coordination/structured_policy_v3.py",
                    "src/maintenance_capacity_coordination/coordination_v1.py",
                    "src/ht_pdm_fjsp/maintenance_dispatch.py",
                    "src/ht_pdm_fjsp/maintenance_requests.py"):
            if sha256(ROOT / rel) != manifest["source_sha256"][rel]:
                raise RuntimeError("frozen v3 policy/physics source mismatch")
    return {"path": str(training_run.resolve()),
            "manifest_sha256": sha256(training_run / "manifest.json"),
            "audit_sha256": sha256(training_run / "audit.json")}


def new_run(output_root, kind, training_run, roots):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / ("capacity_transfer_v7_" + kind + "_"
                         + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(exist_ok=False)
    manifest = {"status": "RUNNING", "kind": kind,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_output("rev-parse", "HEAD"),
                "git_status": git_output("status", "--porcelain"),
                "platform": platform.platform(), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__,
                "protocol_sha256": sha256(PROTOCOL),
                "source_sha256": source_hashes(), "config_sha256": config_hashes(),
                "pairs": PAIRS, "arms": ARMS, "roots": roots,
                "smoke_seeds": SMOKE_SEEDS, "pilot_seeds": PILOT_SEEDS,
                "replication_roots": REPL_TRAIN_ROOTS,
                "replication_dev_seeds": REPL_DEV_SEEDS,
                "replication_eval_seeds": REPL_EVAL_SEEDS,
                "sealed_test_reserved": SEALED_TEST,
                "bootstrap_seed": BOOTSTRAP_SEED, "timeout_seconds": EVAL_TIMEOUT,
                "student_episodes": 0, "contextual_episodes": 0,
                "control_episodes": 0, "H_transitions": 0,
                "tail_transitions": 0, "forecast_transitions": 0}
    if training_run is not None:
        manifest["parent_training"] = verify_parent(training_run, kind == "replication_eval")
        manifest["checkpoints"] = freeze_checkpoints(training_run, roots)
    write_json(run / "manifest.json", manifest)
    return run, manifest


def finish_run(run, manifest, started):
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_sha256"] = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                                 if p.is_file() and p.name != "manifest.json"}
    write_json(run / "manifest.json", manifest)


def evaluate_episode(policy, kind, config, case, source, arm, root, seed, checkpoint, output):
    env = CoordinationEnv(config)
    env.reset(seed)
    controller = None
    if kind in ("F_risk_matching", "F_rollout"):
        controller = RuleController(kind[2:])
    elif kind == "A":
        controller = policy
    actions, submissions = [], []
    selection = 0.
    for _ in range(HORIZON):
        started = time.perf_counter()
        if kind == "student" or kind == "D3":
            decision = choose_v3(policy, env, deterministic=True)
            submit, index = decision["submissions"], decision["matching_index"]
        elif kind == "G":
            decision = choose_g(policy, env, deterministic=True)
            submit, index = decision["submissions"], decision["matching_index"]
        elif kind == "D":
            decision = choose_v1(policy, env, deterministic=True)
            submit, index = decision["submissions"], decision["matching_index"]
        else:
            submit, matching = controller.choose(config, env.inner.state, env.tick,
                                                 case=case, seed=seed)
            index = ACTIONS.index(matching)
        selection += time.perf_counter() - started
        actions.append(int(index))
        submissions.append(list(submit))
        _, done = env.step(submit, index)
    if not done:
        raise RuntimeError("incomplete v7 episode")
    metrics = env.metrics()
    row = {"source_case": source, "case": case, "arm": arm, "root": root,
           "seed": seed, "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
           "checkpoint": checkpoint, "actions": actions, "submissions": submissions,
           "selection_seconds": selection,
           "forecast_transitions": controller.forecast_transitions if kind == "F_rollout" else 0,
           "H_transitions": HORIZON, "tail_transitions": TAIL, **metrics}
    if (metrics["invalid_assignments"] or
            abs(metrics["objective"] - metrics["H_cost"] - metrics["tail_cost"]) > 1e-8 or
            abs(metrics["episode_return"] + metrics["objective"]) > 1e-8):
        raise RuntimeError("v7 H+tail/legality failure")
    append_jsonl(output, row)
    return row


def _bootstrap_interval(per_seed):
    per_seed = np.asarray(per_seed, dtype=float)
    draw = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(per_seed), size=(5000, len(per_seed)))
    return np.quantile(per_seed[draw].mean(axis=1), (.025, .975)).tolist()


def comparison(rows, seeds, roots, contrast):
    keyed = {(r["source_case"], r["case"], r["arm"], r["root"], r["seed"]): r
             for r in rows}
    def delta(source, target, root, seed):
        def cost(case, arm):
            return keyed[source, case, arm, root, seed]["objective"]
        if contrast == "target_S_minus_G3":
            return cost(target, "S") - cost(target, "G3")
        if contrast == "target_S_minus_R":
            return cost(target, "S") - cost(target, "R")
        if contrast == "transfer_difference":
            return (cost(target, "S") - cost(target, "G3")
                    - cost(source, "S") + cost(source, "G3"))
        raise ValueError(contrast)
    per_seed = [float(np.mean([delta(source, target, root, seed)
                               for source, target in PAIRS for root in roots])) for seed in seeds]
    return {"overall": float(np.mean(per_seed)),
            "by_pair": {f"{source}->{target}": float(np.mean([
                delta(source, target, root, seed) for root in roots for seed in seeds]))
                for source, target in PAIRS},
            "by_training_root": {str(root): float(np.mean([
                delta(source, target, root, seed) for source, target in PAIRS for seed in seeds]))
                for root in roots},
            "physical_cluster_ci95": _bootstrap_interval(per_seed)}


def analyze(run, seeds, roots):
    students = read_jsonl(run / "students.jsonl")
    contextual = read_jsonl(run / "contextual.jsonl")
    controls = read_jsonl(run / "controls.jsonl")
    expected = len(PAIRS) * 2 * len(ARMS) * len(roots) * len(seeds)
    if len(students) != expected:
        raise RuntimeError("v7 student panel incomplete")
    comparisons = {name: comparison(students, seeds, roots, name) for name in
                   ("target_S_minus_G3", "transfer_difference", "target_S_minus_R")}
    sg = comparisons["target_S_minus_G3"]
    transfer = comparisons["transfer_difference"]
    sr = comparisons["target_S_minus_R"]
    conditions = {
        "target_S_minus_G3_overall_le_-30": sg["overall"] <= -30,
        "target_S_minus_G3_both_pairs_negative": all(v < 0 for v in sg["by_pair"].values()),
        "target_S_minus_G3_two_roots_negative": sum(v < 0 for v in sg["by_training_root"].values()) >= 2,
        "target_S_minus_G3_ci_upper_lt_0": sg["physical_cluster_ci95"][1] < 0,
        "transfer_overall_le_-10": transfer["overall"] <= -10,
        "transfer_both_pairs_le_0": all(v <= 0 for v in transfer["by_pair"].values()),
        "transfer_ci_upper_lt_0": transfer["physical_cluster_ci95"][1] < 0,
        "target_S_minus_R_overall_le_-10": sr["overall"] <= -10,
        "target_S_minus_R_both_pairs_le_0": all(v <= 0 for v in sr["by_pair"].values()),
        "target_S_minus_R_two_roots_negative": sum(v < 0 for v in sr["by_training_root"].values()) >= 2,
        "target_S_minus_R_ci_upper_lt_0": sr["physical_cluster_ci95"][1] < 0,
    }
    all_rows = students + contextual + controls
    means = {case: {arm: float(np.mean([r["objective"] for r in all_rows
                                       if r["case"] == case and r["arm"] == arm]))
                    for arm in ("S", "R", "G3", "D3", "G", "D", "A", "F_risk_matching", "F_rollout")}
             for pair in PAIRS for case in pair}
    result = {"comparisons": comparisons, "mean_H_tail_cost": means,
              "gate_conditions": conditions, "go": all(conditions.values()),
              "bootstrap_seed": BOOTSTRAP_SEED, "protocol_sha256": sha256(PROTOCOL),
              "student_rows": len(students), "contextual_rows": len(contextual),
              "control_rows": len(controls)}
    write_json(run / "analysis.json", result)
    write_json(run / "replication_go_decision.json",
               {"go": result["go"], "gate_conditions": conditions,
                "protocol_sha256": result["protocol_sha256"]})
    return result


def run_evaluation(output_root, training_run, kind):
    check_seed_roles()
    if kind not in ("smoke", "pilot", "replication_eval"):
        raise ValueError(kind)
    if kind != "smoke" and git_output("status", "--porcelain"):
        raise RuntimeError("scientific v7 evaluation requires clean committed source")
    if kind == "replication_eval":
        parent = json.loads((Path(training_run) / "manifest.json").read_text())
        if "pilot_run" not in parent:
            raise RuntimeError("replication training lacks audited pilot provenance")
    roots = REPL_TRAIN_ROOTS if kind == "replication_eval" else ROOTS
    seeds = SMOKE_SEEDS if kind == "smoke" else (REPL_EVAL_SEEDS if kind == "replication_eval" else PILOT_SEEDS)
    pairs = PAIRS[:1] if kind == "smoke" else PAIRS
    run, manifest = new_run(output_root, kind, training_run, roots)
    started = time.monotonic()
    print(run.resolve(), flush=True)
    try:
        configs_by_case = config_by_name()
        for source, target in pairs:
            for case in (source, target):
                cfg = configs_by_case[case]
                for root in roots:
                    for arm in ARMS:
                        key = f"v3:{source}:{arm}:{root}"
                        identity = manifest["checkpoints"][key]
                        policy = load_v3(identity["path"])
                        for seed in tqdm(seeds, desc=f"V7 {arm} {source}->{case}", leave=False):
                            if time.monotonic() - started > EVAL_TIMEOUT:
                                raise TimeoutError("v7 evaluation timeout")
                            row = evaluate_episode(policy, "student", cfg, case, source, arm, root,
                                                   seed, key, run / "students.jsonl")
                            manifest["student_episodes"] += 1
                            manifest["H_transitions"] += row["H_transitions"]
                            manifest["tail_transitions"] += row["tail_transitions"]
                            write_json(run / "manifest.json", manifest)
                for root in ROOTS:
                    key = f"v3:{source}:D3:{root}"
                    policy = load_v3(manifest["checkpoints"][key]["path"])
                    for seed in tqdm(seeds, desc=f"V7 D3 {source}->{case}", leave=False):
                        row = evaluate_episode(policy, "D3", cfg, case, source, "D3", root,
                                               seed, key, run / "contextual.jsonl")
                        manifest["contextual_episodes"] += 1
                for root in HIST_ROOTS:
                    for arm, loader in (("G", load_v1), ("D", load_v1)):
                        key = f"{arm}:{source}:{root}"
                        policy = loader(manifest["checkpoints"][key]["path"])
                        for seed in tqdm(seeds, desc=f"V7 {arm} {source}->{case}", leave=False):
                            row = evaluate_episode(policy, arm, cfg, case, source, arm, root,
                                                   seed, key, run / "contextual.jsonl")
                            manifest["contextual_episodes"] += 1
                for arm in ("F_risk_matching", "F_rollout"):
                    for seed in tqdm(seeds, desc=f"V7 {arm} {case}", leave=False):
                        row = evaluate_episode(None, arm, cfg, case, source, arm, None,
                                               seed, None, run / "controls.jsonl")
                        manifest["control_episodes"] += 1
                        manifest["forecast_transitions"] += row["forecast_transitions"]
                for root in V14_ROOTS:
                    key = f"A:{root}"
                    controller = V14Controller.from_checkpoint(manifest["checkpoints"][key]["path"])
                    for seed in tqdm(seeds, desc=f"V7 A {case} {root}", leave=False):
                        row = evaluate_episode(controller, "A", cfg, case, source, "A", root,
                                               seed, key, run / "controls.jsonl")
                        manifest["control_episodes"] += 1
                write_json(run / "manifest.json", manifest)
        manifest["H_transitions"] += HORIZON * (manifest["contextual_episodes"] + manifest["control_episodes"])
        manifest["tail_transitions"] += TAIL * (manifest["contextual_episodes"] + manifest["control_episodes"])
        if kind != "smoke":
            analyze(run, seeds, roots)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finish_run(run, manifest, started)
    return run


def verified_pilot(pilot_run):
    if pilot_run is None:
        raise RuntimeError("qualified replication requires --pilot-run")
    pilot_run = Path(pilot_run).resolve()
    manifest = json.loads((pilot_run / "manifest.json").read_text())
    audit = json.loads((pilot_run / "audit.json").read_text())
    decision = json.loads((pilot_run / "replication_go_decision.json").read_text())
    if (manifest["kind"] != "pilot" or manifest["status"] != "COMPLETED"
            or audit["status"] != "PASS" or decision["go"] is not True
            or manifest["protocol_sha256"] != sha256(PROTOCOL)
            or manifest["source_sha256"] != source_hashes()
            or manifest["config_sha256"] != config_hashes()
            or decision["protocol_sha256"] != sha256(PROTOCOL)):
        raise RuntimeError("v7 pilot did not qualify independent replication")
    return {"path": str(pilot_run), "manifest_sha256": sha256(pilot_run / "manifest.json"),
            "audit_sha256": sha256(pilot_run / "audit.json")}


def run_replication_training(output_root, pilot_run):
    check_seed_roles()
    if git_output("status", "--porcelain"):
        raise RuntimeError("scientific v7 training requires clean committed source")
    pilot_identity = verified_pilot(pilot_run)
    run, manifest = new_run(output_root, "replication_train", None, REPL_TRAIN_ROOTS)
    manifest["pilot_run"] = pilot_identity
    manifest["trials"] = []
    write_json(run / "manifest.json", manifest)
    started = time.monotonic()
    print(run.resolve(), flush=True)
    try:
        for source, _ in PAIRS:
            for arm in ARMS:
                for root in REPL_TRAIN_ROOTS:
                    trial = train_v3_trial(run, source, arm, root, steps=8192,
                                           eval_seeds=REPL_DEV_SEEDS)
                    manifest["trials"].append(trial)
                    manifest["H_transitions"] += trial["actual_H_transitions"]
                    write_json(run / "manifest.json", manifest)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        finish_run(run, manifest, started)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "pilot", "replication_train", "replication_eval"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "replication_train":
        main_run = run_replication_training(args.output_root, args.pilot_run)
    else:
        if args.mode == "replication_eval" and args.training_run is None:
            parser.error("replication evaluation requires --training-run")
        training = args.training_run or V3_TRAIN
        main_run = run_evaluation(args.output_root, training, args.mode)
    print(main_run.resolve())


if __name__ == "__main__":
    main()
