"""Prelocked simulator labels for conditional second-edge economic ranking."""
from __future__ import annotations

import argparse
import copy
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

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv, actor_inputs_scaled
from maintenance_capacity_coordination.same_tick_policy_v12 import EDGES, STOP, choose as c12_choose
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v12 as c12_pilot

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/conditional_value_pretraining_v15_prelock.md"
C12_TRAIN = ROOT / "artifacts/same_tick_v12_train_20261023T015159675542Z"
C12_ROOTS = c12_pilot.TRAIN_ROOTS
SMOKE_SEEDS = (77700001, 77700002)
LABEL_SEEDS = tuple(range(77701000, 77701096))
LABEL_TIMEOUT = 7200


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def first_role_seed(raw, case, root):
    return role_seed(raw, f"v15-first:{case}:{root}")


def bank_seed(raw, case, root, tick, slot):
    return role_seed(raw, f"v15-value:{case}:{root}:{tick}:{slot}")


def first_options(mask):
    legal = [i for i, valid in enumerate(mask) if valid]
    out = []
    for first in legal:
        m0, j0 = EDGES[first]
        seconds = tuple(i for i in legal if i > first and EDGES[i][0] != m0 and EDGES[i][1] != j0)
        if len(seconds) >= 2:
            out.append((first, seconds))
    return tuple(out)


def selected_state(policy, case, root, raw):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    found = None
    for tick in range(48):
        decision = c12_choose(policy, env, deterministic=True)
        if found is None and 8 <= tick <= 39:
            options = first_options(decision["edge_mask"])
            if options:
                first, seconds = options[first_role_seed(raw, case, root) % len(options)]
                found = (copy.deepcopy(env), tick, decision["submissions"], first, seconds)
        env.step(decision["submissions"], decision["matching_index"])
    return found


def continuation_cost(base, policy, index, submissions, events, deadline):
    branch = copy.deepcopy(base)
    branch.inner._events = events
    prior = float(branch.inner.metrics["objective"])
    _, done = branch.step(submissions, index)
    while not done:
        if time.monotonic() > deadline:
            raise TimeoutError("v15 label generation exceeded timeout")
        decision = c12_choose(policy, branch, deterministic=True)
        _, done = branch.step(decision["submissions"], decision["matching_index"])
    metrics = branch.metrics()
    total = float(metrics["objective"] - prior)
    h_cost = float(metrics["H_cost"] - prior)
    tail_cost = float(metrics["tail_cost"])
    if not np.isfinite(total) or abs(total - h_cost - tail_cost) > 1e-8:
        raise RuntimeError("v15 branch accounting")
    return total, h_cost, tail_cost


def checkpoint(case, root):
    trial = C12_TRAIN / f"{case}_C12_{root}"
    selected = json.loads((trial / "selection.json").read_text())
    path = trial / selected["checkpoint"]
    if sha256(path) != selected["checkpoint_sha256"]:
        raise RuntimeError("C12 checkpoint hash changed")
    return path, selected["checkpoint_sha256"]


def seed_registry_check(mode):
    raw = SMOKE_SEEDS if mode == "smoke" else LABEL_SEEDS
    assert len(raw) == len(set(raw))
    if not all(77700000 <= s < 77710000 for s in raw):
        raise RuntimeError("v15 label raw role band")
    # v14 already checked all earlier raw and derived roles. Check the new roles
    # against the entire v14 raw/evaluation/training family and v13 bank tapes.
    from experiments import same_tick_pilot_v14 as v14
    v14.seed_registry_check()
    prior_raw = (*range(72600000, 72600005), *range(73100000, 73800000),
                 *range(74700000, 75800000), *range(76700000, 76800000))
    prior_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior_raw}
    new_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    bank = {bank_seed(seed, case, root, tick, slot)
            for seed in raw for case in CASES for root in C12_ROOTS
            for tick in range(8, 40) for slot in range(4)}
    first = {first_role_seed(seed, case, root) for seed in raw for case in CASES for root in C12_ROOTS}
    if len(bank) != len(raw) * len(CASES) * len(C12_ROOTS) * 32 * 4:
        raise RuntimeError("v15 branch tape collision")
    if len(first) != len(raw) * len(CASES) * len(C12_ROOTS):
        raise RuntimeError("v15 first-choice seed collision")
    if (new_physical | bank | first) & prior_physical or (new_physical & bank) or (bank & first):
        raise RuntimeError("v15/prior physical or branch tape overlap")


def evaluate_label(policy, case, root, raw, state, checkpoint_hash, slots, deadline):
    base, tick, submissions, first, seconds = state
    admitted = base.admitted(submissions)
    machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
    # Reuse the actual C12 edge mask; the auditor recomputes it independently.
    from maintenance_capacity_coordination.same_tick_policy_v12 import initial_edge_mask
    mask = initial_edge_mask(base.config, admitted)
    legal = base.matching_mask(submissions)
    m0, j0 = EDGES[first]
    actions = []
    for second in seconds:
        matching = tuple(sorted(((m0, j0), EDGES[second])))
        index = base.actions.index(matching)
        if not legal[index]:
            raise RuntimeError("v15 labelled matching infeasible")
        actions.append(index)
    tapes = []
    seeds = []
    for slot in range(slots):
        seed = bank_seed(raw, case, root, tick, slot)
        physical = DispatchEnv(base.config)
        physical.reset(seed)
        tapes.append(physical.events)
        seeds.append(seed)
    branches = []
    for second, index in zip(seconds, actions):
        costs = [continuation_cost(base, policy, index, submissions, tape, deadline) for tape in tapes]
        branches.append({"second_edge": second, "matching_index": index,
                         "total_costs": [c[0] for c in costs], "H_costs": [c[1] for c in costs],
                         "tail_costs": [c[2] for c in costs]})
    means = np.asarray([np.mean(b["total_costs"]) for b in branches], dtype=float)
    weights = np.exp(-(means - means.min()) / 20.)
    targets = weights / weights.sum()
    return {"case": case, "C12_root": root, "raw_seed": raw, "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
            "tick": tick, "C12_checkpoint_sha256": checkpoint_hash, "submissions": submissions,
            "first_edge": first, "second_edges": seconds, "matching_indices": actions,
            "bank_seeds": seeds, "branches": branches, "mean_costs": means.tolist(),
            "target_probabilities": targets.tolist(), "machine_obs": machine.tolist(),
            "worker_obs": worker.tolist(), "edge_mask": mask.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "labels"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_registry_check(args.mode)
    original = json.loads((C12_TRAIN / "manifest.json").read_text())
    if original["status"] != "COMPLETED" or json.loads((C12_TRAIN / "audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("audited C12 training required")
    for path, expected in original["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"C12 source changed: {path}")
    if args.mode == "labels" and git_output("status", "--porcelain"):
        raise RuntimeError("committed clean source required for scientific labels")
    cases = CASES if args.mode == "labels" else (CASES[0],)
    roots = C12_ROOTS if args.mode == "labels" else (C12_ROOTS[0],)
    raw_seeds = LABEL_SEEDS if args.mode == "labels" else SMOKE_SEEDS
    slots = 4 if args.mode == "labels" else 2
    output = args.output_root / (f"conditional_value_v15_{args.mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    print(output.resolve(), flush=True)
    cfg = configs()
    source = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "experiments/conditional_value_labels_v15.py",
             ROOT / "experiments/scaled_n5k3_v12_config.py", ROOT / "experiments/same_tick_pilot_v12.py",
             ROOT / "experiments/conditional_value_pretrain_v15.py",
             ROOT / "experiments/same_tick_pilot_v15.py", ROOT / "experiments/audit_same_tick_v15.py",
             ROOT / "experiments/audit_conditional_value_v15.py", ROOT / "pyproject.toml"]
    hashes = {f"{case}:{root}": checkpoint(case, root)[1] for case in cases for root in roots}
    manifest = {"status": "RUNNING", "kind": args.mode, "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
                "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "protocol_sha256": sha256(PROTOCOL),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
                "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                                  for case in cases}, "config_seeds": CONFIG_SEEDS,
                "C12_training_manifest_sha256": sha256(C12_TRAIN / "manifest.json"),
                "C12_training_audit_sha256": sha256(C12_TRAIN / "audit.json"),
                "C12_checkpoint_sha256": hashes, "cases": cases, "C12_roots": roots,
                "raw_seeds": raw_seeds, "bank_slots": slots, "timeout_seconds": LABEL_TIMEOUT}
    write_json(output / "manifest.json", manifest)
    append_jsonl(output / "run.log", {"event": "started", "utc": manifest["started_utc"]})
    started = time.monotonic()
    deadline = started + LABEL_TIMEOUT
    coverage = []
    count = 0
    try:
        for case in cases:
            for root in roots:
                path, checkpoint_hash = checkpoint(case, root)
                policy = c12_pilot.load_policy(path)
                for raw in tqdm(raw_seeds, desc=f"v15 labels {case} {root}", leave=False):
                    if time.monotonic() > deadline:
                        raise TimeoutError("v15 label run timeout")
                    selected = selected_state(policy, case, root, raw)
                    coverage_row = {"case": case, "C12_root": root, "raw_seed": raw,
                                    "selected": selected is not None,
                                    "tick": selected[1] if selected else None}
                    append_jsonl(output / "coverage.jsonl", coverage_row)
                    coverage.append(coverage_row)
                    if selected is None:
                        continue
                    row = evaluate_label(policy, case, root, raw, selected, checkpoint_hash, slots, deadline)
                    append_jsonl(output / "labels.jsonl", row)
                    count += 1
                    manifest["labelled_states"] = count
                    write_json(output / "manifest.json", manifest)
        case_counts = {case: sum(r["case"] == case for r in coverage if r["selected"]) for case in cases}
        sufficient = (all(n >= 50 for n in case_counts.values()) and count >= 150) if args.mode == "labels" else count > 0
        summary = {"coverage_episodes": len(coverage), "labelled_states": count, "by_case": case_counts,
                   "label_branches": sum(len(r["branches"]) * slots for r in
                                         (json.loads(x) for x in (output / "labels.jsonl").read_text().splitlines()))
                                     if count else 0,
                   "sufficient_coverage": sufficient}
        write_json(output / "summary.json", summary)
        manifest.update(status="COMPLETED", sufficient_coverage=sufficient, summary=summary)
        append_jsonl(output / "run.log", {"event": "completed", "utc": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        append_jsonl(output / "run.log", {"event": "failed", "error": repr(error),
                                                  "utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
