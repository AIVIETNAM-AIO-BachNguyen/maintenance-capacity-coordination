"""Prelocked bank-split expected-value check for v12 second assignments."""
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
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v12 import EDGES, STOP, choose
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v12 as pilot

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/second_choice_value_v13_prelock.md"
TRAIN = ROOT / "artifacts/same_tick_v12_train_20261023T015159675542Z"
SCIENCE_SEEDS = tuple(range(75711000, 75711016))
SMOKE_SEEDS = (75700001, 75700002)
BOOTSTRAP_SEED = 75713000
TIMEOUT_SECONDS = 1800


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append(path: Path, payload) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def checkpoint_selection(case: str, root: int):
    trial = TRAIN / f"{case}_C12_{root}"
    selection = json.loads((trial / "selection.json").read_text())
    path = trial / selection["checkpoint"]
    if sha256(path) != selection["checkpoint_sha256"]:
        raise RuntimeError("C12 selected checkpoint changed")
    return path, selection


def bank_seed(raw: int, case: str, root: int, tick: int, slot: int) -> int:
    return role_seed(raw, f"v13-second-value:{case}:{root}:{tick}:{slot}")



def seed_registry_check():
    raw = (*SMOKE_SEEDS, *SCIENCE_SEEDS, BOOTSTRAP_SEED)
    if len(raw) != len(set(raw)) or not all(75700000 <= seed < 75800000 for seed in raw):
        raise RuntimeError("v13 raw seed roles overlap")
    base_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    prior_raw = (*range(72600000, 72600005), *range(73100000, 73800000), *range(74700000, 74800000))
    prior_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior_raw}
    if base_physical & prior_physical:
        raise RuntimeError("v13/prior physical tape overlap")
    nested_v12 = {role_seed(role_seed(root, f"same_tick_v12:{case}:{episode}"),
                            BASE_VERSION + ":physical")
                  for root in pilot.TRAIN_ROOTS for case in CASES for episode in range(200)}
    if base_physical & nested_v12:
        raise RuntimeError("v13/v12 training physical tape overlap")
    banks = [bank_seed(raw, case, root, tick, slot)
             for raw in (*SMOKE_SEEDS, *SCIENCE_SEEDS)
             for case in CASES for root in pilot.TRAIN_ROOTS
             for tick in range(8, 40) for slot in range(8)]
    if len(banks) != len(set(banks)) or (set(banks) & (base_physical | prior_physical | nested_v12)):
        raise RuntimeError("v13 future bank tape overlap")


def second_choices(decision):
    first = decision["tokens"][0] if decision["tokens"] else STOP
    if first == STOP:
        return None, ()
    m0, j0 = EDGES[first]
    seconds = tuple(i for i, (m, j) in enumerate(EDGES)
                    if decision["edge_mask"][i] and m != m0 and j != j0)
    return first, seconds


def continuation_cost(base: ScaledCoordinationEnv, policy, matching_index: int,
                      submissions, events, deadline: float) -> tuple[float, float, float]:
    branch = copy.deepcopy(base)
    branch.inner._events = events
    initial_cost = float(branch.inner.metrics["objective"])
    reward, done = branch.step(submissions, matching_index)
    while not done:
        if time.monotonic() > deadline:
            raise TimeoutError("v13 30-minute diagnostic timeout")
        decision = choose(policy, branch, deterministic=True)
        reward, done = branch.step(decision["submissions"], decision["matching_index"])
    metrics = branch.metrics()
    total = float(metrics["objective"] - initial_cost)
    h_cost = float(metrics["H_cost"] - initial_cost)
    tail_cost = float(metrics["tail_cost"])
    if not np.isfinite(total) or abs(total - h_cost - tail_cost) > 1e-8:
        raise RuntimeError("nonfinite or unreconciled branch cost")
    return total, h_cost, tail_cost


def selected_state(policy, case: str, raw: int):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    selected = None
    for tick in range(48):
        decision = choose(policy, env, deterministic=True)
        first, seconds = second_choices(decision)
        if selected is None and 8 <= tick <= 39 and len(seconds) >= 2:
            selected = (copy.deepcopy(env), tick, decision["submissions"], first, seconds)
        env.step(decision["submissions"], decision["matching_index"])
    return selected


def evaluate_state(policy, case: str, root: int, raw: int, selected,
                   checkpoint_sha: str, bank_slots: int, deadline: float):
    base, tick, submissions, first, seconds = selected
    m0, j0 = EDGES[first]
    candidate_actions = {}
    legal = base.matching_mask(submissions)
    for second in seconds:
        matching = tuple(sorted(((m0, j0), EDGES[second])))
        index = base.actions.index(matching)
        if not legal[index]:
            raise RuntimeError("second-edge matching infeasible")
        candidate_actions[second] = index
    tapes = []
    seeds = []
    for slot in range(bank_slots):
        seed = bank_seed(raw, case, root, tick, slot)
        physical = DispatchEnv(base.config)
        physical.reset(seed)
        tapes.append(physical.events)
        seeds.append(seed)
    candidates = []
    for second in seconds:
        costs = [continuation_cost(base, policy, candidate_actions[second], submissions,
                                   tapes[slot], deadline)
                 for slot in range(bank_slots)]
        candidates.append({"second_edge": second, "matching_index": candidate_actions[second],
                           "total_costs": [x[0] for x in costs],
                           "H_costs": [x[1] for x in costs], "tail_costs": [x[2] for x in costs]})
    if bank_slots == 8:
        by_second = {c["second_edge"]: c for c in candidates}
        bank_a = {i: float(np.mean(c["total_costs"][:4])) for i, c in by_second.items()}
        best = min(seconds, key=lambda i: (bank_a[i], i))
        worst = min(seconds, key=lambda i: (-bank_a[i], i))
        validation_gap = float(np.mean(by_second[worst]["total_costs"][4:])
                               - np.mean(by_second[best]["total_costs"][4:]))
        selection_spread = float(bank_a[worst] - bank_a[best])
        bank_b_spread = float(max(np.mean(c["total_costs"][4:]) for c in candidates)
                              - min(np.mean(c["total_costs"][4:]) for c in candidates))
    else:
        best = worst = None
        validation_gap = selection_spread = bank_b_spread = None
    return {"case": case, "root": root, "raw_seed": raw, "tick": tick,
            "checkpoint_sha256": checkpoint_sha, "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
            "first_edge": first, "submissions": submissions, "bank_seeds": seeds,
            "candidates": candidates, "bank_a_best": best, "bank_a_worst": worst,
            "bank_a_spread": selection_spread, "bank_b_spread": bank_b_spread,
            "validation_gap": validation_gap}


def summary_and_gate(rows, coverage):
    case_rows = {case: [r for r in rows if r["case"] == case] for case in CASES}
    case_counts = {case: len(case_rows[case]) for case in CASES}
    case_means = {case: (float(np.mean([r["validation_gap"] for r in case_rows[case]]))
                         if case_rows[case] else None) for case in CASES}
    root_means = {str(root): (float(np.mean([r["validation_gap"] for r in rows if r["root"] == root]))
                              if any(r["root"] == root for r in rows) else None)
                  for root in pilot.TRAIN_ROOTS}
    overall = float(np.mean(list(case_means.values()))) if all(v is not None for v in case_means.values()) else None
    ci = None
    valid_draws = 0
    if rows and all(case_counts.values()):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap = []
        by_seed = {(seed, case): [r["validation_gap"] for r in rows
                                 if r["raw_seed"] == seed and r["case"] == case]
                   for seed in SCIENCE_SEEDS for case in CASES}
        for draw in rng.choice(SCIENCE_SEEDS, size=(5000, len(SCIENCE_SEEDS)), replace=True):
            means = []
            for case in CASES:
                values = [gap for seed in draw for gap in by_seed[seed, case]]
                if not values:
                    break
                means.append(float(np.mean(values)))
            if len(means) == len(CASES):
                bootstrap.append(float(np.mean(means)))
        valid_draws = len(bootstrap)
        if bootstrap:
            ci = [float(x) for x in np.quantile(bootstrap, [.025, .975])]
    conditions = {"at_least_12_states_each_case": all(n >= 12 for n in case_counts.values()),
                  "both_case_validation_means_at_least_10": all(v is not None and v >= 10 for v in case_means.values()),
                  "cluster_95_lower_positive": ci is not None and ci[0] > 0,
                  "all_5000_bootstrap_draws_defined": valid_draws == 5000}
    return {"coverage_episodes": len(coverage), "selected_states": len(rows),
            "case_counts": case_counts, "case_means": case_means,
            "root_means": root_means, "overall_validation_gap": overall,
            "mean_bank_a_spread": float(np.mean([r["bank_a_spread"] for r in rows])) if rows else None,
            "mean_bank_b_spread": float(np.mean([r["bank_b_spread"] for r in rows])) if rows else None,
            "raw_seed_cluster_95_interval": ci, "bootstrap_draws_defined": valid_draws,
            "conditions": conditions, "go": all(conditions.values()),
            "interpretation": "bank-B held-out ranking signal under frozen C12 continuation, not policy gain"}


def run(mode: str, output_root: Path):
    torch.set_num_threads(1)
    seed_registry_check()
    training = json.loads((TRAIN / "manifest.json").read_text())
    if training["status"] != "COMPLETED" or json.loads((TRAIN / "audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("audited C12 training run required")
    for path, expected in training["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"C12 collection source changed: {path}")
    science = mode == "diagnostic"
    if science and git_output("status", "--porcelain"):
        raise RuntimeError("committed clean source required for scientific diagnostic")
    cases = CASES if science else (CASES[0],)
    roots = pilot.TRAIN_ROOTS if science else (pilot.TRAIN_ROOTS[0],)
    seeds = SCIENCE_SEEDS if science else SMOKE_SEEDS
    slots = 8 if science else 2
    output = output_root / (f"second_choice_v13_{mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    cfg = configs()
    source = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "experiments/second_choice_value_v13.py",
            ROOT / "experiments/same_tick_pilot_v12.py", ROOT / "experiments/scaled_n5k3_v12_config.py",
            ROOT / "pyproject.toml"]
    checkpoints = {f"{case}:{root}": checkpoint_selection(case, root)[1]["checkpoint_sha256"]
                   for case in cases for root in roots}
    manifest = {"status": "RUNNING", "kind": mode, "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
                "platform": platform.platform(), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "protocol_sha256": sha256(PROTOCOL),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
                "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                                  for case in cases},
                "training_manifest_sha256": sha256(TRAIN / "manifest.json"),
                "training_audit_sha256": sha256(TRAIN / "audit.json"),
                "checkpoint_sha256": checkpoints, "cases": cases, "roots": roots, "raw_seeds": seeds,
                "bank_slots": slots, "bootstrap_seed": BOOTSTRAP_SEED if science else None,
                "timeout_seconds": TIMEOUT_SECONDS}
    json_write(output / "manifest.json", manifest)
    start = time.monotonic()
    rows, coverage = [], []
    try:
        for case in cases:
            for root in roots:
                checkpoint, selection = checkpoint_selection(case, root)
                policy = pilot.load_policy(checkpoint)
                for raw in tqdm(seeds, desc=f"v13 {mode} {case} {root}"):
                    if time.monotonic() - start > TIMEOUT_SECONDS:
                        raise TimeoutError("v13 30-minute diagnostic timeout")
                    state = selected_state(policy, case, raw)
                    cover = {"case": case, "root": root, "raw_seed": raw,
                             "selected": state is not None, "tick": state[1] if state else None}
                    append(output / "coverage.jsonl", cover)
                    coverage.append(cover)
                    if state is not None:
                        row = evaluate_state(policy, case, root, raw, state, selection["checkpoint_sha256"],
                                             slots, start + TIMEOUT_SECONDS)
                        append(output / "states.jsonl", row)
                        rows.append(row)
        if science:
            summary = summary_and_gate(rows, coverage)
            json_write(output / "summary.json", summary)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["coverage_episodes"] = len(coverage)
        manifest["selected_states"] = len(rows)
        manifest["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}
        json_write(output / "manifest.json", manifest)
    print(output.resolve())
    if science:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "diagnostic"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run(args.mode, args.output_root)
