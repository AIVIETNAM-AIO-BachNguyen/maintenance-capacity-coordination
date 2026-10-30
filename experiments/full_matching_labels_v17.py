"""Fresh simulator labels for a subset of complete legal physical matchings."""
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
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    EDGES, STOP, enumerate_sequences, initial_edge_mask, matching_from_sequence,
)
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v16 as v16

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/full_matching_value_v17_prelock.md"
V16_TRAIN = ROOT / "artifacts/same_tick_v16_train_20261023T055022317165Z"
V16_ROOTS = v16.TRAIN_ROOTS
SMOKE_SEEDS = (79700001, 79700002)
LABEL_SEEDS = tuple(range(79701000, 79701048))
WINDOWS = ((0, 23), (24, 47))
SLOTS = 4
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


def state_seed(raw, case, root, window):
    return role_seed(raw, f"v17-state:{case}:{root}:{window}")


def candidates_seed(raw, case, root, tick):
    return role_seed(raw, f"v17-candidates:{case}:{root}:{tick}")


def bank_seed(raw, case, root, tick, slot):
    return role_seed(raw, f"v17-value:{case}:{root}:{tick}:{slot}")


def first_options(mask):
    legal = [i for i, present in enumerate(mask) if present]
    return tuple(i for i in legal if sum(
        k > i and EDGES[k][0] != EDGES[i][0] and EDGES[k][1] != EDGES[i][1]
        for k in legal) >= 2)


def checkpoint(case, arm, root):
    trial = V16_TRAIN / f"{case}_{arm}_{root}"
    selected = json.loads((trial / "selection.json").read_text())
    path = trial / selected["checkpoint"]
    if sha256(path) != selected["checkpoint_sha256"]:
        raise RuntimeError("v16 checkpoint hash changed")
    return path, selected["checkpoint_sha256"]


def old_policies(case, root):
    c_path, c_hash = checkpoint(case, "C16", root)
    s_path, s_hash = checkpoint(case, "S16", root)
    c = v16.load_policy(c_path)
    s = v16.load_policy(s_path)
    frozen = DynamicSequencePolicy("D16")
    frozen.load_state_dict(c.state_dict(), strict=True)
    frozen.eval()
    return c, frozen, s, {"C16": c_hash, "S16": s_hash}


def selected_states(c_policy, case, root, raw):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    options = {window: [] for window in range(len(WINDOWS))}
    for tick in range(48):
        decision = v16.choose(c_policy, env, True)
        if first_options(decision["edge_mask"]):
            for window, (lo, hi) in enumerate(WINDOWS):
                if lo <= tick <= hi:
                    options[window].append((copy.deepcopy(env), tick, tuple(decision["submissions"])))
        env.step(decision["submissions"], decision["matching_index"])
    chosen = {}
    for window, candidates in options.items():
        chosen[window] = candidates[state_seed(raw, case, root, window) % len(candidates)] if candidates else None
    return chosen, {window: len(items) for window, items in options.items()}


def canonical_actions(env, edge_mask):
    index = {matching: i for i, matching in enumerate(env.actions)}
    result = {}
    for tokens in enumerate_sequences(edge_mask, canonical=True):
        action = index[matching_from_sequence(tokens)]
        if action in result:
            raise RuntimeError("duplicate canonical matching")
        result[action] = tokens
    return result


def choose_candidates(base, c_policy, frozen, s_policy, raw, case, root, tick, submissions, edge_mask):
    legal = np.flatnonzero(base.matching_mask(submissions)).tolist()
    sequences = canonical_actions(base, edge_mask)
    if set(sequences) != set(legal):
        raise RuntimeError("physical/canonical action mismatch")
    recommendations = (
        ("WAIT", base.actions.index(())),
        ("C16", v16.choose(c_policy, base, True)["matching_index"]),
        ("CF16", v16.choose(frozen, base, True)["matching_index"]),
        ("S16", v16.choose(s_policy, base, True)["matching_index"]),
    )
    selected, reasons = [], {}
    def add(index, reason):
        if index not in legal:
            raise RuntimeError("candidate is not physically legal")
        if index not in reasons:
            selected.append(index)
            reasons[index] = []
        reasons[index].append(reason)
    for reason, index in recommendations:
        add(index, reason)
    rng = np.random.default_rng(candidates_seed(raw, case, root, tick))
    for cardinality in (1, 2, 3):
        choices = [i for i in legal if i not in reasons and len(base.actions[i]) == cardinality]
        if choices:
            add(int(rng.choice(choices)), f"cardinality_{cardinality}")
    remaining = [i for i in legal if i not in reasons]
    for index in rng.permutation(remaining):
        if len(selected) >= 8:
            break
        add(int(index), "uniform_fill")
    if len(selected) != min(8, len(legal)):
        raise RuntimeError("candidate count mismatch")
    return selected, [list(sequences[i]) for i in selected], [reasons[i] for i in selected]


def continuation_cost(base, policy, action, submissions, events, deadline):
    branch = copy.deepcopy(base)
    branch.inner._events = events
    prior = float(branch.inner.metrics["objective"])
    _, done = branch.step(submissions, action)
    while not done:
        if time.monotonic() > deadline:
            raise TimeoutError("v17 label generation exceeded timeout")
        decision = v16.choose(policy, branch, True)
        _, done = branch.step(decision["submissions"], decision["matching_index"])
    metrics = branch.metrics()
    total = float(metrics["objective"] - prior)
    h_cost = float(metrics["H_cost"] - prior)
    tail_cost = float(metrics["tail_cost"])
    if not np.isfinite(total) or abs(total - h_cost - tail_cost) > 1e-8:
        raise RuntimeError("v17 branch accounting")
    return total, h_cost, tail_cost


def evaluate_label(c, frozen, s, case, root, raw, window, state, hashes, slots, deadline):
    base, tick, submissions = state
    admitted = base.admitted(submissions)
    machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
    edge_mask = initial_edge_mask(base.config, admitted)
    actions, sequences, reasons = choose_candidates(
        base, c, frozen, s, raw, case, root, tick, submissions, edge_mask)
    bank_seeds, tapes = [], []
    for slot in range(slots):
        seed = bank_seed(raw, case, root, tick, slot)
        physical = DispatchEnv(base.config)
        physical.reset(seed)
        bank_seeds.append(seed)
        tapes.append(physical.events)
    branches = []
    for index in actions:
        costs = [continuation_cost(base, s, index, submissions, tape, deadline) for tape in tapes]
        branches.append({"matching_index": index, "total_costs": [c[0] for c in costs],
                         "H_costs": [c[1] for c in costs], "tail_costs": [c[2] for c in costs]})
    means = np.asarray([np.mean(b["total_costs"]) for b in branches], dtype=float)
    weights = np.exp(-(means - means.min()) / 20.)
    targets = weights / weights.sum()
    return {"case": case, "C16_root": root, "raw_seed": raw,
            "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
            "window": window, "tick": tick, "v16_checkpoint_sha256": hashes,
            "submissions": list(submissions), "matching_indices": actions,
            "sequences": sequences, "candidate_reasons": reasons,
            "bank_seeds": bank_seeds, "branches": branches,
            "mean_costs": means.tolist(), "target_probabilities": targets.tolist(),
            "machine_obs": machine.tolist(), "worker_obs": worker.tolist(),
            "edge_mask": edge_mask.tolist()}


def seed_registry_check(mode):
    v16.seed_registry_check()
    raw = SMOKE_SEEDS if mode == "smoke" else LABEL_SEEDS
    if len(raw) != len(set(raw)) or not all(79700000 <= seed < 79710000 for seed in raw):
        raise RuntimeError("v17 raw seed role band")
    physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    bank = {bank_seed(seed, case, root, tick, slot)
            for seed in raw for case in CASES for root in V16_ROOTS
            for tick in range(48) for slot in range(SLOTS)}
    states = {state_seed(seed, case, root, window)
              for seed in raw for case in CASES for root in V16_ROOTS
              for window in range(len(WINDOWS))}
    candidates = {candidates_seed(seed, case, root, tick)
                  for seed in raw for case in CASES for root in V16_ROOTS for tick in range(48)}
    if len(bank) != len(raw) * len(CASES) * len(V16_ROOTS) * 48 * SLOTS:
        raise RuntimeError("v17 bank tape collision")
    if len(states) != len(raw) * len(CASES) * len(V16_ROOTS) * len(WINDOWS):
        raise RuntimeError("v17 state seed collision")
    if len(candidates) != len(raw) * len(CASES) * len(V16_ROOTS) * 48:
        raise RuntimeError("v17 candidate seed collision")
    if physical & (bank | states | candidates) or bank & (states | candidates) or states & candidates:
        raise RuntimeError("v17 seed roles collide")
    # Prior reserved raw physical roles and actual bank/derived roles are
    # checked together; v16's own registry independently validates them.
    prior_raw = (*range(72600000, 72600005), *range(73100000, 73800000),
                 *range(74700000, 75800000), *range(76700000, 76800000),
                 *range(77700000, 77800000), *range(78700000, 78800000))
    prior = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior_raw}
    from experiments import conditional_value_labels_v15 as v15_labels
    prior |= {v15_labels.bank_seed(seed, case, old_root, tick, slot)
              for seed in (*v15_labels.SMOKE_SEEDS, *v15_labels.LABEL_SEEDS)
              for case in CASES for old_root in v15_labels.C12_ROOTS
              for tick in range(8, 40) for slot in range(4)}
    from experiments.second_choice_value_v13 import bank_seed as v13_bank_seed
    prior |= {v13_bank_seed(seed, case, old_root, tick, slot)
              for seed in (*range(75700001, 75700003), *range(75711000, 75711016))
              for case in CASES for old_root in (74710000, 74710001, 74710002)
              for tick in range(8, 40) for slot in range(8)}
    for version, roots in (
        ("v12", (74700000, 74710000, 74710001, 74710002, 74720000, 74720001, 74720002)),
        ("v14", (76700000, 76710000, 76710001, 76710002, 76720000, 76720001, 76720002)),
        ("v15", (77700000, 77710000, 77710001, 77710002, 77720000, 77720001, 77720002)),
        ("v16", (78700000, *V16_ROOTS, *v16.REPL_TRAIN_ROOTS)),
    ):
        prior |= {role_seed(role_seed(root, f"same_tick_{version}:{case}:{episode}"),
                            BASE_VERSION + ":physical")
                  for root in roots for case in CASES for episode in range(200)}
    if (physical | bank | states | candidates) & prior:
        raise RuntimeError("v17/prior physical or bank tape overlap")
    return prior | physical | bank | states | candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "labels"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_registry_check(args.mode)
    old = json.loads((V16_TRAIN / "manifest.json").read_text())
    if old["status"] != "COMPLETED" or json.loads((V16_TRAIN / "audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("audited v16 training required")
    for path, expected in old["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"v16 source changed: {path}")
    if args.mode == "labels" and git_output("status", "--porcelain"):
        raise RuntimeError("committed clean source required for scientific labels")
    cases = CASES if args.mode == "labels" else (CASES[0],)
    roots = V16_ROOTS if args.mode == "labels" else (V16_ROOTS[0],)
    raw_seeds = LABEL_SEEDS if args.mode == "labels" else SMOKE_SEEDS
    slots = SLOTS if args.mode == "labels" else 2
    output = args.output_root / (f"full_matching_v17_{args.mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    print(output.resolve(), flush=True)
    cfg = configs()
    source = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "experiments/full_matching_labels_v17.py",
        ROOT / "experiments/full_matching_pretrain_v17.py",
        ROOT / "experiments/same_tick_pilot_v17.py",
        ROOT / "experiments/audit_full_matching_v17.py",
        ROOT / "experiments/audit_same_tick_v17.py",
        ROOT / "experiments/scaled_n5k3_v12_config.py",
        ROOT / "pyproject.toml"]
    hashes = {f"{case}:{root}:{arm}": checkpoint(case, arm, root)[1]
              for case in cases for root in roots for arm in ("C16", "S16")}
    manifest = {"status": "RUNNING", "kind": args.mode,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_output("rev-parse", "HEAD"),
                "git_status": git_output("status", "--porcelain"),
                "platform": platform.platform(), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__,
                "protocol_sha256": sha256(PROTOCOL),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
                "config_sha256": {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                                  for case in cases},
                "config_seeds": CONFIG_SEEDS,
                "v16_training_manifest_sha256": sha256(V16_TRAIN / "manifest.json"),
                "v16_training_audit_sha256": sha256(V16_TRAIN / "audit.json"),
                "v16_checkpoint_sha256": hashes,
                "cases": cases, "C16_roots": roots, "raw_seeds": raw_seeds,
                "windows": WINDOWS, "bank_slots": slots, "timeout_seconds": LABEL_TIMEOUT}
    write_json(output / "manifest.json", manifest)
    append_jsonl(output / "run.log", {"event": "started", "utc": manifest["started_utc"]})
    deadline = time.monotonic() + LABEL_TIMEOUT
    started = time.monotonic()
    coverage = []
    count = 0
    try:
        for case in cases:
            for root in roots:
                c, frozen, s, checkpoint_hashes = old_policies(case, root)
                for raw in tqdm(raw_seeds, desc=f"v17 labels {case} {root}", leave=False):
                    if time.monotonic() > deadline:
                        raise TimeoutError("v17 label run timeout")
                    selected, counts = selected_states(c, case, root, raw)
                    for window in range(len(WINDOWS)):
                        state = selected[window]
                        coverage_row = {"case": case, "C16_root": root, "raw_seed": raw,
                                        "window": window, "qualified_ticks": counts[window],
                                        "selected": state is not None,
                                        "tick": state[1] if state else None}
                        append_jsonl(output / "coverage.jsonl", coverage_row)
                        coverage.append(coverage_row)
                        if state is None:
                            continue
                        row = evaluate_label(c, frozen, s, case, root, raw, window, state,
                                             checkpoint_hashes, slots, deadline)
                        append_jsonl(output / "labels.jsonl", row)
                        count += 1
                        manifest["labelled_states"] = count
                        write_json(output / "manifest.json", manifest)
        by_case = {case: sum(r["selected"] and r["case"] == case for r in coverage) for case in cases}
        sufficient = (count >= 300 and all(n >= 120 for n in by_case.values())) if args.mode == "labels" else count > 0
        summary = {"coverage_windows": len(coverage), "labelled_states": count,
                   "by_case": by_case, "sufficient_coverage": sufficient,
                   "label_branches": sum(len(r["branches"]) * slots for r in
                       (json.loads(x) for x in (output / "labels.jsonl").read_text().splitlines())) if count else 0}
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
        manifest["output_sha256"] = {str(p.relative_to(output)): sha256(p)
                                     for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
