"""Independent state, matching, event-tape and H+tail replay for v17 labels."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv, actor_inputs_scaled
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, enumerate_sequences, initial_edge_mask, matching_from_sequence
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v16 as old
from experiments import full_matching_labels_v17 as labels


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearly(a, b):
    return abs(float(a) - float(b)) < 1e-7


def identity(run, manifest):
    assert manifest["protocol_sha256"] == labels.sha256(labels.PROTOCOL)
    assert manifest["config_seeds"] == CONFIG_SEEDS
    for path, expected in manifest["source_sha256"].items():
        assert labels.sha256(labels.ROOT / path) == expected, path
    for path, expected in manifest["output_sha256"].items():
        assert labels.sha256(run / path) == expected, path
    for case, expected in manifest["config_sha256"].items():
        actual = hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
        assert actual == expected
    assert manifest["v16_training_manifest_sha256"] == labels.sha256(labels.V16_TRAIN / "manifest.json")
    assert manifest["v16_training_audit_sha256"] == labels.sha256(labels.V16_TRAIN / "audit.json")
    labels.seed_registry_check(manifest["kind"])


def selected_windows(policy, case, root, raw):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    qualifying = {0: [], 1: []}
    for tick in range(48):
        action = old.choose(policy, env, True)
        mask = initial_edge_mask(env.config, env.admitted(action["submissions"]))
        legal_edges = [i for i, present in enumerate(mask) if present]
        has_competition = any(sum(
            next_edge > first and EDGES[next_edge][0] != EDGES[first][0]
            and EDGES[next_edge][1] != EDGES[first][1]
            for next_edge in legal_edges) >= 2 for first in legal_edges)
        if has_competition:
            window = 0 if tick < 24 else 1
            qualifying[window].append((copy.deepcopy(env), tick, tuple(action["submissions"])))
        env.step(action["submissions"], action["matching_index"])
    selected = {}
    for window in (0, 1):
        choices = qualifying[window]
        key = role_seed(raw, f"v17-state:{case}:{root}:{window}")
        selected[window] = choices[key % len(choices)] if choices else None
    return selected, {window: len(qualifying[window]) for window in (0, 1)}


def candidate_replay(base, c, frozen, s, raw, case, root, tick, submissions, mask):
    valid = [i for i, allowed in enumerate(base.matching_mask(submissions)) if allowed]
    index = {matching: i for i, matching in enumerate(base.actions)}
    canonical = {}
    for path in enumerate_sequences(mask, canonical=True):
        action = index[matching_from_sequence(path)]
        assert action not in canonical
        canonical[action] = list(path)
    assert set(canonical) == set(valid)
    picked, reasons = [], {}
    def add(i, reason):
        assert i in valid
        if i not in reasons:
            picked.append(i)
            reasons[i] = []
        reasons[i].append(reason)
    add(index[()], "WAIT")
    for name, policy in (("C16", c), ("CF16", frozen), ("S16", s)):
        action = old.choose(policy, base, True)
        assert tuple(action["submissions"]) == tuple(submissions)
        add(action["matching_index"], name)
    rng = np.random.default_rng(role_seed(raw, f"v17-candidates:{case}:{root}:{tick}"))
    for size in (1, 2, 3):
        alternatives = [i for i in valid if i not in reasons and len(base.actions[i]) == size]
        if alternatives:
            add(int(rng.choice(alternatives)), f"cardinality_{size}")
    for i in rng.permutation([i for i in valid if i not in reasons]):
        if len(picked) == 8:
            break
        add(int(i), "uniform_fill")
    assert len(picked) == min(8, len(valid))
    return picked, [canonical[i] for i in picked], [reasons[i] for i in picked]


def branch_cost(base, s, submissions, index, tape):
    env = copy.deepcopy(base)
    env.inner._events = tape
    prior = float(env.inner.metrics["objective"])
    _, done = env.step(submissions, index)
    while not done:
        decision = old.choose(s, env, True)
        _, done = env.step(decision["submissions"], decision["matching_index"])
    metrics = env.metrics()
    return metrics["objective"] - prior, metrics["H_cost"] - prior, metrics["tail_cost"]


def replay(run, manifest):
    cases = CASES if manifest["kind"] == "labels" else (CASES[0],)
    roots = labels.V16_ROOTS if manifest["kind"] == "labels" else (labels.V16_ROOTS[0],)
    seeds = labels.LABEL_SEEDS if manifest["kind"] == "labels" else labels.SMOKE_SEEDS
    slots = labels.SLOTS if manifest["kind"] == "labels" else 2
    assert manifest["cases"] == list(cases) and manifest["C16_roots"] == list(roots)
    assert manifest["raw_seeds"] == list(seeds) and manifest["bank_slots"] == slots
    assert manifest["windows"] == [list(x) for x in labels.WINDOWS]
    coverage = rows(run / "coverage.jsonl")
    records = rows(run / "labels.jsonl") if (run / "labels.jsonl").exists() else []
    by_key = {(r["case"], r["C16_root"], r["raw_seed"], r["window"]): r for r in records}
    assert len(by_key) == len(records)
    assert len(coverage) == len(cases) * len(roots) * len(seeds) * 2
    cursor, seen, branches = 0, set(), 0
    for case in cases:
        for root in roots:
            c_path, c_hash = labels.checkpoint(case, "C16", root)
            s_path, s_hash = labels.checkpoint(case, "S16", root)
            assert manifest["v16_checkpoint_sha256"][f"{case}:{root}:C16"] == c_hash
            assert manifest["v16_checkpoint_sha256"][f"{case}:{root}:S16"] == s_hash
            c, s = old.load_policy(c_path), old.load_policy(s_path)
            frozen = DynamicSequencePolicy("D16")
            frozen.load_state_dict(c.state_dict(), strict=True)
            frozen.eval()
            for raw in seeds:
                selected, counts = selected_windows(c, case, root, raw)
                for window in (0, 1):
                    item = coverage[cursor]; cursor += 1
                    key = case, root, raw, window
                    state = selected[window]
                    assert (item["case"], item["C16_root"], item["raw_seed"], item["window"]) == key
                    assert item["qualified_ticks"] == counts[window]
                    assert item["selected"] == (state is not None)
                    assert item["tick"] == (state[1] if state else None)
                    if state is None:
                        assert key not in by_key
                        continue
                    row = by_key[key]; seen.add(key)
                    base, tick, submissions = state
                    assert row["tick"] == tick and tuple(row["submissions"]) == submissions
                    assert row["physical_seed"] == role_seed(raw, BASE_VERSION + ":physical")
                    assert row["v16_checkpoint_sha256"] == {"C16": c_hash, "S16": s_hash}
                    admitted = base.admitted(submissions)
                    machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
                    mask = initial_edge_mask(base.config, admitted)
                    assert np.allclose(row["machine_obs"], machine, atol=1e-7)
                    assert np.allclose(row["worker_obs"], worker, atol=1e-7)
                    assert np.array_equal(row["edge_mask"], mask)
                    actions, paths, reasons = candidate_replay(
                        base, c, frozen, s, raw, case, root, tick, submissions, mask)
                    assert row["matching_indices"] == actions
                    assert row["sequences"] == paths and row["candidate_reasons"] == reasons
                    assert len(set(actions)) == len(actions)
                    tapes = []
                    for slot in range(slots):
                        seed = role_seed(raw, f"v17-value:{case}:{root}:{tick}:{slot}")
                        assert row["bank_seeds"][slot] == seed
                        physical = DispatchEnv(base.config)
                        physical.reset(seed)
                        tapes.append(physical.events)
                    means = []
                    assert len(row["branches"]) == len(actions)
                    for j, action in enumerate(actions):
                        branch = row["branches"][j]
                        assert branch["matching_index"] == action
                        totals = []
                        for slot in range(slots):
                            total, h, tail = branch_cost(base, s, submissions, action, tapes[slot])
                            assert nearly(total, branch["total_costs"][slot])
                            assert nearly(h, branch["H_costs"][slot])
                            assert nearly(tail, branch["tail_costs"][slot])
                            assert nearly(total, h + tail)
                            totals.append(total); branches += 1
                        means.append(float(np.mean(totals)))
                    assert np.allclose(means, row["mean_costs"], atol=1e-7)
                    weights = np.exp(-(np.asarray(means) - min(means)) / 20.)
                    assert np.allclose(weights / weights.sum(), row["target_probabilities"], atol=1e-8)
    assert seen == set(by_key)
    summary = json.loads((run / "summary.json").read_text())
    counts = {case: sum(r["case"] == case for r in records) for case in cases}
    sufficient = (len(records) >= 300 and all(x >= 120 for x in counts.values())) if manifest["kind"] == "labels" else bool(records)
    assert summary == {"coverage_windows": len(coverage), "labelled_states": len(records),
                       "by_case": counts, "sufficient_coverage": sufficient, "label_branches": branches}
    assert manifest["summary"] == summary and manifest["sufficient_coverage"] == sufficient
    events = rows(run / "run.log")
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    return {"coverage_windows_replayed": len(coverage), "labelled_states_replayed": len(records),
            "branches_replayed": branches, "sufficient_coverage": sufficient}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    audit_dir = run.parent / ("full_matching_v17_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    audit_dir.mkdir(exist_ok=False)
    if (run / "audit.json").exists():
        shutil.copy2(run / "audit.json", audit_dir / "previous_audit.json")
    result = {"status": "RUNNING", "kind": manifest["kind"], "source_run": str(run),
              "audit_artifact": str(audit_dir), "manifest_sha256": labels.sha256(run / "manifest.json"),
              "auditor_sha256": labels.sha256(Path(__file__))}
    try:
        assert manifest["status"] == "COMPLETED"
        identity(run, manifest)
        result.update(replay(run, manifest))
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        labels.write_json(audit_dir / "audit.json", result)
        labels.write_json(run / "audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
