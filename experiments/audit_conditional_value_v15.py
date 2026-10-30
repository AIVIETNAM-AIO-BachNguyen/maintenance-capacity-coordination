"""Independent replay audit for v15 conditional second-edge simulator labels."""
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
from maintenance_capacity_coordination.same_tick_policy_v12 import EDGES, initial_edge_mask, choose as choose_c12
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v12 as c12_pilot
from experiments import conditional_value_labels_v15 as labels


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(a, b):
    return abs(float(a) - float(b)) < 1e-7


def independently_selected(policy, case, root, raw):
    env = ScaledCoordinationEnv(configs()[case]); env.reset(raw)
    selected = None
    for tick in range(48):
        choice = choose_c12(policy, env, True)
        if selected is None and 8 <= tick <= 39:
            mask = initial_edge_mask(env.config, env.admitted(choice["submissions"]))
            eligible = []
            for first in range(len(EDGES)):
                if not mask[first]:
                    continue
                m0, j0 = EDGES[first]
                successors = tuple(second for second in range(first + 1, len(EDGES))
                                   if mask[second] and EDGES[second][0] != m0 and EDGES[second][1] != j0)
                if len(successors) >= 2:
                    eligible.append((first, successors))
            if eligible:
                k = role_seed(raw, f"v15-first:{case}:{root}") % len(eligible)
                first, seconds = eligible[k]
                selected = (copy.deepcopy(env), tick, choice["submissions"], first, seconds)
        env.step(choice["submissions"], choice["matching_index"])
    return selected


def independently_branch(base, policy, submissions, index, event_tape):
    branch = copy.deepcopy(base)
    branch.inner._events = event_tape
    prior = float(branch.inner.metrics["objective"])
    _, done = branch.step(submissions, index)
    while not done:
        choice = choose_c12(policy, branch, True)
        _, done = branch.step(choice["submissions"], choice["matching_index"])
    metric = branch.metrics()
    return (float(metric["objective"] - prior), float(metric["H_cost"] - prior),
            float(metric["tail_cost"]))


def identity(run, manifest):
    assert manifest["protocol_sha256"] == labels.sha256(labels.PROTOCOL)
    assert manifest["config_seeds"] == CONFIG_SEEDS
    for path, expected in manifest["source_sha256"].items():
        assert labels.sha256(labels.ROOT / path) == expected, path
    for case, expected in manifest["config_sha256"].items():
        actual = hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
        assert actual == expected
    for path, expected in manifest["output_sha256"].items():
        assert labels.sha256(run / path) == expected, path
    assert manifest["C12_training_manifest_sha256"] == labels.sha256(labels.C12_TRAIN / "manifest.json")
    assert manifest["C12_training_audit_sha256"] == labels.sha256(labels.C12_TRAIN / "audit.json")
    labels.seed_registry_check(manifest["kind"])


def replay(run, manifest):
    cases = CASES if manifest["kind"] == "labels" else (CASES[0],)
    roots = labels.C12_ROOTS if manifest["kind"] == "labels" else (labels.C12_ROOTS[0],)
    seeds = labels.LABEL_SEEDS if manifest["kind"] == "labels" else labels.SMOKE_SEEDS
    slots = 4 if manifest["kind"] == "labels" else 2
    assert manifest["cases"] == list(cases) and manifest["C12_roots"] == list(roots)
    assert manifest["raw_seeds"] == list(seeds) and manifest["bank_slots"] == slots
    coverage = read_lines(run / "coverage.jsonl")
    records = read_lines(run / "labels.jsonl") if (run / "labels.jsonl").exists() else []
    assert len(coverage) == len(cases) * len(roots) * len(seeds)
    by_key = {(r["case"], r["C12_root"], r["raw_seed"]): r for r in records}
    assert len(by_key) == len(records)
    seen, branch_count = set(), 0
    coverage_cursor = 0
    for case in cases:
        for root in roots:
            checkpoint, expected_hash = labels.checkpoint(case, root)
            assert manifest["C12_checkpoint_sha256"][f"{case}:{root}"] == expected_hash
            policy = c12_pilot.load_policy(checkpoint)
            for raw in seeds:
                found = independently_selected(policy, case, root, raw)
                item = coverage[coverage_cursor]; coverage_cursor += 1
                key = case, root, raw
                assert (item["case"], item["C12_root"], item["raw_seed"]) == key
                assert item["selected"] == (found is not None)
                assert item["tick"] == (found[1] if found else None)
                if found is None:
                    assert key not in by_key
                    continue
                row = by_key[key]
                seen.add(key)
                base, tick, submissions, first, seconds = found
                assert row["tick"] == tick and row["first_edge"] == first
                assert tuple(row["second_edges"]) == seconds
                assert tuple(row["submissions"]) == tuple(submissions)
                assert row["physical_seed"] == role_seed(raw, BASE_VERSION + ":physical")
                assert row["C12_checkpoint_sha256"] == expected_hash
                admitted = base.admitted(submissions)
                machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
                mask = initial_edge_mask(base.config, admitted)
                assert np.allclose(np.asarray(row["machine_obs"]), machine, atol=1e-7)
                assert np.allclose(np.asarray(row["worker_obs"]), worker, atol=1e-7)
                assert np.array_equal(np.asarray(row["edge_mask"], dtype=bool), mask)
                assert len(row["branches"]) == len(seconds)
                tapes = []
                for slot in range(slots):
                    seed = role_seed(raw, f"v15-value:{case}:{root}:{tick}:{slot}")
                    assert row["bank_seeds"][slot] == seed
                    physical = DispatchEnv(base.config); physical.reset(seed)
                    tapes.append(physical.events)
                means = []
                for i, second in enumerate(seconds):
                    matching = tuple(sorted((EDGES[first], EDGES[second])))
                    index = base.actions.index(matching)
                    assert base.matching_mask(submissions)[index]
                    assert row["matching_indices"][i] == index
                    branch = row["branches"][i]
                    assert branch["second_edge"] == second and branch["matching_index"] == index
                    totals = []
                    for slot in range(slots):
                        replayed = independently_branch(base, policy, submissions, index, tapes[slot])
                        assert close(replayed[0], branch["total_costs"][slot])
                        assert close(replayed[1], branch["H_costs"][slot])
                        assert close(replayed[2], branch["tail_costs"][slot])
                        assert close(replayed[0], replayed[1] + replayed[2])
                        totals.append(replayed[0]); branch_count += 1
                    means.append(float(np.mean(totals)))
                assert np.allclose(means, row["mean_costs"], atol=1e-7)
                unnormalised = np.exp(-(np.asarray(means) - min(means)) / 20.)
                targets = unnormalised / unnormalised.sum()
                assert np.allclose(targets, row["target_probabilities"], atol=1e-8)
    assert seen == set(by_key)
    summary = json.loads((run / "summary.json").read_text())
    case_counts = {case: sum(r["case"] == case for r in records) for case in cases}
    sufficient = (all(n >= 50 for n in case_counts.values()) and len(records) >= 150) if manifest["kind"] == "labels" else bool(records)
    assert summary == {"coverage_episodes": len(coverage), "labelled_states": len(records),
                       "by_case": case_counts, "label_branches": branch_count,
                       "sufficient_coverage": sufficient}
    assert manifest["summary"] == summary and manifest["sufficient_coverage"] == sufficient
    events = read_lines(run / "run.log")
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    return {"coverage_episodes_replayed": len(coverage), "labelled_states_replayed": len(records),
            "branches_replayed": branch_count, "sufficient_coverage": sufficient}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    audit_dir = run.parent / ("conditional_value_v15_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
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
