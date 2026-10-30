"""Independent state/branch/cost replay audit for v13 second-choice diagnostic."""
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
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v12 import EDGES, STOP, choose
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v12 as pilot
from experiments import second_choice_value_v13 as study


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(a, b):
    return abs(float(a) - float(b)) < 1e-7


def fresh_events(config, seed):
    env = DispatchEnv(config)
    env.reset(seed)
    return env.events


def independent_branch_cost(snapshot, policy, index, submissions, events):
    env = copy.deepcopy(snapshot)
    env.inner._events = events
    before = env.inner.metrics["objective"]
    _, done = env.step(submissions, index)
    while not done:
        decision = choose(policy, env, deterministic=True)
        _, done = env.step(decision["submissions"], decision["matching_index"])
    metrics = env.metrics()
    return (metrics["objective"] - before, metrics["H_cost"] - before, metrics["tail_cost"])


def replay_episode(case, root, raw, policy):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    selected = None
    for tick in range(48):
        decision = choose(policy, env, deterministic=True)
        first = decision["tokens"][0] if decision["tokens"] else STOP
        residual = ()
        if first != STOP:
            m0, j0 = EDGES[first]
            residual = tuple(i for i, (m, j) in enumerate(EDGES)
                             if decision["edge_mask"][i] and m != m0 and j != j0)
        if selected is None and 8 <= tick <= 39 and len(residual) >= 2:
            selected = (copy.deepcopy(env), tick, decision["submissions"], first, residual)
        env.step(decision["submissions"], decision["matching_index"])
    return selected


def independent_summary(state_rows, coverage, science):
    if not science:
        return None
    case_rows = {case: [r for r in state_rows if r["case"] == case] for case in CASES}
    counts = {case: len(case_rows[case]) for case in CASES}
    means = {case: float(np.mean([r["validation_gap"] for r in case_rows[case]]))
             if case_rows[case] else None for case in CASES}
    roots = {str(root): float(np.mean([r["validation_gap"] for r in state_rows if r["root"] == root]))
             if any(r["root"] == root for r in state_rows) else None for root in pilot.TRAIN_ROOTS}
    overall = float(np.mean(list(means.values()))) if all(x is not None for x in means.values()) else None
    ci, valid = None, 0
    if all(counts.values()):
        grouped = {(seed, case): [r["validation_gap"] for r in state_rows
                                  if r["raw_seed"] == seed and r["case"] == case]
                   for seed in study.SCIENCE_SEEDS for case in CASES}
        draws = np.random.default_rng(study.BOOTSTRAP_SEED).choice(
            study.SCIENCE_SEEDS, size=(5000, len(study.SCIENCE_SEEDS)), replace=True)
        boot = []
        for draw in draws:
            case_values = []
            for case in CASES:
                observations = [x for seed in draw for x in grouped[seed, case]]
                if not observations:
                    break
                case_values.append(float(np.mean(observations)))
            if len(case_values) == len(CASES):
                boot.append(float(np.mean(case_values)))
        valid = len(boot)
        if boot:
            ci = [float(x) for x in np.quantile(boot, [.025, .975])]
    conditions = {"at_least_12_states_each_case": all(n >= 12 for n in counts.values()),
                  "both_case_validation_means_at_least_10": all(x is not None and x >= 10 for x in means.values()),
                  "cluster_95_lower_positive": ci is not None and ci[0] > 0,
                  "all_5000_bootstrap_draws_defined": valid == 5000}
    return {"coverage_episodes": len(coverage), "selected_states": len(state_rows),
            "case_counts": counts, "case_means": means, "root_means": roots,
            "overall_validation_gap": overall,
            "mean_bank_a_spread": float(np.mean([r["bank_a_spread"] for r in state_rows])) if state_rows else None,
            "mean_bank_b_spread": float(np.mean([r["bank_b_spread"] for r in state_rows])) if state_rows else None,
            "raw_seed_cluster_95_interval": ci, "bootstrap_draws_defined": valid,
            "conditions": conditions, "go": all(conditions.values()),
            "interpretation": "bank-B held-out ranking signal under frozen C12 continuation, not policy gain"}


def audit(run: Path):
    torch.set_num_threads(1)
    study.seed_registry_check()
    manifest = json.loads((run / "manifest.json").read_text())
    artifact = run.parent / ("second_choice_v13_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    artifact.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "source_run": str(run), "kind": manifest["kind"],
              "manifest_sha256": study.sha256(run / "manifest.json"),
              "auditor_sha256": study.sha256(Path(__file__))}
    try:
        assert manifest["status"] == "COMPLETED"
        assert manifest["protocol_sha256"] == study.sha256(study.PROTOCOL)
        assert manifest["git_commit"] == study.git_output("rev-parse", "HEAD")
        for path, digest in manifest["source_sha256"].items():
            assert study.sha256(study.ROOT / path) == digest, path
        for path, digest in manifest["output_sha256"].items():
            assert study.sha256(run / path) == digest, path
        for case, digest in manifest["config_sha256"].items():
            actual = hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
            assert actual == digest
        assert manifest["training_manifest_sha256"] == study.sha256(study.TRAIN / "manifest.json")
        assert manifest["training_audit_sha256"] == study.sha256(study.TRAIN / "audit.json")
        assert json.loads((study.TRAIN / "audit.json").read_text())["status"] == "PASS"
        assert all(study.sha256(study.ROOT / path) == digest
                   for path, digest in json.loads((study.TRAIN / "manifest.json").read_text())["source_sha256"].items())
        science = manifest["kind"] == "diagnostic"
        cases = CASES if science else (CASES[0],)
        roots = pilot.TRAIN_ROOTS if science else (pilot.TRAIN_ROOTS[0],)
        seeds = study.SCIENCE_SEEDS if science else study.SMOKE_SEEDS
        slots = 8 if science else 2
        assert tuple(manifest["cases"]) == cases and tuple(manifest["roots"]) == roots
        assert tuple(manifest["raw_seeds"]) == seeds and manifest["bank_slots"] == slots
        assert manifest["checkpoint_sha256"] == {
            f"{case}:{root}": study.checkpoint_selection(case, root)[1]["checkpoint_sha256"]
            for case in cases for root in roots}
        coverage = rows(run / "coverage.jsonl")
        state_rows = rows(run / "states.jsonl") if (run / "states.jsonl").exists() else []
        expected_keys = [(case, root, raw) for case in cases for root in roots for raw in seeds]
        assert [(r["case"], r["root"], r["raw_seed"]) for r in coverage] == expected_keys
        observed_states = []
        cursor = 0
        replayed_branches = 0
        for case in cases:
            for root in roots:
                checkpoint, selection = study.checkpoint_selection(case, root)
                policy = pilot.load_policy(checkpoint)
                for raw in seeds:
                    cover = coverage[cursor]
                    selected = replay_episode(case, root, raw, policy)
                    assert cover["selected"] == (selected is not None)
                    assert cover["tick"] == (selected[1] if selected else None)
                    if selected is not None:
                        row = state_rows[len(observed_states)]
                        snapshot, tick, submissions, first, residual = selected
                        assert (row["case"], row["root"], row["raw_seed"], row["tick"]) == (case, root, raw, tick)
                        assert row["checkpoint_sha256"] == selection["checkpoint_sha256"]
                        assert row["physical_seed"] == role_seed(raw, BASE_VERSION + ":physical")
                        assert row["first_edge"] == first and tuple(row["submissions"]) == submissions
                        assert [c["second_edge"] for c in row["candidates"]] == list(residual)
                        bank = [study.bank_seed(raw, case, root, tick, slot) for slot in range(slots)]
                        assert row["bank_seeds"] == bank and len(set(bank)) == slots
                        for candidate in row["candidates"]:
                            m0, j0 = EDGES[first]
                            matching = tuple(sorted(((m0, j0), EDGES[candidate["second_edge"]])))
                            action = snapshot.actions.index(matching)
                            assert candidate["matching_index"] == action
                            assert snapshot.matching_mask(submissions)[action]
                            for slot, seed in enumerate(bank):
                                total, h, tail = independent_branch_cost(
                                    snapshot, policy, action, submissions, fresh_events(snapshot.config, seed))
                                assert close(total, candidate["total_costs"][slot])
                                assert close(h, candidate["H_costs"][slot])
                                assert close(tail, candidate["tail_costs"][slot])
                                assert close(total, h + tail)
                                replayed_branches += 1
                        if science:
                            by_second = {c["second_edge"]: c for c in row["candidates"]}
                            a_mean = {i: float(np.mean(c["total_costs"][:4])) for i, c in by_second.items()}
                            best = min(residual, key=lambda i: (a_mean[i], i))
                            worst = min(residual, key=lambda i: (-a_mean[i], i))
                            assert row["bank_a_best"] == best and row["bank_a_worst"] == worst
                            assert close(row["bank_a_spread"], a_mean[worst] - a_mean[best])
                            gap = np.mean(by_second[worst]["total_costs"][4:]) - np.mean(by_second[best]["total_costs"][4:])
                            assert close(row["validation_gap"], gap)
                            b_spread = max(np.mean(c["total_costs"][4:]) for c in row["candidates"]) \
                                     - min(np.mean(c["total_costs"][4:]) for c in row["candidates"])
                            assert close(row["bank_b_spread"], b_spread)
                        observed_states.append(row)
                    cursor += 1
        assert len(observed_states) == len(state_rows) == manifest["selected_states"]
        assert len(coverage) == manifest["coverage_episodes"]
        if science:
            summary = json.loads((run / "summary.json").read_text())
            expected = independent_summary(state_rows, coverage, True)
            assert summary.keys() == expected.keys()
            for key in summary:
                if key in ("case_means", "root_means"):
                    for name in summary[key]:
                        if summary[key][name] is None:
                            assert expected[key][name] is None
                        else:
                            assert close(summary[key][name], expected[key][name])
                elif isinstance(summary[key], (float, int)) and not isinstance(summary[key], bool):
                    assert close(summary[key], expected[key])
                elif key == "raw_seed_cluster_95_interval" and summary[key] is not None:
                    assert all(close(a, b) for a, b in zip(summary[key], expected[key]))
                else:
                    assert summary[key] == expected[key]
            result["gate"] = summary["go"]
        result.update(status="PASS", coverage_episodes=len(coverage), selected_states=len(state_rows),
                      branches_replayed=replayed_branches)
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        study.json_write(artifact / "audit.json", result)
        study.json_write(run / "audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audit(args.run.resolve())
