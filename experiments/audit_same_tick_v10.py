"""Independent artifact and physical replay audit for the prelocked v10 pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import evaluate_episode, RuleController
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from experiments.same_tick_v10_config import CASES, configs
from maintenance_capacity_coordination.structured_policy_v3 import ACTIONS
from experiments import same_tick_pilot_v10 as pilot
from maintenance_capacity_coordination.same_tick_policy_v8 import EDGES, STOP, initial_edge_mask, matching_from_sequence


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearly(a, b):
    return abs(float(a) - float(b)) < 1e-7


def identity(run, manifest):
    assert manifest["protocol_sha256"] == pilot.sha256(pilot.PROTOCOL)
    for path, expected in manifest["source_sha256"].items():
        assert pilot.sha256(pilot.ROOT / path) == expected, path
    cfg = configs()
    for case, expected in manifest["config_sha256"].items():
        actual = hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
        assert actual == expected
    for path, expected in manifest["output_sha256"].items():
        assert pilot.sha256(run / path) == expected, path
    for path, expected in manifest["v14_checkpoint_sha256"].items():
        assert pilot.sha256(pilot.ROOT / path) == expected, path
    pilot.seed_registry_check()


def episode_accounting(row):
    assert row["invalid_assignments"] == 0
    assert nearly(row["episode_return"] + row["objective"], 0)
    assert nearly(row["H_cost"] + row["tail_cost"], row["objective"])


def audit_training_trace(trial, case, arm, root, steps, episode_rows, record):
    traces = rows(trial / "training_decisions.jsonl")
    assert len(traces) == steps
    env = CoordinationEnv(configs()[case])
    episode, tick, complete = 0, 0, 0
    env.reset(role_seed(root, f"same_tick_v10:{case}:{episode}"))
    passes = 0
    for trace in traces:
        reset = role_seed(root, f"same_tick_v10:{case}:{episode}")
        assert (trace["case"], trace["arm"], trace["train_root"], trace["episode"], trace["tick"], trace["reset_seed"]) == (case, arm, root, episode, tick, reset)
        submissions = tuple(trace["submissions"])
        index = trace["matching_index"]
        assert 0 <= index < len(env.actions)
        if arm in ("B10", "C10", "D10"):
            sequence = trace["sequence"]
            assert isinstance(sequence, list)
            assert matching_from_sequence(sequence) == env.actions[index]
            edge_mask = initial_edge_mask(env.config, env.admitted(submissions))
            used_m, used_w, stopped = set(), set(), False
            for offset, token in enumerate(sequence):
                feasible = [i for i, (m, j) in enumerate(EDGES)
                            if edge_mask[i] and m not in used_m and j not in used_w]
                assert feasible and not stopped
                if token == STOP:
                    assert offset == len(sequence) - 1
                    stopped = True
                else:
                    assert token in feasible
                    m, j = EDGES[token]
                    used_m.add(m)
                    used_w.add(j)
            if not stopped:
                assert not any(edge_mask[i] and m not in used_m and j not in used_w
                               for i, (m, j) in enumerate(EDGES))
        else:
            assert trace["sequence"] is None
        assert np.isfinite(trace["logprob"])
        reward, done = env.step(submissions, index)
        assert nearly(reward, trace["reward"])
        assert done == trace["done"]
        passes += trace["decoder_forward_passes"]
        if done:
            row = episode_rows[complete]
            assert row["episode"] == episode and row["transition"] == complete * 48 + 48
            for metric, value in env.metrics().items():
                if isinstance(value, (int, float)) and value is not None:
                    assert nearly(row[metric], value), (case, arm, root, episode, metric)
                else:
                    assert row[metric] == value
            complete += 1
            episode += 1
            tick = 0
            env.reset(role_seed(root, f"same_tick_v10:{case}:{episode}"))
        else:
            tick += 1
    assert complete == len(episode_rows)
    assert tick == steps % 48
    assert passes == record["decoder_forward_passes"]
    return len(traces)


def audit_training(run, manifest):
    kind = manifest["kind"]
    cases = CASES if kind != "smoke" else (CASES[0],)
    roots = (pilot.SMOKE_ROOT,) if kind == "smoke" else (pilot.TRAIN_ROOTS if kind == "train" else pilot.REPL_TRAIN_ROOTS)
    dev_seeds = pilot.SMOKE_SEEDS if kind == "smoke" else (pilot.DEV_SEEDS if kind == "train" else pilot.REPL_DEV_SEEDS)
    steps = 256 if kind == "smoke" else 8192
    milestones = (steps,) if kind == "smoke" else pilot.MILESTONES
    expected_trials = {(case, arm, root) for case in cases for arm in pilot.ARMS for root in roots}
    assert {(r["case"], r["arm"], r["train_root"]) for r in manifest["trials"]} == expected_trials
    events = rows(run / "run.log")
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    assert sum(e["event"] == "trial_completed" for e in events) == len(expected_trials)
    trials = 0
    for case, arm, root in sorted(expected_trials):
        trial = run / f"{case}_{arm}_{root}"
        record = json.loads((trial / "manifest.json").read_text())
        assert record["status"] == "COMPLETED"
        assert record["actual_H_decisions"] == steps
        policy = pilot.new_policy(arm)
        assert record["actor_parameters"] == policy.actor_parameters
        assert record["critic_parameters"] == policy.critic_parameters
        update_rows = rows(trial / "updates.jsonl")
        assert [r["transition"] for r in update_rows] == list(range(256, steps + 1, 256))
        for r in update_rows:
            assert all(np.isfinite(v) for k, v in r.items() if k != "transition")
        train_episodes = rows(trial / "training_episodes.jsonl")
        assert [r["episode"] for r in train_episodes] == list(range(len(train_episodes)))
        for r in train_episodes:
            episode_accounting(r)
        audit_training_trace(trial, case, arm, root, steps, train_episodes, record)
        dev_rows = rows(trial / "dev_episodes.jsonl")
        assert len(dev_rows) == len(milestones) * len(dev_seeds)
        summaries = json.loads((trial / "dev_milestones.json").read_text())
        assert [r["transition"] for r in summaries] == list(milestones)
        for milestone, summary in zip(milestones, summaries):
            matching = [r for r in dev_rows if r["checkpoint"] == str(milestone)]
            assert [r["seed"] for r in matching] == list(dev_seeds)
            assert nearly(summary["mean_H_tail_cost"], np.mean([r["objective"] for r in matching]))
            checkpoint = trial / summary["checkpoint"]
            assert pilot.sha256(checkpoint) == summary["checkpoint_sha256"]
            loaded = pilot.load_policy(checkpoint)
            assert loaded.actor_parameters == policy.actor_parameters
            for r in matching:
                episode_accounting(r)
                assert r["physical_seed"] == role_seed(r["seed"], BASE_VERSION + ":physical")
        selection = json.loads((trial / "selection.json").read_text())
        assert selection == min(summaries, key=lambda r: (r["mean_H_tail_cost"], r["transition"]))
        assert selection["checkpoint"] == record["selected_checkpoint"]
        assert nearly(selection["mean_H_tail_cost"], record["dev_mean"])
        trials += 1
    return {"trials": trials, "training_H_decisions": trials * steps,
            "dev_episodes": trials * len(dev_seeds) * len(milestones)}


def selected_policies(training_run, panel):
    roots = pilot.TRAIN_ROOTS if panel == "heldout" else pilot.REPL_TRAIN_ROOTS
    result = {}
    for case in CASES:
        for arm in pilot.ARMS:
            for root in roots:
                trial = training_run / f"{case}_{arm}_{root}"
                selected = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selected["checkpoint"]
                assert pilot.sha256(checkpoint) == selected["checkpoint_sha256"]
                result[case, arm, root] = (pilot.load_policy(checkpoint), f"{root}:{selected['transition']}")
    return result


def audit_episodes(run, manifest):
    kind = manifest["kind"]
    panel = "heldout" if kind == "heldout" else "replication_eval"
    seeds = pilot.HELDOUT_SEEDS if panel == "heldout" else pilot.REPL_EVAL_SEEDS
    training_run = Path(manifest["training_run"])
    original = json.loads((training_run / "manifest.json").read_text())
    assert original["status"] == "COMPLETED"
    assert json.loads((training_run / "audit.json").read_text())["status"] == "PASS"
    for key in ("source_sha256", "protocol_sha256", "config_sha256"):
        assert original[key] == manifest[key]
    checkpoints = selected_policies(training_run, panel)
    episode_rows = rows(run / f"{panel}_episodes.jsonl")
    decisions = rows(run / f"{panel}_decisions.jsonl")
    expected = len(checkpoints) * len(seeds)
    assert manifest["learned_episodes"] == expected == len(episode_rows)
    assert len(decisions) == expected * 48
    observed = set()
    cursor = 0
    for row in episode_rows:
        case, arm, seed = row["case"], row["arm"], row["seed"]
        root = int(row["checkpoint"].split(":")[0])
        key = case, arm, root, seed
        assert key not in observed
        observed.add(key)
        assert seed in seeds
        policy, label = checkpoints[case, arm, root]
        assert row["checkpoint"] == label
        assert row["physical_seed"] == role_seed(seed, BASE_VERSION + ":physical")
        env = CoordinationEnv(configs()[case])
        env.reset(seed)
        total_passes, total_cardinality = 0, 0
        for tick in range(48):
            trace = decisions[cursor]
            cursor += 1
            assert (trace["case"], trace["arm"], trace["seed"], trace["tick"]) == (case, arm, seed, tick)
            choice = pilot.choose(policy, env, True)
            assert list(choice["submissions"]) == trace["submissions"]
            assert choice["matching_index"] == trace["matching_index"]
            assert ACTIONS[choice["matching_index"]] == tuple(tuple(p) for p in trace["matching"])
            assert (list(choice.get("tokens")) if choice.get("tokens") is not None else None) == trace["sequence"]
            assert nearly(choice["old_logprob"], trace["logprob"])
            assert choice.get("decoder_forward_passes", 1) == trace["decoder_forward_passes"]
            reward, _ = env.step(choice["submissions"], choice["matching_index"])
            assert nearly(reward, trace["reward"])
            total_passes += trace["decoder_forward_passes"]
            total_cardinality += len(ACTIONS[choice["matching_index"]])
        metrics = env.metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                assert nearly(row[key], value), (case, arm, seed, key)
            else:
                assert row[key] == value, (case, arm, seed, key)
        assert row["decoder_forward_passes"] == total_passes
        assert row["matching_cardinality_sum"] == total_cardinality
        episode_accounting(row)
    assert cursor == len(decisions)
    assert observed == {(case, arm, root, seed) for case, arm, root in checkpoints for seed in seeds}
    controls = rows(run / f"{panel}_controls.jsonl")
    assert len(controls) == manifest["control_episodes"] == len(CASES) * (2 + 5) * len(seeds)
    control_keys = set()
    for r in controls:
        assert r["seed"] in seeds and r["case"] in CASES
        key = (r["case"], r["arm"], r.get("historical_train_seed"), r["seed"])
        assert key not in control_keys
        control_keys.add(key)
        assert r["invalid_assignments"] == 0
        assert nearly(r["H_cost"] + r["tail_cost"], r["objective"])
        if r["arm"] == "A":
            directory = pilot.ROOT / f"baselines/v14/checkpoints/count_calibrated/{r['historical_train_seed']}"
            paths = list(directory.glob("*.pt"))
            assert len(paths) == 1
            controller = V14Controller.from_checkpoint(paths[0])
        else:
            assert r["arm"] in ("F_risk_matching", "F_rollout")
            controller = RuleController(r["arm"][2:])
        replay = evaluate_episode(configs()[r["case"]], r["case"], r["seed"], controller)
        for metric, value in replay.items():
            if metric in ("selection_seconds", "episode_wall_seconds"):
                continue
            if isinstance(value, (int, float)) and value is not None:
                assert nearly(r[metric], value), (key, metric)
            else:
                assert r[metric] == value, (key, metric)
    assert control_keys == {(case, arm, root, seed) for case in CASES
                            for arm, roots in (("A", (72600000, 72600001, 72600002, 72600003, 72600004)),
                                               ("F_risk_matching", (None,)), ("F_rollout", (None,)))
                            for root in roots for seed in seeds}
    return episode_rows, {"learned_episodes_replayed": len(episode_rows),
                          "decision_rows_replayed": len(decisions), "control_episodes_checked": len(controls)}


def clustered_contrast(data, left, right, seeds, bootstrap_seed):
    roots = pilot.TRAIN_ROOTS if len(seeds) == len(pilot.HELDOUT_SEEDS) and bootstrap_seed == pilot.BOOTSTRAP_SEED else pilot.REPL_TRAIN_ROOTS
    cube = np.empty((len(seeds), len(CASES), len(roots)), dtype=float)
    by_key = {(r["seed"], r["case"], r["arm"], int(r["checkpoint"].split(":")[0])): r["objective"] for r in data}
    for si, seed in enumerate(seeds):
        for ci, case in enumerate(CASES):
            for ri, root in enumerate(roots):
                cube[si, ci, ri] = by_key[seed, case, left, root] - by_key[seed, case, right, root]
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), (5000, len(seeds)))
    interval = np.quantile(cube.mean((1, 2))[draws].mean(1), [.025, .975])
    return {"overall": float(cube.mean()), "case_means": {case: float(cube[:, i, :].mean()) for i, case in enumerate(CASES)},
            "training_root_means": {str(root): float(cube[:, :, i].mean()) for i, root in enumerate(roots)},
            "physical_cluster_95_interval": [float(interval[0]), float(interval[1])]}


def gate(data, panel):
    seeds = pilot.HELDOUT_SEEDS if panel == "heldout" else pilot.REPL_EVAL_SEEDS
    bs = pilot.BOOTSTRAP_SEED if panel == "heldout" else pilot.REPL_BOOTSTRAP_SEED
    contrasts = {name: clustered_contrast(data, "C10", name, seeds, bs)
                 for name in ("D10", "B10", "S10", "G10")}
    cd, cb, cg = contrasts["D10"], contrasts["B10"], contrasts["G10"]
    if panel == "heldout":
        conditions = {
            "C10_minus_D10_at_most_minus_20": cd["overall"] <= -20,
            "C10_minus_D10_both_cases_negative": all(x < 0 for x in cd["case_means"].values()),
            "C10_minus_D10_two_roots_negative": sum(x < 0 for x in cd["training_root_means"].values()) >= 2,
            "C10_minus_D10_cluster_upper_negative": cd["physical_cluster_95_interval"][1] < 0,
            "C10_minus_B10_negative": cb["overall"] < 0,
            "C10_minus_G10_negative": cg["overall"] < 0,
        }
    else:
        conditions = {
            "C10_minus_D10_negative": cd["overall"] < 0,
            "C10_minus_G10_negative": cg["overall"] < 0,
            "C10_minus_D10_both_cases_negative": all(x < 0 for x in cd["case_means"].values()),
            "C10_minus_D10_cluster_upper_negative": cd["physical_cluster_95_interval"][1] < 0,
        }
    return {"panel": panel, "contrasts": contrasts, "conditions": conditions,
            "go": all(conditions.values()), "protocol_sha256": pilot.sha256(pilot.PROTOCOL)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    audit_dir = run.parent / ("same_tick_v10_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    audit_dir.mkdir(exist_ok=False)
    if (run / "audit.json").exists():
        shutil.copy2(run / "audit.json", audit_dir / "previous_audit.json")
    result = {"status": "RUNNING", "kind": manifest["kind"],
              "source_run": str(run), "audit_artifact": str(audit_dir),
              "manifest_sha256": pilot.sha256(run / "manifest.json"),
              "auditor_sha256": pilot.sha256(Path(__file__))}
    try:
        assert manifest["status"] == "COMPLETED"
        identity(run, manifest)
        if manifest["kind"] in ("smoke", "train", "replication_train"):
            result.update(audit_training(run, manifest))
        else:
            data, counts = audit_episodes(run, manifest)
            result.update(counts)
            panel = "heldout" if manifest["kind"] == "heldout" else "replication_eval"
            decision = gate(data, panel)
            decision_path = run / "replication_go_decision.json"
            if decision_path.exists():
                shutil.copy2(decision_path, audit_dir / "previous_decision.json")
            pilot.write_json(audit_dir / "replication_go_decision.json", decision)
            pilot.write_json(decision_path, decision)
            result["decision_sha256"] = pilot.sha256(decision_path)
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        pilot.write_json(audit_dir / "audit.json", result)
        pilot.write_json(run / "audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
