"""Independent artifact and physical replay audit for the prelocked v17 pilot."""
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
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from maintenance_capacity_coordination.scaled_matching_policy_v15 import ACTIONS
from experiments import same_tick_pilot_v17 as pilot
from experiments.full_matching_pretrain_v17 import EPOCHS as NEW_EPOCHS, pretrain as new_pretrain
from experiments.conditional_value_pretrain_v15 import EPOCHS as OLD_EPOCHS, pretrain as old_pretrain
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, STOP, SequencePolicy, initial_edge_mask, matching_from_sequence
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy


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
    assert manifest["config_seeds"] == CONFIG_SEEDS
    pilot.seed_registry_check()


def episode_accounting(row):
    assert row["invalid_assignments"] == 0
    assert nearly(row["episode_return"] + row["objective"], 0)
    assert nearly(row["H_cost"] + row["tail_cost"], row["objective"])


def audit_training_trace(trial, case, arm, root, steps, episode_rows, record):
    traces = rows(trial / "training_decisions.jsonl")
    assert len(traces) == steps
    env = ScaledCoordinationEnv(configs()[case])
    episode, tick, complete = 0, 0, 0
    env.reset(role_seed(root, f"same_tick_v17:{case}:{episode}"))
    passes = 0
    for trace in traces:
        reset = role_seed(root, f"same_tick_v17:{case}:{episode}")
        assert (trace["case"], trace["arm"], trace["train_root"], trace["episode"], trace["tick"], trace["reset_seed"]) == (case, arm, root, episode, tick, reset)
        submissions = tuple(trace["submissions"])
        index = trace["matching_index"]
        assert 0 <= index < len(env.actions)
        if arm in ("C17", "D17", "I17", "R17", "B17", "O17", "P17"):
            sequence = trace["sequence"]
            assert isinstance(sequence, list)
            assert matching_from_sequence(sequence) == env.actions[index]
            edge_mask = initial_edge_mask(env.config, env.admitted(submissions))
            used_m, used_w, stopped, last = set(), set(), False, -1
            for offset, token in enumerate(sequence):
                feasible = [i for i, (m, j) in enumerate(EDGES)
                            if edge_mask[i] and m not in used_m and j not in used_w
                            and i > last]
                assert feasible and not stopped
                if token == STOP:
                    assert offset == len(sequence) - 1
                    stopped = True
                else:
                    assert token in feasible
                    m, j = EDGES[token]
                    used_m.add(m)
                    used_w.add(j)
                    last = token
            if not stopped:
                assert not any(edge_mask[i] and m not in used_m and j not in used_w
                               and i > last
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
            env.reset(role_seed(root, f"same_tick_v17:{case}:{episode}"))
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
    label_run = Path(manifest["label_run"])
    assert pilot.sha256(label_run / "manifest.json") == manifest["label_manifest_sha256"]
    assert pilot.sha256(label_run / "audit.json") == manifest["label_audit_sha256"]
    assert pilot.sha256(label_run / "labels.jsonl") == manifest["labels_sha256"]
    assert pilot.sha256(label_run / "summary.json") == manifest["label_summary_sha256"]
    label_manifest = json.loads((label_run / "manifest.json").read_text())
    assert label_manifest["kind"] == ("smoke" if kind == "smoke" else "labels")
    assert label_manifest["sufficient_coverage"]
    assert json.loads((label_run / "audit.json").read_text())["status"] == "PASS"
    labelled = rows(label_run / "labels.jsonl")
    by_case = {case: [item for item in labelled if item["case"] == case] for case in CASES}
    old_run = pilot.OLD_LABEL_RUN
    assert manifest["old_label_run"] == str(old_run.resolve())
    assert pilot.sha256(old_run / "manifest.json") == manifest["old_label_manifest_sha256"] == pilot.OLD_LABEL_MANIFEST_SHA256
    assert pilot.sha256(old_run / "audit.json") == manifest["old_label_audit_sha256"] == pilot.OLD_LABEL_AUDIT_SHA256
    assert pilot.sha256(old_run / "labels.jsonl") == manifest["old_labels_sha256"] == pilot.OLD_LABELS_SHA256
    assert json.loads((old_run / "audit.json").read_text())["status"] == "PASS"
    old_labelled = rows(old_run / "labels.jsonl")
    old_by_case = {case: [item for item in old_labelled if item["case"] == case] for case in CASES}
    trials = 0
    for case, arm, root in sorted(expected_trials):
        trial = run / f"{case}_{arm}_{root}"
        record = json.loads((trial / "manifest.json").read_text())
        assert record["status"] == "COMPLETED"
        assert record["actual_H_decisions"] == steps
        policy = pilot.new_policy(arm)
        assert record["actor_parameters"] == policy.actor_parameters
        assert record["critic_parameters"] == policy.critic_parameters
        pretrain_path = trial / "pretrain_checkpoint.pt"
        assert pilot.sha256(pretrain_path) == record["pretrain_checkpoint_sha256"]
        saved = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        assert (saved["arm"], saved["case"], saved["train_root"]) == (arm, case, root)
        torch.manual_seed(root)
        expected_policy = pilot.new_policy(arm)
        if arm in pilot.PRETRAIN_ARMS:
            expected_seed = role_seed(root, f"v17-pretrain:{case}:{arm}")
            logged = rows(trial / "pretrain_epochs.jsonl")
            replayed = []
            if arm == "O17":
                details = old_pretrain(expected_policy, old_by_case[case], expected_seed,
                                       on_epoch=replayed.append)
                expected_epochs, bank = OLD_EPOCHS, "v15_exact_two_edge"
            else:
                details = new_pretrain(expected_policy, by_case[case], expected_seed,
                                       on_epoch=replayed.append)
                expected_epochs, bank = NEW_EPOCHS, "v17_complete_matching"
            assert len(logged) == len(replayed) == expected_epochs
            for actual, expected in zip(logged, replayed):
                assert actual["epoch"] == expected["epoch"]
                assert actual["cumulative_updates"] == expected["cumulative_updates"]
                assert nearly(actual["mean_listwise_loss"], expected["mean_listwise_loss"])
            for key, value in details.items():
                assert record[key] == value
            assert record["pretraining_seed"] == expected_seed
            assert record["pretraining_label_bank"] == bank
        else:
            assert record["pretraining_records"] == record["pretraining_epochs"] == record["pretraining_updates"] == 0
            assert record["pretraining_seed"] is None and record["pretraining_label_bank"] is None
            assert not (trial / "pretrain_epochs.jsonl").exists()
        for key, value in expected_policy.state_dict().items():
            assert torch.equal(value, saved["state_dict"][key]), (case, arm, root, key)
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
                policy = pilot.load_policy(checkpoint)
                result[case, arm, root] = (policy, f"{root}:{selected['transition']}")
                if arm == "C17":
                    frozen = DynamicSequencePolicy("D16")
                    frozen.load_state_dict(policy.state_dict(), strict=True)
                    frozen.eval()
                    result[case, "CF17", root] = (frozen, f"{root}:{selected['transition']}")
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
        env = ScaledCoordinationEnv(configs()[case])
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
    traces = rows(run / f"{panel}_control_decisions.jsonl")
    assert len(controls) == manifest["control_episodes"] == len(CASES) * 2 * len(seeds)
    assert len(traces) == len(controls) * HORIZON
    control_keys = set()
    control_cursor = 0
    for row in controls:
        case, arm, seed = row["case"], row["arm"], row["seed"]
        key = case, arm, seed
        assert case in CASES and seed in seeds and arm in ("F_risk_matching", "F_rollout")
        assert key not in control_keys
        control_keys.add(key)
        assert row["physical_seed"] == role_seed(seed, BASE_VERSION + ":physical")
        env = ScaledCoordinationEnv(configs()[case]); env.reset(seed)
        controller = RuleController(arm[2:])
        for tick in range(HORIZON):
            trace = traces[control_cursor]; control_cursor += 1
            assert (trace["case"], trace["arm"], trace["seed"], trace["tick"]) == (case, arm, seed, tick)
            submissions, matching = controller.choose(env.config, env.inner.state, tick,
                                                      case=case, seed=seed)
            index = env.actions.index(matching)
            assert list(submissions) == trace["submissions"] and index == trace["matching_index"]
            reward, done = env.step(submissions, index)
            assert nearly(reward, trace["reward"]) and done == trace["done"]
        for metric, value in env.metrics().items():
            if isinstance(value, (int, float)) and value is not None:
                assert nearly(row[metric], value), (key, metric)
            else:
                assert row[metric] == value, (key, metric)
        assert row["forecast_transitions"] == controller.forecast_transitions
        episode_accounting(row)
    assert control_cursor == len(traces)
    assert control_keys == {(case, arm, seed) for case in CASES
                            for arm in ("F_risk_matching", "F_rollout") for seed in seeds}
    return episode_rows, {"learned_episodes_replayed": len(episode_rows),
                          "decision_rows_replayed": len(decisions),
                          "control_episodes_checked": len(controls),
                          "control_decisions_replayed": len(traces)}


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
    bootstrap = pilot.BOOTSTRAP_SEED if panel == "heldout" else pilot.REPL_BOOTSTRAP_SEED
    contrasts = {name: clustered_contrast(data, "C17", name, seeds, bootstrap)
                 for name in ("D17", "CF17", "I17", "R17", "O17", "B17", "S17", "G17", "P17", "Q17")}
    cd, cf, ci, cr, co, cb, cs, cg = (contrasts[name] for name in
        ("D17", "CF17", "I17", "R17", "O17", "B17", "S17", "G17"))
    negative_cases = lambda item: all(value < 0 for value in item["case_means"].values())
    upper_negative = lambda item: item["physical_cluster_95_interval"][1] < 0
    if panel == "heldout":
        conditions = {
            "C17_minus_D17_at_most_minus_20": cd["overall"] <= -20,
            "C17_minus_D17_both_cases_negative": negative_cases(cd),
            "C17_minus_D17_two_roots_negative": sum(value < 0 for value in cd["training_root_means"].values()) >= 2,
            "C17_minus_D17_cluster_upper_negative": upper_negative(cd),
            "C17_minus_CF17_at_most_minus_10": cf["overall"] <= -10,
            "C17_minus_CF17_both_cases_negative": negative_cases(cf),
            "C17_minus_CF17_cluster_upper_negative": upper_negative(cf),
            "C17_minus_I17_at_most_minus_10": ci["overall"] <= -10,
            "C17_minus_I17_both_cases_negative": negative_cases(ci),
            "C17_minus_I17_cluster_upper_negative": upper_negative(ci),
            "C17_minus_R17_at_most_minus_10": cr["overall"] <= -10,
            "C17_minus_R17_both_cases_negative": negative_cases(cr),
            "C17_minus_O17_at_most_minus_10": co["overall"] <= -10,
            "C17_minus_O17_both_cases_negative": negative_cases(co),
            "C17_minus_B17_negative": cb["overall"] < 0,
            "C17_minus_S17_negative": cs["overall"] < 0,
            "C17_minus_G17_negative": cg["overall"] < 0,
        }
    else:
        conditions = {
            "C17_minus_D17_negative": cd["overall"] < 0,
            "C17_minus_CF17_negative": cf["overall"] < 0,
            "C17_minus_I17_negative": ci["overall"] < 0,
            "C17_minus_R17_negative": cr["overall"] < 0,
            "C17_minus_O17_negative": co["overall"] < 0,
            "C17_minus_S17_negative": cs["overall"] < 0,
            "C17_minus_G17_negative": cg["overall"] < 0,
            "C17_minus_D17_both_cases_negative": negative_cases(cd),
            "C17_minus_CF17_both_cases_negative": negative_cases(cf),
            "C17_minus_I17_both_cases_negative": negative_cases(ci),
            "C17_minus_D17_cluster_upper_negative": upper_negative(cd),
            "C17_minus_CF17_cluster_upper_negative": upper_negative(cf),
            "C17_minus_I17_cluster_upper_negative": upper_negative(ci),
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
    audit_dir = run.parent / ("same_tick_v17_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
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
