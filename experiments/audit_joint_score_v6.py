"""Independent replay/accounting audit of the prelocked v6 score pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION, RuleController, configs, evaluate_episode
from ht_pdm_fjsp.maintenance_request_policies import choose as rollout_choose
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.joint_score_v6 import ARMS, ACTIONS, pair_interaction_errors, parameter_count, teacher_targets
from maintenance_capacity_coordination.pilot import CASES, evaluate_policy, load_policy
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs
from experiments.joint_score_v6 import (
    DEV_SEEDS, HELDOUT_SEEDS, REPLICATION_DEV, REPLICATION_EVAL,
    REPLICATION_ROOTS, TRAIN_ROOTS, SMOKE_DEV, SMOKE_ROOT, TEACHER_EPISODES,
    EPOCHS, MILESTONES, BATCH, ANALYSIS_SEED, ROOT, PROTOCOL,
    _comparison, _selected_checkpoint, config_hashes, digest, evaluate_student,
    forecast_seed, load_student, read_jsonl, seed_registry_check, sha256,
    source_hashes, training_raw, write_json,
)
from experiments.threshold_control_v1 import evaluate_g


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_physical(actual, expected, context):
    omitted = {"selection_seconds", "episode_wall_seconds", "controller", "panel",
               "checkpoint", "arm", "root", "historical_train_seed",
               "forecast_transitions", "contextual_train_root", "checkpoint_sha256"}
    for key, value in expected.items():
        if key in omitted:
            continue
        require(key in actual, f"{context}: missing {key}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            require(abs(float(actual[key]) - float(value)) <= 1e-7,
                    f"{context}: {key} differs")
        else:
            require(actual[key] == value, f"{context}: {key} differs")


def audit_header(run, manifest):
    require(manifest["status"] == "COMPLETED", "run incomplete")
    require(manifest["protocol_sha256"] == sha256(PROTOCOL), "protocol hash")
    require(manifest["source_sha256"] == source_hashes(), "source hashes")
    require(manifest["config_sha256"] == config_hashes(), "config hashes")
    seed_registry_check()
    for rel, claimed in manifest.get("output_sha256", {}).items():
        require(sha256(run / rel) == claimed, f"raw output hash: {rel}")


def audit_teacher(dataset, case, root, episodes):
    record = json.loads((dataset / "manifest.json").read_text())
    require(record["status"] == "COMPLETED", "teacher incomplete")
    rows = read_jsonl(dataset / "states.jsonl")
    episode_rows = read_jsonl(dataset / "episodes.jsonl")
    require(len(rows) == episodes * HORIZON and len(episode_rows) == episodes,
            "teacher coverage")
    require(record["states_sha256"] == sha256(dataset / "states.jsonl")
            and record["episodes_sha256"] == sha256(dataset / "episodes.jsonl"),
            "teacher file hashes")
    env = CoordinationEnv(configs()[case])
    forecasts = legal_targets = 0
    for ep in tqdm(range(episodes), desc=f"Audit V6 teacher {case} {root}", leave=False):
        raw = training_raw(root, case, ep)
        env.reset(raw)
        for tick in range(HORIZON):
            row = rows[ep * HORIZON + tick]
            state = env.inner.state
            submissions, matching, diag = rollout_choose(
                env.config, state, tick, "threshold", "rollout",
                policy_horizon=HORIZON, forecast_seed=forecast_seed(case, raw, tick),
                scenarios=16, forecast_horizon=6)
            admitted = env.admitted(submissions)
            machine, worker = actor_inputs(env.config, admitted, tick)
            mask = env.matching_mask(submissions)
            target = teacher_targets(diag["root_scores"], mask)
            index = ACTIONS.index(matching)
            require((row["case"], row["root"], row["episode"], row["tick"], row["raw_seed"])
                    == (case, root, ep, tick, raw), "teacher row identity")
            require(row["physical_seed"] == role_seed(raw, VERSION + ":physical")
                    and row["forecast_seed"] == forecast_seed(case, raw, tick),
                    "teacher seed identities")
            require(row["state_sha256"] == hashlib.sha256(repr(state).encode()).hexdigest(),
                    "teacher state hash")
            require(row["submissions"] == list(submissions)
                    and row["matching"] == [list(edge) for edge in matching]
                    and row["matching_index"] == index and row["mask"] == mask.tolist(),
                    "teacher action/legality")
            require(np.array_equal(np.asarray(row["machine"], dtype=np.float32), machine)
                    and np.array_equal(np.asarray(row["worker"], dtype=np.float32), worker),
                    "teacher inputs")
            require(row["root_scores"] == json.loads(json.dumps(diag["root_scores"]))
                    and row["root_scores_sha256"] == digest(diag["root_scores"]),
                    "teacher all-root scores")
            require(np.array_equal(np.asarray(row["target"], dtype=np.float32), target),
                    "teacher relative target")
            require(row["forecast_transitions"] == diag["forecast_transitions"],
                    "teacher forecast count")
            forecasts += diag["forecast_transitions"]
            legal_targets += int(mask.sum())
            _, done = env.step(submissions, index)
        require(done, "teacher H completion")
        compare_physical(episode_rows[ep], {"case": case, "root": root,
            "episode": ep, "raw_seed": raw,
            "physical_seed": role_seed(raw, VERSION + ":physical"), **env.metrics()},
            "teacher H+tail episode")
    require(record["episodes_completed"] == episodes and record["rows"] == len(rows)
            and record["legal_targets"] == legal_targets
            and record["forecast_transitions"] == forecasts
            and record["H_transitions"] == episodes * HORIZON
            and record["tail_transitions"] == episodes * 16,
            "teacher manifest accounting")
    return forecasts


def audit_training(run, manifest):
    smoke = manifest["kind"] == "smoke"
    replication = manifest["kind"] == "replication_train"
    cases = CASES[:1] if smoke else CASES
    roots = (SMOKE_ROOT,) if smoke else (REPLICATION_ROOTS if replication else TRAIN_ROOTS)
    dev = SMOKE_DEV if smoke else (REPLICATION_DEV if replication else DEV_SEEDS)
    episodes = 2 if smoke else TEACHER_EPISODES
    epochs = 1 if smoke else EPOCHS
    milestones = (1,) if smoke else MILESTONES
    forecasts = dev_h = dev_tail = 0
    for case in cases:
        for root in roots:
            dataset = run / f"teacher_{case}_{root}"
            forecasts += audit_teacher(dataset, case, root, episodes)
            data_hash = sha256(dataset / "states.jsonl")
            legal_targets = sum(sum(row["mask"]) for row in read_jsonl(dataset / "states.jsonl"))
            for arm in ARMS:
                trial = run / f"student_{case}_{arm}_{root}"
                record = json.loads((trial / "manifest.json").read_text())
                require(record["status"] == "COMPLETED", "student trial incomplete")
                require(record["dataset_sha256"] == data_hash and record["examples"] == episodes * HORIZON
                        and record["legal_targets"] == legal_targets, "student dataset/budget")
                require(record["epochs_completed"] == epochs
                        and record["gradient_updates"] == epochs * int(np.ceil(record["examples"] / BATCH)),
                        "student update count")
                require([r["epoch"] for r in read_jsonl(trial / "epochs.jsonl")]
                        == list(range(1, epochs + 1)), "epoch coverage")
                dev_rows = read_jsonl(trial / "dev_episodes.jsonl")
                require(len(dev_rows) == len(milestones) * len(dev), "DEV coverage")
                means = []
                for epoch in milestones:
                    checkpoint = trial / f"checkpoint_epoch_{epoch}.pt"
                    actor, payload = load_student(checkpoint)
                    require((payload["case"], payload["arm"], payload["root"], payload["epoch"])
                            == (case, arm, root, epoch)
                            and payload["dataset_sha256"] == data_hash,
                            "checkpoint identity")
                    require(record["actor_parameters"] == parameter_count(actor),
                            "active model size")
                    selected = [r for r in dev_rows if r["checkpoint"] == checkpoint.name]
                    require(len(selected) == len(dev)
                            and {r["seed"] for r in selected} == set(dev), "DEV seed coverage")
                    replay = evaluate_student(actor, case, dev, arm=arm, root=root,
                                              checkpoint=checkpoint.name)
                    for actual, expected in zip(selected, replay):
                        compare_physical(actual, expected, "DEV student")
                    means.append((epoch, float(np.mean([r["objective"] for r in selected])),
                                  checkpoint.name, sha256(checkpoint)))
                best = min(means, key=lambda item: (item[1], item[0]))
                selection = json.loads((trial / "selection.json").read_text())
                require(selection["epoch"] == best[0]
                        and abs(selection["mean_H_tail_cost"] - best[1]) < 1e-8
                        and selection["checkpoint"] == best[2]
                        and selection["checkpoint_sha256"] == best[3]
                        and record["selected_checkpoint"] == best[2]
                        and record["selected_checkpoint_sha256"] == best[3],
                        "DEV selection")
                require(record["dev_episodes"] == len(dev_rows)
                        and record["H_transitions"] == len(dev_rows) * HORIZON
                        and record["tail_transitions"] == len(dev_rows) * 16,
                        "student transition budget")
                dev_h += record["H_transitions"]
                dev_tail += record["tail_transitions"]
    require(manifest["teacher_forecast_transitions"] == forecasts
            and manifest["teacher_H_transitions"] == len(cases) * len(roots) * episodes * HORIZON
            and manifest["teacher_tail_transitions"] == len(cases) * len(roots) * episodes * 16
            and manifest["student_H_transitions"] == dev_h
            and manifest["student_tail_transitions"] == dev_tail,
            "training run totals")
    return {"teacher_forecast_transitions": forecasts,
            "teacher_H_transitions": manifest["teacher_H_transitions"],
            "student_DEV_episodes": dev_h // HORIZON}


def audit_interactions(run, training_run, seeds, roots, control):
    rows = read_jsonl(run / "heldout_interactions.jsonl")
    require(len(rows) == len(CASES) * len(seeds) * HORIZON,
            "F2-visited interaction coverage")
    policies = {(case, root, arm): _selected_checkpoint(training_run, case, arm, root)[0]
                for case in CASES for root in roots for arm in ARMS}
    index = 0
    forecasts = 0
    for case in CASES:
        cfg = configs()[case]
        env = CoordinationEnv(cfg)
        for seed in tqdm(seeds, desc=f"Audit V6 interaction {case}"):
            env.reset(seed)
            for tick in range(HORIZON):
                actual = rows[index]
                index += 1
                state = env.inner.state
                submissions, matching, diag = rollout_choose(
                    cfg, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                    forecast_seed=forecast_seed(case, seed, tick),
                    scenarios=16, forecast_horizon=6)
                admitted = env.admitted(submissions)
                machine, worker = actor_inputs(cfg, admitted, tick)
                mask = env.matching_mask(submissions)
                pairs = sum(len(r["matching"]) == 2 for r in diag["root_scores"])
                require((actual["case"], actual["seed"], actual["tick"])
                        == (case, seed, tick), "interaction state identity")
                require(actual["physical_seed"] == role_seed(seed, VERSION + ":physical")
                        and actual["forecast_seed"] == forecast_seed(case, seed, tick),
                        "interaction seed roles")
                require(actual["state_sha256"] == hashlib.sha256(repr(state).encode()).hexdigest()
                        and actual["root_scores_sha256"] == digest(diag["root_scores"]),
                        "interaction state/root score hashes")
                require(actual["legal_root_count"] == int(mask.sum())
                        and actual["pair_root_count"] == pairs
                        and actual["forecast_transitions"] == diag["forecast_transitions"],
                        "interaction root/forecast accounting")
                for root in roots:
                    for arm in ARMS:
                        with torch.no_grad():
                            predicted = policies[case, root, arm](torch.from_numpy(machine),
                                                                   torch.from_numpy(worker))
                        expected_errors = pair_interaction_errors(predicted, diag["root_scores"])
                        saved_errors = actual["errors"][str(root)][arm]
                        require(len(saved_errors) == len(expected_errors)
                                and np.allclose(saved_errors, expected_errors, atol=1e-7, rtol=0),
                                "interaction score error replay")
                forecasts += diag["forecast_transitions"]
                env.step(submissions, ACTIONS.index(matching))
            actual_control = control[case, "F_rollout", None, seed]
            replay_metrics = env.metrics()
            require(set(replay_metrics) - set(actual_control)
                    == {"legal_service_waits", "request_creation_free_compatible_mean"},
                    "legacy F2 diagnostic schema")
            compare_physical(actual_control,
                             {"case": case, "seed": seed,
                              "physical_seed": role_seed(seed, VERSION + ":physical"),
                              **{key: value for key, value in replay_metrics.items()
                                 if key in actual_control}}, "interaction F2 physical replay")
    return rows, forecasts


def audit_evaluation(run, manifest):
    replication = manifest["kind"] == "replication_eval"
    seeds = REPLICATION_EVAL if replication else HELDOUT_SEEDS
    roots = REPLICATION_ROOTS if replication else TRAIN_ROOTS
    training_run = Path(manifest["training_run"])
    require(manifest["training_manifest_sha256"] == sha256(training_run / "manifest.json"),
            "training manifest hash")
    training = json.loads((training_run / "manifest.json").read_text())
    require(training["status"] == "COMPLETED" and training.get("audit_status") == "PASS"
            and training["protocol_sha256"] == manifest["protocol_sha256"]
            and training["source_sha256"] == manifest["source_sha256"]
            and training["config_sha256"] == manifest["config_sha256"],
            "training identity/audit")
    students = read_jsonl(run / "heldout_students.jsonl")
    controls = read_jsonl(run / "heldout_controls.jsonl")
    contextual = read_jsonl(run / "heldout_contextual.jsonl")
    require(len(students) == len(CASES) * len(roots) * len(ARMS) * len(seeds),
            "student heldout coverage")
    require(len(controls) == len(CASES) * len(seeds) * 7, "A/F control coverage")
    require(len(contextual) == len(CASES) * len(seeds) * 2 * 3, "G/D context coverage")
    learned = {(r["case"], r["arm"], r["root"], r["seed"]): r for r in students}
    control = {(r["case"], r["arm"], r.get("historical_train_seed"), r["seed"]): r
               for r in controls}
    context = {(r["case"], r["arm"], r["contextual_train_root"], r["seed"]): r
               for r in contextual}
    require(len(learned) == len(students) and len(control) == len(controls)
            and len(context) == len(contextual), "duplicate heldout keys")
    for case in CASES:
        for root in roots:
            for arm in ARMS:
                actor, selection = _selected_checkpoint(training_run, case, arm, root)
                replay = evaluate_student(actor, case, seeds, arm=arm, root=root,
                                          checkpoint=selection["checkpoint"])
                for row in replay:
                    actual = learned[case, arm, root, row["seed"]]
                    compare_physical(actual, row, "heldout student")
                    require(actual["physical_seed"] == role_seed(row["seed"], VERSION + ":physical"),
                            "heldout student physical seed")
        cfg = configs()[case]
        for seed in tqdm(seeds, desc=f"Audit V6 A/F {case}"):
            for name in ("risk_matching", "rollout"):
                replay = evaluate_episode(cfg, case, seed, RuleController(name))
                actual = control[case, "F_" + name, None, seed]
                compare_physical(actual, replay, "heldout rule")
                require(actual["forecast_transitions"] == replay["forecast_transitions"],
                        "heldout F2 forecast count")
            for historical_seed in (72600000, 72600001, 72600002, 72600003, 72600004):
                checkpoint_dir = ROOT / f"baselines/v14/checkpoints/count_calibrated/{historical_seed}"
                candidates = list(checkpoint_dir.glob("*.pt"))
                require(len(candidates) == 1, "V14 checkpoint identity")
                require(sha256(candidates[0]) ==
                        manifest["v14_checkpoint_sha256"][str(historical_seed)],
                        "V14 checkpoint hash")
                replay = evaluate_episode(cfg, case, seed,
                                          V14Controller.from_checkpoint(candidates[0]))
                compare_physical(control[case, "A", historical_seed, seed], replay,
                                 "heldout V14")
        for root in (73110000, 73110001, 73110002):
            for arm, train_dir in (("G", ROOT / "artifacts/threshold_control_v1_train_20261022T063058659512Z"),
                                   ("D", ROOT / "artifacts/pilot_v1_train_20261022T061152746250Z")):
                trial = train_dir / f"{case}_{arm}_{root}"
                selection = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selection["checkpoint"]
                claimed = manifest["contextual_checkpoint_sha256"][f"{case}:{arm}:{root}"]
                require(sha256(checkpoint) == claimed, "contextual checkpoint hash")
                policy = load_policy(checkpoint)
                label = f"{root}:{selection['transition']}"
                replay = (evaluate_g(policy, case, seeds, None, label=label) if arm == "G"
                          else evaluate_policy(policy, case, seeds, None, label=label))
                for row in replay:
                    actual = context[case, arm, root, row["seed"]]
                    compare_physical(actual, row, "contextual G/D")
                    require(actual["checkpoint_sha256"] == claimed
                            and actual["physical_seed"] == role_seed(row["seed"], VERSION + ":physical"),
                            "contextual seed/checkpoint")
    require(manifest["student_H_transitions"] == len(students) * HORIZON
            and manifest["student_tail_transitions"] == len(students) * 16
            and manifest["control_H_transitions"] == (len(controls) + len(contextual)) * HORIZON
            and manifest["control_tail_transitions"] == (len(controls) + len(contextual)) * 16
            and manifest["control_forecast_transitions"] == sum(r["forecast_transitions"] for r in controls),
            "heldout transition totals")
    interaction_rows, interaction_forecasts = audit_interactions(
        run, training_run, seeds, roots, control)
    require(manifest["interaction_forecast_transitions"] == interaction_forecasts,
            "interaction forecast count")
    analysis = json.loads((run / "analysis.json").read_text())
    for other in ("A6", "G6"):
        require(analysis["comparisons_J6_minus"][other] == _comparison(learned, other, seeds, roots),
                "paired physical-cluster analysis")
    for case in CASES:
        for arm in ARMS:
            values = [v for row in interaction_rows if row["case"] == case
                      for root in roots for v in row["errors"][str(root)][arm]]
            reported = analysis["pair_interaction_absolute_errors"][case][arm]
            per_seed = [(sum(v for row in interaction_rows if row["case"] == case
                             and row["seed"] == seed for root in roots
                             for v in row["errors"][str(root)][arm]),
                         sum(len(row["errors"][str(root)][arm]) for row in interaction_rows
                             if row["case"] == case and row["seed"] == seed
                             for root in roots)) for seed in seeds]
            rng = np.random.default_rng(ANALYSIS_SEED)
            sampled = rng.integers(0, len(seeds), size=(5000, len(seeds)))
            pair_array = np.asarray(per_seed, dtype=float)
            numerator = pair_array[sampled, 0].sum(axis=1)
            denominator = pair_array[sampled, 1].sum(axis=1)
            interval = np.quantile(numerator[denominator > 0] / denominator[denominator > 0],
                                   (.025, .975)).tolist()
            require(reported["pair_roots"] == len(values)
                    and abs(reported["mean_absolute_cost_error"] - float(np.mean(values))) < 1e-8
                    and reported["physical_episode_cluster_ci95"] == interval,
                    "pair interaction error aggregation")
    secondary_fields = ("H_availability", "H_failures", "queued_pm_ticks",
                        "queued_cm_ticks", "H_pending", "H_pending_pm", "H_pending_cm",
                        "end_pending", "service_starts", "legal_service_waits")
    for case in CASES:
        for arm in (*ARMS, "A", "F_risk_matching", "F_rollout", "G", "D"):
            source = students if arm in ARMS else (contextual if arm in ("G", "D") else controls)
            selected = [r for r in source if r["case"] == case and r["arm"] == arm]
            for field in secondary_fields:
                reported = analysis["secondary_mean_by_case_arm"][case][arm][field]
                if all(field in r for r in selected):
                    require(abs(reported - float(np.mean([r[field] for r in selected]))) < 1e-8,
                            "secondary aggregation")
                else:
                    require(reported is None, "unavailable secondary aggregation")
    reciprocal = analysis["pair_interaction_absolute_errors"][CASES[0]]
    a_error = reciprocal["A6"]["mean_absolute_cost_error"]
    ratio = reciprocal["J6"]["mean_absolute_cost_error"] / a_error if a_error > 0 else None
    require(analysis["reciprocal_J6_over_A6_pair_error_ratio"] == ratio,
            "pair error ratio")
    conditions = {}
    for other in ("A6", "G6"):
        c = analysis["comparisons_J6_minus"][other]
        conditions[f"J6_minus_{other}_overall_le_-10"] = c["overall"] <= -10
        conditions[f"J6_minus_{other}_reciprocal_le_-20"] = c["by_case"][CASES[0]] <= -20
        conditions[f"J6_minus_{other}_specialist_le_5"] = c["by_case"][CASES[1]] <= 5
        conditions[f"J6_minus_{other}_two_roots_negative"] = sum(
            v < 0 for v in c["by_training_root"].values()) >= 2
        conditions[f"J6_minus_{other}_ci_upper_lt_0"] = c["physical_cluster_ci95"][1] < 0
    conditions["reciprocal_pair_mae_ratio_le_0.8"] = ratio is not None and ratio <= .8
    require(analysis["gate_conditions"] == conditions
            and analysis["go"] == all(conditions.values())
            and analysis["protocol_sha256"] == sha256(PROTOCOL)
            and analysis["analysis_seed"] == ANALYSIS_SEED,
            "prelocked gate arithmetic")
    decision = json.loads((run / "replication_go_decision.json").read_text())
    require(decision["go"] == analysis["go"]
            and decision["gate_conditions"] == conditions
            and decision["protocol_sha256"] == sha256(PROTOCOL),
            "replication decision")
    return {"student_episodes": len(students),
            "F_V14_control_episodes": len(controls),
            "G_D_contextual_episodes": len(contextual),
            "interaction_states": len(interaction_rows),
            "interaction_forecast_transitions": interaction_forecasts,
            "gate_go": decision["go"]}


def audit(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    audit_header(run, manifest)
    started = time.monotonic()
    if manifest["kind"] in ("smoke", "train", "replication_train"):
        detail = audit_training(run, manifest)
    elif manifest["kind"] in ("heldout", "replication_eval"):
        detail = audit_evaluation(run, manifest)
    else:
        raise RuntimeError("unsupported v6 artifact kind")
    result = {"status": "PASS", "kind": manifest["kind"],
              "protocol_sha256": sha256(PROTOCOL), "source_sha256": source_hashes(),
              "elapsed_seconds": time.monotonic() - started, **detail}
    write_json(run / "audit.json", result)
    manifest["audit_status"] = "PASS"
    manifest["output_sha256"] = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                                 if p.is_file() and p.name != "manifest.json"}
    write_json(run / "manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.run), sort_keys=True))


if __name__ == "__main__":
    main()
