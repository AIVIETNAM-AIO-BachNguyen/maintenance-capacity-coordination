"""Independent replay/accounting audit for the prelocked v4 pilot artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION, RuleController, configs, evaluate_episode
from ht_pdm_fjsp.maintenance_request_policies import choose as rollout_choose
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.pilot import CASES
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs
from experiments.forecast_distillation_v4 import (
    ARMS, BACKING_ARM, DEV_SEEDS, HELDOUT_SEEDS, REPLICATION_DEV, REPLICATION_EVAL,
    REPLICATION_ROOTS, TRAIN_ROOTS, TRAIN_TIMEOUT, TEACHER_EPISODES,
    _paired_comparison, config_hashes, digest, evaluate_student, forecast_seed,
    load_student, read_jsonl, seed_registry_check, sha256, source_hashes,
    training_raw, write_json, PROTOCOL,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_physical(actual, expected, context):
    """Compare physical/accounting fields, excluding timing and annotation."""
    omitted = {"selection_seconds", "episode_wall_seconds", "controller", "panel",
               "checkpoint", "arm", "root", "historical_train_seed", "forecast_transitions"}
    for key, value in expected.items():
        if key in omitted:
            continue
        require(key in actual, f"{context}: missing {key}")
        other = actual[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            require(abs(float(other) - float(value)) <= 1e-7, f"{context}: {key} differs")
        else:
            require(other == value, f"{context}: {key} differs")


def audit_header(run, manifest):
    require(manifest["status"] == "COMPLETED", "run is not complete")
    require(manifest["protocol_sha256"] == sha256(PROTOCOL), "protocol hash mismatch")
    require(manifest["source_sha256"] == source_hashes(), "source hash mismatch")
    require(manifest["config_sha256"] == config_hashes(), "config hash mismatch")
    seed_registry_check()
    for rel, expected in manifest.get("output_sha256", {}).items():
        path = run / rel
        require(path.is_file() and sha256(path) == expected, f"output hash mismatch {rel}")


def audit_teacher(dataset, case, root, episodes):
    record = json.loads((dataset / "manifest.json").read_text())
    require(record["status"] == "COMPLETED", "teacher incomplete")
    rows = read_jsonl(dataset / "states.jsonl")
    episode_rows = read_jsonl(dataset / "episodes.jsonl")
    require(len(rows) == episodes * HORIZON and len(episode_rows) == episodes,
            "teacher coverage")
    require(record["states_sha256"] == sha256(dataset / "states.jsonl"), "teacher states hash")
    require(record["episodes_sha256"] == sha256(dataset / "episodes.jsonl"), "teacher episodes hash")
    env = CoordinationEnv(configs()[case])
    forecasts = 0
    for ep in tqdm(range(episodes), desc=f"Audit teacher {case} {root}", leave=False):
        raw = training_raw(root, case, ep)
        env.reset(raw)
        for tick in range(HORIZON):
            row = rows[ep * HORIZON + tick]
            state = env.inner.state
            fseed = forecast_seed(case, raw, tick)
            submissions, matching, diag = rollout_choose(
                env.config, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                forecast_seed=fseed, scenarios=16, forecast_horizon=6)
            admitted = env.admitted(submissions)
            machine, worker = actor_inputs(env.config, admitted, tick)
            mask = env.matching_mask(submissions)
            index = env.actions.index(matching)
            require((row["case"], row["root"], row["episode"], row["tick"], row["raw_seed"])
                    == (case, root, ep, tick, raw), "teacher row identity")
            require(row["physical_seed"] == role_seed(raw, VERSION + ":physical"),
                    "teacher physical seed")
            require(row["forecast_seed"] == fseed, "teacher forecast seed")
            require(row["matching_index"] == index and row["matching"] == [list(a) for a in matching],
                    "teacher selected matching")
            require(row["submissions"] == list(submissions) and row["mask"] == mask.tolist(),
                    "teacher submissions/mask")
            require(np.array_equal(np.asarray(row["machine"], dtype=np.float32), machine)
                    and np.array_equal(np.asarray(row["worker"], dtype=np.float32), worker),
                    "teacher actor observations")
            require(row["root_scores"] == json.loads(json.dumps(diag["root_scores"])),
                    "teacher root scores")
            require(row["scores_sha256"] == digest(diag["root_scores"]), "teacher score hash")
            require(row["forecast_transitions"] == diag["forecast_transitions"],
                    "teacher forecast count")
            require(row["state_sha256"] == hashlib.sha256(repr(state).encode()).hexdigest(),
                    "teacher state hash")
            forecasts += diag["forecast_transitions"]
            _, done = env.step(submissions, index)
        require(done, "teacher episode end")
        compare_physical(episode_rows[ep], {"case": case, "root": root, "episode": ep,
                                            "raw_seed": raw,
                                            "physical_seed": role_seed(raw, VERSION + ":physical"),
                                            **env.metrics()}, "teacher episode")
    require(record["rows"] == len(rows) and record["episodes_completed"] == episodes,
            "teacher manifest coverage")
    require(record["forecast_transitions"] == forecasts
            and record["H_transitions"] == episodes * HORIZON
            and record["tail_transitions"] == episodes * 16, "teacher manifest transitions")
    return forecasts


def audit_training(run, manifest):
    smoke = manifest["kind"] == "smoke"
    replication = manifest["kind"] == "replication_train"
    cases = CASES[:1] if smoke else CASES
    roots = (73100099,) if smoke else (REPLICATION_ROOTS if replication else TRAIN_ROOTS)
    dev_seeds = (73100098,) if smoke else (REPLICATION_DEV if replication else DEV_SEEDS)
    episodes = 2 if smoke else TEACHER_EPISODES
    expected_epochs = 1 if smoke else 32
    milestones = (1,) if smoke else (8, 16, 32)
    forecast_total = student_h = student_tail = 0
    for case in cases:
        for root in roots:
            dataset = run / f"teacher_{case}_{root}"
            forecast_total += audit_teacher(dataset, case, root, episodes)
            dataset_hash = sha256(dataset / "states.jsonl")
            for arm in ARMS:
                trial = run / f"student_{case}_{arm}_{root}"
                record = json.loads((trial / "manifest.json").read_text())
                require(record["status"] == "COMPLETED", "student trial incomplete")
                require(record["dataset_sha256"] == dataset_hash, "student dataset mismatch")
                require(record["examples"] == episodes * HORIZON, "student example coverage")
                require(record["epochs_completed"] == expected_epochs
                        and record["gradient_updates"] == expected_epochs * int(np.ceil(record["examples"] / 128)),
                        "student training budget")
                epochs = read_jsonl(trial / "epochs.jsonl")
                require([row["epoch"] for row in epochs] == list(range(1, expected_epochs + 1)),
                        "student epoch rows")
                dev_rows = read_jsonl(trial / "dev_episodes.jsonl")
                require(len(dev_rows) == len(milestones) * len(dev_seeds), "DEV coverage")
                means = []
                for epoch in milestones:
                    checkpoint = trial / f"checkpoint_epoch_{epoch}.pt"
                    policy, payload = load_student(checkpoint)
                    require((payload["case"], payload["arm"], payload["root"], payload["epoch"])
                            == (case, arm, root, epoch), "checkpoint identity")
                    require(payload["dataset_sha256"] == dataset_hash, "checkpoint dataset identity")
                    selected_rows = [row for row in dev_rows if row["checkpoint"] == checkpoint.name]
                    require(len(selected_rows) == len(dev_seeds)
                            and {row["seed"] for row in selected_rows} == set(dev_seeds),
                            "DEV seed coverage")
                    replay = evaluate_student(policy, case, dev_seeds,
                                              arm=arm, root=root, checkpoint=checkpoint.name)
                    for expected, actual in zip(selected_rows, replay):
                        compare_physical(expected, actual, "DEV episode")
                    means.append((epoch, float(np.mean([row["objective"] for row in dev_rows
                                                       if row["checkpoint"] == checkpoint.name])),
                                  checkpoint.name, sha256(checkpoint)))
                selection = json.loads((trial / "selection.json").read_text())
                best = min(means, key=lambda item: (item[1], item[0]))
                require(selection["epoch"] == best[0] and abs(selection["mean_H_tail_cost"] - best[1]) < 1e-8
                        and selection["checkpoint"] == best[2]
                        and selection["checkpoint_sha256"] == best[3], "DEV selection")
                require(record["selected_checkpoint"] == best[2]
                        and record["selected_checkpoint_sha256"] == best[3], "trial selection manifest")
                require(record["H_transitions"] == len(dev_rows) * HORIZON
                        and record["tail_transitions"] == len(dev_rows) * 16, "trial DEV transitions")
                student_h += record["H_transitions"]
                student_tail += record["tail_transitions"]
    require(manifest["teacher_forecast_transitions"] == forecast_total, "run teacher forecast count")
    require(manifest["teacher_H_transitions"] == len(cases) * len(roots) * episodes * HORIZON,
            "run teacher H count")
    require(manifest["teacher_tail_transitions"] == len(cases) * len(roots) * episodes * 16,
            "run teacher tail count")
    require(manifest["student_H_transitions"] == student_h
            and manifest["student_tail_transitions"] == student_tail, "run DEV transition count")
    return {"teacher_forecast_transitions": forecast_total, "student_H_transitions": student_h}


def teacher_agreement_diagnostic(run, training_run, seeds, roots):
    """Read-only same-state student choices along independent F2 control replays."""
    policies = {}
    for case in CASES:
        for root in roots:
            for arm in ARMS:
                trial = training_run / f"student_{case}_{arm}_{root}"
                selected = json.loads((trial / "selection.json").read_text())
                policies[(case, root, arm)] = load_student(trial / selected["checkpoint"])[0]
    control_rows = read_jsonl(run / "heldout_controls.jsonl")
    control = {(r["case"], r["seed"]): r for r in control_rows if r["arm"] == "F_rollout"}
    stats = {(case, root, arm): {"matches": 0, "decisions": 0, "forecast_regret_sum": 0.0}
             for case in CASES for root in roots for arm in ARMS}
    forecasts = 0
    for case in CASES:
        cfg = configs()[case]
        env = CoordinationEnv(cfg)
        for seed in tqdm(seeds, desc=f"Agreement {case}", leave=False):
            env.reset(seed)
            for tick in range(HORIZON):
                submissions, matching, diag = rollout_choose(
                    cfg, env.inner.state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                    forecast_seed=forecast_seed(case, seed, tick), scenarios=16, forecast_horizon=6)
                forecasts += diag["forecast_transitions"]
                admitted = env.admitted(submissions)
                machine, worker = actor_inputs(cfg, admitted, tick)
                mask = env.matching_mask(submissions)
                teacher_index = env.actions.index(matching)
                scores = {tuple(tuple(edge) for edge in row["matching"]): row["expected_cost"]
                          for row in diag["root_scores"]}
                for root in roots:
                    for arm in ARMS:
                        policy = policies[(case, root, arm)]
                        with torch.no_grad():
                            selected_index = int(policy.sample(
                                torch.from_numpy(machine), torch.from_numpy(worker),
                                torch.from_numpy(mask), deterministic=True))
                        require(mask[selected_index], "agreement student illegal action")
                        entry = stats[(case, root, arm)]
                        entry["matches"] += selected_index == teacher_index
                        entry["decisions"] += 1
                        if scores:
                            entry["forecast_regret_sum"] += (
                                scores[env.actions[selected_index]] - scores[matching])
                _, done = env.step(submissions, teacher_index)
            require(done, "agreement F2 episode end")
            compare_physical(control[(case, seed)],
                             {k: v for k, v in env.metrics().items() if k in control[(case, seed)]},
                             "agreement F2 episode")
    rows = []
    for (case, root, arm), item in stats.items():
        rows.append({"case": case, "root": root, "arm": arm, **item,
                     "agreement_rate": item["matches"] / item["decisions"],
                     "mean_finite_forecast_regret": item["forecast_regret_sum"] / item["decisions"]})
    result = {"rows": rows, "forecast_transitions": forecasts,
              "scope": "F2-visited evaluation states; finite teacher forecast scores"}
    write_json(run / "teacher_agreement.json", result)
    return result


def audit_evaluation(run, manifest):
    replication = manifest["kind"] == "replication_eval"
    seeds = REPLICATION_EVAL if replication else HELDOUT_SEEDS
    roots = REPLICATION_ROOTS if replication else TRAIN_ROOTS
    training_run = Path(manifest["training_run"])
    require(manifest["training_manifest_sha256"] == sha256(training_run / "manifest.json"),
            "training manifest hash")
    training_manifest = json.loads((training_run / "manifest.json").read_text())
    require(training_manifest["status"] == "COMPLETED"
            and training_manifest.get("audit_status") == "PASS"
            and training_manifest["protocol_sha256"] == manifest["protocol_sha256"]
            and training_manifest["source_sha256"] == manifest["source_sha256"]
            and training_manifest["config_sha256"] == manifest["config_sha256"],
            "training run identity/audit")
    learned_rows = read_jsonl(run / "heldout_students.jsonl")
    control_rows = read_jsonl(run / "heldout_controls.jsonl")
    require(len(learned_rows) == len(CASES) * len(roots) * len(ARMS) * len(seeds),
            "held-out student coverage")
    require(len(control_rows) == len(CASES) * len(seeds) * 7, "held-out control coverage")
    learned = {(row["case"], row["arm"], row["root"], row["seed"]): row for row in learned_rows}
    controls = {(row["case"], row["arm"], row.get("historical_train_seed"), row["seed"]): row
                for row in control_rows}
    require(len(learned) == len(learned_rows) and len(controls) == len(control_rows),
            "duplicate evaluation row")
    for case in CASES:
        for root in roots:
            for arm in ARMS:
                trial = training_run / f"student_{case}_{arm}_{root}"
                selection = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selection["checkpoint"]
                require(sha256(checkpoint) == selection["checkpoint_sha256"],
                        "evaluation checkpoint hash")
                policy, payload = load_student(checkpoint)
                require((payload["case"], payload["arm"], payload["root"]) == (case, arm, root),
                        "evaluation checkpoint identity")
                replay = evaluate_student(policy, case, seeds,
                                          arm=arm, root=root, checkpoint=selection["checkpoint"])
                for row in replay:
                    actual = learned[(case, arm, root, row["seed"])]
                    compare_physical(actual, row, "held-out student")
                    require(actual["physical_seed"] == role_seed(row["seed"], VERSION + ":physical"),
                            "held-out student physical seed")
        cfg = configs()[case]
        for seed in tqdm(seeds, desc=f"Audit controls {case}"):
            for name in ("risk_matching", "rollout"):
                replay = evaluate_episode(cfg, case, seed, RuleController(name))
                actual = controls[(case, "F_" + name, None, seed)]
                compare_physical(actual, replay, "held-out rule")
                require(actual["forecast_transitions"] == replay["forecast_transitions"],
                        "held-out control forecast count")
            for historical_seed in (72600000, 72600001, 72600002, 72600003, 72600004):
                from maintenance_capacity_coordination.pilot import ROOT
                checkpoint_dir = ROOT / f"baselines/v14/checkpoints/count_calibrated/{historical_seed}"
                checkpoint = next(checkpoint_dir.glob("*.pt"))
                replay = evaluate_episode(cfg, case, seed, V14Controller.from_checkpoint(checkpoint))
                actual = controls[(case, "A", historical_seed, seed)]
                compare_physical(actual, replay, "held-out V14")
    require(manifest["student_H_transitions"] == len(learned_rows) * HORIZON
            and manifest["student_tail_transitions"] == len(learned_rows) * 16,
            "evaluation student transitions")
    require(manifest["control_H_transitions"] == len(control_rows) * HORIZON
            and manifest["control_tail_transitions"] == len(control_rows) * 16,
            "evaluation control transitions")
    require(manifest["control_forecast_transitions"] == sum(r["forecast_transitions"] for r in control_rows),
            "evaluation forecast total")
    analysis = json.loads((run / "analysis.json").read_text())
    learned_keyed = {(r["case"], r["arm"], r["root"], r["seed"]): r for r in learned_rows}
    controls_keyed = {(r["case"], r["arm"], r["seed"]): r for r in control_rows
                      if r["arm"] in ("F_rollout", "F_risk_matching")}
    for other in ("G4", "R4", "F2"):
        recomputed = _paired_comparison(learned_keyed, controls_keyed, other, seeds, roots)
        require(recomputed == analysis["comparisons_S4_minus"][other], "analysis mismatch")
    agreement = teacher_agreement_diagnostic(run, training_run, seeds, roots)
    manifest["agreement_forecast_transitions"] = agreement["forecast_transitions"]
    gate = json.loads((run / "replication_go_decision.json").read_text())
    require(gate == analysis["gate"] and gate["protocol_sha256"] == sha256(PROTOCOL),
            "gate decision mismatch")
    sg, sr, sf = (analysis["comparisons_S4_minus"][arm] for arm in ("G4", "R4", "F2"))
    recomputed_speed = (np.mean([r["selection_seconds"] for r in learned_rows if r["arm"] == "S4"])
                        / np.mean([r["selection_seconds"] for r in control_rows
                                   if r["arm"] == "F_rollout"]))
    require(abs(recomputed_speed - analysis["selection_time_ratio_S4_over_F2"]) < 1e-10,
            "selection time ratio")
    independently_go = (sg["overall"] <= -30 and all(v < 0 for v in sg["by_case"].values())
                        and sum(v < 0 for v in sg["by_training_root"].values()) >= 2
                        and sg["physical_cluster_ci95"][1] < 0
                        and sr["overall"] <= -10 and all(v < 0 for v in sr["by_case"].values())
                        and sum(v < 0 for v in sr["by_training_root"].values()) >= 2
                        and sr["physical_cluster_ci95"][1] < 0
                        and sf["overall"] <= 20 and all(v <= 30 for v in sf["by_case"].values())
                        and recomputed_speed <= .2)
    require(gate["go"] == independently_go, "gate calculation")
    return {"student_episodes": len(learned_rows), "control_episodes": len(control_rows),
            "gate_go": gate["go"],
            "agreement_forecast_transitions": agreement["forecast_transitions"]}


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
        raise RuntimeError("unsupported v4 artifact kind")
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
