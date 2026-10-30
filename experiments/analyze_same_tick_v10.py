"""Post-pilot descriptive mechanism check; never changes the locked v10 gate.

On each audited C10-visited held-out state, compare the selected C10 checkpoint
with its own identical weights but frozen within-tick context. Only the C10
matching is physically executed. This is a state-level sensitivity analysis,
not an economic counterfactual rollout or independent training replication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.same_tick_v10_config import configs
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v8 import EDGES, STOP, SequencePolicy, choose
from experiments import same_tick_pilot_v10 as pilot

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/same_tick_v10_train_20261022T173636025619Z"
HELDOUT = ROOT / "artifacts/same_tick_v10_heldout_20261022T174659511017Z"


def root_graph_profile(edge_mask):
    """Count residual choices for every legal first edge, independent of policy score."""
    legal = [edge for edge, allowed in enumerate(edge_mask) if allowed]
    with_second = with_multiple_second = 0
    for first in legal:
        m0, w0 = EDGES[first]
        residual = sum(EDGES[next_edge][0] != m0 and EDGES[next_edge][1] != w0
                       for next_edge in legal)
        with_second += residual >= 1
        with_multiple_second += residual >= 2
    return len(legal), with_second, with_multiple_second


def second_step_tv(policy, decision):
    first = decision["tokens"][0] if decision["tokens"] else None
    if first is None or first == STOP:
        return None
    machine = torch.from_numpy(decision["machine_obs"])[None]
    worker = torch.from_numpy(decision["worker_obs"])[None]
    initial = torch.from_numpy(decision["edge_mask"])[None]
    used_m = torch.zeros((1, 4), dtype=torch.bool)
    used_w = torch.zeros((1, 2), dtype=torch.bool)
    selected = torch.zeros((1, STOP), dtype=torch.bool)
    with torch.no_grad():
        encoded = policy.actor.encode(machine, worker)
        _, _, root_context = policy.actor.logits(encoded, initial, used_m, used_w, selected)
        m, j = EDGES[first]
        used_m[0, m], used_w[0, j], selected[0, first] = True, True, True
        updated, feasible, _ = policy.actor.logits(encoded, initial, used_m, used_w, selected)
        if not bool(feasible.any()):
            return None
        frozen, _, _ = policy.actor.logits(encoded, initial, used_m, used_w, selected,
                                           frozen_context=root_context)
        valid = torch.cat((feasible[0], torch.ones(1, dtype=torch.bool)))
        p_updated = torch.softmax(updated[0, valid], 0)
        p_frozen = torch.softmax(frozen[0, valid], 0)
        tv = float(.5 * (p_updated - p_frozen).abs().sum())
        score_delta = updated[0, valid] - frozen[0, valid]
        max_score_delta = float(score_delta.abs().max())
        updated_top = torch.topk(updated[0, valid], 2).values
        updated_margin = float(updated_top[0] - updated_top[1])
        edge_count = int(feasible.sum())
        edge_rank_changed = (edge_count > 1 and not torch.equal(
            torch.argsort(updated[0, :STOP][feasible[0]]), torch.argsort(frozen[0, :STOP][feasible[0]])))
    return dict(tv=tv, max_score_delta=max_score_delta,
                updated_margin=updated_margin, second_feasible_edges=edge_count,
                edge_rank_changed=bool(edge_rank_changed))


def main():
    torch.set_num_threads(1)
    training = json.loads((TRAIN / "manifest.json").read_text())
    heldout = json.loads((HELDOUT / "manifest.json").read_text())
    assert training["status"] == heldout["status"] == "COMPLETED"
    assert training["source_sha256"] == heldout["source_sha256"]
    assert json.loads((TRAIN / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((HELDOUT / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((HELDOUT / "replication_go_decision.json").read_text())["go"] is False
    for path, expected in training["source_sha256"].items():
        assert pilot.sha256(ROOT / path) == expected, path
    source_rows = [json.loads(x) for x in (HELDOUT / "heldout_episodes.jsonl").read_text().splitlines()]
    original = {(r["case"], r["arm"], int(r["checkpoint"].split(":")[0]), r["seed"]): r
                for r in source_rows}
    output = ROOT / "artifacts" / ("same_tick_v10_mechanism_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    manifest = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                "collection_commit": training["git_commit"],
                "collection_manifest_sha256": pilot.sha256(TRAIN / "manifest.json"),
                "heldout_manifest_sha256": pilot.sha256(HELDOUT / "manifest.json"),
                "heldout_audit_sha256": pilot.sha256(HELDOUT / "audit.json"),
                "analysis_source_sha256": pilot.sha256(Path(__file__)),
                "protocol_sha256": training["protocol_sha256"],
                "interpretation": "post-pilot descriptive; C10 checkpoint weights held fixed; no counterfactual physical execution or gate change"}
    pilot.write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    rows = []
    try:
        for case in pilot.CASES:
            for root in pilot.TRAIN_ROOTS:
                trial = TRAIN / f"{case}_C10_{root}"
                selected = json.loads((trial / "selection.json").read_text())
                checkpoint = trial / selected["checkpoint"]
                assert pilot.sha256(checkpoint) == selected["checkpoint_sha256"]
                active = pilot.load_policy(checkpoint)
                d_trial = TRAIN / f"{case}_D10_{root}"
                d_selected = json.loads((d_trial / "selection.json").read_text())
                d_checkpoint = d_trial / d_selected["checkpoint"]
                assert pilot.sha256(d_checkpoint) == d_selected["checkpoint_sha256"]
                trained_d = pilot.load_policy(d_checkpoint)
                frozen = SequencePolicy("D8")
                frozen.load_state_dict(active.state_dict(), strict=True)
                frozen.eval()
                env = CoordinationEnv(configs()[case])
                for seed in tqdm(pilot.HELDOUT_SEEDS, desc=f"v10 mechanism {case} {root}", leave=False):
                    env.reset(seed)
                    opportunities = changed_second = changed_matching = 0
                    trained_first_changed = trained_matching_changed = 0
                    tv_values, score_deltas, margins = [], [], []
                    one_edge = multiple_edges = edge_rank_changes = near_margin = 0
                    root_edges = root_with_second = root_with_multiple_second = 0
                    root_states_with_second = root_states_with_multiple_second = 0
                    for _ in range(48):
                        active_decision = choose(active, env, deterministic=True)
                        n_root, n_second, n_multiple = root_graph_profile(active_decision["edge_mask"])
                        root_edges += n_root
                        root_with_second += n_second
                        root_with_multiple_second += n_multiple
                        root_states_with_second += n_second > 0
                        root_states_with_multiple_second += n_multiple > 0
                        frozen_decision = choose(frozen, env, deterministic=True)
                        trained_d_decision = choose(trained_d, env, deterministic=True)
                        assert active_decision["tokens"][:1] == frozen_decision["tokens"][:1]
                        if active_decision["tokens"][:1] != trained_d_decision["tokens"][:1]:
                            trained_first_changed += 1
                        if active_decision["matching"] != trained_d_decision["matching"]:
                            trained_matching_changed += 1
                        sensitivity = second_step_tv(active, active_decision)
                        if sensitivity is not None:
                            opportunities += 1
                            tv_values.append(sensitivity["tv"])
                            score_deltas.append(sensitivity["max_score_delta"])
                            margins.append(sensitivity["updated_margin"])
                            one_edge += sensitivity["second_feasible_edges"] == 1
                            multiple_edges += sensitivity["second_feasible_edges"] > 1
                            edge_rank_changes += sensitivity["edge_rank_changed"]
                            near_margin += sensitivity["updated_margin"] <= 2 * sensitivity["max_score_delta"]
                            if active_decision["tokens"][1] != frozen_decision["tokens"][1]:
                                changed_second += 1
                        if active_decision["matching"] != frozen_decision["matching"]:
                            changed_matching += 1
                        env.step(active_decision["submissions"], active_decision["matching_index"])
                    expected = original[case, "C10", root, seed]
                    assert abs(env.metrics()["objective"] - expected["objective"]) < 1e-8
                    row = {"case": case, "train_root": root, "seed": seed,
                           "checkpoint_sha256": selected["checkpoint_sha256"],
                           "root_first_edges": root_edges,
                           "root_first_edges_with_second": root_with_second,
                           "root_first_edges_with_multiple_second": root_with_multiple_second,
                           "root_states_with_second": root_states_with_second,
                           "root_states_with_multiple_second": root_states_with_multiple_second,
                           "second_step_opportunities": opportunities,
                           "one_edge_second_opportunities": one_edge,
                           "multiple_edge_second_opportunities": multiple_edges,
                           "second_edge_rank_changes": edge_rank_changes,
                           "margin_within_two_max_logit_deltas": near_margin,
                           "changed_second_token": changed_second,
                           "changed_final_matching": changed_matching,
                           "trained_C10_D10_first_token_difference": trained_first_changed,
                           "trained_C10_D10_final_matching_difference": trained_matching_changed,
                           "mean_second_step_tv": float(np.mean(tv_values)) if tv_values else None,
                           "mean_second_step_max_logit_delta": float(np.mean(score_deltas)) if score_deltas else None,
                           "mean_second_step_top_margin": float(np.mean(margins)) if margins else None,
                           "sum_second_step_tv": float(np.sum(tv_values)),
                           "sum_second_step_max_logit_delta": float(np.sum(score_deltas)),
                           "sum_second_step_top_margin": float(np.sum(margins)),
                           "C10_H_tail_cost": expected["objective"]}
                    pilot.append_jsonl(output / "episodes.jsonl", row)
                    rows.append(row)
        assert len(rows) == len(pilot.CASES) * len(pilot.TRAIN_ROOTS) * len(pilot.HELDOUT_SEEDS)
        summary = {"episodes": len(rows), "H_states": len(rows) * 48,
                   "by_case": {case: {
                       "root_first_edges": sum(r["root_first_edges"] for r in rows if r["case"] == case),
                       "root_first_edges_with_second": sum(r["root_first_edges_with_second"] for r in rows if r["case"] == case),
                       "root_first_edges_with_multiple_second": sum(r["root_first_edges_with_multiple_second"] for r in rows if r["case"] == case),
                       "root_states_with_second": sum(r["root_states_with_second"] for r in rows if r["case"] == case),
                       "root_states_with_multiple_second": sum(r["root_states_with_multiple_second"] for r in rows if r["case"] == case),
                       "second_step_opportunities": sum(r["second_step_opportunities"] for r in rows if r["case"] == case),
                       "one_edge_second_opportunities": sum(r["one_edge_second_opportunities"] for r in rows if r["case"] == case),
                       "multiple_edge_second_opportunities": sum(r["multiple_edge_second_opportunities"] for r in rows if r["case"] == case),
                       "second_edge_rank_changes": sum(r["second_edge_rank_changes"] for r in rows if r["case"] == case),
                       "margin_within_two_max_logit_deltas": sum(r["margin_within_two_max_logit_deltas"] for r in rows if r["case"] == case),
                       "changed_second_token": sum(r["changed_second_token"] for r in rows if r["case"] == case),
                       "changed_final_matching": sum(r["changed_final_matching"] for r in rows if r["case"] == case),
                       "trained_C10_D10_first_token_difference": sum(r["trained_C10_D10_first_token_difference"] for r in rows if r["case"] == case),
                       "trained_C10_D10_final_matching_difference": sum(r["trained_C10_D10_final_matching_difference"] for r in rows if r["case"] == case),
                       "mean_second_step_tv": sum(r["sum_second_step_tv"] for r in rows if r["case"] == case) /
                           max(1, sum(r["second_step_opportunities"] for r in rows if r["case"] == case)),
                       "mean_second_step_max_logit_delta": sum(r["sum_second_step_max_logit_delta"] for r in rows if r["case"] == case) /
                           max(1, sum(r["second_step_opportunities"] for r in rows if r["case"] == case)),
                       "mean_second_step_top_margin": sum(r["sum_second_step_top_margin"] for r in rows if r["case"] == case) /
                           max(1, sum(r["second_step_opportunities"] for r in rows if r["case"] == case)),
                   } for case in pilot.CASES}}
        pilot.write_json(output / "summary.json", summary)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {str(p.relative_to(output)): pilot.sha256(p)
                                     for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
        pilot.write_json(output / "manifest.json", manifest)
    print(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
