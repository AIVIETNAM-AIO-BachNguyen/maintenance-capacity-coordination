"""Replay and independently check the v5 bounded interaction artifact."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION as BASE_VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from pair_commitment_interaction_v5 import (
    CASES, SEEDS, SMOKE_SEEDS, SELECTION_SCENARIOS, DEPTH, PROTOCOL, ROOT,
    additive_decision, analyze, compare_state, digest, forecast_seed, read_jsonl,
    seed_registry_check, sha256, write_json,
)


def equivalent(left, right):
    return digest(left) == digest(right)


def audit(run_path):
    manifest = json.loads((run_path / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["mode"] not in ("smoke", "check"):
        raise RuntimeError("artifact is not terminal and completed")
    cases = CASES if manifest["mode"] == "check" else CASES[:1]
    seeds = SEEDS if manifest["mode"] == "check" else SMOKE_SEEDS
    if tuple(manifest["cases"]) != cases or tuple(manifest["seeds"]) != seeds:
        raise RuntimeError("wrong case or seed panel")
    seed_registry_check(cases, seeds)
    if manifest["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("protocol hash mismatch")
    for filename, claimed in manifest["source_sha256"].items():
        if sha256(ROOT / filename) != claimed:
            raise RuntimeError(f"source hash mismatch: {filename}")
    cfgs = configs()
    for case in cases:
        if manifest["config_sha256"][case] != digest(asdict(cfgs[case])):
            raise RuntimeError(f"config hash mismatch: {case}")
    for filename, claimed in manifest["output_sha256"].items():
        if sha256(run_path / filename) != claimed:
            raise RuntimeError(f"raw output hash mismatch: {filename}")
    states = read_jsonl(run_path / "states.jsonl")
    episodes = read_jsonl(run_path / "episodes.jsonl")
    if len(states) != len(cases) * len(seeds) * HORIZON or len(episodes) != len(cases) * len(seeds):
        raise RuntimeError("incomplete rows")
    state_iter = iter(states)
    episode_iter = iter(episodes)
    selection_transitions = holdout_transitions = 0
    for case in cases:
        cfg = cfgs[case]
        for raw in seeds:
            env = CoordinationEnv(cfg)
            env.reset(raw)
            for tick in range(HORIZON):
                recorded = next(state_iter)
                state = env.inner.state
                submissions, matching, diagnostic = choose(
                    cfg, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                    forecast_seed=forecast_seed(case, raw, tick),
                    scenarios=SELECTION_SCENARIOS, forecast_horizon=DEPTH)
                # Replay from the saved pre-decision state using the independent bank.
                expected = compare_state(cfg, state, tick, case, raw, submissions,
                                         matching, diagnostic)
                if not equivalent(recorded, expected):
                    raise RuntimeError(f"state replay mismatch: {case} {raw} {tick}")
                if recorded["physical_seed"] != role_seed(raw, BASE_VERSION + ":physical"):
                    raise RuntimeError("physical seed mismatch")
                if recorded["selection_root_scores_sha256"] != digest(recorded["selection_root_scores"]):
                    raise RuntimeError("root score hash mismatch")
                if recorded["pair_eligible"] and additive_decision(recorded["selection_root_scores"])[0] != tuple(
                        tuple(edge) for edge in recorded["additive_matching"]):
                    raise RuntimeError("additive decision mismatch")
                selection_transitions += expected["selection_forecast_transitions"]
                holdout_transitions += expected["holdout_forecast_transitions"]
                env.step(submissions, env.actions.index(matching))
            episode = next(episode_iter)
            expected_episode = {"case": case, "raw_seed": raw,
                                "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
                                **env.metrics()}
            if not equivalent(episode, expected_episode):
                raise RuntimeError(f"episode replay mismatch: {case} {raw}")
    if (selection_transitions != manifest["selection_forecast_transitions"]
            or holdout_transitions != manifest["holdout_forecast_transitions"]
            or manifest["states_completed"] != len(states)
            or manifest["episodes_completed"] != len(episodes)):
        raise RuntimeError("transition or coverage count mismatch")
    if manifest["mode"] == "check":
        recorded = json.loads((run_path / "analysis.json").read_text())
        calculated = analyze(states)
        if not equivalent(recorded, calculated):
            raise RuntimeError("analysis or gate mismatch")
    result = {"status": "PASS", "mode": manifest["mode"],
              "states_replayed": len(states), "episodes_replayed": len(episodes),
              "selection_forecast_transitions": selection_transitions,
              "holdout_forecast_transitions": holdout_transitions,
              "checks": ["protocol/source/config hashes", "raw file hashes", "raw/derived seed roles",
                         "exact F2 trajectory/root scores", "additive matching and independent bank",
                         "legal actions and H+tail accounting", "analysis and prelocked gate"]}
    write_json(run_path / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_path), sort_keys=True))


if __name__ == "__main__":
    main()
