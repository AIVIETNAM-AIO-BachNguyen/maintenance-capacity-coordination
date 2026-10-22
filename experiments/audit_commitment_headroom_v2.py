"""Replay/audit the locked v2 headroom check, then apply its fixed gate."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION as BASE_VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose
from ht_pdm_fjsp.maintenance_requests import RequestEnv
from experiments.commitment_headroom_v2 import (
    CASES, FORECAST_DEPTH, PROTOCOL, ROOT, SCENARIOS_HOLDOUT,
    SCENARIOS_SELECTION, SEEDS, TICKS, assert_seed_separation,
    expanded_options, forecast_seed, holdout_seed, sha256, write_json,
)

BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 73172000


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_and_analyze(run_path: Path):
    run_path = run_path.resolve()
    manifest = read_json(run_path / "manifest.json")
    assert manifest["status"] == "COMPLETED" and manifest["kind"] == "check"
    assert tuple(manifest["cases"]) == CASES
    assert tuple(manifest["physical_seeds"]) == SEEDS
    assert tuple(manifest["sampled_ticks"]) == TICKS
    assert manifest["selection_scenarios"] == SCENARIOS_SELECTION
    assert manifest["holdout_scenarios"] == SCENARIOS_HOLDOUT
    assert manifest["forecast_depth"] == FORECAST_DEPTH
    assert manifest["protocol_sha256"] == sha256(PROTOCOL)
    for name, digest in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    cfgs = configs()
    for case in CASES:
        digest = hashlib.sha256(json.dumps(asdict(cfgs[case]), sort_keys=True).encode()).hexdigest()
        assert manifest["config_sha256"][case] == digest
    for name, digest in manifest["output_sha256"].items():
        assert sha256(run_path / name) == digest, name
    assert_seed_separation(CASES, SEEDS, TICKS)

    states = read_jsonl(run_path / "states.jsonl")
    episodes = read_jsonl(run_path / "episodes.jsonl")
    assert len(states) == manifest["sampled_states"] == len(CASES) * len(SEEDS) * len(TICKS)
    assert len(episodes) == manifest["episodes_completed"] == len(CASES) * len(SEEDS)
    keyed = {(row["case"], row["seed"], row["tick"]): row for row in states}
    expected = {(case, seed, tick) for case in CASES for seed in SEEDS for tick in TICKS}
    assert set(keyed) == expected and max(Counter((r["case"], r["seed"], r["tick"])
                                                  for r in states).values()) == 1
    episode_keys = Counter((row["case"], row["seed"]) for row in episodes)
    assert set(episode_keys) == {(case, seed) for case in CASES for seed in SEEDS}
    assert max(episode_keys.values()) == 1
    expected_forecast_transitions = 0
    for case in CASES:
        config = cfgs[case]
        for seed in tqdm(SEEDS, desc=f"Audit replay {case}"):
            env = RequestEnv(config)
            env.reset(role_seed(seed, BASE_VERSION + ":physical"))
            for tick in range(HORIZON):
                selection_id = forecast_seed(case, seed, tick)
                submissions, matching, diagnostic = choose(
                    config, env.state, tick, "threshold", "rollout",
                    policy_horizon=HORIZON, forecast_seed=selection_id,
                    scenarios=SCENARIOS_SELECTION, forecast_horizon=FORECAST_DEPTH)
                expected_forecast_transitions += diagnostic["forecast_transitions"]
                if tick in TICKS:
                    row = keyed[case, seed, tick]
                    threshold, ordinary, expanded, submissions_for = expanded_options(
                        config, env.state, tick)
                    base = tuple(map(tuple, row["baseline_matching"]))
                    candidate = tuple(map(tuple, row["expanded_matching"]))
                    assert row["physical_seed"] == role_seed(seed, BASE_VERSION + ":physical")
                    assert row["selection_forecast_seed"] == selection_id
                    assert row["holdout_forecast_seed"] == holdout_seed(case, seed, tick)
                    assert base == matching and base in ordinary
                    assert tuple(row["baseline_submissions"]) == threshold == submissions
                    assert candidate in expanded
                    candidate_submit, early = submissions_for(candidate)
                    assert tuple(row["expanded_submissions"]) == candidate_submit
                    assert tuple(row["early_pm_selected"]) == early
                    assert row["ordinary_root_count"] == len(ordinary)
                    assert row["extra_root_count"] == len(expanded - ordinary)
                    assert row["expanded_selection_cost"] <= row["baseline_selection_cost"] + 1e-8
                    assert abs(row["holdout_difference"] - row["expanded_holdout_cost"]
                               + row["baseline_holdout_cost"]) < 1e-8
                    assert all(np.isfinite(row[name]) for name in
                               ("baseline_selection_cost", "expanded_selection_cost",
                                "baseline_holdout_cost", "expanded_holdout_cost"))
                    expected_count = (diagnostic["forecast_transitions"]
                                      + (len(expanded - ordinary) * SCENARIOS_SELECTION
                                         + 2 * SCENARIOS_HOLDOUT) * FORECAST_DEPTH)
                    assert row["forecast_transitions"] == expected_count
                    expected_forecast_transitions += expected_count - diagnostic["forecast_transitions"]
                env.step(submissions, matching)
    assert manifest["forecast_transitions"] == expected_forecast_transitions
    for row in episodes:
        matched = [keyed[row["case"], row["seed"], tick] for tick in TICKS]
        assert row["sampled_states"] == len(TICKS)
        assert abs(row["mean_holdout_difference"] - np.mean(
            [item["holdout_difference"] for item in matched])) < 1e-8
        assert row["extra_pm_selected_states"] == sum(bool(item["early_pm_selected"])
                                                       for item in matched)

    audit = {
        "status": "PASS", "run": str(run_path),
        "protocol_source_config_and_raw_hashes": "PASS",
        "state_and_episode_coverage": "PASS",
        "F2_state_replay_and_legal_action": "PASS",
        "forecast_seed_separation_and_transition_accounting": "PASS",
        "cost_and_episode_aggregation": "PASS",
        "sampled_states": len(states), "episodes": len(episodes),
        "forecast_transitions": expected_forecast_transitions,
    }
    write_json(run_path / "audit.json", audit)

    episode_means = np.asarray([[next(r["mean_holdout_difference"] for r in episodes
                                      if r["case"] == case and r["seed"] == seed)
                                 for seed in SEEDS] for case in CASES], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    within_case_draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(SEEDS), size=episode_means.shape)
        within_case_draws[index] = np.take_along_axis(episode_means, sampled, axis=1).mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shared_seed_draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(SEEDS), size=len(SEEDS))
        shared_seed_draws[index] = episode_means[:, sampled].mean()
    case_means = {case: float(episode_means[i].mean()) for i, case in enumerate(CASES)}
    mean = float(episode_means.mean())
    ci = np.quantile(within_case_draws, (0.025, 0.975)).tolist()
    shared_ci = np.quantile(shared_seed_draws, (0.025, 0.975)).tolist()
    extra_fraction = sum(bool(r["early_pm_selected"]) for r in states) / len(states)
    go = bool(mean <= -2 and ci[1] < 0 and all(v < 0 for v in case_means.values())
              and extra_fraction >= 0.05)
    analysis = {
        "status": "AUDITED", "run": str(run_path),
        "estimand": "Expanded minus F2 common-holdout six-tick forecast cost on sampled F2 states",
        "case_mean": case_means, "mean": mean,
        "episode_cluster_bootstrap_95": ci,
        "shared_seed_cross_case_cluster_sensitivity_95": shared_ci,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "analysis_seed": BOOTSTRAP_SEED,
        "extra_pm_selected_fraction": extra_fraction,
        "extra_pm_selected_states": sum(bool(r["early_pm_selected"]) for r in states),
        "sampled_states": len(states), "forecast_transitions": expected_forecast_transitions,
        "prelocked_go": go,
        "scope": "F2-visited states and finite-horizon model forecasts; no learned-policy or H+tail claim",
    }
    write_json(run_path / "analysis.json", analysis)
    return audit, analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    audit, analysis = audit_and_analyze(args.run)
    print(json.dumps({"audit": audit["status"], "analysis": analysis}, indent=2))


if __name__ == "__main__":
    main()
