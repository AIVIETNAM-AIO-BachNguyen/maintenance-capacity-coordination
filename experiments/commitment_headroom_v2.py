"""Prelocked, bounded v2 action-set headroom check on strong-rollout states."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, VERSION as BASE_VERSION, configs
from ht_pdm_fjsp.maintenance_request_policies import choose, request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import RequestEnv, legal_matchings, submit_requests, transition
from maintenance_capacity_coordination.coordination_v1 import eligible_requests

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/commitment_headroom_v2.md"
CASES = ("reciprocal_scale2", "specialist_generalist_scale2")
SEEDS = tuple(range(73170000, 73170016))
TICKS = tuple(range(0, HORIZON, 6))
SCENARIOS_SELECTION = 16
SCENARIOS_HOLDOUT = 32
FORECAST_DEPTH = 6
TIMEOUT_SECONDS = 1200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def forecast_seed(case, seed, tick):
    return role_seed(62300000, f"{BASE_VERSION}:forecast:{case}:{seed}:{tick}")


def holdout_seed(case, seed, tick):
    return role_seed(73171000, f"commitment_headroom_v2:holdout:{case}:{seed}:{tick}")


def scenario_bank(config, seed, depth, scenarios):
    rng = random.Random(seed)
    return tuple(tuple(tuple(rng.random() < config.failure_probability
                             for _ in range(config.machines)) for _ in range(depth))
                 for _ in range(scenarios))


def expanded_options(config, state, tick):
    """Return threshold admissions, full legal roots, and just-in-time submissions.

    Provisional requests are used only to enumerate options. No returned state
    is installed into the real environment.
    """
    threshold = request_action(config, state, tick, "threshold")
    admitted = submit_requests(config, state, tick, threshold)
    ordinary = set(legal_matchings(config, admitted))
    extras = tuple(m for m, can in enumerate(eligible_requests(config, admitted)) if can)
    provisional = submit_requests(config, admitted, tick, extras)
    expanded = set(legal_matchings(config, provisional))
    assert ordinary <= expanded and () in expanded

    def submissions_for(matching):
        early = tuple(sorted(m for m, _ in matching if admitted.requests[m] is None))
        return tuple(sorted((*threshold, *early))), early

    return threshold, ordinary, expanded, submissions_for


def forecast_cost(config, state, tick, submissions, matching, bank, *, check_timeout=lambda: None):
    total = 0.0
    for tape in bank:
        check_timeout()
        simulated = state
        for offset, events in enumerate(tape):
            t = tick + offset
            if offset == 0:
                submit, action = submissions, matching
            else:
                submit = request_action(config, simulated, t, "threshold")
                pending = submit_requests(config, simulated, t, submit)
                action = risk_matching(config, pending, t)
            simulated, cost, _ = transition(config, simulated, t, submit, action, events)
            total += cost
    return total / len(bank)


def compare_state(config, state, tick, case, seed, baseline, diagnostic, *, check_timeout=lambda: None):
    threshold, ordinary, expanded, submissions_for = expanded_options(config, state, tick)
    baseline_submit, baseline_matching = baseline
    assert baseline_submit == threshold and baseline_matching in ordinary
    depth = min(FORECAST_DEPTH, config.horizon - tick)
    selection_id = forecast_seed(case, seed, tick)
    holdout_id = holdout_seed(case, seed, tick)
    assert selection_id != holdout_id
    selection_bank = scenario_bank(config, selection_id, depth, SCENARIOS_SELECTION)
    holdout_bank = scenario_bank(config, holdout_id, depth, SCENARIOS_HOLDOUT)
    reported = next((item["expected_cost"] for item in diagnostic["root_scores"]
                     if item["matching"] == baseline_matching), None)
    base_selection_cost = forecast_cost(config, state, tick, threshold, baseline_matching,
                                        selection_bank, check_timeout=check_timeout)
    if reported is not None:
        assert abs(reported - base_selection_cost) < 1e-8
    else:
        assert len(ordinary) == 1 and diagnostic["forecast_transitions"] == 0
    winner, winner_submit, winner_early = baseline_matching, threshold, ()
    winner_selection_cost = base_selection_cost
    extra_roots = sorted(expanded - ordinary)
    for matching in extra_roots:
        check_timeout()
        submissions, early = submissions_for(matching)
        value = forecast_cost(config, state, tick, submissions, matching, selection_bank,
                              check_timeout=check_timeout)
        if value < winner_selection_cost - 1e-10:
            winner, winner_submit, winner_early = matching, submissions, early
            winner_selection_cost = value
    base_holdout = forecast_cost(config, state, tick, threshold, baseline_matching,
                                 holdout_bank, check_timeout=check_timeout)
    expanded_holdout = forecast_cost(config, state, tick, winner_submit, winner,
                                     holdout_bank, check_timeout=check_timeout)
    assert np.isfinite(base_holdout) and np.isfinite(expanded_holdout)
    return {
        "case": case, "seed": seed, "tick": tick,
        "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
        "selection_forecast_seed": selection_id, "holdout_forecast_seed": holdout_id,
        "baseline_matching": baseline_matching, "expanded_matching": winner,
        "baseline_submissions": threshold, "expanded_submissions": winner_submit,
        "early_pm_selected": winner_early,
        "ordinary_root_count": len(ordinary), "extra_root_count": len(extra_roots),
        "baseline_selection_cost": base_selection_cost,
        "expanded_selection_cost": winner_selection_cost,
        "baseline_holdout_cost": base_holdout,
        "expanded_holdout_cost": expanded_holdout,
        "holdout_difference": expanded_holdout - base_holdout,
        "forecast_transitions": diagnostic["forecast_transitions"]
                                + (len(extra_roots) * SCENARIOS_SELECTION
                                   + 2 * SCENARIOS_HOLDOUT) * depth,
    }


def assert_seed_separation(cases, seeds, ticks):
    prior_raw = (set(range(73110000, 73110003)) | set(range(73120000, 73120016))
                 | set(range(73130000, 73130016)) | set(range(73140000, 73140032))
                 | set(range(73160000, 73160032)))
    if set(seeds) & prior_raw:
        raise RuntimeError("raw seed overlap")
    prior_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in prior_raw}
    current = {role_seed(seed, BASE_VERSION + ":physical") for seed in seeds}
    forecast = {forecast_seed(case, seed, tick) for case in cases for seed in seeds for tick in ticks}
    holdout = {holdout_seed(case, seed, tick) for case in cases for seed in seeds for tick in ticks}
    if len(current) != len(seeds) or len(forecast) != len(cases) * len(seeds) * len(ticks) \
       or len(holdout) != len(forecast) or current & prior_physical \
       or forecast & holdout or forecast & current or holdout & current \
       or forecast & prior_physical or holdout & prior_physical:
        raise RuntimeError("derived seed overlap")


def run(mode, output_root):
    if mode not in ("smoke", "check"):
        raise ValueError(mode)
    cases = CASES if mode == "check" else CASES[:1]
    seeds = SEEDS if mode == "check" else (73100031,)
    ticks = TICKS if mode == "check" else (0, 6)
    assert_seed_separation(cases, seeds, ticks)
    run_path = output_root / ("commitment_headroom_v2_" + mode + "_"
                              + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run_path.mkdir(parents=True, exist_ok=False)
    source = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    config_by_case = configs()
    manifest = {
        "status": "RUNNING", "kind": mode, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source},
        "config_sha256": {case: hashlib.sha256(json.dumps(asdict(config_by_case[case]),
                                                        sort_keys=True).encode()).hexdigest()
                          for case in cases},
        "cases": cases, "physical_seeds": seeds, "sampled_ticks": ticks,
        "selection_scenarios": SCENARIOS_SELECTION, "holdout_scenarios": SCENARIOS_HOLDOUT,
        "forecast_depth": FORECAST_DEPTH, "timeout_seconds": TIMEOUT_SECONDS,
        "sampled_states": 0, "episodes_completed": 0, "forecast_transitions": 0,
    }
    write_json(run_path / "manifest.json", manifest)
    started = time.monotonic()
    deadline = started + TIMEOUT_SECONDS

    def check_timeout():
        if time.monotonic() > deadline:
            raise TimeoutError("bounded headroom check timed out")

    try:
        for case in cases:
            config = config_by_case[case]
            for seed in tqdm(seeds, desc=f"Headroom {case}"):
                check_timeout()
                env = RequestEnv(config)
                env.reset(role_seed(seed, BASE_VERSION + ":physical"))
                deltas = []
                extra_count = 0
                for tick in range(HORIZON):
                    assert env.time == tick
                    fseed = forecast_seed(case, seed, tick)
                    submissions, matching, diagnostic = choose(
                        config, env.state, tick, "threshold", "rollout",
                        policy_horizon=HORIZON, forecast_seed=fseed,
                        scenarios=SCENARIOS_SELECTION, forecast_horizon=FORECAST_DEPTH,
                        check_timeout=check_timeout)
                    manifest["forecast_transitions"] += diagnostic["forecast_transitions"]
                    if tick in ticks:
                        row = compare_state(config, env.state, tick, case, seed,
                                            (submissions, matching), diagnostic,
                                            check_timeout=check_timeout)
                        append_jsonl(run_path / "states.jsonl", row)
                        deltas.append(row["holdout_difference"])
                        extra_count += bool(row["early_pm_selected"])
                        manifest["sampled_states"] += 1
                        manifest["forecast_transitions"] += (row["forecast_transitions"]
                                                             - diagnostic["forecast_transitions"])
                        write_json(run_path / "manifest.json", manifest)
                    env.step(submissions, matching)
                summary = {"case": case, "seed": seed, "sampled_states": len(deltas),
                           "mean_holdout_difference": float(np.mean(deltas)),
                           "extra_pm_selected_states": extra_count}
                append_jsonl(run_path / "episodes.jsonl", summary)
                manifest["episodes_completed"] += 1
                write_json(run_path / "manifest.json", manifest)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {p.name: sha256(p) for p in run_path.iterdir()
                                      if p.is_file() and p.name != "manifest.json"}
        write_json(run_path / "manifest.json", manifest)
    return run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "check"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(run(args.mode, args.output_root).resolve())


if __name__ == "__main__":
    main()
