"""Prelocked, bounded complete-matching interaction diagnostic (no learner/TEST)."""
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
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests, transition
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/pair_commitment_interaction_check_v5.md"
CASES = ("reciprocal_scale2", "specialist_generalist_scale2")
SEEDS = tuple(range(73200000, 73200016))
SMOKE_SEEDS = (73203000, 73203001)
SELECTION_SCENARIOS = 16
HOLDOUT_SCENARIOS = 32
DEPTH = 6
TIMEOUT = 1200
BOOTSTRAP_SEED = 73202000
DRAWS = 5000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_output(*args):
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def forecast_seed(case, raw, tick):
    return role_seed(62300000, f"{BASE_VERSION}:forecast:{case}:{raw}:{tick}")


def holdout_seed(case, raw, tick):
    return role_seed(73201000, f"pair_commitment_v5:holdout:{case}:{raw}:{tick}")


def scenario_bank(config, seed, scenarios=HOLDOUT_SCENARIOS, depth=DEPTH):
    rng = random.Random(seed)
    return tuple(tuple(tuple(rng.random() < config.failure_probability
                             for _ in range(config.machines)) for _ in range(depth))
                 for _ in range(scenarios))


def forecast_cost(config, state, tick, submissions, matching, bank, check_timeout=lambda: None):
    total = 0.
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


def additive_decision(root_scores):
    """Use F2 enumeration order, including its risk-matching-first tie rule."""
    scores = {tuple(tuple(edge) for edge in row["matching"]): row["expected_cost"]
              for row in root_scores}
    if len(scores) != len(root_scores) or () not in scores:
        raise RuntimeError("invalid F2 root score set")
    pair_interactions = []
    best, best_score = None, float("inf")
    for row in root_scores:
        action = tuple(tuple(edge) for edge in row["matching"])
        value = row["expected_cost"]
        if len(action) == 2:
            if (action[0],) not in scores or (action[1],) not in scores:
                raise RuntimeError("pair without legal singleton scores")
            approximate = scores[(action[0],)] + scores[(action[1],)] - scores[()]
            pair_interactions.append(value - approximate)
            value = approximate
        elif len(action) > 2:
            raise RuntimeError("v5 decoder supports exactly two technicians")
        if value < best_score - 1e-10:
            best, best_score = action, value
    return best, pair_interactions


def seed_registry_check(cases=CASES, seeds=SEEDS):
    old_raw = (set(range(73110000, 73110003)) | set(range(73120000, 73120016))
               | set(range(73130000, 73130016)) | set(range(73140000, 73140032))
               | set(range(73160000, 73160032)) | set(range(73170000, 73170016))
               | set(range(73180000, 73180003)) | set(range(73181000, 73181016))
               | set(range(73182000, 73182032)) | set(range(73183000, 73183003))
               | set(range(73184000, 73184016)) | set(range(73185000, 73185032))
               | set(range(73186000, 73186032)) | set(range(73190000, 73190003))
               | set(range(73191000, 73191016)) | set(range(73192000, 73192032))
               | set(range(73193000, 73193003)) | set(range(73194000, 73194016))
               | set(range(73195000, 73195032)) | set(range(73196000, 73196032)))
    old_episode_raw = set()
    for root in range(73110000, 73110003):
        for case in CASES:
            old_episode_raw.update(role_seed(root, f"pilot_train:{case}:{ep}")
                                   for ep in range(200))
    for root in (*range(73180000, 73180003), *range(73183000, 73183003)):
        for case in CASES:
            old_episode_raw.update(role_seed(root, f"structured_train_v3:{case}:{ep}")
                                   for ep in range(200))
    for root in (*range(73190000, 73190003), *range(73193000, 73193003)):
        for case in CASES:
            old_episode_raw.update(role_seed(root, f"forecast_distill_v4:{case}:{ep}")
                                   for ep in range(16))
    raw = set(seeds) | set(SMOKE_SEEDS)
    if len(set(seeds)) != len(seeds) or raw & (old_raw | old_episode_raw):
        raise RuntimeError("v5 raw seed overlap")
    all_prior = old_raw | old_episode_raw
    physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in raw}
    old_physical = {role_seed(seed, BASE_VERSION + ":physical") for seed in all_prior}
    if len(physical) != len(raw) or physical & old_physical:
        raise RuntimeError("v5 physical seed overlap")
    forecasts = {forecast_seed(case, seed, tick)
                 for case in cases for seed in raw for tick in range(HORIZON)}
    holdouts = {holdout_seed(case, seed, tick)
                for case in cases for seed in raw for tick in range(HORIZON)}
    expected = len(cases) * len(raw) * HORIZON
    old_forecasts = {forecast_seed(case, seed, tick)
                     for case in CASES for seed in all_prior for tick in range(HORIZON)}
    old_holdouts = {role_seed(73171000, f"commitment_headroom_v2:holdout:{case}:{seed}:{tick}")
                    for case in CASES for seed in range(73170000, 73170016)
                    for tick in range(0, HORIZON, 6)}
    if (len(forecasts) != expected or len(holdouts) != expected
            or forecasts & holdouts or (forecasts | holdouts) & (physical | old_physical
                                                               | old_forecasts | old_holdouts)):
        raise RuntimeError("v5 forecast/holdout seed overlap")
    return True


def compare_state(config, state, tick, case, raw, submissions, matching, diagnostic,
                  check_timeout=lambda: None):
    admitted = submit_requests(config, state, tick, submissions)
    legal = legal_matchings(config, admitted)
    if matching not in legal or submissions != request_action(config, state, tick, "threshold"):
        raise RuntimeError("F2 contract mismatch")
    roots = diagnostic["root_scores"]
    if len(legal) > 1 and len(roots) != len(legal):
        raise RuntimeError("F2 root score coverage")
    pair_eligible = any(len(a) == 2 for a in legal)
    if len(legal) == 1:
        additive, interactions = matching, []
    else:
        additive, interactions = additive_decision(roots)
    if additive not in legal or (not pair_eligible and additive != matching):
        raise RuntimeError("additive choice invalid")
    hold_seed = holdout_seed(case, raw, tick)
    row = {
        "case": case, "raw_seed": raw, "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
        "tick": tick, "state_sha256": hashlib.sha256(repr(state).encode()).hexdigest(),
        "selection_forecast_seed": forecast_seed(case, raw, tick),
        "holdout_forecast_seed": hold_seed, "submissions": submissions,
        "legal_root_count": len(legal), "pair_eligible": pair_eligible,
        "selection_root_scores": roots, "selection_root_scores_sha256": digest(roots),
        "f2_matching": matching, "additive_matching": additive,
        "selection_forecast_transitions": diagnostic["forecast_transitions"],
        "pair_interactions": interactions, "holdout_bank_sha256": None,
        "f2_holdout_score": None, "additive_holdout_score": None,
        "holdout_difference": 0., "holdout_forecast_transitions": 0,
    }
    if pair_eligible:
        bank = scenario_bank(config, hold_seed)
        row["holdout_bank_sha256"] = digest(bank)
        row["f2_holdout_score"] = forecast_cost(config, state, tick, submissions, matching,
                                                bank, check_timeout)
        row["additive_holdout_score"] = forecast_cost(config, state, tick, submissions,
                                                      additive, bank, check_timeout)
        row["holdout_difference"] = row["additive_holdout_score"] - row["f2_holdout_score"]
        row["holdout_forecast_transitions"] = 2 * HOLDOUT_SCENARIOS * DEPTH
    return row


def analyze(rows):
    by = {(case, seed): [] for case in CASES for seed in SEEDS}
    for row in rows:
        by[row["case"], row["raw_seed"]].append(row)
    if any(len(items) != HORIZON for items in by.values()):
        raise RuntimeError("incomplete v5 panel")
    cases = {}
    for case in CASES:
        data = [r for seed in SEEDS for r in by[case, seed]]
        eligible = [r for r in data if r["pair_eligible"]]
        if not eligible:
            raise RuntimeError("no pair-eligible states")
        cases[case] = {
            "states": len(data), "pair_eligible": len(eligible),
            "pair_frequency": len(eligible) / len(data),
            "choice_disagreement": sum(r["additive_matching"] != r["f2_matching"]
                                       for r in eligible) / len(eligible),
            "conditional_mean": float(np.mean([r["holdout_difference"] for r in eligible])),
            "all_state_mean": float(np.mean([r["holdout_difference"] for r in data])),
            "pair_interaction_mean": float(np.mean([v for r in eligible
                                                    for v in r["pair_interactions"]])),
            "episode_summaries": [{"raw_seed": seed,
                "pair_count": sum(r["pair_eligible"] for r in by[case, seed]),
                "all_state_mean": float(np.mean([r["holdout_difference"]
                                                  for r in by[case, seed]])),
                "conditional_mean": (float(np.mean([r["holdout_difference"]
                    for r in by[case, seed] if r["pair_eligible"]]))
                    if any(r["pair_eligible"] for r in by[case, seed]) else None)}
                for seed in SEEDS],
        }
    per_seed = np.asarray([[sum(r["holdout_difference"] for r in by[case, seed]),
                            sum(r["pair_eligible"] for r in by[case, seed])]
                           for seed in SEEDS for case in CASES], dtype=float).reshape(len(SEEDS), 2, 2)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(SEEDS), size=(DRAWS, len(SEEDS)))
    sums = per_seed[draws, 0, 0].sum(axis=1)
    counts = per_seed[draws, 0, 1].sum(axis=1)
    valid = counts > 0
    interval = np.quantile(sums[valid] / counts[valid], (.025, .975)).tolist()
    cases[CASES[0]]["physical_cluster_95"] = interval
    primary = cases[CASES[0]]
    conditions = {
        "pair_frequency_ge_0.10": primary["pair_frequency"] >= .10,
        "disagreement_ge_0.20": primary["choice_disagreement"] >= .20,
        "conditional_mean_ge_1.5": primary["conditional_mean"] >= 1.5,
        "all_state_mean_ge_0.20": primary["all_state_mean"] >= .20,
        "cluster_lower_gt_zero": interval[0] > 0.,
    }
    return {"cases": cases, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": DRAWS, "gate_conditions": conditions, "go": all(conditions.values()),
            "estimand": "independent-bank additive minus F2 six-tick forecast cost on F2-visited states"}


def run(mode, output_root=Path("artifacts")):
    if mode not in ("smoke", "check"):
        raise ValueError(mode)
    cases = CASES if mode == "check" else CASES[:1]
    seeds = SEEDS if mode == "check" else SMOKE_SEEDS
    seed_registry_check(cases, seeds)
    run_path = output_root / ("pair_commitment_v5_" + mode + "_"
                              + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run_path.mkdir(parents=True, exist_ok=False)
    cfgs = configs()
    sources = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    manifest = {
        "status": "RUNNING", "mode": mode, "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "config_sha256": {case: digest(asdict(cfgs[case])) for case in cases},
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "cases": cases, "seeds": seeds,
        "selection_scenarios": SELECTION_SCENARIOS,
        "holdout_scenarios": HOLDOUT_SCENARIOS, "depth": DEPTH,
        "timeout_seconds": TIMEOUT, "episodes_completed": 0,
        "states_completed": 0, "selection_forecast_transitions": 0,
        "holdout_forecast_transitions": 0,
    }
    write_json(run_path / "manifest.json", manifest)
    started = time.monotonic()
    deadline = started + TIMEOUT
    def check_timeout():
        if time.monotonic() > deadline:
            raise TimeoutError("v5 bounded check timed out")
    try:
        for case in cases:
            cfg = cfgs[case]
            for raw in tqdm(seeds, desc=f"V5 {case}"):
                env = CoordinationEnv(cfg)
                env.reset(raw)
                for tick in range(HORIZON):
                    check_timeout()
                    state = env.inner.state
                    submissions, matching, diagnostic = choose(
                        cfg, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
                        forecast_seed=forecast_seed(case, raw, tick),
                        scenarios=SELECTION_SCENARIOS, forecast_horizon=DEPTH,
                        check_timeout=check_timeout)
                    row = compare_state(cfg, state, tick, case, raw, submissions, matching,
                                        diagnostic, check_timeout)
                    append_jsonl(run_path / "states.jsonl", row)
                    manifest["states_completed"] += 1
                    manifest["selection_forecast_transitions"] += row["selection_forecast_transitions"]
                    manifest["holdout_forecast_transitions"] += row["holdout_forecast_transitions"]
                    env.step(submissions, env.actions.index(matching))
                append_jsonl(run_path / "episodes.jsonl", {"case": case, "raw_seed": raw,
                             "physical_seed": role_seed(raw, BASE_VERSION + ":physical"),
                             **env.metrics()})
                manifest["episodes_completed"] += 1
                write_json(run_path / "manifest.json", manifest)
        if mode == "check":
            analysis = analyze(read_jsonl(run_path / "states.jsonl"))
            write_json(run_path / "analysis.json", analysis)
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
