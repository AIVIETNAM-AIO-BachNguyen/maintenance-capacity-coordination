"""Prelocked simulation-only structural opportunity qualification for v9."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_request_policies import family_configs, risk_matching
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v8 import EDGES, initial_edge_mask

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/same_tick_simulation_regime_v9.md"
CASES = ("interchangeable_scale1", "reciprocal_scale1")
SEEDS = tuple(range(73501000, 73501032))
THRESHOLD = 47


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append_row(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


def opportunity(mask):
    legal = tuple(i for i, allowed in enumerate(mask) if allowed)
    qualified = []
    for first in legal:
        machine, worker = EDGES[first]
        residual = tuple(second for second in legal
                         if EDGES[second][0] != machine and EDGES[second][1] != worker)
        if len(residual) >= 2:
            qualified.append(first)
    return legal, tuple(qualified)


def new_dir(kind, parent):
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / ("same_tick_v9_" + kind + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(exist_ok=False)
    return directory


def source_hashes():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "pyproject.toml"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def config_hashes():
    cfg = family_configs("full")
    return {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
            for case in CASES}


def check_seeds():
    roles = (*range(73500000, 73500003), *SEEDS, *range(73510000, 73510003),
             *range(73511000, 73511016), *range(73512000, 73512032), 73513000,
             *range(73520000, 73520003), *range(73521000, 73521016),
             *range(73522000, 73522032), 73523000, *range(73530000, 73530032))
    assert len(roles) == len(set(roles))
    prior = {role_seed(seed, BASE_VERSION + ":physical") for seed in range(73100000, 73500000)}
    qualified = {role_seed(seed, BASE_VERSION + ":physical") for seed in SEEDS}
    assert len(qualified) == len(SEEDS) and not (prior & qualified)


def collect(parent, seeds=SEEDS, kind="qualification"):
    check_seeds()
    run = new_dir(kind, parent)
    git = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True)
    manifest = {"status": "RUNNING", "kind": kind, "started_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "git_commit": git.stdout.strip(),
                "source_sha256": source_hashes(), "protocol_sha256": digest(PROTOCOL),
                "config_sha256": config_hashes(), "cases": CASES, "seeds": seeds,
                "threshold_count_per_case": THRESHOLD if kind == "qualification" else None}
    write_json(run / "manifest.json", manifest)
    start = time.monotonic()
    print(run.resolve(), flush=True)
    try:
        configurations = family_configs("full")
        counts = {case: 0 for case in CASES}
        for case in CASES:
            env = CoordinationEnv(configurations[case])
            for seed in tqdm(seeds, desc=f"v9 {kind} {case}"):
                env.reset(seed)
                for tick in range(48):
                    submissions = env.fixed_submissions("threshold")
                    admitted = env.admitted(submissions)
                    mask = initial_edge_mask(env.config, admitted)
                    legal, qualifying_first = opportunity(mask)
                    matching = risk_matching(env.config, admitted, env.tick)
                    index = env.actions.index(matching)
                    reward, done = env.step(submissions, index)
                    counts[case] += bool(qualifying_first)
                    append_row(run / "decisions.jsonl", {
                        "case": case, "seed": seed, "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
                        "tick": tick, "submissions": submissions, "root_edges": legal,
                        "qualifying_first_edges": qualifying_first, "matching_index": index,
                        "matching": matching, "reward": reward, "done": done})
                assert done
                append_row(run / "episodes.jsonl", {"case": case, "seed": seed, **env.metrics()})
        manifest["counts"] = counts
        manifest["denominator_per_case"] = 48 * len(seeds)
        manifest["go"] = (all(count >= THRESHOLD for count in counts.values())
                          if kind == "qualification" else None)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)
    return run


def audit(run, parent):
    review = new_dir("qualification_audit", parent)
    report = {"status": "RUNNING", "source_run": str(run.resolve()),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(review / "audit.json", report)
    start = time.monotonic()
    print(review.resolve(), flush=True)
    try:
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "COMPLETED"
        assert manifest["source_sha256"] == source_hashes()
        assert manifest["protocol_sha256"] == digest(PROTOCOL)
        assert manifest["config_sha256"] == config_hashes()
        for name, expected in manifest["output_sha256"].items():
            assert digest(run / name) == expected
        seeds = tuple(manifest["seeds"])
        assert seeds == SEEDS if manifest["kind"] == "qualification" else len(seeds) <= len(SEEDS)
        decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
        episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        assert len(decisions) == len(CASES) * len(seeds) * 48
        assert len(episodes) == len(CASES) * len(seeds)
        counts = {case: 0 for case in CASES}
        offset = 0
        episode_offset = 0
        for case in CASES:
            env = CoordinationEnv(family_configs("full")[case])
            for seed in seeds:
                env.reset(seed)
                for tick in range(48):
                    row = decisions[offset]
                    offset += 1
                    submissions = env.fixed_submissions("threshold")
                    admitted = env.admitted(submissions)
                    legal, qualifying_first = opportunity(initial_edge_mask(env.config, admitted))
                    matching = risk_matching(env.config, admitted, env.tick)
                    index = env.actions.index(matching)
                    assert (row["case"], row["seed"], row["tick"]) == (case, seed, tick)
                    assert row["physical_seed"] == role_seed(seed, BASE_VERSION + ":physical")
                    assert row["submissions"] == list(submissions)
                    assert row["root_edges"] == list(legal)
                    assert row["qualifying_first_edges"] == list(qualifying_first)
                    assert row["matching_index"] == index
                    assert row["matching"] == [list(pair) for pair in matching]
                    reward, done = env.step(submissions, index)
                    assert abs(row["reward"] - reward) < 1e-8 and row["done"] == done
                    counts[case] += bool(qualifying_first)
                episode = episodes[episode_offset]
                episode_offset += 1
                assert (episode["case"], episode["seed"]) == (case, seed)
                for key, value in env.metrics().items():
                    if isinstance(value, (int, float)) and value is not None:
                        assert abs(episode[key] - value) < 1e-7, (case, seed, key)
                    else:
                        assert episode[key] == value, (case, seed, key)
                assert episode["invalid_assignments"] == 0
                assert abs(episode["objective"] - episode["H_cost"] - episode["tail_cost"]) < 1e-7
                assert abs(episode["episode_return"] + episode["objective"]) < 1e-7
        assert counts == manifest["counts"]
        assert manifest["denominator_per_case"] == 48 * len(seeds)
        if manifest["kind"] == "qualification":
            assert manifest["go"] == all(value >= THRESHOLD for value in counts.values())
        report.update(status="PASS", counts=counts, go=manifest["go"],
                      run_manifest_sha256=digest(run / "manifest.json"))
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(review / "audit.json", report)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "qualification", "audit"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.mode == "audit":
        if args.run is None:
            parser.error("--run required for audit")
        audit(args.run, args.output_root)
    else:
        collect(args.output_root, (73500001, 73500002) if args.mode == "smoke" else SEEDS,
                args.mode)


if __name__ == "__main__":
    main()
