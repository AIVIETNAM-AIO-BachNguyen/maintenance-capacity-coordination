"""Prospective nonlearning diagnostic for the explicit shared request lifecycle."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_request_policies import (
    DISPATCHERS, REQUESTERS, choose, family_configs, request_action,
)
from ht_pdm_fjsp.maintenance_requests import RequestEnv, queue_snapshot

VERSION = "maintenance_request_lifecycle_v1"
ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / f"{VERSION}_plan.md"
REGISTRY = ROOT / "configs" / f"{VERSION}_seed_registry.json"
SOURCES = ("maintenance_dispatch.py", "maintenance_requests.py",
           "maintenance_request_policies.py", "maintenance_request_experiment.py", "__init__.py")
EXECUTION_INPUTS = ("src", "configs", "pyproject.toml", "uv.lock", "AGENTS.md",
                    f"docs/{VERSION}_plan.md")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_preflight(root=ROOT):
    """Protect every runtime/config input, without blocking unrelated reports.

    HEAD diff includes index and working-tree changes. Untracked files under
    input directories are included because imports and registry discovery can
    consume them. No files are modified or removed by this check.
    """
    def read(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True)
    changed = read("diff", "--name-only", "-z", "HEAD", "--", *EXECUTION_INPUTS)
    untracked = read("ls-files", "--others", "--exclude-standard", "-z", "--", *EXECUTION_INPUTS)
    blocking = sorted(set(path for path in (changed + untracked).split("\0") if path))
    return dict(commit=read("rev-parse", "HEAD").strip(),
                branch=read("branch", "--show-current").strip(),
                dirty=bool(read("status", "--porcelain").strip()),
                source_clean=not blocking, blocking_paths=blocking,
                guarded_paths=list(EXECUTION_INPUTS))


def settings(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    registry = json.loads(REGISTRY.read_text())
    raw = registry[profile]
    return dict(profile=profile, policy_horizon=12 if profile == "smoke" else 48,
                tail_ticks=8 if profile == "smoke" else 16,
                seeds=list(range(raw["evaluation_start"], raw["evaluation_start"] + raw["evaluation_count"])),
                forecast_seed=registry["forecast_seed"], bootstrap_seed=registry["bootstrap_seed"],
                scenarios=2 if profile == "smoke" else 16, forecast_horizon=3 if profile == "smoke" else 6,
                bootstrap_samples=200 if profile == "smoke" else 2000,
                max_seconds=600 if profile == "smoke" else 14400,
                training="NOT_APPLICABLE", development="NOT_APPLICABLE",
                interpretation="ENGINEERING_ONLY" if profile == "smoke" else "EXPLORATORY_DIAGNOSTIC")


def registered_integers(value):
    """Conservatively include every integer and expand historical start/count panels."""
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result |= registered_integers(child)
            if key.endswith("_start"):
                count = value.get(key[:-6] + "_count")
                if type(child) is int and type(count) is int and count >= 0:
                    result.update(range(child, child + count))
    elif isinstance(value, list):
        for child in value:
            result |= registered_integers(child)
    elif type(value) is int:
        result.add(value)
    return result


def seed_audit():
    registry = json.loads(REGISTRY.read_text())
    groups = [set(settings(p)["seeds"]) for p in ("smoke", "full")]
    groups += [{registry["forecast_seed"]}, {registry["bootstrap_seed"]}]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("seed roles overlap")
    historical, hashes = set(), {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path == REGISTRY:
            continue
        historical |= registered_integers(json.loads(path.read_text()))
        hashes[path.name] = digest(path)
    own = set.union(*groups)
    if own & historical:
        raise ValueError(f"historical raw seed overlap: {sorted(own & historical)}")
    return dict(status="PASS", raw_unique=len(own), historical_registry_hashes=hashes,
                intentional_pairing="same physical raw seed across family cells and controllers within each split",
                derived_namespace=VERSION)


def physical_seed(seed):
    return role_seed(seed, VERSION + ":physical")


def forecast_seed(cfg, cell, seed, tick):
    return role_seed(cfg["forecast_seed"], f"{VERSION}:forecast:{cell}:{seed}:{tick}")


def controller(cfg, config, state, tick, requester, dispatcher, cell, seed, check_timeout):
    if tick >= cfg["policy_horizon"]:
        requester, dispatcher = "threshold", "risk_matching"
    return choose(config, state, tick, requester, dispatcher,
                  policy_horizon=cfg["policy_horizon"],
                  forecast_seed=forecast_seed(cfg, cell, seed, tick),
                  scenarios=cfg["scenarios"], forecast_horizon=cfg["forecast_horizon"],
                  check_timeout=check_timeout)


def episode_row(env, cutoff, pending_h, cell, requester, dispatcher, seed, seconds, forecasts, policy_horizon):
    metrics = dict(env.metrics)
    completed = metrics["requests_completed"]
    return dict(cell=cell, requester=requester, dispatcher=dispatcher, seed=seed,
                physical_seed=physical_seed(seed), **metrics,
                policy_cost=cutoff["objective"], tail_cost=metrics["objective"] - cutoff["objective"],
                availability=1 - cutoff["unavailable_ticks"] / (env.config.machines * policy_horizon),
                policy_failures=cutoff["failures"],
                completed_wait_mean=metrics["completed_wait_total"] / completed if completed else 0.,
                completed_wait_defined=bool(completed),
                **{"H_" + k: v for k, v in pending_h.items()},
                **{"end_" + k: v for k, v in queue_snapshot(env.state, env.time).items()},
                controller_seconds=seconds, forecast_transitions=forecasts)


def run_episode(cfg, config, cell, requester, dispatcher, seed, journal, check_timeout):
    env = RequestEnv(config)
    env.reset(physical_seed(seed))
    elapsed, forecasts = 0., 0
    for tick in range(config.horizon):
        check_timeout()
        before = env.state
        started = time.perf_counter()
        submissions, matching, diagnostic = controller(
            cfg, config, before, tick, requester, dispatcher, cell, seed, check_timeout)
        elapsed += time.perf_counter() - started
        forecasts += diagnostic["forecast_transitions"]
        _, reward, done, info = env.step(submissions, matching)
        journal.write(json.dumps(dict(cell=cell, requester=requester, dispatcher=dispatcher,
            seed=seed, tick=tick, phase="policy" if tick < cfg["policy_horizon"] else "tail",
            state=asdict(before), submissions=submissions, matching=matching,
            events=env._events[tick], following=asdict(env.state), reward=reward,
            info=info, diagnostic=diagnostic), separators=(",", ":"), allow_nan=False) + "\n")
        if tick + 1 == cfg["policy_horizon"]:
            cutoff = dict(env.metrics)
            pending_h = queue_snapshot(env.state, env.time)
    assert done
    return episode_row(env, cutoff, pending_h, cell, requester, dispatcher, seed, elapsed, forecasts,
                       cfg["policy_horizon"])


def audit_run(out, cfg, configs, rows, check_timeout):
    """Replay every physical transition against its seeded tape, then reconcile rows.

    This is deterministic replay against the production kernel, not an
    independent physical-model implementation. Unit fixtures provide separate
    hand-calculated checks. Forecast winners are checked against logged root
    scores; smoke also reconstructs all hypothetical banks and scores.
    """
    expected = {(c, r, d, s) for c, r, d, s in itertools.product(
        configs, REQUESTERS, DISPATCHERS, cfg["seeds"])}
    by_key = {(r["cell"], r["requester"], r["dispatcher"], r["seed"]): r for r in rows}
    if set(by_key) != expected or len(rows) != len(expected):
        raise AssertionError("episode grid mismatch")
    previous, seen, transitions = None, set(), 0
    with (out / "decisions.jsonl").open() as stream:
        for line in stream:
            check_timeout()
            row = json.loads(line)
            key = (row["cell"], row["requester"], row["dispatcher"], row["seed"])
            if key != previous:
                if previous is not None:
                    finish()
                if key not in expected or key in seen:
                    raise AssertionError("unexpected or repeated episode")
                seen.add(key)
                env = RequestEnv(configs[key[0]])
                env.reset(physical_seed(key[3]))
                previous = key
                forecast_total = 0

                def finish():
                    if env.time != env.config.horizon:
                        raise AssertionError("incomplete episode")
                    actual = episode_row(env, cutoff, pending_h, *previous,
                                         by_key[previous]["controller_seconds"], forecast_total, cfg["policy_horizon"])
                    if actual != by_key[previous]:
                        raise AssertionError("episode metrics differ from replay")

            if (row["tick"] != env.time or row["state"] != json.loads(json.dumps(asdict(env.state)))
                    or row["phase"] != ("policy" if env.time < cfg["policy_horizon"] else "tail")):
                raise AssertionError("state/clock replay mismatch")
            if row["events"] != list(env._events[env.time]):
                raise AssertionError("physical shock tape differs")
            rule = key[1] if env.time < cfg["policy_horizon"] else "threshold"
            if row["submissions"] != list(request_action(env.config, env.state, env.time, rule)):
                raise AssertionError("request rule mismatch")
            # Cheap controllers are always reconstructed. All smoke rollout banks
            # are reconstructed too; full root scores remain a logged-score audit.
            if cfg["profile"] == "smoke" or key[2] != "rollout" or env.time >= cfg["policy_horizon"]:
                sub, match, diag = controller(cfg, env.config, env.state, env.time,
                    key[1], key[2], key[0], key[3], check_timeout)
                actual = json.loads(json.dumps(dict(submissions=sub, matching=match, diagnostic=diag)))
                if any(actual[k] != row[k] for k in actual):
                    raise AssertionError("controller reconstruction differs")
            scores = row["diagnostic"]["root_scores"]
            if scores:
                winner = scores[0]
                for candidate in scores[1:]:
                    if candidate["expected_cost"] < winner["expected_cost"] - 1e-10:
                        winner = candidate
                if winner["matching"] != row["matching"]:
                    raise AssertionError("rollout winner differs from logged scores")
            forecast_total += row["diagnostic"]["forecast_transitions"]
            _, reward, _, info = env.step(row["submissions"], row["matching"])
            if (row["following"] != json.loads(json.dumps(asdict(env.state)))
                    or row["info"] != info or row["reward"] != reward):
                raise AssertionError("transition/cost/lifecycle replay mismatch")
            if env.time == cfg["policy_horizon"]:
                cutoff = dict(env.metrics)
                pending_h = queue_snapshot(env.state, env.time)
            transitions += 1
        if previous is not None:
            finish()
    if seen != expected:
        raise AssertionError("missing episodes")
    return dict(status="PASS", episodes=len(seen), physical_transitions=transitions,
                invalid_assignments=0, physical_replay="ALL",
                forecast_reconstruction="ALL" if cfg["profile"] == "smoke" else "LOGGED_SCORE_WINNERS_ONLY",
                independent_model_verification=False)


def summarize(cfg, configs, rows):
    keys = ("objective", "availability", "policy_failures", "request_wait_ticks",
            "queued_pm_ticks", "queued_cm_ticks", "max_wait_age", "H_pending", "end_pending",
            "H_in_service", "end_in_service", "tail_cost", "controller_seconds")
    means, contrasts = [], []
    for cell in configs:
        grouped = {}
        for r, d in itertools.product(REQUESTERS, DISPATCHERS):
            panel = sorted((row for row in rows if (row["cell"], row["requester"], row["dispatcher"]) == (cell, r, d)),
                           key=lambda row: row["seed"])
            grouped[r, d] = np.asarray([row["objective"] for row in panel])
            means.append(dict(cell=cell, requester=r, dispatcher=d, count=len(panel),
                              **{k: float(np.mean([row[k] for row in panel])) for k in keys}))
        pairs = [((r, "rollout"), (r, "risk_matching")) for r in REQUESTERS]
        pairs += [(("resource_aware", d), ("threshold", d)) for d in DISPATCHERS]
        for treatment, baseline in pairs:
            delta = grouped[treatment] - grouped[baseline]
            rng = np.random.default_rng(role_seed(cfg["bootstrap_seed"], f"{VERSION}:{cfg['profile']}:{cell}:{treatment}:{baseline}"))
            indices = rng.integers(0, len(delta), (cfg["bootstrap_samples"], len(delta)))
            lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            contrasts.append(dict(cell=cell, treatment="/".join(treatment), baseline="/".join(baseline),
                                  mean_cost_difference=float(delta.mean()), ci95_exploratory=[float(lo), float(hi)],
                                  interpretation="negative favors treatment; paired raw-seed unit; no multiplicity guarantee"))
    return dict(interpretation=cfg["interpretation"], means=means, contrasts=contrasts,
                waiting_penalty=0, novel_algorithm_claim=False, timing_optimality_claim=False,
                caveats=["Free early submission can weakly expand centralized dispatch options.",
                         "Bounded-window rollout is not an optimal oracle and can lose.",
                         "Finite common tail mitigates but does not eliminate endpoint bias.",
                         "Synthetic fixed family does not establish industrial validity."])


def run(profile, output_dir, device="cpu"):
    if device != "cpu":
        raise ValueError("CPU only")
    cfg, configs = settings(profile), family_configs(profile)
    seed_info = seed_audit()
    git = git_preflight()
    if profile == "full" and not git["source_clean"]:
        raise RuntimeError("full run requires committed execution inputs; changed paths: "
                           + ", ".join(git["blocking_paths"])
                           + ". Existing files were preserved; run the committed revision in a clean worktree.")
    out = Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("refusing to overwrite run")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = dict(protocol=VERSION, status="RUNNING", git=git, settings=cfg,
                    started_utc=datetime.now(timezone.utc).isoformat(), seed_audit=seed_info,
                    runtime=dict(platform=platform.platform(), python=platform.python_version(),
                                 numpy=np.__version__, device=device))
    write_json(out / "manifest.json", manifest)

    def check_timeout():
        if time.monotonic() - started > cfg["max_seconds"]:
            raise TimeoutError("registered hard timeout")

    try:
        shutil.copy2(PLAN, out / "protocol.md")
        shutil.copy2(REGISTRY, out / "seed_registry.json")
        write_json(out / "configs.json", {name: asdict(c) for name, c in configs.items()})
        snap = out / "source_snapshot"
        snap.mkdir()
        for name in SOURCES:
            shutil.copy2(Path(__file__).parent / name, snap / name)
        sealed_paths = [PLAN, REGISTRY, *(Path(__file__).parent / name for name in SOURCES)]
        source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sealed_paths}
        manifest["source_hashes"] = source_hashes
        write_json(out / "manifest.json", manifest)
        grid = list(itertools.product(configs, REQUESTERS, DISPATCHERS, cfg["seeds"]))
        rows = []
        with (out / "decisions.jsonl").open("x") as journal, (out / "episodes.partial.csv").open("x", newline="") as partial:
            writer = None
            for cell, requester, dispatcher, seed in tqdm(grid, desc="Shared requests", unit="episode"):
                row = run_episode(cfg, configs[cell], cell, requester, dispatcher, seed, journal, check_timeout)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(partial, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                partial.flush()
                journal.flush()
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        audit = audit_run(out, cfg, configs, rows, check_timeout)
        if {str(p.relative_to(ROOT)): digest(p) for p in sealed_paths} != source_hashes:
            raise RuntimeError("source changed during execution")
        write_json(out / "audit.json", audit)
        write_json(out / "summary.json", summarize(cfg, configs, rows))
        manifest.update(status="COMPLETED", counts=audit, elapsed_seconds=time.monotonic() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        artifact_hashes={p.name: digest(p) for p in out.iterdir()
                                         if p.is_file() and p.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise
    print(json.dumps(dict(output_dir=str(out), status="COMPLETED", episodes=len(rows),
                         elapsed_seconds=manifest["elapsed_seconds"])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output_dir, args.device)


if __name__ == "__main__":
    main()
