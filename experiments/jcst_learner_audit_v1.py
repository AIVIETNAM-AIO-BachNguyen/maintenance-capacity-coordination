"""Independent physical replay, catalogue, selection and contrast audit for JCST L1."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from jcst_replay_audit import catalogue, independent_step
from jcst_campaign_audit_v1 import expected_risk_action, expected_baseline_action
from jcst_learner_campaign_v1 import (
    ARMS, BANK, BASE_SEED, B, CASES, CHECKPOINTS, H, ROOT, ROOTS,
    action_universe, case_config, encode, load_policy, seed, sha,
)
from ht_pdm_fjsp.component_service_v1 import State, config_json, initial_state, state_from_json
from ht_pdm_fjsp.jcst_learner_v1 import tensors

BOOTSTRAP_SEED = BASE_SEED + 900_000_000
BOOTSTRAP_REPS = 2000
Q_LOW, Q_HIGH = 0.004166666666666667, 0.9958333333333333


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def audit_episode(cfg: dict, row: dict, expected_seed: int, *,
                  policy=None, universe=None, config=None):
    if row["terminal_status"] != "COMPLETE" or row["seed"] != expected_seed:
        raise ValueError("episode status/seed")
    trace = row["trace"]
    if len(trace) != H + B or row["simulator_ticks"] != H + B or row["policy_decisions"] != H:
        raise ValueError("tick count")
    state = json.loads(json.dumps(initial_state(config), default=lambda x: x.__dict__))
    totals = {"H": 0.0, "tail": 0.0}
    pieces = dict(preventive=0.0, corrective=0.0, unavailable=0.0, failure=0.0)
    wait = pm = cm = starts = failures = completions = occupancy = unavailable = 0
    for tick, item in enumerate(trace):
        expected_phase = "H" if tick < H else "tail"
        if item["tick"] != tick or item["phase"] != expected_phase:
            raise ValueError("tick/phase")
        possible = catalogue(cfg, state)
        action = tuple(tuple(edge) for edge in item["action"])
        if action not in possible or item["catalogue_size"] != len(possible):
            raise ValueError("full catalogue/legality")
        if tick >= H:
            if action != expected_risk_action(cfg, state, H, B) or item["index"] is not None:
                raise ValueError("fixed tail action")
        elif policy is not None:
            stat = state_from_json(config, json.dumps(state))
            encoded = encode(config, stat, universe, H, B)
            if item["index"] is None or action != universe[item["index"]] or not encoded.mask[item["index"]]:
                raise ValueError("matching index/mask")
            c, w, flat, mask = tensors([encoded])
            with torch.no_grad():
                greedy = int(policy.logits(c, w, flat, mask).argmax(dim=-1).item())
            if action != universe[greedy]:
                raise ValueError("deterministic policy checkpoint action")
        else:
            arm = row["arm"]
            if arm in ARMS:
                if universe is None or item["index"] is None or action != universe[item["index"]]:
                    raise ValueError("training action index")
            elif arm == "rollout2":
                if row["forecast_transitions"] <= 0:
                    raise ValueError("rollout forecast counter")
            else:
                expected = expected_baseline_action(cfg, state, arm)
                if expected is not None and action != expected:
                    raise ValueError("baseline action")
                if arm != "risk" and row["forecast_transitions"] != 0:
                    raise ValueError("unexpected forecast compute")
        following, breakdown, (started, completed, failed, occupied) = independent_step(
            cfg, state, action, expected_seed)
        if breakdown != item["breakdown"] or sum(breakdown.values()) != item["cost"]:
            raise ValueError("cost components")
        totals[expected_phase] += item["cost"]
        for key in pieces:
            pieces[key] += breakdown[key]
        wait += tick < H and not action
        pm += sum(kind == "PM" for _, _, _, _, kind in started)
        cm += sum(kind == "CM" for _, _, _, _, kind in started)
        starts += len(started)
        failures += len(failed)
        completions += len(completed)
        occupancy += len(occupied)
        unavailable += round(breakdown["unavailable"] / cfg["unavailable_cost"])
        state = following
    if (state != row["final_state"] or totals["H"] != row["H_cost"]
            or totals["tail"] != row["tail_cost"]
            or sum(totals.values()) != row["total_cost"] or pieces != row["components"]):
        raise ValueError("H/tail/terminal accounting")
    if row["ppo_samples"] not in (0, H):
        raise ValueError("PPO sample count")
    expected_metrics = dict(pm_starts=pm, cm_starts=cm, failures=failures,
                            completions=completions, unavailable_machine_intervals=unavailable,
                            occupied_technician_intervals=occupancy, wait_H=wait,
                            unfinished_jobs=sum(job is not None for job in state["jobs"]),
                            failed_components_at_cutoff=sum(item["failed"] for group in state["components"] for item in group))
    if row["metrics"] != expected_metrics:
        raise ValueError("episode event metrics")
    return {"wait_H": wait, "starts": starts, "failures": failures,
            "occupied_technician_intervals": occupancy}


def audit_file(path: Path, case: str, arm: str, bank: str, count: int, *,
               root: int = 0, trial_dir: Path | None = None,
               checkpoint: int | None = None):
    config, _ = case_config(case)
    cfg = json.loads(config_json(config))
    policy = universe = None
    if arm in ARMS:
        universe = action_universe(config)
        if bank not in ("train", "rep_train", "engineering"):
            if trial_dir is None or checkpoint is None:
                raise ValueError("checkpoint provenance")
            _, universe, policy = load_policy(trial_dir, checkpoint)
    seen = set()
    costs = []
    metrics = defaultdict(int)
    for row in iter_rows(path):
        ep = len(seen)
        if ep >= count:
            raise ValueError("too many episodes")
        physical = seed(bank, case, root if bank in ("train", "rep_train", "engineering") else 0, ep)
        if physical in seen:
            raise ValueError("duplicate physical seed")
        if row.get("case", case) != case or row.get("arm", arm) != arm or row.get("bank", bank) != bank:
            raise ValueError("case/arm/bank metadata")
        if arm in ARMS and row.get("root", root) != root:
            raise ValueError("training-root metadata")
        if checkpoint is not None and row.get("checkpoint") != checkpoint:
            raise ValueError("checkpoint metadata")
        result = audit_episode(cfg, row, physical, policy=policy,
                               universe=universe, config=config)
        seen.add(physical)
        costs.append(row["total_cost"])
        for key, value in result.items():
            metrics[key] += value
    if len(seen) != count:
        raise ValueError(f"incomplete episode file {path}: {len(seen)} != {count}")
    return costs, dict(metrics)


def source_check(manifest: dict):
    if manifest["schema"] != "JCST-framework-v1" or manifest["status"] != "COMPLETE":
        raise ValueError("source run incomplete")
    if sha(ROOT / "docs/jcst_framework_protocol_v1.md") != manifest["protocol_hash"]:
        raise ValueError("protocol hash mismatch")
    if sha(ROOT / "docs/joint_component_service_contract_v1.md") != manifest["contract_hash"]:
        raise ValueError("contract hash mismatch")
    for rel, digest in manifest["source_hashes"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f"source hash mismatch: {rel}")
    if tuple(manifest["parameters"]["cases"]) != CASES or tuple(manifest["parameters"]["arms"]) != ARMS:
        raise ValueError("case/arm registry")
    if manifest["seed_roles"]["base"] != BASE_SEED or manifest["seed_roles"]["banks"] != BANK:
        raise ValueError("seed namespace")
    if manifest["parameters"]["gamma"] != 1.0 or manifest["parameters"]["reward_divisor"] != 100.0:
        raise ValueError("objective parameters")


def crossed_bootstrap(s: np.ndarray, comparator: np.ndarray, *,
                      comparator_learned: bool, rng: np.random.Generator):
    # Arrays: learned 5 x 128, control 128. Physical indices are paired;
    # training indices are independently resampled across learned arms.
    difference = float(s.mean() - comparator.mean())
    nroot, nphysical = s.shape
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for b in range(BOOTSTRAP_REPS):
        p = rng.integers(0, nphysical, nphysical)
        rs = rng.integers(0, nroot, nroot)
        left = s[np.ix_(rs, p)].mean()
        if comparator_learned:
            rc = rng.integers(0, nroot, nroot)
            right = comparator[np.ix_(rc, p)].mean()
        else:
            right = comparator[p].mean()
        draws[b] = left - right
    return difference, float(np.quantile(draws, Q_LOW)), float(np.quantile(draws, Q_HIGH))


def analyze_confirmation(run_dir: Path, arrays: dict, source_train_dir: str):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    case_success = {}
    for case in CASES:
        criteria = []
        s = arrays[(case, "S")]
        for comparator, margin, kind in (("F", -10, "gain"), ("A", -10, "gain"),
                                          ("threshold5", 10, "noninferiority")):
            other = arrays[(case, comparator)]
            mean, lo, hi = crossed_bootstrap(s, other,
                                              comparator_learned=comparator in ARMS, rng=rng)
            if kind == "gain":
                passed = mean <= margin and hi < 0
            else:
                passed = mean <= margin and hi <= margin
            rows.append({"case": case, "contrast": f"S-{comparator}", "mean": mean,
                         "adjusted_lo": lo, "adjusted_hi": hi, "practical_margin": margin,
                         "pass": passed})
            criteria.append(passed)
        case_success[case] = all(criteria)
    result = {"status": "AUDIT_PASS", "source_train_dir": source_train_dir,
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_reps": BOOTSTRAP_REPS,
              "family_intervals": 6, "rows": rows, "case_success": case_success,
              "replication_eligible": all(case_success.values())}
    return result


def audit_run(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    source_check(manifest)
    stage = manifest["stage"]
    audited = 0
    summaries = {}
    arrays = {}
    if stage in ("smoke", "train_dev", "replication"):
        for info in tqdm(manifest["completed_trials"], desc="training physical replay"):
            case, arm, root = info["case"], info["arm"], info["root"]
            bank = "engineering" if stage == "smoke" else "rep_train" if stage == "replication" else "train"
            trial = run_dir / "trials" / case / arm / str(root)
            costs, _ = audit_file(trial / "train.jsonl.gz", case, arm, bank,
                                  info["episodes"], root=root)
            if info["samples"] != H * len(costs):
                raise ValueError("training sample count")
            checkpoints = (48, 96) if stage == "smoke" else CHECKPOINTS
            for cp in checkpoints:
                if not (trial / f"checkpoint-{cp}.pt").is_file():
                    raise ValueError("missing checkpoint")
            audited += len(costs)
        expected_trials = 2 * 3 if stage == "smoke" else 2 * 3 * 5
        if len(manifest["completed_trials"]) != expected_trials:
            raise ValueError("incomplete trial panel")
    if stage == "smoke":
        for case in CASES:
            for arm in ARMS:
                path = run_dir / "evaluations" / "engineering_eval" / case / arm / "0" / "episodes.jsonl.gz"
                trial = run_dir / "trials" / case / arm / "0"
                audit_file(path, case, arm, "engineering_eval", 2,
                           trial_dir=trial, checkpoint=96)
                audited += 2
        bank = "engineering_eval"
        count = 1
        cases = CASES
    elif stage == "train_dev":
        bank, count, cases = "dev", 16, CASES
    elif stage == "confirmation":
        bank, count, cases = "confirmation", 128, CASES
    elif stage == "replication":
        bank, count, cases = "rep_eval", 128, CASES
    else:
        raise ValueError(stage)
    if stage in ("train_dev", "confirmation", "replication"):
        train_dir = run_dir if stage != "confirmation" else ROOT / manifest["train_dir"]
        selection = None
        if stage == "confirmation":
            selection = json.loads((train_dir / "selection.json").read_text())
            if selection["status"] != "AUDIT_PASS" or selection["source_manifest_hash"] != sha(train_dir / "manifest.json"):
                raise ValueError("unapproved checkpoint selection")
        if stage == "replication":
            prior = ROOT / manifest["confirmation_dir"]
            prior_manifest = json.loads((prior / "manifest.json").read_text())
            prior_train = ROOT / prior_manifest["train_dir"]
            selection = json.loads((prior_train / "selection.json").read_text())
        for case in cases:
            for arm in ARMS:
                for cp in CHECKPOINTS if stage == "train_dev" else (selection["selected"][case][arm],):
                    root_rows = []
                    for root in ROOTS:
                        trial = train_dir / "trials" / case / arm / str(root)
                        if stage == "train_dev":
                            path = run_dir / "evaluations" / bank / case / arm / str(root) / "episodes.jsonl.gz"
                            # train_dev stores all checkpoint evaluations; see checkpoint-specific files.
                            path = run_dir / "evaluations" / bank / case / arm / str(root) / f"episodes-{cp}.jsonl.gz"
                        else:
                            path = run_dir / "evaluations" / bank / case / arm / str(root) / "episodes.jsonl.gz"
                        costs, _ = audit_file(path, case, arm, bank, count,
                                              trial_dir=trial, checkpoint=cp)
                        root_rows.append(costs)
                        audited += count
                    arrays[(case, arm, cp)] = np.asarray(root_rows, dtype=float)
                    summaries[(case, arm, cp)] = float(arrays[(case, arm, cp)].mean())
        if stage == "train_dev":
            selected = {case: {arm: min(CHECKPOINTS, key=lambda cp: (summaries[(case, arm, cp)], cp))
                               for arm in ARMS} for case in CASES}
            atomic = {"status": "AUDIT_PASS", "selected": selected,
                      "means": {case: {arm: {str(cp): summaries[(case, arm, cp)]
                                             for cp in CHECKPOINTS} for arm in ARMS} for case in CASES},
                      "source_manifest_hash": sha(run_dir / "manifest.json")}
            (run_dir / "selection.json").write_text(json.dumps(atomic, indent=2) + "\n")
        else:
            for case in cases:
                for arm in ARMS:
                    arrays[(case, arm)] = arrays[(case, arm, selection["selected"][case][arm])]
    for case in cases:
        for arm in ("threshold5", "risk", "corrective", "rollout2", "wait"):
            path = run_dir / "controls" / bank / case / arm / "episodes.jsonl.gz"
            if not path.is_file() and stage == "train_dev":
                continue
            costs, _ = audit_file(path, case, arm, bank, count)
            arrays[(case, arm)] = np.asarray(costs, dtype=float)
            audited += count
    report = {"status": "PASS", "stage": stage, "episodes_replayed": audited,
              "source_manifest_hash": sha(run_dir / "manifest.json")}
    (run_dir / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if stage in ("confirmation", "replication"):
        analysis = analyze_confirmation(run_dir, arrays, manifest.get("train_dir"))
        (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit_run(args.run_dir)))


if __name__ == "__main__":
    main()
