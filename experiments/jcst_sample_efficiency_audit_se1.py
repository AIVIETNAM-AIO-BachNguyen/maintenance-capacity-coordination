"""Independent SE1 provenance, full evaluation replay, and fixed TRAIN trace audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource

import numpy as np
import torch
from tqdm import tqdm

import jcst_learnability_audit_l2 as replay
from jcst_learnability_l2 import action_universe, case_config, load_policy

ROOT = Path(__file__).resolve().parents[1]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
FULL_CHECKPOINTS = (24000, 48000, 72000, 96000, 144000, 192000, 288000, 384000)
TRACE_EPISODES = {0, 1, 2, 3, 499, 999, 1499, 1999, 2999, 3999, 5999, 7997, 7998, 7999}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(bank: str, case: str, arm: str | None, root: int, episode: int) -> int:
    ci = CASES.index(case)
    if bank == "train":
        return 2**54 + 10**13 + ci * 10**12 + ARMS.index(arm) * 10**11 + root * 10**8 + episode
    if bank == "engineering_train":
        return 2**54 + 3 * 10**13 + ci * 10**12 + ARMS.index(arm) * 10**11 + root * 10**8 + episode
    offset = {"eval": 2, "engineering": 3, "calibration": 4}[bank] * 10**13
    return 2**54 + offset + ci * 10**12 + episode


def action_seed(bank: str, case: str, arm: str, root: int, checkpoint: int,
                physical_index: int, repetition: int) -> int:
    key = f"JCST-SE1:{case}:{arm}:{root}:{checkpoint}:{physical_index}:{repetition}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def check_manifest(run: Path) -> dict:
    data = json.loads((run / "manifest.json").read_text())
    if data.get("schema") != "JCST-SE1" or data.get("status") != "COLLECTED":
        raise ValueError("SE1 collection incomplete")
    for key, path in (("protocol_hash", "docs/jcst_sample_efficiency_protocol_se1.md"),
                      ("design_lock_hash", "docs/jcst_sample_efficiency_design_lock_se1_20261028_v2.json"),
                      ("scheduling_amendment_hash", "docs/jcst_sample_efficiency_parallel_amendment_se1_20261028.md"),
                      ("contract_hash", "docs/joint_component_service_contract_v1.md")):
        if data[key] != sha(ROOT / path):
            raise ValueError(f"{key} mismatch")
    for path, digest in data["source_hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"source hash changed: {path}")
    if data["parameters"]["H"] != 48 or data["parameters"]["B"] != 16:
        raise ValueError("H/tail contract")
    if data["seed_roles"]["physical_base"] != 2**54:
        raise ValueError("seed base")
    if data["parameters"]["torch_threads"] != 1:
        raise ValueError("Torch threads")
    return data


def audit_calibration(run: Path, data: dict) -> int:
    lock = json.loads((run / "target-lock.json").read_text())
    if lock["status"] != "TARGET_LOCKED" or data["target_lock_sha256"] != sha(run / "target-lock.json"):
        raise ValueError("target lock")
    count = 3 if data["smoke"] else 128
    bank = "engineering" if data["smoke"] else "calibration"
    total = 0
    for case in CASES:
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        path = run / "calibration" / case / "episodes.jsonl"
        if lock["cases"][case]["rows_sha256"] != sha(path):
            raise ValueError("calibration hash")
        costs = []
        for i, row in enumerate(replay.lines(path)):
            if i >= count or row["physical_index"] != i or row["bank"] != bank:
                raise ValueError("calibration index")
            replay.physical_replay(row, cfg, physical(bank, case, None, 0, i), universe)
            costs.append(row["total_cost"])
        if len(costs) != count or not np.isclose(np.mean(costs), lock["cases"][case]["mean_threshold5_cost"]):
            raise ValueError("calibration mean/coverage")
        if not np.isclose(.95 * np.mean(costs), lock["cases"][case]["target_cost"]):
            raise ValueError("target arithmetic")
        total += count
    return total


def audit_training(run: Path, data: dict) -> tuple[int, int]:
    roots = range(2) if data["smoke"] else range(10)
    checkpoints = (1200, 2400) if data["smoke"] else FULL_CHECKPOINTS
    expected = {(case, arm, root) for case in CASES for arm in ARMS for root in roots}
    rows = data["completed_trials"]
    if len(rows) != len(expected) or {(x["case"], x["arm"], x["root"]) for x in rows} != expected:
        raise ValueError("training trial coverage")
    scanned = replayed = 0
    for item in tqdm(rows, desc="SE1 TRAIN audit", leave=False):
        case, arm, root = item["case"], item["arm"], item["root"]
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        trial = run / "trials" / case / arm / str(root)
        expected_bank = "engineering" if data["smoke"] else "train"
        trace_indices = set(range(checkpoints[-1] // 48)) if data["smoke"] else TRACE_EPISODES
        n = 0
        for row in replay.lines(trial / "train.jsonl"):
            seed_bank = "engineering_train" if data["smoke"] else "train"
            seed = physical(seed_bank, case, arm, root, n)
            if (row["case"], row["arm"], row["root"], row["bank"], row["episode_index"], row["seed"]) != (case, arm, root, expected_bank, n, seed):
                raise ValueError("TRAIN metadata/seed")
            if not np.isclose(row["H_cost"] + row["tail_cost"], row["total_cost"]):
                raise ValueError("TRAIN H+tail")
            if "trace" in row:
                if n not in trace_indices:
                    raise ValueError("unlocked TRAIN trace")
                replay.physical_replay(row, cfg, seed, universe)
                replayed += 1
            elif n in trace_indices:
                raise ValueError("missing TRAIN trace")
            n += 1
        if n != checkpoints[-1] // 48 or item["episodes"] != n or item["samples"] != checkpoints[-1]:
            raise ValueError("TRAIN episode count")
        previous_elapsed = -1.0
        for cp in checkpoints:
            meta = json.loads((trial / f"checkpoint-{cp}.json").read_text())
            if meta["weights_sha256"] != sha(trial / f"checkpoint-{cp}.pt"):
                raise ValueError("checkpoint hash")
            if meta["samples"] != cp or meta["episodes"] != cp // 48 or meta["updates"] != cp // 1200:
                raise ValueError("checkpoint counters")
            elapsed = meta["elapsed_training_seconds"]
            phase = meta["phase_seconds"]
            if not math.isfinite(elapsed) or elapsed <= previous_elapsed:
                raise ValueError("nonmonotonic elapsed timing")
            if any(not math.isfinite(v) or v < 0 for v in phase.values()):
                raise ValueError("phase timing")
            if sum(phase.values()) > elapsed + max(.02, elapsed * .01):
                raise ValueError("phase time exceeds elapsed")
            previous_elapsed = elapsed
        scanned += n
    return scanned, replayed


def audit_evaluations(run: Path, data: dict) -> int:
    roots = range(2) if data["smoke"] else range(10)
    checkpoints = (1200, 2400) if data["smoke"] else FULL_CHECKPOINTS
    combos = tuple((cp, "greedy") for cp in checkpoints) + tuple(
        (cp, "stochastic") for cp in ((2400,) if data["smoke"] else (96000, 384000)))
    count = 3 if data["smoke"] else 128
    bank = "engineering" if data["smoke"] else "eval"
    expected = {(case, arm, root, cp, mode) for case in CASES for arm in ARMS
                for root in roots for cp, mode in combos}
    rows = data["completed_evaluations"]
    indexed = {(x["case"], x["arm"], x["root"], x["checkpoint"], x["mode"]): x for x in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("evaluation coverage/duplicates")
    replay.action_seed = action_seed
    total = 0
    for key in tqdm(sorted(expected), desc="SE1 evaluation replay", leave=False):
        case, arm, root, cp, mode = key
        item = indexed[key]
        reps = 4 if mode == "stochastic" else 1
        if item["bank"] != bank or item["physical_seeds"] != count or item["action_repetitions"] != reps:
            raise ValueError("evaluation budget")
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        trial = run / "trials" / case / arm / str(root)
        _, _, policy = load_policy(trial, cp)
        digest = sha(trial / f"checkpoint-{cp}.pt")
        costs = []
        n = 0
        for row in replay.lines(ROOT / item["file"]):
            i, rep = divmod(n, reps)
            if i >= count:
                raise ValueError("extra evaluation row")
            expected_seed = physical(bank, case, None, 0, i)
            if (row["case"], row["arm"], row["root"], row["checkpoint"], row["mode"],
                row["bank"], row["physical_index"], row["action_repetition"], row["seed"],
                row["checkpoint_sha256"]) != (case, arm, root, cp, mode, bank, i, rep, expected_seed, digest):
                raise ValueError("evaluation metadata/seed")
            replay.physical_replay(row, cfg, expected_seed, universe)
            replay.policy_replay(row, cfg, universe, policy)
            costs.append(row["total_cost"])
            n += 1
        if n != count * reps or not np.isclose(np.mean(costs), item["mean_cost"]):
            raise ValueError("evaluation rows/mean")
        total += n
    return total


def audit_controls(run: Path, data: dict) -> int:
    bank = "engineering" if data["smoke"] else "eval"
    count = 3 if data["smoke"] else 128
    names = ("threshold5", "risk", "corrective", "wait")
    expected = {(case, name) for case in CASES for name in names}
    indexed = {(x["case"], x["arm"]): x for x in data["completed_controls"]}
    if len(indexed) != len(expected) or set(indexed) != expected:
        raise ValueError("control coverage")
    total = 0
    for case, name in sorted(expected):
        item = indexed[(case, name)]
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        costs = []
        for i, row in enumerate(replay.lines(run / "controls" / bank / case / name / "episodes.jsonl")):
            if i >= count:
                raise ValueError("extra control row")
            replay.physical_replay(row, cfg, physical(bank, case, None, 0, i), universe)
            costs.append(row["total_cost"])
        if len(costs) != count or not np.isclose(np.mean(costs), item["mean_cost"]):
            raise ValueError("control rows/mean")
        total += count
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run_dir.resolve()
    out = run / ("audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_run": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "auditor_sha256": sha(Path(__file__))}
    try:
        data = check_manifest(run)
        report["smoke"] = data["smoke"]
        report["calibration_episodes_replayed"] = audit_calibration(run, data)
        report["training_rows_scanned"], report["training_episodes_replayed"] = audit_training(run, data)
        report["evaluation_episodes_replayed"] = audit_evaluations(run, data)
        report["control_episodes_replayed"] = audit_controls(run, data)
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["failure"] = repr(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(out)


if __name__ == "__main__":
    main()
