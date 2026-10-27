"""Independent raw-row provenance, physical and policy replay for L3 S/F/A."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import jcst_learnability_audit_l2 as replay
from jcst_learnability_l2 import action_universe, case_config, load_policy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/jcst_matched_architecture_protocol_l3.md"
ARTIFACTS = ROOT / "artifacts/jcst_matched_architecture"
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
CHECKPOINTS = (24000, 96000, 384000)
COMBINATIONS = ((24000, "greedy"), (96000, "greedy"),
                (384000, "greedy"), (384000, "stochastic"))
TRACE_EPISODES = {0, 1, 2, 3, 499, 1999, 7997, 7998, 7999}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_physical(bank: str, case: str, arm: str, root: int, episode: int) -> int:
    ci = CASES.index(case)
    if bank == "train":
        return (2**52 + 10**13 + ci * 10**12 + ARMS.index(arm) * 10**11
                + root * 10**8 + episode)
    return 2**52 + (2 if bank == "eval" else 3) * 10**13 + ci * 10**12 + episode


def independent_action_seed(bank: str, case: str, arm: str, root: int,
                            checkpoint: int, physical_index: int, repetition: int) -> int:
    assert bank == "eval"
    key = ':'.join(map(str, ("JCST-L3", case, arm, root, checkpoint, physical_index, repetition)))
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def check_manifest(run: Path) -> dict:
    data = json.loads((run / "manifest.json").read_text())
    if data["schema"] != "JCST-matched-architecture-L3" or data["status"] != "COMPLETE":
        raise ValueError("run incomplete or wrong schema")
    if data["protocol_hash"] != sha(PROTOCOL):
        raise ValueError("protocol hash")
    if data["contract_hash"] != sha(ROOT / "docs/joint_component_service_contract_v1.md"):
        raise ValueError("contract hash")
    for name, digest in data["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source hash: {name}")
    if data["parameters"]["H"] != 48 or data["parameters"]["B"] != 16:
        raise ValueError("horizon")
    if data["seed_roles"]["physical_base"] != 2**52:
        raise ValueError("seed base")
    override = data.get("parallel_l2_override")
    if data["stage"] == "collect":
        if override is None and data.get("l2_closure") is None:
            raise ValueError("missing L2 scheduling authorization")
        if override is not None:
            amendment = ROOT / override["amendment"]
            if (override["amendment_sha256"] != sha(amendment) or
                    override["l2_status_at_launch"] != "RUNNING" or
                    override["concurrent_scientific_cpu_workers_cap"] != 2):
                raise ValueError("parallel amendment provenance")
    return data


def audit_training(run: Path, data: dict) -> tuple[int, int]:
    smoke = data["stage"] == "smoke"
    roots = range(1) if smoke else range(10)
    checkpoints = (48, 96) if smoke else CHECKPOINTS
    expected = {(case, arm, root) for case in CASES for arm in ARMS for root in roots}
    if {(x["case"], x["arm"], x["root"]) for x in data["completed_trials"]} != expected:
        raise ValueError("training coverage")
    if len(data["completed_trials"]) != len(expected):
        raise ValueError("duplicate training trial")
    scanned = replayed = 0
    for item in tqdm(data["completed_trials"], desc="L3 TRAIN replay"):
        case, arm, root = item["case"], item["arm"], item["root"]
        cfg, _ = case_config(case)
        universe = action_universe(cfg)
        trial = run / "trials" / case / arm / str(root)
        bank = "engineering" if smoke else "train"
        n = 0
        for row in replay.lines(trial / "train.jsonl"):
            seed = expected_physical(bank, case, arm, root, n)
            if (row["case"], row["arm"], row["root"], row["bank"], row["episode_index"], row["seed"]) != (case, arm, root, bank, n, seed):
                raise ValueError("TRAIN metadata or seed")
            if not np.isclose(row["H_cost"] + row["tail_cost"], row["total_cost"]):
                raise ValueError("TRAIN H/tail")
            if "trace" in row:
                if not smoke and n not in TRACE_EPISODES:
                    raise ValueError("unlocked TRAIN trace sample")
                replay.physical_replay(row, cfg, seed, universe)
                replayed += 1
            elif not smoke and n in TRACE_EPISODES:
                raise ValueError("missing locked TRAIN trace")
            n += 1
        if n != checkpoints[-1] // 48 or item["episodes"] != n or item["samples"] != checkpoints[-1]:
            raise ValueError("TRAIN length")
        for cp in checkpoints:
            meta = json.loads((trial / f"checkpoint-{cp}.json").read_text())
            if meta["weights_sha256"] != sha(trial / f"checkpoint-{cp}.pt"):
                raise ValueError("checkpoint hash")
        scanned += n
    return scanned, replayed


def audit_evaluations(run: Path, data: dict) -> int:
    smoke = data["stage"] == "smoke"
    roots = range(1) if smoke else range(10)
    combos = ((96, "greedy"),) if smoke else COMBINATIONS
    count = 2 if smoke else 128
    bank = "engineering" if smoke else "eval"
    expected = {(case, arm, root, cp, mode) for case in CASES for arm in ARMS
                for root in roots for cp, mode in combos}
    rows = {(x["case"], x["arm"], x["root"], x["checkpoint"], x["mode"]): x
            for x in data["completed_evaluations"]}
    if set(rows) != expected or len(rows) != len(data["completed_evaluations"]):
        raise ValueError("evaluation coverage or duplicate")
    replay.action_seed = independent_action_seed
    total = 0
    for key in tqdm(sorted(expected), desc="L3 evaluation replay"):
        case, arm, root, cp, mode = key
        item = rows[key]
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
            seed = expected_physical(bank, case, arm, root, i)
            if (row["case"], row["arm"], row["root"], row["checkpoint"], row["mode"],
                row["bank"], row["physical_index"], row["action_repetition"], row["seed"],
                row["checkpoint_sha256"]) != (case, arm, root, cp, mode, bank, i, rep, seed, digest):
                raise ValueError("evaluation provenance")
            replay.physical_replay(row, cfg, seed, universe)
            replay.policy_replay(row, cfg, universe, policy)
            costs.append(row["total_cost"])
            n += 1
        if n != count * reps or not np.isclose(np.mean(costs), item["mean_cost"]):
            raise ValueError("evaluation length/mean")
        total += n
    return total


def audit_controls(run: Path, data: dict) -> int:
    smoke = data["stage"] == "smoke"
    bank = "engineering" if smoke else "eval"
    count = 2 if smoke else 128
    names = ("threshold5", "risk", "corrective", "wait")
    expected = {(case, name) for case in CASES for name in names}
    rows = {(x["case"], x["arm"]): x for x in data["completed_controls"]}
    if set(rows) != expected or len(rows) != len(data["completed_controls"]):
        raise ValueError("control coverage")
    total = 0
    for case, name in sorted(expected):
        item = rows[(case, name)]
        if item["bank"] != bank or item["physical_seeds"] != count:
            raise ValueError("control budget")
        cfg, _ = case_config(case)
        costs = []
        for i, row in enumerate(replay.lines(run / "controls" / bank / case / name / "episodes.jsonl")):
            if i >= count:
                raise ValueError("extra control row")
            replay.physical_replay(row, cfg, expected_physical(bank, case, name, 0, i))
            costs.append(row["total_cost"])
        if len(costs) != count or not np.isclose(np.mean(costs), item["mean_cost"]):
            raise ValueError("control length/mean")
        total += len(costs)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run_dir.resolve()
    out = ARTIFACTS / ("audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_run": str(run.relative_to(ROOT)),
              "source_manifest_sha256": sha(run / "manifest.json"),
              "auditor_sha256": sha(Path(__file__))}
    try:
        data = check_manifest(run)
        report["stage"] = data["stage"]
        report["training_rows_scanned"], report["training_episodes_replayed"] = audit_training(run, data)
        report["evaluation_episodes_replayed"] = audit_evaluations(run, data)
        report["control_episodes_replayed"] = audit_controls(run, data)
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
