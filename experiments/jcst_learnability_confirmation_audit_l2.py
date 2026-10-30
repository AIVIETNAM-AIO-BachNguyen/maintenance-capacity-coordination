"""Independent replay and provenance audit for fresh L2 confirmation/replication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jcst_learnability_audit_l2 import lines, physical_replay, policy_replay
from jcst_learnability_l2 import (ARMS, CASES, ROOT, action_universe, case_config,
                                  load_policy, physical_seed, sha)


def check_manifest(run: Path, train: Path, selection_file: Path) -> tuple[dict, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    source = json.loads((train / "manifest.json").read_text())
    selection = json.loads(selection_file.read_text())
    if manifest["status"] != "COMPLETE" or manifest["stage"] not in ("confirmation", "replication"):
        raise ValueError("incomplete/wrong-stage run")
    if source["status"] != "COMPLETE" or source["stage"] != "train_dev":
        raise ValueError("incomplete TRAIN/DEV source")
    if (manifest["source_train_dir"] != str(train.relative_to(ROOT)) or
            selection["status"] != "AUDIT_PASS" or
            selection["source_manifest_hash"] != sha(train / "manifest.json")):
        raise ValueError("selection/source provenance")
    if not selection["dev_eligible"]:
        raise ValueError("DEV gate did not open confirmation")
    if manifest["source_hashes"] != source["source_hashes"] or manifest["protocol_hash"] != source["protocol_hash"]:
        raise ValueError("source/protocol changed after TRAIN/DEV")
    for name, digest in manifest["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source hash changed: {name}")
    if manifest["stage"] == "replication":
        if len(manifest["completed_trials"]) != 30:
            raise ValueError("independent replication trial panel incomplete")
    elif manifest["completed_trials"]:
        raise ValueError("confirmation cannot train new roots")
    return manifest, selection


def audit_replication_training(run: Path, manifest: dict, selection: dict) -> tuple[int, int]:
    if manifest["stage"] != "replication":
        return 0, 0
    scanned = replayed = 0
    covered = set()
    for info in tqdm(manifest["completed_trials"], desc="replication TRAIN audit"):
        case, arm, root = info["case"], info["arm"], info["root"]
        if case not in CASES or arm not in ARMS or root not in range(5, 10):
            raise ValueError("replication role")
        if (case, arm, root) in covered:
            raise ValueError("duplicate replication trial")
        covered.add((case, arm, root))
        checkpoint = int(selection["selected"][case][arm]["checkpoint"])
        if info["samples"] != checkpoint or info["episodes"] != checkpoint // 48:
            raise ValueError("replication sample budget")
        config, _ = case_config(case)
        universe = action_universe(config)
        trial = run / "trials" / case / arm / str(root)
        seen = 0
        for row in lines(trial / "train.jsonl"):
            if row["episode_index"] != seen or row["bank"] != "rep_train":
                raise ValueError("replication TRAIN sequence/bank")
            physical = physical_seed("rep_train", case, root, seen)
            if row["seed"] != physical or row["case"] != case or row["arm"] != arm or row["root"] != root:
                raise ValueError("replication TRAIN seed/metadata")
            if row["H_cost"] + row["tail_cost"] != row["total_cost"]:
                raise ValueError("replication TRAIN cost")
            if "trace" in row:
                physical_replay(row, config, physical, universe)
                replayed += 1
            seen += 1
        if seen != info["episodes"]:
            raise ValueError("replication TRAIN rows incomplete")
        for cp in {24000, checkpoint}:
            meta = json.loads((trial / f"checkpoint-{cp}.json").read_text())
            if meta["weights_sha256"] != sha(trial / f"checkpoint-{cp}.pt"):
                raise ValueError("replication checkpoint hash")
        scanned += seen
    expected = {(case, arm, root) for case in CASES for arm in ARMS for root in range(5, 10)}
    if covered != expected:
        raise ValueError("replication root coverage")
    return scanned, replayed


def audit_models(run: Path, train: Path, manifest: dict, selection: dict) -> int:
    stage = manifest["stage"]
    bank = "confirmation" if stage == "confirmation" else "rep_eval"
    roots = range(5) if stage == "confirmation" else range(5, 10)
    model_source = train if stage == "confirmation" else run
    expected = {(case, arm, root, cp, selection["selected"][case][arm]["mode"])
                for case in CASES for arm in ARMS for root in roots
                for cp in {24000, int(selection["selected"][case][arm]["checkpoint"])}}
    seen_files = set()
    episodes = 0
    for info in tqdm(manifest["completed_evaluations"], desc="confirmation policy replay"):
        case, arm, root = info["case"], info["arm"], info["root"]
        cp, mode = info["checkpoint"], info["mode"]
        key = (case, arm, root, cp, mode)
        if key not in expected or key in seen_files or info["bank"] != bank:
            raise ValueError("evaluation selection/coverage")
        seen_files.add(key)
        repeats = 1 if mode == "greedy" else 4
        if info["physical_seeds"] != 128 or info["action_repetitions"] != repeats:
            raise ValueError("evaluation seed budget")
        config, _ = case_config(case)
        universe = action_universe(config)
        trial = model_source / "trials" / case / arm / str(root)
        _, _, policy = load_policy(trial, cp)
        checkpoint_hash = sha(trial / f"checkpoint-{cp}.pt")
        costs = []
        row_count = 0
        for row in lines(ROOT / info["file"]):
            i, rep = divmod(row_count, repeats)
            if i >= 128 or row["physical_index"] != i or row["action_repetition"] != rep:
                raise ValueError("evaluation row order/repetition")
            physical = physical_seed(bank, case, 0, i)
            if (row["seed"] != physical or row["checkpoint_sha256"] != checkpoint_hash or
                    row["case"] != case or row["arm"] != arm or row["root"] != root or
                    row["checkpoint"] != cp or row["mode"] != mode or row["bank"] != bank):
                raise ValueError("evaluation checkpoint/seed provenance")
            physical_replay(row, config, physical, universe)
            policy_replay(row, config, universe, policy)
            costs.append(row["total_cost"])
            row_count += 1
        if row_count != 128 * repeats or not np.isclose(np.mean(costs), info["mean_cost"]):
            raise ValueError("evaluation rows/summary")
        episodes += row_count
    if seen_files != expected:
        raise ValueError("missing evaluation combinations")
    return episodes


def audit_controls(run: Path, manifest: dict) -> int:
    bank = "confirmation" if manifest["stage"] == "confirmation" else "rep_eval"
    expected = {(case, name) for case in CASES for name in ("threshold5", "risk", "corrective", "wait")}
    covered = set()
    episodes = 0
    for info in manifest["completed_controls"]:
        case, arm = info["case"], info["arm"]
        if (case, arm) not in expected or (case, arm) in covered or info["bank"] != bank:
            raise ValueError("control coverage")
        covered.add((case, arm))
        config, _ = case_config(case)
        costs = []
        for i, row in enumerate(lines(run / "controls" / bank / case / arm / "episodes.jsonl")):
            if i >= 128:
                raise ValueError("extra control episode")
            physical_replay(row, config, physical_seed(bank, case, 0, i))
            costs.append(row["total_cost"])
        if len(costs) != 128 or not np.isclose(np.mean(costs), info["mean_cost"]):
            raise ValueError("control rows/summary")
        episodes += len(costs)
    if covered != expected:
        raise ValueError("missing controls")
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    run, train, selection_file = (path.resolve() for path in (args.run_dir, args.train_dir, args.selection))
    manifest, selection = check_manifest(run, train, selection_file)
    output = ROOT / "artifacts/jcst_learnability" / ("confirmation-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_run_dir": str(run.relative_to(ROOT)),
              "source_manifest_hash": sha(run / "manifest.json"),
              "training_source_manifest_hash": sha(train / "manifest.json"),
              "selection_hash": sha(selection_file), "audit_code_hash": sha(Path(__file__)),
              "stage": manifest["stage"]}
    try:
        scanned, replayed = audit_replication_training(run, manifest, selection)
        model_episodes = audit_models(run, train, manifest, selection)
        control_episodes = audit_controls(run, manifest)
        report.update(status="PASS", training_rows_scanned=scanned,
                      training_episodes_replayed=replayed,
                      learned_evaluation_episodes_replayed=model_episodes,
                      control_episodes_replayed=control_episodes)
    except BaseException as exc:
        report.update(status="FAIL", failure=repr(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
