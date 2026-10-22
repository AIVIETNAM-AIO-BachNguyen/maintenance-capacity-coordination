"""Bounded V14 training validation and replay, using only this local archive."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_framework_v14_campaign as campaign
from ht_pdm_fjsp.maintenance_framework_v1 import CASES, configs, evaluate_episode


def save(destination, value):
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def training_smoke(output, deadline):
    registry, groups = campaign.seed_registry()
    seed = groups[2][0]
    result = []
    for arm in tqdm(campaign.ARMS, desc="V14 original training smoke"):
        actor = campaign.actor_new(seed, arm)
        initial = campaign.actor_digest(actor)
        records, coverage, tape = campaign.collect(actor, seed, 1, 2, 3, deadline)
        if not records:
            raise RuntimeError("no training states")
        values = []
        transitions = 0
        for record in tqdm(records, desc=f"V14 {arm} Q bank", leave=False):
            q_values, count, _ = campaign.counterfactual_values(
                actor, record, registry["forecast_seed"], seed, 1, 2, deadline)
            values.append(q_values)
            transitions += count
        update = campaign.update_actor(actor, records, values, arm, seed, 1, deadline)
        if update["updates"] < 1 or not all(torch.isfinite(parameter).all() for parameter in actor.parameters()):
            raise RuntimeError("V14 optimizer smoke failed")
        checkpoint = output / arm / "model.pt"
        campaign.checkpoint(actor, checkpoint, {"arm": arm, "training_seed": seed, "iteration": 1})
        restored = campaign.load_checkpoint(checkpoint)
        if campaign.actor_digest(actor) != campaign.actor_digest(restored):
            raise RuntimeError("V14 checkpoint roundtrip failed")
        episodes = []
        for case in CASES:
            deadline()
            episode = evaluate_episode(configs()[case], case, groups[2][1], campaign.V14Controller(restored, arm))
            if episode["invalid_assignments"] or episode["episode_return"] != -episode["objective"]:
                raise RuntimeError("V14 deployment/accounting failed")
            episodes.append(episode)
        result.append({"arm": arm, "initial_digest": initial, "final_digest": campaign.actor_digest(actor),
                       "source_episodes": len(tape), "selected_states": len(records), "coverage": coverage,
                       "forecast_transitions": transitions, "update": update, "episodes": episodes})
        save(output / "training_smoke.json", result)
    if result[0]["initial_digest"] != result[1]["initial_digest"]:
        raise RuntimeError("paired initial weights differ")
    return {"arms": len(result), "selected_states": sum(item["selected_states"] for item in result),
            "replayed_episodes": 8, "checkpoint_roundtrip": "PASS"}


def replay(root, output, deadline):
    assets = root / "baselines/v14"
    selection = json.loads((assets / "selection.json").read_text())
    historical = Path(__file__).resolve().parent / "research_result/framework_v14_dev_20261020T185303Z"
    with (historical / "episodes.csv").open() as stream:
        reference = {(row["controller"], row["train_seed"], row["case"], int(row["seed"])): row
                     for row in csv.DictReader(stream) if row["controller"] in campaign.ARMS}
    registry = json.loads((assets / "seed_registry.json").read_text())
    panel = registry["development"]
    seeds = range(panel["start"], panel["start"] + panel["count"])
    grid = [(arm, seed, item, case, physical) for arm, items in selection.items()
            for seed, item in items.items() for case in CASES for physical in seeds]
    actors = {}
    episodes = []
    mismatches = []
    numeric_keys = ("objective", "H_cost", "tail_cost", "H_availability", "H_failures",
                    "queued_pm_ticks", "queued_cm_ticks", "episode_return", "invalid_assignments")
    for arm, seed, item, case, physical in tqdm(grid, desc="Frozen V14 historical replay", unit="episode"):
        deadline()
        key = (arm, seed)
        if key not in actors:
            checkpoint = assets / item["checkpoint"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("selected V14 checkpoint hash mismatch")
            actors[key] = campaign.load_checkpoint(checkpoint)
        row = evaluate_episode(configs()[case], case, physical, campaign.V14Controller(actors[key], arm))
        expected = reference[(arm, seed, case, physical)]
        differences = {}
        for name in numeric_keys:
            if not np.isclose(row[name], float(expected[name]), rtol=0, atol=1e-9):
                differences[name] = {"current": row[name], "historical": float(expected[name])}
        if differences:
            mismatches.append({"arm": arm, "training_seed": int(seed), "case": case,
                               "physical_seed": physical, "differences": differences})
        row["train_seed"] = int(seed)
        episodes.append(row)
    save(output / "episodes.json", episodes)
    save(output / "historical_mismatches.json", mismatches)
    means = {arm: float(np.mean([row["objective"] for row in episodes if row["controller"] == arm]))
             for arm in campaign.ARMS}
    return {"replayed_episodes": len(episodes), "checkpoints": len(actors),
            "historical_metrics": "PASS" if not mismatches else "MISMATCH",
            "historical_mismatch_episodes": len(mismatches), "mean_cost": means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("evaluate", "training-smoke", "tests", "audit", "campaign"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", default="smoke", choices=("smoke", "dev"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.root / "artifacts" / ("v14_" + args.operation + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "RUNNING", "operation": args.operation, "scientific_scope": "historical reproduction/engineering only"}
    save(output / "manifest.json", manifest)
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 180:
            raise TimeoutError("180-second bounded V14 operation exceeded")

    try:
        if args.operation == "training-smoke":
            manifest.update(training_smoke(output, deadline))
        elif args.operation == "evaluate":
            manifest.update(replay(args.root, output, deadline))
        elif args.operation == "tests":
            subprocess.run([sys.executable, "-m", "pytest", "-c", "/dev/null", "-p", "no:cacheprovider", "-q", "tests/test_framework_v14_capacity_control.py"], check=True, timeout=180)
        elif args.operation == "audit":
            with (output / "console.log").open("w") as log:
                subprocess.run([sys.executable, "research_result/verify_framework_v14.py", "--run-dir",
                                "research_result/framework_v14_dev_20261020T185303Z", "--output-dir", str(output / "audit")],
                               check=True, timeout=300, stdout=log, stderr=subprocess.STDOUT)
        else:
            subprocess.run([sys.executable, "-m", "ht_pdm_fjsp.maintenance_framework_v14_campaign",
                            "--profile", args.profile, "--output-dir", str(output / "campaign")], check=True,
                           timeout=1250 if args.profile == "smoke" else 14450)
        manifest["status"] = "COMPLETED"
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        save(output / "manifest.json", manifest)
        print(output)


if __name__ == "__main__":
    main()
