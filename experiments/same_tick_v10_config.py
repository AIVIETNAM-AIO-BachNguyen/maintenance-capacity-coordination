"""Versioned high-demand N4/K2 simulator configurations and same-panel controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import RuleController, evaluate_episode
from ht_pdm_fjsp.maintenance_request_policies import family_configs
from maintenance_capacity_coordination.baselines import V14Controller

ROOT = Path(__file__).resolve().parents[1]
CASES = ("reciprocal_age3_scale1", "specialist_generalist_age3_scale1")
BASE_CASES = {"reciprocal_age3_scale1": "reciprocal_scale1",
              "specialist_generalist_age3_scale1": "specialist_generalist_scale1"}
V14_ROOTS = (72600000, 72600001, 72600002, 72600003, 72600004)


def configs():
    family = family_configs("full")
    return {case: replace(family[base], failure_age=3)
            for case, base in BASE_CASES.items()}


def evaluate_controls(run, seeds, *, panel):
    config_by_case = configs()
    rows = []
    destination = run / f"{panel}_controls.jsonl"
    for case in CASES:
        for name in ("risk_matching", "rollout"):
            for seed in tqdm(seeds, desc=f"v10 control {case} {name}"):
                controller = RuleController(name)
                row = {"panel": panel, "arm": "F_" + name,
                       **evaluate_episode(config_by_case[case], case, seed, controller)}
                rows.append(row)
                with destination.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        for train_seed in V14_ROOTS:
            directory = ROOT / f"baselines/v14/checkpoints/count_calibrated/{train_seed}"
            checkpoints = list(directory.glob("*.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"expected one V14 checkpoint in {directory}")
            for seed in tqdm(seeds, desc=f"v10 control {case} V14 {train_seed}"):
                controller = V14Controller.from_checkpoint(checkpoints[0])
                row = {"panel": panel, "arm": "A", "historical_train_seed": train_seed,
                       **evaluate_episode(config_by_case[case], case, seed, controller)}
                rows.append(row)
                with destination.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return rows
