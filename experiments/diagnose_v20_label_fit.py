"""Descriptive in-sample fit to v20 training-only matching-value labels.

This cannot estimate out-of-bank generalization or justify new checkpoint selection.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from experiments import same_tick_pilot_v19 as v19
from experiments import same_tick_pilot_v20 as v20
from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss, tensorize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pair_reliability(case_rows: list[dict]) -> list[dict]:
    """Exploratory second-half sign agreement binned only by first-half gaps."""
    bins = [(0, 10), (10, 20), (20, 40), (40, float("inf"))]
    observations = {lo: [] for lo, _ in bins}
    for row in case_rows:
        costs = np.asarray([[h + t for h, t in zip(branch["H_costs"], branch["tail_costs"])]
                            for branch in row["branches"]], dtype=float)
        for i, j in itertools.combinations(range(len(costs)), 2):
            difference = costs[i] - costs[j]
            first = float(difference[:8].mean())
            second = float(difference[8:].mean())
            gap = abs(first)
            agreement = bool(np.sign(first) == np.sign(second) and np.sign(first) != 0)
            lo = next(lo for lo, hi in bins if lo <= gap < hi)
            observations[lo].append(agreement)
    return [{"first_half_gap_lower": lo, "first_half_gap_upper": hi if math.isfinite(hi) else None,
             "pairs": len(observations[lo]),
             "second_half_sign_agreement": float(np.mean(observations[lo])) if observations[lo] else None}
            for lo, hi in bins]


def main() -> None:
    torch.set_num_threads(1)
    labels = Path("artifacts/recent_policy_v20_labels_20261024T113110729833Z")
    training = Path("artifacts/same_tick_v20_train_20261024T155517615436Z")
    assert json.loads((labels / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((training / "audit.json").read_text())["status"] == "PASS"
    rows = [json.loads(x) for x in (labels / "labels.jsonl").read_text().splitlines()]
    out = Path("artifacts") / ("v20_label_fit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for case in v20.CASES:
            case_rows = [row for row in rows if row["case"] == case]
            data = tensorize(case_rows)
            targets = data["targets"]
            entropy = float(-(targets * targets.clamp_min(1e-12).log()).sum(1).mean())
            uniform = float(np.mean([math.log(len(row["matching_indices"])) for row in case_rows]))
            for arm in tqdm(("C20", "D20", "C19", "D19"), desc=f"v20 label fit {case}"):
                for root in v20.TRAIN_ROOTS:
                    if arm in ("C20", "D20"):
                        trial = training / f"{case}_{arm}_{root}"
                        selection = json.loads((trial / "selection.json").read_text())
                        checkpoints = (("pretrain", trial / "pretrain_checkpoint.pt"),
                                       ("selected", trial / selection["checkpoint"]))
                    else:
                        trial = training / "references" / f"{case}_{arm}_{root}"
                        selection = json.loads((trial / "selection.json").read_text())
                        checkpoints = (("selected", Path(selection["checkpoint"])),)
                    for stage, checkpoint in checkpoints:
                        if stage == "selected":
                            assert digest(checkpoint) == selection["checkpoint_sha256"]
                        policy = v20.load_policy(checkpoint) if arm in ("C20", "D20") else v19.load_policy(checkpoint)
                        with torch.no_grad():
                            scores = candidate_logits(policy, data)
                            loss = float(listwise_loss(policy, data))
                            agreement = float((scores.argmax(1) == targets.argmax(1)).float().mean())
                        record = {
                            "case": case, "arm": arm, "root": root, "stage": stage,
                            "checkpoint_sha256": digest(checkpoint), "states": len(case_rows),
                            "cross_entropy": loss, "top1_agreement": agreement,
                            "target_entropy": entropy, "uniform_cross_entropy": uniform,
                        }
                        records.append(record)
                        with (out / "rows.jsonl").open("a") as target:
                            target.write(json.dumps(record, sort_keys=True) + "\n")
        summary = {"status": "COMPLETED", "rows": len(records),
                   "labels_sha256": digest(labels / "labels.jsonl"),
                   "training_manifest_sha256": digest(training / "manifest.json"),
                   "script_sha256": digest(Path(__file__)),
                   "by_case": {case: {arm: {stage: {
                       "cross_entropy": float(np.mean([x["cross_entropy"] for x in records if x["case"] == case and x["arm"] == arm and x["stage"] == stage])),
                       "top1_agreement": float(np.mean([x["top1_agreement"] for x in records if x["case"] == case and x["arm"] == arm and x["stage"] == stage])),
                   } for stage in (("pretrain", "selected") if arm in ("C20", "D20") else ("selected",))}
                   for arm in ("C20", "D20", "C19", "D19")}
                   for case in v20.CASES},
                   "baselines": {case: {
                       "target_entropy": next(x["target_entropy"] for x in records if x["case"] == case),
                       "uniform_cross_entropy": next(x["uniform_cross_entropy"] for x in records if x["case"] == case),
                   } for case in v20.CASES},
                   "exploratory_pair_reliability": {
                       case: pair_reliability([row for row in rows if row["case"] == case])
                       for case in v20.CASES},
                   "limitation": "In-sample training-label fit and exploratory tape-gap bins only; no out-of-bank ranking or physical-cost estimate."}
        save(out / "summary.json", summary)
    except Exception as exc:
        save(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise
    print(out.resolve())


if __name__ == "__main__":
    main()
