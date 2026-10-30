"""Post-pilot in-sample label-fit diagnostic; never changes v17 gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments import same_tick_pilot_v17 as pilot
from experiments.full_matching_pretrain_v17 import tensorize, candidate_logits, listwise_loss

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / "artifacts/full_matching_v17_labels_20261023T080338842451Z"
TRAIN = ROOT / "artifacts/same_tick_v17_train_20261023T144446781419Z"
HELDOUT = ROOT / "artifacts/same_tick_v17_heldout_20261023T154122079855Z"


def main():
    torch.set_num_threads(1)
    for run in (LABEL, TRAIN, HELDOUT):
        assert json.loads((run / "manifest.json").read_text())["status"] == "COMPLETED"
        assert json.loads((run / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((HELDOUT / "replication_go_decision.json").read_text())["go"] is False
    original = json.loads((TRAIN / "manifest.json").read_text())
    assert original["source_sha256"] == json.loads((HELDOUT / "manifest.json").read_text())["source_sha256"]
    for path, expected in original["source_sha256"].items():
        assert pilot.sha256(ROOT / path) == expected
    labels = [json.loads(line) for line in (LABEL / "labels.jsonl").open()]
    grouped = {case: [row for row in labels if row["case"] == case] for case in pilot.CASES}
    output = ROOT / "artifacts" / ("v17_label_fit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    records = []
    try:
        for case in pilot.CASES:
            data = tensorize(grouped[case])
            target = data["targets"]
            entropy = float(-(target * target.clamp_min(1e-12).log()).sum(1).mean())
            uniform_ce = float(np.mean([math.log(len(row["matching_indices"])) for row in grouped[case]]))
            for arm in tqdm(pilot.ARMS, desc=f"v17 label fit {case}", leave=False):
                for root in pilot.TRAIN_ROOTS:
                    trial = TRAIN / f"{case}_{arm}_{root}"
                    selection = json.loads((trial / "selection.json").read_text())
                    for stage, path in (
                        ("pretrain", trial / "pretrain_checkpoint.pt"),
                        ("selected", trial / selection["checkpoint"]),
                    ):
                        policy = pilot.load_policy(path)
                        with torch.no_grad():
                            scores = candidate_logits(policy, data)
                            loss = float(listwise_loss(policy, data))
                            top1 = float((scores.argmax(1) == target.argmax(1)).float().mean())
                        row = {"case": case, "arm": arm, "train_root": root, "stage": stage,
                               "checkpoint_sha256": pilot.sha256(path),
                               "training_records": len(grouped[case]),
                               "listwise_cross_entropy": loss, "top1_match_rate": top1,
                               "target_entropy": entropy, "uniform_cross_entropy": uniform_ce}
                        pilot.append_jsonl(output / "rows.jsonl", row)
                        records.append(row)
        summary = {arm: {stage: {
            "mean_cross_entropy": float(np.mean([r["listwise_cross_entropy"] for r in records
                if r["arm"] == arm and r["stage"] == stage])),
            "mean_top1_match_rate": float(np.mean([r["top1_match_rate"] for r in records
                if r["arm"] == arm and r["stage"] == stage])),
        } for stage in ("pretrain", "selected")} for arm in pilot.ARMS}
        summary["label_baselines"] = {
            "target_entropy_equal_case": float(np.mean([
                next(r["target_entropy"] for r in records if r["case"] == case)
                for case in pilot.CASES])),
            "uniform_cross_entropy_equal_case": float(np.mean([
                next(r["uniform_cross_entropy"] for r in records if r["case"] == case)
                for case in pilot.CASES])),
        }
        pilot.write_json(output / "summary.json", summary)
        status = "COMPLETED"
    except Exception:
        status = "FAILED"
        raise
    finally:
        pilot.write_json(output / "manifest.json", {
            "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "analysis_source_sha256": pilot.sha256(Path(__file__)),
            "label_manifest_sha256": pilot.sha256(LABEL / "manifest.json"),
            "label_audit_sha256": pilot.sha256(LABEL / "audit.json"),
            "train_manifest_sha256": pilot.sha256(TRAIN / "manifest.json"),
            "train_audit_sha256": pilot.sha256(TRAIN / "audit.json"),
            "heldout_manifest_sha256": pilot.sha256(HELDOUT / "manifest.json"),
            "heldout_audit_sha256": pilot.sha256(HELDOUT / "audit.json"),
            "protocol_sha256": pilot.sha256(pilot.PROTOCOL),
            "interpretation": "in-sample descriptive training-label fit, no model selection, generalization or gate change",
            "output_sha256": {p.name: pilot.sha256(p) for p in output.iterdir()
                              if p.is_file() and p.name != "manifest.json"},
        })
    print(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
