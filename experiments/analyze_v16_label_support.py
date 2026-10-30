"""Descriptive coverage of v15 training labels versus audited C16 decisions.

No policy is trained or selected, no future events are read, and this analysis
does not change the v16 gate. It compares only observable support dimensions.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments import same_tick_pilot_v16 as pilot


ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / "artifacts/conditional_value_v15_labels_20261023T032931104204Z"
TRAIN = ROOT / "artifacts/same_tick_v16_train_20261023T055022317165Z"
HELDOUT = ROOT / "artifacts/same_tick_v16_heldout_20261023T062526347509Z"


def rows(path):
    return [json.loads(line) for line in path.open()]


def main():
    assert json.loads((LABEL / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((TRAIN / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((HELDOUT / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((HELDOUT / "replication_go_decision.json").read_text())["go"] is False
    labels = rows(LABEL / "labels.jsonl")
    decisions = rows(HELDOUT / "heldout_decisions.jsonl")
    output = ROOT / "artifacts" / ("v16_label_support_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    by_case = {}
    for case in pilot.CASES:
        lab = [r for r in labels if r["case"] == case]
        c = [r for r in decisions if r["case"] == case and r["arm"] == "C16"]
        second = [r for r in c if len(r["sequence"]) >= 2]
        first_label = Counter(int(r["first_edge"]) for r in lab)
        first_c = Counter(int(r["sequence"][0]) for r in second)
        assert len(c) == len(pilot.HELDOUT_SEEDS) * len(pilot.TRAIN_ROOTS) * 48
        by_case[case] = {
            "label_states": len(lab),
            "label_first_edge_counts": dict(sorted(first_label.items())),
            "label_tick_min_max": [min(r["tick"] for r in lab), max(r["tick"] for r in lab)],
            "C16_H_decisions": len(c),
            "C16_second_token_opportunities": len(second),
            "C16_second_first_edge_counts": dict(sorted(first_c.items())),
            "C16_second_first_edge_unseen_by_labels": sum(n for edge, n in first_c.items() if edge not in first_label),
            "C16_second_outside_label_tick_window_8_39": sum(not 8 <= r["tick"] <= 39 for r in second),
            "C16_second_STOP": sum(r["sequence"][1] == 15 for r in second),
            "C16_third_token": sum(len(r["sequence"]) >= 3 for r in second),
        }
    summary = {
        "interpretation": "Descriptive support on v15 training labels and C16-visited held-out states; no economic effect or generalization estimate",
        "by_case": by_case,
        "totals": {name: sum(v[name] for v in by_case.values()) for name in (
            "label_states", "C16_H_decisions", "C16_second_token_opportunities",
            "C16_second_first_edge_unseen_by_labels", "C16_second_outside_label_tick_window_8_39",
            "C16_second_STOP", "C16_third_token")},
    }
    pilot.write_json(output / "summary.json", summary)
    pilot.write_json(output / "manifest.json", {
        "status": "COMPLETED", "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": pilot.sha256(Path(__file__)),
        "label_manifest_sha256": pilot.sha256(LABEL / "manifest.json"),
        "label_audit_sha256": pilot.sha256(LABEL / "audit.json"),
        "train_manifest_sha256": pilot.sha256(TRAIN / "manifest.json"),
        "train_audit_sha256": pilot.sha256(TRAIN / "audit.json"),
        "heldout_manifest_sha256": pilot.sha256(HELDOUT / "manifest.json"),
        "heldout_audit_sha256": pilot.sha256(HELDOUT / "audit.json"),
        "protocol_sha256": pilot.sha256(pilot.PROTOCOL),
        "summary_sha256": pilot.sha256(output / "summary.json"),
    })
    print(output.resolve())
    print(json.dumps(summary["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
