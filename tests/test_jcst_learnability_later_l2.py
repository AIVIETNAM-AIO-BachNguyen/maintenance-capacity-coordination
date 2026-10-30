"""Engineering-only checks for preconfirmation provenance and statistical gates."""
from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import jcst_learnability_l2 as runner
from jcst_learnability_analysis_l2 import analyze
from jcst_learnability_stats_audit_l2 import verify
from jcst_learnability_stage3_l2 import load_lock
import jcst_learnability_preconfirmation_lock_l2 as prelock


def test_confirmation_manifest_uses_repo_relative_serializable_paths():
    root = Path(__file__).resolve().parents[1]
    args = Namespace(train_dir="artifacts/jcst_learnability/synthetic-train",
                     confirmation_dir=None)
    manifest = runner.manifest("confirmation", root / "artifacts/jcst_learnability/synthetic-confirmation", args)
    assert manifest["source_train_dir"] == args.train_dir
    json.dumps(manifest)


def test_independent_bootstrap_and_useful_learning_gate_agree_on_synthetic_panel():
    selection = {"selected": {case: {arm: {"checkpoint": 96000 if case == "base" and arm == "S" else 24000,
                                               "mode": "greedy"}
                                     for arm in runner.ARMS} for case in runner.CASES}}
    arrays = {}
    raw = {}
    for case in runner.CASES:
        for arm in runner.ARMS:
            low = np.full((5, 128), 1000.0)
            arrays[(case, arm, 24000)] = low
            if case == "base" and arm == "S":
                arrays[(case, arm, 96000)] = np.full((5, 128), 900.0)
            for root in range(5):
                raw[(case, arm, 24000, root)] = low[root]
                if case == "base" and arm == "S":
                    raw[(case, arm, 96000, root)] = arrays[(case, arm, 96000)][root]
        for name, cost in (("wait", 1100.0), ("threshold5", 800.0),
                           ("risk", 950.0), ("corrective", 1050.0)):
            arrays[(case, name)] = np.full(128, cost)
            raw[(case, name)] = arrays[(case, name)]
    preliminary = analyze(arrays, selection, "confirmation")
    assert len(preliminary["contrasts"]) == 18
    assert preliminary["useful_learning"]["base"]["S"]
    assert not preliminary["useful_learning"]["generalist"]["S"]
    assert preliminary["replication_eligible"]
    assert verify(preliminary, selection, raw, "confirmation")["status"] == "PASS"


def test_stage3_lock_accepts_only_frozen_source_and_selection(tmp_path):
    root = Path(__file__).resolve().parents[1]
    smoke = tmp_path / "synthetic_source"
    smoke.mkdir()
    (smoke / "manifest.json").write_text(json.dumps({
        "schema": "JCST-learnability-L2", "status": "COMPLETE",
        "protocol_hash": runner.sha(runner.PROTOCOL),
        "source_hashes": {name: runner.sha(root / name) for name in runner.SOURCES},
    }))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "AUDIT_PASS", "dev_eligible": True,
                                     "source_manifest_hash": runner.sha(smoke / "manifest.json")}))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "status": "PRECONFIRMATION_LOCKED", "dev_eligible": True,
        "train_manifest_sha256": runner.sha(smoke / "manifest.json"),
        "selection_sha256": runner.sha(selection),
        "stage3_runner_sha256": runner.sha(root / "experiments/jcst_learnability_stage3_l2.py"),
        "analysis_and_audit_sha256": {}
    }))
    source, chosen = load_lock(lock, smoke, selection)
    assert source["status"] == "COMPLETE" and chosen["dev_eligible"]
    lock.write_text(lock.read_text().replace("PRECONFIRMATION_LOCKED", "INCOMPLETE"))
    try:
        load_lock(lock, smoke, selection)
    except ValueError:
        pass
    else:
        raise AssertionError("unlocked confirmation accepted")


def test_preconfirmation_lock_requires_audited_eligible_complete_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(prelock, "ROOT", tmp_path)
    protocol = tmp_path / "docs/jcst_learnability_protocol_l2.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("synthetic engineering protocol")
    amendment = tmp_path / "docs/jcst_learnability_interpretation_amendment_l2_20261026.md"
    amendment.write_text("synthetic engineering amendment")
    for name in (*prelock.FROZEN_LATER_CODE, "experiments/jcst_learnability_stage3_l2.py"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("synthetic engineering code")
    train = tmp_path / "train"
    train.mkdir()
    manifest = {"status": "COMPLETE", "stage": "train_dev",
                "completed_trials": [{}] * 30,
                "completed_evaluations": [{}] * 240,
                "completed_controls": [{}] * 8,
                "source_hashes": {}, "protocol_hash": prelock.sha(protocol),
                "started_utc": datetime.now(timezone.utc).isoformat()}
    (train / "manifest.json").write_text(json.dumps(manifest))
    source_hash = prelock.sha(train / "manifest.json")
    paths = []
    for name, status, key in (("audit.json", "PASS", "source_manifest_hash"),
                              ("selection.json", "AUDIT_PASS", "source_manifest_hash"),
                              ("report.json", "COMPLETE", "source_manifest_sha256"),
                              ("optimization.json", "PASS", "source_manifest_sha256"),
                              ("inference.json", "COMPLETE", "source_manifest_sha256")):
        file = tmp_path / name
        payload = {"status": status, key: source_hash}
        if name == "selection.json":
            payload.update(dev_eligible=True, eligible_arm_cases=[["base", "S"]],
                           selected={"base": {"S": {"checkpoint": 96000, "mode": "greedy"}}})
        file.write_text(json.dumps(payload))
        paths.append(file)
    lock = prelock.build_lock(train, *paths)
    assert lock["status"] == "PRECONFIRMATION_LOCKED"
    assert lock["dev_eligible"]
    selection = json.loads(paths[1].read_text())
    selection["dev_eligible"] = False
    paths[1].write_text(json.dumps(selection))
    try:
        prelock.build_lock(train, *paths)
    except ValueError:
        pass
    else:
        raise AssertionError("ineligible DEV panel accepted")
