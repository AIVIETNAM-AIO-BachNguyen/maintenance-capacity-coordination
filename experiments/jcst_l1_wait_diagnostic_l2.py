"""Read-only diagnosis of frozen L1 DEV WAIT actions on selected checkpoints."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from jcst_replay_audit import independent_step
from jcst_learner_campaign_v1 import (ARMS, CASES, ROOT, action_universe,
                                       case_config, encode, load_policy, seed, sha)
from ht_pdm_fjsp.component_service_v1 import config_json, initial_state, state_from_json
from ht_pdm_fjsp.jcst_learner_v1 import tensors

L1_TRAIN = ROOT / "artifacts/jcst_framework/train_dev-20261025T175233.722390Z"


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def mean(xs) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main():
    selection = json.loads((L1_TRAIN / "selection.json").read_text())
    source = json.loads((L1_TRAIN / "manifest.json").read_text())
    if selection["status"] != "AUDIT_PASS" or source["status"] != "COMPLETE":
        raise ValueError("L1 source not frozen/audited")
    if selection["source_manifest_hash"] != sha(L1_TRAIN / "manifest.json"):
        raise ValueError("L1 selection provenance")
    output = ROOT / "artifacts/jcst_learnability" / ("l1-wait-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    decision_rows = []
    root_rows = []
    for case in CASES:
        config, _ = case_config(case)
        cfg = json.loads(config_json(config))
        universe = action_universe(config)
        for arm in ARMS:
            checkpoint = selection["selected"][case][arm]
            for root in tqdm(range(5), desc=f"L1 DEV {case}/{arm}", leave=False):
                trial = L1_TRAIN / "trials" / case / arm / str(root)
                _, _, policy = load_policy(trial, checkpoint)
                path = L1_TRAIN / "evaluations" / "dev" / case / arm / str(root) / f"episodes-{checkpoint}.jsonl.gz"
                start = len(decision_rows)
                for episode_index, episode in enumerate(rows(path)):
                    physical = seed("dev", case, 0, episode_index)
                    if episode["seed"] != physical:
                        raise ValueError("L1 DEV physical seed")
                    state = asdict(initial_state(config))
                    for tick in range(48):
                        item = episode["trace"][tick]
                        encoded = encode(config, state_from_json(config, json.dumps(state)), universe)
                        c, w, flat, mask = tensors([encoded])
                        with torch.no_grad():
                            distribution = policy.distribution(c, w, flat, mask)
                            probabilities = distribution.probs[0]
                            legal = mask[0]
                            top = torch.topk(distribution.logits[0][legal], min(2, int(legal.sum()))).values
                            gap = float(top[0] - top[1]) if len(top) > 1 else 1e9
                            chosen = int(distribution.logits.argmax(dim=-1).item())
                        action = tuple(tuple(edge) for edge in item["action"])
                        if chosen != item["index"] or action != universe[chosen]:
                            raise ValueError("L1 policy/action mismatch")
                        legal_actions = [candidate for candidate, valid in zip(universe, encoded.mask) if valid]
                        decision_rows.append({
                            "case": case, "arm": arm, "root": root,
                            "checkpoint": checkpoint, "physical_index": episode_index,
                            "tick": tick, "wait": not action,
                            "forced_wait": not action and len(legal_actions) == 1,
                            "discretionary_wait": not action and len(legal_actions) > 1,
                            "wait_failed_free": not action and any(
                                any(state["components"][m][cc]["failed"] for m, cc, _ in candidate)
                                for candidate in legal_actions if candidate),
                            "wait_all_healthy": not action and all(
                                not component["failed"] for group in state["components"] for component in group),
                            "wait_all_busy": not action and all(job is not None for job in state["jobs"]),
                            "wait_probability": float(probabilities[0]),
                            "chosen_probability": float(probabilities[chosen]),
                            "entropy": float(distribution.entropy().item()),
                            "logit_gap": gap, "near_tie": gap <= .01,
                            "exact_tie": gap <= 1e-6,
                            "legal_count": int(legal.sum()),
                        })
                        state, _, _ = independent_step(cfg, state, action, physical)
                subset = decision_rows[start:]
                if len(subset) != 16 * 48:
                    raise ValueError("L1 DEV episode count")
                root_rows.append({"case": case, "arm": arm, "root": root,
                                  "checkpoint": checkpoint, "decisions": len(subset),
                                  **{name: mean([row[name] for row in subset]) for name in
                                     ("wait", "forced_wait", "discretionary_wait",
                                      "wait_failed_free", "wait_all_healthy", "wait_all_busy",
                                      "wait_probability", "chosen_probability", "entropy",
                                      "logit_gap", "near_tie", "exact_tie", "legal_count")}})
    with (output / "decisions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    with (output / "per_root.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(root_rows[0]))
        writer.writeheader()
        writer.writerows(root_rows)
    summary = []
    for case in CASES:
        for arm in ARMS:
            subset = [row for row in root_rows if row["case"] == case and row["arm"] == arm]
            summary.append({"case": case, "arm": arm, "roots": 5,
                            **{name: mean([row[name] for row in subset]) for name in
                               ("wait", "forced_wait", "discretionary_wait",
                                "wait_failed_free", "wait_all_healthy", "wait_all_busy",
                                "wait_probability", "chosen_probability", "entropy",
                                "logit_gap", "near_tie", "exact_tie", "legal_count")}})
    with (output / "summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "scope": "historical L1 DEV descriptive diagnosis only",
        "source_manifest_sha256": sha(L1_TRAIN / "manifest.json"),
        "source_selection_sha256": sha(L1_TRAIN / "selection.json"),
        "diagnostic_code_sha256": sha(Path(__file__)),
        "physical_episodes": 2 * 3 * 5 * 16, "policy_decisions": len(decision_rows),
    }, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
