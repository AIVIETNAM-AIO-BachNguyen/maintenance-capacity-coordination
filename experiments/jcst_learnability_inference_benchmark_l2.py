"""Matched-state L2 actor+encoding inference benchmark (no physical outcomes)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics
import time

import torch
from tqdm import tqdm

from jcst_replay_audit import independent_step
from jcst_learnability_l2 import ARMS, CASES, ROOT, action_universe, case_config, load_policy, sha
from ht_pdm_fjsp.component_service_v1 import config_json, initial_state, state_from_json
from ht_pdm_fjsp.jcst_learner_v1 import encode, tensors


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def matched_states(train: Path, case: str):
    config, _ = case_config(case)
    cfg = json.loads(config_json(config))
    path = train / "controls" / "dev" / case / "threshold5" / "episodes.jsonl"
    states = []
    count = 0
    for episode in read_rows(path):
        state = asdict(initial_state(config))
        for tick in range(48):
            states.append(state_from_json(config, json.dumps(state)))
            action = tuple(tuple(edge) for edge in episode["trace"][tick]["action"])
            state, _, _ = independent_step(cfg, state, action, episode["seed"])
        count += 1
    if count != 16 or len(states) != 768:
        raise ValueError("matched DEV state panel")
    return config, states


def benchmark(policy, config, universe, states) -> tuple[float, float]:
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        for state in states:
            encoded = encode(config, state, universe)
            c, w, flat, mask = tensors([encoded])
            with torch.no_grad():
                index = int(policy.logits(c, w, flat, mask).argmax(dim=-1).item())
            if not encoded.mask[index]:
                raise ValueError("benchmark policy illegal")
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), max(durations) - min(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train, selection_file = args.train_dir.resolve(), args.selection.resolve()
    manifest = json.loads((train / "manifest.json").read_text())
    selection = json.loads(selection_file.read_text())
    if (manifest["status"] != "COMPLETE" or selection["status"] != "AUDIT_PASS" or
            selection["source_manifest_hash"] != sha(train / "manifest.json")):
        raise ValueError("TRAIN/DEV source/selection not audit-passed")
    output = ROOT / "artifacts/jcst_learnability" / ("inference-benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {"status": "RUNNING", "source_train_dir": str(train.relative_to(ROOT)),
              "source_manifest_sha256": sha(train / "manifest.json"),
              "selection_sha256": sha(selection_file),
              "code_sha256": sha(Path(__file__)),
              "state_source": "threshold5 DEV physical paths, 16 seeds per case",
              "measurement": "encode plus actor masked-logit greedy argmax, excludes critic/logging/physical transition"}
    try:
        for case in CASES:
            config, states = matched_states(train, case)
            universe = action_universe(config)
            for arm in ARMS:
                checkpoint = int(selection["selected"][case][arm]["checkpoint"])
                for root in tqdm(range(5), desc=f"inference {case}/{arm}", leave=False):
                    trial = train / "trials" / case / arm / str(root)
                    _, _, policy = load_policy(trial, checkpoint)
                    median_seconds, spread = benchmark(policy, config, universe, states)
                    rows.append({"case": case, "arm": arm, "root": root,
                                 "selected_checkpoint": checkpoint,
                                 "state_count": len(states), "repetitions": 3,
                                 "median_total_seconds": median_seconds,
                                 "seconds_per_decision": median_seconds / len(states),
                                 "three_repeat_spread_seconds": spread})
        with (output / "per_root.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report.update(status="COMPLETE", rows=len(rows),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        report.update(status="INCOMPLETE", failure=repr(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
