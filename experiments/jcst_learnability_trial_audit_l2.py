"""Early independent integrity audit of one completed L2 training root."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from jcst_learnability_audit_l2 import lines, physical_replay
from jcst_learnability_l2 import CHECKPOINTS, ROOT, TRACE_EPISODES, action_universe, case_config, physical_seed, sha


def audit_trial(run: Path, case: str, arm: str, root: int) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["schema"] != "JCST-learnability-L2" or manifest["stage"] != "train_dev":
        raise ValueError("wrong source run")
    matching = [x for x in manifest["completed_trials"] if (x["case"], x["arm"], x["root"]) == (case, arm, root)]
    if len(matching) != 1:
        raise ValueError("trial not completed in manifest")
    info = matching[0]
    if info["samples"] != CHECKPOINTS[-1] or info["episodes"] != CHECKPOINTS[-1] // 48:
        raise ValueError("training budget incomplete")
    config, _ = case_config(case)
    universe = action_universe(config)
    trial = run / "trials" / case / arm / str(root)
    seen = 0
    replayed = 0
    for row in lines(trial / "train.jsonl"):
        if (row["episode_index"] != seen or row["seed"] != physical_seed("train", case, root, seen)
                or row["case"] != case or row["arm"] != arm or row["root"] != root
                or row["bank"] != "train" or row["terminal_status"] != "COMPLETE"):
            raise ValueError("training row sequence/seed/metadata")
        if (row["H_cost"] + row["tail_cost"] != row["total_cost"] or
                sum(row["components"].values()) != row["total_cost"]):
            raise ValueError("training H/tail/component arithmetic")
        if seen in TRACE_EPISODES:
            if "trace" not in row:
                raise ValueError("fixed replay trace missing")
            physical_replay(row, config, row["seed"], universe)
            replayed += 1
        elif "trace" in row:
            raise ValueError("unlocked training trace sampling")
        seen += 1
    if seen != info["episodes"] or replayed != len(TRACE_EPISODES):
        raise ValueError("trial row/trace completeness")
    for checkpoint in CHECKPOINTS:
        meta = json.loads((trial / f"checkpoint-{checkpoint}.json").read_text())
        if (meta["samples"] != checkpoint or meta["episodes"] != checkpoint // 48
                or meta["weights_sha256"] != sha(trial / f"checkpoint-{checkpoint}.pt")):
            raise ValueError("checkpoint provenance")
    update_rows = list(lines(trial / "updates.jsonl"))
    if len(update_rows) != 1280:
        raise ValueError("PPO update count")
    for index, row in enumerate(update_rows, start=1):
        if (row["update"] != index or row["samples"] != 1200 * index or
                row["sampled_telemetry"] != (index == 1 or index % 20 == 0)
                or not math.isfinite(row["loss"])):
            raise ValueError("PPO update schedule/finite loss")
    return {"status": "PASS", "case": case, "arm": arm, "root": root,
            "training_episodes_scanned": seen, "physical_episodes_replayed": replayed,
            "checkpoints_verified": len(CHECKPOINTS), "updates_verified": len(update_rows),
            "source_manifest_sha256_at_audit": sha(run / "manifest.json"),
            "trial_model_sha256": sha(trial / "model.json"),
            "audit_code_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("case", choices=("base", "generalist"))
    parser.add_argument("arm", choices=("S", "F", "A"))
    parser.add_argument("root", type=int)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = ROOT / "artifacts/jcst_learnability" / ("trial-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "run_dir": str(run.relative_to(ROOT)),
              "case": args.case, "arm": args.arm, "root": args.root}
    try:
        report.update(audit_trial(run, args.case, args.arm, args.root))
    except BaseException as exc:
        report.update(status="FAIL", failure=repr(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
