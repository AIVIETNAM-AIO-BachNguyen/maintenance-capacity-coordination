"""Bounded Linux JCST/1 engineering smoke. No learner, checkpoint, or TEST."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from tqdm import tqdm

from ht_pdm_fjsp.component_service_v1 import config_json, engineering_config
from ht_pdm_fjsp.component_controllers_v1 import (
    BoundedRollout, Corrective, RiskMatching, Threshold, Wait, OBSERVATION_MANIFEST
)
from ht_pdm_fjsp.component_framework_v1 import run_episode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "jcst_engineering"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def timeout_handler(signum, frame):
    raise TimeoutError("bounded smoke timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--tail", type=int, default=4)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-forecast-transitions", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if any(x < 1 for x in (args.horizon,args.scenarios,args.depth,
                            args.max_forecast_transitions,args.timeout_seconds)) or args.tail < 0:
        parser.error("invalid bounds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.output_root.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = {
        "status":"running", "kind":"engineering_smoke", "contract":"JCST/1.1",
        "timestamp_utc":stamp, "command":sys.argv, "cwd":str(Path.cwd()),
        "git_head":git("rev-parse","HEAD"), "git_status":git("status","--short"),
        "platform":platform.platform(), "python":sys.version,
        "dependencies":{name:importlib.metadata.version(name) for name in
                        ("numpy","torch","gymnasium","tqdm","pytest")},
        "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in
                         [ROOT/"src/ht_pdm_fjsp/component_service_v1.py",
                          ROOT/"src/ht_pdm_fjsp/component_controllers_v1.py",
                          ROOT/"src/ht_pdm_fjsp/component_framework_v1.py",
                          ROOT/"experiments/jcst_engineering_smoke.py",
                          ROOT/"experiments/jcst_replay_audit.py"]},
        "contract_hash":sha(ROOT/"docs/joint_component_service_contract_v1.md"),
        "observation_manifest":OBSERVATION_MANIFEST,
        "budget":vars(args) | {"output_root":str(args.output_root)},
        "episodes":[],
    }
    def save():
        (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    save()
    config = engineering_config()
    (run_dir/"config.json").write_text(config_json(config)+"\n")
    manifest["config_hash"] = sha(run_dir/"config.json")
    save()
    controllers = [Wait(), Corrective(), Threshold(2), RiskMatching(),
                   BoundedRollout(args.scenarios,args.depth,args.max_forecast_transitions)]
    event_counts = {"start":0, "failure":0, "completion":0, "occupancy":0, "WAIT":0}
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)
        for controller in tqdm(controllers, desc="JCST engineering controllers"):
            episode_dir = run_dir/controller.name
            episode_dir.mkdir(exist_ok=False)
            (episode_dir/"config.json").write_text(config_json(config)+"\n")
            entry = dict(controller=controller.name, path=str(episode_dir), status="running")
            manifest["episodes"].append(entry)
            save()
            with (episode_dir/"partial.jsonl").open("w") as partial:
                def persist(record):
                    partial.write(json.dumps(record)+"\n")
                    partial.flush()
                    os.fsync(partial.fileno())
                episode = run_episode(config,controller,args.seed,args.horizon,args.tail,
                                      record_callback=persist)
            (episode_dir/"episode.json").write_text(json.dumps(episode,indent=2)+"\n")
            for record in episode["records"]:
                event_counts["WAIT"] += not record["action"]
                for event in record["events"]:
                    event_counts[event["type"]] += 1
            entry.update(status="complete", cost=episode["total_cost"],
                         compute=episode["compute"],
                         episode_wall_seconds=episode["episode_wall_seconds"])
            manifest["event_coverage"] = event_counts
            save()
        if not all(event_counts.values()):
            raise AssertionError("required smoke event coverage absent")
        signal.alarm(0)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        if manifest["episodes"] and manifest["episodes"][-1]["status"] == "running":
            manifest["episodes"][-1]["status"] = "partial"
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - start
        save()
        print(run_dir)


if __name__ == "__main__":
    main()
