"""Durable SE1 stage driver: collect, independent replay, analysis, stats audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/jcst_sample_efficiency"
PROTOCOL = ROOT / "docs/jcst_sample_efficiency_protocol_se1.md"
DESIGN_LOCK = ROOT / "docs/jcst_sample_efficiency_design_lock_se1_20261028_v2.json"
AMENDMENT = ROOT / "docs/jcst_sample_efficiency_parallel_amendment_se1_20261028.md"
COLLECTOR = ROOT / "experiments/jcst_sample_efficiency_se1.py"
AUDITOR = ROOT / "experiments/jcst_sample_efficiency_audit_se1.py"
ANALYZER = ROOT / "experiments/jcst_sample_efficiency_analyze_se1.py"
STATS = ROOT / "experiments/jcst_sample_efficiency_stats_audit_se1.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def require_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    if lock.get("status") != "PRELAUNCH_LOCKED":
        raise ValueError("SE1 prelaunch lock missing")
    for key, source in (("protocol_sha256", PROTOCOL), ("design_lock_sha256", DESIGN_LOCK),
                        ("scheduling_amendment_sha256", AMENDMENT)):
        if lock[key] != sha(source):
            raise ValueError(f"prelaunch lock mismatch: {key}")
    for name, digest in lock["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"prelaunch source mismatch: {name}")
    if not lock.get("engineering_qualified"):
        raise ValueError("engineering qualification not PASS")
    return lock


def stage(run: Path, state: dict, name: str, command: list[str], deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("SE1 campaign 72-hour cap before " + name)
    log = run / f"{name}.log"
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               SE1_DEADLINE_MONOTONIC=str(deadline))
    entry = {"name": name, "command": command, "log": str(log.relative_to(ROOT)),
             "started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
    state["stages"].append(entry)
    write_json(run / "pipeline-status.json", state)
    with open(log, "x", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=False)
        entry["pid"] = proc.pid
        write_json(run / "pipeline-status.json", state)
        try:
            code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            entry["status"] = "INCOMPLETE"
            entry["failure"] = "72-hour elapsed cap"
            write_json(run / "pipeline-status.json", state)
            raise TimeoutError("SE1 campaign cap during " + name)
    entry["exit_code"] = code
    entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
    entry["status"] = "COMPLETE" if code == 0 else "FAILED"
    write_json(run / "pipeline-status.json", state)
    if code:
        raise RuntimeError(f"{name} exited {code}; see {log}")


def latest(run: Path, pattern: str) -> Path:
    found = sorted(run.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"expected one {pattern}; found {len(found)}")
    return found[0]


def summarize(run: Path, state: dict, smoke: bool, audit: Path, analysis: Path, stats: Path) -> None:
    audit_data = json.loads(audit.read_text())
    stats_data = json.loads(stats.read_text())
    report = json.loads(analysis.read_text())
    if audit_data["status"] != "PASS" or stats_data["status"] != "PASS":
        raise ValueError("cannot summarize unaudited results")
    lines = ["# SE1 completion summary", "", f"Status: {'engineering smoke' if smoke else 'scientific'} COMPLETE with independent replay and statistics PASS.",
             "", f"Run: `{run.relative_to(ROOT)}`", f"Replay: `{audit.relative_to(ROOT)}`",
             f"Analysis: `{analysis.relative_to(ROOT)}`", f"Statistics audit: `{stats.relative_to(ROOT)}`", ""]
    for case in ("base", "generalist"):
        lines += [f"## {case}", ""]
        for name, value in report["primary"][case]["contrasts"].items():
            lines.append(f"- {name}: {value['mean_difference']:.3f} "
                         f"[{value['adjusted_interval'][0]:.3f}, {value['adjusted_interval'][1]:.3f}], "
                         f"practical pass={value['practical_pass']}")
        lines.append("")
    (run / "completion-summary.md").write_text("\n".join(lines) + "\n")
    state["status"] = "COMPLETE"
    state["audit_report"] = str(audit.relative_to(ROOT))
    state["analysis_report"] = str(analysis.relative_to(ROOT))
    state["stats_audit_report"] = str(stats.relative_to(ROOT))
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run / "pipeline-status.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prelaunch-lock", type=Path)
    parser.add_argument("--session", default="foreground-engineering")
    args = parser.parse_args()
    if not args.smoke:
        if args.prelaunch_lock is None:
            parser.error("scientific pipeline requires immutable prelaunch lock")
        lock = require_lock(args.prelaunch_lock.resolve())
    else:
        lock = None
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        lock_stream = open(ARTIFACTS / ".scientific.lock", "a+", encoding="utf-8")
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        lock_stream = None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = ARTIFACTS / (f"{'smoke' if args.smoke else 'run'}-{stamp}")
    if run.exists():
        raise FileExistsError(run)
    # The collector creates the fresh directory and its immutable collection manifest.
    deadline = time.monotonic() + (3600 if args.smoke else 72 * 3600)
    state = {"schema": "JCST-SE1-pipeline", "status": "RUNNING", "smoke": args.smoke,
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "run_dir": str(run.relative_to(ROOT)), "tmux_session": args.session,
             "driver_pid": os.getpid(), "driver_ppid": os.getppid(),
             "driver_command": sys.argv, "stages": [],
             "prelaunch_lock": None if lock is None else str(args.prelaunch_lock.resolve().relative_to(ROOT)),
             "prelaunch_lock_sha256": None if lock is None else sha(args.prelaunch_lock.resolve())}
    try:
        # First stage log lives at the artifact parent until collector creates run.
        bootstrap_log = ARTIFACTS / f"bootstrap-{stamp}.log"
        command = [sys.executable, "-u", str(COLLECTOR), "--run-dir", str(run)]
        if args.smoke:
            command.append("--smoke")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
                   SE1_DEADLINE_MONOTONIC=str(deadline))
        if not args.smoke:
            env["SE1_PRELAUNCH_LOCK"] = str(args.prelaunch_lock.resolve())
        with open(bootstrap_log, "x", encoding="utf-8") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state["stages"].append({"name": "collection", "pid": proc.pid, "command": command,
                                    "log": str(bootstrap_log.relative_to(ROOT)), "status": "RUNNING",
                                    "started_utc": datetime.now(timezone.utc).isoformat()})
            try:
                code = proc.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                state["stages"][-1]["status"] = "INCOMPLETE"
                state["stages"][-1]["failure"] = "72-hour elapsed cap"
                raise TimeoutError("SE1 campaign cap during collection")
        if run.exists():
            write_json(run / "pipeline-status.json", state)
        if code:
            raise RuntimeError(f"collection exited {code}; see {bootstrap_log}")
        state["stages"][-1]["status"] = "COMPLETE"
        write_json(run / "pipeline-status.json", state)
        stage(run, state, "replay", [sys.executable, "-u", str(AUDITOR), str(run)], deadline)
        audit = latest(run, "audit-*/report.json")
        if json.loads(audit.read_text())["status"] != "PASS":
            raise ValueError("independent replay failed")
        stage(run, state, "analysis", [sys.executable, "-u", str(ANALYZER), str(run),
                                       "--audit", str(audit)], deadline)
        analysis = latest(run, "analysis-*/report.json")
        stage(run, state, "stats-audit", [sys.executable, "-u", str(STATS), str(run),
                                          str(analysis)], deadline)
        stats = latest(run, "stats-audit-*/report.json")
        summarize(run, state, args.smoke, audit, analysis, stats)
    except BaseException as exc:
        if run.exists():
            state["status"] = "INCOMPLETE" if len(state["stages"]) <= 1 else "INVALID"
            state["failure"] = repr(exc)
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(run / "pipeline-status.json", state)
        raise
    finally:
        if lock_stream is not None:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()
        print(run)


if __name__ == "__main__":
    main()
