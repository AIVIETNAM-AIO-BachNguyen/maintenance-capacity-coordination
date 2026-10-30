"""Locked C1 physical collection. One worker; no learner or sealed TEST."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from ht_pdm_fjsp.component_service_v1 import config_json
from ht_pdm_fjsp.component_controllers_v1 import (
    BoundedRollout, Corrective, Periodic, RiskMatching, Threshold, Wait,
    OBSERVATION_MANIFEST,
)
from ht_pdm_fjsp.component_framework_v1 import run_episode
from ht_pdm_fjsp.jcst_characterization_v1 import (
    CASE_NAMES, SEED_RANGES, SENSITIVITY_CASES, SENSITIVITY_H_TAIL,
    ROLLOUT_DIAGNOSTIC_CASES,
    case_config, episode_metrics, physical_seed,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"artifacts"/"jcst_characterization"
SOURCES = [
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/component_framework_v1.py",
    "src/ht_pdm_fjsp/jcst_characterization_v1.py",
    "experiments/jcst_campaign_collect_v1.py",
    "experiments/jcst_replay_audit.py",
    "experiments/jcst_campaign_audit_v1.py",
]
PROTOCOL = "docs/jcst_characterization_protocol_v1.md"
STAGE_HOURS = {"characterization":4, "selection":2, "diagnostic":2, "diagnostic_rollout":4,
               "confirmation":8, "sensitivity":4}


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arms_for(stage: str, case: str, selected: dict | None):
    if stage == "selection":
        return [(f"periodic{p}",Periodic(p)) for p in (4,8)] + [
            (f"threshold{a}",Threshold(a)) for a in (2,5)]
    if stage == "diagnostic":
        return [("corrective",Corrective()),("risk",RiskMatching())]
    if stage == "diagnostic_rollout":
        return [("rollout2",BoundedRollout(2,2,20000))]
    if stage == "sensitivity":
        return [("corrective",Corrective()),("risk",RiskMatching()),
                ("rollout2",BoundedRollout(2,2,20000))]
    if stage == "characterization":
        return [("wait",Wait()),("corrective",Corrective()),
                ("periodic4",Periodic(4)),("periodic8",Periodic(8)),
                ("threshold2",Threshold(2)),("threshold5",Threshold(5)),
                ("risk",RiskMatching())]
    if selected is None or case not in selected:
        raise ValueError("audited baseline selection required for confirmation")
    return [("wait",Wait()),("corrective",Corrective()),
            (f"periodic{selected[case]['period']}",Periodic(selected[case]["period"])),
            (f"threshold{selected[case]['age']}",Threshold(selected[case]["age"])),
            ("risk",RiskMatching()),("rollout2",BoundedRollout(2,2,20000))]


def provenance(args, run_dir):
    hashes={rel:hash_file(ROOT/rel) for rel in SOURCES}
    configs={case:hashlib.sha256(config_json(case_config(case)[0]).encode()).hexdigest()
             for case in CASE_NAMES}
    return {
        "schema":"JCST-C1-physical-v1", "status":"running", "stage":args.stage,
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "run_dir":str(run_dir), "cwd":str(Path.cwd()), "command":sys.argv,
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "platform":platform.platform(), "python":sys.version,
        "dependencies":{name:importlib.metadata.version(name) for name in
                        ("numpy","torch","gymnasium","tqdm","pytest")},
        "source_hashes":hashes,
        "contract_hash":hash_file(ROOT/"docs/joint_component_service_contract_v1.md"),
        "protocol_hash":hash_file(ROOT/PROTOCOL),
        "config_hashes":configs,
        "observation_manifest":OBSERVATION_MANIFEST,
        "seed_range":((104729, args.seeds) if args.engineering_smoke else
                      (12_000_000,64) if args.stage=="diagnostic_rollout" else
                      SEED_RANGES[args.stage]),
        "kind":"engineering_smoke" if args.engineering_smoke else "scientific",
        "horizon":args.horizon, "tail":args.tail, "tail_rule":"risk",
        "cases":list(SENSITIVITY_CASES if args.stage=="sensitivity" else
                     ROLLOUT_DIAGNOSTIC_CASES if args.stage=="diagnostic_rollout" else CASE_NAMES),
        "episode_timeout_seconds":180 if args.stage in ("confirmation","sensitivity","diagnostic_rollout") else 30,
        "stage_timeout_seconds":STAGE_HOURS[args.stage]*3600,
        "disk_cap_bytes":(20 if args.stage in ("sensitivity","diagnostic_rollout") else 30)*1024**3, "worker_count":1,
        "completed":0, "expected":0, "failed":0, "selection_file":args.selection_file,
        "engineering_selection_fixture":bool(args.stage=="confirmation" and
                                             args.engineering_smoke and not args.selection_file),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=tuple(STAGE_HOURS),required=True)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--selection-file",type=str)
    parser.add_argument("--case",choices=CASE_NAMES,help="engineering-only subset; prohibited for scientific collection")
    parser.add_argument("--seeds",type=int,help="engineering-only subset; prohibited for scientific collection")
    parser.add_argument("--engineering-smoke",action="store_true")
    parser.add_argument("--horizon",type=int,default=48)
    parser.add_argument("--tail",type=int,default=16)
    args=parser.parse_args()
    if args.stage=="sensitivity":
        if (args.horizon,args.tail) not in SENSITIVITY_H_TAIL:
            parser.error("sensitivity H/tail must be a locked pair")
    elif (args.horizon,args.tail)!=(48,16):
        parser.error("main stages use locked H48+B16")
    if (args.case or args.seeds) and not args.engineering_smoke:
        parser.error("scientific stages must use all locked cases and seeds")
    selected=None
    if args.stage=="confirmation":
        if args.engineering_smoke and not args.selection_file:
            selected={case:dict(period=4,age=2) for case in CASE_NAMES}
        else:
            if not args.selection_file:
                parser.error("scientific confirmation requires audited selection file")
            selection_path=Path(args.selection_file)
            selection_doc=json.loads(selection_path.read_text())
            if selection_doc.get("status")!="AUDIT_PASS":
                parser.error("selection file is not audit-passed")
            selection_audit_path=selection_path.parent/"selection_analysis_audit.json"
            if not selection_audit_path.is_file():
                parser.error("independent selection arithmetic audit missing")
            selection_audit=json.loads(selection_audit_path.read_text())
            if (selection_audit.get("status")!="PASS" or
                    selection_audit.get("selection_file_hash")!=hash_file(selection_path)):
                parser.error("selection arithmetic audit/hash mismatch")
            source=Path(selection_doc["source_run"])
            if (selection_doc["protocol_hash"]!=hash_file(ROOT/PROTOCOL)
                    or hash_file(source/"manifest.json")!=selection_doc["source_manifest_hash"]
                    or hash_file(source/"independent_audit.json")!=selection_doc["source_audit_hash"]
                    or set(selection_doc["selected"])!=set(CASE_NAMES)
                    or any(value["period"] not in (4,8) or value["age"] not in (2,5)
                           for value in selection_doc["selected"].values())):
                parser.error("selection provenance or registry mismatch")
            selected=selection_doc["selected"]
    if args.resume:
        run_dir=args.resume.resolve()
        if not run_dir.is_dir():
            parser.error("resume path missing")
    else:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir=args.output_root.resolve()/f"{args.stage}-{stamp}"
        run_dir.mkdir(parents=True,exist_ok=False)
    fresh=provenance(args,run_dir)
    manifest_path=run_dir/"manifest.json"
    if args.resume:
        prior=json.loads(manifest_path.read_text())
        for key in ("stage","schema","source_hashes","contract_hash","protocol_hash","config_hashes"):
            if prior[key]!=fresh[key]:
                parser.error(f"resume {key} mismatch")
        if prior["status"]=="complete":
            parser.error("cannot resume complete run")
        manifest=prior
        manifest["status"]="running"
        manifest["resumed_utc"]=datetime.now(timezone.utc).isoformat()
    else:
        manifest=fresh
    def save_manifest():
        temp=manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest,indent=2,default=str)+"\n")
        os.replace(temp,manifest_path)
    save_manifest()
    cases=(args.case,) if args.case else (
        SENSITIVITY_CASES if args.stage=="sensitivity" else
        ROLLOUT_DIAGNOSTIC_CASES if args.stage=="diagnostic_rollout" else CASE_NAMES)
    locked_n=64 if args.stage=="diagnostic_rollout" else SEED_RANGES[args.stage][1]
    n=args.seeds if args.seeds else locked_n
    if n<1 or n>locked_n:
        parser.error("invalid seed count")
    tasks=[(case,arm,controller,(104729+i if args.engineering_smoke else physical_seed("diagnostic" if args.stage=="diagnostic_rollout" else args.stage,i)))
           for case in cases for arm,controller in arms_for(args.stage,case,selected)
           for i in range(n)]
    task_spec=dict(cases=list(cases),seed_count=n,
                   engineering_smoke=args.engineering_smoke,
                   horizon=args.horizon,tail=args.tail)
    if args.resume and (manifest.get("task_spec")!=task_spec or manifest["expected"]!=len(tasks)):
        parser.error("resume task specification mismatch")
    manifest["task_spec"]=task_spec
    manifest["expected"]=len(tasks)
    index_path=run_dir/"index.jsonl"
    indexed=[json.loads(line) for line in index_path.read_text().splitlines()] if index_path.exists() else []
    indexed_keys={(row["case"],row["arm"],row["seed"]) for row in indexed}
    if len(indexed_keys)!=len(indexed):
        parser.error("duplicate index records")
    manifest["completed"]=len(indexed)
    disk_bytes=sum(path.stat().st_size for path in (run_dir/"episodes").rglob("*") if path.is_file()) if (run_dir/"episodes").exists() else 0
    save_manifest()
    started=time.monotonic()
    deadline=started+manifest["stage_timeout_seconds"]
    try:
        for case,arm,controller,seed in tqdm(tasks,desc=f"C1 {args.stage}"):
            if time.monotonic()>deadline:
                raise TimeoutError("stage timeout")
            c,state=case_config(case)
            target=run_dir/"episodes"/case/arm/f"{seed}.json"
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_suffix(".partial.jsonl")
            identity=(case,arm,seed)
            if target.exists():
                existing=json.loads(target.read_text())
                if existing["seed"]!=seed or existing["case"]!=case or existing["arm"]!=arm or existing["terminal_status"]!="complete":
                    raise ValueError("duplicate/mismatched completed episode")
                if identity not in indexed_keys:
                    row=dict(case=case,arm=arm,seed=seed,status="complete",
                             episode_file=str(target.relative_to(run_dir)),
                             episode_hash=hash_file(target),cost=existing["total_cost"],
                             seconds=None)
                    with index_path.open("a") as ix:
                        ix.write(json.dumps(row)+"\n")
                        ix.flush(); os.fsync(ix.fileno())
                    indexed_keys.add(identity)
                    manifest["completed"]+=1
                    save_manifest()
                continue
            if identity in indexed_keys:
                raise ValueError("indexed episode file missing")
            if partial.exists():
                if not args.resume:
                    raise ValueError("unexpected partial collision")
                preserved=partial.with_name(partial.name+"."+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+".preserved")
                partial.rename(preserved)
            (run_dir/"configs").mkdir(exist_ok=True)
            config_path=run_dir/"configs"/f"{case}.json"
            if not config_path.exists():
                config_path.write_text(config_json(c)+"\n")
            else:
                if hashlib.sha256(config_path.read_text().strip().encode()).hexdigest()!=fresh["config_hashes"][case]:
                    raise ValueError("config file changed")
            ep_start=time.monotonic()
            def alarm_handler(signum,frame):
                raise TimeoutError("episode timeout")
            signal.signal(signal.SIGALRM,alarm_handler)
            signal.alarm(manifest["episode_timeout_seconds"])
            with partial.open("x") as stream:
                def persist(record):
                    stream.write(json.dumps(record,separators=(",",":"))+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                episode=run_episode(c,controller,seed,args.horizon,args.tail,
                                    state=state,record_callback=persist)
            signal.alarm(0)
            episode["case"]=case
            episode["arm"]=arm
            episode["initial_state"]=asdict(state)
            episode["metrics"]=episode_metrics(c,episode)
            temp=target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(episode,separators=(",",":"))+"\n")
            os.replace(temp,target)
            row=dict(case=case,arm=arm,seed=seed,status="complete",
                     episode_file=str(target.relative_to(run_dir)),
                     episode_hash=hash_file(target),cost=episode["total_cost"],
                     seconds=time.monotonic()-ep_start)
            with index_path.open("a") as ix:
                ix.write(json.dumps(row)+"\n")
                ix.flush()
                os.fsync(ix.fileno())
            indexed_keys.add(identity)
            manifest["completed"]+=1
            disk_bytes+=target.stat().st_size+partial.stat().st_size
            if manifest["completed"]%20==0:
                save_manifest()
            if disk_bytes>manifest["disk_cap_bytes"]:
                raise OSError("disk cap exceeded")
        manifest["status"]="complete"
    except BaseException as exc:
        signal.alarm(0)
        manifest["status"]="INCOMPLETE"
        manifest["failed"]+=1
        manifest["error"]=repr(exc)
        raise
    finally:
        manifest["elapsed_seconds"]=time.monotonic()-started
        manifest["ended_utc"]=datetime.now(timezone.utc).isoformat()
        save_manifest()
        print(run_dir)


if __name__=="__main__":
    main()
