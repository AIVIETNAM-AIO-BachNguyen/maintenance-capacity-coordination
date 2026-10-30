"""C1 conditional economic diagnostic, using audited physical source states."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from tqdm import tqdm

from ht_pdm_fjsp.component_service_v1 import config_from_json, state_from_json, legal_actions, transition
from ht_pdm_fjsp.component_controllers_v1 import RiskMatching, observe
from ht_pdm_fjsp.jcst_diagnostics_v1 import (
    MOTIFS, ROOT_LIMIT, forecast_branch, joint_quartets, motif_flags,
    priority, role_seed, witness_pairs,
)

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts"/"jcst_diagnostics"
PROTOCOL=ROOT/"docs"/"jcst_characterization_protocol_v1.md"
SOURCES=[
    "src/ht_pdm_fjsp/component_service_v1.py",
    "src/ht_pdm_fjsp/component_controllers_v1.py",
    "src/ht_pdm_fjsp/jcst_diagnostics_v1.py",
    "experiments/jcst_diagnostic_collect_v1.py",
    "experiments/jcst_diagnostic_audit_v1.py",
]
BUDGETS={"R2":(2,2),"R4":(4,4),"R8":(8,6)}


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_list(action):
    return [list(edge) for edge in action]


def encoded_arm(action,wait_for=None):
    return dict(action=action_list(action),wait_for=list(wait_for) if wait_for else None)


def arm_key(action,wait_for=None):
    return (tuple(action),tuple(wait_for) if wait_for else None)


def pair_arms(kind,pair):
    if kind=="Q":
        _,slow,target=pair
        return (arm_key((),target),arm_key(slow))
    return (arm_key(pair[0]),arm_key(pair[1]))


def pair_json(kind,pair):
    left,right=pair_arms(kind,pair)
    return dict(kind=kind,left=encoded_arm(*left),right=encoded_arm(*right))


def short_branch(config,state,root,seed,depth,horizon=48,tail=16):
    current=state
    chosen=root
    costs=[]
    path=[]
    risk=RiskMatching()
    for _ in range(depth):
        before=current
        out=transition(config,current,chosen,seed)
        costs.append(out.cost)
        path.append(dict(tick=before.tick,action=action_list(chosen),cost=out.cost,
                         after_hash=hashlib.sha256(
                             json.dumps(asdict(out.state),sort_keys=True).encode()).hexdigest()))
        current=out.state
        if current.tick>=horizon+tail: break
        chosen=risk.choose(observe(config,current,horizon,tail)).action
    return dict(cost=sum(costs),transitions=len(costs),path=path)


def selected_source_states(source:Path, case:str, reference:str):
    cfg=config_from_json((source/"configs"/f"{case}.json").read_text())
    per_episode=[]
    structural=[]
    for path in sorted((source/"episodes"/case/reference).glob("*.json")):
        episode=json.loads(path.read_text())
        seed=episode["seed"]
        counts={key:0 for key in MOTIFS}
        candidate=[]
        for record in episode["records"][:48]:
            state=state_from_json(cfg,json.dumps(record["before"]))
            actions=legal_actions(cfg,state)
            flags=motif_flags(cfg,state,actions)
            for key,seen in flags.items(): counts[key]+=int(seen)
            if any(flags.values()):
                candidate.append((priority("C1-state",case,reference,seed,state.tick),state.tick,
                                  record["before"],flags))
        structural.append(dict(case=case,reference=reference,seed=seed,
                               H_ticks=48,motif_ticks=counts,
                               motif_any={key:counts[key]>0 for key in MOTIFS},
                               eligible_ticks=len(candidate)))
        if candidate:
            item=min(candidate,key=lambda x:x[0])
            per_episode.append((item[0],seed,item[1],item[2],item[3],sha(path)))
    chosen=sorted(per_episode,key=lambda x:x[0])[:24]
    return cfg,structural,chosen


def choose_pairs(config,state,actions,A_means,extra_means,case,reference,physical_seed,tick):
    all_pairs=witness_pairs(config,state,actions)
    selected={}
    for kind,pairs in all_pairs.items():
        if not pairs:
            selected[kind]=dict(count=0,extreme=None,sentinel=None)
            continue
        def score(pair):
            arms=pair_arms(kind,pair)
            return [extra_means[arms[0]] if arms[0] in extra_means else A_means[arms[0]],
                    extra_means[arms[1]] if arms[1] in extra_means else A_means[arms[1]]]
        extreme=min(pairs,key=lambda pair:(
            -abs(score(pair)[0]-score(pair)[1]),repr(pair)))
        sentinel=min(pairs,key=lambda pair:priority(
            "C1-sentinel",case,reference,physical_seed,tick,kind,repr(pair)))
        left,right=score(extreme)
        selected[kind]=dict(
            count=len(pairs),extreme=pair_json(kind,extreme),
            extreme_A_costs=[left,right],
            orientation="left-minus-right" if left>=right else "right-minus-left",
            sentinel=pair_json(kind,sentinel))
    return selected


def collect_state(run_dir:Path, source:Path, case:str,reference:str,
                  config,seed:int,tick:int,before:dict,flags:dict,source_hash:str,
                  bank_stage:str="diagnostic",validation_bank:str="B"):
    state=state_from_json(config,json.dumps(before))
    actions=legal_actions(config,state)
    if len(actions)>ROOT_LIMIT: raise ValueError("root cap")
    folder=run_dir/"states"/case/reference/f"{seed}-{tick}"
    folder.mkdir(parents=True,exist_ok=False)
    state_info=dict(case=case,reference=reference,physical_seed=seed,tick=tick,
                    state=before,source_episode_hash=source_hash,
                    flags=flags,roots=len(actions),bank_stage=bank_stage,
                    validation_bank=validation_bank)
    (folder/"state.json").write_text(json.dumps(state_info,separators=(",",":"))+"\n")
    A_scores=defaultdict(list)
    with (folder/"A.jsonl").open("x") as stream:
        for root in actions:
            for tape in range(4):
                forecast=role_seed(bank_stage,case,seed,reference,tick,"A",tape)
                result=forecast_branch(config,state,root,forecast)
                A_scores[arm_key(root)].append(result["total_cost"])
                row=dict(bank="A",tape=tape,forecast_seed=forecast,
                         arm=encoded_arm(root),**result)
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
    A_means={key:sum(values)/len(values) for key,values in A_scores.items()}
    pairs=witness_pairs(config,state,actions)
    q_arms={pair_arms("Q",pair)[0] for pair in pairs["Q"]}
    extra=defaultdict(list)
    with (folder/"AQ.jsonl").open("x") as stream:
        for action,wait_for in sorted(q_arms):
            for tape in range(4):
                forecast=role_seed(bank_stage,case,seed,reference,tick,"A",tape)
                result=forecast_branch(config,state,action,forecast,wait_for=wait_for)
                extra[(action,wait_for)].append(result["total_cost"])
                stream.write(json.dumps(dict(bank="A-Q",tape=tape,forecast_seed=forecast,
                    arm=encoded_arm(action,wait_for),**result),separators=(",",":"))+"\n")
    extra_means={key:sum(values)/len(values) for key,values in extra.items()}
    selection=choose_pairs(config,state,actions,A_means,extra_means,case,reference,seed,tick)
    risk=RiskMatching().choose(observe(config,state,48,16)).action
    P={}
    if any(len(actions)*scenarios*depth>20000 for scenarios,depth in BUDGETS.values()):
        raise ValueError("locked per-selection forecast cap exceeded")
    with (folder/"P.jsonl").open("x") as stream:
        for label,(scenarios,depth) in BUDGETS.items():
            started=time.perf_counter()
            score=defaultdict(float)
            half=[defaultdict(float),defaultdict(float)]
            transitions=0
            for root in actions:
                for tape in range(scenarios):
                    forecast=role_seed(bank_stage,case,seed,reference,tick,"P",tape)
                    branch=short_branch(config,state,root,forecast,depth)
                    score[root]+=branch["cost"]/scenarios
                    half[0 if tape<scenarios//2 else 1][root]+=branch["cost"]/(scenarios//2)
                    transitions+=branch["transitions"]
                    stream.write(json.dumps(dict(budget=label,tape=tape,forecast_seed=forecast,
                        arm=encoded_arm(root),**branch),separators=(",",":"))+"\n")
            chosen=min(actions,key=lambda a:(score[a],a))
            half_roots=[min(actions,key=lambda a:(half[h][a],a)) for h in (0,1)]
            P[label]=dict(action=action_list(chosen),
                          half_actions=[action_list(a) for a in half_roots],
                          root_actions_evaluated=len(actions),scenarios=scenarios,
                          simulated_transitions=transitions,
                          selection_seconds=time.perf_counter()-started)
    quartets=joint_quartets(config,actions)
    selected_quartet=(min(quartets,key=lambda x:priority(
        "C1-joint",case,reference,seed,tick,repr(x))) if quartets else None)
    B_arms={arm_key(risk)}
    if selected_quartet:
        B_arms.update(arm_key(action) for action in selected_quartet)
    for item in selection.values():
        for label in ("extreme","sentinel"):
            pair=item[label]
            if pair:
                for side in ("left","right"):
                    arm=pair[side]
                    B_arms.add(arm_key(tuple(tuple(x) for x in arm["action"]),
                                       tuple(arm["wait_for"]) if arm["wait_for"] else None))
    for item in P.values():
        B_arms.add(arm_key(tuple(tuple(x) for x in item["action"])))
        for action in item["half_actions"]:
            B_arms.add(arm_key(tuple(tuple(x) for x in action)))
    if len(B_arms)*12>600:
        raise ValueError("locked validation branch cap exceeded")
    B_costs=defaultdict(list)
    with (folder/f"{validation_bank}.jsonl").open("x") as stream:
        for action,wait_for in sorted(B_arms,key=repr):
            for tape in range(12):
                forecast=role_seed(bank_stage,case,seed,reference,tick,validation_bank,tape)
                result=forecast_branch(config,state,action,forecast,wait_for=wait_for)
                B_costs[(action,wait_for)].append(result["total_cost"])
                stream.write(json.dumps(dict(bank=validation_bank,tape=tape,forecast_seed=forecast,
                    arm=encoded_arm(action,wait_for),**result),separators=(",",":"))+"\n")
    def meanB(arm):
        key=arm_key(tuple(tuple(x) for x in arm["action"]),
                    tuple(arm["wait_for"]) if arm["wait_for"] else None)
        return sum(B_costs[key])/len(B_costs[key])
    for item in selection.values():
        for label in ("extreme","sentinel"):
            pair=item[label]
            if pair: pair["validation_costs"]=[meanB(pair["left"]),meanB(pair["right"])]
    for item in P.values():
        item["validation_cost"]=meanB(encoded_arm(tuple(tuple(x) for x in item["action"])))
    selection["frontier"]=P
    selection["risk_root"]=action_list(risk)
    selection["risk_validation_cost"]=meanB(encoded_arm(risk))
    if selected_quartet:
        selection["joint_quartet"]=[action_list(action) for action in selected_quartet]
        selection["joint_validation_costs"]=[meanB(encoded_arm(action)) for action in selected_quartet]
        selection["joint_interaction_validation"]=(selection["joint_validation_costs"][3]
            -selection["joint_validation_costs"][1]-selection["joint_validation_costs"][2]
            +selection["joint_validation_costs"][0])
    else:
        selection["joint_quartet"]=None
    (folder/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    return dict(case=case,reference=reference,seed=seed,tick=tick,
                folder=str(folder.relative_to(run_dir)),roots=len(actions),
                A_branches=sum(len(v) for v in A_scores.values())+sum(len(v) for v in extra.values()),
                P_branches=sum(len(actions)*v[0] for v in BUDGETS.values()),
                validation_branches=sum(len(v) for v in B_costs.values()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    parser.add_argument("--engineering-smoke",action="store_true")
    parser.add_argument("--confirmation-bank",action="store_true")
    parser.add_argument("--case",help="engineering-only subset")
    parser.add_argument("--states",type=int,help="engineering-only subset per case/reference")
    args=parser.parse_args()
    if (args.case or args.states) and not args.engineering_smoke:
        parser.error("scientific diagnostics use all locked cases/states")
    source=args.source_run.resolve()
    source_manifest=json.loads((source/"manifest.json").read_text())
    source_audit=json.loads((source/"independent_audit.json").read_text())
    bank_stage="confirmation" if args.confirmation_bank else "diagnostic"
    validation_bank="C" if args.confirmation_bank else "B"
    if source_manifest["status"]!="complete" or source_audit["status"]!="PASS" or source_manifest["stage"]!=bank_stage:
        parser.error("matching source bank must be complete and independently audited")
    if source_manifest["protocol_hash"]!=sha(PROTOCOL):
        parser.error("source and economic protocol hashes differ")
    for rel in ("src/ht_pdm_fjsp/component_service_v1.py",
                "src/ht_pdm_fjsp/component_controllers_v1.py",
                "src/ht_pdm_fjsp/component_framework_v1.py",
                "src/ht_pdm_fjsp/jcst_characterization_v1.py"):
        if source_manifest["source_hashes"][rel]!=sha(ROOT/rel):
            parser.error(f"source physics/controller hash differs: {rel}")
    if args.engineering_smoke != (source_manifest["kind"]=="engineering_smoke"):
        parser.error("engineering/scientific source role mismatch")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir=args.output_root.resolve()/f"{bank_stage}-economic-{stamp}"
    run_dir.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema="JCST-C1-diagnostics-v1",status="running",
                  bank_stage=bank_stage,validation_bank=validation_bank,
                  kind="engineering_smoke" if args.engineering_smoke else "scientific",
                  source_run=str(source),source_manifest_hash=sha(source/"manifest.json"),
                  source_audit_hash=sha(source/"independent_audit.json"),
                  protocol_hash=sha(PROTOCOL),
                  contract_hash=sha(ROOT/"docs/joint_component_service_contract_v1.md"),
                  source_hashes={rel:sha(ROOT/rel) for rel in SOURCES},
                  command=sys.argv,git_head=subprocess.check_output(
                      ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  git_status=subprocess.check_output(
                      ["git","status","--short"],cwd=ROOT,text=True),
                  platform=platform.platform(),started_utc=datetime.now(timezone.utc).isoformat(),
                  cases=[args.case] if args.case else list(source_manifest["cases"]),
                  budgets=BUDGETS,bank_A=4,bank_B=12,state_limit=args.states or 24,
                  root_cap=ROOT_LIMIT,validation_branch_cap=600,
                  forecast_transition_cap_per_selection=20000,
                  worker_count=1,timeout_seconds=8*3600,
                  disk_cap_bytes=20*1024**3,completed_states=0)
    def save():
        temp=run_dir/"manifest.tmp"
        temp.write_text(json.dumps(manifest,indent=2,default=str)+"\n")
        os.replace(temp,run_dir/"manifest.json")
    save()
    started=time.monotonic()
    disk_bytes=0
    try:
        tasks=[]
        with (run_dir/"structural.jsonl").open("x") as stream:
            for case in manifest["cases"]:
                for reference in ("corrective","risk"):
                    cfg,rows,chosen=selected_source_states(source,case,reference)
                    for row in rows: stream.write(json.dumps(row)+"\n")
                    for _,seed,tick,before,flags,source_hash in chosen[:args.states or 24]:
                        tasks.append((case,reference,cfg,seed,tick,before,flags,source_hash))
        with (run_dir/"index.jsonl").open("x") as index:
            for case,reference,cfg,seed,tick,before,flags,source_hash in tqdm(
                    tasks,desc="C1 economic diagnostics"):
                if time.monotonic()-started>manifest["timeout_seconds"]:
                    raise TimeoutError("C1 economic diagnostic stage timeout")
                row=collect_state(run_dir,source,case,reference,cfg,seed,tick,before,flags,source_hash,
                                  bank_stage,validation_bank)
                index.write(json.dumps(row)+"\n"); index.flush(); os.fsync(index.fileno())
                manifest["completed_states"]+=1
                folder=run_dir/row["folder"]
                disk_bytes+=sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
                if disk_bytes>manifest["disk_cap_bytes"]:
                    raise OSError("diagnostic disk cap exceeded")
                save()
        manifest["expected_states"]=len(tasks)
        manifest["status"]="complete"
    except BaseException as exc:
        manifest["status"]="INCOMPLETE"
        manifest["error"]=repr(exc)
        raise
    finally:
        manifest["elapsed_seconds"]=time.monotonic()-started
        save()
        print(run_dir)


if __name__=="__main__":
    main()
