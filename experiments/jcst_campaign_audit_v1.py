"""Independent physical replay and provenance audit for locked JCST C1."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from tqdm import tqdm

from jcst_replay_audit import catalogue, independent_step, u01

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_risk_action(cfg: dict, before: dict, horizon: int, tail: int):
    all_actions=catalogue(cfg,before)
    values={}
    for action in all_actions:
        for m,c,j in action:
            edge=(m,c,j)
            if edge in values: continue
            spec=cfg["components"][m][c]
            age=before["components"][m][c]["age"]
            duration=cfg["duration"][m][c][j]
            if before["components"][m][c]["failed"]:
                benefit=cfg["unavailable_cost"]*(horizon+tail-before["tick"]+1)+cfg["failure_cost"]
                start_cost=cfg["corrective_cost"]
            else:
                delta=((age+1)/spec["scale"])**spec["shape"]-(age/spec["scale"])**spec["shape"]
                probability=1-math.exp(-delta)
                benefit=probability*(cfg["failure_cost"]+cfg["unavailable_cost"]*max(1,duration))
                start_cost=cfg["preventive_cost"]
            values[edge]=(benefit-start_cost-cfg["unavailable_cost"]*duration)/duration
    return min(all_actions,key=lambda a:(-sum(values[x] for x in a),a))



def expected_forecast_seed(tick:int,scenario:int)->int:
    base=tick*1000003+7919
    digest=hashlib.sha256(f"JCST/1:forecast:{base}:{scenario}".encode()).digest()
    return (1<<64)+int.from_bytes(digest[:8],"big")


def audit_decision(config,before,record,physical_seed,H,B):
    decision=record.get("decision")
    if decision is None or decision["action"]!=record["action"]:
        raise ValueError("decision record/action")
    roots=catalogue(config,before)
    forecasts=decision["forecast_records"]
    if forecasts:
        scenarios=decision["forecast_scenarios"]
        if scenarios!=2 or decision["root_actions_evaluated"]!=len(roots):
            raise ValueError("rollout budget/root count")
        if len(forecasts)!=len(roots)*scenarios:
            raise ValueError("forecast branch coverage")
        totals={root:0. for root in roots}
        seen=set()
        transitions=0
        for branch in forecasts:
            root=tuple(tuple(e) for e in branch["root"])
            scenario=branch["scenario"]
            seed=expected_forecast_seed(before["tick"],scenario)
            if root not in roots or scenario not in (0,1) or branch["seed"]!=seed or seed==physical_seed:
                raise ValueError("forecast role/legality")
            if (root,scenario) in seen:
                raise ValueError("duplicate forecast root/scenario")
            seen.add((root,scenario))
            state=json.loads(json.dumps(before))
            action=root
            cost=0.
            steps=min(2,H+B-before["tick"])
            for k in range(steps):
                state,parts,_=independent_step(config,state,action,seed)
                cost+=sum(parts.values())
                if k+1<steps:
                    action=expected_risk_action(config,state,H,B)
            if branch["cost"]!=cost or branch["transitions"]!=steps:
                raise ValueError("forecast branch replay")
            totals[root]+=cost
            transitions+=steps
        if seen!={(root,scenario) for root in roots for scenario in (0,1)}:
            raise ValueError("incomplete forecast root/scenario coverage")
        if decision["simulated_transitions"]!=transitions:
            raise ValueError("forecast transition counter")
        chosen=min(roots,key=lambda a:(totals[a],a))
        if chosen!=tuple(tuple(e) for e in record["action"]):
            raise ValueError("rollout selection arithmetic")
    elif decision["forecast_scenarios"] or decision["simulated_transitions"]:
        raise ValueError("unrecorded forecasts")
    if not 0<=decision["selection_seconds"]<=decision["wall_seconds"]+1e-9:
        raise ValueError("decision timing counter")
    return decision


def expected_baseline_action(config,before,arm):
    actions=catalogue(config,before)
    if arm=="wait": return ()
    if arm=="risk": return expected_risk_action(config,before,48,16)
    if arm=="corrective":
        candidates=[a for a in actions if all(before["components"][m][c]["failed"] for m,c,j in a)]
        return min(candidates,key=lambda a:(-len(a),
            sum(config["duration"][m][c][j] for m,c,j in a),a))
    if arm.startswith("threshold"):
        age=int(arm[len("threshold"):])
        candidates=[a for a in actions if all(
            before["components"][m][c]["failed"] or before["components"][m][c]["age"]>=age
            for m,c,j in a)]
        return min(candidates,key=lambda a:(
            -sum(10 if before["components"][m][c]["failed"] else 1 for m,c,j in a),
            sum(config["duration"][m][c][j] for m,c,j in a),a))
    if arm.startswith("periodic"):
        period=int(arm[len("periodic"):])
        candidates=[a for a in actions if all(
            before["components"][m][c]["failed"] or
            ((before["tick"]-(m+c))%period==0) for m,c,j in a)]
        return min(candidates,key=lambda a:(
            -sum(10 if before["components"][m][c]["failed"] else 1 for m,c,j in a),
            sum(config["duration"][m][c][j] for m,c,j in a),a))
    if arm=="rollout2": return None
    raise ValueError("unknown controller arm")

def audit_episode(config: dict, episode: dict) -> dict:
    H=episode["horizon"]; B=episode["tail"]
    if (H,B) not in ((48,16),(24,8),(48,0),(96,32)) or episode["adapter_version"]!="JCST/1.1-H-tail-risk-v1":
        raise ValueError("H/tail/adapter mismatch")
    if episode["terminal_status"]!="complete" or len(episode["records"])!=H+B:
        raise ValueError("incomplete episode")
    previous=episode["initial_state"]
    costs={"H":0.,"tail":0.}
    counts=Counter()
    compute=Counter()
    for n,record in enumerate(episode["records"]):
        if record["tick"]!=n or record["before"]!=previous or record["phase"]!=("H" if n<H else "tail"):
            raise ValueError(f"state chain/timing at {n}")
        complete=catalogue(config,previous)
        observed={tuple(tuple(e) for e in a) for a in record["catalogue"]}
        if len(observed)!=len(record["catalogue"]) or observed!=complete:
            raise ValueError(f"action catalogue at {n}")
        action=tuple(tuple(e) for e in record["action"])
        if action not in complete:
            raise ValueError(f"illegal action at {n}")
        if n>=H and action!=expected_risk_action(config,previous,H,B):
            raise ValueError(f"tail policy at {n}")
        if n<H:
            arm=episode["arm"]
            expected_control=expected_baseline_action(config,previous,arm)
            if arm=="risk":
                expected_control=expected_risk_action(config,previous,H,B)
            if expected_control is not None and action!=expected_control:
                raise ValueError(f"controller tie/priority at {n}")
            decision=audit_decision(config,previous,record,episode["seed"],H,B)
            if arm=="risk" and decision["root_actions_evaluated"]!=len(complete):
                raise ValueError("risk root evaluation count")
            if arm not in ("risk","rollout2") and decision["root_actions_evaluated"]!=0:
                raise ValueError("restricted controller root count")
            for key in ("root_actions_evaluated","forecast_scenarios",
                        "simulated_transitions","selection_seconds","wall_seconds"):
                compute[key]+=decision[key]
        following,breakdown,(starts,completions,failures,occupancy)=independent_step(
            config,previous,action,episode["seed"])
        if following!=record["after"] or breakdown!=record["breakdown"] or sum(breakdown.values())!=record["cost"]:
            raise ValueError(f"transition/cost at {n}")
        events=record["events"]
        if len(events)!=len(starts)+len(completions)+len(failures)+len(occupancy):
            raise ValueError(f"event count at {n}")
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["kind"],e["duration"])
                  for e in events if e["type"]=="start") != sorted(
                    (t,m,c,j,k,config["duration"][m][c][j]) for t,m,c,j,k in starts):
            raise ValueError(f"starts at {n}")
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["kind"])
                  for e in events if e["type"]=="completion")!=sorted(completions):
            raise ValueError(f"completions at {n}")
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["remaining"])
                  for e in events if e["type"]=="occupancy")!=sorted(occupancy):
            raise ValueError(f"occupancy at {n}")
        failure_events=[e for e in events if e["type"]=="failure"]
        if sorted((e["tick"],e["machine"],e["component"]) for e in failure_events)!=sorted(failures):
            raise ValueError(f"failures at {n}")
        if any(e["uniform"]!=u01(episode["seed"],n,e["machine"],e["component"])
               for e in failure_events):
            raise ValueError(f"failure uniform at {n}")
        for e in events: counts[e["type"]]+=1
        costs["H" if n<H else "tail"]+=record["cost"]
        previous=following
    if (previous!=episode["final_state"] or costs["H"]!=episode["controlled_cost"]
            or costs["tail"]!=episode["tail_cost"]
            or sum(costs.values())!=episode["total_cost"]):
        raise ValueError("terminal/H-tail totals")
    for key in compute:
        if compute[key]!=episode["compute"][key]:
            raise ValueError("episode compute counter")
    metrics=episode["metrics"]
    if (metrics["component_failures"]!=counts["failure"] or metrics["pm_starts"]+
            metrics["cm_starts"]!=counts["start"] or metrics["completions"]!=counts["completion"]
            or metrics["occupied_technician_intervals"]!=counts["occupancy"]
            or metrics["H_cost"]!=costs["H"] or metrics["tail_cost"]!=costs["tail"]):
        raise ValueError("metric arithmetic")
    if metrics["working_intervals"]+metrics["breakdown_intervals"]+metrics["service_intervals"]!=len(config["components"])*(H+B):
        raise ValueError("mode occupancy")
    return dict(cost=episode["total_cost"],events=dict(counts),records=H+B)


def audit_run(run_dir: Path) -> dict:
    manifest=json.loads((run_dir/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise ValueError("collection incomplete")
    if manifest["schema"]!="JCST-C1-physical-v1":
        raise ValueError("schema")
    if digest(ROOT/"docs/jcst_characterization_protocol_v1.md")!=manifest["protocol_hash"]:
        raise ValueError("protocol hash differs from collection")
    for rel,sha in manifest["source_hashes"].items():
        if digest(ROOT/rel)!=sha:
            raise ValueError(f"source hash differs: {rel}")
    if digest(ROOT/"docs/joint_component_service_contract_v1.md")!=manifest["contract_hash"]:
        raise ValueError("contract hash differs")
    rows=[json.loads(s) for s in (run_dir/"index.jsonl").read_text().splitlines()]
    if len(rows)!=manifest["expected"] or manifest["completed"]!=len(rows):
        raise ValueError("coverage count")
    keys=[(r["case"],r["arm"],r["seed"]) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError("duplicate identity")
    cases=set(manifest["cases"])
    first,count=manifest["seed_range"]
    if manifest["kind"]=="scientific" and first not in (
            10_000_000,11_000_000,12_000_000,13_000_000,14_000_000):
        raise ValueError("physical seed role")
    total=Counter()
    started=time.monotonic()
    for row in tqdm(rows,desc="C1 independent physical audit"):
        if row["status"]!="complete" or row["case"] not in cases or not first<=row["seed"]<first+count:
            raise ValueError("index identity/seed role")
        path=run_dir/row["episode_file"]
        if not path.is_file() or digest(path)!=row["episode_hash"]:
            raise ValueError("episode hash")
        config_path=run_dir/"configs"/f"{row['case']}.json"
        cfg=json.loads(config_path.read_text())
        canonical=json.dumps(cfg,sort_keys=True,separators=(",",":")).encode()
        if hashlib.sha256(canonical).hexdigest()!=manifest["config_hashes"][row["case"]]:
            raise ValueError("config hash")
        ep=json.loads(path.read_text())
        if (ep["seed"],ep["case"],ep["arm"])!=(row["seed"],row["case"],row["arm"]):
            raise ValueError("episode identity")
        if (ep["horizon"],ep["tail"])!=(manifest["horizon"],manifest["tail"]):
            raise ValueError("episode H/tail differs from run manifest")
        result=audit_episode(cfg,ep)
        total["episodes"]+=1
        total["records"]+=result["records"]
        for event,n in result["events"].items(): total[event]+=n
    result=dict(status="PASS",run_dir=str(run_dir),stage=manifest["stage"],
                audited=dict(total),elapsed_seconds=time.monotonic()-started,
                protocol_hash=manifest["protocol_hash"],
                source_hashes=manifest["source_hashes"])
    (run_dir/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_run(args.run_dir)))


if __name__=="__main__":
    main()
