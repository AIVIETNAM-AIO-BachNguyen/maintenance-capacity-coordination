"""Independent C1 forecast-bank replay and selection arithmetic audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from tqdm import tqdm

from jcst_replay_audit import catalogue, independent_step

ROOT=Path(__file__).resolve().parents[1]


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(*parts)->str:
    return hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()


def seed_for(stage,case,physical_seed,reference,tick,bank,tape):
    text=f"C1|{stage}|{case}|{physical_seed}|{reference}|{tick}|{bank}|{tape}"
    return (1<<64)+int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")


def arm_key(arm):
    return (tuple(tuple(x) for x in arm["action"]),
            tuple(arm["wait_for"]) if arm["wait_for"] else None)


def _risk_value(config,state,edge,H=48,B=16):
    m,c,j=edge
    spec=config["components"][m][c]
    duration=config["duration"][m][c][j]
    if state["components"][m][c]["failed"]:
        benefit=config["unavailable_cost"]*(H+B-state["tick"]+1)+config["failure_cost"]
        start=config["corrective_cost"]
    else:
        age=state["components"][m][c]["age"]
        delta=((age+1)/spec["scale"])**spec["shape"]-(age/spec["scale"])**spec["shape"]
        risk=1-math.exp(-delta)
        benefit=risk*(config["failure_cost"]+config["unavailable_cost"]*max(1,duration))
        start=config["preventive_cost"]
    return (benefit-start-config["unavailable_cost"]*duration)/duration


def risk_action(config,state,excluded_machine=None,required_edge=None):
    actions=catalogue(config,state)
    choices=[a for a in actions if
             (excluded_machine is None or all(e[0]!=excluded_machine for e in a))
             and (required_edge is None or required_edge in a)]
    if not choices: raise ValueError("no constrained risk action")
    values={edge:_risk_value(config,state,edge) for a in actions for edge in a}
    return min(choices,key=lambda a:(-sum(values[e] for e in a),a))


def audit_branch(config,state,row,case,reference,physical_seed,tick,bank,
                 bank_stage="diagnostic",depth=None):
    expected=seed_for(bank_stage,case,physical_seed,reference,tick,bank,row["tape"])
    if row["forecast_seed"]!=expected or expected<2**64:
        raise ValueError("forecast role seed")
    root=arm_key(row["arm"])[0]
    wait_for=arm_key(row["arm"])[1]
    if root not in catalogue(config,state):
        raise ValueError("branch root legality")
    current=json.loads(json.dumps(state))
    held=wait_for is not None
    Hcost=Bcost=0.
    path=row["path"]
    expected_len=min(depth if depth else 48+16-tick,48+16-tick)
    if len(path)!=expected_len or row["transitions"]!=len(path):
        raise ValueError("branch length")
    for i,entry in enumerate(path):
        action=tuple(tuple(x) for x in entry["action"])
        if entry["tick"]!=current["tick"] or action not in catalogue(config,current):
            raise ValueError("branch action/tick")
        if i==0 and action!=root:
            raise ValueError("forced root")
        if i>0:
            if current["tick"]>=48:
                expected_action=risk_action(config,current)
            elif held:
                target=wait_for
                if target in {e for a in catalogue(config,current) for e in a}:
                    expected_action=risk_action(config,current,required_edge=target)
                    held=False
                else:
                    expected_action=risk_action(config,current,excluded_machine=target[0])
            else:
                expected_action=risk_action(config,current)
            if action!=expected_action:
                raise ValueError("forecast continuation")
        following,breakdown,_=independent_step(config,current,action,expected)
        cost=sum(breakdown.values())
        after_hash=hashlib.sha256(json.dumps(following,sort_keys=True).encode()).hexdigest()
        if cost!=entry["cost"] or after_hash!=entry["after_hash"]:
            raise ValueError("forecast transition/cost")
        if current["tick"]<48: Hcost+=cost
        else: Bcost+=cost
        current=following
    if depth:
        if Hcost+Bcost!=row["cost"]:
            raise ValueError("short budget cost")
    else:
        if Hcost!=row["controlled_cost"] or Bcost!=row["tail_cost"] or Hcost+Bcost!=row["total_cost"]:
            raise ValueError("forecast H/tail")
    return Hcost+Bcost


def independent_pairs(config,state,actions):
    """Independent complete-action pair enumeration for C1 motif audit."""
    roots=sorted(actions,key=lambda a:(len(a),a))
    aset=set(roots)
    result={k:set() for k in ("T","C","W","J","A","R","Q","P")}
    for a in roots:
        for edge in a:
            m,c,j=edge
            rest=tuple(x for x in a if x!=edge)
            if rest in aset and not state["components"][m][c]["failed"]:
                pair=tuple(sorted((a,rest)))
                result["T"].add(pair)
                if config["duration"][m][c][j]>1: result["P"].add(pair)
            for c2 in range(c+1,len(config["components"][m])):
                b=tuple(sorted(rest+((m,c2,j),)))
                if b in aset: result["C"].add(tuple(sorted((a,b))))
            for j2 in range(j+1,len(state["jobs"])):
                b=tuple(sorted(rest+((m,c,j2),)))
                if b in aset:
                    pair=tuple(sorted((a,b)))
                    result["W"].add(pair)
                    if config["duration"][m][c][j]!=config["duration"][m][c][j2]:
                        result["R"].add(pair)
    singles={a[0] for a in roots if len(a)==1}
    together={(a[0][0],a[1][0]) for a in roots if len(a)==2}
    for a,b in itertools.combinations(sorted(singles),2):
        if a[0]!=b[0] and tuple(sorted((a[0],b[0]))) not in together:
            result["J"].add(tuple(sorted(((a,),(b,)))))
    grouped={}
    for action in roots:
        if len(action)>=2:
            grouped.setdefault(tuple(m for m,c,j in action),[]).append(action)
    for machine_set,group in grouped.items():
        for a,b in itertools.combinations(group,2):
            da=tuple((m,config["duration"][m][c][j]) for m,c,j in a)
            db=tuple((m,config["duration"][m][c][j]) for m,c,j in b)
            if da!=db: result["A"].add((a,b))
    for m,c,jfree in singles:
        slow=config["duration"][m][c][jfree]
        for jbusy,job in enumerate(state["jobs"]):
            if job is not None and 0<config["duration"][m][c][jbusy]<slow:
                result["Q"].add(((),((m,c,jfree),),(m,c,jbusy)))
    return {k:tuple(sorted(v)) for k,v in result.items()}


def pair_arms(kind,pair):
    if kind=="Q":
        return (((),pair[2]),(pair[1],None))
    return ((pair[0],None),(pair[1],None))


def pair_from_json(kind,item):
    left=arm_key(item["left"]); right=arm_key(item["right"])
    if kind=="Q":
        if left[0]!=() or left[1] is None or right[1] is not None:
            raise ValueError("Q strategy encoding")
        return ((),right[0],left[1])
    if left[1] is not None or right[1] is not None:
        raise ValueError("unexpected strategy")
    return tuple(sorted((left[0],right[0])))


def audit_state(folder:Path,config,info,bank_stage,validation_bank):
    state=info["state"]
    if info["bank_stage"]!=bank_stage or info["validation_bank"]!=validation_bank:
        raise ValueError("state bank identity")
    case,reference,physical_seed,tick=(
        info["case"],info["reference"],info["physical_seed"],info["tick"])
    actions=catalogue(config,state)
    if len(actions)!=info["roots"] or len(actions)>500:
        raise ValueError("root count")
    selected=json.loads((folder/"selection.json").read_text())
    costs={"A":defaultdict(dict),"AQ":defaultdict(dict),validation_bank:defaultdict(dict)}
    counts={}
    all_seeds=set()
    for name,bank,depth in (("A","A",None),("AQ","A",None),("P","P",None),
                            (validation_bank,validation_bank,None)):
        rows=[json.loads(line) for line in (folder/(name+".jsonl")).read_text().splitlines()]
        counts[name]=len(rows)
        for row in rows:
            if name=="P":
                label=row["budget"]
                if label not in ("R2","R4","R8"):
                    raise ValueError("unknown frontier budget")
                scenarios,steps={"R2":(2,2),"R4":(4,4),"R8":(8,6)}[label]
                if not 0<=row["tape"]<scenarios: raise ValueError("P tape")
                branch_cost=audit_branch(config,state,row,case,reference,physical_seed,tick,bank,bank_stage,steps)
                costs.setdefault(label,defaultdict(dict))[arm_key(row["arm"])][row["tape"]]=branch_cost
            else:
                limit=4 if bank=="A" else 12
                if not 0<=row["tape"]<limit: raise ValueError("A/B tape")
                branch_cost=audit_branch(config,state,row,case,reference,physical_seed,tick,bank,bank_stage)
                costs[name][arm_key(row["arm"])][row["tape"]]=branch_cost
            all_seeds.add((bank,row["tape"],row["forecast_seed"]))
    if len({seed for _,_,seed in all_seeds})!=len({(bank,tape) for bank,tape,_ in all_seeds}):
        raise ValueError("forecast bank separation")
    def mean(bank,arm):
        values=costs[bank][arm]
        expected=4 if bank in ("A","AQ") else 12
        if set(values)!=set(range(expected)): raise ValueError("forecast tape coverage")
        return sum(values.values())/expected
    pairs=independent_pairs(config,state,actions)
    for kind,candidates in pairs.items():
        item=selected[kind]
        if item["count"]!=len(candidates): raise ValueError("pair count")
        if not candidates:
            if item["extreme"] is not None or item["sentinel"] is not None:
                raise ValueError("absent pair selected")
            continue
        def score(pair):
            a,b=pair_arms(kind,pair)
            return (mean("AQ" if a[1] else "A",a),
                    mean("AQ" if b[1] else "A",b))
        extreme=min(candidates,key=lambda pair:(-abs(score(pair)[0]-score(pair)[1]),repr(pair)))
        sentinel=min(candidates,key=lambda pair:priority(
            "C1-sentinel",case,reference,physical_seed,tick,kind,repr(pair)))
        if pair_from_json(kind,item["extreme"])!=extreme or pair_from_json(kind,item["sentinel"])!=sentinel:
            raise ValueError("selection arithmetic")
        if item["extreme_A_costs"]!=list(score(extreme)):
            raise ValueError("A score arithmetic")
        for label in ("extreme","sentinel"):
            frozen=item[label]
            arms=(arm_key(frozen["left"]),arm_key(frozen["right"]))
            if frozen["validation_costs"]!=[mean(validation_bank,arm) for arm in arms]:
                raise ValueError("validation score arithmetic")
    for label,(scenarios,steps) in {"R2":(2,2),"R4":(4,4),"R8":(8,6)}.items():
        budget=costs[label]
        if set(budget)!={(a,None) for a in actions}:
            raise ValueError("P root coverage")
        if any(set(tapes)!=set(range(scenarios)) for tapes in budget.values()):
            raise ValueError("P tape coverage")
        def meanP(action,tapes):
            return sum(budget[(action,None)][i] for i in tapes)/len(tapes)
        chosen=min(actions,key=lambda a:(meanP(a,range(scenarios)),a))
        half=[min(actions,key=lambda a:(meanP(a,range(h*(scenarios//2),(h+1)*(scenarios//2))),a))
              for h in (0,1)]
        row=selected["frontier"][label]
        if (row["action"]!=[list(e) for e in chosen]
                or row["half_actions"]!=[[list(e) for e in a] for a in half]
                or row["root_actions_evaluated"]!=len(actions)
                or row["scenarios"]!=scenarios
                or row["simulated_transitions"]!=sum(len(r["path"]) for r in
                    (json.loads(line) for line in (folder/"P.jsonl").read_text().splitlines())
                    if r["budget"]==label)):
            raise ValueError("P selection/compute arithmetic")
        if row["validation_cost"]!=mean(validation_bank,(chosen,None)):
            raise ValueError("P independent cost")
    if selected["risk_root"]!=[list(e) for e in risk_action(config,state)]:
        raise ValueError("risk root")
    if selected["risk_validation_cost"]!=mean(validation_bank,
            (tuple(tuple(e) for e in selected["risk_root"]),None)):
        raise ValueError("risk validation cost")
    quartet=selected["joint_quartet"]
    if quartet is not None:
        arms=[tuple(tuple(e) for e in a) for a in quartet]
        if not (arms[0]==() and len(arms[1])==len(arms[2])==1 and
                len(arms[3])==2 and set(arms[3])==set(arms[1]+arms[2])):
            raise ValueError("joint quartet shape")
        values=[mean(validation_bank,(a,None)) for a in arms]
        if values!=selected["joint_validation_costs"] or values[3]-values[1]-values[2]+values[0]!=selected["joint_interaction_validation"]:
            raise ValueError("joint interaction arithmetic")
    return counts



def independent_flags(config,state,actions):
    singles={a[0] for a in actions if len(a)==1}
    free={j for j,job in enumerate(state["jobs"]) if job is None}
    busy={j for j,job in enumerate(state["jobs"]) if job is not None}
    flags={k:False for k in ("T","C","W","J","A","R","Q","P")}
    flags["T"]=any(not state["components"][m][c]["failed"] for m,c,j in singles)
    flags["C"]=any((m,c2,j) in singles for m,c,j in singles
                   for c2 in range(c+1,len(config["components"][m])))
    flags["W"]=any((m,c,j2) in singles for m,c,j in singles for j2 in free if j2>j)
    flags["R"]=any((m,c,j2) in singles and config["duration"][m][c][j]!=config["duration"][m][c][j2]
                   for m,c,j in singles for j2 in free if j2>j)
    flags["P"]=any(not state["components"][m][c]["failed"] and config["duration"][m][c][j]>1
                   for m,c,j in singles)
    flags["Q"]=any(0<config["duration"][m][c][jb]<config["duration"][m][c][jf]
                   for m,c,jf in singles for jb in busy)
    machines={m for m,c,j in singles}
    joint_machines={tuple(sorted((a[0][0],a[1][0]))) for a in actions if len(a)==2}
    flags["J"]=any((m,n) not in joint_machines for m,n in itertools.combinations(sorted(machines),2))
    lengths={}
    for action in actions:
        if len(action)>1:
            machines=tuple(m for m,c,j in action)
            vector=tuple((m,config["duration"][m][c][j]) for m,c,j in action)
            lengths.setdefault(machines,set()).add(vector)
    flags["A"]=any(len(v)>1 for v in lengths.values())
    return flags


def audit_structural_selection(run_dir,source,manifest,index):
    stored=[json.loads(line) for line in (run_dir/"structural.jsonl").read_text().splitlines()]
    keyed={(r["case"],r["reference"],r["seed"]):r for r in stored}
    if len(keyed)!=len(stored):
        raise ValueError("duplicate structural episode")
    selected={(r["case"],r["reference"],r["seed"],r["tick"]) for r in index}
    expected_selected=set()
    expected_count=0
    for case in manifest["cases"]:
        cfg=json.loads((source/"configs"/f"{case}.json").read_text())
        for reference in ("corrective","risk"):
            choices=[]
            episode_paths=sorted((source/"episodes"/case/reference).glob("*.json"))
            if not episode_paths:
                raise ValueError("missing structural source population")
            for path in episode_paths:
                episode=json.loads(path.read_text())
                seed=episode["seed"]
                counts={k:0 for k in ("T","C","W","J","A","R","Q","P")}
                candidates=[]
                for record in episode["records"][:48]:
                    before=record["before"]
                    actions=catalogue(cfg,before)
                    flags=independent_flags(cfg,before,actions)
                    for kind,flag in flags.items():
                        counts[kind]+=int(flag)
                    if any(flags.values()):
                        rank=priority("C1-state",case,reference,seed,record["tick"])
                        candidates.append((rank,seed,record["tick"]))
                expected=dict(case=case,reference=reference,seed=seed,H_ticks=48,
                              motif_ticks=counts,
                              motif_any={k:counts[k]>0 for k in counts},
                              eligible_ticks=len(candidates))
                if keyed.get((case,reference,seed))!=expected:
                    raise ValueError("structural prevalence arithmetic")
                expected_count+=1
                if candidates:
                    choices.append(min(candidates))
            for rank,seed,tick in sorted(choices)[:manifest["state_limit"]]:
                expected_selected.add((case,reference,seed,tick))
    if len(stored)!=expected_count or selected!=expected_selected:
        raise ValueError("selected-state coverage/priority")
    return dict(source_episodes=expected_count,selected_states=len(expected_selected))

def audit_run(run_dir:Path):
    manifest=json.loads((run_dir/"manifest.json").read_text())
    if manifest["status"]!="complete": raise ValueError("incomplete diagnostic collection")
    if manifest["schema"]!="JCST-C1-diagnostics-v1": raise ValueError("schema")
    bank_stage=manifest["bank_stage"]
    validation_bank=manifest["validation_bank"]
    if (bank_stage,validation_bank) not in (("diagnostic","B"),("confirmation","C")):
        raise ValueError("bank role")
    if sha(ROOT/"docs/jcst_characterization_protocol_v1.md")!=manifest["protocol_hash"]:
        raise ValueError("protocol hash")
    for rel,digest in manifest["source_hashes"].items():
        if sha(ROOT/rel)!=digest: raise ValueError(f"source hash {rel}")
    source=Path(manifest["source_run"])
    source_meta=json.loads((source/"manifest.json").read_text())
    if (source_meta["stage"]!=bank_stage or source_meta["kind"]!=manifest["kind"]
            or source_meta["protocol_hash"]!=manifest["protocol_hash"]):
        raise ValueError("source/validation role or protocol")
    if sha(source/"manifest.json")!=manifest["source_manifest_hash"] or sha(source/"independent_audit.json")!=manifest["source_audit_hash"]:
        raise ValueError("source audit/provenance hash")
    index=[json.loads(line) for line in (run_dir/"index.jsonl").read_text().splitlines()]
    if len(index)!=manifest["expected_states"] or len(index)!=manifest["completed_states"]:
        raise ValueError("state coverage")
    if len({(r["case"],r["reference"],r["seed"],r["tick"]) for r in index})!=len(index):
        raise ValueError("duplicate state")
    started=time.monotonic()
    structural=audit_structural_selection(run_dir,source,manifest,index)
    total=defaultdict(int)
    for row in tqdm(index,desc="C1 independent diagnostic audit"):
        folder=run_dir/row["folder"]
        info=json.loads((folder/"state.json").read_text())
        if (info["case"],info["reference"],info["physical_seed"],info["tick"])!=(
                row["case"],row["reference"],row["seed"],row["tick"]):
            raise ValueError("state identity")
        episode_path=source/"episodes"/row["case"]/row["reference"]/f"{row['seed']}.json"
        if sha(episode_path)!=info["source_episode_hash"]:
            raise ValueError("source episode hash")
        episode=json.loads(episode_path.read_text())
        if episode["records"][row["tick"]]["before"]!=info["state"]:
            raise ValueError("source state mismatch")
        cfg=json.loads((source/"configs"/f"{row['case']}.json").read_text())
        counts=audit_state(folder,cfg,info,bank_stage,validation_bank)
        for key,value in counts.items(): total[key]+=value
    result=dict(status="PASS",bank_stage=bank_stage,validation_bank=validation_bank,
                states=len(index),structural=structural,branches=dict(total),
                elapsed_seconds=time.monotonic()-started,
                protocol_hash=manifest["protocol_hash"])
    (run_dir/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_run(args.run_dir)))


if __name__=="__main__":
    main()
