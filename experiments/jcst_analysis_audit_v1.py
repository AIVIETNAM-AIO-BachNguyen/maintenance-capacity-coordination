"""Independent arithmetic audit for C1 analysis artifacts.

This checker recomputes point estimates and decisions from raw audited banks.
It does not use the collector transition or the analysis functions being checked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def close(a,b,tol=1e-9):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=tol,abs_tol=tol)


def require(run):
    manifest=json.loads((run/"manifest.json").read_text())
    audit=json.loads((run/"independent_audit.json").read_text())
    if manifest["status"]!="complete" or audit["status"]!="PASS":
        raise ValueError("unverified input run")
    if manifest["protocol_hash"]!=audit["protocol_hash"]:
        raise ValueError("protocol/audit hash")
    return manifest


def physical(run):
    data=defaultdict(dict)
    for line in (run/"index.jsonl").read_text().splitlines():
        row=json.loads(line)
        path=run/row["episode_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["episode_hash"]:
            raise ValueError("raw episode hash")
        ep=json.loads(path.read_text())
        key=(row["case"],row["arm"])
        if row["seed"] in data[key]:raise ValueError("duplicate physical episode")
        data[key][row["seed"]]=ep
    return data


def audit_selection(selection_file):
    doc=json.loads(selection_file.read_text())
    run=Path(doc["source_run"])
    meta=require(run)
    if meta["stage"]!="selection" or meta["kind"]!="scientific":
        raise ValueError("selection source")
    if (hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest()!=doc["source_manifest_hash"]
            or hashlib.sha256((run/"independent_audit.json").read_bytes()).hexdigest()!=doc["source_audit_hash"]):
        raise ValueError("selection provenance")
    data=physical(run)
    for case,choice in doc["selected"].items():
        period={}
        threshold={}
        for p in (4,8):
            values=list(data[(case,f"periodic{p}")].values())
            if len(values)!=64:raise ValueError("period tuning coverage")
            period[p]=statistics.fmean(ep["total_cost"] for ep in values)
        for age in (2,5):
            values=list(data[(case,f"threshold{age}")].values())
            if len(values)!=64:raise ValueError("threshold tuning coverage")
            threshold[age]=statistics.fmean(ep["total_cost"] for ep in values)
        if choice!=(dict(period=min(period,key=lambda p:(period[p],p)),
                         age=min(threshold,key=lambda a:(threshold[a],a)))):
            raise ValueError("baseline selection arithmetic")
    return dict(status="PASS",kind="baseline_selection",cases=len(doc["selected"]),
                selection_file_hash=hashlib.sha256(selection_file.read_bytes()).hexdigest(),
                protocol_hash=meta["protocol_hash"])


def audit_atlas(directory):
    summary=json.loads((directory/"atlas.json").read_text())
    run=Path(summary["source_run"])
    meta=require(run)
    if meta["stage"]!="characterization":raise ValueError("atlas source")
    data=physical(run)
    for row in summary["rows"]:
        episodes=list(data[(row["case"],row["controller"])].values())
        if len(episodes)!=128:raise ValueError("atlas coverage")
        checks={
            "total_cost":statistics.fmean(ep["total_cost"] for ep in episodes),
            "H_cost":statistics.fmean(ep["controlled_cost"] for ep in episodes),
            "tail_cost":statistics.fmean(ep["tail_cost"] for ep in episodes),
            "failure_count":statistics.fmean(ep["metrics"]["component_failures"] for ep in episodes),
            "PM_count":statistics.fmean(ep["metrics"]["pm_starts"] for ep in episodes),
            "CM_count":statistics.fmean(ep["metrics"]["cm_starts"] for ep in episodes),
            "failed_waiting":statistics.fmean(ep["metrics"]["failed_waiting_intervals"] for ep in episodes),
        }
        for key,value in checks.items():
            if not close(row[key],value):raise ValueError(f"atlas estimate {key}")
        if not close(row["total_cost"],row["H_cost"]+row["tail_cost"]):
            raise ValueError("atlas H/tail")
    return dict(status="PASS",kind="atlas",rows=len(summary["rows"]))


def arm_key(arm):
    return (tuple(tuple(x) for x in arm["action"]),
            tuple(arm["wait_for"]) if arm["wait_for"] else None)


def audit_diagnostics(directory):
    summary=json.loads((directory/"summary.json").read_text())
    run=Path(summary["source_run"])
    meta=require(run)
    if meta["schema"]!="JCST-C1-diagnostics-v1":
        raise ValueError("diagnostic source")
    bank=meta["validation_bank"]
    structural=[json.loads(line) for line in (run/"structural.jsonl").read_text().splitlines()]
    grouped=defaultdict(list)
    for row in structural:grouped[(row["case"],row["reference"])].append(row)
    for item in summary["prevalence"]:
        source=grouped[(item["case"],item["reference"])]
        kind=item["motif"]
        estimate=statistics.fmean(r["motif_ticks"][kind]/48 for r in source)
        if len(source)!=128 or not close(item["tick_fraction"],estimate):
            raise ValueError("prevalence estimator")
    selected={}
    for line in (run/"index.jsonl").read_text().splitlines():
        row=json.loads(line)
        folder=run/row["folder"]
        pairs=json.loads((folder/"selection.json").read_text())
        costs=defaultdict(dict)
        for raw in (folder/f"{bank}.jsonl").read_text().splitlines():
            branch=json.loads(raw)
            costs[arm_key(branch["arm"])][branch["tape"]]=branch["total_cost"]
        selected[(row["case"],row["reference"],row["seed"])]=(pairs,costs)
    for item in summary["conditional_cost"]:
        values=[]
        for (case,reference,seed),(pairs,costs) in selected.items():
            if (case,reference)!=(item["case"],item["reference"]):continue
            frozen=pairs[item["motif"]]["extreme"]
            if frozen is None:continue
            a,b=arm_key(frozen["left"]),arm_key(frozen["right"])
            sign=1 if pairs[item["motif"]]["orientation"]=="left-minus-right" else -1
            values.append(statistics.fmean((costs[a][t]-costs[b][t])*sign for t in range(12)))
        expected=statistics.fmean(values) if values else float("nan")
        if len(values)!=item["states_with_pair"] or not close(
                item["validation_extreme_gap"],expected):
            raise ValueError("conditional cost estimator")
    for item in summary["compute_frontier"]:
        gaps=[]
        transitions=[]
        for (case,reference,seed),(pairs,costs) in selected.items():
            if (case,reference)!=(item["case"],item["reference"]):continue
            row=pairs["frontier"][item["budget"]]
            gaps.append(pairs["risk_validation_cost"]-row["validation_cost"])
            transitions.append(row["simulated_transitions"])
        expected=statistics.fmean(gaps) if gaps else float("nan")
        if not close(item["risk_minus_selector_validation_cost"],expected):
            raise ValueError("frontier cost estimator")
        if transitions and not close(item["mean_forecast_transitions"],statistics.fmean(transitions)):
            raise ValueError("frontier compute estimator")
    return dict(status="PASS",kind="economic_diagnostics",
                prevalence_rows=len(summary["prevalence"]),
                cost_rows=len(summary["conditional_cost"]))


def audit_physical_r2(directory):
    report=json.loads((directory/"physical_R2_frontier.json").read_text())
    diagnostic=Path(report["diagnostic_run"])
    rollout=Path(report["rollout_run"])
    dmeta=require(diagnostic);rmeta=require(rollout)
    if (dmeta["stage"]!="diagnostic" or rmeta["stage"]!="diagnostic_rollout"
            or dmeta["protocol_hash"]!=rmeta["protocol_hash"]
            or report["protocol_hash"]!=dmeta["protocol_hash"]):
        raise ValueError("physical R2 source roles")
    d=physical(diagnostic);r=physical(rollout)
    expected_cases={"base","homogeneous","scarce","restricted","generalist","scale4"}
    if {(x["case"],x["reference"]) for x in report["rows"]} != {
            (case,arm) for case in expected_cases for arm in ("risk","corrective")}:
        raise ValueError("physical R2 table coverage")
    for row in report["rows"]:
        case,arm=row["case"],row["reference"]
        seeds=set(range(12_000_000,12_000_064))
        if not seeds.issubset(d[(case,arm)]) or set(r[(case,"rollout2")])!=seeds:
            raise ValueError("physical R2 seed pairing")
        diff=statistics.fmean(d[(case,arm)][seed]["total_cost"]-
            r[(case,"rollout2")][seed]["total_cost"] for seed in seeds)
        compute=statistics.fmean(r[(case,"rollout2")][seed]["compute"]["simulated_transitions"]
                                 for seed in seeds)
        if not close(row["reference_minus_R2"],diff) or not close(
                row["mean_R2_forecast_transitions"],compute):
            raise ValueError("physical R2 point/compute estimator")
    return dict(status="PASS",kind="physical_r2",rows=len(report["rows"]))


def audit_confirmation(directory):
    report=json.loads((directory/"confirmation.json").read_text())
    physical_run=Path(report["physical_run"])
    economic_run=Path(report["economic_run"])
    require(physical_run);require(economic_run)
    data=physical(physical_run)
    case_cost={case:json.loads((physical_run/"configs"/f"{case}.json").read_text())["unavailable_cost"]
               for case in {row["case"] for row in report["benchmark_roles"]}}
    for item in report["confirmation_motifs"]:
        expected=(item["episodes_with_motif"]>=12
                  and item["selected_states_with_pair"]>=2
                  and item["oriented_C_gap"]>=case_cost[item["case"]]
                  and math.isfinite(item["adjusted_lo"]) and item["adjusted_lo"]>0)
        if item["confirmed"]!=expected:
            raise ValueError("motif gate arithmetic")
    for path_text in report["sensitivity_runs"]:
        run=Path(path_text)
        meta=require(run)
        if meta["stage"]!="sensitivity":raise ValueError("sensitivity stage")
        panel=physical(run)
        for row in report["sensitivity"]:
            if (row["horizon"],row["tail"])!=(meta["horizon"],meta["tail"]):
                continue
            group=panel[(row["case"],row["controller"])]
            if len(group)!=64:raise ValueError("sensitivity coverage")
            M=len(next(iter(group.values()))["initial_state"]["components"])
            normalized=statistics.fmean(ep["total_cost"]/(M*(meta["horizon"]+meta["tail"]))
                                        for ep in group.values())
            if not close(row["mean_cost_per_machine_interval"],normalized):
                raise ValueError("sensitivity normalization")
    for row in report["physical_controls"]:
        group=data[(row["case"],row["controller"])]
        risk=data[(row["case"],"risk")]
        ids=set(group)
        if len(ids)!=128 or ids!=set(risk):
            raise ValueError("confirmation physical pairing")
        mean=statistics.fmean(ep["total_cost"] for ep in group.values())
        diff=statistics.fmean(risk[s]["total_cost"]-group[s]["total_cost"] for s in ids)
        if not close(row["mean_total_cost"],mean) or not close(row["risk_minus_controller"],diff):
            raise ValueError("confirmation physical estimator")
        if row["controller"]=="rollout2":
            cU=case_cost[row["case"]]
            expected=diff>=cU and row["adjusted_lo"]>0
            if row["strong_control_advantage"]!=expected:
                raise ValueError("strong-control gate arithmetic")
    motif={(r["case"],r["reference"],r["motif"]):r for r in report["confirmation_motifs"]}
    for row in report["benchmark_roles"]:
        case=row["case"]
        reachable=all(any(motif[(case,reference,kind)]["episodes_with_motif"]>=12
                          for reference in ("risk","corrective")) for kind in ("T","C","W"))
        economic=any(motif[(case,reference,kind)]["confirmed"]
                     for reference in ("risk","corrective") for kind in ("T","C","W","J"))
        expected=row["prespecified_role"]=="core_candidate" and reachable and economic
        if (row["t_c_w_reachable"],row["confirmed_conditional_consequence"],
                row["recommended_core"])!=(reachable,economic,expected):
            raise ValueError("benchmark decision arithmetic")
    return dict(status="PASS",kind="confirmation",cases=len(report["benchmark_roles"]))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="kind",required=True)
    for kind in ("selection","atlas","diagnostics","physical_r2","confirmation"):
        part=sub.add_parser(kind);part.add_argument("path",type=Path)
    args=parser.parse_args()
    result={"selection":audit_selection,"atlas":audit_atlas,
            "diagnostics":audit_diagnostics,"physical_r2":audit_physical_r2,
            "confirmation":audit_confirmation}[args.kind](args.path)
    target=args.path.parent if args.path.is_file() else args.path
    (target/(f"{args.kind}_analysis_audit.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
