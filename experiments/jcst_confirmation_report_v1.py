"""C1 independent confirmation decisions and benchmark synthesis from audited raw data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from jcst_diagnostic_analyze_v1 import CASES, MOTIFS, interval, key

CANDIDATE_ROLE={
    "base":"core_candidate","generalist":"core_candidate","restricted":"core_candidate",
    "shortlife":"stress_candidate","scarce":"stress_candidate","scale4":"stress_candidate",
    "pm_expensive":"stress_candidate","downtime_high":"stress_candidate",
    "staggered":"stress_candidate","memoryless":"negative_control",
    "homogeneous":"negative_control","surplus":"negative_control",
    "longlife":"negative_control",
}


def require_audited(run:Path,stage:str|None=None):
    manifest=json.loads((run/"manifest.json").read_text())
    audit=json.loads((run/"independent_audit.json").read_text())
    if manifest["status"]!="complete" or audit["status"]!="PASS" or manifest["kind"]!="scientific":
        raise ValueError(f"run not scientifically audit-passed: {run}")
    if stage and manifest.get("stage")!=stage and manifest.get("bank_stage")!=stage:
        raise ValueError("run stage mismatch")
    if manifest["protocol_hash"]!=audit["protocol_hash"]:
        raise ValueError("run protocol/audit mismatch")
    return manifest


def physical_by_case(run:Path):
    data=defaultdict(dict)
    for line in (run/"index.jsonl").read_text().splitlines():
        index=json.loads(line)
        episode=json.loads((run/index["episode_file"]).read_text())
        identity=(index["case"],index["arm"])
        if index["seed"] in data[identity]:
            raise ValueError("duplicate physical episode")
        data[identity][index["seed"]]=episode
    return data


def adjusted_motif_intervals(economic:Path,case:str,reference:str,draws:int=2000):
    indexed=[json.loads(line) for line in (economic/"index.jsonl").read_text().splitlines()]
    chosen=[r for r in indexed if r["case"]==case and r["reference"]==reference]
    vectors={kind:{} for kind in ("T","C","W","J")}
    for row in chosen:
        folder=economic/row["folder"]
        selected=json.loads((folder/"selection.json").read_text())
        tapes=defaultdict(dict)
        for line in (folder/"C.jsonl").read_text().splitlines():
            branch=json.loads(line)
            tapes[key(branch["arm"])][branch["tape"]]=branch["total_cost"]
        for kind in vectors:
            pair=selected[kind]["extreme"]
            if pair is None: continue
            left,right=key(pair["left"]),key(pair["right"])
            sign=1 if selected[kind]["orientation"]=="left-minus-right" else -1
            if set(tapes[left])!=set(range(12)) or set(tapes[right])!=set(range(12)):
                raise ValueError("confirmation tape coverage")
            vectors[kind][row["seed"]]=np.asarray(
                [(tapes[left][t]-tapes[right][t])*sign for t in range(12)],dtype=float)
    means={kind:float(np.mean([v.mean() for v in grouped.values()]))
           for kind,grouped in vectors.items() if grouped}
    union=sorted({seed for grouped in vectors.values() for seed in grouped})
    if not union: return {kind:dict(n=0,mean=float("nan"),lo=float("nan"),hi=float("nan"))
                          for kind in vectors}
    rng=np.random.default_rng(20261025+CASES.index(case)*37+(0 if reference=="risk" else 1))
    samples={kind:[] for kind in vectors}
    for _ in range(draws):
        resampled=rng.choice(union,size=len(union),replace=True)
        for kind,grouped in vectors.items():
            values=[]
            for seed in resampled:
                if seed in grouped:
                    tape_indices=rng.integers(0,12,size=12)
                    values.append(float(grouped[seed][tape_indices].mean()))
            samples[kind].append(float(np.mean(values)) if values else float("nan"))
    standard={kind:(float(np.nanstd(np.asarray(draw,dtype=float),ddof=1))
                    if len(vectors[kind])>=2 else float("nan"))
              for kind,draw in samples.items()}
    available=[kind for kind in vectors if len(vectors[kind])>=2 and
               np.isfinite(standard[kind]) and standard[kind]>0]
    max_t=[]
    for i in range(draws):
        max_t.append(max((abs((samples[kind][i]-means[kind])/standard[kind])
                          for kind in available if np.isfinite(samples[kind][i])),
                         default=0.))
    critical=float(np.quantile(max_t,.95))
    result={}
    for kind,grouped in vectors.items():
        n=len(grouped)
        mean=means.get(kind,float("nan"))
        se=standard[kind]
        result[kind]=dict(n=n,mean=mean,
            lo=mean-critical*se if kind in available else float("nan"),
            hi=mean+critical*se if kind in available else float("nan"),
            adjusted_critical=critical,
            tape_se=statistics.fmean(float(np.std(v,ddof=1)/np.sqrt(12))
                                     for v in grouped.values()) if grouped else float("nan"))
    return result


def summarize(physical:Path,economic:Path,sensitivity:list[Path],out:Path):
    phys_meta=require_audited(physical,"confirmation")
    econ_meta=require_audited(economic,"confirmation")
    if econ_meta["validation_bank"]!="C" or econ_meta["source_run"]!=str(physical.resolve()):
        raise ValueError("confirmation physical/economic source mismatch")
    sens_meta=[require_audited(path,"sensitivity") for path in sensitivity]
    if {(m["horizon"],m["tail"]) for m in sens_meta}!={(24,8),(48,0),(96,32)}:
        raise ValueError("three locked sensitivity settings required")
    protocol_hash=phys_meta["protocol_hash"]
    if any(m["protocol_hash"]!=protocol_hash for m in [econ_meta,*sens_meta]):
        raise ValueError("protocol mismatch across stages")
    data=physical_by_case(physical)
    structural=[json.loads(line) for line in (economic/"structural.jsonl").read_text().splitlines()]
    by_stratum=defaultdict(list)
    for row in structural: by_stratum[(row["case"],row["reference"])].append(row)
    costs=[]
    motifs=[]
    benchmark=[]
    for case in CASES:
        arms={arm:episodes for (c,arm),episodes in data.items() if c==case}
        if len(arms)!=6 or any(len(group)!=128 for group in arms.values()):
            raise ValueError("confirmation physical coverage")
        ids=set(next(iter(arms.values())))
        if any(set(group)!=ids for group in arms.values()):
            raise ValueError("unpaired physical seeds")
        risk=arms["risk"]
        for arm,episodes in sorted(arms.items()):
            values={seed:episodes[seed]["total_cost"] for seed in ids}
            paired={seed:risk[seed]["total_cost"]-episodes[seed]["total_cost"]
                    for seed in ids}
            lo,hi=interval(list(paired.values()),20261025+CASES.index(case)*23+len(arm))
            costs.append(dict(case=case,controller=arm,episodes=128,
                mean_total_cost=statistics.fmean(values.values()),
                risk_minus_controller=statistics.fmean(paired.values()),
                paired_lo=lo,paired_hi=hi,
                adjusted_lo=float("nan"),adjusted_hi=float("nan"),
                strong_control_advantage=False,
                mean_H_cost=statistics.fmean(ep["controlled_cost"] for ep in episodes.values()),
                mean_tail_cost=statistics.fmean(ep["tail_cost"] for ep in episodes.values()),
                mean_forecast_transitions=statistics.fmean(
                    ep["compute"]["simulated_transitions"] for ep in episodes.values()),
                mean_selection_seconds=statistics.fmean(
                    ep["compute"]["selection_seconds"] for ep in episodes.values()),
                mean_episode_wall_seconds=statistics.fmean(
                    ep["episode_wall_seconds"] for ep in episodes.values())))
        # Simultaneous physical-seed max-|t| interval for the 13 rollout2-vs-risk cases.
        confirmed={}
        for reference in ("corrective","risk"):
            group=by_stratum[(case,reference)]
            if len(group)!=128: raise ValueError("confirmation structural coverage")
            adjusted=adjusted_motif_intervals(economic,case,reference)
            cU=json.loads((physical/"configs"/f"{case}.json").read_text())["unavailable_cost"]
            for kind in ("T","C","W","J"):
                presence=sum(row["motif_any"][kind] for row in group)
                estimate=adjusted[kind]
                passed=(presence>=12 and estimate["n"]>=2 and estimate["mean"]>=cU
                        and np.isfinite(estimate["lo"]) and estimate["lo"]>0)
                motifs.append(dict(case=case,reference=reference,motif=kind,
                    episodes_with_motif=presence,selected_states_with_pair=estimate["n"],
                    oriented_C_gap=estimate["mean"],adjusted_lo=estimate["lo"],
                    adjusted_hi=estimate["hi"],mean_tape_se=estimate["tape_se"],
                    cU_scale=cU,confirmed=passed))
                confirmed[(reference,kind)]=passed
        role=CANDIDATE_ROLE[case]
        t_c_w_reachable=all(any(row["case"]==case and row["motif"]==kind and
                                   row["episodes_with_motif"]>=12 for row in motifs)
                            for kind in ("T","C","W"))
        any_consequence=any(value for value in confirmed.values())
        recommended=(role=="core_candidate" and t_c_w_reachable and any_consequence)
        benchmark.append(dict(case=case,prespecified_role=role,
            t_c_w_reachable=t_c_w_reachable,confirmed_conditional_consequence=any_consequence,
            recommended_core=recommended,
            final_role=("core" if recommended else
                        "unqualified_core_candidate" if role=="core_candidate" else role)))
    ids=sorted(data[("base","risk")])
    matrix=np.asarray([[data[(case,"risk")][seed]["total_cost"]-
                        data[(case,"rollout2")][seed]["total_cost"]
                        for seed in ids] for case in CASES],dtype=float)
    observed=matrix.mean(axis=1)
    rng=np.random.default_rng(20261025)
    draws=rng.integers(0,len(ids),size=(2000,len(ids)))
    boot=matrix[:,draws].mean(axis=2)
    standard=boot.std(axis=1,ddof=1)
    t=np.abs((boot-observed[:,None])/np.where(standard[:,None]>0,standard[:,None],np.inf))
    critical=float(np.quantile(np.max(t,axis=0),.95))
    for row in costs:
        if row["controller"]=="rollout2":
            i=CASES.index(row["case"])
            row["adjusted_lo"]=float(observed[i]-critical*standard[i])
            row["adjusted_hi"]=float(observed[i]+critical*standard[i])
            cU=json.loads((physical/"configs"/f"{row['case']}.json").read_text())["unavailable_cost"]
            row["strong_control_advantage"]=(row["risk_minus_controller"]>=cU
                                               and row["adjusted_lo"]>0)
    sensitivity_rows=[]
    for run,meta in zip(sensitivity,sens_meta):
        population=physical_by_case(run)
        H,B=meta["horizon"],meta["tail"]
        for (case,arm),episodes in sorted(population.items()):
            if len(episodes)!=64:raise ValueError("sensitivity coverage")
            M=len(next(iter(episodes.values()))["initial_state"]["components"])
            values=[ep["total_cost"]/(M*(H+B)) for ep in episodes.values()]
            lo,hi=interval(values,20261025+H*101+B*13+CASES.index(case))
            sensitivity_rows.append(dict(case=case,controller=arm,horizon=H,tail=B,
                episodes=64,mean_cost_per_machine_interval=statistics.fmean(values),
                physical_lo=lo,physical_hi=hi,
                mean_H_cost=statistics.fmean(ep["controlled_cost"] for ep in episodes.values()),
                mean_tail_cost=statistics.fmean(ep["tail_cost"] for ep in episodes.values())))
    out.mkdir(parents=True,exist_ok=False)
    def write(name,items):
        with (out/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]))
            writer.writeheader();writer.writerows(items)
    write("physical_controls.csv",costs)
    write("confirmation_motifs.csv",motifs)
    write("sensitivity.csv",sensitivity_rows)
    write("benchmark_roles.csv",benchmark)
    result=dict(schema="C1-confirmation-v1",protocol_hash=protocol_hash,
                analysis_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numpy_version=np.__version__,
                physical_run=str(physical.resolve()),economic_run=str(economic.resolve()),
                sensitivity_runs=[str(x.resolve()) for x in sensitivity],
                physical_controls=costs,confirmation_motifs=motifs,
                sensitivity=sensitivity_rows,benchmark_roles=benchmark)
    (out/"confirmation.json").write_text(json.dumps(result,indent=2)+"\n")
    render(result,out)
    write_report(result,out)
    return result


def render(result,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5))
    for arm in ("corrective","risk","rollout2"):
        group=[next(r for r in result["physical_controls"] if r["case"]==case and
                    r["controller"]==arm) for case in CASES]
        ax.plot(CASES,[r["mean_total_cost"] for r in group],marker="o",label=arm)
    ax.set_ylabel("Paired H+tail cost, simulation units")
    ax.set_title("Independent confirmation: physical controls")
    ax.tick_params(axis="x",rotation=50)
    ax.legend();fig.tight_layout()
    fig.savefig(out/"confirmed_physical_cost.png",dpi=180)
    fig.savefig(out/"confirmed_physical_cost.pdf")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for case in ("base","shortlife","homogeneous","restricted","generalist","scale4"):
        group=sorted((r for r in result["sensitivity"] if r["case"]==case and
                      r["controller"]=="risk"),key=lambda r:r["horizon"])
        ax.plot([r["horizon"]+r["tail"] for r in group],
                [r["mean_cost_per_machine_interval"] for r in group],
                marker="o",label=case)
    ax.set_xlabel("Total evaluated intervals H+B")
    ax.set_ylabel("Cost per machine interval")
    ax.set_title("Prelocked horizon/tail sensitivity under risk control")
    ax.legend();fig.tight_layout()
    fig.savefig(out/"horizon_sensitivity.png",dpi=180)
    fig.savefig(out/"horizon_sensitivity.pdf")
    plt.close(fig)


def write_report(result,out):
    qualified=[r["case"] for r in result["benchmark_roles"] if r["recommended_core"]]
    lines=[
        "# JCST/1.1 C1 independent confirmation",
        "",
        "All numbers below use simulation units. Confirmation is separate from sealed TEST.",
        "Physical episodes are paired by raw seed; conditional branches and tapes are not physical replicates.",
        "",
        "## Prespecified benchmark roles",
        "",
        "| Case | Prior role | Confirmed conditional contrast | Core recommendation |",
        "| --- | --- | --- | --- |",
    ]
    for row in result["benchmark_roles"]:
        lines.append(f"| {row['case']} | {row['prespecified_role']} | {row['confirmed_conditional_consequence']} | {row['recommended_core']} |")
    lines.extend(["","## Scope and decision",""])
    if qualified:
        lines.append("Core recommendation candidates supported by the locked rule: "+", ".join(qualified)+".")
        lines.append("One possible next method step, requiring its own authorization: a matched-compute direct-state versus demand–capacity representation pilot on the confirmed core population.")
    else:
        lines.append("No core candidate passed the locked confirmation rule. Stop before a learned-method pilot; improve operational calibration or redefine the benchmark through a new protocol.")
    lines.extend([
        "",
        "The full physical, motif, sensitivity and compute tables are adjacent CSV/JSON artifacts.",
        "A positive local contrast does not establish achievable episode gain. A rollout loss does not prove a need for RL.",
        "Intervals exclude hazard-model misspecification, monetary calibration and field external validity.",
    ])
    (out/"confirmation_report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--physical",type=Path,required=True)
    parser.add_argument("--economic",type=Path,required=True)
    parser.add_argument("--sensitivity",type=Path,nargs=3,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=summarize(args.physical,args.economic,args.sensitivity,args.output)
    print(json.dumps(dict(status="created",output=str(args.output),
                          core=[r["case"] for r in report["benchmark_roles"] if r["recommended_core"]])))


if __name__=="__main__":
    main()
