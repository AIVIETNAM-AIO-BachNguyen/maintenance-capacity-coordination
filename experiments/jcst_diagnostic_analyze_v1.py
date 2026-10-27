"""C1 conditional consequence and compute-frontier estimates from audited banks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

MOTIFS=("T","C","W","J","A","R","Q","P")
CASES=("base","longlife","shortlife","memoryless","homogeneous","surplus",
       "scarce","restricted","generalist","pm_expensive","downtime_high",
       "staggered","scale4")


def interval(values,role,draws=2000):
    x=np.asarray(values,dtype=float)
    if len(x)<2: return (float("nan"),float("nan"))
    rng=np.random.default_rng(role)
    means=x[rng.integers(0,len(x),size=(draws,len(x)))].mean(axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def nested_interval(tape_vectors,role,draws=1000):
    if len(tape_vectors)<2: return (float("nan"),float("nan"))
    a=np.asarray(tape_vectors,dtype=float)
    rng=np.random.default_rng(role)
    values=[]
    for _ in range(draws):
        physical=rng.integers(0,len(a),size=len(a))
        tape=rng.integers(0,a.shape[1],size=(len(a),a.shape[1]))
        values.append(float(np.take_along_axis(a[physical],tape,axis=1).mean()))
    return float(np.quantile(values,.025)),float(np.quantile(values,.975))


def key(arm):
    return (tuple(tuple(e) for e in arm["action"]),
            tuple(arm["wait_for"]) if arm["wait_for"] else None)


def summarize(run:Path,out:Path):
    manifest=json.loads((run/"manifest.json").read_text())
    audit=json.loads((run/"independent_audit.json").read_text())
    if manifest["status"]!="complete" or audit["status"]!="PASS" or manifest["kind"]!="scientific":
        raise ValueError("scientific diagnostic bank not independently audited")
    bank=manifest["validation_bank"]
    stage=manifest["bank_stage"]
    if (stage,bank) not in (("diagnostic","B"),("confirmation","C")):
        raise ValueError("bank role")
    structural=[json.loads(s) for s in (run/"structural.jsonl").read_text().splitlines()]
    index=[json.loads(s) for s in (run/"index.jsonl").read_text().splitlines()]
    structural_by=defaultdict(list)
    for row in structural:
        structural_by[(row["case"],row["reference"])].append(row)
    states=defaultdict(list)
    for row in index:
        folder=run/row["folder"]
        selected=json.loads((folder/"selection.json").read_text())
        B=defaultdict(dict)
        for line in (folder/f"{bank}.jsonl").read_text().splitlines():
            branch=json.loads(line)
            B[key(branch["arm"])][branch["tape"]]=branch["total_cost"]
        if any(set(v)!=set(range(12)) for v in B.values()):
            raise ValueError("B tape coverage")
        states[(row["case"],row["reference"])].append((row,selected,B))
    prevalence=[]
    consequence=[]
    frontier=[]
    joint=[]
    for case in CASES:
        for reference in ("corrective","risk"):
            group=structural_by[(case,reference)]
            if len(group)!=128:
                raise ValueError("structural episode coverage")
            sample=states[(case,reference)]
            for kind in MOTIFS:
                ticks=[r["motif_ticks"][kind]/48 for r in group]
                any_episode=[int(r["motif_any"][kind]) for r in group]
                lo,hi=interval(ticks,20261025+CASES.index(case)*101+MOTIFS.index(kind)*7)
                alo,ahi=interval(any_episode,20261026+CASES.index(case)*101+MOTIFS.index(kind)*7)
                prevalence.append(dict(case=case,reference=reference,motif=kind,
                    episodes=128,tick_fraction=statistics.fmean(ticks),
                    tick_lo=lo,tick_hi=hi,episode_fraction=statistics.fmean(any_episode),
                    episode_lo=alo,episode_hi=ahi))
                extreme=[];sentinel=[];A_gaps=[];tapes=[]
                pair_count=[]
                for row,selected,B in sample:
                    item=selected[kind]
                    pair_count.append(item["count"])
                    if item["extreme"] is None: continue
                    e=item["extreme"];s=item["sentinel"]
                    left,right=key(e["left"]),key(e["right"])
                    sign=1 if item["orientation"]=="left-minus-right" else -1
                    vector=np.asarray([B[left][t]-B[right][t] for t in range(12)])*sign
                    extreme.append(float(vector.mean()))
                    tapes.append(vector)
                    A_gaps.append(abs(item["extreme_A_costs"][0]-item["extreme_A_costs"][1]))
                    sl,sr=key(s["left"]),key(s["right"])
                    sentinel.append(statistics.fmean(B[sl][t]-B[sr][t] for t in range(12)))
                lo,hi=interval(extreme,20261027+CASES.index(case)*101+MOTIFS.index(kind)*7)
                nlo,nhi=nested_interval(tapes,20261028+CASES.index(case)*101+MOTIFS.index(kind)*7)
                slo,shi=interval(sentinel,20261029+CASES.index(case)*101+MOTIFS.index(kind)*7)
                consequence.append(dict(case=case,reference=reference,motif=kind,
                    selected_states=len(sample),states_with_pair=len(extreme),
                    mean_pair_count=statistics.fmean(pair_count) if pair_count else float("nan"),
                    mean_A_selected_gap=statistics.fmean(A_gaps) if A_gaps else float("nan"),
                    validation_extreme_gap=statistics.fmean(extreme) if extreme else float("nan"),
                    physical_lo=lo,physical_hi=hi,nested_lo=nlo,nested_hi=nhi,
                    mean_forecast_tape_SE=statistics.fmean(
                        float(np.std(v,ddof=1)/np.sqrt(12)) for v in tapes) if tapes else float("nan"),
                    sentinel_signed_gap=statistics.fmean(sentinel) if sentinel else float("nan"),
                    sentinel_lo=slo,sentinel_hi=shi))
            for label in ("R2","R4","R8"):
                gaps=[];compute=[];latency=[];changes=[]
                for row,selected,B in sample:
                    arm=selected["frontier"][label]
                    gaps.append(selected["risk_validation_cost"]-arm["validation_cost"])
                    compute.append(arm["simulated_transitions"])
                    latency.append(arm["selection_seconds"])
                    changes.append(int(arm["half_actions"][0]!=arm["half_actions"][1]))
                lo,hi=interval(gaps,20261030+CASES.index(case)*101+len(label))
                frontier.append(dict(case=case,reference=reference,budget=label,
                    selected_states=len(sample),risk_minus_selector_validation_cost=statistics.fmean(gaps) if gaps else float("nan"),
                    physical_lo=lo,physical_hi=hi,
                    mean_forecast_transitions=statistics.fmean(compute) if compute else float("nan"),
                    mean_selection_seconds=statistics.fmean(latency) if latency else float("nan"),
                    half_root_disagreement=statistics.fmean(changes) if changes else float("nan")))
            joint_values=[selected["joint_interaction_validation"] for row,selected,B in sample
                          if selected["joint_quartet"] is not None]
            lo,hi=interval(joint_values,20261031+CASES.index(case)*101)
            joint.append(dict(case=case,reference=reference,selected_states=len(sample),
                joint_quartets=len(joint_values),
                conditional_nonadditivity=statistics.fmean(joint_values) if joint_values else float("nan"),
                physical_lo=lo,physical_hi=hi))
    out.mkdir(parents=True,exist_ok=False)
    def write(name,items):
        with (out/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]))
            writer.writeheader();writer.writerows(items)
    write("prevalence.csv",prevalence)
    write("conditional_cost.csv",consequence)
    write("compute_frontier.csv",frontier)
    write("joint_interaction.csv",joint)
    (out/"summary.json").write_text(json.dumps(dict(schema="C1-diagnostic-summary-v1",
        bank_stage=stage,validation_bank=bank,source_run=str(run.resolve()),protocol_hash=manifest["protocol_hash"],
        prevalence=prevalence,conditional_cost=consequence,
        compute_frontier=frontier,joint_interaction=joint),indent=2)+"\n")
    render(prevalence,consequence,frontier,out)
    return out


def render(prevalence,consequence,frontier,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,6))
    data=np.asarray([[next(r["tick_fraction"] for r in prevalence
                   if r["case"]==case and r["reference"]=="risk" and r["motif"]==motif)
                   for motif in MOTIFS] for case in CASES])
    im=ax.imshow(data,aspect="auto",cmap="viridis",vmin=0,vmax=1)
    ax.set_xticks(range(len(MOTIFS)),MOTIFS)
    ax.set_yticks(range(len(CASES)),CASES)
    ax.set_title("Structural witness prevalence on risk trajectories (H ticks)")
    fig.colorbar(im,ax=ax,label="Episode-averaged tick fraction")
    fig.tight_layout()
    fig.savefig(out/"witness_prevalence.png",dpi=180)
    fig.savefig(out/"witness_prevalence.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,motif in zip(axes,("T","W")):
        group=[next(r for r in consequence if r["case"]==case and r["reference"]=="risk"
                    and r["motif"]==motif) for case in CASES]
        mean=np.asarray([r["validation_extreme_gap"] for r in group],dtype=float)
        low=np.asarray([r["nested_lo"] for r in group],dtype=float)
        high=np.asarray([r["nested_hi"] for r in group],dtype=float)
        ax.errorbar(range(len(CASES)),mean,yerr=np.vstack((mean-low,high-mean)),
                    fmt="o",capsize=3)
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_title(f"{motif}: Bank-A-selected contrast on independent validation bank")
        ax.set_xticks(range(len(CASES)),CASES,rotation=60,ha="right")
        ax.set_ylabel("Conditional cost gap, simulation units")
    fig.tight_layout()
    fig.savefig(out/"conditional_gaps.png",dpi=180)
    fig.savefig(out/"conditional_gaps.pdf")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for case in ("base","homogeneous","scarce","restricted","generalist","scale4"):
        rows=[next(r for r in frontier if r["case"]==case and r["reference"]=="risk"
                   and r["budget"]==label) for label in ("R2","R4","R8")]
        ax.plot([r["mean_forecast_transitions"] for r in rows],
                [r["risk_minus_selector_validation_cost"] for r in rows],marker="o",label=case)
    ax.axhline(0,color="black",linewidth=.8)
    ax.set_xlabel("Mean simulated forecast transitions per selection")
    ax.set_ylabel("Risk root minus selector root Bank-B cost")
    ax.set_title("Present-information compute frontier on sampled states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"compute_frontier.png",dpi=180)
    fig.savefig(out/"compute_frontier.pdf")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(summarize(args.run_dir,args.output))


if __name__=="__main__":
    main()
