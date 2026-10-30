"""C1 predeclared selection and descriptive analysis. Reads audited raw episodes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

CASES=(
    "base","longlife","shortlife","memoryless","homogeneous","surplus",
    "scarce","restricted","generalist","pm_expensive","downtime_high",
    "staggered","scale4",
)


def audited_run(path:Path, stage:str):
    manifest=json.loads((path/"manifest.json").read_text())
    audit=json.loads((path/"independent_audit.json").read_text())
    if manifest["stage"]!=stage or manifest["status"]!="complete" or audit["status"]!="PASS":
        raise ValueError("run not complete and independently audited")
    if audit["protocol_hash"]!=manifest["protocol_hash"]:
        raise ValueError("audit/protocol mismatch")
    return manifest


def rows(path:Path):
    for line in (path/"index.jsonl").read_text().splitlines():
        index=json.loads(line)
        episode_path=path/index["episode_file"]
        if hashlib.sha256(episode_path.read_bytes()).hexdigest()!=index["episode_hash"]:
            raise ValueError("raw episode changed after audit")
        yield index,json.loads(episode_path.read_text())


def paired_interval(values:dict[int,float], seed:int, draws:int=2000):
    ids=sorted(values)
    array=np.asarray([values[i] for i in ids],dtype=float)
    if len(array)<2:
        return [float("nan"),float("nan")]
    rng=np.random.default_rng(seed)
    indexes=rng.integers(0,len(array),size=(draws,len(array)))
    means=array[indexes].mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def select_baselines(path:Path,out:Path):
    manifest=audited_run(path,"selection")
    if manifest["kind"]!="scientific":
        raise ValueError("baseline selection needs the locked scientific seed bank")
    grouped=defaultdict(dict)
    for index,episode in rows(path):
        grouped[(index["case"],index["seed"])][index["arm"]]=episode["total_cost"]
    expected={f"periodic{p}" for p in (4,8)}|{f"threshold{a}" for a in (2,5)}
    chosen={}
    table=[]
    for case in CASES:
        seeds=[k[1] for k in grouped if k[0]==case]
        if len(seeds)!=64 or any(set(grouped[(case,seed)])!=expected for seed in seeds):
            raise ValueError("selection coverage")
        means={arm:statistics.fmean(grouped[(case,seed)][arm] for seed in seeds)
               for arm in sorted(expected)}
        period=min((4,8),key=lambda p:(means[f"periodic{p}"],p))
        age=min((2,5),key=lambda a:(means[f"threshold{a}"],a))
        chosen[case]=dict(period=period,age=age)
        table.append(dict(case=case,**means,selected_period=period,selected_age=age))
    doc=dict(status="AUDIT_PASS",stage="baseline_selection",
             source_run=str(path.resolve()),
             source_manifest_hash=hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest(),
             source_audit_hash=hashlib.sha256((path/"independent_audit.json").read_bytes()).hexdigest(),
             protocol_hash=manifest["protocol_hash"],
             selected=chosen,selection_table=table)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(doc,indent=2)+"\n")
    return doc



def render_atlas(table,out:Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    controllers=("wait","corrective","periodic4","periodic8",
                 "threshold2","threshold5","risk")
    positions=np.arange(len(CASES))
    fig,ax=plt.subplots(figsize=(14,7))
    width=.10
    for j,arm in enumerate(controllers):
        values=[next(r for r in table if r["case"]==case and r["controller"]==arm)
                for case in CASES]
        mean=np.asarray([r["total_cost"] for r in values])
        low=np.asarray([r["total_cost_lo"] for r in values])
        high=np.asarray([r["total_cost_hi"] for r in values])
        ax.errorbar(positions+(j-3)*width,mean,yerr=np.vstack((mean-low,high-mean)),
                    fmt="o",capsize=2,markersize=3,label=arm)
    ax.set_xticks(positions,CASES,rotation=45,ha="right")
    ax.set_ylabel("H+tail cost (simulation units)")
    ax.set_title("JCST/1.1 operating-regime atlas: physical-seed 95% intervals")
    ax.grid(axis="y",alpha=.2)
    ax.legend(ncol=4,fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"cost_atlas.png",dpi=180)
    fig.savefig(out/"cost_atlas.pdf")
    plt.close(fig)
    risk=[next(r for r in table if r["case"]==case and r["controller"]=="risk")
          for case in CASES]
    fig,ax=plt.subplots(figsize=(13,5))
    bottom=np.zeros(len(CASES))
    for label,color in (("working_fraction","#2c7a5b"),
                        ("breakdown_fraction","#b94747"),
                        ("service_fraction","#5473a3")):
        values=np.asarray([r[label] for r in risk])
        ax.bar(CASES,values,bottom=bottom,label=label.split("_")[0],color=color)
        bottom+=values
    ax.set_ylim(0,1)
    ax.set_ylabel("Fraction of machine intervals")
    ax.set_title("Derived machine modes under the fixed risk control")
    ax.tick_params(axis="x",rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"mode_atlas.png",dpi=180)
    fig.savefig(out/"mode_atlas.pdf")
    plt.close(fig)

def characterize(path:Path,out:Path):
    manifest=audited_run(path,"characterization")
    if manifest["kind"]!="scientific":
        raise ValueError("engineering smoke is not scientific characterization")
    episode_data=defaultdict(dict)
    for index,episode in rows(path):
        key=(index["case"],index["arm"])
        if index["seed"] in episode_data[key]:
            raise ValueError("duplicate physical seed")
        episode_data[key][index["seed"]]=episode
    table=[]
    for case in CASES:
        for arm in ("wait","corrective","periodic4","periodic8",
                    "threshold2","threshold5","risk"):
            episodes=episode_data[(case,arm)]
            if len(episodes)!=128:
                raise ValueError("characterization coverage")
            row=dict(case=case,controller=arm,episodes=128)
            metrics=(
                ("total_cost",lambda ep:ep["total_cost"]),
                ("H_cost",lambda ep:ep["controlled_cost"]),
                ("tail_cost",lambda ep:ep["tail_cost"]),
                ("working_fraction",lambda ep:ep["metrics"]["working_intervals"]/(64*len(ep["initial_state"]["components"]))),
                ("breakdown_fraction",lambda ep:ep["metrics"]["breakdown_intervals"]/(64*len(ep["initial_state"]["components"]))),
                ("service_fraction",lambda ep:ep["metrics"]["service_intervals"]/(64*len(ep["initial_state"]["components"]))),
                ("failure_count",lambda ep:ep["metrics"]["component_failures"]),
                ("PM_count",lambda ep:ep["metrics"]["pm_starts"]),
                ("CM_count",lambda ep:ep["metrics"]["cm_starts"]),
                ("failed_waiting",lambda ep:ep["metrics"]["failed_waiting_intervals"]),
                ("technician_utilization",lambda ep:ep["metrics"]["occupied_technician_intervals"]/
                    (64*len(ep["initial_state"]["jobs"]))),
                ("root_actions",lambda ep:ep["metrics"]["legal_actions_sum"]/64),
                ("unfinished_jobs",lambda ep:ep["metrics"]["unfinished_jobs"]),
                ("failed_at_cutoff",lambda ep:ep["metrics"]["failed_components_at_cutoff"]),
                ("forecast_transitions",lambda ep:ep["compute"]["simulated_transitions"]),
                ("episode_wall_seconds",lambda ep:ep["episode_wall_seconds"]),
            )
            for name,extract in metrics:
                values={seed:float(extract(ep)) for seed,ep in episodes.items()}
                row[name]=statistics.fmean(values.values())
                lo,hi=paired_interval(values,20261025+CASES.index(case)*100+len(arm)+len(name))
                row[name+"_lo"]=lo;row[name+"_hi"]=hi
            for cost in ("preventive","corrective","unavailable","failure"):
                values={seed:float(ep["metrics"]["costs"][cost]) for seed,ep in episodes.items()}
                row["cost_"+cost]=statistics.fmean(values.values())
            table.append(row)
    out.mkdir(parents=True,exist_ok=False)
    with (out/"atlas.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    (out/"atlas.json").write_text(json.dumps(
        dict(schema="C1-atlas-v1",source_run=str(path.resolve()),
             protocol_hash=manifest["protocol_hash"],
             analysis_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             numpy_version=np.__version__,rows=table),indent=2)+"\n")
    render_atlas(table,out)
    return table


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("select-baselines","characterize"):
        p=sub.add_parser(command)
        p.add_argument("run_dir",type=Path)
        p.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.command=="select-baselines":
        doc=select_baselines(args.run_dir,args.output)
        print(json.dumps(dict(status=doc["status"],cases=len(doc["selected"]),output=str(args.output))))
    else:
        table=characterize(args.run_dir,args.output)
        print(json.dumps(dict(status="created",rows=len(table),output=str(args.output))))


if __name__=="__main__":
    main()
