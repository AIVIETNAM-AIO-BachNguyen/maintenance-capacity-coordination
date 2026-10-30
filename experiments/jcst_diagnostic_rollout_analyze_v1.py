"""Paired physical R2 diagnostic frontier on the locked six-case, 64-seed panel."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

CASES=("base","homogeneous","scarce","restricted","generalist","scale4")
ARMS=("corrective","risk","rollout2")


def verified(run:Path,stage:str)->dict:
    meta=json.loads((run/"manifest.json").read_text())
    audit=json.loads((run/"independent_audit.json").read_text())
    if (meta["status"]!="complete" or meta["stage"]!=stage or
            meta["kind"]!="scientific" or audit["status"]!="PASS" or
            audit["protocol_hash"]!=meta["protocol_hash"]):
        raise ValueError("unaudited physical source")
    return meta


def episodes(run:Path,arms:tuple[str,...])->dict:
    out={}
    for line in (run/"index.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row["case"] not in CASES or row["arm"] not in arms or row["seed"]>=12_000_064:
            continue
        path=run/row["episode_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["episode_hash"]:
            raise ValueError("episode hash changed")
        key=(row["case"],row["arm"],row["seed"])
        if key in out:raise ValueError("duplicate episode")
        out[key]=json.loads(path.read_text())
    return out


def summarize(diagnostic:Path,rollout:Path,output:Path)->dict:
    dm=verified(diagnostic,"diagnostic")
    rm=verified(rollout,"diagnostic_rollout")
    if dm["protocol_hash"]!=rm["protocol_hash"] or tuple(rm["cases"])!=CASES:
        raise ValueError("protocol or case registry mismatch")
    base=episodes(diagnostic,("corrective","risk"))
    selected=episodes(rollout,("rollout2",))
    rows=[]
    for case in CASES:
        for arm in ("risk","corrective"):
            seeds=range(12_000_000,12_000_064)
            if any((case,arm,s) not in base or (case,"rollout2",s) not in selected for s in seeds):
                raise ValueError("incomplete paired diagnostic frontier")
            diff=np.asarray([base[(case,arm,s)]["total_cost"]-
                             selected[(case,"rollout2",s)]["total_cost"] for s in seeds],dtype=float)
            rng=np.random.default_rng(20261025+CASES.index(case)*19+len(arm))
            draws=diff[rng.integers(0,64,size=(2000,64))].mean(axis=1)
            compute=[selected[(case,"rollout2",s)]["compute"] for s in seeds]
            rows.append(dict(case=case,reference=arm,seeds=64,
                reference_minus_R2=float(diff.mean()),
                paired_lo=float(np.quantile(draws,.025)),
                paired_hi=float(np.quantile(draws,.975)),
                mean_R2_forecast_transitions=statistics.fmean(x["simulated_transitions"] for x in compute),
                mean_R2_selection_seconds=statistics.fmean(x["selection_seconds"] for x in compute),
                mean_R2_wall_seconds=statistics.fmean(
                    selected[(case,"rollout2",s)]["episode_wall_seconds"] for s in seeds)))
    output.mkdir(parents=True,exist_ok=False)
    with (output/"physical_R2_frontier.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    result=dict(schema="C1-physical-R2-frontier-v1",protocol_hash=dm["protocol_hash"],
        diagnostic_run=str(diagnostic.resolve()),rollout_run=str(rollout.resolve()),
        analysis_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
    (output/"physical_R2_frontier.json").write_text(json.dumps(result,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for arm,marker in (("risk","o"),("corrective","s")):
        subset=[next(r for r in rows if r["case"]==case and r["reference"]==arm) for case in CASES]
        ax.errorbar(range(len(CASES)),[r["reference_minus_R2"] for r in subset],
            yerr=[[r["reference_minus_R2"]-r["paired_lo"] for r in subset],
                  [r["paired_hi"]-r["reference_minus_R2"] for r in subset]],
            marker=marker,capsize=3,label=arm)
    ax.axhline(0,color="black",linewidth=.8)
    ax.set_xticks(range(len(CASES)),CASES,rotation=35,ha="right")
    ax.set_ylabel("Reference minus physical R2 H+tail cost (simulation units)")
    ax.set_title("Paired diagnostic physical R2 panel: 64 seeds per case")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"physical_R2_frontier.png",dpi=180)
    fig.savefig(output/"physical_R2_frontier.pdf")
    plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--diagnostic",type=Path,required=True)
    parser.add_argument("--rollout",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.diagnostic,args.rollout,args.output)
    print(json.dumps(dict(status="created",rows=len(result["rows"]),output=str(args.output))))


if __name__=="__main__":
    main()
