"""Independent JCST/1.1 trace audit. Does not call the collector's transition."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def catalogue(cfg: dict, state: dict) -> set[tuple]:
    busy_m = {j["machine"] for j in state["jobs"] if j is not None}
    free = {j for j, job in enumerate(state["jobs"]) if job is None}
    edges = [(m,c,j) for m, comps in enumerate(cfg["components"]) if m not in busy_m
             for c in range(len(comps)) for j in free if cfg["duration"][m][c][j] > 0]
    actions = {()}
    for count in range(1, min(len(cfg["components"]), len(free)) + 1):
        for chosen in itertools.combinations(edges, count):
            if len({x[0] for x in chosen}) == count and len({x[2] for x in chosen}) == count:
                actions.add(tuple(sorted(chosen)))
    return actions


def u01(seed: int, tick: int, m: int, c: int) -> float:
    raw = hashlib.sha256(f"JCST/1|{seed}|{tick}|{m}|{c}".encode()).digest()[:8]
    return (int.from_bytes(raw, "big") + .5) / 2**64


def independent_step(cfg: dict, before: dict, action: tuple, seed: int):
    state = json.loads(json.dumps(before))
    t = state["tick"]
    jobs = state["jobs"]
    components = state["components"]
    starts = []
    completions = []
    failures = []
    occupancy = []
    pm = cm = 0
    for m,c,j in action:
        kind = "CM" if components[m][c]["failed"] else "PM"
        jobs[j] = dict(machine=m, component=c, technician=j, kind=kind,
                       remaining=cfg["duration"][m][c][j])
        starts.append((t,m,c,j,kind))
        pm += kind == "PM"
        cm += kind == "CM"
    busy_m = {job["machine"] for job in jobs if job is not None}
    producing = [m for m,row in enumerate(components)
                 if m not in busy_m and not any(item["failed"] for item in row)]
    unavailable = len(components) - len(producing)
    for m in producing:
        for c,item in enumerate(components[m]):
            age = item["age"]
            spec = cfg["components"][m][c]
            alpha, beta = spec["scale"], spec["shape"]
            q = 1 - math.exp(-(((age+1)/alpha)**beta - (age/alpha)**beta))
            item["age"] += 1
            if u01(seed,t,m,c) < q:
                item["failed"] = True
                failures.append((t+1,m,c))
    for j,job in enumerate(jobs):
        if job is None:
            continue
        m,c = job["machine"],job["component"]
        occupancy.append((t,m,c,j,job["remaining"]))
        job["remaining"] -= 1
        if job["remaining"] == 0:
            components[m][c] = dict(age=0, failed=False)
            completions.append((t+1,m,c,j,job["kind"]))
            jobs[j] = None
    state["tick"] += 1
    breakdown = dict(preventive=pm*cfg["preventive_cost"],
                     corrective=cm*cfg["corrective_cost"],
                     unavailable=unavailable*cfg["unavailable_cost"],
                     failure=len(failures)*cfg["failure_cost"])
    events = (starts, completions, failures, occupancy)
    return state, breakdown, events


def audit(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    episode = json.loads((run_dir / "episode.json").read_text())
    if episode["terminal_status"] != "complete" or episode["adapter_version"] != "JCST/1.1-H-tail-risk-v1":
        raise ValueError("incomplete or wrong adapter")
    records = episode["records"]
    H, B = episode["horizon"], episode["tail"]
    if len(records) != H+B:
        raise ValueError("wrong trace length")
    sums = {"H":0., "tail":0.}
    previous = None
    for n,record in enumerate(records):
        phase = "H" if n < H else "tail"
        if record["tick"] != n or record["phase"] != phase:
            raise ValueError(f"tick/phase at {n}")
        if previous is not None and record["before"] != previous:
            raise ValueError(f"broken state chain at {n}")
        action = tuple(sorted(tuple(x) for x in record["action"]))
        full_catalogue = catalogue(cfg,record["before"])
        recorded_catalogue = {tuple(tuple(x) for x in a) for a in record["catalogue"]}
        if recorded_catalogue != full_catalogue or len(record["catalogue"]) != len(full_catalogue):
            raise ValueError(f"catalogue/mask mismatch at {n}")
        if tuple(tuple(x) for x in record["action"]) != action or action not in full_catalogue:
            raise ValueError(f"illegal or noncanonical action at {n}")
        after, costs, event_groups = independent_step(cfg,record["before"],action,episode["seed"])
        if after != record["after"] or costs != record["breakdown"] or sum(costs.values()) != record["cost"]:
            raise ValueError(f"state/cost mismatch at {n}")
        starts, completions, failures, occupancy = event_groups
        observed = record["events"]
        signature = lambda e: (e["tick"],e["machine"],e["component"])
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["kind"])
                  for e in observed if e["type"] == "start") != sorted(starts):
            raise ValueError(f"starts mismatch at {n}")
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["kind"])
                  for e in observed if e["type"] == "completion") != sorted(completions):
            raise ValueError(f"completions mismatch at {n}")
        if sorted(signature(e) for e in observed if e["type"] == "failure") != sorted(failures):
            raise ValueError(f"failures mismatch at {n}")
        if sorted((e["tick"],e["machine"],e["component"],e["technician"],e["remaining"])
                  for e in observed if e["type"] == "occupancy") != sorted(occupancy):
            raise ValueError(f"occupancy mismatch at {n}")
        sums[phase] += record["cost"]
        previous = after
    if (sums["H"] != episode["controlled_cost"] or sums["tail"] != episode["tail_cost"]
            or sum(sums.values()) != episode["total_cost"] or previous != episode["final_state"]):
        raise ValueError("H/tail/terminal mismatch")
    return dict(status="PASS", records=len(records), controlled_cost=sums["H"],
                tail_cost=sums["tail"], total_cost=sum(sums.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    (args.run_dir / "replay_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
