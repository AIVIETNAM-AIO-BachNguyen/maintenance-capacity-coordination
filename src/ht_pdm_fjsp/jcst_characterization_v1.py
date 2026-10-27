"""Locked C1 configuration registry and physical episode summaries."""
from __future__ import annotations

from dataclasses import asdict
from .component_service_v1 import ComponentConfig, ComponentState, Config, State, initial_state

SENSITIVITY_CASES = ("base","shortlife","homogeneous","restricted","generalist","scale4")
ROLLOUT_DIAGNOSTIC_CASES = ("base","homogeneous","scarce","restricted","generalist","scale4")
SENSITIVITY_H_TAIL = ((24,8),(48,0),(96,32))

CASE_NAMES = (
    "base", "longlife", "shortlife", "memoryless", "homogeneous",
    "surplus", "scarce", "restricted", "generalist", "pm_expensive",
    "downtime_high", "staggered", "scale4",
)
SEED_RANGES = {
    "characterization": (10_000_000, 128),
    "selection": (11_000_000, 64),
    "diagnostic": (12_000_000, 128),
    "confirmation": (13_000_000, 128),
    "sensitivity": (14_000_000, 64),
}


def case_config(name: str) -> tuple[Config, State]:
    if name not in CASE_NAMES:
        raise ValueError("unknown locked case")
    m = 4 if name == "scale4" else 3
    k = 1 if name == "scarce" else 3 if name in ("surplus","generalist","scale4") else 2
    if name == "longlife":
        scales = (12,16)
    elif name == "shortlife":
        scales = (4,5)
    elif name == "homogeneous":
        scales = (6,6)
    else:
        scales = (6,8)
    shape = 1 if name == "memoryless" else 5
    components = tuple((ComponentConfig(scales[0],shape),
                        ComponentConfig(scales[1],shape)) for _ in range(m))
    if name == "scarce":
        rows = ((2,),(8,))
    elif name == "restricted":
        rows = ((2,0),(0,2))
    elif name in ("homogeneous","surplus"):
        rows = (tuple(2 for _ in range(k)),tuple(2 for _ in range(k)))
    elif k == 3:
        rows = ((2,8,5),(8,2,5))
    else:
        rows = ((2,8),(8,2))
    duration = tuple(rows for _ in range(m))
    config = Config(components,duration,
                    preventive_cost=8 if name == "pm_expensive" else 1,
                    unavailable_cost=12 if name == "downtime_high" else 6)
    config.validate()
    if name == "staggered":
        state = State(0,tuple(tuple(ComponentState((machine+2*c)%5,False)
                                  for c in range(2)) for machine in range(m)),
                      tuple(None for _ in range(k)))
    else:
        state = initial_state(config)
    return config,state


def physical_seed(stage: str, index: int) -> int:
    first,count = SEED_RANGES[stage]
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("physical seed index")
    return first+index


def episode_metrics(config: Config, episode: dict) -> dict:
    out = dict(
        working_intervals=0, breakdown_intervals=0, service_intervals=0,
        component_failures=0, pm_starts=0, cm_starts=0, completions=0,
        occupied_technician_intervals=0, idle_technician_intervals=0,
        failed_waiting_intervals=0, multi_failed_machine_intervals=0,
        legal_actions_sum=0, legal_actions_max=0,
        duration_histogram={}, release_histogram={},
        H_cost=0.0, tail_cost=0.0,
        costs=dict(preventive=0.,corrective=0.,unavailable=0.,failure=0.),
    )
    for record in episode["records"]:
        before = record["before"]
        starts = {e["machine"] for e in record["events"] if e["type"] == "start"}
        active = {job["machine"] for job in before["jobs"] if job is not None} | starts
        for m,row in enumerate(before["components"]):
            failed = sum(item["failed"] for item in row)
            if m in active:
                out["service_intervals"] += 1
            elif failed:
                out["breakdown_intervals"] += 1
            else:
                out["working_intervals"] += 1
            out["failed_waiting_intervals"] += int(failed > 0 and m not in active)
            out["multi_failed_machine_intervals"] += int(failed > 1)
        occupied = sum(e["type"] == "occupancy" for e in record["events"])
        out["occupied_technician_intervals"] += occupied
        out["idle_technician_intervals"] += config.technicians-occupied
        for e in record["events"]:
            kind=e["type"]
            if kind == "failure": out["component_failures"] += 1
            elif kind == "completion":
                out["completions"] += 1
                key=str(e["tick"])
                out["release_histogram"][key]=out["release_histogram"].get(key,0)+1
            elif kind == "start":
                out["pm_starts" if e["kind"]=="PM" else "cm_starts"] += 1
                key=str(e["duration"])
                out["duration_histogram"][key]=out["duration_histogram"].get(key,0)+1
        n=len(record["catalogue"])
        out["legal_actions_sum"] += n
        out["legal_actions_max"] = max(n,out["legal_actions_max"])
        out["H_cost" if record["phase"]=="H" else "tail_cost"] += record["cost"]
        for key,value in record["breakdown"].items():
            out["costs"][key] += value
    final = episode["final_state"]
    out["unfinished_jobs"] = sum(job is not None for job in final["jobs"])
    out["failed_components_at_cutoff"] = sum(item["failed"] for row in final["components"] for item in row)
    out["ages_at_cutoff"] = [[item["age"] for item in row] for row in final["components"]]
    out["episode_total_cost"] = episode["total_cost"]
    out["compute"] = episode["compute"]
    out["episode_wall_seconds"] = episode["episode_wall_seconds"]
    return out
