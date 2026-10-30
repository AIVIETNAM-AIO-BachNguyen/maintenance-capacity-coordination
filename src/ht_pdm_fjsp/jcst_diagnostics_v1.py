"""C1 structural witnesses and role-separated conditional forecasts.

No sampled physical future is passed to a controller. Witness construction uses
only the present state and legal catalogue, before any outcome evaluation.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Iterable

from .component_service_v1 import Action, Config, State, legal_actions, transition
from .component_controllers_v1 import RiskMatching, _edge_value, observe

MOTIFS = ("T","C","W","J","A","R","Q","P")
ROOT_LIMIT = 500


def role_seed(stage: str, case: str, physical_seed: int, reference: str,
              tick: int, bank: str, tape: int) -> int:
    if bank not in ("A","B","P","C") or type(physical_seed) is not int or not 0 <= physical_seed < 2**64:
        raise ValueError("forecast seed role")
    raw=f"C1|{stage}|{case}|{physical_seed}|{reference}|{tick}|{bank}|{tape}".encode()
    return (1<<64)+int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")


def priority(*parts) -> str:
    return hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()


def _single_edges(actions: Iterable[Action]) -> set[tuple[int,int,int]]:
    return {a[0] for a in actions if len(a)==1}


def motif_flags(config: Config, state: State, actions: tuple[Action,...] | None = None) -> dict[str,bool]:
    if actions is None:
        actions=legal_actions(config,state)
    if len(actions)>ROOT_LIMIT:
        raise ValueError("full root catalogue exceeds locked cap")
    singles=_single_edges(actions)
    free={j for j,job in enumerate(state.jobs) if job is None}
    busy={j for j,job in enumerate(state.jobs) if job is not None}
    flags={m:False for m in MOTIFS}
    flags["T"]=any(not state.components[m][c].failed for m,c,j in singles)
    flags["C"]=any((m,c2,j) in singles for m,c,j in singles
                   for c2 in range(c+1,len(config.components[m])))
    flags["W"]=any((m,c,j2) in singles for m,c,j in singles for j2 in free
                   if j2>j)
    flags["R"]=any((m,c,j2) in singles and config.duration[m][c][j]!=config.duration[m][c][j2]
                   for m,c,j in singles for j2 in free if j2>j)
    flags["P"]=any(not state.components[m][c].failed and config.duration[m][c][j]>1
                   for m,c,j in singles)
    flags["Q"]=any(config.duration[m][c][jb]>0 and config.duration[m][c][jb]<config.duration[m][c][jf]
                   for m,c,jf in singles for jb in busy)
    by_machine={}
    for m,c,j in singles: by_machine.setdefault(m,[]).append((m,c,j))
    together={(a[0][0],a[1][0]) for a in actions if len(a)==2}
    flags["J"]=any((m1,m2) not in together
                   for m1 in by_machine for m2 in by_machine if m1<m2)
    signatures={}
    for action in actions:
        if len(action)<2: continue
        machines=tuple(m for m,c,j in action)
        duration_by_machine=tuple((m,config.duration[m][c][j]) for m,c,j in action)
        signatures.setdefault(machines,set()).add(duration_by_machine)
    flags["A"]=any(len(group)>1 for group in signatures.values())
    return flags


def witness_pairs(config: Config, state: State, actions: tuple[Action,...] | None = None) -> dict[str,tuple]:
    """Return every eligible pair, deduplicated in canonical root order.

    Q pairs carry a waiting strategy target; all other roots use risk continuation.
    The full A family may be large; it is enumerated rather than outcome-pruned.
    """
    if actions is None:
        actions=legal_actions(config,state)
    if len(actions)>ROOT_LIMIT:
        raise ValueError("full root catalogue exceeds locked cap")
    aset=set(actions)
    pairs={kind:set() for kind in MOTIFS}
    for action in actions:
        for edge in action:
            m,c,j=edge
            rest=tuple(x for x in action if x!=edge)
            if not state.components[m][c].failed and rest in aset:
                pair=tuple(sorted((rest,action)))
                pairs["T"].add(pair)
                if config.duration[m][c][j]>1: pairs["P"].add(pair)
            for c2 in range(c+1,len(config.components[m])):
                replacement=tuple(sorted(rest+((m,c2,j),)))
                if replacement in aset:
                    pairs["C"].add(tuple(sorted((action,replacement))))
            for j2 in range(j+1,config.technicians):
                replacement=tuple(sorted(rest+((m,c,j2),)))
                if replacement in aset:
                    pair=tuple(sorted((action,replacement)))
                    pairs["W"].add(pair)
                    if config.duration[m][c][j]!=config.duration[m][c][j2]:
                        pairs["R"].add(pair)
    singles={a[0] for a in actions if len(a)==1}
    by_machine={}
    for edge in singles: by_machine.setdefault(edge[0],[]).append(edge)
    together={(a[0][0],a[1][0]) for a in actions if len(a)==2}
    for m1 in by_machine:
        for m2 in by_machine:
            if m1<m2 and (m1,m2) not in together:
                for a in by_machine[m1]:
                    for b in by_machine[m2]:
                        pairs["J"].add(tuple(sorted((((a,),((b,)))))))
    by_machine_set={}
    for action in actions:
        if len(action)>=2:
            by_machine_set.setdefault(tuple(m for m,c,j in action),[]).append(action)
    for machine_set,group in by_machine_set.items():
        signatures={action:tuple((m,config.duration[m][c][j]) for m,c,j in action)
                    for action in group}
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if signatures[a]!=signatures[b]:
                    pairs["A"].add((a,b))
    for m,c,jfree in singles:
        slow=config.duration[m][c][jfree]
        for jbusy,job in enumerate(state.jobs):
            if job is not None and 0<config.duration[m][c][jbusy]<slow:
                # Q key stores roots plus explicit desired fast worker.
                pairs["Q"].add(((),((m,c,jfree),),(m,c,jbusy)))
    return {kind:tuple(sorted(items)) for kind,items in pairs.items()}


def joint_quartets(config: Config, actions: tuple[Action,...]) -> tuple[tuple[Action,Action,Action,Action],...]:
    aset=set(actions)
    result=[]
    for full in actions:
        if len(full)!=2: continue
        a=(full[0],); b=(full[1],)
        if a in aset and b in aset:
            result.append(((),a,b,full))
    return tuple(result)


def _risk_with_constraints(config: Config, state: State, horizon: int, tail: int,
                           excluded_machine: int | None = None,
                           required_edge: tuple[int,int,int] | None = None) -> Action:
    obs=observe(config,state,horizon,tail)
    candidates=(a for a in obs.actions if
                (excluded_machine is None or all(e[0]!=excluded_machine for e in a))
                and (required_edge is None or required_edge in a))
    values={edge:_edge_value(obs,edge) for action in obs.actions for edge in action}
    return min(candidates,key=lambda a:(-sum(values[e] for e in a),a))


def forecast_branch(config: Config, state: State, root: Action, seed: int,
                    horizon: int = 48, tail: int = 16,
                    wait_for: tuple[int,int,int] | None = None) -> dict:
    if state.tick>=horizon or seed<2**64:
        raise ValueError("forecast state/seed")
    actions=legal_actions(config,state)
    if root not in actions:
        raise ValueError("forecast root not legal")
    current=state
    chosen=root
    held=wait_for is not None
    records=[]
    hcost=tailcost=0.
    while current.tick<horizon+tail:
        before=current
        outcome=transition(config,current,chosen,seed)
        cost=outcome.cost
        if before.tick<horizon: hcost+=cost
        else: tailcost+=cost
        records.append(dict(tick=before.tick,action=[list(x) for x in chosen],
                            cost=cost,after_hash=hashlib.sha256(
                                json.dumps(asdict(outcome.state),sort_keys=True).encode()).hexdigest()))
        current=outcome.state
        if current.tick>=horizon+tail: break
        if current.tick>=horizon:
            chosen=RiskMatching().choose(observe(config,current,horizon,tail)).action
        elif held:
            m,c,j=wait_for
            target=(m,c,j)
            if target in {e for action in legal_actions(config,current) for e in action}:
                chosen=_risk_with_constraints(config,current,horizon,tail,required_edge=target)
                held=False
            else:
                chosen=_risk_with_constraints(config,current,horizon,tail,excluded_machine=m)
        else:
            chosen=RiskMatching().choose(observe(config,current,horizon,tail)).action
    return dict(controlled_cost=hcost,tail_cost=tailcost,total_cost=hcost+tailcost,
                transitions=len(records),path=records,wait_strategy=wait_for)
