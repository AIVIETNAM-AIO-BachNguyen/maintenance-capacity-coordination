"""Outcome-blind C1 witness and forecast invariants."""
from ht_pdm_fjsp.component_service_v1 import (
    ComponentConfig, ComponentState, Config, ServiceJob, State,
    engineering_config, initial_state, legal_actions,
)
from ht_pdm_fjsp.jcst_diagnostics_v1 import (
    forecast_branch, joint_quartets, motif_flags, role_seed, witness_pairs,
)


def test_full_root_witnesses_and_negative_controls():
    c=engineering_config(); s=initial_state(c)
    flags=motif_flags(c,s)
    assert flags["T"] and flags["C"] and flags["W"] and flags["A"] and flags["R"] and flags["P"]
    assert not flags["J"] and not flags["Q"]
    pairs=witness_pairs(c,s)
    assert all(pairs[k] for k in ("T","C","W","A","R","P"))
    assert not pairs["J"] and not pairs["Q"]
    assert joint_quartets(c,legal_actions(c,s))
    homogeneous=Config(tuple((ComponentConfig(6,5),ComponentConfig(6,5)) for _ in range(2)),
                       tuple(((2,2),(2,2)) for _ in range(2)))
    assert not motif_flags(homogeneous,initial_state(homogeneous))["R"]
    assert not motif_flags(homogeneous,initial_state(homogeneous))["A"]


def test_busy_fast_worker_defines_wait_strategy_and_disjoint_forecast_banks():
    c=Config(((ComponentConfig(6,5),),(ComponentConfig(6,5),)),
             (((2,8),),((2,8),)))
    s=State(0,((ComponentState(),),(ComponentState(),)),
            (ServiceJob(1,0,0,"PM",1),None))
    flags=motif_flags(c,s)
    assert flags["Q"]
    pair=witness_pairs(c,s)["Q"][0]
    assert pair[0]==() and pair[2]==(0,0,0)
    seeds=[role_seed("diagnostic","base",12_000_000,"risk",0,bank,0)
           for bank in ("A","B","P","C")]
    assert len(set(seeds))==4 and min(seeds)>=2**64
    result=forecast_branch(c,s,(),seeds[0],horizon=4,tail=1,wait_for=(0,0,0))
    assert result["path"][0]["action"]==[]
    assert [0,0,0] in result["path"][1]["action"]
    assert result["transitions"]==5
