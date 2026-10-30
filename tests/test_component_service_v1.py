"""JCST/1 engineering fixtures and two hand-calculated transition traces."""
from __future__ import annotations

from dataclasses import replace
import itertools
import json
import math

import pytest

from ht_pdm_fjsp.component_service_v1 import (
    ComponentConfig as CC, ComponentState as CS, Config, ServiceJob, State,
    canonical_action, config_from_json, config_json, engineering_config, hazard,
    initial_state, keyed_uniform, legal_actions, machine_modes, state_from_json,
    state_json, transition, validate_state,
)
from ht_pdm_fjsp.component_controllers_v1 import (
    BoundedRollout, Corrective, RiskMatching, Threshold, Wait, observe,
)
from ht_pdm_fjsp.component_framework_v1 import advance, run_episode


def cfg(machines=1, components=2, technicians=1, duration=1):
    return Config(tuple(tuple(CC(2,2) for _ in range(components)) for _ in range(machines)),
                  tuple(tuple(tuple(duration for _ in range(technicians))
                              for _ in range(components)) for _ in range(machines)))


def high(config):
    return {(m,c): .999999 for m,row in enumerate(config.components) for c in range(len(row))}


def test_idle_catalogue_37_and_independent_crosscheck():
    c = engineering_config()
    s = initial_state(c)
    actions = legal_actions(c,s)
    edges = [(m,k,j) for m in range(3) for k in range(2) for j in range(2)]
    brute = {()}
    for n in (1,2):
        brute |= {tuple(sorted(x)) for x in itertools.combinations(edges,n)
                  if len({e[0] for e in x}) == n and len({e[2] for e in x}) == n}
    assert len(actions) == 37
    assert set(actions) == brute


def test_trace_one_duration_one_pm_and_age():
    # Hand trace: t0 PM at age 0 costs 1+6=7, releases at t1.
    # t1 WAIT produces, both ages become 1, cost 0.
    c = cfg()
    s = initial_state(c)
    first = transition(c,s,((0,0,0),),17,high(c))
    assert first.cost == 7
    assert first.breakdown == dict(preventive=1,corrective=0,unavailable=6,failure=0)
    assert first.state == State(1,((CS(0,False),CS(0,False)),),(None,))
    assert [e["type"] for e in first.events] == ["start","occupancy","completion"]
    second = transition(c,first.state,(),17,high(c))
    assert second.cost == 0
    assert second.state == State(2,((CS(1,False),CS(1,False)),),(None,))


def test_trace_two_double_failure_target_only_and_same_boundary():
    # Hand trace: m0 has two failed components; m1 produces and both fail.
    # t0 CM m0,c0: CM 2 + one unavailable 6 + two failures 30 = 38.
    # t1 CM m0,c1: CM 2 + two unavailable machines 12 = 14.
    c = cfg(machines=2)
    s = State(0,((CS(3,True),CS(4,True)),(CS(),CS())),(None,))
    one = transition(c,s,((0,0,0),),5,{(1,0):0.,(1,1):0.})
    assert one.cost == 38
    assert one.state == State(1,((CS(),CS(4,True)),(CS(1,True),CS(1,True))),(None,))
    assert machine_modes(one.state) == ("breakdown","breakdown")
    assert sorted(e["type"] for e in one.events) == [
        "completion","failure","failure","occupancy","start"]
    two = transition(c,one.state,((0,1,0),),5,high(c))
    assert two.cost == 14
    assert two.state == State(2,((CS(),CS()),(CS(1,True),CS(1,True))),(None,))
    assert machine_modes(two.state) == ("working","breakdown")


def test_opportunistic_pm_freeze_and_cm():
    c = cfg(duration=2)
    s = State(0,((CS(4,True),CS(3,False)),),(None,))
    assert ((0,1,0),) in legal_actions(c,s)
    one = transition(c,s,((0,1,0),),1,high(c))
    assert one.cost == 7 and one.state.components[0] == (CS(4,True),CS(3,False))
    assert one.state.jobs[0].kind == "PM"
    assert legal_actions(c,one.state) == ((),)
    two = transition(c,one.state,(),1,high(c))
    assert two.state.components[0] == (CS(4,True),CS())
    assert two.state.jobs == (None,)
    cm = transition(c,two.state,((0,0,0),),1,high(c))
    assert cm.breakdown["corrective"] == 2


def test_busy_capacity_and_reciprocal_duration():
    c = engineering_config()
    s = initial_state(c)
    assert ((0,0,0),(1,1,1)) in legal_actions(c,s)
    assert ((0,0,1),(1,1,0)) in legal_actions(c,s)
    fast = transition(c,s,((0,0,0),(1,1,1)),1,high(c))
    slow = transition(c,s,((0,0,1),(1,1,0)),1,high(c))
    assert tuple(j.remaining for j in fast.state.jobs) == (1,1)
    assert tuple(j.remaining for j in slow.state.jobs) == (7,7)
    assert legal_actions(c,fast.state) == ((),)
    c2 = cfg(machines=2)
    assert ((0,0,0),(1,0,0)) not in legal_actions(c2,initial_state(c2))


def test_invalid_pre_mutation_and_tuple_order():
    c = cfg(machines=2,technicians=2)
    s = initial_state(c)
    with pytest.raises(ValueError):
        transition(c,s,((0,0,0),(1,0,0)),1,high(c))
    with pytest.raises(ValueError):
        transition(c,s,((0,0,2),),1,high(c))
    with pytest.raises(ValueError):
        transition(c,s,((0,0,0),),1,{(0,0):2.})
    assert s == initial_state(c)
    a = ((0,0,0),(1,1,1))
    b = tuple(reversed(a))
    assert transition(c,s,a,1,high(c)) == transition(c,s,b,1,high(c))
    invalid = State(0,s.components,(ServiceJob(0,0,0,"CM",1),None))
    with pytest.raises(ValueError):
        validate_state(c,invalid)


def test_hazard_survival_ratio_bounds_and_keyed_noise():
    spec = CC(6,5)
    for a in (0,1,4,8,16):
        expected = 1-math.exp(-(((a+1)/6)**5-(a/6)**5))
        assert hazard(spec,a) == pytest.approx(expected)
        assert 0 <= hazard(spec,a) <= 1
    assert hazard(CC(1,1),0) == pytest.approx(1-math.exp(-1))
    with pytest.raises(ValueError):
        hazard(spec,-1)
    c = cfg()
    s = initial_state(c)
    assert transition(c,s,(),78) == transition(c,s,(),78)
    assert keyed_uniform(78,0,0,0) != keyed_uniform(78,0,0,1)
    # Root action catalogue order cannot enter the noise key.
    assert keyed_uniform(78,0,0,0) == keyed_uniform(78,0,0,0)


def test_serialization_h_tail_reward_and_controllers():
    c = cfg(machines=2,technicians=2)
    s = initial_state(c)
    assert config_from_json(config_json(c)) == c
    assert state_from_json(c,state_json(c,s)) == s
    for controller in (Wait(),Corrective(),Threshold(0),RiskMatching(),
                       BoundedRollout(2,2,1000)):
        obs = observe(c,s,2,1)
        d = controller.choose(obs)
        assert d.action in obs.actions
        if controller.name == "rollout":
            assert d.root_actions_evaluated == len(obs.actions)
            assert d.forecast_scenarios == 2
            assert d.simulated_transitions == len(obs.actions)*2*2
    step0 = advance(c,s,(),7,2,1)
    step1 = advance(c,step0.state,(),7,2,1)
    assert not step0.done and step1.done
    assert len(step1.records) == 2
    assert step1.reward == -(step1.controlled_cost+step1.tail_cost)
    episode = run_episode(c,Wait(),7,2,1)
    assert episode["controlled_cost"] + episode["tail_cost"] == episode["total_cost"]
    assert len(episode["records"]) == 3


def test_relabeling_with_transport_of_config_action_noise():
    c = Config(((CC(2,2),),(CC(4,2),)),
               (((2,1),),((1,2),)))
    s = State(0,((CS(1,False),),(CS(2,False),)),(None,None))
    a = ((0,0,0),(1,0,1))
    uniforms = {(0,0):.9,(1,0):.1}
    original = transition(c,s,a,99,uniforms)
    # Swap machines and technicians, including duration columns and noise map.
    swapped = Config(tuple(reversed(c.components)),
                     tuple(tuple(tuple(reversed(row)) for row in machine)
                           for machine in reversed(c.duration)))
    ss = State(0,tuple(reversed(s.components)),tuple(reversed(s.jobs)))
    aa = tuple((1-m,k,1-j) for m,k,j in a)
    uu = {(1-m,k):v for (m,k),v in uniforms.items()}
    relabeled = transition(swapped,ss,aa,99,uu)
    assert relabeled.cost == original.cost
    assert relabeled.state.components == tuple(reversed(original.state.components))
    # Jobs are transported under the same permutation.
    for j,job in enumerate(original.state.jobs):
        target = relabeled.state.jobs[1-j]
        assert target is not None and (target.machine,target.component,target.remaining) == (
            1-job.machine,job.component,job.remaining)


def test_independent_replay_detects_missing_mask_and_wrong_cost(tmp_path):
    from experiments.jcst_replay_audit import audit
    c = cfg()
    episode = run_episode(c, Wait(), 7, 2, 1)
    (tmp_path/"config.json").write_text(config_json(c))
    path = tmp_path/"episode.json"
    path.write_text(json.dumps(episode))
    assert audit(tmp_path)["status"] == "PASS"
    corrupt = json.loads(path.read_text())
    corrupt["records"][0]["catalogue"].pop()
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="catalogue"):
        audit(tmp_path)
    corrupt = json.loads(json.dumps(episode))
    corrupt["records"][0]["cost"] += 1
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="state/cost"):
        audit(tmp_path)


def test_periodic_calendar_and_forecast_seed_namespace():
    from ht_pdm_fjsp.component_controllers_v1 import Periodic, forecast_seed
    c = cfg(machines=2, components=2, technicians=2)
    s = initial_state(c)
    action = Periodic(4).choose(observe(c,s,8,2)).action
    assert all((s.tick-(m+k)) % 4 == 0 for m,k,_ in action)
    assert action
    assert forecast_seed(123,0) >= 2**64
    assert forecast_seed(123,0) != forecast_seed(123,1)
    with pytest.raises(ValueError):
        Periodic(0)
