"""Exact finite-horizon DP only for a tiny 1M/1C/1K JCST engineering case."""
from __future__ import annotations

from functools import lru_cache
import json

from ht_pdm_fjsp.component_service_v1 import (
    ComponentConfig, Config, State, hazard, initial_state, legal_actions, transition,
)


def tiny_config(scale:float=4,shape:float=2,duration:int=2)->Config:
    config=Config(((ComponentConfig(scale,shape),),),(((duration,),),))
    config.validate()
    return config


def branch_expectation(config:Config,state:State,action):
    if config.machines!=1 or len(config.components[0])!=1 or config.technicians!=1:
        raise ValueError("tiny DP supports only 1M/1C/1K")
    busy=state.jobs[0] is not None
    failed=state.components[0][0].failed
    produces=not busy and not failed and not action
    p=hazard(config.components[0][0],state.components[0][0].age) if produces else 0.
    no_failure=transition(config,state,action,0,{(0,0):1-2**-53})
    if not produces:
        return ((1.,no_failure),)
    failure=transition(config,state,action,0,{(0,0):0.})
    return ((1-p,no_failure),(p,failure))


def exact_dp(config:Config,horizon:int,state:State|None=None):
    """Expected simulation-unit cost to horizon, with no tail."""
    if horizon<1:raise ValueError("positive horizon")
    if state is None:state=initial_state(config)
    if state.tick>horizon:raise ValueError("state beyond horizon")

    @lru_cache(None)
    def value(s:State):
        if s.tick==horizon:return 0.,()
        best=None
        for action in legal_actions(config,s):
            total=sum(weight*(outcome.cost+value(outcome.state)[0])
                      for weight,outcome in branch_expectation(config,s,action))
            candidate=(total,action)
            if best is None or candidate<best:best=candidate
        return best

    optimum,first=value(state)
    return dict(optimal_expected_cost=optimum,first_action=first,
                dynamic_states=value.cache_info().currsize,horizon=horizon)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--horizon",type=int,default=8)
    parser.add_argument("--scale",type=float,default=4.)
    parser.add_argument("--shape",type=float,default=2.)
    parser.add_argument("--duration",type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(exact_dp(tiny_config(args.scale,args.shape,args.duration),args.horizon)))


if __name__=="__main__":
    main()
