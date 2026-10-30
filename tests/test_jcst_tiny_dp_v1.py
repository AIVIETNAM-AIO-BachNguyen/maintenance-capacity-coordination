"""Tiny exact-horizon control is scoped and consistent with JCST transitions."""
from functools import lru_cache
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from jcst_tiny_dp_v1 import branch_expectation, exact_dp, tiny_config
from ht_pdm_fjsp.component_service_v1 import initial_state, legal_actions


def test_exact_tiny_dp_dominates_forced_wait_and_is_legal():
    config=tiny_config(scale=4,shape=2,duration=2)
    initial=initial_state(config)
    @lru_cache(None)
    def wait_value(state):
        if state.tick==6:return 0.
        return sum(weight*(outcome.cost+wait_value(outcome.state))
                   for weight,outcome in branch_expectation(config,state,()))
    result=exact_dp(config,6)
    assert result["first_action"] in legal_actions(config,initial)
    assert result["optimal_expected_cost"]<=wait_value(initial)+1e-9
    assert result["dynamic_states"]>6


def test_tiny_horizon_one_prefers_wait_for_low_failure_risk():
    config=tiny_config(scale=100,shape=1,duration=2)
    result=exact_dp(config,1)
    assert result["first_action"]==()
    assert result["optimal_expected_cost"]>0
