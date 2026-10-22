import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location("threshold_control_v1", Path(__file__).parents[1] / "experiments/threshold_control_v1.py")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
choose_g = _module.choose_g
from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_request_policies import request_action
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.pilot import load_policy
from maintenance_capacity_coordination.pilot_policy import PilotPolicy


def test_fixed_threshold_coordinator_has_exact_masked_logprob_and_no_future_tape(tmp_path):
    torch.manual_seed(73100021)
    env = CoordinationEnv(configs()['specialist_generalist_scale2'])
    env.reset(73100022)
    policy = PilotPolicy('D')
    first = choose_g(policy, env, deterministic=True)
    assert first['submissions'] == request_action(env.config, env.inner.state, env.tick, 'threshold')
    assert bool(first['matching_mask'][first['matching_index']])
    logits = policy.coordinator(torch.from_numpy(first['coordinator_obs']))
    logits = logits.masked_fill(~torch.from_numpy(first['matching_mask']), -1e9)
    expected = Categorical(logits=logits).log_prob(torch.tensor(first['matching_index']))
    assert first['old_logprob'] == pytest.approx(float(expected.detach()), abs=1e-6)
    env.inner._events = tuple(tuple(not x for x in row) for row in env.inner._events)
    second = choose_g(policy, env, deterministic=True)
    np.testing.assert_array_equal(first['pre_obs'], second['pre_obs'])
    assert first['matching_index'] == second['matching_index']
    torch.save({'arm': 'D', 'state_dict': policy.state_dict()}, tmp_path/'checkpoint.pt')
    loaded = load_policy(tmp_path/'checkpoint.pt')
    assert choose_g(loaded, env, deterministic=True)['matching_index'] == first['matching_index']
    reward, done = env.step(first['submissions'], first['matching_index'])
    assert np.isfinite(reward) and not done
