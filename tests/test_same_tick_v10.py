"""High-demand configuration and unchanged v8 decoder integration checks."""
from dataclasses import replace

import torch

from experiments.same_tick_v10_config import CASES, BASE_CASES, configs
from experiments.same_tick_pilot_v10 import ARMS, choose, new_policy, seed_registry_check
from ht_pdm_fjsp.maintenance_request_policies import family_configs
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v8 import enumerate_sequences, initial_edge_mask, matching_from_sequence


def test_v10_cases_only_change_declared_failure_age():
    base = family_configs("full")
    for name in CASES:
        assert configs()[name] == replace(base[BASE_CASES[name]], failure_age=3)
    seed_registry_check()


def test_v10_decoder_covers_kernel_actions_and_keeps_one_physical_tick():
    for case in CASES:
        env = CoordinationEnv(configs()[case])
        env.reset(73600001)
        policy = new_policy("C10")
        for _ in range(48):
            submissions = env.fixed_submissions("threshold")
            admitted = env.admitted(submissions)
            mask = initial_edge_mask(env.config, admitted)
            assert {matching_from_sequence(s) for s in enumerate_sequences(mask)} == set(legal_matchings(env.config, admitted))
            state, tick = env.inner.state, env.tick
            decision = choose(policy, env, True)
            assert env.inner.state == state and env.tick == tick
            assert env.actions[decision["matching_index"]] == decision["matching"]
            env.step(decision["submissions"], decision["matching_index"])
            assert env.tick == tick + 1
        assert abs(env.metrics()["episode_return"] + env.metrics()["objective"]) < 1e-8


def test_v10_policy_mapping_and_context_ablation_identity():
    assert ARMS == ("C10", "D10", "B10", "S10", "G10")
    c, d = new_policy("C10"), new_policy("D10")
    d.load_state_dict(c.state_dict())
    assert c.actor_parameters == d.actor_parameters
    machine, worker = torch.randn(1, 4, 21), torch.randn(1, 2, 4)
    mask = torch.ones(1, 8, dtype=torch.bool)
    assert torch.equal(c.sample(machine[0], worker[0], mask[0], deterministic=True)[0][:1],
                       d.sample(machine[0], worker[0], mask[0], deterministic=True)[0][:1])
