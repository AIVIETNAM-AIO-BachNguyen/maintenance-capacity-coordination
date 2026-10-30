"""Scaled adapter retains the inherited physical and information contracts."""
from dataclasses import replace

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import family_config, role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL, configs, vector_observation
from ht_pdm_fjsp.maintenance_request_policies import request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import RequestEnv, legal_matchings, submit_requests
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, requester_features
from maintenance_capacity_coordination.scaled_coordination_v11 import (
    ScaledCoordinationEnv, actor_inputs_scaled, global_dim, local_dim,
    vector_observation_scaled, machine_features_scaled,
)
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs


def scaled_config(seed=73740000):
    return replace(family_config("n5_k3_sparse", seed, "full"), horizon=HORIZON + TAIL)


def test_scaled_observations_exactly_preserve_n4k2_coordinates():
    config = configs()["reciprocal_scale2"]
    env = ScaledCoordinationEnv(config)
    env.reset(73700001)
    for _ in range(12):
        submissions = env.fixed_submissions("threshold")
        admitted = env.admitted(submissions)
        assert np.array_equal(vector_observation_scaled(config, admitted, env.tick),
                              vector_observation(config, admitted, env.tick))
        assert np.array_equal(machine_features_scaled(config, admitted, env.tick),
                              requester_features(config, admitted, env.tick, "C"))
        old_machine, old_worker = actor_inputs(config, admitted, env.tick)
        new_machine, new_worker = actor_inputs_scaled(config, admitted, env.tick)
        assert np.array_equal(new_machine, old_machine)
        assert np.array_equal(new_worker, old_worker)
        matching = risk_matching(config, admitted, env.tick)
        env.step(submissions, env.actions.index(matching))


def test_n5k3_full_legal_action_coverage_and_exact_kernel_h_tail():
    config = scaled_config()
    assert (config.machines, config.technicians) == (5, 3)
    assert global_dim(5, 3) == 81 and local_dim(3) == 24
    env = ScaledCoordinationEnv(config)
    seed = 73700002
    env.reset(seed)
    kernel = RequestEnv(config)
    kernel.reset(role_seed(seed, BASE_VERSION + ":physical"))
    assert len(env.actions) == 136
    for tick in range(HORIZON):
        assert env.tick == kernel.time == tick
        assert env.inner.state == kernel.state
        submissions = request_action(config, kernel.state, tick, "threshold")
        admitted = submit_requests(config, kernel.state, tick, submissions)
        legal = legal_matchings(config, admitted)
        mask = env.matching_mask(submissions)
        assert {a for a, allowed in zip(env.actions, mask) if allowed} == set(legal)
        matching = risk_matching(config, admitted, tick)
        index = env.actions.index(matching)
        reward, done = env.step(submissions, index)
        _, kernel_reward, _, _ = kernel.step(submissions, matching)
        if tick < HORIZON - 1:
            assert not done and reward == kernel_reward
    for _ in range(TAIL):
        tick = kernel.time
        submissions = request_action(config, kernel.state, tick, "threshold")
        admitted = submit_requests(config, kernel.state, tick, submissions)
        matching = risk_matching(config, admitted, tick)
        kernel.step(submissions, matching)
    assert done and env.inner.state == kernel.state
    assert env.metrics()["objective"] == kernel.metrics["objective"]
    assert abs(env.metrics()["episode_return"] + env.metrics()["objective"]) < 1e-8
    assert abs(env.metrics()["H_cost"] + env.metrics()["tail_cost"] - env.metrics()["objective"]) < 1e-8


def test_scaled_observations_ignore_future_failure_tape():
    env = ScaledCoordinationEnv(scaled_config(73740001))
    env.reset(73700003)
    before = (env.pre_observation().copy(), env.inner.state, env.tick)
    events = env.inner._events
    env.inner._events = tuple(reversed(events))
    after = env.pre_observation()
    env.inner._events = events
    assert np.array_equal(before[0], after)
    assert env.inner.state == before[1] and env.tick == before[2]
