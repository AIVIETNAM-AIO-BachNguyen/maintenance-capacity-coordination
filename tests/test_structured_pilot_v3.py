"""Mechanism and contract tests for the structured centralized v3 pilot."""
from __future__ import annotations

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.structured_policy_v3 import MatchingPolicy, actor_inputs
from maintenance_capacity_coordination.structured_pilot_v3 import (
    choose, load_policy, seed_registry_check,
)


def test_shared_machine_encoder_and_matched_information():
    env = CoordinationEnv(configs()["reciprocal_scale2"])
    env.reset(73100051)
    admitted = env.admitted(env.fixed_submissions("threshold"))
    machine, worker = actor_inputs(env.config, admitted, env.tick)
    assert machine.shape == (4, 21) and worker.shape == (2, 4)
    structured = MatchingPolicy("S")
    direct = MatchingPolicy("G3")
    assert abs(structured.actor_parameters - direct.actor_parameters) / structured.actor_parameters < .10
    assert structured.critic_parameters == direct.critic_parameters
    x = torch.from_numpy(machine)
    encoded = structured.actor.machine(x)
    assert torch.allclose(structured.actor.machine(x[[3, 1, 0, 2]]), encoded[[3, 1, 0, 2]])
    assert direct.actor.net[0].in_features == machine.size + worker.size


def test_remaining_time_ablation_only_masks_numeric_commitment_fields():
    torch.manual_seed(1)
    full = MatchingPolicy("S")
    ablated = MatchingPolicy("R")
    assert full.actor_parameters == ablated.actor_parameters
    machine = torch.randn(4, 21)
    worker = torch.randn(2, 4)
    m, w = ablated.actor_view(machine, worker)
    expected_m, expected_w = machine.clone(), worker.clone()
    expected_m[:, 7] = 0
    expected_m[:, 20] = 0
    expected_w[:, 1] = 0
    assert torch.equal(m, expected_m) and torch.equal(w, expected_w)
    assert torch.equal(machine, full.actor_view(machine, worker)[0])


def test_masked_joint_matching_distribution_and_wait():
    env = CoordinationEnv(configs()["specialist_generalist_scale2"])
    env.reset(73100052)
    submissions = env.fixed_submissions("threshold")
    admitted = env.admitted(submissions)
    machine, worker = actor_inputs(env.config, admitted, env.tick)
    mask = env.matching_mask(submissions)
    legal = set(legal_matchings(env.config, admitted))
    assert mask[0] and {env.actions[i] for i in np.flatnonzero(mask)} == legal
    for arm in ("S", "R", "G3", "D3"):
        policy = MatchingPolicy(arm)
        distribution = policy.distribution(torch.from_numpy(machine), torch.from_numpy(worker),
                                           torch.from_numpy(mask))
        probabilities = distribution.probs.detach().numpy()
        assert abs(probabilities.sum() - 1) < 1e-6
        assert np.all(probabilities[~mask] == 0)
        decision = choose(policy, env, deterministic=True)
        assert decision["matching_mask"][decision["matching_index"]]


def test_save_load_determinism_future_tape_independence_and_tail(tmp_path):
    config = configs()["reciprocal_scale2"]
    a, b = CoordinationEnv(config), CoordinationEnv(config)
    a.reset(73100053)
    b.reset(73100053)
    policy = MatchingPolicy("S")
    checkpoint = tmp_path / "policy.pt"
    torch.save({"arm": "S", "state_dict": policy.state_dict()}, checkpoint)
    restored = load_policy(checkpoint)
    assert choose(policy, a, deterministic=True)["matching_index"] == choose(restored, b, deterministic=True)["matching_index"]
    b.inner._events = tuple((False,) * config.machines for _ in b.inner._events)
    assert choose(restored, a, deterministic=True)["matching_index"] == choose(restored, b, deterministic=True)["matching_index"]
    for _ in range(48):
        decision = choose(restored, a, deterministic=True)
        _, done = a.step(decision["submissions"], decision["matching_index"])
    assert done
    metrics = a.metrics()
    assert abs(metrics["episode_return"] + metrics["objective"]) < 1e-8
    assert abs(metrics["H_cost"] + metrics["tail_cost"] - metrics["objective"]) < 1e-8
    assert metrics["invalid_assignments"] == 0


def test_new_raw_and_derived_seed_roles_are_disjoint():
    seed_registry_check()
