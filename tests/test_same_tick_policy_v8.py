"""Probability, feasibility and unchanged physical-contract checks for v8."""
from collections import defaultdict
import math

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs, catalogue
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v8 import (
    EDGES, STOP, SequencePolicy, choose, enumerate_sequences, initial_edge_mask,
    matching_from_sequence,
)
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs


def test_decoder_covers_exact_legal_physical_matchings_and_executes_once():
    env = CoordinationEnv(configs()["reciprocal_scale2"])
    env.reset(73400000)
    policy = SequencePolicy("C8")
    for _ in range(48):
        submissions = env.fixed_submissions("threshold")
        admitted = env.admitted(submissions)
        edge_mask = initial_edge_mask(env.config, admitted)
        legal = set(legal_matchings(env.config, admitted))
        sequences = enumerate_sequences(edge_mask)
        assert {matching_from_sequence(seq) for seq in sequences} == legal
        assert () in legal
        decision = choose(policy, env, deterministic=True)
        assert decision["matching"] in legal
        assert decision["matching_index"] == env.actions.index(decision["matching"])
        before = env.tick
        env.step(decision["submissions"], decision["matching_index"])
        assert env.tick == before + 1
    assert abs(env.metrics()["episode_return"] + env.metrics()["objective"]) < 1e-8


def test_exact_sequence_probability_normalization_and_ratio():
    env = CoordinationEnv(configs()["specialist_generalist_scale2"])
    env.reset(73400001)
    submissions = env.fixed_submissions("threshold")
    admitted = env.admitted(submissions)
    machine, worker = actor_inputs(env.config, admitted, env.tick)
    mask = np.ones(STOP, dtype=bool)  # nontrivial two-worker order multiplicity
    m, w, e = torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(mask)
    policy = SequencePolicy("C8")
    sequences = enumerate_sequences(mask)
    padded = torch.as_tensor([list(seq) + [-1] * (3 - len(seq)) for seq in sequences])
    lp, entropy = policy.logprob_entropy(m.expand(len(sequences), -1, -1),
                                         w.expand(len(sequences), -1, -1),
                                         e.expand(len(sequences), -1), padded)
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.), atol=1e-6)
    assert bool((entropy >= 0).all())
    marginals = defaultdict(float)
    for seq, probability in zip(sequences, lp.exp().detach().tolist()):
        marginals[matching_from_sequence(seq)] += probability
    assert set(marginals) == set(catalogue(4, 2))
    assert math.isclose(sum(marginals.values()), 1., abs_tol=1e-6)
    sampled, sampled_lp, _, _ = policy.sample(m, w, e)
    recomputed, _ = policy.logprob_entropy(m, w, e, sampled)
    assert torch.allclose(sampled_lp, recomputed[0], atol=1e-6)
    old = recomputed.detach()
    with torch.no_grad():
        policy.actor.stop_score[-1].bias.add_(0.1)
    updated, _ = policy.logprob_entropy(m, w, e, sampled)
    assert torch.allclose(torch.exp(updated - old), updated.exp() / old.exp(), atol=1e-6)


def test_context_ablation_only_changes_later_scores_and_matched_parameters():
    c, d, b = SequencePolicy("C8"), SequencePolicy("D8"), SequencePolicy("B8")
    d.load_state_dict(c.state_dict())
    assert c.actor_parameters == d.actor_parameters
    assert abs(c.actor_parameters - b.actor_parameters) / c.actor_parameters < .1
    machine = torch.randn(1, 4, 21)
    worker = torch.randn(1, 2, 4)
    mask = torch.ones(1, STOP, dtype=torch.bool)
    encoded_c, encoded_d = c.actor.encode(machine, worker), d.actor.encode(machine, worker)
    used_m = torch.zeros(1, 4, dtype=torch.bool)
    used_w = torch.zeros(1, 2, dtype=torch.bool)
    selected = torch.zeros(1, STOP, dtype=torch.bool)
    root_c, _, root_context = c.actor.logits(encoded_c, mask, used_m, used_w, selected)
    root_d, _, _ = d.actor.logits(encoded_d, mask, used_m, used_w, selected)
    assert torch.allclose(root_c, root_d)
    selected[:, 0] = True
    used_m[:, EDGES[0][0]] = True
    used_w[:, EDGES[0][1]] = True
    later_c, feasible_c, _ = c.actor.logits(encoded_c, mask, used_m, used_w, selected)
    later_d, feasible_d, _ = d.actor.logits(encoded_d, mask, used_m, used_w, selected,
                                           frozen_context=root_context)
    assert torch.equal(feasible_c, feasible_d)
    assert not torch.allclose(later_c[:, :STOP][:, feasible_c[0]], later_d[:, :STOP][:, feasible_d[0]])


def test_stop_singleton_forced_end_and_checkpoint(tmp_path):
    assert matching_from_sequence((STOP,)) == ()
    assert matching_from_sequence((0, STOP)) == ((0, 0),)
    assert matching_from_sequence((0, 3)) == ((0, 0), (1, 1))
    mask = np.zeros(STOP, dtype=bool)
    mask[0] = True
    sequences = enumerate_sequences(mask)
    assert set(sequences) == {(STOP,), (0,)}
    policy = SequencePolicy("D8")
    machine, worker = torch.randn(4, 21), torch.randn(2, 4)
    edge_mask = torch.from_numpy(mask)
    first = policy.sample(machine, worker, edge_mask, deterministic=True)
    path = tmp_path / "model.pt"
    torch.save(policy.state_dict(), path)
    restored = SequencePolicy("D8")
    restored.load_state_dict(torch.load(path, weights_only=True))
    second = restored.sample(machine, worker, edge_mask, deterministic=True)
    assert torch.equal(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_invalid_sequence_and_future_tape_independence():
    env = CoordinationEnv(configs()["reciprocal_scale2"])
    env.reset(73400002)
    policy = SequencePolicy("C8")
    initial_tick = env.tick
    initial_state = env.inner.state
    first = choose(policy, env, deterministic=True)
    # The actor only receives present arrays; removing future events must not
    # change the current internal decision, and choose must not step physics.
    future = env.inner._events
    env.inner._events = tuple(reversed(future))
    second = choose(policy, env, deterministic=True)
    env.inner._events = future
    assert first["tokens"] == second["tokens"]
    assert first["matching"] == second["matching"]
    assert env.tick == initial_tick and env.inner.state == initial_state
    machine, worker = torch.randn(4, 21), torch.randn(2, 4)
    mask = torch.zeros(STOP, dtype=torch.bool)
    mask[0] = True
    with np.testing.assert_raises(ValueError):
        policy.logprob_entropy(machine, worker, mask, torch.tensor([1, -1, -1]))
    with np.testing.assert_raises(ValueError):
        policy.logprob_entropy(machine, worker, mask, torch.tensor([0, STOP, -1]))
