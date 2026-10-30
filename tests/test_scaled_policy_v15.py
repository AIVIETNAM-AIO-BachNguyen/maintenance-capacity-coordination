"""N5/K3 sequential and complete-action probability/feasibility contracts."""
from collections import defaultdict
import math
from dataclasses import replace

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import family_config
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.scaled_coordination_v11 import (
    ScaledCoordinationEnv, actor_inputs_scaled,
)
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    EDGES, STOP, SequencePolicy, choose as choose_sequence,
    enumerate_sequences, initial_edge_mask, matching_from_sequence,
)
from maintenance_capacity_coordination.scaled_matching_policy_v15 import (
    ACTIONS, MatchingPolicy, choose as choose_matching,
)


def env():
    config = replace(family_config("n5_k3_sparse", 74740000, "full"), horizon=HORIZON + TAIL)
    result = ScaledCoordinationEnv(config)
    result.reset(74700004)
    return result


def test_sequence_probability_matching_coverage_and_exact_ratio():
    e = env()
    submissions = e.fixed_submissions("threshold")
    admitted = e.admitted(submissions)
    machine, worker = actor_inputs_scaled(e.config, admitted, e.tick)
    mask = np.ones(STOP, dtype=bool)
    sequences = enumerate_sequences(mask)
    assert {matching_from_sequence(sequence) for sequence in sequences} == set(ACTIONS)
    padded = torch.as_tensor([list(seq) + [-1] * (4 - len(seq)) for seq in sequences])
    m = torch.from_numpy(machine).expand(len(sequences), -1, -1)
    w = torch.from_numpy(worker).expand(len(sequences), -1, -1)
    masks = torch.from_numpy(mask).expand(len(sequences), -1)
    policy = SequencePolicy("C15")
    lp, entropy = policy.logprob_entropy(m, w, masks, padded)
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.), atol=1e-5)
    assert bool((entropy >= 0).all())
    matching_mass = defaultdict(float)
    for path, mass in zip(sequences, lp.exp().tolist()):
        matching_mass[matching_from_sequence(path)] += mass
    assert len(matching_mass) == len(ACTIONS) == 136
    assert math.isclose(sum(matching_mass.values()), 1., abs_tol=1e-5)
    sampled, sampled_lp, _, _ = policy.sample(torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(mask))
    recomputed, _ = policy.logprob_entropy(torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(mask), sampled)
    assert torch.allclose(sampled_lp, recomputed[0], atol=1e-6)
    old = recomputed.detach()
    with torch.no_grad():
        policy.actor.stop_score[-1].bias.add_(0.1)
    new, _ = policy.logprob_entropy(torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(mask), sampled)
    assert torch.allclose(torch.exp(new - old), new.exp() / old.exp(), atol=1e-6)


def test_context_and_complete_action_controls_share_legal_physical_set():
    e = env()
    c, d, b = SequencePolicy("C15"), SequencePolicy("D15"), SequencePolicy("B15")
    d.load_state_dict(c.state_dict())
    assert c.actor_parameters == d.actor_parameters
    assert abs(c.actor_parameters - b.actor_parameters) / c.actor_parameters < .1
    submissions = e.fixed_submissions("threshold")
    admitted = e.admitted(submissions)
    machine, worker = actor_inputs_scaled(e.config, admitted, e.tick)
    mask = initial_edge_mask(e.config, admitted)
    assert {matching_from_sequence(s) for s in enumerate_sequences(mask)} == set(legal_matchings(e.config, admitted))
    cm, dw, edge_mask = torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(mask)
    first_c = c.sample(cm, dw, edge_mask, deterministic=True)[0][0]
    first_d = d.sample(cm, dw, edge_mask, deterministic=True)[0][0]
    assert first_c == first_d
    state, tick = e.inner.state, e.tick
    for policy in (c, d, b):
        choice = choose_sequence(policy, e, True)
        assert e.actions[choice["matching_index"]] in legal_matchings(e.config, admitted)
        assert e.inner.state == state and e.tick == tick
    for arm in ("S15", "G15"):
        policy = MatchingPolicy(arm)
        choice = choose_matching(policy, e, True)
        assert choice["submissions"] == submissions
        assert e.actions[choice["matching_index"]] in legal_matchings(e.config, admitted)
    q = MatchingPolicy("Q15")
    q_choice = choose_matching(q, e, True)
    assert q_choice["submissions"] == e.fixed_submissions("D")
    assert e.matching_mask(q_choice["submissions"])[q_choice["matching_index"]]


def test_future_tape_does_not_enter_scaled_policy():
    e = env()
    policy = SequencePolicy("C15")
    state, tick = e.inner.state, e.tick
    before = choose_sequence(policy, e, True)
    events = e.inner._events
    e.inner._events = tuple(reversed(events))
    after = choose_sequence(policy, e, True)
    e.inner._events = events
    assert before["tokens"] == after["tokens"]
    assert e.inner.state == state and e.tick == tick

def test_c_d_later_context_differs_with_identical_weights():
    c, d = SequencePolicy("C15"), SequencePolicy("D15")
    d.load_state_dict(c.state_dict())
    machine, worker = torch.randn(1, 5, 24), torch.randn(1, 3, 4)
    mask = torch.ones(1, STOP, dtype=torch.bool)
    encoded_c, encoded_d = c.actor.encode(machine, worker), d.actor.encode(machine, worker)
    used_m = torch.zeros(1, 5, dtype=torch.bool)
    used_w = torch.zeros(1, 3, dtype=torch.bool)
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
    assert not torch.allclose(later_c[:, :STOP][:, feasible_c[0]],
                              later_d[:, :STOP][:, feasible_d[0]])


def test_scaled_checkpoint_reload_and_stop_forced_end(tmp_path):
    mask = np.zeros(STOP, dtype=bool)
    mask[0] = True
    assert set(enumerate_sequences(mask)) == {(STOP,), (0,)}
    assert matching_from_sequence((STOP,)) == ()
    assert matching_from_sequence((0, STOP)) == ((0, 0),)
    policy = SequencePolicy("D15")
    machine, worker = torch.randn(5, 24), torch.randn(3, 4)
    first = policy.sample(machine, worker, torch.from_numpy(mask), deterministic=True)
    path = tmp_path / "scaled.pt"
    torch.save(policy.state_dict(), path)
    restored = SequencePolicy("D15")
    restored.load_state_dict(torch.load(path, weights_only=True))
    second = restored.sample(machine, worker, torch.from_numpy(mask), deterministic=True)
    assert torch.equal(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_candidate_residual_graph_summary_matches_explicit_enumeration():
    actor = SequencePolicy("C15").actor
    edges = torch.arange(STOP * 32, dtype=torch.float32).reshape(1, STOP, 32) / 100.
    feasible = torch.zeros((1, STOP), dtype=torch.bool)
    feasible[0, [0, 1, 3, 4, 6, 10]] = True
    cur_mean, cur_max, cur_count, after_mean, after_max, after_count = actor.residual_summaries(edges, feasible)
    expected_cur = [i for i in range(STOP) if feasible[0, i]]
    assert torch.allclose(cur_mean[0], edges[0, expected_cur].mean(0))
    assert torch.allclose(cur_max[0], edges[0, expected_cur].amax(0))
    assert torch.isclose(cur_count[0], torch.tensor(len(expected_cur) / STOP))
    for candidate, (m, j) in enumerate(EDGES):
        remaining = [i for i in expected_cur if EDGES[i][0] != m and EDGES[i][1] != j and i > candidate]
        assert torch.isclose(after_count[0, candidate], torch.tensor(len(remaining) / STOP))
        if remaining:
            assert torch.allclose(after_mean[0, candidate], edges[0, remaining].mean(0))
            assert torch.allclose(after_max[0, candidate], edges[0, remaining].amax(0))
        else:
            assert torch.equal(after_mean[0, candidate], torch.zeros(32))
            assert torch.equal(after_max[0, candidate], torch.zeros(32))


def test_canonical_bijection_and_unrestricted_multiplicity():
    from collections import Counter
    full = np.ones(STOP, dtype=bool)
    canonical = enumerate_sequences(full)
    unrestricted = enumerate_sequences(full, canonical=False)
    assert len(canonical) == len(ACTIONS) == 136
    assert {matching_from_sequence(s) for s in canonical} == set(ACTIONS)
    assert len(unrestricted) > len(canonical)
    counts = Counter(matching_from_sequence(s) for s in unrestricted)
    assert max(counts.values()) == 6
    for sequence in canonical:
        edges = [t for t in sequence if t != STOP]
        assert edges == sorted(edges)
    c = SequencePolicy("C15")
    u = SequencePolicy("P15")
    assert c.actor_parameters == u.actor_parameters
    assert u.actor.canonical


def test_canonical_residual_mask_keeps_only_higher_edge_indices():
    policy = SequencePolicy("C15")
    machine, worker = torch.randn(1, 5, 24), torch.randn(1, 3, 4)
    encoded = policy.actor.encode(machine, worker)
    mask = torch.ones(1, STOP, dtype=torch.bool)
    used_m = torch.zeros(1, 5, dtype=torch.bool)
    used_w = torch.zeros(1, 3, dtype=torch.bool)
    selected = torch.zeros(1, STOP, dtype=torch.bool)
    selected[0, 4] = True
    used_m[0, EDGES[4][0]] = True
    used_w[0, EDGES[4][1]] = True
    _, feasible, _ = policy.actor.logits(encoded, mask, used_m, used_w, selected)
    assert not feasible[0, :5].any()
    assert feasible[0, 5:].any()
