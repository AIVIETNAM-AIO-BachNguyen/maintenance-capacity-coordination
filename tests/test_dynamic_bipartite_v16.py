"""Engineering checks for v16 information flow and canonical physical decisions."""
from dataclasses import replace

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import family_config
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    EDGES, STOP, SequencePolicy, choose, enumerate_sequences, matching_from_sequence,
)
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv


def _fixture():
    machine = torch.zeros((1, 5, 24))
    worker = torch.zeros((1, 3, 4))
    service = torch.tensor([[3., 2., 0.], [0., 2., 3.], [3., 0., 3.],
                            [0., 2., 2.], [3., 3., 0.]]) / 12
    machine[:, :, 14:17] = service
    machine[:, :, 17:20] = .5
    machine[:, :, 3] = torch.tensor([1., 1., 0., 0., 1.])
    machine[:, :, 1] = torch.tensor([.5, .375, .125, .25, .5])
    machine[:, :, 20:23] = torch.tensor([[1., 0., 1.], [0., 0., 1.],
                                         [1., 0., 1.], [0., 0., 1.], [1., 0., 1.]])
    worker[:, :, 0] = torch.tensor([1., 0., 1.])
    worker[:, :, 1] = torch.tensor([0., 2 / 12, 0.])
    mask = torch.zeros((1, STOP), dtype=torch.bool)
    mask[0, [0, 5, 12]] = True
    used_m = torch.zeros((1, 5), dtype=torch.bool)
    used_w = torch.zeros((1, 3), dtype=torch.bool)
    selected = torch.zeros((1, STOP), dtype=torch.bool)
    return machine, worker, mask, used_m, used_w, selected


def _logits(policy, machine, worker, mask, used_m, used_w, selected, frozen=None):
    actor = policy.actor
    return actor.logits(actor.encode(machine, worker), mask, used_m, used_w, selected,
                        frozen_context=frozen)


def test_busy_time_and_unrequested_machine_have_effective_actor_path():
    machine, worker, mask, used_m, used_w, selected = _fixture()
    later_worker = worker.clone()
    later_worker[0, 1, 1] = 4 / 12
    older_machine = machine.clone()
    older_machine[0, 2, 1] = .25
    old = SequencePolicy("C15")
    old_base = _logits(old, machine, worker, mask, used_m, used_w, selected)[0]
    assert torch.equal(old_base, _logits(old, machine, later_worker, mask, used_m, used_w, selected)[0])
    assert torch.equal(old_base, _logits(old, older_machine, worker, mask, used_m, used_w, selected)[0])
    for arm in ("C16", "I16"):
        torch.manual_seed(78700000)
        policy = DynamicSequencePolicy(arm)
        base = _logits(policy, machine, worker, mask, used_m, used_w, selected)[0]
        busy = _logits(policy, machine, later_worker, mask, used_m, used_w, selected)[0]
        age = _logits(policy, older_machine, worker, mask, used_m, used_w, selected)[0]
        assert float((base - busy).abs().max().detach()) > 1e-6
        assert float((base - age).abs().max().detach()) > 1e-6


def test_same_weight_root_and_refresh_only_change_scores_not_legality():
    machine, worker, mask, used_m, used_w, selected = _fixture()
    c, d = DynamicSequencePolicy("C16"), DynamicSequencePolicy("D16")
    d.load_state_dict(c.state_dict(), strict=True)
    assert c.actor_parameters == d.actor_parameters
    root_c, _, root_context = _logits(c, machine, worker, mask, used_m, used_w, selected)
    root_d, _, _ = _logits(d, machine, worker, mask, used_m, used_w, selected)
    assert torch.equal(root_c, root_d)
    selected[:, 0] = True
    used_m[:, EDGES[0][0]] = True
    used_w[:, EDGES[0][1]] = True
    later_c, feasible_c, _ = _logits(c, machine, worker, mask, used_m, used_w, selected)
    later_d, feasible_d, _ = _logits(d, machine, worker, mask, used_m, used_w, selected,
                                    frozen=root_context)
    assert torch.equal(feasible_c, feasible_d)
    assert not feasible_c[:, :3].any()
    assert float((later_c[:, :STOP][feasible_c] - later_d[:, :STOP][feasible_d])
                 .abs().max().detach()) > 1e-8


def test_canonical_probability_normalization_reload_and_physical_action(tmp_path):
    machine, worker, mask, *_ = _fixture()
    policy = DynamicSequencePolicy("C16")
    paths = enumerate_sequences(mask[0].numpy())
    assert len(paths) > 2
    padded = torch.as_tensor([list(path) + [-1] * (4 - len(path)) for path in paths])
    logprob, _ = policy.logprob_entropy(machine.expand(len(paths), -1, -1),
                                        worker.expand(len(paths), -1, -1),
                                        mask.expand(len(paths), -1), padded)
    assert torch.allclose(logprob.exp().sum(), torch.tensor(1.), atol=1e-5)
    sequence, sampled_lp, _, _ = policy.sample(machine[0], worker[0], mask[0])
    replay_lp, _ = policy.logprob_entropy(machine[0], worker[0], mask[0], sequence)
    assert torch.allclose(sampled_lp, replay_lp[0], atol=1e-6)
    checkpoint = tmp_path / "graph.pt"
    torch.save(policy.state_dict(), checkpoint)
    reloaded = DynamicSequencePolicy("C16")
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    a = policy.sample(machine[0], worker[0], mask[0], deterministic=True)
    b = reloaded.sample(machine[0], worker[0], mask[0], deterministic=True)
    assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])

    config = replace(family_config("n5_k3_sparse", 74740000, "full"), horizon=HORIZON + TAIL)
    env = ScaledCoordinationEnv(config)
    env.reset(78700011)
    submissions = env.fixed_submissions("threshold")
    admitted = env.admitted(submissions)
    before = (env.inner.state, env.tick)
    decision = choose(policy, env, True)
    assert env.actions[decision["matching_index"]] in legal_matchings(config, admitted)
    assert matching_from_sequence(decision["tokens"]) == env.actions[decision["matching_index"]]
    assert before == (env.inner.state, env.tick)
    assert np.isfinite(decision["old_logprob"])


def test_same_count_bipartite_motifs_remain_distinct_and_future_tape_is_absent():
    star = {(0, 0), (1, 1), (2, 1), (3, 1)}
    split = {(0, 0), (1, 1), (2, 2), (3, 2)}
    def make(pairs):
        machine = torch.zeros((1, 5, 24))
        worker = torch.zeros((1, 3, 4))
        worker[:, :, 0] = 1
        machine[:, :, 3] = 1
        machine[:, :, 1] = .375
        for m, j in pairs:
            machine[0, m, 14 + j] = .25
            machine[0, m, 17 + j] = .5
            machine[0, m, 20 + j] = 1
        mask = torch.tensor([[edge in pairs for edge in EDGES]], dtype=torch.bool)
        return machine, worker, mask
    def residual_capacity(pairs):
        remaining = [(m, j) for m, j in pairs if (m, j) != (0, 0)]
        return 2 if any(m != other_m and j != other_j
                        for m, j in remaining for other_m, other_j in remaining) else 1
    assert len(star) == len(split) == 4
    assert residual_capacity(star) == 1
    assert residual_capacity(split) == 2
    used_m = torch.zeros((1, 5), dtype=torch.bool)
    used_w = torch.zeros((1, 3), dtype=torch.bool)
    selected = torch.zeros((1, STOP), dtype=torch.bool)
    selected[0, 0] = used_m[0, 0] = used_w[0, 0] = True
    torch.manual_seed(78700000)
    actor = DynamicSequencePolicy("C16").actor
    m1, w1, mask1 = make(star)
    m2, w2, mask2 = make(split)
    _, summary1 = actor._node_edge_embeddings(m1, w1, mask1, used_m, used_w, selected)
    _, summary2 = actor._node_edge_embeddings(m2, w2, mask2, used_m, used_w, selected)
    assert not torch.allclose(summary1, summary2)
    assert int(mask1.sum()) == int(mask2.sum())

    config = replace(family_config("n5_k3_sparse", 74740000, "full"), horizon=HORIZON + TAIL)
    env = ScaledCoordinationEnv(config)
    env.reset(78700011)
    policy = DynamicSequencePolicy("C16")
    before = choose(policy, env, True)
    events = env.inner._events
    env.inner._events = tuple(reversed(events))
    after = choose(policy, env, True)
    env.inner._events = events
    assert before["tokens"] == after["tokens"]
