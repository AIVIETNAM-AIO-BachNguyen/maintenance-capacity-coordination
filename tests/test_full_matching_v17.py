"""Focused engineering checks for v17 matching-level value supervision."""
from __future__ import annotations

import copy

import numpy as np
import torch

from experiments import full_matching_labels_v17 as labels
from experiments import same_tick_pilot_v17 as pilot
from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss, tensorize
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    ACTIONS, EDGES, MAX_TOKENS, STOP, enumerate_sequences, matching_from_sequence,
)


def fixture():
    machine = torch.zeros((5, 24))
    worker = torch.zeros((3, 4))
    machine[:, 3] = 1.
    machine[:, 14:17] = .25
    machine[:, 17:20] = .5
    worker[:, 0] = 1.
    mask = torch.zeros(STOP, dtype=torch.bool)
    mask[[0, 5, 12]] = True
    return machine, worker, mask


def record():
    machine, worker, mask = fixture()
    paths = enumerate_sequences(mask.numpy())
    lookup = {matching_from_sequence(path): path for path in paths}
    matchings = ((), (EDGES[0],), tuple(sorted((EDGES[0], EDGES[5]))))
    assert all(m in lookup for m in matchings)
    return {
        "case": "n5k3_residual_a", "machine_obs": machine.tolist(),
        "worker_obs": worker.tolist(), "edge_mask": mask.tolist(),
        "matching_indices": [ACTIONS.index(m) for m in matchings],
        "sequences": [list(lookup[m]) for m in matchings],
        "target_probabilities": [.2, .3, .5],
    }


def test_exact_canonical_matching_probability_and_supervised_gradient():
    row = record()
    data = tensorize([row])
    c = pilot.new_policy("C17")
    paths = enumerate_sequences(np.asarray(row["edge_mask"], bool))
    machine, worker, mask = fixture()
    all_tokens = torch.as_tensor(
        [list(path) + [-1] * (MAX_TOKENS - len(path)) for path in paths])
    logp, _ = c.logprob_entropy(machine.expand(len(paths), -1, -1),
                                worker.expand(len(paths), -1, -1),
                                mask.expand(len(paths), -1), all_tokens)
    assert torch.allclose(logp.exp().sum(), torch.tensor(1.), atol=1e-5)
    scores = candidate_logits(c, data)
    for j, tokens in enumerate(row["sequences"]):
        padded = torch.as_tensor(tokens + [-1] * (MAX_TOKENS - len(tokens)))
        exact, _ = c.logprob_entropy(machine, worker, mask, padded)
        assert torch.allclose(scores[0, j], exact[0], atol=1e-6)
    assert row["sequences"][0] == [STOP]  # WAIT when a root edge exists.
    assert row["sequences"][-1][-1] in (STOP, EDGES.index(EDGES[5]))
    loss = listwise_loss(c, data)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in c.actor.parameters())
    s = pilot.new_policy("S17")
    assert torch.isfinite(listwise_loss(s, data))


def test_same_weight_frozen_root_and_label_candidates_do_not_advance_time():
    c = pilot.new_policy("C17")
    d = pilot.new_policy("D17")
    d.load_state_dict(c.state_dict(), strict=True)
    machine, worker, mask = fixture()
    zeros_m = torch.zeros((1, 5), dtype=torch.bool)
    zeros_w = torch.zeros((1, 3), dtype=torch.bool)
    zeros_e = torch.zeros((1, STOP), dtype=torch.bool)
    cm = c.actor.logits(c.actor.encode(machine[None], worker[None]), mask[None],
                        zeros_m, zeros_w, zeros_e)[0]
    dm = d.actor.logits(d.actor.encode(machine[None], worker[None]), mask[None],
                        zeros_m, zeros_w, zeros_e)[0]
    assert torch.equal(cm, dm)
    assert c.actor_parameters == d.actor_parameters

    # Find one natural simulator state under a fresh engineering seed.
    case, root, raw = "n5k3_residual_a", 78710000, 79700001
    selected, _ = labels.selected_states(c, case, root, raw)
    state = next((value for value in selected.values() if value is not None), None)
    assert state is not None
    env, tick, submissions = state
    from maintenance_capacity_coordination.same_tick_policy_v15 import initial_edge_mask
    mask_np = initial_edge_mask(env.config, env.admitted(submissions))
    s = pilot.new_policy("S17")
    before = (env.tick, copy.deepcopy(env.inner.state), env.inner.metrics["objective"])
    actions, sequences, reasons = labels.choose_candidates(
        env, c, d, s, raw, case, root, tick, submissions, mask_np)
    assert len(actions) == len(set(actions)) == min(8, int(env.matching_mask(submissions).sum()))
    assert actions[0] == env.actions.index(())
    for index, tokens in zip(actions, sequences):
        assert matching_from_sequence(tokens) == env.actions[index]
        assert env.matching_mask(submissions)[index]
    assert (env.tick, env.inner.state, env.inner.metrics["objective"]) == before
    old_events = env.inner._events
    env.inner._events = tuple(reversed(old_events))
    replayed = labels.choose_candidates(env, c, d, s, raw, case, root, tick, submissions, mask_np)
    env.inner._events = old_events
    assert replayed == (actions, sequences, reasons)


def test_protocol_arms_and_gate_include_old_label_and_structured_controls():
    assert set(pilot.ARMS) == {"C17", "D17", "I17", "R17", "B17", "S17",
                               "G17", "O17", "P17", "Q17"}
    from experiments.audit_same_tick_v17 import gate
    assert callable(gate)
