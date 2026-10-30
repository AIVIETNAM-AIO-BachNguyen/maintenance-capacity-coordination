"""Engineering assertions for the JCST L1 actor, objective and action contract."""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp.component_service_v1 import (
    ComponentState, ServiceJob, State, initial_state, legal_actions,
)
from ht_pdm_fjsp.component_framework_v1 import advance
from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import (
    Policy, Transition, action_universe, choose, encode, ppo_update, tensors,
)


@pytest.mark.parametrize("case", ["base", "generalist"])
def test_full_catalogue_and_early_pm(case):
    config, state = case_config(case)
    universe = action_universe(config)
    observation = encode(config, state, universe)
    assert universe[0] == ()
    assert all(observation.mask)
    assert len(universe) == (37 if case == "base" else 139)
    assert ((0, 0, 0),) in universe  # healthy age-zero PM remains eligible
    occupied = State(0, state.components,
                     (ServiceJob(1, 0, 0, "PM", 2),) +
                     tuple(None for _ in range(config.technicians - 1)))
    encoded = encode(config, occupied, universe)
    assert {a for a, valid in zip(universe, encoded.mask) if valid} == set(legal_actions(config, occupied))
    assert encoded.mask[0]
    assert not any(a for a, valid in zip(universe, encoded.mask) if valid
                   if any(m == 1 or j == 0 for m, c, j in a))


@pytest.mark.parametrize("case", ["base", "generalist"])
def test_policy_parameter_parity_mask_and_gradients(case):
    config, state = case_config(case)
    universe = action_universe(config)
    item = encode(config, state, universe)
    c, w, flat, mask = tensors([item])
    counts = {}
    for arm in ("S", "F", "A"):
        torch.manual_seed(11)
        policy = Policy(arm, config, universe)
        counts[arm] = policy.parameter_counts()
        dist = policy.distribution(c, w, flat, mask)
        index = torch.tensor([1])
        assert torch.isfinite(dist.log_prob(index)).all()
        assert torch.isfinite(dist.entropy()).all()
        loss = -dist.log_prob(index).mean() + policy.value(flat).square().mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.actor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic.parameters())
        first, _, _ = choose(policy, item, deterministic=True)
        again, _, _ = choose(policy, item, deterministic=True)
        assert first == again and item.mask[first]
    assert counts["S"]["actor"] == counts["A"]["actor"]
    assert counts["S"]["critic"] == counts["F"]["critic"] == counts["A"]["critic"]
    assert abs(counts["S"]["actor"] - counts["F"]["actor"]) / counts["S"]["actor"] < .10


def test_busy_remaining_time_reaches_actor_and_no_future_noise():
    config, state = case_config("base")
    universe = action_universe(config)
    torch.manual_seed(19)
    policy = Policy("S", config, universe)
    assert policy.actor.ablated is False
    assert not any("seed" in key or "uniform" in key for key in vars(encode(config, state, universe)))
    states = [State(3, state.components,
                    (ServiceJob(1, 0, 0, "PM", remaining), None))
              for remaining in (2, 1)]
    encodings = [encode(config, x, universe) for x in states]
    assert np.array_equal(encodings[0].mask, encodings[1].mask)
    outputs = []
    for item in encodings:
        c, w, flat, mask = tensors([item])
        outputs.append(policy.logits(c, w, flat, mask).detach().numpy())
    valid = encodings[0].mask
    assert not np.allclose(outputs[0][0, valid], outputs[1][0, valid])
    # Busy-worker features influence component messages in S; A deliberately
    # removes that pre-scoring cross input while retaining global worker access.
    c, w, flat, mask = tensors([encodings[0]])
    w.requires_grad_(True)
    chosen = int(np.flatnonzero(valid)[1])
    policy.logits(c, w, flat, mask)[0, chosen].backward()
    assert w.grad is not None and w.grad[0, 0, 1].abs() > 0


def test_terminal_tail_once_and_undiscounted_ppo_update():
    config, state = case_config("base")
    universe = action_universe(config)
    for tick in range(48):
        step = advance(config, state, (), 99117, 48, 16)
        if tick < 47:
            assert step.tail_cost == 0 and not step.done
        else:
            assert step.done and len(step.records) == 17
            assert step.reward == -(step.controlled_cost + step.tail_cost)
            assert step.tail_cost == sum(x["cost"] for x in step.records[1:])
        state = step.state
    torch.manual_seed(7)
    policy = Policy("S", config, universe)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    item = encode(config, initial_state(config), universe)
    c, w, flat, mask = tensors([item])
    with torch.no_grad():
        dist = policy.distribution(c, w, flat, mask)
        lp = float(dist.log_prob(torch.tensor([0])).item())
        value = float(policy.value(flat).item())
    data = [Transition(item, 0, lp, value, -0.07, False),
            Transition(item, 0, lp, value, -0.14, True)]
    before = [p.detach().clone() for p in policy.parameters()]
    stats = ppo_update(policy, optimizer, data, epochs=1, minibatch=2, lam=.95)
    assert np.isfinite(stats["loss"])
    assert any(not torch.equal(x, y) for x, y in zip(before, policy.parameters()))


def test_checkpoint_roundtrip(tmp_path):
    config, state = case_config("base")
    universe = action_universe(config)
    torch.manual_seed(17)
    policy = Policy("S", config, universe)
    encoded = encode(config, state, universe)
    c, w, flat, mask = tensors([encoded])
    expected = policy.logits(c, w, flat, mask).detach()
    path = tmp_path / "checkpoint.pt"
    torch.save(policy.state_dict(), path)
    copy = Policy("S", config, universe)
    copy.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(expected, copy.logits(c, w, flat, mask).detach())
