"""Engineering equivalence and seed-separation checks for L2 diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from jcst_learnability_l2 import (ACTION_BASE, BANK, evaluated_episode,
                                  physical_seed, policy_for, action_seed)

from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import (Policy, Transition, action_universe,
                                         choose, encode, ppo_update)
from ht_pdm_fjsp.jcst_learner_l2 import policy_diagnostic, ppo_update_diagnostic


@pytest.mark.parametrize("arm", ["S", "F", "A"])
def test_diagnostic_update_matches_frozen_l1_recipe(arm):
    torch.set_num_threads(1)
    config, state = case_config("base")
    universe = action_universe(config)
    torch.manual_seed(171)
    first = Policy(arm, config, universe)
    second = Policy(arm, config, universe)
    second.load_state_dict(first.state_dict())
    first_opt = torch.optim.Adam(first.parameters(), lr=3e-4)
    second_opt = torch.optim.Adam(second.parameters(), lr=3e-4)
    encoded = encode(config, state, universe)
    index, logprob, value = choose(first, encoded, deterministic=True)
    data = [Transition(encoded, index, logprob, value, -i / 100, i in (3, 7))
            for i in range(8)]
    rng = torch.random.get_rng_state()
    expected = ppo_update(first, first_opt, data, epochs=2, minibatch=4)
    torch.random.set_rng_state(rng)
    observed = ppo_update_diagnostic(second, second_opt, data, epochs=2, minibatch=4)
    assert expected["loss"] == pytest.approx(observed["loss"], abs=1e-7)
    assert expected["advantage_std"] == pytest.approx(observed["advantage_std"], abs=1e-7)
    assert observed["actor_gradient_pre"] > 0
    assert observed["critic_gradient_pre"] > 0
    assert observed["actor_relative_update"] > 0
    assert observed["critic_relative_update"] > 0
    assert np.isfinite(observed["approx_kl"])
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_dedicated_action_rng_and_physical_seed_roles():
    config, state = case_config("base")
    universe = action_universe(config)
    policy = policy_for("S", config, universe, "engineering", "base", 0)
    encoded = encode(config, state, universe)
    before = torch.random.get_rng_state().clone()
    seed = action_seed("dev", "base", "S", 0, 24000, 0, 0)
    first = policy_diagnostic(policy, encoded, deterministic=False,
                              generator=torch.Generator().manual_seed(seed))
    again = policy_diagnostic(policy, encoded, deterministic=False,
                              generator=torch.Generator().manual_seed(seed))
    assert first == again
    assert torch.equal(before, torch.random.get_rng_state())
    assert encoded.mask[first["index"]]
    assert 0 <= first["wait_probability"] <= 1
    assert first["legal_count"] == int(encoded.mask.sum())
    bank_seeds = {physical_seed(bank, "base", 0, 0) for bank in BANK}
    assert len(bank_seeds) == len(BANK)
    assert ACTION_BASE not in bank_seeds


@pytest.mark.parametrize("case", ["base", "generalist"])
def test_evaluation_replay_determinism_and_cost_accounting(case):
    config, _ = case_config(case)
    universe = action_universe(config)
    policy = policy_for("S", config, universe, "engineering", case, 0)
    physical = physical_seed("engineering", case, 0, 30)
    a = evaluated_episode(config, universe, physical, policy=policy)
    b = evaluated_episode(config, universe, physical, policy=policy)
    assert a["trace"] == b["trace"]
    assert a["H_cost"] + a["tail_cost"] == a["total_cost"]
    assert sum(a["components"].values()) == a["total_cost"]
    assert a["metrics"]["forced_wait"] + a["metrics"]["discretionary_wait"] == a["metrics"]["wait_H"]
    assert len(a["diagnostics"]) == 48
    assert len(a["trace"]) == 64


@pytest.mark.parametrize("case", ["base", "generalist"])
def test_flat_action_edges_are_unique_complete_matching_codes(case):
    config, _ = case_config(case)
    universe = action_universe(config)
    flat = Policy("F", config, universe).actor
    bits = flat.action_bits.detach().numpy()
    assert np.unique(bits, axis=0).shape[0] == len(universe)
