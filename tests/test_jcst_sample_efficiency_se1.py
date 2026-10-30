"""Focused engineering tests for SE1 isolation, timing and locked arithmetic."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import jcst_learner_campaign_v1 as campaign
import jcst_sample_efficiency_se1 as se1
from jcst_sample_efficiency_analyze_se1 import log_weights, crossed_bootstrap
from ht_pdm_fjsp.jcst_characterization_v1 import case_config
from ht_pdm_fjsp.jcst_learner_v1 import Policy, action_universe


def test_seed_roles_are_separate_and_paired_evaluation():
    seen = set()
    for case in se1.CASES:
        for arm in se1.ARMS:
            se1._arm_context[0] = arm
            for root in range(10):
                for episode in (0, 499, 1999, 7999):
                    seed = se1.physical_seed("train", case, root, episode)
                    assert seed not in seen and seed < 2**64
                    seen.add(seed)
                assert se1.torch_seed("train", case, arm, root) < 2**64
        se1._arm_context[0] = None
        for bank in ("eval", "calibration"):
            for episode in range(128):
                seed = se1.physical_seed(bank, case, 0, episode)
                assert seed not in seen and seed < 2**64
                seen.add(seed)
        assert se1.physical_seed("eval", case, 0, 7) == se1.physical_seed("eval", case, 9, 7)
    assert se1.action_seed("eval", "base", "S", 0, 96000, 7, 0) != se1.action_seed(
        "eval", "base", "S", 0, 96000, 7, 1)


def test_phase_clocks_preserve_episode_actions_and_cost(monkeypatch):
    config, _ = case_config("base")
    universe = action_universe(config)
    torch.manual_seed(71)
    policy = Policy("S", config, universe)
    torch.manual_seed(882)
    original, _ = campaign.episode(config, universe, 2**54 + 3 * 10**13 + 3, None,
                                   train=True, policy=policy)
    for attr, name in (("observe", "catalogue"), ("encode", "encoding_mask"),
                       ("choose", "inference"), ("advance", "simulator")):
        monkeypatch.setattr(campaign, attr, se1._clocked(name, getattr(campaign, attr)))
    torch.manual_seed(882)
    timed, _ = campaign.episode(config, universe, 2**54 + 3 * 10**13 + 3, None,
                                train=True, policy=policy)
    assert [row["action"] for row in original["trace"]] == [row["action"] for row in timed["trace"]]
    assert original["H_cost"] == timed["H_cost"]
    assert original["tail_cost"] == timed["tail_cost"]
    assert original["final_state"] == timed["final_state"]


def test_locked_log_area_and_bootstrap_gate_on_constant_panel():
    checkpoints = se1.CHECKPOINTS
    weights = log_weights(checkpoints)
    assert math.isclose(sum(weights), 1, abs_tol=1e-12)
    assert math.isclose(float(np.dot(weights, np.full(len(weights), 120.0))), 120.0)
    cube = {case: {"S": np.full((10, 8, 8), 80.0),
                   "F": np.full((10, 8, 8), 100.0),
                   "A": np.full((10, 8, 8), 90.0)} for case in se1.CASES}
    report = crossed_bootstrap(cube, checkpoints, 40)
    for case in se1.CASES:
        assert report[case]["contrasts"]["S-F"]["practical_pass"]
        assert report[case]["contrasts"]["A-F"]["practical_pass"]
        assert report[case]["contrasts"]["S-A"]["practical_pass"]


def test_cap_rejects_expired_deadline(tmp_path):
    with pytest.raises(TimeoutError):
        se1.cap_check(tmp_path, -1)
