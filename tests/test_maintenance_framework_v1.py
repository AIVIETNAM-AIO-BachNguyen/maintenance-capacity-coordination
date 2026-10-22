import numpy as np
import pytest

from ht_pdm_fjsp.maintenance_framework_v1 import (
    FrameworkEnv, RuleController, catalogue, configs, evaluate_episode,
)
from ht_pdm_fjsp.maintenance_requests import legal_matchings


def test_catalogue_and_mask_cover_exact_legal_matchings():
    assert len(catalogue()) == 21
    assert catalogue()[0] == ()
    for config in configs().values():
        env = FrameworkEnv(config)
        env.reset(seed=62500000)
        for _ in range(48):
            _, admitted = env._admitted()
            legal = set(legal_matchings(config, admitted))
            assert set(action for action, ok in zip(env.actions, env.action_masks()) if ok) == legal
            action = next(i for i, ok in enumerate(env.action_masks()) if ok and env.actions[i] == ())
            _, _, done, _, _ = env.step(action)
        assert done


def test_wait_tail_and_cost_reconciliation():
    env = FrameworkEnv()
    env.reset(seed=62500001)
    total = 0
    for tick in range(48):
        wait = env.actions.index(())
        assert env.action_masks()[wait]
        _, reward, done, _, info = env.step(wait)
        total += reward
        assert done == (tick == 47)
    assert env.inner.time == 64
    assert total == env.episode_return == -env.inner.metrics["objective"]
    assert info["episode_metrics"]["H_cost"] + info["episode_metrics"]["tail_cost"] == -total
    with pytest.raises(RuntimeError):
        env.step(wait)


def test_seed_determinism_and_controller_contract():
    config = configs()["reciprocal_scale2"]
    a = evaluate_episode(config, "reciprocal_scale2", 62500000, RuleController("risk_matching"))
    b = evaluate_episode(config, "reciprocal_scale2", 62500000, RuleController("risk_matching"))
    for key in ("objective", "H_cost", "tail_cost", "H_availability", "H_failures",
                "end_pending", "episode_return"):
        assert a[key] == b[key]
    assert a["invalid_assignments"] == 0
    assert a["episode_return"] == -a["objective"]
    assert 0 <= a["selection_seconds"] <= a["episode_wall_seconds"]


def test_mask_rejects_infeasible_action():
    env = FrameworkEnv(configs()["specialist_generalist_scale2"])
    env.reset(seed=62500000)
    bad = next(i for i, ok in enumerate(env.action_masks()) if not ok)
    with pytest.raises(ValueError):
        env.step(bad)


def test_observation_does_not_depend_on_future_event_tape():
    env = FrameworkEnv()
    original, _ = env.reset(seed=62500000)
    env.inner._events = tuple(tuple(not x for x in row) for row in env.inner._events)
    np.testing.assert_array_equal(env._observation(), original)
    assert env.action_masks().any()
