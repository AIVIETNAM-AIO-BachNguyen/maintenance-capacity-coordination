"""Focused v6 mechanism and protocol checks; fixtures are not scientific evidence."""
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.joint_score_v6 import (
    ACTIONS, actor_for, masked_score_mse, pair_interaction_errors,
    parameter_count, teacher_targets,
)
from maintenance_capacity_coordination.structured_policy_v3 import actor_inputs
import experiments.joint_score_v6 as v6
from experiments.joint_score_v6 import (
    CASES, TRAIN_ROOTS, forecast_seed, seed_registry_check, training_raw,
    save_student, load_student, _require_pilot_go,
)


def example_inputs():
    env = CoordinationEnv(configs()[CASES[0]])
    env.reset(73217001)
    submissions = env.fixed_submissions("threshold")
    admitted = env.admitted(submissions)
    return actor_inputs(env.config, admitted, env.tick)


def test_exact_active_parameter_matching_and_wait_zero():
    machine, worker = example_inputs()
    counts = {arm: parameter_count(actor_for(arm)) for arm in ("J6", "A6", "G6")}
    assert counts == {"J6": 4850, "A6": 4825, "G6": 4905}
    for arm in counts:
        score = actor_for(arm)(torch.from_numpy(machine), torch.from_numpy(worker))
        assert score.shape == (len(ACTIONS),)
        assert score[0].item() == 0.


def test_additive_ablation_is_exactly_edge_additive():
    torch.manual_seed(7)
    actor = actor_for("A6")
    machine, worker = example_inputs()
    with torch.no_grad():
        score = actor(torch.from_numpy(machine), torch.from_numpy(worker))
    for pair in (a for a in ACTIONS if len(a) == 2):
        i = ACTIONS.index(pair)
        a = ACTIONS.index((pair[0],))
        b = ACTIONS.index((pair[1],))
        assert torch.isclose(score[i], score[a] + score[b], atol=1e-6)


def test_joint_term_changes_only_two_edge_actions():
    actor = actor_for("J6")
    with torch.no_grad():
        for p in actor.pair.parameters():
            p.zero_()
        actor.pair[-1].bias.fill_(2.)
    machine, worker = example_inputs()
    with torch.no_grad():
        score = actor(torch.from_numpy(machine), torch.from_numpy(worker))
    for action in ACTIONS:
        i = ACTIONS.index(action)
        if len(action) == 2:
            a = ACTIONS.index((action[0],))
            b = ACTIONS.index((action[1],))
            assert torch.isclose(score[i] - score[a] - score[b], torch.tensor(2.), atol=1e-6)
        if len(action) == 0:
            assert score[i].item() == 0.


def test_teacher_target_is_relative_wait_score_and_masks_illegal_roots():
    first = ACTIONS.index(((0, 0),))
    second = ACTIONS.index(((1, 1),))
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[[0, first, second]] = True
    rows = [{"matching": ACTIONS[first], "expected_cost": 70.},
            {"matching": (), "expected_cost": 90.},
            {"matching": ACTIONS[second], "expected_cost": 110.}]
    target = teacher_targets(rows, mask)
    assert target[0] == 0 and target[first] == 1 and target[second] == -1
    assert np.all(target[~mask] == 0)
    with pytest.raises(ValueError, match="incomplete"):
        teacher_targets(rows[:2], mask)


def test_one_action_has_zero_loss_and_gradient():
    mask = torch.zeros((3, len(ACTIONS)), dtype=torch.bool)
    mask[:, 0] = True
    predicted = torch.zeros((3, len(ACTIONS)), requires_grad=True)
    target = torch.zeros_like(predicted)
    loss = masked_score_mse(predicted, target, mask)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(predicted.grad) == 0
    assert np.all(teacher_targets([], mask[0].numpy()) == 0)


def test_pair_interaction_error_detects_exact_and_additive_forecasts():
    pair = ((0, 0), (1, 1))
    rows = [{"matching": (), "expected_cost": 100.},
            {"matching": ((0, 0),), "expected_cost": 90.},
            {"matching": ((1, 1),), "expected_cost": 90.},
            {"matching": pair, "expected_cost": 70.}]
    additive = torch.zeros(len(ACTIONS))
    additive[ACTIONS.index(((0, 0),))] = .5
    additive[ACTIONS.index(((1, 1),))] = .5
    additive[ACTIONS.index(pair)] = 1.
    assert pair_interaction_errors(additive, rows) == [10.]
    joint = additive.clone()
    joint[ACTIONS.index(pair)] = 1.5
    assert pair_interaction_errors(joint, rows) == [0.]


def test_checkpoint_roundtrip_and_fresh_seed_roles(tmp_path):
    assert seed_registry_check()
    raw = training_raw(TRAIN_ROOTS[0], CASES[0], 0)
    assert raw not in TRAIN_ROOTS
    assert forecast_seed(CASES[0], raw, 1) != forecast_seed(CASES[0], raw, 2)
    actor = actor_for("J6")
    path = tmp_path / "student.pt"
    save_student(path, actor, CASES[0], "J6", TRAIN_ROOTS[0], 16, "dataset")
    restored, payload = load_student(path)
    machine, worker = example_inputs()
    with torch.no_grad():
        expected = actor(torch.from_numpy(machine), torch.from_numpy(worker))
        actual = restored(torch.from_numpy(machine), torch.from_numpy(worker))
    assert torch.equal(actual, expected)
    assert payload["dataset_sha256"] == "dataset"


def test_replication_refuses_missing_or_failed_pilot(tmp_path):
    with pytest.raises(RuntimeError, match="requires"):
        _require_pilot_go(None)
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": "heldout", "status": "COMPLETED",
                                                   "audit_status": "PASS"}))
    (tmp_path / "replication_go_decision.json").write_text(json.dumps({"go": False,
        "protocol_sha256": "irrelevant"}))
    with pytest.raises(RuntimeError, match="gate is closed"):
        _require_pilot_go(tmp_path)


def test_actor_inputs_ignore_future_physical_events():
    env = CoordinationEnv(configs()[CASES[0]])
    env.reset(73217001)
    submissions = env.fixed_submissions("threshold")
    before = actor_inputs(env.config, env.admitted(submissions), env.tick)
    original = env.inner._events
    env.inner._events = tuple(tuple(not value for value in tick) for tick in original)
    after = actor_inputs(env.config, env.admitted(submissions), env.tick)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert env.matching_mask(submissions).shape == (len(ACTIONS),)


def test_timestamped_run_refuses_overwrite(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    class FrozenDateTime:
        @staticmethod
        def now(_timezone):
            return datetime(2026, 10, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(v6, "datetime", FrozenDateTime)
    first, _ = v6.new_run(tmp_path, "smoke")
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        v6.new_run(tmp_path, "smoke")
