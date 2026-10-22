import numpy as np
import pytest
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_requests import legal_matchings
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.pilot_policy import PilotPolicy


def test_order_masks_wait_lifecycle_and_h_tail():
    env = CoordinationEnv(configs()["specialist_generalist_scale2"])
    env.reset(73100001)
    total = 0.
    for tick in range(48):
        submissions = env.fixed_submissions("D")
        admitted = env.admitted(submissions)
        assert set(action for action, ok in zip(env.actions, env.matching_mask(submissions)) if ok) == set(
            legal_matchings(env.config, admitted))
        assert env.actions[0] == () and env.matching_mask(submissions)[0]
        with pytest.raises(ValueError):
            env.step(tuple(submissions) + tuple(submissions[:1]) if submissions else (-1,), 0)
        reward, done = env.step(submissions, 0)
        total += reward
        assert done == (tick == 47)
        assert env.inner.time == (64 if done else tick + 1)
        for m, request in enumerate(env.inner.state.requests):
            if env.inner.state.physical.failed[m]:
                assert request is not None and request.kind == "corrective"
    metrics = env.metrics()
    assert metrics["H_cost"] + metrics["tail_cost"] == metrics["objective"]
    assert total == metrics["episode_return"] == -metrics["objective"]
    assert metrics["invalid_assignments"] == 0
    assert metrics["legal_service_waits"] > 0


def test_capacity_only_changes_c_and_future_tape_is_hidden():
    env = CoordinationEnv(configs()["reciprocal_scale2"])
    env.reset(73100002)
    before = {arm: env.machine_observation(arm).copy() for arm in ("B", "C", "E")}
    global_before = env.pre_observation().copy()
    env.inner._events = tuple(tuple(not flag for flag in row) for row in env.inner._events)
    for arm in before:
        np.testing.assert_array_equal(env.machine_observation(arm), before[arm])
    np.testing.assert_array_equal(env.pre_observation(), global_before)
    assert np.all(before["B"][:, -3:] == 0)
    assert before["C"].shape == (4, 21)
    assert before["E"].shape == (4, 77)


def test_exact_joint_log_probability_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(9)
    env = CoordinationEnv(configs()["reciprocal_scale2"])
    env.reset(73100003)
    policy = PilotPolicy("C")
    machine = torch.from_numpy(env.machine_observation("C"))
    eligible = torch.from_numpy(env.request_mask())
    bits = policy.sample_requests(machine, eligible)
    submissions = tuple(np.flatnonzero(bits.numpy()).tolist())
    coordinator = torch.from_numpy(env.coordinator_observation(submissions))
    mask = torch.from_numpy(env.matching_mask(submissions))
    choice = policy.sample_matching(coordinator, mask)
    logprob, _ = policy.logprob_entropy(machine, eligible, bits, coordinator, mask, choice)
    request_dist, matching_dist = policy.distributions(machine, eligible, coordinator, mask)
    expected = (request_dist.log_prob(bits) * eligible.float()).sum() + matching_dist.log_prob(choice)
    torch.testing.assert_close(logprob, expected)
    assert bool(mask[int(choice)])
    torch.save(policy.state_dict(), tmp_path / "policy.pt")
    loaded = PilotPolicy("C")
    loaded.load_state_dict(torch.load(tmp_path / "policy.pt", weights_only=True))
    torch.testing.assert_close(policy.coordinator(coordinator), loaded.coordinator(coordinator))
    torch.testing.assert_close(policy.machine(machine), loaded.machine(machine))


def test_adapter_replays_existing_threshold_risk_controller_exactly():
    from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv
    from ht_pdm_fjsp.maintenance_request_policies import risk_matching

    config = configs()["reciprocal_scale2"]
    old = FrameworkEnv(config)
    new = CoordinationEnv(config)
    old.reset(seed=73100004)
    new.reset(73100004)
    for _ in range(48):
        assert old.inner.state == new.inner.state
        submissions = new.fixed_submissions("threshold")
        matching = risk_matching(config, new.admitted(submissions), new.tick)
        index = new.actions.index(matching)
        _, old_reward, old_done, _, _ = old.step(index)
        new_reward, new_done = new.step(submissions, index)
        assert (old_reward, old_done) == (new_reward, new_done)
    assert old.inner.state == new.inner.state
    assert old.episode_metrics()["objective"] == new.metrics()["objective"]
    assert old.episode_metrics()["H_cost"] == new.metrics()["H_cost"]


def test_two_stage_policy_normalizes_over_all_feasible_joint_actions():
    from itertools import product

    torch.manual_seed(73100011)
    env = CoordinationEnv(configs()["specialist_generalist_scale2"])
    env.reset(73100012)
    eligible = env.request_mask()
    eligible_indices = np.flatnonzero(eligible).tolist()
    for arm in ("B", "C", "D", "E"):
        policy = PilotPolicy(arm)
        machine = None if arm == "D" else torch.from_numpy(env.machine_observation(arm))
        request_vectors = [tuple(eligible_indices)] if arm == "D" else [
            tuple(m for m, bit in zip(eligible_indices, bits) if bit)
            for bits in product((False, True), repeat=len(eligible_indices))
        ]
        total = 0.
        for submissions in request_vectors:
            bits = torch.tensor([float(m in submissions) for m in range(env.config.machines)])
            coordinator = torch.from_numpy(env.coordinator_observation(submissions))
            mask = torch.from_numpy(env.matching_mask(submissions))
            assert bool(mask[0])  # WAIT is always available.
            for index in torch.where(mask)[0]:
                logprob, _ = policy.logprob_entropy(
                    machine, torch.from_numpy(eligible), bits, coordinator, mask, index)
                total += float(logprob.exp().detach())
        assert total == pytest.approx(1., abs=1e-5)
