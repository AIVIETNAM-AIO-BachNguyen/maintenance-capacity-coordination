"""Focused engineering checks for the prelocked v4 teacher and students."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_request_policies import choose as rollout_choose
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv
from maintenance_capacity_coordination.structured_policy_v3 import MatchingPolicy, actor_inputs
from experiments.forecast_distillation_v4 import (
    ARMS, collect_teacher, create_run, evaluate_student, forecast_seed,
    load_student, new_student, save_student, seed_registry_check,
)


def test_seed_roles_are_disjoint():
    assert seed_registry_check()


def test_teacher_label_uses_exact_threshold_mask_and_root_scores():
    cfg = configs()["reciprocal_scale2"]
    env = CoordinationEnv(cfg)
    raw = 73100097
    env.reset(raw)
    tick = env.tick
    submissions, matching, diag = rollout_choose(
        cfg, env.inner.state, tick, "threshold", "rollout", policy_horizon=48,
        forecast_seed=forecast_seed("reciprocal_scale2", raw, tick),
        scenarios=16, forecast_horizon=6)
    assert submissions == env.fixed_submissions("threshold")
    mask = env.matching_mask(submissions)
    assert mask[env.actions.index(matching)] and mask[0]
    if mask.sum() == 1:
        assert diag["root_scores"] == [] and diag["forecast_transitions"] == 0
    else:
        assert {tuple(tuple(p) for p in score["matching"]) for score in diag["root_scores"]} == {
            action for action, legal in zip(env.actions, mask) if legal}
        assert diag["forecast_transitions"] == int(mask.sum()) * 16 * 6


def test_actor_inputs_are_identical_except_declared_r4_zeroing():
    cfg = configs()["specialist_generalist_scale2"]
    env = CoordinationEnv(cfg)
    env.reset(73100097)
    admitted = env.admitted(env.fixed_submissions("threshold"))
    machine, worker = actor_inputs(cfg, admitted, env.tick)
    m = torch.from_numpy(machine)
    w = torch.from_numpy(worker)
    s = MatchingPolicy("S")
    r = MatchingPolicy("R")
    g = MatchingPolicy("G3")
    sm, sw = s.actor_view(m, w)
    gm, gw = g.actor_view(m, w)
    rm, rw = r.actor_view(m, w)
    assert torch.equal(sm, gm) and torch.equal(sw, gw)
    expected_m = m.clone()
    expected_w = w.clone()
    expected_m[..., [7, 20]] = 0
    expected_w[..., 1] = 0
    assert torch.equal(rm, expected_m) and torch.equal(rw, expected_w)


def test_masked_one_action_loss_checkpoint_and_deterministic_eval(tmp_path: Path):
    cfg = configs()["reciprocal_scale2"]
    env = CoordinationEnv(cfg)
    env.reset(73100097)
    submissions = env.fixed_submissions("threshold")
    machine, worker = actor_inputs(cfg, env.admitted(submissions), env.tick)
    mask = env.matching_mask(submissions)
    policy = new_student("S4", 73190000)
    logits = policy.logits(torch.from_numpy(machine), torch.from_numpy(worker),
                           torch.from_numpy(mask))
    label = int(np.flatnonzero(mask)[0])
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
    assert torch.isfinite(loss)
    one = torch.zeros_like(torch.from_numpy(mask))
    one[0] = True
    one_logits = policy.logits(torch.from_numpy(machine), torch.from_numpy(worker), one)
    assert float(F.cross_entropy(one_logits.unsqueeze(0), torch.tensor([0])).detach()) == pytest.approx(0.0)
    checkpoint = tmp_path / "student.pt"
    save_student(checkpoint, policy, "reciprocal_scale2", "S4", 73190000, 8, "dataset-hash")
    restored, payload = load_student(checkpoint)
    assert payload["dataset_sha256"] == "dataset-hash"
    assert torch.equal(logits, restored.logits(torch.from_numpy(machine), torch.from_numpy(worker),
                                            torch.from_numpy(mask)))
    rows_a = evaluate_student(restored, "reciprocal_scale2", (73100098,),
                              arm="S4", root=73190000, checkpoint="student.pt")
    rows_b = evaluate_student(restored, "reciprocal_scale2", (73100098,),
                              arm="S4", root=73190000, checkpoint="student.pt")
    assert rows_a[0]["objective"] == rows_b[0]["objective"]
    assert rows_a[0]["episode_return"] == pytest.approx(-rows_a[0]["objective"])
    assert rows_a[0]["objective"] == pytest.approx(rows_a[0]["H_cost"] + rows_a[0]["tail_cost"])
    assert rows_a[0]["invalid_assignments"] == 0


def test_teacher_dataset_refuses_overwrite(tmp_path: Path):
    run, _ = create_run(tmp_path, "unit")
    dataset = run / "teacher_reciprocal_scale2_73100099"
    dataset.mkdir()
    with pytest.raises(FileExistsError):
        collect_teacher(run, "reciprocal_scale2", 73100099, episodes=2)


def test_teacher_retains_risk_baseline_on_exact_forecast_ties(monkeypatch):
    from ht_pdm_fjsp import maintenance_request_policies as policies
    from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

    cfg = configs()["reciprocal_scale2"]
    env = CoordinationEnv(cfg)
    env.reset(73100097)
    monkeypatch.setattr(policies, "request_action", lambda *_args: (0, 1))
    monkeypatch.setattr(policies, "transition",
                        lambda _cfg, state, _tick, _submissions, _matching, _events: (state, 0.0, {}))
    admitted = submit_requests(cfg, env.inner.state, 0, (0, 1))
    assert len(legal_matchings(cfg, admitted)) > 1
    baseline = policies.risk_matching(cfg, admitted, 0)
    _, selected, diag = policies.choose(
        cfg, env.inner.state, 0, "threshold", "rollout", policy_horizon=48,
        forecast_seed=73100096, scenarios=2, forecast_horizon=1)
    assert selected == baseline
    assert all(score["expected_cost"] == 0.0 for score in diag["root_scores"])


def test_teacher_forecasts_do_not_read_realized_future_tape():
    from ht_pdm_fjsp.maintenance_requests import submit_requests

    cfg = configs()["reciprocal_scale2"]
    left = CoordinationEnv(cfg)
    right = CoordinationEnv(cfg)
    left.reset(73100097)
    right.reset(73100097)
    state = submit_requests(cfg, left.inner.state, 0, (0, 1))
    left.inner.state = state
    right.inner.state = state
    right.inner._events = tuple(tuple(False for _ in range(cfg.machines)) for _ in right.inner._events)
    assert left.inner._events != right.inner._events
    seed = forecast_seed("reciprocal_scale2", 73100097, 0)
    args = (cfg, state, 0, "threshold", "rollout")
    kwargs = dict(policy_horizon=48, forecast_seed=seed, scenarios=16, forecast_horizon=6)
    one = rollout_choose(*args, **kwargs)
    two = rollout_choose(*args, **kwargs)
    assert one == two


@pytest.mark.parametrize("roots", [(73190000, 73190001, 73190002),
                                   (73193000, 73193001, 73193002)])
def test_panel_analysis_pairs_physical_seeds_and_replication_roots(tmp_path: Path, roots):
    from experiments.forecast_distillation_v4 import analyze_panel, append_jsonl

    seeds = (73100095, 73100096)
    for case in ("reciprocal_scale2", "specialist_generalist_scale2"):
        for seed in seeds:
            for arm, cost in (("F_rollout", 95.0), ("F_risk_matching", 110.0)):
                append_jsonl(tmp_path / "heldout_controls.jsonl",
                             {"case": case, "seed": seed, "arm": arm,
                              "objective": cost, "selection_seconds": .1})
            for root in roots:
                for arm, cost in (("S4", 100.0), ("R4", 120.0), ("G4", 140.0)):
                    append_jsonl(tmp_path / "heldout_students.jsonl",
                                 {"case": case, "seed": seed, "root": root, "arm": arm,
                                  "objective": cost, "selection_seconds": .01})
    result = analyze_panel(tmp_path, seeds, roots)
    assert result["comparisons_S4_minus"]["G4"]["overall"] == pytest.approx(-40)
    assert result["comparisons_S4_minus"]["R4"]["overall"] == pytest.approx(-20)
    assert result["comparisons_S4_minus"]["F2"]["overall"] == pytest.approx(5)
    assert result["selection_time_ratio_S4_over_F2"] == pytest.approx(.1)
    assert result["gate"]["go"] is True
