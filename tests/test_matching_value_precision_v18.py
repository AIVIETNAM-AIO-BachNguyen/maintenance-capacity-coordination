"""Focused v18 qualification, arm and locked-gate regression checks."""
from __future__ import annotations

from experiments import audit_same_tick_v18 as audit
from experiments import full_matching_labels_v18 as labels
from experiments import same_tick_pilot_v18 as pilot
from experiments.scaled_n5k3_v12_config import CASES
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy


def test_precision_and_four_tape_control_have_identical_actor_structure():
    c = pilot.new_policy('C18')
    low = pilot.new_policy('L18')
    frozen = pilot.new_policy('D18')
    assert isinstance(c, DynamicSequencePolicy)
    assert type(c.actor) is type(low.actor) is type(frozen.actor)
    assert c.actor_parameters == low.actor_parameters == frozen.actor_parameters
    assert set(c.state_dict()) == set(low.state_dict()) == set(frozen.state_dict())


def test_stability_uses_disjoint_eight_tape_halves_and_excludes_ties():
    rows = []
    for case in CASES:
        rows.append({'case': case, 'branches': [
            {'total_costs': [10.] * 16},
            {'total_costs': [30.] * 16},
            {'total_costs': [40.] * 8 + [10.] * 8},
        ]})
    out = labels.stability(rows)
    for case in CASES:
        assert out[case]['winner_agreement'] == 1.
        assert out[case]['pair_sign_count'] == 2
        assert out[case]['pair_ties_excluded'] == 1
        assert out[case]['pair_sign_agreement'] == .5


def synthetic_panel(cf_cost=1040.):
    arms = (*pilot.ARMS, 'CF18')
    result = []
    for seed in pilot.HELDOUT_SEEDS:
        for case in CASES:
            for root in pilot.TRAIN_ROOTS:
                for arm in arms:
                    objective = 1000. if arm == 'C18' else (cf_cost if arm == 'CF18' else 1040.)
                    result.append({'seed': seed, 'case': case, 'arm': arm,
                                   'checkpoint': f'{root}:8192', 'objective': objective})
    return result


def test_prelocked_joint_gate_requires_same_weight_feedback():
    good = audit.gate(synthetic_panel(), 'heldout')
    assert good['go'] is True
    assert good['conditions']['C18_minus_CF18_at_most_minus_10']
    assert good['conditions']['C18_minus_L18_at_most_minus_10']
    assert good['conditions']['C18_minus_D18_at_most_minus_20']
    no_feedback = audit.gate(synthetic_panel(cf_cost=1000.), 'heldout')
    assert no_feedback['go'] is False
    assert no_feedback['conditions']['C18_minus_CF18_at_most_minus_10'] is False
    assert no_feedback['conditions']['C18_minus_CF18_cluster_upper_negative'] is False
