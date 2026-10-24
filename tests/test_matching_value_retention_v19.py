"""V19 protocol, same-weight and locked-gate regression checks."""
from __future__ import annotations

import torch

from experiments import audit_same_tick_v19 as audit
from experiments import same_tick_pilot_v19 as pilot
from experiments.scaled_n5k3_v12_config import CASES
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy


def test_dynamic_and_frozen_actors_have_same_weights_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(41)
    active = pilot.new_policy('C19')
    frozen = pilot.new_policy('D19')
    assert isinstance(active, DynamicSequencePolicy)
    assert active.actor_parameters == frozen.actor_parameters
    assert set(active.state_dict()) == set(frozen.state_dict())
    frozen.load_state_dict(active.state_dict(), strict=True)
    assert all(torch.equal(value, frozen.state_dict()[key])
               for key, value in active.state_dict().items())
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'arm': 'C19', 'case': CASES[0], 'train_root': pilot.TRAIN_ROOTS[0],
                'state_dict': active.state_dict()}, checkpoint)
    restored = pilot.load_policy(checkpoint)
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in active.state_dict().items())


def test_v19_seed_registry_is_disjoint_from_old_evaluation_and_label_roles():
    pilot.seed_registry_check()


def synthetic_panel(cf_cost=1040.):
    arms = (*pilot.NEW_ARMS, 'CF19', *pilot.OLD_ARMS)
    result = []
    for seed in pilot.HELDOUT_SEEDS:
        for case in CASES:
            for root in pilot.TRAIN_ROOTS:
                for arm in arms:
                    objective = 1000. if arm == 'C19' else (cf_cost if arm == 'CF19' else 1040.)
                    result.append({'seed': seed, 'case': case, 'arm': arm,
                                   'checkpoint': f'{root}:8192', 'objective': objective})
    controls = [{'seed': seed, 'case': case, 'arm': 'F_rollout', 'objective': 1000.}
                for seed in pilot.HELDOUT_SEEDS for case in CASES]
    return result, controls


def test_v19_gate_requires_same_weight_feedback_and_rehearsal_gain():
    panel, controls = synthetic_panel()
    good = audit.gate(panel, controls, 'heldout')
    assert good['go'] is True
    assert good['conditions']['C19_minus_C18_at_most_minus_10']
    assert good['conditions']['C19_minus_D19_at_most_minus_20']
    assert good['conditions']['C19_minus_CF19_at_most_minus_10']
    no_feedback, controls = synthetic_panel(cf_cost=1000.)
    bad = audit.gate(no_feedback, controls, 'heldout')
    assert bad['go'] is False
    assert not bad['conditions']['C19_minus_CF19_at_most_minus_10']
    assert not bad['conditions']['C19_minus_CF19_cluster_upper_negative']
