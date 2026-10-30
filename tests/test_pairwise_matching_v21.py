"""Focused arithmetic and optimizer checks for the prelocked v21 loss."""
import math

import numpy as np
import pytest
import torch

from experiments.full_matching_pretrain_v17 import pretrain as listwise_pretrain
from experiments.pairwise_matching_v21 import (
    pairwise_loss_from_scores, pretrain, tensorize,
)
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy


def row(costs=(10., 20., 20.), varying=False):
    branches = []
    for index, mean in enumerate(costs):
        h = [mean + (t - 7.5) * (index if varying else 0) for t in range(16)]
        branches.append({'matching_index': index, 'H_costs': h, 'tail_costs': [0.] * 16})
    return {
        'machine_obs': np.zeros((5, 24)).tolist(),
        'worker_obs': np.zeros((3, 4)).tolist(),
        'edge_mask': [True, True] + [False] * 13,
        'matching_indices': list(range(len(costs))),
        'sequences': [[15], [0], [1]][:len(costs)],
        'target_probabilities': [1 / len(costs)] * len(costs),
        'branches': branches,
    }


def test_paired_tape_target_se_weight_and_uniform_scale():
    data = tensorize([row(varying=True)])
    delta = -10.
    paired = np.asarray([10.] * 16) - np.asarray([20. + (t - 7.5) for t in range(16)])
    se = paired.std(ddof=1) / 4
    assert float(data['pair_targets'][0, 0, 1]) == pytest.approx(1 / (1 + math.exp(delta / 20)))
    assert float(data['pair_weights'][0, 0, 1]) == pytest.approx(10 / (10 + 2 * se + 1e-8))
    assert float(data['pair_weights'][0, 1, 2]) == pytest.approx(0.)
    scores = torch.zeros((1, 3), requires_grad=True)
    loss = pairwise_loss_from_scores(scores, data)
    assert float(loss.detach()) == pytest.approx(math.log(3), abs=1e-6)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_zero_gap_row_falls_back_to_equal_weights_and_rejects_bad_tapes():
    from experiments.audit_same_tick_v21 import audit_pair_targets
    data = tensorize([row((12., 12., 12.))])
    audit_pair_targets([row((12., 12., 12.))])
    assert int(data['pair_valid'].sum()) == 3
    assert int((data['pair_weights'] > 0).sum()) == 3
    assert float(pairwise_loss_from_scores(torch.zeros((1, 3)), data)) == pytest.approx(math.log(3), abs=1e-6)
    malformed = row()
    malformed['branches'][0]['H_costs'] = [1.] * 15
    with pytest.raises(ValueError, match='16 finite paired tapes'):
        tensorize([malformed])


def test_listwise_control_matches_existing_pretraining_update():
    records = [row((10., 20., 30.)), row((18., 22., 27.))]
    torch.manual_seed(82800000)
    old = DynamicSequencePolicy('C16')
    control = DynamicSequencePolicy('C16')
    control.load_state_dict(old.state_dict(), strict=True)
    listwise_pretrain(old, records, 82800001, epochs=1, batch_size=2)
    pretrain(control, records, 82800001, objective='listwise', epochs=1, batch_size=2)
    for key, value in old.state_dict().items():
        assert torch.equal(value, control.state_dict()[key]), key


def test_pairwise_ppo_keeps_sequence_ratio_and_actor_only_supervision():
    import copy
    from experiments.pairwise_matching_v21 import LabelCycle, pairwise_loss, ppo_update
    from maintenance_capacity_coordination.same_tick_policy_v15 import (
        GLOBAL_DIM, SequenceTransition, ppo_update as original_ppo_update,
    )
    torch.set_num_threads(1)
    torch.manual_seed(82800003)
    policy = DynamicSequencePolicy('C16')
    records = [row((10., 30., 20.)), row((25., 15., 35.))]
    labels = tensorize(records)
    supervised = pairwise_loss(policy, labels)
    assert torch.isfinite(supervised)
    assert all(gradient is None for gradient in torch.autograd.grad(
        supervised, tuple(policy.critic.parameters()), allow_unused=True))
    machine = np.zeros((5, 24), dtype=np.float32)
    worker = np.zeros((3, 4), dtype=np.float32)
    mask = np.asarray(records[0]['edge_mask'], dtype=bool)
    pre = np.zeros(GLOBAL_DIM, dtype=np.float32)
    transitions = []
    for i in range(4):
        with torch.no_grad():
            sequence, logprob, _, _ = policy.sample(torch.from_numpy(machine), torch.from_numpy(worker),
                                                     torch.from_numpy(mask), deterministic=False)
            value = float(policy.value(torch.from_numpy(pre)))
        replay, _ = policy.logprob_entropy(torch.from_numpy(machine), torch.from_numpy(worker),
                                           torch.from_numpy(mask), sequence)
        assert torch.allclose(logprob, replay[0], atol=1e-6)
        transitions.append(SequenceTransition(pre, machine, worker, mask, sequence.numpy(),
                                               float(logprob), value, float(-1 - i), i == 3))
    old, zero, pair = (copy.deepcopy(policy) for _ in range(3))
    old_opt = torch.optim.Adam(old.parameters(), lr=3e-4)
    zero_opt = torch.optim.Adam(zero.parameters(), lr=3e-4)
    pair_opt = torch.optim.Adam(pair.parameters(), lr=3e-4)
    torch.manual_seed(82800004)
    expected = original_ppo_update(old, old_opt, transitions, 0., epochs=1, minibatch=4)
    torch.manual_seed(82800004)
    actual = ppo_update(zero, zero_opt, transitions, 0., labels, LabelCycle(2, 1),
                        weight=0., epochs=1, minibatch=4)
    assert expected == actual
    assert all(torch.equal(value, zero.state_dict()[key]) for key, value in old.state_dict().items())
    logged = []
    torch.manual_seed(82800004)
    result = ppo_update(pair, pair_opt, transitions, 0., labels, LabelCycle(2, 1),
                        weight=.5, objective='pairwise', epochs=1, minibatch=4,
                        label_batch_size=2, on_minibatch=logged.append)
    assert result['rehearsal_minibatches'] == len(logged) == 1
    assert result['rehearsal_candidate_paths'] == 6
    assert np.isfinite(result['sequence_ratio_abs_deviation_mean'])
    assert any(not torch.equal(value, pair.actor.state_dict()[key])
               for key, value in zero.actor.state_dict().items())


def test_v21_locked_gate_and_same_weight_root_identity():
    from experiments.audit_same_tick_v21 import gate
    from experiments import same_tick_pilot_v21 as pilot
    from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
    costs = {'C21': 1000, 'L21': 1015, 'D21': 1025, 'CF21': 1015,
             'C20': 1030, 'C19': 1030, 'D19': 1030,
             'B18': 1050, 'S18': 1050, 'G18': 1050, 'P18': 1050, 'Q18': 1200}
    data = [{'case': case, 'arm': arm, 'seed': seed, 'checkpoint': f'{root}:8192',
             'objective': cost}
            for case in pilot.CASES for root in pilot.TRAIN_ROOTS
            for seed in pilot.HELDOUT_SEEDS for arm, cost in costs.items()]
    controls = [{'case': case, 'seed': seed, 'arm': 'F_rollout', 'objective': 1000}
                for case in pilot.CASES for seed in pilot.HELDOUT_SEEDS]
    assert gate(data, controls, 'heldout')['go'] is True
    for entry in data:
        if entry['arm'] == 'L21':
            entry['objective'] = 1000
    decision = gate(data, controls, 'heldout')
    assert decision['go'] is False
    assert decision['conditions']['C21_minus_L21_at_most_minus_10'] is False
    active = pilot.new_policy('C21')
    frozen = DynamicSequencePolicy('D16')
    frozen.load_state_dict(active.state_dict(), strict=True)
    assert all(torch.equal(value, frozen.state_dict()[key]) for key, value in active.state_dict().items())
