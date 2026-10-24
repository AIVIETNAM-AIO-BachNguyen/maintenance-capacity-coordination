"""Focused v19 rehearsal update tests on the unchanged canonical policy."""
from __future__ import annotations

import copy
import numpy as np
import torch

from experiments.full_matching_pretrain_v17 import listwise_loss, tensorize
from experiments.matching_value_rehearsal_v19 import LabelCycle, ppo_update
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    GLOBAL_DIM, LOCAL_DIM, STOP, SequenceTransition, ppo_update as original_ppo_update,
)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    policy = DynamicSequencePolicy('C16')
    machine = np.zeros((5, LOCAL_DIM), dtype=np.float32)
    machine[0, 14] = machine[1, 15] = 3.
    machine[0, 17] = machine[1, 18] = .5
    worker = np.zeros((3, 4), dtype=np.float32)
    mask = np.zeros(STOP, dtype=bool)
    mask[[0, 4]] = True
    pre = np.zeros(GLOBAL_DIM, dtype=np.float32)
    records = []
    for probabilities in ([.9, .1], [.1, .9]):
        records.append({'matching_indices': [0, 1], 'sequences': [[STOP], [0, 4]],
                        'target_probabilities': probabilities, 'machine_obs': machine.tolist(),
                        'worker_obs': worker.tolist(), 'edge_mask': mask.tolist()})
    labels = tensorize(records)
    transitions = []
    for i in range(8):
        with torch.no_grad():
            sequence, logprob, _, _ = policy.sample(torch.from_numpy(machine), torch.from_numpy(worker),
                                                     torch.from_numpy(mask), deterministic=False)
            value = float(policy.value(torch.from_numpy(pre)))
        transitions.append(SequenceTransition(pre, machine, worker, mask, sequence.numpy(),
                                               float(logprob), value, float(-1 - i), i == 7))
    return policy, labels, transitions


def test_label_cycle_uses_all_rows_without_consuming_torch_randomness():
    cycle = LabelCycle(5, 37)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first = cycle.take(5)
    second = cycle.take(5)
    assert sorted(first.tolist()) == sorted(second.tolist()) == list(range(5))
    assert torch.equal(before, torch.get_rng_state())


def test_zero_rehearsal_is_exact_original_ppo_update():
    policy, labels, transitions = fixture()
    reference = copy.deepcopy(policy)
    wrapped = copy.deepcopy(policy)
    opt_ref = torch.optim.Adam(reference.parameters(), lr=3e-4)
    opt_wrap = torch.optim.Adam(wrapped.parameters(), lr=3e-4)
    torch.manual_seed(199)
    expected = original_ppo_update(reference, opt_ref, transitions, 0., epochs=1, minibatch=4)
    torch.manual_seed(199)
    actual = ppo_update(wrapped, opt_wrap, transitions, 0., labels, LabelCycle(2, 1),
                        weight=0., epochs=1, minibatch=4)
    assert actual == expected
    assert all(torch.equal(value, wrapped.state_dict()[key])
               for key, value in reference.state_dict().items())


def test_rehearsal_changes_actor_gradient_and_logs_label_rows():
    policy, labels, transitions = fixture()
    loss = listwise_loss(policy, labels)
    assert torch.isfinite(loss)
    critic_grad = torch.autograd.grad(loss, tuple(policy.critic.parameters()), allow_unused=True)
    assert all(gradient is None for gradient in critic_grad)
    reference = copy.deepcopy(policy)
    rehearsed = copy.deepcopy(policy)
    opt_ref = torch.optim.Adam(reference.parameters(), lr=3e-4)
    opt_rehearsed = torch.optim.Adam(rehearsed.parameters(), lr=3e-4)
    torch.manual_seed(199)
    ppo_update(reference, opt_ref, transitions, 0., labels, LabelCycle(2, 1),
               weight=0., epochs=1, minibatch=4)
    logged = []
    torch.manual_seed(199)
    result = ppo_update(rehearsed, opt_rehearsed, transitions, 0., labels, LabelCycle(2, 1),
                        weight=.5, label_batch_size=2, epochs=1, minibatch=4,
                        on_minibatch=logged.append)
    assert result['rehearsal_minibatches'] == len(logged) == 2
    assert all(sorted(row['label_row_indices']) == [0, 1] for row in logged)
    assert result['rehearsal_candidate_paths'] == 8
    assert any(not torch.equal(value, rehearsed.actor.state_dict()[key])
               for key, value in reference.actor.state_dict().items())
