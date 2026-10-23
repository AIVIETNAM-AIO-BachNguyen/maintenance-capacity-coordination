"""Conditional economic target and matched pretraining contracts."""
import torch

from experiments.conditional_value_pretrain_v15 import candidate_logits, listwise_loss, pretrain, tensorize
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, ACTIONS, STOP, SequencePolicy
from maintenance_capacity_coordination.scaled_matching_policy_v15 import MatchingPolicy

def _records():
    torch.manual_seed(1234)
    machine = torch.randn(5, 24).tolist()
    worker = torch.randn(3, 4).tolist()
    out = []
    for first, seconds, targets in ((0, (4, 8), (.7, .3)), (1, (3, 8), (.4, .6))):
        matching = [ACTIONS.index(tuple(sorted((EDGES[first], EDGES[second])))) for second in seconds]
        out.append({"machine_obs": machine, "worker_obs": worker, "edge_mask": [True] * STOP,
                    "first_edge": first, "second_edges": seconds, "matching_indices": matching,
                    "target_probabilities": targets})
    return out

def test_all_pretrained_arms_use_same_records_with_finite_scores():
    data = tensorize(_records())
    for arm in ("C15", "D15", "B15", "S15", "G15"):
        policy = SequencePolicy(arm) if arm in ("C15", "D15", "B15") else MatchingPolicy(arm)
        scores = candidate_logits(policy, data)
        assert scores.shape == data["targets"].shape
        assert torch.isfinite(scores[data["candidate_valid"]]).all()
        assert torch.isfinite(listwise_loss(policy, data))


def test_c_d_have_same_root_weights_but_conditional_second_logits_differ():
    data = tensorize(_records())
    c, d = SequencePolicy("C15"), SequencePolicy("D15")
    d.load_state_dict(c.state_dict())
    root = c.actor.encode(data["machine"], data["worker"])
    z_m = torch.zeros((len(_records()), 5), dtype=torch.bool)
    z_w = torch.zeros((len(_records()), 3), dtype=torch.bool)
    z_e = torch.zeros((len(_records()), 15), dtype=torch.bool)
    a, _, _ = c.actor.logits(root, data["edge_mask"], z_m, z_w, z_e)
    b, _, _ = d.actor.logits(root, data["edge_mask"], z_m, z_w, z_e)
    assert torch.allclose(a, b)
    assert not torch.allclose(candidate_logits(c, data), candidate_logits(d, data))


def test_auxiliary_training_changes_actor_and_leaves_critic_untouched():
    records = _records()
    for arm in ("C15", "D15", "B15", "S15", "G15"):
        torch.manual_seed(123)
        policy = SequencePolicy(arm) if arm in ("C15", "D15", "B15") else MatchingPolicy(arm)
        before_actor = [x.clone() for x in policy.actor.parameters()]
        before_critic = [x.clone() for x in policy.critic.parameters()]
        result = pretrain(policy, records, 123, epochs=2, batch_size=2)
        assert result == {"pretraining_records": 2, "pretraining_epochs": 2, "pretraining_updates": 2}
        assert any(not torch.equal(x, y) for x, y in zip(before_actor, policy.actor.parameters()))
        assert all(torch.equal(x, y) for x, y in zip(before_critic, policy.critic.parameters()))
