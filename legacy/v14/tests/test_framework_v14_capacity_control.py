"""V14 equal-capacity identity-control and campaign contracts."""
import torch

from ht_pdm_fjsp.maintenance_framework_v14_policy import ACTIONS, OpportunityCostActor
from ht_pdm_fjsp.maintenance_framework_v14_campaign import ARMS, VERSION, masked_probabilities, seed_registry


def observation():
    value = torch.zeros((1, 56), dtype=torch.float32)
    value[:, 2:34] = torch.linspace(.05, .8, 32)
    value[:, 34] = value[:, 37] = 1.
    value[:, 40:56:2] = 1.
    value[:, 41:56:2] = .5
    return value


def matched():
    torch.manual_seed(72600000)
    left = OpportunityCostActor('competition')
    right = OpportunityCostActor('count_calibrated')
    right.load_state_dict(left.state_dict())
    return left, right


def test_registry_initial_tensors_and_all_parameters_active():
    registry, (train, dev, smoke) = seed_registry()
    assert registry['campaign_version'] == VERSION
    assert ARMS == ('competition', 'count_calibrated')
    assert len(train) == 5 and len(dev) == 16 and len(smoke) == 2
    assert train[0] == 72600000 and dev[0] == 72610000
    left, right = matched()
    assert all(torch.equal(left.state_dict()[key], right.state_dict()[key])
               for key in left.state_dict())
    for actor in (left, right):
        assert sum(p.numel() for p in actor.parameters()) == 3219
        actor(observation()).sum().backward()
        assert sum(p.numel() for p in actor.parameters()
                   if p.grad is not None and bool(torch.any(p.grad != 0))) == 3219


def test_one_service_and_uniform_demand_actions_match():
    left, right = matched()
    x = observation()
    l, lp, _ = left.components(x)
    r, rp, _ = right.components(x)
    for index, action in enumerate(ACTIONS):
        if len(action) <= 1:
            assert torch.allclose(lp[:, index], rp[:, index], atol=1e-6)
            assert torch.allclose(l[:, index], r[:, index], atol=1e-6)
    with torch.no_grad():
        for actor in (left, right):
            actor.demand_head[2].weight.zero_()
            actor.demand_head[2].bias.zero_()
    l, lp, _ = left.components(x)
    r, rp, _ = right.components(x)
    assert torch.allclose(lp, rp, atol=1e-6)
    assert torch.allclose(l, r, atol=1e-6)


def test_other_served_machine_identity_changes_joint_penalty():
    left, right = matched()
    with torch.no_grad():
        for actor in (left, right):
            for parameter in actor.parameters():
                parameter.zero_()
            actor.machine_encoder[0].weight[0, 0] = 1.
            actor.demand_head[0].weight[0, 0] = 4.
            actor.demand_head[2].weight[0, 0] = 3.
    x = observation()
    x[:, 2:34] = 0.
    x[:, 2 + 2*8] = 1.  # machine 2 has high exposed demand
    _, joint, demand = left.components(x)
    _, count, _ = right.components(x)
    serve_01 = ACTIONS.index(((0, 0), (1, 1)))
    serve_02 = ACTIONS.index(((0, 0), (2, 1)))
    assert demand[0, 2, 0] > demand[0, 1, 0]
    assert joint[0, serve_01] > count[0, serve_01]
    assert joint[0, serve_02] < count[0, serve_02]
    assert not torch.allclose(joint, count)


def test_legal_mask_and_finite_gradients():
    left, right = matched()
    x = observation()
    legal = [ACTIONS.index(()), ACTIONS.index(((0, 0),))]
    for actor in (left, right):
        probs = masked_probabilities(actor, x.numpy()[0], legal)
        assert abs(probs.sum() - 1.) < 1e-6
        assert all(probs[i] == 0 for i in range(len(ACTIONS)) if i not in legal)
        actor.zero_grad()
        logits, penalty, demand = actor.components(x)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(penalty).all()
        assert torch.isfinite(demand).all()
        logits.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in actor.parameters())
