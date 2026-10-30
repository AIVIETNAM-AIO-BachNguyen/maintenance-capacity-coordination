"""Focused checks for the locked v20 label-bank construction."""
from __future__ import annotations

from collections import Counter

from experiments import recent_policy_labels_v20 as labels


def test_stratified_selection_is_deterministic_and_balanced():
    pool = []
    case = labels.CASES[0]
    for root in labels.ROOTS:
        for arm in labels.SOURCE_ARMS:
            for window in range(2):
                for raw in range(30):
                    pool.append({'case': case, 'root': root, 'source_arm': arm,
                                 'window': window, 'raw_seed': raw, 'priority': raw,
                                 'state': object()})
    first = labels.selected_pool(pool, case, 273)
    second = labels.selected_pool(list(reversed(pool)), case, 273)
    assert [(r['root'], r['source_arm'], r['window'], r['raw_seed']) for r in first] == [
        (r['root'], r['source_arm'], r['window'], r['raw_seed']) for r in second]
    counts = Counter((r['root'], r['source_arm'], r['window']) for r in first)
    assert len(first) == 273 and len(counts) == 12
    assert max(counts.values()) - min(counts.values()) == 1
    assert len({(r['root'], r['source_arm'], r['window'], r['raw_seed']) for r in first}) == 273


def test_distinct_label_seed_roles():
    raw = labels.SMOKE_SEEDS[0]
    case = labels.CASES[0]
    root = labels.ROOTS[0]
    arm = labels.SOURCE_ARMS[0]
    seeds = {labels.state_seed(raw, case, root, arm, 0),
             labels.priority_seed(raw, case, root, arm, 0),
             labels.candidate_seed(raw, case, root, arm, 7),
             *[labels.bank_seed(raw, case, root, arm, 7, slot) for slot in range(16)]}
    assert len(seeds) == 19


def test_v20_locked_pilot_gate_requires_all_primary_effects():
    from experiments.audit_same_tick_v20 import gate
    from experiments import same_tick_pilot_v20 as pilot

    costs = {'C20': 1000, 'C19': 1015, 'D20': 1025, 'CF20': 1015,
             'D19': 1030, 'B18': 1050, 'S18': 1050, 'G18': 1050,
             'P18': 1050, 'Q18': 1200}
    data = [{'case': case, 'arm': arm, 'seed': seed, 'checkpoint': f'{root}:8192',
             'objective': cost}
            for case in pilot.CASES for root in pilot.TRAIN_ROOTS
            for seed in pilot.HELDOUT_SEEDS for arm, cost in costs.items()]
    controls = [{'case': case, 'seed': seed, 'arm': 'F_rollout', 'objective': 1000}
                for case in pilot.CASES for seed in pilot.HELDOUT_SEEDS]
    assert gate(data, controls, 'heldout')['go'] is True
    for row in data:
        if row['arm'] == 'C19':
            row['objective'] = 1000
    decision = gate(data, controls, 'heldout')
    assert decision['go'] is False
    assert decision['conditions']['C20_minus_C19_at_most_minus_10'] is False


def test_v20_same_weight_frozen_actor_identity_at_root():
    import torch
    from experiments.scaled_n5k3_v12_config import configs
    from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
    from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
    from maintenance_capacity_coordination.same_tick_policy_v15 import choose
    from experiments import same_tick_pilot_v20 as pilot

    torch.manual_seed(20)
    active = pilot.new_policy('C20')
    frozen = DynamicSequencePolicy('D16')
    frozen.load_state_dict(active.state_dict(), strict=True)
    assert all(torch.equal(value, frozen.state_dict()[key])
               for key, value in active.state_dict().items())
    env = ScaledCoordinationEnv(configs()[pilot.CASES[0]])
    env.reset(pilot.SMOKE_SEEDS[0])
    a, b = choose(active, env, True), choose(frozen, env, True)
    assert a['tokens'][:1] == b['tokens'][:1]
    assert env.matching_mask(a['submissions'])[a['matching_index']]
    assert env.matching_mask(b['submissions'])[b['matching_index']]
