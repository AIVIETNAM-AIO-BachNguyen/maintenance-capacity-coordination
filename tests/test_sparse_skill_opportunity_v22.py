from ht_pdm_fjsp.maintenance_dispatch import family_config
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES

from experiments.sparse_skill_opportunity_v22 import CASES, CONFIG_SEEDS, SKILL_BITS, configs, structure


def test_sparse_skill_config_changes_only_compatibility_and_preserves_coverage():
    for case in CASES:
        original = family_config('n5_k3_sparse', CONFIG_SEEDS[case], 'full')
        changed = configs()[case]
        assert changed.machines == original.machines == 5
        assert changed.technicians == original.technicians == 3
        assert changed.failure_age == original.failure_age
        assert changed.failure_probability == original.failure_probability
        assert changed.restoration == original.restoration
        assert changed.preventive_cost == original.preventive_cost
        assert changed.corrective_cost == original.corrective_cost
        assert changed.unavailable_cost == original.unavailable_cost
        assert changed.failure_cost == original.failure_cost
        assert changed.horizon == 64
        assert sum(changed.service_time[m][w] > 0 for m, w in EDGES) == 8
        for i, (m, w) in enumerate(EDGES):
            expected = (original.service_time[m][w] or 3) if SKILL_BITS[case] & (1 << i) else 0
            assert changed.service_time[m][w] == expected
        assert all(any(changed.service_time[m][w] > 0 for w in range(3)) for m in range(5))
        assert all(any(changed.service_time[m][w] > 0 for m in range(5)) for w in range(3))


def test_physical_same_count_capacity_contrast_is_computed_without_order_filter():
    for case in CASES:
        mask = [bool(SKILL_BITS[case] & (1 << i)) for i in range(len(EDGES))]
        result = structure(mask)
        assert result['qualifying']
        assert result['witnesses']
        for left, right, count in result['witnesses']:
            a, b = result['choices'][[x['first_edge'] for x in result['choices']].index(left)], result['choices'][[x['first_edge'] for x in result['choices']].index(right)]
            assert a['physical_count'] == b['physical_count'] == count >= 2
            assert {a['physical_capacity'], b['physical_capacity']} == {1, 2}
