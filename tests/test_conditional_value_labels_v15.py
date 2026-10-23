"""Prelocked v15 conditional-label selection and target contracts."""
import numpy as np

from experiments import conditional_value_labels_v15 as labels
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, STOP


def test_canonical_first_options_match_explicit_successors():
    full = np.ones(STOP, dtype=bool)
    options = labels.first_options(full)
    assert options
    for first, seconds in options:
        m0, j0 = EDGES[first]
        explicit = tuple(second for second in range(first + 1, STOP)
                         if EDGES[second][0] != m0 and EDGES[second][1] != j0)
        assert seconds == explicit
        assert len(seconds) >= 2
    empty = np.zeros(STOP, dtype=bool)
    assert labels.first_options(empty) == ()


def test_label_seed_roles_and_future_tapes_are_distinct():
    labels.seed_registry_check("smoke")
    case, root = labels.CASES[0], labels.C12_ROOTS[0]
    raw = labels.SMOKE_SEEDS[0]
    seeds = [labels.bank_seed(raw, case, root, tick, slot)
             for tick in (8, 9) for slot in range(4)]
    assert len(seeds) == len(set(seeds))
    assert labels.first_role_seed(raw, case, root) not in seeds


def test_label_target_is_invariant_to_common_cost_shift():
    costs = np.array([101., 115., 130.])
    target = np.exp(-(costs - costs.min()) / 20.)
    target /= target.sum()
    shifted = costs + 500.
    target_shifted = np.exp(-(shifted - shifted.min()) / 20.)
    target_shifted /= target_shifted.sum()
    assert np.allclose(target, target_shifted)
    assert np.isclose(target.sum(), 1.)
    assert np.argmax(target) == np.argmin(costs)
