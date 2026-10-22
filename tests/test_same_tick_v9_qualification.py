"""Independent small-graph checks for the locked v9 opportunity definition."""
from experiments.qualify_same_tick_v9 import opportunity


def test_opportunity_requires_two_residual_second_edges():
    # Three first edges use worker 0; worker 1 edge has three distinct-machine alternatives.
    legal, qualifying = opportunity([1, 0, 1, 0, 0, 1, 1, 0])
    assert legal == (0, 2, 5, 6)
    assert qualifying == (5,)


def test_opportunity_with_no_pair_or_one_residual():
    assert opportunity([1, 0, 1, 0, 0, 0, 0, 0]) == ((0, 2), ())
    assert opportunity([1, 0, 0, 1, 0, 0, 0, 0]) == ((0, 3), ())
