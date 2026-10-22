"""Fixed pre-existing N5/K3 sparse family configurations for engineering work."""
from dataclasses import replace

from ht_pdm_fjsp.maintenance_dispatch import family_config
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL

CASES = ("n5k3_sparse_a", "n5k3_sparse_b")
CONFIG_SEEDS = {"n5k3_sparse_a": 73740000, "n5k3_sparse_b": 73740001}


def configs():
    return {case: replace(family_config("n5_k3_sparse", seed, "full"),
                          horizon=HORIZON + TAIL)
            for case, seed in CONFIG_SEEDS.items()}
