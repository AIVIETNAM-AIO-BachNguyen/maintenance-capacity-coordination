"""Fixed pre-existing N5/K3 sparse family configurations for engineering work."""
from dataclasses import replace

from ht_pdm_fjsp.maintenance_dispatch import family_config
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL

CASES = ("n5k3_residual_a", "n5k3_residual_b")
CONFIG_SEEDS = {"n5k3_residual_a": 74740000, "n5k3_residual_b": 74740001}


def configs():
    return {case: replace(family_config("n5_k3_sparse", seed, "full"),
                          horizon=HORIZON + TAIL)
            for case, seed in CONFIG_SEEDS.items()}


def evaluate_controls(run, seeds, *, panel):
    """Evaluate present-state F1/F2 on the same N5/K3 physical tapes."""
    import json
    import time
    from tqdm.auto import tqdm
    from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController
    from maintenance_capacity_coordination.scaled_coordination_v12 import ScaledCoordinationEnv
    from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
    from ht_pdm_fjsp.maintenance_dispatch import role_seed

    rows = []
    destination = run / f"{panel}_controls.jsonl"
    trace = run / f"{panel}_control_decisions.jsonl"
    for case in CASES:
        for name in ("risk_matching", "rollout"):
            for seed in tqdm(seeds, desc=f"v12 control {case} {name}"):
                env = ScaledCoordinationEnv(configs()[case])
                env.reset(seed)
                controller = RuleController(name)
                selection_seconds = 0.
                for tick in range(HORIZON):
                    started = time.monotonic()
                    submissions, matching = controller.choose(env.config, env.inner.state, tick,
                                                              case=case, seed=seed)
                    selection_seconds += time.monotonic() - started
                    index = env.actions.index(matching)
                    reward, done = env.step(submissions, index)
                    with trace.open("a") as stream:
                        stream.write(json.dumps({"case": case, "arm": "F_" + name,
                                                 "seed": seed, "tick": tick,
                                                 "submissions": submissions,
                                                 "matching_index": index, "reward": reward,
                                                 "done": done}, sort_keys=True, allow_nan=False) + "\n")
                assert done
                row = {"case": case, "arm": "F_" + name, "seed": seed,
                       "panel": panel, "physical_seed": role_seed(seed, BASE_VERSION + ":physical"),
                       "selection_seconds": selection_seconds,
                       "forecast_transitions": controller.forecast_transitions,
                       **env.metrics()}
                rows.append(row)
                with destination.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return rows
