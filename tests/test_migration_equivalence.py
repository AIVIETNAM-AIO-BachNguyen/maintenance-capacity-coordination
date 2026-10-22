import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, RuleController, configs, evaluate_episode
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from maintenance_capacity_coordination.baselines import V14Controller

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "provenance/reference_fixtures.json").read_text())


def test_imported_module_bytes_are_unchanged():
    manifest = json.loads((ROOT / "provenance/import_manifest.json").read_text())
    for record in manifest["imports"]:
        if record["destination_path"].startswith("src/"):
            actual = hashlib.sha256((ROOT / record["destination_path"]).read_bytes()).hexdigest()
            assert actual == record["source_sha256"]


@pytest.mark.parametrize("expected", REFERENCE["episodes"], ids=lambda row: row["case"] + "/" + row["controller"])
def test_original_rule_episode_metrics(expected):
    actual = evaluate_episode(configs()[expected["case"]], expected["case"], REFERENCE["seed"],
                              RuleController(expected["controller"]))
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-9)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("expected", REFERENCE["actors"], ids=lambda row: row["case"] + "/" + row["mode"])
def test_original_actor_observation_and_logits(expected):
    torch.set_num_threads(1)
    torch.manual_seed(REFERENCE["seed"])
    actor = OpportunityCostActor(expected["mode"])
    env = FrameworkEnv(configs()[expected["case"]])
    observation, _ = env.reset(seed=REFERENCE["seed"])
    np.testing.assert_array_equal(observation, np.asarray(expected["observation"], dtype=np.float32))
    with torch.no_grad():
        actual = actor(torch.from_numpy(observation).reshape(1, 56))[0].numpy()
    np.testing.assert_allclose(actual, expected["logits"], rtol=1e-6, atol=1e-6)


def test_v14_checkpoint_controller_selects_only_legal_actions(tmp_path):
    torch.manual_seed(REFERENCE["seed"])
    actor = OpportunityCostActor("count_calibrated")
    checkpoint = tmp_path / "actor.pt"
    torch.save({"metadata": {"arm": actor.mode}, "state_dict": actor.state_dict()}, checkpoint)
    controller = V14Controller.from_checkpoint(checkpoint)
    for case, config in configs().items():
        result = evaluate_episode(config, case, REFERENCE["seed"], controller)
        assert result["invalid_assignments"] == 0
        assert result["episode_return"] == -result["objective"]


def test_v14_retains_active_capacity_and_finite_gradients():
    actor = OpportunityCostActor("count_calibrated")
    observation = torch.tensor(REFERENCE["actors"][0]["observation"]).reshape(1, 56)
    logits = actor(observation)
    logits.sum().backward()
    assert sum(parameter.numel() for parameter in actor.parameters()) == 3219
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in actor.parameters())
