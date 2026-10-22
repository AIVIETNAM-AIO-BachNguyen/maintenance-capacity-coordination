import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs, evaluate_episode
from maintenance_capacity_coordination.baselines import V14Controller

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "baselines/v14"
REFERENCE = json.loads((ASSETS / "reference_episodes.json").read_text())


@pytest.fixture(scope="session")
def original_runtime_reference():
    program = """
import json
import torch
from pathlib import Path
from ht_pdm_fjsp.maintenance_framework_v14_campaign import load_checkpoint, V14Controller
from ht_pdm_fjsp.maintenance_framework_v1 import configs, evaluate_episode
torch.set_num_threads(1)
assets = Path.cwd() / 'baselines/v14'
selection = json.loads((assets / 'selection.json').read_text())
rows = []
for arm, items in selection.items():
    for seed, item in items.items():
        actor = load_checkpoint(assets / item['checkpoint'])
        for case, config in configs().items():
            row = evaluate_episode(config, case, 72610000, V14Controller(actor, arm))
            row['train_seed'] = int(seed)
            rows.append(row)
print(json.dumps(rows))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "legacy/v14/src")
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                            check=True, capture_output=True, text=True, timeout=30)
    return {(row["controller"], row["train_seed"], row["case"]): row for row in json.loads(result.stdout)}


@pytest.mark.parametrize("expected", REFERENCE, ids=lambda row: f'{row["controller"]}/{row["train_seed"]}/{row["case"]}')
def test_selected_trained_actor_matches_isolated_original_on_current_host(expected, original_runtime_reference):
    torch.set_num_threads(1)
    selection = json.loads((ASSETS / "selection.json").read_text())
    item = selection[expected["controller"]][str(expected["train_seed"])]
    checkpoint = ASSETS / item["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == item["sha256"]
    controller = V14Controller.from_checkpoint(checkpoint)
    actual = evaluate_episode(configs()[expected["case"]], expected["case"], expected["seed"], controller)
    original = original_runtime_reference[(expected["controller"], expected["train_seed"], expected["case"])]
    for name in ("objective", "H_cost", "tail_cost", "H_availability", "H_failures",
                 "queued_pm_ticks", "queued_cm_ticks", "episode_return", "invalid_assignments"):
        assert actual[name] == pytest.approx(original[name], rel=0, abs=1e-9)
