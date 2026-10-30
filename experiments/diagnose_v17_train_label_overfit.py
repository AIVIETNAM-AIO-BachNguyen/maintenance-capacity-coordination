"""Bounded in-sample capacity check on the audited v17 simulator label bank.

Uses only 32 preselected training rows, never DEV, held-out or TEST. This is
an engineering diagnostic, not a new policy evaluation or model selection.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments import same_tick_pilot_v17 as pilot
from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss, tensorize

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / 'artifacts/full_matching_v17_labels_20261023T080338842451Z'
SEED = 79714000
RECORDS_PER_CASE = 16
UPDATES = 512
MILESTONES = (0, 32, 64, 128, 256, 512)
ARMS = ('C17', 'G17')


def measurement(policy, data):
    with torch.no_grad():
        logits = candidate_logits(policy, data)
        return {
            'cross_entropy': float(listwise_loss(policy, data)),
            'top1_match_rate': float((logits.argmax(1) == data['targets'].argmax(1)).float().mean()),
        }


def main():
    torch.set_num_threads(1)
    label_manifest = json.loads((LABEL / 'manifest.json').read_text())
    label_audit = json.loads((LABEL / 'audit.json').read_text())
    assert label_manifest['status'] == 'COMPLETED' and label_audit['status'] == 'PASS'
    for path, expected in label_manifest['source_sha256'].items():
        assert pilot.sha256(ROOT / path) == expected
    rows = [json.loads(line) for line in (LABEL / 'labels.jsonl').open()]
    selected = []
    selected_indices = {}
    for case in pilot.CASES:
        case_rows = [row for row in rows if row['case'] == case]
        indices = np.linspace(0, len(case_rows) - 1, RECORDS_PER_CASE, dtype=int).tolist()
        selected.extend(case_rows[i] for i in indices)
        selected_indices[case] = indices
    data = tensorize(selected)
    entropy = float((-(data['targets'] * data['targets'].clamp_min(1e-12).log()).sum(1)).mean())
    uniform_ce = float(np.mean([math.log(len(row['matching_indices'])) for row in selected]))
    run = ROOT / 'artifacts' / ('v17_train_label_overfit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(exist_ok=False)
    started = time.monotonic()
    status = 'FAILED'
    try:
        results = {}
        for arm in ARMS:
            torch.manual_seed(SEED)
            policy = pilot.new_policy(arm)
            optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
            curve = {'0': measurement(policy, data)}
            for step in tqdm(range(1, UPDATES + 1), desc=f'v17 train-only overfit {arm}', leave=False):
                loss = listwise_loss(policy, data)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss {arm} step {step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
                optimizer.step()
                if step in MILESTONES:
                    curve[str(step)] = measurement(policy, data)
            results[arm] = curve
            pilot.write_json(run / 'partial_results.json', results)
        pilot.write_json(run / 'summary.json', {
            'interpretation': 'training-only overfit capacity check; no physical policy evaluation, selection or gate change',
            'arms': results,
            'training_rows': len(selected),
            'indices_by_case': selected_indices,
            'target_entropy': entropy,
            'uniform_cross_entropy': uniform_ce,
            'optimizer': {'updates': UPDATES, 'adam_lr': 3e-4, 'gradient_clip': .5, 'batch_size': len(selected)},
        })
        status = 'COMPLETED'
    finally:
        pilot.write_json(run / 'manifest.json', {
            'status': status,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started,
            'label_manifest_sha256': pilot.sha256(LABEL / 'manifest.json'),
            'label_audit_sha256': pilot.sha256(LABEL / 'audit.json'),
            'labels_sha256': pilot.sha256(LABEL / 'labels.jsonl'),
            'analysis_source_sha256': pilot.sha256(Path(__file__)),
            'protocol_sha256': pilot.sha256(pilot.PROTOCOL),
            'seed': SEED, 'arms': ARMS, 'updates': UPDATES,
            'output_sha256': {p.name: pilot.sha256(p) for p in run.iterdir() if p.is_file() and p.name != 'manifest.json'},
        })
    print(run.resolve())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
