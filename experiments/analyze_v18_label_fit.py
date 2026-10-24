"""Descriptive in-sample v18 matching-label fit; no selection or gate change."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments import same_tick_pilot_v18 as pilot
from experiments.full_matching_pretrain_v17 import tensorize, candidate_logits, listwise_loss

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / 'artifacts/full_matching_v18_labels_20261024T035233500615Z'
TRAIN = ROOT / 'artifacts/same_tick_v18_train_20261024T045926267434Z'
HELDOUT = ROOT / 'artifacts/same_tick_v18_heldout_20261024T090154402501Z'


def main():
    torch.set_num_threads(1)
    for run in (LABEL, TRAIN, HELDOUT):
        assert json.loads((run / 'manifest.json').read_text())['status'] == 'COMPLETED'
        assert json.loads((run / 'audit.json').read_text())['status'] == 'PASS'
    assert json.loads((HELDOUT / 'replication_go_decision.json').read_text())['go'] is False
    collection = json.loads((TRAIN / 'manifest.json').read_text())
    assert collection['source_sha256'] == json.loads((HELDOUT / 'manifest.json').read_text())['source_sha256']
    for path, expected in collection['source_sha256'].items():
        assert pilot.sha256(ROOT / path) == expected
    labels = [json.loads(x) for x in (LABEL / 'labels.jsonl').read_text().splitlines()]
    output = ROOT / 'artifacts' / ('v18_label_fit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    status = 'FAILED'
    records = []
    baselines = {}
    try:
        for case in pilot.CASES:
            case_rows = [row for row in labels if row['case'] == case]
            variants = {
                'sixteen_tape': tensorize(case_rows),
                'four_tape': tensorize([{**row, 'target_probabilities': row['old_4_target_probabilities']}
                                       for row in case_rows]),
            }
            baselines[case] = {}
            for variant, data in variants.items():
                target = data['targets']
                baselines[case][variant] = {
                    'target_entropy': float(-(target * target.clamp_min(1e-12).log()).sum(1).mean()),
                    'uniform_cross_entropy': float(np.mean([math.log(len(row['matching_indices'])) for row in case_rows])),
                }
            for arm in tqdm(pilot.ARMS, desc=f'v18 label fit {case}', leave=False):
                for root in pilot.TRAIN_ROOTS:
                    trial = TRAIN / f'{case}_{arm}_{root}'
                    selection = json.loads((trial / 'selection.json').read_text())
                    for stage, checkpoint in (
                        ('pretrain', trial / 'pretrain_checkpoint.pt'),
                        ('selected', trial / selection['checkpoint']),
                    ):
                        policy = pilot.load_policy(checkpoint)
                        targets = variants if arm in ('C18', 'L18') else {'sixteen_tape': variants['sixteen_tape']}
                        for variant, data in targets.items():
                            with torch.no_grad():
                                scores = candidate_logits(policy, data)
                                loss = float(listwise_loss(policy, data))
                                top1 = float((scores.argmax(1) == data['targets'].argmax(1)).float().mean())
                            row = {
                                'case': case, 'arm': arm, 'train_root': root, 'stage': stage,
                                'target_variant': variant, 'checkpoint_sha256': pilot.sha256(checkpoint),
                                'training_records': len(case_rows), 'listwise_cross_entropy': loss,
                                'top1_match_rate': top1, **baselines[case][variant],
                            }
                            pilot.append_jsonl(output / 'rows.jsonl', row)
                            records.append(row)
        summary = {'baselines': {variant: {
            metric: float(np.mean([baselines[case][variant][metric] for case in pilot.CASES]))
            for metric in ('target_entropy', 'uniform_cross_entropy')}
            for variant in ('sixteen_tape', 'four_tape')}, 'arms': {}}
        for arm in pilot.ARMS:
            summary['arms'][arm] = {}
            for variant in (('sixteen_tape', 'four_tape') if arm in ('C18', 'L18') else ('sixteen_tape',)):
                summary['arms'][arm][variant] = {}
                for stage in ('pretrain', 'selected'):
                    values = [r for r in records if r['arm'] == arm and r['target_variant'] == variant and r['stage'] == stage]
                    summary['arms'][arm][variant][stage] = {
                        'mean_cross_entropy': float(np.mean([r['listwise_cross_entropy'] for r in values])),
                        'mean_top1_match_rate': float(np.mean([r['top1_match_rate'] for r in values])),
                    }
        pilot.write_json(output / 'summary.json', summary)
        status = 'COMPLETED'
    finally:
        pilot.write_json(output / 'manifest.json', {
            'status': status, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started,
            'analysis_source_sha256': pilot.sha256(Path(__file__)),
            'label_manifest_sha256': pilot.sha256(LABEL / 'manifest.json'),
            'label_audit_sha256': pilot.sha256(LABEL / 'audit.json'),
            'train_manifest_sha256': pilot.sha256(TRAIN / 'manifest.json'),
            'train_audit_sha256': pilot.sha256(TRAIN / 'audit.json'),
            'heldout_manifest_sha256': pilot.sha256(HELDOUT / 'manifest.json'),
            'heldout_audit_sha256': pilot.sha256(HELDOUT / 'audit.json'),
            'protocol_sha256': pilot.sha256(pilot.PROTOCOL),
            'interpretation': 'in-sample training-label fit only; no generalization, policy selection or gate change',
            'output_sha256': {p.name: pilot.sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'},
        })
    print(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
