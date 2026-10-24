"""Describe v18 training-label fit at every immutable PPO milestone.

This reads only audited training labels, audited train/DEV checkpoints and DEV
selection data. It does not select a model, tune a gate or evaluate held-out.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments import same_tick_pilot_v18 as pilot
from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss, tensorize

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / 'artifacts/full_matching_v18_labels_20261024T035233500615Z'
TRAIN = ROOT / 'artifacts/same_tick_v18_train_20261024T045926267434Z'
ARMS = ('C18', 'D18', 'L18')
STAGES = ('pretrain', 'checkpoint_2048.pt', 'checkpoint_4096.pt', 'checkpoint_8192.pt')


def main():
    torch.set_num_threads(1)
    label_manifest = json.loads((LABEL / 'manifest.json').read_text())
    train_manifest = json.loads((TRAIN / 'manifest.json').read_text())
    assert label_manifest['status'] == train_manifest['status'] == 'COMPLETED'
    assert json.loads((LABEL / 'audit.json').read_text())['status'] == 'PASS'
    assert json.loads((TRAIN / 'audit.json').read_text())['status'] == 'PASS'
    assert train_manifest['label_manifest_sha256'] == pilot.sha256(LABEL / 'manifest.json')
    assert train_manifest['label_audit_sha256'] == pilot.sha256(LABEL / 'audit.json')
    for path, expected in train_manifest['source_sha256'].items():
        assert pilot.sha256(ROOT / path) == expected, path
    labels = [json.loads(line) for line in (LABEL / 'labels.jsonl').read_text().splitlines()]
    output = ROOT / 'artifacts' / ('v18_checkpoint_path_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    status = 'FAILED'
    rows = []
    try:
        for case in pilot.CASES:
            case_rows = [row for row in labels if row['case'] == case]
            data = {
                'sixteen_tape': tensorize(case_rows),
                'four_tape': tensorize([{**row, 'target_probabilities': row['old_4_target_probabilities']}
                                       for row in case_rows]),
            }
            for arm in tqdm(ARMS, desc=f'checkpoint path {case}', leave=False):
                for root in pilot.TRAIN_ROOTS:
                    trial = TRAIN / f'{case}_{arm}_{root}'
                    dev = {int(item['transition']): item for item in json.loads((trial / 'dev_milestones.json').read_text())}
                    selection = json.loads((trial / 'selection.json').read_text())
                    for stage in STAGES:
                        checkpoint = trial / ('pretrain_checkpoint.pt' if stage == 'pretrain' else stage)
                        if stage != 'pretrain':
                            step = int(stage.removeprefix('checkpoint_').removesuffix('.pt'))
                            assert pilot.sha256(checkpoint) == dev[step]['checkpoint_sha256']
                        policy = pilot.load_policy(checkpoint)
                        for variant in ('sixteen_tape', 'four_tape'):
                            d = data[variant]
                            with torch.no_grad():
                                scores = candidate_logits(policy, d)
                                loss = float(listwise_loss(policy, d))
                                top1 = float((scores.argmax(1) == d['targets'].argmax(1)).float().mean())
                            row = {'case': case, 'arm': arm, 'train_root': root,
                                   'stage': stage, 'target_variant': variant,
                                   'training_records': len(case_rows),
                                   'checkpoint_sha256': pilot.sha256(checkpoint),
                                   'train_label_cross_entropy': loss,
                                   'train_label_top1_rate': top1,
                                   'dev_mean_H_tail_cost': None if stage == 'pretrain' else dev[step]['mean_H_tail_cost'],
                                   'dev_selected': stage != 'pretrain' and selection['checkpoint'] == stage}
                            pilot.append_jsonl(output / 'rows.jsonl', row)
                            rows.append(row)
        summary = {'interpretation': 'post-pilot training-label/DEV trajectory only; no held-out or checkpoint selection',
                   'arms': {}}
        for arm in ARMS:
            summary['arms'][arm] = {}
            for case in pilot.CASES:
                summary['arms'][arm][case] = {}
                for stage in STAGES:
                    d = [r for r in rows if r['arm'] == arm and r['case'] == case and r['stage'] == stage
                         and r['target_variant'] == 'sixteen_tape']
                    summary['arms'][arm][case][stage] = {
                        'sixteen_tape_train_cross_entropy': float(np.mean([r['train_label_cross_entropy'] for r in d])),
                        'sixteen_tape_top1_rate': float(np.mean([r['train_label_top1_rate'] for r in d])),
                        'dev_mean_H_tail_cost': None if stage == 'pretrain' else float(np.mean([r['dev_mean_H_tail_cost'] for r in d])),
                        'dev_selected_roots': sum(r['dev_selected'] for r in d),
                    }
        pilot.write_json(output / 'summary.json', summary)
        assert len(rows) == len(pilot.CASES) * len(ARMS) * len(pilot.TRAIN_ROOTS) * len(STAGES) * 2
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
            'protocol_sha256': pilot.sha256(pilot.PROTOCOL),
            'output_sha256': {p.name: pilot.sha256(p) for p in output.iterdir()
                              if p.is_file() and p.name != 'manifest.json'},
        })
    print(output.resolve())
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
