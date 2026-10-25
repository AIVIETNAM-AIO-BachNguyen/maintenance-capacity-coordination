"""Descriptive v21 fit on the already-audited training-only matching bank.

Candidate-score top choice is not the deployed greedy-sequence action. The
reported candidate regret is in-sample branch continuation, not episode gain.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from experiments import same_tick_pilot_v21 as v21
from experiments.full_matching_pretrain_v17 import candidate_logits, listwise_loss
from experiments.pairwise_matching_v21 import pairwise_loss_from_scores, tensorize

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'artifacts/recent_policy_v20_labels_20261024T113110729833Z'
TRAIN = ROOT / 'artifacts/same_tick_v21_train_20261024T180209859141Z'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def candidate_costs(row: dict) -> np.ndarray:
    return np.asarray([np.mean(np.asarray(branch['H_costs']) + np.asarray(branch['tail_costs']))
                       for branch in row['branches']], dtype=float)


def evaluate(policy, data, case_rows: list[dict]) -> dict:
    with torch.no_grad():
        scores = candidate_logits(policy, data)
        pair_loss = float(pairwise_loss_from_scores(scores, data))
        list_loss = float(listwise_loss(policy, data))
    costs = [candidate_costs(row) for row in case_rows]
    top1 = []
    regrets = []
    agreement_all = []
    agreement_gap20 = []
    for i, (row, value) in enumerate(zip(case_rows, costs, strict=True)):
        n = len(row['matching_indices'])
        logits = scores[i, :n].numpy()
        chosen = int(np.argmax(logits))
        best = int(np.argmin(value))
        top1.append(chosen == best)
        regrets.append(float(value[chosen] - value[best]))
        for left in range(n):
            for right in range(left + 1, n):
                gap = float(value[left] - value[right])
                if gap == 0:
                    continue
                aligned = bool(np.sign(logits[left] - logits[right]) == -np.sign(gap))
                agreement_all.append(aligned)
                if abs(gap) >= 20:
                    agreement_gap20.append(aligned)
    return {'pairwise_loss': pair_loss, 'listwise_loss': list_loss,
            'candidate_top1_agreement': float(np.mean(top1)),
            'candidate_mean_cost_gap_to_bank_best': float(np.mean(regrets)),
            'pair_sign_agreement_all': float(np.mean(agreement_all)),
            'pair_sign_agreement_gap_at_least_20': float(np.mean(agreement_gap20)),
            'pairs_all': len(agreement_all), 'pairs_gap_at_least_20': len(agreement_gap20)}


def main() -> None:
    torch.set_num_threads(1)
    label_audit = json.loads((LABELS / 'audit.json').read_text())
    train_audit = json.loads((TRAIN / 'audit.json').read_text())
    train_manifest = json.loads((TRAIN / 'manifest.json').read_text())
    assert label_audit['status'] == train_audit['status'] == 'PASS'
    assert train_manifest['status'] == 'COMPLETED'
    assert train_manifest['labels_sha256'] == digest(LABELS / 'labels.jsonl')
    all_rows = [json.loads(line) for line in (LABELS / 'labels.jsonl').read_text().splitlines()]
    out = ROOT / 'artifacts' / ('v21_training_fit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(exist_ok=False)
    manifest = {'status': 'RUNNING', 'kind': 'training_only_descriptive',
                'source_sha256': digest(Path(__file__)),
                'labels_sha256': digest(LABELS / 'labels.jsonl'),
                'label_audit_sha256': digest(LABELS / 'audit.json'),
                'training_manifest_sha256': digest(TRAIN / 'manifest.json'),
                'training_audit_sha256': digest(TRAIN / 'audit.json'),
                'v21_protocol_sha256': digest(ROOT / 'docs/reliability_weighted_pairwise_v21_prelock.md'),
                'git_commit': train_manifest['git_commit'],
                'limitation': 'In-sample audited training bank only; candidate-score choice differs from deployed greedy sequence; no physical episode-gain estimate.'}
    write(out / 'manifest.json', manifest)
    results = []
    try:
        for case in v21.CASES:
            case_rows = [row for row in all_rows if row['case'] == case]
            data = tensorize(case_rows)
            uniform = float(np.mean([np.log(len(row['matching_indices'])) for row in case_rows]))
            for arm in tqdm(v21.NEW_ARMS, desc=f'v21 training fit {case}'):
                for root in v21.TRAIN_ROOTS:
                    trial = TRAIN / f'{case}_{arm}_{root}'
                    selected = json.loads((trial / 'selection.json').read_text())
                    checkpoints = [('pretrain', trial / 'pretrain_checkpoint.pt'),
                                   ('selected', trial / selected['checkpoint'])]
                    for stage, checkpoint in checkpoints:
                        expected = (json.loads((trial / 'manifest.json').read_text())['pretrain_checkpoint_sha256']
                                    if stage == 'pretrain' else selected['checkpoint_sha256'])
                        assert digest(checkpoint) == expected
                        record = {'case': case, 'arm': arm, 'root': root, 'stage': stage,
                                  'states': len(case_rows), 'checkpoint_sha256': expected,
                                  'uniform_loss_reference': uniform,
                                  **evaluate(v21.load_policy(checkpoint), data, case_rows)}
                        results.append(record)
                        with (out / 'rows.jsonl').open('a') as handle:
                            handle.write(json.dumps(record, sort_keys=True) + '\n')
        manifest.update(status='COMPLETED', rows=len(results),
                        rows_sha256=digest(out / 'rows.jsonl'))
        write(out / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error))
        write(out / 'manifest.json', manifest)
        raise
    print(out)


if __name__ == '__main__':
    main()
