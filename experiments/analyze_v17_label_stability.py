"""Training-bank-only four-tape split-half ranking stability for v17 labels."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from experiments import same_tick_pilot_v17 as pilot

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / 'artifacts/full_matching_v17_labels_20261023T080338842451Z'


def main():
    manifest = json.loads((LABEL / 'manifest.json').read_text())
    audit = json.loads((LABEL / 'audit.json').read_text())
    assert manifest['status'] == 'COMPLETED' and audit['status'] == 'PASS'
    assert manifest['summary']['sufficient_coverage']
    for path, expected in manifest['source_sha256'].items():
        assert pilot.sha256(ROOT / path) == expected
    run = ROOT / 'artifacts' / ('v17_label_stability_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(exist_ok=False)
    started = time.monotonic()
    status = 'FAILED'
    try:
        all_rows = []
        for line in (LABEL / 'labels.jsonl').open():
            row = json.loads(line)
            costs = np.asarray([branch['total_costs'] for branch in row['branches']], dtype=float)
            assert costs.shape == (len(row['matching_indices']), 4)
            left, right = costs[:, :2].mean(1), costs[:, 2:].mean(1)
            full = costs.mean(1)
            pair_agreement = []
            for i in range(len(full)):
                for j in range(i + 1, len(full)):
                    if left[i] != left[j] and right[i] != right[j]:
                        pair_agreement.append(int(np.sign(left[i] - left[j]) == np.sign(right[i] - right[j])))
            sorted_cost = np.sort(full)
            entry = {
                'case': row['case'], 'raw_seed': row['raw_seed'], 'tick': row['tick'],
                'candidate_count': len(full), 'split_winner_agrees': int(np.argmin(left) == np.argmin(right)),
                'pair_sign_agree_count': sum(pair_agreement), 'pair_sign_count': len(pair_agreement),
                'full_mean_top_gap': float(sorted_cost[1] - sorted_cost[0]),
                'mean_within_candidate_tape_sd': float(costs.std(1).mean()),
                'target_top_probability': max(row['target_probabilities']),
            }
            pilot.append_jsonl(run / 'rows.jsonl', entry)
            all_rows.append(entry)
        summary = {}
        for case in pilot.CASES:
            subset = [r for r in all_rows if r['case'] == case]
            summary[case] = {
                'states': len(subset),
                'split_winner_agreement_rate': float(np.mean([r['split_winner_agrees'] for r in subset])),
                'pooled_pair_sign_agreement_rate': sum(r['pair_sign_agree_count'] for r in subset) / sum(r['pair_sign_count'] for r in subset),
                'median_full_mean_top_gap': float(np.median([r['full_mean_top_gap'] for r in subset])),
                'mean_within_candidate_tape_sd': float(np.mean([r['mean_within_candidate_tape_sd'] for r in subset])),
                'mean_target_top_probability': float(np.mean([r['target_top_probability'] for r in subset])),
            }
        pilot.write_json(run / 'summary.json', summary)
        status = 'COMPLETED'
    finally:
        pilot.write_json(run / 'manifest.json', {
            'status': status, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started,
            'label_manifest_sha256': pilot.sha256(LABEL / 'manifest.json'),
            'label_audit_sha256': pilot.sha256(LABEL / 'audit.json'),
            'labels_sha256': pilot.sha256(LABEL / 'labels.jsonl'),
            'analysis_source_sha256': pilot.sha256(Path(__file__)),
            'interpretation': 'training-label split-half descriptive stability; no held-out outcomes, model selection, or gate change',
            'output_sha256': {p.name: pilot.sha256(p) for p in run.iterdir() if p.is_file() and p.name != 'manifest.json'},
        })
    print(run.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
