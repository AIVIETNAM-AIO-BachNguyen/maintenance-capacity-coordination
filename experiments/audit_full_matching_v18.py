"""Independent v18 label identity, tape, cost and qualification replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, role_seed
from experiments.scaled_n5k3_v12_config import CASES, configs
from experiments import audit_full_matching_v17 as audit17
from experiments import full_matching_labels_v17 as old
from experiments import full_matching_labels_v18 as labels
from experiments import same_tick_pilot_v16 as v16


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def identity(run, manifest):
    assert manifest['status'] == 'COMPLETED'
    assert manifest['protocol_sha256'] == old.sha256(labels.PROTOCOL)
    for path, expected in manifest['source_sha256'].items():
        assert old.sha256(labels.ROOT / path) == expected, path
    for path, expected in manifest['output_sha256'].items():
        assert old.sha256(run / path) == expected, path
    for case, expected in manifest['config_sha256'].items():
        actual = hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
        assert actual == expected
    assert old.sha256(labels.OLD_LABELS / 'manifest.json') == manifest['old_manifest_sha256']
    assert old.sha256(labels.OLD_LABELS / 'audit.json') == manifest['old_audit_sha256']
    assert old.sha256(labels.OLD_LABELS / 'labels.jsonl') == manifest['old_labels_sha256']
    assert json.loads((labels.OLD_LABELS / 'audit.json').read_text())['status'] == 'PASS'


def replay(run, manifest):
    old_rows = rows(labels.OLD_LABELS / 'labels.jsonl')
    expected = old_rows[:1] if manifest['kind'] == 'smoke' else old_rows
    found = rows(run / 'labels.jsonl')
    assert len(found) == len(expected) == manifest['selected_states']
    slots = (4, 5) if manifest['kind'] == 'smoke' else tuple(range(4, 16))
    assert manifest['extra_slots'] == list(slots)
    policies, cache = {}, {}
    pairs = defaultdict(lambda: [0, 0, 0, 0])
    by_case = defaultdict(int)
    branch_count = 0
    for original, row in zip(expected, found):
        case, root, raw, window = (original[k] for k in ('case', 'C16_root', 'raw_seed', 'window'))
        assert tuple(row[k] for k in ('case', 'C16_root', 'raw_seed', 'window')) == (case, root, raw, window)
        for key in ('tick', 'submissions', 'matching_indices', 'sequences', 'candidate_reasons',
                    'machine_obs', 'worker_obs', 'edge_mask', 'physical_seed', 'v16_checkpoint_sha256'):
            assert row[key] == original[key], key
        assert row['old_4_target_probabilities'] == original['target_probabilities']
        key = case, root
        if key not in policies:
            c_path, c_hash = old.checkpoint(case, 'C16', root)
            s_path, s_hash = old.checkpoint(case, 'S16', root)
            assert row['v16_checkpoint_sha256'] == {'C16': c_hash, 'S16': s_hash}
            policies[key] = v16.load_policy(c_path), v16.load_policy(s_path)
        c, s = policies[key]
        state_key = case, root, raw
        if state_key not in cache:
            cache[state_key] = audit17.selected_windows(c, case, root, raw)[0]
        state = cache[state_key][window]
        assert state is not None and state[1] == row['tick'] and list(state[2]) == row['submissions']
        base, _, submissions = state
        assert row['bank_seeds'][:4] == original['bank_seeds']
        assert len(row['bank_seeds']) == 4 + len(slots)
        assert len(row['branches']) == len(original['branches'])
        tapes = []
        for slot, recorded in zip(slots, row['bank_seeds'][4:]):
            anchor = 80700001 + (slot - 4) % 2 if manifest['kind'] == 'smoke' else raw
            seed = role_seed(anchor, f'v18-extra:{case}:{root}:{row["tick"]}:{slot}')
            assert recorded == seed
            env = DispatchEnv(base.config)
            env.reset(seed)
            tapes.append(env.events)
        means = []
        for old_branch, branch in zip(original['branches'], row['branches']):
            assert branch['matching_index'] == old_branch['matching_index']
            for metric in ('total_costs', 'H_costs', 'tail_costs'):
                assert branch[metric][:4] == old_branch[metric]
                assert len(branch[metric]) == 4 + len(slots)
            action = branch['matching_index']
            for offset, tape in enumerate(tapes, start=4):
                total, h, tail = audit17.branch_cost(base, s, submissions, action, tape)
                assert abs(total - branch['total_costs'][offset]) < 1e-7
                assert abs(h - branch['H_costs'][offset]) < 1e-7
                assert abs(tail - branch['tail_costs'][offset]) < 1e-7
                assert abs(total - h - tail) < 1e-7
                branch_count += 1
            means.append(float(np.mean(branch['total_costs'])))
        assert np.allclose(means, row['mean_costs'], atol=1e-7)
        weights = np.exp(-(np.asarray(means) - min(means)) / 20.)
        assert np.allclose(weights / weights.sum(), row['target_probabilities'], atol=1e-8)
        by_case[case] += 1
        if len(slots) == 12:
            costs = np.asarray([branch['total_costs'] for branch in row['branches']], dtype=float)
            first, last = costs[:, :8].mean(1), costs[:, 8:].mean(1)
            pairs[case][0] += int(np.argmin(first) == np.argmin(last))
            for i in range(len(first)):
                for j in range(i + 1, len(first)):
                    a, b = first[i] - first[j], last[i] - last[j]
                    if a == 0 or b == 0:
                        pairs[case][3] += 1
                    else:
                        pairs[case][2] += 1
                        pairs[case][1] += int(np.sign(a) == np.sign(b))
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['states'] == len(found)
    assert summary['by_case'] == {case: by_case[case] for case in CASES}
    assert summary['new_branches'] == branch_count
    assert summary['total_branches'] == sum(len(r['branches']) * len(r['bank_seeds']) for r in found)
    if manifest['kind'] == 'labels':
        assert len(found) == 508 and [by_case[case] for case in CASES] == [273, 235]
        stability = {}
        for case in CASES:
            wins, agree, total, ties = pairs[case]
            stability[case] = {'states': by_case[case], 'winner_agreement': wins / by_case[case],
                               'pair_sign_agreement': agree / total, 'pair_sign_count': total,
                               'pair_ties_excluded': ties}
        qualified = all(stability[case]['winner_agreement'] >= .30 and
                        stability[case]['pair_sign_agreement'] >= .63 for case in CASES)
    else:
        stability, qualified = {}, bool(found)
    assert summary['stability'] == stability
    assert summary['qualified'] == qualified == manifest['sufficient_coverage']
    assert manifest['summary'] == summary
    events = rows(run / 'run.log')
    assert events[0]['event'] == 'started' and events[-1]['event'] == 'completed'
    return {'states_replayed': len(found), 'new_branches_replayed': branch_count,
            'qualified': qualified, 'stability': stability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    audit_dir = run.parent / ('full_matching_v18_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    audit_dir.mkdir(exist_ok=False)
    if (run / 'audit.json').exists():
        shutil.copy2(run / 'audit.json', audit_dir / 'previous_audit.json')
    result = {'status': 'RUNNING', 'source_run': str(run), 'audit_artifact': str(audit_dir),
              'manifest_sha256': old.sha256(run / 'manifest.json'),
              'auditor_sha256': old.sha256(Path(__file__))}
    try:
        identity(run, manifest)
        result.update(replay(run, manifest))
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        old.write_json(audit_dir / 'audit.json', result)
        old.write_json(run / 'audit.json', result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
