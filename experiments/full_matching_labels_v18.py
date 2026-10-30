"""Extend the audited v17 training-only matching labels with 12 paired tapes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from experiments.scaled_n5k3_v12_config import CASES, configs
from experiments import full_matching_labels_v17 as old
from experiments import same_tick_pilot_v17 as pilot17

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/matching_value_precision_v18_prelock.md'
OLD_LABELS = ROOT / 'artifacts/full_matching_v17_labels_20261023T080338842451Z'
EXTRA_SLOTS = tuple(range(4, 16))
LABEL_TIMEOUT = 10800


def extra_seed(row, slot, *, smoke=False):
    raw = 80700001 + (slot - 4) % 2 if smoke else row['raw_seed']
    return role_seed(raw, f"v18-extra:{row['case']}:{row['C16_root']}:{row['tick']}:{slot}")


def source_hashes():
    files = sorted((ROOT / 'src').rglob('*.py')) + [
        ROOT / 'experiments/full_matching_labels_v18.py',
        ROOT / 'experiments/audit_full_matching_v18.py',
        ROOT / 'experiments/full_matching_labels_v17.py',
        ROOT / 'experiments/audit_full_matching_v17.py',
        ROOT / 'experiments/same_tick_pilot_v16.py',
        ROOT / 'experiments/scaled_n5k3_v12_config.py',
        ROOT / 'pyproject.toml',
    ]
    return {str(p.relative_to(ROOT)): old.sha256(p) for p in files}


def old_bank():
    m = json.loads((OLD_LABELS / 'manifest.json').read_text())
    a = json.loads((OLD_LABELS / 'audit.json').read_text())
    if m['status'] != 'COMPLETED' or a['status'] != 'PASS':
        raise RuntimeError('audited v17 training label bank required')
    if any(old.sha256(ROOT / path) != expected for path, expected in m['source_sha256'].items()):
        raise RuntimeError('v17 collection source changed')
    if any(old.sha256(OLD_LABELS / path) != expected for path, expected in m['output_sha256'].items()):
        raise RuntimeError('v17 label bank output changed')
    rows = [json.loads(x) for x in (OLD_LABELS / 'labels.jsonl').read_text().splitlines()]
    if len(rows) != 508 or [sum(r['case'] == case for r in rows) for case in CASES] != [273, 235]:
        raise RuntimeError('v17 training bank identity changed')
    return m, a, rows


def state_for(row, c, cache):
    key = row['case'], row['C16_root'], row['raw_seed']
    if key not in cache:
        cache[key] = old.selected_states(c, row['case'], row['C16_root'], row['raw_seed'])[0]
    state = cache[key][row['window']]
    if state is None or state[1] != row['tick'] or list(state[2]) != row['submissions']:
        raise RuntimeError('v17 selected state no longer matches')
    return state


def extend_row(row, state, s, deadline, *, smoke=False):
    base, _, submissions = state
    extra_seeds = [extra_seed(row, slot, smoke=smoke) for slot in EXTRA_SLOTS if not smoke or slot < 6]
    tapes = []
    for seed in extra_seeds:
        physical = DispatchEnv(base.config)
        physical.reset(seed)
        tapes.append(physical.events)
    branches = []
    for original in row['branches']:
        branch = {key: list(value) if isinstance(value, list) else value for key, value in original.items()}
        for tape in tapes:
            total, h, tail = old.continuation_cost(base, s, branch['matching_index'], submissions, tape, deadline)
            branch['total_costs'].append(total)
            branch['H_costs'].append(h)
            branch['tail_costs'].append(tail)
        branches.append(branch)
    means = np.asarray([np.mean(b['total_costs']) for b in branches], dtype=float)
    weights = np.exp(-(means - means.min()) / 20.)
    target = weights / weights.sum()
    result = dict(row)
    result['old_4_target_probabilities'] = row['target_probabilities']
    result['target_probabilities'] = target.tolist()
    result['mean_costs'] = means.tolist()
    result['bank_seeds'] = row['bank_seeds'] + extra_seeds
    result['branches'] = branches
    return result


def stability(rows):
    result = {}
    for case in CASES:
        case_rows = [r for r in rows if r['case'] == case]
        if not case_rows:
            continue
        pairs_match = pairs_total = winners_match = 0
        ties = 0
        for row in case_rows:
            costs = np.asarray([b['total_costs'] for b in row['branches']], dtype=float)
            if costs.shape[1] != 16:
                continue
            first, last = costs[:, :8].mean(1), costs[:, 8:].mean(1)
            winners_match += int(np.argmin(first) == np.argmin(last))
            for i in range(len(first)):
                for j in range(i + 1, len(first)):
                    di, dj = first[i] - first[j], last[i] - last[j]
                    if di == 0 or dj == 0:
                        ties += 1
                    else:
                        pairs_total += 1
                        pairs_match += int(np.sign(di) == np.sign(dj))
        result[case] = {
            'states': len(case_rows), 'winner_agreement': winners_match / len(case_rows),
            'pair_sign_agreement': pairs_match / pairs_total if pairs_total else None,
            'pair_sign_count': pairs_total, 'pair_ties_excluded': ties,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'labels'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    torch.set_num_threads(1)
    old_manifest, old_audit, old_rows = old_bank()
    if args.mode == 'labels' and pilot17.git_output('status', '--porcelain'):
        raise RuntimeError('committed clean source required for scientific v18 labels')
    selected = old_rows[:1] if args.mode == 'smoke' else old_rows
    old_registry = old.seed_registry_check('labels')
    extras = [extra_seed(row, slot, smoke=args.mode == 'smoke') for row in selected
              for slot in EXTRA_SLOTS if args.mode != 'smoke' or slot < 6]
    if len(extras) != len(set(extras)) or set(extras) & old_registry:
        raise RuntimeError('v18 extra tape seed collision')
    v18_raw = [*range(80700000, 80700013), *range(80710000, 80710003),
               *range(80711000, 80711016), *range(80712000, 80712032), 80713000,
               *range(80720000, 80720003), *range(80721000, 80721016),
               *range(80722000, 80722032), 80723000, *range(80730000, 80730032)]
    reserved_physical = {role_seed(raw, BASE_VERSION + ':physical') for raw in v18_raw}
    if set(extras) & reserved_physical or old_registry & reserved_physical:
        raise RuntimeError('v18 extra/prior/reserved physical tape collision')
    run = args.output_root / ('full_matching_v18_' + args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True, exist_ok=False)
    print(run.resolve(), flush=True)
    started = time.monotonic()
    deadline = started + LABEL_TIMEOUT
    cfg = configs()
    manifest = {
        'status': 'RUNNING', 'kind': args.mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': pilot17.git_output('rev-parse', 'HEAD'),
        'git_status': pilot17.git_output('status', '--porcelain'),
        'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__,
        'protocol_sha256': old.sha256(PROTOCOL), 'source_sha256': source_hashes(),
        'config_sha256': {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest() for case in CASES},
        'old_label_run': str(OLD_LABELS),
        'old_manifest_sha256': old.sha256(OLD_LABELS / 'manifest.json'),
        'old_audit_sha256': old.sha256(OLD_LABELS / 'audit.json'),
        'old_labels_sha256': old.sha256(OLD_LABELS / 'labels.jsonl'),
        'cases': list(CASES), 'old_states': len(old_rows),
        'selected_states': len(selected), 'extra_slots': [4, 5] if args.mode == 'smoke' else list(EXTRA_SLOTS),
        'timeout_seconds': LABEL_TIMEOUT,
    }
    old.write_json(run / 'manifest.json', manifest)
    old.append_jsonl(run / 'run.log', {'event': 'started', 'utc': manifest['started_utc']})
    cache = {}
    policies = {}
    output_rows = []
    try:
        for row in tqdm(selected, desc=f'v18 {args.mode} extra tapes'):
            key = row['case'], row['C16_root']
            if key not in policies:
                c, _, s, hashes = old.old_policies(*key)
                policies[key] = c, s, hashes
            c, s, hashes = policies[key]
            if hashes != row['v16_checkpoint_sha256']:
                raise RuntimeError('v16 checkpoint hash mismatch')
            state = state_for(row, c, cache)
            extended = extend_row(row, state, s, deadline, smoke=args.mode == 'smoke')
            old.append_jsonl(run / 'labels.jsonl', extended)
            output_rows.append(extended)
            manifest['completed_states'] = len(output_rows)
            old.write_json(run / 'manifest.json', manifest)
        by_case = {case: sum(r['case'] == case for r in output_rows) for case in CASES}
        rank = stability(output_rows) if args.mode == 'labels' else {}
        qualified = (by_case == {CASES[0]: 273, CASES[1]: 235}
                     and all(rank[case]['winner_agreement'] >= .30 and rank[case]['pair_sign_agreement'] >= .63
                             for case in CASES)) if args.mode == 'labels' else bool(output_rows)
        summary = {
            'states': len(output_rows), 'by_case': by_case,
            'new_branches': sum(len(r['branches']) * len(r['bank_seeds'][4:]) for r in output_rows),
            'total_branches': sum(len(r['branches']) * len(r['bank_seeds']) for r in output_rows),
            'stability': rank, 'qualified': bool(qualified),
        }
        old.write_json(run / 'summary.json', summary)
        manifest.update(status='COMPLETED', summary=summary, sufficient_coverage=bool(qualified))
        old.append_jsonl(run / 'run.log', {'event': 'completed', 'utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error))
        old.append_jsonl(run / 'run.log', {'event': 'failed', 'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {str(p.relative_to(run)): old.sha256(p) for p in run.rglob('*')
                                     if p.is_file() and p.name != 'manifest.json'}
        old.write_json(run / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
