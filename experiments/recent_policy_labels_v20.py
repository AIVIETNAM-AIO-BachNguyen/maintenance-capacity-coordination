"""Build v20 training-only complete-matching labels at recent-policy states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
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
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv, actor_inputs_scaled
from maintenance_capacity_coordination.same_tick_policy_v15 import initial_edge_mask
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import full_matching_labels_v17 as old
from experiments import same_tick_pilot_v19 as pilot

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/on_policy_matching_value_v20_prelock.md'
PRIOR_TRAIN = ROOT / 'artifacts/same_tick_v19_train_20261024T094616510975Z'
ROOTS = pilot.TRAIN_ROOTS
SOURCE_ARMS = ('C19', 'D19')
WINDOWS = ((0, 23), (24, 47))
LABEL_SEEDS = tuple(range(81801000, 81801048))
SMOKE_SEEDS = (81800011, 81800012)
TARGET_COUNTS = {CASES[0]: 273, CASES[1]: 235}
SLOTS = 16
TIMEOUT = 14400

sha256 = old.sha256
write_json = old.write_json
append_jsonl = old.append_jsonl
git_output = old.git_output


def state_seed(raw, case, root, arm, window):
    return role_seed(raw, f'v20-state:{case}:{root}:{arm}:{window}')


def priority_seed(raw, case, root, arm, window):
    return role_seed(raw, f'v20-priority:{case}:{root}:{arm}:{window}')


def candidate_seed(raw, case, root, arm, tick):
    return role_seed(raw, f'v20-candidate:{case}:{root}:{arm}:{tick}')


def bank_seed(raw, case, root, arm, tick, slot):
    return role_seed(raw, f'v20-value:{case}:{root}:{arm}:{tick}:{slot}')


def source_paths():
    fixed = [ROOT / name for name in (
        'experiments/recent_policy_labels_v20.py',
        'experiments/audit_recent_policy_labels_v20.py',
        'experiments/same_tick_pilot_v19.py',
        'experiments/same_tick_pilot_v18.py',
        'experiments/full_matching_labels_v17.py',
        'experiments/same_tick_pilot_v16.py',
        'experiments/scaled_n5k3_v12_config.py',
        'pyproject.toml')]
    return sorted((ROOT / 'src').rglob('*.py')) + fixed


def source_checkpoint(case, root, arm):
    trial = PRIOR_TRAIN / f'{case}_{arm}_{root}'
    selected = json.loads((trial / 'selection.json').read_text())
    checkpoint = trial / selected['checkpoint']
    if sha256(checkpoint) != selected['checkpoint_sha256']:
        raise RuntimeError('v19 source checkpoint hash changed')
    return pilot.load_policy(checkpoint), selected['checkpoint_sha256']


def s16_checkpoint(case, root):
    old_root = 78710000 + ROOTS.index(root)
    path, digest = old.checkpoint(case, 'S16', old_root)
    from experiments import same_tick_pilot_v16 as v16
    return v16.load_policy(path), digest


def prior_check():
    manifest = json.loads((PRIOR_TRAIN / 'manifest.json').read_text())
    audit = json.loads((PRIOR_TRAIN / 'audit.json').read_text())
    if manifest['status'] != 'COMPLETED' or audit['status'] != 'PASS':
        raise RuntimeError('audited v19 train/DEV required')
    for path, digest in manifest['source_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f'v19 source changed: {path}')
    return manifest, audit


def seed_registry_check(mode):
    raw = SMOKE_SEEDS if mode == 'smoke' else LABEL_SEEDS
    if len(raw) != len(set(raw)) or not all(81800000 <= x < 81900000 for x in raw):
        raise RuntimeError('v20 label raw seed role collision')
    prior = old.seed_registry_check('labels') | old.seed_registry_check('smoke')
    # Include v18 additional bank tapes and physical seeds reserved by v18/v19.
    from experiments import full_matching_labels_v18 as v18
    rows = [json.loads(line) for line in (v18.OLD_LABELS / 'labels.jsonl').read_text().splitlines()]
    prior |= {v18.extra_seed(row, slot) for row in rows for slot in v18.EXTRA_SLOTS}
    old_raw = (*pilot.SMOKE_SEEDS, *pilot.DEV_SEEDS, *pilot.HELDOUT_SEEDS,
               *pilot.REPLICATION_DEV_SEEDS, *pilot.REPLICATION_EVAL_SEEDS, *pilot.SEALED_TEST)
    prior |= {role_seed(x, BASE_VERSION + ':physical') for x in old_raw}
    current = set()
    for seed in raw:
        physical = role_seed(seed, BASE_VERSION + ':physical')
        if physical in prior or physical in current:
            raise RuntimeError('v20 label physical tape collision')
        current.add(physical)
    cases = CASES if mode == 'labels' else (CASES[0],)
    roots = ROOTS if mode == 'labels' else (ROOTS[0],)
    arms = SOURCE_ARMS if mode == 'labels' else (SOURCE_ARMS[0],)
    for case in cases:
        for root in roots:
            for arm in arms:
                for seed in raw:
                    for window in range(2):
                        for value in (state_seed(seed, case, root, arm, window),
                                      priority_seed(seed, case, root, arm, window)):
                            if value in prior or value in current:
                                raise RuntimeError('v20 state/priority seed collision')
                            current.add(value)
                    for tick in range(48):
                        value = candidate_seed(seed, case, root, arm, tick)
                        if value in prior or value in current:
                            raise RuntimeError('v20 candidate seed collision')
                        current.add(value)
                        for slot in range(SLOTS if mode == 'labels' else 2):
                            value = bank_seed(seed, case, root, arm, tick, slot)
                            if value in prior or value in current:
                                raise RuntimeError('v20 label bank seed collision')
                            current.add(value)
    return prior | current


def selected_windows(policy, case, root, arm, raw):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    candidates = {0: [], 1: []}
    for tick in range(48):
        decision = old.v16.choose(policy, env, True)
        if old.first_options(decision['edge_mask']):
            window = 0 if tick < 24 else 1
            candidates[window].append((copy.deepcopy(env), tick, tuple(decision['submissions'])))
        env.step(decision['submissions'], decision['matching_index'])
    selected = {}
    for window, choices in candidates.items():
        selected[window] = choices[state_seed(raw, case, root, arm, window) % len(choices)] if choices else None
    return selected, {window: len(choices) for window, choices in candidates.items()}


def selected_pool(pool, case, target):
    by_stratum = defaultdict(list)
    for item in pool:
        if item['case'] == case and item['state'] is not None:
            key = item['root'], item['source_arm'], item['window']
            by_stratum[key].append(item)
    strata = [(root, arm, window) for root in ROOTS for arm in SOURCE_ARMS for window in range(2)]
    if any(not by_stratum[key] for key in strata):
        raise RuntimeError(f'v20 empty source stratum: {case}')
    for key in strata:
        by_stratum[key].sort(key=lambda item: (item['priority'], item['raw_seed']))
    selected, positions = [], {key: 0 for key in strata}
    while len(selected) < target:
        before = len(selected)
        for key in strata:
            if len(selected) == target:
                break
            if positions[key] < len(by_stratum[key]):
                selected.append(by_stratum[key][positions[key]])
                positions[key] += 1
        if len(selected) == before:
            raise RuntimeError(f'v20 insufficient eligible state supply: {case}')
    return selected


def candidate_actions(base, policies, raw, case, root, source_arm, tick, submissions, edge_mask):
    legal = np.flatnonzero(base.matching_mask(submissions)).tolist()
    sequences = old.canonical_actions(base, edge_mask)
    if set(sequences) != set(legal):
        raise RuntimeError('canonical and physical matching sets differ')
    selected, reasons = [], {}
    def add(index, reason):
        if index not in legal:
            raise RuntimeError('recommended matching is illegal')
        if index not in reasons:
            selected.append(index)
            reasons[index] = []
        reasons[index].append(reason)
    add(base.actions.index(()), 'WAIT')
    for arm in ('C19', 'D19', 'CF19'):
        choice = old.v16.choose(policies[arm], base, True)
        if tuple(choice['submissions']) != tuple(submissions):
            raise RuntimeError('source actor admission differs')
        add(choice['matching_index'], arm)
    rng = np.random.default_rng(candidate_seed(raw, case, root, source_arm, tick))
    for cardinality in (1, 2, 3):
        options = [i for i in legal if i not in reasons and len(base.actions[i]) == cardinality]
        if options:
            add(int(rng.choice(options)), f'cardinality_{cardinality}')
    for index in rng.permutation([i for i in legal if i not in reasons]):
        if len(selected) >= 8:
            break
        add(int(index), 'uniform_fill')
    if len(selected) != min(8, len(legal)):
        raise RuntimeError('candidate count mismatch')
    return selected, [list(sequences[i]) for i in selected], [reasons[i] for i in selected]


def evaluate_label(item, policies, s16, slots, deadline):
    base, tick, submissions = item['state']
    case, root, arm, raw = (item[key] for key in ('case', 'root', 'source_arm', 'raw_seed'))
    admitted = base.admitted(submissions)
    machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
    mask = initial_edge_mask(base.config, admitted)
    actions, sequences, reasons = candidate_actions(
        base, policies, raw, case, root, arm, tick, submissions, mask)
    bank_seeds, tapes = [], []
    for slot in range(slots):
        seed = bank_seed(raw, case, root, arm, tick, slot)
        physical = DispatchEnv(base.config)
        physical.reset(seed)
        bank_seeds.append(seed)
        tapes.append(physical.events)
    branches = []
    for index in actions:
        total, hcost, tailcost = [], [], []
        for tape in tapes:
            cost, h, tail = old.continuation_cost(base, s16, index, submissions, tape, deadline)
            total.append(cost); hcost.append(h); tailcost.append(tail)
        branches.append({'matching_index': index, 'total_costs': total,
                         'H_costs': hcost, 'tail_costs': tailcost})
    means = np.asarray([np.mean(branch['total_costs']) for branch in branches], dtype=float)
    weights = np.exp(-(means - means.min()) / 20.)
    targets = weights / weights.sum()
    return {'case': case, 'train_root': root, 'source_arm': arm, 'raw_seed': raw,
            'physical_seed': role_seed(raw, BASE_VERSION + ':physical'),
            'window': item['window'], 'tick': tick, 'selection_priority': item['priority'],
            'source_checkpoint_sha256': item['source_checkpoint_sha256'],
            'S16_checkpoint_sha256': item['S16_checkpoint_sha256'],
            'submissions': list(submissions), 'matching_indices': actions,
            'sequences': sequences, 'candidate_reasons': reasons,
            'bank_seeds': bank_seeds, 'branches': branches,
            'mean_costs': means.tolist(), 'target_probabilities': targets.tolist(),
            'machine_obs': machine.tolist(), 'worker_obs': worker.tolist(),
            'edge_mask': mask.tolist()}


def stability(rows):
    result = {}
    for case in CASES:
        group = [r for r in rows if r['case'] == case]
        if not group:
            continue
        winners = agrees = pairs = ties = 0
        for row in group:
            costs = np.asarray([b['total_costs'] for b in row['branches']], dtype=float)
            if costs.shape[1] != 16:
                continue
            first, last = costs[:, :8].mean(1), costs[:, 8:].mean(1)
            winners += int(np.argmin(first) == np.argmin(last))
            for i in range(len(first)):
                for j in range(i + 1, len(first)):
                    left, right = first[i] - first[j], last[i] - last[j]
                    if left == 0 or right == 0:
                        ties += 1
                    else:
                        pairs += 1; agrees += int(np.sign(left) == np.sign(right))
        result[case] = {'states': len(group), 'winner_agreement': winners / len(group),
                        'pair_sign_agreement': agrees / pairs if pairs else None,
                        'pair_sign_count': pairs, 'pair_ties_excluded': ties}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'labels'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior_check()
    seed_registry_check(args.mode)
    if args.mode == 'labels' and git_output('status', '--porcelain'):
        raise RuntimeError('clean committed source required for scientific v20 labels')
    cases = CASES if args.mode == 'labels' else (CASES[0],)
    roots = ROOTS if args.mode == 'labels' else (ROOTS[0],)
    arms = SOURCE_ARMS if args.mode == 'labels' else (SOURCE_ARMS[0],)
    seeds = LABEL_SEEDS if args.mode == 'labels' else SMOKE_SEEDS
    slots = SLOTS if args.mode == 'labels' else 2
    run = args.output_root / ('recent_policy_v20_' + args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True, exist_ok=False)
    print(run.resolve(), flush=True)
    started = time.monotonic()
    deadline = started + TIMEOUT
    cfg = configs()
    manifest = {'status': 'RUNNING', 'kind': args.mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--porcelain'),
                'platform': platform.platform(), 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__,
                'protocol_sha256': sha256(PROTOCOL),
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths()},
                'config_sha256': {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest() for case in cases},
                'config_seeds': CONFIG_SEEDS, 'prior_train_run': str(PRIOR_TRAIN),
                'prior_train_manifest_sha256': sha256(PRIOR_TRAIN / 'manifest.json'),
                'prior_train_audit_sha256': sha256(PRIOR_TRAIN / 'audit.json'),
                'cases': cases, 'roots': roots, 'source_arms': arms,
                'raw_seeds': seeds, 'windows': WINDOWS, 'bank_slots': slots,
                'target_counts': TARGET_COUNTS if args.mode == 'labels' else {cases[0]: 1},
                'timeout_seconds': TIMEOUT}
    write_json(run / 'manifest.json', manifest)
    append_jsonl(run / 'run.log', {'event': 'started', 'utc': manifest['started_utc']})
    pool, rows = [], []
    try:
        for case in cases:
            for root in roots:
                c, c_hash = source_checkpoint(case, root, 'C19')
                d, d_hash = source_checkpoint(case, root, 'D19')
                cf = DynamicSequencePolicy('D16')
                cf.load_state_dict(c.state_dict(), strict=True); cf.eval()
                s16, s_hash = s16_checkpoint(case, root)
                policies = {'C19': c, 'D19': d, 'CF19': cf}
                for arm in arms:
                    for raw in tqdm(seeds, desc=f'v20 pool {case} {root} {arm}', leave=False):
                        if time.monotonic() > deadline:
                            raise TimeoutError('v20 pool timeout')
                        selected, counts = selected_windows(policies[arm], case, root, arm, raw)
                        for window in range(2):
                            state = selected[window]
                            item = {'case': case, 'root': root, 'source_arm': arm, 'raw_seed': raw,
                                    'window': window, 'qualified_ticks': counts[window],
                                    'tick': state[1] if state else None,
                                    'priority': priority_seed(raw, case, root, arm, window),
                                    'source_checkpoint_sha256': c_hash if arm == 'C19' else d_hash,
                                    'S16_checkpoint_sha256': s_hash, 'state': state}
                            pool.append(item)
        chosen = []
        for case in cases:
            if args.mode == 'labels':
                chosen += selected_pool(pool, case, TARGET_COUNTS[case])
            else:
                chosen += next(item for item in pool if item['case'] == case and item['state'] is not None),
        chosen_keys = {(r['case'], r['root'], r['source_arm'], r['raw_seed'], r['window']) for r in chosen}
        for item in pool:
            key = item['case'], item['root'], item['source_arm'], item['raw_seed'], item['window']
            append_jsonl(run / 'pool.jsonl', {k: v for k, v in item.items() if k != 'state'} |
                         {'eligible': item['state'] is not None, 'selected': key in chosen_keys})
        for item in tqdm(chosen, desc='v20 forced branches'):
            if time.monotonic() > deadline:
                raise TimeoutError('v20 label timeout')
            case, root = item['case'], item['root']
            c, _ = source_checkpoint(case, root, 'C19')
            d, _ = source_checkpoint(case, root, 'D19')
            cf = DynamicSequencePolicy('D16'); cf.load_state_dict(c.state_dict(), strict=True); cf.eval()
            s16, _ = s16_checkpoint(case, root)
            row = evaluate_label(item, {'C19': c, 'D19': d, 'CF19': cf}, s16, slots, deadline)
            append_jsonl(run / 'labels.jsonl', row)
            rows.append(row)
            manifest['labelled_states'] = len(rows)
            write_json(run / 'manifest.json', manifest)
        summary = {'pool_rows': len(pool), 'eligible_states': sum(item['state'] is not None for item in pool),
                   'labelled_states': len(rows),
                   'by_case': {case: sum(row['case'] == case for row in rows) for case in cases},
                   'label_branches': sum(len(row['branches']) * slots for row in rows),
                   'stability': stability(rows) if args.mode == 'labels' else {}}
        write_json(run / 'summary.json', summary)
        manifest.update(status='COMPLETED', summary=summary)
        append_jsonl(run / 'run.log', {'event': 'completed', 'utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error))
        append_jsonl(run / 'run.log', {'event': 'failed', 'error': repr(error),
                                     'utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {str(p.relative_to(run)): sha256(p) for p in run.rglob('*')
                                     if p.is_file() and p.name != 'manifest.json'}
        write_json(run / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
