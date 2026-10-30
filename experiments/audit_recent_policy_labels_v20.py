"""Independent physical replay of v20 recent-policy matching-value labels."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv, actor_inputs_scaled
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    EDGES, enumerate_sequences, initial_edge_mask, matching_from_sequence, choose,
)
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import recent_policy_labels_v20 as labeler
from experiments import same_tick_pilot_v19 as pilot
from experiments import full_matching_labels_v17 as old


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(a, b):
    return abs(float(a) - float(b)) < 1e-7


def identity(run, manifest):
    assert manifest['protocol_sha256'] == labeler.sha256(labeler.PROTOCOL)
    assert manifest['config_seeds'] == CONFIG_SEEDS
    for path, digest in manifest['source_sha256'].items():
        assert labeler.sha256(labeler.ROOT / path) == digest, path
    for path, digest in manifest['output_sha256'].items():
        assert labeler.sha256(run / path) == digest, path
    for case, digest in manifest['config_sha256'].items():
        actual = hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
        assert actual == digest
    assert manifest['prior_train_manifest_sha256'] == labeler.sha256(labeler.PRIOR_TRAIN / 'manifest.json')
    assert manifest['prior_train_audit_sha256'] == labeler.sha256(labeler.PRIOR_TRAIN / 'audit.json')
    labeler.prior_check()
    labeler.seed_registry_check(manifest['kind'])


def source_policy(case, root, arm):
    trial = labeler.PRIOR_TRAIN / f'{case}_{arm}_{root}'
    selected = json.loads((trial / 'selection.json').read_text())
    checkpoint = trial / selected['checkpoint']
    assert labeler.sha256(checkpoint) == selected['checkpoint_sha256']
    return pilot.load_policy(checkpoint), selected['checkpoint_sha256']


def continuation_policy(case, root):
    old_root = 78710000 + labeler.ROOTS.index(root)
    path, digest = old.checkpoint(case, 'S16', old_root)
    from experiments import same_tick_pilot_v16 as v16
    return v16.load_policy(path), digest


def source_windows(policy, case, raw, root, arm):
    env = ScaledCoordinationEnv(configs()[case])
    env.reset(raw)
    options = [[], []]
    for tick in range(48):
        action = choose(policy, env, True)
        legal = [i for i, yes in enumerate(action['edge_mask']) if yes]
        eligible = any(sum(
            j > i and EDGES[j][0] != EDGES[i][0] and EDGES[j][1] != EDGES[i][1]
            for j in legal) >= 2 for i in legal)
        if eligible:
            options[tick // 24].append((copy.deepcopy(env), tick, tuple(action['submissions'])))
        env.step(action['submissions'], action['matching_index'])
    chosen = {}
    for window in (0, 1):
        seed = role_seed(raw, f'v20-state:{case}:{root}:{arm}:{window}')
        chosen[window] = options[window][seed % len(options[window])] if options[window] else None
    return chosen, [len(x) for x in options]


def choose_bank(pool, case, target, roots, arms):
    groups = {}
    for root in roots:
        for arm in arms:
            for window in (0, 1):
                group = [r for r in pool if r['case'] == case and r['root'] == root
                         and r['source_arm'] == arm and r['window'] == window and r['eligible']]
                assert group
                groups[root, arm, window] = sorted(group, key=lambda r: (r['priority'], r['raw_seed']))
    order = [(root, arm, window) for root in roots for arm in arms for window in (0, 1)]
    selected = []
    layer = 0
    while len(selected) < target:
        added = 0
        for key in order:
            if len(selected) == target:
                break
            if layer < len(groups[key]):
                selected.append(groups[key][layer]); added += 1
        assert added
        layer += 1
    return selected


def matching_candidates(base, policies, raw, case, root, arm, tick, submissions, mask):
    valid = [i for i, yes in enumerate(base.matching_mask(submissions)) if yes]
    index = {matching: i for i, matching in enumerate(base.actions)}
    paths = {}
    for path in enumerate_sequences(mask, canonical=True):
        action = index[matching_from_sequence(path)]
        assert action not in paths
        paths[action] = list(path)
    assert set(paths) == set(valid)
    chosen, reasons = [], {}
    def add(action, reason):
        assert action in valid
        if action not in reasons:
            chosen.append(action)
            reasons[action] = []
        reasons[action].append(reason)
    add(index[()], 'WAIT')
    for name in ('C19', 'D19', 'CF19'):
        action = choose(policies[name], base, True)
        assert tuple(action['submissions']) == submissions
        add(action['matching_index'], name)
    rng = np.random.default_rng(role_seed(raw, f'v20-candidate:{case}:{root}:{arm}:{tick}'))
    for size in (1, 2, 3):
        alternatives = [i for i in valid if i not in reasons and len(base.actions[i]) == size]
        if alternatives:
            add(int(rng.choice(alternatives)), f'cardinality_{size}')
    for action in rng.permutation([i for i in valid if i not in reasons]):
        if len(chosen) == 8:
            break
        add(int(action), 'uniform_fill')
    assert len(chosen) == min(8, len(valid))
    return chosen, [paths[i] for i in chosen], [reasons[i] for i in chosen]


def physical_cost(base, policy, submissions, action, events):
    env = copy.deepcopy(base)
    env.inner._events = events
    before = float(env.inner.metrics['objective'])
    _, done = env.step(submissions, action)
    while not done:
        decision = old.v16.choose(policy, env, True)
        _, done = env.step(decision['submissions'], decision['matching_index'])
    final = env.metrics()
    return final['objective'] - before, final['H_cost'] - before, final['tail_cost']


def stability(records, cases):
    result = {}
    for case in cases:
        group = [r for r in records if r['case'] == case]
        winners = agrees = pairs = ties = 0
        for row in group:
            costs = np.asarray([branch['total_costs'] for branch in row['branches']])
            assert costs.shape[1] == 16
            first, last = costs[:, :8].mean(1), costs[:, 8:].mean(1)
            winners += int(int(first.argmin()) == int(last.argmin()))
            for i in range(len(first)):
                for j in range(i + 1, len(first)):
                    a, b = first[i] - first[j], last[i] - last[j]
                    if a == 0 or b == 0:
                        ties += 1
                    else:
                        pairs += 1; agrees += int(np.sign(a) == np.sign(b))
        result[case] = {'states': len(group), 'winner_agreement': winners / len(group),
                        'pair_sign_agreement': agrees / pairs if pairs else None,
                        'pair_sign_count': pairs, 'pair_ties_excluded': ties}
    return result


def replay(run, manifest):
    full = manifest['kind'] == 'labels'
    cases = CASES if full else (CASES[0],)
    roots = labeler.ROOTS if full else (labeler.ROOTS[0],)
    arms = labeler.SOURCE_ARMS if full else (labeler.SOURCE_ARMS[0],)
    seeds = labeler.LABEL_SEEDS if full else labeler.SMOKE_SEEDS
    slots = 16 if full else 2
    assert manifest['cases'] == list(cases) and manifest['roots'] == list(roots)
    assert manifest['source_arms'] == list(arms) and manifest['raw_seeds'] == list(seeds)
    assert manifest['bank_slots'] == slots and manifest['windows'] == [[0,23],[24,47]]
    pool_rows, label_rows = rows(run / 'pool.jsonl'), rows(run / 'labels.jsonl')
    assert len(pool_rows) == len(cases) * len(roots) * len(arms) * len(seeds) * 2
    pool_by_key = {(r['case'], r['root'], r['source_arm'], r['raw_seed'], r['window']): r for r in pool_rows}
    label_by_key = {(r['case'], r['train_root'], r['source_arm'], r['raw_seed'], r['window']): r for r in label_rows}
    assert len(pool_by_key) == len(pool_rows) and len(label_by_key) == len(label_rows)
    state_by_key = {}
    for case in cases:
        for root in roots:
            c, c_hash = source_policy(case, root, 'C19')
            d, d_hash = source_policy(case, root, 'D19')
            s, s_hash = continuation_policy(case, root)
            for arm in arms:
                for raw in seeds:
                    selected, counts = source_windows({'C19': c, 'D19': d}[arm], case, raw, root, arm)
                    for window in (0, 1):
                        key = case, root, arm, raw, window
                        item = pool_by_key[key]
                        state = selected[window]
                        priority = role_seed(raw, f'v20-priority:{case}:{root}:{arm}:{window}')
                        assert item['qualified_ticks'] == counts[window]
                        assert item['eligible'] == (state is not None)
                        assert item['tick'] == (state[1] if state else None)
                        assert item['priority'] == priority
                        assert item['source_checkpoint_sha256'] == (c_hash if arm == 'C19' else d_hash)
                        assert item['S16_checkpoint_sha256'] == s_hash
                        if state is not None:
                            state_by_key[key] = state
    selected = []
    for case in cases:
        if full:
            selected += choose_bank(pool_rows, case, labeler.TARGET_COUNTS[case], roots, arms)
        else:
            selected += [next(r for r in pool_rows if r['case'] == case and r['eligible'])]
    keys = [(r['case'], r['root'], r['source_arm'], r['raw_seed'], r['window']) for r in selected]
    assert len(keys) == len(set(keys))
    assert {key for key, row in pool_by_key.items() if row['selected']} == set(keys)
    assert list(label_by_key) == keys
    branches = 0
    for key in keys:
        row = label_by_key[key]
        case, root, arm, raw, window = key
        base, tick, submissions = state_by_key[key]
        assert row['tick'] == tick and tuple(row['submissions']) == submissions
        assert row['selection_priority'] == pool_by_key[key]['priority']
        assert row['physical_seed'] == role_seed(raw, BASE_VERSION + ':physical')
        c, c_hash = source_policy(case, root, 'C19')
        d, d_hash = source_policy(case, root, 'D19')
        cf = DynamicSequencePolicy('D16'); cf.load_state_dict(c.state_dict(), strict=True); cf.eval()
        s, s_hash = continuation_policy(case, root)
        assert row['source_checkpoint_sha256'] == (c_hash if arm == 'C19' else d_hash)
        assert row['S16_checkpoint_sha256'] == s_hash
        admitted = base.admitted(submissions)
        machine, worker = actor_inputs_scaled(base.config, admitted, base.tick)
        mask = initial_edge_mask(base.config, admitted)
        assert np.allclose(row['machine_obs'], machine, atol=1e-7)
        assert np.allclose(row['worker_obs'], worker, atol=1e-7)
        assert np.array_equal(row['edge_mask'], mask)
        actions, paths, reasons = matching_candidates(base, {'C19': c, 'D19': d, 'CF19': cf},
                                                      raw, case, root, arm, tick, submissions, mask)
        assert row['matching_indices'] == actions and row['sequences'] == paths
        assert row['candidate_reasons'] == reasons and len(set(actions)) == len(actions)
        tapes = []
        for slot in range(slots):
            seed = role_seed(raw, f'v20-value:{case}:{root}:{arm}:{tick}:{slot}')
            assert row['bank_seeds'][slot] == seed
            physical = DispatchEnv(base.config); physical.reset(seed)
            tapes.append(physical.events)
        means = []
        assert len(row['branches']) == len(actions)
        for j, action in enumerate(actions):
            branch = row['branches'][j]
            assert branch['matching_index'] == action
            totals = []
            for slot, events in enumerate(tapes):
                total, h, tail = physical_cost(base, s, submissions, action, events)
                assert close(total, branch['total_costs'][slot])
                assert close(h, branch['H_costs'][slot])
                assert close(tail, branch['tail_costs'][slot]) and close(total, h + tail)
                totals.append(total); branches += 1
            means.append(float(np.mean(totals)))
        assert np.allclose(means, row['mean_costs'], atol=1e-7)
        weights = np.exp(-(np.asarray(means) - min(means)) / 20.)
        assert np.allclose(weights / weights.sum(), row['target_probabilities'], atol=1e-8)
    summary = json.loads((run / 'summary.json').read_text())
    expected = {'pool_rows': len(pool_rows), 'eligible_states': sum(r['eligible'] for r in pool_rows),
                'labelled_states': len(label_rows),
                'by_case': {case: sum(r['case'] == case for r in label_rows) for case in cases},
                'label_branches': branches,
                'stability': stability(label_rows, cases) if full else {}}
    assert summary == expected and manifest['summary'] == expected
    assert rows(run / 'run.log')[0]['event'] == 'started'
    assert rows(run / 'run.log')[-1]['event'] == 'completed'
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    audit_dir = run.parent / ('recent_policy_v20_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    audit_dir.mkdir(exist_ok=False)
    if (run / 'audit.json').exists():
        shutil.copy2(run / 'audit.json', audit_dir / 'previous_audit.json')
    result = {'status': 'RUNNING', 'kind': manifest['kind'], 'source_run': str(run),
              'audit_artifact': str(audit_dir), 'manifest_sha256': labeler.sha256(run / 'manifest.json'),
              'auditor_sha256': labeler.sha256(Path(__file__))}
    try:
        assert manifest['status'] == 'COMPLETED'
        identity(run, manifest)
        summary = replay(run, manifest)
        result.update(summary)
        if manifest['kind'] == 'labels':
            conditions = {f'{case}_winner_agreement': value['winner_agreement'] >= .30
                          for case, value in summary['stability'].items()}
            conditions.update({f'{case}_pair_sign_agreement': value['pair_sign_agreement'] >= .63
                               for case, value in summary['stability'].items()})
            decision = {'go': all(conditions.values()), 'conditions': conditions,
                        'protocol_sha256': manifest['protocol_sha256']}
            labeler.write_json(audit_dir / 'label_go_decision.json', decision)
            labeler.write_json(run / 'label_go_decision.json', decision)
            result['decision_sha256'] = labeler.sha256(run / 'label_go_decision.json')
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        labeler.write_json(audit_dir / 'audit.json', result)
        labeler.write_json(run / 'audit.json', result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
