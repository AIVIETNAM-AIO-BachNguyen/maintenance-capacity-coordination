"""Replay and audit the prelocked v7 transfer artifacts before interpreting gates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, TAIL, RuleController
from maintenance_capacity_coordination.baselines import V14Controller
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.pilot import choose as choose_v1, load_policy as load_v1
from maintenance_capacity_coordination.structured_pilot_v3 import choose as choose_v3, load_policy as load_v3
from maintenance_capacity_coordination.structured_policy_v3 import ACTIONS
from experiments.threshold_control_v1 import choose_g
from experiments import capacity_transfer_v7 as v7


def load(path):
    return json.loads(Path(path).read_text())


def near(left, right, atol=1e-7):
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(float(left) - float(right)) <= atol
    return left == right


def assert_artifact_header(run, manifest):
    assert manifest['status'] == 'COMPLETED'
    assert manifest['protocol_sha256'] == v7.sha256(v7.PROTOCOL)
    assert manifest['source_sha256'] == v7.source_hashes()
    assert manifest['config_sha256'] == v7.config_hashes()
    assert manifest['git_status'] == '' or manifest['kind'] == 'smoke'
    assert manifest['sealed_test_reserved'] == list(v7.SEALED_TEST)
    for filename, expected in manifest['output_sha256'].items():
        assert v7.sha256(run / filename) == expected, filename
    v7.check_seed_roles()


def verify_checkpoint_identity(manifest):
    training = Path(manifest['parent_training']['path'])
    parent = v7.verify_parent(training, manifest['kind'] == 'replication_eval')
    assert parent == manifest['parent_training']
    roots = v7.REPL_TRAIN_ROOTS if manifest['kind'] == 'replication_eval' else v7.ROOTS
    frozen = v7.freeze_checkpoints(training, roots)
    assert frozen == manifest['checkpoints']
    for identity in frozen.values():
        assert v7.sha256(identity['path']) == identity['sha256']


def policy_for(row, manifest, cache):
    arm, key = row['arm'], row['checkpoint']
    if key is None:
        assert arm in ('F_risk_matching', 'F_rollout')
        return RuleController(arm[2:])
    assert key in manifest['checkpoints']
    if key not in cache:
        path = manifest['checkpoints'][key]['path']
        if arm in ('S', 'R', 'G3', 'D3'):
            cache[key] = load_v3(path)
        elif arm in ('G', 'D'):
            cache[key] = load_v1(path)
        elif arm == 'A':
            cache[key] = V14Controller.from_checkpoint(path)
        else:
            raise AssertionError(arm)
    return cache[key]


def replay_row(row, manifest, cache):
    source, case, arm, root, seed = (row[k] for k in
                                    ('source_case', 'case', 'arm', 'root', 'seed'))
    assert (source, case) in {(s, c) for s, t in v7.PAIRS for c in (s, t)}
    assert row['physical_seed'] == role_seed(seed, BASE_VERSION + ':physical')
    assert len(row['actions']) == len(row['submissions']) == HORIZON
    assert row['H_transitions'] == HORIZON and row['tail_transitions'] == TAIL
    assert row['selection_seconds'] >= 0
    env = CoordinationEnv(v7.config_by_name()[case])
    env.reset(seed)
    policy = policy_for(row, manifest, cache)
    done = False
    for tick in range(HORIZON):
        if arm in ('S', 'R', 'G3', 'D3'):
            d = choose_v3(policy, env, deterministic=True)
            submit, index = d['submissions'], d['matching_index']
        elif arm == 'G':
            d = choose_g(policy, env, deterministic=True)
            submit, index = d['submissions'], d['matching_index']
        elif arm == 'D':
            d = choose_v1(policy, env, deterministic=True)
            submit, index = d['submissions'], d['matching_index']
        else:
            submit, matching = policy.choose(env.config, env.inner.state, env.tick,
                                             case=case, seed=seed)
            index = ACTIONS.index(matching)
        assert index == row['actions'][tick]
        assert list(submit) == row['submissions'][tick]
        assert env.matching_mask(submit)[index]
        _, done = env.step(submit, index)
    assert done
    metrics = env.metrics()
    for field, actual in metrics.items():
        assert near(row[field], actual), (field, row[field], actual)
    assert row['invalid_assignments'] == 0
    assert near(row['objective'], row['H_cost'] + row['tail_cost'])
    assert near(row['episode_return'], -row['objective'])
    assert near(row['objective'], row['maintenance_cost'] + row['failure_cost']
                + row['unavailability_cost'])
    assert 0 <= row['H_availability'] <= 1
    if arm == 'F_rollout':
        assert row['forecast_transitions'] == policy.forecast_transitions
    else:
        assert row['forecast_transitions'] == 0


def expected_keys(kind):
    roots = v7.REPL_TRAIN_ROOTS if kind == 'replication_eval' else v7.ROOTS
    seeds = v7.SMOKE_SEEDS if kind == 'smoke' else (v7.REPL_EVAL_SEEDS if kind == 'replication_eval' else v7.PILOT_SEEDS)
    pairs = v7.PAIRS[:1] if kind == 'smoke' else v7.PAIRS
    students, contextual, controls = set(), set(), set()
    for source, target in pairs:
        for case in (source, target):
            for seed in seeds:
                students.update((source, case, arm, root, seed) for arm in v7.ARMS for root in roots)
                contextual.update((source, case, 'D3', root, seed) for root in v7.ROOTS)
                contextual.update((source, case, arm, root, seed) for arm in ('G', 'D') for root in v7.HIST_ROOTS)
                controls.update((source, case, arm, None, seed) for arm in ('F_risk_matching', 'F_rollout'))
                controls.update((source, case, 'A', root, seed) for root in v7.V14_ROOTS)
    return {'students': students, 'contextual': contextual, 'controls': controls}


def independent_analysis(rows, seeds, roots):
    keyed = {(r['source_case'], r['case'], r['arm'], r['root'], r['seed']): r['objective'] for r in rows}
    matrix = np.empty((len(v7.PAIRS), len(roots), len(seeds), 3), dtype=float)
    for ci, (source, target) in enumerate(v7.PAIRS):
        for ri, root in enumerate(roots):
            for si, seed in enumerate(seeds):
                def cost(case, arm):
                    return keyed[source, case, arm, root, seed]
                sg_target = cost(target, 'S') - cost(target, 'G3')
                matrix[ci, ri, si] = (sg_target,
                                       sg_target - cost(source, 'S') + cost(source, 'G3'),
                                       cost(target, 'S') - cost(target, 'R'))
    results = {}
    for i, name in enumerate(('target_S_minus_G3', 'transfer_difference', 'target_S_minus_R')):
        panel = matrix[:, :, :, i]
        per_seed = panel.mean(axis=(0, 1))
        draws = np.random.default_rng(v7.BOOTSTRAP_SEED).integers(0, len(seeds), (5000, len(seeds)))
        results[name] = {
            'overall': float(panel.mean()),
            'by_pair': {f'{s}->{t}': float(panel[ci].mean()) for ci, (s, t) in enumerate(v7.PAIRS)},
            'by_training_root': {str(root): float(panel[:, ri, :].mean()) for ri, root in enumerate(roots)},
            'physical_cluster_ci95': np.quantile(per_seed[draws].mean(axis=1), (.025, .975)).tolist(),
        }
    return results


def audit_evaluation(run):
    run = Path(run).resolve()
    manifest = load(run / 'manifest.json')
    assert manifest['kind'] in ('smoke', 'pilot', 'replication_eval')
    assert_artifact_header(run, manifest)
    verify_checkpoint_identity(manifest)
    expected = expected_keys(manifest['kind'])
    cache = {}
    totals = Counter()
    all_rows = []
    for panel in ('students', 'contextual', 'controls'):
        rows = v7.read_jsonl(run / f'{panel}.jsonl')
        keys = [(r['source_case'], r['case'], r['arm'], r['root'], r['seed']) for r in rows]
        assert len(keys) == len(set(keys)) == len(expected[panel])
        assert set(keys) == expected[panel]
        for row in rows:
            replay_row(row, manifest, cache)
            totals['H_transitions'] += row['H_transitions']
            totals['tail_transitions'] += row['tail_transitions']
            totals['forecast_transitions'] += row['forecast_transitions']
        totals[{'students': 'student_episodes', 'contextual': 'contextual_episodes',
                'controls': 'control_episodes'}[panel]] = len(rows)
        all_rows.extend(rows)
    assert totals['student_episodes'] == manifest['student_episodes']
    assert totals['contextual_episodes'] == manifest['contextual_episodes']
    assert totals['control_episodes'] == manifest['control_episodes']
    for field in ('H_transitions', 'tail_transitions', 'forecast_transitions'):
        assert totals[field] == manifest[field], field
    if manifest['kind'] != 'smoke':
        roots = v7.REPL_TRAIN_ROOTS if manifest['kind'] == 'replication_eval' else v7.ROOTS
        seeds = v7.REPL_EVAL_SEEDS if manifest['kind'] == 'replication_eval' else v7.PILOT_SEEDS
        independent = independent_analysis(v7.read_jsonl(run / 'students.jsonl'), seeds, roots)
        saved = load(run / 'analysis.json')
        for name, item in independent.items():
            recorded = saved['comparisons'][name]
            for field in ('overall', 'by_pair', 'by_training_root', 'physical_cluster_ci95'):
                if isinstance(item[field], dict):
                    assert all(near(value, recorded[field][key]) for key, value in item[field].items())
                elif isinstance(item[field], list):
                    assert all(near(a, b) for a, b in zip(item[field], recorded[field]))
                else:
                    assert near(item[field], recorded[field])
        sg, tr, sr = (independent[n] for n in ('target_S_minus_G3', 'transfer_difference', 'target_S_minus_R'))
        conditions = (sg['overall'] <= -30, all(x < 0 for x in sg['by_pair'].values()),
                      sum(x < 0 for x in sg['by_training_root'].values()) >= 2,
                      sg['physical_cluster_ci95'][1] < 0,
                      tr['overall'] <= -10, all(x <= 0 for x in tr['by_pair'].values()),
                      tr['physical_cluster_ci95'][1] < 0,
                      sr['overall'] <= -10, all(x <= 0 for x in sr['by_pair'].values()),
                      sum(x < 0 for x in sr['by_training_root'].values()) >= 2,
                      sr['physical_cluster_ci95'][1] < 0)
        assert list(saved['gate_conditions'].values()) == list(conditions)
        assert saved['go'] is all(conditions)
        decision = load(run / 'replication_go_decision.json')
        assert decision['go'] is all(conditions)
        assert decision['gate_conditions'] == saved['gate_conditions']
    audit = {'status': 'PASS', 'kind': manifest['kind'], 'replayed_episodes': len(all_rows),
             'complete_panel_and_checkpoint_identity': 'PASS', 'source_protocol_config_seed_roles': 'PASS',
             'legality_cost_accounting_and_gate_arithmetic': 'PASS'}
    v7.write_json(run / 'audit.json', audit)
    return audit


def audit_training(run):
    run = Path(run).resolve()
    manifest = load(run / 'manifest.json')
    assert manifest['kind'] == 'replication_train'
    assert_artifact_header(run, manifest)
    pilot = Path(manifest['pilot_run']['path'])
    assert v7.sha256(pilot / 'manifest.json') == manifest['pilot_run']['manifest_sha256']
    assert v7.sha256(pilot / 'audit.json') == manifest['pilot_run']['audit_sha256']
    assert load(pilot / 'audit.json')['status'] == 'PASS'
    assert load(pilot / 'replication_go_decision.json')['go'] is True
    expected = {(source, arm, root) for source, _ in v7.PAIRS for arm in v7.ARMS for root in v7.REPL_TRAIN_ROOTS}
    trials = {(t['case'], t['arm'], t['train_seed']): t for t in manifest['trials']}
    assert len(trials) == len(expected) and set(trials) == expected
    total_training = 0
    total_dev = 0
    for (source, arm, root), summary in trials.items():
        trial = run / f'{source}_{arm}_{root}'
        item = load(trial / 'manifest.json')
        assert item == summary and item['status'] == 'COMPLETED'
        assert item['requested_H_transitions'] == item['actual_H_transitions'] == 8192
        assert item['elapsed_seconds'] < 7200 and item['forecast_transitions'] == 0
        assert item['ppo'] == {'n_steps': 256, 'minibatch': 64, 'epochs': 4, 'learning_rate': 3e-4,
                                'gamma': .99, 'gae_lambda': .95, 'clip': .2, 'entropy_coef': .01}
        train_rows = v7.read_jsonl(trial / 'training_episodes.jsonl')
        assert len(train_rows) == 8192 // HORIZON
        for row in train_rows:
            assert row['invalid_assignments'] == 0
            assert near(row['objective'], row['H_cost'] + row['tail_cost'])
            assert near(row['episode_return'], -row['objective'])
        dev = v7.read_jsonl(trial / 'dev_episodes.jsonl')
        expected_dev = {(str(m), s) for m in (2048, 4096, 8192) for s in v7.REPL_DEV_SEEDS}
        assert len(dev) == len(expected_dev)
        assert {(r['checkpoint'], r['seed']) for r in dev} == expected_dev
        for row in dev:
            assert row['case'] == source and row['arm'] == arm
            assert row['physical_seed'] == role_seed(row['seed'], BASE_VERSION + ':physical')
            assert row['invalid_assignments'] == 0
            assert near(row['objective'], row['H_cost'] + row['tail_cost'])
        candidates = []
        for milestone in (2048, 4096, 8192):
            checkpoint = trial / f'checkpoint_{milestone}.pt'
            assert checkpoint.exists() and checkpoint.stat().st_size > 0
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert (payload['arm'], payload['case'], payload['train_seed'], payload['transition']) == (arm, source, root, milestone)
            mean = float(np.mean([r['objective'] for r in dev if r['checkpoint'] == str(milestone)]))
            candidates.append((mean, milestone, checkpoint.name))
        best = min(candidates)
        choice = load(trial / 'selection.json')
        assert choice['checkpoint'] == item['selected_checkpoint'] == best[2]
        assert choice['transition'] == best[1] and near(choice['mean_H_tail_cost'], best[0])
        total_training += 8192
        total_dev += len(dev)
    audit = {'status': 'PASS', 'kind': 'replication_train', 'trials': len(trials),
             'H_training_decisions': total_training, 'DEV_episodes': total_dev,
             'source_protocol_config_pilot_gate': 'PASS', 'checkpoint_selection_accounting': 'PASS'}
    v7.write_json(run / 'audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    kind = load(args.run / 'manifest.json')['kind']
    result = audit_training(args.run) if kind == 'replication_train' else audit_evaluation(args.run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
