"""Independent artifact, physical replay and locked-gate audit for v21."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    EDGES, STOP, initial_edge_mask, matching_from_sequence,
)
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from experiments.pairwise_matching_v21 import pretrain, tensorize
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs
from experiments import same_tick_pilot_v18 as prior
from experiments import same_tick_pilot_v19 as v19
from experiments import same_tick_pilot_v20 as v20
from experiments import same_tick_pilot_v21 as pilot


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearly(a, b):
    return abs(float(a) - float(b)) < 1e-7


def check_metrics(row, env):
    metrics = env.metrics()
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and value is not None:
            assert nearly(row[key], value), (row['case'], row['arm'], row['seed'], key)
        else:
            assert row[key] == value
    assert row['invalid_assignments'] == 0
    assert nearly(row['episode_return'] + row['objective'], 0)
    assert nearly(row['H_cost'] + row['tail_cost'], row['objective'])


def identity(run, manifest):
    assert manifest['protocol_sha256'] == pilot.sha256(pilot.PROTOCOL)
    assert manifest['config_seeds'] == CONFIG_SEEDS
    assert manifest['config_sha256'] == {
        case: hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
        for case in CASES}
    for path, expected in manifest['source_sha256'].items():
        assert pilot.sha256(pilot.ROOT / path) == expected, path
    for path, expected in manifest['output_sha256'].items():
        assert pilot.sha256(run / path) == expected, path
    assert manifest['label_manifest_sha256'] == pilot.sha256(pilot.LABEL / 'manifest.json')
    assert manifest['label_audit_sha256'] == pilot.sha256(pilot.LABEL / 'audit.json')
    assert manifest['labels_sha256'] == pilot.sha256(pilot.LABEL / 'labels.jsonl')
    assert json.loads((pilot.LABEL / 'audit.json').read_text())['status'] == 'PASS'
    assert manifest['v20_train_manifest_sha256'] == pilot.sha256(pilot.V20_TRAIN / 'manifest.json')
    assert manifest['v20_train_audit_sha256'] == pilot.sha256(pilot.V20_TRAIN / 'audit.json')
    assert manifest['prior_train_manifest_sha256'] == pilot.sha256(pilot.PRIOR_TRAIN / 'manifest.json')
    assert manifest['prior_train_audit_sha256'] == pilot.sha256(pilot.PRIOR_TRAIN / 'audit.json')
    assert manifest['older_train_manifest_sha256'] == pilot.sha256(pilot.OLDER_TRAIN / 'manifest.json')
    assert manifest['older_train_audit_sha256'] == pilot.sha256(pilot.OLDER_TRAIN / 'audit.json')
    pilot.label_bank(manifest['kind'])
    pilot.old_train_check()
    if manifest['kind'] != 'smoke':
        assert manifest['label_decision_sha256'] == pilot.sha256(pilot.LABEL / 'label_go_decision.json')
    pilot.seed_registry_check()


def legal_sequence(env, trace):
    submissions = tuple(trace['submissions'])
    index = trace['matching_index']
    assert 0 <= index < len(env.actions)
    sequence = trace['sequence']
    assert isinstance(sequence, list)
    assert matching_from_sequence(sequence) == env.actions[index]
    edge_mask = initial_edge_mask(env.config, env.admitted(submissions))
    used_m, used_w, last, stopped = set(), set(), -1, False
    for position, token in enumerate(sequence):
        feasible = [i for i, (m, j) in enumerate(EDGES)
                    if edge_mask[i] and m not in used_m and j not in used_w and i > last]
        assert feasible and not stopped
        if token == STOP:
            assert position == len(sequence) - 1
            stopped = True
        else:
            assert token in feasible
            m, j = EDGES[token]
            used_m.add(m)
            used_w.add(j)
            last = token
    if not stopped:
        assert not any(edge_mask[i] and m not in used_m and j not in used_w and i > last
                       for i, (m, j) in enumerate(EDGES))
    assert env.matching_mask(submissions)[index]
    return submissions, index


def audit_training_trace(trial, case, arm, root, kind, steps, record):
    traces = rows(trial / 'training_decisions.jsonl')
    assert len(traces) == steps
    episodes = rows(trial / 'training_episodes.jsonl')
    env = ScaledCoordinationEnv(configs()[case])
    episode = tick = completed = passes = 0
    env.reset(pilot.reset_seed(root, case, episode, kind))
    for trace in traces:
        assert (trace['case'], trace['arm'], trace['train_root'], trace['episode'], trace['tick'],
                trace['reset_seed']) == (case, arm, root, episode, tick,
                                         pilot.reset_seed(root, case, episode, kind))
        submissions, index = legal_sequence(env, trace)
        assert np.isfinite(trace['logprob'])
        reward, done = env.step(submissions, index)
        assert nearly(reward, trace['reward']) and done == trace['done']
        passes += trace['decoder_forward_passes']
        if done:
            row = episodes[completed]
            assert row['episode'] == episode and row['transition'] == (completed + 1) * 48
            for key, value in env.metrics().items():
                if isinstance(value, (float, int)) and value is not None:
                    assert nearly(row[key], value)
                else:
                    assert row[key] == value
            completed += 1
            episode += 1
            tick = 0
            env.reset(pilot.reset_seed(root, case, episode, kind))
        else:
            tick += 1
    assert completed == len(episodes) and tick == steps % 48
    assert passes == record['decoder_forward_passes']
    return len(traces)


def audit_rehearsal_log(trial, case, arm, root, record, bank_count):
    updates = rows(trial / 'updates.jsonl')
    expected_steps = list(range(256, record['actual_H_decisions'] + 1, 256))
    assert [row['transition'] for row in updates] == expected_steps
    for row in updates:
        assert all(np.isfinite(value) for key, value in row.items() if key != 'transition')
    if record['rehearsal_weight'] == 0:
        assert not (trial / 'rehearsal_minibatches.jsonl').exists()
        assert record['rehearsal_minibatches'] == 0
        return
    logged = rows(trial / 'rehearsal_minibatches.jsonl')
    assert len(logged) == record['rehearsal_minibatches'] == len(updates) * 16
    assert sum(row['rehearsal_candidate_paths'] for row in updates) == record['rehearsal_candidate_paths']
    assert sum(row['rehearsal_token_positions'] for row in updates) == record['rehearsal_token_positions']
    seed = role_seed(root, f'v21-label-cycle:{case}')
    assert record['label_cycle_seed'] == seed
    rng = np.random.default_rng(seed)
    order = np.empty(0, dtype=np.int64)
    position = 0
    def next_indices():
        nonlocal order, position
        result = []
        while len(result) < 32:
            if position == len(order):
                order = rng.permutation(bank_count)
                position = 0
            count = min(32 - len(result), len(order) - position)
            result.extend(int(x) for x in order[position:position + count])
            position += count
        return result
    for i, row in enumerate(logged):
        assert row['transition'] == expected_steps[i // 16]
        assert row['label_row_indices'] == next_indices()
        assert nearly(row['combined_loss'], row['ppo_loss'] + .5 * row['rehearsal_loss'])
    for j, update in enumerate(updates):
        batch = logged[j * 16:(j + 1) * 16]
        assert update['rehearsal_minibatches'] == 16
        assert nearly(update['rehearsal_loss_mean'], np.mean([x['rehearsal_loss'] for x in batch]))
        assert nearly(update['ppo_loss_mean'], np.mean([x['ppo_loss'] for x in batch]))


def replay_dev_episode(policy, row):
    env = ScaledCoordinationEnv(configs()[row['case']])
    env.reset(row['seed'])
    for _ in range(HORIZON):
        choice = prior.choose(policy, env, True)
        env.step(choice['submissions'], choice['matching_index'])
    check_metrics(row, env)
    assert row['physical_seed'] == role_seed(row['seed'], BASE_VERSION + ':physical')


def audit_new_trial(trial, case, arm, root, kind, steps, dev_seeds, bank):
    record = json.loads((trial / 'manifest.json').read_text())
    assert record['status'] == 'COMPLETED' and record['actual_H_decisions'] == steps
    policy = pilot.new_policy(arm)
    assert record['actor_parameters'] == policy.actor_parameters
    assert record['critic_parameters'] == policy.critic_parameters
    pretrain_path = trial / 'pretrain_checkpoint.pt'
    assert pilot.sha256(pretrain_path) == record['pretrain_checkpoint_sha256']
    saved = torch.load(pretrain_path, map_location='cpu', weights_only=True)
    assert (saved['arm'], saved['case'], saved['train_root']) == (arm, case, root)
    torch.manual_seed(root)
    expected = pilot.new_policy(arm)
    preseed = role_seed(root, f'v21-pretrain:{case}')
    epochs = 2 if kind == 'smoke' else 96
    objective = 'listwise' if arm == 'L21' else 'pairwise'
    replayed = []
    details = pretrain(expected, bank, preseed, objective=objective, epochs=epochs, on_epoch=replayed.append)
    assert rows(trial / 'pretrain_epochs.jsonl') == replayed
    assert record['pretraining_seed'] == preseed and record['pretraining_epochs'] == epochs
    assert all(record[key] == value for key, value in details.items())
    assert all(torch.equal(value, saved['state_dict'][key]) for key, value in expected.state_dict().items())
    rng_sha = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert record['torch_rng_after_pretrain_sha256'] == rng_sha
    assert record['pretraining_label_bank'] == 'v20_recent_policy_16_tape'
    assert record['supervised_objective'] == objective
    assert record['pretraining_records'] == len(bank)
    assert record['rehearsal_weight'] == .5
    audit_training_trace(trial, case, arm, root, kind, steps, record)
    audit_rehearsal_log(trial, case, arm, root, record, len(bank))
    milestones = (steps,) if kind == 'smoke' else pilot.MILESTONES
    dev_rows = rows(trial / 'dev_episodes.jsonl')
    assert len(dev_rows) == len(milestones) * len(dev_seeds)
    summaries = json.loads((trial / 'dev_milestones.json').read_text())
    assert [row['transition'] for row in summaries] == list(milestones)
    for milestone, summary in zip(milestones, summaries):
        matching = [row for row in dev_rows if row['checkpoint'] == str(milestone)]
        assert [row['seed'] for row in matching] == list(dev_seeds)
        assert nearly(summary['mean_H_tail_cost'], np.mean([row['objective'] for row in matching]))
        checkpoint = trial / summary['checkpoint']
        assert pilot.sha256(checkpoint) == summary['checkpoint_sha256']
        selected = pilot.load_policy(checkpoint)
        for row in matching:
            replay_dev_episode(selected, row)
    selection = json.loads((trial / 'selection.json').read_text())
    assert selection == min(summaries, key=lambda row: (row['mean_H_tail_cost'], row['transition']))
    assert record['selected_checkpoint'] == selection['checkpoint']
    assert nearly(record['dev_mean'], selection['mean_H_tail_cost'])
    return record


def audit_old_references(run, dev_seeds):
    selected = {}
    for case in CASES:
        for arm in pilot.OLD_ARMS:
            for root in pilot.TRAIN_ROOTS:
                source_root = v20.TRAIN_ROOTS[pilot.TRAIN_ROOTS.index(root)]
                trial = run / 'references' / f'{case}_{arm}_{root}'
                summaries = json.loads((trial / 'dev_milestones.json').read_text())
                dev_rows = rows(trial / 'dev_episodes.jsonl')
                assert len(dev_rows) == len(pilot.MILESTONES) * len(dev_seeds)
                source_run = (pilot.V20_TRAIN if arm == 'C20' else pilot.PRIOR_TRAIN
                              if arm in ('C19', 'D19') else pilot.OLDER_TRAIN)
                for milestone, summary in zip(pilot.MILESTONES, summaries):
                    assert summary['transition'] == milestone and summary['source_root'] == source_root
                    checkpoint = Path(summary['checkpoint'])
                    expected = source_run / f'{case}_{arm}_{source_root}' / f'checkpoint_{milestone}.pt'
                    assert checkpoint == expected and pilot.sha256(checkpoint) == summary['checkpoint_sha256']
                    matching = [row for row in dev_rows if row['checkpoint'] == str(milestone)]
                    assert [row['seed'] for row in matching] == list(dev_seeds)
                    assert nearly(summary['mean_H_tail_cost'], np.mean([row['objective'] for row in matching]))
                    policy = (v20.load_policy(checkpoint) if arm == 'C20' else
                              v19.load_policy(checkpoint) if arm in ('C19', 'D19') else
                              prior.load_policy(checkpoint))
                    for row in matching:
                        replay_dev_episode(policy, row)
                choice = json.loads((trial / 'selection.json').read_text())
                assert choice == min(summaries, key=lambda row: (row['mean_H_tail_cost'], row['transition']))
                selected[case, arm, root] = choice
    return selected


def audit_pair_targets(records):
    """Recompute every paired-tape target and weight independently of the trainer."""
    packed = tensorize(records)
    for r, record in enumerate(records):
        branches = record['branches']
        expected_weights = np.zeros((len(branches), len(branches)), dtype=float)
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                left = np.asarray(branches[i]['H_costs'], dtype=float) + np.asarray(branches[i]['tail_costs'], dtype=float)
                right = np.asarray(branches[j]['H_costs'], dtype=float) + np.asarray(branches[j]['tail_costs'], dtype=float)
                assert len(left) == len(right) == 16
                differences = left - right
                delta = float(differences.sum() / 16)
                se = float(np.sqrt(((differences - delta) ** 2).sum() / 15) / 4)
                expected_target = 1 / (1 + math.exp(delta / 20))
                expected_weights[i, j] = abs(delta) / (abs(delta) + 2 * se + 1e-8)
                assert abs(float(packed['pair_targets'][r, i, j]) - expected_target) < 1e-6
                assert bool(packed['pair_valid'][r, i, j])
        if expected_weights.sum() == 0:
            expected_weights[np.triu_indices(len(branches), k=1)] = 1
        assert np.allclose(packed['pair_weights'][r, :len(branches), :len(branches)].numpy(),
                           expected_weights, atol=1e-6)
        assert abs(float(packed['pair_scale'][r]) - math.log(len(branches)) / math.log(2)) < 1e-6


def audit_training(run, manifest):
    kind = manifest['kind']
    cases = (CASES[0],) if kind == 'smoke' else CASES
    arms = pilot.REPLICATION_ARMS if kind == 'replication_train' else pilot.NEW_ARMS
    roots = (pilot.SMOKE_ROOT,) if kind == 'smoke' else (
        pilot.TRAIN_ROOTS if kind == 'train' else pilot.REPLICATION_ROOTS)
    dev_seeds = pilot.SMOKE_SEEDS if kind == 'smoke' else (
        pilot.DEV_SEEDS if kind == 'train' else pilot.REPLICATION_DEV_SEEDS)
    steps = 256 if kind == 'smoke' else 8192
    expected = {(case, arm, root) for case in cases for arm in arms for root in roots}
    assert {(row['case'], row['arm'], row['train_root']) for row in manifest['trials']} == expected
    events = rows(run / 'run.log')
    assert events[0]['event'] == 'started' and events[-1]['event'] == 'completed'
    assert sum(event['event'] == 'trial_completed' for event in events) == len(expected)
    bank = pilot.label_bank(kind)
    for case in cases:
        audit_pair_targets(bank[case])
    for case, arm, root in sorted(expected):
        trial = run / f'{case}_{arm}_{root}'
        audit_new_trial(trial, case, arm, root,
                        kind if kind != 'replication_train' else 'replication',
                        steps, dev_seeds, bank[case])
    if kind == 'train':
        audit_old_references(run, dev_seeds)
    return {'trials': len(expected), 'training_H_decisions': len(expected) * steps,
            'new_dev_episodes': len(expected) * len(dev_seeds) * (1 if kind == 'smoke' else 3),
            'old_reference_dev_episodes': 0 if kind != 'train' else len(CASES) * len(pilot.OLD_ARMS) *
                                          len(roots) * len(dev_seeds) * 3}


def selected_policies(training_run, panel):
    return {(case, arm, root): (policy, label)
            for case, arm, root, policy, label in pilot.selected_policies(
                training_run, 'heldout' if panel == 'heldout' else 'replication_eval')}


def audit_episodes(run, manifest):
    panel = manifest['kind']
    seeds = pilot.HELDOUT_SEEDS if panel == 'heldout' else pilot.REPLICATION_EVAL_SEEDS
    training_run = Path(manifest['training_run'])
    source = json.loads((training_run / 'manifest.json').read_text())
    assert source['status'] == 'COMPLETED' and json.loads((training_run / 'audit.json').read_text())['status'] == 'PASS'
    for key in ('source_sha256', 'protocol_sha256', 'config_sha256', 'labels_sha256'):
        assert source[key] == manifest[key]
    policies = selected_policies(training_run, panel)
    episode_rows = rows(run / f'{panel}_episodes.jsonl')
    decisions = rows(run / f'{panel}_decisions.jsonl')
    assert len(episode_rows) == manifest['learned_episodes'] == len(policies) * len(seeds)
    assert len(decisions) == len(episode_rows) * HORIZON
    observed = set()
    cursor = 0
    for row in episode_rows:
        case, arm, seed = row['case'], row['arm'], row['seed']
        root = int(row['checkpoint'].split(':')[0])
        key = case, arm, root, seed
        assert key not in observed and seed in seeds
        observed.add(key)
        policy, label = policies[case, arm, root]
        assert row['checkpoint'] == label
        env = ScaledCoordinationEnv(configs()[case])
        env.reset(seed)
        passes = cardinality = 0
        for tick in range(HORIZON):
            trace = decisions[cursor]; cursor += 1
            assert (trace['case'], trace['arm'], trace['seed'], trace['tick']) == (case, arm, seed, tick)
            choice = prior.choose(policy, env, True)
            assert list(choice['submissions']) == trace['submissions']
            assert choice['matching_index'] == trace['matching_index']
            assert [list(pair) for pair in env.actions[choice['matching_index']]] == trace['matching']
            assert (list(choice.get('tokens')) if choice.get('tokens') is not None else None) == trace['sequence']
            assert nearly(choice['old_logprob'], trace['logprob'])
            assert choice.get('decoder_forward_passes', 1) == trace['decoder_forward_passes']
            reward, _ = env.step(choice['submissions'], choice['matching_index'])
            assert nearly(reward, trace['reward'])
            passes += trace['decoder_forward_passes']
            cardinality += len(env.actions[choice['matching_index']])
        check_metrics(row, env)
        assert row['physical_seed'] == role_seed(seed, BASE_VERSION + ':physical')
        assert row['decoder_forward_passes'] == passes
        assert row['matching_cardinality_sum'] == cardinality
    assert cursor == len(decisions)
    assert observed == {(case, arm, root, seed) for case, arm, root in policies for seed in seeds}
    controls = rows(run / f'{panel}_controls.jsonl')
    traces = rows(run / f'{panel}_control_decisions.jsonl')
    assert len(controls) == manifest['control_episodes'] == len(CASES) * 2 * len(seeds)
    assert len(traces) == len(controls) * HORIZON
    control_cursor = 0
    for row in controls:
        case, arm, seed = row['case'], row['arm'], row['seed']
        env = ScaledCoordinationEnv(configs()[case]); env.reset(seed)
        controller = RuleController(arm[2:])
        for tick in range(HORIZON):
            trace = traces[control_cursor]; control_cursor += 1
            assert (trace['case'], trace['arm'], trace['seed'], trace['tick']) == (case, arm, seed, tick)
            submissions, matching = controller.choose(env.config, env.inner.state, tick,
                                                      case=case, seed=seed)
            index = env.actions.index(matching)
            assert list(submissions) == trace['submissions'] and index == trace['matching_index']
            reward, done = env.step(submissions, index)
            assert nearly(reward, trace['reward']) and done == trace['done']
        check_metrics(row, env)
        assert row['forecast_transitions'] == controller.forecast_transitions
    assert control_cursor == len(traces)
    return episode_rows, {'learned_episodes_replayed': len(episode_rows),
                          'decision_rows_replayed': len(decisions),
                          'control_episodes_checked': len(controls),
                          'control_decisions_replayed': len(traces)}


def contrast(data, right, panel):
    seeds = pilot.HELDOUT_SEEDS if panel == 'heldout' else pilot.REPLICATION_EVAL_SEEDS
    roots = pilot.TRAIN_ROOTS if panel == 'heldout' else pilot.REPLICATION_ROOTS
    bootstrap = pilot.BOOTSTRAP_SEED if panel == 'heldout' else pilot.REPLICATION_BOOTSTRAP_SEED
    values = {(row['seed'], row['case'], row['arm'], int(row['checkpoint'].split(':')[0])): row['objective']
              for row in data}
    cube = np.array([[[values[seed, case, 'C21', root] - values[seed, case, right, root]
                       for root in roots] for case in CASES] for seed in seeds], dtype=float)
    draws = np.random.default_rng(bootstrap).integers(0, len(seeds), (5000, len(seeds)))
    interval = np.quantile(cube.mean((1, 2))[draws].mean(1), [.025, .975])
    return {'overall': float(cube.mean()),
            'case_means': {case: float(cube[:, i, :].mean()) for i, case in enumerate(CASES)},
            'training_root_means': {str(root): float(cube[:, :, i].mean()) for i, root in enumerate(roots)},
            'physical_cluster_95_interval': [float(interval[0]), float(interval[1])]}


def gate(data, controls, panel):
    names = (('L21', 'D21', 'CF21', 'C20', 'C19', 'D19', 'B18', 'S18', 'G18', 'P18', 'Q18')
             if panel == 'heldout' else ('L21', 'D21', 'CF21'))
    contrasts = {name: contrast(data, name, panel) for name in names}
    neg_cases = lambda c: all(value < 0 for value in c['case_means'].values())
    upper_neg = lambda c: c['physical_cluster_95_interval'][1] < 0
    if panel == 'heldout':
        loss, trained, same_weight = (contrasts[name] for name in ('L21', 'D21', 'CF21'))
        control = {(row['case'], row['seed']): row['objective'] for row in controls if row['arm'] == 'F_rollout'}
        f2 = float(np.mean([row['objective'] - control[row['case'], row['seed']]
                            for row in data if row['arm'] == 'C21']))
        conditions = {
            'C21_minus_L21_at_most_minus_10': loss['overall'] <= -10,
            'C21_minus_L21_both_cases_negative': neg_cases(loss),
            'C21_minus_L21_cluster_upper_negative': upper_neg(loss),
            'C21_minus_D21_at_most_minus_20': trained['overall'] <= -20,
            'C21_minus_D21_both_cases_negative': neg_cases(trained),
            'C21_minus_D21_two_roots_negative': sum(x < 0 for x in trained['training_root_means'].values()) >= 2,
            'C21_minus_D21_cluster_upper_negative': upper_neg(trained),
            'C21_minus_CF21_at_most_minus_10': same_weight['overall'] <= -10,
            'C21_minus_CF21_both_cases_negative': neg_cases(same_weight),
            'C21_minus_CF21_cluster_upper_negative': upper_neg(same_weight),
            'C21_minus_C19_negative': contrasts['C19']['overall'] < 0,
            'C21_minus_G18_negative': contrasts['G18']['overall'] < 0,
            'C21_minus_B18_negative': contrasts['B18']['overall'] < 0,
            'C21_minus_S18_negative': contrasts['S18']['overall'] < 0,
            'C21_minus_F2_at_most_plus_20': f2 <= 20,
        }
    else:
        conditions = {f'C21_minus_{name}_negative': contrasts[name]['overall'] < 0 for name in names}
        conditions.update({f'C21_minus_{name}_both_cases_negative': neg_cases(contrasts[name]) for name in names})
        conditions.update({f'C21_minus_{name}_cluster_upper_negative': upper_neg(contrasts[name]) for name in names})
        f2 = None
    return {'panel': panel, 'contrasts': contrasts, 'conditions': conditions,
            'C21_minus_F2_overall': f2, 'go': all(conditions.values()),
            'protocol_sha256': pilot.sha256(pilot.PROTOCOL)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    pilot.LABEL = Path(manifest['label_run'])
    audit_dir = run.parent / ('same_tick_v21_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    audit_dir.mkdir(exist_ok=False)
    if (run / 'audit.json').exists():
        shutil.copy2(run / 'audit.json', audit_dir / 'previous_audit.json')
    result = {'status': 'RUNNING', 'kind': manifest['kind'], 'source_run': str(run),
              'audit_artifact': str(audit_dir), 'manifest_sha256': pilot.sha256(run / 'manifest.json'),
              'auditor_sha256': pilot.sha256(Path(__file__))}
    try:
        assert manifest['status'] == 'COMPLETED'
        identity(run, manifest)
        if manifest['kind'] in ('smoke', 'train', 'replication_train'):
            result.update(audit_training(run, manifest))
        else:
            data, counts = audit_episodes(run, manifest)
            result.update(counts)
            decision = gate(data, rows(run / f"{manifest['kind']}_controls.jsonl"), manifest['kind'])
            path = run / 'replication_go_decision.json'
            if path.exists():
                shutil.copy2(path, audit_dir / 'previous_decision.json')
            pilot.write_json(audit_dir / 'replication_go_decision.json', decision)
            pilot.write_json(path, decision)
            result['decision_sha256'] = pilot.sha256(path)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        pilot.write_json(audit_dir / 'audit.json', result)
        pilot.write_json(run / 'audit.json', result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
