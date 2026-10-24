"""Prelocked v19 matching-value rehearsal pilot. There is no TEST mode."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv
from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.same_tick_policy_v15 import SequenceTransition
from experiments.scaled_n5k3_v12_config import CASES, CONFIG_SEEDS, configs, evaluate_controls
from experiments.full_matching_pretrain_v17 import pretrain, tensorize
from experiments.matching_value_rehearsal_v19 import LabelCycle, ppo_update
from experiments import same_tick_pilot_v18 as prior
from experiments import full_matching_labels_v17 as old_labeler
from experiments import full_matching_labels_v18 as labeler

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/matching_value_retention_v19_prelock.md'
LABEL = ROOT / 'artifacts/full_matching_v18_labels_20261024T035233500615Z'
PRIOR_TRAIN = ROOT / 'artifacts/same_tick_v18_train_20261024T045926267434Z'
NEW_ARMS = ('C19', 'D19')
OLD_ARMS = ('C18', 'D18', 'B18', 'S18', 'G18', 'P18', 'Q18')
REPLICATION_ARMS = ('C19', 'D19', 'C0', 'D0')
TRAIN_ROOTS = (80710000, 80710001, 80710002)
REPLICATION_ROOTS = (80820000, 80820001, 80820002)
SMOKE_ROOT = 80800000
SMOKE_SEEDS = (80800011, 80800012)
DEV_SEEDS = tuple(range(80811000, 80811016))
HELDOUT_SEEDS = tuple(range(80812000, 80812032))
BOOTSTRAP_SEED = 80813000
REPLICATION_DEV_SEEDS = tuple(range(80821000, 80821016))
REPLICATION_EVAL_SEEDS = tuple(range(80822000, 80822032))
REPLICATION_BOOTSTRAP_SEED = 80823000
SEALED_TEST = tuple(range(80830000, 80830032))
MILESTONES = (2048, 4096, 8192)
TRIAL_TIMEOUT = 7200

write_json = prior.write_json
append_jsonl = prior.append_jsonl
sha256 = prior.sha256


def git_output(*args):
    result = subprocess.run(('git', *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def actor_kind(arm):
    return 'C' if arm in ('C19', 'C0') else 'D'


def new_policy(arm):
    if arm not in (*NEW_ARMS, 'C0', 'D0'):
        raise ValueError(arm)
    return DynamicSequencePolicy(actor_kind(arm) + '16')


def load_policy(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    policy = new_policy(payload['arm'])
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    return policy


def source_paths():
    fixed = [ROOT / name for name in (
        'experiments/same_tick_pilot_v19.py', 'experiments/audit_same_tick_v19.py',
        'experiments/matching_value_rehearsal_v19.py', 'experiments/same_tick_pilot_v18.py',
        'experiments/full_matching_pretrain_v17.py', 'experiments/full_matching_labels_v18.py',
        'experiments/full_matching_labels_v17.py', 'experiments/scaled_n5k3_v12_config.py',
        'pyproject.toml')]
    if not all(path.is_file() for path in fixed):
        raise RuntimeError('v19 runner and independent auditor required')
    return sorted((ROOT / 'src').rglob('*.py')) + fixed


def seed_registry_check():
    prior.seed_registry_check()
    groups = ((SMOKE_ROOT,), SMOKE_SEEDS, DEV_SEEDS, HELDOUT_SEEDS,
              (BOOTSTRAP_SEED,), REPLICATION_ROOTS, REPLICATION_DEV_SEEDS,
              REPLICATION_EVAL_SEEDS, (REPLICATION_BOOTSTRAP_SEED,), SEALED_TEST)
    raw = [seed for group in groups for seed in group]
    if len(raw) != len(set(raw)) or not all(80800000 <= seed < 80900000 for seed in raw):
        raise RuntimeError('v19 raw seed roles overlap')
    old = old_labeler.seed_registry_check('labels') | old_labeler.seed_registry_check('smoke')
    label_rows = [json.loads(line) for line in (LABEL / 'labels.jsonl').read_text().splitlines()]
    old |= {labeler.extra_seed(row, slot) for row in label_rows for slot in labeler.EXTRA_SLOTS}
    prior_raw = (*prior.SMOKE_SEEDS, *prior.DEV_SEEDS, *prior.HELDOUT_SEEDS,
                 *prior.REPL_DEV_SEEDS, *prior.REPL_EVAL_SEEDS, *prior.SEALED_TEST)
    old |= {role_seed(seed, BASE_VERSION + ':physical') for seed in prior_raw}
    current = set()
    for seed in (*SMOKE_SEEDS, *DEV_SEEDS, *HELDOUT_SEEDS, *REPLICATION_DEV_SEEDS,
                 *REPLICATION_EVAL_SEEDS, *SEALED_TEST):
        physical = role_seed(seed, BASE_VERSION + ':physical')
        if physical in old or physical in current:
            raise RuntimeError('v19 evaluation physical tape collision')
        current.add(physical)
    for root in (SMOKE_ROOT, *REPLICATION_ROOTS):
        for case in CASES:
            for episode in range(200):
                namespace = 'smoke' if root == SMOKE_ROOT else 'replication'
                reset = role_seed(root, f'same_tick_v19:{namespace}:{case}:{episode}')
                physical = role_seed(reset, BASE_VERSION + ':physical')
                if physical in old or physical in current:
                    raise RuntimeError('v19 training physical tape collision')
                current.add(physical)


def label_bank():
    manifest = json.loads((LABEL / 'manifest.json').read_text())
    audit = json.loads((LABEL / 'audit.json').read_text())
    if (manifest['kind'] != 'labels' or manifest['status'] != 'COMPLETED'
            or audit['status'] != 'PASS' or not audit['qualified']
            or any(sha256(LABEL / path) != expected for path, expected in manifest['output_sha256'].items())
            or any(sha256(ROOT / path) != expected for path, expected in manifest['source_sha256'].items())):
        raise RuntimeError('immutable audited v18 training label bank required')
    rows = [json.loads(line) for line in (LABEL / 'labels.jsonl').read_text().splitlines()]
    by_case = {case: [row for row in rows if row['case'] == case] for case in CASES}
    if [len(by_case[case]) for case in CASES] != [273, 235]:
        raise RuntimeError('v19 training label case coverage changed')
    return by_case


def old_train_check():
    manifest = json.loads((PRIOR_TRAIN / 'manifest.json').read_text())
    audit = json.loads((PRIOR_TRAIN / 'audit.json').read_text())
    if (manifest['kind'] != 'train' or manifest['status'] != 'COMPLETED'
            or audit['status'] != 'PASS' or manifest['labels_sha256'] != sha256(LABEL / 'labels.jsonl')
            or manifest['config_sha256'] != config_hashes()
            or any(sha256(ROOT / path) != expected for path, expected in manifest['source_sha256'].items())):
        raise RuntimeError('audited v18 train/DEV source required')
    return manifest


def config_hashes():
    return {case: hashlib.sha256(json.dumps(asdict(configs()[case]), sort_keys=True).encode()).hexdigest()
            for case in CASES}


def new_run(root, kind):
    root.mkdir(parents=True, exist_ok=True)
    run = root / ('same_tick_v19_' + kind + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(exist_ok=False)
    manifest = {'status': 'RUNNING', 'kind': kind, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'platform': platform.platform(), 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
                'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--porcelain'),
                'protocol_sha256': sha256(PROTOCOL),
                'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()},
                'config_sha256': config_hashes(), 'config_seeds': CONFIG_SEEDS,
                'label_run': str(LABEL), 'label_manifest_sha256': sha256(LABEL / 'manifest.json'),
                'label_audit_sha256': sha256(LABEL / 'audit.json'),
                'labels_sha256': sha256(LABEL / 'labels.jsonl'),
                'prior_train_run': str(PRIOR_TRAIN), 'prior_train_manifest_sha256': sha256(PRIOR_TRAIN / 'manifest.json'),
                'prior_train_audit_sha256': sha256(PRIOR_TRAIN / 'audit.json'),
                'cases': CASES, 'new_arms': NEW_ARMS, 'old_arms': OLD_ARMS,
                'train_roots': TRAIN_ROOTS, 'replication_roots': REPLICATION_ROOTS,
                'dev_seeds': DEV_SEEDS, 'heldout_seeds': HELDOUT_SEEDS,
                'bootstrap_seed': BOOTSTRAP_SEED, 'replication_dev_seeds': REPLICATION_DEV_SEEDS,
                'replication_eval_seeds': REPLICATION_EVAL_SEEDS,
                'replication_bootstrap_seed': REPLICATION_BOOTSTRAP_SEED,
                'sealed_test_seeds': SEALED_TEST, 'smoke_root': SMOKE_ROOT,
                'smoke_seeds': SMOKE_SEEDS, 'milestones': MILESTONES,
                'trial_timeout_seconds': TRIAL_TIMEOUT, 'rehearsal_weight': .5,
                'rehearsal_label_batch_size': 32}
    write_json(run / 'manifest.json', manifest)
    append_jsonl(run / 'run.log', {'event': 'started', 'kind': kind, 'utc': manifest['started_utc']})
    return run, manifest


def reset_seed(root, case, episode, kind):
    if kind == 'train':
        return role_seed(root, f'same_tick_v18:{case}:{episode}')
    return role_seed(root, f'same_tick_v19:{kind}:{case}:{episode}')


def train_trial(run, case, arm, root, records, *, kind, steps, dev_seeds):
    trial = run / f'{case}_{arm}_{root}'
    trial.mkdir(exist_ok=False)
    torch.manual_seed(root)
    np.random.seed(root % 2**32)
    policy = new_policy(arm)
    env = ScaledCoordinationEnv(configs()[case])
    episode = 0
    env.reset(reset_seed(root, case, episode, kind))
    started = time.monotonic()
    deadline = started + TRIAL_TIMEOUT
    record = {'status': 'RUNNING', 'case': case, 'arm': arm, 'train_root': root,
              'requested_H_decisions': steps, 'actual_H_decisions': 0,
              'decoder_forward_passes': 0, 'rehearsal_candidate_paths': 0,
              'rehearsal_token_positions': 0, 'rehearsal_minibatches': 0,
              'forecast_transitions': 0, 'actor_parameters': policy.actor_parameters,
              'critic_parameters': policy.critic_parameters, 'rehearsal_weight': 0. if arm in ('C0', 'D0') else .5,
              'ppo': {'n_steps': 256, 'minibatch': 64, 'epochs': 4,
                      'learning_rate': 3e-4, 'gamma': .99, 'gae_lambda': .95,
                      'clip': .2, 'entropy_coef': .01}}
    write_json(trial / 'manifest.json', record)
    try:
        old_arm = actor_kind(arm) + '18'
        if kind == 'train':
            preseed = role_seed(root, f'v18-pretrain:{case}:{old_arm}')
            epochs = 96
        elif kind == 'smoke':
            preseed = role_seed(root, f'v19-smoke-pretrain:{case}:{arm}')
            epochs = 2
        else:
            preseed = role_seed(root, f'v19-replication-pretrain:{case}:{actor_kind(arm)}')
            epochs = 96
        details = pretrain(policy, records, preseed, epochs=epochs,
                           on_epoch=lambda row: append_jsonl(trial / 'pretrain_epochs.jsonl', row))
        record.update(details, pretraining_seed=preseed,
                      pretraining_label_bank='v18_sixteen_tape')
        if kind == 'train':
            old_checkpoint = PRIOR_TRAIN / f'{case}_{old_arm}_{root}' / 'pretrain_checkpoint.pt'
            old_manifest = json.loads((old_checkpoint.parent / 'manifest.json').read_text())
            if old_manifest['pretrain_checkpoint_sha256'] != sha256(old_checkpoint):
                raise RuntimeError('audited original pretraining checkpoint hash changed')
            old = torch.load(old_checkpoint, map_location='cpu', weights_only=True)
            if not all(torch.equal(value, old['state_dict'][key])
                       for key, value in policy.state_dict().items()):
                raise RuntimeError('v19 pretraining failed exact v18 state replay')
            record['source_pretrain_checkpoint_sha256'] = sha256(old_checkpoint)
        record['torch_rng_after_pretrain_sha256'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        pretrain_checkpoint = trial / 'pretrain_checkpoint.pt'
        torch.save({'arm': arm, 'case': case, 'train_root': root,
                    'state_dict': policy.state_dict()}, pretrain_checkpoint)
        record['pretrain_checkpoint_sha256'] = sha256(pretrain_checkpoint)
        write_json(trial / 'manifest.json', record)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        tensor_bank = tensorize(records)
        cycle_seed = role_seed(root, f'v19-label-cycle:{case}:{actor_kind(arm)}')
        cycle = LabelCycle(len(records), cycle_seed)
        record['label_cycle_seed'] = cycle_seed
        milestones = (steps,) if kind == 'smoke' else MILESTONES
        summaries = []
        with tqdm(total=steps, desc=f'v19 {kind} {case} {arm} {root}') as bar:
            for milestone in milestones:
                while record['actual_H_decisions'] < milestone:
                    trajectory = []
                    for _ in range(min(256, milestone - record['actual_H_decisions'])):
                        if time.monotonic() > deadline:
                            raise TimeoutError('v19 trial timeout during PPO')
                        tick = env.tick
                        choice = prior.choose(policy, env, False)
                        reward, done = env.step(choice['submissions'], choice['matching_index'])
                        append_jsonl(trial / 'training_decisions.jsonl', {
                            'case': case, 'arm': arm, 'train_root': root, 'episode': episode,
                            'tick': tick, 'reset_seed': reset_seed(root, case, episode, kind),
                            'submissions': choice['submissions'],
                            'matching_index': choice['matching_index'],
                            'sequence': choice['tokens'], 'logprob': choice['old_logprob'],
                            'reward': reward, 'done': done,
                            'decoder_forward_passes': choice['decoder_forward_passes']})
                        trajectory.append(SequenceTransition(
                            choice['pre_obs'], choice['machine_obs'], choice['worker_obs'],
                            choice['edge_mask'], choice['sequence'], choice['old_logprob'],
                            choice['value'], reward, done))
                        record['decoder_forward_passes'] += choice['decoder_forward_passes']
                        record['actual_H_decisions'] += 1
                        bar.update(1)
                        if done:
                            append_jsonl(trial / 'training_episodes.jsonl',
                                         {'episode': episode, 'transition': record['actual_H_decisions'],
                                          **env.metrics()})
                            episode += 1
                            env.reset(reset_seed(root, case, episode, kind))
                    with torch.no_grad():
                        bootstrap = float(policy.value(torch.from_numpy(env.pre_observation())))
                    update_step = record['actual_H_decisions']
                    update = ppo_update(policy, optimizer, trajectory, bootstrap, tensor_bank, cycle,
                        weight=record['rehearsal_weight'],
                        on_minibatch=lambda row: append_jsonl(trial / 'rehearsal_minibatches.jsonl',
                            {'transition': update_step, **row}))
                    for key in ('rehearsal_candidate_paths', 'rehearsal_token_positions', 'rehearsal_minibatches'):
                        record[key] += update.get(key, 0)
                    append_jsonl(trial / 'updates.jsonl', {'transition': update_step, **update})
                    write_json(trial / 'manifest.json', record)
                checkpoint = trial / f'checkpoint_{milestone}.pt'
                torch.save({'arm': arm, 'case': case, 'train_root': root,
                            'transition': milestone, 'state_dict': policy.state_dict()}, checkpoint)
                restored = load_policy(checkpoint)
                dev_rows = prior.evaluate_policy(restored, arm, case, dev_seeds,
                    trial / 'dev_episodes.jsonl', label=str(milestone), deadline=deadline)
                summaries.append({'transition': milestone,
                                  'mean_H_tail_cost': float(np.mean([row['objective'] for row in dev_rows])),
                                  'checkpoint': checkpoint.name,
                                  'checkpoint_sha256': sha256(checkpoint)})
                write_json(trial / 'dev_milestones.json', summaries)
        selected = min(summaries, key=lambda row: (row['mean_H_tail_cost'], row['transition']))
        write_json(trial / 'selection.json', selected)
        record.update(status='COMPLETED', selected_checkpoint=selected['checkpoint'],
                      dev_mean=selected['mean_H_tail_cost'])
    except Exception as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(trial / 'manifest.json', record)
    return record


def reselect_old_controls(run, dev_seeds):
    entries = []
    for case in CASES:
        for arm in OLD_ARMS:
            for root in TRAIN_ROOTS:
                old_trial = PRIOR_TRAIN / f'{case}_{arm}_{root}'
                old_milestones = json.loads((old_trial / 'dev_milestones.json').read_text())
                destination = run / 'references' / f'{case}_{arm}_{root}'
                destination.mkdir(parents=True, exist_ok=False)
                new_milestones = []
                for old_entry in tqdm(old_milestones, desc=f'v19 reselect {case} {arm} {root}', leave=False):
                    checkpoint = old_trial / old_entry['checkpoint']
                    if sha256(checkpoint) != old_entry['checkpoint_sha256']:
                        raise RuntimeError('v18 reference checkpoint changed')
                    policy = prior.load_policy(checkpoint)
                    rows = prior.evaluate_policy(policy, arm, case, dev_seeds,
                        destination / 'dev_episodes.jsonl', label=str(old_entry['transition']))
                    new_milestones.append({'transition': old_entry['transition'],
                                           'mean_H_tail_cost': float(np.mean([row['objective'] for row in rows])),
                                           'checkpoint': str(checkpoint.resolve()),
                                           'checkpoint_sha256': sha256(checkpoint)})
                write_json(destination / 'dev_milestones.json', new_milestones)
                selected = min(new_milestones,
                               key=lambda row: (row['mean_H_tail_cost'], row['transition']))
                write_json(destination / 'selection.json', selected)
                entries.append({'case': case, 'arm': arm, 'train_root': root,
                                'selected': selected})
    return entries


def selected_policies(training_run, kind):
    roots = TRAIN_ROOTS if kind == 'heldout' else REPLICATION_ROOTS
    arms = NEW_ARMS if kind == 'heldout' else REPLICATION_ARMS
    for case in CASES:
        for root in roots:
            for arm in arms:
                trial = training_run / f'{case}_{arm}_{root}'
                selected = json.loads((trial / 'selection.json').read_text())
                checkpoint = trial / selected['checkpoint']
                if sha256(checkpoint) != selected['checkpoint_sha256']:
                    raise RuntimeError('selected v19 checkpoint hash changed')
                policy = load_policy(checkpoint)
                yield case, arm, root, policy, f"{root}:{selected['transition']}"
                if arm == 'C19':
                    frozen = DynamicSequencePolicy('D16')
                    frozen.load_state_dict(policy.state_dict(), strict=True)
                    frozen.eval()
                    yield case, 'CF19', root, frozen, f"{root}:{selected['transition']}"
            if kind == 'heldout':
                for arm in OLD_ARMS:
                    ref = training_run / 'references' / f'{case}_{arm}_{root}'
                    selected = json.loads((ref / 'selection.json').read_text())
                    checkpoint = Path(selected['checkpoint'])
                    if sha256(checkpoint) != selected['checkpoint_sha256']:
                        raise RuntimeError('selected v18 reference checkpoint hash changed')
                    yield case, arm, root, prior.load_policy(checkpoint), f"{root}:{selected['transition']}"


def evaluate_selected(run, training_run, kind):
    manifest = json.loads((training_run / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or manifest['kind'] != ('train' if kind == 'heldout' else 'replication_train'):
        raise RuntimeError('wrong or incomplete training run')
    audit = json.loads((training_run / 'audit.json').read_text())
    if audit['status'] != 'PASS':
        raise RuntimeError('passing independent training audit required')
    own = json.loads((run / 'manifest.json').read_text())
    for key in ('source_sha256', 'protocol_sha256', 'config_sha256', 'labels_sha256'):
        if manifest[key] != own[key]:
            raise RuntimeError(f'v19 source/input changed since training: {key}')
    if kind == 'heldout':
        qualification_path = training_run / 'retention_go_decision.json'
        qualification = json.loads(qualification_path.read_text())
        if (qualification['go'] is not True
                or qualification['protocol_sha256'] != own['protocol_sha256']
                or audit.get('decision_sha256') != sha256(qualification_path)):
            raise RuntimeError('audited training-bank retention gate did not pass')
        seeds = HELDOUT_SEEDS
        panel = 'heldout'
    else:
        seeds = REPLICATION_EVAL_SEEDS
        panel = 'replication_eval'
    count = 0
    for case, arm, root, policy, label in selected_policies(training_run, kind):
        count += len(prior.evaluate_policy(policy, arm, case, seeds,
            run / f'{panel}_episodes.jsonl', label=label,
            trace_path=run / f'{panel}_decisions.jsonl'))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'train', 'heldout', 'replication_train', 'replication_eval'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--pilot-heldout-run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_registry_check()
    label_rows = label_bank()
    old_train_check()
    run, manifest = new_run(args.output_root, args.mode)
    print(run.resolve(), flush=True)
    started = time.monotonic()
    try:
        if args.mode != 'smoke' and manifest['git_status']:
            raise RuntimeError('committed clean v19 source required')
        if args.mode in ('smoke', 'train', 'replication_train'):
            if args.mode == 'replication_train':
                if args.pilot_heldout_run is None:
                    raise RuntimeError('audited held-out gate required for replication')
                heldout = args.pilot_heldout_run.resolve()
                decision_path = heldout / 'replication_go_decision.json'
                decision = json.loads(decision_path.read_text())
                heldout_audit = json.loads((heldout / 'audit.json').read_text())
                if (heldout_audit['status'] != 'PASS'
                        or heldout_audit.get('decision_sha256') != sha256(decision_path)
                        or decision['go'] is not True
                        or decision['protocol_sha256'] != manifest['protocol_sha256']):
                    raise RuntimeError('v19 independent replication gate did not pass')
                manifest['pilot_heldout_run'] = str(heldout)
            roots = (SMOKE_ROOT,) if args.mode == 'smoke' else (TRAIN_ROOTS if args.mode == 'train' else REPLICATION_ROOTS)
            cases = (CASES[0],) if args.mode == 'smoke' else CASES
            arms = NEW_ARMS if args.mode != 'replication_train' else REPLICATION_ARMS
            dev_seeds = SMOKE_SEEDS if args.mode == 'smoke' else (DEV_SEEDS if args.mode == 'train' else REPLICATION_DEV_SEEDS)
            steps = 256 if args.mode == 'smoke' else 8192
            trials = []
            for case in cases:
                for arm in arms:
                    for root in roots:
                        record = train_trial(run, case, arm, root, label_rows[case],
                                             kind=args.mode if args.mode != 'replication_train' else 'replication',
                                             steps=steps, dev_seeds=dev_seeds)
                        trials.append(record)
                        manifest['trials'] = trials
                        write_json(run / 'manifest.json', manifest)
                        append_jsonl(run / 'run.log', {'event': 'trial_completed', 'case': case,
                                                      'arm': arm, 'train_root': root,
                                                      'utc': datetime.now(timezone.utc).isoformat()})
            if args.mode == 'train':
                manifest['old_reference_selections'] = reselect_old_controls(run, DEV_SEEDS)
        else:
            if args.training_run is None:
                raise RuntimeError('--training-run required for evaluation')
            training_run = args.training_run.resolve()
            manifest['training_run'] = str(training_run)
            manifest['learned_episodes'] = evaluate_selected(run, training_run, args.mode)
            panel = args.mode
            seeds = HELDOUT_SEEDS if panel == 'heldout' else REPLICATION_EVAL_SEEDS
            controls = evaluate_controls(run, seeds, panel=panel)
            manifest['control_episodes'] = len(controls)
        manifest['status'] = 'COMPLETED'
        append_jsonl(run / 'run.log', {'event': 'completed', 'utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error))
        append_jsonl(run / 'run.log', {'event': 'failed', 'error': repr(error),
                                      'utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {str(path.relative_to(run)): sha256(path)
                                     for path in run.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        write_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
