"""Post-pilot threshold-requester/learned-coordinator attribution diagnostic.

Pilot v1 source, checkpoints and sealed TEST roles remain untouched.
"""
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
from ht_pdm_fjsp.maintenance_framework_v1 import configs
from maintenance_capacity_coordination.coordination_v1 import CoordinationEnv, BASE_VERSION
from maintenance_capacity_coordination.pilot import (
    ARMS, CASES, DEV_SEEDS, MILESTONES, ROOT, TRAIN_SEEDS, TRIAL_TIMEOUT,
    append_jsonl, evaluate_policy, load_policy, sha256, write_json,
)
from maintenance_capacity_coordination.pilot_policy import PilotPolicy, Transition, ppo_update

PROTOCOL = ROOT / 'docs/threshold_requester_diagnostic_v1.md'
SCRIPT = Path(__file__).resolve()
PANEL_SEEDS = tuple(range(73160000, 73160032))
ANALYSIS_SEED = 73161000


def git_output(*args):
    result = subprocess.run(('git', *args), cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes():
    files = sorted((ROOT / 'src').rglob('*.py')) + [SCRIPT, ROOT / 'pyproject.toml']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def create_run(output_root, mode):
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / ('threshold_control_v1_' + mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(exist_ok=False)
    cfg = configs()
    manifest = {
        'status': 'RUNNING', 'mode': mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
        'git_commit': git_output('rev-parse', 'HEAD'), 'git_status': git_output('status', '--porcelain'),
        'protocol_sha256': sha256(PROTOCOL), 'source_sha256': source_hashes(),
        'config_sha256': {case: hashlib.sha256(json.dumps(asdict(cfg[case]), sort_keys=True).encode()).hexdigest()
                          for case in CASES},
        'cases': CASES, 'arm': 'G', 'reference_arms': ARMS,
        'train_seeds': TRAIN_SEEDS, 'dev_seeds': DEV_SEEDS, 'diagnostic_seeds': PANEL_SEEDS,
        'milestones': MILESTONES, 'trial_timeout_seconds': TRIAL_TIMEOUT,
        'forecast_transitions': 0,
    }
    write_json(run / 'manifest.json', manifest)
    return run, manifest


def choose_g(policy, env, deterministic):
    """Threshold submissions are fixed; only masked matching is sampled."""
    if policy.arm != 'D':
        raise ValueError('G uses the coordinator-only D architecture')
    pre_obs = env.pre_observation()
    eligible = env.request_mask()
    submissions = env.fixed_submissions('threshold')
    bits = np.asarray([float(m in submissions) for m in range(env.config.machines)], dtype=np.float32)
    coordinator_obs = env.coordinator_observation(submissions)
    matching_mask = env.matching_mask(submissions)
    with torch.no_grad():
        index = policy.sample_matching(torch.from_numpy(coordinator_obs), torch.from_numpy(matching_mask), deterministic)
        logprob, _ = policy.logprob_entropy(
            None, torch.from_numpy(eligible), torch.from_numpy(bits),
            torch.from_numpy(coordinator_obs), torch.from_numpy(matching_mask), index)
        value = policy.value(torch.from_numpy(pre_obs))
    return {
        'pre_obs': pre_obs, 'machine_obs': None, 'eligible': eligible, 'request_bits': bits,
        'submissions': submissions, 'coordinator_obs': coordinator_obs,
        'matching_mask': matching_mask, 'matching_index': int(index),
        'old_logprob': float(logprob), 'value': float(value),
    }


def evaluate_g(policy, case, seeds, episode_path=None, *, label=None, deadline=None):
    env = CoordinationEnv(configs()[case])
    rows = []
    for seed in tqdm(seeds, desc=f'Evaluate G {case} {label}', leave=False):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('trial timeout during evaluation')
        env.reset(seed)
        for _ in range(48):
            decision = choose_g(policy, env, deterministic=True)
            _, done = env.step(decision['submissions'], decision['matching_index'])
        assert done
        row = {'case': case, 'arm': 'G', 'seed': seed,
               'physical_seed': role_seed(seed, BASE_VERSION + ':physical'),
               'checkpoint': label, **env.metrics()}
        if row['invalid_assignments'] != 0 or abs(row['episode_return'] + row['objective']) > 1e-8:
            raise RuntimeError('G feasibility/accounting gate')
        rows.append(row)
        if episode_path is not None:
            append_jsonl(episode_path, row)
    return rows


def train_trial(run, case, seed, *, steps=8192, eval_seeds=DEV_SEEDS):
    trial = run / f'{case}_G_{seed}'
    trial.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    policy = PilotPolicy('D')
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    env = CoordinationEnv(configs()[case])
    episode_index = 0
    env.reset(role_seed(seed, f'pilot_train:{case}:{episode_index}'))
    started = time.monotonic()
    deadline = started + TRIAL_TIMEOUT
    record = {'status': 'RUNNING', 'case': case, 'arm': 'G', 'train_seed': seed,
              'requested_transitions': steps, 'actual_transitions': 0,
              'parameter_count': policy.parameter_count, 'forecast_transitions': 0,
              'ppo': {'n_steps': 256, 'minibatch': 64, 'epochs': 4, 'learning_rate': 3e-4,
                      'gamma': .99, 'gae_lambda': .95, 'clip': .2, 'entropy_coef': .01}}
    write_json(trial / 'manifest.json', record)
    dev_means = []
    try:
        with tqdm(total=steps, desc=f'Train {case} G {seed}') as bar:
            milestones = (steps,) if steps < MILESTONES[0] else MILESTONES
            for milestone in milestones:
                while record['actual_transitions'] < milestone:
                    trajectory = []
                    for _ in range(min(256, milestone - record['actual_transitions'])):
                        if time.monotonic() > deadline:
                            raise TimeoutError('two-hour G trial timeout')
                        decision = choose_g(policy, env, deterministic=False)
                        reward, done = env.step(decision['submissions'], decision['matching_index'])
                        trajectory.append(Transition(
                            decision['pre_obs'], None, decision['eligible'], decision['request_bits'],
                            decision['coordinator_obs'], decision['matching_mask'], decision['matching_index'],
                            decision['old_logprob'], decision['value'], reward, done))
                        record['actual_transitions'] += 1
                        bar.update(1)
                        if done:
                            append_jsonl(trial / 'training_episodes.jsonl',
                                         {'episode': episode_index, 'transition': record['actual_transitions'],
                                          **env.metrics()})
                            episode_index += 1
                            env.reset(role_seed(seed, f'pilot_train:{case}:{episode_index}'))
                    with torch.no_grad():
                        bootstrap = float(policy.value(torch.from_numpy(env.pre_observation())))
                    update = ppo_update(policy, optimizer, trajectory, bootstrap)
                    append_jsonl(trial / 'updates.jsonl', {'transition': record['actual_transitions'], **update})
                    write_json(trial / 'manifest.json', record)
                checkpoint = trial / f'checkpoint_{milestone}.pt'
                torch.save({'arm': 'D', 'diagnostic_arm': 'G', 'case': case, 'train_seed': seed,
                            'transition': milestone, 'state_dict': policy.state_dict()}, checkpoint)
                restored = load_policy(checkpoint)
                rows = evaluate_g(restored, case, eval_seeds, trial / 'dev_episodes.jsonl',
                                  label=str(milestone), deadline=deadline)
                mean = float(np.mean([row['objective'] for row in rows]))
                dev_means.append({'transition': milestone, 'mean_H_tail_cost': mean,
                                  'checkpoint': checkpoint.name})
                write_json(trial / 'dev_milestones.json', dev_means)
        selected = min(dev_means, key=lambda x: (x['mean_H_tail_cost'], x['transition']))
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


def evaluate_panel(run, diagnostic_train, v1_train):
    source = json.loads((diagnostic_train / 'manifest.json').read_text())
    v1 = json.loads((v1_train / 'manifest.json').read_text())
    if source['status'] != 'COMPLETED' or source['mode'] != 'train' or v1['status'] != 'COMPLETED':
        raise RuntimeError('incomplete source training run')
    if source['source_sha256'] != source_hashes() or source['protocol_sha256'] != sha256(PROTOCOL):
        raise RuntimeError('G source/protocol changed since training')
    if v1['protocol_sha256'] != sha256(ROOT / 'docs/coordination_pilot_v1.md'):
        raise RuntimeError('v1 protocol changed')
    for filename, digest in v1['source_sha256'].items():
        if sha256(ROOT / filename) != digest:
            raise RuntimeError('v1 research source changed')
    counts = {'G': 0, **{arm: 0 for arm in ARMS}}
    for case in CASES:
        for seed in TRAIN_SEEDS:
            g_trial = diagnostic_train / f'{case}_G_{seed}'
            selected = json.loads((g_trial / 'selection.json').read_text())
            policy = load_policy(g_trial / selected['checkpoint'])
            rows = evaluate_g(policy, case, PANEL_SEEDS, run / 'diagnostic_episodes.jsonl',
                              label=f'{seed}:{selected["transition"]}')
            counts['G'] += len(rows)
            for arm in ARMS:
                trial = v1_train / f'{case}_{arm}_{seed}'
                selected = json.loads((trial / 'selection.json').read_text())
                policy = load_policy(trial / selected['checkpoint'])
                rows = evaluate_policy(policy, case, PANEL_SEEDS, run / 'diagnostic_episodes.jsonl',
                                       label=f'{seed}:{selected["transition"]}')
                counts[arm] += len(rows)
    return counts


def analyze_panel(run):
    rows = [json.loads(line) for line in (run / 'diagnostic_episodes.jsonl').read_text().splitlines()]
    keyed = {}
    for row in rows:
        train_seed = int(row['checkpoint'].split(':', 1)[0])
        key = (row['case'], row['arm'], train_seed, row['seed'])
        if key in keyed or row['invalid_assignments'] != 0:
            raise RuntimeError('duplicate/infeasible diagnostic episode')
        if abs(row['objective'] - row['H_cost'] - row['tail_cost']) > 1e-8:
            raise RuntimeError('H+tail mismatch')
        if abs(row['episode_return'] + row['objective']) > 1e-8:
            raise RuntimeError('return/cost mismatch')
        keyed[key] = row
    expected = {(case, arm, train_seed, seed) for case in CASES for arm in ('G', *ARMS)
                for train_seed in TRAIN_SEEDS for seed in PANEL_SEEDS}
    if set(keyed) != expected:
        raise RuntimeError(f'diagnostic coverage mismatch: {len(expected - set(keyed))} missing')
    rng = np.random.default_rng(ANALYSIS_SEED)
    comparisons = {}
    for other in ('C', 'D'):
        differences = np.asarray([[[keyed[case, 'G', train_seed, seed]['objective']
                                    - keyed[case, other, train_seed, seed]['objective']
                                    for seed in PANEL_SEEDS] for case in CASES]
                                  for train_seed in TRAIN_SEEDS], dtype=float)
        draws = np.empty(5000)
        for i in range(len(draws)):
            sampled = rng.integers(0, len(PANEL_SEEDS),
                                   size=(len(TRAIN_SEEDS), len(CASES), len(PANEL_SEEDS)))
            draws[i] = np.take_along_axis(differences, sampled, axis=2).mean()
        comparisons[other] = {
            'mean_G_minus_other': float(differences.mean()),
            'by_case': {case: float(differences[:, j, :].mean()) for j, case in enumerate(CASES)},
            'by_training_seed': {str(seed): float(differences[j].mean())
                                 for j, seed in enumerate(TRAIN_SEEDS)},
            'physical_seed_bootstrap_95': np.quantile(draws, (.025, .975)).tolist(),
        }
    means = {arm: {case: float(np.mean([keyed[case, arm, train_seed, seed]['objective']
                                       for train_seed in TRAIN_SEEDS for seed in PANEL_SEEDS]))
                   for case in CASES} for arm in ('G', *ARMS)}
    result = {'status': 'AUDITED', 'scope': 'post hoc attribution diagnostic, not pilot v1 confirmation',
              'episode_count': len(rows), 'cases': CASES, 'training_seeds': TRAIN_SEEDS,
              'diagnostic_physical_seeds': PANEL_SEEDS, 'bootstrap_draws': 5000,
              'analysis_seed': ANALYSIS_SEED, 'cost_means': means, 'comparisons': comparisons}
    write_json(run / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'train', 'evaluate'))
    parser.add_argument('--output-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--diagnostic-training-run', type=Path)
    parser.add_argument('--v1-training-run', type=Path)
    args = parser.parse_args()
    if args.mode == 'evaluate' and (args.diagnostic_training_run is None or args.v1_training_run is None):
        parser.error('evaluate requires both --diagnostic-training-run and --v1-training-run')
    torch.set_num_threads(1)
    run, manifest = create_run(args.output_root, args.mode)
    started = time.monotonic()
    print(run.resolve(), flush=True)
    try:
        if args.mode == 'smoke':
            trial = train_trial(run, CASES[0], 73100019, steps=256, eval_seeds=(73100020,))
            manifest['trials'] = [trial]
        elif args.mode == 'train':
            trials = []
            for case in CASES:
                for seed in TRAIN_SEEDS:
                    trials.append(train_trial(run, case, seed))
                    manifest['trials'] = trials
                    write_json(run / 'manifest.json', manifest)
        else:
            diagnostic_train = args.diagnostic_training_run.resolve()
            v1_train = args.v1_training_run.resolve()
            manifest['diagnostic_training_run'] = str(diagnostic_train)
            manifest['v1_training_run'] = str(v1_train)
            manifest['episodes_by_arm'] = evaluate_panel(run, diagnostic_train, v1_train)
            manifest['analysis'] = analyze_panel(run)
        manifest['status'] = 'COMPLETED'
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {str(p.relative_to(run)): sha256(p) for p in run.rglob('*')
                                     if p.is_file() and p.name != 'manifest.json'}
        write_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
