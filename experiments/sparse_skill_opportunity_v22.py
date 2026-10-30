"""Prelocked v22 structural opportunity check on unchanged simulator physics."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from experiments import same_tick_pilot_v21 as v21
from ht_pdm_fjsp.maintenance_dispatch import family_config, role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController, TAIL
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, initial_edge_mask
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/sparse_skill_opportunity_v22_prelock.md'
CASES = ('n5k3_sparse_skill_a', 'n5k3_sparse_skill_b')
CONFIG_SEEDS = dict(zip(CASES, (74760000, 74760001), strict=True))
SKILL_BITS = dict(zip(CASES, (4767, 4829), strict=True))
SMOKE_SEED = 83800011
QUAL_SEEDS = tuple(range(83801000, 83801032))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def append_json(path: Path, value: dict) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def source_paths() -> list[Path]:
    fixed = [ROOT / name for name in (
        'experiments/sparse_skill_opportunity_v22.py',
        'experiments/audit_sparse_skill_opportunity_v22.py',
        'experiments/same_tick_pilot_v21.py',
        'docs/sparse_skill_opportunity_v22_prelock.md',
        'pyproject.toml')]
    return sorted((ROOT / 'src').rglob('*.py')) + fixed


def configs():
    result = {}
    for case in CASES:
        base = family_config('n5_k3_sparse', CONFIG_SEEDS[case], 'full')
        bits = SKILL_BITS[case]
        duration = tuple(tuple((base.service_time[m][j] or 3) if bits & (1 << (m * 3 + j)) else 0
                               for j in range(3)) for m in range(5))
        config = replace(base, service_time=duration, horizon=HORIZON + TAIL)
        config.validate()
        result[case] = config
    return result


def exact_capacity(edges: list[tuple[int, int]]) -> int:
    for count in (2, 1):
        if any(len({m for m, _ in choice}) == count and len({w for _, w in choice}) == count
               for choice in itertools.combinations(edges, count)):
            return count
    return 0


def structure(mask) -> dict:
    valid = [i for i, flag in enumerate(mask) if flag]
    choices = []
    for first in valid:
        m, w = EDGES[first]
        physical = [EDGES[i] for i in valid if i != first and EDGES[i][0] != m and EDGES[i][1] != w]
        canonical = [EDGES[i] for i in valid if i > first and EDGES[i][0] != m and EDGES[i][1] != w]
        choices.append({'first_edge': first, 'physical_count': len(physical),
                        'physical_capacity': exact_capacity(physical),
                        'canonical_count': len(canonical),
                        'canonical_capacity': exact_capacity(canonical)})
    witnesses = [[left['first_edge'], right['first_edge'], left['physical_count']]
                 for left, right in itertools.combinations(choices, 2)
                 if left['physical_count'] == right['physical_count'] >= 2
                 and left['physical_capacity'] != right['physical_capacity']]
    return {'choices': choices, 'witnesses': witnesses, 'qualifying': bool(witnesses)}


def seed_check() -> None:
    v21.seed_registry_check()
    raw = (SMOKE_SEED, *QUAL_SEEDS)
    if len(raw) != len(set(raw)) or not all(83800000 <= seed < 83900000 for seed in raw):
        raise RuntimeError('v22 raw roles overlap')
    physical = [role_seed(seed, BASE_VERSION + ':physical') for seed in raw]
    if len(physical) != len(set(physical)):
        raise RuntimeError('v22 physical tape overlap')


def manifest(mode: str, run: Path, cfgs) -> dict:
    source = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    return {'status': 'RUNNING', 'kind': mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'run': str(run.resolve()), 'git_commit': git_commit, 'git_status': git_status,
            'protocol_sha256': sha256(PROTOCOL), 'source_sha256': source,
            'config_sha256': {case: hashlib.sha256(repr(cfgs[case]).encode()).hexdigest() for case in CASES},
            'cases': CASES if mode == 'qualification' else (CASES[0],),
            'config_seeds': CONFIG_SEEDS, 'skill_bits': SKILL_BITS,
            'smoke_seed': SMOKE_SEED, 'qualification_seeds': QUAL_SEEDS,
            'physical_seeds': {str(seed): role_seed(seed, BASE_VERSION + ':physical')
                               for seed in (SMOKE_SEED, *QUAL_SEEDS)},
            'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'torch': torch.__version__}


def run(mode: str) -> Path:
    seed_check()
    cfgs = configs()
    if mode not in ('smoke', 'qualification'):
        raise ValueError(mode)
    if mode == 'qualification':
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        if status:
            raise RuntimeError('qualification requires committed clean source')
    output = ROOT / 'artifacts' / ('sparse_skill_v22_' + mode + '_' +
                                   datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    info = manifest(mode, output, cfgs)
    write_json(output / 'manifest.json', info)
    started = time.monotonic()
    totals = {case: {'episodes': 0, 'H_decisions': 0, 'qualifying_ticks': 0,
                     'episodes_with_qualifying_tick': 0} for case in info['cases']}
    try:
        append_json(output / 'run.log', {'event': 'started', 'mode': mode})
        seeds = QUAL_SEEDS if mode == 'qualification' else (SMOKE_SEED,)
        for case in info['cases']:
            for seed in tqdm(seeds, desc=f'v22 {mode} {case}'):
                env = ScaledCoordinationEnv(cfgs[case])
                env.reset(seed)
                controller = RuleController('risk_matching')
                episode_qualified = False
                for tick in range(HORIZON):
                    submissions = env.fixed_submissions('threshold')
                    admitted = env.admitted(submissions)
                    mask = initial_edge_mask(env.config, admitted)
                    structural = structure(mask)
                    chosen_submissions, matching = controller.choose(
                        env.config, env.inner.state, tick, case=case, seed=seed)
                    if tuple(chosen_submissions) != tuple(submissions):
                        raise RuntimeError('F1 admission differs from threshold contract')
                    matching_index = env.actions.index(matching)
                    reward, done = env.step(chosen_submissions, matching_index)
                    append_json(output / 'decisions.jsonl', {
                        'case': case, 'seed': seed, 'physical_seed': role_seed(seed, BASE_VERSION + ':physical'),
                        'tick': tick, 'edge_mask': mask.tolist(), 'structure': structural,
                        'submissions': list(chosen_submissions), 'matching_index': matching_index,
                        'reward': float(reward), 'done': bool(done)})
                    totals[case]['H_decisions'] += 1
                    totals[case]['qualifying_ticks'] += int(structural['qualifying'])
                    episode_qualified |= structural['qualifying']
                if not done:
                    raise RuntimeError('episode did not end at H')
                totals[case]['episodes'] += 1
                totals[case]['episodes_with_qualifying_tick'] += int(episode_qualified)
                append_json(output / 'run.log', {'event': 'episode_completed', 'case': case, 'seed': seed})
        gate = {case: totals[case]['qualifying_ticks'] >= 16
                and totals[case]['episodes_with_qualifying_tick'] >= 12
                for case in CASES} if mode == 'qualification' else None
        append_json(output / 'run.log', {'event': 'completed', 'mode': mode})
        info.update(status='COMPLETED', finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, totals=totals,
                    gate=gate, go=all(gate.values()) if gate is not None else None,
                    output_sha256={name: sha256(output / name) for name in ('decisions.jsonl', 'run.log')})
    except Exception as error:
        info.update(status='FAILED', finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, totals=totals, error=repr(error))
        raise
    finally:
        write_json(output / 'manifest.json', info)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'qualification'))
    args = parser.parse_args()
    print(run(args.mode).resolve())


if __name__ == '__main__':
    main()
