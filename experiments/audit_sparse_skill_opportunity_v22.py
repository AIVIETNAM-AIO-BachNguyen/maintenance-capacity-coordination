"""Independent physical replay of the prelocked v22 opportunity check."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, family_config, role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, RuleController, TAIL
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION
from maintenance_capacity_coordination.scaled_coordination_v11 import ScaledCoordinationEnv

from experiments import sparse_skill_opportunity_v22 as pilot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def config(case: str):
    base = family_config('n5_k3_sparse', pilot.CONFIG_SEEDS[case], 'full')
    bits = pilot.SKILL_BITS[case]
    durations = tuple(tuple((base.service_time[m][w] or 3) if ((bits >> (3 * m + w)) & 1) else 0
                            for w in range(3)) for m in range(5))
    result = replace(base, horizon=HORIZON + TAIL, service_time=durations)
    result.validate()
    return result


def capacity(edges: list[tuple[int, int]]) -> int:
    for count in (2, 1):
        for combo in itertools.combinations(edges, count):
            if len(set(m for m, _ in combo)) == count and len(set(w for _, w in combo)) == count:
                return count
    return 0


def structure(mask: list[bool]) -> dict:
    edges = [(m, w) for m in range(5) for w in range(3)]
    present = [i for i, value in enumerate(mask) if value]
    choices = []
    for first in present:
        m0, w0 = edges[first]
        physical = [edges[i] for i in present if i != first and edges[i][0] != m0 and edges[i][1] != w0]
        ordered = [edges[i] for i in present if i > first and edges[i][0] != m0 and edges[i][1] != w0]
        choices.append({'first_edge': first, 'physical_count': len(physical),
                        'physical_capacity': capacity(physical),
                        'canonical_count': len(ordered), 'canonical_capacity': capacity(ordered)})
    witnesses = []
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            a, b = choices[i], choices[j]
            if a['physical_count'] >= 2 and a['physical_count'] == b['physical_count'] \
                    and a['physical_capacity'] != b['physical_capacity']:
                witnesses.append([a['first_edge'], b['first_edge'], a['physical_count']])
    return {'choices': choices, 'witnesses': witnesses, 'qualifying': bool(witnesses)}


def source_identity(run: Path, manifest: dict) -> None:
    assert manifest['status'] == 'COMPLETED'
    assert manifest['protocol_sha256'] == sha256(pilot.PROTOCOL)
    assert manifest['source_sha256'] == {str(p.relative_to(pilot.ROOT)): sha256(p) for p in pilot.source_paths()}
    assert manifest['config_sha256'] == {case: hashlib.sha256(repr(config(case)).encode()).hexdigest()
                                          for case in pilot.CASES}
    assert manifest['cases'] == (list(pilot.CASES) if manifest['kind'] == 'qualification'
                                  else [pilot.CASES[0]])
    assert manifest['config_seeds'] == pilot.CONFIG_SEEDS and manifest['skill_bits'] == pilot.SKILL_BITS
    assert manifest['smoke_seed'] == pilot.SMOKE_SEED and manifest['qualification_seeds'] == list(pilot.QUAL_SEEDS)
    assert manifest['output_sha256'] == {name: sha256(run / name) for name in ('decisions.jsonl', 'run.log')}
    if manifest['kind'] == 'qualification':
        assert manifest['git_status'] == ''
    pilot.seed_check()


def replay(run: Path, manifest: dict) -> tuple[dict, dict | None]:
    cases = pilot.CASES if manifest['kind'] == 'qualification' else (pilot.CASES[0],)
    seeds = pilot.QUAL_SEEDS if manifest['kind'] == 'qualification' else (pilot.SMOKE_SEED,)
    rows = read_rows(run / 'decisions.jsonl')
    assert len(rows) == len(cases) * len(seeds) * HORIZON
    log = read_rows(run / 'run.log')
    assert log[0] == {'event': 'started', 'mode': manifest['kind']}
    assert log[-1] == {'event': 'completed', 'mode': manifest['kind']}
    assert len(log) == 2 + len(cases) * len(seeds)
    totals = {case: {'episodes': 0, 'H_decisions': 0, 'qualifying_ticks': 0,
                     'episodes_with_qualifying_tick': 0} for case in cases}
    position = 0
    event = 1
    for case in cases:
        cfg = config(case)
        for seed in seeds:
            env = ScaledCoordinationEnv(cfg)
            env.reset(seed)
            controller = RuleController('risk_matching')
            episode_qualified = False
            for tick in range(HORIZON):
                row = rows[position]
                position += 1
                expected_sub = env.fixed_submissions('threshold')
                admitted = env.admitted(expected_sub)
                available = eligible_edges(cfg, admitted.physical)
                mask = [admitted.requests[m] is not None
                        and admitted.requests[m].started_at is None
                        and bool(available[m][w]) for m in range(5) for w in range(3)]
                assert row['case'] == case and row['seed'] == seed and row['tick'] == tick
                assert row['physical_seed'] == role_seed(seed, BASE_VERSION + ':physical')
                assert row['edge_mask'] == mask
                expected_structure = structure(mask)
                assert row['structure'] == expected_structure
                submissions, matching = controller.choose(cfg, env.inner.state, tick, case=case, seed=seed)
                assert tuple(row['submissions']) == tuple(expected_sub) == tuple(submissions)
                assert row['matching_index'] == env.actions.index(matching)
                reward, done = env.step(row['submissions'], row['matching_index'])
                assert math.isclose(row['reward'], reward, abs_tol=1e-8)
                assert row['done'] == done == (tick == HORIZON - 1)
                totals[case]['H_decisions'] += 1
                totals[case]['qualifying_ticks'] += int(expected_structure['qualifying'])
                episode_qualified |= expected_structure['qualifying']
            assert log[event] == {'event': 'episode_completed', 'case': case, 'seed': seed}
            event += 1
            totals[case]['episodes'] += 1
            totals[case]['episodes_with_qualifying_tick'] += int(episode_qualified)
    assert position == len(rows) and event == len(log) - 1
    assert totals == manifest['totals']
    gate = ({case: totals[case]['qualifying_ticks'] >= 16 and
             totals[case]['episodes_with_qualifying_tick'] >= 12 for case in cases}
            if manifest['kind'] == 'qualification' else None)
    assert manifest['gate'] == gate
    assert manifest['go'] == (all(gate.values()) if gate is not None else None)
    decision = None if gate is None else {'status': 'AUDITED', 'go': all(gate.values()),
                                          'conditions': gate, 'totals': totals,
                                          'protocol_sha256': sha256(pilot.PROTOCOL)}
    return totals, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    audit = pilot.ROOT / 'artifacts' / ('sparse_skill_v22_audit_' +
                                       datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    audit.mkdir(exist_ok=False)
    if (run / 'audit.json').exists():
        shutil.copy2(run / 'audit.json', audit / 'previous_audit.json')
    result = {'status': 'RUNNING', 'source_run': str(run), 'kind': manifest['kind'],
              'manifest_sha256': sha256(run / 'manifest.json'), 'auditor_sha256': sha256(Path(__file__))}
    try:
        source_identity(run, manifest)
        totals, decision = replay(run, manifest)
        result.update(status='PASS', totals=totals, decisions_replayed=sum(x['H_decisions'] for x in totals.values()))
        if decision is not None:
            pilot.write_json(audit / 'go_decision.json', decision)
            pilot.write_json(run / 'go_decision.json', decision)
            result['decision_sha256'] = sha256(run / 'go_decision.json')
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        pilot.write_json(audit / 'audit.json', result)
        pilot.write_json(run / 'audit.json', result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
