"""Focused contract and arithmetic tests for the prelocked v7 transfer check."""
from dataclasses import asdict
import json

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from maintenance_capacity_coordination.coordination_v1 import BASE_VERSION, CoordinationEnv
from maintenance_capacity_coordination.structured_pilot_v3 import choose, load_policy
from maintenance_capacity_coordination.structured_policy_v3 import ACTIONS, actor_inputs
from experiments import capacity_transfer_v7 as v7
from experiments.audit_capacity_transfer_v7 import independent_analysis


def test_only_legal_service_durations_change_and_complete_action_catalogue():
    configs = v7.config_by_name()
    assert len(ACTIONS) == 21 and ACTIONS[0] == ()
    for source, target in v7.PAIRS:
        left, right = asdict(configs[source]), asdict(configs[target])
        assert [key for key in left if left[key] != right[key]] == ['service_time']
        for case in (source, target):
            env = CoordinationEnv(configs[case])
            env.reset(v7.SMOKE_SEEDS[0])
            submitted = env.fixed_submissions('threshold')
            machine, worker = actor_inputs(env.config, env.admitted(submitted), env.tick)
            mask = env.matching_mask(submitted)
            assert machine.shape == (4, 21) and worker.shape == (2, 4)
            assert mask.shape == (21,) and mask[0]


def test_source_checkpoint_identity_and_dev_selection():
    parent = v7.verify_parent(v7.V3_TRAIN, False)
    assert parent['audit_sha256'] == v7.sha256(v7.V3_TRAIN / 'audit.json')
    for source, _ in v7.PAIRS:
        for arm in v7.ARMS:
            for root in v7.ROOTS:
                identity = v7.checkpoint_identity(v7.V3_TRAIN, source, arm, root)
                trial = v7.V3_TRAIN / f'{source}_{arm}_{root}'
                rows = v7.read_jsonl(trial / 'dev_episodes.jsonl')
                means = [(float(np.mean([row['objective'] for row in rows
                                         if row['checkpoint'] == str(step)])), step)
                         for step in (2048, 4096, 8192)]
                best = min(means)
                assert identity['transition'] == best[1]
                assert identity['sha256'] == v7.sha256(identity['path'])


def test_physical_pairing_and_no_future_tape_actor_input():
    configs = v7.config_by_name()
    source, target = v7.PAIRS[0]
    seed = v7.SMOKE_SEEDS[0]
    a, b = CoordinationEnv(configs[source]), CoordinationEnv(configs[target])
    a.reset(seed)
    b.reset(seed)
    expected = role_seed(seed, BASE_VERSION + ':physical')
    assert expected == role_seed(seed, BASE_VERSION + ':physical')
    assert a.inner.state.physical == b.inner.state.physical
    policy = load_policy(v7.checkpoint_identity(v7.V3_TRAIN, source, 'S', v7.ROOTS[0])['path'])
    original = choose(policy, a, deterministic=True)
    a.inner._events = tuple((False,) * a.config.machines for _ in a.inner._events)
    changed = choose(policy, a, deterministic=True)
    assert original['matching_index'] == changed['matching_index']
    assert original['submissions'] == changed['submissions']


def test_cluster_arithmetic_and_gate_signs_agree_independently():
    rows = []
    seeds = (11, 12, 13, 14)
    roots = (1, 2, 3)
    for source, target in v7.PAIRS:
        for case in (source, target):
            for root in roots:
                for seed in seeds:
                    baseline = 100 + (seed - 11) * 7 + root
                    for arm in v7.ARMS:
                        offset = {'S': -20 if case == source else -40,
                                  'G3': 0, 'R': -5 if case == source else -10}[arm]
                        rows.append({'source_case': source, 'case': case,
                                     'root': root, 'seed': seed, 'arm': arm,
                                     'objective': baseline + offset})
    audited = independent_analysis(rows, seeds, roots)
    for name, value in [('target_S_minus_G3', -40), ('transfer_difference', -20),
                        ('target_S_minus_R', -30)]:
        from_runner = v7.comparison(rows, seeds, roots, name)
        assert audited[name] == from_runner
        assert from_runner['overall'] == value
        assert from_runner['physical_cluster_ci95'] == [value, value]


def test_seed_registry_and_fresh_artifacts(tmp_path):
    assert v7.check_seed_roles(False)
    first, manifest = v7.new_run(tmp_path, 'smoke', None, v7.ROOTS)
    (first / 'sentinel.txt').write_text('preserve')
    second, _ = v7.new_run(tmp_path, 'smoke', None, v7.ROOTS)
    assert first != second and (first / 'sentinel.txt').read_text() == 'preserve'
    assert json.loads((first / 'manifest.json').read_text())['status'] == 'RUNNING'
    assert manifest['sealed_test_reserved'] == v7.SEALED_TEST
