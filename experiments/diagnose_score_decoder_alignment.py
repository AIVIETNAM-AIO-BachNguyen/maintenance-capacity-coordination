"""Frozen C21 score versus deployed greedy decoder on the audited training bank.

This is Stage A only. Its labels trained C21 and cannot qualify a decoder change.
No DEV, held-out or TEST state or cost is read.
"""
from __future__ import annotations

import argparse
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

from experiments import same_tick_pilot_v21 as v21
from maintenance_capacity_coordination.same_tick_policy_v15 import (
    ACTIONS, EDGES, MAX_TOKENS, STOP, enumerate_sequences, matching_from_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/score_decoder_alignment_diagnostic_prelock.md'
LABELS = ROOT / 'artifacts/recent_policy_v20_labels_20261024T113110729833Z'
TRAIN = ROOT / 'artifacts/same_tick_v21_train_20261024T180209859141Z'
ACTION_INDEX = {tuple(action): index for index, action in enumerate(ACTIONS)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def padded(paths: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    result = torch.full((len(paths), MAX_TOKENS), -1, dtype=torch.long)
    for i, path in enumerate(paths):
        if len(path) > MAX_TOKENS:
            raise ValueError('path too long')
        result[i, :len(path)] = torch.as_tensor(path, dtype=torch.long)
    return result


def reconstruct_logprob(policy, machine, worker, mask, path: tuple[int, ...]) -> float:
    """Independent token-by-token masked log-softmax reconstruction."""
    with torch.no_grad():
        m = torch.as_tensor(machine, dtype=torch.float32)[None]
        w = torch.as_tensor(worker, dtype=torch.float32)[None]
        edge_mask = torch.as_tensor(mask, dtype=torch.bool)[None]
        encoded = policy.actor.encode(m, w)
        used_m = torch.zeros((1, 5), dtype=torch.bool)
        used_w = torch.zeros((1, 3), dtype=torch.bool)
        selected = torch.zeros((1, STOP), dtype=torch.bool)
        total = 0.
        stopped = False
        for token in path:
            if stopped:
                raise ValueError('token after STOP')
            logits, feasible, _ = policy.actor.logits(encoded, edge_mask, used_m, used_w, selected)
            if not bool(feasible.any()):
                raise ValueError('path token after implicit termination')
            if token != STOP and not bool(feasible[0, token]):
                raise ValueError('masked edge token')
            total += float(torch.log_softmax(logits, dim=1)[0, token])
            if token == STOP:
                stopped = True
            else:
                machine_index, worker_index = EDGES[token]
                used_m[0, machine_index] = True
                used_w[0, worker_index] = True
                selected[0, token] = True
        if not stopped:
            _, feasible, _ = policy.actor.logits(encoded, edge_mask, used_m, used_w, selected)
            if bool(feasible.any()):
                raise ValueError('missing STOP')
        return total


def score_state(policy, row: dict) -> dict:
    mask = np.asarray(row['edge_mask'], dtype=bool)
    paths = enumerate_sequences(mask, canonical=True)
    indices = [ACTION_INDEX[matching_from_sequence(path)] for path in paths]
    if len(indices) != len(set(indices)):
        raise ValueError('duplicate canonical path for physical matching')
    if set(indices) != set(ACTION_INDEX[i] for i in ACTION_INDEX if all(
            mask[machine * 3 + worker] for machine, worker in i)):
        # The physical matching mask is checked by independent Stage-B replay.
        raise ValueError('canonical coverage mismatch')
    machine = torch.as_tensor(row['machine_obs'], dtype=torch.float32)
    worker = torch.as_tensor(row['worker_obs'], dtype=torch.float32)
    edge_mask = torch.as_tensor(mask, dtype=torch.bool)
    with torch.no_grad():
        scores, _ = policy.logprob_entropy(
            machine.repeat(len(paths), 1, 1),
            worker.repeat(len(paths), 1, 1),
            edge_mask.repeat(len(paths), 1),
            padded(paths),
        )
        greedy, greedy_score, _, _ = policy.sample(machine, worker, edge_mask, deterministic=True)
    scores = scores.detach().cpu().numpy()
    greedy_path = tuple(int(x) for x in greedy.tolist() if x >= 0)
    greedy_index = ACTION_INDEX[matching_from_sequence(greedy_path)]
    score_position = min(range(len(paths)), key=lambda k: (-float(scores[k]), indices[k]))
    score_index = indices[score_position]
    greedy_position = indices.index(greedy_index)
    if abs(float(greedy_score) - float(scores[greedy_position])) > 1e-5:
        raise ValueError('greedy path score mismatch')
    for pos in {score_position, greedy_position}:
        if abs(float(scores[pos]) - reconstruct_logprob(
                policy, row['machine_obs'], row['worker_obs'], mask, paths[pos])) > 1e-5:
            raise ValueError('independent score reconstruction mismatch')
    candidates = [int(x) for x in row['matching_indices']]
    if len(candidates) != len(set(candidates)) or not set(candidates).issubset(indices):
        raise ValueError('invalid bank candidates')
    record = {
        'case': row['case'], 'source_raw_seed': row['raw_seed'], 'source_tick': row['tick'],
        'legal_count': len(paths), 'candidate_count': len(candidates),
        'greedy_matching_index': greedy_index, 'score_best_matching_index': score_index,
        'greedy_tokens': list(greedy_path), 'score_best_tokens': list(paths[score_position]),
        'greedy_logprob': float(scores[greedy_position]),
        'score_best_logprob': float(scores[score_position]),
        'score_gap': float(scores[score_position] - scores[greedy_position]),
        'disagree': greedy_index != score_index,
        'greedy_in_bank': greedy_index in candidates,
        'score_best_in_bank': score_index in candidates,
    }
    if record['score_gap'] < -1e-6:
        raise ValueError('score-best below greedy')
    if greedy_index in candidates and score_index in candidates:
        costs = {int(branch['matching_index']): np.asarray(branch['total_costs'], dtype=float)
                 for branch in row['branches']}
        if set(costs) != set(candidates) or any(len(values) != 16 for values in costs.values()):
            raise ValueError('malformed branch labels')
        difference = costs[greedy_index] - costs[score_index]
        record['training_bank_cost_gap_A'] = float(np.mean(difference[:8]))
        record['training_bank_cost_gap_B'] = float(np.mean(difference[8:]))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='two rows per case, root 0 only')
    parser.add_argument('--stage-a', action='store_true', help='all audited training rows and roots')
    args = parser.parse_args()
    if args.smoke == args.stage_a:
        parser.error('choose exactly one of --smoke or --stage-a')
    torch.set_num_threads(1)
    label_audit = json.loads((LABELS / 'audit.json').read_text())
    train_audit = json.loads((TRAIN / 'audit.json').read_text())
    train_manifest = json.loads((TRAIN / 'manifest.json').read_text())
    if label_audit['status'] != 'PASS' or train_audit['status'] != 'PASS':
        raise ValueError('input audit not PASS')
    if train_manifest['status'] != 'COMPLETED' or train_manifest['labels_sha256'] != digest(LABELS / 'labels.jsonl'):
        raise ValueError('training/label provenance mismatch')
    rows = [json.loads(line) for line in (LABELS / 'labels.jsonl').read_text().splitlines()]
    git_status = subprocess.run(('git', 'status', '--short'), cwd=ROOT, capture_output=True, text=True).stdout
    out = ROOT / 'artifacts' / ('score_decoder_' + ('smoke_' if args.smoke else 'stage_a_')
                                 + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'RUNNING', 'kind': 'engineering_smoke' if args.smoke else 'training_bank_descriptive',
        'scope': 'C21 frozen checkpoints; no independent validation or physical policy gain',
        'source_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL),
        'labels_sha256': digest(LABELS / 'labels.jsonl'),
        'label_audit_sha256': digest(LABELS / 'audit.json'),
        'training_manifest_sha256': digest(TRAIN / 'manifest.json'),
        'training_audit_sha256': digest(TRAIN / 'audit.json'),
        'git_commit': subprocess.run(('git', 'rev-parse', 'HEAD'), cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        'git_status': git_status, 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'started_utc': datetime.now(timezone.utc).isoformat(),
    }
    write(out / 'manifest.json', manifest)
    started = time.monotonic()
    results = []
    try:
        for case in v21.CASES:
            case_rows = [row for row in rows if row['case'] == case]
            if args.smoke:
                case_rows = case_rows[:2]
            roots = v21.TRAIN_ROOTS[:1] if args.smoke else v21.TRAIN_ROOTS
            for root in roots:
                trial = TRAIN / f'{case}_C21_{root}'
                selected = json.loads((trial / 'selection.json').read_text())
                checkpoint = trial / selected['checkpoint']
                if digest(checkpoint) != selected['checkpoint_sha256']:
                    raise ValueError('checkpoint hash mismatch')
                policy = v21.load_policy(checkpoint)
                for row in tqdm(case_rows, desc=f'score-decoder {case}/{root}'):
                    record = score_state(policy, row)
                    record.update(root=root, checkpoint_sha256=selected['checkpoint_sha256'])
                    results.append(record)
                    with (out / 'rows.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, sort_keys=True) + '\n')
        summary = []
        for case in v21.CASES:
            for root in (v21.TRAIN_ROOTS[:1] if args.smoke else v21.TRAIN_ROOTS):
                group = [r for r in results if r['case'] == case and r['root'] == root]
                paired = [r for r in group if 'training_bank_cost_gap_B' in r]
                summary.append({
                    'case': case, 'root': root, 'rows': len(group),
                    'disagree': sum(r['disagree'] for r in group),
                    'greedy_in_bank': sum(r['greedy_in_bank'] for r in group),
                    'score_best_in_bank': sum(r['score_best_in_bank'] for r in group),
                    'paired_label_rows': len(paired),
                    'mean_score_gap': float(np.mean([r['score_gap'] for r in group])),
                    'in_sample_mean_cost_gap_B': float(np.mean(
                        [r['training_bank_cost_gap_B'] for r in paired])) if paired else None,
                })
        write(out / 'summary.json', {'scope': manifest['scope'], 'case_root': summary})
        manifest.update(status='COMPLETED', rows=len(results), rows_sha256=digest(out / 'rows.jsonl'),
                        summary_sha256=digest(out / 'summary.json'), runtime_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error), runtime_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'manifest.json', manifest)
        raise
    print(out)


if __name__ == '__main__':
    main()
