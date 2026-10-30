"""Independent L2 physical replay, policy-action verification and DEV selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from jcst_learner_audit_v3 import audit_episode
from jcst_replay_audit import catalogue, independent_step
from jcst_learnability_l2 import (ARMS, BANK, CASES, CHECKPOINTS, ROOT, action_seed,
                                  action_universe, case_config, load_policy,
                                  physical_seed, sha)
from ht_pdm_fjsp.component_service_v1 import config_json, initial_state, state_from_json
from ht_pdm_fjsp.jcst_learner_v1 import encode, tensors

OLD_METRICS = ("pm_starts", "cm_starts", "failures", "completions",
               "unavailable_machine_intervals", "occupied_technician_intervals",
               "wait_H", "unfinished_jobs", "failed_components_at_cutoff")


def lines(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def physical_replay(row: dict, config, expected_seed: int, universe=None) -> None:
    legacy = dict(row)
    legacy["metrics"] = {key: row["metrics"][key] for key in OLD_METRICS}
    cfg = json.loads(config_json(config))
    audit_episode(cfg, legacy, expected_seed, universe=universe, config=config)
    state = asdict(initial_state(config))
    waits = dict(forced_wait=0, discretionary_wait=0, wait_failed_free=0,
                 wait_all_healthy=0, wait_all_busy=0)
    for tick in range(48):
        action = tuple(tuple(edge) for edge in row["trace"][tick]["action"])
        if not action:
            available = catalogue(cfg, state)
            waits["forced_wait" if len(available) == 1 else "discretionary_wait"] += 1
            if any(any(state["components"][m][c]["failed"] for m, c, _ in candidate)
                   for candidate in available if candidate):
                waits["wait_failed_free"] += 1
            if all(not item["failed"] for group in state["components"] for item in group):
                waits["wait_all_healthy"] += 1
            if all(job is not None for job in state["jobs"]):
                waits["wait_all_busy"] += 1
        state, _, _ = independent_step(cfg, state, action, expected_seed)
    if all(name in row["metrics"] for name in waits):
        if any(row["metrics"][name] != value for name, value in waits.items()):
            raise ValueError("WAIT context classification mismatch")


def policy_replay(row: dict, config, universe, policy) -> None:
    state = asdict(initial_state(config))
    generator = None
    if row["mode"] == "stochastic":
        generator = torch.Generator().manual_seed(action_seed(
            row["bank"], row["case"], row["arm"], row["root"],
            row["checkpoint"], row["physical_index"], row["action_repetition"]))
    if len(row["diagnostics"]) != 48:
        raise ValueError("policy diagnostic tick count")
    for tick in range(48):
        trace = row["trace"][tick]
        state_obj = state_from_json(config, json.dumps(state))
        encoded = encode(config, state_obj, universe)
        c, w, flat, mask = tensors([encoded])
        with torch.no_grad():
            dist = policy.distribution(c, w, flat, mask)
            probabilities = dist.probs[0]
            if generator is None:
                index = int(dist.logits.argmax(dim=-1).item())
            else:
                index = int(torch.multinomial(probabilities, 1, generator=generator).item())
            legal = mask[0]
            logits = torch.topk(dist.logits[0][legal], min(2, int(legal.sum()))).values
            probs = torch.topk(probabilities[legal], min(2, int(legal.sum()))).values
            gap = float(logits[0] - logits[1]) if len(logits) > 1 else 1e9
            prob_gap = float(probs[0] - probs[1]) if len(probs) > 1 else 1e9
            d = row["diagnostics"][tick]
            if index != trace["index"] or index != d["index"] or not encoded.mask[index]:
                raise ValueError("policy action/index/RNG mismatch")
            scalars = {"logprob": float(dist.log_prob(torch.tensor([index])).item()),
                       "entropy": float(dist.entropy().item()),
                       "wait_probability": float(probabilities[0]),
                       "chosen_probability": float(probabilities[index]),
                       "greedy_probability": float(probabilities.max()),
                       "logit_gap": gap, "probability_gap": prob_gap,
                       "legal_count": int(legal.sum())}
            for name, expected in scalars.items():
                if not np.isclose(d[name], expected, atol=2e-6, rtol=1e-6):
                    raise ValueError(f"policy diagnostic mismatch: {name}")
            if d["near_tie"] != (gap <= .01) or d["exact_tie"] != (gap <= 1e-6):
                raise ValueError("near-tie rule mismatch")
        action = tuple(tuple(edge) for edge in trace["action"])
        state, _, _ = independent_step(json.loads(config_json(config)), state, action, row["seed"])


def source_check(run_dir: Path) -> dict:
    data = json.loads((run_dir / "manifest.json").read_text())
    if data["schema"] != "JCST-learnability-L2" or data["status"] != "COMPLETE":
        raise ValueError("incomplete L2 source")
    if data["protocol_hash"] != sha(ROOT / "docs/jcst_learnability_protocol_l2.md"):
        raise ValueError("protocol changed")
    if data["contract_hash"] != sha(ROOT / "docs/joint_component_service_contract_v1.md"):
        raise ValueError("contract changed")
    for name, digest in data["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    if data["parameters"]["H"] != 48 or data["parameters"]["B"] != 16:
        raise ValueError("operational horizon")
    if data["seed_roles"]["banks"] != BANK:
        raise ValueError("seed bank")
    return data


def audit_training(run_dir: Path, manifest: dict) -> tuple[int, int]:
    expected = 6 if manifest["stage"] == "smoke" else 30
    if len(manifest["completed_trials"]) != expected:
        raise ValueError("trial panel incomplete")
    scanned = replayed = 0
    for info in tqdm(manifest["completed_trials"], desc="audit training trials"):
        case, arm, root = info["case"], info["arm"], info["root"]
        bank = "engineering" if manifest["stage"] == "smoke" else "train"
        config, _ = case_config(case)
        universe = action_universe(config)
        trial = run_dir / "trials" / case / arm / str(root)
        seen = 0
        for row in lines(trial / "train.jsonl"):
            if row["episode_index"] != seen or row["case"] != case or row["arm"] != arm or row["root"] != root:
                raise ValueError("training row sequence/metadata")
            physical = physical_seed(bank, case, root, seen)
            if row["seed"] != physical or row["bank"] != bank:
                raise ValueError("training physical seed")
            if row["H_cost"] + row["tail_cost"] != row["total_cost"]:
                raise ValueError("training H/tail sum")
            if sum(row["components"].values()) != row["total_cost"]:
                raise ValueError("training component sum")
            if "trace" in row:
                physical_replay(row, config, physical, universe)
                replayed += 1
            seen += 1
        if seen != info["episodes"] or seen * 48 != info["samples"]:
            raise ValueError("training episode/sample completeness")
        checkpoints = (48, 96) if manifest["stage"] == "smoke" else CHECKPOINTS
        for checkpoint in checkpoints:
            file = trial / f"checkpoint-{checkpoint}.pt"
            meta = json.loads((trial / f"checkpoint-{checkpoint}.json").read_text())
            if meta["weights_sha256"] != sha(file):
                raise ValueError("checkpoint hash")
        scanned += seen
    return scanned, replayed


def audit_evaluation(run_dir: Path, manifest: dict) -> tuple[int, dict]:
    bank = "engineering" if manifest["stage"] == "smoke" else "dev"
    count = 2 if manifest["stage"] == "smoke" else 16
    expected_files = 12 if manifest["stage"] == "smoke" else 240
    if len(manifest["completed_evaluations"]) != expected_files:
        raise ValueError("evaluation panel incomplete")
    replayed = 0
    cube = {}
    for info in tqdm(manifest["completed_evaluations"], desc="audit policy evaluation"):
        case, arm, root = info["case"], info["arm"], info["root"]
        checkpoint, mode = info["checkpoint"], info["mode"]
        if info["bank"] != bank or info["physical_seeds"] != count:
            raise ValueError("evaluation bank/count")
        config, _ = case_config(case)
        universe = action_universe(config)
        _, _, policy = load_policy(run_dir / "trials" / case / arm / str(root), checkpoint)
        reps = 1 if mode == "greedy" else 4
        costs = np.empty((count, reps), dtype=float)
        starts = np.empty((count, reps), dtype=float)
        seen = 0
        for row in lines(ROOT / info["file"]):
            i, rep = divmod(seen, reps)
            if i >= count or row["physical_index"] != i or row["action_repetition"] != rep:
                raise ValueError("evaluation row sequence")
            physical = physical_seed(bank, case, 0, i)
            if (row["seed"] != physical or row["checkpoint_sha256"] !=
                    sha(run_dir / "trials" / case / arm / str(root) / f"checkpoint-{checkpoint}.pt")):
                raise ValueError("evaluation seed/checkpoint provenance")
            physical_replay(row, config, physical, universe)
            policy_replay(row, config, universe, policy)
            costs[i, rep] = row["total_cost"]
            starts[i, rep] = row["metrics"]["pm_starts"] + row["metrics"]["cm_starts"]
            seen += 1
        if seen != count * reps or not np.isclose(costs.mean(), info["mean_cost"]):
            raise ValueError("evaluation completeness/summary")
        cube[(case, arm, root, checkpoint, mode)] = (costs.mean(axis=1), starts.mean(axis=1))
        replayed += seen
    return replayed, cube


def audit_controls(run_dir: Path, manifest: dict) -> tuple[int, dict]:
    bank = "engineering" if manifest["stage"] == "smoke" else "dev"
    count = 1 if manifest["stage"] == "smoke" else 16
    expected = 4 if manifest["stage"] == "smoke" else 8
    if len(manifest["completed_controls"]) != expected:
        raise ValueError("control panel incomplete")
    replayed = 0
    arrays = {}
    for info in manifest["completed_controls"]:
        case, arm = info["case"], info["arm"]
        config, _ = case_config(case)
        path = run_dir / "controls" / bank / case / arm / "episodes.jsonl"
        costs = []
        for i, row in enumerate(lines(path)):
            if i >= count:
                raise ValueError("extra control row")
            physical_replay(row, config, physical_seed(bank, case, 0, i))
            costs.append(row["total_cost"])
        if len(costs) != count or not np.isclose(np.mean(costs), info["mean_cost"]):
            raise ValueError("control completeness/summary")
        arrays[(case, arm)] = np.asarray(costs)
        replayed += count
    return replayed, arrays


def select(manifest: dict, cube: dict, controls: dict, source_manifest_hash: str) -> dict:
    selected = {}
    eligible_cases = []
    table = []
    for case in CASES:
        selected[case] = {}
        for arm in ARMS:
            candidates = []
            for checkpoint in CHECKPOINTS:
                for mode in ("greedy", "stochastic"):
                    costs = np.stack([cube[(case, arm, root, checkpoint, mode)][0]
                                      for root in range(5)])
                    starts = np.stack([cube[(case, arm, root, checkpoint, mode)][1]
                                       for root in range(5)])
                    candidates.append((float(costs.mean()), checkpoint,
                                       0 if mode == "greedy" else 1, mode, costs, starts))
                    table.append({"case": case, "arm": arm, "checkpoint": checkpoint,
                                  "mode": mode, "mean_cost": float(costs.mean()),
                                  "mean_starts": float(starts.mean())})
            mean, checkpoint, _, mode, costs, starts = min(candidates, key=lambda item: item[:3])
            low = np.stack([cube[(case, arm, root, 24000, mode)][0]
                            for root in range(5)])
            root_gains = low.mean(axis=1) - costs.mean(axis=1)
            gain = float(low.mean() - mean)
            wait_gain = float(controls[(case, "wait")].mean() - mean)
            qualifying_roots = int(np.sum(root_gains >= 10))
            active_roots = int(np.sum(starts.mean(axis=1) > 2))
            eligible = bool(checkpoint >= 96000 and gain >= 20 and wait_gain >= 20
                            and qualifying_roots >= 4 and active_roots >= 4)
            selected[case][arm] = {"checkpoint": checkpoint, "mode": mode,
                                   "mean_cost": mean, "gain_vs_24k": gain,
                                   "gain_vs_wait": wait_gain,
                                   "roots_gain_at_least_10": qualifying_roots,
                                   "roots_more_than_2_starts": active_roots,
                                   "dev_eligible": eligible}
            if eligible:
                eligible_cases.append((case, arm))
    return {"status": "AUDIT_PASS", "source_manifest_hash": source_manifest_hash,
            "selected": selected, "dev_eligible": bool(eligible_cases),
            "eligible_arm_cases": eligible_cases, "learning_table": table,
            "selection_rule": "min mean cost, tie smaller checkpoint then greedy"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = args.run_dir.resolve()
    manifest = source_check(source)
    output = ROOT / "artifacts/jcst_learnability" / ("audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "source_run_dir": str(source.relative_to(ROOT)),
              "source_manifest_hash": sha(source / "manifest.json"),
              "audit_code_hash": sha(Path(__file__)), "stage": manifest["stage"]}
    try:
        scanned, train_replayed = audit_training(source, manifest)
        evaluated, cube = audit_evaluation(source, manifest)
        controlled, controls = audit_controls(source, manifest)
        report.update(status="PASS", training_rows_scanned=scanned,
                      training_episodes_replayed=train_replayed,
                      evaluation_episodes_replayed=evaluated,
                      control_episodes_replayed=controlled)
        if manifest["stage"] == "train_dev":
            selection = select(manifest, cube, controls, report["source_manifest_hash"])
            (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    except BaseException as exc:
        report.update(status="FAIL", failure=repr(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(output)


if __name__ == "__main__":
    main()
