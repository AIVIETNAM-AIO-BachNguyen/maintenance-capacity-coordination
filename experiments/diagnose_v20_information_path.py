"""Training-bank-only descriptive probe of the selected C20 information path.

Observation perturbations below are engineering interventions, not physically
valid transitions or economic counterfactuals. No held-out or TEST data are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from maintenance_capacity_coordination.dynamic_bipartite_policy_v16 import DynamicSequencePolicy
from maintenance_capacity_coordination.same_tick_policy_v15 import EDGES, STOP


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, item: dict) -> None:
    path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")


def embeddings(actor, machine, worker, mask, used_m, used_w, selected):
    outputs = []
    hook = actor.update_machine.register_forward_hook(
        lambda _module, _inputs, output: outputs.append(output.detach().clone()))
    try:
        actor._node_edge_embeddings(machine, worker, mask, used_m, used_w, selected)
    finally:
        hook.remove()
    assert len(outputs) == 1
    return outputs[0][0]


def evaluate_row(row: dict, actor) -> dict:
    machine = torch.as_tensor(row["machine_obs"], dtype=torch.float32)[None]
    worker = torch.as_tensor(row["worker_obs"], dtype=torch.float32)[None]
    mask = torch.as_tensor(row["edge_mask"], dtype=torch.bool)[None]
    used_m = torch.zeros((1, 5), dtype=torch.bool)
    used_w = torch.zeros((1, 3), dtype=torch.bool)
    selected = torch.zeros((1, STOP), dtype=torch.bool)
    allowed = torch.cat((mask[0], torch.ones(1, dtype=torch.bool)))
    with torch.no_grad():
        encoded = actor.encode(machine, worker)
        root_logits, _, root_context = actor.logits(encoded, mask, used_m, used_w, selected)
        root_machine = embeddings(actor, machine, worker, mask, used_m, used_w, selected)
        busy_deltas = []
        for j in range(3):
            if worker[0, j, 1] <= 0 or worker[0, j, 1] > 11 / 12:
                continue
            changed_worker = worker.clone()
            changed_worker[0, j, 1] += 1 / 12
            changed_logits, _, _ = actor.logits(
                actor.encode(machine, changed_worker), mask, used_m, used_w, selected)
            changed_machine = embeddings(actor, machine, changed_worker, mask, used_m, used_w, selected)
            norms = (changed_machine - root_machine).norm(dim=1).numpy()
            compatible = (machine[0, :, 14 + j] > 0).numpy()
            busy_deltas.append({
                "worker": j,
                "compatible_machines": int(compatible.sum()),
                "compatible_embedding_max_delta": float(norms[compatible].max()) if compatible.any() else 0.,
                "incompatible_embedding_max_delta": float(norms[~compatible].max()) if (~compatible).any() else 0.,
                "valid_logit_max_delta": float((changed_logits[0] - root_logits[0]).abs()[allowed].max()),
                "root_greedy_token_changed": bool(changed_logits[0].argmax() != root_logits[0].argmax()),
            })
        first = int(root_logits[0].argmax())
        partial = None
        if first != STOP:
            m, j = EDGES[first]
            used_m[0, m] = True
            used_w[0, j] = True
            selected[0, first] = True
            after_logits, feasible, _ = actor.logits(
                encoded, mask, used_m, used_w, selected)
            frozen_logits, frozen_feasible, _ = actor.logits(
                encoded, mask, used_m, used_w, selected, frozen_context=root_context)
            assert torch.equal(feasible, frozen_feasible)
            changed_machine = embeddings(actor, machine, worker, mask, used_m, used_w, selected)
            machine_deltas = (changed_machine - root_machine).norm(dim=1).numpy()
            sole_lost = [i for i in range(5) if i != m and mask[0, i * 3 + j]
                         and int(mask[0, i * 3:i * 3 + 3].sum()) == 1]
            legal_after = torch.cat((feasible[0], torch.ones(1, dtype=torch.bool)))
            partial = {
                "first_edge": first,
                "legal_second_edges": int(feasible.sum()),
                "sole_worker_lost_machines": sole_lost,
                "sole_worker_lost_max_embedding_delta": float(max((machine_deltas[i] for i in sole_lost), default=0.)),
                "unselected_machine_max_embedding_delta": float(max(machine_deltas[i] for i in range(5) if i != m)),
                "dynamic_frozen_valid_logit_max_delta": float((after_logits[0] - frozen_logits[0]).abs()[legal_after].max()),
                "dynamic_frozen_next_token_changed": bool(after_logits[0].argmax() != frozen_logits[0].argmax()),
            }
    return {
        "case": row["case"], "train_root": row["train_root"],
        "raw_seed": row["raw_seed"], "tick": row["tick"],
        "source_arm": row["source_arm"], "busy_time_interventions": busy_deltas,
        "greedy_partial": partial,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args()
    labels = args.label_run.resolve()
    training = args.training_run.resolve()
    assert json.loads((labels / "audit.json").read_text())["status"] == "PASS"
    assert json.loads((training / "audit.json").read_text())["status"] == "PASS"
    torch.set_num_threads(1)
    out = Path("artifacts") / ("v20_information_path_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    paths = {}
    rows = []
    try:
        with (labels / "labels.jsonl").open() as source, (out / "rows.jsonl").open("w") as target:
            for line in tqdm(source, total=508, desc="C20 training-bank information path"):
                row = json.loads(line)
                key = (row["case"], row["train_root"])
                if key not in paths:
                    trial = training / f"{key[0]}_C20_{key[1]}"
                    selection = json.loads((trial / "selection.json").read_text())
                    checkpoint = trial / selection["checkpoint"]
                    assert digest(checkpoint) == selection["checkpoint_sha256"]
                    policy = DynamicSequencePolicy("C16")
                    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"], strict=True)
                    policy.eval()
                    paths[key] = policy.actor
                measured = evaluate_row(row, paths[key])
                rows.append(measured)
                target.write(json.dumps(measured, sort_keys=True) + "\n")
        busy = [item for row in rows for item in row["busy_time_interventions"]]
        partial = [row["greedy_partial"] for row in rows if row["greedy_partial"] is not None]
        sole = [item for item in partial if item["sole_worker_lost_machines"]]
        summary = {
            "status": "COMPLETED", "kind": "training_bank_descriptive_information_path",
            "labels_sha256": digest(labels / "labels.jsonl"),
            "training_manifest_sha256": digest(training / "manifest.json"),
            "script_sha256": digest(Path(__file__)),
            "rows": len(rows), "busy_worker_interventions": len(busy),
            "busy_valid_logit_change_count": sum(x["valid_logit_max_delta"] > 1e-7 for x in busy),
            "busy_valid_logit_delta_median": float(np.median([x["valid_logit_max_delta"] for x in busy])) if busy else None,
            "busy_root_greedy_change_count": sum(x["root_greedy_token_changed"] for x in busy),
            "compatible_machine_embedding_change_count": sum(x["compatible_embedding_max_delta"] > 1e-7 for x in busy),
            "incompatible_machine_embedding_max_delta": max((x["incompatible_embedding_max_delta"] for x in busy), default=0.),
            "greedy_first_edge_rows": len(partial),
            "greedy_second_edge_opportunities": sum(x["legal_second_edges"] > 0 for x in partial),
            "sole_worker_lost_rows": len(sole),
            "sole_worker_lost_embedding_change_count": sum(x["sole_worker_lost_max_embedding_delta"] > 1e-7 for x in sole),
            "dynamic_frozen_next_token_change_count": sum(x["dynamic_frozen_next_token_changed"] for x in partial),
            "dynamic_frozen_valid_logit_change_count": sum(x["dynamic_frozen_valid_logit_max_delta"] > 1e-7 for x in partial),
            "limitations": "Observation interventions and in-bank greedy partials are descriptive; no physical counterfactual or held-out economic estimate.",
        }
        save_json(out / "summary.json", summary)
    except Exception as exc:
        save_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise
    print(out.resolve())


if __name__ == "__main__":
    main()
