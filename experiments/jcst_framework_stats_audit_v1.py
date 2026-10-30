"""Independent raw-row arithmetic check for L1 confirmation; not a selector."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("base", "generalist")
ARMS = ("S", "F", "A")
Q = (.004166666666666667, .9958333333333333)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def matrix(run, case, arm):
    values = []
    seeds = None
    for root in range(5):
        path = run / "evaluations/confirmation" / case / arm / str(root) / "episodes.jsonl.gz"
        rows = read(path)
        expected = [(1 << 42) + 300_000_000 + CASES.index(case) * 10_000_000 + i
                    for i in range(128)]
        observed = [r["seed"] for r in rows]
        if observed != expected or any(r["H_cost"] + r["tail_cost"] != r["total_cost"] for r in rows):
            raise ValueError("raw learned physical panel")
        if any(r["arm"] != arm or r["root"] != root or r["case"] != case for r in rows):
            raise ValueError("learned metadata")
        if seeds is None:
            seeds = observed
        elif seeds != observed:
            raise ValueError("physical pairing")
        values.append([r["total_cost"] for r in rows])
    return np.asarray(values, dtype=float)


def control(run, case):
    rows = read(run / "controls/confirmation" / case / "threshold5/episodes.jsonl.gz")
    expected = [(1 << 42) + 300_000_000 + CASES.index(case) * 10_000_000 + i
                for i in range(128)]
    if [r["seed"] for r in rows] != expected:
        raise ValueError("control physical pairing")
    return np.asarray([r["total_cost"] for r in rows], dtype=float)


def independent_interval(left, right, learned, rng, draws=4000):
    # Resample a common physical index and separate training-root indices.
    # Threshold is one physical measurement per seed and is never replicated
    # as an independent training root.
    results = np.empty(draws)
    for n in range(draws):
        physical = rng.choice(128, size=128, replace=True)
        lroots = rng.choice(5, size=5, replace=True)
        lhs = float(left[lroots][:, physical].mean())
        if learned:
            rroots = rng.choice(5, size=5, replace=True)
            rhs = float(right[rroots][:, physical].mean())
        else:
            rhs = float(right[physical].mean())
        results[n] = lhs - rhs
    return np.quantile(results, Q).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_dir", type=Path)
    args = parser.parse_args()
    run = (ROOT / args.confirmation_dir).resolve()
    audit = json.loads((run / "independent_audit.json").read_text())
    analysis = json.loads((run / "analysis.json").read_text())
    if audit["status"] != "PASS" or analysis["status"] != "AUDIT_PASS":
        raise ValueError("unapproved source")
    rng = np.random.default_rng(0xC1A1A5E)
    checked = []
    for case in CASES:
        banks = {arm: matrix(run, case, arm) for arm in ARMS}
        banks["threshold5"] = control(run, case)
        for other, margin in (("F", -10), ("A", -10), ("threshold5", 10)):
            original = next(x for x in analysis["rows"]
                            if x["case"] == case and x["contrast"] == f"S-{other}")
            difference = float(banks["S"].mean() - banks[other].mean())
            if abs(difference - original["mean"]) > 1e-9:
                raise ValueError("raw primary mean mismatch")
            lo, hi = independent_interval(banks["S"], banks[other],
                                          other in ARMS, rng)
            original_gate = (difference <= margin and
                             (original["adjusted_hi"] < 0 if other in ARMS
                              else original["adjusted_hi"] <= margin))
            if original_gate != original["pass"]:
                raise ValueError("locked gate arithmetic mismatch")
            checked.append({
                "case": case, "contrast": f"S-{other}",
                "raw_mean": difference,
                "auditor_interval": [original["adjusted_lo"], original["adjusted_hi"]],
                "independent_4000_draw_interval": [lo, hi],
                "locked_margin": margin, "auditor_gate": original["pass"],
            })
    if analysis["replication_eligible"] != all(x["auditor_gate"] for x in checked):
        raise ValueError("replication arithmetic")
    result = {"status": "PASS", "scope": "raw-row primary means, physical pairing, alternative crossed bootstrap, gate arithmetic",
              "source_confirmation_manifest_hash": sha(run / "manifest.json"),
              "source_replay_audit_hash": sha(run / "independent_audit.json"),
              "source_analysis_hash": sha(run / "analysis.json"),
              "script_hash": sha(Path(__file__)),
              "bootstrap_seed": 0xC1A1A5E, "bootstrap_resamples": 4000,
              "simultaneous_family": 6, "percentile_tails": Q,
              "replication_eligible": analysis["replication_eligible"],
              "contrasts": checked}
    output = ROOT / "artifacts/jcst_framework" / (
        "stats-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
