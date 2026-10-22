# Maintenance Capacity Coordination

A learning-based framework for coordinating maintenance demand and heterogeneous technician capacity.

This repository contains a tested two-stage demand/capacity framework and one prelocked pilot. The pilot has mixed results; it does not establish a general learning benefit or a new RL algorithm. Read [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md), [CURRENT_STATE.md](CURRENT_STATE.md), then [AGENTS.md](AGENTS.md).

## Quick start

Python 3.12 is required.

```bash
uv sync --extra dev
uv run pytest -q
uv run mcc-smoke
```

`mcc-smoke` runs a bounded engineering check, trains a tiny MaskablePPO baseline, checks checkpoint round trips, and evaluates rules with progress bars. It writes a fresh UTC-stamped directory under `artifacts/`. Its outputs are not research performance estimates.

## Reused components

- Unchanged physical kernel, persistent PM/CM request lifecycle and rule/rollout baselines.
- Unchanged N4/K2 Gym adapter: 56 observations, 21 partial matchings, H48 plus a common 16-tick risk-matching tail.
- Unchanged V14 actor, including the count-calibrated baseline. V14's original training was counterfactual-Q/KL, not PPO.
- Focused physical/request tests and migration-equivalence fixtures.

The five imported modules retain their `ht_pdm_fjsp` namespace for byte-for-byte provenance. New code lives in `maintenance_capacity_coordination`; the old campaigns are not dependencies. See [docs/MIGRATION.md](docs/MIGRATION.md).

Ten selected pretrained V14 checkpoints are bundled under `baselines/v14`: both arms across all five training seeds, hash-checked against the original selection. `V14Controller` loads them directly. The complete original training runtime, protocol, registries, verifier and tests are isolated under `legacy/v14`; historical run data and V7 comparator assets are also included in a portable reproduction bundle. See [docs/V14.md](docs/V14.md). The imported Gym adapter still fixes the requester to a threshold. The new `CoordinationEnv` is a separate two-stage adapter over the same simulator; it is intentionally limited to the two prelocked N4/K2 pilot cases. The legacy V14 actor does not establish variable-size support.

## Coordination pilot v1

Read [the prelocked protocol](docs/coordination_pilot_v1.md), [the audited replication findings](docs/pilot_v1_replication_findings.md), and [the sealed TEST findings](docs/pilot_v1_test_findings.md) before interpreting the pilot. Shared machine requesters submit persistent PM demand; a full-current-information coordinator chooses a feasible partial matching or WAIT. B uses local machine information, C adds present compatible-technician capacity, D always requests, and E is a full-information centralized control. All learned arms use the same economic reward and standard PPO with the exact joint request/matching likelihood.

```bash
uv run --extra dev pytest -q
uv run mcc-pilot smoke
uv run mcc-v14 audit
uv run mcc-pilot train
uv run mcc-pilot replicate --training-run artifacts/PILOT_TRAIN_RUN
uv run mcc-pilot-analyze --training-run artifacts/PILOT_TRAIN_RUN --replication-run artifacts/PILOT_REPLICATION_RUN
uv run mcc-pilot test --training-run artifacts/PILOT_TRAIN_RUN --replication-run artifacts/PILOT_REPLICATION_RUN
uv run python experiments/audit_pilot_test_v1.py --training-run artifacts/PILOT_TRAIN_RUN --replication-run artifacts/PILOT_REPLICATION_RUN --test-run artifacts/PILOT_TEST_RUN
```

`train`, `replicate`, and `test` are research runs with fresh timestamped directories. `smoke` is engineering validation. The sealed TEST command requires a positive recorded replication decision and explicit authorization; the prelocked v1 TEST has now been run and must not be reused for tuning. Frozen V14, rules and rollout are evaluated on the same physical panels. The Ubuntu V14 verifier passed unchanged; the historical Mac replay mismatch remains documented in [docs/V14.md](docs/V14.md).

## Attribution follow-up

A [versioned fixed-threshold requester diagnostic](docs/threshold_requester_diagnostic_v1.md) tested the open question left by the pilot: whether C's gain over always-request D came from learned capacity-aware demand or simply gating PM requests. The [audited findings](docs/threshold_requester_diagnostic_findings.md) favor fixed threshold gating at this budget. This follow-up is exploratory and did not change the sealed pilot v1 TEST decision or its analysis.

## Revised demand and technician-commitment direction

The [v2 proposal](docs/demand_capacity_commitment_proposal_v2.md) distinguishes machine demand representation from the physical matching decision. Its [prelocked bounded headroom check](docs/commitment_headroom_v2.md) tested whether just-in-time early PM choices improve over strong threshold plus rollout on paired current states and independent common forecasts. The [audited findings](docs/commitment_headroom_v2_findings.md) show a small six-tick forecast gain below the locked minimum; the v2 learning-pilot gate failed and no structured learner was trained. The check is not an H+tail policy comparison or an optimality result.

```bash
uv run --extra dev pytest -q
uv run python -m experiments.audit_commitment_headroom_v2 --run artifacts/commitment_headroom_v2_check_20261022T070841552901Z
```

The v1 `replicate` command above evaluated fixed trained checkpoints on fresh physical seeds; independent training replication would require new training seeds and fresh models. No new TEST is authorized.

## Structured matching pilot v3

A [separate prelocked pilot](docs/structured_demand_commitment_pilot_v3.md) tests whether a [shared machine-demand/worker-commitment actor](docs/structured_demand_commitment_proposal_v3.md) improves PPO learning against a flat full-information actor on the same threshold-admitted matching set. It includes a remaining-time ablation and contextual always-request/strong rule controls. The v2 early-PM no-go result is unchanged. A bounded Ubuntu smoke, 114 tests, DEV training and held-out audits passed. [The v3 findings](docs/structured_demand_commitment_pilot_v3_findings.md) show a modest S gain over matched G3 that missed the locked practical gate, and no measured remaining-time ablation benefit. Independent training replication and new TEST remain unopened.
