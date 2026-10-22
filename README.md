# Maintenance Capacity Coordination

A learning-based framework for coordinating maintenance demand and heterogeneous technician capacity.

This repository is a clean research starting point, not an already validated two-policy framework. Read [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md), [CURRENT_STATE.md](CURRENT_STATE.md), then [AGENTS.md](AGENTS.md).

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

Ten selected pretrained V14 checkpoints are bundled under `baselines/v14`: both arms across all five training seeds, hash-checked against the original selection. `V14Controller` loads them directly. The complete original training runtime, protocol, registries, verifier and tests are isolated under `legacy/v14`; historical run data and V7 comparator assets are also included in a portable reproduction bundle. See [docs/V14.md](docs/V14.md). The actual learned requester/coordinator framework remains to be designed.

The imported Gym adapter fixes the requester to a threshold. A new two-policy adapter must be separately specified and tested; passing a new requester to this adapter is not supported. The physical kernel is configurable, but the legacy observation and V14 actor do not establish variable-size support.
