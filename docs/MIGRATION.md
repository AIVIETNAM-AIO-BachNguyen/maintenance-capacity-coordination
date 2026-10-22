# Selective migration

## Source and preservation

Original repository: `bachng23/ht-pdm-fjsp`.
Pinned fetched revision: `2343d79d70d1c1fc2f857da2a5afe35c4d501527`, branch `codex/machine-capacity-formulation-gate`.

The Mac's active old checkout was an older unrelated experiment with user changes. Importing from that working tree would have selected the wrong simulator generation. The five selected modules instead come from the exact fetched lab commit, not from the dirty checkout. No old user files were edited or removed.

`provenance/import_manifest.json` records each selected source path and hash. Source modules are byte-preserved. Their original namespace and legacy version/forecast seed strings are deliberately retained to avoid changing behavior. New code uses `maintenance_capacity_coordination`.

## Included

| Component | Purpose |
| --- | --- |
| `maintenance_dispatch.py` | Physical state, seeded failure tapes, service timing, restoration and costs |
| `maintenance_requests.py` | Persistent PM/CM lifecycle, feasible matching, request accounting |
| `maintenance_request_policies.py` | Threshold/resource-aware rules, FCFS, risk matching, bounded rollout, profile family |
| `maintenance_framework_v1.py` | Legacy Gym/observation/action contract, common tail and evaluation |
| `maintenance_framework_v14_policy.py` | Structured V14 actor, including count-calibrated mode |
| Focused inherited tests | Simulator, request lifecycle and adapter regression coverage |
| New baseline adapters and smoke | MaskablePPO setup, V14 loading, serialization and small engineering evaluation |
| Original-reference fixtures | Isolated-original rule metrics, observations and actor logits for equivalence checks |
| Small historical manual-screen evidence | Existing negative requester result with original manifest and raw paired logs |

The one removed inherited test imported an old campaign solely to inspect that campaign's Git preflight. Neither the campaign nor its historical registry is part of this project. All physical/request tests are retained unchanged. New tests cover imported-module hashes, twelve original rule episodes, eight actor/observation fixtures, V14 legal deployment and finite gradients.

## Deliberately excluded

Except for the explicitly isolated V14 reproduction runtime described below, old V1-V20 campaigns/gates, watchers, broad journals, FJSP datasets, production scheduling modules and old output trees are not copied. V14's necessary older runtime dependencies are historical support, not active research goals. The top-level dependency lock removes unrelated OR-Tools/FJSP dependencies.

Correction after a deeper checkpoint search: the historical V14 weights WERE present locally, below the earlier search depth. Ten selected trained checkpoints are now included under baselines/v14, together with their original selection/metadata. Use `V14Controller.from_checkpoint`; do not substitute a random actor for this baseline.

The complete original trainer and eleven-module dependency closure are now byte-preserved in the isolated legacy/v14 runtime, along with protocols, historical seed registries, the corrected verifier and original training tests. Original V14/V7 runs and the old audit are preserved locally and in the portable reproduction bundle. This explicitly scoped exception to the original campaign exclusion makes V14 self-contained without making its old research gates the new project's objective. See docs/V14.md and provenance/v14_extension_manifest.json.

The new MaskablePPO helper is an engineering baseline configuration, not the original trained PPO checkpoint or a prelocked scientific training protocol.

## Evidence and validation

Read docs/KNOWN_FINDINGS.md for the limits of prior requester and capacity findings. The raw small requester screen is copied under provenance/legacy_manual_screen. The synthetic witness is summarized with scope; its full raw artifacts are not bundled and its audit is not re-executed here.

Original-reference fixtures were generated in a separate process using byte-preserved original modules and the old Python environment. Destination tests use the new independently installed lock. Numerical equivalence plus source hashes provide migration evidence, not independent confirmation that the physical model represents a real plant.

`provenance/validation.json` records executed tests/smoke and software versions. Artifacts live under `artifacts/`, ignored by Git. The new repo has no research training outcome, no opened TEST and no claim of a completed framework.

## First research task

Follow CURRENT_STATE.md: a bounded two-policy pilot with matched-information centralized and always-request controls. Formulation is still a decision to make, not copied wholesale from the old proposal's requester-necessity gate.
