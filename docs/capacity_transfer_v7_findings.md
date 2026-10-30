# Service-duration transfer diagnostic v7 — audited findings

Date: 2026-10-22 (Asia/Taipei). The question and decision rule were locked in `capacity_transfer_proposal_v7.md` and `capacity_transfer_check_v7.md` (protocol SHA-256 `aff9addbf88ba54223170202266b86c089a54ddd686327727bdc711c6d8fc5e8`) before collection. This is a zero-shot N4/K2 service-duration robustness check of already-selected v3 PPO matching policies on unchanged simulator physics and H+tail economic cost. S is a structured centralized shared machine/worker actor, R removes numerical remaining-time inputs, and G3 is a flat full-information actor with the same legal matching/WAIT action set and 8,192 source training decisions per case/root. They are **not** two separate machine and coordinator policies. The fixed threshold PM request contract and automatic CM remain in force.

## Verification and provenance

- Runner, auditor and five focused tests were committed/pushed as `538e6bb` before any pilot outcome. The full repository suite passed **144 tests**. The bounded Ubuntu engineering smoke `artifacts/capacity_transfer_v7_smoke_20261022T115120375560Z` replay-audited **PASS** for 100 episodes on two engineering seeds and one source→target pair. An earlier complete smoke `artifacts/capacity_transfer_v7_smoke_20261022T115008681146Z` is retained; its first audit attempt exposed an auditor episode-count typo, corrected before the final smoke and pilot. Neither smoke is scientific evidence. This was Ubuntu validation, not macOS validation.
- The committed collection code `538e6bb` produced `artifacts/capacity_transfer_v7_pilot_20261022T115212108531Z`: **COMPLETED**, clean Git tree at start, 32 new paired raw seeds, two preselected source→target pairs, three frozen v3 roots per learned arm, and 3,200 episodes (1,152 S/R/G3, 1,152 D3/historical G/D, 896 F1/F2/V14). It recorded 153,600 H decisions, 51,200 tail transitions, 1,345,824 F2 forecast transitions and 197.08 seconds of collection wall time. All raw episode action/submission traces, manifests, checkpoint/source/config/protocol hashes and partial-progress bookkeeping remain in the timestamped directory.
- The first pilot-audit process was stopped after inspection found an auditor-only bug: JSON serialization sorts the condition names, while the auditor compared their values by position. Commit `658800a` compares by condition name and checks every collection-source file against the exact `538e6bb` Git content; it changes no raw row, simulator, actor, checkpoint, estimator, threshold or pilot decision. The complete rerun of the amended auditor replayed **all 3,200 episodes and PASSed** checkpoint/source/protocol identities, physical pairing, legal actions, H+tail accounting, F2 forecasts, panel coverage and independent gate arithmetic. The audit records its own SHA-256. The original pilot manifest and raw files were not rewritten.

## Paired H+tail comparisons

Lower cost is better. The two target cases are `reciprocal_scale1` and `specialist_generalist_scale3`, paired with their respective scale-2 source cases on the same 32 physical seed IDs. Estimates average cases, three fixed training roots and seeds equally. The 95% percentile intervals use 5,000 resamples of **raw physical seed IDs as clusters** across cases, source/target domains, arms and roots. They condition on these trained checkpoints; three root means do not measure the variation of a broader training population.

| Locked comparison | Overall | Reciprocal target/pair | Specialist target/pair | Cluster 95% interval | Training-root overall means |
| --- | ---: | ---: | ---: | --- | --- |
| Target S−G3 | −26.52 | −74.41 | **+21.38** | [−41.45, −11.56] | −24.50, −38.94, −16.11 |
| Transfer difference: target (S−G3) minus source (S−G3) | −0.43 | −32.60 | **+31.75** | [−17.91, +18.04] | +14.97, −29.45, +13.20 |
| Target S−R | −0.16 | 0.00 | −0.31 | [−0.47, 0.00] | −0.38, −0.09, 0.00 |

S beats G3 on the shorter reciprocal target, but loses on the longer specialist/generalist target. The transfer contrast is essentially zero overall and changes sign by pair. S and R choose nearly the same behavior: their complete action/submission traces match in 86/96 reciprocal-target episodes and 95/96 specialist-target episodes; their target cost gap is only −0.16, with an interval touching zero. This does not establish a numerical technician-commitment benefit. The target S−G3 overall estimate also misses the locked practical ≤−30 threshold despite its negative interval.

| Arm | Reciprocal source | Reciprocal target | Specialist source | Specialist target |
| --- | ---: | ---: | ---: | ---: |
| S: structured PPO | 803.21 | 490.66 | 988.84 | 1197.82 |
| R: numerical-time ablation | 804.54 | 490.66 | 988.84 | 1198.14 |
| G3: flat full-information PPO | 845.01 | 565.06 | 999.22 | 1176.45 |
| D3: v3 always-request PPO, contextual | 1368.45 | 1316.81 | 1336.90 | 1382.26 |
| G: historical fixed-threshold learned coordinator, contextual | 818.23 | 563.79 | 996.92 | 1164.49 |
| D: historical always-request learned coordinator, contextual | 1368.02 | 1200.14 | 1343.02 | 1399.08 |
| A: frozen V14 five-checkpoint reference | 792.89 | 490.44 | 994.26 | 1186.84 |
| F1: risk matching | 830.59 | 490.72 | 1010.25 | 1242.25 |
| F2: finite six-tick rollout | 735.94 | 466.78 | 973.41 | 1129.00 |

The reference arms are descriptive. D3/G/D differ in request contract or training budget, V14 has a historical regime, and F2 spends known-model forecasts; none is an equal-budget architecture attribution control. On both targets F2 has the lowest listed mean cost, but is a finite rollout reference, not an optimum. S/R actors each have 8,553 parameters versus G3's 8,705; all three have the same 7,873-parameter critic and use zero deployment forecasts. Mean per-episode H selection time on reciprocal target was about 0.04/0.05/0.02 seconds for S/R/G3 versus 0.98 seconds for F2; on specialist target it was 0.04/0.05/0.02 versus 0.42 seconds. Runtime is machine-specific.

Secondary target behavior helps locate the failure without proving a causal mechanism. On reciprocal target, S had mean 1.66 failures and 3.76 queued PM ticks versus G3's 6.68 and 14.67; S matched F1/V14 closely. On specialist target, S had mean 14.10 failures, 32.44 queued PM ticks and 31.66 queued CM ticks versus G3's 12.86, 30.01 and 34.78. S availability was 0.374 versus G3 0.377. These values come from the same completed episode panel. They do not justify changing the locked gate or selecting only the reciprocal shift.

## Gate and decision

The audited `replication_go_decision.json` is **false**. Target S−G3 fails the ≤−30 overall and both-target-sign conditions. Transfer fails the ≤−10 overall, both-pair-sign and interval-upper-below-zero conditions. Target S−R fails the ≤−10 overall and interval conditions. Other passed subconditions do not override these failures. Under the locked protocol, **no fresh training replication and no new TEST** are eligible; neither was run. The reserved TEST seed role remains unopened.

V7 does not provide a stable service-duration-transfer advantage over the same-information/action/budget flat PPO control, or evidence that numerical remaining-time features contributed. This is a negative result under two existing N4/K2 duration shifts and frozen v3 source policies. Same-simulator replay validates implementation/accounting, not physical realism or performance at other skills, N/K or real sites. The broader demand–capacity framework contribution remains unproven. A further experiment would need a separately grounded operational contract or external domain evidence, then a new protocol; the present data should not be used to tune another scorer or reopen TEST.
