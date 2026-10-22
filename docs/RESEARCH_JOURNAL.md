# Research journal

## 2026-10-22 — pilot v1 prelock and engineering gate

Decision: test finite-budget coordination in two existing N4/K2 cases. Shared machine requesters in B/C choose PM timing; C alone receives present compatible-technician capacity. D always requests and E uses full pre-decision state. All coordinators observe the admitted full current state and use the same feasible matching decoder. The environment and economic reward remain unchanged. The exact joint PPO likelihood includes both sampled stages. Protocol, seed roles, budgets, checkpoint rule, controls and TEST gate are in `coordination_pilot_v1.md`.

Evidence before research outcomes: 104 repository tests pass; all B/C/D/E arms completed a 256-transition Ubuntu engineering smoke in each case with checkpoint reload and deterministic evaluation. The unchanged V14 verifier passed on Ubuntu (240 full-Q replays, 640 selected episode replays). Smoke seeds are open engineering seeds and establish no learning benefit. The V14 Mac replay mismatch remains disclosed; fresh pilot panels use the Ubuntu baseline.

Next uncertainty: whether C improves paired H+tail cost or finite-budget learning over B, D and E. Run the locked DEV panel, audit it, then replicate without tuning. A negative result remains a result.

## 2026-10-22 — DEV and replication outcome

DEV completed all 24 locked trials and passed the artifact audit. Replication passed coverage, source/protocol, legality, H+tail and seed-separation audits. The predeclared C–D cost gate passed in both cases and all three training seeds (overall C−D −182.09); C–B and C–E did not pass. Frozen V14 and strong rules/rollout remain much better than the learned arms. D's low availability and excessive PM starts explain much of its gap, consistent with finite-budget optimization difficulty rather than structural need for a requester. Full data, uncertainty and limitations are in `pilot_v1_replication_findings.md`. TEST has not been opened. A supplementary same-panel replay recovered control WAIT/request-creation diagnostics and exactly matched original cost/service starts.

Decision: retain the negative B/E and strong-control comparisons. The locked gate makes TEST eligible, but the explicit lab authorization did not name TEST. Request permission before opening it. Do not change source or checkpoints meanwhile. A fixed-threshold learned-coordinator diagnostic could isolate attribution after TEST or if TEST is declined; it would require a versioned follow-up and cannot be treated as a prelocked v1 arm.

A post-run test independently enumerated the feasible joint request/matching action space for B/C/D/E and confirmed normalization to one. It changes only test code, not the locked research source or protocol. The full suite passed 104 tests; the focused test passed without warnings.

A further replication audit verified the exact derived physical tape for all 608 episodes and the identity/hash of every selected learned checkpoint; no sealed TEST seed appeared in episode rows.

## 2026-10-22 — fixed-threshold coordinator attribution diagnostic

Question: did C beat always-request D because its requester learned useful capacity-aware timing, or because it gated PM submissions at all? The separately locked `threshold_requester_diagnostic_v1.md` introduced G: inherited fixed threshold requests plus D's coordinator-only PPO, with identical H budget and matched three training seeds. G training, DEV selection and the new 32-seed physical panel completed; audits passed all six trials and 960 paired G/B/C/D/E episodes. G−C was −263.16 and G−D −446.51, negative in both cases and all three training seeds. An explicitly post-outcome descriptive supplement evaluated frozen V14 and rule/rollout on the same panel; its 448 episodes passed accounting and physical-pairing audit. G was close to V14/risk matching, with rollout still lower. Details and limitations: `threshold_requester_diagnostic_findings.md`.

Decision: the pilot does not support a learned capacity-aware requester advantage. Fixed threshold request gating plus a learned coordinator is the simpler effective control at this budget. Retain the negative result and do not add prediction, pricing, reward shaping or scale expansion without a new grounded question. Pilot v1 TEST stays sealed pending the already-requested explicit approval; no source or checkpoint was changed to fit the diagnostic result.

## 2026-10-22 — authorized sealed TEST

The user explicitly authorized the prelocked 32-seed TEST and creation/push of `codex/coordination-pilot`. The branch was created without changing source; source and protocol hashes matched the completed DEV and replication manifests before TEST. The runner evaluated unchanged selected checkpoints and A/F controls on a fresh, complete panel. The independent TEST audit passed all 768 learned and 448 control episodes, including pairing, raw hashes, feasibility and economic cost decomposition. C−D was −187.42, favorable in both cases and every training seed. C−B was mixed by case/seed; C−E was +17.84; frozen V14, risk matching and rollout were all substantially lower cost. See `pilot_v1_test_findings.md` and the preserved TEST artifact for intervals and secondary measures.

Decision: the pilot confirms only a narrow finite-budget C−D difference. The post hoc G diagnostic and strong same-panel TEST controls prevent attributing a general benefit to learned capacity feedback. Close TEST to tuning, publish the verified pilot faithfully, and require a newly locked failure-mode hypothesis before further research runs.

## 2026-10-22 — revised demand/technician-commitment proposal and headroom prelock

The new objective separates the physical maintenance/service commitment problem from v1's free binary request gate and its finite-budget learning difficulty. The proposed machine component is a shared demand representation, not an independent policy or calibrated economic value. A full-information coordinator will eventually score complete feasible matchings including WAIT. The v2 candidate contract keeps ordinary threshold requests and allows just-in-time submission for an additional PM candidate only if it is selected for immediate service; the inherited simulator still executes the physical transition. Exact G/D remain important controls but differ in action set, so architecture attribution requires same-action direct-health PPO and capacity-context ablation. Full proposal: `demand_capacity_commitment_proposal_v2.md`.

Before any v2 learning outcome, `commitment_headroom_v2.md` locked one bounded check against strong F2 rollout on paired states and common 16-scenario selection forecasts, with a separate common 32-scenario holdout bank. It samples fixed H ticks in 16 new physical episodes per case and uses episode-cluster uncertainty. The gate requires a meaningful paired held-out reduction, both case means below zero, and actual use of extra PM options. The diagnostic is not an optimum oracle or an H+tail claim. Four focused tests, the full 109-test suite and a bounded Ubuntu smoke passed before opening the scientific panel.

## 2026-10-22 — bounded v2 headroom outcome

The locked check completed 32 F2 episodes and 256 sampled states on new physical seeds. Replay audit passed legality, fixed selection/holdout forecast seed roles, hashes, coverage, cost and 572,832 forecast transitions. The expanded early-PM catalogue was selected in 32 states (12.5%), but its paired independent six-tick holdout difference was only −0.44 cost units, with a predeclared within-case episode-cluster interval [−0.77, −0.14]. A shared-raw-seed cross-case cluster sensitivity interval was [−0.80, −0.14]. The locked minimum meaningful gain was −2.0, so the go gate failed. Full results and limits are in `commitment_headroom_v2_findings.md`.

Decision: do not open a v2 learning pilot for this action-set expansion, even though there is a small positive model-forecast effect. Do not reinterpret this as evidence that demand representation or capacity feedback can never help. The next formulation must expose a concrete, potentially material long-term commitment mechanism against strong controls; it requires a new proposal, protocol and seeds. V1 TEST remains closed.

## 2026-10-22 — distinct structured matching pilot v3 prelock

After the v2 early-PM option failed its own headroom gate, the separate v3 hypothesis asks whether machine/worker factorization improves finite-budget PPO learning on the **unchanged threshold-request matching set**. It does not enlarge physical options or rescue the v1 binary requester. The shared machine features are uncalibrated learned representations; the actor has one masked matching action. `structured_demand_commitment_proposal_v3.md` states the mechanism and falsifiers; `structured_demand_commitment_pilot_v3.md` locks fresh seed roles, 8,192-H-decision budgets, DEV selection, held-out and independent training replication gates, strong controls, and shared-physical-seed cluster analysis. S/R actors each have 8,553 parameters and flat G3/D3 each 8,705; all critics have 7,873. The full suite passed 114 tests; the fresh eight-trial Ubuntu smoke and its source/selection/cost audit passed before DEV. No v3 learning outcome has been opened.

## 2026-10-22 — v3 DEV and held-out no-go

All 24 locked S/R/G3/D3 trials completed 196,608 H decisions and 1,152 DEV episodes; the training audit passed source/config/protocol, selection, seed, legality and H+tail checks. The held-out panel used unchanged checkpoints on 32 fresh physical seeds per case and passed audit for 768 learned plus 448 A/F control episodes. S−G3 mean was −22.27 with raw-physical-seed cluster interval [−33.92, −11.12]; the three training-seed means were −37.00, +6.30, −36.11. S−R was −0.22 with interval [−0.95, +0.32]. A and F2 remained lower cost overall. Learning areas and runtime are reported separately in `structured_demand_commitment_pilot_v3_findings.md`.

Decision: the practical H2 gate required at least −30, so it failed despite a negative conditional physical interval. H3 did not support a numerical remaining-time contribution. No independent training replication or TEST is eligible under v3. Preserve negative/mixed results and formulate a new operational question rather than adjusting architecture, thresholds or seed selection to produce a win.

## 2026-10-22 — primary-literature positioning and campaign decision

The author-hosted Rodríguez paper and primary AirME/HetGPO paper both address maintenance scheduling with shared service resources. Rodríguez gives machine agents technician/component or delay actions; AirME uses a resource/decision graph for crew allocation. Neither permits this project to claim novelty from machine agents, capacity context or an entity encoder alone. The comparison and evidence boundaries are recorded in `research_positioning_and_decision.md`.

Decision: no further scientific run follows automatically from v2's limited early-PM headroom or v3's failed practical and capacity-ablation gates. A later explicit user authorization to run a 32-seed sealed TEST is conditional on the locked protocol's eligibility; v1 TEST is already complete, and v3 TEST remains ineligible. The next research decision is to identify a distinct operational mechanism with a fresh falsifiable bounded headroom protocol, or retain the negative/mixed result. No threshold, seed role or earlier TEST is reopened.

## 2026-10-22 — service-horizon mechanism screen

A source/config screen tested whether F2's six-tick lookahead truncates the technician commitments that dominate v3's cost gap. Reciprocal service times are only 2 or 6 ticks, yet its held-out S−F2 gap is +88.66; specialist/generalist has one 8-tick edge with a 2-tick alternative and a much smaller S−F2 gap of +7.65. This does not measure longer-horizon forecasts, but it rules out unfinished service beyond tick six as an explanation of the largest observed gap. Decision: do not open a horizon-length campaign on this rationale alone; retain the need for a distinct operational mechanism and fresh prelock before any new run.

## 2026-10-22 — forecast-informed coordination v4 prelock

A distinct, same-contract question follows from the audited v3 cost/runtime tradeoff: F2's 16-scenario six-tick rollout was lower cost than S on the same panel but required much more action-selection compute, and its implementation exposes a forecast score for every legal matching. The opened v2 state panel also showed multiple legal roots and 10–12 distinct F2 choices by case. This supports testing whether a shared machine/worker policy can amortize a strong decision rule, rather than adding a requester gate or changing physics.

`forecast_demand_commitment_proposal_v4.md` specifies the forecast-distilled structured centralized mechanism. `forecast_demand_commitment_pilot_v4.md` locks new training/DEV/held-out/replication/TEST seed roles, identical F2 labels and masked cross-entropy budgets for S4/R4/G4, same-panel F2/A/F1 controls, H+tail estimands, physical-seed clustering, compute accounting and practical gates before any v4 teacher label or outcome. V1–v3 gates and sealed TEST remain unchanged. Next: implement and test only this pilot, run bounded Ubuntu smoke, then collect qualified v4 data.

## 2026-10-22 — v4 engineering gate

The forecast-distillation runner and independent replay auditor now implement the locked F2 teacher, identical state/label budgets for S4/R4/G4, deterministic H+tail evaluation, fresh timestamped manifests, paired physical-seed analysis and no TEST entry point. Seven focused checks plus two paired-analysis root variants passed; the full repository suite passed 123 tests. The final bounded Ubuntu engineering smoke `artifacts/forecast_distillation_v4_smoke_20261022T081720658860Z` used one case, two teacher episodes and one epoch for each arm. Its audit replayed all teacher labels, root scores, DEV episodes, hashes and transition counts and reported PASS. The earlier smoke artifact is preserved but not used as the final engineering gate. No scientific v4 training/DEV/held-out outcome has been opened yet.

Decision: commit/push verified v4 code, then run only the prelocked training panel and its audit. Held-out evaluation requires that training audit to pass. No new TEST is eligible.

## 2026-10-22 — v4 training and held-out no-go

The prelocked v4 source/protocol hashes remained unchanged through the scientific panel. F2 generated 4,608 teacher state/label rows from 96 episodes using 944,064 selection forecast transitions; all 18 S4/R4/G4 trials used the same case/root data and fixed 32-epoch budget. Training plus DEV-selection replay audit passed. The 32-seed-per-case held-out panel completed 576 student and 448 F2/F1/V14 control episodes, then passed independent replay of actions, checkpoint selection/source identity, physical pairing, H+tail costs, forecast counts and gate arithmetic. Artifacts are `artifacts/forecast_distillation_v4_train_20261022T082000203886Z` and `artifacts/forecast_distillation_v4_heldout_20261022T082342680537Z`; full estimates and limits are in `forecast_demand_commitment_pilot_v4_findings.md`.

S4−G4 was +0.57 H+tail (cluster interval [−10.33,+12.07]); S4−R4 −0.37 ([−1.15,+0.22]); S4−F2 +35.93 ([+15.11,+54.44]). S4 used 2.51% of F2's action-selection CPU time per held-out episode but missed the predeclared ≤+20 cost gap, especially in reciprocal (+67.90). The structural and numerical-commitment gates also failed. Decision: retain the no-go, do not open independent training replication or v4 TEST, and do not tune labels/architecture/seed/checkpoint on this panel. The desired validated capacity–commitment contribution remains unproven; any further scientific experiment needs a new operational mechanism with independent justification.
