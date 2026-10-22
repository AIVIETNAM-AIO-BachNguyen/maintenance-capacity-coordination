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
