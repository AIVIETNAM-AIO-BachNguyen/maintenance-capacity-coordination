# V2 demand/commitment headroom check — findings

Date: 2026-10-22 (Asia/Taipei). This report evaluates the single bounded check locked in `commitment_headroom_v2.md` before its outcome. It does not use or reopen pilot v1 TEST. The design proposal is `demand_capacity_commitment_proposal_v2.md`; its structured learner has **not** been implemented or trained.

## Evidence and audit

The bounded Ubuntu engineering smoke completed at `artifacts/commitment_headroom_v2_smoke_20261022T070658153963Z`; four focused tests and the full 109-test suite passed before the scientific panel. The check artifact is `artifacts/commitment_headroom_v2_check_20261022T070841552901Z`. Its manifest has terminal `COMPLETED` status, 32 F2 physical episodes, all 256 predeclared H states (16 raw seeds × eight ticks × two cases), and 572,832 forecast transitions. Raw state/episode JSONL, protocol/source/config hashes, versions, timing and partials are preserved. `experiments/audit_commitment_headroom_v2.py` replayed all 32 F2 trajectories and passed protocol/source/config/raw hashes, exact state coverage, legal threshold and expanded matchings, just-in-time submissions, physical/selection/holdout seed identities, six-tick cost arithmetic, episode aggregation and forecast-transition accounting. Its `audit.json` and `analysis.json` are inside the same artifact. The replay is an independent run of the same simulator/forecast implementation, not an independent physical model.

## Predeclared paired outcome

The comparison is expanded immediate-PM matching minus F2's threshold-admitted matching, scored on an independent **common** 32-scenario holdout forecast bank after choices were made on a common 16-scenario bank. The quantity is six-tick model forecast cost on F2-visited states, not realized H+tail episode cost or an optimum gap. Within each case, all eight overlapping sampled windows are clustered by physical episode for 5,000 bootstrap draws, seed 73172000. The same raw physical seed appears in both cases, so a second sensitivity interval resamples its two case episodes together.

| Case | F2 holdout mean | Expanded holdout mean | Paired difference | Extra PM chosen |
| --- | ---: | ---: | ---: | ---: |
| Reciprocal scale 2 | 67.16 | 66.43 | −0.73 | 9/128 states (7.0%) |
| Specialist/generalist scale 2 | 92.28 | 92.12 | −0.15 | 23/128 states (18.0%) |
| Equal-case mean | 79.72 | 79.28 | **−0.44** | 32/256 states (12.5%) |

The predeclared within-case episode-cluster 95% interval is [−0.77, −0.14]; the paired cross-case raw-seed cluster sensitivity interval is [−0.80, −0.14]. On the selection bank, the average paired changes were −0.86 and −0.58 by case, somewhat larger than independent holdout changes. Among states where extra PM was selected, the holdout change averaged −10.38 in the reciprocal case but only −0.85 in specialist/generalist; this selected subset is descriptive, not a separate gate. Some selected extra actions were worse on holdout, especially in specialist/generalist.

The locked go gate required overall mean at most **−2.0** economic units per six-tick decision, interval upper endpoint below zero, both case means below zero and extra PM selection on at least 5% of states. The effect-size requirement failed (−0.44 > −2.0), while the other conditions passed. **No v2 learning pilot opens under this protocol.** The negative gate is retained despite a small conditional forecast benefit. The check shows that expanding immediate early-PM eligibility alone leaves limited measured headroom over strong F2 in these sampled states; it does not rule out other demand/commitment formulations, longer-term interactions, or a finite-budget learning benefit. It also does not justify a claim that capacity feedback is useless.

## Research decision

Do not implement/train the proposed structured learner under this versioned contract or tune the −2.0 gate, sampled states, scenarios or v1 TEST. Reconsider the operational formulation: the next question must identify a decision or commitment mechanism whose potential gain over strong controls can be demonstrated without manufacturing new cost or information restrictions. A new proposal and seed registry are required before another scientific check. The v1 pilot, G diagnostic and this failed headroom check remain separate evidence with their original scope.
