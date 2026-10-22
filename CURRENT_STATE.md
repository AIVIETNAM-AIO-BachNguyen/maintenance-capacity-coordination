# Current state

Date: 2026-10-22, Asia/Taipei.

- Pilot v1 is closed. Its frozen TEST and audits remain at `artifacts/pilot_v1_test_20261022T064459750562Z`: C−D was −187.42 H+tail cost, while C−B varied by case/seed and E, V14 and strong controls were better. The separately prelocked G diagnostic favored fixed threshold requests plus a learned coordinator over C on its own panel. V1 “replication” was held-out physical evaluation of old checkpoints, not independent training replication. Neither v1 TEST nor G's diagnostic panel can select a new design.
- The revised objective is a structured centralized demand/technician-commitment framework. Its candidate contract retains threshold requests and permits just-in-time PM submission only for additional healthy machines actually selected for service. The unchanged simulator executes all physics, CM, non-preemption and economic costs. The proposal is `docs/demand_capacity_commitment_proposal_v2.md`; no v2 learner has been implemented or evaluated.
- The sole current experiment, the prelocked bounded headroom check in `docs/commitment_headroom_v2.md`, is complete. The 32-episode/256-state panel and independent replay audit passed. Extra immediate PM was selected in 12.5% of states, but its paired six-tick holdout gain was only 0.44 cost units against strong F2, below the locked 2.0-unit minimum. The v2 pilot gate failed; no structured learner has been implemented or trained. See `docs/commitment_headroom_v2_findings.md`.

- A distinct same-action structural hypothesis is now locked in `docs/structured_demand_commitment_pilot_v3.md`: S shared machine/worker matching scorer versus full-information flat G3, remaining-time ablation R, and contextual always-request D3. It retains the fixed threshold request pool, so it does not reopen the v2 early-PM action-set gate. The learner and four arms are implemented; 114 repository tests passed, and a fresh eight-trial Ubuntu smoke plus artifact audit passed (`artifacts/structured_pilot_v3_smoke_20261022T072512442867Z`). No v3 DEV or held-out result has been opened.

## One next experiment

Run the locked v3 DEV training on fresh seeds, audit every trial and selected checkpoint, then evaluate the predeclared held-out physical panel without tuning. Open independent training replication only if the written S−G3 gate passes. Keep all TEST roles sealed.
