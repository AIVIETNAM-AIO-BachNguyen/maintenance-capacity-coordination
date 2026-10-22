# Current state

Date: 2026-10-22, Asia/Taipei.

- Pilot v1 is closed. Its frozen TEST and audits remain at `artifacts/pilot_v1_test_20261022T064459750562Z`: C−D was −187.42 H+tail cost, while C−B varied by case/seed and E, V14 and strong controls were better. The separately prelocked G diagnostic favored fixed threshold requests plus a learned coordinator over C on its own panel. V1 “replication” was held-out physical evaluation of old checkpoints, not independent training replication. Neither v1 TEST nor G's diagnostic panel can select a new design.
- The revised objective is a structured centralized demand/technician-commitment framework. Its candidate contract retains threshold requests and permits just-in-time PM submission only for additional healthy machines actually selected for service. The unchanged simulator executes all physics, CM, non-preemption and economic costs. The proposal is `docs/demand_capacity_commitment_proposal_v2.md`; no v2 learner has been implemented or evaluated.
- The sole current experiment is the prelocked bounded headroom check in `docs/commitment_headroom_v2.md`. Its implementation is `experiments/commitment_headroom_v2.py`; four focused tests, the 109-test full suite, and a bounded Ubuntu smoke passed. No scientific headroom outcome has been opened yet.

## One next experiment

Run and audit the locked headroom check on 16 new physical seeds per case. Open a v2 learning pilot only if its predeclared held-out forecast gate passes; otherwise retain the negative result and revisit the formulation without tuning on v1 TEST.
