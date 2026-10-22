# Current state

Date: 2026-10-22, Asia/Taipei. V1–v3 are completed with their original gates and raw artifacts preserved. V4 forecast-informed demand/commitment coordination is the sole newly locked experiment; see `docs/forecast_demand_commitment_proposal_v4.md` and `docs/forecast_demand_commitment_pilot_v4.md`.

- V1 binary-requester pilot and its authorized 32-seed sealed TEST are complete and closed. C beat learned always-request D, but capacity feedback did not establish a robust advantage over matched/full-information and strong controls. Its labelled “replication” was held-out evaluation of fixed checkpoints, not independent training replication. The separate fixed-threshold G diagnostic favored G over C on its own panel.
- V2's prelocked early-PM headroom check passed audit but missed its practical forecast-gain gate (−0.44 versus −2.0). No v2 learner was trained.
- V3 retained fixed-threshold requests and the same complete matching action set for S, R and flat full-information G3. All 24 training trials, Ubuntu smoke, 114 tests and the held-out artifact audits passed. S−G3 was −22.27 H+tail, short of its locked −30 gate; S−R was −0.22 and did not support a numerical remaining-time benefit. Frozen V14 and F2 were lower cost overall. Independent training replication and v3 TEST remain closed by protocol. The user's later TEST authorization does not change the failed gate.
- Raw artifacts, failed gates and the v1 TEST are preserved. The current branch is `codex/forecast-demand-commitment-v4`; prior v1–v3 branches remain pushed.

## One next experiment

The prelocked v4 teacher-amortization runner and auditor are implemented. F2 supplies same-contract matching labels; S4, G4 and R4 receive identical data and budgets. The full 123-test suite and final bounded Ubuntu smoke/audit passed (`artifacts/forecast_distillation_v4_smoke_20261022T081720658860Z`). Next, commit/push this verified code, run only the locked training panel, and audit it before held-out. No new TEST is currently eligible.
