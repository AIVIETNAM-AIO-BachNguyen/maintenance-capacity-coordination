# Current state

Date: 2026-10-22, Asia/Taipei. V1–v3 are completed with their original gates and raw artifacts preserved. V4 forecast-informed demand/commitment coordination is the sole current experiment and is now complete; see `docs/forecast_demand_commitment_pilot_v4_findings.md`.

- V1 binary-requester pilot and its authorized 32-seed sealed TEST are complete and closed. C beat learned always-request D, but capacity feedback did not establish a robust advantage over matched/full-information and strong controls. Its labelled “replication” was held-out evaluation of fixed checkpoints, not independent training replication. The separate fixed-threshold G diagnostic favored G over C on its own panel.
- V2's prelocked early-PM headroom check passed audit but missed its practical forecast-gain gate (−0.44 versus −2.0). No v2 learner was trained.
- V3 retained fixed-threshold requests and the same complete matching action set for S, R and flat full-information G3. All 24 training trials, Ubuntu smoke, 114 tests and the held-out artifact audits passed. S−G3 was −22.27 H+tail, short of its locked −30 gate; S−R was −0.22 and did not support a numerical remaining-time benefit. Frozen V14 and F2 were lower cost overall. Independent training replication and v3 TEST remain closed by protocol. The user's later TEST authorization does not change the failed gate.
- V4's prelocked forecast-distillation panel completed with 123 tests and final Ubuntu smoke PASS, 4,608 teacher state/label rows, 18 student trials and 32 paired held-out physical seeds per case. Training and held-out replay audits PASS. S4−G4 was +0.57, S4−R4 −0.37, and S4−F2 +35.93 H+tail cost; only the deployment-speed condition passed. The replication gate failed. All raw artifacts and negative findings remain in `docs/forecast_demand_commitment_pilot_v4_findings.md`; no v4 independent training replication or TEST is eligible.
- Raw artifacts, failed gates and the v1 TEST are preserved. The current branch is `codex/forecast-demand-commitment-v4`; prior v1–v3 branches remain pushed.

## One next research decision

Reassess whether a distinct operational mechanism with credible headroom remains in this simulator. Do not tune v4 on its held-out panel or open another architecture/run solely to find a positive result. If no justified mechanism exists, report the negative/mixed framework evidence rather than claiming a validated capacity–commitment benefit.
