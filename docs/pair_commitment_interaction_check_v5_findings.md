# V5 joint service-start interaction — audited bounded findings

Date: 2026-10-22 (Asia/Taipei). The proposal and diagnostic protocol were locked and pushed in commit `7836d69` before v5 outcomes; protocol SHA-256 `5b1fedab55c9e85bc94499f5fe17eec04d3e2e6dd0137c124158cc6bdc21b345`. Verified code was pushed in `d4a12f1` before the scientific check. V1 TEST and v2–v4 verdicts remain closed.

## Verification and scope

The full repository suite passed 129 tests. The two-episode Ubuntu engineering smoke `artifacts/pair_commitment_v5_smoke_20261022T084352280973Z` passed replay audit and is not scientific evidence. The locked check artifact `artifacts/pair_commitment_v5_check_20261022T084457062057Z` completed 32 F2 trajectories on 16 new paired raw physical seeds per case, recording all 1,536 H states and complete H+tail episode costs. It spent 324,192 F2 selection forecast transitions plus 84,096 independent-bank transitions at two-start-eligible states; wall time was 26.25 s. Its replay audit passed source/protocol/config/raw hashes, fresh raw/derived seed roles, exact F2 trajectory/root scores, independent bank, additive choice, matching legality, H+tail accounting, complete coverage and gate arithmetic. The audit replay uses the same simulator/model, not an independently validated physical model.

The diagnostic compares **additive minus F2** finite six-tick forecast cost on the *same F2-visited states*. The additive rule uses F2's WAIT and singleton root scores from the same 16-scenario selection bank; two-start scores are approximated as the sum of singleton effects. Both selected roots are evaluated on one independent common 32-scenario bank. Positive difference favors assessment of the complete matching. This is neither realized H+tail advantage nor a learned-policy comparison. F2 is a finite-scenario reference, not an optimum.

| Predeclared stratum | Two-start states / all H states | Choice disagreement within stratum | Independent-bank mean within stratum | Independent-bank mean over all H states | Mean selection-bank pair interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reciprocal (primary) | 151/768 (19.66%) | 50.99% | **+2.91** | **+0.57** | −19.06 |
| Specialist/generalist (contrast) | 68/768 (8.85%) | 19.12% | +0.20 | +0.02 | −0.75 |

The reciprocal conditional 95% interval is **[+1.53,+4.65]**, from 5,000 draws resampling the 16 raw physical seed IDs as clusters. All 48 overlapping H states and both cases stay together within a sampled seed; scenarios and state rows are not independent observations. Episode counts and means are retained in `analysis.json`. The primary gate required reciprocal pair frequency ≥10%, disagreement ≥20%, conditional mean ≥+1.5, all-state mean ≥+0.20, and interval lower endpoint >0; **all passed**. The specialist case was descriptive and did not alter the gate.

## Interpretation and next decision

On this finite F2 state distribution, reciprocal simultaneous-service starts exhibit material nonadditivity under the same forecast conditions. The independent bank confirms a bounded representation opportunity relative to a specific additive decomposition, while the specialist case shows the mechanism is not equally prominent in both case structures. The large negative interaction score includes future risk-matching continuation and forecast sampling; it cannot be attributed solely to physical service coupling or called a calibrated machine value. A learned joint scorer may still fail through function approximation, training limits or on-policy distribution shift. This result does not establish headroom *over F2*, a realized episode cost improvement, a general N/K result, or a successful framework.

The locked pass permits a **new, separately prelocked** same-information, same-matching learner pilot with an explicit joint matching term and an edge-additive ablation. It does not reopen v4 replication/TEST, authorize any new TEST, or relax the need for flat centralized, frozen V14, risk and F2 controls. Preserve all raw rows and the original v5 gate.
