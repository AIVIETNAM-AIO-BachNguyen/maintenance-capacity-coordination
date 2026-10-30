# Coordination pilot v1 — DEV and replication findings

Date: 2026-10-22 (Asia/Taipei). Here “replication” means held-out physical-seed evaluation of fixed selected checkpoints, not independent training replication. Protocol: [`coordination_pilot_v1.md`](coordination_pilot_v1.md). These findings use the unchanged `shared_maintenance_requests_v1` and `pending_matching_completion_v1` simulators. No sealed TEST seed has been evaluated.

## Audit and panel

The Ubuntu V14 verifier passed unchanged: 240 full-Q replays and 640 selected-policy episode replays, including source/artifact hashes (`artifacts/v14_audit_20261022T053515407392Z`). All 104 repository tests passed. A bounded Ubuntu smoke completed for B/C/D/E in both cases (`artifacts/pilot_v1_smoke_20261022T053635426675Z`); its seeds are engineering checks only.

The DEV training run is `artifacts/pilot_v1_train_20261022T061152746250Z`. Its independent artifact audit passed: all 24 trials completed their locked 8,192 H decisions (196,608 total), all 1,152 learned DEV episodes and 224 control episodes are present, and every choice, return, H+tail decomposition, seed role, checkpoint-selection rule, source hash and protocol hash passed audit. These H counts exclude the common 16 physical tail ticks after each completed training episode; 170 complete episodes per trial add 65,280 tail physical transitions across training. The replication run is `artifacts/pilot_v1_replicate_20261022T061743244003Z`; its audit passed for all 384 learned and 224 control episodes. Both audits are stored as `audit.json` in their runs. A separate replication pairing audit verified all 608 episode physical-tape seeds and every selected checkpoint identity/hash (`pairing_audit.json`). No TEST seed was evaluated, and no result was deleted or tuned after the panel opened.

## Primary cost result

Mean H+tail economic cost on the same 16 replication physical seeds (lower is better):

| Arm | Reciprocal scale 2 | Specialist/generalist scale 2 | Overall mean |
| --- | ---: | ---: | ---: |
| A: threshold + frozen V14 count (5 historical training seeds) | 813.31 | 999.39 | 906.35 |
| B: shared local requester + learned coordinator (3 training seeds) | 1151.69 | 1211.83 | 1181.76 |
| C: capacity-aware shared requester + learned coordinator (3 training seeds) | 1137.42 | 1219.04 | 1178.23 |
| D: always-request + learned coordinator (3 training seeds) | 1369.54 | 1351.10 | 1360.32 |
| E: full-information centralized requester + learned coordinator (3 training seeds) | 1094.71 | 1200.90 | 1147.80 |
| F1: threshold + risk matching | 863.19 | 994.56 | 928.88 |
| F2: threshold + 6-tick rollout | 701.31 | 979.00 | 840.16 |

The prelocked primary paired C-minus-control estimates, equally weighting case and training seed, are:

| Comparison | Mean difference | 95% paired physical-seed bootstrap, conditional on the 3 training seeds | Per-training-seed differences | Prelocked gate |
| --- | ---: | ---: | --- | --- |
| C − B | −3.53 | [−17.59, 9.73] | −5.91, −29.28, +24.59 | Fail |
| C − D | −182.09 | [−196.51, −167.28] | −222.94, −154.41, −168.94 | Pass |
| C − E | +30.43 | [+15.80, +44.84] | +11.94, −7.94, +87.28 | Fail |

The bootstrap resamples physical seeds within case and training seed (5,000 draws, seed 73150000); it does not make three training seeds a large population. As a sensitivity check, t intervals over the three seed-level differences are C−B [−70.65, 63.59], C−D [−271.80, −92.38] and C−E [−94.36, 155.22]. The latter two interval types address different uncertainty and should not be read interchangeably. The replication gate in the protocol passes **only** for C versus D, and `test_go_decision.json` records that. TEST remains sealed pending explicit authorization to open it.

C does not show a consistent benefit from capacity feedback over B: its reciprocal-case difference is −14.27 but specialist/generalist difference is +7.21, and one of three training seeds favors B. E has the lowest mean learned cost on both cases, though the three-seed sensitivity interval does not establish a population-level E advantage. A and F1/F2 substantially outperform all learned arms in this finite-budget pilot. A uses historically trained checkpoints with a different training budget; F2 consumes planning forecasts. Those rows are controls, not sample-matched training comparisons.

## Behavior, learning and compute

D's H availability is only 0.082 on reciprocal and 0.154 on specialist/generalist, versus C's 0.513 and 0.558. D starts about 52 and 34 PM jobs per H+tail episode, compared with C's 25 and 6; D has fewer failures but much greater unavailability cost. This is consistent with an undertrained always-request coordinator over-serving free PM demand. It does **not** disprove the known expressivity result that a sufficiently capable always-request coordinator can emulate another requester schedule.

C's H failures average 20.0 and 25.9 by case, versus A's 6.4 and 9.2 and rollout's 1.9 and 8.1. The current learned framework is not economically competitive with strong controls. Legal-service WAIT and backlog remained feasible; the replication audit found no hidden invalid-action or cost-accounting explanation for its gap. A supplementary replay of all 224 control episodes exactly matched their costs and service starts while recording WAIT and capacity at each PM request creation (`supplementary_control_wait.jsonl` in the replication run). Mean legal-service WAIT for A was 1.83/3.63 by case, for F1 0/0, and for F2 1.38/2.13; C was 2.23/0.10. The full per-episode secondary metrics, including queues, PM/CM waits, H availability, service starts and forecast transitions, remain in the JSONL artifacts.

DEV mean costs at 2,048 / 4,096 / 8,192 H transitions, averaging the six case/seed training trials per arm, were B: 1237.73 / 1198.26 / 1193.20; C: 1207.35 / 1186.75 / 1187.73; D: 1366.61 / 1399.04 / 1394.21; E: 1257.20 / 1245.54 / 1155.48. Trapezoidal DEV learning-curve areas over 2,048–8,192 are B 1203.15, C 1190.51, D 1392.02 and E 1217.46 cost units. These are opened DEV descriptions, not independent proof of faster learning. Final replication cost and learning-curve area are distinct estimands.

B/C each have 22,679 trainable parameters, D 17,046, and E 26,263. Each learned trial used 8,192 H decisions, identical PPO settings and no forecast transitions. Mean per-trial wall time including DEV evaluation was B 12.67 s, C 12.74 s, D 8.97 s and E 13.06 s on CPU. The 32 replication F2 rollout episodes used 325,248 forecast transitions (mean 0.653 s per episode); F1 uses none. The frozen V14 actor has 3,219 effective parameters in the independently verified historical audit, but historical training samples and compute are not matched to the new arms.

## Interpretation and next decision

The pilot supports a narrow finite-budget difference between C and the learned always-request D in these two cases. It does not support a capacity-feedback benefit over a local requester or superiority to a full-information centralized learner, frozen V14, or strong rules. The large D gap is associated with excessive PM service and severe H unavailability. A threshold-requester + learned-coordinator diagnostic would isolate whether fixed requester timing or coordinator learning explains the remaining gap; if pursued, it needs a versioned follow-up protocol and opened DEV/replication labels, never a silent addition to pilot v1 or TEST tuning.

The prior work already includes partial-observation multi-agent predictive maintenance with technician assignment (Ruiz Rodríguez et al., [author repository](https://orbilu.uni.lu/handle/10993/54345)) and shared-crew stochastic maintenance scheduling with PPO-derived heterogeneous graph policies (Wang and Gombolay, [ICAPS paper](https://ojs.aaai.org/index.php/ICAPS/article/view/19839)). This pilot makes no first-use claim about MARL, shared resources or capacity feedback. The possible contribution remains the explicit request/capacity contract and matched finite-budget controls; its strongest general claims are currently unsupported.

## Post hoc attribution follow-up

A separately versioned fixed-threshold requester / learned-coordinator diagnostic was prelocked after this replication outcome and evaluated on new, non-TEST physical seeds. G cost was much lower than C and D in both cases and all three training seeds, suggesting request gating explains the finite-budget C–D gap. This does not alter any v1 result or its TEST gate. See [`threshold_requester_diagnostic_findings.md`](threshold_requester_diagnostic_findings.md) for the audited panel, same-panel strong controls and post hoc limits.
