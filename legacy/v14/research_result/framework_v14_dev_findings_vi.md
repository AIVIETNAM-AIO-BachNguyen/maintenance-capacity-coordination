# V14 capacity-matched identity control — independently audited DEV no-go

## Method, fairness and provenance

The V13 joint unserved-machine opportunity penalty showed a small advantage over a no-penalty arm, but its 3,219 versus 2,354 gradient-active parameter difference left capacity confounding. V14 prospectively locked a more precise scratch-RL ablation: `competition` uses the identity of machines unserved by each joint matching; `count_calibrated` prices the same demand by the number of unserved machines, without their identities. Both actors have the same 3,219 gradient-active parameters, current 56-coordinate observation, legal actions, initial tensors, training physical-role streams and six-update full-H64 credit budget. V14 source/protocol was pushed at `3abbf14` before the full run. The five independent training seeds are 72600000–04, selection/DEV physical seeds 72610000–15, and no prior TEST panel was reused.

Full `framework_v14_dev_20261020T185303Z` is **COMPLETED** in 4,365.5 s. Independent `framework_v14_audit_20261020T200552Z/independent_audit.json` is **PASS** for source/artifact/V7 hashes, initial active capacity, all 15,360 Q rows and 60 trace cells, 240 full-bank Q and source-state replays (163,316 forecast transitions), all 640 selected actor episodes, 4,480 checkpoint-selection episodes, 1,792 final comparator episodes, selection, cost/request conservation and bootstrap. Both arms trained on 7,680 selected states and 92,160 H48 source steps each; forecast transitions were 5.539M versus 5.557M. No actor received future shocks or online planner calls.

Checkpoint selection and final DEV comparison reuse the same 16 physical seeds, as disclosed in the locked protocol. These are exploratory DEV numbers, not fresh confirmation. Earlier V1–V13 development also means a favorable DEV comparison alone would not settle the goal.

## Locked primary ablation and competitive gate

Equal-weight four-case H64 cost; lower is better. Candidate minus capacity-matched control is **−1.72**, paired crossed-seed/tape CI95 **[−7.08,+2.80]**. Train-seed deltas are **+2.23,−6.23,+0.17,−0.80,−3.98**. Only three of five improve, including only **one of the three** strongest control seeds. The locked H1 required at least −10, four of five improvements including all three strong control seeds, and CI upper <0. It **fails every material part**. The full DEV go gate is **FALSE**.

| Controller | Four-case cost | Selection seconds/episode |
|---|---:|---:|
| V14 joint identity | 852.21 | 0.0223 |
| V14 count calibrated | 853.93 | 0.0237 |
| Bounded rollout | **847.70** | 0.6505 |
| Risk matching | 887.30 | **0.0013** |
| Flat PPO | 903.42 | 0.0199 |
| V7 shared canonical | 945.75 | 0.0278 |

V14 joint is 0.53% above rollout and 29.2× faster, and risk matching does not dominate it in both cost and runtime. These point estimates satisfy V14's auxiliary H2 headroom, but the identity-specific component has no supported benefit and there is no independent replication or confirmatory TEST. The original goal's A route would require ≤805.32 on this DEV panel to beat its strongest locked comparator by 5%; B would require ≤856.18 plus speed and non-dominance **on a valid frozen fresh evaluation**, alongside a supported core-method ablation and replication. V14 cannot be promoted from a within-1% DEV point estimate alone.

## Cases and operating behavior

| Case | Joint cost | Count cost | Rollout cost | Joint H48 failures | Rollout failures | Joint H48 PM queue ticks | Rollout PM queue ticks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 730.81 | 731.21 | **683.75** | 2.88 | 0.88 | 8.58 | 2.06 |
| Reciprocal scale2 | 747.84 | 750.05 | **699.00** | 3.26 | 1.38 | 11.15 | 3.56 |
| Reciprocal scale3 | **933.01** | 937.91 | 1023.56 | 5.89 | 8.00 | 17.59 | 24.31 |
| Specialist/generalist scale2 | 997.16 | **996.54** | 984.50 | 8.99 | 7.75 | 30.10 | 25.38 |

The joint/count differences by case are −0.40,−2.21,−4.90,+0.63 in table order. The candidate's rollout gap is concentrated in interchangeable and reciprocal2, where failures and queued PM time remain substantially higher despite similar total service starts. It improves over rollout in reciprocal3, and specialist is slightly worse than both count and rollout. That case mix explains why the pooled cost looks close while the identity-specific ablation is weak. Training-seed failure risk is not concentrated in only one catastrophic replica in V14, but the small, sign-mixed differences do not identify a robust identity mechanism. No generalization, permutation or N/K scaling claim was tested.

## Secondary identity and uncertainty diagnostics

The prelocked same-checkpoint Q-state diagnostic first failed before producing a summary because a mathematically equal one-service score had two float32 near-ties. The failed `v14_identity_toggle_dev_20261020T200742Z` and watcher log are preserved. The transparent v2 numerical-tie amendment was pushed at `84503f4`; it keeps the original raw action/Q estimand and records two one-service near-ties separately. Fresh `v14_identity_toggle_dev_v2_20261021T014826Z` and independent audit **PASS** for all 7,680 training states, 30 old checkpoints, decisions, penalties, Q values and bootstrap. Raw joint-minus-count Q is **+0.0163** CI95 **[−0.0601,+0.0949]**; 39/7,680 actions differ (0.51%), with 19 improved and 20 worsened. The two numeric-tie rows sum to +51.5 Q cost; setting only these to zero in a secondary mechanism account gives mean **+0.0096**. The locked directional Q signal is FALSE. This diagnostic uses old joint-policy continuation and is not a closed-loop ablation.

A separate pre-score addendum clarified crossed versus nested physical bootstrap. `v14_bootstrap_sensitivity_dev_20261021T015027Z` and independent audit PASS from the same immutable paired episodes: nested CI95 **[−5.92,+2.03]**, versus original crossed [−7.08,+2.80]. Both include zero; the primary prelocked gate is unchanged. A reciprocal3-only nested interval is negative, but that is secondary among four cases and does not rescue the joint aggregate method.

## Decision

Reject V14's **identity-specific joint unserved-machine term** as a supported improvement over the active-capacity-matched count control on this DEV campaign. Retire the unused V14-specific 727xxxxx replication and 728xxxxx TEST reservations **unopened**; no V14 replication or fresh TEST is justified. Preserve the near-rollout speed/cost observation as a lead for a distinct method hypothesis, not as proof of the user's completion conditions. The V13 one-edge cancellation also failed causally while its actor reissued every removed service; a next design must address repeated temporal decisions and/or coordinated machine-set planning without claiming a simple technician hold is enough. Any new scratch-RL method needs a new proposal, distinct seed registry, matched control, strong comparators and fresh evaluation before its own single confirmation.
