# Joint demand–capacity score pilot v6 — audited findings

Date: 2026-10-22 (Asia/Taipei). The hypothesis, operating contract, seed roles, estimator and gates were locked in `joint_demand_capacity_pilot_v6.md` (SHA-256 `12a2060bd8e3a93c096b48e9cf88a3f21016760ae8510d17f285b1b8201416b8`) before any v6 outcomes. This pilot uses unchanged physical and economic rules, fixed threshold PM requests, automatic CM, complete legal matching/WAIT, H=48 and the inherited 16-tick tail. J6 scores machine–technician edges plus a two-edge interaction; A6 scores the same actions additively; G6 is a flat full-information actor. All three see the same present inputs, legal mask, F2 teacher states and all legal relative root-score targets. They are centralized supervised policies, not a two-policy MARL framework or a calibrated machine-value model.

## Verification, artifacts and budget

- Final source commit `cd470f9` was pushed before the final scientific collection. The full suite passed 139 tests. Final two-episode, one-epoch Ubuntu smoke `artifacts/joint_score_v6_smoke_20261022T093212430941Z` and independent replay audit PASS.
- Final training `artifacts/joint_score_v6_train_20261022T093257608647Z`: COMPLETED and audit PASS. Three roots × two cases × 16 F2 teacher episodes produced 4,608 H state rows and 977,952 teacher forecast transitions. Within a case/root, J6/A6/G6 each saw the same 768 rows, trained for exactly 64 epochs and 384 gradient updates, and selected checkpoints among epochs 16/32/64 on 16 DEV physical seeds. All 18 student trials completed; 864 student DEV episodes were replay audited. Active actor parameters: J6 4,850; A6 4,825; G6 4,905. Training wall time 105.66 s, including teacher and DEV work. Mean per-trial student training plus DEV time was 2.78/2.92/1.12 s for J6/A6/G6; equal rows and updates do not imply equal compute.
- Final held-out `artifacts/joint_score_v6_heldout_20261022T093629493268Z`: COMPLETED and independent replay audit PASS. It contains 576 student episodes over 32 new paired physical seeds per case, 448 F2/F1/V14 control episodes, 384 frozen historical G/D contextual episodes, and 3,072 F2-visited interaction states. Controls and interaction replay each spent 654,912 forecast transitions; students used zero deployment forecasts. Held-out wall time was 125.65 s; independent audit 127.61 s. Source/protocol/config/checkpoint identities, legal actions, physical pairing, H+tail accounting, interaction scores, complete coverage and gate arithmetic passed.
- Earlier raw artifacts remain: `artifacts/joint_score_v6_train_20261022T091108837568Z` and `artifacts/joint_score_v6_heldout_20261022T091449323292Z` (the latter FAILED after writing raw panels because legacy controls omitted `legal_service_waits`); then `artifacts/joint_score_v6_train_20261022T092058361101Z` and `artifacts/joint_score_v6_heldout_20261022T092437227540Z` (the latter COMPLETED but formal audit stopped because F2 legacy rows omit two adapter diagnostics). These were retained, not used to select or interpret the final outcome. Reporting and audit schema repairs changed no learner, target, budget, simulator, estimator, seed or gate. The final panels were rerun from training under one committed source hash.

## Realized H+tail cost

Lower is better. Student means average three fresh training roots and 32 seeds in each case. F2/F1 have no training root; V14 averages five fixed historical checkpoints. G/D replay three fixed historical checkpoints with different training budgets, and D uses always-request submissions, so they are context rather than equal-budget attribution arms.

| Arm | Reciprocal | Specialist/generalist | Equal-case mean |
| --- | ---: | ---: | ---: |
| J6: explicit pair interaction | 847.08 | 991.75 | 919.42 |
| A6: additive ablation | 1335.00 | 990.91 | 1162.95 |
| G6: flat full-information | 789.72 | 987.52 | 888.62 |
| F2: six-tick rollout teacher | 724.78 | 982.09 | 853.44 |
| F1: risk matching | 816.97 | 1000.97 | 908.97 |
| A: frozen V14 | 775.66 | 994.73 | 885.19 |
| G: historical threshold + learned coordinator | 815.01 | 993.66 | 904.33 |
| D: historical always-request + learned coordinator | 1364.99 | 1340.17 | 1352.58 |

The predeclared paired comparisons resample raw physical seed IDs as clusters across both cases and the three trained roots, with 5,000 draws. Intervals condition on these trained models and do not quantify training-root variation; three roots are too few for a broad model-training population claim.

| Paired difference | Overall | Reciprocal | Specialist/generalist | Physical-cluster 95% interval | Per-root overall means |
| --- | ---: | ---: | ---: | --- | --- |
| J6−A6 | −243.54 | −487.92 | +0.84 | [−262.52, −225.48] | −231.80, −251.89, −246.92 |
| J6−G6 | +30.80 | +57.36 | +4.23 | [+15.28, +46.67] | +40.05, +18.34, +34.00 |

J6's large gain over A6 is real within this paired panel, but A6's reciprocal behavior is poor: mean H availability 0.08 versus J6 0.58 and G6 0.60; mean legal-service WAIT 47.59 versus 7.66 and 2.50; mean queued CM ticks 181.25 versus 9.03 and 0.92; and service starts 14.00 versus 49.49 and 50.56. These are descriptive closed-loop behavior differences. The additive restriction may contribute, but this panel does not isolate why A6 collapsed. J6 does not improve over the equally informed flat G6 or F2. In specialist/generalist, all three students are close in cost; J6−G6 is +4.23.

## Forecast interaction diagnostic and gate

On F2-visited held-out reciprocal states, the pooled 2,406 legal two-start roots had mean absolute error in the F2 finite-forecast pair-interaction cost of **32.71 J6**, **17.40 A6** and **6.35 G6**. Physical-episode-cluster 95% intervals were [31.39, 33.79], [16.68, 18.20] and [5.77, 7.02]. J6/A6 error ratio was **1.88**, versus the locked ≤0.80 requirement. The specialist/generalist case had 807 legal pair roots and errors 4.37/2.51/4.14 for J6/A6/G6. This teacher-state diagnostic is separate from realized H+tail cost and cannot explain policy trajectory differences causally.

The locked gate required J6−A6 and J6−G6 each to meet overall ≤−10, reciprocal ≤−20, specialist ≤+5, at least two negative training-root means, and a cluster interval upper endpoint below zero; it also required the reciprocal pair-error ratio ≤0.80. **J6−A6 passed all its cost conditions. J6−G6 failed overall, reciprocal, root direction and interval conditions. Pair-error fidelity failed.** The audited `replication_go_decision.json` is `false`. No independent training replication or v6 TEST is eligible or run.

Mean H action-selection time per episode was 0.0159 s J6, 0.0125 s A6, 0.0056 s G6 and 0.6311 s F2, measured in this environment. J6 amortizes F2's forecast computation at deployment but does not meet the predeclared cost/fidelity conditions. The teacher is a finite six-tick rollout reference, not an optimum; its known-model forecasts are a large training resource. All evidence is simulator-based, with only two N4/K2 cases and no independent physical validation.

## Decision

The explicit pair scorer is implemented and audited, and it avoids the additive ablation's reciprocal failure. The same-budget, same-information flat actor is better on realized cost, and the pair scorer fits the proposed interaction worse than the additive ablation. Thus v5's forecast headroom did not translate into the claimed v6 learning benefit. Preserve the negative result, do not tune v6 on these held-out seeds, and keep replication and TEST closed. The broader maintenance-demand/technician-capacity coordination contribution remains unproven; any next research experiment needs a distinct hypothesis and new prelocked evidence plan.
