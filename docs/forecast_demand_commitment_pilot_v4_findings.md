# Forecast-informed demand/commitment pilot v4 — audited findings

Date: 2026-10-22 (Asia/Taipei). The distinct formulation and all gates were locked in `forecast_demand_commitment_pilot_v4.md` (SHA-256 `ccbeb00d031bb5a9657b166c996fc2c233317fea2c08e78bcd3ddb25cc52b204`, commit `560fd53`) before any v4 teacher label or DEV/held-out outcome. Verified source was committed as `b5aef85` before scientific training. V1 TEST and the v2/v3 failed gates remain unchanged. V4 uses fixed threshold requests, the original physics/economic costs and one complete masked matching/WAIT action. It is **structured centralized policy distillation**, not two-policy MARL or a learned calibrated machine value.

## Verification and budgets

- The full repository suite passed 123 tests. Final bounded Ubuntu smoke `artifacts/forecast_distillation_v4_smoke_20261022T081720658860Z` used one case, two teacher episodes and one training epoch per arm; independent replay/accounting audit PASS. The earlier smoke artifact is retained but did not serve as the final engineering gate.
- Scientific training: `artifacts/forecast_distillation_v4_train_20261022T082000203886Z`. Six F2 teacher datasets (three fresh training roots × two cases) generated 96 complete episodes and 4,608 H state/label rows using 944,064 forecast transitions. S4/R4/G4 each trained for exactly 32 epochs on the same 768 rows within each case/root, 192 gradient updates per trial. All 18 student trials completed; 864 DEV episodes across three milestones selected checkpoints by locked H+tail mean. Teacher, all root scores, masks, selected labels, checkpoint identities, DEV rows, source/config/protocol hashes, costs and 4,608 H plus 1,536 tail teacher transitions replayed/audited PASS. Training wall time was 99.45 s; independent audit 81.98 s.
- Held-out: `artifacts/forecast_distillation_v4_heldout_20261022T082342680537Z`. Selected checkpoints were evaluated on 32 new paired raw physical seeds per case: 576 student episodes and 448 F2/F1/V14 control episodes. Replay audit PASS for complete coverage, exact checkpoint/source identity, physical pairing, legal assignments, H+tail returns/costs, forecast counts, analysis and gate calculation. Evaluation wall time was 66.34 s; independent audit, including teacher-agreement replay, 117.00 s. The held-out control panel spent 637,344 F2 forecast transitions; the separate agreement audit spent another 637,344. Students used zero deployment forecasts.

All actors saw the same current raw inputs and legal mask within S4/G4; R4 zeroed only numerical remaining-time actor fields. S4/R4 each had 8,553 actor parameters, G4 8,705. Critic parameters were unused. Teacher label generation and its known-model forecasts are an additional training resource, so this is **not** an equal-information/equal-compute comparison with historical model-free PPO v3. F2 is a finite six-tick rollout reference, not an optimum.

## Primary paired H+tail cost

Lower is better. S4/R4/G4 averages use three newly trained roots × 32 physical seeds/case. A averages its five frozen historical checkpoints on the same physical seeds. F2/F1 have no training root. Each student and control in a case shares the raw physical seed panel.

| Arm | Reciprocal | Specialist/generalist | Equal-case mean |
| --- | ---: | ---: | ---: |
| S4: structured forecast-distilled student | 800.71 | 986.36 | 893.54 |
| R4: S4 without numerical remaining-time inputs | 801.40 | 986.42 | 893.91 |
| G4: flat full-information forecast-distilled student | 784.79 | 1001.15 | 892.97 |
| F2: six-tick rollout teacher | 732.81 | 982.41 | 857.61 |
| F1: risk matching | 808.28 | 1002.94 | 905.61 |
| A: frozen V14 count-calibrated | 794.12 | 990.62 | 892.37 |

Paired intervals resample the **raw physical seed ID as one cluster across both cases and all three training roots**, 5,000 draws with locked analysis seed 73198000. These intervals condition on the trained models; training-root means are reported separately and three roots are too few for a strong distributional claim about future training.

| Comparison | Overall | By case (reciprocal / specialist) | Cluster 95% interval | Training-root means |
| --- | ---: | ---: | ---: | --- |
| S4−G4 | **+0.57** | +15.92 / −14.78 | [−10.33, +12.07] | −2.97, +7.39, −2.72 |
| S4−R4 | **−0.37** | −0.69 / −0.05 | [−1.15, +0.22] | −1.44, +0.17, +0.16 |
| S4−F2 | **+35.93** | +67.90 / +3.96 | [+15.11, +54.44] | +30.02, +49.08, +28.69 |

The locked replication gate required **all**: S4−G4 ≤−30 with negative case/root direction and CI upper <0; S4−R4 ≤−10 with analogous support; and S4−F2 ≤+20 overall, ≤+30 per case, with S4 selection time ≤20% of F2. The first two cost/effect gates failed. The F2 cost gate also failed overall and in the reciprocal case. Therefore `replication_go_decision.json` is `false`: **no independent training replication or v4 TEST opens**. The speed condition passed separately: S4 took 0.0156 s versus F2 0.6193 s of H action selection per episode, ratio 0.0251 (about 40× faster), but this does not establish a successful cost–speed tradeoff under the locked threshold. G4 was still faster than S4 (~0.0047 s per episode) on this panel.

## Behavior and limits of attribution

On F2-visited held-out states, teacher-action agreement for S4 was about 0.94 in the reciprocal case and 0.85–0.86 in the specialist case, depending on training root; G4 and R4 were similar. These rates include states with only one legal action. The corresponding finite six-tick teacher-score regret was 0.28–0.51 and 0.54–0.71 cost units per visited decision for S4. These are on **teacher-visited** states and do not measure student on-policy distribution shift or explain the large closed-loop reciprocal H+tail gap. Rare consequential disagreements and trajectory divergence are plausible interpretations, not established causes.

Numerical commitment context again had little measured effect: S4/R4 had exactly equal paired episode objectives in 88/96 reciprocal and 95/96 specialist episodes. R4 still sees free/busy flags and the matching mask, so this result does not test whether *all* capacity information is unnecessary. S4's reciprocal H failures averaged 5.49 versus F2's 2.62, and queued PM ticks averaged 16.34 versus 6.44; specialist H failures were 7.81 versus 8.44. These descriptive outcomes are consistent with the case-specific cost gaps, but do not isolate a single causal mechanism. Full secondary rows include availability, CM/PM waiting, backlog, starts and legal WAIT.

The dataset comes from F2 trajectories, uses the same simulator/model for teacher and replay audit, and covers only two N4/K2 cases. There is no independent physical model validation, no v4 TEST and no independent fresh-training replication. The six-tick teacher scores are finite-scenario estimates; no machine embedding was verified as an economic value. Total training/research forecast compute is substantial even though deployment student forecasts are zero.

## Decision

The forecast-distilled structured actor is implemented and independently replay-audited, but does not show the predeclared structural or numerical capacity-context gain, nor sufficient cost fidelity to F2. Preserve the result and all raw/failed artifacts. Do not adjust labels, optimizer, sample count, checkpoint rule, threshold or seeds on this held-out outcome; do not open replication or TEST under v4. A further campaign would require a separate operational mechanism and fresh protocol, not another attempt to obtain a favorable architecture result from this panel. The project's desired validated capacity–commitment contribution remains unproven.
