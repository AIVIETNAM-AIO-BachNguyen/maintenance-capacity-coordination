# Structured demand–commitment pilot v3 — audited findings

Date: 2026-10-22 (Asia/Taipei). The distinct v3 question and all gates were locked in `structured_demand_commitment_pilot_v3.md` (SHA-256 `9ac479d70e65601ab3f39bf1f99476267d8359e44c2fa90c50efcda8a504b97c`) and committed as `285392d` before any v3 DEV or held-out outcome. V1 TEST stayed closed; the v2 early-PM action-set gate and its negative finding were not changed. The v3 actor is a structured **centralized** matching policy: shared machine/worker encoders produce learned features, not calibrated economic values or independent machine actions. The unchanged request simulator executed threshold submissions, matching/WAIT, automatic CM, non-preemptive service, H+tail cost and physical transitions.

## Verification and artifacts

- Fresh bounded Ubuntu smoke: `artifacts/structured_pilot_v3_smoke_20261022T072512442867Z`; all eight arm/case trials and its artifact audit passed. The full repository suite passed 114 tests.
- DEV training: `artifacts/structured_pilot_v3_train_20261022T072700626808Z`; 24/24 trials completed the locked 8,192 H decisions, total 196,608 H decisions plus 65,280 tail physical transitions after 4,080 complete training episodes. All 1,152 DEV episodes, three milestones, cost/legality checks, source/config/protocol hashes and selected checkpoints passed audit. Wall time: 294.74 s.
- Held-out physical panel: `artifacts/structured_pilot_v3_heldout_20261022T073218111740Z`; all 768 learned and 448 A/F control episodes passed audit on the same 32 raw physical seeds per case. The audit checked exact checkpoint selection, source/config/protocol identity, paired derived physical seeds, zero invalid assignments, returns and cost decomposition. Wall time: 87.05 s. `analysis.json`, `secondary_analysis.json` and `replication_go_decision.json` preserve all estimates and the written no-go result.

These are internal artifact and replay/accounting checks of the inherited simulator, not an independently implemented physical model. Results are restricted to the two N4/K2 cases and finite training budget.

## Primary H+tail cost

Lower is better. S, R, G3 and D3 each average three **newly trained** seeds on 32 paired physical seeds per case. S/R/G3 share fixed threshold requests, complete matching mask, raw present information and 8,192 H decisions per trial. D3 uses always-request and has a different request history/action set; A is frozen historical V14, and F2 spends forecast compute.

| Arm | Reciprocal | Specialist/generalist | Equal-case mean |
| --- | ---: | ---: | ---: |
| S: structured machine/worker matching | 813.19 | 987.46 | 900.32 |
| R: S without numerical remaining-time actor inputs | 813.64 | 987.46 | 900.55 |
| G3: direct-health full-information PPO | 853.67 | 991.52 | 922.59 |
| D3: always-request coordinator PPO | 1364.77 | 1338.06 | 1351.42 |
| A: threshold + frozen V14 count | 802.08 | 982.21 | 892.14 |
| F1: threshold + risk matching | 859.88 | 1007.25 | 933.56 |
| F2: threshold + six-tick rollout | 724.53 | 979.81 | 852.17 |

The predeclared comparisons equal-weight cases and training seeds. Each 95% interval is a 5,000-draw bootstrap of the **raw physical seed ID as one cluster across arms, cases and the three trained models**. It is conditional on those trained models; the three training-seed means are shown separately.

| Comparison | Overall | By case (reciprocal / specialist) | Cluster 95% interval | Training-seed means |
| --- | ---: | ---: | ---: | --- |
| S−G3 | **−22.27** | −40.48 / −4.06 | [−33.92, −11.12] | −37.00, +6.30, −36.11 |
| S−R | −0.22 | −0.45 / 0.00 | [−0.95, +0.32] | −0.33, −0.44, +0.09 |

The locked independent-training-replication gate required S−G3 ≤ −30.0, both case means below zero, at least two of three training-seed means below zero, and the clustered interval upper endpoint below zero. The practical-effect requirement **failed** (−22.27 > −30.0), although the other conditions passed. A simple t sensitivity interval over the three S−G3 seed means spans roughly [−83.74, +39.20], emphasizing weak evidence about new training seeds. **No independent training replication or new TEST opens.** The held-out panel is evaluation of the DEV-selected trained models, not independent training replication.

H3 received no support: S−R was near zero, 84.4% of paired reciprocal episode costs were identical, and all specialist/generalist paired costs were identical. R retains free/busy flags and the feasibility mask, so this only tests numerical remaining-time context, not every form of capacity information. S is lower cost than G3 on this panel but not by the predeclared practical amount, and it does not beat frozen V14 or F2 overall. Beating D3 alone cannot establish the proposed structural contribution because D3's request contract differs.

## Learning, behavior and compute

DEV mean H+tail cost at 2,048 / 4,096 / 8,192 decisions (six case/seed trials per arm) was S: 1006.30 / 916.60 / 917.60; R: 1006.86 / 916.17 / 917.19; G3: 1035.13 / 940.89 / 932.03; D3: 1374.91 / 1351.16 / 1404.92. Trapezoidal DEV learning areas over 2,048–8,192 were S 931.89, R 931.62, G3 953.64 and D3 1373.03 cost units. This opened DEV panel also selected checkpoints and is descriptive; it is not independent proof of sample efficiency.

S/R actors each had 8,553 trainable parameters, G3/D3 8,705 (1.8% more); every critic had 7,873. Mean CPU trial wall time including DEV was S 15.47 s, R 15.79 s, G3 8.74 s and D3 8.96 s. Thus equal transition budget was **not** equal compute time. On held-out episodes S policy selection averaged 0.044 s versus G3 0.021 s and F2 rollout 0.743/0.545 s by case. F2 used 385,920/277,248 forecast transitions over its 32 reciprocal/specialist episodes; learned actors used zero forecasts. S was faster to deploy than F2 but its mean cost was higher, so no cost-and-speed superiority follows.

S's H availability was 0.573/0.498 with 5.74/7.93 H failures, compared with G3's 0.576/0.506 and 7.17/8.81. S started 48.97/40.19 service jobs and had H pending backlog 0.41/0.48, compared with G3's 48.06/40.88 starts and 0.55/0.61 pending. S's queued PM ticks were 16.21/26.15 and queued CM ticks 3.54/10.50; G3 had 21.02/27.32 and 5.49/9.42. S recorded zero legal-service WAIT in both cases, while G3 averaged 1.73/0.00. The full per-episode secondary rows, including end backlog, are preserved in the held-out JSONL. These behavior summaries do not isolate a causal advantage of any one feature.

## Decision

This versioned structural-learning pilot is implemented and audited, but its primary practical gate failed and the remaining-time ablation did not support the claimed capacity-context mechanism. Preserve the result; do not lower the −30 gate, retrain on selected seeds, open independent replication/TEST, add a prediction/price/loss module, or claim a successful two-policy MARL framework. The next research step requires a distinct operational mechanism with evidence that it can materially improve strong same-contract controls; the current data alone do not justify scaling N/K.
