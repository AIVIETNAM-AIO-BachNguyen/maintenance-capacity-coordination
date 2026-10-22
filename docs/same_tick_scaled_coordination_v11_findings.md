# Scaled N5/K3 same-tick coordination v11 — audited no-go

Date: 2026-10-23 (Asia/Taipei). Protocol: `same_tick_scaled_coordination_v11_pilot.md`, SHA-256 `3959709edeb8e93c451c48fef594af8646d631a8ba973ad71ea6bfa603c9a9d9`. Collection source: pushed commit `fe584560d0e0d9bcc132ecdc651447cee8f527b6`. All training and evaluation trajectories came from the simulator. These two fixed N5/K3 configurations are simulated stress cases, not empirical factory calibration. No TEST was opened.

## Verification and comparison scope

Before scientific outcomes, 165 tests and the final bounded Ubuntu smoke/replay audit passed. The locked training/DEV run `artifacts/same_tick_v11_train_20261022T181041551780Z` completed 36 trials (two cases × six scratch arms × three independent training roots), 294,912 training H decisions and 1,728 DEV episodes. Its independent audit `artifacts/same_tick_v11_audit_20261022T182306383031Z` passed every training decision, physical transition, accounting and checkpoint selection. Only then was the fresh held-out panel `artifacts/same_tick_v11_heldout_20261022T182429355817Z` collected: 1,344 learned/CF11 episodes on paired physical seeds and 128 F1/F2 episodes. Its independent audit `artifacts/same_tick_v11_audit_20261022T182726308160Z` passed replay of all 64,512 learned H decisions and 6,144 control decisions, source/config/checkpoint identity, physical pairing and locked gate arithmetic. The held-out manifest records the same source commit and protocol hash. Failures and partials remain preserved as artifacts.

Lower H+tail economic cost is better. The C11−control estimates below average the two fixed cases, three training roots and 32 fresh paired raw physical seeds equally. The intervals are 5,000 percentile resamples of **raw physical seed ID as a cluster** across cases, arms and roots; they condition on these training roots. Per-root estimates are reported separately.

| C11 minus control | Overall H+tail | N5/K3 case a | N5/K3 case b | Physical-cluster 95% interval | Three training-root means |
| --- | ---: | ---: | ---: | --- | --- |
| **D11, independently trained frozen context** | **+0.67** | +3.80 | −2.47 | [−5.81, +7.45] | +7.28, −6.55, +1.27 |
| **CF11, same C11 weights frozen at inference** | **−0.12** | +1.78 | −2.02 | [−4.20, +3.63] | 0.00, +0.78, −1.14 |
| B11, direct sequential | −10.15 | −5.41 | −14.89 | [−18.20, −2.05] | −14.11, −12.64, −3.69 |
| S11, structured complete action | −2.75 | +2.17 | −7.67 | [−9.78, +4.32] | −0.45, −13.88, +6.08 |
| G11, flat full information | −28.02 | −26.49 | −29.54 | [−37.98, −18.48] | −23.84, −21.41, −38.80 |
| Q11, flat always-request admission | −220.96 | −248.13 | −193.79 | [−230.59, −210.97] | −238.50, −226.48, −197.89 |

The locked independent-retraining gate requires C11−D11 ≤−20, both cases negative, two of three roots negative and the interval upper endpoint negative; C11−CF11 must also be ≤−10 with both cases and its upper endpoint negative. The practical magnitudes, both-case signs and both interval-upper conditions fail; only one D11 root and one case for each contrast have a negative estimate. C11−B11 and C11−G11 are negative, but they do not establish contextual-feedback benefit. The audited `replication_go_decision.json` records `go: false`. There was no independent retraining, checkpoint retuning or TEST.

| Arm | Mean H+tail case a | Mean H+tail case b |
| --- | ---: | ---: |
| C11 | 1596.86 | 1598.57 |
| D11 | 1593.06 | 1601.04 |
| CF11 | 1595.08 | 1600.59 |
| B11 | 1602.27 | 1613.46 |
| S11 | 1594.70 | 1606.24 |
| G11 | 1623.35 | 1628.11 |
| Q11 | 1844.99 | 1792.36 |
| F1 risk matching | 1634.19 | 1647.22 |
| F2 six-tick rollout | 1594.91 | 1587.72 |

F1/F2 each have 32 episodes per case rather than three learned training roots. Averaging the displayed case means, C11 is about 6.4 cost units above F2, within the protocol's 20-unit deployment margin, but the feedback-attribution gate fails. Q11 changes PM admission and over-serves in this panel: its H availability was 0.077/0.109 versus C11's 0.356/0.371. The negative C11−Q11 sign means C11 was **better** than Q11. Q11 is a request-policy control, not an architecture-isolated contrast. The historical N4/K2 V14 checkpoint cannot score N5/K3 and was correctly kept as an external reference, not a same-panel control.

## Behavior, compute and mechanism

C11 had mean H failures 14.68/15.58, total H+tail service starts 47.14/56.31, and selected H matching cardinality sums 35.31/42.00 per episode in cases a/b. It made zero discretionary service WAITs. C11/D11 each had 12,322 actor parameters and 9,473 critic parameters; B11 had 11,770 actor parameters, S11 8,649, and G11/Q11 16,880. Mean training-trial wall times were 22.59/22.25/21.54/34.98/10.43/10.84 seconds for C/D/B/S/G/Q respectively. Mean C11 held-out selection took 0.065/0.068 seconds per episode versus G11 0.024/0.023 and F2 1.031/1.449; F2 used about 14,202/18,624 extra forecast transitions. For greedy C11, one root scorer call plus one after each selected edge gives 48 + the H matching-cardinality sum, or about **83.31/90.00 scorer calls per episode** in cases a/b; a flat complete-action policy uses 48. The inherited `decoder_forward_passes` field omits calls when no edge remains and must not be used as an exact count.

A separately timestamped **post-pilot descriptive** replay, `artifacts/same_tick_v11_mechanism_20261022T183233028522Z`, held each selected C11 checkpoint's weights fixed and froze only its within-tick context on **C11-visited states**. It replayed all 192 C11 held-out episodes, confirmed exact episode H+tail costs against the audited held-out rows, and its source/output hashes and episode-to-summary aggregate counts were checked. On 9,216 visited H states, some legal first edge left multiple feasible second edges in 271 case-a and 365 case-b states (636 total, 6.9%). After C11's *chosen* first edge, there were 151/621 case-a and 275/845 case-b opportunities with multiple feasible second edges; the other opportunities had one edge plus STOP. Same-weight freezing changed the final greedy matching in **212 case-a and 4 case-b states**, 216/9,216 overall. It changed second-edge rankings in only 3/4 opportunities by case; STOP-versus-edge or later-step choices can also change a final matching. Independently trained D11 checkpoints differed from C11 on 403/211 C11-visited matchings, but this also reflects different weights and learning trajectories.

Thus N5/K3 increased simultaneous-choice exposure relative to N4/K2, and context did affect some final actions, especially in case a. The paired **episode-level** C11−CF11 cost was nevertheless −0.12 overall, with an interval spanning zero and opposite case signs. Action-change frequency is not economic benefit. The descriptive replay is conditional on C11 trajectories and cannot assign a local cost to each changed token; the audited paired CF11 physical rollouts are the outcome comparison.

## Decision

Close v11 as **no-go** for the predeclared useful contextual-feedback claim. The simulator-based RL framework and scalable matching controls function correctly, but C11 neither beats its independently trained D11 ablation nor its same-weight CF11 inference ablation by the locked practical margins. Its gain over B11/G11 is an architecture-level observation and does not rescue the feedback hypothesis. Preserve v8–v11 gates and raw artifacts. Any further scientific experiment needs a distinct, independently justified hypothesis and new prelock; do not retune this case, gate or checkpoint from held-out outcomes. Real maintenance logs are optional external calibration, never a prerequisite for simulator RL. A new TEST still requires separate explicit authorization.
