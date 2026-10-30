# Candidate-conditioned residual-option coordination v12 — audited no-go

Date: 2026-10-23 (Asia/Taipei). The locked design and protocol are `residual_option_coordination_v12_design.md` (SHA-256 `22330d862fc888f52e7bd9ca9369b64f7b388aa77df3ae619ba7f5a607e38b93`). Corrected collection source was pushed as `3f5a1e2fd0cbacc08c1e013501927af76207652b`. All RL trajectories were simulator generated on two fixed, fresh N5/K3 configurations under the unchanged physical transition. No TEST panel was opened.

## Collection integrity and scope

Before scientific collection, **175 tests** passed, including explicit candidate-residual graph enumeration, exact sequence-probability/PPO-ratio contracts, same-weight context freezing, gate wiring and an executed F1/F2 control wrapper on both cases. The corrected-source bounded Ubuntu smoke `artifacts/same_tick_v12_smoke_20261023T015054764437Z` and independent replay audit `artifacts/same_tick_v12_audit_20261023T015119578248Z` passed 1,536 training H decisions and 12 DEV episodes. Smoke results were engineering validation only.

The first train/DEV artifact `artifacts/same_tick_v12_train_20261022T184245591036Z` completed and replay-audited PASS, but its first held-out attempt `artifacts/same_tick_v12_heldout_20261023T014800119791Z` failed after all learned-policy episodes and before F1/F2 because a control-wrapper import referred to a nonexistent adapter. The failed manifest and partials remain intact. **No cost or gate from that attempt was read or used for a design decision.** The one-line import correction left the policy, physics, cases, seeds, budgets and protocol unchanged but changed the source hash. We therefore repeated **all** training and held-out collection under one committed source identity; checkpoints from the earlier run were not mixed into the final panel.

Final train/DEV artifact `artifacts/same_tick_v12_train_20261023T015159675542Z` completed 36 trials (two cases × six scratch arms × three independent training roots), 294,912 H decisions and 1,728 DEV episodes. Independent replay audit `artifacts/same_tick_v12_audit_20261023T020654752328Z` **PASS** checked every training decision, physical transition, H+tail accounting, source/config/protocol identity and checkpoint-selection rule. Only then did final held-out `artifacts/same_tick_v12_heldout_20261023T020820954549Z` complete 1,344 learned/CF12 episodes and 128 risk/F2 episodes on 32 paired physical seeds per case. Its independent audit `artifacts/same_tick_v12_audit_20261023T021134725701Z` **PASS** replayed all 64,512 learned H decisions and 6,144 control decisions and verified checkpoint identity, same-weight CF12, physical pairing, source/config/protocol hashes and locked gate arithmetic. The final manifest source commit is `3f5a1e2`; failed and partial artifacts were not overwritten.

C12/D12/B12 use ordered internal-sequence PPO with one physical reward for the completed matching. Sequence log probability and entropy refer to token paths, not marginal matching probabilities. S12/G12/Q12 use complete-matching categorical PPO. All same-admission scratch arms share present-state information, the 136-action physical matching/WAIT catalogue and H-decision budgets; their parameter and compute budgets differ. Historical V14 is N4/K2-dimension locked and remains only an external reference. F2 is a known-model rollout control, not sample-matched PPO.

## Prelocked economic gate

Lower H+tail cost is better. Each learned contrast averages two fixed cases, three training roots and 32 paired raw physical seeds equally. The 95% intervals use 5,000 bootstrap resamples of **raw physical seed ID as the cluster** across cases/arms/roots, conditional on trained roots; the root means separately show training variability.

| C12 minus control | Overall H+tail | Case a | Case b | Physical-cluster 95% interval | Three training-root means |
| --- | ---: | ---: | ---: | --- | --- |
| **D12, independently trained frozen context** | **−1.32** | +0.82 | −3.46 | [−8.81,+6.00] | −19.27,+1.97,+13.34 |
| **CF12, same C12 weights frozen at inference** | **−1.36** | +0.21 | −2.93 | [−6.19,+3.28] | −5.84,−3.97,+5.73 |
| B12, direct residual-option sequential | −1.98 | −3.85 | −0.10 | [−9.28,+5.08] | −14.44,+12.67,−4.17 |
| S12, structured complete action | +2.51 | +0.71 | +4.30 | [−5.71,+10.66] | +2.73,+7.73,−2.95 |
| G12, flat full information | −17.52 | −27.42 | −7.61 | [−28.24,−6.87] | −15.41,−2.77,−34.38 |
| Q12, flat always-request admission | −134.20 | −142.83 | −125.56 | [−143.03,−125.45] | −76.36,−148.56,−177.67 |

The gate requires C12−D12 ≤−20 with both cases, two training roots and the cluster interval upper endpoint negative; C12−CF12 must be ≤−10 with both cases and its cluster upper endpoint negative. **The mandatory practical magnitudes, both-case signs, D12 root sign count and both interval-upper conditions failed.** C12−B12 and C12−G12 were negative, but their signs do not establish useful contextual feedback. The independently audited `replication_go_decision.json` records `go: false`. No independent retraining or TEST was run; no threshold, seed, case or checkpoint-selection rule was revised after outcomes.

| Arm | Mean H+tail case a | Mean H+tail case b |
| --- | ---: | ---: |
| C12 | 1651.60 | 1552.68 |
| D12 | 1650.78 | 1556.14 |
| CF12 | 1651.40 | 1555.60 |
| B12 | 1655.46 | 1552.78 |
| S12 | 1650.90 | 1548.38 |
| G12 | 1679.02 | 1560.29 |
| Q12 | 1794.44 | 1678.24 |
| F1 risk matching | 1697.12 | 1573.38 |
| F2 six-tick rollout | 1650.69 | 1515.53 |

F1/F2 have 32 episodes per case, without training-root variation. Mean C12 cost was about 19.03 above F2 overall, just within the predeclared 20-unit deployment comparison margin; in case b F2 was about 37.15 lower. This does not rescue the failed feedback-attribution gate. Q12 changes PM admission, so its much higher cost is not an architecture-isolated comparison.

## Behavior, compute and mechanism

C12 mean H availability was 0.353/0.373, H failures 16.30/13.52, and selected H matching cardinality sums 38.52/42.61 per episode in cases a/b. It made zero discretionary service WAITs. C12/D12 each had 13,858 actor and 9,473 critic parameters; B12 had 13,498 actor parameters, S12 8,649, G12/Q12 16,880. Mean training-trial wall times were 33.00/25.95/32.32/34.76/10.39/10.74 seconds for C/D/B/S/G/Q. C12 mean held-out selection took 0.085/0.093 seconds per episode versus G12 0.023/0.023 and F2 1.209/1.473; F2 performed about 16,077/19,527 extra forecast transitions per episode. A greedy C12 episode requires one root scorer call plus one after every selected edge, about 86.52/90.61 scorer calls per episode in cases a/b versus 48 complete-action calls for G12. The inherited `decoder_forward_passes` field omits scorer calls after feasible edges are exhausted and must not be interpreted as that exact total.

A separate **post-pilot descriptive** same-weight replay `artifacts/same_tick_v12_mechanism_20261023T021514895450Z` followed all 192 C12 held-out trajectories and confirmed their exact H+tail costs against the audited episode rows. Its output hashes and episode-to-summary aggregate counts were independently checked. On 9,216 C12-visited H states, some legal first edge left multiple feasible second edges in 357 case-a and 236 case-b states (593 total, 6.4%). After C12's chosen first edge, 254/780 case-a and 188/840 case-b second-step opportunities had multiple feasible edges. Freezing the **same C12 weights** changed the final greedy matching in 47 and 25 visited states (72/9,216 overall); second-edge rankings changed in 57 and 48 opportunities. Independently trained D12 checkpoints differed from C12 on 345/298 C12-visited matchings, which also reflects different weights and learning trajectories.

The candidate-conditioned residual graph is active and sometimes changes decisions, but its paired **episode-level** C12−CF12 contrast is only −1.36, has opposite case signs and a cluster interval crossing zero. Counts of altered actions are not economic gain; the descriptive replay is conditional on C12 trajectories and does not assign local regret or a causal outcome to each changed token. The audited paired CF12 physical rollouts are the outcome comparison.

## Decision and remaining uncertainty

Close v12 as **no-go** for a useful same-tick feedback contribution under this fixed simulation contract and budget. The framework code correctly recomputes candidate-specific residual options, but it did not deliver the predeclared benefit over independently trained D12 or same-weight CF12. Its gain over G12 and its near parity with B12/S12 do not demonstrate the proposed feedback mechanism or a distinct useful demand representation. These results do **not** prove that all same-tick coordination is useless: the current question is whether the residual alternatives on visited states have sufficiently different *long-horizon economic values* for feedback to matter, and whether this training budget can learn that difference. Feasibility counts alone cannot resolve either uncertainty. Preserve v1–v12 verdicts and all artifacts. Any further experiment needs an independently justified, distinct hypothesis and a fresh protocol rather than another outcome-guided scorer adjustment or recycled TEST data. Real logs remain optional calibration; simulator trajectories are sufficient for RL training. New TEST still requires separate user authorization.
