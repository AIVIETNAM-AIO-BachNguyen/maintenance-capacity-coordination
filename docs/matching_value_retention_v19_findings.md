# V19 matching-value retention: audited findings

Status: **closed, no-go** on 2026-10-24. The protocol was locked in `docs/matching_value_retention_v19_prelock.md` at commit `64a68d4` before implementation or outcomes. Scientific collection used clean commit `a769831cd055ea452ea3c60ed4aa7be66ff5d9ac`. This experiment used simulator-generated trajectories throughout. No independent retraining or new TEST ran.

## Question and design

The v18 training-bank replay showed the matching-label ranking loss rising during PPO. V19 tested whether a fixed 0.5-weight listwise rehearsal term could retain that ranking and improve physical H48+tail16 cost. C19 and D19 started from the corresponding v18 pretraining states, used the same training tapes and 8,192 PPO H decisions per case/root, and differed from C18/D18 only by the auxiliary training loss. D19 froze within-tick scoring context while C19 refreshed it. CF19 loaded each selected C19 state dict into a frozen-context actor, isolating feedback at identical weights. All checkpoint controls were reselected on fresh v19 DEV. The physical simulator, threshold PM admission, complete matching action set and STOP remained unchanged.

## Provenance and qualification

The full source suite passed 227 tests. The final bounded Ubuntu smoke and its independent audit passed. One earlier smoke audit failure caused by float32 logging of the summed scalar loss is preserved; the correction used double scalar addition without changing gradients, and the final-source smoke audit passed.

| Stage | Artifact | Independently audited result |
| --- | --- | --- |
| Final Ubuntu smoke | `artifacts/same_tick_v19_smoke_20261024T094327865804Z` | PASS; 2 trials, 512 H decisions, 4 DEV episodes. |
| Train and fresh DEV | `artifacts/same_tick_v19_train_20261024T094616510975Z` | PASS; 12 trials, 98,304 H decisions, 576 C19/D19 DEV episodes and 2,016 reselected reference DEV episodes. |
| Fresh paired held-out | `artifacts/same_tick_v19_heldout_20261024T102118303621Z` | PASS; 1,920 learned/CF episodes, 128 F1/F2 episodes, 92,160 learned/CF and 6,144 control H decisions replayed. |

On the immutable 16-tape *training* label bank, the selected checkpoints' equal-root mean cross-entropies were C19 **2.050/2.054** versus reselected C18 **2.965/2.650**, and D19 **2.059/2.072** versus D18 **2.641/2.928**, in cases a/b. All eight prelocked retention conditions passed, so the fresh held-out panel was eligible. This verifies local ranking retention, not generalization or physical benefit.

## Fresh held-out economic result

Each learned arm used the same 32 raw physical seeds in each case and training root. Differences are C19 minus comparator H+tail cost; negative favors C19. The 95% intervals resample raw physical seed clusters shared across arms, cases and roots. The three primary contrasts are:

| Contrast | Overall | Case a / b | Training roots 0 / 1 / 2 | Cluster 95% interval | Locked practical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| C19 − C18, rehearsal effect | **+1.79** | +4.14 / −0.56 | +2.08 / +6.28 / −3.00 | [−5.51, +8.56] | ≤−10, both cases negative, upper <0: **fail** |
| C19 − D19, trained feedback comparison | **−15.21** | −18.98 / −11.44 | −28.09 / −3.59 / −13.94 | [−21.39, −9.21] | ≤−20, both cases negative, 2 roots negative, upper <0: **fail on magnitude** |
| C19 − CF19, same-weight feedback | **−3.30** | +0.72 / −7.31 | +0.03 / −9.84 / −0.08 | [−9.61, +2.69] | ≤−10, both cases negative, upper <0: **fail** |

C19's absolute mean cost was **1562.64** versus C18 **1560.85**, D19 **1577.85**, CF19 **1565.94**, G18 **1590.52**, B18 **1569.11**, S18 **1567.21**, and F2 six-tick rollout **1574.69**. The pooled C19−G18/B18/S18 means were −27.88/−6.47/−4.57, satisfying their contextual sign checks; C19−F2 was −12.05, satisfying the no-more-than-20-worse check. These do not override the failed primary gates. The independent `replication_go_decision.json` records `go: false`.

## Secondary outcomes and compute

Descriptive held-out means per learned episode, pooled across cases and training roots:

| Metric | C19 | C18 | D19 | CF19 |
| --- | ---: | ---: | ---: | ---: |
| H availability | 0.349 | 0.346 | 0.360 | 0.350 |
| H failures | 11.44 | 10.89 | 13.01 | 11.66 |
| Queued CM ticks | 29.45 | 28.99 | 23.56 | 28.88 |
| Queued PM ticks | 29.92 | 28.70 | 33.18 | 30.36 |
| Service starts | 52.37 | 53.19 | 53.73 | 52.39 |
| Legal service waits | 4.54 | 3.70 | 2.20 | 4.55 |
| H matching cardinality sum | 38.22 | 39.18 | 39.59 | 38.22 |
| Selection seconds / H episode | 0.133 | 0.134 | 0.097 | 0.098 |
| Decoder forward passes / H episode | 47.80 | 45.45 | 45.01 | 47.28 |

Each new actor had 22,178 parameters and critic 9,473. The two new arms used 512 rehearsal minibatches per trial, 6,144 across 12 trials, with roughly 1.56 million candidate paths and 2.79 million canonical token positions evaluated for the auxiliary loss. Mean trial wall time including PPO and DEV was 169.7 seconds for C19 and 78.3 seconds for D19; the full train/DEV run took 1,677.6 seconds, and held-out collection 283.6 seconds. These counts demonstrate extra training compute; equal environment decisions do not imply equal runtime.

## Post-pilot sensitivity and label-fit diagnostics

A separate descriptive replay, `artifacts/same_tick_v19_mechanism_20261024T105649637120Z`, evaluated the selected C19 actor and the identical-weight frozen-context actor at each of 9,216 C19-visited H states. It executed only C19's matching; all 192 episode costs exactly matched the independently audited held-out rows. File hashes, episode count, case summaries and cost identities were checked independently of the replay script. There were 2,219 second-token opportunities, including 785 with multiple remaining feasible edges. Freezing context changed the final greedy matching at 261/9,216 states (2.83%; 112 in case a, 149 in b), compared with 89/9,216 (0.97%) in the earlier v18 descriptive replay. C19 and separately trained D19 differed at 2,032 final matchings on C19-visited states, but their weights differ and that count cannot isolate feedback. These sensitivity counts show that rehearsal made same-weight feedback affect more selected matchings; they do not establish whether the changes improve episode cost. The paired physical C19−CF19 contrast above remains the economic test.

For scale, the audited 16-tape training targets have mean entropies 1.935/1.870 in cases a/b. A uniform distribution over each row's candidate list has cross-entropy 2.063/2.079. The selected C19 loss of 2.050/2.054 is much closer to the uniform reference than to the target entropy, even though it is far better than the drifted C18 checkpoints. This is an in-sample reference calculation, not a held-out prediction-quality estimate. It narrows the learning uncertainty: rehearsal prevented degradation but did not extract most of the soft target ranking signal in the fixed training bank.

## Decision and remaining uncertainty

V19 successfully prevented the measured training-label drift, but it did **not** improve cost against the matched no-rehearsal C18 control. C19 beat separately trained D19 by a statistically negative pooled difference, yet missed the locked practical magnitude. The same-weight feedback effect missed its magnitude, both-case and interval checks. Thus v19 does not establish that retaining these simulator matching labels makes within-tick feedback economically useful. The concrete uncertainty is whether the existing 16-tape continuation labels align with the on-policy decisions that drive held-out cost, and whether the current actor uses the graph relationship strongly enough at those decisions. These data do not identify which explanation dominates. V19 is closed without independent retraining or TEST; a successor would need a distinct, prelocked hypothesis rather than a revised v19 gate.
