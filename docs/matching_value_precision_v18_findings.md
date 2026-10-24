# V18 matching-value precision: audited findings

Status: **closed, no-go** on 2026-10-24. This report follows the separately locked protocol in `docs/matching_value_precision_v18_prelock.md`. V1–v17 verdicts remain closed. V18 used simulator-generated training and evaluation trajectories; no operational log or public dataset was required. No independent retraining or new TEST ran.

## Question and evidence boundary

V17's four-tape full-matching continuation labels covered WAIT, singleton and multi-pair matchings, but their training-only split showed unstable rankings. V18 reused exactly those 508 audited *training* states and candidate sets, adding 12 paired future tapes to the original four. C18 and L18 had the same dynamic graph actor and fitting budget; C18 used 16-tape labels and L18 the original four. D18 had the same actor parameters and information as C18 but froze scoring context within the tick. CF18 loaded each selected C18 checkpoint into that frozen actor, making C18−CF18 the same-weight physical feedback contrast. The simulator, physical action set, PM admission, H48+tail16 cost, canonical sequence policy and PPO budget did not change.

The possible framework contribution was a useful economic gain from updating machine-demand and technician-capacity context after each provisional assignment. The held-out physical C18−D18, C18−CF18 and C18−L18 contrasts, not label reliability or in-sample fit, test that claim. Negative cost differences favor C18.

## Provenance and audits

The protocol was committed as `1167a71` before v18 implementation or outcomes; scientific collection used clean source commit `25ef7acbf94b805aa7974cfffe363dfe4d341baf`. The full source suite passed 221 tests. Bounded Ubuntu label and PPO smokes and their independent audits passed; an earlier failed smoke audit after a smoke-only summary correction remains preserved and was not used as scientific evidence.

| Stage | Timestamped artifact | Result |
| --- | --- | --- |
| Label extension | `artifacts/full_matching_v18_labels_20261024T035233500615Z` | Completed; 508 states (273/235), 48,384 new forced branches, 64,512 branches including old tapes; independent replay audit PASS. |
| Train and DEV | `artifacts/same_tick_v18_train_20261024T045926267434Z` | Completed; 60 trials, 491,520 H decisions, 2,880 DEV episodes; independent replay and checkpoint-selection audit PASS. |
| Fresh paired held-out | `artifacts/same_tick_v18_heldout_20261024T090154402501Z` | Completed; 2,112 learned/CF episodes and 128 F1/F2 episodes; independent physical replay of 101,376 learned/CF decisions and gate audit PASS. |
| Same-weight mechanism replay | `artifacts/same_tick_v18_mechanism_20261024T091310597633Z` | Completed; descriptive replay on 192 C18 episodes, with C18 physical costs checked against audited held-out rows. |
| Training-label fit | `artifacts/v18_label_fit_20261024T091458253872Z` | Completed; descriptive in-sample calculation from immutable audited labels and checkpoints. |

The label auditor found first-eight/last-eight winner agreement **0.326/0.379** and candidate-pair sign agreement **0.648/0.672** in cases a/b. Both passed the prelocked ≥0.30 and ≥0.63 qualification thresholds. These numbers establish only that v18 was eligible to train; they do not show that the actor learned a useful decision rule. They should not be compared as an effect size against v17's 2-versus-2 split because the split sizes differ.

## Primary physical result

Equal-case, equal-training-root means use the same 32 raw physical seeds for every learned arm. The 95% intervals resample raw physical seed clusters, preserving repeated cases and roots. The source of truth is the independently audited `replication_go_decision.json` in the held-out artifact.

| C18 minus | H+tail mean | 95% seed-cluster interval | Case a / b | Locked interpretation |
| --- | ---: | ---: | ---: | --- |
| D18, separately trained frozen context | −9.18 | [−17.59, −0.95] | −16.77 / −1.58 | Missed practical ≤−20 gate. |
| CF18, **same weights**, frozen context | −0.27 | [−3.84, +3.60] | +3.13 / −3.66 | Missed ≤−10, both-case sign and upper <0 gates. |
| L18, matched four-tape labels | −13.33 | [−22.66, −4.82] | −27.92 / +1.25 | Pooled gain, but missed both-case sign gate. |
| I18, no incidence message | −11.81 | [−19.95, −3.91] | −25.17 / +1.54 | Missed both-case sign gate. |
| R18, no remaining-time magnitude | −5.58 | [−12.60, +1.44] | −9.56 / −1.60 | Missed ≤−10 gate. |
| B18, direct pair actor | −2.52 | [−12.14, +6.52] | −15.08 / +10.05 | Pooled sign negative. |
| S18, structured matching actor | −6.54 | [−15.45, +2.57] | −27.50 / +14.42 | Pooled sign negative. |
| G18, flat full-information actor | −24.74 | [−34.27, −15.52] | −46.47 / −3.01 | Pooled sign negative. |
| P18, graph PPO without labels | −28.30 | [−36.00, −20.59] | −40.21 / −16.39 | Pooled sign negative. |
| Q18, always-request flat PPO | −207.06 | [−214.63, −199.41] | −205.53 / −208.58 | Contextual control; different request pool. |

C18's absolute mean H+tail cost was **1575.68**, versus D18 **1584.85**, CF18 **1575.94**, L18 **1589.01**, flat G18 **1600.42** and graph PPO P18 **1603.97**. C18−F1 risk matching was −52.37 and C18−F2 six-tick rollout was −6.26 on the shared physical seeds; for F2, the case contrasts were −37.91/+25.39. F1/F2 are external controls, not sample-matched trained arms. The direct same-weight comparison shows no established economic feedback gain. C18−D18's smaller improvement can reflect different learned weights and does not attribute benefit to refreshing context.

The gate file records `go: false`. Required C−CF conditions all failed, as did the C−D effect threshold, C−L both-case sign and selected ablation conditions. Independent retraining was therefore ineligible, and TEST remained unopened. No threshold was changed after outcomes.

## Descriptive mechanism and fitting checks

On 9,216 C18-visited H states, C18 made a first assignment with a feasible second decision on 1,901 states; 713 of those had more than one second edge. Holding C18 weights fixed but freezing context changed the second token on 82 states and the final greedy matching on **89/9,216 (0.97%)**. Case a/b final changes were 49/40. Mean second-step total-variation distance between updated and frozen action distributions was 0.070/0.022 by case. Separately trained C18 and D18 differed in final matching on 1,468/9,216 C18-visited states, but their weights differ, so this is not a causal feedback estimate. Only paired C18−CF18 physical episodes measure economic effect; action changes alone cannot establish value.

On the *training labels*, the equal-case/equal-root average 16-tape candidate-list cross-entropy was **2.044** for C18 after pretraining, compared with target entropy **1.902** and uniform predictor **2.071**. At each trial's DEV-selected PPO checkpoint, it was **2.652**. The matching four-tape L18 checkpoint had four-tape loss **2.478**, compared with four-tape target entropy **1.749** and the same uniform baseline **2.071**. C18's 16-tape top-candidate match rate was 0.330 after pretraining and 0.264 at selection. These are in-sample descriptive calculations on the fixed training bank; the increase after PPO does not by itself identify destructive forgetting, label quality, rollout mismatch or representation as the cause of the physical result. DEV cost, not label loss, selected checkpoints under the locked protocol.

The discrepancy between stable-enough continuation targets and weak supervised fit, followed by low same-weight action and physical effects, identifies a concrete unresolved uncertainty: whether the current actor/training path can preserve and use matching-level economic ranking information after PPO. That uncertainty warrants an independently justified training-only diagnosis before another architecture or scientific campaign. It does not justify reusing held-out outcomes to tune v18.

### Checkpoint-path addendum (descriptive, 2026-10-24)

The later training-bank-only artifact `artifacts/v18_checkpoint_path_20261024T092054623292Z` evaluated every immutable pretraining and PPO milestone checkpoint on the same audited v18 training labels, alongside the already logged DEV cost. It used no held-out or TEST state for model choice. Mean 16-tape training-label cross-entropy for C18 was 2.041 → 2.223 → 2.439 → 2.965 in case a and 2.046 → 2.307 → 2.615 → 2.974 in case b at pretrain → 2,048 → 4,096 → 8,192 PPO decisions. D18 and L18 showed the same monotone rise in both cases. Thus the loss increase at the selected checkpoint was not an artifact of choosing one unusual milestone; it began by the first measured PPO checkpoint. DEV H+tail cost did not move monotonically with training-label loss. This localizes supervised-ranking drift but does not establish that preventing it will improve physical cost.

## Secondary outcomes and compute

All rows below are held-out descriptive means per episode, pooled across the two cases and three training roots. Availability is the simulator's H availability; queues are ticks through H+tail. They are not independent endpoints for the prelocked gate.

| Metric | C18 | D18 | CF18 | L18 |
| --- | ---: | ---: | ---: | ---: |
| H availability | 0.342 | 0.350 | 0.342 | 0.343 |
| Failures | 17.57 | 18.77 | 17.55 | 18.44 |
| Queued CM ticks | 30.34 | 24.93 | 30.38 | 28.90 |
| Queued PM ticks | 29.07 | 30.96 | 29.37 | 30.56 |
| Service starts | 52.86 | 54.22 | 52.80 | 53.46 |
| Legal service waits | 3.95 | 1.61 | 4.07 | 2.65 |
| H matching cardinality sum | 38.80 | 40.20 | 38.73 | 39.31 |
| Selection seconds per H48 episode | 0.135 | 0.099 | 0.099 | 0.135 |
| Decoder forward passes per H48 episode | 45.06 | 44.18 | 45.12 | 45.59 |

The fixed skill matrix has compatible-machine degrees [5,3,4] in case a and [5,4,5] in case b. From audited H traces, C18 allocated 1,226/3,358 starts (36.5%) to the sole degree-five technician in a, versus D18 1,341/3,623 (37.0%). In b, the two degree-five technicians together received 2,842/4,091 starts (69.5%) for C18, versus D18 2,838/4,096 (69.3%). These counts describe allocation, not opportunity-adjusted quality.

Each of the 60 trials used 8,192 H PPO decisions: 491,520 total. C18/D18/L18 each had 22,178 actor and 9,473 critic parameters, and each used 4,896 supervised updates across its six case/root trials (864 per a trial, 768 per b trial). C18 logged 46,344 decoder forward passes during PPO collection versus D18 45,508 and L18 46,262; these counters exclude PPO minibatch/replay forward work. Mean trial wall time including pretraining, PPO and DEV was 121.2 s for C18, 59.7 s for D18 and 121.6 s for L18; the whole train/DEV run took 4,093 s. On the held-out panel, F2 used 1,108,416 forecast transitions over 64 episodes and averaged 1.319 s selection time per H48 episode; F1 used none and averaged 0.002 s. C18 used no forecast transitions. DEV milestone cost curves and every checkpoint are retained in the train artifact; the selected C18 milestones were two at 2,048, two at 4,096 and two at 8,192 H decisions, so no single late milestone dominated selection.

## Decision

The 16-tape labels qualified and C18 had a pooled advantage over L18, G18 and P18. The required same-weight feedback contrast was near zero, the C−D practical effect fell short, and several comparisons changed sign by case. **V18 does not establish a useful learned same-tick feedback framework.** Its negative gate and all artifacts remain preserved. A subsequent experiment needs a new prelock grounded in the unresolved learning-path uncertainty; no new TEST is authorized.
