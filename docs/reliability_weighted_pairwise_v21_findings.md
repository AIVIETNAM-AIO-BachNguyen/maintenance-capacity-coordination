# V21 reliability-weighted pairwise supervision: audited findings

Status: **closed, no-go** on 2026-10-25 (Asia/Taipei). The protocol in `docs/reliability_weighted_pairwise_v21_prelock.md` was locked before v21 code or outcomes. Scientific collection used clean, pushed source `ab00d2dab73708ce11ba19cb94f4323b938d4417`. All training and evaluation data came from the unchanged simulator. No independent retraining or new TEST ran.

## Question and comparison

V20 established an effective route from busy-technician time and provisional loss of a sole compatible worker into the actor, but its learned matching-label rankings were near uniform and its physical gates failed. V21 changed only the supervised matching-label objective during pretraining and PPO rehearsal. C21 and D21 used paired-tape reliability-weighted candidate comparisons on the same audited 16-tape v20 training bank; L21 used the prior listwise objective on that bank with C21's dynamic actor, initialization, roots and budget. D21 froze root context while updating legality. CF21 used selected C21 weights with context frozen only at inference. C21−L21 isolates the learning-signal intervention, C21−D21 compares separately trained graph feedback, and C21−CF21 measures same-weight physical feedback. All use the same present-state information, legal threshold-admitted matching/WAIT action set and unchanged physics.

## Provenance and independent audits

The final-source full suite passed 236 tests. Bounded Ubuntu smoke `artifacts/same_tick_v21_smoke_20261024T175637312635Z` and independent smoke audit `artifacts/same_tick_v21_audit_20261024T175658339373Z` passed. An engineering reference-selection smoke checked 48 case/arm/root combinations and 144 DEV episodes. Scientific stages were audited before held-out cost interpretation:

| Stage | Artifact | Independent result |
| --- | --- | --- |
| C21/D21/L21 train and fresh DEV | `artifacts/same_tick_v21_train_20261024T180209859141Z` | PASS; 18 trials, 147,456 H decisions, 864 new-arm DEV and 2,304 old-reference DEV episodes. |
| Fresh paired held-out | `artifacts/same_tick_v21_heldout_20261025T043815311226Z` | PASS; 2,304 learned/CF and 128 F1/F2 episodes; 110,592 learned/CF and 6,144 control H decisions replayed. |

The train audit is `artifacts/same_tick_v21_audit_20261025T041105806790Z`; the held-out audit is `artifacts/same_tick_v21_audit_20261025T044406682463Z`. The first train-audit attempt created an empty timestamped directory and stopped without a verdict; it remains preserved. A fresh audit then passed. The independent held-out auditor also recomputed the locked cluster-bootstrap contrasts and `replication_go_decision.json`.

## Fresh held-out economic result

Each learned arm used the same 32 new raw physical seeds in both cases and all three training roots. Numbers are paired C21 minus comparator H48+tail16 cost, so negative favors C21. Intervals resample **raw physical seed clusters**, preserving case/root/arm pairing; training-root means are shown separately.

| Contrast | Overall | Case a / b | Training roots 0 / 1 / 2 | Cluster 95% interval | Locked decision |
| --- | ---: | ---: | ---: | ---: | --- |
| C21−L21, pairwise versus listwise | **−4.01** | −10.26 / +2.24 | +10.88 / −5.58 / −17.33 | [−10.72,+2.33] | Fail: required ≤−10, both cases negative and upper <0. |
| C21−D21, separately trained feedback | **−1.69** | −1.64 / −1.75 | +7.72 / −1.81 / −10.98 | [−9.19,+5.41] | Fail: required ≤−20 and upper <0. |
| C21−CF21, same-weight feedback | **+1.24** | +2.02 / +0.46 | −1.69 / +4.83 / +0.58 | [−3.05,+5.42] | Fail: required ≤−10, both cases negative and upper <0. |

C21's pooled absolute mean cost was **1568.24**, versus L21 **1572.25**, D21 **1569.93** and CF21 **1567.00**. The contextual reference C19 was **1569.19**, so C21−C19 was −0.95 [−8.41,+6.53]. C21−B18 was −6.47 [−13.93,+1.51], C21−S18 −16.59 [−23.36,−9.71], and C21−G18 −29.72 [−38.31,−21.40]; their locked sign conditions passed. C21−F2 rollout was −14.40, satisfying the allowed ≤+20 difference. These reference results cannot rescue the three failed primary conditions. F1 risk matching and F2 rollout were contextual controls, not sample-matched scratch learners.

The small C21−L21 mean favors pairwise supervision overall but changes sign in case b and root 0; its interval spans zero. C21−D21 is negative in both cases and two roots, yet misses the practical magnitude and its interval spans zero. The same-weight C21−CF21 comparison is slightly adverse overall and in both cases. Thus this training-loss intervention does not establish an economic benefit from refreshed within-tick context. The independently audited replication decision is **`go: false`**.

## Secondary outcomes and compute

Descriptive held-out means per learned episode, pooled across both cases and three roots:

| Metric | C21 | L21 | D21 | CF21 |
| --- | ---: | ---: | ---: | ---: |
| H availability | 0.351 | 0.345 | 0.346 | 0.350 |
| H failures | 11.38 | 11.09 | 11.21 | 11.39 |
| Queued CM ticks | 28.72 | 30.80 | 30.03 | 28.19 |
| Queued PM ticks | 29.80 | 29.48 | 29.00 | 29.55 |
| Service starts | 53.04 | 52.63 | 52.47 | 53.15 |
| Legal service waits | 4.10 | 4.16 | 4.92 | 4.18 |
| H matching-cardinality sum | 38.76 | 38.52 | 38.22 | 38.83 |
| Selection seconds / H episode | 0.129 | 0.129 | 0.095 | 0.094 |
| Decoder forward passes / H episode | 47.74 | 48.01 | 48.14 | 47.33 |

Each new actor had 22,178 parameters and its critic 9,473. Each C21/L21/D21 arm consumed 49,152 physical PPO H decisions over six trials. Total training decoder forward passes were 48,950 / 48,608 / 48,893, respectively. Mean trial wall time including DEV was 164.4 / 165.4 / 75.6 seconds, reflecting the higher compute of dynamic graph feedback. Full train/DEV took 2,658 seconds; held-out collection took 332 seconds. Mean DEV H+tail costs across six case/root trials at milestones 2,048 / 4,096 / 8,192 were C21 1570.57 / 1581.46 / 1577.17, L21 1579.77 / 1574.41 / 1580.50 and D21 1574.46 / 1577.01 / 1594.29. Checkpoints were selected independently per trial by the locked minimum DEV rule; these curves were not used to revise the protocol. Secondary associations do not decompose causal cost differences.

## Decision and remaining uncertainty

V21 passes engineering, train/DEV and fresh held-out replay audits, but all three required economic tests of pairwise learning and within-tick feedback fail. The pairwise loss did not produce a qualified physical improvement over matched listwise L21; dynamic feedback did not beat separately trained frozen D21 by the required amount; and freezing C21's own context slightly improved pooled cost. No independent training replication or TEST is eligible under the locked protocol.

The evidence narrows this intervention: a reliability-weighted pairwise target on the current eight-candidate, 16-tape training bank is insufficient. It does **not** show that input information is absent or that all feedback encoders are ineffective. Training-root variability, uncertain small-gap labels, representation of scarce alternatives, and economic alignment remain unresolved; this experiment cannot identify which dominates. A follow-up needs a separately justified hypothesis and prelock, not an outcome-guided v21 loss/threshold adjustment. Simulator-generated trajectories remain sufficient for RL training; real operational logs are optional later calibration.
