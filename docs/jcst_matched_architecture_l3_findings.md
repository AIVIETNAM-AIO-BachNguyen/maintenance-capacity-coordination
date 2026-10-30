# JCST/1.1 L3 matched-architecture findings

Status: **complete and independently audited**, 2026-10-28 UTC. This is the prelocked [L3 protocol](jcst_matched_architecture_protocol_l3.md) run under the pre-outcome [parallel scheduling amendment](jcst_matched_architecture_parallel_amendment_l3_20261027.md). The [collection manifest](../artifacts/jcst_matched_architecture/collect-20261027T094717.495781Z/manifest.json) contains 60/60 new training roots and 240/240 evaluation combinations. The separate [physical/policy replay](../artifacts/jcst_matched_architecture/audit-20261028T002118.862567Z/report.json) and [raw-row statistical audit](../artifacts/jcst_matched_architecture/stats-audit-20261028T012845.622876Z/report.json) both PASSed. The [analysis report](../artifacts/jcst_matched_architecture/analysis-20261028T012826.787058Z/report.json) is the numerical authority and retains all ten root means for each arm/case/checkpoint/mode. Collection and audit completed within the 30-hour L3 cap.

## Locked primary decision

The endpoint is 96,000 controlled decisions and greedy deployment on the same 128 fresh physical seeds per case. Negative S-minus-comparator H+tail cost favors S. Intervals are two-sided 98.75% Bonferroni-adjusted crossed-bootstrap intervals (4,000 draws; 4 primary contrasts). A contrast passes only with mean at most −10 and upper bound below zero.

| Case | Contrast | Mean H+tail difference | Adjusted interval | Gate |
| --- | --- | ---: | ---: | --- |
| Base | S−F | −69.20 | [−87.51, −50.25] | PASS |
| Base | S−A | +1.08 | [−7.65, +9.89] | FAIL |
| Generalist | S−F | −312.32 | [−343.92, −269.64] | PASS |
| Generalist | S−A | +1.50 | [−5.93, +8.11] | FAIL |

**The structured-actor package passes the two-case, per-decision sample-efficiency gate relative to F at 96k. The cross-entity message mechanism does not pass its two-case gate relative to A.** S−F compares different actor packages with unequal parameter count and compute; it is not an isolated message effect. S and A use the same structured actor size, with A's cross-entity inputs zeroed, and have nearly identical primary costs. Standard PPO and the simulator contract were unchanged.

## Cost curves and operating behavior

All numbers below are mean H+tail cost over ten roots and the fixed 128-seed physical panel. The 384k stochastic evaluation also averages four independent action repetitions within each physical seed; it is secondary.

| Case | Checkpoint/mode | S | F | A |
| --- | --- | ---: | ---: | ---: |
| Base | 24k greedy | 1052.55 | 1097.89 | 987.65 |
| Base | 96k greedy | 743.23 | 812.43 | 742.15 |
| Base | 384k greedy | 735.65 | 738.89 | 740.42 |
| Base | 384k stochastic | 739.82 | 748.76 | 741.25 |
| Generalist | 24k greedy | 1057.21 | 1056.75 | 1056.75 |
| Generalist | 96k greedy | 728.92 | 1041.24 | 727.43 |
| Generalist | 384k greedy | 723.94 | 742.55 | 726.60 |
| Generalist | 384k stochastic | 728.37 | 754.15 | 729.88 |

The large S−F gap at 96k contracts by 384k: base −3.24 and generalist −18.61 descriptive units. This is consistent with faster per-decision learning by S, while the endpoint-specific gate does not imply a lasting asymptotic advantage. At 96k in generalist, F spends 27.91 of 48 horizon ticks in discretionary WAIT versus 20.89 for S; its mean unavailable cost is 852.29 versus S's 535.61. At 96k in base, F has 13.15 failures versus S's 9.87 and mean failure cost 197.20 versus 148.05. These audited episode summaries describe behavior; they do not isolate a causal mechanism.

At the primary endpoint, the complete H/tail and operational summaries are:

| Case/arm | H | Tail | Preventive | Corrective | Unavailable | Failure | PM / CM starts | Failures | Unavailable intervals | Forced / discretionary WAIT | Unfinished jobs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base S | 507.64 | 235.59 | 32.06 | 18.49 | 544.63 | 148.05 | 32.06 / 9.24 | 9.87 | 90.77 | 6.61 / 14.07 | 0.76 |
| Base F | 576.33 | 236.10 | 29.28 | 24.94 | 561.01 | 197.20 | 29.28 / 12.47 | 13.15 | 93.50 | 4.98 / 13.54 | 0.74 |
| Base A | 508.53 | 233.62 | 33.39 | 17.32 | 552.17 | 139.27 | 33.39 / 8.66 | 9.28 | 92.03 | 7.66 / 13.31 | 0.75 |
| Generalist S | 508.59 | 220.34 | 32.50 | 18.03 | 535.61 | 142.78 | 32.50 / 9.02 | 9.52 | 89.27 | 0.29 / 20.89 | 0.68 |
| Generalist F | 808.19 | 233.06 | 26.26 | 18.30 | 852.29 | 144.40 | 26.26 / 9.15 | 9.63 | 142.05 | 0.00 / 27.91 | 0.75 |
| Generalist A | 510.88 | 216.55 | 32.87 | 17.64 | 536.98 | 139.93 | 32.87 / 8.82 | 9.33 | 89.50 | 0.45 / 22.51 | 0.65 |

The unchanged always-request and full-information controls were evaluated on the same 128 physical seeds per case, once per case:

| Case | Threshold5 | Risk/duration | Corrective only | WAIT |
| --- | ---: | ---: | ---: | ---: |
| Base | 839.93 | 964.91 | 1011.06 | 1070.99 |
| Generalist | 774.34 | 897.47 | 981.51 | 1056.75 |

S and A beat threshold5 descriptively at 96k in both cases; these are secondary control comparisons without primary adjusted intervals. The learner result does not require requester-specific necessity or a new RL algorithm.

## Compute, verification, and limits

At 96k decisions, median measured training seconds per root were base S/F/A **145.90 / 97.56 / 145.48**, and generalist **266.01 / 200.05 / 303.45**. At 384k they were base **589.56 / 400.87 / 593.72** and generalist **1101.91 / 844.91 / 1259.43**. Actor parameter counts were base S/F/A **13,521 / 12,391 / 13,521**, with a shared 10,113-parameter critic, and generalist **14,577 / 14,281 / 14,577**, with a shared 11,457-parameter critic. Mean measured selection time per 48-decision episode at 96k was base **0.0519 / 0.0312 / 0.0533 s**, and generalist **0.0626 / 0.0421 / 0.0646 s**. These figures rule out a matched-wall-time or equal-compute interpretation of the per-decision S−F gate.

The collector took 51,977.7 s (14.44 h), peaked at 868,960 KiB RSS (about 849 MiB), and its artifact directory is about 1.8 GiB, within the 30 h, 8 GiB process and 5 GiB new-artifact caps. The independent replay scanned **480,000 TRAIN rows**, physically replayed **540 locked TRAIN traces**, physically and policy replayed **53,760 evaluation episodes**, and physically replayed **1,024 controls**. The independent statistics audit recomputed all four primary contrasts, their bootstrap intervals and gates, and checked 24 secondary arm/case/checkpoint/mode cells from raw rows. The separate postprocess [status](../artifacts/jcst_matched_architecture/postprocess-20261028T002110.175190Z/status.json) is COMPLETE. No result was selected, tuned or replaced after observing outcomes. The L2 replication and SE1 learning-curve campaign have separate seed banks and remain pending; L3's 96k verdict must not be substituted for their prelocked gates. Sealed TEST remains unopened.

## Reproduction

From the repository root with the frozen sources and one Torch thread, the postcollection sequence is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_matched_architecture_audit_l3.py artifacts/jcst_matched_architecture/collect-20261027T094717.495781Z
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_matched_architecture_analysis_l3.py artifacts/jcst_matched_architecture/collect-20261027T094717.495781Z --audit artifacts/jcst_matched_architecture/audit-20261028T002118.862567Z/report.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_matched_architecture_stats_audit_l3.py artifacts/jcst_matched_architecture/collect-20261027T094717.495781Z artifacts/jcst_matched_architecture/analysis-20261028T012826.787058Z/report.json
```

These commands create fresh timestamped audit and analysis directories; the links above point to the completed audited chain.
