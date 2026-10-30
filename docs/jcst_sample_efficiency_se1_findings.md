# JCST/1.1 SE1 sample-efficiency findings

Status: **scientific panel COMPLETE; independent physical/policy replay PASS; independent raw-row statistics audit PASS**, 2026-10-28 UTC. This is the local Linux campaign frozen in the [SE1 protocol](jcst_sample_efficiency_protocol_se1.md), [canonical design lock](jcst_sample_efficiency_design_lock_se1_20261028_v2.json), [prelaunch qualification lock](jcst_sample_efficiency_prelaunch_lock_se1_20261028.json) and pre-outcome [parallel scheduling amendment](jcst_sample_efficiency_parallel_amendment_se1_20261028.md). The amendment changed only scheduling: L2/L3 were live at SE1 launch, their partial outcomes did not choose SE1 settings, and SE1 used its own seed banks and artifacts. The [launch receipt](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/launch-receipt.json) records the verified tmux worker. The durable [pipeline](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/pipeline-status.json) ran to COMPLETE, in 18.19 h versus its 72 h cap.

The [collection manifest](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/manifest.json) holds 60/60 fresh training roots (two cases × S/F/A × ten roots), all eight nested 24k–384k checkpoints, 600/600 learned evaluation combinations and eight controls. The separate [replay report](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/audit-20261028T100319.754813Z/report.json) PASSed after scanning 480,000 TRAIN rows, replaying 840 fixed TRAIN traces, 122,880 learned evaluation episodes, 256 calibration episodes and 1,024 control episodes. The independent [statistical audit](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/stats-audit-20261028T122412.221311Z/report.json) PASSed all six raw-row area contrasts, 10,000-draw crossed-bootstrap intervals, gates, target arithmetic and 480 root/checkpoint rows. The [analysis report](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/report.json) is the numerical authority. Additional read-only cross-checks recomputed all 480 greedy root/checkpoint means and times, all 60 target trajectories, eight controls, six compute cells, 120 stochastic cells and calibration intervals from terminal raw rows; all matched. Engineering smoke rows were excluded from the science.

**Qualification timing deviation.** The prelaunch lock records 279/279 regression tests, the exact S/F/A two-case mini-panel, replay PASS and statistics audit PASS. Its focused SE1 tests exercised an expired deadline, seed isolation, timing identity and locked area arithmetic, but did not explicitly exercise RAM-cap, disk-cap, stage-failure preservation or in-flight timeout handling before scientific launch. Four supplemental synthetic failure-path tests were added after launch in [the resilience test file](../tests/test_jcst_sample_efficiency_resilience_se1.py) and passed 4/4 in the final audit. Those later tests do not retroactively satisfy the prelaunch timing requirement. Frozen scientific source, data, hypotheses and gates were unchanged; the deviation limits the claim about *prelaunch engineering coverage*, while the terminal collection and independent result audits remain as reported.

## Primary: area over the entire learning window

The prelocked primary metric is each root's H+tail cost area under the **24k–384k log-sample learning curve**, trapezoidally integrated and divided by log-window width. Lower is better. The six contrasts form one family, with two-sided 99.1667% Bonferroni-adjusted crossed-bootstrap intervals over independent training roots and paired physical seeds. A per-case gate passes only if the mean difference is at most −10 cost units and the adjusted upper bound is below zero. An across-case claim requires both cases to pass. Checkpoints and stochastic action repetitions are not independent roots.

| Case | S area | A area | F area | Contrast | Difference | Adjusted interval | Gate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Base | 785.17 | 779.92 | 900.03 | S−F | −114.86 | [−130.33, −99.72] | PASS |
| Base |  |  |  | A−F | −120.11 | [−131.13, −109.50] | PASS |
| Base |  |  |  | S−A | +5.25 | [−8.85, +19.83] | FAIL |
| Generalist | 799.15 | 784.99 | 955.35 | S−F | −156.20 | [−182.19, −129.92] | PASS |
| Generalist |  |  |  | A−F | −170.36 | [−190.76, −152.18] | PASS |
| Generalist |  |  |  | S−A | +14.16 | [−5.69, +35.85] | FAIL |

**Both structured packages, S and A, pass the two-case per-decision sample-efficiency gate against F. The incremental cross-message gate S−A fails in both cases.** S−F is the whole structured-actor package, A−F is the structured package without cross-entity messages, and S−A is the incremental message contrast. The S−A intervals contain zero and also allow some benefit or harm; they do **not** establish equivalence or that messages are useless. There is no new RL algorithm claim.

## Where the curves differ

The table reports the equal-root mean of greedy H+tail cost over the same 128 physical evaluation seeds at every checkpoint. The complete [480-row root/checkpoint table](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/root_checkpoint.csv) retains all root trajectories, H/tail splits and measured training elapsed times. [Base sample curve](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/learning_curve_samples_base.png) and [generalist sample curve](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/learning_curve_samples_generalist.png) are standalone 1440×900 PNGs with every root faintly drawn and the arm means emphasized.

| Case | Decisions | S | A | F |
| --- | ---: | ---: | ---: | ---: |
| Base | 24k | 1014.9 | 1012.1 | 1101.3 |
| Base | 48k | 791.5 | 758.9 | 1077.3 |
| Base | 72k | 742.4 | 746.3 | 1000.9 |
| Base | 96k | 746.4 | 746.0 | 820.5 |
| Base | 144k | 742.1 | 742.1 | 765.6 |
| Base | 192k | 738.8 | 741.3 | 753.5 |
| Base | 288k | 736.5 | 741.2 | 743.5 |
| Base | 384k | 736.3 | 741.1 | 740.8 |
| Generalist | 24k | 1053.9 | 1057.7 | 1058.3 |
| Generalist | 48k | 881.0 | 808.3 | 1056.6 |
| Generalist | 72k | 740.4 | 731.0 | 1057.6 |
| Generalist | 96k | 725.6 | 730.9 | 1051.0 |
| Generalist | 144k | 726.2 | 726.6 | 928.3 |
| Generalist | 192k | 723.3 | 726.5 | 805.4 |
| Generalist | 288k | 726.9 | 725.1 | 757.5 |
| Generalist | 384k | 724.3 | 725.1 | 746.9 |

S/A separate from F by 48k in both cases. F catches up sharply between 96k and 384k: in base its mean is within 4.5 of S at 384k and 0.3 below A, so the A/F mean curves cross near the end. In generalist F remains 22.6 above S and 21.8 above A at 384k. S/A mean ordering changes along the grid; there is no persistent S advantage. These observations apply only to the locked 24k–384k window.

Root-area ranges do not overlap between either structured arm and F: base S 756.56–816.37, A 764.20–787.03, F 885.22–913.59; generalist S 772.42–839.97, A 770.15–801.03, F 930.02–1000.40. The primary gains are not driven by one or two favorable roots. At 48k, generalist S is variable (740.6–1058.1), but by 96k its roots are 719.9–731.4. F generalist's 48k mean is 1056.6 with 48.0 of 48 horizon decisions spent in WAIT, but its 384k roots are 721.9–780.8, showing recovery rather than a permanent collapse. No S/A root's 384k mean exceeds its own 96k mean by more than 10 units. Local nonmonotonicity remains; the panel does not prove monotone learning.

The [greedy cost-component and action table](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/greedy_diagnostics.csv) includes all arms/cases/checkpoints. At 96k generalist, F incurs 895.0 unavailable cost and 32.5 WAIT decisions versus S 534.6/20.2 and A 532.3/20.8. At 96k base, F's mean failure cost is 198.1 versus S 137.5 and A 144.5. These are behavioral descriptions, not isolated causal explanations for the architecture gap.

The predeclared [stochastic table](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/stochastic_checkpoint.csv) covers 96k and 384k, four action repetitions nested in each of 128 physical seeds. For example, 96k generalist stochastic means are S 739.9, A 739.8 and F 922.7; F improves relative to its 1051.0 greedy mean, but this secondary deployment mode cannot replace the primary greedy area. Across roots, mean within-physical-seed **sample** SD is about 34–42 cost units depending on arm/case/checkpoint; action repetitions are not new physical replicates. All 120 stochastic cells and their within-seed SDs were rechecked against raw rows.

## Independently fixed target and censoring

Before learner evaluation, a separate 128-seed threshold5 calibration panel fixed target H+tail cost at 95% of its calibration mean. Base threshold5 calibration mean was **847.79** (95% physical-seed bootstrap interval [833.89, 861.73]), target **805.40** (conditional interval [792.20, 818.65]). Generalist calibration mean was **762.58** ([753.27, 771.88]), target **724.45** ([715.61, 733.28]). The target lock was timestamped before training/evaluation; calibration and evaluation seed roles are separate. Target intervals describe uncertainty in this conditional reference, not uncertainty in a learner contrast.

A root attains only if **two consecutive greedy checkpoint means** are at or below its case target; the table records the first checkpoint in that pair. Confirmation is only known after the next checkpoint. Unattained roots remain censored at 384k. No interpolation or best-root choice was used.

| Case | Arm | Attained / 10 by 384k | Median first qualifying samples | Median measured training seconds | Restricted mean samples to 384k | Restricted mean training seconds to 384k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | S | 10 | 48k | 80.28 | 55.2k | 92.55 |
| Base | A | 10 | 48k | 80.31 | 48.0k | 80.52 |
| Base | F | 10 | 144k | 162.36 | 124.8k | 142.12 |
| Generalist | S | 3 | not reached | not reached | 321.6k | 1003.45 |
| Generalist | A | 4 | not reached | not reached | 283.2k | 860.86 |
| Generalist | F | 0 | not reached | not reached | 384.0k | 780.49 |

| Decisions | Base S / A / F attained | Generalist S / A / F attained |
| ---: | ---: | ---: |
| 24k | 0 / 0 / 0 | 0 / 0 / 0 |
| 48k | 7 / 10 / 0 | 0 / 0 / 0 |
| 72k | 10 / 10 / 0 | 0 / 2 / 0 |
| 96k | 10 / 10 / 4 | 1 / 3 / 0 |
| 144k | 10 / 10 / 10 | 2 / 3 / 0 |
| 192k | 10 / 10 / 10 | 2 / 3 / 0 |
| 288k | 10 / 10 / 10 | 3 / 4 / 0 |
| 384k | 10 / 10 / 10 | 3 / 4 / 0 |

These are retrospective first-pair checkpoints; an online decision to stop training would require observing the successor checkpoint. Base's conditional target is reached earlier and in fewer **measured training seconds** by S/A than F. Generalist's target is stringent relative to all arms: F has no attainment and S/A have less than half their roots attaining by 384k, so no uncensored median or cross-case time-to-quality success claim is available. The restricted means keep every root, including censored ones, but are not an uncensored mean time-to-attainment. Checkpoint spacing limits resolution. The complete per-root attainment and censoring records are in the [analysis report](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/report.json).

## Controls, compute and contention

The same evaluation physical panel includes full-information threshold5, risk/duration matching, corrective-only and WAIT controls. Mean H+tail cost:

| Case | Threshold5 | Risk/duration | Corrective only | WAIT |
| --- | ---: | ---: | ---: | ---: |
| Base | 850.98 | 967.67 | 1011.30 | 1069.73 |
| Generalist | 761.05 | 899.59 | 975.22 | 1056.59 |

At 384k, all three learned arms have lower mean cost than threshold5 in both cases on this bank; this is descriptive, not a multiplicity-adjusted primary control gate. The primary inference remains the six area contrasts. No requester restrictions, fees, hidden information or forced service were introduced.

Every arm trained with one Torch intra/inter-op thread and the same PPO recipe, sample budget and complete legal-action catalogue. Full-run mean measured **384k training elapsed per root** was base S/A/F 663.9 / 665.8 / 445.9 s and generalist 1199.3 / 1172.7 / 780.5 s. Structured arms take about 1.5× F's training time at equal samples. Their actor parameter counts were base S/A/F 13,521 / 13,521 / 12,391 (shared-architecture critic 10,113) and generalist 14,577 / 14,577 / 14,281 (critic 11,457). Mean root phase seconds at 384k show that actor inference and PPO updates account for most of the difference: base S inference/update 345.4/178.6, A 351.8/175.6, F 181.8/130.7; generalist S 373.3/574.2, A 367.8/564.0, F 198.4/358.4. Simulator, catalogue, encoding/mask and checkpoint/logging phases are separated in the [compute report](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/report.json). Summed across all 60 roots, measured training occupied 49,279.6 s, learned-policy evaluation 7,704.6 s, and calibration/controls/other collector overhead about 34.8 s. Independent replay took 8,342.0 s, analysis/plots 111.5 s and statistical audit 25.7 s; total pipeline elapsed was 65,499.9 s (18.19 h).

A separate held-out-state inference microbenchmark used the same 32 states per case, three warmup passes and 20 timed passes per root. Mean model-selection latency in microseconds was base S/A/F **545.7 / 564.0 / 182.9** and generalist **642.2 / 658.7 / 252.3**. This controlled microbenchmark shows F's cheaper inference; it does not control whole-training wall time. The [base compute-axis plot](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/learning_curve_time_base.png) and [generalist compute-axis plot](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/learning_curve_time_generalist.png) use measured checkpoint training elapsed, with all roots shown.

SE1 deliberately ran concurrently with L2/L3. Two saved [contention snapshots](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/contention-snapshots.jsonl) confirm three roughly one-CPU workers early in the campaign; L3 and then L2 collection ended before SE1's later stages. Consequently the observed base time-to-target difference is descriptive under changing shared-host contention. **There is no matched-wall-time or controlled end-to-end training-speed claim across both cases.** The collector peaked at 751,168 KiB RSS (about 734 MiB), independent replay at 522,484 KiB, and the new SE1 run directory is about 3.6 GiB, below the locked 8 GiB process and 20 GiB artifact caps. SE1 used one scientific collector or audit worker at a time, one Torch thread and no GPU.

## Answers and limits

1. **Do S/A learn faster than F?** Yes by the prelocked, adjusted log-area criterion in both base and generalist. This is a per-simulator-decision result over 24k–384k, not a claim at every budget.
2. **Where does the advantage appear?** It becomes pronounced by 48k–96k. F approaches S/A by 384k in base, but remains behind in generalist. The base F/A mean curves cross near the last checkpoint.
3. **Does S outperform A?** The two S−A area gates fail. The intervals allow positive or negative incremental effects; cross-message value remains unproven rather than disproven.
4. **Does fewer samples save real time?** Base's conditional target is reached earlier in measured training time by S/A, but S/A are slower per equal sample and generalist has heavy censoring. Shared-host contention prevents a matched-wall-time causal claim.
5. **Is the result stable?** Both S−F and A−F pass in both cases, with disjoint observed root-area ranges. Generalist target attainment is unstable across roots, and S−A is inconclusive.
6. **What is the simplest supported architecture?** A is the parsimonious supported structured choice: it retains the two-case A−F area advantage without cross-entity message inputs. This does not prove A and S equivalent or that messages can never help.
7. **What remains unknown?** Controlled full-training wall-time benefit, behavior beyond 384k decisions or outside the two core JCST/1.1 cases, an isolated encoder mechanism, and whether cross-messages help under a different prelocked design. No extra architecture search or sealed TEST was opened to answer these questions.

The separate [L2 replication findings](jcst_learnability_l2_replication_findings.md) show useful learning and threshold5 competitiveness on independent retraining but no selected-recipe S−F/S−A gate; [L3 findings](jcst_matched_architecture_l3_findings.md) show an adjusted S−F advantage at the fixed 96k endpoint but no S−A advantage. SE1 adds the prelocked full-curve inference and conditional time-to-quality assessment. The studies have different seed banks and estimands and are not pooled post hoc.

## Reproduction and handoff

The [prelaunch lock](jcst_sample_efficiency_prelaunch_lock_se1_20261028.json) and [launch receipt](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/launch-receipt.json) preserve exact source/config/protocol hashes, qualification evidence, command, session, PID, budgets and environment. The completed [pipeline status](../artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/pipeline-status.json) records stage commands and timing. From the repository root with the frozen sources and one Torch thread, the postcollection commands are:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_sample_efficiency_audit_se1.py artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_sample_efficiency_analyze_se1.py artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z --audit artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/audit-20261028T100319.754813Z/report.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_sample_efficiency_stats_audit_se1.py artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z artifacts/jcst_sample_efficiency/run-20261027T181257.930696Z/analysis-20261028T122222.042632Z/report.json
```

Each auditor/analysis invocation makes a fresh timestamped directory. These commands reproduce the checks; they do not retrain or modify the completed scientific panel. The finished tmux session no longer exists because its Python driver exited normally. Raw results, logs, target lock, heartbeat, pipeline status, four standalone PNGs and all audits remain in the run directory. `CURRENT_STATE.md` is the concise project handoff.
