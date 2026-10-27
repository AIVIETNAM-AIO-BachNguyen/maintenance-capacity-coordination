# JCST/1.1 L1: first demand–capacity structured PPO campaign

**Status 2026-10-26 Asia/Taipei: closed, no-go under the prelocked confirmation gate.** L1 implemented a real centralized demand–capacity actor and evaluated five independent training roots for each of structured S, flat full-information F and message-ablation A in both base and generalist. The campaign did not establish the full framework objective. All numbers below are uncalibrated simulator H+tail cost units; lower is better. No sealed TEST or independent training replication ran.

## Question, design and provenance

The [pre-outcome protocol](jcst_framework_protocol_v1.md) and [Stage 0 audit](jcst_framework_stage0.md) fixed the unchanged JCST/1.1 H48 plus fixed-risk B16 problem, full legal complete matchings and WAIT, five training roots per learned arm/case, 24,000 controlled simulator decisions per root, three equal-sample checkpoints, 16 DEV seeds and 128 fresh confirmation seeds. Gamma=1 and the final controlled reward includes tail cost once. S passes duration-conditioned component–technician messages, including busy worker state, before scoring whole matchings. A retains the same information, modules, parameter count and joint matching scorer but removes cross-entity inputs to those messages. F scores the same flat full state and a complete-action one-hot vector. All share PPO and a full-information critic. The actors have 13,521/12,391/13,521 parameters in base and 14,577/14,281/14,577 in generalist (S/F/A), within the locked 10% matching target; critic sizes are identical within cases.

[Prelaunch source/protocol hashes](../artifacts/jcst_framework/prelaunch-lock-20261025T175225.414966Z.json) followed 263/263 regression tests and a final-source bounded Linux [smoke](../artifacts/jcst_framework/smoke-20261025T175204.916360Z/manifest.json) with [independent replay PASS](../artifacts/jcst_framework/smoke-20261025T175204.916360Z/independent_audit.json) on 34 engineering episodes. The [TRAIN/DEV run](../artifacts/jcst_framework/train_dev-20261025T175233.722390Z/manifest.json) completed 30 trials, 720,000 PPO samples / controlled simulator decisions, 15,000 training episodes, 1,440 learned DEV episodes and 160 control DEV episodes. Its [v3 independent audit](../artifacts/jcst_framework/train_dev-20261025T175233.722390Z/independent_audit.json) replayed all 16,600 physical episodes and selected one checkpoint per arm/case after PASS. Selected samples were base S/F/A 24,000/24,000/6,000 and generalist S/F/A 12,000/24,000/6,000. These counts, the original source, the v3 auditor, analysis code and confirmation seed role were frozen in the [preconfirmation lock](../artifacts/jcst_framework/confirmation-lock-20261025T182413.224980Z.json).

The [confirmation run](../artifacts/jcst_framework/confirmation-20261025T182420.983964Z/manifest.json) evaluated all selected roots and the five controls on 128 new paired physical seeds per case. Its [independent replay](../artifacts/jcst_framework/confirmation-20261025T182420.983964Z/independent_audit.json) passed 5,120 complete episodes, legal catalogues, costs, H/tail accounting, fixed tail actions and deterministic checkpoint selections. The [prelocked analysis](../artifacts/jcst_framework/confirmation-20261025T182420.983964Z/analysis.json) crossed independent training-root and physical-seed variation, using 2,000 two-way bootstrap resamples and six Bonferroni-adjusted two-sided intervals. A separate [raw-row statistics audit](../artifacts/jcst_framework/stats-audit-20261025T183421.896046Z/audit.json) independently reconstructed the paired means, ran 4,000 alternative resamples and checked gate arithmetic; PASS. Physical seeds were shared for pairing but threshold episodes were never multiplied into five independent samples.

The original audit attempt stopped at a numerical near-tie because it compared raw actor logits while the frozen collector chooses the argmax of PyTorch Categorical-normalized logits. A second audit attempt omitted the current training-root index from one metadata check. Both failed attempts are preserved as [auditor-only errata v2](../artifacts/jcst_framework/train_dev-20261025T175233.722390Z/audit_erratum_v2.json) and [v3](../artifacts/jcst_framework/train_dev-20261025T175233.722390Z/audit_erratum_v3.json). The corrected v3 auditor replayed the *entire* source panel. Neither fix changed collector, physical traces, checkpoints, estimands, gates or protocol.

## Primary confirmation result

The primary outcome is undiscounted physical H+tail total cost. Negative differences favor S. Each interval is the locked 99.17% two-sided family-adjusted crossed-bootstrap interval.

| Case | Contrast | Mean difference | Adjusted interval | Locked criterion | Result |
| --- | --- | ---: | ---: | --- | --- |
| base | S−F | −71.18 | [−146.15, −20.92] | mean ≤−10, upper <0 | Pass in base |
| base | S−A | −4.24 | [−92.80, +83.70] | mean ≤−10, upper <0 | Fail |
| base | S−threshold5 | +173.96 | [+95.67, +226.08] | mean ≤+10, upper ≤+10 | Fail |
| generalist | S−F | −8.07 | [−20.98, 0.00] | mean ≤−10, upper <0 | Fail |
| generalist | S−A | −5.13 | [−16.46, 0.00] | mean ≤−10, upper <0 | Fail |
| generalist | S−threshold5 | +297.51 | [+283.69, +311.18] | mean ≤+10, upper ≤+10 | Fail |

S therefore has a **case-specific base advantage over F**, but the two-case representation criterion is not met. The specific cross-message mechanism is not supported over A in either case. S is not economically competitive with the strong selected threshold5 controller in either case; it is also worse than risk and corrective on the confirmation bank. The complete first-framework objective is **not met**. The prelocked replication eligibility requires all three confirmation criteria in both cases, so no new independent training replication was opened. A fresh physical evaluation of the same checkpoints would not have been training replication.

The [per-case summary](../artifacts/jcst_framework/report-20261025T183313.694605Z/summary.csv), [per-training-root table](../artifacts/jcst_framework/report-20261025T183313.694605Z/per_training_seed.csv), [paired physical-seed table](../artifacts/jcst_framework/report-20261025T183313.694605Z/paired_physical.csv) and [raw confirmation episodes](../artifacts/jcst_framework/report-20261025T183313.694605Z/confirmation_episodes.csv) preserve H, tail, all four cost components, action/availability metrics and variation. Mean total cost was base S/F/A/threshold5 = 1016.85/1088.02/1021.08/842.88; generalist = 1054.52/1062.60/1059.66/757.01. These confirmation controls are close to, but independently seeded from, C1's threshold means.

## What the policies did and what remains uncertain

S made legal complete-matching decisions and received busy-worker information through a tested actor path, but it **waited too often** relative to threshold5: mean controlled WAIT decisions were 35.45/48 versus 28.48/48 in base and 44.87/48 versus 27.59/48 in generalist. S started fewer PM and CM jobs and accumulated 142.52 versus 98.09 unavailable machine-intervals in base and 148.92 versus 88.88 in generalist. The lower S failure charge partly reflects machines spending more time unavailable and therefore not accruing operating-age failures; it does not offset lost production. Generalist A chose WAIT for all 48 controlled ticks in all five roots; three generalist S roots also matched that behavior. These are descriptive mechanisms in the observed rollout population, not proof that the architecture cannot learn under another prelocked campaign.

The [sample learning curves](../artifacts/jcst_framework/report-20261025T183313.694605Z/learning_by_samples.pdf) show base S DEV mean falling from 1034.6 at 6,000 samples to 1015.5 at 24,000, while generalist S stayed near 1055. The [compute-axis curves](../artifacts/jcst_framework/report-20261025T183313.694605Z/learning_by_compute.pdf) use checkpoint/model filesystem times as approximate elapsed measures. Selected base S per-root confirmation means ranged 914.53–1071.79 (SD 60.49); generalist S ranged 1046.44–1059.66 (SD 7.04), with the narrow range largely reflecting WAIT collapse. Initial-root variation is reported, but **retraining stability of a beneficial effect was not established** because replication was ineligible.

The [training compute table](../artifacts/jcst_framework/report-20261025T183313.694605Z/training_compute.csv) records roughly 36 s versus 24 s per base S/F root and 66 s versus 46 s per generalist S/F root at equal 24,000 samples. S selection time averaged 0.037/0.042 s per 48-decision episode (base/generalist), versus F 0.018/0.023 s and threshold5 0.00054/0.00133 s; the full [cost plot](../artifacts/jcst_framework/report-20261025T183313.694605Z/confirmation_cost.pdf) is descriptive. R2 spent 1,482/3,931 forecast transitions per episode; S/F/A used zero forecast transitions and did not access actual future noise. Equal simulator samples are not equal wall time or model access. The TRAIN/DEV and confirmation stages took 1,479 s and 315 s, with peak process RSS about 842 MiB and 507 MiB, well within caps. No efficiency-success claim was prelocked or made.

## Decision and limits

L1 is closed after one predeclared S/F/A design round and a complete audited confirmation. Its unused time and maximum iteration allowance are not a reason to tune after confirmation. The negative result is specific to the fixed JCST/1.1 base/generalist fixtures, initial healthy/free population, 24,000-sample PPO budget, actors and checkpoint rule. It does not show that demand–capacity structure is useless in general; it also does not support a claim that structural requester roles, changed physics, hidden information, fees or new RL algorithms are needed. C1 and historical v1–v22 outcomes are not L1 evidence, and sealed TEST remains unopened. A further learning-signal or sample-budget hypothesis would require a new protocol and authorization, with the present no-go preserved.

## Reproduction map

From the repository root, the manifests contain exact original commands, source/protocol/config hashes, package versions and seed roles. Fresh reproductions use new timestamped directories:

```sh
uv run --no-sync pytest -q
uv run --no-sync python experiments/jcst_learner_campaign_v1.py --stage smoke
uv run --no-sync python experiments/jcst_learner_campaign_v1.py --stage train_dev
uv run --no-sync python experiments/jcst_learner_audit_v3.py NEW_TRAIN_DEV_DIR
uv run --no-sync python experiments/jcst_learner_campaign_v1.py --stage confirmation --train-dir NEW_TRAIN_DEV_DIR
uv run --no-sync python experiments/jcst_learner_audit_v3.py NEW_CONFIRMATION_DIR
uv run --no-sync python experiments/jcst_framework_stats_audit_v1.py NEW_CONFIRMATION_DIR
uv run --no-sync --with matplotlib==3.11.2 python experiments/jcst_framework_report_v1.py --train-dir NEW_TRAIN_DEV_DIR --confirmation-dir NEW_CONFIRMATION_DIR
```

The existing audited source runs should be read rather than re-audited in place, since the auditor writes its audit and selection files into the run directory. No branch, commit/push, SSH, rsync or remote launch occurred.
