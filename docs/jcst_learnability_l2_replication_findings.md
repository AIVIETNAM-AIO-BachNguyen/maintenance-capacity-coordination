# JCST/1.1 L2 independent-training replication findings

Status: **COMPLETE, independent replay PASS, independent statistics audit PASS**, 2026-10-28 UTC. This is the five-new-root-per-arm/case replication opened by the prelocked useful-learning gate in the earlier [confirmation findings](jcst_learnability_l2_confirmation_findings.md). The [replication manifest](../artifacts/jcst_learnability/replication-20261027T091525.639861Z/manifest.json) contains 30/30 independently retrained trials, 60/60 selected/24k evaluation combinations and eight controls, with the frozen checkpoints and deployment modes in the [selection](../artifacts/jcst_learnability/audit-20261027T075634.481426Z/selection.json). The [physical/policy replay](../artifacts/jcst_learnability/confirmation-audit-20261028T121304.786186Z/report.json) and [raw-row statistics audit](../artifacts/jcst_learnability/stats-audit-20261028T123207.421198Z/report.json) both PASSed. The [audited analysis](../artifacts/jcst_learnability/replication-20261027T091525.639861Z/analysis.json) is the numerical authority.

All six arm/cases meet the prelocked useful-learning criterion again on new training roots and new physical evaluation seeds. S remains competitive with threshold5 in both cases. The prelocked S−F representation and S−A mechanism gates **do not pass** in either case. Thus the useful-learning result replicates, while a distinct selected-recipe S advantage does not emerge. An interval spanning zero does not establish equivalence.

| Case | Contrast | Mean H+tail difference | Family-adjusted interval | Gate |
| --- | --- | ---: | ---: | --- |
| Base | S−24k | −311.23 | [−342.58, −251.92] | Useful learning PASS |
| Base | S−F | +5.23 | [−7.75, +18.99] | Representation FAIL |
| Base | S−A | −0.75 | [−13.76, +11.56] | Mechanism FAIL |
| Base | S−threshold5 | −117.63 | [−140.15, −94.52] | Competitive PASS |
| Generalist | S−24k | −259.81 | [−268.02, −251.73] | Useful learning PASS |
| Generalist | S−F | −0.73 | [−23.23, +15.86] | Representation FAIL |
| Generalist | S−A | +7.28 | [−2.55, +16.64] | Mechanism FAIL |
| Generalist | S−threshold5 | −42.93 | [−57.84, −28.31] | Competitive PASS |

The complete audited analysis retains all **18** primary contrasts, including F/A gains over 24k and all learned arms versus WAIT, with 4,000 crossed-bootstrap draws and the prelocked family adjustment. The base selected S/F checkpoints are both 1.536m greedy, but A is 1.536m stochastic. In generalist S is 384k stochastic while F/A are 1.536m greedy. Consequently these are frozen selected-recipe comparisons, not same-budget or same-mode architecture contrasts; the separate [L3 findings](jcst_matched_architecture_l3_findings.md) answer the 96k matched-budget question.

The replay scanned **840,000 TRAIN rows**, physically replayed **285 fixed TRAIN traces**, physically and policy replayed **15,360 learned evaluation episodes**, and physically replayed **1,024 controls**. The independent statistics audit recomputed all 18 contrasts, intervals and gate arithmetic. The replication collection took 82,298 s (22.86 h), peaked at 869,224 KiB RSS, and finished before the original 72-hour L2 campaign cap; the full TRAIN/DEV-to-replication-audit interval was about 58.4 h. All failed/partial earlier attempts remain preserved. No model, checkpoint, mode, seed budget or simulator contract was changed because of replication outcomes; sealed TEST remains unopened.

The durable [postprocess status](../artifacts/jcst_learnability/postprocess-20261028T121255.248253Z/status.json) records the completed replay → analysis → statistics audit chain. To reproduce it from the repository root with the frozen sources and one Torch thread, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_learnability_confirmation_audit_l2.py artifacts/jcst_learnability/replication-20261027T091525.639861Z --train-dir artifacts/jcst_learnability/train_dev-20261026T020729.745460Z --selection artifacts/jcst_learnability/audit-20261027T075634.481426Z/selection.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_learnability_analysis_l2.py artifacts/jcst_learnability/replication-20261027T091525.639861Z --train-dir artifacts/jcst_learnability/train_dev-20261026T020729.745460Z --selection artifacts/jcst_learnability/audit-20261027T075634.481426Z/selection.json --physical-audit artifacts/jcst_learnability/confirmation-audit-20261028T121304.786186Z/report.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/jcst_learnability_stats_audit_l2.py artifacts/jcst_learnability/replication-20261027T091525.639861Z --selection artifacts/jcst_learnability/audit-20261027T075634.481426Z/selection.json
```

Those commands are the recorded sequence, not an invitation to overwrite the completed `analysis.json`; a fresh reproduction must use a fresh run directory because the analysis writes with exclusive creation.
