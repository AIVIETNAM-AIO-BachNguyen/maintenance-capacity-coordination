# L2 TRAIN/DEV qualification (preconfirmation evidence)

Date: 2026-10-27 UTC. This is the sole active JCST/1.1 L2 campaign. L1 remains closed no-go; sealed TEST is unopened. The unchanged S/F/A PPO ladder trained five roots per arm/case at nested 24k, 96k, 384k and 1.536m controlled decisions. The [collector manifest](../artifacts/jcst_learnability/train_dev-20261026T020729.745460Z/manifest.json) is COMPLETE. Interrupted partials remain preserved under the recovery record and are excluded from the complete panel.

The [independent physical and policy replay](../artifacts/jcst_learnability/audit-20261027T075634.481426Z/report.json) PASSed: 960,000 TRAIN rows scanned, 300 fixed TRAIN traces, 9,600 DEV episodes and 128 control episodes replayed. The [optimization audit](../artifacts/jcst_learnability/optimization-audit-20261027T080746.557954Z/report.json) PASSed across all 38,400 PPO updates; sampled telemetry included 1,950 rows, with no zero actor or critic update. The first report attempt remained INCOMPLETE because matplotlib was absent; it was installed from the offline cache and the [fresh report](../artifacts/jcst_learnability/report-20261027T080840.450249Z/manifest.json) is COMPLETE. These are integrity and engineering results, not economic confirmation.

## Locked DEV selection and learning curve

The [independent selection](../artifacts/jcst_learnability/audit-20261027T075634.481426Z/selection.json) applies the prelocked lowest mean cost rule across four checkpoints and two deployment modes per arm/case. Costs below are equal-root, equal-physical DEV H+tail means at the *selected mode* for that arm/case. Lower is better. Checkpoints in the same row are nested in one training trajectory, not independent roots.

| Case | Arm | Mode | 24k | 96k | 384k | 1.536m | Selected checkpoint |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | S | greedy | 990.2 | 747.7 | 742.5 | 734.4 | 1.536m |
| base | F | greedy | 1096.9 | 818.5 | 740.3 | 730.4 | 1.536m |
| base | A | stochastic | 989.6 | 750.7 | 739.1 | 737.7 | 1.536m |
| generalist | S | stochastic | 987.3 | 743.3 | 725.4 | 725.6 | 384k |
| generalist | F | greedy | 1060.0 | 1053.5 | 730.1 | 716.8 | 1.536m |
| generalist | A | greedy | 1055.1 | 740.7 | 733.9 | 724.4 | 1.536m |

All six arm/cases meet the prelocked DEV eligibility rule: selected checkpoint at least 96k, at least 20 lower cost than 24k and WAIT, all five roots individually gain at least 10 over 24k, and all five roots average more than two service starts. The DEV threshold5 controls cost 838.94 (base) and 783.44 (generalist); WAIT controls cost 1075.56 and 1056.88. Selected DEV means are below threshold5 in both cases, but selection occurred on this DEV panel and supplies no confirmation claim.

The selected generalist S checkpoint/mode differs from F and A, so their selected-bundle cost differences are selected-recipe contrasts, not a pure representation or cross-message comparison. Matched-checkpoint/mode contrasts remain in `paired_arm_physical.csv` of the report. No representation or mechanism advantage is inferred from DEV selection.

## WAIT and decoding diagnosis

On the selected DEV panels, S averages 21.5 WAIT ticks in base (7.0 forced) and 22.1 in generalist (approximately zero forced), versus 36.1 and 30.8 at its 24k checkpoint in the selected modes. Generalist F and A each fall from about 43 WAIT ticks at 24k to 22.3 and 21.0 at selection, with approximately zero forced WAIT. Selected generalist S/F/A mean WAIT probabilities are 0.46/0.46/0.44, so L2 has learned active service behavior despite substantial probability mass on WAIT. Near-tie fractions at selection are 0.065/0/0.059. These are observed policy diagnostics, not proof of optimal service choice.

Paired stochastic-minus-greedy DEV cost at selected checkpoints is +1.72 (base S), +4.13 (base F), -2.57 (base A), -1.64 (generalist S), +7.42 (generalist F) and +0.40 (generalist A). These small descriptive paired means do not establish a general stochastic deployment advantage. Full per-root, per-physical, cost-component, optimization and compute results are retained in the [report directory](../artifacts/jcst_learnability/report-20261027T080840.450249Z/manifest.json).

## Next locked decision

The DEV gate is open for the entire fixed S/F/A two-case bundle. The [preconfirmation lock](../artifacts/jcst_learnability/preconfirmation-lock-20261027T081035.520081Z.json) freezes selected checkpoints/modes, source, protocol, analysis code and fresh seed roles. The concrete confirmation hypothesis is useful learning versus 24k and WAIT under the prelocked margins, with separate threshold5, S-F and S-A contrasts. No TRAIN/DEV intervention is justified before testing this qualified bundle. Confirmation and any conditional independent retraining must follow the original fixed gates; no model or mode will be reselected on confirmation.
