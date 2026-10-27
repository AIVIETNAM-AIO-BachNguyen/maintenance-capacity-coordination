# L2 independent-physical confirmation (before training replication)

Date: 2026-10-27 UTC. This is a predeclared confirmation of the L2 S/F/A bundle selected solely on DEV, not sealed TEST. The [preconfirmation lock](../artifacts/jcst_learnability/preconfirmation-lock-20261027T081035.520081Z.json) fixed checkpoints, deployment modes, 128 new physical seeds per case, four nested action seeds for stochastic deployment, 18 Bonferroni-adjusted primary intervals and all gates before this panel opened. The [raw confirmation manifest](../artifacts/jcst_learnability/confirmation-20261027T083357.339193Z/manifest.json) is COMPLETE.

The first frozen-source policy replay failed at a thread-sensitive near-tie because its process inherited eight Torch CPU threads, unlike the one-thread collector. The FAIL is preserved and the [auditor erratum](jcst_learnability_l2_confirmation_audit_erratum.md) was recorded before outcome analysis. Rerunning the **same frozen auditor source** with `OMP_NUM_THREADS=1` [PASSed](../artifacts/jcst_learnability/confirmation-audit-20261027T085605.819381Z/report.json): 15,360 learned and 1,024 control episodes physically and policy-action replayed. The independent [raw-row statistics audit](../artifacts/jcst_learnability/stats-audit-20261027T091448.274840Z/report.json) then PASSed all 18 contrast means, 4,000-draw crossed-bootstrap intervals and gate decisions. The audited [analysis](../artifacts/jcst_learnability/confirmation-20261027T083357.339193Z/analysis.json) is the numerical authority.

## Confirmed cost contrasts

Lower H+tail cost is better; negative contrast favors the first policy. Intervals are simultaneous family-adjusted 95% two-sided percentile intervals over five training roots and 128 paired physical seeds. Stochastic repetitions are averaged within each physical seed.

| Case | Selected S / F / A | threshold5 | S−24k | S−WAIT | S−F | S−A | S−threshold5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 731.93 / 735.27 / 735.46 | 859.01 | −257.54 [−328.41,−137.54] | −339.62 [−350.77,−328.31] | −3.34 [−17.65,+9.22] | −3.53 [−13.12,+6.02] | −127.08 [−148.97,−105.21] |
| generalist | 727.56 / 718.03 / 716.11 | 775.80 | −257.80 [−267.88,−249.24] | −330.28 [−338.67,−321.63] | +9.53 [+0.38,+18.69] | +11.45 [+4.16,+19.44] | −48.24 [−62.35,−33.44] |

The analogous F−24k/A−24k adjusted means are −360.53/−253.71 in base and −343.11/−339.80 in generalist; all their intervals are below zero. All S/F/A selected−WAIT adjusted intervals are also below zero. All five roots per arm/case individually improve by at least 10 over their 24k checkpoint on the paired confirmation panel. The prelocked useful-learning criterion passes for all six arm/cases; S passes the threshold5 competitiveness criterion in both cases. These findings establish physical-seed generalization for the *existing checkpoints*, not yet independent training-recipe replication.

The prelocked S−F representation and S−A cross-message gates fail in both cases. In generalist the selected S recipe costs more than selected F/A with adjusted intervals above zero. Generalist S was selected at 384k with stochastic deployment, while F/A were selected at 1.536m with greedy deployment. Per the pre-outcome interpretation amendment, those generalist differences are selected-recipe contrasts, not pure matched-budget actor/mechanism effects. In base the checkpoint matches but A's selected mode differs, and S−F has no adjusted gain. No distinct cross-message benefit is established.

## Training-root variation and next gate

Selected per-root confirmation means are base S 723.98/736.11/731.19/736.02/732.33, F 743.82/728.38/735.19/731.62/737.34, A 736.59/735.74/734.84/735.20/734.93; generalist S 728.54/725.87/728.31/728.22/726.88, F 718.54/712.57/714.31/722.70/722.02, A 714.26/715.19/712.74/716.88/721.51. These are five original training roots per arm/case, not new training replication.

The audited useful-learning result opens the **prelocked independent retraining gate**. All S/F/A arms in both cases are being trained from scratch on five new roots per arm/case at their frozen selected checkpoint, then evaluated on another 128 new physical seeds per case. The new run is `artifacts/jcst_learnability/replication-20261027T091525.639861Z/`; it remains subject to the original 72-hour campaign cap. No hyperparameter, mode, winner, physical contract or gate was changed after confirmation. If replication fails, report nonreplication without replacing the selected bundle or extending seeds.
