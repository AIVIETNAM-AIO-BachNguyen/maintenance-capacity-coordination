# V15 conditional-value pretraining: audited findings

Status: **closed, no-go** on 2026-10-23. The prelock is `conditional_value_pretraining_v15_prelock.md`; its gates were not changed after outcomes. This is a simulator-only N5/K3 research panel, not a TEST result. Source commit: `1436a13a25c2359e7ed584c85f611c6e6c7a4ee2`; protocol SHA-256: `80fbdad6d80cd7cbdb11f64464fad4613803466831b9b45cc08f394d8b840314`.

## Execution and audit

- The independent label audit passed for `artifacts/conditional_value_v15_labels_20261023T032931104204Z`: 576 source episodes, 457 selected states (268/189 by case), and 3,832 forced branches. These are training labels, not evaluation outcomes.
- `artifacts/same_tick_v15_train_20261023T045702062834Z` completed 42 trials, 344,064 H training decisions and 2,016 DEV episodes. Its independent pretraining/training/DEV replay audit passed in `artifacts/same_tick_v15_audit_20261023T051703827186Z`.
- `artifacts/same_tick_v15_heldout_20261023T052206566840Z` completed 1,536 learned/CF and 128 F1/F2 episodes on the locked fresh paired physical panel. `artifacts/same_tick_v15_audit_20261023T052544231412Z` replayed 73,728 learned decisions and 6,144 control decisions and passed. The audited decision is in the held-out artifact's `replication_go_decision.json`.
- The final source had 208 passing tests; both bounded Ubuntu engineering smokes and their separate audits passed. The scientific results below were interpreted only after the independent held-out audit.

## Prelocked primary contrasts

All contrasts are C15 minus comparator H+tail economic cost; negative favors C15. Intervals are 95% bootstrap intervals clustered by raw physical seed across cases and training roots.

| Contrast | Overall | Cluster interval | Locked requirement | Result |
| --- | ---: | ---: | --- | --- |
| C15−D15 | +2.39 | [−4.63,+9.66] | ≤−20, both cases negative, ≥2 roots negative, interval upper <0 | Fail |
| C15−CF15, same weights | +1.42 | [−2.05,+5.38] | ≤−10, both cases negative, interval upper <0 | Fail |
| C15−P15, PPO-only | −10.03 | [−18.42,−1.13] | ≤−10 and both cases negative | Fail: case b +1.03 |
| C15−B15, direct | −1.13 | [−7.32,+5.30] | Overall negative | Pass alone |
| C15−G15, flat | −30.81 | [−39.48,−22.26] | Overall negative | Pass alone |

C15−D15 is −5.82 in case a and +10.59 in case b; C15−CF15 is −0.48 and +3.31. Two of three C−D root means are negative, but the overall cost and case/interval requirements fail. The joint replication gate is **false**. No independent retraining and no TEST were opened.

The conditional labels supplied a train-time economic signal and C15 beat flat G15 on this panel. The result does **not** establish a useful within-tick feedback effect: freezing C15's own context does not worsen held-out cost, and separately trained D15 is not worse overall. C15−P15 is case-dependent, so the pretraining contribution also fails its locked joint gate. The label is a C12-continuation estimate for exact two-edge branches, not an optimal Q function; root choice, STOP and third assignments remained PPO problems. These are possible explanations, not verified causes of the no-go result.

The separate source and representation audit in `representation_path_audit_v15.md` identifies an additional, verifiable actor information-path defect. It motivates a distinct successor hypothesis without changing this verdict or the old v1–v14 gates. Simulator trajectories remain sufficient for RL; operational logs are optional calibration. No public FJSP dataset or real maintenance log is needed to train this simulator policy.
