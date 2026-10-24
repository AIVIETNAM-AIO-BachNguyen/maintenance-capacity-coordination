# Current state

Date: 2026-10-25 (Asia/Taipei). **V20 is closed, no-go; no experiment is currently prelocked.** The goal of a useful learned maintenance-demand/technician-capacity coordination framework remains unproven. All RL training data come from the simulator; real logs and public datasets are optional later calibration.

## Current research boundary

The simulator's failure law, restoration, heterogeneous skills, non-preemptive service, PM/CM lifecycle, threshold PM admission, H48+tail16 cost and complete matching/WAIT action set remain unchanged. The canonical decoder builds a provisional matching within a tick; the simulator advances and charges cost once after matching completion. Sequential decoding, STOP and graph message passing are prior art. The contribution at issue is whether refreshing a representation of machine demand and technician capacity after a provisional pair improves physical decisions over matched frozen-context and strong full-information controls. No two-policy requester/coordinator claim has been established.

## Latest audited evidence

V20's protocol is `docs/on_policy_matching_value_v20_prelock.md`; its full report is `docs/on_policy_matching_value_v20_findings.md`. The clean, pushed collection source was `0645bad8d855265f63c2baf5af2a3f67a702840a`. The final 231-test suite and bounded Ubuntu label/PPO engineering smokes passed. The scientific label bank contains 508 C19/D19-visited training states and 64,512 forced matching branches; its independent audit passed both case reliability gates. Train/DEV completed 12 C20/D20 trials and 98,304 H decisions, with independent audit PASS. Fresh paired held-out completed 1,920 learned/CF episodes and 128 F1/F2 control episodes; independent replay of 92,160 learned/CF and 6,144 control H decisions passed before cost interpretation.

On held-out H+tail cost (negative favors C20):

| Contrast | Pooled difference | Physical-cluster 95% interval | Locked decision |
| --- | ---: | ---: | --- |
| C20−C19, recent-policy versus old label bank | +9.57 | [+1.94,+17.57] | Fail: worse, required ≤−10. |
| C20−D20, separately trained dynamic versus frozen context | +18.68 | [+11.48,+25.89] | Fail: worse, required ≤−20. |
| C20−CF20, same weights with feedback versus frozen context | −7.37 | [−12.97,−2.05] | Fail on practical magnitude, required ≤−10. |

C20−CF20 is negative in both cases and its interval is below zero, evidence of a modest same-weight benefit for these weights and this panel. It does not override the failed practical gate or C20's losses to C19/D20. The audited `replication_go_decision.json` is `go: false`. No independent retraining or TEST ran. The combined new bank does not separate state coverage from candidate coverage. Label economic alignment, generalization and the graph actor's treatment of scarce machine–technician alternatives remain unresolved; the current data do not identify which dominates. A post-v20 training-bank-only information-path probe (`docs/v20_information_path_diagnostic.md`) found busy-time changes reached valid C20 logits in 279/279 local observation interventions and compatible machine embeddings in 279/279; a machine losing its sole feasible worker after a provisional edge changed embedding in 14/14 observed instances. Dynamic versus same-weight frozen second-token logits differed in 508/508 bank states and greedy next tokens in 66/508. These conditioned descriptive checks show a functioning information path, not economic use or a population rate.

## Preserved prior verdicts

- V1–v7 remain closed. V1 beat always-request on its sealed TEST but did not establish stable superiority over strong same-information controls. V5 found conditional representation headroom, not realized gain. V6/V7 did not qualify a useful framework effect.
- V8–v12 same-tick feedback pilots did not pass joint dynamic-versus-frozen and practical cost gates. V13 independently validated a conditional second-edge H+tail value gap, not a learned policy gain.
- V14 canonical matching PPO, v15 simulator-label pretraining, v16 dynamic bipartite representation and v17 complete-matching value coverage all missed locked practical feedback gates. See their respective findings in `docs/`.
- V18 improved label precision but C18−D18 was −9.18 and same-weight C18−CF18 −0.27; no-go. V19 retained training-bank labels during PPO but C19−C18 was +1.79, C19−D19 −15.21 and C19−CF19 −3.30; no-go. Their full reports are `docs/matching_value_precision_v18_findings.md` and `docs/matching_value_retention_v19_findings.md`.

Follow `docs/experiment_workflow.md`: separately prelock any new scientific experiment, test and run bounded Ubuntu smoke, audit before outcome interpretation, preserve failed and partial artifacts, and commit/push only relevant verified changes. The user's explicit lab-run authorization applies here, but no new TEST, SSH/rsync or destructive Git action is authorized.
