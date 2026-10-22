# Fixed-threshold requester / learned-coordinator diagnostic — findings

Date: 2026-10-22 (Asia/Taipei). This is a **post hoc attribution diagnostic** motivated by pilot v1's C-versus-D replication difference. Its question, G architecture, budget, seed roles, selection rule and analysis were locked in [`threshold_requester_diagnostic_v1.md`](threshold_requester_diagnostic_v1.md) before any G outcome. It does not revise pilot v1, its TEST gate, or its sealed TEST panel.

## Verification and scope

G keeps the inherited threshold requester fixed and trains only the masked coordinator plus centralized critic. It uses the same 17,046-parameter coordinator architecture and PPO settings as D, 8,192 H decisions per case/training seed, and the same three training seeds. The already-open pilot v1 DEV seeds select G checkpoints by the same rule as the v1 arms; DEV cannot provide independent evidence. A bounded Ubuntu smoke and focused fixed-requester/log-probability/future-tape/checkpoint test passed. The full suite passed 105 tests.

Training artifact: `artifacts/threshold_control_v1_train_20261022T063058659512Z`. Its audit passed all six trials (49,152 H training decisions), 288 DEV episodes, checkpoint selection, source/protocol hashes, feasibility and H+tail accounting. The independent diagnostic physical panel 73160000–73160031 was opened only afterward. Evaluation artifact: `artifacts/threshold_control_v1_evaluate_20261022T063211658343Z`. Its audit passed all 960 paired G/B/C/D/E episodes, including physical-tape and selected-checkpoint hashes, legal choices, cost decomposition, and separation from pilot v1 TEST seeds. No v1 code, protocol or checkpoint was altered.

## Same-panel cost

Mean H+tail economic cost, lower is better. G/B/C/D/E each use three training seeds × 32 physical seeds per case. A uses five historic training seeds × the same physical seeds. F1/F2 are deterministic rules on the same seeds. A/F1/F2 were added **after seeing G outcomes**, solely as a descriptive same-panel supplement; their 448 episodes passed a separate feasibility, cost and physical-seed audit.

| Arm | Reciprocal scale 2 | Specialist/generalist scale 2 |
| --- | ---: | ---: |
| G: fixed threshold + learned coordinator | 818.58 | 1007.02 |
| B: local learned requester + learned coordinator | 1168.18 | 1219.19 |
| C: capacity-aware learned requester + learned coordinator | 1128.01 | 1223.92 |
| D: always-request + learned coordinator | 1371.64 | 1346.99 |
| E: full-information centralized requester + learned coordinator | 1092.06 | 1219.63 |
| A: fixed threshold + frozen V14 count | 804.16 | 1004.23 |
| F1: fixed threshold + risk matching | 833.88 | 1010.19 |
| F2: fixed threshold + six-tick rollout | 722.63 | 970.25 |

The prelocked paired estimates, equally weighting cases and training seeds, are G−C **−263.16** (95% physical-seed bootstrap conditional on training seed [−274.37, −251.85]) and G−D **−446.51** ([−455.36, −437.48]). G−C is −309.43 in the reciprocal case and −216.90 in specialist/generalist; its training-seed differences are −271.56, −248.94 and −268.98. G−D is −553.05 and −339.97 by case; its training-seed differences are −504.27, −399.75 and −435.52. These intervals resample physical seeds (5,000 draws, seed 73161000) while conditioning on the three trained models; they do not establish generalization to arbitrary training seeds or environments. The diagnostic arm was chosen after seeing v1 results, so its contrast is exploratory even on a newly reserved physical panel.

## Mechanism and limits

G's reciprocal/specialist H availability is 0.59/0.50, with 6.88/8.98 H failures. C has 0.51/0.56 availability and 19.97/25.86 failures; D has only 0.08/0.16 availability despite 2.98/5.32 failures. G starts about 41.79/27.89 PM jobs per H+tail episode, C 26.01/5.69 and D 51.96/34.04. In these cases and at this budget, B/C under-request PM while D over-serves free PM requests. Fixed threshold gating plus a learned coordinator is much more effective than either learned requester variant. This supports the hypothesis that **request gating**, not learned capacity-aware requester decisions, drives the observed C–D difference. It does not prove the requester structurally unnecessary or that no learned requester can improve with a different justified training design.

G's same-panel cost is close to frozen V14 and risk matching; rollout remains lower in both cases and spends 621,408 forecast transitions across its 64 diagnostic episodes. These A/F comparisons were added after G outcomes and have different training or planning budgets. They are context, not a prelocked superiority claim. G has 17,046 trainable parameters, no forecast transitions and mean 8.43 s per training trial including DEV evaluation on CPU. The full per-episode secondary metrics and partials remain in the run artifacts.

## Research decision

The capacity-feedback requester has no demonstrated learning or economic advantage over the local requester, centralized learner, or the simpler fixed-threshold/learned-coordinator control in this pilot. The diagnostic gives no basis for expanding N/K, adding prediction/pricing/reward modules, or claiming a successful two-policy contribution. Preserve pilot v1's positive C−D gate and its negative B/E/strong-control results unchanged. Its sealed TEST still awaits the explicit authorization already requested; no TEST seed has been evaluated. A next formulation should be proposed only if it addresses a specific failure mode with a new prelocked comparison, rather than searching for a favorable scorer variant.
