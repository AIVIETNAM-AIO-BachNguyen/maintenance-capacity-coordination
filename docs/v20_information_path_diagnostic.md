# C20 trained-encoder information-path diagnostic

Status: descriptive engineering probe after the audited v20 no-go; **not** a new performance experiment, gate or checkpoint selection. Script: `experiments/diagnose_v20_information_path.py`. Output: `artifacts/v20_information_path_20261024T173053498042Z`. It reads only the independently audited v20 **training** label bank and DEV-selected C20 checkpoints. It does not read v20 held-out trajectories, costs or TEST. The script and input hashes are in the output summary; a separate JSONL aggregate check passed.

## Why this check

The earlier source audit (`docs/representation_path_audit_v15.md`) established that the v15 feasible-edge actor could receive technician busy times in raw input while its feasible-edge scoring path was invariant to their positive values. V16 introduced machine→technician→machine compatible-edge messages, including busy technicians, and tests showed an available path in an untrained actor and engineering motifs. The audited v16–v20 physical pilots did not establish a practical feedback gain. This check asks a narrower question of the **trained C20 weights**: do busy-time changes and a provisional assignment reach specific machine representations and actor scores on observed simulator training states?

## Method and observed counts

The probe uses all 508 audited v20 label states (273/235 by case) and their matching case/root DEV-selected C20 checkpoints. Torch uses one thread, matching the scientific runner. Each hypothetical intervention changes **only an observation tensor**; it does not advance or rerun the simulator. Therefore these interventions need not be physically self-consistent and cannot estimate H+tail cost.

| Probe | Result | Interpretation limit |
| --- | ---: | --- |
| Increase one positive busy-technician remaining-time feature by one normalized tick; compare valid root logits | 279/279 interventions changed some valid logit; median maximum absolute change 0.01693 | There is a nonzero path from busy time to score. |
| Same intervention; inspect machine embeddings by compatibility with that technician | Compatible machine embeddings changed in 279/279; maximum change among incompatible machine embeddings was exactly 0 | This perturbation travels through compatible edges to the relevant machine nodes in the observed states. |
| Root greedy token after the busy-time intervention | Changed in 1/279 | Logit sensitivity rarely changed the root greedy choice under this local perturbation. |
| Execute C20's greedy first provisional edge in the decoder, with no physical transition; inspect a different machine that thereby loses its sole currently feasible technician | 14 states had such a machine; its embedding changed in 14/14 | The selected-worker loss is reflected in that machine's updated representation on these training states. |
| Compare second-token logits under C20 dynamic refresh and the same weights with root context frozen | Valid logits differed in 508/508; greedy next token differed in 66/508 | Feedback changes scores and sometimes the next token in this selected state bank. These counts are not episode-gain estimates. |

All 508 bank states had a C20 greedy first edge and a feasible second edge. That is a property of the label-state selection and C20 behavior on this bank, not a population frequency. The 14 sole-worker-loss cases are a subset of this conditioned bank. The next-token comparison holds selected weights and the first edge fixed; it does not execute the alternative matching. The audited physical C20−CF20 result, **−7.37 H+tail** [−12.97,−2.05], remains the economic evidence and missed its locked ≤−10 practical gate.

## Boundary of the conclusion

The v16 graph mechanism and trained C20 weights do carry positive busy time and at least some technician substitution relationships into machine embeddings and logits. Thus the simple claim that C20 cannot represent either quantity at all is contradicted by this probe. It does **not** prove that the encoder preserves every competition pattern, that it estimates the economic value of alternatives, or that its updated scores are calibrated to long-horizon cost. The remaining uncertainty is how well these relationships are learned and used, especially when several machines compete for scarce technicians and 16-tape matching labels have limited reliability. A future scientific intervention needs its own question and prelock, fresh simulator training/DEV/held-out roles, matched frozen-context and strong controls, and a practical economic gate. No new TEST is authorized.
