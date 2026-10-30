# V21 training-bank ranking fit diagnostic

Status: descriptive post-pilot check on **training data only**. V21 remains closed, no-go under `docs/reliability_weighted_pairwise_v21_findings.md`. This diagnostic neither changes selected checkpoints nor reopens held-out, replication or TEST.

`experiments/diagnose_v21_training_fit.py` evaluated the 18 pretrain and 18 DEV-selected C21/L21/D21 checkpoints on the immutable, independently audited v20 matching-label **training** bank (273 states in case a, 235 in b; 491 rows have eight candidates, 15 have six and two have seven). The script reads no held-out episodes. Artifact: `artifacts/v21_training_fit_20261025T045620646862Z`. Its manifest is `COMPLETED` with 36 unique rows; a separate completeness check verified source, label, train-audit and output hashes, all case/arm/root/stage combinations, finite metrics and nonnegative candidate gaps. These checks validate the recorded calculation's basic provenance and consistency, not out-of-bank prediction.

The pairwise loss is the v21 reliability-weighted surrogate; uniform candidate scores give the mean log-candidate-count reference. Values below are averages over the three training roots within each case. Candidate-score top choice is the maximum **complete canonical-sequence score among the training bank's eight candidates**, which may differ from deployed tokenwise greedy decoding.

| Case | Arm/stage | Pairwise loss | Listwise loss | Pair-sign agreement, all / gap ≥20 | Candidate top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| a | C21 pretrain | 2.001 | 2.041 | 0.635 / 0.795 | 0.300 |
| a | C21 selected | 2.008 | 2.048 | 0.633 / 0.787 | 0.304 |
| a | L21 selected | 2.023 | 2.053 | 0.618 / 0.758 | 0.287 |
| a | D21 selected | 2.020 | 2.058 | 0.623 / 0.764 | 0.283 |
| b | C21 pretrain | 2.005 | 2.044 | 0.628 / 0.730 | 0.278 |
| b | C21 selected | 2.010 | 2.048 | 0.628 / 0.724 | 0.294 |
| b | L21 selected | 2.022 | 2.052 | 0.615 / 0.701 | 0.287 |
| b | D21 selected | 2.020 | 2.054 | 0.621 / 0.723 | 0.275 |

Uniform loss references are 2.063/2.079 in cases a/b. C21 fits the training-bank pair comparisons slightly better than matched listwise L21 at selected checkpoints, especially larger cost gaps, but its pairwise loss rises from pretrain to DEV-selected PPO in both cases. The differences are small relative to the approximately two-unit uniform reference. Pair observations within a state are correlated; these rates have no independent uncertainty interval. The eight-candidate bank is not the full physical action set, and a branch continuation cost gap is not an episode-level causal gain.

This check narrows the interpretation of v21: the new surrogate was not completely ignored in-sample, yet its modest ranking advantage did not translate into the audited held-out economic gates. It cannot identify whether off-bank generalization, scarcity-relation encoding or alignment of continuation labels with the deployed greedy matching is the main limit. V20 already showed an information path from worker busy-time and loss of a sole worker into trained embeddings, while v16's engineering motif test showed its incidence-aware graph summary can distinguish a same-edge-count capacity-one/capacity-two construction. Neither establishes economically useful relation encoding on physical states. Any next pilot needs a separately justified mechanism and fresh prelock, rather than tuning v21 on its held-out results.
