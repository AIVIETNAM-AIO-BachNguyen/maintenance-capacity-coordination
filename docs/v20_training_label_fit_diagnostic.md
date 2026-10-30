# V20 training-label ranking diagnostic

Status: post-v20 descriptive analysis of **training-only** simulator labels, not a new checkpoint-selection rule or performance experiment. Script: `experiments/diagnose_v20_label_fit.py`; final artifact: `artifacts/v20_label_fit_20261024T173522344667Z`. The script loads the independently audited v20 label bank, C20/D20 pretrain and fresh-DEV-selected checkpoints, and v20-DEV-reselected C19/D19 controls. It does not read physical held-out results or TEST. All 36 checkpoint/stage rows are saved; a separate row-to-summary arithmetic check passed.

## In-sample ranking fit

Cross-entropy is measured on the same matching-value bank used to pretrain/rehearse C20 and D20. Lower is better. The target entropy and uniform-candidate cross-entropy are references for that **training bank**, not generalization bounds.

| Case | Target entropy | Uniform | C20 pretrain → selected | D20 pretrain → selected | C19 selected | D19 selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | 1.930 | 2.063 | 2.038 → **2.053** | 2.046 → **2.051** | 2.062 | 2.064 |
| b | 1.870 | 2.079 | 2.045 → **2.052** | 2.050 → **2.066** | 2.069 | 2.074 |

Selected C20 fits the new bank slightly better than reselected old-bank C19, but remains near uniform: its cross-entropy improves on uniform by only 0.010/0.028 in cases a/b, versus 0.132/0.210 possible between uniform and target entropy. PPO increased C20's in-sample loss from its pretrain checkpoint in both cases despite fixed rehearsal. This narrows the training-path question but does not show why physical C20 cost worsened. Top-candidate agreement at selected C20 checkpoints was 0.283/0.299 (case a/b); it is also in-sample and cannot estimate out-of-bank ranking.

## Exploratory future-tape sign stability

For each candidate pair, the absolute mean-cost gap on the **first eight** simulator continuation tapes determines its bin. Agreement asks whether the pair's cost sign repeats on the **other eight** tapes. This avoids choosing bins from the same tapes used for the agreement check. Exact zero signs count as nonagreement. Pair comparisons within a state are correlated; these bins have no inferential confidence interval and are not a new qualification gate.

| First-half absolute cost gap | Case-a pairs / sign agreement | Case-b pairs / sign agreement |
| --- | ---: | ---: |
| [0,10) | 3,008 / 0.549 | 2,193 / 0.544 |
| [10,20) | 2,226 / 0.650 | 1,751 / 0.660 |
| [20,40) | 1,864 / 0.748 | 1,989 / 0.758 |
| ≥40 | 337 / 0.864 | 647 / 0.861 |

Large first-half gaps carry more reproducible sign information, while many small-gap comparisons are weak. These observations could motivate a separately prelocked uncertainty-weighted learning-signal study, but they do not establish that emphasizing large gaps would improve H+tail cost. The current actor has an effective information path for busy times and provisional substitution (`docs/v20_information_path_diagnostic.md`), yet its selected in-sample matching-ranking fit remains weak. Whether that reflects optimization, noisy or economically misaligned targets, or limits of graph representation is unresolved. V20's audited no-go and no-TEST decision are unchanged.
