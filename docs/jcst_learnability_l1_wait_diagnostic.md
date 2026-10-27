# Historical L1 DEV WAIT diagnosis for the L2 question

This is a read-only, post-L1 diagnosis of the **selected L1 DEV checkpoints**, not L2 scientific evidence and not a reanalysis of L1 confirmation for tuning. The new [decision-level artifact](../artifacts/jcst_learnability/l1-wait-diagnostic-20261026T021556.293476Z/decisions.csv) contains the reconstruction; the [summary](../artifacts/jcst_learnability/l1-wait-analysis-20261026T021646.598572Z/summary.csv) excludes one-legal-action states when averaging logit gaps. Each case/arm has five frozen training roots × 16 L1 DEV physical seeds × 48 decisions = 3,840 decisions. The script independently reconstructed state transitions and verified every selected greedy action against the frozen checkpoint's normalized categorical logits.

| Case / arm | Greedy WAIT | Forced WAIT | Discretionary WAIT | Mean WAIT probability | Mean WAIT probability when discretionary WAIT chosen | Entropy / log(legal actions) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base S | 73.9% | 9.6% | 64.3% | 20.3% | 13.5% | 97.8% |
| base F | 67.5% | 0.0% | 67.5% | 14.3% | 18.6% | 94.5% |
| base A | 64.4% | 20.0% | 44.4% | 29.2% | 16.4% | 99.4% |
| generalist S | 93.4% | 0.0% | 93.4% | 4.9% | 5.2% | 97.3% |
| generalist F | 90.0% | 0.0% | 90.0% | 8.5% | 8.8% | 90.3% |
| generalist A | 100% | 0.0% | 100% | 1.7% | 1.7% | 98.7% |

The generalist L1 greedy WAIT pattern is **not caused by no legal service choice** in these trajectories: forced WAIT is zero for all learned arms. Nor is it a distribution concentrated on WAIT: generalist S chooses WAIT on 93.4% of decisions while assigning it only 4.9% mean probability, and A chooses WAIT on every decision while assigning it 1.7%. High entropy relative to the legal catalogue and the large action count explain how a low-probability action can still rank first. Numerically near-tied top-two logits (gap ≤0.01) occurred on about 11.0% of all generalist S decisions, so near-ties contribute but cannot by themselves explain its 93.4% greedy WAIT frequency. On 83.1% of all generalist S decisions, the policy chose WAIT despite at least one legal action serving a failed component with a free compatible worker.

This evidence supports a **greedy-versus-distribution mismatch hypothesis**, without establishing that stochastic deployment is economically better. Sampling may start more service yet choose poor complete matchings. L2 evaluates greedy and stochastic actions on the *same fresh physical DEV seeds* with nested action seeds, and its cost outcome—not the historical WAIT frequency—will decide learnability. No L1 checkpoint, raw artifact, gate or verdict was modified.
