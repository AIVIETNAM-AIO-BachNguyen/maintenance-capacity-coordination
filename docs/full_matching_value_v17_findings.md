# V17 complete-matching value coverage: audited findings

Status: **closed, replication gate false**. This is the separately prelocked v17 experiment in `full_matching_value_v17_prelock.md`. All labels, PPO trajectories and evaluations came from the unchanged simulator; operational logs and public datasets were not needed. No independent retraining or new TEST ran.

## Question and provenance

V15 showed that numbers present in raw input need not reach feasible-edge actor logits. V16 repaired that path with compatible-pair graph messages but missed both its practical C−D and same-weight C−CF gates. V17 held the v16 encoder, PPO and physical contract fixed, and changed the supervised target to the value of **complete legal matchings**, including WAIT, singleton STOP and three-pair choices throughout H. C17 refreshed graph messages after a provisional pair; D17 used the same architecture but froze its within-tick scoring context; CF17 evaluated each selected C17 checkpoint with that frozen behavior. I/R/B/S/G, old-label O17, PPO-only P17, always-request Q17 and F1/F2 supplied the locked comparisons. C−O contrasts training packages, since the label bank, epochs and compute also differ.

The collection source was commit `f4b9ffc3ba9da40ad120cdf73cf13e81140d21ef`; locked protocol SHA-256 `a34eff4aa509102656b21f55f02cdecf71cf4af3dccea4e84ab7d39831a29bf8`. The 218-test suite passed. Ubuntu label and PPO engineering smokes passed their independent replay audits before scientific collection. Physical horizon was H48 plus tail16, with the same two fixed N5/K3 configurations, threshold PM, automatic CM, failure/service/cost laws and legal physical matchings as v16. Decoder tokens were provisional; only the completed matching advanced the simulator.

| Artifact | Content and independent verdict |
| --- | --- |
| `artifacts/full_matching_v17_labels_20261023T080338842451Z` | 508 training states (273/235 by case), 16,128 four-tape continuation branches and sufficient coverage; `artifacts/full_matching_v17_audit_20261023T081541174639Z` **PASS**. |
| `artifacts/same_tick_v17_train_20261023T144446781419Z` | 60 trials, 491,520 H training decisions and 2,880 DEV episodes; `artifacts/same_tick_v17_audit_20261023T152825638675Z` **PASS**. |
| `artifacts/same_tick_v17_heldout_20261023T154122079855Z` | 2,112 learned/CF and 128 F1/F2 episodes on 32 fresh paired raw seeds per case; `artifacts/same_tick_v17_audit_20261023T154644545364Z` **PASS**, replaying 101,376 learned/CF and 6,144 control decisions. |
| `artifacts/same_tick_v17_mechanism_20261023T155332059598Z`; `artifacts/v17_label_fit_20261023T155711937170Z` | Post-pilot descriptive same-weight action replay and in-sample training-label fit. Both verify audit/source hashes and preserve fresh timestamped outputs; neither changes the gate. |

The first scientific train attempt (`artifacts/same_tick_v17_train_20261023T082916366007Z`) stopped after three trials when its session vanished. Its partials and interruption record are preserved; it supplied no DEV selection or scientific result. The fresh completed train above started from scratch under the locked source and seeds.

## Held-out physical result

The estimates below are C17 minus each arm's H+tail cost, averaged equally over cases, three training roots and paired held-out physical seeds. Lower is better; negative favors C17. The 95% intervals resample the 32 raw physical seeds as clusters across cases and roots. DEV chose checkpoints only at the locked 2,048/4,096/8,192-decision milestones.

| Comparison | Δ H+tail cost | Cluster 95% interval | Interpretation |
| --- | ---: | ---: | --- |
| C17−D17 | **+8.17** | [+0.28,+15.95] | Missed ≤−20; both cases were worse (+8.19/+8.15). Root means were +0.22/−1.33/+25.61. |
| C17−CF17 | **−1.01** | [−5.27,+3.40] | Missed ≤−10 and upper <0; case means −0.33/−1.69. Same-weight physical feedback benefit is not established. |
| C17−I17 | +4.98 | [−3.62,+14.20] | Missed ≤−10; case signs differed (−4.54/+14.51). |
| C17−R17 | +23.44 | [+16.53,+29.67] | Old feasible-edge actor did better in both cases. |
| C17−B17 | +8.90 | [+0.68,+17.37] | Direct sequential actor did better in both cases. |
| C17−S17 | +27.31 | [+19.43,+35.38] | Structured complete-matching control did better in both cases. |
| C17−G17 | +6.77 | [−1.60,+14.77] | Flat full-information PPO did better overall; case signs differed. |
| C17−O17 | −31.28 | [−40.95,−21.54] | New-label training package did better overall, but case signs differed (+7.82/−70.39); this is not pure label-coverage attribution. |

C17−P17 was −13.00 [−21.17,−5.05] and C17−Q17 −178.55 [−187.80,−169.09]. C17−F1 was −38.06 [−51.77,−23.73], whereas C17−F2 was **+8.75** [−2.38,+19.25], with case means −5.46/+22.96. These are secondary contrasts, not substitutions for the feedback gates. Mean absolute H+tail costs were C17 1,599.86, D17 1,591.69, CF17 1,600.87, I17 1,594.88, R17 1,576.42, B17 1,590.96, S17 1,572.55, G17 1,593.09, O17 1,631.14, P17 1,612.86, Q17 1,778.41, F1 1,637.92 and F2 1,591.11.

C17's mean H availability was 0.336; it had 18.49 failures, 31.79 queued CM ticks, 30.65 queued PM ticks and 52.76 service starts per episode. These metrics help describe outcomes but the economic objective is H+tail cost. The independent `replication_go_decision.json` records `go: false`: the primary C−D and C−CF practical gates failed, as did the C−I/R/B/S/G conditions and O17's both-case condition. Under the prelock this closes v17 without independent retraining or TEST.

## Did the representation and labels affect decisions?

The earlier source-path audit showed a real distinction between raw availability and actor access: busy-time magnitude existed in v15 input but could not reach its feasible-edge score. V16/v17 graph paths allow those values and unrequested-machine features to affect logits, and engineering star/split motifs are distinguishable. That establishes an information path, not that the trained policy preserved or valued the relations in economically useful decisions. The v15 audit did not prove an actual mean/max/count collision of learned embeddings.

On 9,216 C17-visited H states, 2,082 had an eligible second token after C17's first edge, including 689 with multiple eligible second edges. Holding each C17 checkpoint's weights fixed but freezing within-tick graph updates changed 132 second tokens and **137/9,216 final greedy matchings (1.49%)**. Mean second-step total-variation distance over eligible opportunities was 0.068. Separately trained C17 and D17 differed on 2,462/9,216 final matchings, but their different weights prevent attributing that count to inference-time feedback. The descriptive replay re-executed only C17 choices and matched every audited C17 episode cost; the C17−CF17 paired physical episodes provide the economic comparison.

The fresh bank covers WAIT, singleton, two-pair and three-pair candidate matchings (508 WAIT candidates, 1,580 singleton, 1,679 two-pair, 265 three-pair); it includes 270/238 selected states in the first/second H halves. Coverage passed, but coverage alone did not yield a useful policy. A descriptive fit check on this **training** bank found C17's mean candidate-list cross-entropy 2.074 immediately after pretraining and 2.641 at the DEV-selected PPO checkpoint; top-target match rates were 0.213 and 0.195. The uniform candidate baseline was 2.071 cross-entropy and target entropy 1.749. D17 fit was 2.088/2.647; the flat G17 arm fit 2.010/2.194. Thus C17's pretraining did not outperform a uniform candidate prediction on its own training labels by this metric, and selected PPO checkpoints fit these labels worse. This is an in-sample diagnostic on a listwise surrogate. It does not identify whether optimization, target noise, continuation mismatch, representation, or policy selection caused the physical no-go, and it does not estimate generalization. P17/Q17 had no supervised pretraining, so their “pretrain” columns are random initialization rather than trained fits.

## Compute and boundary

C/D/O/P had 22,178 actor and 9,473 critic parameters; I 15,394, R 13,858, B 13,498, S 8,649 and G/Q 16,880 actor parameters. New-label arms each had 272 pretraining updates per trial; O17 had 256 old-label updates; P/Q had none. Every scientific arm then saw six PPO trials of 8,192 H decisions. The completed train/DEV run took 2,586 seconds CPU wall time; held-out took 311 seconds. Mean C17 trial time was 72.4 s, D17 42.0 s, S17 35.2 s and G17 10.8 s. Held-out selection time per episode was C17 0.138 s, D17 0.099 s, S17 0.061 s, G17 0.024 s and F2 1.339 s. F2 used 1,106,304 simulated forecast transitions over 64 episodes. The logged choice-step `decoder_forward_passes` undercounts actual actor scorer calls: C17 made 16,633 scorer/graph-refresh calls across 9,216 H states, while D17 made 16,600 scorer calls but recomputed graph context only once per tick. These times are environment-specific.

The narrow conclusion is that v17 repaired label **coverage** while preserving a path for relational information, yet the learned dynamic-feedback framework still did not meet the locked practical or same-weight economic thresholds. A next research step needs a distinct prelocked uncertainty: whether these simulator continuation labels offer a learnable ranking to the actor under controlled fitting, and whether improving such fit transfers to physical decisions. Do not infer an architecture cure from these held-out outcomes, revise v17 gates, use TEST for design, or require real-world logs for RL training. The research objective remains unproven.
