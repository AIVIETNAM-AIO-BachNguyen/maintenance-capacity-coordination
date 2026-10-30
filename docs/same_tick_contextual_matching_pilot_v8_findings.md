# Same-tick contextual matching v8 — audited pilot findings

Date: 2026-10-22 (Asia/Taipei). The proposal and locked protocol are `same_tick_contextual_matching_proposal_v8.md` and `same_tick_contextual_matching_pilot_v8.md` (SHA-256 `74d56ad194391be334bf876d3dbef21eeb1fc48620b21d02db3c682d1d5d644b`). Collection used pushed commit `df6cdd782eb19e7a8d9bba84fc50f932f0f2d6f2`. The N4/K2 simulator, fixed threshold PM admission, automatic CM, costs, failure tape, service physics and H/tail stayed unchanged. Within-tick decoder steps are provisional; the completed matching is executed once. No TEST panel was opened.

## Verification and coverage

The full repository suite passed **149 tests** before collection. The final bounded **Ubuntu** engineering smoke `artifacts/same_tick_v8_smoke_20261022T142448175514Z` and its independent replay audit `artifacts/same_tick_v8_audit_20261022T142459464474Z` passed; smoke is engineering validation only.

The locked training/DEV run `artifacts/same_tick_v8_train_20261022T143004651031Z` completed all 30 case × arm × root trials: 245,760 H training decisions, 5,100 complete training episodes, 81,600 tail ticks, and preserved partial episodes, and 1,440 DEV episodes. Wall time was 506.05 s. Independent audit `artifacts/same_tick_v8_audit_20261022T143843612434Z` **PASS** independently replayed every training decision and verified source/config/protocol/checkpoint identities, reward and H/tail accounting, physical seeds and DEV selection.

Only after that audit, `artifacts/same_tick_v8_heldout_20261022T143941233791Z` completed 960 selected learned-policy and 448 strong-control episodes on 32 fresh paired raw physical seeds per case; wall time was 111.47 s. Independent audit `artifacts/same_tick_v8_audit_20261022T144142398189Z` **PASS** replayed all 46,080 learned H decisions and 448 control episodes, with legal matching, exact outcomes, source/checkpoint hashes, physical pairing and gate arithmetic. F2 spent 657,984 additional forecast transitions on this panel. Each audit and run has its own timestamped artifact; failures/partials remain preserved.

C8/D8/B8 are ordered-sequence PPO policies. A sequence, including STOP where available, is the PPO random variable; a physical matching can have multiple orderings. Their recorded entropy is the sum of conditional **sequence** entropies on the sampled path, not matching entropy. Deterministic inference greedily chooses each token and need not yield the highest-probability physical matching. S8/G8 use one masked complete-matching categorical. All five scratch arms use equal H-decision budgets and the same physical action set; they do not use equal actor compute.

## Locked H+tail comparisons

Lower economic cost is better. Each contrast averages cases, three selected training roots and paired physical seeds equally. The 95% percentile intervals resample the **raw physical seed ID as a cluster** across cases, arms and roots (5,000 resamples); they condition on the three trained roots. Root means below show training variability separately.

| C8 minus control | Overall H+tail | Reciprocal | Specialist/generalist | Cluster 95% interval | Three training-root overall means |
| --- | ---: | ---: | ---: | --- | --- |
| **D8, frozen within-tick context** | **−6.18** | −0.78 | −11.58 | [−9.41, −3.00] | −9.38, −4.22, −4.95 |
| B8, direct-feature sequential | −2.31 | −15.52 | **+10.91** | [−10.23, +4.74] | −19.97, +1.30, +11.75 |
| S8, structured complete-action | +4.30 | −1.32 | +9.92 | [−0.74, +9.46] | +1.06, −4.09, +15.92 |
| G8, flat complete-action | −16.51 | −24.98 | −8.03 | [−26.49, −6.66] | −19.52, −18.77, −11.23 |

The locked independent-training-replication gate requires C8−D8 ≤−20 overall, both case means negative, at least two root means negative, interval upper endpoint <0, plus C8−B8 and C8−G8 overall negative. **Only the practical C8−D8 threshold failed, but it is mandatory.** The audited `replication_go_decision.json` is `go: false`. The modest C8−D8 conditional difference is insufficient for the predeclared useful feedback claim. Neither independent retraining replication nor TEST was run. C8's advantage over G8 cannot substitute for the primary C8−D8 test; B8 reverses C8's ordering in the specialist case.

| Arm | Reciprocal mean cost | Specialist/generalist mean cost | Role |
| --- | ---: | ---: | --- |
| C8 | 796.28 | 1009.67 | Full sequential contextual policy |
| D8 | 797.06 | 1021.25 | Same architecture/parameters, context frozen within tick |
| B8 | 811.80 | 998.76 | Direct-feature sequential control |
| S8 | 797.60 | 999.75 | Fresh structured complete-action PPO |
| G8 | 821.26 | 1017.70 | Fresh flat full-information PPO |
| Frozen V14 (five roots) | 794.88 | 996.26 | Historical pretrained reference |
| F1 risk matching | 830.59 | 1009.69 | Fixed rule reference |
| F2 six-tick rollout | 704.44 | 978.59 | Known-model compute-heavy reference |

V14/F1/F2 are contextual strong controls, not sample-matched scratch actors. F2 has the lowest listed mean cost in both cases and spends forecast compute. V14's historical training differs from the v8 budget. These comparisons limit practical claims without identifying the effect of within-tick feedback.

## Behavior and compute

C8's reciprocal H availability was 0.585, failures 5.14/episode, queued PM 16.27 ticks, queued CM 1.88 ticks, and service starts 50.14; corresponding D8 values were 0.584, 5.17, 16.08, 2.11 and 50.10. In specialist/generalist, C8 had availability 0.500, failures 9.28, queued PM 29.41, queued CM 8.47 and service starts 40.16; D8 had 0.503, 9.97, 30.29, 7.89 and 40.55. These descriptive aggregates do not establish which local assignment caused an episode's cost difference.

C8 selected 37.35 and 30.07 edges per 48-H-tick episode in reciprocal and specialist/generalist, respectively. Its two-start ticks averaged 8.48 and 2.80; zero-edge decisions averaged 19.12 and 20.73. C8/D8/B8/S8 selected **zero discretionary WAITs** on heldout in both cases; their zero-edge decisions were forced by feasibility. Thus this pilot gives no evidence that learned STOP after a feasible edge or at the root was useful. In the specialist case, generalist worker 0 served exclusive machine 0 for 9.15 starts and shared machines 1–3 for 13.07 starts per C8 episode. C8's sampled-path sequence entropy summed to 31.57 and 18.47 by case; it is not physical-matching entropy.

C8/D8 actors have 12,226 parameters, B8 11,410, S8 8,553 and G8 8,705; all critics have 7,873. Mean training trial wall times were 20.01/19.58/18.51/16.48/9.59 s for C8/D8/B8/S8/G8. Mean H-selection seconds per heldout episode in reciprocal were 0.063/0.062/0.061/0.043/0.021; specialist/generalist 0.057/0.056/0.054/0.042/0.021. F2 used 0.742 and 0.545 s and averaged 11,853 and 8,709 forecast transitions per episode, respectively. These timings are specific to this Ubuntu CPU run.

**Telemetry correction:** the raw field `decoder_forward_passes` for C8/D8/B8 increments only when a stochastic token choice has feasible edges; it does *not* count every `actor.logits` call. Source inspection gives one scorer call at root and one after every selected edge, including the last edge that forces termination. The correct per-episode count is therefore `48 + matching_cardinality_sum`: C8 means **85.35** and **78.07** scorer calls by case, versus 48 one-action scorings for S8/G8. The original field values (37.35 and 30.07 for C8) remain in immutable collection artifacts but must not be cited as scorer-call counts.

DEV learning-area means (trapezoid over 2,048–8,192 H decisions) were reciprocal C8/D8/B8/S8/G8 = 802.45/802.52/814.76/801.69/852.47, and specialist/generalist = 1005.23/1000.59/983.94/995.64/1007.48. These selection-panel summaries are descriptive; no architecture or gate was changed after seeing them.

## Post-pilot mechanism sensitivity and decision

After the no-go, descriptive artifact `artifacts/same_tick_v8_mechanism_20261022T145734194701Z` replays **C8-visited** heldout states and checks exact C8 episode costs against the audited rows. On 192 C8 episodes (9,216 H states), C8's selected checkpoint was evaluated again with **identical weights** but frozen within-tick context. There were 814 reciprocal and 269 specialist states with a feasible second-token choice after the first edge. The second-step probability distributions differed slightly (opportunity-weighted mean total variation 0.0060 and 0.0108), yet the greedy second token changed **0** times and the completed matching changed **0** times in both cases. Separately trained D8 checkpoints, evaluated on those same C8-visited states without executing their actions, differed from C8 on 16 reciprocal and 144 specialist final matchings (and 285/152 first tokens). Those differences may reflect training trajectories and are **not** same-weight feedback effects. The opportunity profile narrows why this tested feedback rarely matters. A further root-graph enumeration (`artifacts/same_tick_v8_mechanism_20261022T154546273317Z`) checked **every legal first edge**, independent of C8’s preferred first choice: only 15 reciprocal and 12 specialist C8-visited states had any first edge leaving at least two feasible second edges. Thus on this visited panel the scarcity is present in the root feasibility graph, not caused by C8 selecting a first edge that removes alternative second choices. This is still a post-pilot, policy-visited structural description; it does not estimate benefit in another simulated regime. After C8's first edge, 799/814 reciprocal and 257/269 specialist second-step opportunities had **exactly one** feasible remaining edge, leaving only that edge or STOP. Only **15 reciprocal and 12 specialist** states, 27/9,216 H states overall (0.29%), had multiple feasible second edges to rank. No second-edge ranking changed under same-weight context freezing in those 27 states. The mean top-token logit margin was 1.90/2.58 by case, whereas the mean largest absolute context-induced logit change was 0.102/0.080; this descriptive comparison does not itself bound all possible flips. The sampled C8 policy also never chose discretionary STOP on heldout. These frequencies are conditional on the selected C8 policy's visited states and first choices; they do not measure headroom for other policies or operating regimes.

This analysis has its own source/output hashes and completed manifest, but is post-pilot descriptive, not an independently prelocked causal gate or a D8 counterfactual rollout. A first indexing error in this optional profile was preserved as failed artifact `artifacts/same_tick_v8_mechanism_20261022T145711420091Z`; the corrected fresh run above completed and passed separate hash/aggregate checks. Neither analysis establishes episode gain from contextual updates.

Decision: retain v8 as a **no-go** for the proposed useful same-tick feedback contribution. The model and simulator code remain available as a tested research implementation; its hypothesized mechanism did not pass the locked practical comparison. Preserve the v1–v7 verdicts, v8 raw artifacts and reserved replication/TEST seed roles. Any later campaign needs independent evidence that partial-matching context changes economically relevant feasible choices, followed by a fresh prelock. Do not tune v8 on this heldout panel, lower its gate, or infer a validated two-policy framework from one coordinator actor.
