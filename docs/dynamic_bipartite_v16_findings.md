# V16 dynamic bipartite encoder: audited findings

Status: **closed, replication gate false**. This is the one prelocked v16 pilot in `dynamic_bipartite_v16_prelock.md`. The simulator produced every training label and trajectory. No operational logs, independent retraining or new TEST were used.

## Question and provenance

The v15 path audit showed a difference between a value being present in raw tensors and reaching the actor score: positive busy-technician remaining times and unrequested-machine features were often screened out by feasible-edge pooling. A separate structural concern was that edge mean/max/count did not explicitly encode which machines competed for the same technician. These were hypotheses about usefulness, not demonstrations that the old policy conflated particular real states. The four primary precedents and their scope are in `representation_path_audit_v15.md`.

V16 tested one machine→technician→machine sum-message pass over all compatible pairs. C16 recomputed node and candidate context after each provisional pair. D16 used identical modules, input and root logits but froze within-tick scoring context; CF16 evaluated each DEV-selected C16 checkpoint with D16 behavior. I16 pooled all entity input without incidence messages; R16 reused the old feasible-edge actor; B16 was direct sequential; S16/G16 were complete-matching controls; P16 omitted supervised labels; Q16 used always-request admission. All training data and physical actions followed the locked simulator contract.

Collection commit `5cac0dff315b6881861ef495a4f44a034a50f121`; protocol SHA-256 `e068e3aa550dbd24bf8fd6f4485f7495680ac2cee47074da61ff98e12d38adbe`. Full suite **215 passed**. The bounded Ubuntu engineering smoke and its independent audit passed before science. The audited v15 label bank supplied 457 selected training states and 3,832 continuation branches to the seven pretraining arms; it did not supply DEV or held-out outcomes.

| Artifact | Content and replay verdict |
| --- | --- |
| `artifacts/same_tick_v16_train_20261023T055022317165Z` | 54 trials, 442,368 H decisions, 2,592 DEV episodes; independent audit `artifacts/same_tick_v16_audit_20261023T062112225957Z` **PASS**. |
| `artifacts/same_tick_v16_heldout_20261023T062526347509Z` | 1,920 learned/CF and 128 F1/F2 episodes on 32 fresh paired raw seeds per case; independent audit `artifacts/same_tick_v16_audit_20261023T063023558789Z` **PASS**, replaying 92,160 learned/CF decisions and 6,144 control decisions. |
| `artifacts/same_tick_v16_mechanism_20261023T063802565715Z` | Post-pilot, same-state/same-weight descriptive replay on C16-visited H states. It rechecked each C16 episode cost against audited held-out rows; it did not change the gate or physically execute the frozen choice. |

Checkpoint selection used only the locked 2,048/4,096/8,192-decision DEV evaluations. H is 48 ticks plus 16-tick tail; lower H+tail economic cost is better. The following estimates are C16 minus each arm, averaged equally over both cases, three training roots and paired held-out seeds. Intervals resample the 32 raw physical seeds as clusters across cases and roots. A negative number favors C16.

| Comparison | Δ H+tail cost | Physical-seed-cluster 95% interval | Locked gate interpretation |
| --- | ---: | ---: | --- |
| C16−D16 | −11.55 | [−19.26, −4.20] | Both case means and two of three root means were negative, but missed ≤−20. Root means were −40.19, +13.36, −7.81. |
| C16−CF16 | −2.26 | [−6.47, +1.64] | Missed ≤−10 and interval upper <0; the direct same-weight feedback evidence is inconclusive. |
| C16−I16 | +8.16 | [−0.51, +16.27] | Missed ≤−10 and reversed sign across cases. I16 has fewer parameters, so this is an architecture package comparison. |
| C16−R16 | +16.48 | [+9.67, +23.70] | Old feasible-edge actor was better in both cases; missed ≤−10. |
| C16−B16 | +9.70 | [+3.37, +15.85] | Direct sequential control was better. |
| C16−S16 | +21.11 | [+15.75, +26.68] | Structured complete-matching control was better. |
| C16−G16 | −41.71 | [−50.55, −34.03] | C16 beat flat full-information PPO, insufficient for a feedback claim. |

Other secondary contrasts: C16−P16 −15.35 [−24.38, −7.23]; C16−Q16 −178.60 [−186.48, −170.89]. Against risk matching F1, C16−F1 was −33.80 [−43.70, −23.95]. Against six-tick rollout F2, C16−F2 was **+19.02** [+9.27, +28.86], with opposite case signs (−12.18, +50.21). F2 used 1,092,384 forecast transitions across its 64 held-out episodes. The F1/F2 intervals used the same raw-seed-cluster bootstrap rule as the gate but are secondary calculations after audit.

Mean absolute H+tail costs were C16 1,596.72, D16 1,608.27, CF16 1,598.98, I16 1,588.56, R16 1,580.23, B16 1,587.02, S16 1,575.60, G16 1,638.43 and F2 1,577.70. C16's H availability was 0.325, failures 17.33, queued CM ticks 37.71, queued PM ticks 29.02 and service starts 52.34 per episode; corresponding D16 values were 0.350, 20.19, 26.30, 33.33 and 53.91. These secondary metrics can move in different directions and do not replace H+tail cost.

**Gate verdict:** `replication_go_decision.json` has `go: false`. The primary C−D practical threshold and same-weight C−CF threshold failed, as did the C−I/R/B requirements. Independent retraining and TEST remain closed under the prelock. V1–v15 verdicts are unchanged.

## What the feedback changed

On the 9,216 C16-visited H states, only 1,628 had an eligible second token after C16's first edge; 512 had more than one eligible second edge. Freezing the same C16 checkpoint changed the second token in 73 states and the final greedy matching in **76/9,216 (0.82%)**. The mean second-step total-variation distance over the 1,628 opportunities was 0.058. Separately trained C16 and D16 chose different final matchings in 2,937/9,216 states, but their weights differ, so that number does not isolate inference-time feedback. The C16 versus CF16 paired *physical* rollouts, above, are the relevant economic comparison. The descriptive replay executes only the C16 matching and exactly matches its audited episode cost.

Unit and source-path tests verify that busy-time and unrequested-machine changes can reach C16's actor logits, that same-count star/split engineering motifs are distinguishable, and that C16/D16 have identical root logits but can diverge after a provisional choice. These checks show an available representation path, not that the trained policy used each relation optimally. Conversely, the v15 mean/max/count actor also has candidate-specific counts and static compatibility columns, so this pilot does not prove it is blind to all alternative structure.

## Compute and limits

C16 and D16 each had 22,178 actor and 9,473 critic parameters; I16 had 15,394 actor parameters, R16 13,858, B16 13,498, S16 8,649 and G16 16,880. Seven supervised arms each received 64 pretraining epochs per trial; P16/Q16 received none. All nine arms had six trials of 8,192 H training decisions. The complete train/DEV run took 1,839 seconds CPU wall time; mean trial wall time was C16 49.7 s, D16 35.5 s, I16 39.8 s, R16 36.1 s, B16 34.6 s, S16 42.3 s and G16 10.8 s. Held-out mean policy-selection time per episode was C16 0.135 s, D16 0.097 s, I16 0.102 s, R16 0.091 s, S16 0.060 s, G16 0.024 s and F2 1.287 s on this CPU environment.

The logged `decoder_forward_passes` field counts only non-forced choices; it undercounts scorer calls. For greedy canonical sequences, C16 made **16,549** actor scorer/graph-refresh calls over 9,216 held-out H decisions (`48 + matching_cardinality_sum` per episode). D16 made 16,843 scorer calls but only recomputed its graph at the root of each tick. Flat/structured policies make one complete-matching score call per tick. Wall times are environment-specific, not a formal complexity benchmark.

The result supports a narrow observation: relation-aware input repair and graph feedback are implementable without changing simulator physics, but **this trained v16 did not deliver a practically useful coordination gain under the locked contract**. The remaining uncertainty is whether economically binding alternative assignments are too scarce on these C16-visited states, whether the current one-pass encoder/label bank fails to extract their long-horizon value, or both. A new module or changed demand regime would need its own justified, versioned prelock and controls; this negative held-out result cannot be used to lower v16 gates or tune on TEST.
