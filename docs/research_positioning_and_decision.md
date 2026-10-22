# Research positioning and next decision

Date: 2026-10-22 (Asia/Taipei). This is a post-pilot research decision, not a new experiment or an amendment to any locked gate.

## Position against primary work

| Work | Decision and resource mechanism | Implication here |
| --- | --- | --- |
| [Ruiz Rodríguez et al. (2022)](https://orbilu.uni.lu/bitstream/10993/54345/1/1-s2.0-S0736584522000928-main.pdf) | Each machine agent chooses a technician/component maintenance action or delay under partial observation. Technician skills and assignment conflicts matter. Their evaluation compares learned behavior with corrective, preventive and random policies. | Machine-level maintenance actions, heterogeneous technician allocation and MARL are prior work. Our free-request contract, full-information controls and economic comparisons differ; these are evaluation conditions, not an established novelty claim. |
| [Wang and Gombolay, AirME/HetGPO (ICAPS 2022)](https://cdn.aaai.org/ojs/19839/19839-40-23852-1-2-20220613.pdf) | A scheduler allocates crews to aircraft for predictive maintenance under stochastic failures and service times. Its heterogeneous graph represents resources and scheduling decisions, and it evaluates several objectives and scales. | A learned entity/decision representation for maintenance scheduling is also prior work. Our shared machine/worker encoder cannot be claimed as the first structured resource scheduler. |

The present simulator has its own failure law, request lifecycle, technician skills and H+tail economic cost. Cross-paper numerical scores are not comparable. A contribution here would need a specified operational mechanism and evidence against strong **same-information, same-action-set** controls, not a claim that maintenance MARL, capacity feedback or graph/entity representations are new.

## What the local evidence permits

- V1's capacity-aware binary requester C beat the learned always-request D on its sealed TEST, but did not establish a stable advantage over local B, full-information E, frozen V14 or rollout. The separate fixed-threshold G diagnostic was better than C on its own panel. The v1 TEST remains closed to new design choices.
- V2's just-in-time early-PM action-set expansion improved six-tick independent holdout forecast cost by only 0.44 on F2-visited states, below its locked 2.0 minimum. This is a bounded forecast result, not a realized H+tail or universal action-set bound.
- V3's structured same-action PPO S improved on flat full-information G3 by 22.27 H+tail cost units on its held-out panel, below the locked 30-unit practical gate. S and remaining-time ablation R differed by only 0.22. R retained free/busy status and the feasibility mask, so this tests numerical commitment time only. Frozen V14 and F2 had lower overall cost. No independent training replication or v3 TEST is eligible under the locked protocol.

The v3 actor uses present machine features, worker features and complete feasible-match logits. Its embeddings are neither calibrated economic values nor independent machine policies. The current evidence supports a negative or mixed finite-budget result for this implementation; it does not prove that coordination or PPO is intrinsically ineffective.

## Source-level screen of a longer rollout horizon

The F2 control uses a six-tick forecast horizon. In the reciprocal case, all legal service times are 2 or 6 ticks; in the specialist/generalist case, one legal machine–technician edge lasts 8 ticks, but the same machine has a 2-tick alternative. On the v3 held-out panel, S cost 813.19 versus F2 724.53 in the reciprocal case (an 88.66 gap), compared with 987.46 versus 979.81 (a 7.65 gap) in the specialist/generalist case. Thus a service commitment extending beyond F2's six-tick horizon cannot explain the largest observed S–F2 gap. This is a configuration and existing-outcome screen, not a measured 12-tick forecast result or proof that longer-horizon interactions never matter. It does not justify a new horizon campaign by itself.

## Decision

Do not launch another architecture or seed campaign from these outcomes alone. The next research step is to specify a **distinct operational decision or commitment mechanism** whose potential advantage over strong controls can be measured without changing physics, creating artificial costs, concealing health or reusing sealed TEST. Before any scientific run, write a new proposal and locked bounded protocol with fresh seed roles, a same-contract F2 comparison and a falsifiable practical headroom threshold. If no such mechanism can be justified, report the negative/mixed result as the outcome of this project rather than claiming a validated capacity-coordination benefit.

The user's later permission to run a 32-seed sealed TEST is recorded. It authorizes an eligible TEST under a locked protocol; it does not override v3's failed predeclared replication gate. The v1 32-seed TEST was already completed and must not be rerun or used for tuning.
