# Joint service-start interaction: v5 proposal

Status: exploratory mechanism proposal, 2026-10-22 (Asia/Taipei). V1 TEST and v2–v4 gates remain closed. No simulator contract, request rule, cost or physical action changes.

## Operational mechanism

Two free technicians can start service on different machines in the same tick. Each start immediately causes machine downtime and commits a technician for its edge-specific duration. The combined decision can change which machines fail or wait while both technicians are occupied and which technician becomes available for the next job. This is a joint demand–capacity interaction: the value of a complete matching may differ from the sum of independently scored machine–technician edges.

Under the fixed threshold request contract, the F2 six-tick rollout already evaluates complete legal matchings. Its root scores permit a diagnostic decomposition without inventing new actions. For each present state, compare its selected matching with an additive decision formed from the *same* F2 WAIT and single-edge scores. For a legal two-edge matching {e,f}, the additive score is Q({e}) + Q({f}) − Q(WAIT). The difference Q({e,f}) − Q({e}) − Q({f}) + Q(WAIT) measures interaction within F2's finite forecast and continuation rule. It is not a calibrated economic value, optimal Q, or proof of physical causality. One-tick costs are additive; multi-tick commitments and risk-matching continuation can create nonadditivity.

An exploratory calculation on the **opened v4 training teacher rows**, after v4 had failed, found 382/2304 reciprocal states with a legal two-start matching. In those states the additive rule differed from F2 in 158 and incurred mean +2.00 on the *same selection forecast scores*; the specialist/generalist case had 202/2304 such states and mean +0.10. These opened, correlated forecast scores only motivate a fresh bounded check. They cannot establish independent headroom, H+tail benefit, or a v5 learner result. The reciprocal case is the predeclared primary stratum; specialist/generalist is reported as a contrast, not selected after the new outcome.

## Falsifier and next decision

The companion locked protocol uses new F2 trajectories and an independent common forecast bank to test whether the additive approximation loses a practically meaningful amount against F2 where two starts are feasible. A failed gate ends this mechanism without a learner. A passed gate establishes only a bounded *representation opportunity* and permits a separate proposal for an explicit joint matching representation, controlled against the existing structured actor, flat full-information actor, frozen V14 and F2 under the same information and action set. It does not authorize a new TEST or imply that a trained actor can realize the forecast difference. No v4 labels, optimizer or checkpoints will be tuned on prior held-out outcomes.
