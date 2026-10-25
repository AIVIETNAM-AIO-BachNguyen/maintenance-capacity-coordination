# V22 sparse-skill structural opportunity qualification: audited findings

Status: **closed, no-go** on 2026-10-25 (Asia/Taipei). The protocol in `docs/sparse_skill_opportunity_v22_prelock.md` fixed both sparse N5/K3 skill masks, seed roles, exact structural statistic and joint gate before simulator outcomes. A focused test exposed zero duration on a declared compatible edge in the base generator; the deterministic three-tick fallback was recorded and pushed as a protocol amendment **before any v22 simulator run**. V1–v21 verdicts remain closed. No v22 learner, economic held-out, independent retraining or TEST ran.

## Question and setup

The concern was whether the current simulator bank contains enough reachable states to evaluate a decoder's treatment of **who can substitute for whom**, beyond how many edges remain. Two fixed skill graphs were selected using static topology only: case a has worker degrees (4,3,1), case b (5,2,1). Each graph has eight compatible machine–technician pairs, full matching capacity three and two first physical assignments with equal residual edge counts but different residual maximum matching capacities when all those pairs are immediately available. The simulator kept its failure/restoration/cost law, threshold PM admission, automatic CM, non-preemptive service and H48+tail16. F1 risk matching determined the visited present states; all structurally feasible first pairs at each visited state were assessed without executing counterfactual actions. No cost was used for configuration or gate selection.

For a visited state, a qualifying tick required two different feasible first assignments that leave the same **physical** residual edge count of at least two, but residual maximum matching capacities one versus two, after removing the assigned machine and technician. This count ignores canonical decoder index ordering; canonical residual counts were recorded separately. The prelocked gate required at least **16 qualifying ticks and 12 distinct qualifying episodes in each case** among 32 fresh raw seeds per case.

## Provenance and replay

Clean pushed source `3435e51` ran after a 238-test full suite. Bounded Ubuntu engineering smoke `artifacts/sparse_skill_v22_smoke_20261025T051415159290Z` completed 48 H decisions; independent audit `artifacts/sparse_skill_v22_audit_20261025T051514147820Z` **PASS**. The scientific structural run `artifacts/sparse_skill_v22_qualification_20261025T052105725623Z` completed 64 F1 episodes and 3,072 H decisions. Independent replay `artifacts/sparse_skill_v22_audit_20261025T052215669114Z` **PASS** verified source/protocol/config/output hashes, seed roles, every threshold admission, immediate edge mask, feasible F1 physical action, reward/done record, structural witness and exact gate count. This is a structural qualification on simulator trajectories, not a policy-performance experiment.

| Case | Qualifying ticks / required | Episodes with qualifying tick / required | Locked condition |
| --- | ---: | ---: | --- |
| `n5k3_sparse_skill_a` | **4 / 16** | **4 / 12** | Fail. |
| `n5k3_sparse_skill_b` | **9 / 16** | **8 / 12** | Fail. |

The independently audited `go_decision.json` is **`go: false`**. Per case, there were 1,536 visited H states. In case a/b, 508/537 states had at least two feasible immediate edges, 396/268 had feasible edges involving at least two technicians, and 387/259 had at least two possible first assignments each leaving some physical second edge. Thus generic multiple-choice assignment states occurred, but the *specific* equal-count/different-capacity within-state contrast appeared only 4/9 times. These are F1-visited, threshold-admitted counts; they are not population prevalence or economic value estimates. The qualification did not compare learned actor embeddings or physical episode costs.

## Decision and limit

Both prelocked case gates failed, so this v22 sparse-skill qualification closes. The source, raw rows and audits are preserved. No alternate masks are selected using this result, and no full v22 label/PPO/DEV/held-out pilot is opened. The result shows that even skill matrices with a static capacity-one/two contrast seldom exposed that **precise** contrast along these F1 trajectories. It does not show that other machine–technician feedback choices lack value: many ticks allowed multiple assignments, and v13 previously validated a conditional second-edge economic signal under a different sampling rule. It also cannot determine whether the trained v16/v21 encoder retains useful scarcity relations. A future distinct learning hypothesis would need a fresh prelock and a criterion tied to economically binding decisions rather than relaxing this gate after seeing it.
