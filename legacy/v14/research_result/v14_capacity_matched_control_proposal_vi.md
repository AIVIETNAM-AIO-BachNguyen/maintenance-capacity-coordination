# Conditional V14 control: count-calibrated demand versus joint opportunity cost

Written while V13 full DEV is still RUNNING, before reading any V13 DEV outcome. This is a conditional follow-up design, not a claim that V13 succeeded or authorization to open a new TEST.

## Trigger and question

Run this control only if V13's prelocked **ablation hypothesis H1** passes on DEV: competition improves case-balanced cost by at least 10 against no_competition, wins at least four of five train seeds including all three best control seeds. If H1 fails, record this plan as unused and pursue a new failure diagnosis instead. V13's planner/speed gate remains separate and unchanged.

V13's no_competition arm instantiates 3,219 parameters but its demand head does not receive a gradient; only 2,354 parameters are initially active. This makes a positive V13 contrast insufficient to attribute benefit to the proposed matching-conditioned term. The question here is whether the *joint exclusion of machines already served by the candidate matching* helps beyond a trainable demand price with the same active capacity and service-cardinality scale.

## Mechanism and fair control

For candidate matching a, V13 uses edge utility minus

`P_joint(a) = sum_(m,j in a) duration(m,j) * sum_(n not served by a) d(n,j,s)`.

The capacity-matched anonymous control uses the same encoders, edge head, demand head, WAIT head, current 56-vector and legal catalogue, but subtracts

`P_count(a) = sum_(m,j in a) duration(m,j) * [(4 - |served(a)|) / 3] * sum_(n != m) d(n,j,s)`.

Here d is the same learned nonnegative machine-technician demand. Both arms have every parameter active, use identical initial tensors per paired training seed, and price technician occupation against other machines. For a one-machine assignment these penalties are equal. For a two-machine assignment, P_count preserves the number of unserved machines but substitutes their mean demand over all machines other than the edge being served. If demand is constant across machines, P_count and P_joint are exactly equal for every legal action. P_count knows matching cardinality and global demand, but not the *identity* of the other machine served by this candidate matching. This stronger control isolates identity-specific joint opportunity cost, rather than a generic preference for one versus two concurrent services. It is an ablation of mechanism, not an assumed weaker policy.

## Prelocked prospective campaign if triggered

Use the unchanged four V1 cases, requester, physics, H48+H16 tail economic objective, all legal matchings, six updates, 16 on-policy source episodes/case/update, 64 selected states/case/update, four common stochastic forecast scenarios/root, H64 old-actor continuation, lambda 0.5 assignment contrast, Adam 3e-4, five epochs, KL threshold 0.02 and identical checkpoint selection on 16 fresh DEV physical seeds. Pair physical seed roles and initial tensors across arms; on-policy states can diverge. Train five new independent seeds 72600000-04, use DEV seeds 72610000-15, smoke seeds 72620000-01, forecast anchor 72630000, bootstrap 72640000. Verify nonoverlap with historical registries before execution. Re-evaluate locked flat PPO, previous structured policies, FCFS, risk matching and bounded rollout on the new DEV physical seeds. No opened TEST seeds are used.

Primary estimand is case-balanced mean `competition - count_calibrated` H+tail economic cost, with 20,000-draw hierarchical 95% bootstrap CI over training seed and physical tape. Mechanism gate: mean <= -10, gains in >=4/5 train seeds including all three best count_calibrated seeds, and CI upper bound < 0. Also require <=1.02x bounded rollout cost, >=10x faster selection and no risk-matching cost/runtime dominance before considering fresh TEST promotion. Report every case and seed, operating metrics, effective trainable parameters by gradient, action and service-cardinality frequencies, inference/training compute and failure modes. A favorable V14 DEV result still needs independently locked confirmation and replication under the original goal's A/B criteria; this campaign alone cannot complete the goal.

Before full execution: implement in new V14 files without editing V13 source during its run; add numerical tests for equal active parameter count, equality on one-service actions and uniform-demand states, count calibration and identity-sensitive joint score, finite gradients and legal masking; run Ubuntu smoke; commit/push verified source/protocol/tests; create new timestamped raw directory and independent verifier. Four-hour full timeout, explicit artifact hashes/provenance and tqdm remain mandatory.
