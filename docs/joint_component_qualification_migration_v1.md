# JCST/1 qualification and migration — revision 1.1, inactive proposal

Status 2026-10-25: the [JCST/1.1 contract](joint_component_service_contract_v1.md), separate component simulator, engineering controls, tests and bounded Linux smoke are implemented. No learner training, scientific calibration/qualification, TEST, branch, commit, push or remote execution is authorized or started. The historical DC-F0/1 simulator and v1–v22 verdicts remain unchanged. This document is a future protocol design, not an active prelock.

## Evidence and claim boundaries

Rodríguez (2022) already includes technician–component selection or delay. JCST/1 adds an explicit, reproducible operational contract for this repository, with full current information and economic accounting; it is not a replication of that paper's observations, reward or collision handling. Engineering fixtures demonstrate only code behavior. The two possible later method claims are separate: lower economic H+tail cost at suitable matched information/action/compute budgets, or cost within a prespecified acceptable margin using less training data, forecast compute, inference time or wall time. Neither claim requires a new RL optimizer, a second requester policy or a win over F2 on economic cost as a universal entry gate.

Scientific work must separate five questions:

| Stage | Estimand | What a positive result can support |
| --- | --- | --- |
| R, reachability | Frequency of timing, component and technician alternatives on predeclared reference trajectories, episode-clustered. | Choices actually arise in this state population. |
| E, conditional consequence | Paired common-noise H+tail cost of preselected legal action pairs under a fixed continuation. | The conditional choice has economic consequence, not that a policy can find it. |
| I, selection | Held-out cost/ranking of a selector using only current observation and its declared model budget. | Present-information exploitation at that budget. |
| F, strong-control headroom | Paired physical H+tail comparisons against full-information direct, risk/duration matching and matched-compute rollout. | A solution effect within JCST/1. |
| A, amortization | Cost difference together with measured training decisions, forecast transitions, inference latency and wall time. | An acceptable-cost efficiency claim if its margin and resource savings both qualify. |

A loss to rollout does not negate reachability or consequence. Root catalogue enumeration is exact at the current state only; bounded forecast rollout is not exact stochastic-horizon optimization. Historical DC-F0/1 absolute cost is not a JCST comparator.

## Coordination witnesses

Enumerate the full legal root catalogue before inspecting outcomes. Keep witness identities and priorities invariant under relabeling.

- T: add/remove a healthy-target PM start, holding other starts fixed.
- C: change the target component on the same machine and technician, holding other starts fixed; disclose any PM/CM mix.
- W: change technician for the same machine/component, including duration or skill differences.
- J: **same-tick capacity contention** where two machines cannot both start, even after reassignment. J is one motif, not the definition of all coordination.
- A: same number of starts but different complete assignments and total or individual service durations; compare future occupancy and cost.
- R: legal roots with different technician release boundaries, including a duration-one release versus a longer commitment.
- Q: WAIT for a better matched currently busy technician versus start now with an available slower technician; compare under common forecasts and disclose the wait risk.
- P: START versus WAIT when the starts alter the future feasible catalogue or technician capacity at a prespecified later boundary.

For A/R/Q/P, record both root action and future commitment path, and report whether the comparison actually changes future feasible actions. A motif's existence alone is not a policy gain. No new scientific gates are assigned by this list.

## Inactive staged protocol design

**Calibration/estimation first.** Freeze source/contract/config hashes, initial age and failure distributions, compatibility and duration tables, hazard law, costs, H/tail and sensitivity grid before collection. Prespecify distinct engineering, calibration, selector-training, DEV, confirmation and sealed TEST seed roles. Calibration seeds estimate operating lifetime relative to service duration, PM/CM/WAIT tradeoffs, heterogeneous skills and future commitment exposure, horizon/tail sensitivity, and no-bottleneck or nearly separable negative controls. Report component failure and service-time distributions, mode occupancy, cost components, action catalogue sizes and motifs by episode. This stage estimates scale and uncertainty; it has no positive performance gate and cannot select a configuration because it favored a proposed learner. If a case is changed, version the case and use independent confirmation seeds.

**Conditional qualification next, only after a separate prelock.** Declare reference controllers, physical seeds, selected witness rules, candidate count, continuation controller, independent forecast banks, forecast scenarios, depth/transition caps and timeout *before* viewing economic outcomes. Pair physical trajectories and counterfactual noise by raw seed; keep training roots distinct from physical seeds; resample at the physical-episode cluster level, not individual branches. Select pairs on one bank and estimate costs on an independent bank. Report uncertainty from physical clusters and forecast tapes, rank stability, extreme-pair selection bias and cost-independent sentinel pairs. Stop as INVALID on any state, action-mask, event, cost or replay discrepancy; stop as INCOMPLETE on budget/timeout. Preserve partials. No cost finding is interpreted before independent audit.

**Selection and pilot later.** A current-information selector first has a declared forecast/inference budget and frozen decision rule. A physical pilot, if justified, compares a method with direct full-state, risk/duration and matched-compute rollout controls under the same JCST action set and observation. Threshold, periodic and corrective-only remain restricted-behavior controls, not eligibility rules for the method. Training seeds, sample counts, architecture/parameter budget, checkpoint selection on DEV, independent held-out episodes, pairwise estimator and uncertainty method must be locked before training. F2 is a strong cost control, not a mandatory cost-win gate for a separately defined efficiency claim. Independent replication and TEST require later explicit authorization.

**Practical thresholds.** Before scientific collection, estimate scale from operational cost units and a separate calibration panel: e.g. how many unavailable-machine intervals are material, what extra latency or training time an operator could accept, and how sensitive these are to H/tail. Then prelock a cost margin, an acceptable-cost tolerance for efficiency, minimum resource savings and uncertainty criteria on fresh confirmation seeds. Do not inherit v1–v22 numerical gates by habit or choose thresholds from engineering smoke rankings. If practical scale is unknown, report descriptive intervals and seek external or domain calibration before a positive method claim.

## Engineering status and remaining decisions

The current default fixture is 3 machines × 2 components × 2 technicians, initially healthy age zero, all compatible, reciprocal 2/8 and 8/2 durations, Weibull (6,5)/(8,5), costs 1/2/6/15, controlled H=48 and risk tail B=16. It is an engineering fixture, not a calibrated benchmark. Linux smoke uses shorter explicit H/B only to exercise transitions. The code preserves target-only restoration, series downtime, age freeze, duration-one boundary release and keyed physical noise. The full catalogue, model-access manifest and compute counters make controller behavior reviewable.

Before scientific collection, decide and freeze initial-state population, at least two independently motivated cases, compatibility/skill plausibility, hazard and cost calibration or simulation-unit interpretation, H/tail sensitivity, relevant operational claim (same-tick J, multi-period A/R/Q/P, or both), reference population and practical margins. If real operations cannot permit healthy PM or observe component states, define a new contract version rather than silently restricting JCST/1. Real logs are optional for a simulator-internal study but necessary for field-calibrated claims.
