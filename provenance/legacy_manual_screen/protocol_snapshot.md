# Formulation gate: machine demand and technician capacity

Date: 2026-10-22. Status: prospective formulation and mechanism protocol, **not** a
positive MARL result. This goal supersedes the V14 scorer-improvement direction
for new work; all V14–V20 verdicts and artifacts remain historical evidence.

## 1. Source contract and question

The active kernel is `maintenance_requests.py` over `maintenance_dispatch.py`.
At a tick boundary, operating unassigned machine `m` without an active request
may submit one PM request. Admission does not stop the machine, reserve a
technician, change degradation/failure dynamics, or charge a fee. A queued PM
machine keeps operating and may fail; the same request then becomes CM while
retaining `created_at`. Failure without a PM request creates CM automatically.
The coordinator may select any legal partial machine–technician matching,
including the empty WAIT matching. Service starts immediately, occupies both
resources for a non-preemptive duration, and restores health only on completion.
Eligibility follows the existing positive `service_time[m][j]` skill edge and
idle-technician constraints. The objective remains the episode H48 plus common
tail16 economic cost: maintenance + unavailability + failure; queue age has no
independent monetary cost.

Question: can a machine's capacity-aware *request decision*, followed by a
resource coordinator's matching decision, improve physical/economic outcomes
beyond a coordinator given all eligible requests and the same observable
machine/resource state? An improved timestamp or request-wait statistic alone
does not answer this question.

## 2. Candidate decision contract, before architecture

At each H48 decision tick, resource status is published first: compatibility,
worker idle/busy, remaining service time, and current PM/CM queue. A shared
machine policy sees its own observable age/failed/service state and that
present-time resource status; it emits `REQUEST` or `DEFER` only when operating,
unassigned and request-free. Existing requests stay active until service
completion or PM-to-CM upgrade. The first test has no withdrawal, priority
message, prediction, or future realization. A failed machine has an automatic
CM request and no agent veto. The coordinator sees admitted requests, present
machine health, compatibility and technician state; it chooses a legal partial
matching or WAIT. The chosen matching alone enters the physical transition.
Both policies would be trained against the same economic episode return;
execution uses only declared current observations. A centralized critic may
see the joint present state during training. No reward shaping, message fee,
failure suppression, production stoppage on request, or changed physics is
introduced to manufacture a requester effect.

The lifecycle is therefore: eligible request-free PM machine -> submit or
defer -> persistent queued PM -> optional automatic PM-to-CM on failure ->
service start on a legal edge -> non-preemptive in-service state -> completion
and restoration. Creation, kind/upgrade, start, completion, H/tail residual
queue and in-service counts remain separate recorded events. A queued PM may
wait arbitrarily and is not a commitment by the coordinator. The requester is
not allowed to conceal CM or erase its cost.

This is a genuine two-actor action contract, but whether its machine actor
contributes a necessary decision is an empirical and structural question. A
machine embedding without a separate action would be called a structured
centralized policy, regardless of the number of network modules.

## 3. Structural null: free requests can be redundant

For this kernel, consider any deterministic or stochastic requester and
coordinator policy `P` using only the joint present observation. Define policy
`A` to submit PM on the first eligible tick for every machine. Give `A` a
coordinator that internally tracks which requests `P` would have admitted and
selects exactly `P`'s matching, ignoring extra real requests and their earlier
timestamps. Induction over ticks gives identical physical states, service
starts, costs, and exogenous event tapes: an ignored request changes only
request metadata, and automatic CM occurs under either policy. The virtual
requester's state can be reconstructed from the same observed physical state
and policy history. Thus the always-request action set weakly contains every
physical service schedule reachable by `P`, subject to the same observation
and coordinator expressivity. This is an expressivity argument, not a claim
that a particular finite trained coordinator will optimize perfectly.

The existing `risk_matching` and FCFS rules use request age, so early
submission may alter *their* decisions. A learned coordinator can also have
training/representation differences. Those effects do not prove a fundamental
benefit of delayed submission. The completed lifecycle diagnostic observed
identical physical schedules under two requesters on all 64 seeds in 6/9
profiles with risk matching and 7/9 with rollout, despite different PM
timestamps/backlogs. This supports running the null test first.

If always-request plus a sufficiently capable, matched-information
coordinator is equivalent or better, retire the claim that PM submission
timing is a necessary MARL agent in this simulator. Retain the result as a
negative finding. A two-way framework needs a documented operational reason
such as genuinely private machine information, an externally defined
service-consent/production constraint, or a real communication limit. None is
present in this kernel, and none will be added merely to improve a score.

## 4. Prospective small mechanism screen

This section locks the questions and pass conditions before any new outcome.
The screen is engineering/mechanism evidence, not a distributional superiority
claim. Use the unchanged N4/K2 H48+tail16 kernel and two profiles:
`reciprocal_scale2` (anchor) and `interchangeable_scale1` (symmetric service
time). The hand-computable engineering panel uses raw physical seeds
76000000..76000003 in each profile, checked for collisions with existing
registries before execution; these are opened mechanism seeds and never a
future TEST panel. Fixed budget is exactly eight H48+tail16 paired traces,
no early outcome stop, with a 120-second hard timeout. First, construct
same-tape paired executions with (i) threshold and (ii) always-request,
forcing the same
legal matching trace whenever possible. Compare at every tick the physical
state, feasible matching superset, cost components, request metadata, H/tail
queue and service starts. Pass the structural invariance check only if every
forced physical state and economic component agrees; report any divergence
with the first causal action and preserve the failed artifact.
Write one new UTC timestamped directory with `manifest.json` (status, source
commit/hash, seed roles, case grid and terminal counts), `paired_ticks.jsonl`
(physical states, submissions, matching, costs and request metadata), and
`summary.json` (equality flags and request-metadata differences); preserve
partial output if a check fails. A second small reference-state check needs a
separately locked amendment before execution and is not hidden inside this
eight-pair screen.

Second, enumerate or manually construct present-state pairs with the same
target-machine health but different observed worker remaining times/queue.
Evaluate whether a fixed economic full-horizon reference prefers different
*service starts*. Do not infer a requester benefit from that fact: the
coordinator may make both choices under always-request. Third, compare
threshold/requester-signal/always-request under the same fixed coordinator
and a coordinator that has full current information. Report signal-use as
action differences and physical/cost consequences on paired tapes, not as
attention weights. A manual state witness is labeled manual and separate
from episode-distribution estimates.

Mechanism decision gate: proceed to two-policy training only if a legal,
source-grounded operational asymmetry makes requester timing or message
content change the *available physical decisions* or supplies information
unavailable to the coordinator, and the screen demonstrates a nontrivial
physical/cost effect. If the structural null holds under the current contract,
do not spend a full training budget to claim requester necessity. Study a
capacity-aware centralized coordinator as a distinct project or first obtain
an independently justified operational contract.

## 5. Conditional training/benchmark protocol, not yet released to run

If the mechanism gate is met, freeze a separate full protocol before coding
or training: hypothesis and estimand, profiles, cost/health gates, exact seed
registry, matched architecture/parameter counts, checkpoint rule, update and
wall-clock budgets, timeout, independent audit, and source/config hashes.
Training uses at least five seeds, paired physical DEV scenarios, independent
replication seeds, and sealed TEST only after a predeclared DEV and replication
gate. Report each case and training seed, hierarchical uncertainty across
both levels, and no winning-seed selection or TEST tuning. Primary metric is
H+tail economic cost. Secondary metrics: H availability/failures; PM/CM wait
with creation times; H/end backlog/in-service; service starts; legal-service
WAIT; inference latency; training compute. Keep original A/B legacy targets
separate from this goal's prospective gates.

Minimum arms: (A) threshold requester + frozen V14 count-calibrated coordinator;
(B) learned local-only requester + learned coordinator; (C) capacity-aware
requester + learned coordinator; (D) always-request + the same learned
coordinator; (E) centralized full-information RL with comparable capacity
and budget; (F) FCFS/risk, existing MaskablePPO and compatible rollout.
Also cross a fixed requester with learned/frozen coordinators to separate
coordinator learning. B/C must share architecture, train seeds, updates and
information except capacity feedback. D and E are essential tests of the
requester's incremental value. If V14 is reused, its logged-Q/KL training is
identified as such; importing `MaskablePPO` only for a baseline does not make
V14 a PPO method. If PPO/CTDE replaces it, that algorithm change gets its own
matched control. V14's current fixed N4/K2 observation/action catalogue does
not establish support at larger N/K; scaling waits for a legal dynamic decoder
and rechecked baseline/environment compatibility. Retraining and zero-shot
transfer are reported separately.

Every code experiment requires the repository's macOS smoke gate, tests, and
verified commit/push before a qualified lab run. This session is on the Ubuntu
lab; an Ubuntu test cannot be relabeled as macOS smoke. Each diagnostic/run
gets a new UTC timestamped directory with RUNNING/COMPLETED/FAILED manifest,
protocol and Git/source/config hashes, seed roles, logs/partials, counts and
audit status; no existing artifact is overwritten. Training and multi-seed
evaluation show `tqdm`. The user-authorized exception permits this lab agent
to launch qualified diagnostic/DEV/replication after those gates; it does not
authorize SSH or Mac rsync. If Mac smoke is unavailable, prepare a precise
patch and Mac command for the user before any full run.

## 6. Prior work and contribution boundary

[Ruiz Rodríguez et al. (2022)](https://orbilu.uni.lu/handle/10993/54345)
already study machine-level partially observed MARL for predictive maintenance
with technicians of different skills. [Wang and Gombolay (ICAPS 2022)](https://ojs.aaai.org/index.php/ICAPS/article/view/19839)
already study AirME/HetGPO, stochastic maintenance resource allocation,
heterogeneous graph policy and PPO-derived training. Hence neither MARL
maintenance, hierarchy, resource awareness, nor graph matching is claimed as
new. A defensible contribution, conditional on new evidence, would be an
audited *operational contract and evaluation* that separates machine demand
control from capacity allocation under persistent PM/CM, demonstrates when
the split is useful or redundant against always-request/full-information
controls, and reports the cost–runtime/robustness boundary. The structural
null is publishable as a limitation even if it ends this particular MARL
formulation.
