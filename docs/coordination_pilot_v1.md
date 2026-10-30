# Coordination pilot v1 — prelocked protocol

Status: LOCKED BEFORE NEW RESEARCH OUTCOMES. Date: 2026-10-22 (Asia/Taipei).
This is the sole current experiment. Code or engineering smoke may expose defects;
any scientific protocol revision must be versioned and justified before looking at
DEV, replication, or TEST results. Historical V14 panels are already open.

## Question and falsifiable hypothesis

Does a shared machine PM requester using present technician capacity improve
H+tail economic cost, learning speed, or training-seed robustness over a
same-budget local-only requester, an always-request coordinator, and a
full-information centralized learner? The hypothesis fails for a claimed
benefit when its paired replication interval does not support that benefit;
negative and inconclusive results are retained.

This tests finite-policy learning. Free persistent PM submissions mean an
informed always-request coordinator can reproduce requester schedules. No
structural necessity claim follows from a positive result.

## Operating contract and decision order

Use the unmodified `shared_maintenance_requests_v1` transition and
`pending_matching_completion_v1` physics. Pilot cases are the existing N4/K2
`reciprocal_scale2` and `specialist_generalist_scale2`, H=48 plus 16 tail
ticks. At each H tick: (1) observe current physical state and persistent
requests, (2) each eligible machine with no active request independently
chooses SUBMIT_PM or NO_SUBMIT using one shared requester network, (3) admit
all submissions simultaneously, (4) coordinator observes the admitted full
current state and chooses one feasible partial matching, including WAIT,
(5) start chosen service and execute exactly one physical interval. Eligibility
and action masks are computed before sampling. Coordinator sees all machine
health in every learned arm; it never sees future event tapes. The shared
requester sees its own age, failed/occupied/request status, own skill edges,
service times and restoration, current tick, and case constants. The C arm
also sees current availability and remaining busy time of compatible workers;
B replaces those capacity fields with zero. Neither requester sees other
machines' health. The centralized E requester sees the entire pre-decision
state. Every learned coordinator sees the same admitted global state.

A PM request is free, persistent, and does not stop a machine or reserve a
technician. No repeat submission is allowed while a request exists. A failed
machine automatically gets CM, or its queued PM becomes CM. Service is
non-preemptive; skills, restoration, costs, completion timing, failure law,
H/tail accounting, and the existing deterministic threshold + risk-matching
tail rule stay unchanged. All arms use reward `-economic_cost` only. A
centralized critic may use full present state before requester decisions,
never future realizations. Training uses the exact joint log probability:
sum of eligible machine Bernoulli terms plus the masked coordinator
categorical term conditional on admitted requests. The always-request arm
has only the coordinator term. No PPO ratio may omit a sampled action term.

## Arms and estimand

| Arm | H requester | H coordinator |
| --- | --- | --- |
| A | fixed threshold | frozen selected V14 count-calibrated (five historic seeds) |
| B | learned shared local-only | learned masked coordinator |
| C | learned shared local + present capacity | learned masked coordinator |
| D | fixed always-request | learned masked coordinator |
| E | learned centralized full-information requester | learned masked coordinator |
| F1/F2 | fixed threshold | risk matching / six-tick, 16-scenario rollout |

E uses the same two-stage action space and conditional coordinator decoder as
B/C, but its requester has global input. This makes E a full-information
centralized control with the same physical choices, not an information-
restricted strawman. D's extra requests can be ignored by WAIT. F2's
forecast transitions and decision time are reported, so it is a
compute-using reference rather than a sample-budget match. A uses the
unmodified selected checkpoint behavior; its five historic training seeds
are not presented as newly trained replicates. A fixed-threshold + learned
coordinator is an optional diagnostic if B/C attribution remains ambiguous;
it cannot replace D or E.

The primary estimand is the paired difference in mean H+tail cost on identical
physical seeds and case, C minus each of B/D/E. Report per-case and equally
weighted two-case estimates, with training seeds as independent replicates
for B–E and physical-seed paired bootstrap intervals conditional on training
seed. A/F comparisons are descriptive controls. Secondary metrics are H
availability, H failures, PM/CM waiting, backlog at H and end, service
starts, legal-service WAIT, learning curve, forecast transitions, parameter
count, environment transitions, and wall time. Request-wait is sampled at
request creation and is interpreted only alongside service and cost metrics.
Learning speed is area under the DEV cost curve at fixed environment sample
milestones; final cost is reported separately.

## Seeds, budgets, selection and gates

Training seeds: 73110000, 73110001, 73110002. Derive episode reset seeds from
training seed and episode index with `role_seed`; never reuse evaluation seed
roles. DEV physical seeds: 73120000–73120015. Replication: 73130000–73130015.
Sealed TEST: 73140000–73140031, inaccessible to model selection and code
changes until replication audit and a written go/no-go decision. Each arm/case
gets 8,192 H transitions per training seed. Evaluate deterministic policies
at 2,048, 4,096 and 8,192 transitions on DEV. Select each arm/case/training seed's checkpoint with lowest DEV H+tail
mean for that case, tie to earlier checkpoint.
No learning-rate or architecture tuning on DEV or TEST. All B–E arms use the
same PPO clipping, discount, GAE, update epochs, minibatch size and transition
budget; report actor/critic parameter count because input sizes differ.

Engineering gate before any DEV: action/log-probability consistency, exact
masks, lifecycle, CM upgrade, non-preemption, accounting, H/tail, save/load,
deterministic evaluation, and future-tape independence tests; then bounded
Ubuntu smoke, explicitly labeled Ubuntu. Baseline gate: run unchanged V14
verifier on this Ubuntu host; if historical replay differs, use only fresh
same-panel A measurements and disclose the discrepancy. Per-run timeout is
two hours per arm/case/training seed, including DEV. Stop a run at timeout or
non-finite loss, retain partials, and do not replace seeds. Replication opens
only after all planned DEV runs have terminal manifests and feasibility and
cost audits pass. TEST opens only if C has lower equal-case replication H+tail mean
than at least one of B, D, or E for each of the three paired training seeds,
and a 95% physical-seed stratified paired bootstrap interval for that
comparison has upper endpoint below zero. Use 5,000 bootstrap draws and
fixed analysis seed 73150000. Otherwise report the pilot as
negative/inconclusive and leave TEST sealed. No gate is lowered after
seeing results.

## Artifacts and interpretation

Every invocation creates a new timestamped directory (never overwrite) with
manifest, frozen protocol/config/source hashes, Git commit/status, platform
and package versions, arm/case, train and physical seeds by role, parameter
count, transition budget/actual count, forecast transitions, checkpoint
selection rule, wall time, checkpoint files, per-episode JSONL, milestone
summaries, stdout/stderr log or error, partial outputs, and terminal status.
Training and multi-seed evaluation show `tqdm`. Audits check legal choices,
cost decomposition, episode return, H+tail and paired panel coverage before
scientific interpretation. No cross-panel improvement claim is allowed.

This pilot does not establish novelty, scale transfer, or a winning algorithm.
Expansion to more N/K or skill structures requires a new protocol after the
adapter and decoder support those cases and pilot evidence motivates it.
