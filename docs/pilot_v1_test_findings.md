# Pilot v1 sealed TEST findings

Date: 2026-10-22 (Asia/Taipei). The user explicitly authorized opening the
prelocked 32-seed TEST panel after the replication gate passed. The unchanged
protocol is [`coordination_pilot_v1.md`](coordination_pilot_v1.md), with SHA-256
`ab2cae361f78596828fdcb49f2bd0bbd298e316441a73972e8aa3dc98938923c`.
No policy, checkpoint, simulator, source file, or selection rule changed between
training, replication and TEST. The optional G diagnostic used separate seeds
and did not affect the v1 gate or TEST analysis.

## Artifacts and audit

- DEV training: `artifacts/pilot_v1_train_20261022T061152746250Z`.
- Replication and written positive C–D gate:
  `artifacts/pilot_v1_replicate_20261022T061743244003Z`.
- TEST: `artifacts/pilot_v1_test_20261022T064459750562Z`.

The TEST run completed in 84.24 seconds on Ubuntu/Linux. It contains all 768
B/C/D/E episodes (2 cases × 4 arms × 3 training seeds × 32 physical seeds) and
448 A/F control episodes. The independent `experiments/audit_pilot_test_v1.py`
audit passed exact source/config/protocol hashes; completed prior runs and
positive gate; raw JSONL hashes; complete, unique paired panels; derived
physical-tape seeds; selected checkpoint identities and hashes; zero invalid
assignments; episode return; economic cost decomposition; and H+tail accounting.
Its result is `audit.json` and its fixed-protocol summary is `analysis.json`
inside the TEST artifact. All raw episode rows and selected checkpoints remain
preserved. This audit checks internal accounting and pairing; it is not an
independent implementation of the simulator.

## Primary outcome

Mean H+tail economic cost; lower is better. B/C/D/E average three training
seeds on the same 32 physical seeds per case. A averages five frozen historical
seeds; F1/F2 are deterministic controls. A/F comparisons are descriptive
because their historical training and/or planning budgets differ.

| Arm | Reciprocal | Specialist/generalist | Equal-case mean |
| --- | ---: | ---: | ---: |
| A: fixed threshold + frozen V14 | 776.71 | 998.05 | 887.38 |
| B: local learned requester + coordinator | 1159.27 | 1209.55 | 1184.41 |
| C: capacity-aware learned requester + coordinator | 1120.07 | 1217.79 | 1168.93 |
| D: always-request + coordinator | 1371.60 | 1341.10 | 1356.35 |
| E: centralized full-information learner | 1092.64 | 1209.54 | 1151.09 |
| F1: fixed threshold + risk matching | 797.75 | 995.09 | 896.42 |
| F2: fixed threshold + rollout | 688.94 | 989.34 | 839.14 |

The predeclared paired estimand uses equal case and training-seed weights.
Intervals are 5,000 physical-seed stratified paired bootstrap draws, using the
predeclared analysis seed 73150000. They are conditional on the three observed
training seeds; three training seeds do not establish population robustness.

| Comparison | TEST C minus other | By case (reciprocal / specialist) | 95% physical-seed interval | Three training-seed differences |
| --- | ---: | ---: | ---: | --- |
| C−B | −15.48 | −39.20 / +8.24 | [−24.37, −6.83] | −12.05, −48.27, +13.88 |
| C−D | −187.42 | −251.53 / −123.31 | [−195.36, −179.39] | −233.95, −158.88, −169.44 |
| C−E | +17.84 | +27.44 / +8.25 | [+8.24, +27.49] | +0.56, −11.14, +64.11 |

C−D confirms the sole comparison that passed the replication gate: the
finite-budget C policy costs less than learned always-request D in both cases
and all three training seeds. C−B has a negative conditional pooled interval,
but the specialist case and one training seed favor B; it does not establish
a stable benefit from present capacity feedback. E has a lower learned mean
than C on TEST, and all three A/F controls cost much less. No claim that the
two-policy framework is economically competitive with the strong controls,
or that the learned requester is necessary, follows from C−D.

## Behavior and interpretation

D again starts more service than C and has very low H availability:
0.079/0.160 by case, versus C's 0.510/0.561. Its H failures are fewer
(2.92/5.24 versus C's 19.86/25.58), but service downtime and maintenance
cost still produce higher total economic cost. This is consistent with the
free always-request signal causing an undertrained coordinator to overserve
PM. It does not show an intrinsic limitation of always-request policies:
their coordinator can ignore free requests. The separate, explicitly post hoc
fixed-threshold requester + learned-coordinator diagnostic favors fixed
request gating over C on its own panel; see
[`threshold_requester_diagnostic_findings.md`](threshold_requester_diagnostic_findings.md).
Its panel must not be pooled with TEST or presented as a confirmatory arm.

The v1 pilot therefore documents an implemented and audited two-stage
coordination framework, a reproducible narrow C−D finite-budget effect, and
negative or mixed comparisons against the other essential controls. The
current evidence does not support expanding N/K, adding prediction/pricing
modules, or claiming a general capacity-feedback advantage. Any revised
formulation needs a new prelocked question and new seed roles; TEST is closed
to tuning.
