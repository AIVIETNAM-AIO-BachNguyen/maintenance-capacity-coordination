# Evidence needed for a pre-start commitment formulation

Status: research handoff, **not** an experiment protocol or a simulator-change authorization. Date: 2026-10-22 (Asia/Taipei). V1–v7 results and closed gates remain unchanged. The current simulator creates a PM request without reserving a technician or stopping production; only a feasible matching at the current tick starts non-preemptive service. A new planning layer is justified only if the target operation has a consequential decision before that start.

## One operational question

Does a planner or technician make a **binding assignment, appointment or capacity commitment before maintenance service starts** for the machines this framework is meant to represent? “Binding” means that the choice changes later available technician time, machine production, cancellation/replanning obligations or economic cost even if service has not started. A tentative display on a dashboard with no operational consequence does not meet this condition.

## Minimum evidence to establish the contract

An operations owner can provide a current written procedure or a small de-identified event log plus an explanation of the fields. One concrete workflow example is sufficient to decide whether a versioned contract is warranted; statistical benefit is a separate experiment. For each example, identify:

| Fact to establish | Evidence to capture | Why it matters |
| --- | --- | --- |
| Decision and timing | Who chooses a machine–technician pairing, at what time, and when service actually begins | Separates advance commitment from the existing start-service matching |
| Binding effect | What the technician and machine can or cannot do between commitment and start | Defines physical capacity and production consequences without inventing them |
| Change and cancellation | Who can revise the commitment, at what time, and with what measured operational/economic consequence | Prevents adding an arbitrary penalty or forced shutdown |
| Skill and duration | Technician compatibility, expected duration at commitment, actual duration and non-preemption rule | Preserves heterogeneous service and commitment accounting |
| Information available | Health/failure state, backlog, roster and duration estimates visible when the choice is made | Defines fair same-information centralized and structured controls |
| Objective and scale | Observed downtime, failure, maintenance, travel/overtime or other costs that the operation actually counts | Keeps the economic comparison tied to the target system |

A useful de-identified event sequence would include `machine_id`, `technician_id`, `request_time`, `commitment_time`, `planned_start`, `actual_start`, `completion_time`, `change_or_cancel_time`, contemporaneous machine/technician status and the recorded consequence of a change. The owner should state which timestamps are authoritative, whether missing entries mean “no event”, and whether technician capacity is shared across sites. No raw personal data is needed for the formulation decision.

## Decision rule after reviewing evidence

1. If there is **no binding pre-start effect**, keep the existing start-service contract and report the audited negative/mixed v1–v7 findings. Do not open another scorer, seed panel or TEST to search for a win.
2. If the effect is documented, specify a separate versioned operational contract using only the observed decision order, state transitions and cost terms. First test whether the added decision creates practical headroom against a strong full-information current-contract planner on common states and forecast conditions. Keep feasible matching/WAIT, automatic CM, non-preemptive service and the original contract as a control.
3. Only a passed prelocked headroom check could justify a learner pilot. That pilot must compare a shared machine-demand/technician-capacity representation with a same-information, same-action, near-matched-budget centralized policy and an always-request control. Reserve fresh DEV, replication and TEST roles before collecting outcomes; TEST still needs separate authorization.

The AirME/HetGPO and aircraft scheduling literature discussed in `post_v6_formulation_decision.md` motivates asking about this workflow. It does not prove that the target parallel-machine operation has it. This handoff contains no proposed fee, failure-law change or synthetic commitment event.
