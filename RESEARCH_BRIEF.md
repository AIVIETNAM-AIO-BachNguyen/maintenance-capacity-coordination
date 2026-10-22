# Research brief

## Objective

Develop and evaluate a framework coordinating machine maintenance demand with heterogeneous technician capacity. A standard RL algorithm is acceptable; inventing a new RL optimizer is not the objective.

Main question: does organizing learning into machine demand decisions and resource allocation improve economic outcomes, learning efficiency or robustness compared with matched-information centralized and always-request baselines?

## Candidate framework

Shared machine policies express PM demand using local health and present capacity information. A coordinator chooses feasible partial machine-technician matchings or WAIT. CM is automatic after failure and cannot be concealed. Service remains non-preemptive. Information, message semantics and request lifecycle must be specified before training. An embedding-only machine module is a structured centralized policy, not automatically a separate MARL actor.

There is no established operational restriction hiding machine health from the coordinator. Do not invent one to make agents necessary. Full-information centralized control is an essential comparator.

## Established limitation

Under the inherited contract, PM submission is free, does not stop a machine or reserve a worker, and extra requests can be ignored. Always-request plus a sufficiently expressive informed coordinator can emulate other requester schedules. This limits claims of requester necessity, but does not prove equal optimization or sample efficiency for finite learned policies.

The old forced-action screen found physical/cost equality on eight paired traces. A synthetic capacity witness found different optimal service starts under different competing demand. Neither is a trained-framework result. Details and scope are in docs/KNOWN_FINDINGS.md.

## Research scope

1. Specify roles, observations, action timing, persistent request semantics and the common objective.
2. Prelock a bounded pilot comparing threshold + frozen V14, learned local requester, capacity-aware requester, always-request + learned coordinator, and full-information centralized RL. Add fixed-requester controls to isolate coordinator learning.
3. Match information and report model size, environment samples, forecast use and wall time. Distinguish information gains from architecture/optimization gains.
4. Measure total H+tail cost, availability, failures, queues/backlog, service starts, WAIT, learning curves and runtime across training and physical seeds.
5. Expand size/skill structure only after an implemented adapter/decoder and compatible baselines exist. Report retraining separately from zero-shot transfer.

Do not add request fees, forced shutdowns, reward shaping or hidden health restrictions merely to create a positive result. Do not claim novelty or successful coordination without ablations and supporting results.

## Success and limits

A useful result establishes a clear operating contract and reproducible evidence about when coordination helps or is redundant. A negative pilot is valid and must be retained. Paper novelty and NSTC acceptance are not assumed. Old V14-V20 performance gates are historical, not active success criteria here; lock project-specific scientific gates before new full outcomes.
