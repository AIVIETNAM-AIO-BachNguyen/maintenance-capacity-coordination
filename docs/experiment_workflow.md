# Experiment workflow

This repository starts a new framework project. Historical branch gates and lab launch authorizations do not silently carry over.

1. Before research outcomes, lock a protocol in docs/: hypothesis, estimator, operating contract, arms, primary/secondary metrics, train/selection/DEV/sealed TEST seed roles, budgets, checkpoint rule, stopping rule, timeout, gates and artifact schema.
2. Implement focused tests and run a bounded local macOS smoke. Smoke seeds are opened engineering seeds, not final evaluation. Preserve dynamics and verify masks, costs, H/tail accounting and checkpoint round trips.
3. Commit/push verified files when authorized. Do not assume copying a project authorizes publishing it.
4. Provide the exact compound command for the user to run in the Ubuntu terminal at bachng@100.111.83.52. Do not SSH or launch full runs unless the user explicitly authorizes that agent/session. Resolve the new lab checkout path before issuing commands; do not reuse ~/ht-pdm-fjsp by accident.
5. Use tqdm for training and multi-seed evaluation. Write a new timestamped run directory, recording source/protocol/config hashes, Git state, versions, seed roles, checkpoints, episode/partial outputs, logs and terminal status. Refuse overwrite. Keep failed/incomplete artifacts.
6. Audit feasibility, cost accounting, coverage and seed separation before interpreting performance. Distinguish independent replay from self-checks. Never tune on TEST or erase negative results. Replication/TEST eligibility must be predeclared.
7. After the lab run, provide an exact user-run Mac command of the form `rsync -avhP --partial REMOTE_RUN/ LOCAL_RUN/`; resolve both paths, retain the run directory name, and never start the transfer on the user's behalf.

## Engineering migration check

The migration protocol is docs/migration_validation_protocol.md. It does not start a new research campaign, open TEST, require a lab run, or authorize committing/pushing. `uv run mcc-smoke` validates the inherited adapter and baseline pipeline only.
