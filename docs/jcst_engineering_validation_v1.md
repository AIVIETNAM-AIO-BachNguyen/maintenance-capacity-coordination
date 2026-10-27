# JCST/1.1 engineering validation, 2026-10-25

This is an engineering record, not scientific qualification or policy ranking. The old simulator and v1–v22 verdicts are unchanged. The component simulator is in src/ht_pdm_fjsp/component_service_v1.py, controllers and observation/model-access manifest in component_controllers_v1.py, H/tail adapter in component_framework_v1.py, smoke collector in experiments/jcst_engineering_smoke.py, and independent replay in experiments/jcst_replay_audit.py. Importing the modules starts no run and loads no checkpoint.

## Two hand-calculated traces

These expected states/costs were specified in tests/test_component_service_v1.py before comparison with transition output. Costs are PM 1, CM 2, unavailable machine 6 per interval, and new failure 15 per component.

**Trace A: early PM and duration-one release.** One machine has two healthy age-zero components and one free technician; both service durations are one. At boundary 0 start (m0,c0,j0). It is PM, the machine is unavailable in [0,1), no component ages, and the job finishes at boundary 1. Expected cost: 1+6=7. Boundary-1 state is [(age 0, healthy),(age 0, healthy)], technician free. WAIT during [1,2) with nonfailing uniforms costs 0 and yields both ages 1 at boundary 2. This verifies PM before a threshold, release timing, age freeze and reward = negative cost.

**Trace B: two failed targets and cross-machine boundary events.** Two machines have two components each, one duration-one technician. At boundary 0, m0 components are failed at ages 3 and 4; m1 components are healthy at age 0. Start CM on (m0,c0,j0). During [0,1), only m0 is unavailable. Both m1 components fail at the end using explicit uniform zero. The job completes at the same boundary, resetting only m0,c0. Expected cost: CM 2 + unavailable 6 + two failures 30 = 38. Boundary-1 state is m0 [(0,healthy),(4,failed)], m1 [(1,failed),(1,failed)], technician free; both machines are in breakdown. At boundary 1 start CM on m0,c1. Neither machine produces in [1,2). Expected cost: CM 2 + two unavailable machines 12 = 14. Boundary-2 state is m0 fully healthy, m1 still doubly failed; technician free. This verifies target-only repair, single downtime charge per machine, end failure timing and simultaneous completion/failure on different machines.

The focused tests (18 passed; full regression 256 passed with two pre-existing GPU compatibility warnings) also cover a tiny 1-machine/1-component/1-technician exact finite-horizon DP reference, scoped to its engineering fixture, and the idle 37-action catalogue against independent combination enumeration; all-busy WAIT; reciprocal durations; illegal action/state rejection before mutation; tuple-order and relabeling invariance; opportunistic PM; hazard/survival identity; deterministic keyed noise; serialization; H/tail decomposition; rollout counters; and an independent audit that rejects missing action-mask entries and corrupted cost rows.

## Bounded local Linux smoke and replay

Latest run: artifacts/jcst_engineering/20261025T074317.372109Z. It used one engineering seed (104729) for each of WAIT, corrective, threshold age 2, risk/duration matching and bounded rollout. Each episode had H=12, tail B=4, rollout 2 scenarios × depth 2, maximum 10,000 forecast transitions per decision and a 120-second process timeout. These bounds were set before this final run. The timestamped directory was created without overwrite; earlier development smoke directories were retained. Each episode has full before/action/catalogue/after, cost components, events, compute, partial JSONL and terminal status. The manifest contains contract/config/source hashes, Git state, exact command, budgets and runtime.

Platform: Linux 7.0.0-38-generic x86_64, Python 3.12.3, numpy 2.5.4, torch 2.14.0, gymnasium 1.3.0, tqdm 4.70.1, pytest 8.4.2. Across five completed 16-interval traces the automatic coverage check records 31 starts, 25 failures, 27 completions, 78 occupied job intervals and 55 WAIT actions. All five independent replay audits PASSed. The replay separately brute-enumerates the full legal catalogue at every state, reconstructs each transition and event boundary from the stored pre-state and keyed seed, recalculates each cost component, and checks H/tail totals and final state. It does not call the collector transition. Exact cost values are fixture behavior, not a comparative finding.

The latest run followed prior smokes. Its fresh directory records the current source hashes, full catalogue audit and incremental partial retention; no earlier artifacts were overwritten. A failed or timed-out smoke leaves its manifest and flushed partial records. The auditor rejects incomplete episodes.

## Exact local commands

From /home/bachng/Bryan/maintenance-capacity-coordination:

    uv run --no-sync pytest -q tests/test_component_service_v1.py tests/test_jcst_characterization_v1.py tests/test_jcst_diagnostics_v1.py tests/test_jcst_tiny_dp_v1.py
    uv run --no-sync pytest -q
    uv run --no-sync python experiments/jcst_engineering_smoke.py --horizon 12 --tail 4 --seed 104729 --scenarios 2 --depth 2 --max-forecast-transitions 10000 --timeout-seconds 120

The smoke prints a new UTC timestamped directory. Replay each episode in that printed directory:

    RUN_DIR=/home/bachng/Bryan/maintenance-capacity-coordination/artifacts/jcst_engineering/20261025T074317.372109Z
    for d in "$RUN_DIR"/{wait,corrective,threshold,risk,rollout}; do uv run --no-sync python experiments/jcst_replay_audit.py "$d" || exit; done

For a new smoke, replace RUN_DIR with its newly printed absolute directory.

## Limits and research handoff

The initial state is all healthy age zero; Weibull parameters, reciprocal durations and costs are engineering fixtures without field calibration. A scientific pre-outcome protocol is drafted but inactive. No scientific seed bank or sealed TEST has been opened; fixture costs and ages have not been field calibrated. The risk score is a declared heuristic. Rollout evaluates every root for a bounded short forecast and does not optimize the entire stochastic horizon. Model access is public and full-current-state; actual future uniforms are never controller inputs. A scientific study needs the inactive calibration/qualification plan and independent prelock described in docs/joint_component_qualification_migration_v1.md.
