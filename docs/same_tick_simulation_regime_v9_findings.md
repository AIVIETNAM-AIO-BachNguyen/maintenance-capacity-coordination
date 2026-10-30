# Simulation-regime v9 qualification — audited no-go

Date: 2026-10-22 (Asia/Taipei). Protocol: `same_tick_simulation_regime_v9.md`. Source commit `3f6f8e6`; fresh qualification artifact `artifacts/same_tick_v9_qualification_20261022T155229970484Z`; separate replay audit `artifacts/same_tick_v9_qualification_audit_20261022T155236209115Z` **PASS**. The audit rederived every threshold request, legal root edge, qualifying first edge, risk matching, reward, physical tape, complete H+tail episode metrics, source/config/protocol/output hashes and gate arithmetic. The full suite passed 151 tests and the bounded Ubuntu smoke and replay audit passed before this panel. Smoke was engineering validation only.

The fixed risk-matching simulator trajectory used 32 new raw physical seeds in each of two predeclared existing N4/K2 configurations (1,536 H states per case). A state qualified if some feasible first machine–technician edge left at least two feasible second edges for a different machine and worker.

| Existing simulated case | Qualifying H states | Frequency | Locked minimum |
| --- | ---: | ---: | ---: |
| `interchangeable_scale1` | 6 / 1,536 | 0.39% | 47 / 1,536 (3%) |
| `reciprocal_scale1` | 84 / 1,536 | 5.47% | 47 / 1,536 (3%) |

The gate required **both** cases to pass. It failed, so the conditional C9/D9/B9/S9/G9 PPO pilot, independent retraining and TEST were **not run**. No case, gate or seed was changed after observing these counts. The higher opportunity frequency in reciprocal is a structural observation on frozen risk-policy trajectories, not economic headroom or evidence that contextual update improves RL. The low interchangeable count does not imply that simulated data are insufficient for RL generally; it means this predeclared panel did not adequately exercise the proposed second-step reranking in both cases.

Decision: close v9 no-go. RL training data for this project remain simulator rollouts. A later simulation design should justify demand intensity, compatibility and staffing from an explicit queueing/operations scenario, version any N/K or arrival-contract change, then lock a *new* comparison before outcomes. Real logs are optional calibration or external validation and are not required for simulator training. Preserve v8/v9 verdicts and unopened TEST roles.
