# Same-tick high-demand simulation v10 — audited pilot no-go

Date: 2026-10-23 (Asia/Taipei). The locked protocol is `same_tick_high_demand_v10.md` (SHA-256 `b9a01fe4ec119169335d92bebdc26fa4019886281245abab383dc0bcfd65a61b`). Collection source was pushed as `715654e791dc63cdb180d49fbaddd98446761894`. This is simulation-only PPO under the same N4/K2 transition mechanism with the declared `failure_age=3` parameter and pre-existing reciprocal/specialist-generalist scale-1 skill graphs. V8 and v9 verdicts remain closed; no TEST panel was opened.

## Verification and scope

Before scientific outcomes, **154 repository tests** passed. Bounded Ubuntu smoke `artifacts/same_tick_v10_smoke_20261022T155831351226Z` and separate replay audit `artifacts/same_tick_v10_audit_20261022T155842497561Z` passed. A two-case engineering integration check completed 14 V14/risk/F2 episodes with valid H+tail accounting. Smoke observations were not scientific evidence.

Training/DEV artifact `artifacts/same_tick_v10_train_20261022T173636025619Z` completed all 30 case × arm × root trials, 245,760 H decisions, 5,100 complete training episodes and 1,440 DEV episodes; every trial used the locked 8,192-H-decision budget. Its independent replay audit `artifacts/same_tick_v10_audit_20261022T174601066382Z` **PASS** verified every training decision, physical transition, reward, H+tail accounting, source/config/protocol/checkpoint identity, seed role and DEV checkpoint selection. Only then did held-out artifact `artifacts/same_tick_v10_heldout_20261022T174659511017Z` run 960 selected learner episodes and 448 V14/risk/F2 episodes on 32 new paired raw physical seeds per case. Independent audit `artifacts/same_tick_v10_audit_20261022T174956000563Z` **PASS** replayed all 46,080 learner H decisions and every control episode and checked the locked gate arithmetic. Collection and audit artifacts are timestamped and preserved.

C10/D10/B10 use ordered internal-sequence PPO: sequence log probability and entropy refer to the sampled token path, not the marginal physical matching. Physical time and cost advance only after the final matching. S10/G10 use complete-matching categorical PPO. All five scratch arms share present-state information, physical matching/WAIT actions and H-decision budgets, but not actor compute. V14/F1/F2 are contextual references with different training or forecast compute.

## Locked H+tail comparison

Lower cost is better. Each learned contrast averages two cases, three independently trained roots and paired physical seeds equally. Intervals are 5,000 percentile bootstrap resamples of **raw physical seed ID as the cluster**, across both cases/arms/roots; they condition on the trained roots. Root means separately show training variability.

| C10 minus control | Overall H+tail | Reciprocal age 3 | Specialist/generalist age 3 | Cluster 95% interval | Three training-root overall means |
| --- | ---: | ---: | ---: | --- | --- |
| **D10, frozen context** | **−12.58** | −1.65 | −23.52 | [−16.70, −8.93] | −49.91, −4.69, +16.84 |
| B10, direct sequential | −1.60 | −3.49 | +0.29 | [−6.55, +3.63] | −15.81, +14.33, −3.31 |
| S10, structured complete action | +7.48 | −24.07 | +39.03 | [−1.41, +16.88] | +8.31, −5.30, +19.42 |
| G10, flat complete action | **+22.60** | +36.29 | +8.91 | [+11.27, +35.75] | +25.03, +36.17, +6.59 |

The locked independent-retraining gate requires C10−D10 ≤−20, both case means negative, at least two root means negative, cluster upper endpoint <0, plus C10−B10 and C10−G10 overall negative. **The practical C10−D10 threshold and C10−G10 both failed.** The audited `replication_go_decision.json` is `go: false`. The negative C10−D10 estimate does not validate useful contextual feedback at the predeclared magnitude, and flat full-information G10 has lower H+tail cost overall. No independent retraining or TEST was run; no threshold, case or checkpoint rule was changed.

| Arm | Reciprocal mean H+tail | Specialist/generalist mean H+tail | Interpretation |
| --- | ---: | ---: | --- |
| C10 | 641.74 | 708.79 | Contextual sequential PPO |
| D10 | 643.39 | 732.31 | Frozen-context sequential PPO |
| B10 | 645.23 | 708.50 | Direct-feature sequential PPO |
| S10 | 665.81 | 669.76 | Structured complete-action PPO |
| G10 | 605.45 | 699.89 | Flat complete-action PPO |
| Frozen V14 (five roots) | 663.92 | 669.83 | Historically trained transfer reference |
| F1 risk matching | 663.72 | 670.41 | Fixed rule reference |
| F2 six-tick rollout | 605.72 | 708.34 | Known-model, forecast-heavy reference |

The best listed arm differs by case: G10 is lowest in reciprocal and S10 is lowest in specialist/generalist. F2 is close to G10 in reciprocal but higher than several scratch arms in specialist/generalist; it is not an oracle. V14 was selected on other configurations, so its transfer comparison is contextual rather than sample matched.

## Behavior, compute and mechanism

C10's mean H availability was 0.650/0.604 and H failures 1.39/1.78 in reciprocal/specialist; D10 had 0.649/0.595 and 1.40/2.55. C10 started more services than D10 in the specialist case (75.90 versus 74.02 total H+tail starts) but B10 cost was effectively identical there (708.50 versus C10 708.79), limiting representation attribution. All five learned arms made **zero discretionary service WAITs** on held-out, so this pilot provides no positive STOP-use evidence. The C10 selected H matching cardinality averaged 60.84/56.16 edges per 48-tick episode.

C10/D10 actors each have 12,226 parameters, B10 11,410, S10 8,553 and G10 8,705; critics have 7,873 each. Mean training-trial wall time was 22.79/22.08/21.72/16.58/9.71 s for C10/D10/B10/S10/G10. Mean C10 held-out selection time was 0.085/0.081 s per episode by case versus G10 0.023/0.023 and F2 1.49/1.08 s. F2 used about 20,568/15,051 extra forecast transitions per episode. The v8-inherited `decoder_forward_passes` telemetry still undercounts all scorer invocations. For sequential greedy episodes count one root call plus one after each selected edge, including the final forced-end call: C10 averaged **108.84/104.16 scorer calls** per episode versus 48 for S10/G10. Timings are Ubuntu CPU measurements on this run.

A separate **post-pilot descriptive** same-weight profile `artifacts/same_tick_v10_mechanism_20261022T175323635845Z` replayed all 192 C10 held-out episodes and checked their exact H+tail costs against the audited collection rows. Its source/output hashes and independent aggregate-count check passed. On 9,216 C10-visited H states, the root feasibility graph had some first edge leaving at least two distinct feasible second edges in only **50 reciprocal and 36 specialist** states (86/9,216, 0.93%). After C10's chosen first edge, only 50/1,773 reciprocal and 34/1,021 specialist second-step opportunities had multiple feasible edges; the rest offered just one edge or STOP. Holding C10 weights fixed while freezing its within-tick context changed greedy final matching in **2 reciprocal and 0 specialist** states; second-edge rankings changed in 8 reciprocal and 0 specialist states. Independently trained D10 checkpoints differed from C10 on 57 and 545 C10-visited matchings by case, but this includes different weights and learning trajectories. The specialist C10−D10 episode cost gap therefore cannot be attributed to an observed same-weight greedy context-update action on C10's visited states. These are conditional on C10's trajectory, not a universal opportunity bound or a new causal gate. No post-pilot contrast revises the locked verdict.

## Decision

Close v10 as **no-go** for a useful same-tick contextual-feedback contribution under this predeclared simulated high-demand setting. The implementation is functioning and the physical comparisons are audited, but C10 missed its practical ablation threshold and lost to G10; same-weight context rarely altered matching. Preserve v8/v9/v10 no-go verdicts and all raw/partial artifacts. A future framework experiment would need a distinct operationally justified simulated regime with enough concurrent matching alternatives and controls capable of that N/K scale, or a distinct mechanism beyond another scorer retune. Actual maintenance logs remain optional calibration rather than a prerequisite for simulator-based RL. No new TEST is authorized.
