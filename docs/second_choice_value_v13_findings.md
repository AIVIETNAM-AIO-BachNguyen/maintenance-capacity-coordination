# V13 second-choice economic signal — audited qualification passed

Date: 2026-10-23 (Asia/Taipei). Prelocked protocol: `second_choice_value_v13_prelock.md` (SHA-256 `a9671b8e2622d9527c791312b947b7bb3fbfd074a88f0bbd17cba5b867aa53dd`). Collection source: pushed commit `ed4cd38`. The only objective was to test whether alternative second assignments have a stable **conditional economic ranking** under a fixed C12 continuation on simulator-generated data. No learner, independent retraining or TEST was opened; v1–v12 no-go verdicts remain closed.

## Integrity and design

Before scientific collection, 178 tests passed, including seed-bank separation, exact branch replay and gate wiring. Bounded Ubuntu engineering smoke `artifacts/second_choice_v13_smoke_20261023T022533181188Z` selected two states, and its separate audit PASS replayed eight forced branches. Smoke is not scientific evidence.

Scientific artifact `artifacts/second_choice_v13_diagnostic_20261023T022715247813Z` used all two fixed N5/K3 v12 cases, all three audited DEV-selected C12 checkpoints per case, and 16 fresh paired raw physical seeds (`75711000..75711015`). Each episode followed greedy C12 and took its first qualifying tick in 8..39: C12's first chosen edge left at least two legal second edges. Every candidate then forced a **two-edge** physical matching sharing that first edge, after which the same frozen C12 checkpoint controlled the episode through H and the inherited risk tail. Future event tapes were sampled independently of the current state and held common across candidate matchings. Bank A's four tapes selected the lowest- and highest-cost second edge; four independent Bank B tapes evaluated those fixed identities. The policy never received future events as input.

Independent replay audit `artifacts/second_choice_v13_audit_20261023T022843727273Z` **PASS** checked protocol/source/config/checkpoint identity, all 96 followed episodes, 69 selected states, 1,216 forced branches, matching legality, tape seeds and every H+tail branch cost. Source/output hashes and prelocked gate arithmetic matched. The diagnostic directory and all raw rows are preserved.

## Locked bank-B validation gap

Lower H+tail cost is better. The state-level validation gap is Bank-B mean remaining cost of the Bank-A-worst second edge minus Bank-B mean remaining cost of the Bank-A-best second edge. Positive means Bank A's ranking generalized to independent future events. It is **not** a learned-policy improvement or an achievable oracle return. Each case is weighted equally in the overall mean; the 95% interval bootstraps the raw physical seed ID as one cluster across cases and training roots (5,000 draws).

| Prelocked measure | Case a | Case b | Overall |
| --- | ---: | ---: | ---: |
| Qualifying states / followed episodes | 43 / 48 | 26 / 48 | 69 / 96 |
| Bank-B validation gap, H+tail | **11.12** | **15.33** | **13.22** |
| Mean Bank-A best/worst spread | 22.55 | 28.88 | 24.93 across selected states |
| Mean Bank-B full-candidate spread | 23.49 | 24.12 | 23.72 across selected states |

Overall raw-seed-cluster 95% interval for the validation gap: **[7.96, 18.62]**. Three C12 training-root means were +7.25, +11.44 and +18.75. Among the 69 selected states, 57 had two, ten had three and two had four second-edge choices; selected ticks ranged 8..38 (mean 19.17). Validation gap was positive/zero/negative in 29/1/13 case-a states and 19/0/7 case-b states. The case-b coverage is lower, but its 26 states exceed the locked minimum of 12. All 5,000 bootstrap draws were defined.

The locked gate required ≥12 qualifying states in **each** case, Bank-B mean gap ≥10 in **each** case, and a positive overall cluster-interval lower endpoint. All conditions passed; audited `go: true`. This is evidence of a **stable economic signal among legal second assignments** in this particular fixed two-edge comparison and C12 continuation regime. It resolves the v12 uncertainty that mere feasibility counts could not answer.

## Limits and next decision

Bank A uses simulated future information to select an edge pair, so its selection procedure is a diagnostic oracle, not a deployable policy. Bank B validates whether those identities differ in expected outcomes, not whether a present-state learner can discover the ranking. Forcing exactly two edges omits optional STOP and a possible third assignment, and selecting the first qualifying tick conditions the sample on C12-visited states. The interval clusters physical seeds but conditions on the three trained C12 checkpoints and these two configurations. No local branch gap was summed into episode-level policy gain.

The positive signal justifies **one separately prelocked learning pilot** aimed at a distinct learnability obstacle: ordered-edge PPO in v8–v12 assigns multiple sequence paths to the same physical matching. A canonical increasing-edge decoder can give each matching one path while retaining edge/STOP decisions and within-tick context refresh. Its own C/D, same-weight CF, direct-feature, structured, flat and F2 controls must establish realized paired H+tail benefit before the framework goal can be claimed. Do not change v13's gate, select cases after outcomes, or open TEST without separate authorization. Real operational data are optional calibration; simulation remains sufficient for RL training.
