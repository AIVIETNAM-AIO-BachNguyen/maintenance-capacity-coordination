# Canonical-path same-tick PPO v14 — audited no-go for contextual feedback

Date: 2026-10-23 (Asia/Taipei). The prelocked protocol is `canonical_matching_v14_prelock.md` (SHA-256 `3a1d45a7d7888fa912c92d6306f0b7c4791530a80428855b01254479e8e04220`); scientific collection source was pushed at `9e67b68`. All RL train, DEV, held-out and mechanism-replay trajectories came from the simulator. No real maintenance log was needed, and no new TEST was opened.

## Design and integrity

V13 found an independently validated economic difference between alternative second edges under a frozen continuation; it did not establish learnability. V14 tested whether removing ordered-path multiplicity helps PPO learn that value. The canonical C14/D14/B14 decoder admits only increasing edge indices, giving each of the 136 N5/K3 physical matching/WAIT actions one path. U14 retains unrestricted orders. C14 recomputes selected-edge and candidate residual context after provisional assignments; D14 freezes root context with the same modules and information; CF14 freezes each selected C14 checkpoint at inference. B14 is direct-feature sequential; S14 structured complete-action; G14 flat full-information; Q14 flat always-request. F1 risk and F2 known-model rollout are same-panel references. Q14 differs in admission and is not an isolated architecture control; historical N4/K2 V14 cannot score this panel.

The fixed `n5k3_residual_a/b` configurations kept threshold PM, automatic CM, simulator physics and H48+tail16. PPO trained each of seven scratch arms for 8,192 H decisions × two cases × three training roots, with DEV checkpoint selection at 2,048/4,096/8,192. Full suite **190 passed**. Bounded Ubuntu smoke `artifacts/same_tick_v14_smoke_20261023T023828168282Z` completed seven 256-decision trials/14 DEV episodes; separate replay audit `artifacts/same_tick_v14_audit_20261023T023849704411Z` **PASS**. The smoke was engineering validation only.

Scientific train/DEV `artifacts/same_tick_v14_train_20261023T024008448285Z` completed **42 trials, 344,064 H training decisions and 2,016 DEV episodes**. Independent replay audit `artifacts/same_tick_v14_audit_20261023T025810820082Z` **PASS** verified every training transition, accounting, checkpoint hash, selection rule and source/config/protocol identity. Only then did held-out `artifacts/same_tick_v14_heldout_20261023T025946488294Z` evaluate 1,536 learned/CF14 episodes and 128 F1/F2 episodes on 32 new paired raw physical seeds per case. Independent audit `artifacts/same_tick_v14_audit_20261023T030326819183Z` **PASS** replayed all 73,728 learned decisions and 6,144 control decisions, verified pairing and same-weight CF14 identity, and calculated the locked gate. Raw artifacts are preserved outside Git.

## Prelocked economic gate

Lower H+tail cost is better. Each learned contrast averages both fixed cases, three training roots and 32 paired raw physical seeds equally. The 95% interval uses 5,000 bootstrap resamples of the **raw physical seed ID as a cluster** across cases, roots and arms, conditional on the trained roots. Training-root means are reported separately.

| C14 minus control | Overall H+tail | Case a | Case b | Physical-cluster 95% interval | Three training-root means |
| --- | ---: | ---: | ---: | --- | --- |
| **D14, independently trained frozen context** | **−4.99** | −7.75 | −2.24 | [−12.00,+1.75] | −11.36,−3.91,+0.28 |
| **CF14, same C14 weights frozen at inference** | **−0.24** | +0.48 | −0.97 | [−2.88,+2.47] | +0.52,+1.12,−2.38 |
| B14, direct sequential | −12.68 | −17.42 | −7.94 | [−19.85,−5.52] | −11.06,−35.45,+8.48 |
| U14, unrestricted paths | −3.36 | −5.28 | −1.44 | [−11.92,+5.27] | +0.06,−6.41,−3.73 |
| S14, structured complete action | −7.39 | −20.76 | +5.98 | [−16.44,+1.90] | −0.27,−20.95,−0.95 |
| G14, flat full information | −29.28 | −39.89 | −18.67 | [−37.76,−20.71] | −27.75,−37.30,−22.78 |
| Q14, flat always-request admission | −186.41 | −188.21 | −184.60 | [−194.24,−179.21] | −182.14,−199.05,−178.03 |

The locked replication gate required C−D ≤−20, both cases negative, at least two roots negative and the interval upper endpoint <0; C−CF ≤−10, both cases negative and upper endpoint <0; and C−U/B/G negative overall. C−D missed the practical magnitude and interval conditions. C−CF missed its practical magnitude, both-case sign and interval conditions. The C−U/B/G sign conditions passed, but C−U's interval crosses zero and its root means are mixed. The audited `replication_go_decision.json` says **`go: false`**. Per protocol, **no independent retraining or TEST** was run; no gate, seed or checkpoint rule changed after outcomes.

| Arm | Mean H+tail case a | Mean H+tail case b |
| --- | ---: | ---: |
| C14 | 1630.76 | 1544.57 |
| D14 | 1638.51 | 1546.81 |
| CF14 | 1630.28 | 1545.54 |
| B14 | 1648.18 | 1552.51 |
| U14 | 1636.04 | 1546.01 |
| S14 | 1651.52 | 1538.59 |
| G14 | 1670.65 | 1563.24 |
| Q14 | 1818.97 | 1729.18 |
| F1 risk matching | 1708.78 | 1577.25 |
| F2 six-tick rollout | 1639.03 | 1514.97 |

C14 was 8.27 below F2 in case a but 29.60 above it in case b; equal-case C14−F2 was about +10.67. F2 is a compute-heavy oracle-style forecast reference, not sample-matched PPO. This contrast does not rescue the feedback gate.

## Behavior, compute and mechanism

C14 mean H availability was 0.346/0.372, H failures 14.29/13.01 and selected H matching cardinality 38.58/42.22 per episode in cases a/b. It made zero discretionary service WAITs. C14/D14/U14 each had 13,858 actor and 9,473 critic parameters; B14 had 13,498 actor parameters, S14 8,649 and G14/Q14 16,880. Mean training-trial wall time was 32.69/26.26/31.71/32.82/34.55/10.21/10.57 seconds for C/D/B/U/S/G/Q. C14 mean held-out action selection took 0.093/0.092 seconds per episode versus G14 0.022/0.023 and F2 1.188/1.452; F2 used about 15,558/18,963 forecast transitions per episode. A greedy C14 episode invokes the scorer once at each H root and after each selected edge, about 86.58/90.22 calls per episode versus 48 complete-action calls for G14. The inherited `decoder_forward_passes` field counts sampled conditionals and **understates** those actual scorer calls when termination is forced.

A separate post-pilot **descriptive** same-weight replay, `artifacts/same_tick_v14_mechanism_20261023T030757777783Z`, followed all 192 audited C14 held-out trajectories; each episode's H+tail cost matched the audited row. Output hashes and case aggregates were checked independently. On 9,216 C14-visited H states, some legal first edge left multiple higher-index second edges in 318 case-a and 218 case-b states (536 total, 5.8%). After C14's chosen first edge, 149/619 case-a and 131/872 case-b second-step opportunities had multiple feasible edges. Freezing the same C14 weights changed the final greedy matching in 11 and 21 visited states (32/9,216 overall); second-edge rankings changed in the same 11/21 opportunities. Independently trained D14 differed on 309/298 C14-visited matchings, which also reflects different weights and training trajectories.

These counts show the feedback computation sometimes changes actions, but they are conditional on C14-visited states and cannot be summed into episode gain. The paired **physical** C14−CF14 rollouts show near-zero economic effect here. V13's positive conditional second-edge value signal therefore did not become a useful learned feedback benefit under this canonical PPO formulation and fixed budget. V14's gain over flat G14, direct B14 and average U14 suggests a useful sequential/representation package on this panel, but it does not isolate contextual updating; neither canonicalization alone nor embedding novelty is established by the failed attribution gate.

## Decision

Close v14 as **no-go for the proposed contextual-feedback contribution** under this simulator contract and budget. Preserve the v13 positive economic-signal diagnostic and every v8–v12 no-go verdict; they answer different questions. The specific unresolved gap is how a policy can learn to **use** the conditional value differences in provisional matching choices without depending on future-event information available only to an offline diagnostic. Any successor needs independent justification and a fresh prelock, not a revised v14 threshold or favorable-case selection. Simulator trajectories remain sufficient for RL training; real logs are optional calibration. A new TEST still requires separate authorization.
