# Forecast-informed demand/commitment coordination: v4 proposal

Status: distinct proposal, 2026-10-22 (Asia/Taipei). V1 TEST, the v2 action-set no-go and the v3 practical/ablation no-go stay closed. This proposal does not claim a new RL algorithm or alter simulator physics.

## Operational question and measured opportunity

The physical decision remains a complete feasible technician–machine matching or WAIT after fixed-threshold PM admission and automatic CM. Starting service interrupts production and commits a skilled technician for multiple ticks; postponement risks failure and leaves capacity available. V3's shared machine/worker actor learned from economic PPO reward alone and was faster than the finite six-tick F2 rollout, but cost more on the same held-out panel (S 900.32, F2 852.17 equal-case H+tail means). When multiple matchings are legal, F2 emits a forecast cost for every legal root. It took roughly 0.55–0.74 s per episode to select actions versus 0.044 s for S. This creates a concrete **cost–deployment-compute** question: can a structured policy amortize F2's same-contract decision rule while preserving its capacity-sensitive choices?

## One minimal mechanism

On new F2-generated training trajectories, collect each present observation, legal matching mask and F2's chosen matching, with all available forecast root scores retained for audit. Train the existing v3 structured machine/worker matching actor as S4 by ordinary masked cross-entropy against the teacher choice. The shared machine encoder represents present maintenance demand; the worker encoder represents free/busy status, assignment and remaining commitment; the coordinator scores complete feasible matchings. No machine action, prediction head, extra cost, future realized event or modified decoder is introduced. These embeddings are learned features, **not calibrated economic values**. This is structured centralized forecast distillation, not two-policy MARL.

The exact same teacher states, labels, masks, optimization steps and H+tail selection panel train (i) flat full-information G4 with the same raw actor inputs and slightly larger parameter count, and (ii) R4, the S4 architecture with only numerical remaining-time inputs zeroed in the actor. This isolates the finite-budget structural and numerical-commitment hypotheses; R4 retains free/busy status and the action mask. F2 is the costly teacher/control, not an optimum. Frozen V14 and risk matching remain same-panel cost references. PPO v3 is a historical same-panel reference with different training information, not an equal-budget v4 arm.

## Falsification

The primary question is whether the selected structured student reduces paired H+tail cost materially versus G4 and R4, while coming close to F2 cost with much less deployment forecast compute. The pilot also reports teacher-label generation forecasts and wall time, so speed is not confused with total learning cost. If S4 only imitates F2 as well as G4 or R4, the structure/capacity claim fails. If S4 is fast but far above F2 cost, the amortization claim fails. If a pilot gate fails, do not tune labels, temperature, sample counts, masks, checkpoint rules or seeds on held-out outcomes; retain the result and do not open replication/TEST under this protocol.
