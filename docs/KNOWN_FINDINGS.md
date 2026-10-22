# Findings inherited from the old project

These are historical evidence, not success claims for this repository.

## Free requester timing can be redundant

The current contract charges maintenance, failure and unavailability, not request submission. Queued PM keeps operating and is not a resource reservation. An informed, sufficiently expressive always-request coordinator can reproduce schedules of a different requester by ignoring extra requests and tracking virtual history. This is an expressivity argument, not equal finite-policy learning performance.

The local old artifact `lab_results/machine_capacity_manual_screen_20261022T044828898572Z` contains eight forced-matching paired traces / 512 ticks. Physical state and cost agree throughout; request metadata differs and additional queued-PM time totals 1,086 machine-ticks. The two trajectories deliberately share service actions. This is not a learned-policy comparison or proof that two independently acting coordinators behave alike.

## Capacity matters at the coordinator

The fetched lab report `research_result/machine_capacity_exact_witness_findings_vi.md` describes a synthetic N2/K1, six-tick witness. With target age unchanged, serving the target costs 26.00 versus 40.20 for the healthy competitor; when the competitor is failed, serving it costs 41.20 versus 45.00 for the target. The reported audit recalculates the decision layer using the same physical kernel. This is neither independent physics validation nor an episode-level gain.

## V14 and V20

V14 count-calibrated is a useful structured baseline, not an established successful new mechanism. The identity-specific ablation did not demonstrate the required aggregate benefit. V20 activation-margin training failed its DEV gates and exhibited excessive WAIT in one case/seed. Neither justifies another automatic scorer-tuning campaign here.

On the old V20 DEV panel, reported cost was 880.79 for frozen V14 count, 908.51 for V20 and 865.88 for rollout. These values are historical and panel-specific. No trained checkpoints are included in this migration.

## Consequence for this project

Test coordination's learning efficiency/robustness against matched centralized and always-request controls. Do not create physical costs or information restrictions merely to force requester necessity. Preserve negative outcomes and distinguish model representation, training effects and information advantages.
