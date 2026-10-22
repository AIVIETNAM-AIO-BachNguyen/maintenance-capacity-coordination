# Current state

Date: 2026-10-22, Asia/Taipei.

- Stage: selective migration COMPLETE; 99 top-level tests, four original V14 tests and bounded macOS training smokes pass. No framework research training opened.
- Imported legacy source is pinned in provenance/import_manifest.json.
- Rules, simulator and legacy Gym adapter are available. V14 now includes ten selected pretrained checkpoints, the original trainer with its dependency closure, protocols/registries, verifier and historical replay data. See docs/V14.md; no access to the old checkout is required.
- The current adapter has a fixed threshold requester. A learned requester, capacity feedback and two-policy training are NOT implemented yet.
- This project tests usefulness of coordinated learning, not structural necessity of requester timing.

## One next research step

Write one pilot protocol resolving machine actions, coordinator inputs, message timing, request lifecycle, fixed-information controls, budgets, seed roles, stopping rules and artifact schema. Then implement the minimal two-policy adapter and matched centralized/always-request controls. Do not open another V14-V20 scorer campaign or accumulate synthetic witnesses as a substitute for a pilot.

Check provenance/validation.json for migration validation. The destination smoke completed 64 PPO training steps and 16 rule/PPO evaluation episodes with checkpoint round trips. Engineering smoke seeds and fixtures are opened checks, never sealed evaluation seeds.

V14 extension validation is in provenance/v14_validation.json. All ten selected actors and fifteen V7 comparator checkpoints load with matching hashes. Original V14 trainer smoke passes. New adapter and isolated original runtime agree on the forty fixed same-host episode checks. Historical Ubuntu replay differs on 73/640 episodes on this Mac; the unchanged full verifier consequently does NOT pass here. Preserve these mismatches and recheck on Ubuntu before claiming exact historical reproduction. Source/weights have not been altered to force agreement.
