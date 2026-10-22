# V14 completion engineering check

Locked before executing this migration validation. Objective: make V14 usable without the old checkout, preserving its scientific method and original evidence. This is not a new model-improvement campaign.

- Import every selected actor for both arms/all five seeds; verify original checkpoint hashes and manifest source-input hashes.
- Re-run the original four capacity/control tests inside the isolated runtime.
- Test direct deployment of all ten selected actors across the four original cases on historical DEV seed 72610000 against the original episode CSV values.
- Replay all 640 selected episodes on the original 16-physical-seed panel with a 180-second budget, checking economic/physical metrics. Do not interpret replay as fresh evaluation.
- Run the corrected original independent verifier with a 300-second timeout. Its fixed 240 complete Q-bank samples and 640 selected episodes are an implementation replay, not independent retraining or a different simulator.
- Run one original smoke iteration per arm, same paired initialization and original smoke training seed 72620000, two source episodes per case, cap three selected states per case, two forecast scenarios, original update defaults (five epochs, Adam/KL/assignment credit), then checkpoint roundtrip and four deployment episodes per arm on smoke seed 72620001. Budget 180 seconds. No tuning or extension based on cost.
- Re-run the existing 59 top-level tests plus the new trained-baseline tests. Source kernels must remain unchanged.

Artifacts: fresh top-level artifacts/v14_<operation>_<UTC> directories with terminal manifest, checkpoint/Q training evidence where applicable, episodes, replay/audit results and errors. Hashes and final validation status live in provenance/v14_extension_manifest.json and provenance/v14_validation.json. Existing raw artifacts remain unchanged; portable bundle restoration verifies hashes and never overwrites a differing file.

Pass only when trained checkpoints, trainer core, tests, economic replay and verifier work within the new repository. Preserve negative V14 research gates. Do not commit/push or launch the full historical DEV campaign as part of this engineering check.

## Disclosed amendment after initial validation failure

The original protocol is retained above. Initial strict historical equality failed on macOS for five of forty smoke-panel episodes, and the unchanged independent verifier failed during selected-episode replay after completing all 240 source/Q-bank checks. This is a failed historical-reproducibility gate, not permission to loosen numeric tolerances or claim audit success.

Follow-up migration checks distinguish two questions: (1) whether the new adapter matches the byte-preserved original runtime on the same host (forty fixed original-panel episodes), and (2) whether that runtime reproduces the original Ubuntu CSV (report all 640 episodes and every mismatch). Keep original reference data unchanged and preserve failed artifacts. Report software migration usability separately from cross-platform historical reproducibility. No research method or full training protocol changes.
