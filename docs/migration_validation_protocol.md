# Migration engineering protocol — locked before validation

Hypothesis: the five byte-preserved imported modules retain their original behavior. This is software equivalence, not a scientific policy-improvement hypothesis.

Reference: source commit in provenance/import_manifest.json, loaded in an isolated Python process. Generate original reference fixtures before running destination checks. Record exact hashes and compare imported source byte-for-byte.

Coverage: inherited physical/request tests; original rule episode metrics for all four legacy cases with FCFS, risk matching and bounded rollout; V14 initial-state logits and observations for both modes; mask and cost accounting; tiny PPO train/save/reload and V14 actor serialization. Remove only the legacy campaign-specific preflight test whose campaign is intentionally excluded, and record this adaptation.

Opened engineering seed: 62500000 for reference episodes and model initialization; PPO training seed 62500001, evaluation seed 62500002. These are reused engineering roles from old development, not independent/fresh scientific panels. No train/DEV/TEST claims. Stop at the fixed coverage; no result-dependent extension. Smoke PPO budget: exactly 64 requested timesteps. Evaluate three rules across four cases plus PPO across four cases, one evaluation seed each. Verify PPO and V14 predictions after save/reload. Single-thread CPU. Hard process budget: 120 seconds; failures remain recorded.

Pass: all imported-module hashes equal, focused tests pass, original-reference numeric fixtures match (floating tolerances declared in tests), finite values, zero invalid assignments, reward equals negative H+tail cost, checkpoint actions agree, manifest COMPLETED. Performance values carry no research interpretation.

Artifacts: new UTC timestamped artifacts/migration_smoke_* directory containing RUNNING/COMPLETED/FAILED manifest, episodes.json, PPO checkpoint, initialized V14 checkpoint and error if any. Validation record in provenance/validation.json includes platform, runtime, counts and source equivalence. No existing full artifacts are copied or overwritten.
