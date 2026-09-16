# Profile bridge preparation

This directory is intentionally **preparation only** while collection runs. No native/simulator/model result was read. `bridge.py` inspects protocol/code identities without inspecting `runs/` or extraction. All outputs are exclusive creation; no existing artifact is overwritten.

The active collector is `collection_r2`, protocol `operator-matrix-collection-protocol/v2`, probe `operator_probe/r3`. Its raw app schema remains `single-operator-surface-probe/v2`; additional clock evidence is part of collection v2. The existing resolver currently requires collection protocol v1 and has no clock telemetry fields in its measurement bundle. These two blockers are recorded in `readiness.json`. **Do not rewrite the frozen collector's schema to v1 or drop its clock gates to make it load.**

`bridge.py --output <new readiness.json>` is safe during collection. Only after root confirms complete extraction can `--extraction <collector extraction directory> --effective-hardware <root-provided actual complete configuration> --collection-complete-confirmed --output <new inventory.json>` inventory all26 outcomes and original evidence. Failed outcomes remain in the denominator; validation/aligned controls never become fitting candidates. No guessed effective configuration, execution K, phase owner or caller execution context is created.

The inventory preserves refs to original SQLite, raw event/application files, mapped records, quality files, process receipts, tool logs, telemetry, clock receipt and its command outputs. Candidate entries are limited to accepted training MMVQ main. When zero candidates survive, output is explicitly `no_usable_calibration`. Even nonzero candidates are not promoted: no profile/coefficient is emitted until the schema adapter and independently recomputed clock/K semantics are complete.

Outstanding root coordination:
1. Accept collector protocol v2 only with independent telemetry/clock binding validation in resolver (outside this directory's current ownership).
2. Extend measurement bundle with paired telemetry and immutable clock receipt references; rederive all formal bracketing/SM-clock checks, never trust `clock_domain_validated=true` alone.
3. Once completed extraction is authorized and the independent v2 review is resolved, finish the emission stage using actual observed single-kernel shape/variant/launch geometry and role timing; call the semantic loader before declaring a candidate usable.
4. Include the complete effective simulator hardware document supplied by actual planner context. Do not infer HBF/HBM configuration from GPU UUID or peak specifications.

Current lightweight checks only verify preparation guards and no-overwrite behavior. They do not claim complete profile generation or real evidence compatibility. No GPU execution, compilation or large test suite is performed here.
