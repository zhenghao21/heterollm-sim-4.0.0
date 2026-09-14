# GitHub submission scope

This repository contains the simulator source, tests, task brief, compact evidence summaries, and the semantic llama.cpp server patch used for the current Engine-boundary work.

Large local-only inputs are intentionally excluded from Git history: GGUF weights, NSYS reports, SQLite exports, CSV profiler dumps, raw per-event trace JSON, generated binaries, and build directories. Their capture-time paths and SHA-256 identities remain recorded in the manifests under `artifacts/development/` and in `docs/TASK_BRIEF_AUDIT_20260914.md`. A compact manifest is not a substitute for the raw capture; the raw files must be restored from the recorded local evidence store before reproducing a profile or formal freeze.

The current task is not yet an acceptance pass. The task brief records the failing model groups, identity gates, coverage limits, and the next automatic optimization step.
