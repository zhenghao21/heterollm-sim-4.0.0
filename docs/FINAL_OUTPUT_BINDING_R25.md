# R25 optional final-output binding

Default remains off. Enable only on a new frozen selection with
`--final-output-selection`, verified `--host-offload-source-contract`,
`--sampling-contract`, `--tensor-storage-contract`, and
`--tensor-storage-f32-hidden`. Existing frozen predictions are not modified.

The narrow adapter in `tools/native_final_output_binding.py` binds reviewed
source fingerprints, re-derived historical runtime module lineage, actual
GGUF SHA/architecture, selected configuration and verified static sampling
settings. Reviewed collectors establish the completion endpoint; reviewed
server task/context branches and the non-speculative sampling qualification
establish the ordinary output-selection branch. Unsupported or missing scope
stays uncovered; altered identity/digest or conflicting declarations fail.
Workers never open native timing responses for this feature.

Architecture include bodies were not hashed in the historical Unity build.
Their current reviewed bytes do not prove historical compilation. Qualification
therefore remains conditional, historical_include_content_proven=false and
native_dispatch_proven=false. No new cost coefficient is introduced.

Production GGUF canonical names qwen2 and llama are explicitly supported, as
are the existing decoder graph names. qwen35 (including this project's Qwen3.8
GGUFs) resolves to qwen3_5_hybrid_transformer. Selection is never chosen by
model display name. The canonical declaration is ModelSpec.metadata's top-level
key; matching legacy mirrors are tolerated, but conflicts and nested-only
owners fail rather than silently disappear. The production binder writes only
that canonical owner. Final replan includes the changed model fingerprint.

Qwen2/llama retain full attention/KV work and select rows before the last FFN.
Qwen35 retains full last FFN and full final norm, then selects the LM-head rows.
The LM head already uses logit rows. Decode identity indices and GET_ROWS are
retained. Consequently this is a semantic correction, not a guaranteed decrease
in TTFT/TPOT. Accuracy must be reported separately from semantic correctness.

Validation performed in the isolated worktree:
- New binding tests: 34 passed (small synthetic real GGUF builder objects,
  production scenario builder/final replan, intercepted run_scenario boundary,
  row counts/dependencies, corruption, default-off, serialization and MTP).
- Relevant preexisting selector/planner/static-predictor/GPU-invocation/GGUF
  tests: 155 passed, 33 skipped for unavailable local source/model artifacts.
- Read-only real existing R23 runtime/source contract roundtrip and one static
  cell proof re-verification passed. No full GGUF data hash scan was introduced
  by this check, no predictions or native execution, no target timing read.
- git diff --check passed.

Integration touches the predictor's static_inputs, freeze_selection,
verify_freeze_references, worker_cell, predict_cell and CLI parser. Apply the
worker hook after all existing static bindings and before the final replan.
Concurrent edits to that file should be merged narrowly. The new module is
included automatically by the existing source_freeze Python-file sweep.

Additional static physical-dispatch regression: ordinary Q5_K prefill B=64/R=1 retains attention M=64 while both final FFN invocations use M=1. The down projection re-enters the MMVQ branch. Fused up/gate remains explicitly uncovered for single-physical-matrix MMQ qualification; this is not claimed as dispatch proof. No timing calibration or native run was used.

Reviewed effective-configuration source chain (2026-09-16): server-task.h need_embd returns false for completion and need_logits returns true; server-context.cpp sets embeddings from the batched task and marks the last prompt token as output after prompt completion. llama-context.cpp initializes embeddings_nextn_masked=false; the reviewed speculative.cpp masked mutations belong to draft/MTP contexts, while the bound sampling evidence requires speculative.types=none,none. qwen35.cpp performs final norm before GET_ROWS when masked=false. This supports the conditional ordinary-completion branch; historical included-source identity and native dispatch remain unproven. Static source evidence does not upgrade qualification to verified.

After merging the reviewed R24 identity repair b53b250 into the isolated candidate, combined binding/selector/planner/retained-adapter regression: 114 passed, 1 skipped. Main R24 frozen execution source was not modified.
