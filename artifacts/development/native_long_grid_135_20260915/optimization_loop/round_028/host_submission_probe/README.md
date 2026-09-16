# Target-DLL host submission probe — prepared, no GPU run

This is an external synthetic GGML probe, not an LLM, serving benchmark, calibration, or cost-model change. The actual target DLL bytes, installed headers/CUPTI, source slices, target device symbol and GPU identity are pinned in `protocol.json` and `locked_identity.h`. `target_resource_usage.txt` was obtained by static cuobjdump inspection only. The target DLL is never patched or rebuilt.

## Fixed experiment and qualification gate

One F32 RMS_NORM operator type, 4096 columns × 1 row, epsilon=1e-5, with exactly1/4/16 executable nodes. Development graphs are dependency chains; held-out graphs are independent fanouts from the same input, with every output separately rooted. No fused MUL/ADD follows a norm. Source predicts exactlyN unfused `rms_norm_f32<1024,false,false>` kernels, each grid1×1×1, block1024×1×1, shared128B. This expectation is not accepted until runtime capture confirms every node, pointer relationship, source-qualified launch API/PDL attribute, stream, return code and every output's CPU-reference numerical check.

`path` is a separate, untimed CUPTI process. Legacy cudaLaunchKernel and exact cudaLaunchKernelExC are decoded; unknown launches/memory operations, missing EXIT, count/geometry/pointer mismatches or numerical failures reject qualification. Both chain and fanout1/4/16 must qualify with the same compiled probe and protocol before any `timing` process. No callback calls CUDA or reads device memory. Captured API timings are not performance evidence.

`timing` is a new process without CUPTI. Known profiling modules and injection environment are rejected.16 warmup graph replays per case;31 samples;128 graph replays per submit sample;5 independent processes per topology; count order1,4,16; chain processes1..5 then held-out fanout1..5. Calling thread pinned to logicalCPU0. All failures/raw samples retained; no retry or outlier removal. Acceptance: every case/phase has full coverage, within-process sampleCV<=5%, and maximum deviation of5 process medians from their median<=5%. These rules do not admit any cost coefficient. GPU clocks/thermal state are not assumed fixed by the identity snapshot; instability stays visible.

## Honest phase and clock ownership

- Construction: external GGML metadata/context/tensor graph creation. **Not** llama_context can_reuse or model.build_graph.
- Allocation: `ggml_backend_alloc_ctx_tensors`; includes backend allocation and possible driver waits, not separated LLM scheduler splitting/arena allocation.
- Submission: wall and calling-thread counters around128 `ggml_backend_graph_compute_async` calls; includes GGML dispatch and runtime/API enqueue. It does not isolate driver CPU work from other host logic.
- Synchronization: wall and threadCPU counters around one `ggml_backend_synchronize`. GPU wait is not a separately additive service cost.
- GPU event envelope: the public target ABI exposes no internal stream. A separate stream's start event completes before submission, and its end event is recorded after backend synchronization. The GPU event duration includes host/queue/synchronization gaps. `pure_GPU_service_ms` is null; no same-stream kernel time is invented.

QPC wall, GetThreadTimes user/kernel counters (nominal100ns units), and QueryThreadCycleTime are recorded separately. CPU cycles are never converted to ns. Counter-zero or <100 nominal ticks is visibility-limited; never infer zero CPU service. Meeting that threshold is not proof of resolution or service accuracy; thread_CPU_precision_validated remains false. Other driver/worker threads are outside the calling-thread CPU counter. Counter-read overhead is not subtracted, and QPC/GetThreadTimes reads are not simultaneous; short intervals may be dominated by observation overhead. No wall-minus-GPU residual is computed.

The existing simulator1000ns/kernel frontend,250ns/group submit and128+12G command-build assumptions can overlap these intervals. Evidence is **not connected to a cost model**, not added on top, and does not prove those values should increase. LLM internal build/reuse timing requires separate source-instrumented evidence.

## Build and host validation only

From this directory:

```powershell
E:\anaconda\python.exe build.py verify
E:\anaconda\python.exe build.py build
E:\anaconda\python.exe build.py host-test --manifest <build_manifest.NNNN.json>
E:\anaconda\python.exe -m pytest -q test_host_probe.py
```

The host decoder test has no CUDA/CUPTI/GGML imports; its import table is checked before execution. The target executable is compiled but never executed by these commands. Failed build attempts and manifests are append-only. `build.py` has no GPU execution mode.

GPU execution is a separate root-reviewed execution step within the existing user-authorized research scope; no additional user confirmation is required. The only runner is `run_probe.py`, which requires the program execution switch `--authorize-gpu-execution`, an explicit build manifest and mode/topology/index. Root must schedule it serially after native/simulator/probe work has finished; never run it while a simulator campaign is active. Do not use it during preparation. Both path receipts are required before timing, and development runs precede holdout runs. Failed attempts remain immutable. Example command syntax is `run_probe.py path --topology chain --index 1 --manifest <build_manifest.NNNN.json> --authorize-gpu-execution`; it documents the execution switch, not a request for new user authorization.

Results must pass `analyze.validate_record`; `analyze.summarize_processes` applies preregistered stability rules and always reports `cost_model_admitted=false`. Protocol, code, dependencies and target identities must match the build manifest before and after each eventual run. No existing campaign or old failure is overwritten.

The runner constructs a minimal OS environment with locked target DLL directories and explicitly requests `GGML_CUDA_DISABLE_GRAPHS=1`. Inherited dispatch/profiling/injection controls are rejected unless they exactly match that explicit setting; unrelated parent variables are omitted and their names recorded. This is a new controlled child environment, not verification of the inherited configuration. Process inventories before launch and after natural completion reject live native/simulator/probe work, while permitting recognized compilation and host-only tests. These two snapshots do not prove continuous exclusion or isolation from desktop GPU activity; desktop GPU processes are not a numerical/path qualification condition. No child is automatically killed. Launch failures, missing/invalid records, interruptions and post-run identity errors leave append-only failed terminal receipts and block reuse. Runner, analyzer, build helper, all build inputs, protocol, manifest and target/header identities are checked against the frozen pre-run snapshot.
