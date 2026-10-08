# CUDA Graph structure preparation and native diagnostics

The supported llama.cpp revision is `d3146f2b56c2db4711ac8391871c9e529d1946d7`.
The producer calls that native runtime to build the target GGUF's graph and uses
the real CUDA runtime to capture kernel/copy calls. Capture-only does not
instantiate or launch the captured CUDA body, sample output tokens, or consume
model latency measurements. Model loading, CPU-side preparation, allocations
and input copies can still execute. It requires the target weights, native
libraries and a compatible NVIDIA GPU/driver. This is a **source-assisted,
GPU-dependent route**, not a CPU-only independent predictor.

Live diagnostics actually execute the model. They are separate validation,
never a stage of the automatic structure-preparation tool.

## Build dependencies and restoration

The validated Windows build uses VS 2022 Build Tools/MSVC 14.44.35207,
CUDA 12.8.1 at `E:\cuda`, CMake/Ninja, Release, `GGML_CUDA=ON`,
`GGML_CUDA_GRAPHS=ON`, `CMAKE_CUDA_ARCHITECTURES=120a-real`, with both
`GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` off. The include uses
`cudaGraphGetEdges_v2` and Windows driver introspection from `nvcuda.dll`.
Other compiler/platform combinations have not been established by this run.

Local source: `F:\codex_project\_runtime_sources\llama.cpp`.
The configured build is `build-native-5080-sm120` below it. The separate
`build-native-5080-sm120-graph-diag\bin` directory holds diagnostic binaries;
it is a copied binary bundle, **not** a separate configured CMake build tree.

Rebuilding the existing local diagnostic bundle:

1. Preserve `ggml/src/ggml-cuda/ggml-cuda.cu` and all original `bin` files before
   changing them. This run kept `ggml-cuda.cu.before_graph_trace` and
   `original_native_bin` under
   `F:\codex_project\_scratch\37-native-validation-raw\cuda_graph_2026-10-08`.
2. In the matching source checkout, run `git apply --check` and `git apply`
   with this repository's `tools/native_cuda_graph_trace.patch`. Copy
   `tools/native_cuda_graph_trace.inc` beside `ggml-cuda.cu`; the patch uses a
   relative include.
3. The actual local build commands, from a VS x64 developer environment, were:

   ```bat
   call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   "F:\codex_project\_scratch\llama-cpp-build-env\Lib\site-packages\cmake\data\bin\cmake.exe" --build "F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120" --target llama-server -j 12
   ```

   Later instrumentation-only changes used `--target ggml-cuda` on that same
   build. A fresh machine must first configure the matching dependencies and
   architecture; the existing `CMakeCache.txt` is not portable.
4. Compile the harness against the matching native import libraries:

   ```bat
   cl /nologo /EHsc /O2 /MD /std:c++17 /I"F:\codex_project\_runtime_sources\llama.cpp\include" /I"F:\codex_project\_runtime_sources\llama.cpp\ggml\include" "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\tools\cuda_graph_structure_probe.cpp" /Fe:"F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120\bin\cuda_graph_structure_probe.exe" /Fo:"F:\codex_project\_scratch\37-native-validation-raw\cuda_graph_2026-10-08\cuda_graph_structure_probe.obj" /link "F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120\src\llama.lib" "F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120\ggml\src\ggml.lib" "F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120\ggml\src\ggml-base.lib"
   ```

5. Copy the probe/server and their matching DLLs to the separate diagnostic
   directory. Restore the preserved original source and original timing
   executable/DLLs; remove only the added include/probe artifacts from the
   original source/bin. Check the source diff without resetting unrelated
   changes. Rebuilding the restored original tree is another way to produce
   uninstrumented binaries; never mix the two DLL bundles.

The original checkout and timing binaries were restored after this run.
Recorded `.cmd` files in the raw directory document local history; the patch,
include and harness in this repository are the maintained build inputs.
A source revision tag does not independently establish an arbitrary supplied
binary's identity: use the corresponding instrumented build.

## Automatic preparation without prior native measurement

`compile_cuda_graph_structure.py` takes any supported GGUF path and its imported
base scenario. It reads metadata/tensor geometry, checks the complete model
graph against the scenario, then runs separate **dry** and **capture-only**
processes. It never reads `native_*.json`, live events or model latency data.

Example, from this repository's root:

```powershell
.\.venv\Scripts\python.exe tools/compile_cuda_graph_structure.py `
  --model F:/codex_project/37_LLMsim/models/Qwen3-0.6B-f16.gguf `
  --scenario docs/cuda_graph_validation_2026-10-08/scenario_qwen3_0_6b_f16_512_128.json `
  --source-root F:/codex_project/_runtime_sources/llama.cpp `
  --probe F:/codex_project/_runtime_sources/llama.cpp/build-native-5080-sm120-graph-diag/bin/cuda_graph_structure_probe.exe `
  --output-dir F:/codex_project/_scratch/new-graph-structure `
  --prompt-tokens 512 --output-tokens 128 --context 768 --warmups 2 --repetitions 1 `
  --cuda-device 0
```

Use a new output directory. Output contains `dry.jsonl`, `capture.jsonl`, the
two logs, and `program_fragment.json`. Failed runs preserve logs but do not
publish a completed fragment. The producer writes the same explicit
model/config/request contract into both newly generated JSONL files and the
fragment. A changed scenario requires recompilation, not relabeling old files.
The fragment references those files by path; it is not a portable bundle.

Supported scope is Qwen3/Qwen35 decoder text models, one CUDA GPU, one sequence,
no MTP, prompt 1–512 tokens, positive output length and
`prompt + output <= context`. Context must be a multiple of 256 to match native
padding. Explicit valid GGUF BOS and EOS IDs are required rather than guessed
tokenizer defaults. Fixed harness settings are GPU layers −1, batch/ubatch 512,
16 threads in both stages, FP16 KV, Flash Attention off, unified KV,
KQV/operator offload on, and graph reuse enabled. The requested physical CUDA
device is the sole device exposed to each child. Unsupported inputs fail.
The current bundled producer also checks `nvidia-smi` identity and supports
only the RTX 5080/sm120 scenario profile on a machine with one physical RTX 5080,
compute capability 12.0 and driver 617.14. Multi-GPU ordinal mapping is not
qualified by this tool. A different GPU, driver, or simulated hardware profile
requires a separately supported build/dispatch contract; it is rejected here.

`prepare_cuda_graph_cases.py` now declares the pinned server's F32 host-logit
and temperature-zero CPU sampler settings directly: top-k 40, top-p .95,
min-p .05, min-keep 0. Preparing those settings no longer requires a completed
native measurement. Later paired validation must still check the observed
server settings. Its existing `--attach-case`, `--program-fragment`, and
`--experimental-costs` options attach structure and independent runtime costs;
that local report utility's case list remains the five validation cases.

## Harness invocation and mode semantics

```text
cuda_graph_structure_probe MODEL.gguf PROMPT_TOKENS OUTPUT_TOKENS CONTEXT WARMUPS REPEATS
```

Startup first submits the available BOS/EOS pair (`model_warmup`), clears
memory, then submits the server's two-token sequence-removal startup probe
(`model_seq_rm_probe`). Warmup and ordinary requests use fixed token ID 0,
never sampled output, and each completed sequence is cleared. Prefill produces
the first output position, leaving `OUTPUT_TOKENS - 1` decode calls. For the
five validated models, 512/128 with two warmups and one ordinary request gives
`2 + 3 * 128 = 386` CUDA backend invocations.

Set `HETEROLLM_CUDA_GRAPH_TRACE` before launch. Its JSONL file is overwritten
when first opened. Mode flags test variable **presence**, so remove unused
flags instead of setting them to `0`:

| Mode | Flag | Behavior |
| --- | --- | --- |
| Dry | `HETEROLLM_CUDA_GRAPH_DRY_RUN=1` | Records GGML properties, returns before CUDA lifecycle decisions/evaluation. |
| Capture-only | `HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY=1` | Captures and exports CUDA nodes/edges; no executable instantiate/update/launch. |
| Live validation | Neither flag | Executes the model, recording actual direct/capture/update/replay events. |

Dry and capture-only cannot be combined. Both still load the model and may
allocate GPU memory/copy inputs. Unsupported stream-capture paths fail.
The automatic wrapper never selects live mode and changes only child-local
environment variables. Windows labels use `SetEnvironmentVariableA` and
`GetEnvironmentVariableA` across DLL runtime boundaries.

Remove `GGML_CUDA_DISABLE_GRAPHS` for live Graph-on: even `0` disables it in
this runtime. `GGML_CUDA_GRAPH_OPT` is a separate optimizer and is unset here.

## Optional server verification and timing

`diagnose_native_cuda_graph.py` **executes** the instrumented server with the
request tokens/settings from an existing native timing case. It runs the
recorded warmup count plus one ordinary request, checks lengths/cache/context
and Graph events, and discards timing values. This live validation is not a
dependency of compilation:

```powershell
.\.venv\Scripts\python.exe tools/diagnose_native_cuda_graph.py `
  --server F:/codex_project/_runtime_sources/llama.cpp/build-native-5080-sm120-graph-diag/bin/llama-server.exe `
  --case docs/cuda_graph_validation_2026-10-08/native_qwen3_0_6b_f16_graph_on.json `
  --trace F:/codex_project/_scratch/server-graph-check.jsonl `
  --output F:/codex_project/_scratch/server-graph-check.json
```

The controlled experiment explicitly passes `--ctx-checkpoints 0`; the
harness's recurrent configuration models that variant, including startup
sequence removal. Default server checkpoints are a different configuration:
in the original 27B run they split the 512-token prefill into 508+4. Those
results remain in `_checkpoint_default.json` files and are not interchangeable.

`native_benchmark.py` defaults to context **640**, Graph mode **auto**, and no
explicit checkpoint setting. This experiment overrides context to 768, checks
effective context 768, selects Graph `on`/`off`, and supplies
`--ctx-checkpoints 0`, two warmups and five repetitions. It rejects inherited
trace/dry/capture flags and uses only the original uninstrumented server for
timing. Its Graph option changes only the child environment and records it.

## Records and interpretation

`node_property` entries preserve the entire zero-initialized
`ggml_cuda_graph::node_properties` value, including tensor bytes and source
data pointers/shapes/strides. `snapshot.node_property_refs` references these
bytes losslessly. Addresses are local to a process; validation compares
identity/equality relationships across runs, not literal addresses.

`cuda_structure` includes typed nodes and dependency edges. Kernel properties
include function identity, grid/block dimensions, shared memory and cooperative
attributes. Memcpy fields are nested under `node.copy`: dimensions, memory
types, pitches, contexts and device ordinals. `src_device`/`dst_device` are
pointer addresses, not device numbers. Driver queries handle cuBLAS kernels
whose functions cannot be resolved by the runtime query. GGML node count and
captured CUDA node count are different quantities.

Live `event` records are observed validation outcomes. Neither these outcomes
nor native timing values may become prediction inputs. Successful structure
preparation does not qualify an independent cost profile or prove latency
accuracy on unmeasured structures.
