# B-10 Host Boundary Marker Audit (2026-09-13)

## 状态

已完成源码审计与开发编译门控 marker 补丁；默认 binary 与独立 `LLAMA_SERVER_HOST_TRACE` 开发 binary 均已编译；已完成一条 Qwen2.5-0.5B NSYS 开发采集。未重测 Qwen3.8-27B，未使用目标 LLM TTFT/TPOT/E2E 拟合 host 参数。

## Marker map

| boundary | source location | marker |
|---|---|---|
| HTTP handler/JSON request | `tools/server/server.cpp`, `ex_wrapper` | `http.handler.begin`, `http.handler.end` |
| input tokenize | `tools/server/server-context.cpp`, `tokenize_cli_input` | `json.tokenize` |
| admission/slot launch | `server-context.cpp`, `launch_slot_with_task` | `slot.launch` |
| response queue | `server-context.cpp`, every `queue_results.send` path | `response.queue` |
| stream sink write | `server-stream.cpp`, `stream_pipe_producer::write` | `sink.write` |

Markers call the existing `llama_trace_mark()` API and are gated by `LLAMA_SERVER_HOST_TRACE`; request/prefill ranges use the matching gated range macros. Non-trace builds compile to no-op. The underlying API emits NVTX only with `GGML_CUDA_NVTX`.

## Evidence / blocked reason

The checked-in `build-semantic-direct` tree is present, but this host has no `cmake` executable (`cmake --build ...` fails with command-not-found). Therefore no developer binary or NVTX/NSYS run was produced. Host parameter calibration remains disabled and must stay zero until same-binary repeated traces plus independent host microbenchmarks are available.

