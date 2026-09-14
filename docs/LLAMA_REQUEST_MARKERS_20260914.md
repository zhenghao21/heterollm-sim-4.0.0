# llama.cpp 请求级 NVTX marker（2026-09-14）

本次改动只修改项目 37 的 `source/llama.cpp-semantic`，没有读取或修改项目 33。请求生命周期由 `tools/server/server-context.cpp` 发出，NVTX 实现集中在 `src/llama-context.cpp`，通过 `src/llama-ext.h` 提供 no-op-safe 包装。

## marker 语义

| 标签前缀 | 产生位置 | 语义边界 |
|---|---|---|
| `request_begin` | `launch_slot_with_task()` 将任务绑定到 slot 后 | 请求进入推理调度；同时开始一个异步 NVTX range |
| `prefill_begin` | slot 从 `STARTED` 即将处理首个 prompt batch 时 | 首个 prompt batch 的 host/CUDA 工作开始 |
| `prefill_end` | 最后一个 prompt batch 的 `llama_decode()` 成功并完成需要的同步后 | prompt logits 可用于采样；结束 prefill range |
| `first_token` | 首次采样 token（含 speculative accept 路径）后 | 第一个生成 token 已确定 |
| `request_end` | slot `release()` 开始时 | 请求生命周期结束；先关闭未结束的 prefill range，再结束 request range |

标签以 `|` 携带 slot 和形状信息，例如：

```text
request_begin|slot=0|prompt_tokens=2|predict_tokens=8
prefill_begin|slot=0
prefill_end|slot=0
first_token|slot=0
request_end|slot=0
```

`request_begin` 和 `request_end` 使用 `nvtxRangeStartA`/`nvtxRangeEnd`，因此不会依赖线程局部 push/pop 栈；多个并发 slot 可以安全交错。点事件使用 `nvtxMarkA`。Nsight SQLite 中点事件为 `eventType=34`（`end=NULL`），异步范围为 `eventType=60`。`request_begin` 和 `prefill_begin` 标签同时出现在点与范围中；计算 marker 间隔时只取点事件，范围用于归属 CUDA/CPU 工作。非 `GGML_CUDA_NVTX` 构建下三个包装均为空操作，默认行为和 ABI 调用路径不变。

## 构建锁定

构建目录：`source/llama.cpp-semantic/build-semantic-direct`

```text
Release / Ninja
GGML_CUDA=ON
GGML_CUDA_NVTX=ON
GGML_CUDA_FA=OFF
GGML_CUDA_GRAPHS=OFF
CMAKE_CUDA_ARCHITECTURES=120a-real
GGML_NVTX_INCLUDE_DIR=E:/cuda/include
MSVC 19.44.35207
CUDA 12.8.93
```

使用 VS DevCmd 初始化环境后执行：

```text
ninja -C build-semantic-direct llama-server -j 8
```

结果为成功。构建后的 `llama-server.exe` SHA256 为
`4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5eb829337`，`llama.dll` SHA256 为
`dba8eb57633ffa2e2d7ad35b1e856376503f5208937a3b9cd6416cbba8a7d55a`。

完整 bin 已复制到 `artifacts/multimodel_next/llama-semantic-request-markers-v1/`，构建锁定 JSON 为 `artifacts/multimodel_next/llama_semantic_request_markers_build_v1.json`。

## 验证

- `dumpbin /exports llama.dll` 可见 `llama_trace_mark`、`llama_trace_range_end`、`llama_trace_range_start` 三个导出符号。
- `E:/anaconda/python.exe -m pytest tests/test_nsys_trace_extract.py -q`：`4 passed（新增 request_markers 汇总后由采样代理继续执行 extractor 回归）`。
- `tools/extract_nsys_trace.py` 会在 `nvtx_events[]` 写入 `request_marker` 字段，供 Nsight SQLite 回放和 request boundary 校准使用。
- 当前尚未声称已有新的 Nsight marker 时间证据；需使用上述锁定 binary 重新采集同一模型/GGUF/硬件配置的 train/holdout trace 后，才能把 marker 间隔用于仿真器 request boundary 校准。

