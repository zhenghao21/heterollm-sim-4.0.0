# Semantic direct 重建与 fused MMQ owner 覆盖（2026-09-13）

## 构建

- 构建目录：`source/llama.cpp-semantic/build-semantic-direct`
- 生成器：Ninja
- 配置：Release，CUDA arch `120a-real`
- `GGML_CUDA=ON`
- `GGML_CUDA_GRAPHS=OFF`
- `GGML_CUDA_NVTX=ON`
- `GGML_CUDA_FA=OFF`
- 编译器：MSVC 19.44 + CUDA 12.8
- binary SHA256：`DD8B158CEF066071E133CA8EBCC49A672D50DAAF3FCCD36676BCE76E174D54C0`
- 日志：[semantic_direct_rebuild_20260913_vs.log](../artifacts/semantic_direct_rebuild_20260913_vs.log)

首次直接调用 cmake 未加载 VS Developer 环境，nvcc 报 `Cannot find compiler cl.exe in PATH`；随后通过 `VsDevCmd.bat -arch=x64 -host_arch=x64` 重新执行并成功完成 115 个目标。

## 采集配置

使用新 binary 启动 llama-server，并由 Nsight Systems `--trace=cuda,nvtx,wddm` 采集。模型为 Qwen2.5-0.5B Q4_K_M；prompt token 数 9；`ctx=512,batch=64,ubatch=64,threads=16,gpu_layers=-1,np=1,FA=off,mmap,on op-offload,on KV offload,seed=42,predict=2`。warmup 为 1 token，formal completion 为 2 token，采集仅覆盖 formal 请求。

## 产物

- profile：[native_profile_semantic_direct_rebuild_20260913.json](../artifacts/native_profile_semantic_direct_rebuild_20260913.json)
- SQLite：[native_profile_semantic_direct_rebuild_20260913.sqlite](../artifacts/native_profile_semantic_direct_rebuild_20260913.sqlite)
- trace：[native_profile_semantic_direct_rebuild_20260913.trace.json](../artifacts/native_profile_semantic_direct_rebuild_20260913.trace.json)
- 覆盖摘要：[native_profile_semantic_direct_rebuild_20260913.coverage.json](../artifacts/native_profile_semantic_direct_rebuild_20260913.coverage.json)

## 结果

| 类别 | 数量 | semantic owner 匹配 | 时长覆盖 |
|---|---:|---:|---:|
| 所有 CUDA kernel | 1348 | 1348 (100%) | 100% |
| MMQ (`mul_mat_q`,`mul_mat_vec_q`) | 478 | 478 (100%) | 100% |
| quantize | 313 | 313 (100%) | 100% |

trace 共 2812 events、1499 NVTX ranges、0 graph node。每个 kernel 都携带稳定 `semantic_operator_id`，例如 `MUL_MAT:ffn_gate-16`；MMQ kernel 通过同一 NVTX scope 关联到对应的 projection invocation，未使用时间重叠推断。旧 `native_profile_semantic_direct_true_v1.trace.json` 中 MMQ 仅 183/614、全 kernel 1012/2166 匹配，新 binary 显著提高了 owner 可见性。

注意 extractor 的 `operator_id` 对 kernel 仍保留在 `semantic_operator_id` 字段（`operator_id` 仅在 NVTX event 上），这是现有 schema 设计；下游应使用 `semantic_status=matched` 与 `semantic_operator_id`。

## Stage classifier v2

重新抽取 trace 时，在 NVTX/runtime correlation 完成后再执行一次语义 stage refine。`stage=unknown` 只会在有 `semantic_status=matched` 时按稳定 owner 名称保守映射；`node_N` 等无语义名称继续保留 unknown。最新 stage 计数为：attention_qkv 384、ffn 211、attention_output 168、kv 48、lm_head 6、normalization 98、quantize 313、unknown 120。

最新留出校准：[native_semantic_calibration_direct_rebuild_v3.json](../artifacts/native_semantic_calibration_direct_rebuild_v3.json)。其 unknown/uncovered 为 0；313 个量化/非目标阶段 kernel 记录为 explicit_excluded，不阻塞四个目标阶段的 calibrated 状态。
