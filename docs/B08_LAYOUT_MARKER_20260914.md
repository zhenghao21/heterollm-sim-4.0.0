# B-08 显式 layout marker 与 extractor 验证（2026-09-14）

## 修改

在 `source/llama.cpp-semantic/ggml/src/ggml-cuda/ggml-cuda.cu` 的 `ggml_cuda_nvtx_scope` 中增加 `layout=` 字段。分类只依赖 tensor stride 语义：`contiguous`、`transposed`、`permuted`、`strided`；不写入地址或其他场景答案。`tools/extract_nsys_trace.py` 将该字段保存为 `semantic_layout`。

## 固定身份与场景

- llama-server.exe SHA-256：`4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337`；ggml-cuda.dll SHA-256：`eef90dd2e43863d2f5692ee6818c65ca036e359065332b0caf3db715b11b1f99`。
- GGUF SHA-256：`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`；Nsight SHA-256：`9e4d325628a774f358be8ff068b8d1ff284416479b5f72a44fcfe632d046ff6b`。
- `ctx=512,batch=64,ubatch=64,threads=16,gpu_layers=-1,np=1,KV=f16,FA=off,CUDA graphs=OFF,warmup=1`。
- 仅采集 Qwen2.5-0.5B：prompt/output=16/4 与 32/8；无 27B native。

## 结果

- `b08_qwen25_layout_m16_t4_v1.trace.json`：kernel 2320，matched 2320，layout missing 0，layout 分布 {'contiguous': 2320}，联合 key 1266。
- `b08_qwen25_layout_m32_t8_v1.trace.json`：kernel 4384，matched 4384，layout missing 0，layout 分布 {'contiguous': 4384}，联合 key 1362。
- 两条 trace 的含 layout 联合 key 交集/并集：462/2166（21.33%）；layout 字段本身 100% 可提取且当前均为 contiguous。
- calibration coverage=covered（owner/stage/phase/shape 均完整），但不同 M/T 的 kernel shape 联合覆盖仍不足，不能启用全局 stage/operator rate。

## 判定

B-08 的显式 layout 语义链路已通过：源码 marker → Nsight NVTX → extractor `semantic_layout` 全部闭合。21.33% 的跨 M/T 联合 key 交集说明 layout 字段修复没有消除 shape/kernel 路径稀疏；候选成本 profile 继续 fail-closed。下一步只能对 exact `(stage,phase,shape,dtype,layout,kernel_family)` 命中应用，未命中保持分析回退或补采独立 shape 证据。

## 产物

- `artifacts/development/b08_qwen25_layout_m16_t4_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b08_qwen25_layout_m32_t8_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b08_qwen25_layout_calibration_v1.json`
- `artifacts/development/b08_qwen25_layout_evidence_v1.json`

结论：B-08 layout marker/extractor 完成并通过小模型验证；全局成本 profile 仍因 shape 联合覆盖不足阻断。