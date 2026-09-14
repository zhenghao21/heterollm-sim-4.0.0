# 仿真器数据流完整审计与误差归因（2026-09-13）

## 审计范围

数据流按以下边界核对：GGUF header/tensor directory → ModelSpec/IR → projection descriptors → llama.cpp placement lowering → prefill/decode task graph → resource/event scheduler → TTFT/TPOT/E2E reporting。原生侧使用同一 GGUF、本机 RTX 5080、同一 ctx/batch/ubatch/threads/KV/offload 配置。

## 已确认并修复的根因

1. **GGUF tensor enum 曾把 file type 与 tensor type 混用。** IQ3_S 实际为 ggml type 21，IQ4_XS 实际为 23，IQ4_NL 为 20；旧映射会把 Qwen3.8-27B 的权重字节与量化投影算错。现在按项目内 llama.cpp `ggml.h` 的 enum、block size 和 block bytes 读取，并禁止对未对齐 tensor 使用 ceil 伪造大小。
2. **Qwen3.8-27B 的 nextn 辅助层曾进入主干。** GGUF `block_count=65`、`nextn_predict_layers=1`，llama.cpp 主干执行层是 64 层；现在保留 `n_layer_all=65`、`n_layer_nextn=1` 审计字段，ModelSpec 只生成 64 个 trunk layers，其中 48 个 linear-attention、16 个 full-attention。
3. **Qwen3.5 full-attention 的真实语义不完整。** 现已记录 head_dim=256、rotary_dim=64、Q/K RMSNorm、sigmoid query gate、qk scale=1/sqrt(256)，并用 q/k/v 精确 segment id 与 K 维 output shard。
4. **Qwen3.5 linear-attention 的权重不完整。** 现已显式保留 fused QKV、SSM output、output gate、alpha、beta projection；IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS 等格式进入 projection registry。
5. **tensor 查找可能误把 `attn_q_norm.weight` 当成 Q projection。** 现在使用明确的 tensor suffix 绑定，避免目录顺序改变语义。
6. **lm-head 的显式 CPU/GPU placement 曾被忽略。** planner 只在 CIM 情况尊重配置，CPU/GPU 会回退到当前 rank；因此 `gpu_layers=0` 仍会把 lm-head 放到 GPU。现在 CPU/GPU 显式 target 直接生效，非计算组件 fail-closed，并有回归测试。

## 数据流仍存在的误差来源

- llama.cpp direct holdout trace 中，kernel 时间主要由没有可靠 NVTX 语义绑定的 MMQ kernel 构成。当前统计中约 77.5% 的 kernel 时间落入 `unknown`，主要形式为 `mul_mat_vec_q`；quantize 约 11.1%，normalization 约 8.0%，已明确映射的 QKV 约 3.1%。因此不能把少量 QKV operator 的校准系数推广到全部 FFN/KV/lm-head MMQ。
- `cudaStreamSynchronize` 的平均 API 时间约 0.078–0.112 ms/次，当前只进入 calibration metadata，没有作为独立 runtime action 下沉。它会造成少量 TTFT/E2E 低估，但 trace 中同步调用彼此重叠，不能把总和逐算子相加。
- simulator 使用 native 实际 prompt/output token 数构建场景，所以 token parity 是 bound/native-derived，不能当作独立 tokenizer 验证。提前 EOS 的 cell 会被排除出 TPOT 聚合。
- TTFT/E2E 的 native 与 simulator 边界仍不完全相同；TPOT 在 output token>1 时最适合比较。所有 timing 结论仍标为 diagnostic-only。

## 修复后抽样复测

- Qwen2.5-0.5B、full GPU、medium prompt、predict=8：TTFT `-77.58%`、TPOT `-38.07%`、E2E `-49.99%`。
- Qwen2.5-0.5B、CPU-only、short prompt、predict=8：TTFT `-61.42%`、TPOT `-47.50%`、E2E `-49.77%`。该结果比旧矩阵更低，是因为修复后 CPU-only 不再错误计入 GPU lm-head/权重搬运；旧的接近零误差结果不能继续使用。
- Qwen3.8-27B、CPU-only、short prompt、predict=2：TTFT `-43.59%`、TPOT `-31.12%`、E2E `-38.03%`。修复 65→64 trunk、真实 IQ bytes 和 attention projection 后，误差较旧 smoke 的 `-45.42%/-36.90%/-41.50%` 收窄。

## 验证结果

在显式设置项目 37 `PYTHONPATH`、避免项目 33 dist 污染的情况下：

```text
797 passed, 1 skipped
```

专项结果：

```text
GGUF parity + projection：11 passed
planner/output placement：9 passed
```

Windows 全量测试过程中仍会打印一次 pyarrow `0xc0000139` 的 native loader 噪声；pytest 最终结果为全绿，这属于测试环境依赖加载问题，不是仿真器断言失败。

## 结论与下一步

目前能够确认的误差已从几何、权重物理字节、层数、attention 语义和 lm-head 放置错误中分离出来。剩余 30–80% 量级误差主要来自 shape-aware MMQ kernel 的有效吞吐与 operator 语义覆盖，而不是 roofline 单位或事件调度算术。下一步应在 direct semantic build 中让每个 `mul_mat_vec_q` 的 launch correlation 带上稳定 operator id，按模型×placement×prefill/decode×shape 产生系数，再把系数绑定到对应 projection invocation；同步则按真实边界计数增加独立阶段。


## 修复后热力图与样本

只使用上述修复之后重新运行的样本生成的误差数据：
[error_samples_v2.json](../artifacts/audit_20260913/error_samples_v2.json)。

- [TTFT 绝对误差热力图](../artifacts/audit_20260913/heatmaps_v2/ttft_ms_abs_heatmap.png)
- [TPOT 绝对误差热力图](../artifacts/audit_20260913/heatmaps_v2/tpot_ms_abs_heatmap.png)
- [E2E 绝对误差热力图](../artifacts/audit_20260913/heatmaps_v2/e2e_ms_abs_heatmap.png)
- [TTFT 有符号误差热力图](../artifacts/audit_20260913/heatmaps_v2/ttft_ms_signed_heatmap.png)
- [TPOT 有符号误差热力图](../artifacts/audit_20260913/heatmaps_v2/tpot_ms_signed_heatmap.png)
- [E2E 有符号误差热力图](../artifacts/audit_20260913/heatmaps_v2/e2e_ms_signed_heatmap.png)

此前的 22-cell 多模型矩阵基于旧的 GGUF enum/nextn/lm-head 数据流，已降级为历史基线，不再与修复后样本混合聚合。


## 本轮继续下沉的改动

- `build_matching_scenario()` 现在按 `gpu_layers` 设置 KV owner：CPU-only 使用 `hostmem0`，CUDA offload 使用 `hbm0`，并在 decode task 中保留 KV read/append 的物理字节。partial offload 仍是单 cache owner 近似，尚未冒充逐层 `model.dev_layer()`。
- semantic CUDA graph/fused path 增加 NVTX scope，绕过普通 `ggml_cuda_compute_forward()` 的 fused MMQ/MMVQ launch 现在也有显式 owner 证据；extractor 为每个 operator 保留稳定 `operator_id`/`semantic_operator_id`。旧 trace 不会自动改变，需用新 semantic binary 重采集后才能确认 unknown 覆盖率下降。
- stage calibration 已接入 planner，但默认关闭；只有显式 `--apply-stage-calibration`、精确 stage/phase/shape/projection 命中时才改 compute demand，unknown、缺失 shape、identity mismatch 均保持原始分析值。同步系数仍只接受显式 `sync_boundary_count`。

新增验证：

```text
tests/test_native_kv_placement.py + llama scenario/writeback：11 passed
tests/test_native_evidence.py + runtime parity + KV placement：23 passed
```

修复后 Qwen2.5 CPU-only short 的 sim KV owner 已切至 hostmem；Qwen3.8-27B 的主干层数、IQ 物理字节和 attention descriptors 已在 v4 smoke 中通过。
