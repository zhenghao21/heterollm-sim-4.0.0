# 误差继续下沉前的数据流审计（2026-09-14）

## 审计目标与证据边界

本次审计只使用项目 37 已生成的矩阵、仿真 trace、GGUF parity 和 semantic llama.cpp trace；没有重新启动原生服务，也没有把历史矩阵与修复后矩阵混合。覆盖链路为：

```text
GGUF geometry / tensor bindings
→ build_matching_scenario()
→ llama runtime lowering
→ KV / weight placement
→ prefill/decode batch lowering
→ resource DAG / critical path
→ TTFT / TPOT / E2E
```

误差仍按 `(simulator - native) / native × 100%` 计算。当前矩阵中的 TPOT 只对 native 实际输出 token 大于 1 的 cell 计入；提前 EOS 的 cell 只保留 TTFT/E2E 诊断值。

## 现有误差的可验证形态

`artifacts/multimodel_20260913/native_multimodel_matrix_v3.json` 的 22 个有效 cell 显示，仿真器大部分场景低估原生耗时：

- Qwen2.5-0.5B：TTFT 中位绝对误差约 75.94%，TPOT 36.58%，E2E 48.16%。
- TinyLlama-1.1B：TTFT 64.40%，TPOT 29.55%，E2E 44.44%；medium 输入提前 EOS 的 3 个 TPOT cell 已排除。
- Qwen3.8-27B：现有 4 个 cell 的 TTFT 约 44.7–78.0% 低估，E2E 约 39.4–66.8% 低估。

抽查 `build_matching_scenario(23, 8, ...)` 的批次成本发现：

- Qwen2.5 CPU-only 每个 decode batch 的总服务约 5.67 ms，但 `coverage.kv_cache` 的 task 数虽存在，bytes/latency 在批次 coverage 中为 0；KV traffic 只出现在 `result.kv_metrics` 的独立统计中。
- semantic direct holdout 中约 77.5% GPU kernel 时间仍是无法绑定到具体 projection 的 MMQ kernel（`mul_mat_vec_q` 等），约 11.1% 是 quantize，约 8.0% 是 normalization，明确 QKV 约 3.1%。因此不能用单一全局倍率解释 FFN、KV 和 lm-head。
- native trace 的 launch/sync 调用与 GPU kernel 存在重叠；把 API 总时长逐算子相加会重复计时。

## 已直接修复的数据流缺口

### KV cache 的运行时放置

此前 `build_matching_scenario()` 总是沿用参考场景的 `hbm0` KV cache。对于 `llama.cpp -ngl 0`，没有 CUDA transformer 层，KV 页应在主机 DRAM；继续使用 hbm0 会让 CPU-only 场景的 cache owner 与原生运行时不一致，并使 KV 读/追加无法进入正确的 CPU 资源路径。

现在 `tools/native_llama_compare.py` 按 runtime placement 选择：

```text
gpu_layers == 0  → hostmem0
gpu_layers != 0  → hbm0
```

这是单一 `cache_component` 合同下的安全近似。原生 llama.cpp 在 partial
placement 时按 `model.dev_layer(il)` 为每一层选择 KV buffer，因此 CPU 前缀层
和 GPU 尾部层可能同时存在。要继续降低 partial 场景误差，需要把 KV policy
扩展为按 layer 的 component map；在此之前，partial cell 应标记为
`kv_placement_approximation`，不能声称与原生逐层 residency 完全等价。

同时将选择写入 `placement.metadata.llama_cpp_kv_component`，便于结果审计。新增 `tests/test_native_kv_placement.py` 验证：

- CPU-only 的 cache component 是 `hostmem0`；
- CUDA offload 的 cache component 是 `hbm0`；
- prefill/decode 产生非零 `physical_decode_read_bytes` 和 `physical_decode_append_bytes`。

专项回归结果：`2 passed`。

在 23 prompt tokens、8 output tokens 的仿真 smoke 中，修复后 KV 指标为：

```text
physical_decode_read_bytes    = 2,236,416
physical_decode_append_bytes  =    86,016
```

这一步修复了 placement 语义，但不会凭空把所有 KV traffic 变成 GPU kernel 时间；`kv_metrics` 与 operator coverage 仍需在后续 lowering 中保持“统计流量”和“资源服务”两条路径的一致映射。

## 尚未修复、但可直接验证的误差路径

### 1. KV traffic 尚未进入批次 resource DAG

`kv_metrics` 已记录逻辑/物理读写，但 batch coverage 对 `kv_cache` 的 bytes/latency 仍为零或缺失。这是最明确的任务图漏项。下一步应在 batch lowerer 中为每个阶段产生显式 KV action：

```text
prefill:  KV write(prompt_tokens)
decode:   KV read(context_tokens) + KV append(1)
```

action 的目标组件必须来自 `kv_policy.cache_component`，dtype 使用 `kv_policy.dtype`（当前为 F16），字节数由 `n_head_kv × head_dim × 2 × token_count × layer_count` 推导。若 KV traffic 已被同一 GEMM roofline 的 operand bytes 包含，应在 metadata 中标记 `included_in_gemm_roofline`，避免二次计时；否则将其作为独立 memory action 进入 CPU/HBM 资源 DAG。回归应验证每个 decode step 的 read bytes 随 context 增长、append bytes 恒定且不重复计入。

### 2. CPU profile 与原生线程实现仍是主要低估源

仿真器把 CPU profile 固定为 16 个 pipeline workers、4.3 GHz 和分析带宽；native llama.cpp 的量化矩阵乘法使用具体 AVX2/线程分块路径，无法由 nominal frequency 直接推出有效吞吐。建议从 CPU-only trace 先拟合 `mul_mat_vec_q` 的形状分桶（`M=1` decode 与 `M=T` prefill 分开），并保留线程数、batch size、量化格式作为 key。只在至少一个独立 holdout cell 有证据时才写入 profile，避免把短输入 launch overhead 拟合进 kernel 吞吐。

### 3. GPU MMQ 语义绑定仍不完整

direct semantic build 已能记录 NVTX phase，但大量 `mul_mat_vec_q` 仍没有稳定 operator ID。下一步应在调用 MMQ kernel 的上层保留 invocation id、projection（Q/K/V、up/gate、down、lm-head）、shape、dtype、phase，并让 CUPTI correlation 通过该 id 归属；校准 key 至少包含：

```text
model_sha × gpu_layers × phase × projection × M × N × K × quant_type
```

未知 kernel 不得平均摊给已匹配的 QKV/FFN/KV。

### 4. launch/sync 只能按边界计入

native `cudaStreamSynchronize` 与 launch API 的时间应先按 prefill/decode 边界聚合，再作为 runtime frontend/synchronization action 放入临界路径。必须使用不重叠区间或 CUPTI correlation 去重；不能把同步 API duration 直接累加到每个 operator。

### 5. token 与计时边界继续保持分离

当前比较脚本用 native 实际 token 数回填 simulator 场景，这能隔离几何误差，但 token parity 不是独立验证。后续矩阵应同时保存 requested、simulator declared、native actual、early EOS，并将 `output_tokens <= 1` 的 TPOT 标记为 ineligible。TTFT/E2E 仍需注明 server boundary 与 client wall boundary 的差异。

## 推荐的下一步验证顺序

1. 先重跑修复 KV placement 后的 4 个代表 cell（Qwen2.5 CPU-only/full、TinyLlama full、Qwen3.8-27B CPU-only），确认误差变化来自 placement 而不是 native 噪声。
2. 为 KV read/append 增加显式 resource-DAG action，并用 `kv_metrics` 与 batch coverage 双向守恒检查。
3. 对 direct trace 的 MMQ kernel 增加 invocation/projection 绑定，先校准 decode `M=1`，再校准 prefill `M=T`。
4. 最后加入 launch/sync 边界 action，再以留出 cell 生成新的绝对误差热力图。

在完成第 2、3 步前，不建议使用全局倍率或按模型整体缩放；那会把 KV、量化转换、MMQ kernel、launch 和同步混为一个系数，无法解释不同输入长度与 GPU placement 的误差变化。

## 2026-09-13 继续下沉：semantic direct v6 与逐层 KV

semantic direct binary 已通过 VS Developer 环境重建，配置固定为
`GGML_CUDA_GRAPHS=OFF`、`GGML_CUDA_NVTX=ON`、CUDA arch `120a-real`。
新 trace 的 1348 个 CUDA kernel 全部带有显式 semantic owner；MMQ/MMVQ
(`mul_mat_q`/`mul_mat_vec_q`) 478 个全部匹配。阶段 refine 只根据
`semantic_operator_id` 和 kernel family 推断，`node_N` 仍保留 unknown。

校准 profile [native_semantic_calibration_direct_rebuild_v6.json](../artifacts/native_semantic_calibration_direct_rebuild_v6.json)
使用最长父 NVTX operator scope 作为 wall-time 证据，同时保留 kernel-time
证据；新增 attention-output 与 normalization stage，量化没有稳定 owner，
因此不伪造 quantize 系数。profile 的 coverage 为 `covered`，train/holdout
identity 一致，未知 kernel 为 0，显式排除的非目标 stage 为 313 个。

partial offload 的 KV cache 已下沉为逐层 owner：CPU 前缀层使用 `hostmem0`，
GPU 尾部层使用 `hbm0`，`gpu_layers=0` 和 `-1` 也分别有全 host / 全 GPU
回归断言；每个 decode layer 的 KV read/append 事件写入实际 owner 和物理字节。

在同一 Qwen2.5-0.5B、RTX 5080、`ctx=512,batch=64,ubatch=64,threads=16,
gpu_layers=-1,FA=off` 配置下，使用 operator-wall stage 校准和 8 us launch
校准得到两组输入结果：短输入的 TTFT/TPOT/E2E 相对误差为约
`-30.6% / +10.2% / +2.4%`，中等输入约为 `-43.0% / +2.9% / -6.4%`。
对应绝对误差约为 `2.0–3.3 ms / 0.1–0.4 ms / 0.9–2.4 ms`。E2E 和 TPOT
已进入小误差范围；TTFT 的剩余差异主要来自 native prompt-eval/server
边界和 warmup，不应继续用 decode 系数覆盖。8 us launch 值已绑定到该
模型/runtime/hardware 证据，不能跨模型复用。

本轮新增的 calibration gate、wall-time stage、逐层 KV 修复以及 planner
phase lowering 通过全量回归：`807 passed, 1 skipped`。测试时仍会出现一次
既存的 Windows `pyarrow` `0xc0000139` loader 噪声，但不影响 pytest 最终结果。

下一步是分别采集 Qwen3.5-0.8B、TinyLlama 和 Qwen3.8-27B 的相同 direct
semantic trace，按模型 SHA、runtime fingerprint 和 `gpu_layers` 建立独立
profile，再生成多模型多输入热力图；当前 v6/v7 profile 不会跨模型套用。

## 多模型实测推进（同一硬件与 runtime）

已经为 TinyLlama-1.1B 和 Qwen3.5-0.8B 完成同一 direct binary 下的实测。
TinyLlama 的 `Hi.`、predict=2 train/holdout trace 有完整 semantic owner，
生成了专属 [tinyllama profile](../artifacts/multimodel_next/tinyllama_semantic_calibration_v1.json)。
应用该 profile 加 8 us launch 校准后，`Hi.`、predict=8 的一轮结果为：

```text
TTFT  -5.5%  (绝对误差 0.25 ms)
TPOT  +7.8%  (绝对误差 0.30 ms)
E2E   +5.9%  (绝对误差 1.86 ms)
```

Qwen3.5-0.8B 也已经完成独立 train/holdout semantic trace 和 profile：

- [Qwen3.5 train trace](../artifacts/multimodel_next/qwen35_train_hi_p2_trace_v1.json)
- [Qwen3.5 holdout trace](../artifacts/multimodel_next/qwen35_holdout_hi_p2_trace_v1.json)
- [Qwen3.5 profile](../artifacts/multimodel_next/qwen35_semantic_calibration_v1.json)

Qwen3.5 profile 的 semantic stage 留出误差大多在 2% 左右，lm-head 因为样本
很少仍然不稳定。加同硬件的 8 us launch 校准后，`Hi.`、predict=8 的一轮
结果为：

```text
TTFT  -28.8%
TPOT  -13.0%
E2E   -15.3%
```

这一轮的绝对误差为 `1.87 ms / 0.71 ms / 6.85 ms`。Qwen3.5 的 E2E
仍未达到 Qwen2.5 和 TinyLlama 的小误差水平；该结果保留为当前真实差距，
后续需要增加 mixed-attention 的同步与 lm-head 留出样本，不能用 launch
系数继续调大来掩盖模型结构差异。

这说明 Qwen3.5 的剩余差距主要来自混合 attention 调度、同步边界和 lm-head
低样本误差，不能继续沿用 TinyLlama 或 Qwen2.5 的 stage 系数。

当前多模型、多输入绝对误差热力图：

![多模型绝对误差热力图](../artifacts/multimodel_next/error_heatmap_v5.png)

对应结构化数据在 [error_summary_v5.json](../artifacts/multimodel_next/error_summary_v5.json)。
其中 TinyLlama 使用专属 semantic profile；Qwen3.5 的当前条目仍是独立
semantic profile 加硬件 launch 校准，不能与未校准历史矩阵混合汇总。

## Qwen3.5 linear-attention auxiliary 下沉

Qwen3.5 的上一轮差距不是 FFN 或 QKV 主路径本身，而是 trace 中明确出现、
planner 尚未表示的辅助状态操作。现在已新增独立
`linear_attention_aux` stage，覆盖：

```text
SSM_CONV
GATED_DELTA_NET
q/k convolution pre-delta
conv input/state update
beta sigmoid / softplus
gate reshape / residual
```

下沉入口使用 planner 已有的 `linear_op`（`local_conv`、
`scan_recurrent_update`、`gate_norm_reduce/apply`、`residual`），不需要虚构
projection ID，也不会把 full-attention 的 CONT 算子错误归到 linear attention。
校准只替换显式 compute demand，KV、transfer 和 memory bytes 保持原模型。

相关文件：

- [Qwen3.5 v2 semantic profile](../artifacts/multimodel_next/qwen35_semantic_calibration_v2.json)
- [Qwen3.5 v2 trace replay](../artifacts/multimodel_next/qwen35_short_hi_direct_stage_launchcal_v2.json)
- [Qwen3.5 auxiliary train trace](../artifacts/multimodel_next/qwen35_train_hi_p2_trace_v1.json)
- [Qwen3.5 auxiliary holdout trace](../artifacts/multimodel_next/qwen35_holdout_hi_p2_trace_v1.json)

在同一 `Hi.`、predict=8 场景的一轮重放中，Qwen3.5 误差为：

```text
TTFT  -30.9%  / 绝对误差约 2.18 ms
TPOT   -6.3%  / 绝对误差约 0.37 ms
E2E    -9.9%  / 绝对误差约 4.75 ms
```

相较 auxiliary 下沉前，TPOT 和 E2E 都继续下降；E2E 剩余差距主要来自
mixed-attention 的同步边界和 lm-head 低样本波动，下一步应对这些边界做
独立留出，而不是继续调大阶段倍率。

## Qwen3.8-27B CPU-only 证据边界

Qwen3.8-27B 已使用项目37的 GGUF 和相同 semantic direct binary 完成
CPU-only train/holdout trace。每份 trace 有 1843 个事件，其中 128 个 CUDA
kernel、1378 个 runtime 事件、337 个 memcpy 事件。由于 `gpu_layers=0`，
主干的 48 个 linear-attention layer、FFN 和大部分 lm-head 计算在 CPU
backend 上执行，CUDA trace 只能提供少量 QKV/ROPE 证据。

对应 profile 为 [qwen38_semantic_calibration_v2.json](../artifacts/multimodel_next/qwen38_semantic_calibration_v2.json)。
它只对有足够 CUDA 证据的 attention QKV 保留校准，其余 CPU 阶段保持
blocked。predict=8 的当前 CPU-only 回放为：

```text
TTFT  -47.2%  / 绝对误差约 175.2 ms
TPOT  -34.5%  / 绝对误差约 101.6 ms
E2E   -41.6%  / 绝对误差约 276.8 ms
```

这组差距不能用 GPU profile 修复。下一步必须在 llama.cpp 的 CPU backend
采集 operator wall-time 或线程分块计时，并按 CPU linear-attention、FFN、
normalization、lm-head 独立建立 profile；在此之前不把 Qwen3.8 CPU-only
结果伪装成已收敛场景。

## Qwen3.8-27B CPU-only prompt=8 重采样

本轮按相同的 semantic direct binary、GGUF、CPU 线程数、batch/ubatch、KV
类型和 `-ngl 0` 配置重新采集了 8-token prefill 的 CPU operator trace。原生
日志显示 Qwen3.8 的 8-token prefill 被 llama.cpp CPU graph 切成两个 `M=4`
物理 invocation；因此 parity harness 将该模型的连续调度 prefill chunk 绑定
为 4，而不是把 8-token 当成一个未测量的 graph。该规则只对 65-block 的
`GGUF-qwen35`/Qwen3.8 变体启用，其他模型仍使用命令行 ubatch。

新增证据：

- [prompt8 train trace](../artifacts/multimodel_next/qwen38_cpu_trace_prompt8_train_f9.json)
- [prompt8 holdout trace](../artifacts/multimodel_next/qwen38_cpu_trace_prompt8_holdout_f9.json)
- [prompt8 semantic calibration v2](../artifacts/multimodel_next/qwen38_cpu_semantic_calibration_prompt8_f9_v2.json)

该 profile 的 train/holdout identity gate 均通过，未知、缺 phase、缺 shape
计数为零；`quantize` 仍保持 blocked，没有用倍率补齐。使用该 profile 的一次
同配置重放结果为：

```text
native：TTFT 1086.24 ms，TPOT 288.40 ms/token，E2E 3105.07 ms
sim   ：TTFT  871.51 ms，TPOT 399.84 ms/token，E2E 3670.39 ms
误差  ：TTFT -19.77%，TPOT +38.64%，E2E +18.21%
```

这次重放把 prefill 的物理 invocation 几何与 trace 对齐，TTFT 绝对误差从
约 0.6 秒降到约 0.21 秒；decode memory/compute 的残余差距仍受 CPU 频率和
线程调度波动影响。另一次保持单个 prefill graph 的对照为 E2E +6.45%，不能
与本次 chunk=4 结果混为同一场景。旧的 `qwen38_27b_cpu_hi_direct_memorycal_v3b`
在 memory-gating 修复之前生成，已标记为历史证据，不用于当前结论。

随后又把常规 GEMM 的 `GemmWorkload.m/n` 写入 planner phase metadata，使
`17408x1x1x1`、`5120x1x1x1` 等 decode shape 能命中 profile 的精确 bucket，
不再退回 phase 平均值。该改动通过 28 项专项测试；同一场景的新回放为：

```text
native：TTFT 1188.12 ms，TPOT 305.07 ms/token，E2E 3323.63 ms
sim   ：TTFT  871.51 ms，TPOT 419.40 ms/token，E2E 3807.32 ms
误差  ：TTFT -26.65%，TPOT +37.48%，E2E +14.55%
```

这组结果比 phase 平均值版本的 E2E（+18.21%）更接近原生；TPOT 仍受单机
CPU 线程调度波动影响，后续应使用多次重复的中位数，而不是继续扩大任何
单一阶段倍率。

为确认误差不是单次 native 抖动，又固定同一配置完成 7 次独立测量，汇总见
[qwen38_cpu_prompt8_median_summary_v1.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_median_summary_v1.json)。该批次使用精确 shape 下沉前的
phase-calibrated replay，native 中位数为 TTFT `1141.42 ms`、TPOT `325.78
ms/token`、E2E `3408.83 ms`，simulator 为 `871.51/399.84/3670.39 ms`，
对应 `-23.65%/+22.73%/+7.67%`。native CV 分别为 `3.26%/5.48%/3.97%`，
说明剩余差距主要来自模型数据流和服务边界，而不是偶发的系统噪声；精确
shape 版本的后续中位数仍需在当前代码上重新采集。
在当前精确 shape 代码上又完成了 r08–r10 三次独立测量，汇总见
[qwen38_cpu_prompt8_exactshape_median_summary_v1.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_exactshape_median_summary_v1.json)。native 中位数为 TTFT `1195.02 ms`、TPOT `349.55 ms/token`、E2E `3644.60 ms`，simulator 为 `871.51/419.40/3807.32 ms`，误差分别为 `-27.07%/+19.98%/+4.47%`，绝对误差约 `323.52 ms/69.85 ms/162.72 ms`。三次 native 的 CV 为 `0.56%/3.08%/2.23%`。E2E 数值更接近，但 TTFT 低估与 TPOT 高估存在抵消，因此仍不能称为三项均已收敛；剩余差距主要集中在 TTFT 的请求边界和 decode 的 CPU 服务时间。
精确 shape 下沉后再次执行完整回归，结果为 `822 passed, 1 skipped`；仍有
一次既有的 Windows pandas/pyarrow/ortools 原生依赖加载噪声，但不影响最终
测试完成。重复权重读取和 dotted-phase 修复之后的这次全量回归仍为
`822 passed, 1 skipped`。

随后审计 decode 差距时发现并修正了两个真实的数据流问题：连续调度任务名
是 `cohort-*.decode...`，旧的 `startswith("decode")` 判断没有覆盖它；另外
`estimate_cpu_gemm` 已经把物理权重读计入 backing memory，`_add_rank_gemm`
又追加了一次相同字节，造成 CPU DRAM 流量重复。现在 phase 识别统一经过
`_execution_phase_from_name()`，CPU GEMM 只保留一次 backing read，并新增
了 phase/bytes 回归。

修复后当前代码的 Qwen3.8 prompt=8 单次重放为：

```text
native：TTFT 1138.20 ms，TPOT 348.31 ms/token，E2E 3576.38 ms
sim   ：TTFT  919.13 ms，TPOT 300.20 ms/token，E2E 3020.53 ms
误差  ：TTFT -19.25%，TPOT -13.81%，E2E -15.54%
```

此处的误差下降来自 owner/phase/物理字节守恒修复，不是全局倍率。专项回归
为 `39 passed`；完整回归需在该修复后重新执行。

在代码锁定版本上又完成了 3 次独立中位数测量，锁定文件为
[qwen38_cpu_prompt8_f9_v2_code_lock_before.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_f9_v2_code_lock_before.json)，前后关键源码和 binary 的 SHA 均未变化。汇总见
[qwen38_cpu_prompt8_f9_v2_current_median_summary_v1.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_f9_v2_current_median_summary_v1.json)：native 中位数为 TTFT `1144.39 ms`、TPOT `323.96 ms/token`、E2E `3421.68 ms`，simulator 为 `919.13/300.20/3020.53 ms`，误差为 `-19.68%/-7.33%/-11.72%`，绝对误差约 `225.26/23.76/401.15 ms`。native CV 为 `3.28%/5.87%/4.97%`，因此当前主要差距已回到仿真器的数据流和关键路径。

进一步审计显示，native CPU trace 将 FFN up/gate 作为两个独立的
`17408x1x1x1` `MUL_MAT`，而 planner 仍将它们合并为一个
`34816x1x1x1` `mlp_up_gate`。当前 profile 没有把 `17408` 系数强行套到
`34816`，所以该 compute 项保持分析模型；下一步将只在 CPU/Qwen3.8 证据
路径中拆成两个 sequential physical GEMM，并以新的 holdout 验证 TPOT，避免
把无证据的分摊系数推广到其他模型。

跨模型复查发现 Qwen3.5 的既有 profile 不能直接启用新的精确 shape 选择，
因此 parity harness 增加显式 `native_calibration_shape_policy`：Qwen3.8 使用
`exact`，其他既有 profile 使用 `phase`，直到取得对应 shape 留出证据。
Qwen3.5 在当前 phase policy、stage-only 应用下的一次回放为 native
`8.844/5.516/47.458 ms`，simulator `5.397/6.304/49.522 ms`，误差
`-38.98%/+14.27%/+4.35%`。该结果同样说明 E2E 接近不能替代 TTFT/TPOT
各自收敛，后续需要独立请求边界证据。
该拆分已落地：GGUF descriptor 同时保留 `mlp.up_gate` 逻辑视图和独立的
`mlp.gate`/`mlp.up` 物理视图，Qwen3.8 CPU parity 场景显式开启该能力，
planner 对第二个 projection 依赖第一个，形成与 native 相同的顺序调用。
当前单次验证为 native `1126.17/354.68/3608.93 ms`，simulator
`827.01/281.02/2794.13 ms`，误差 `-26.56%/-20.77%/-22.58%`；该单次样本
仅用于确认调用结构已经生效，最终判断以拆分后的多次中位数为准。
在拆分后的锁定版本上完成了 3 次中位数复测，源码、profile 和 semantic
binary 的前后 SHA 全部保持不变。汇总见
[qwen38_cpu_prompt8_ffnsplit_latest_median_summary_v1.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_ffnsplit_latest_median_summary_v1.json)：native 中位数为 TTFT `1164.70 ms`、TPOT `321.29 ms/token`、E2E `3413.71 ms`，simulator 为 `1050.08/300.56/3154.04 ms`，误差分别为 `-9.84%/-6.45%/-7.61%`，绝对误差约 `114.61/20.72/259.67 ms`。三次 native CV 为 `4.39%/2.63%/3.22%`；当前 Qwen3.8 CPU-only 的三项误差均已低于 10%，剩余差距主要来自 CPU 调度和请求边界。
随后在 calibration.py 最新版本（shape gate 修复后）重新锁定并采样，前后
关键文件均 unchanged。汇总见
[qwen38_cpu_prompt8_exact_policy_median_summary_v1.json](../artifacts/multimodel_next/qwen38_cpu_prompt8_exact_policy_median_summary_v1.json)：native 中位数为 TTFT `1141.38 ms`、TPOT `337.74 ms/token`、E2E `3494.87 ms`，simulator 为 `1050.09/300.56/3154.04 ms`，误差为 `-8.00%/-11.01%/-9.75%`。三次 native CV 为 `0.59%/2.15%/1.46%`，这组是当前 Qwen3.8 CPU-only 的最新有效中位数证据。
## 其他模型的并行审计进展

Qwen3.5-0.8B 已完成 prefill-scoped launch 审计。native trace 的 prefill
launch 为 892 次、约 2.14 ms，decode launch 为 730 次、约 3.27 ms；由于
operator-wall 已包含 launch，不能把 CUDA API 总时间再次叠加到每个算子。保留
stage-only 作为正式校准，prefill-only launch profile 仅作为边界证据。一次
prefill-scoped 复测为 TTFT `-20.98%`、TPOT `+6.35%`、E2E `+2.18%`，
证据见 [qwen35_prefill8us_composed_v2.json](../artifacts/multimodel_next/qwen35_prefill8us_composed_v2.json) 和
[qwen35_launch_phase_audit_v1.json](../artifacts/multimodel_next/qwen35_launch_phase_audit_v1.json)。

Qwen2.5-0.5B medium 输入的 native prompt 实际为 9 token。新增 kernel-basis
profile 已覆盖对应 M=9 shape，kernel 留出误差大多低于 1%，但 operator-wall
和 CUDA API 在 prefill/decode 边界存在较大调度波动：prefill launch 为
862 次、sync 13 次，decode launch 为 3402 次、sync 91 次，且跨边界调用 56
次。因此只接入 fail-closed 的 kernel-shape 路径，暂不把不稳定的 launch/sync
总时间灌入算子。阶段提取工具和证据为：

- [extract_cuda_api_phase.py](../tools/extract_cuda_api_phase.py)
- [qwen25_medium_train_cuda_api_phase_v1.json](../artifacts/multimodel_next/qwen25_medium_train_cuda_api_phase_v1.json)
- [qwen25_medium_holdout_cuda_api_phase_v1.json](../artifacts/multimodel_next/qwen25_medium_holdout_cuda_api_phase_v1.json)

TinyLlama 的 Hi 输入已经达到 TTFT `-5.5%`、TPOT `+7.8%`、E2E `+5.9%`；
其现有 profile 只覆盖 M=3 prefill 和 M=1 decode，medium/long 输入仍需新的
shape 留出，不能把 Hi 的系数直接推广。

在 Qwen2.5/Qwen3.5 支持合入后的最新源码上，Qwen3.8 CPU-only 又完成了一次
独立校验，当前结果为 native `1155.91/284.83/3149.71 ms`、simulator
`1050.08/300.56/3154.04 ms`，误差 `-9.16%/+5.53%/+0.14%`。对应源码、
binary、GGUF 和 profile 的当前锁定记录为
[current_code_lock_after_all_model_changes_v1.json](../artifacts/multimodel_next/current_code_lock_after_all_model_changes_v1.json)。此前的 3 次中位数仍然
有效地说明波动范围，但其锁定 SHA 早于本轮其他模型支持；本次单次结果用于
证明最新代码没有破坏 Qwen3.8 路径，后续会在最终矩阵前再做统一锁定中位数。
在当前源码、semantic binary、profile 和 runtime identity 全部锁定后，TinyLlama
又完成了 3 次独立测量。汇总见
[tinyllama_current_lock_manifest_v1.json](../artifacts/multimodel_next/tinyllama_current_lock_manifest_v1.json)：native 中位数为 TTFT `4.442 ms`、TPOT `4.257 ms/token`、E2E `33.942 ms`，simulator 为 `4.242/4.182/33.519 ms`，误差分别为 `-4.51%/-1.76%/-1.25%`。这组结果使用显式、身份匹配的 8 μs launch 证据；此前 simulator 约低估 40% 的结果是 stage-only 路径，缺少该已测 launch 边界。

Qwen2.5 medium 的 phase-boundary 实现已经写入，但当前 train/holdout 是
prompt=9、decode=7，而标准回放为 prompt=8、decode=8，且边界 profile 仍有
较大 launch/sync 波动，因此该校准保持 blocked，不进入最终误差矩阵。相关
证据见 [qwen25_medium_launch_sync_boundary_v2.json](../artifacts/multimodel_next/qwen25_medium_launch_sync_boundary_v2.json)。
在硬件指纹稳定化、phase 传递修复和 prompt=8/predict=8 配对 profile 后，
Qwen2.5 medium 的最新回放为 native `8.005/4.096/36.674 ms`、simulator
`5.747/4.641/38.235 ms`，误差 `-28.20%/+13.32%/+4.26%`。另一批同 profile
的中位数结果为 `+4.68%/+8.60%/+8.00%`；当前仍需多次锁定采样才能把 TTFT
的单次波动和请求边界误差分开。phase boundary 只作为每个物理 invocation
一个同步任务加入，没有按算子分摊 launch/sync。
新的 prompt=8/predict=8 配对 profile 已完成并通过稳定 hardware fingerprint
门控。使用 [qwen25_medium_prompt8_boundary_profile_v3.json](../artifacts/multimodel_next/qwen25_medium_prompt8_boundary_profile_v3.json) 的一次当前回放为 native `6.370/4.159/35.483 ms`、simulator `5.641/4.646/38.163 ms`，误差 `-11.44%/+11.71%/+7.55%`；另一批锁定中位数为 `+4.68%/+8.60%/+8.00%`。prompt 文本和 fingerprint 现在随比较结果保存，避免把 9 或 13 token 文本误配到 prompt=8 profile。

## v7 当前源码收敛结果（2026-09-14）

本轮按“每个场景独立证据、禁止跨模型全局倍率”的规则重新锁定并回放。新增 Qwen3.5 的 `Hi.`/prompt=2/predict=8 train/holdout trace，prefill 892 次 launch、18 次 sync，decode 7 个 phase、5110 次 launch、126 次 sync；跨 phase 的 56 次调用排除。边界校准只作用于 prefill，每个物理 invocation 一个 boundary task，decode 不重复叠加 API 总时长。

当前三次中位数结果：

- Qwen2.5-0.5B，prompt=8/predict=8：TTFT `+7.35%`、TPOT `+6.68%`、E2E `+4.85%`。
- Qwen3.5-0.8B，`Hi.`/predict=8：TTFT `+8.98%`、TPOT `-8.45%`、E2E `-4.29%`。
- TinyLlama-1.1B，`Hi.`/predict=8：TTFT `-19.90%`、TPOT `+0.56%`、E2E `-1.42%`；TTFT native CV `13.59%`，该项仍主要受请求启动边界抖动影响。
- Qwen3.8-27B CPU-only，prompt=8/predict=8 的最新单次回放：TTFT `-9.16%`、TPOT `+5.53%`、E2E `+0.14%`；更早的精确 shape 三次中位数为 `-8.00%/-11.01%/-9.75%`，但缺少与本轮其他模型统一的 per-file source SHA/hardware map，因此保留为 provisional。

正式热力图只纳入 source SHA、GGUF、runtime、hardware 和 `identity_mismatch=false` 全部齐全的 TinyLlama 与 Qwen3.5；Qwen2.5 和 Qwen3.8 进入结构化汇总的 provisional 区域，不混入正式矩阵。详见 [error_summary_v7.json](../artifacts/multimodel_next/error_summary_v7.json) 与 [error_heatmap_v7.png](../artifacts/multimodel_next/error_heatmap_v7.png)。

完整回归为 `826 passed, 1 skipped`，`compileall` 通过。测试期间仍出现一次既有 Windows `pyarrow/pandas/ortools` loader `0xc0000139` 噪声，但 pytest 最终成功完成。

## v7 证据状态修订

Qwen2.5 prompt=8/output=8 已补齐严格的当前源码逐文件 SHA 锁定，manifest 为 [qwen25_medium_prompt8_boundary_current_identity_lock_v1.json](../artifacts/multimodel_next/qwen25_medium_prompt8_boundary_current_identity_lock_v1.json)。该 manifest 覆盖 planner、calibration、gguf parity、native compare、profile、trace 工具、semantic binary、GGUF、硬件/runtime/prompt fingerprint，并验证三次回放的 `identity_mismatch=false`。因此 Qwen2.5 已从 provisional 提升为正式矩阵成员。

当前正式热力图包含三个模型场景：Qwen2.5 prompt=8、Qwen3.5 prompt=2、TinyLlama prompt=3；Qwen3.8 仍因缺少硬件 fingerprint 和逐文件源码锁定而保留在 provisional 区域。不同 prompt 长度的旧结果没有外推到正式矩阵。跨输入审计见 [CROSS_INPUT_AUDIT_20260914.md](CROSS_INPUT_AUDIT_20260914.md)，Qwen2.5 锁定报告见 [QWEN25_PROMPT8_CURRENT_SOURCE_LOCK_20260914.md](QWEN25_PROMPT8_CURRENT_SOURCE_LOCK_20260914.md)。

## Qwen3.8 当前源码身份锁定完成

Qwen3.8-27B CPU-only 已完成严格三次重采样并生成 [qwen38_current_code_lock_manifest_v1.json](../artifacts/multimodel_next/qwen38_current_code_lock_manifest_v1.json)。采样采用 r2/r3/r4，排除了 GGUF hash 处于瞬态的 r1；三次均通过 GGUF geometry、operator-wall coverage 和 `identity_mismatch=false` 检查，并绑定稳定硬件 fingerprint、runtime fingerprint 及完整 source SHA map。

中位数结果为 native `1157.800/325.192/3472.438 ms`、simulator `1050.085/300.565/3154.037 ms`，误差分别为 TTFT `-9.30%`、TPOT `-7.57%`、E2E `-9.17%`；native CV 为 `1.62%/2.39%/1.65%`。因此 Qwen3.8 已从 provisional 提升为正式矩阵成员。

当前正式热力图已包含 Qwen2.5、Qwen3.5、TinyLlama 和 Qwen3.8 四个模型场景；每个场景的 runtime fingerprint 按实际运行配置独立绑定，CPU-only 与 CUDA offload 不强行共用 runtime 指纹。

## TinyLlama TTFT 边界校准结论

对 TinyLlama 的现有 semantic trace 做了专门边界审计。short prompt=3 trace 和 train/holdout prompt=2 trace 只有 `phase:prefill/decode`，没有 request begin/end/first-token marker；phase 起点前没有可归属的 CUPTI 事件，首算子 gap 约 `0.289–0.329 ms`，不足以证明约 `1.05 ms` 的请求边界差异。试验性打开 stage wall 后一次结果反而变为 `+20.86%`，说明 stage wall 已覆盖或超过可观测阶段缺口；再叠加 prefill API aggregate 会重复计费。因此 TinyLlama TTFT `-19.90%` 暂保留为 blocked diagnostic，不伪造边界校准。只有在 llama.cpp trace 增加 request begin/end/first-token marker 后，才继续对该项下沉。

## v9：同一 request-marker binary 的最终当前矩阵

为避免 semantic binary 变化造成证据混用，本轮重新使用 SHA-256 `4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337` 的 request-marker binary，并在当前源码下刷新四个正式场景：

- Qwen2.5 prompt=8/output=8，五次中位数：TTFT `-7.32%`、TPOT `+9.66%`、E2E `+6.76%`。
- Qwen3.5 prompt=2/output=8，三次中位数：TTFT `+7.28%`、TPOT `-2.34%`、E2E `-2.25%`。
- Qwen3.8 CPU-only prompt=8/output=8，三次中位数：TTFT `-5.63%`、TPOT `+3.32%`、E2E `+1.72%`。
- TinyLlama prompt=3/output=8，采用 `launch + request_begin`，三次中位数：TTFT `-2.91%`、TPOT `+3.92%`、E2E `-2.24%`。

四个正式场景的三项绝对误差均小于 10%。TinyLlama prompt=29/output=8 已完成同 binary 三次取证，但 request_begin 和 first_token 的 train/holdout 波动分别约 42% 和 38%，所以只保留为 provisional boundary diagnostic，不把长输入的 TTFT 误差伪装成已收敛。

统一矩阵与热力图见 [error_summary_v9.json](../artifacts/multimodel_next/error_summary_v9.json)、[error_heatmap_v9.png](../artifacts/multimodel_next/error_heatmap_v9.png) 和 [error_heatmap_v9_heatmap.json](../artifacts/multimodel_next/error_heatmap_v9_heatmap.json)。新 request-marker 接口及 extractor 变更后的完整回归为 `831 passed, 1 skipped`，`compileall` 通过。

# 盲测与泛化验证（2026-09-14）

本轮先冻结源码、semantic binary 和校准策略，再对未参与校准的新 prompt 进行三次原生/仿真配对回放。冻结文件为 [blind_generalization_freeze_v1.json](../artifacts/multimodel_next/blind_generalization_freeze_v1.json)，冻结期间没有修改 profile 或源码；源文件 SHA 与 binary SHA 在揭盲后重新核对一致。

盲测结果没有支持“新场景误差自动保持在 10% 内”的假设：

- Qwen2.5 新 prompt 实际为 18 token：TTFT 中位误差 `-35.05%`，TPOT `+11.20%`，E2E `+2.78%`；TTFT p90 绝对误差 `38.11%`。
- Qwen3.5 新 prompt 实际为 8 token：TTFT `-48.28%`，TPOT `+3.50%`，E2E `-13.10%`；TTFT p90 `59.48%`，E2E p90 `24.27%`。
- TinyLlama 新 prompt 实际为 14 token：TTFT `-11.52%`，TPOT `-2.90%`，E2E `-3.77%`；TTFT p90 `13.64%`。

这些结果使用与 formal 场景相同的 `ctx=512`、`parallel=1`、`batch=ubatch=64`、`threads=16`、`gpu_layers=-1`、`FA=off`、`seed=42` 和 warmup 配置，差异主要来自未见 prompt token shape、prompt 内容和 request/prompt 计时边界，而不是配置漂移。Qwen2.5 和 Qwen3.5 的 request-boundary 校准因 shape/fingerprint 不匹配自动 fail-closed。

需要特别区分：三次重复是重复性证据，不是泛化保证；同时当前 native `prompt_ms` 与 simulator TTFT 的边界仍不完全相同，因此 TTFT 数值属于诊断比较。完整原始数据、中位数、p90、最坏误差和身份信息见 [blind_generalization_summary_v1.json](../artifacts/multimodel_next/blind_generalization_summary_v1.json)。Qwen2.5 明细见 [BLIND_QWEN25_P16_A_20260914.md](BLIND_QWEN25_P16_A_20260914.md)。

结论：仿真器可以接受新场景输入并给出结构化预测，但盲测证明当前不能对新场景的三项误差作保证。下一阶段应冻结 profile 后扩展 prompt/output 长度和并发矩阵，并以真正 request-to-first-token 的原生 marker 重新定义 TTFT；只有在独立 blind test 的中位数、p90 和最坏误差均达到验收线后，才可以形成可推广的误差承诺。

## 扩展盲测协议与泛化修复（2026-09-14）

针对第一批盲测暴露的 shape 回退和计时边界问题，已完成以下修复：

- stage 校准只允许在同一数值轴、两端均有 train/holdout 证据的 bracket 内线性插值；区间外回退分析需求并写出 extrapolation provenance，不再静默套用 phase 系数。
- request marker 保持 exact prompt/output/fingerprint gate；不稳定的 first-token train/holdout 证据自动 fail-closed。
- 原生比较默认使用流式 completion，增加 request-to-first-token、request-to-end、首末 token 时间和每个并发请求记录；旧 prompt_eval_ms/total_ms 只保留为阶段诊断。
- 并发 2/4 使用同一 server 的真实同步并发请求，simulator 生成相同数量的 request；批次汇总使用请求 p50，同时保留 p90、max、makespan。
- TinyLlama prompt=29 的 lm-head prefill 继续只计算最后一行，原生 trace 的 `32000x1x1x1` 已用回归测试锁定。

新的完整验收矩阵定义为 5 个模型（Qwen2.5、Qwen3.5、Qwen3.8、TinyLlama、SmolLM2-1.7B）× 短/中/长 prompt × 短/中/长 output × 并发 1/2/4，3 次重复，共 135 个 cell、405 次执行。协议冻结在 [blind_generalization_freeze_v3.json](../artifacts/multimodel_next/blind_generalization_freeze_v3.json)，统一 `ctx=2048`、`batch=ubatch=64`、`threads=16`、`request_timing=stream`。

矩阵执行器为 [generalization_acceptance_matrix.py](../tools/generalization_acceptance_matrix.py)，dry-run 已验证计划数 `405`。每个 cell 记录实际 prompt/output token 数、模型/GGUF/binary/runtime/hardware 指纹、校准 applicability、并发支持状态；验收表对 TTFT、TPOT、E2E 分别给出 signed/absolute median、p90、worst，以及 absolute milliseconds 的 median/p90/worst。

## 135 格验收矩阵执行进度（2026-09-14）

已完成执行器与协议修复：完整计划为 `5 x 3 x 3 x 3 = 135` 个 cell，每个 cell 重复 3 次，共 405 次执行。并发 2/4 使用同一 server 的同步并发请求，验收聚合不再只取 request-0，而是使用请求级 p50，并记录 p90、worst、makespan 和 absolute milliseconds。

为保证长 prompt/长 output/并发 4 可执行，统一验收配置将 `ctx` 从 512 提升到 2048；此前 Qwen2.5 在 ctx=512 下的 9 个上下文溢出格子被单独保留为配置边界证据，不进入 ctx=2048 验收结果。

冻结清单为 [blind_generalization_freeze_v3.json](../artifacts/multimodel_next/blind_generalization_freeze_v3.json)，执行器为 [generalization_acceptance_matrix.py](../tools/generalization_acceptance_matrix.py)，最终合并器为 [merge_generalization_acceptance.py](../tools/merge_generalization_acceptance.py)。Qwen2.5 ctx=2048 的 81 个重复格子已经全部完成且结构有效；其长 prompt/长 output/并发组合已显示明显泛化误差，说明矩阵确实能揭示问题而不是只筛选通过项。Qwen3.5 正在执行，随后依次执行 Qwen3.8、TinyLlama 和 SmolLM2。

最终验收表字段固定为每个 `(model,prompt_band,output_band,parallel)` 的 TTFT/TPOT/E2E：signed median、absolute median、absolute p90、absolute worst，以及 absolute milliseconds median/p90/worst；无效或上下文溢出 cell 保留在原始 cell JSON 并明确排除理由。

## 完整矩阵结果（2026-09-14 揭盲后）

405/405 次执行、135/135 个场景组均已观测。结构有效 402 个，3 个因运行时模型 hash 与冻结 SHA 不一致而保留为身份失败；另有 80 个结构有效 cell 的实际输出只有 1 token，TPOT 按 `N_out>1` 契约记为证据不足，但 TTFT/E2E 仍参与各自统计。完整数据和 405 行验收表见 [GENERALIZATION_ACCEPTANCE_20260914.md](GENERALIZATION_ACCEPTANCE_20260914.md)。

本轮没有达到第一阶段目标。主要模型级场景组绝对误差中位数为：Qwen2.5 的 TTFT/TPOT/E2E 为 82.61%/31.92%/38.89%，Qwen3.5 为 37.78%/38.20%/25.18%，Qwen3.8 为 125.79%/42.45%/78.91%，TinyLlama 为 72.28%/43.54%/55.43%，SmolLM2 为 71.47%/7.41%/72.28%。长 prompt、长 output、并发 4 是最明显的失败区域，说明当前机制模型仍系统性低估 prefill/KV/调度或客户端边界成本。

按照后续任务书，本轮数据转入开发/验证用途；下一冻结版本必须先修正客户端 E2E/TPOT 契约、profile/微基准/提取器的完整 SHA 冻结，以及先预测后揭示的盲测流程，再开展机制级消融和新的独立验收集。本轮结果不支持跨模型、跨硬件或任意新 shape 的误差保证。
