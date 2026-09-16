# R23 MMVQ HBM 几何审计

日期：2026-09-16  
性质：只读、分析性审计；未运行 native、未读取目标 LLM 时延、未修改 profile、系数或主分支。

## 结论

当前 MMVQ 路径的 HBM 字节账本没有发现错误，但 `estimate_gpu_gemm()` 把 **MMA 输出 tile 波次利用率**用于 MMVQ HBM 有明确的语义错配：`mmvq_work` 已表明实际内核是 vector CTA/warp/K-loop 路径，而 HBM 带宽仍由 `mma_output_tile_wave_proxy()` 的 `M×N` MMA 输出 tile 数决定。

这不是已证实“应把带宽调高”的证据。仅凭当前 source geometry，能严格导出 CTA、warp、rows/CTA、K-block 循环和逻辑 weight bytes；**不能**导出 resident CTA/SM、register 限制、真实 load coalescing、L2 命中、memory partition 分布、outstanding request 数或达到饱和带宽所需并行度。因此不能把 CTA 数直接换算成新的 HBM 效率，也不能按 native actual 选择更快或更慢的方向。

最严谨的近期行为是：保留现有值时明确其为 legacy MMA fallback，不能声称它来自 MMVQ source geometry；若后续要消除语义错配，应把 MMVQ HBM 并行度降级为 `unqualified`，只输出 source-derived geometry 与带宽下界，而非伪造有限的 source-derived effective bandwidth。

## 当前公式和代码位置

`round_022/cta_issue/source/src/heterollm_sim/cost_models.py`：

- `mma_output_tile_wave_proxy()`，约 2385–2424 行：
  - `independent_output_tile_count = ceil(M/mma_m) * ceil(N/mma_n)`；
  - `parallel_tile_slots = floor(SM × tensor_cores_per_SM × occupancy)`；
  - `output_tile_wave_utilization = independent_output_tile_count / (waves × slots)`。
- `estimate_gpu_gemm()`，约 2683–2703 行：
  - `shape_effective_hbm_bandwidth = hbm.effective_bandwidth × output_tile_wave_utilization`；
  - 即使 `workload.mmvq_work is not None`，仍使用该 MMA proxy；
  - 元数据同时明确写入 `source_geometry_hbm_concurrency_applied=false`，并声明 MMVQ CTA/warp geometry 只是 unpriced metadata。

同一函数中，MMVQ issue-bound 的 compute demand 已换成 `gpu.scalar`（约 2632、2767 行），但 memory demand 没有相应换成 MMVQ-specific model。故 compute 和 HBM 两个子模型的几何基础不一致。

## 同一冻结 qwen25 decode cohort 的可复算反例

冻结 cohort：`qwen25_p512_o32_c1__fixed_runtime`，`cohort-000008`，decode `M=1`，上下文 512。

GPU profile 的 MMA proxy 输入：

- GPU SM 数：84；
- tensor cores/SM：4；
- occupancy：0.85；
- `parallel_tile_slots = floor(84 × 4 × 0.85) = 285`；
- MMA tile 为 `16×16×16`；
- 两个算子都为 `M=1, N=896`，故 `ceil(1/16) × ceil(896/16) = 56` 个 independent output tiles；
- 只有一个 output wave，MMA utilization 均为 `56 / 285 = 0.1964912281`；
- HBM profile 的 peak-effective bandwidth 为 `960 × 0.75 = 720 GB/s`，因此二者都得到 `720 × 0.1964912281 = 141.473684 GB/s`。

这两个 MMVQ 算子的 source launch geometry 明显不同：

| 算子 | GGUF physical weight | source `grid` | block / warps | rows/CTA | CTA 数 | launched warps | K blocks/row | source vector-dot calls | 当前 MMA HBM utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `layer-000.attention_q` | Q5_0，551,936 B | `(224,1,1)` | `(32,4,1)` / 4 | 4 | 224 | 896 | 28 | 25,088 | 0.1964912281 |
| `layer-000.mlp_down` | Q6_K，3,575,040 B | `(896,1,1)` | `(32,4,1)` / 4 | 1 | 896 | 3,584 | 19 | 544,768 | 0.1964912281 |

`rows_per_cta` 的变化使 source grid 从 224 CTA 变成 896 CTA；总 launched warps 也从 896 变成 3,584。当前 HBM utilization 却只观察相同的 `M=1,N=896`，所以完全相同。该反例证明 MMA output-tile 因子**不能是 MMVQ launch 并行度的来源**。

这不证明 896 CTA 一定比 224 CTA 有更高实际带宽：CTA residence、register pressure、CTA scheduler 分布、cache/partition traffic 与 memory transaction 仍未知。

## 实际 source loads 与当前字节模型的边界

当前 source work 可靠给出：

- Q5_0：28 K blocks/row，22 B/block；物理 weight = `896 × 28 × 22 = 551,936 B`。
- Q6_K：19 K blocks/row，210 B/block；物理 weight = `896 × 19 × 210 = 3,575,040 B`。
- 两个 main GEMM 的 HBM bytes 分别为：
  - attention Q：`1,008 B Q8_1 input + 551,936 B weight + 3,584 B output = 556,528 B`；
  - MLP down：`5,472 B Q8_1 input + 3,575,040 B weight + 3,584 B output = 3,584,096 B`。

这与 GGUF block layout 一致，且不可把 `source_vector_dot_calls × weight_block_bytes` 当成 DRAM bytes：它是 source-level logical vector-dot work，不证明每个 vector-dot 都造成独立 global transaction。这样相乘会把同一 weight block 的 warp/lane work 误记为多次物理读。

现有 HBM service 是：

- `556,528 / 141.473684 = 3,933.791667 ns`；
- `3,584,096 / 141.473684 = 25,334.011905 ns`。

因此当前问题不是量化 block bytes、Q8_1 bytes 或 weight 读被错误按 FP16/INT8 收费；问题是这些已正确的 bytes 进入了一个与实际 MMVQ kernel family 不同的并行度 proxy。

## 可从 source 导出的参数化上界

可安全保留的 source-derived metrics：

```text
cta_count = grid.x
launched_warps = cta_count × warps_per_cta
rows_per_cta
active_k_threads_per_cta
blocks_per_row
loop_iterations_by_thread
source_vector_dot_calls
logical_weight_bytes / Q8_1 consumer bytes / F32 output bytes
```

若以后需要表达并行度，只能写成有未知项的 envelope，例如：

```text
resident_ctas = min(cta_count, SM_count × resident_ctas_per_SM)
resident_warps = min(launched_warps, SM_count × resident_warps_per_SM)
```

但 `resident_ctas_per_SM`、`resident_warps_per_SM` 和“多少 resident work 足以饱和 HBM”均非当前 source contract 所给。也不能把 MMA 的 285 个 tensor-core warp-equivalent slots当成 MMVQ CTA slots：前者是 tensor-core proxy，后者是 4-warp integer/vector CTAs，资源单位不同。

唯一无额外性能假设的 HBM 时间下界是：

```text
T_memory >= compulsory_backing_bytes / peak_effective_bandwidth
```

这里 peak-effective 为 720 GB/s；它给出理想化最短时间，不是可验证的预测时间。当前 source 不能提供有限的、可信的 MMVQ HBM 上界。

## 不应更改或不应声称的部分

- 不应把 CTA_count 或 launched_warps 直接乘到 720 GB/s，或据此选取新的效率；这是无来源的饱和模型。
- 不应因为替换 MMA factor 会使预测更快，就把它视为修复。若该方向使 native 欠估扩大，也只能如实报告，不能按 native actual 倒选方向。
- 不应将 `source_vector_dot_calls` 解释为 DRAM transaction count。
- 不应消除 Q8_1 conversion predecessor 的独立读写，或把它重新加回 main GEMM 的 F32 input；现有字节账本已确认该处没有重复收费。
- 不应以 RTX Blackwell 的 integer issue partition contract 推导 HBM concurrency；该合同只约束条件化 integer warp issue lower bound。

## 代码方案（仅供后续审阅，不执行）

如果后续独立的 MMVQ bandwidth/occupancy 微基准提供了 source-compatible saturation evidence，可新增独立 `mmvq_hbm_parallelism` contract，并使其显式要求 runtime binary、kernel family、shape/format、CTA geometry 和硬件匹配。

在此之前，结构上较诚实的改法是：当 `workload.mmvq_work` 存在时，停止把 `mma_output_tile_wave_proxy_v1` 标为其并行度依据；元数据改为 `hbm_parallelism_status=unqualified_source_geometry_available`，同时输出上述 CTA/warp envelope 与 peak-effective lower bound。若产品必须给出单点 HBM 预测，应继续把现有 141.474 GB/s 标为 legacy analytical fallback，而不是更名为 MMVQ source-based estimate。
