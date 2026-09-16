# R23 独立 MMVQ 访存微基准设计

**状态：仅设计，等待 R23 结束后串行执行。** 不运行 native、不修改 simulator/core、不读取 LLM 实测时延；结果只可验证物理带宽曲面，不能拟合 LLM 答案。

## 目的与复用边界

目的不是继续从 host submit/sync 提取常数，而是测量固定 `ggml-cuda.dll` / source binding 下 **MMVQ 主 kernel** 的 device elapsed time，并按量化物理 weight working-set 区分“L2 warm”与“超过 L2 的旋转 weight set”。这样避免把一个很小、反复读取同一 weight 的测试误作 LLM decode 的 DRAM 带宽。

复用 `operator_microbench_v2` 的 build receipt、静态 device property、module identity、NVTX/CUPTI export 和 event-timing scaffold；不直接复用 `generic-gemm-microbench.exe` 作为 MMVQ 证据：它的 format allowlist 没有 `Q5_0`，且它只证明 generic GGML graph compute，不能证明目标是 `cuda_mmvq_vector_dp4a`。

已记录的静态 device property：CC 12.0、84 SM、warp 32、`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=67,108,864 B`。设计使用该实测属性，不猜 L2 容量。

## 被测 contract

主 kernel 只接受 source-qualified physical format 与 shape：

| 格式 | physical block | source M 域 | K 约束 | CTA/warp envelope |
|---|---:|---:|---:|---|
| Q5_0 | 32 elem / 22 B | 1–8 | K mod 32 = 0 | block `(32,warps,1)`；M≤4 为 4 warps，M>4 为 2；`grid.x=N/rows_per_cta` |
| Q8_0 | 32 elem / 34 B | 1–8 | K mod 32 = 0 | 同上 |
| Q4_K | 256 elem / 144 B | 1–5 | K mod 256 = 0 | 同上，需显式 K-format opt-in |
| Q6_K | 256 elem / 210 B | 1–7 | K mod 256 = 0 | 同上，需显式 K-format opt-in |

`rows_per_cta` 必须按锁定 source 的 small-K predicate 计算并写入结果；M=1 时可能为 4（small K）或 1，M=2–4 为 2。每个结果保存 `M,N,K`、weight bytes、`grid/block`、CTA count、launched warps、每 thread K-loop、source vector-dot calls 和 locked DLL SHA。

**LLM exact-shape anchor 与合成容量 shape 分开保存、分开报告：**

- exact M=1 anchors：Q5_0 `K896,N896`；Q8_0 `K896,N128`；Q4_K `K896,N4864`；Q6_K `K4864,N896`。这些只用于验证 source dispatch/physical-byte 对齐，不进入带宽拟合。
- synthetic capacity shapes：每格式 M=1 与 M=4 各一组，K/N 在上述整除域内选取，使单一 physical weight 约 8–10 MiB；例如 Q5_0 `K4096,N3072`、Q8_0 `K4096,N2048`、Q4_K `K4096,N4096`、Q6_K `K4096,N2560`。实际 freeze 前再次用 ggml block byte 公式校验全部 payload、CTA envelope、显存预算；不得按结果改 shape。

## 两种 working-set 状态与计时边界

每个 exact/synthetic cell 均有两种预注册状态：

1. **warm-L2 label**：同一 weight buffer 连续复用。它只能称为 warm/reuse 条件，绝不能报告为 HBM 带宽。
2. **L2-exceeding rotation label**：同 shape 的 weight buffers 构成 deterministic ring，`ring_count × physical_weight_bytes ≥ 2 × 67,108,864 B`；每次 formal invocation 轮转下一个 buffer。它只证明工作集超过已记录 L2 容量，仍不能宣称所有读取都来自 DRAM。

主 MMVQ timing：在**同一 CUDA stream**上，用 CUDA event 仅包围已验证为 MMVQ 的 main kernel chain。F32→Q8_1 conversion 必须另起 event/NVTX range：

```text
MMVQ_CONVERT: F32 input -> Q8_1 input       # 单独报告
MMVQ_MAIN:    Q8_1 + packed weight -> F32 O # 主带宽候选
```

main payload 记录为 `Q8_1 consumer bytes + physical packed weight bytes + F32 output bytes`；不得把 conversion 的 F32 read/padded Q8_1 write 混入主 kernel bytes。若 event 区间含 conversion、H2D/D2H、多个 stream、未匹配 kernel 或 graph replay，该样本不属于 main-kernel 带宽样本。

## 采样、CUPTI 与独立性

每个 `format × M × shape-class × working-set` cell：3 个全新 direct process；每 process 为 first 1、warmup 5、formal 30，正式值为 device-event elapsed。CUPTI/profile 使用独立 process，只做 kernel family、launch correlation、single-stream、grid/block、kernel count、无 CUDA Graph replay/额外 memcpy 的资格核验；profile timing 不得替代 direct device event，也不得与 direct 合并。

所有 cell 预先冻结为开发与 holdout 两组：开发组为四格式的 M=1 exact anchors及 M=1 synthetic capacity shapes；holdout 为四格式 M=4 synthetic capacity shapes，以及每格式一个未用于选择 ring count 的 L2-exceeding rotation cell。holdout 仅评分/验证，不反向改格式、shape、ring 或质量门。

质量门逐 process 强制执行：锁定 DLL/source/device identity、SM clock bracket、数值正确、CUPTI correlation 完整、单 stream、expected MMVQ family/geometry、无额外 copy、所有正式 device-event 值有效。formal P90/P10、三个 process median 偏差和 direct/profile shape-family 一致性均须预先固定阈值；任何失败保留为失败或“证据不足”，不能改用平均值追认。

## 结果解释限制

输出可形成 `format × M × CTA envelope × working-set label` 的物理 bytes/device-time 曲面。它可以检验当前 simulator 是否误把 MMA output-tile utilization 当作 MMVQ HBM 并行度；不能自动生成全局 HBM efficiency、host launch 常数或 LLM TTFT/TPOT 修正。若 rotation 结果比 warm 慢或快，都如实记录；不得使用目标 LLM actual 决定采用哪一侧。
