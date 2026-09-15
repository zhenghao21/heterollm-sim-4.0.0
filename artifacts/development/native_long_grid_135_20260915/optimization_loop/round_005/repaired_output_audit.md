# R5 修正版输出独立审核

结论：三组修正版各 14 个锚点完整，126 个请求的 token 守恒与绝对计时边界检查全部通过。初版两个 inactive 候选、补跑 baseline_r2 与原 baseline 的逐请求时间及三指标完全一致；与 R4 共享的 6 个对照全部一致。支持 full-F32＋gather 的全 131 格冻结诊断，但不支持准确性晋级。

## 守恒与 engine 边界

逐请求 prompt 实际行数=P，decode 实际行数=O−1，可见输出=O；首次输出来自 prefill。事件 committed_tokens 为增量，committed 为累计计数，二者分开核对。上下文和 completion cursor 连续，没有丢 token、重复处理或少生成输出。
TTFT=(first−begin)，E2E=(last−begin)，TPOT=(last−first)/(O−1)，均直接由仿真绝对纳秒换算。begin 仍是首次 prompt 批次处理，first/last 匹配 token 提交事件。8 个运行时/调度源码文件以及 request_timepoints、engine_cohort_span 两个提取函数与 R4 完全一致。
begin 与首个 prompt batch start 的最大差值为 0.00001621246337890625 ns，仅为浮点舍入；审核容差是 0.001 ns。没有把 arrival 换成 engine begin，也没有移动首 token/末 token 边界。
active 候选改变上游执行成本，后续 slot 的绝对开始时间因此可能移动；这不等于更改 engine 定义。此处仅对同到达、fresh cohort、固定 admission 顺序的已资格集合成立。不能把这项结论外推到动态到达、slot 复用、抢占或 aging。

## 行流量、容量、跨设备复制

两个 active 候选各有 2,399 个 qualified/applied 物理 embedding 工作项。物理 ubatch 数可以超过逻辑 batch 数；所有每批和汇总字节均通过以下整数等式：选中行数=C×(P+O−1)，权重读取=行数×packed 行字节，I32 索引读取=4×行数，F32 输出写入=4×行数×hidden。
表容量取最大 footprint，不能按调用累加。完整表暂存按 weight-read invocation 计一次，不能把同一表经过多个链路的流量再算成不同逻辑暂存容量。真实图锚点的 embedding 整表暂存均为 0；非零跨路完整表传输仍只有此前合成结构测试的证据，本审核不把它冒充真实图观测。
**logical selected-row bytes 不等于 DRAM 事务。** 重复索引没有去重，缓存行、页、写分配和解量化 compute 仍未证明或未计价；不能据 logical bytes 直接宣称 DRAM 带宽准确。resource_accounted_bytes 聚合多个缓存/资源层级，也不是唯一 DRAM 字节。
新输出保留 logical storage 账本和批次成本，没有保存每个新真实 GGUF 物理 task 的完整资源 demand。因此已复核计数代数、源码 MemoryWorkload 接线和旧合成 demand 证据，但不能声称已逐项独立审核所有实际收费。inactive/null 账本也不表示容量或流量为零。

## F32 的作用和仍缺的证据

冻结源码中 14 处普通 GEMM 调用均显式传递 F32 存储参数，覆盖 attention QKV/Q/K/V/output、线性 attention 投影及 gate、MLP up/gate/down、LM head。公共 _layer_gemm 为 CPU/GPU 设置 4×M×K 输入存储和 F32 输出；普通 norm/residual/activation/传输按 hidden storage 位宽计算。packed 权重布局与容量单独保留。
KV cache 类型及其转换/回写、持久 recurrent state、logits graph dtype、算术精度、实际 kernel dispatch 和解包吞吐率仍是分别约束的合同，不能把 full-F32 解释为全部张量无条件 F32。两种 active 候选的逐批 KV 追加逻辑/物理字节和 linear state 记录完全一致。
Qwen3.5 专用 QKV 之前就显式 F32；CPU27B 的 R4 host-offload 合同也已开启 hidden F32，gather-only 保留此前设置，所以该组 4 个锚点的 gather-only/full-F32 逐请求结果完全相同。这一解释直接来自源码与已记录的 prior marker，而非误差推测 dispatch。
对其余未先前启用 F32 的组，gather-only 只是隔离消融：embedding 输出变为 F32，但后续普通 hidden storage 未全面接上。full-F32 更符合 GET_ROWS、MUL_MAT、FlashAttention 输出和普通 hidden 类型传递的源码语义。
**仍没有穷尽的逐 projection 输入/输出字节及跨设备 activation copy 观测表。** enabled marker 加调用点源码清单不能替代每个实际 CPU/GPU 算子的 IO 核对。为遵守只读任务，本审核没有新编译或重跑补证。

## 14 个锚点评价

下表是每格 TTFT / TPOT / E2E 绝对百分比误差，再在组内取中位数。只评价结果，不根据误差反推 dispatch。

| 组 | 格数 | baseline_r2 | gather_only_r2 | f32_and_gather_r2 |
|---|---:|---|---|---|
| qwen25 | 1 | 76.05% / 56.56% / 61.54% | 87.35% / 75.14% / 78.25% | 86.74% / 74.65% / 77.73% |
| qwen35 | 3 | 24.90% / 18.99% / 19.16% | 78.55% / 67.54% / 68.07% | 68.03% / 64.23% / 64.32% |
| qwen38 | 4 | 10.56% / 14.65% / 6.57% | 10.38% / 22.25% / 7.38% | 10.38% / 22.25% / 7.38% |
| qwen38_gpu | 4 | 30.60% / 56.84% / 49.73% | 24.31% / 14.15% / 12.97% | 25.00% / 16.76% / 17.50% |
| smollm2 | 1 | 43.26% / 3.48% / 16.98% | 54.85% / 25.04% / 35.16% | 53.44% / 24.51% / 34.34% |
| tinyllama | 1 | 56.14% / 41.86% / 42.64% | 63.15% / 53.74% / 54.24% | 60.34% / 53.08% / 53.46% |

## 全 131 格建议

**继续以 f32_and_gather_r2 做全 131 格冻结诊断，保留 gather-only 为机制消融；不能晋级为准确性通过。** GPU27B 某些 gather-only 格子的三项误差更低，不足以支持保留后续错误 hidden storage。小模型、Qwen3.5 的明显退化说明成本缺项仍大，不能重新加入错误整表读取或错误位宽来抵消误差。
14 个锚点全部已出现在 R3 或 R4；新增两个 CPU27B 控制也在 R4 用过，不能称为未见 holdout。
本审核另纳入固定快照中的 91 个已生成非锚点 full-F32 输出，逻辑 token、报告的时间点和 storage 资格检查通过。它们没有新增 gather-only 配对，也不是未接触过的独立原生 holdout；未要求 diagnostic events 的新格子不具备独立逐 token 事件复核。
全 131 仍在推进，后来生成的文件不进入本审计快照；不等待全量完成，也不把单边输出流当成候选选择依据。
后续成本工作应使用已界定的原生源码、逐调用 shape/dispatch 和合成算子证据，保持解量化未计价等缺项显式；本审核不拟合任何成本参数。

仅新增 repaired_output_audit.json/.md，未运行仿真或 native，未修改任何 src/tools。
