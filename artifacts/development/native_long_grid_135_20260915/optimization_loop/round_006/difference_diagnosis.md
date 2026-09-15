# R6 小模型剩余差距：只读边界诊断

对象：`round_006/physical_mapping_mmq` 已评分的旧 12 锚点快照，其中 4 个 Qwen2.5/TinyLlama 控制。并行代理正在修复的 4 个 R6 边界问题尚不属于此快照。没有新跑 LLM、模拟或测试，没有拟合系数。

## 已确认事实

| 场景 | TTFT 原生 → 模拟 ms | TPOT 原生 → 模拟 ms | E2E 原生 → 模拟 ms |
|---|---:|---:|---:|
| qwen25_p512_o32_c1 | 59.568 → 15.046 | 3.609 → 2.021 | 171.446 → 77.699 |
| qwen25_p1536_o128_c4 | 301.602 → 54.661 | 7.204 → 2.402 | 1221.405 → 360.175 |
| tinyllama_p512_o32_c1 | 54.738 → 27.224 | 3.149 → 2.382 | 152.215 → 101.072 |
| tinyllama_p128_o128_c4 | 22.391 → 10.862 | 4.439 → 2.865 | 588.323 → 374.529 |

1. 四格原生 raw SHA 重新核对一致。所有正式请求的 prompt_n、predicted_n、cache_n=0 与输入一致，token 时间戳个数等于输出数；原生 prompt-last 恰好等于首 token。没有证据表明是少 token 或把客户端时钟混成 engine 时钟。不同进程的绝对 epoch 不可直接相减，只比较请求内部区间。
2. 四格冻结配置均为 **compiled_cuda_graphs=false**。即使环境中 GGML_CUDA_DISABLE_GRAPHS 未设置，也不能推断启用了 CUDA Graph。缺失 graph replay 不是当前有证据支持的原因；CPU 图构建/复用另当别论。
3. **kernel launch 没有全漏计**：GPUProfile 保留每次 1,000 ns 的分析值，成本通过 gpu0.frontend 收取。Qwen2.5 P512/C1 的 prompt 和 decode frontend busy 分别为 4.035 ms、17.205 ms，TinyLlama 对应 5.811 ms、15.779 ms。它们不是原生 kernel 数，也不能再加到已含这些成本的 engine 时延上。
4. 旧 Qwen2.5/TinyLlama profile 被 binary/runtime 身份门阻断，eligible count=0；本快照 calibration_applied=false。同步/阶段边界/首次 decode/request marker 等额外经验项不再应用，保留输出的 synchronization critical-path 成本为 0。**同步依赖仍等待模拟 GPU 工作完成；缺的是未闭合的额外 host/driver 边界，不是把 GPU 执行等待全部跳过。**

## prompt ubatch：匹配到哪一级

两个 C1/P512 控制的模拟均为 8×64 prompt 行加 31 个 decode 行。结合原生 cache_n=0、prompt_n=512、b=ub=64，可推导“8 个满 prompt chunk”与源码约束一致。但原生日志均为 0 字节，没有原生逐 ubatch/kernel 事件，因此这是推导，不是直接计数证明。
C4 混合 decode 会占用 batch 的 token 预算，不能简单按每 slot 的 ceil(P/64) 当成完整物理次数：Qwen2.5 P1536 的模拟四请求为 24/25/26/26 个 prompt item；TinyLlama P128 为 2/3/3/3。每请求行数仍完整。原生精确混合顺序、backend split 和物理 ubatch 数仍未知。

## 旧 profile 阻断后的实际回退

| 项 | 当前处理 | 不能据此声称 |
|---|---|---|
| GPU launch | 固定分析值 1 µs，frontend 服务非零 | 已测得真实 host enqueue 或每 kernel 固定开销 |
| GEMM 主计算 | 结构峰值 ×0.65 efficiency ×0.85 occupancy，频率按冻结采样映射 | 当前 native kernel 可达吞吐已经校准；或无折扣地直接取峰值 |
| HBM | 960 GB/s ×0.75=720 GB/s 基础值，再乘 tile-wave 几何利用率 | logical bytes 就是真实 DRAM 事务 |
| 同步/phase/startup/marker | 无合格 profile 时不增加其经验项 | 可把完整 cudaStreamSynchronize 墙钟再加一次；该墙钟包含 GPU 等待 |
| 量化/转换 | GET_ROWS compute 未计价；generic quantized dot 有分析标量成本；MMQ 分支部分具备 conversion/fixup | 全部解量化都为零，或全部 MMVQ 已被 MMQ 成本修复 |
| host/transfer | prompt cohort 约 8.6 µs orchestration，DMA/PCIe 服务非零 | native CPU 图处理、驱动供给间隔和实际同步节奏已经闭合 |

Qwen2.5 的静态 tensor 目录明确包含 K=896 的 Q5_0/Q8_0 QKV/up/gate 投影。当前 source-MMQ 成本要求 K 是 256 的倍数：P512/C1 的 7,712 个模拟 GEMM 中仅 192 个使用该成本，1,152 个以此对齐原因未覆盖，3,720 个标为 mmvq_precedes_mmq。TinyLlama P512/C1 为 1,232/7,072 个 MMQ 已应用，3,410 个 MMVQ 优先。
因此，不能把“开启 MMQ source cost”解读成小模型 decode 的实际向量核已得到独立建模。这里引用的是尚待核心统计修复的模拟账本，不是 native dispatch 证据；未覆盖路径仍有 generic roofline 成本，不是整条路径为零。

## 通用 probe 能说明什么

刚完成 launch probe 的 480 个正式样本 raw 有效，但 **16 个配置全部 unresolved_diagnostic，transferable_kernel_launch_ns 全部为 null**。例如 empty kernel、burst=64：continuous enqueue 的 host total 中位数约 228.2 µs，逐次 stream-sync 为 1776.75 µs。供给与同步节奏明显改变区间；不能把两者差值或任何单配置中位数直接转成 LLM launch 常数。
先前 Q4_K 合成 MMQ trace 能区分 host graph_compute+sync 与 CUDA kernel 并集，但 Nsight 对 CUDA13.4 驱动使用了 12.8 采集库，兼容性门未通过。其区间仍只作诊断，不能把 host wall 与 GPU union 相加，也不能作为当前 LLM 的 kernel 真值。

## 最小下一步验证项

1. **直接提交的 GGML 依赖链与供给边界**：固定 DLL/时钟条件，区分连续 enqueue、host-paced enqueue 和真正依赖点同步。分别保留 QPC enqueue/sync/total 和 CUDA event 区间，避免对 GPU 等待双计；probe 未稳定前不写 profile。
2. **低 M MMVQ 与 K896 量化矩阵**：合成 Q5_0/Q8_0 K896、代表性 N128/896/4864，M1/2/4，另加 M64 对照和对齐 Q4_K/Q6_K 对照。验证合法 padding、实际 kernel 路径和数值正确；源公式及未见合成尺寸验证先于成本入库。
3. **转换与实际 backend split 边界**：合成 GET_ROWS、转置 KV 写入、H2D 与 logits return，保留索引形状与依赖链。分别检查转换计算和复制/同步位置，而非从模型总差额反求补偿。

这些是建议的最小验证范围，本任务没有执行它们。当前差距还不能在提交节奏、同步、MMVQ/非对齐 shape、图规划、cache conversion 和实际 residency 之间可靠分摊；-ngl -1 的实际加载层数仍标为 conditional。

完整逐请求时间戳、模拟成本分解、源文件行号/SHA、profile 阻断理由和 probe 引用见 difference_diagnosis.json。没有修改任何原预测/评分/native/source/state/任务书。
