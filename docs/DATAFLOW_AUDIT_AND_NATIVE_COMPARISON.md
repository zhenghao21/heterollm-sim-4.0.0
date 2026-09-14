# 数据流审计与原生 llama.cpp 对照

## 已修复的数据流问题

1. **关键路径漏边**：旧逻辑只有在依赖恰好在任务启动时刻完成时才连依赖边。资源排队或 `earliest_start_ns` 造成等待时，依赖类别会从 critical path 消失。`metrics.py` 与 `streaming_des.py` 现在对所有已验证依赖和资源前驱连零时长因果边。
2. **KV 页/流量语义复核**：最初把 `bytes_per_page // tokens_per_page` 误判为容量错误。当前 `bytes_per_page` 本身由每 token 物理量子乘页长精确构造；流量计数表示每 token 的访问服务量，因此保留该量子；驻留和交换则按 `ceil(tokens / tokens_per_page)` 的完整页数计算。新增回归覆盖两种语义，避免把页分配字节重复计入每个 token。
3. **吞吐窗口错误**：旧逻辑用 `[0, makespan]`，非零到达时间会把空闲前缀计入分母。现在使用已完成请求的 `[first_arrival, last_done]` 活跃窗口；全部从零到达时结果保持不变。
4. **llama.cpp 层数参数**：`gpu_layers` 现在只允许 `-1`（全部层）或非负显式层数，拒绝未被本机构建支持的 `-2`。
5. **运行时配置没有进入场景身份**：新增 `LlamaCppRuntimeConfig`，覆盖线程、batch/ubatch、context、并发 slot、GPU 层数、Flash Attention、KV 类型、unified KV、continuous batching、warmup 和 seed；配置会进入 `ScenarioConfig`、任务语义和 SHA-256 fingerprint。
6. **资源容量与在线/静态 DES 不一致**：`ScheduleIR`、`StreamingScheduleIR`、统一事件内核和结果 trace 现在携带同一份 `resource_capacities`，多 lane 利用率按 `busy / (makespan × capacity)` 归一化。
7. **coherent DMA 复合事务过于乐观**：`TopologyRouter` 增加 `coherent_dma_mode`。默认 `pipelined` 保持旧兼容行为；设置为 `strict_serialized` 时，端点读 → DMA → 链路 → DMA → 端点写按显式依赖链累加延迟。
8. **模型与 GGUF 身份未校验**：新增纯标准库 GGUF 解析器，校验 SHA256、architecture、层数、隐藏维度、注意力头、KV 头、词表、上下文和量化类型；native harness 在启动前 fail-closed，token 数在仿真后再次校验。

## 仍需在比较时明确的建模边界

- `LlamaCppAdapter` 的 batch、ubatch、context、并发 slot、KV 类型和 GPU 层数目前是 metadata-only，不能改变任务 demand；本次 harness 因此把这些字段写入命令、场景假设和结果，而不宣称已完成内核级校准。
- coherent DMA 默认仍为 `pipelined` 复合事务；需要保守上界时在硬件或 placement metadata 中设置 `coherent_dma_mode: strict_serialized`，需要真实链路 trace 时再替换成分块 pipeline。
- 模型权重图已通过 GGUF 几何和身份门槛，逐层 projection descriptor 保存了实际 tensor 类型、块大小和物理字节；CUDA kernel 的真实吞吐、launch 和 cache 行为仍未校准。因此结果状态仍固定为 `timing_comparison_only`，不可直接作为模型校准结论。
- 已加入 Nsight Systems 采集入口 [`tools/native_llama_profile.py`](../tools/native_llama_profile.py)。它沿用同一 llama.cpp 命令，分别执行 uncached warmup 和 formal completion，并保存 `.nsys-rep/.sqlite`、CUDA API、kernel、memcpy 汇总及 llama.cpp stage timing。当前 Windows 中等权限会明确记录 WDDM trace disabled 或空 kernel 结果；这类结果只能作为“未获得 kernel 证据”的审计记录，不能被解释成无 CUDA kernel。
- `native_llama_compare.py` 现在同时保存 `/metrics` 前后计数器差分、`/slots` 快照和 stderr 中的 `graphs reused`、prompt/decode/total 行。这些字段用于区分 prompt、decode、CUDA graph reuse、采样/host output 和服务端墙钟，仍保持 `timing_comparison_only`，直到 profiler 覆盖目标阶段。

## 本机原生对照

脚本：[`tools/native_llama_compare.py`](../tools/native_llama_compare.py)

```powershell
py -3.12 tools/native_llama_compare.py --output artifacts/native_compare.json
```

需要采集 CUDA API/kernel/memcpy 证据时，在相同参数下运行：

```powershell
py -3.12 tools/native_llama_profile.py --output artifacts/native_profile_session.json
```

该命令需要本机 Nsight Systems 路径；脚本默认使用已安装的 2024.6.2 版本。WDDM 采集通常需要管理员会话，若权限不足，结果会保留告警和空 kernel 汇总，作为证据缺口而不是零耗时。

从 profiling artifact 生成可复用的最小校准配置：

```powershell
py -3.12 tools/build_calibration_profile.py artifacts/native_profile_session.json --output artifacts/native_calibration.json
py -3.12 tools/native_llama_compare.py --predict 4 --calibration-profile artifacts/native_calibration.json --output artifacts/native_compare_calibrated.json
```

`native-calibration/v1` 当前包含 prompt/decode 每 token 观测值、CUDA launch 平均调用时延、同步平均调用时延和来源文件 SHA。默认应用配置只保存校准证据；只有显式传入 `--apply-launch-calibration` 时，`launch_ns_per_call` 才会进入 `gpu.frontend` 的 `kernel_launch_ns`。这是因为 CUDA graph 的 API 时间不能证明是每个仿真算子的独立启动成本，默认注入会重复计时。其余系数不会自动乘到所有 GEMM 上，必须先有带 NVTX 或 CUPTI 时间戳的阶段映射。

校准文件同时包含 `native-kernel-mapping/v1` 阶段证据。映射器优先接受显式 `graph_stage/op_stage` 标记，再对 kernel 名称使用保守启发式；无法证明归属的记录进入 `unknown`。当前 profile 中约 73.8% 的 kernel 时间仍属于 `unknown`，因此这些记录不会自动计入 Q/K/V、FFN 或 KV 的任务级需求。获得 NVTX 或 CUPTI graph-node 关联后，才可以把阶段聚合提升为任务级 `ResourceDemand` 校准。

需要保留逐事件证据时运行：

```powershell
py -3.12 tools/extract_nsys_trace.py artifacts/native_profile_session.sqlite --output artifacts/native_profile_session.trace.json
```

输出的 `native-nsys-trace/v1` 会保存每个 kernel、memcpy 和 CUDA runtime 事件的 `start_ns/end_ns`、device、stream、correlation、graph node、kernel 名称、阶段标签和置信度。`task_mapping.py` 只接受唯一或显式 task id 匹配；同一阶段存在多个仿真任务时返回 `ambiguous`，不会把一个 native kernel 时间复制到多个任务。

将逐事件 trace 与相同 token 数、相同 runtime 配置生成的静态仿真图连接：

```powershell
py -3.12 tools/join_native_simulator.py `
  artifacts/native_profile_session.trace.json `
  --profile artifacts/native_profile_session.json `
  --output artifacts/native_task_mapping.json
```

该报告同时保存 matched、ambiguous、unmatched 事件和 kernel 阶段计数。本机当前没有 graph-node 到 `TaskSpec.task_id` 的显式关联，因此 4,134 个 native 事件中仍会有大量 ambiguous/unmatched；这反映的是证据不足，不是把它们当成零耗时。


本次 profile 快照（`artifacts/native_profile_session.json`）得到：formal prompt 8 tokens / 6.38 ms，decode 4 tokens / 12.468 ms；CUDA kernel 汇总 28 行，API 汇总 11 行，memcpy 汇总 3 行。代表性 API 总量为 `cudaLaunchKernelExC_v11060` 与 `cudaLaunchKernel` 共 1,503 次、平均约 2.85 μs，`cudaStreamSynchronize` 80 次、平均约 78.23 μs；D2H memcpy 4 次。由于当前二进制没有按 Q/K/V、FFN、KV、logits 插入 NVTX 范围，不能把这些调用平均值直接当成每个仿真任务的独立服务时间；校准配置只启用 launch 前端这一项，且仍保留 `timing_comparison_only` 证据等级。

本次运行固定了以下两侧共同参数：`ctx=512`、`parallel=1`、`batch=64`、`ubatch=64`、`threads=16`、`threads_batch=16`、`gpu_layers=-1`、`flash_attn=off`、`ctk=ctv=f16`、unified KV、continuous batching、显式 warmup（不计入正式 timing）、无 MTP、静态无抢占、greedy sampling（temperature 0 / top-k 1）。

硬件快照：AMD Ryzen 9 9950X3D（16C/32T，4.3 GHz）、NVIDIA GeForce RTX 5080（16,303 MiB，驱动 616.64）。llama.cpp：`0.3.0-dev build1 commit 0f3a71b`。

主存由 WMI 在每次对照启动时重新采集；当前机器报告 `134,939,398,144 B`，4 条 32 GiB 模组，配置时钟 5600 MT/s。GPU PCIe 当前链路为 Gen5 ×8（最大 Gen5 ×16），因此模型配置同时记录了理论端口能力和本次硬件实测链路状态。

结果文件 [`artifacts/native_compare.json`](../artifacts/native_compare.json) 记录了完整命令、实际 `prompt_n/predicted_n`、逐张量 GGUF parity、原生 prompt/decode timing、warmup timing、HTTP 墙钟诊断、仿真器 TTFT/TPOT/E2E 和相对误差。当前快照为：输入 8、输出 8；原生 prompt eval 5.944 ms、decode 18.978 ms、TPOT 2.711 ms/token、总计 24.922 ms；仿真器 TTFT 7.817 ms、TPOT 7.691 ms、E2E 61.657 ms。相对误差分别为 +31.52%、+183.70%、+147.40%。这次仿真使用逐张量 GGUF 绑定后的真实权重字节、实测 Gen5 ×8 PCIe 链路和 `aggregate` 运行模式；剩余偏差主要来自 decode kernel、调度、launch、cache 和 host output 路径。llama.cpp timing 会随 GPU 时钟、后台负载和显存状态变化，复现实验应以 JSON 中保存的命令和配置为准。

复制到本项目的原生证据包括 [`artifacts/native_benchmark_20260912/ENVIRONMENT.json`](../artifacts/native_benchmark_20260912/ENVIRONMENT.json)、[`MODEL_SOURCES.json`](../artifacts/native_benchmark_20260912/MODEL_SOURCES.json)、Qwen2.5-0.5B GGUF 权重、smoke 日志和历史原生矩阵。当前正式对照只使用项目37内的 Qwen GGUF；历史矩阵原有预测侧使用旧的 8×HBM/CIM reference 拓扑，已标记为无效校准证据。
