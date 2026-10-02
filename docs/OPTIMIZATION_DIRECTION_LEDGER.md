# 仿真器优化方向台账

本台账防止优化循环长期停留在单一问题域。它记录“最近检查过什么”和“下一步应该看什么”，不把任何方向标记为已经穷尽。

方向定义与轮换规则见 [OPTIMIZATION_WORKFLOW.md](OPTIMIZATION_WORKFLOW.md)。每次 prepare 必须先更新本文件，再创建候选记录；负向扫描也要登记。

| 方向 | 关注内容 | 最近完成轮次 | 当前状态 | 下一步要求 |
|---|---|---:|---|---|
| 仿真器正确性与语义 | 状态、调度、资源守恒、错误、报告契约、API | H31 | H31 重新扫描 Web/serving/reporting/资源 fail-closed 路径，未发现新的可复现错误到成功折叠；保留有限负结论 | 下一轮转计算成本模型；新的语义候选需有更窄的独立复现 |
| 运行速度与内存 | 仿真 wall time、峰值内存、图编译、缓存、批处理吞吐 | H36（活动） | H29 收口 CLI 文本路径重复 `report_dict` 构建；H36 重新检查 run_scenario/批处理执行和内存路径 | 先取得独立 wall-time/峰值内存候选或负结论，不重复 CLI 修复 |
| 计算成本模型 | GEMM、归约、内存、链路、kernel、并发和 shape 泛化 | H32 | H32 独立 arithmetic/link/owner/pipeline probes 全通过，未发现排除 H27 后的新可复现公式缺陷；MMA occupancy/collective 仍明确是未实测或参数化 | 下一轮转开发工具链/可复现性；新的成本候选需更窄的执行证据 |
| 存储硬件建模粒度 | DRAM 家族（DDR/LPDDR/HBM）和 NAND 家族（SSD/NVMe/HBF）的共享介质模型、变体 profile、访问粒度、队列和内部成本 | H30 | H30 将已知 MemoryPosition offset 传入共享 NAND page/RMW/program 计费，并补齐 endpoint/legacy alias 可选入口；geometry 数值仍 profile-specific | 下一轮转评估/API或正确性语义；只有新的来源与实际 erase/FTL 调用链才重开存储方向 |
| 预测精度与泛化 | Native 成对误差、跨模型/硬件/shape 泛化 | H34（活动） | H28 收口请求集合覆盖契约；H34 重新检查输入身份、shape/hardware coverage 和 fallback 泛化边界，无 Native 时只做机制验证 | 先做独立输入/泛化 source scan；不重复 H28 请求覆盖修复 |
| 评估、API 与可视化口径 | 结果边界、评分、失败/缺失、序列化、UI、文本 | H35（活动） | H28 收口 request coverage；H35 重新检查结果状态、失败/缺失和可视化/API 一致性 | 先独立评估/API source scan；不重复 H28 request coverage |
| 开发工具链与可复现性 | 测试隔离、缓存身份、实验记录、证据哈希、恢复 | H21-H23 记录流程 | 局部维护 | 只在证据或恢复失败时重开 |

## 轮换记录

- H18–H21：主要处理仿真器语义、资源守恒和报告/执行链正确性；这些轮次没有完成运行速度或成本模型的系统性测量。
- H22：处理评估边界一致性；没有 Native 精度结论。
- H23：在有限执行链路上做负向发现，未产生源码候选；负结论不代表其他方向已检查。
- H24：在不新增 DRAMProfile 的前提下，HBM/HostMemoryProfile 共用 `memory_service`，新增可选 `parallel_lanes=1` 与真实 endpoint 计费透传；focused 222 passed、全量 3048 passed/4 skipped，无 Native 精度结论。
- H25：运行速度方向负向扫描关闭当前 report aggregate 假设；没有现有 aggregate-only 生产调用，未改源码，H24 的 23.9x probe 仅保留为开发基线。
- H26：将现有 HBF cold-page 路径最小泛化为 opt-in `nand_media_v1`，SSD/high_io_ssd/NVMe/HBF 显式 contract 共用 page/RMW/program/plane/queue 链路；focused 131 passed、全量 3057 passed/4 skipped，无 Native/cycle-accuracy 结论。
- H27：修复 serialized/overlapped 内存成本中 service physical bytes/energy 与 ResourceDemand/cache metadata 仍使用 logical payload 的不一致；focused 302 passed，排除无关 SQLite 测试的套件 2912 passed/4 skipped，SQLite 测试单独 1 passed，无 Native 精度结论。
- H28：修复 `/api/simulate-score` 只遍历 simulated request ID 导致 Native 多请求被静默忽略的问题；加入双方请求覆盖计数，部分集合返回 `partial_reference` 且不允许 `passed=true`；focused 38 passed，相关回归 225 passed，排除无关 SQLite 测试的套件 2913 passed/4 skipped，SQLite 测试单独 1 passed，无 Native 精度结论。
- H29：已登记运行速度/内存方向；先独立扫描真实执行链路，不把 H25 的 aggregate 负结论扩展到全域。
- H29：修复 CLI 文本输出重复 `report_dict` 构建，`format_report(data=...)` 复用同一 payload；独立父/候选探针中位数 123.2554ms→62.5194ms，文本 SHA 一致，完整套件 3068 passed/4 skipped。
- H30：已登记存储硬件建模粒度方向，保持 NAND 家族（SSD/HBF/NVMe）共享边界，检查 profile-specific geometry、地址映射和访问拆分调用链。
- H30：修复已知 `MemoryPosition.offset_bytes` 在 NAND 计费入口被丢弃的问题；128B@页尾跨页读从 4096B 变为 8192B，写从 8192B 变为 16384B，并补齐 endpoint/legacy alias 可选偏移；focused 83 passed、全量 3070 passed/4 skipped，无 Native 精度结论。
- H31：已登记评估/API或正确性语义方向，离开已收口存储模型。
- H31：已登记正确性/语义方向，检查独立执行链路中的错误、缺失和 unknown 边界，不重复 H30 存储 geometry。
- H31：独立扫描未发现新的可复现错误到成功折叠；Web typed errors、Native coverage fail-closed、serving/reporting unknown/degraded/incomplete 和资源 fail-closed 路径均保留有限负结论，未改源码。
- H32：已登记计算成本模型方向，检查 shape/并发/owner 守恒，避免重复 H27 已收口的 backing demand。
- H32：独立成本扫描未发现新缺陷；GEMM/reduction shape bytes、链路单位、owner capacity/physical service 和 finite pipeline bounds 全部通过解析探针，未改源码，保留 MMA occupancy/collective 未实测边界。
- H33：已登记开发工具链与可复现性方向，检查实验身份、证据哈希、恢复和基线配对链路。
- H33：独立扫描确认 H30-H32 记录的 artifact 哈希、round/source/parent/remote 身份一致；未发现新的可复现配对或恢复缺陷，未改源码，保留环境性 `pytest-of-A` 权限警告边界。
- H34：已登记预测精度与泛化方向，检查输入身份、shape/hardware coverage 和 fallback 泛化，不重复 H28 请求覆盖修复。
- H34：独立扫描确认 scenario/hardware/scheduler/mapping fingerprints 随 shape/input 变化，model/runtime coverage 与 unknown/fallback 显式；无 Native 精度结论，未改源码。
- H35：已登记评估/API与可视化口径方向，检查结果状态、失败/缺失和 UI/API 一致性，不重复 H28 request coverage。
- H35：独立 partial/empty reference controls、HttpError/API/report/UI 状态扫描均通过，未发现新评估口径缺陷，未改源码。
- H36：已登记运行速度/内存方向，检查实际 `run_scenario`/批处理执行和峰值内存路径，不重复 H29 CLI report reuse。
- 当前推进点：H36 活动，等待独立 runtime execution source scan。
- 新增存储硬件定向计划：任务一建立有来源的分层 geometry schema 和参数目录；任务二把 geometry 接入地址映射、访问拆分、队列/冲突、读写/擦除成本和资源计费。详见 [STORAGE_MODELING_WORKPLAN.md](STORAGE_MODELING_WORKPLAN.md)。
