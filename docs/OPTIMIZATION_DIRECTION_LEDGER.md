# 仿真器优化方向台账

本台账防止优化循环长期停留在单一问题域。它记录“最近检查过什么”和“下一步应该看什么”，不把任何方向标记为已经穷尽。

方向定义与轮换规则见 [OPTIMIZATION_WORKFLOW.md](OPTIMIZATION_WORKFLOW.md)。每次 prepare 必须先更新本文件，再创建候选记录；负向扫描也要登记。

| 方向 | 关注内容 | 最近完成轮次 | 当前状态 | 下一步要求 |
|---|---|---:|---|---|
| 仿真器正确性与语义 | 状态、调度、资源守恒、错误、报告契约、API | H23 | H23 在有限执行链路上为负结论；不能视为全域关闭 | 只有新的契约证据或独立复现才能重开 |
| 运行速度与内存 | 仿真 wall time、峰值内存、图编译、缓存、批处理吞吐 | H24（仅基线） | 已有独立 wall/memory/cProfile 基线；尚未接纳源码候选 | H25 验证 aggregate 报告是否可安全走 `online_summary_dict`，先完成等价性与 wall-time 评估 |
| 计算成本模型 | GEMM、归约、内存、链路、kernel、并发和 shape 泛化 | 未完成系统性轮次 | 未检查 | 与运行速度方向并列优先 |
| 存储硬件建模粒度 | DRAM 家族（DDR/LPDDR/HBM）和 NAND 家族（SSD/NVMe/HBF）的共享介质模型、变体 profile、访问粒度、队列和内部成本 | H24 | DRAM/HBM 共享 aggregate lane 参数已接入；未声称周期精度；NAND 家族未开始 | 下一次存储轮次再处理 NAND/SSD/HBF page/plane；先完成 H25 运行速度候选，避免连续停留在同一方向 |
| 预测精度与泛化 | Native 成对误差、跨模型/硬件/shape 泛化 | 未完成独立验收轮次 | 证据不足 | 需要独立 Native 数据，不能用单测代替 |
| 评估、API 与可视化口径 | 结果边界、评分、失败/缺失、序列化、UI、文本 | H22 | H22 已统一 engine/arrival 边界 | 等运行/成本方向至少各检查一轮后再重开 |
| 开发工具链与可复现性 | 测试隔离、缓存身份、实验记录、证据哈希、恢复 | H21-H23 记录流程 | 局部维护 | 只在证据或恢复失败时重开 |

## 轮换记录

- H18–H21：主要处理仿真器语义、资源守恒和报告/执行链正确性；这些轮次没有完成运行速度或成本模型的系统性测量。
- H22：处理评估边界一致性；没有 Native 精度结论。
- H23：在有限执行链路上做负向发现，未产生源码候选；负结论不代表其他方向已检查。
- H24：在不新增 DRAMProfile 的前提下，HBM/HostMemoryProfile 共用 `memory_service`，新增可选 `parallel_lanes=1` 与真实 endpoint 计费透传；focused 222 passed、全量 3048 passed/4 skipped，无 Native 精度结论。
- 当前推进点：下一轮登记“运行速度与内存”方向，验证报告只需 aggregate summary/requests 时能否使用现有 `online_summary_dict`；随后再回到 NAND 家族建模，保持方向轮换。
- 新增存储硬件定向计划：任务一建立有来源的分层 geometry schema 和参数目录；任务二把 geometry 接入地址映射、访问拆分、队列/冲突、读写/擦除成本和资源计费。详见 [STORAGE_MODELING_WORKPLAN.md](STORAGE_MODELING_WORKPLAN.md)。
