# V4 执行与观察层

V4 只有一个 `UnifiedEventKernel`。控制面 Runtime DAG 与 GPU/模型 inference lowering 使用同一内核、依赖语义和共享资源竞争；模块拆分不代表多个后端或多套模拟器。

控制面从容量检查开始，经过 placement、allocation、权重 cache/NVMe/IOMMU/DMA/PCIe、批次与算子调度、命令构建/提交以及 GPU command/MMU/L2/VRAM controller，最后把因果控制权交给计算 lowering。NVMe、IOMMU、DMA 和 GPU controller 的队列/并发限制在聚合事务层参与调度。

观察层由 `RetentionPolicy` 表示；它是统一内核的 retention policy，而不是后端选择：

- `exact`：保留当前任务模型中已实现的任务/区间；
- `streaming`：精确归约指标，但仅保留有界代表轨迹；
- `aggregate`：保留 cohort、batch 和控制器事务的聚合观察。

这些标签只描述保留粒度。即便是 `exact`，也不代表逐 CPU 指令、缓存行、DMA descriptor、packet/flit 或周期级硬件模型。所有预测仍属于 `ANALYTICAL`，除非有独立的适用校准证据。
