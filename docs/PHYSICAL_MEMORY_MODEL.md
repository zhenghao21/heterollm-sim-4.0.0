# 物理层级读写事务模型

仿真器现在提供一条轻量级的物理事务路径。它保存组织参数、地址映射、资源可用时间和行/Page Buffer 状态，不保存真实存储内容，也不实现 FTL、GC、刷新控制器、缓存替换或完整协议状态机。

五个模块各自只有一个职责：

- `memory_types.py` 定义 `AccessRequest`、DDR/LPDDR/HBM 的 `DramConfig`、SSD/HBF 的 `NandConfig`，以及统一的事务和批次结果。
- `memory_mapping.py` 将显式字节地址映射到通道、Rank、Bank、Die、Plane、Block、Page，并在 Burst 或页/块边界拆分请求。
- `dram_core.py` 生成开行、换行、读写流水和 Burst 数据阶段。Bank 保存打开的 Row，命令发起、首数据延迟和数据总线分别计时，因此同一 Row 的连续 Burst 不会重复支付首数据延迟。
- `nand_core.py` 生成页读、页编程和块擦除阶段。阵列执行单元、Page Buffer、内部通道和主机接口分别竞争；部分页写按配置执行简化读改写，擦除不产生数据搬运。
- `memory_transfer.py` 按唯一 `resource_id` 保留通路的可用时间，并从已完成事务汇总实际带宽。

请求拆分使用生成器逐段执行。`max_expanded_segments` 只限制结果中保留的
mapping/stage 详情；超过上限后仍会继续计时和汇总计数，但清空逐段详情，避免
为整 GiB 访问保留数千万个对象。上游仍应优先按地址区间分批提交。

物理状态由显式的 `PhysicalRuntimeContext` 持有。同一 context 中，同一物理
owner 只能绑定一套几何配置；不同 context 可以交错执行而互不污染。没有提供
runtime 的 `PhysicalService.price()` 和 `endpoint_service()` 是纯预估，会使用临时
核心副本；正式提交必须传入 `runtime=context` 和实际事件 `arrival_ns`。没有显式
地址的物理事务会被拒绝，避免把所有访问静默映射到地址 0。

最小调用示例：

```python
from heterollm_sim.memory_types import AccessRequest, make_hbm_config, make_hbf_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore
from heterollm_sim.data_motion import PhysicalRuntimeContext

hbm = DramCore(make_hbm_config(interface_bandwidth_gb_s=1000.0))
dram_batch = hbm.run([
    AccessRequest("r0", "read", 0, 4096),
    AccessRequest("r1", "read", 8192, 4096),
])

hbf = NandCore(make_hbf_config())
nand_result = hbf.execute(AccessRequest("page", "read", 0, 4096))

# 规划阶段预估不提交状态；事件执行阶段显式提交到本次运行的 context。
runtime = PhysicalRuntimeContext()
# service.price(..., runtime=runtime, arrival_ns=event_start_ns)
```

接入现有数据移动流程时，统一使用 `AccessRequest` 和组件元数据中的
`physical_memory_config` 配置；配置值直接使用 `DramConfig`/`NandConfig` 及其
canonical 字段。`AccessRequest` 的地址和连续长度决定映射、拆分与资源竞争，
不会隐式假设均匀分散访问。

结果中的 `logical_bytes` 用于实际带宽，`host_transfer_bytes`、`internal_transfer_bytes`、
`physical_read_bytes`、`physical_write_bytes`、`pages_read`、`pages_programmed` 和
`erase_operations` 用于核对物理代价。调用方使用上述统一入口以及
`make_ddr_config`、`make_lpddr_config`、`make_hbm_config`、`make_ssd_config`、
`make_hbf_config`。

规划器生成的 `TaskSpec` 可以携带 `metadata["memory_access"]` 描述（`operation`、`address`、`byte_count`、`physical_owner`）及对应的 `physical_memory_config`。`UnifiedEventKernel` 在任务真正出队的事件时刻调用 `resolve_physical_task`，并把本次运行唯一的 `PhysicalRuntimeContext` 传给核心；核心返回的 `physical_execution`、`physical_arrival_ns` 与 `physical_completion_ns` 是该访问的权威物理结果。静态 `ResourceDemand` 在这一类任务中只用于任务索引和报告占位，不会再次把设备阶段排队或重复计费。

因此，规划阶段生成的 endpoint demand 只是预估；正式运行必须经过事件内核的 `memory_access` 描述，才能更新 Row / Page Buffer、通道时间线与请求接纳状态。事件内核会用核心返回的实际完成时间推进依赖，并将核心的每资源占用写入报告；没有该描述的普通通信阶段仍按原有静态 demand 调度。
