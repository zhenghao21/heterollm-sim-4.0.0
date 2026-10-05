# 物理层级读写事务模型

仿真器现在提供一条轻量级的物理事务路径。它保存组织参数、地址映射、资源可用时间和行/Page Buffer 状态，不保存真实存储内容，也不实现 FTL、GC、刷新控制器、缓存替换或完整协议状态机。

五个模块各自只有一个职责：

- `memory_types.py` 定义 `AccessRequest`、DDR/LPDDR/HBM/GDDR 的 `DramConfig`、SSD/HBF 的 `NandConfig`，以及统一的事务和批次结果。GDDR6、GDDR6X、GDDR7 共享 `DramCore`，代际只通过配置与元数据区分。
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
`make_ddr_config`、`make_lpddr_config`、`make_hbm_config`、`make_gddr_config`、`make_ssd_config`、
`make_hbf_config`。

GDDR 配置约定：`data_lanes`/`data_width_bits` 表示数据组织，`interface_bandwidth_gb_s`
表示整套接口的十进制 GB/s；若使用 `effective_pin_data_rate_gbps` 与总位宽，带宽按
`effective_pin_data_rate_gbps × data_width_bits × data_lanes ÷ 8` 计算一次。GDDR6 使用 NRZ、GDDR6X 使用 PAM4、
GDDR7 使用 PAM3 仅作为来源说明，不能再对有效数据率乘一次编码系数。组件拓扑使用
`kind="gddr"`，物理配置使用 `kind="GDDR"` 并保留 `generation`。

组件预设的 `burst_interval_ns` 是每条 32-bit 数据通路的分析调度预算，按
`burst_bytes / (pin_data_rate_gbps × data_width_bits / 8)` 推导。它用于避免
命令资源人为地把接口峰值压到一个更低的共同上限，不代表 JEDEC 的 tCCD 或器件实测
时序；真实控制器时序仍应通过显式配置覆盖。启用 `stateful_l2` 时，规划器先保留
物理地址锚点并附加 L2 contract，事件出队后先更新 L2 状态，再把 `hbm_read_bytes` /
`hbm_write_bytes` 的未命中和写回流量重建为 GDDR 物理描述。纯命中任务不会提交空的
DRAM 事务，但仍保留 L2 的状态更新和缓存耗时。

规划器生成的 `TaskSpec` 可以携带 `metadata["memory_access"]` 描述（`operation`、`address`、`byte_count`、`physical_owner`）及对应的 `physical_memory_config`。`UnifiedEventKernel` 在任务真正出队的事件时刻调用 `resolve_physical_task`，并把本次运行唯一的 `PhysicalRuntimeContext` 传给核心；核心返回的 `physical_execution`、`physical_arrival_ns` 与 `physical_completion_ns` 是该访问的权威物理结果。静态 `ResourceDemand` 在这一类任务中只用于任务索引和报告占位，不会再次把设备阶段排队或重复计费。

当算子级方向字节经过 Rank 分片或缓存裁剪后与本地 demand 不一致时，规划器会按原始读写比例重建本地读写字节，并在 `gddr_directional_reconstruction` 中保留原值、重建值和来源；显式 `buffer_accesses` 则要求字节数严格守恒。两条路径都不会把不确定流量静默改成全读。

因此，规划阶段生成的 endpoint demand 只是预估；正式运行必须经过事件内核的 `memory_access` 描述，才能更新 Row / Page Buffer、通道时间线与请求接纳状态。事件内核会用核心返回的实际完成时间推进依赖，并将核心的每资源占用写入报告；没有该描述的普通通信阶段仍按原有静态 demand 调度。

## 正式接入路径与验收不变量

生产接入只有一条物理事务路径：`TopologyRouter` 先解析端点和链路，`expand_access` 将
`DataAccess` 展开为有依赖关系的 `MotionPhase`，`ExpandedMotion.to_tasks` 生成携带
`physical_memory_config` 与 `memory_access` 的 `TaskSpec`，然后由 `UnifiedEventKernel`
在任务的实际到达时刻调用 `resolve_physical_task`。规划阶段的价格只用于排序和预估，不能
代替这次提交；内核返回的 `physical_execution.completion_ns` 是任务完成时间的唯一来源。

每个资源 ID 都必须绑定一个明确的物理 owner。未声明共享关系的设备使用 owner 命名空间，
因此两个独立设备不会因为都叫 `dram:bank:0` 而互相阻塞；链路若声明相同的
`resource_id`，则必须在同一个 `PhysicalRuntimeContext` 中复用同一条资源日历。报告中的
资源区间应直接来自核心返回的 reservation 起止时刻，不能用“任务开始时间 + 累计 busy”
重新拼接出一条看似连续的区间。

`TaskSpec` 可以同时保留物理访问和其它需求，例如 `gpu.compute` 或链路 demand。解析物理
任务时只替换内存占位需求，不能静默删除同一任务中的其它资源；如果某个阶段确实只能是
纯内存任务，应在规划时拆成显式 DAG 节点。

### `storage_probe` 最小复测格式

小型回归场景可在 `workload.metadata.storage_probe` 中声明一组规范化访问。每项至少包含
`component_id`、`operation`（`read`、`write` 或 `erase`）、`byte_count` 和
`page_offset_bytes`。这些条目由 planner 追加到普通任务图，仍经过 endpoint、内核和报告
链路；它们不是绕过调度器的 direct-core fixture。相邻条目可以验证同一 DRAM row 的命中、
换行冲突，以及 NAND 的页读、编程和块擦除计数。

同一组请求从单请求价格接口、批量接口和事件内核提交时，应具有相同的访问次数、物理字节、
完成时间和 owner。若显式地址缺失，物理配置请求必须失败并指出地址错误；普通（没有
`physical_memory_config`）端点不得携带 `memory_access` 标记，以免事件内核误分派到物理分支。

### 事件内核和报告的边界

endpoint 生成的静态 `ResourceDemand` 是规划占位。任务带有 `memory_access` 后，`UnifiedEventKernel` 在实际到达时刻调用同一个 `PhysicalRuntimeContext`；解析器必须只替换该访问对应的占位需求，并保留同一任务中的 `gpu.compute`、普通链路等需求。因此，改变规划阶段的预估时长不能改变行命中后的物理完成时间或其它并行工作的结果。

主机传输若声明了例如 `shared:pcie`，普通 `ResourceDemand` 和 NAND 的 `HOST_TRANSFER` reservation 必须写入同一条资源日历。提交顺序不改变“同一时刻区间不可重叠”的约束；没有显式共享 ID 的设备则使用各自 owner 命名空间，可以并行。

一次访问组的结果归约按以下规则执行：读写字节、页/突发次数和操作数逐项累加；整体到达与完成时间取组边界；资源区间直接使用核心 reservation 的真实起止时间，并保留搬运字节和能耗。单请求、批量接口和混合读写从同一配置提交时，应得到相同的分类计数与物理字节。

复合访问先完成所有子项的操作、地址、范围和 owner 校验，再登记 runtime、Row/Page Buffer 或资源时间线。任一子项被拒绝时，context 的 timeline 与 runtime 注册必须保持调用前状态；事件内核可将任务重新放回 ready 队列，修正输入后重试不会重复执行已提交前缀。
