<!-- Slide number: 1 -->
HeteroLLM-Sim  /  覆盖总览
我们已完成哪些建模层级？
HBM / HBF 已连接硬件组织、数据地址、介质时序、资源竞争与推理指标
物理组织也已进入模型

硬件组织与参数
几何、容量、方向带宽、时序、模式

1


HBM
数据与物理地址
张量放置、地址与缓冲生命周期

2
Stack → Channel → PC
→ SID / BG → Bank / Row

介质状态与时序
HBM 行与命令；HBF 页、块、读缓存

3


共享资源与事件
有状态 Core、准入、排队、任务依赖

4
HBF
Stack / 实际 Die → Plane
→ Block → Page → 页内字节

观测与性能输出
流量、busy、延迟与 Token 吞吐

5
“有层级”与“有独立执行资源”需要分开
01 / 05
当前工作区实现  ·  几何示例与模式范围均明确标注

### Notes:
本页描述五个已实现的抽象层级，不是一次访存的五个时间步骤。硬件组织和参数层提供显式几何与物理预算；数据地址层管理张量端点、owner、分配代次和物理范围；介质状态层维护HBM命令与HBF媒体/缓存；共享时间线与事件内核表达资源竞争和任务依赖；性能层再结合计算与链路模型输出系统指标。层级可被寻址，不意味着每一级都有独立总线或队列。已有机制模型不等于已校准目标芯片。来源：src/heterollm_sim/physical_contract.py:154–200、data_motion.py:314,1151,1250、memory_transfer.py:263–335、event_kernel.py:685,1036、memory_service.py:26–84、reporting.py:632,780。

<!-- Slide number: 2 -->
HeteroLLM-Sim  /  物理组织 ①
HBM：已覆盖到 Bank / Row / Column
16H HBM4 参考组织；以下状态与时序以 jedec_hbm 模式说明

组织层级 ≠ 总线数量
Device / 内存端点（一个 DramCore）

Channel
Stack
32 Channel / Stack
两个 PC 共享行命令与列命令总线

Channel
共享 Row CA + Column CA
PC


PC0
每个 PC 独立数据通道 DQ 及其时序历史
独立 DQ
PC1
SID × 4

SID / Bank Group
同样的
SID / BG / Bank 组织
Bank Group × 2 / SID


地址与时序区分维度；没有额外 DQ
Bank × 8 / Group


Row → Column

另有独立 DQ
Bank / Row
独立开行状态；行命中 / 冲突
以及 PRE / ACT 就绪时刻
32 B burst 是传输粒度；堆叠 die 已通过 SID 展开，容量不再重复乘一次 die 数
02 / 05
当前工作区实现  ·  几何示例与模式范围均明确标注

### Notes:
HBM正确组织是Device/内存端点→Stack→Physical Channel→PC→SID→Bank Group→Bank→Row→Column。图中只展开PC0，PC1拥有同样的组织。以16H HBM4参考为例：每Stack 32个物理Channel，每Channel两个PC；每PC四个SID，每SID两组，每组八个Bank，即每PC 64个Bank。bank_group在PC中连续编号，SID从bank/group地址推导，AddressMapping没有单独SID字段。SID/BG不提供额外DQ；Channel共享Row CA与Column CA，各PC有独立DQ，Bank有独立打开行与就绪状态。32B burst是服务粒度，不是新增硬件容器。当前per_stack几何中die映射为0，不能在PC与SID间再画一层独立Die执行资源或重复乘容量。HBM3/4的两个有效1KiB寻址页可共享2KiB激活页；Column是burst位置。analytical为另一简化模式，不能套用本图全部正常工作命令时序。来源：src/heterollm_sim/hbm_geometry.py:132,225,250,267、memory_types.py:577,589,593,629、memory_mapping.py:47,78,90,173、_hbm_control.py:70,79,95,206,309,318,364,486。

<!-- Slide number: 3 -->
HeteroLLM-Sim  /  物理组织 ②
HBF：已覆盖到 Plane / Block / Page
论文原生配置：实际 Die 映射为内部 Channel；Stack 通过通道分组表达

HBF 设备 / 一个 NandCore
host：1 个聚合接口
下面是组成与地址映射展开，不是读写步骤




实际 Die
Plane × 25
Stack 分组
Block：擦除单位
⇔ Core Channel
独立阵列执行资源
介质暂存就绪状态





Page：Read / Program












内部搬运总线
4 KiB 页 → 页内字节




每 Plane 两个读槽
16 个实际 Die / 堆
同 Die 的 Plane 共享


Block / Page 可寻址与计费，
不各自新增一条阵列总线
通用 NAND 坐标：Channel → Target → Core die → LUN → Plane → Block → Page → offset
本配置 Target / Core die / LUN 每级数量均为 1；7 堆 = 112 个内部通道、2,800 个 Plane 阵列
03 / 05
当前工作区实现  ·  几何示例与模式范围均明确标注

### Notes:
本页区分物理HBF层级与通用NAND核心坐标。HBF设备由一个NandCore绑定owner并拥有聚合host接口。NandConfig没有独立stack几何字段；论文原生builder用连续16个Core channel表示一堆，每个实际Die映为一个Core channel并拥有内部搬运总线。Target、Core die、LUN作为通用几何坐标已经实现，但该配置都取1，不能再乘实际die数。每实际Die有25个Plane，各Plane独立阵列日历、媒体暂存就绪状态与两个读槽。同Die的Plane共享内部搬运总线，流量最后汇入设备host。Block是擦除、顺序写前沿和可选管理单位，Page是读/program、缓存标签和字节计费单位；Block/Page不各自拥有独立媒体执行日历。本例Page为4KiB，7堆形成112内部通道和2800个Plane。source：scripts/exploring_hbf_paper_config.py:272–342、src/heterollm_sim/memory_types.py:659,677,684,813、memory_mapping.py:124、nand_core.py:380–409,537,549,553,641,647,680。

<!-- Slide number: 4 -->
HeteroLLM-Sim  /  建模深度
每一层，都维护了哪些状态与资源？
有地址层级，不代表每一层都有独立执行器；共享关系决定并行与等待
| 作用层级 | HBM：jedec\_hbm | HBF：论文原生页服务 |
| --- | --- | --- |
| 数据 / 地址 | 张量 → owner 与缓冲地址 32 B burst 拆分与物理映射 | 相同访问契约，按 Page 拆分 页内偏移、对齐、部分写 RMW |
| 介质状态 | Bank 开行；ACT / PRE / RD / WR / REF 命令间隔、方向切换、刷新 | Plane 读 / Program / Erase、介质暂存 双槽标签 / ready / reader\_end / LRU |
| 共享资源 | Channel 共享 Row CA / Column CA PC 各有 DQ；Bank 各有行状态 | 每实际 Die 的内部通道由 Plane 共享 每 Plane 独立阵列，设备 host 汇聚 |
| 事件 / 观测 | 有状态 Core → 资源日历 → 任务依赖 完成时间、流量、命令、等待、busy | 再结合计算与链路模型 → 系统指标 makespan、TTFT、TPOT、Token 吞吐 |
统计口径分开：逻辑 / 物理字节；服务跨度 service_ns / 资源 busy / 总耗时 makespan
04 / 05
当前工作区实现  ·  几何示例与模式范围均明确标注

### Notes:
这是一张已实现范围的映射表，不是新增功能计划。数据地址层对权重、KV、激活和缓冲代次、owner、范围进行绑定和解析；两种介质按burst或page拆分。HBM的jedec_hbm维护Bank开行和正常工作命令、时序间隔、方向切换与刷新。HBF维护Plane媒体日历、媒体暂存和两个读缓存槽，槽标签、ready、最晚reader_end和LRU均有状态。资源层体现共享：Channel的两PC共用Row/Column CA但各有DQ；同实际Die的Plane共用内部通道，而Plane阵列资源独立，最后汇聚host。有状态Core结果在实际任务dispatch时回写事件内核，结合计算与链路模型产出系统指标。任务依赖来自任务图和访问描述，不应宣传为自动发现所有跨任务依赖。service_ns是可能重叠的完成跨度，不能当资源busy或整体makespan。来源：src/heterollm_sim/data_motion.py:314,1151,1250,1330–1496、event_kernel.py:685、memory_transfer.py:263–335、memory_service.py:26–84、_hbm_control.py:70–206、nand_core.py:380–409,647,1901–1956、reporting.py:632,780。

<!-- Slide number: 5 -->
HeteroLLM-Sim  /  范围与边界
已实现范围与模型边界
完成的是可配置机制模型；不同控制与协议模式启用不同能力


媒体核心：已实现
可选控制 / 管理：已实现
HBM：组织、地址、行状态、命令与资源
strict OCP：页命令准入、顺序写、自动擦除
HBF：组织、块页操作、缓存与总线
OCP HOST 暂存：页合并 + 日志式 FTL + GC
论文 generic 基线不启用上层 FTL / GC
另有可选磨损反馈；均按配置启用
当前边界

1

2

3
电路与内容
完整协议
参数与校准
不模拟逐 bit / cell 内容、
ECC 与 retention 等行为
HBM 正常工作命令子集；
不宣称完整 JEDEC / AXI 复现
标准参考、论文输入与研究假设
不等于目标芯片实测校准
05 / 05
当前工作区实现  ·  几何示例与模式范围均明确标注

### Notes:
范围需要分为媒体核心、可选管理和未覆盖细节。HBM有analytical与jedec_hbm，后者覆盖所示正常工作命令与时序，但不覆盖训练、自动预充电和完整刷新推迟策略。NandCore媒体核心本体不内置FTL/GC；论文generic_nand基线不启用上层管理。系统上层已存在strict ocp_hbf_0_7专用的HbfStagingController，提供有限HOST DRAM页合并、小型日志式FTL、异地写映射、free-block管理、victim选择、有效页复制及GC；strict OCP还约束页命令准入、完整对齐写、block顺序program和page-0自动erase。可选磨损反馈不等同于硬件级wear-level remapping。不要把模型已实现说成所有模式默认全部启用。仍不模拟逐bit/cell内容、电气电路、ECC/retention、完整AXI IDs与错误响应，也没有独立OOB/TSV阶段。参数有标准参考、论文输入和本地假设之分，尤其HBM4论文对照的tRRD/tFAW借用HBM3E参考；尚未目标芯片实测校准。来源：src/heterollm_sim/_hbm_control.py:1、dram_core.py:90、nand_core.py:353,608,680,716、memory_types.py:740–771、hbf_staging.py:476,491,503,864,908、scripts/exploring_hbf_paper_config.py:207–258,314。