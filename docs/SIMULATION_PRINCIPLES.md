# HeteroLLM Simulator 4.0.0 仿真原理

**文档口径：** HeteroLLM Simulator 4.0.0｜Authoring Schema 4.0.0｜Canonical IR 1.1｜系统级、任务/事务级解析仿真。

## 因果范围

一次运行既包含模型推理的计算、内存与互连需求，也包含把工作交给 GPU 前的动态 CPU control plane。控制面不是零成本前置步骤：容量核算、placement、allocation、权重 cache/NVMe 路径、IOMMU、DMA、PCIe、批次/算子调度、命令构建/提交和 GPU controllers 都以有依赖的资源需求进入事件内核。

控制面 Runtime DAG 和推理 lowering 统一提交给 `UnifiedEventKernel`，因此控制动作、控制器事务与计算任务在同一时间轴上竞争共享资源，而不是先在模拟器外算出静态映射。

GPU 执行侧仍按 GEMM、element-wise、reduction、memory、communication 分项降低。KV Cache/Offload、TP/PP/EP、collective、Chunked Prefill、连续批处理、抢占和模型内 MTP 的指标依赖同一因果资源图，不能把 FLOPs/TOPS 或单一带宽除法当作端到端时间。

## 聚合而非微架构展开

控制面 Runtime IR 只表示：同类 `InstructionBatch`、某 controller 的 `ControllerTransactionBatch` 和 batch 级 RuntimeAction。计数与字节数记录在聚合对象上，而不生成每条 CPU instruction、cache line、page、DMA descriptor、NVMe command 或 PCIe packet 的任务。该粒度可表现控制器队列、带宽、延迟和共享资源竞争，却不声称模拟操作系统、驱动、page replacement、packet/flit、NVMe die/channel/GC 或周期级 cache/NoC。

CPU profile 表示 cache/DRAM、channel、队列与请求 batch；NVMe/profile 表示 page cache、页粒度、读写带宽、延迟和 IO 队列；PCIe/DMA/IOMMU profile 表示 lane、DMA engine、队列和 translation walk；每 GPU controller profile 表示 command processor、MMU/TLB、L2 和 VRAM controller。这些都是模型输入，而非已校准硬件计数器。

## placement 与状态

V4 不提供旧手动/自动映射流程。动态 control plane 对当前场景进行 placement，容量、组件能力、权重驻留和拓扑可达性始终是硬约束。决策生成 placement patch 和可审计的 control-plane metadata；求解墙钟时间不计入模拟时间。

权重、KV 和状态必须分别通过容量、可写性和路径检查。HBF/SSD/NVMe 是后备或事务端点，不能伪装成活动 HBM/CIM 容量。GPU/HBM、CPU/Host Memory、端点、DMA 和链路的竞争会分别进入报告。

## 证据与限制

结果默认 `ANALYTICAL`，不是校准或实测。`RetentionPolicy`（retention policy）的 `exact`、`streaming`、`aggregate` 只说明观察保留策略，均不等价于 cycle/silicon accuracy。除非额外导入有适用范围、holdout 验证与明确 calibration profile 的数据，报告不得宣称预测已校准。
