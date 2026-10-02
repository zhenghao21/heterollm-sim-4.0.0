# 存储硬件建模定向工作计划

本计划是“存储硬件建模粒度”方向下的两个定向任务。它们不能与普通正确性修复混成一个候选：任务一建立有来源的结构事实，任务二才把结构事实接入访问、排队和成本模型。

## 适用范围和证据边界

覆盖以下存储介质：

- 易失性内存：DDR4/DDR5、LPDDR、HBM2E/HBM3/HBM3E/HBM4（按实际可取得的标准版本和产品资料分层）。
- 非易失性存储：NAND Flash、企业级 SSD、NVMe SSD、CXL Type-3 中的存储介质，以及项目已有的 HBF 组件。
- 项目中的 HBF 不默认存在 JEDEC 内部阵列标准。HBF 的主来源应是 OCP HBF Architecture Specification；底层 NAND 接口、可靠性或产品参数再分别引用 ONFI、JEDEC 或厂商公开资料。OCP v0.7.0 是架构规范，不是已量产器件的独立实测数据。

来源优先级：

1. 公开的 JEDEC/OCP/ONFI 规范或正式勘误；
2. 芯片、封装、SSD 或控制器厂商的数据表、产品手册和编程手册；
3. 有明确实验方法的论文或公开测量；
4. 参数化假设，只能作为未证实配置，不能伪装成器件事实。

JEDEC 文档若需要登录、付费或授权，记录标准编号、版本、可见摘要和获取状态；缺失正文的字段保留 `unknown`，不能从容量、带宽或端到端时延反推 plane、bank、page 或 die 数量。

## 定向任务一：建立分层存储结构和来源目录

目标是把现有“一个 memory component + 容量/带宽”抽象扩展为可审计的层级结构。此任务只建立事实、单位、来源和不确定性，不改变运行时调度。

每类器件至少调查并区分：

| 介质 | 需要调查的内部层级和粒度 |
|---|---|
| DDR/LPDDR DRAM | channel、sub-channel、rank、bank group、bank、row/column；burst length、访问/传输粒度、读写方向、刷新、时序约束、控制器队列 |
| HBM | stack、base/die、channel、pseudo-channel、独立端口、burst/最小传输单位、堆叠带宽分配、读写方向和可公开的 bank/时序字段；未公开字段保持 unknown |
| NAND/SSD | controller、host channel、package、die、CE/LUN、plane、block、page、页读/编程/块擦除；page read/program/erase 粒度、multi-plane、并行度、FTL、GC、磨损和队列 |
| HBF | base die、NAND/core-die stack、host channel、plane、block、page、AXI/UCIe 接口、对齐与小写合并、program/erase 约束、可靠性和遥测；以 OCP 规范为主，不冒充 JEDEC 事实 |

交付物：

- 一个版本化的 storage geometry schema，明确 `known`、`derived`、`parameterized`、`unknown` 四类状态；
- 每个 preset 的层级、容量守恒、带宽守恒、单位和来源 URL/标准号；
- 不能确定时的显式限制和待补资料清单；
- geometry 读取 API 与序列化兼容测试；旧的 aggregate preset 必须能继续加载。

任务一的验收条件：同一个字段不能同时以“厂商事实”和“参数假设”出现；容量、通道、die、plane、page 的推导必须有公式和独立来源；没有来源的字段不能进入精确预测路径。

## 定向任务二：把层级结构接入访问和成本模型

目标是让存储层级实际影响仿真器的访问拆分、排队、冲突、资源计费和结果解释，而不是只增加 metadata。

至少覆盖：

- 地址到 channel/rank/bank/plane/die/page 的可复现映射；
- 读、写、刷新、program、erase 的不同粒度和方向成本；
- page boundary、row/bank conflict、plane conflict、die/queue 并行和控制器队列；
- SSD/NAND 的 FTL、GC、写放大、后台擦除和 host/device queue；
- HBF 的 page 对齐、小写合并、顺序 program、host channel 并行和 UCIe/AXI 到 base die 的边界；
- DMA、PCIe/CXL/UCIe/内存控制器等外部链路与内部介质成本的分层计费，避免同一 payload 重复收费；
- 输出每次访问的 logical bytes、physical bytes、page/program/erase 次数、排队等待、冲突原因、命中的层级和证据来源。

任务二的验收方法：

1. 先用解析场景验证容量、page 数、plane 并行和写放大守恒；
2. 用同一 geometry 的 aggregate 模型作为控制，比较层级模型的 wall time、峰值内存、队列等待和资源账本；
3. 为每类介质准备整页、跨页、跨 plane、读写混合、队列饱和、刷新/GC 触发和空闲控制；
4. 若有独立设备或论文测量，再做成对延迟/吞吐误差；没有实测时只报告机制一致性，不称为精度改善；
5. 最终回归必须覆盖旧 aggregate preset、已有 HBM/HBF 架构组合和不支持/未知字段的 fail-closed 行为。

任务二的禁止事项：不能用端到端目标时延反推内部 page/plane 数；不能把 HBF 的架构规格当成量产器件实测；不能为未公开的 DRAM/HBM bank 或 SSD NAND geometry 填入看似精确的默认值。

## 执行顺序

1. 先做任务一的 schema、来源目录和当前 preset 盘点；
2. 独立审查任务一的字段来源、单位和守恒公式；
3. 选择一个有完整来源的介质作为任务二第一条执行链，优先已有 HBM 或 HBF preset；
4. 接入运行时和成本模型，补控制场景后再扩展到其他介质；
5. 两个任务分别记录为独立候选，但只有任务一稳定后任务二才允许修改实际计费链路。
