# 存储硬件建模定向任务

存储建模按两种**介质家族**推进，而不是为 DDR、HBM、SSD、HBF 各写一套模型：

- DDR、LPDDR、HBM 都是 DRAM 家族，复用一个 DRAM 访问和成本模型；
- SATA/NVMe SSD、NAND 设备和 HBF 都是 NAND 家族，复用一个 NAND 介质模型；
- 家族变体只提供接口、组织方式和控制器差异，不复制底层介质逻辑。

目标是让存储访问的粒度和并行度进入真实的访问、排队和成本链路。未知字段保持 `unknown`，参数化字段必须标为 `parameterized`；不为未公开的内部结构填入看似精确的默认值。

来源优先级为公开 JEDEC/OCP/ONFI 规范、厂商数据表或编程手册、可复现实验论文、最后才是明确标注的参数化假设。JEDEC 正文若需要登录或授权，只记录标准号、版本、公开摘要和获取状态。HBF 的主来源是 OCP HBF 架构规范；不要把 HBF 宣传资料当成量产器件实测。

## 任务一：统一 DRAM 家族的粒度和成本模型

### 最小模型

复用一个 DRAM 模型，至少支持这些可选参数：

- channel/sub-channel；
- rank 或等价逻辑分区；
- bank/bank-group（来源没有时为 `unknown`）；
- burst 或最小传输字节；
- row hit、row miss、读写切换、刷新和控制器队列；
- channel/bank 并行度、读写带宽和时序参数。

DDR/LPDDR/HBM 的区别放入 profile：

- DDR/LPDDR：模块/封装接口、channel/sub-channel、rank、bank-group 和刷新配置；
- HBM：stack、channel、pseudo-channel、封装带宽和接口组织；不能强行套用 DDR DIMM 的 rank 语义；
- 三者共享 DRAM 的 row/bank/burst/读写切换/刷新/队列成本逻辑。

### 接入和验证

同一个 DRAM 访问入口必须使用 profile 解析地址、拆分 burst、选择 channel/bank、产生队列等待，并把读写切换、刷新和冲突成本计入资源账本。先覆盖整 burst、跨 burst、row hit/miss、读写混合、channel 饱和和刷新控制；旧的 aggregate DRAM/HBM preset 必须继续工作。

验收只要求证明：访问字节守恒、burst 拆分正确、并行 channel 不重复收费、旧 profile 与新 profile 的控制行为可解释。没有独立设备测量时，只报告机制一致性和仿真 wall time/内存，不声称 Native 精度改善。

## 任务二：统一 NAND 家族的粒度和成本模型

### 最小模型

复用一个 NAND 模型，至少支持这些可选参数：

- channel、package/die、plane、block、page；
- page read、program、block erase 的粒度和成本；
- page boundary、plane/die 并行和顺序 program 约束；
- host 访问粒度、队列深度和后台工作标记。

SSD/HBF 的区别放入 profile：

- SATA/NVMe SSD：控制器、FTL、host/device queue、GC 和写放大；没有来源时把 FTL/GC 建模为显式参数或关闭，不推断内部实现；
- HBF：base die、NAND/core-die stack、AXI/UCIe 主机边界、page 对齐、小写合并、host channel 并行和顺序 program；以 OCP 规格中已公开的字段为准；
- 两者共享 NAND 的 page/block/plane/die 访问和 program/erase 成本逻辑。

### 接入和验证

同一个 NAND 访问入口必须按 page/block/plane/die 拆分访问，区分 read/program/erase，表达队列等待和并行度，并输出 logical bytes、physical bytes、page/program/erase 次数、等待时间和触发原因。先覆盖整页、跨页、跨 plane、读写混合、队列饱和和后台工作控制；旧 SSD/HBF aggregate preset 必须继续加载。

验收只要求证明：page 和 block 守恒、跨页拆分正确、program/erase 不被当成普通 read、并行度和写放大不重复计费、未知字段 fail-closed。没有独立设备或论文测量时，只报告机制一致性和仿真工程指标。

## 统一边界

- 不先建立四个独立模型，也不为每个供应商复制一套 planner/cost model；使用一个家族模型加 profile。
- 不在第一阶段建完整 DRAM 控制器、SSD FTL 或 NAND 固件仿真；只有独立证据证明它们是主要误差来源时，才增加最小必要状态。
- 不用端到端时延反推 page、plane、bank、die 或 queue 参数。
- 两个任务可以分别验收，但都必须到达真实访问/计费链路；只增加 metadata 不算完成。

## 顺序

1. 先盘点现有 DRAM/HBM/SSD/HBF aggregate preset，给字段标来源和证据状态；
2. 完成任务一的最小 DRAM family profile 和访问控制；
3. 独立审查任务一后，再完成任务二的最小 NAND family profile 和访问控制；
4. 最后才考虑 FTL/GC、刷新策略或更细的 bank/plane 状态，并以实际瓶颈证据决定是否加入。
