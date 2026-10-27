# 1_HBM / HBF 建模流程

这份演示解释当前工作区中的 HBM 和 HBF 建模实现。一条逻辑访存首先解析成带地址和字节数的物理请求，再分别进入 DramCore 或 NandCore。两条介质路径共同使用资源时间线，将介质状态、资源竞争与任务依赖转化成完成事件。理解结果时，需要同时追踪地址、状态与时间，而不能只看标称带宽。内容依据当前工作区源码；本次制作没有新增性能仿真，也不代表厂商芯片校准。核心入口为 src/heterollm_sim/data_motion.py:314、1330，以及 dram_core.py 与 nand_core.py。

---

# 2_仿真器的完整链路

完整链路分六步。模型、批量、上下文和并行配置决定权重、KV 与激活访问；planner 再组织任务、放置和依赖。分配器将缓冲区与偏移解析成物理 owner 和地址范围。任务真正到达时，运行上下文调用保持状态的物理 Core，而不是直接接受规划时的预览延迟。Core 返回完成时间和资源需求后，事件内核推进后继任务，reporting 汇总流量、资源忙碌与 Token 指标。对应源码是 src/heterollm_sim/planner.py:8975、16306、19067，data_motion.py:1151、1330–1496，event_kernel.py:685，以及 reporting.py:632、780。

---

# 3_两种介质的建模对象

HBM 由 DramConfig 配置、DramCore 执行，JEDEC HBM 路径按三十二字节 burst 服务，关键状态是打开的行、命令总线、数据总线以及刷新。HBF 使用 NandConfig 和 NandCore，当前论文原生配置按四 KiB 页服务，关键状态是 plane 阵列、读缓冲和各段传输。两条路径共同接收操作、地址、字节数和到达时间，最终返回完成时间。三十二字节属于 JEDEC HBM 模式，四 KiB 属于这里举例的论文原生配置，不能扩展成所有配置的固定要求。见 src/heterollm_sim/memory_types.py:243、404、659，memory_mapping.py:172、212。

---

# 4_配置先定义物理约束

三组输入分别决定需求、位置与服务约束。工作负载中，模型结构、精度、批量、上下文与并行配置决定访问量。放置配置给每个 rank 指定权重、KV 和激活端点，并约束分配释放以及缓冲区代次。硬件配置提供容量、通道与 bank 或 plane 几何、方向带宽、时延、队列和协议策略。容量决定是否可以容纳，地址与时序决定请求如何执行，接口峰值限制传输上限。组件汇总带宽不能覆盖物理 Core 的实际服务过程。见 src/heterollm_sim/physical_contract.py:154–200、260–292，以及 memory_types.py。

---

# 5_张量访问到物理地址

权重读取使用相应张量和 rank 分片的真实字节；KV 区分历史读取和新 token 追加；激活依据元素数量和存储位宽换算。每条 memory_accesses 描述给出 operation、physical_owner、byte_count，以及显式 address 或 buffer_id 加 offset_bytes。分配器在解析时检查容量、生命周期与分配代次。随后 HBM 拆 burst，HBF 拆 page，并映射到各自资源。逻辑有效字节和物理流量分别统计，因为部分 burst、传输粒度以及读改写可能增加实际流量。见 src/heterollm_sim/planner.py:16306、19067，data_motion.py:1151、1207–1235，以及 memory_mapping.py:172、212。

---

# 6_HBM 地址映射与并行度

图中采用 HBM4 参考组织：每堆三十二个物理 channel，每个 channel 两个 pseudo-channel，也就是 PC。bank 地址继续包含 SID 与 bank group，再确定 row。标准组织中的 SID 已扩展 bank 地址，不能再把容量重复乘 die 数。默认 bank_interleaved 先分散 lane，然后推进 bank、column 和 row。当前 bank 步进在同一 SID 内先轮换 bank group，每组八个 bank 的例子是零、八、一、九。这样会改变同组与异组的连续访问关系，也会影响时序和并行机会。见 src/heterollm_sim/hbm_geometry.py:102、137，memory_mapping.py:18、47，以及 _hbm_control.py:60、105。

---

# 7_HBM 的两种控制模式

analytical 和 jedec_hbm 是两种不同的控制精度。analytical 使用简化 DRAM 服务规则，不模拟刷新与 tRAS。jedec_hbm 显式执行 ACT、PRE、RD、WR 和全 bank REF，将打开行、共享命令总线、PC 数据总线、列命令间隔、ACT 速率和方向切换都纳入时间线。后面几页解释的是 jedec_hbm。它覆盖正常工作命令，仍不包含训练、自动预充电和完整刷新推迟策略等电路与协议细节。执行加速只替换执行方式，不把 JEDEC 模型偷偷改成 analytical。见 src/heterollm_sim/dram_core.py:90、423、483，以及 _hbm_control.py:1。

---

# 8_HBM 行状态与命令链

行命中时目标行已经打开，读请求执行 RD，经过读流水延迟 RL，再占用 DQ 传输数据。bank 关闭时，要先发 ACT，等待 tRCD 后才能 RD。行冲突则先 PRE 关闭旧行，等待 tRP，再 ACT 打开新行，随后执行相同读取链。写操作对应 WR 与 WL。这说明相同字节数并不保证相同延迟。图只表达顺序，不按真实时长比例，且 ACT 的精细阵列锚点由控制器处理。一个请求完成于所有 burst 数据结束时间的最大值。见 src/heterollm_sim/_hbm_control.py:145、157、183、490、547、584。

---

# 9_HBM 共享资源时间线

同一个物理 channel 的两个 PC 共享 Row CA 与 Column CA，但各自拥有 DQ，所以两条数据传输可以部分重叠。命令并非到达就能发射，而是综合请求就绪、bank 可用、总线空闲与各项时序约束，再做时钟对齐。约束包括 tCCD、tRRD、四激活窗口 tFAW 和方向切换。当前实现有限前瞻后续 burst，提前 PRE 后续冲突行；all_bank 刷新也占用资源，tRASmax 即使关闭刷新也生效。图没有数值刻度，只展示共享和重叠。submit_many 的 FCFS、行命中优先或读写批处理选择的是整条请求，并不是逐命令 FR-FCFS。见 src/heterollm_sim/_hbm_control.py:25、121、206、279、318、364、458、569，dram_core.py:483。

---

# 10_HBF 几何与资源映射

当前论文原生 HBF 配置每堆十六个 die，每个 die 二十五个 plane，每页四 KiB。代码将每个实际 die 映成一个内部 channel，每个 plane 提供独立阵列资源，而不是根据目标带宽反推出并行度。连续页先分配到 base units，也就是这里的 die 通道，再轮转 plane，之后推进 block 内 page。因此七堆配置有一百一十二个内部通道和两千八百个独立阵列单元。内部通道与 host 出口还有自己的带宽限制；并行资源多也不会消除页读取延迟。见 scripts/exploring_hbf_paper_config.py:153、289、317，src/heterollm_sim/memory_mapping.py:124、132，memory_types.py:813。

---

# 11_HBF 读请求的三段服务

请求先通过队列 credit 准入，再按页拆分、映射并查找对应 plane 的两个 slot。缓存未命中时，先在 plane 阵列执行 PAGE_READ，再通过内部通道执行 INTERNAL_TRANSFER，将数据填到 slot，最后 HOST_TRANSFER 返回数据。每段分别预约资源，同资源排队，独立资源可以在依赖允许时重叠。缓存命中会跳过阵列读取与内部搬运，但仍要等待 slot 填充就绪和 host 资源。请求完成时间取各页完成时间最大值，延迟从原始到达算起，因此包含准入等待。见 src/heterollm_sim/nand_core.py:537、549、553、1900、1919、1923、1937。

---

# 12_每个 plane 的双缓冲

每个实际 plane 维护两个读缓存 slot，每槽保存 page 标识、填充完成时间和读者完成时间。命中时先等 ready；未命中时选空槽或最近最少使用的槽，覆盖前的内部搬运必须等待旧读者结束。这既允许流水重叠，也保留真实的覆盖等待。每堆容量为十六乘二十五乘二乘四千零九十六，总共三百二十七万六千八百字节，即三点一二五 MiB。它属于媒体页缓冲，与 Plus 配置每堆四十 MB 的中间数据 SRAM 分开；本基线也没有自主预测预取控制器，读请求驱动填充。见 src/heterollm_sim/nand_core.py:647、666、1900、1925，scripts/exploring_hbf_paper_config.py:276、325、334。

---

# 13_HBF 写入与协议边界

完整页写入依次经历 host 传输、内部传输、阵列 program，最后一个阶段结束才算完成。部分页写按配置拒绝，或者执行付费 read-modify-write，也就是先读旧页并内部搬运，再执行新页编程。显式 erase 按物理 block 去重并占用阵列资源。当前论文原生基线使用 generic_nand，是直接地址页服务，没有默认 FTL、垃圾回收和自动擦除。NandCore 也支持 OCP 模式，但 OCP 的 page 零自动 erase、顺序 program 等约束不能混进 generic 基线。见 src/heterollm_sim/nand_core.py:1861、1958、1964、1982，memory_mapping.py:216，memory_types.py:739。

---

# 14_跨端点数据搬运与依赖

跨端点 COPY 展开为源 Core 读取、链路传输、目标 Core 写入，目标写完成后搬运才结束。三个阶段分别支付介质与链路成本，物理 owner 的别名共享资源归属。不同请求在独立资源上可以重叠，同资源需要等待。同一地址的先写后读、先读后写和连续写必须保留 RAW、WAR、WAW 依赖，只读请求之间没有这类完成依赖。任务实际 dispatch 时提交有状态 Core，完成事件再释放后继。部分原生 kernel 使用其他资源跨度与 Core 内存跨度的最大值，加 dependency_ns 形成包络，仍属于任务级模型，不能解释为 GPU 指令级回放。见 src/heterollm_sim/data_motion.py:1261–1282、1330–1496、1730–1775、2573–2647，memory_transfer.py:263–335。

---

# 15_输出指标与瓶颈判断

Core 同时输出完成时间、逻辑与物理流量、等待和各资源占用。HBM 关注行命中、冲突和 ACT、PRE、RD、WR、REF 次数，以及命令总线和数据总线忙碌。HBF 关注缓存命中、实际读页与 program 次数，以及 array、internal 和 host 三种资源。系统层再统计整体 makespan、吞吐以及边界一致的 TTFT 和 TPOT。请求延迟等于完成减到达，有效带宽是逻辑字节除以相应时间窗。service_ns 累加各任务各 owner 的服务跨度，这些跨度可能重叠，不能当整体耗时或总线忙碌；资源忙碌应读 resource_totals。见 src/heterollm_sim/memory_types.py:899、911，memory_service.py:26–84，reporting.py:632、2113、2174–2246。

---

# 16_快速仿真与物理语义

仿真器通过减少执行开销来加速。HBM 可使用 Numba 数值路径、数组常驻状态和批量提交，仍执行所选 JEDEC 时序与资源预约。NAND 编译逐页循环，减少对象与明细成本，同时保留每页缓存访问、填充完成和读者覆盖依赖。两者都有适用条件与参考实现，不适用时使用参考执行方式。执行加速不意味着切换到 analytical，也不意味着参数已经与真实硬件测量校准。本页强调的是软件执行和物理模型的区分。见 src/heterollm_sim/dram_core.py:423、527，_compiled_hbm.py:92，_nand_acceleration.py:1、59，以及 tests/test_nand_generic_buffered_acceleration.py:63、109、203。

---

# 17_一组配置输入示例

这一页只列项目输入，未展示本次测量结果。hardware_catalog 的 HBM4 配置每 GPU 八堆 HBM，读写接口预算都是十二点八 TB 每秒。HBF 配置每 GPU 七堆 HBF 加一堆 HBM，HBF 聚合读带宽十一点二 TB 每秒，写带宽一百一十二 GB 每秒；每堆十六 die、二十五 plane，读页一微秒、program 一百微秒。输入有来源层次。HBM4 的 tRRD 同异组与 tFAW 借用 HBM3E 参考，分别为四、六、二十纳秒。NAND 的二百五十六页每 block、两毫秒 erase、每 die 通道一百 GB 每秒，属于显式本地研究选择。它们不是已校准的厂商实测参数，聚合峰值也不是持续吞吐。见 scripts/exploring_hbf_paper_config.py:144–156、207–258、272–327。

---

# 18_理解模型的四个抓手

理解仿真结果可以始终沿四个抓手检查：请求描述、物理地址、介质状态和资源时间线。HBM 重点看行状态、命令与共享总线；HBF 重点看页、双缓冲与阵列到内部再到 host 的三段传输。出现性能变化时，先确认访问量和放置，再追到地址分布、命中与冲突，最后确认等待落在哪种资源。配置和映射入口为 src/heterollm_sim/memory_types.py 与 memory_mapping.py，HBM 控制器为 _hbm_control.py，HBF 为 nand_core.py，系统连接与统计为 data_motion.py 和 reporting.py。每条解释都应有这条完整因果链支撑。
