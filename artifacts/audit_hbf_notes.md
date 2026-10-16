# HBF / NandCore 当前版本审计笔记

审计对象：当前工作区 `c864890`，对照附件报告 `5a13823`。只读源码并运行小规模定向复现，未修改实现、未跑全套实验。以下优先级表示对研究结论的影响，不把已明示的模型边界自动当作代码缺陷。

## 结论

主实验仍使用 64 个等效阵列执行单元、16 KiB 页、4 µs 读、75 µs 编程。读带宽 1600 GB/s 只是主机接口上限；阵列读上限为 262.144 GB/s，约为接口值的 16.384%。写阵列上限也只有 13.981 GB/s，低于配置的 16 GB/s。64 MiB 顺序整页复现得到读 261.474 GB/s、写 13.793 GB/s，明细与汇总路径在浮点舍入范围内一致。

旧报告的“主实验缺少擦除管理、主机读写共享资源、没有持久读页缓存”仍成立。但“所有 HBF 行为完全没变”和“没有任何 buffer 命中”需要改写：当前正数 `write_buffer_bytes` 一律启用写暂存、页合并、主机早确认和暂存读命中，旧的正缓冲容量但仍直写的兼容模式已经删除。主实验缓冲为 0，因此其时序不受这次模式统一影响。

## 配置与来源

| 参数 | 主实验值 | 证据 |
| --- | --- | --- |
| 阵列单元 / 内部通道 | 64 / 64 | `scripts/hbf_hbm_experiments.py:61-71`；`src/heterollm_sim/physical_presets.py:344-350` |
| 页 | 16384 B | `scripts/hbf_hbm_experiments.py:40,71` |
| 主机传输粒度 | 4096 B | `src/heterollm_sim/physical_presets.py:351-352` |
| 页读 / 编程 / 擦除 | 4000 / 75000 / 3000000 ns | 主实验覆盖读和编程；擦除来自 `physical_presets.py:65-66` |
| 主机读 / 写 | 1600 / 16 GB/s | `scripts/hbf_hbm_experiments.py:67-70` |
| 单个内部通道带宽 | 1600000 GB/s | `physical_presets.py:353`，主机聚合带宽乘 1000 |
| 内部传输 | 每次整页，16384 B | `memory_types.py:690-694`；`memory_mapping.py:188-193` |
| 主机写缓冲 | 0 | `scripts/calibrated_hbf_profiles.py:447-462` |
| 并发上限 | 32 个主机请求 | `physical_presets.py:69`；`nand_core.py:493-516` |
| 磨损模型 | 关闭 | `physical_presets.py:67` |
| 部分页写 | RMW | `physical_presets.py:68` |
| 几何依据 | 等效解析映射，非公开器件版图 | `physical_presets.py:112-117` |

注意“64 单元 / 64 通道”是主实验的等效内部组织，不等于 OCP 的 16 个主机通道。默认的一个主机共享资源和 64 个内部通道分别在 `nand_core.py:116-121` 定义。

## 一次访问流程

1. 按整个上层请求执行准入控制，最多 32 个；不是逐通道、逐页请求队列（`nand_core.py:493-516`）。
2. 地址按 NAND 页轮转内部单元，随后分解 plane、page、block；不经过 FTL（`memory_mapping.py:69-102`）。
3. 拆成整页操作，主机字节按 4 KiB 向外取整，内部传输按整页计（`memory_mapping.py:175-193`）。
4. 读：等阵列与页缓冲可用，执行 PAGE_READ，整页内部传输，再经共享主机资源返回请求字节；下一个同单元读需等上一次内部传输完成（`nand_core.py:613-614,687-719`）。
5. 整页写：主机传输可先进行，内部传输等待缓冲空闲，然后 PAGE_PROGRAM；写完成是编程完成（`nand_core.py:720-745,777-786`）。
6. 部分页写：先读旧页并内部传输，再写完整物理页；没有检查该页是否为空、是否已写，或写入顺序是否合法（`nand_core.py:650-668,720-745`）。
7. 只有显式 ERASE 请求产生 BLOCK_ERASE；整页写及 RMW 都不自动擦除（`nand_core.py:615-623`）。
8. 若写缓冲容量为正：写入到主机暂存就返回，重叠页合并；读已暂存范围可直接命中，部分覆盖读则读取介质再叠加暂存数据；显式刷新或容量压力才安排介质写入（`nand_core.py:280-320,371-488,628-648,669-715`）。

资源预订使用每个资源的最新可用时间，没有空洞回填或 FR-FCFS 调度；读写使用同一主机资源，direction 参数并不创建独立读写总线，NAND 路径也没有传入非零换向惩罚（`memory_transfer.py:99-166`）。这属于当前确定性队列抽象。

## 待处理项，按影响排序

### P1：1600 GB/s 不代表当前主实验能提供 1600 GB/s HBF 服务

类型：研究配置 / 结果解释风险，不是算术实现错误。

64 × 16384 B ÷ 4000 ns = 262.144 GB/s。要仅凭这些页与延迟参数支撑 1600 GB/s，至少需要 391 个等效读单元，尚未计接口和排队开销。当前显式指定 64，覆盖了预设根据带宽与延迟反推并行度的规则（`physical_presets.py:344-350`）。因此“HBM 和 HBF 配置同为 1600 GB/s”不能直接解释成两者有效读服务相等。

写也受阵列限制：64 × 16384 ÷ 75000 = 13.981013 GB/s。部分页写还支付 RMW 和整页编程，主机有效载荷带宽更低。建议报告同时列接口带宽和实际介质约束，不以一个带宽参数概括。

### P1：写入合法性和自动擦除不建模，持续写结果不能代表完整 HBF 写行为

类型：明确模型边界；只有声称遵循 OCP 写协议时才构成符合性缺陷。

`NandCore` 类说明直接声明没有 FTL、GC、cell-content validity（`nand_core.py:33-37`）。重复写同一物理页、跳页写、新块首次写都可直接编程，只有显式 ERASE 消耗 3 ms。小复现中连续两次写相同 4 KiB，均发生一次 RMW 和一次 PAGE_PROGRAM，而两次 `erase_operations` 都为 0。

影响不是单一方向：省略自动擦除、有效性约束会偏乐观；每个 4 KiB 小写都立即进行 16 KiB RMW/编程又可能偏悲观。不可从当前短写测试外推真实设备持续写吞吐、写放大或寿命。OCP 原文对 page 0 自动擦除、写页顺序、4 KiB 累积的具体差异由主报告统一引用。

### P2：默认主机读写共用单一串行资源

类型：简化模型限制。

`_host_id()` 没有 channel 或 direction 维度（`nand_core.py:120-121`）；读和写都预订该资源（`nand_core.py:708-724`），默认 ResourceTimeline 创建单 lane（`memory_transfer.py:109-111`）。方向独立的读 1600 / 写 16 GB/s 只决定每次传输时长，不代表全双工。高读写混合负载会相互阻塞。时间线底层支持外部多 lane，但主实验没有配置成 OCP 的每通道全双工结构。

### P2：没有介质读页缓存、ECC、重试和读刷新；已有脏写暂存命中

类型：简化模型限制；旧报告措辞需修正。

普通 READ 总会执行 PAGE_READ，即使上一次读取相同页（`nand_core.py:687-719`）。`_buffer_ready` 是时序占用状态，不记录读缓存页号。唯一读命中路径来自 `_staged_pages` 中尚未落盘的脏写数据（`nand_core.py:669-686`）；它不是 OCP 每 bank 两个读 cache buffer。全套 NandConfig 也没有 ECC、retry、retention/read-refresh 配置字段（`memory_types.py:566-603`）。

论文标定资料的 `buffering.controller_page_cache` 写有 two page-cache buffers per bank，但配置执行只读取粒度、延迟、并行度等字段，`_hbf_core` 明确写缓冲为 0（`scripts/calibrated_hbf_profiles.py:314-316,447-462,603-626`）。该描述不能当作已执行的缓存能力。

### P2：正写缓冲容量的完成语义与旧版不同，且是研究用主机早确认

类型：有意行为改变 / 协议抽象，主实验不启用。

当前 `NandConfig.buffered_write_mode` 完全由 `write_buffer_bytes > 0` 决定，旧 metadata 开关已拒绝（`memory_types.py:660-661,703-706`）。`nand_core.py:628-648` 仅传到主机暂存便返回，报告明确区分 host ack 与 media drain（`nand_core.py:777-786`）。若将此返回值直接解释为 OCP 正常写完成，会遗漏“数据已写到 Core die”的要求。新模式相对旧版本已删除的正容量直写配置不能直接比较完成时延。

### P2：wear 只是显式擦除计数驱动的编程延迟敏感性

类型：明确模型边界。

每块 P/E = 初始计数 + 显式 ERASE 次数，只有 PAGE_PROGRAM 延迟按比例放大并封顶；不会因写入自动增加 P/E，没有介质损坏或寿命终止（`nand_core.py:158-180`）。主实验关闭磨损。第 14 组的 trace 显式创建擦除，serving 对照只是预老化参数；脚本自己的限制说明也指出没有自动 GC、ECC、retention（`scripts/hbf_hbm_wear_study.py:64-65,79-80,113,150-151`）。

## HBF-2 论文参数组

仍是 128 个等效单元、4 KiB 页、20 µs 读、200 µs 写，主机读写上限都 1200 GB/s（`scripts/calibrated_hbf_profiles.py:276-316`）。其读阵列上限 26.2144 GB/s，约主机值 2.18%；写阵列上限 2.62144 GB/s。并行单元在源码中明确标记为作者选定的敏感性锚点、不是论文披露的物理通道数。因此可以说“借用了论文部分参数”，不能说“复现了论文 HBF-2 的有效 1200 GB/s 服务”。

## 最小复现结果

配置由当前 `resolve_physical_config("published_hbf", overrides=...)` 产生；覆盖主实验的 64 单元 / 16 KiB / 4 µs / 75 µs / 1600 / 16、容量 512 GiB，其他沿用主实验预设。运行了一个 64 MiB 连续整页读、一个同样大小写，分别使用全明细和汇总执行。

| 操作 | 明细完成时间 ns | 汇总完成时间 ns | 有效 GB/s | 页操作 |
| --- | ---: | ---: | ---: | --- |
| 64 MiB 读 | 256656.01535999958 | 256656.01536 | 261.4739573 | 4096 reads |
| 64 MiB 写 | 4865536.655360003 | 4865536.65536 | 13.79269519 | 4096 programs，0 erases |

另运行同页两次 4 KiB 写加一次 4 KiB 读：

- 缓冲为 0：两次写各含 RMW_PAGE_READ、RMW_INTERNAL_TRANSFER、HOST_TRANSFER、INTERNAL_TRANSFER、PAGE_PROGRAM；读含普通 PAGE_READ；没有擦除。
- 缓冲为 64 KiB：两次写只含 HOST_TRANSFER；读为 STAGING_READ，`staging_read_hits=1`；显式 flush 产生 1 次 RMW、1 次 PAGE_PROGRAM，合并 4096 B，介质完成时间 79514.58048 ns。

上述检查只证明这些具体输入的行为，以及这两个顺序请求的明细/汇总一致性，不代表全面验证或实际硬件标定。

## 本轮代码缺陷判断

在检查的 HBF 路径中，没有发现可以仅凭当前实现契约确定的新功能性 bug。主要结论是配置瓶颈、协议覆盖范围和完成语义，应在报告中与代码缺陷分开。当前路径的已知限制有明确实现或注释依据，不建议把它们包装成已经修复、也不应直接把整个模型称为 OCP HBF 的完整实现。

## 追加：OCP v0.7.0 原文逐项对照（2026-10-15）

此节直接复核 [OCP 官方 v0.7.0 PDF](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)，下列 p. 均为印刷页码（PDF 零基索引再加 1）。规范列仅列核查关键词，右列分析来自项目代码。比较对象为普通 HBF 闪存路径；高层 AccessRequest 可以是聚合请求，不能因它大于协议单包上限，就直接判定协议报文错误。

| 协议点 | 极简规范要求 | 精确原文位置 | 当前实现 / 主实验 | 判断 |
| --- | --- | --- | --- | --- |
| 主机通道 | 最多 16；独立、全双工 | §4，p.15 第 1–2 段；§4.1 Host Interface Features | `nand_core.py:120-121` 每设备只有一个默认 host ID；`708-724` 读写共用；`memory_transfer.py:109-111` 默认单 lane。主实验 64 是内部等效单元，不能解释为 64 条 OCP host channel。 | 主机接口抽象，未复现该拓扑与双工性。 |
| NAND 页 | 4 KiB | §4.1，p.15，NAND Interface Features 最后一项；§4.2，p.16 表 3 的 Page size | `physical_presets.py:61`、`hbf_hbm_experiments.py:40,71`：主实验 16 KiB；HBF-2 组为 4 KiB。 | 主实验是不同的等效页配置；不是该版规范页几何。 |
| 主机读传输 | 64 B 对齐；64 B–4 KiB；不跨 4 KiB | §4.1，p.15，读 burst 项；§5.3.1，p.56，第 1、4 项 | `memory_mapping.py:136-146` 只检查容量和可选部分页写拒绝；`175-193` 按 16 KiB 物理页拆分，主机字节默认按 4 KiB 取整。没有 64 B AXI 子事务和 4 KiB 协议分包；主实验启用上述聚合抽象。 | 不建模报文合法性；小读流量放大与 OCP 64 B 主机访问不同。 |
| 小写累积 | 64 B 片段收齐 4 KiB；超时错误 | §5.4.1，p.58，第 3、6 项；§5.6，p.61 表 13 的 accumulation timeout | 主实验 buffer=0：`nand_core.py:650-668` 每个部分物理页写先 RMW，再编程 16 KiB；正 buffer 则 `_stage_page` 按配置物理页合并（`241-265`），没有 4 KiB 累积计时器/超时状态。 | 无 OCP 写累积协议；可选暂存并不等价。 |
| 普通写完成 | Core die 写完才应答 | §5.4.1，p.58，第 1–2 项 | 主实验 buffer=0，`nand_core.py:735-745` 返回 PAGE_PROGRAM 结束；正 buffer 时 `628-648` 主机传输后早确认，落盘另计（`777-786`）。 | 主实验在此完成边界上相符；可选早确认研究语义偏离，不能混称。 |
| 顺序编程 | 连续本地页序，不跳页 | §5.4.1，p.58，第 4、9 项；§5.6，p.61 表 13 的 Write order violation | `memory_mapping.py:136-146` 无前序页状态；`nand_core.py:720-745` 原地写和任意合法容量内地址均可编程；主实验启用。 | 未覆盖写页序与相应错误。 |
| 自动擦除 | page 0 写触发块擦除 | §5.4.1，p.58，第 5 项 | 仅 `nand_core.py:615-623` 的显式 ERASE 执行擦除；WRITE 无 page-0 判断；主实验无自动擦除。 | 明确协议覆盖缺口，不能外推持续写成本。 |
| 读页缓存 | 每 bank 两个页缓冲；命中直返 | §5.3.1，p.57，第 5a、7、8 项；§5.9.2.1，p.70，NCBB[39:36] | `nand_core.py:687-719` 普通读总执行 PAGE_READ；`_buffer_ready` 只记时间，未记缓存页号。`calibrated_hbf_profiles.py:314-316` 的两页缓存只在论文描述字段；`447-462,603-626` 不构造读缓存。 | 执行未覆盖；不得把文案视为执行能力。 |
| 暂存已到部分的读 | 可转发 | §5.4.3，p.59 | 正 buffer 时 `nand_core.py:669-686` 对已覆盖范围作 STAGING_READ，主实验 buffer=0 不启用。 | 此项行为方向一致；不等于持久读页缓存。 |
| 暂存尚缺部分的读 | 返回 0xA | §5.4.3，p.59；§5.5，p.60 表 12 的 Read from a pending Write | 正 buffer 时未完全覆盖就进入 `687-715` 介质读和覆盖合并，而非协议错误；没有 AXI 命令状态。主实验 buffer=0 不产生该暂存状态。 | 可选模式的另一协议差异，不能记成主实验当前错误。 |
| ECC / 重试 | ECC 状态；主机或 Base die 重试 | §5.3.1，p.57，第 9 项；§5.5，p.59–60 表 12 的 0x4/0x5/0x6；§11.5.1.6，p.120 | `memory_types.py:566-603` NandConfig 无相关参数；`nand_core.py:687-719` 确定性正常读，不生成纠错、重试或错误阶段；主实验不覆盖。 | 未覆盖可靠性与尾延迟机制；不是要求虚构固定 ECC 时延。 |
| 数据刷新 | 读计数；主机安排刷新 | §4.1，p.15，read refresh counter；§11.5，p.118，Data refresh conditions 两项 | NandCore 无保持时间/读扰动阈值/刷新状态或操作；`memory_types.py:44-47` 只有 READ/WRITE/ERASE，实验没有刷新策略。 | 设备与主机配合机制均未覆盖；不应写成必须由设备自动刷新。 |
| outstanding | 每通道、产品定义 | §5.4.1，p.58，第 7–8 项；§5.9.2.1，p.70，MOCS[35:32]；§5.9.2.3，p.71–72，MOCS[27:24] | `nand_core.py:493-516` 为整设备上层请求准入；主实验 32；没有按 AXI ID、每通道 4 KiB logical-unit 分别限额。 | 队列粒度不同；不能将 32 与规范示例 64/128 直接判为数值不合规。 |
| scratchpad | 可选、产品定义 | §5.3.3，p.57–58；§5.4.2，p.58；§11.3，p.116；§5.9.2.1，p.71，SMEMS[4] | HBF NandCore 无独立 SRAM 地址区、命令、优先级和寄存器；主实验不启用。通用 SRAM profile 或写暂存并不实现 HBF scratchpad。 | 可选能力未实现，不能仅据此判不符合规范。 |
| GC | HBF 不提供设备内 GC | §11.4，p.117，第 2 段 | `nand_core.py:33-37` 无 GC。 | **不是 OCP 缺失项**；应删除“需要补设备 GC 才符合 HBF”的暗示。 |
| wear leveling | Base die 或主机重映射 | §11.4、§11.4.1，p.117；§5.9.2.1，p.71，WLS[7:6]；§5.9.2.3，p.72，WLS[7:6] | `nand_core.py:158-180` 仅显式擦除计数放大编程延迟，没有 zone 地址重映射或磨损均衡；主实验 wear 关闭。 | 磨损延迟敏感性不是磨损均衡；主机责任与设备责任需分开。 |

补充注意：同一 pending 4 KiB 单元的重复片段写，规范另有 overlap 错误（§5.6，p.61 表 13，0x2）；当前 `_stage_page` 会合并重叠范围并累加主机字节（`nand_core.py:245-264`）。因此正缓冲实验还不能当作完整的 OCP 写聚合器。

**对前文 GC 措辞的限定：**“不含 GC”可以描述 NandCore 的功能范围，但它不应作为 OCP HBF 的符合性缺口。持续写审计应检查自动擦除、页序、有效数据管理、主机重写/重映射和维护开销；不要把 SSD 式设备内 GC 自动套到本版 HBF。
