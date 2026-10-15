# DRAM / NAND Core 执行语义与成本审计

这是第一轮历史审计。后续修复与最新验证见 [第二、三轮深度审计](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm/core_semantics_deep_audit.md)。

日期：2026-10-13。范围为当前工作区源码，包括此前尚未提交的 HBM Core 路径修改。

本轮核查了地址拆分、物理资源共享、请求准入、读写完成条件、缓冲与回滚、快慢路径等价性，以及上层时间和流量的记账边界。共修复 12 处问题，最终全套回归 241 项测试、22 项子测试通过。没有引入外部硬件数据校准。

这里的 Core 是带地址映射和资源状态的物理时序模型。它们仍有明确的简化边界，不能因为“走了 Core”就称为完整硬件周期仿真。

## 已确认的问题与修复

P2 表示在特定配置、调用顺序或环境下产生错误行为或统计；P3 表示接口错误处理不一致。下表按对执行结果的影响排列。

| 编号 | 级别 | 复现条件与原错误 | 修复及实际影响 |
|---|---|---|---|
| 1 | P2 | 显式排空 NAND 写缓冲后，源数据仍在排空，空间却立即释放，后续写可提前确认 | 保留排空中占用直到对应 program 完成；有限缓冲的后续准入会等待。可能改变这种调用序列的时延 |
| 2 | P2 | NAND 主机资源有多个服务槽时，读取部分已写、部分未写的页面，media fallback 可能在重叠的暂存写确认前返回 | 合并读取等待相交 dirty range 的数据就绪；不相交的读取不增加这项等待 |
| 4 | P2 | 独立物理任务先登记分配、后验证，失败时留下 runtime/分配；1 MiB 内存中的越界请求会残留 1024 B 分配 | 将独立入口的回滚覆盖到分配、验证、执行和账单生成全过程；事件内核继续使用已有外层事务 |
| 5 | P2 | 同一 rollback point 恢复后，Core 的可变字典与保存点共享；再次执行后第二次恢复不能还原行状态 | 恢复时复制可变状态；DRAM 行状态、NAND 暂存/排空/磨损和时间线能一致恢复 |
| 6 | P2 | 4 个独立 NAND channel、每通道 1 GB/s 时，实际约 3.995 GB/s，却报告上限 1 GB/s | 介质路径传输上限改为 min(方向主机带宽, channel 数 × 每 channel 带宽)。只修统计，不改变资源计时 |
| 7 | P2 | 全部由暂存缓冲服务的写确认/读取仍用 NAND 介质带宽上限，可能显示“超上限” | 使用方向主机带宽，并标明 host acknowledgement 与 media drain 的窗口区别；批量上限采用实际请求的上限 |
| 8 | P2 | KV 页拆成 4 个串行写入时，把 host outstanding 峰值相加，实际峰值 1 却报告 4 | 峰值取 max，样本数和字节数继续求和；不改变时延 |
| 9 | P2 | ERASE 触发部分 dirty page 排空时，实际发生 RMW read，但 rmw_read_operations 为 0 | 保留由排空产生的 RMW 读次数；物理读写字节不重复记账 |
| 10 | P2 | DRAM 加速模块不可用时，单条可退回 Python，批量入口却直接导入失败 | 批量路径采用相同的 ImportError/OSError 回退，保持请求次序和状态；本机已安装加速模块，旧运行不受此错误影响 |
| 11 | P2 | 同一物理任务多次访问 NAND 时，写缓冲等待、合并等计数只保留最后结果；实测累计等待约 159.7 μs 却只报告约 79.9 μs，按 owner 的账单还缺失该字段 | 累计等待、字节、合并及暂存操作求和，缓冲峰值取 max，并保留 owner 汇总；Core 完成时间不变 |
| 12 | P3 | DRAM 非法请求对象先触发 AttributeError，绕过统一类型验证 | 先验证请求，单条和批量均一致拒绝，且不改变状态 |

实现位置：[NandCore](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py)、[DramCore](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/dram_core.py)、[编译入口](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_compiled_dram.py)、[物理任务与回滚](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/data_motion.py)、[KV 计数汇总](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/kv_transfer.py)。

## DRAM 的实际执行与计费

1. 请求必须有明确地址和正字节数；零字节、越界和 ERASE 被拒绝。上层零字节价格查询可返回零成本，但不会制造 Core 请求。
2. 每个请求拆成完整 burst。未对齐的两个逻辑字节可能触及两个 burst，因此逻辑流量与物理流量必须分开。小几何的全地址枚举确认映射没有别名，末地址合法且不会越界。
3. 每个 bank 维护 open row。首次访问需要 ACTIVATE；冲突需要 PRECHARGE 后 ACTIVATE；行命中不再次支付开行成本。
4. PRE/ACT 占用 bank；列命令占用每 lane 的 command 资源；数据占用每 lane 的共享读写总线。独立 bank/channel 可重叠，共享命令资源和数据总线会串行。stack/die 增加容量和 bank，不自动重复放大外部总带宽。
5. 读写数据从列命令时刻加对应 pipeline latency 后就绪；方向切换按 read_to_write/write_to_read 约束。pipeline latency 是依赖等待，不再次记为 command 独占忙时。
6. 写请求完成包含 write recovery。恢复时间限制后续换行，当前模型允许同一行的列访问流水。因此同一时刻提交的后续读取可在前一笔写的 recovery 结束前完成，但不能早于数据总线依赖。
7. max_outstanding_requests 限制主机请求准入；保留提交次序，不进行优化重排。Core 管资源约束，上层任务依赖负责数据因果关系。

手算回归中，四笔“读、同行写、同行读、冲突行读”的完成时间是 16、30、28、53 ns；数据忙时合计 4 ns，命令忙时 8 ns，bank 的 PRE/ACT 忙时 27 ns。等待、pipeline 和 recovery 没有作为另一笔数据传输重复收费。

**边界：**没有完整 JEDEC 刷新、tFAW/tRRD/tRAS、bank-group 特定时序，也没有 FR-FCFS 调度。PRE/ACT 不另外占命令总线。这些是当前模型的简化，不能将缺少的约束解释为已经模拟且开销为零。

## NAND 的实际执行与计费

1. 地址映射到 channel/target/die/LUN/plane/block/page。planes_independent 决定 plane 是否独立争用；parallel_units 只限制服务槽，不会凭空增加物理阵列。
2. 普通读的路径是 page read → 内部 channel 传输 → host 传输。前端延迟按请求施加；多页请求在资源允许时重叠。
3. 全页写经过 host 传输、内部传输和 program。部分页写采用简化 RMW：读旧页及其内部传输，再写完整页；主机只传对齐后的有效请求。跨页的两字节写可能触发两次整页读和两次整页 program。
4. ERASE 按完整物理 block 坐标去重，不能简单把逻辑连续区间当作一个 block；它产生阵列忙时，不产生普通读写传输字节。
5. host 是共享接口，内部 channel 是独立资源，array/page buffer 有单独就绪状态。持续带宽由这些实际资源和页操作时延共同决定，接口上限不是实际吞吐。
6. 默认写是 media program 完成才返回。开启 buffered_write_mode 后，host ack 与 media drain 分离；写入可按页合并，覆盖范围内的读可由暂存区服务，空间不足会排空并阻塞。显式排空后仍在使用的空间现在也受到准入保护。
7. 缓冲容量按主机对齐后的提交载荷计数，重复覆盖仍占提交载荷预算；merge_bytes 表示向已有 dirty page 追加的载荷，不等于去重后的唯一脏字节。它是提交缓冲模型，不是完整去重写缓存。
8. wear 的 P/E 次数来自初始值和显式 ERASE，不会把 program 数自动换算成擦写次数。

**边界：**没有 FTL、GC 状态机、坏块、ECC/retry、数据内容与 NAND 单向编程合法性检查；多次 program 同一物理页仍可计时。没有完整 NAND 命令协议。第 5 组的额外 GC 写放大仍是公式叠加，不能当成 Core 执行了 GC。

## 上层成本的边界

- **时间：**规划预估使用独立 preview，不提交到在线 Core。正式任务在持久 Core 上预约资源，事件内核只登记其实际忙时，不再预约同一笔内存资源。任务完成取计算和物理访存完成的较大值，并保留任务间显式依赖。这是计算/访存可重叠的简化，不是逐指令依赖仿真。
- **迁移：**KV staging 的源端读、独立链路、目标端写保持顺序依赖；relay 额外支付中继写与读。引用已有内存服务的 service_ref 不作为另一条物理链路再收费。direct 与 staging 的流量差异是架构变量。
- **流量：**logical_bytes 是请求载荷；DRAM physical bytes 按 burst，NAND physical read/write bytes 按页，internal transfer 可再按传输粒度取整，host bytes 独立记录。NAND transfer_bytes 表示内部传输量，不能再加 host bytes 当作“程序请求字节”。
- **队列：**queue_wait_ns 是请求准入等待，包含写缓冲准入等待；write_buffer_wait_ns 是其中一部分，不能再次相加。host outstanding 峰值也不是 NAND page/program 队列深度。可选 NAND observations 才记录页操作的 eligible-to-array-start 等待。
- **带宽：**交易实际带宽为逻辑字节 / 请求到完成的时间；批量为总逻辑字节 / 最早到达至最后完成窗口。不能用各请求延迟之和作为并行批量耗时，也不能用 host ack 带宽代表介质持久写带宽。

相关依据：[事件内核正式记账](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/event_kernel.py:2605)、[批量统计](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_transfer.py:169)。

## 验证与既有实验的影响

新增回归覆盖 DRAM 四路径差分、手算时序、NAND 多页/RMW/擦除/缓冲、快慢路径的后续状态、重复回滚、KV 峰值、多访问缓冲账单。另以有界混合配置检查了 DRAM 80 组和 NAND 100 组快慢路径；比较时间、流量及资源状态，不只比较最终吞吐。

最终验证：`python -m pytest tests -q`，**241 passed、22 subtests passed，82.60 秒**。本轮源文件和测试的差异格式检查通过。快慢路径对照使用浮点容差；不承诺任意极端时间戳下逐位一致。

可重复执行的新增回归：[DRAM 语义](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_dram_core_semantics.py)、[NAND 语义](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_nand_core_semantics.py)、[任务计费与回滚](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_core_cost_accounting.py)、[KV 迁移汇总](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_kv_transfer_accounting.py)。

此前 19 组大型实验本轮没有全部重跑，原 JSON/图表仍是之前那次执行的产物。本次正常 Core 时序检查没有发现默认 DRAM 路径重复计费或快慢路径漏算。脚本中的显式 NAND 排空位于写入结束。因此不能把本次所有修复笼统归因为旧 TPS 图错误，也不能把旧结果称为本轮完整重跑验证。

以后使用多 channel 瓶颈统计、暂存路径带宽、分块队列峰值时应采用修复后的统计；使用显式排空后继续写、多个 host 槽的部分暂存合并读时，应重新执行对应配置获取性能结果。
