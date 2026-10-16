# 在本仿真器中采用 Ramulator2 来源时序参数的对照

执行版本：10；状态：incomplete。
计划/已执行：独立访存 252/252；完整服务 48/40。
固定来源：[Ramulator2 HBM3 源码](https://github.com/CMU-SAFARI/ramulator2/blob/54e90367191062067e9be74840439b0daca40343/python/ramulator/dram/hbm3.py)，提交 `54e90367191062067e9be74840439b0daca40343`。

本实验把 Ramulator2 HBM3 的可映射时序输入放入本项目 HBM 命令模型；没有运行 Ramulator2 本体。
两者的命令计时锚点和控制器、刷新调度实现不同，因此相同输入周期数不代表逐命令执行等价。
两组都不是厂商标定。来源包含上游估算、上游采用的标准值和本模型补充规则，逐字段来源见 results.json。
固定条件：32 Gb × 16 层、512 GiB、CK=0.625 ns、读写方向预算各 1600 GB/s、bank_interleaved、FCFS、device 队列深度 64。
完整服务先运行 Prefill，再独立统计 Decode；A 为全 HBM，B 为 KV 使用 HBF。
实际执行路径：微基准 73836/73926 个请求使用编译数值核；服务 Core 审计中 20060/27232 个 HBM 事务使用该路径。小探针为保留命令事件使用详细参考核，其他不适用编译路径的事务也使用同一 JEDEC 命令模型。

## 参数输入与执行边界

源值/单位、配置值、时钟边界上的最早有效下限分开保存。下表配置和有效下限的时间单位为 ns，原始输入按源单位标注；有效值不是含资源冲突的最终命令间隔。
tRP 等受命令相位影响，后续 ACT 还需对齐上升沿；tRAS/tRP 修改会联动 tRC，RL/WL 修改会联动 tRTW。另设 tRTW 公式对照：保持参考组 RL=20 ns、WL=5 ns，仅采用上游公式得 31 CK=19.375 ns；不把全组不同 RL/WL 下的 17 CK 直接移入参考组。

| 参数 | 上游原始输入 | 参考配置 | Ramulator2 配置 | 参考有效下限 | 对照有效下限 | 单项覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| nRC (activate_cycle_ns) | 71 nCK（公式） | 48.125 | 44.375 | 48.125 | 44.375 | minimum_active_ns, precharge_to_activate_ns |
| nRRDS (activate_different_group_ns) | 4 nCK | 4.0 | 2.5 | 4.375 | 2.5 | activate_different_group_ns |
| nRRDL (activate_same_group_ns) | 5 nCK | 6.0 | 3.125 | 6.25 | 3.125 | activate_same_group_ns |
| nRCDRD (activate_to_read_ns) | 31 nCK | 14.0 | 19.375 | 14.375 | 19.375 | activate_to_read_ns |
| nRCDWR (activate_to_write_ns) | 15 nCK | 14.0 | 9.375 | 14.375 | 9.375 | activate_to_write_ns |
| nCCDS (column_different_group_ns) | 2 nCK | 1.25 | 1.25 | 1.25 | 1.25 | 两组相同 |
| nCCDR (column_different_sid_ns) | 3 nCK | 2.5 | 1.875 | 2.5 | 1.875 | column_different_sid_ns |
| nCCDL (column_same_group_ns) | 4 nCK | 2.5 | 2.5 | 2.5 | 2.5 | 两组相同 |
| nFAW (four_activate_window_ns) | 24 nCK | 20.0 | 15.0 | 20.0 | 15.0 | four_activate_window_ns |
| tRASmax (maximum_active_ns) | formula（本模型补充） | 35100.0 | 35100.0 | 35100.0 | 35100.0 | 两组相同 |
| nRAS (minimum_active_ns) | 45 nCK | 33.125 | 28.125 | 33.125 | 28.125 | minimum_active_ns |
| nRP (precharge_to_activate_ns) | 26 nCK | 15.0 | 16.25 | 15.0 | 16.25 | precharge_to_activate_ns |
| nPPD (precharge_to_precharge_ns) | 2 nCK | 1.25 | 1.25 | 1.25 | 1.25 | 两组相同 |
| nCL (read_latency_ns) | 20 cycles | 20.0 | 12.5 | 20.0 | 12.5 | read_latency_ns |
| nRTP (read_to_precharge_ns) | 9 nCK | 3.75 | 5.625 | 3.75 | 5.625 | read_to_precharge_ns |
| nRTW (read_to_write_command_ns) | 17 nCK（公式） | 18.125 | 10.625 | 18.125 | 10.625 | read_latency_ns, read_to_write_command_ns, write_latency_ns |
| nRFC (refresh_cycle_ns) | 880 nCK | 500.0 | 550.0 | 500.0 | 550.0 | refresh_cycle_ns |
| nREFI (refresh_interval_ns) | 6240 nCK | 3900.0 | 3900.0 | 3900.0 | 3900.0 | 两组相同 |
| REFab (refresh_policy) | all_bank policy（本模型补充） | all_bank | all_bank | all_bank | all_bank | 两组相同 |
| nCWL (write_latency_ns) | 10 cycles | 5.0 | 6.25 | 5.0 | 6.25 | write_latency_ns |
| nWR (write_to_precharge_ns) | 33 nCK | 15.0 | 20.625 | 15.0 | 20.625 | write_to_precharge_ns |
| nWTRS (write_to_read_different_group_ns) | 7 nCK | 3.75 | 4.375 | 3.75 | 4.375 | write_to_read_different_group_ns |
| nWTRL (write_to_read_same_group_ns) | 10 nCK | 5.0 | 6.25 | 5.0 | 6.25 | write_to_read_same_group_ns |

上游组有 7 项配置值超出现有 fast/slow 两端：nRRDL, nRCDRD, nRCDWR, nCL, nRTP, nRTW, nWR。原三档只是选定工程场景，不能当作厂商保证边界或所有公开模型的包络；本次保留原预设并增加独立对照组。

## 逐项影响

每一行只人工替换所列输入，自动联动字段另列。微负载值选取该组相对参考带宽变化绝对值最大的已执行负载；这不是 LLM 加速比例。

| 人工替换输入 | 替换值 ns | 自动联动 | 影响最大的微负载 | 带宽变化 | Small A TPOT变化 | Small B TPOT变化 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| activate_different_group_ns | 2.5 | 无 | activation_pressure | +0.8050% | -0.5565% | -0.1763% |
| activate_same_group_ns | 3.125 | 无 | activation_pressure | +1.6231% | -0.5526% | -0.1765% |
| activate_to_read_ns | 19.375 | 无 | cold_read | -12.1212% | -0.0078% | -0.0021% |
| activate_to_write_ns | 9.375 | 无 | cold_write | +30.7692% | +0.0046% | +0.0000% |
| column_different_sid_ns | 1.875 | 无 | 全部已执行负载均为零 | +0.0000% | +0.0000% | +0.0000% |
| four_activate_window_ns | 15 | 无 | 全部已执行负载均为零 | +0.0000% | +0.0000% | +0.0000% |
| minimum_active_ns | 28.125 | activate_cycle_ns | hot_row_conflicts | +10.0039% | +0.0000% | +0.0000% |
| precharge_to_activate_ns | 16.25 | activate_cycle_ns | activation_pressure | -6.3798% | +0.0013% | +0.0000% |
| read_latency_ns | 12.5 | read_to_write_command_ns | mixed_same_pc | +35.1487% | -0.7979% | -0.1780% |
| read_to_precharge_ns | 5.625 | 无 | late_read_conflict | -1.1952% | +0.0010% | -0.0002% |
| read_to_write_command_ns（公式对照） | 19.375 | 无 | mixed_same_pc | -4.1545% | -0.0042% | +0.0002% |
| refresh_cycle_ns | 550 | 无 | sequential_read | -1.1403% | -0.0231% | -0.0059% |
| write_latency_ns | 6.25 | read_to_write_command_ns | cold_write | -5.5556% | -0.5829% | +0.0000% |
| write_to_precharge_ns | 20.625 | 无 | write_row_conflicts | -11.4939% | +0.0000% | +0.0000% |
| write_to_read_different_group_ns | 4.375 | 无 | different_group_rw | -2.1147% | +0.0000% | +0.0000% |
| write_to_read_same_group_ns | 6.25 | 无 | mixed_same_pc | -2.1213% | +0.0000% | +0.0000% |

不同参数的影响不可直接相加；全组替换可能改变最先阻塞的约束。零变化只表示这些固定条件下未改变观测结果。
尤其是跨 SID 的列命令间隔可能被每 PC 的方向预算遮蔽：32 B / 6.25 GB/s = 5.12 ns，已大于两组 tCCDR；因此专门增加跨 SID 探针，并保留其零影响。
冷读/冷写分别检验 RCD 与管线延迟；晚到读后换行检验 RTP；写行冲突检验 WR；异组读写检验 WTR_S。所有探针保持相同非时序配置、请求地址和到达时间。

## 完整服务

TPOT 越小越好。Small 的输入/批量/输出为 1024/4/8；Qwen2.5-0.5B、Llama3.1-70B 和 Qwen3-235B-A22B 均为 128/1/4。命名模型只做 reference 与 Ramulator2 全组对照。
模型结构与工作量由已执行记录提取；命名模型按 BF16 计算权重存储量，Small 使用 INT8，未下载或计算真实权重数值。

| 模型 | 层数 | 模型权重 GiB | 输入 token | 并发请求 | 每请求输出 token | 架构预设支持级别 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| small | 1 | 0.00110 | 1024 | 4 | 8 | synthetic |
| qwen2_5-0_5b | 24 | 0.92013 | 128 | 1 | 4 | exact |

Llama3.1-70B 为稠密模型；Qwen3-235B-A22B 为 MoE，总参数约 235B、每 token 名义激活约 22B，每层选择 128 个专家中的 8 个。其专家路由是分析近似，不是运行真实权重得出的路由。架构预设的 exact 只表示结构字段覆盖，不能表示性能经过实机标定。


| 模型 | 对照组 | A TPOT µs | B TPOT µs | B/A | 更快 | A变化 | B变化 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| small | reference | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| qwen2_5-0_5b | reference | 1286.50312 | 3260.96146 | 2.53475 | A | +0.0000% | +0.0000% |
| small | ramulator2 | 27.45152 | 125.08045 | 4.55641 | A | +0.0065% | -0.0031% |
| qwen2_5-0_5b | ramulator2 | 1306.50000 | 3272.10229 | 2.50448 | A | +1.5544% | +0.3416% |
| small | single_activate_different_group_ns | 27.29696 | 124.86375 | 4.57427 | A | -0.5565% | -0.1763% |
| small | single_activate_same_group_ns | 27.29804 | 124.86348 | 4.57408 | A | -0.5526% | -0.1765% |
| small | single_activate_to_read_ns | 27.44759 | 125.08170 | 4.55711 | A | -0.0078% | -0.0021% |
| small | single_activate_to_write_ns | 27.45098 | 125.08429 | 4.55664 | A | +0.0046% | +0.0000% |
| small | single_column_different_sid_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| small | single_four_activate_window_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| small | single_minimum_active_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| small | single_precharge_to_activate_ns | 27.45009 | 125.08429 | 4.55679 | A | +0.0013% | +0.0000% |
| small | single_read_latency_ns | 27.23071 | 124.86170 | 4.58533 | A | -0.7979% | -0.1780% |
| small | single_read_to_precharge_ns | 27.45000 | 125.08402 | 4.55679 | A | +0.0010% | -0.0002% |
| small | single_read_to_write_command_ns | 27.44857 | 125.08455 | 4.55705 | A | -0.0042% | +0.0002% |
| small | single_refresh_cycle_ns | 27.44339 | 125.07688 | 4.55763 | A | -0.0231% | -0.0059% |
| small | single_write_latency_ns | 27.28973 | 125.08429 | 4.58357 | A | -0.5829% | +0.0000% |
| small | single_write_to_precharge_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| small | single_write_to_read_different_group_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |
| small | single_write_to_read_same_group_ns | 27.44973 | 125.08429 | 4.55685 | A | +0.0000% | +0.0000% |

### 命名模型的首 token 与完整请求吞吐

TTFT 为从请求到达到首 token 的时间，越小越好。完整请求吞吐为输出 token 总数除以从请求到达到仿真结束的时间，包含初始化等待与 Prefill；它不是稳定 Decode 吞吐，不能用 1/TPOT 替代。

| 模型 | 对照组 | A TTFT ms | B TTFT ms | A TTFT变化 | B TTFT变化 | A完整吞吐 token/s | B完整吞吐 token/s | A吞吐变化 | B吞吐变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2_5-0_5b | reference | 3.49279 | 5.53496 | +0.0000% | +0.0000% | 544.04786 | 261.13336 | +0.0000% | +0.0000% |
| qwen2_5-0_5b | ramulator2 | 3.50457 | 5.54810 | +0.3375% | +0.2374% | 538.78771 | 260.34199 | -0.9669% | -0.3031% |

## 全组替换的独立访存结果

| 负载 | 带宽变化 | P99 延迟变化 ns | 最大请求完成时刻变化 ns |
| --- | ---: | ---: | ---: |
| activation_pressure | -2.9285% | +102.50000 | 120.00000 |
| cold_read | +7.4074% | -2.50000 | 2.50000 |
| cold_write | +21.4286% | -3.75000 | 3.75000 |
| different_group_rw | +24.1358% | -707.50000 | 720.00000 |
| different_sid_reads | +0.4845% | -7.50000 | 7.50000 |
| hot_row_conflicts | +6.1080% | -795.00000 | 1297.50000 |
| late_read_conflict | -0.4016% | -2.50000 | 7.50000 |
| mixed_rw | +7.5472% | -44.37500 | 50.62500 |
| mixed_same_pc | +22.4041% | -671.87500 | 676.87500 |
| random_read | +6.0217% | -41.87500 | 45.62500 |
| row_locality | +0.3336% | -2.50000 | 2.50000 |
| sequential_read | -1.0385% | +138.75000 | 138.75000 |
| sequential_write | -0.8504% | +113.75000 | 113.75000 |
| write_row_conflicts | -5.7972% | +891.87500 | 1268.75000 |

## 解释边界与复现

- 逐项组保留 reference 的其他规则，尤其 RL/WL 自动联动仍使用 reference 的 PHY 假设；它们不应被解释为完整上游参数组。
- 上游未映射的参数及原因保存在 upstream_profile；本轮不新增 REFpb、RFM、低功耗、初始化或错误注入。
- 70B/235B 扩展只测试短输入、单请求、4 个输出 token，因此只有 3 个 Decode 间隔；未覆盖长上下文、高并发、容量不足或长期稳态，更不能据此外推所有大模型和 HBF 容量收益。
- 从 0.5B 换成 70B/235B 时，层数、隐藏维度、注意力结构和稠密/MoE 结构也随模型改变；本实验扩大模型覆盖范围，不是仅改变参数数量的因果实验。
- 本次复现命令：`.venv/Scripts/python.exe scripts/hbm_ramulator2_study.py --stage all --workers 8`。大模型扩展命令：`.venv/Scripts/python.exe scripts/hbm_ramulator2_study.py --stage large --workers 8 --resume`。使用项目虚拟环境以启用数值执行核，并发默认值为 2。可用 micro/serving/named/large/full-service/report 分阶段执行；`--resume` 复用本次目录中已完成且版本、变体、实际输入一致的点。
- 部分执行状态始终为 incomplete；只有全部预定独立访存与完整服务点完成并通过实际执行检查，才标记 passed。
