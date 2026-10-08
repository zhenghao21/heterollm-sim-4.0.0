# 本轮交付与本机历史产物

本目录根部的五模型、Graph 开关共十份 `scenario_*_graph_*.json` 是统一的场景入口。
前端提交、任务记录、结果、native 原始测量、独立成本及最终比较报告仍按原路径保存。
只有 `comparison_*.json` 明确标记为 `compared` 的结果才构成有效配对；文件存在不代表运行已完成。
最终十组均已完成并通过配对，累计 3,860 个批次全部使用物理执行路径。
运行汇总见 [execution_summary.json](execution_summary.json)，最终误差见 [report.html](report.html)。
矩阵从首组启动到最后一组结束耗时约 79 分钟；45 秒间隔监控观察到的专用仿真进程最高
private memory 为 20.92 GiB，系统可用内存最低为 95.28 GiB。这是采样值，不是连续峰值，
也不包括先前中止尝试。十组无失败，报告采集完成后已停止本轮 8780–8789 专用服务。

Graph 开启时五模型 E2E 有符号误差依次为 +25.38%、+13.97%、−4.18%、+4.87%、−6.23%。
native Graph 耗时降幅为 16.63%–30.82%，仿真降幅约 −0.05% 至 −0.02%，
降幅误差约 −30.87 至 −16.66 个百分点。生命周期及配置通过核验不等于预测精度合格；
本轮实验成本没有获得泛化资格，没有根据目标模型实测时延调参。

前端真实导入、校验与提交已经验证，回归画面见 [frontend_regression.png](frontend_regression.png)。
本地 HTML 报告的浏览器预览被工具的 URL 安全策略拒绝（不允许 `file://`），未绕过限制；
报告生成及其三项自动检查通过，但不宣称完成了该报告的浏览器视觉检查。
本次整理没有删除、移动或重写运行中的输入、索引和结果。

以下内容保留在本机，通过本目录 `.gitignore` 排除重复或中间产物；不是最终比较依据：

| 本机目录或文件 | 原因及保留方式 |
| --- | --- |
| `before_runtime_owner_fix/` | driver 提交成本所有权修复前的输入、运行和比较。旧流程同时保留原提交项及独立 CUDA API 计价；修复将已验证的 driver 提交项交给独立成本所有者，保留 CPU 命令构建、GPU/DRAM 工作。旧结果不能冒充修复后的预测。 |
| `cancelled_memory_pressure_27b/` | 27B 在物理图缓存保留过多、内存压力下中止的记录。缓存修复释放可重建的物理执行图并限制保留数量，按相同输入重建；这里的中止运行不是最终结果。 |
| `replay_inputs/scenario_*.json` | 根目录场景的逐字节相同副本。保留本机便利入口，仓库只保存根目录一套；需要这些路径时重新复制对应根文件即可。等价摘要仍随仓库保存，摘要中的 `ready_scenarios` 是本次本机副本位置。 |
| `source_program_fragments/qwen*_update_pairs.json` | 五份中间文件重复嵌入完整 CUDA 节点记录；保留本机用于诊断。提交四类唯一旧→新结构的 `all_update_pairs.json`，独立测量不依赖这五份完整副本。 |

上述相对路径在本次机器上的共同根目录为
`F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/docs/cuda_graph_validation_2026-10-08/`。
受控编译的 dry/capture 原始结构位于
`F:/codex_project/_scratch/37-native-validation-raw/cuda_graph_2026-10-08/bound_{slug}/`。
这些本机路径不是可移植下载地址；更换机器应使用本仓库的生成工具和指定模型重新编译。

重新生成时，先按 [结构编译说明](../../tools/native_cuda_graph_trace.md) 使用
`compile_cuda_graph_structure.py`，输入对应基础 `scenario_*_512_128.json`、GGUF 和固定 native 构图器，
执行 dry/capture-only。将新 fragment 与 `runtime_typed_chain_measurements.json` 交给
`prepare_cuda_graph_cases.py --attach-case`，在独立输出目录生成 on/off 场景，避免覆盖正在使用的结果。
测试直接读取根目录最终 0.6B 场景及基础场景，不依赖被排除的本机副本，也不在测试中重贴 producer contract。

完整 update-pair 中间记录的生成方法是：逐调用读取 dry 的节点属性，按
`CudaGraphRuntime.prepare/commit` 推导生命周期；以 `(context_id, graph_key)` 保存上一份 executable 对应的
capture-only 拓扑。采用 `cuda_graph_serving.py` 相同规则判定更新兼容性，节点数改变产生
`constraints_failure`；完全覆盖且 signature 相同才接受成功，其余情况拒绝。只在源码推导的
`update_failure` 转换处导出前后拓扑、节点数及对应捕获节点，将
`(old.topology, old.node_count, new.topology, new.node_count)` 去重得到 `all_update_pairs.json`。
本轮唯一配对为 533→425、729→594、937→762、2372→1956；不读取 native 执行决策或模型时延。
`run_cuda_graph_structure_microbench.py --capture-trace ... --update-pairs source_program_fragments/all_update_pairs.json`
可按这些结构重做独立小 CUDA 程序测量；无需重建完整中间副本。

五模型受控重编译均为 386 次调用，新旧源码生命周期与 typed topology 差异均为 0，
记录见 [编译等价摘要](replay_inputs/compilation_equivalence_summary.json)。
最终 driver 所有权修复后的专项回归为 35 个 Python 文件、458 项通过。

主代理完成的真实前端回归记录：刷新本机 8788 页面后导入 8B Graph-off JSON；DOM 确认导入期间
`runEnabled=false`、`importEnabled=false`。导入完成后一次点击运行即打开 386 批次确认对话框；
只关闭对话框，没有创建额外任务，浏览器 error 日志为空。生命周期代理随后验证的前端 JavaScript
回归为 29 项通过。该检查验证导入与启动交互，不等同于十组仿真全部完成。

## driver 成本所有权修复前后的实际数值差异

[逐项差值记录](driver_owner_observed_delta.json) 包含六组双方均完成的结果：
Qwen3 0.6B F16 与 1.7B Q8_0 的 Graph off/on，以及 4B、8B Q4_K_M 的 Graph on。
4B、8B 的旧 Graph off 运行没有完成，不能用于前后比较。
这里比较的是修复前后仿真结果，不是 native 预测误差，也不是 Graph 开关收益。

六组正式请求 `04_measured0` 的 engine TTFT 均不变，TPOT 均减少约 250 ns，
E2E 均减少约 31.75 μs（尾数存在浮点差异）。不能以 128 × 250 ns 的服务时间之和
直接宣称 E2E 减少 32 μs；实际端到端差值取决于依赖和重叠。

包含启动、预热与正式请求的完整 386 批次报告中，逻辑/物理读写字节、burst 数、
row hits/misses/conflicts 均精确相同。已比较的批次数、物理执行批次数、任务数、
资源计费字节和总能耗也均相同；每个 DRAM 资源的 `bytes_moved`、`service_ns`、
`energy_pj` 均精确相同，资源集合及其 owner 没有变化。
但顶层 `dram_traffic.service_ns` 和 `queue_wait_ns` 并非完全不变：

| 模型与 Graph 状态 | service 差值（ns） | service 变化幅度 | queue 差值（ns） | queue 变化幅度 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B F16 off | −500.000064850 | 0.000079463660% | −1000.000000000 | 0.000011591246% |
| 0.6B F16 on | −499.998296857 | 0.000079386294% | −1000.000000000 | 0.000011589604% |
| 1.7B Q8_0 off | −499.998642683 | 0.000056640438% | −1000.000000000 | 0.000012814505% |
| 1.7B Q8_0 on | −499.998812437 | 0.000056593915% | −1000.000000000 | 0.000012812122% |
| 4B Q4_K_M on | −500.000337362 | 0.000040703680% | −999.995452881 | 0.000008188955% |
| 8B Q4_K_M on | −499.998838425 | 0.000023119304% | −1000.000000000 | 0.000008228128% |

差值定义为“最终值减旧值”，变化幅度为 `abs(差值) / 旧值 × 100%`。
最大 service 相对变化为 **0.000079463660%**（0.6B off），
最大 queue 相对变化为 **0.000012814505%**（1.7B off）。

这两个 service 字段口径不同：
[`data_motion.py`](../../src/heterollm_sim/data_motion.py) 中事务 `service_ns` 定义为
`completion_ns - arrival_ns`，可能包含排队；
[`planner.py`](../../src/heterollm_sim/planner.py) 的 `_summarize_dram_task_traffic`
分别汇总事务时间与资源需求中的忙碌时间，
[`reporting.py`](../../src/heterollm_sim/reporting.py) 的 `_sum_batch_dram_traffic` 再汇总各批次。
因此资源忙碌服务时间相同，不代表事务等待和到达至完成时间必然相同；这些累计量也不等于端到端时延。

对 0.6B off、1.7B on 两组原始批次记录的进一步核对表明，大于 0.1 ns 的 DRAM 时间变化
集中在 `cohort-000001`（`01_model_seq_rm_probe`）和 `cohort-000002`（`02_warmup0`）
两个 prefill 批次：每批 service 约减少 250 ns、queue 减少 500 ns。
这部分变化不能全部解释为浮点舍入；其余聚合尾数存在微小浮点差异。
尚未对其他四组逐批定位，也未将差值单独归因到具体物理 task。

归因仍有边界：六组记录均标为 `compared_with_input_differences`，因为结构程序输入补充了
`contract`、`request_stages`，并更新 dry/capture 记录路径。模型、硬件及其他 workload 输入相同，
[编译等价摘要](replay_inputs/compilation_equivalence_summary.json) 显示源码生命周期与 typed topology
差异为 0，但这仍不是只改变 driver 项的独立实验。
前置提交时间变化可以影响物理访问的到达和等待状态，是合理的机制解释；
现有证据不足以证明上述全部差值仅由 driver 成本所有权修复引起。
`read_write_switches`、`refresh_wait_ns`、`turnaround_wait_ns` 在两侧均未提供，
保留为 unavailable/null，不能按零处理或声称这些指标不变。
