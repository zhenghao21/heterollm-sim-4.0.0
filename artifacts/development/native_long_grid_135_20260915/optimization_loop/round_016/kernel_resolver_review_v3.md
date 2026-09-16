# Kernel resolver v3 增量复核

日期：2026-09-16。审查源码SHA `84ca007b6f79da222b3ae906f7fc69bb98dfaf3e19a1c0b433ff2f004fcf0ac6`，测试SHA `44d6a3aab76f9e73dfe42f781a9c9bf593c49d697b80504ddfeb788bf1946619`。仅写本报告，不改resolver或冻结采集证据，不运行GPU、不读LLM actual。

**结论：上一轮四项P1的定向反例已封闭；但真实已完成pair暴露两项P1兼容性阻塞，仍不能宣布真实profile可导入。** 这两项是提取语义兼容问题，不是降低质量阈值；修复后当前首配置也必须保持质量拒绝。

## 旧P1复核

仅执行6个定向测试，没有重跑root负责的完整141项：

`python -m pytest tests/test_kernel_calibration.py -q -k 'extra_unowned_gpu_kernel_overlap or freeze_false_receipt or explicit_clock_false_quality or one_pair_replayed_three_times or clock_original_readback or specific_json_limits'`

结果：**6 passed / 135 deselected，1.53秒**。

- 额外同设备未归属kernel/memcpy/memset区间重叠现在拒绝。
- clock false、freeze_after false现在拒绝，正式区间原始telemetry重新bracket。
- 同一pair重复引用现在拒绝，原SQLite SHA、路径和process identity参与区分。
- 原始JSON按kind提高到128MiB，profile/owner仍16MiB，累计JSON1GiB门存在。
- SQLite显式closing已加入；MMVQ源码已从两个词升级为blocks_per_row_x=ncols_x/qk和按blocks_per_iter循环的具体anchor。

## 真实首pair结构核验

使用已完成的 `collection_r2/runs/train_Q5_0_m1_n128_k896/pair_01`，没有查看LLM结果。

- profile/direct的36次原始数值/QPC重新检查通过，各30次formal。
- 两个过程的spec、supervisor-started、complete、外部批准freeze和clock绑定核验通过。
- 原始telemetry重新验证：profile/direct均30/30 formal有clock bracket，最小和最大SM频率均2392MHz，属于预先冻结的2400±30MHz。
- 原始SQLite与保留raw_events逐表比对通过。
- 但 `_correlate_kernel_samples` 在真实API记录上失败，详见下两项。

实际配置quality（3个pair聚合）为 `diagnostic_quality_rejected`，`measurement_cost_eligible=false`，原因：

1. pair0 direct_host_p90_p10
2. pair1 direct_host_p90_p10
3. pair1 profile_direct_perturbation
4. pair2 direct_host_p90_p10
5. across_profile_process_medians

**这组数据不可导入accepted profile。下面的格式修复只能恢复准确诊断，不能删掉这些质量失败。**

## P1-A：正常事件尚未就绪被带版本API名误判失败

位置：`_correlate_kernel_samples` runtime returnValue门，约592行。

代码允许`name_text == "cudaEventQuery" && returnValue==600`，但实际Nsight2026.5记录的名称是 **`cudaEventQuery_v3020`**。首pair全部36次调用各有一次这样的正常未就绪观察，主机raw同时保留`cuda_query_before_wait=600`。当前loader返回`CUDA runtime API failed`。

应使用经过确认的CUDA API规范名称归一化（例如精确匹配`cudaEventQuery`或`cudaEventQuery_v<digits>`），仅为这个查询API保留600状态；其他API非零、名称未知、异常后缀仍拒绝。添加真实row的fixture，不能仅测不带后缀的构造名称。

只在内存把这36条API名变成无后缀，再调用原函数，可通过该门并暴露下一个问题；未修改原始JSON/SQLite。

## P1-B：所有Info诊断也被一概拒绝

位置：`_correlate_kernel_samples`结尾，约658行：`if tables.get("DIAGNOSTIC_EVENT"): raise ...`。

实际首pair有14条诊断，**全部severity=1**；同一SQLite的 `ENUM_DIAGNOSTIC_SEVERITY_LEVEL`明确id1为`Info`。包括Profiling started/stopped、injection initialized、Loaded CUPTI path和事件统计，collector_r2按冻结策略得到`warning_diagnostics_verbatim=[]`。

在上一步仅内存归一化API名后，resolver返回`profiler diagnostic requires explicit independent review`。因此即便未来质量全部合格，正常带Info记录的Nsight trace也无法进入成本profile。

应从同份实际枚举表解释severity：已知Info/Verbose可以保留原文不阻断，Warning/Error/Fatal和未知等级拒绝；不要只按severity数值硬编码，也不要清空原始diagnostic表。若希望需要人工审核Info，需有明确哈希绑定review记录及消费路径，不能仅写一个没有可满足入口的拒绝条件。添加真实14条Info和未知级别/Warning负例。

## 设备、时钟、kernel和缓存严格度

目前可支持的是**同一被观测设备/运行库、同一已冻结时钟域、相同联合shape/kernel/launch geometry下的条件性设备时间下界**，不是通用跨硬件校准：

- 实际GPU UUID、SM数、L2、CC来自SQLite和过程观测，与有效硬件文档比对；硬件配置hash绑定HBM/HBF等调用方完整配置。不同配置回退是正确行为。
- 2400±30MHz由原始每过程telemetry包围所有formal区间验证。采样5ms、最近前后允许25ms，证明的是冻结观测门通过；并非每纳秒频率恒定，也不证明没有采样间瞬态。
- MMVQ main的实际符号、量化类型、同stream、grid/block/shared-memory与键精确相等。源代码K循环anchor仅证明锁定源码声明，不能补全原始native build缺失的源码/二进制对应关系。
- cold-sweep>=4×L2是明确实验条件，不等于测到每次HBM事务、也不等于LLM中相同矩阵处于相同缓存状态。当前LLM cache未知时必须回退；不能因为形状相同就使用cold样本，更不能改查询cache字段硬凑命中。
- 完整设备elapsed作为同阶段下界仍不是tensor/HBM资源占用分解；`validated_llm_scope=false`、`resource_occupancy_calibrated=false`等标记应保留。

没有额外放宽阈值建议。Bridge目前应对现有首配置输出拒绝条目；若所有已结束配置都失败，输出“无可用标定”，不生成假的通过profile。

## 其它注意

源匹配仍明确是条件性同DLL观察。effective hardware factory仍要求planner传入完整有效配置，不能用手写的少数字段冒充完整状态。当前1GiB聚合上限需要bridge按可用训练域合理分片；不能在同一profile中重复加入相同证据路径。新P1修复需要更新源码/测试及新冻结身份，不得原位改已冻结采集文件。
