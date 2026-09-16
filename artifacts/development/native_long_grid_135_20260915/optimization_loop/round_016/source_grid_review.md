# R16 source_grid 并行成本与账本修复审查

日期：2026-09-16。只读审查 `cost_models.py`、`tests/test_source_grid_cost.py`、root更新的账本与汇总；仅运行几个内存成本计算和账本反例，不修改core、冻结、native，不运行GPU或simulator场景预测。

**结论：CTA→可同时活跃SM数上界有依据；当前乘法实现可以作为默认关闭、明确假设的保守分析候选，但不能描述成无附加假设的物理必要吞吐上界。建议使用“峰值cap”语义，或在冻结前明确声明现有效率是每个活跃SM的局部效率。20锚点冻结可设计，但本次不执行。**

## 1. CTA几何上界与现有乘法的区别

令S为SM总数，B为这次普通kernel启动的CTA总数，A=min(B,S)，p为每个SM上某类执行资源的峰值吞吐。假设每CTA在任一时刻只占用一个SM，则同时能有该kernel工作的SM最多A，因而该kernel该资源瞬时总吞吐不超过A×p。

CTA被暂停或在别的时刻重新调度也不会令同一时刻一个CTA占用多个SM；不需要额外证明它终身不迁移。这个推导只适用普通CTA所属SM的执行资源，不适用把HBM带宽按SM数线性分摊。root保持memory_bandwidth_fraction=1是正确的。

现有`GPUProfile.elementwise_gops`和`special_function_gops`已经是：

`C_old = S × p × attainable_efficiency × occupancy`。

新实现又乘`A/S`，得到：

`C_new = A × p × η`，其中η=0.65×0.85=0.5525。

所以新实现不仅应用`C<=A×p`的物理上界，还假定所有活跃SM的可用资源效率仍等于旧的η。若旧occupancy描述每个活跃SM驻留warp/发射效率，且旧attainable_efficiency也是局部因素，该乘积有一致的建模解释；若旧η已经吸收小grid空闲SM/全设备利用率，乘A/S就重复惩罚。代码中的旧字段没有强制说明这一分解，不能声称无重复风险。

### 推荐语义

如目标是**只加入源码几何证明的必要峰值cap**，使用：

`C_candidate = min(C_old, A × p_peak)`，等价`T_compute=max(W/C_old, W/(A×p_peak))`。

它保留现有分析效率，不再无证据地假定η按活跃SM重新折扣。比如S84/B4时：当前乘法把服务时间放大21倍；仅峰值cap相对旧模型放大84×0.5525/4≈11.60倍。这不是根据实测挑数字，而是两种假设的数学差异。

若保留当前21倍式，应把metadata写成`active_sm_scaled_inherited_local_efficiency`，明确`local_efficiency_transfer_assumed=true`，不能仅写成“必要上界”。现有`occupancy_measured=false/timing_calibrated=false`很好，但还应标明η的归属假设。

## 2. SFU、warp数、依赖和资源争用

- 在同质SM且操作确实使用每SM SFU的前提下，A×每SM SFU峰值同样是总吞吐上界。每CTA的warp数不同可能让真实利用率更低，不会使这条上界失效。
- warp数不足、寄存器/shared限制、每CTA尾部退出、依赖链/issue端口仍未被B单独表达。`min(B,S)`是最多活跃SM数，不是实际occupancy，也不保证每个活跃SM已经吃满SFU。
- `transcendental_operations`与profile的SFU吞吐单位需要匹配真实指令类别。把所有除法/round/convert当SFU操作并不自动成立；本改动没有补上这个证明。
- dependency_depth/frequency原项保持不变，没有把grid因子再乘到依赖延迟；这避免同一等待重复放大。
- 单一scalar/SFU资源的独占计时需求仍不是按SM分片的并发资源占用。两个各只用4CTA的kernel若同时就绪，现有单资源引擎可能仍串行它们。源grid上界只改善隔离kernel服务估计，不能声称并发GPU利用率模型也修好。
- launch-only目前早返回，不消费source_grid_ctas。零算术fixup的设备控制服务仍未定价，不能把这一字段说成已解决此问题。

## 3. 代码和轻量验证

源码变更仅修改GPU typed tensor kernel路径：source_grid_ctas未给出时返回原estimate；>=SM时乘数1；只缩放scalar/SFU吞吐，不改变memory需求、字节、能量、launch、依赖链。

root提供的6项测试覆盖scalar服务倍数、memory/energy/launch保留、饱和grid和非法值，没有重复跑。补做一个不运行GPU的小计算：operations=0、read_bytes=4、transcendental_operations=1000，84SM/B4时SFU服务由0.897795ns变18.853695ns，恰21倍；说明SFU分支确实应用同一因子。

建议冻结前补三项精确行为测试：SFU主导路径、memory-only路径完全不变、缺字段/饱和grid逐需求与旧结果相同。测试应检查绑定资源和phase，不只检查总耗时；不能为了曲线平滑强制实际路径处处单调。

一个现存非本改动问题：`TensorKernelWorkload(operations=0,read=0,write=0,transcendental_operations>0)`构造允许，但typed_roofline拒绝“未声明ops/bytes”。本次SFU验证用真实4字节输入绕开；这不是source_grid引入的回归，可以另列小结构修复。

## 4. 四项账本P2复核

### 已修复

1. 核心缺失原因/unrepresented数量现在合并到紧凑导出。
2. typed geometry现在验证格式、设备、字节、accumulator与bool字段；只有M/N/K的记录不会再complete。
3. 同task_id而shape/target/consumer不同会抛`conflicting geometry`。
4. 转换前字节改名logical_input_storage_bytes；转换后的main consumer字节在最终workload确定后捕获，M1/K896分别3584/1008，不再混为一个维度。

### 剩余两个小语义点

- 紧凑汇总`missing_batch_ledgers = len(outer_ledgers)-len(valid_query_ledgers)`不计整个gpu_invocations外层也缺失的batch。内存反例：两个batch，一个完整、一个cost.metadata为空，父级missing_batch_summaries=1且gpu_gemm_tasks=None正确，嵌套missing_batch_ledgers却=0。完整性flag仍false，没有误宣称通过，但字段易误读。建议改名`missing_query_ledgers_within_present_invocation_ledgers`，或增加total_missing_batches=len(batches)-len(valid_query_ledgers)。unrepresented_tasks应明确为observed而非完整总分母。
- 当前要求weight_formats非空会把合法unpacked/动态RHS GPU GEMM列为unrepresented；这可以作为“量化kernel查询域”设计，但schema需要写清。如果目标是所有GPU GEMM几何，应保留显式`unpacked/dynamic_rhs`类别，不要把空格式当无效数据，也不应凭空补量化格式。

另外identity冲突fingerprint目前不含mmq_source_work的path/执行K；同task id同shape但不同audit可能只保留首条。正常planner应无此冲突，可在将ledger用于校准时再补路径冲突检查。

## 5. 建议R17二十锚点冻结方案（不执行）

建议输出根：`artifacts/development/native_long_grid_135_20260915/optimization_loop/round_017/source_grid_candidate/`。这只是后续路径建议，本次没有创建目录或运行预测。

- 先冻结本次假设：选择`active_sm_peak_cap`（推荐）或`active_sm_scaled_inherited_local_efficiency`；不得看锚点误差后悄悄替换语义并仍使用同一版本号。
- 新入口默认off，例如`--source-grid-compute-bound`，只给锁定源码可导出的普通GPU tensor conversion/fixup等调用显式B。没有source proof或不在支持域继续None，不从model名称或误差推算B。
- 冻结内容：核心源码、工作量派生器、flag与默认值、有效GPU/CPU/内存runtime profile、既有native selection与raw timestamp/extractor身份、聚合脚本、锚点清单和统计阈值。源grid参数不是native重新标定。
- 20锚点复用既有固定native，只跑simulator。按预先声明的静态M/N/K、model分组、prompt/output/concurrency覆盖选取；若沿用已存在20-anchor清单，保留其原id和身份，不按这次error结果删格换格。所有锚点已经揭示，属于开发/回归，不是盲测。
- 同一20格保存三路：A当前分析基线（flag off），B峰值cap，C乘η版本仅作预注册机制消融（如果团队决定比较两种假设）。若只批准B，则C不执行，不能引入额外隐含模型选择。
- 修改前先逐任务检查None路径的所有非metadata字段不变、内存字节/需求不变、每source绑定B可追溯、kernel/control计费不重复。profiling样本没有被接受，不能增加“测量下界”作为第四条来提高通过率。
- 每格Engine TTFT/TPOT/E2E分别报告signed/APE/absolute ms；分组median/P90/worst与每格三项<10%的数量共同报告。覆盖失败与未表示查询保留分母，MMVQ main仍未改变的格子应如实标明无机制覆盖。
- 自动保留/回滚依据：机制正确、未见静态shape的方向是否合理、最差组与覆盖不退化。不能仅因20格平均改善就宣称目标达成。完整131/162对照和下一独立验收集仍是后续工作。

## 6. 对当前目标的影响判断

本字段主要能提高小grid conversion的分析时间，不降低高估的MMVQ main。本轮设备诊断已经表明这两项方向相反；只补conversion可能增大某些总误差。R17必须把角色成本变化与Engine结果联系起来解释，而不是把结果不好归因于噪声。当前仍未达到每格Engine三项<10%的目标。
