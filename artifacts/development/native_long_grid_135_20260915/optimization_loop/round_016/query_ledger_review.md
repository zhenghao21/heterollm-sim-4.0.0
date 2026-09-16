# R16 查询账本与MMVQ几何独立复核

2026-09-16。只读审查 `kernel_query_ledger.py`、planner新增查询metadata、`predict_stable_native_dataset.py`紧凑汇总和两组测试；只写本报告，没有修改实现或冻结证据，没有GPU、目标LLM实测或时延拟合。

**结论：没有发现本次新增账本改变当前计算时延或资源需求；K来自typed workload，缓存和原生派发仍明确未知。MMVQ源码公式与已完成14个配置-pair/504次kernel的几何吻合。账本存在4个P2完整性/语义问题，应在将其用于自动覆盖判定前修复；它们当前不构成已启用的成本profile泄漏，因为calibration_eligible始终为false。**

## 1. 计时不变与K来源

读取`git show HEAD:src/heterollm_sim/planner.py`到独立内存模块，分别用修改前/后的planner编译同一个参考场景，没有修改checkout。

结果：

- 326个任务的全部非metadata字段相同：task id、类别、依赖、所有ResourceDemand（含service_ns/bytes/energy/work）等未改变。
- resource_capacities、resource_owners相同。
- 随后的模拟trace非metadata统计相同。

这是一项参考场景结构回归，不等于所有模型/所有调度的精度验证。静态diff也只包含metadata赋值、账本汇总和导出，没有调用新MMVQ成本或改成本系数。

`planner.py:9378`在合并外部operation_metadata之后，直接覆盖`kernel_query_geometry`，其中M/N/K均取`workload`。因此外部metadata中的错误`gemm_k`不能替换账本的`k_logical`。新增`gemm_k`仍用setdefault保留既有specialized metadata，但账本不读取它；两条语义没有混用。

## 2. 独立原生几何交叉检查

只读取已结束的`collection_r2/analysis_0001`与`collection_r3/analysis_0001`中pair_01的`mapped_calls.json`，不取duration生成参数。

比对14个配置-pair、504次MMVQ main（其中11种不同shape/format）：全部grid、block、dynamicSharedMemory匹配新`derive_mmvq_work`公式；实际staticSharedMemory也与源归约数组大小一致。

| shape域 | 本次实测几何匹配 |
|---|---|
| Q5_0 M1，N128/896/4864，K896 | grid=N/4，block=(32,4,1)，small_k=true，shared=1536B；r2/r3均匹配 |
| Q5_0 M4，N128/896/4864，K896 | grid=N/2，block=(32,4,1)，small_k=false，shared=3072B |
| Q8_0 M1，N128/896/4864，K896 | grid=N/4，block=(32,4,1)，small_k=true，shared=1536B |
| Q8_0 M4，N128/896，K896 | grid=N/2，block=(32,4,1)，small_k=false，shared=3072B |

本次没有M2/M5–8、K1024 small-K边界的实际geometry覆盖；这些仍是源码推导/单元测试范围。部分Q8数值或计时质量失败不会改变“观测到这些kernel几何”的事实，但**几何吻合不使失败样本变成可用计时标定**。

新MMVQ模块没有延迟数值、HBM效率或占用率；`native_dispatch_proven=false`、`binary_source_equivalence_proven=false`、`cost_model_applied=false`正确保留。不能把CTA数直接除以SM数作为HBM效率，不能因504次几何相同就证明实际资源服务率。

## 3. P2：导出汇总丢失未表示任务数量和缺失原因

核心`summarize_kernel_queries`正确返回`gpu_gemm_tasks`、`represented_tasks`、`unrepresented_tasks`和`missing_reason_counts`，但`retained_dispatch_summary`新增嵌套汇总只保留represented_tasks和complete，没合并后两项。

最小反例：同batch包含1个有效GPU GEMM和1个缺geometry任务。核心账本正确记录总数2、表示1、未表示1、missing_typed_geometry=1；导出嵌套账本只有represented=1、complete=false，缺失原因和未表示数量消失。父级`gpu_invocations.gpu_gemm_tasks=2`尚能恢复总分母，所以当前不是静默删掉整个失败样本；但消费者单读嵌套账本无法解释缺口。

建议在嵌套账本保留observed_gpu_gemm_tasks、unrepresented_tasks、missing_reason_counts、missing_ledger_batches及分母是否完整。缺batch历史时不能将未知未表示数量写0；保留observed值与完整总量None的区分。增加带missing geometry、missing batch ledger和history truncation的聚合测试。

## 4. P2：complete_geometry只检验M/N/K，会把缺格式/设备/字节标为完整

目前只有m/n/k_logical被要求正整数；随后weight_formats、activation_storage_bytes、output_storage_bytes、accumulator_bits、读写属性和target_component均通过.get进入键。

最小反例：geometry只有m=1/n=128/k_logical=896，其他字段全部缺失。结果represented_tasks=1、unrepresented_tasks=0、complete_geometry=true，key内其余物理字段都是null。

完整shape可以成立，但完整kernel query geometry不能成立。建议区分`complete_mnk`与`complete_typed_geometry`，或严格检查该schema声明的必要字段。模型权重格式必须非空且合法；运行时RHS非量化路径允许空格式时应单独标明类别。typed bits/bytes、model_weight_read/rhs_is_activation、target_component缺失或类型错误应记录原因，不靠null假装完整。

这不会误开当前calibration门，但会高估未来静态联合查询覆盖率。测试应包括bool-as-int、缺format、缺target、缺bytes及动态RHS合法空format。

## 5. P2：同task_id但不同geometry会被静默去重

目前seen是set，第二次相同task_id无条件continue。重复引用同一个对象正确；相同id携带不同K或目标却属于数据冲突，不应静默采用第一个。

最小反例：task_id相同，第一个K896、第二个K1024，得到gpu_gemm_tasks=1、complete_geometry=true，仅保留K896。

建议seen存每个id的canonical物理facts，重复相同值去重，重复不同值报错或记conflicting_task_identity并让完整性失败。不同batch重复使用相同局部task id是另一回事，汇总按batch求和仍正确，不能跨batch盲目去重。

## 6. P2：activation_storage_bytes捕获发生在转换之前，却在main账本中被解释为kernel输入

`planner.py:9378`写geometry时workload还是原F32输入。后续约9755行MMVQ把`activation_storage_bytes`改成Q8_1临时输入，MMQ约9778行改成consumer_unique_bytes；main cost使用的是转换后的workload，但geometry没有刷新。

因此例如M1/K896记录activation_storage_bytes=3584，而MMVQ main实际被计价的consumer输入为1008字节。当前字段也可能是合理的“GGML逻辑算子输入”，但schema scope写的是“main physical geometry only”，不能把两者当一个字节维度。

建议明确保留两类字段：`logical_input_f32_bytes`（转换前）与`main_consumer_storage_bytes`/`internal_activation_layout`（转换后）。后者在最终GPU workload定型后赋值，K/M/N和逻辑shape不改变。转换角色自己的账本应独立；不能为了修报表修改既有成本需求或把两份输入都计费。

## 7. 正确的未覆盖标记

这些已有处理应保留：

- cache_state=unknown、layout_proven=false、native_dispatch_proven=false、calibration_eligible=false固定写入，不能让外部metadata覆写为true。
- MMQ的k_executed只读取applied源工作量；MMVQ仍为None，不从转换padding推断执行K。
- schema没有model_name、prompt fingerprint或目标时延索引。
- ledger main task数量明确不是原生kernel总数，conversion/fixup/launch不偷偷乘进去。

## 8. 下一步机制方向

查询账本完善后可以客观回答哪些真实typed shape落入源码几何域。当前MMVQ几何已有多shape实测结构佐证，但计时仍须独立资源证据。下一轮可在flag关闭不变的前提下比较MMA-wave代理与MMVQ CTA/warp/Kloop物理并行上界，禁止将后者直接变成线性HBM比例。转换和MMQ主/fixup应分角色消融；不能通过补高conversion掩盖已高估的MMVQ main。

本报告没有要求重新运行任何目标LLM native，也没有改变当前验收阈值。
