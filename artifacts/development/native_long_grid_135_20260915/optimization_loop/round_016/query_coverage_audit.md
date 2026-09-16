# R16 合成查询覆盖审计：R14 tail_candidate_r2

更新结论：**仅用 retained prediction 不能恢复精确逐调用账本，但冻结模型路径指向的 GGUF 元数据目录足以导出静态 M/N/K/格式签名。** 新增 metadata-only 导出后，154 个含模型/投影类别的静态签名中有 21 个精确匹配 R16 training 配置；在“层物理矩阵 × 保留批次 M”计数口径下为 299,952/522,804（57.374%）。这是静态潜在查询覆盖，不是实际调用回放。缓存、真实派发/融合与连续频率条件仍未证明，完整域可接受查询仍为 0；不能因静态形状匹配直接使用样本成本。

审计只打开该候选的 20 个既有 prediction JSON、其 manifest、冻结预测器源代码及 R16 r2 protocol。没有打开 native/raw、评分 errors、comparison 或 strict_gate，没有读取 LLM actual，没有运行仿真/原生/GPU，也没有使用延迟拟合。下列是**模拟物理任务账本**统计，不是 LLM 原生逐内核实测。

## 范围与证据完整性

- 20 个已预测配置：qwen25 17 个，qwen35、smollm2、tinyllama 各 1 个；候选 manifest 固定分母为 131，已成功 20，尚未预测 111。不能将本子集推及完整网格。
- 总计 3,111 个批次，全部批次历史保留且 details_truncated=false；但 cost_metadata 每批仍省略 72/74 个字段，累计 48,608 个 execution_tasks 条目被替换为 depth_limit 标记。
- 全部 20 个 prediction 的 diagnostic_events_requested=false；predictions 目录没有诊断事件文件。遍历现存结构，没有任何完整 M/N/K 三元组，也没有逐查询 weight_format/weight_type/quant_type 或 cache_policy/cache_state/cold_cache。
- native_dispatch_proven_tasks 总数为 **0**。MMVQ/MMQ 计数只是模拟器按绑定规则选择的路径，不能替代真实内核符号与角色证据。

| 模型组 | 配置 | 批次 | GPU GEMM 任务 | 合格物理投影任务 | MMQ source-work applied | MMVQ precedes MMQ |
|---|---:|---:|---:|---:|---:|---:|
| qwen25 | 17 | 2,994 | 626,277 | 479,784 | 85,776 | 370,224 |
| qwen35 | 1 | 39 | 7,010 | 6,510 | 1,488 | 4,278 |
| smollm2 | 1 | 39 | 7,712 | 5,808 | 1,344 | 3,720 |
| tinyllama | 1 | 39 | 7,072 | 5,324 | 1,232 | 3,410 |
| 合计 | 20 | 3,111 | 648,071 | 497,426 | 89,840 | 381,632 |

GEMM 任务数不是 NVIDIA 内核数；conversion/main/fixup 也不能未经角色账本映射就各计一次真实调用。

## R16 能证明的形状范围

R16 固定 26 配置覆盖 Q5_0/Q8_0 两格式，输入/输出 F32、普通连续二维布局：

- training 18 项：M={1,4,64} × N={128,896,4864}，K=896，两格式。
- validation 4 项：M={2,32}，N=1792、K=896，两格式；只作留出验证，不参与拟合。
- aligned_control 4 项：M={4,64}、N=896、K=1024，两格式；不自动扩大训练范围。

R14 能读取的 physical_batch_rows 分布如下。它是批次属性，**不能保证等于每个投影任务的实际 M**，尤其存在融合、输出行筛选及非模型权重 RHS 时。

| 批次物理行数 M | 批次数 | 所在批次的 GPU GEMM 任务数 | R16 的 M 取值是否出现 |
|---|---:|---:|---|
| 1 | 1,060 | 203,526 | training |
| 2 | 826 | 179,242 | 仅 validation，且只测 N=1792/K=896 |
| 3 | 75 | 16,275 | 未覆盖 |
| 4 | 612 | 132,804 | training / aligned_control |
| 64 | 528 | 114,054 | training / aligned_control |
| 9、13、22、25、49 | 各 2，共 10 | 各 434，共 2,170 | 未覆盖 |

仅按 M 做宽松筛查，3,026/3,111 批次（97.268%）落在 R16 任一 M 取值；其中仅 2,200/3,111（70.717%）落在 training 的 M 取值。**这两个百分比都不是查询覆盖率**：没有 N/K、格式、角色、布局、缓存身份，无法形成可用查询键；M=2 对应的 826 个批次也不能套用 M=1/4 的训练结果。

## 必须回退的条件与最高优先缺口

1. **先补非计时查询账本，而不是扩大形状猜测。** 下一次预测应保存每个物理调用或去重签名的 M/N/K、实际权重类型、输入/输出 dtype、layout、batch/row-selection、设备与运行时身份、模型权重/运行时 RHS 分类、kernel family/role 和计数。保留原始调用到去重签名的对应；不能复用被截断 execution_tasks 去猜。
2. **缓存未知全部回退。** R16 是每调用之前至少 max(128 MiB,4×L2) 扫读的显式 cold-cache 条件；R14 没有对应声明。native_configuration.cache_ram_mib 是服务缓存配置，不能据此宣称 CUDA L2 cold-cache。匹配形状也不能默认把 cold 样本转用于未知或热缓存查询。
3. **M=3 与残余 M 是已证实的取值缺口。** 75 个 M=3 批次及 10 个 M=9/13/22/25/49 批次所涉及的 18,445 个 GEMM 任务，连批次 M 筛查也无法进入 R16 取值。是否优先补哪组 N/K，应等完整物理查询账本后按调用数量决定，不从时延误差选择。
4. **MMVQ 是高频候选路径，但尚未证明格式/形状覆盖。** 381,632 个模拟路径判定为 mmvq_precedes_mmq；89,840 个使用 MMQ source-work。R16 两/三内核链只能在真实捕获符号、量化类型和源路径参考一致时接受，不能由这里的 expected path 直接晋级。
5. **现有未覆盖原因必须保持回退。** MMQ 工作账本另外记录 176,599 个 uncovered：147,768 个 runtime RHS 非物理模型权重，25,378 个单一物理投影未证明，2,877 个权重格式不支持/混合，576 个源 allocation high-water 不覆盖消费范围。这些不能用 26 个普通连续 Q5_0/Q8_0 样本越过原有门。

完整静态签名缺失、非目标量化类型、N/K 不在已接受证据范围、MMVQ/MMQ/融合语义不一致、未知缓存、验证集查询误当训练以及未通过配置级数值/波动/扰动门，都保持解析成本或原回退路径；不以近邻形状、模型名或 LLM 答案补值。


## Metadata-only 补充：可直接计算的静态查询范围

新增 `static_query_coverage.py` / `static_query_coverage.json`。只解析四个实际已预测模型的 GGUF header、metadata 和 tensor directory，复用冻结 `gguf_parity.py` 的元数据标量解析器；没有调用其完整模型读取/哈希函数，没有读取 tensor payload，也没有打开 27B 模型。总计仅读取 **20,401,554 字节**前缀，并分别记录 prefix SHA256、前后文件 stat 和对冻结文件大小的核对。历史完整模型 SHA 作为继承身份保留，**此次没有重新证明全文件 SHA**。

每个层矩阵按唯一物理 tensor name 计一次，GGUF shape[0]=K、shape[1]=N，再与已保留 batch 的 physical_batch_rows=M 组合。没有导入/执行 planner、run_scenario 或成本估计器；也没有根据模型名、时延或误差选择形状。严格排除 output head、运行时 RHS attention GEMM、非矩阵 SSM 操作及 MTP。该统计的别名去重规则明确，但 M=1 gate/up 融合、设备分配和输出行选择仍可能改变实际调用数量，故不把“矩阵×批次”命名为实测/回放调用次数。

| 模型 | 每批已识别层物理矩阵 | 静态矩阵×批次 | 去重签名 | 精确 M/N/K/格式匹配计数 |
|---|---:|---:|---:|---:|
| qwen25 | 168 | 502,992 | 90 | 299,952 |
| qwen35 | 186 | 7,254 | 28 | 0 |
| smollm2 | 168 | 6,552 | 18 | 0 |
| tinyllama | 154 | 6,006 | 18 | 0 |
| 合计 | — | 522,804 | 154 | 299,952 |

精确匹配覆盖 **12 个不同 M/N/K/格式键**（因为同键可对应不同 projection，按模型/投影展开为 21 个签名）：M={1,4,64}；K=896；Q5_0 的 N={128,896,4864} 与 Q8_0 的 N=128。全部落在 training，validation/aligned_control 没有精确匹配。R14 GGUF 文件级名称含 Q4_K_M，但逐矩阵真实类型显著不同：Q5_0 395,208、Q8_0 37,332、Q4_K 50,592、Q6_K 38,268、Q5_K 1,404（均为静态矩阵×批次）。因此不能用文件名或模型名代替物理格式键。

最高优先静态缺口已经可以明确：

- **M=2 / K=896 / N=128、896、4864 的 Q5_0，以及 N=128 的 Q8_0**：当前缺少精确配置，合计 118,944 个静态矩阵×批次。现有 M=2 validation 只覆盖 N=1792，不能替代这些键。这比只观察“M 集合存在2”更能识别实际缺口。
- **qwen25 down projection 的 K=4864、N=896，Q4_K/Q6_K**：71,856 个静态矩阵×批次；既不在目标格式，也不在现有 K 域，必须原路径回退。
- 另外仍有 M=3 与残余 M=9/13/22/25/49；现在可查看 JSON 中逐个完整静态键，而不是只列批次 M。
- qwen35、smollm2、tinyllama 的现存静态键没有一个精确落入 26 配置；不能用 qwen25 结果代表它们。

已保留硬件读回样本为 **2370–2392 MHz**，数值落在新 2400±30 域的边界内；它们仅为历史采样，不能证明每个查询在目标频率域运行。原预测没有 cold-cache 声明，新 probe 则每次 cold sweep；全部 522,804 个静态潜在查询仍携带 cache_domain_not_proven、native_dispatch_not_observed、frequency_domain_not_proven 的回退标记。

这次补充取代“只有重新预测才能获得 N/K/格式”的过强说法：**静态几何可从既有 GGUF 目录恢复，实际调用计数/融合/缓存仍需要更完整账本。** 不把 57.374% 作为真实 GEMM 覆盖率，也不据此拟合或开放查询。

## analytical_predictions.py 只读审查

脚本只接受合成 protocol 的 26 个静态配置，未访问 LLM actual/profile；明确 2400 MHz，分 conversion/main/fixup 与 launch，MMVQ 主体仍是 generic proxy。这一用途与上述静态查询导出相容，但它不负责生成 R14 物理查询账本。

硬件引用路径已实际核对存在：`optimization_loop/operator_microbench_v2/driver_device_properties.json`。发现一项实现身份边界（未修改脚本、未运行预测）：protocol 先读 JSON、最后才计算 protocol_ref，未在开始/结束对同一原始 bytes 绑定；源码也在 modules 导入后才采集摘要。应按已经读入的 bytes 绑定 protocol 起始身份并末尾核对，避免中途修改导致“输出是旧内容、引用是新文件”。当前 sources 未完整列出其所有间接公式依赖，不能把该清单描述为完整实现闭包。

此外 stateless_compulsory_GEMM_IO 是明确解析假设，不等于测得 cold HBM；profiles 中的效率常量仍属结构假设。保留脚本已有不拟合/不外推声明即可，不把相近结果升级为源/运行时等价证明。

## 身份与复核

- R14 manifest SHA256：`5b767c86d769b5f741319be55f42f75211c7dd2e7778802183d6f87bdff62fc6`。
- R14 20 个 prediction 的按文件名字典序 SHA256 清单规范化摘要：`5f31f96718be6d9997c0979fa0fdd60086da55ed5335dbc26129406f0226f2b1`。清单成员仅为 predictions/*.prediction.json；未纳入评分或 native 原始记录。
- R16 r2 protocol SHA256：`36bcb1a9c758765801e71fc8f20c1b0964322a485bab5c96f50f94e92dfe33f2`。
- 统计字段：dispatch_summary.gpu_invocations / mmq_source_work、batch_schedule.batches[].cost_metadata.physical_batch_rows、execution_stages[].execution_tasks。没有使用 latency/ns 数值作为筛选或权重。

补充静态导出 JSON SHA256：`c7991bc23c7d7bd01dbf0327199ddd5593f0429ddc59a585926eaf7221d5a568`。
