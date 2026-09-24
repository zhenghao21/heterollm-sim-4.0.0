# Qwen3.8-27B 多介质场景：P0 / P7 / P8 工具

> 面向应用的结论见“两个应用场景的第一版答案”；公开来源参数与下一轮默认扫描见文末“公开标准与论文参考参数”。前面保留按时间推进的历史实验，旧版 BLOCKED/单请求限制不代表当前全部能力；新硬件精度仍全部 UNVALIDATED。

## 定位与证据边界

所有新硬件场景均为 **UNVALIDATED**：使用显式、可替换的分析参数，未测量 HBF、3D DRAM 或 SRAM-CIM。机制检查 PASS 不等于硬件预测精度合格；不输出凭空计算的硬件误差、Spearman ρ 或“<25%”精度结论。不启动 native、不读取评分答案、不改固定 R0、optimization_loop 或既有 artifacts。

模型直接由项目 GGUF metadata sidecar 经 `read_gguf_metadata_cache(strict=False)` 和 `build_model_from_gguf` 构造。校验 sidecar 内容、解析器身份、源文件 stat 与缓存 SHA 绑定；**不重新读取/计算 14.8GB 权重哈希**。缺失/过期 sidecar 直接失败，不偷偷重建。

核实的模型：64 层主干，48 linear attention + 16 full attention；隐藏维度5120。保留 GGUF 每 tensor 混合量化类型，不把整个文件的 MOSTLY_IQ3_M 标签当作统一位宽。`nextn_predict_layers=1` 单独记录，本轮 workload 关闭 MTP。

## P0 身份

`artifacts/memory_tier_scenarios_20260922/p0_frozen_head_summary.json` 是真正 before-change 小基线：HEAD `60b07028b3f7942638c8d70613fa44eb068d85a6` 的全部 `heterollm_sim` Python 模块通过只读 `git archive` 在内存隔离载入，没有复制工作目录或改 R0。8 prompt / 4 output / batch1，保留源身份、sidecar摘要、请求完成和指标摘要。

此前 `p0_baseline_manifest.json` / `p0_baseline_report.json` 是工作树观测，**不是冻结 HEAD before**；初次请求 exact retention 被 continuous 调度器拒绝，失败文件保留，随后使用合法 aggregate 路径成功。before结果：TTFT 53.233459552ms、TPOT 52.669464844ms、E2E 211.241854084ms，均是 reference analytical 预测而非实测。

## 入口

在仓库根运行（Python3.12现有虚拟环境，不新增依赖）：

```powershell
.venv\Scripts\python.exe tools/qwen38_memory_scenario.py --scenario baseline --dry-run --output artifacts/memory_tier_scenarios_20260922/baseline_plan.json
.venv\Scripts\python.exe tools/qwen38_memory_scenario.py --scenario hbf_active_split --output artifacts/memory_tier_scenarios_20260922/split_short.json
.venv\Scripts\python.exe tools/qwen38_memory_scenario.py --scenario baseline --prompt-tokens 512 --output-tokens 128 --output artifacts/memory_tier_scenarios_20260922/baseline_long.json
.venv\Scripts\python.exe tools/memory_tier_sweep.py --scenario all --dry-run --output artifacts/memory_tier_scenarios_20260922/matrix_plan.json
.venv\Scripts\python.exe tools/memory_tier_sweep.py --scenario hbf_idle --output artifacts/memory_tier_scenarios_20260922/idle_sweep.json
```

`--output` 必须位于新结果目录，文件 exclusive-create，重跑应换新文件名。`--dry-run` 只构造/校验配置，不执行引擎。正式矩阵应在核心源稳定后启动；运行前后源码摘要变化标 `INVALIDATED`，不能将并行变更混入最终验收。退出码0表示模拟/计划完成，不代表精度验证；缺能力/检查失败/源变更返回2。

主函数：`load_model()`、`build_scenario(model, scenario, prompt_tokens=8, output_tokens=4, batch=1)`、`execute_scenario(scenario)`。复用 reference、V4 config reader、CPU control-plane placement、`run_scenario(... retention_policy='aggregate')` 与真实报告。只存紧凑资源聚合，不重复保存整份任务轨迹。

## 小而全场景

| scenario | 实际意图 | 边界 |
|---|---|---|
| baseline | 原reference硬件+真实27B，短prefill/decode | 自动placement，UNVALIDATED |
| hbf_idle | baseline增加未用Flash端点 | 与自身延迟扫描比较，映射必须不变 |
| hbf_remote_flash | 缩小专用KV HBM容量，两请求竞争，KV/state交换到Flash | 必须swap in/out和HBF read/write均大于0才算覆盖；缺合法linear-state-offload约束即BLOCKED |
| hbf_remote_weights | FFN权重Flash按调用冷读，其他权重HBM | 真实HBF读流量；不是KV交换覆盖 |
| hbf_active_memory | rank-local HBF active profile，FFN权重HBF、其余HBM | `access_mode=memory`，真实controller路径 |
| hbf_active_weights | HBM执行，FFN权重HBF | 跨tier暂存不等于page级直访 |
| hbf_active_kv | KV放HBF，linear state放HBM | 分开核算，不混为KV |
| hbf_active_split | full-attention KV与linear state分别按层分配HBM/HBF | 固定层级放置，非动态page级迁移 |
| dual_dram_cim | SoC+两DRAM+CIM，独立PHY，固定部分权重到CIM | 需要实际cim0.array demand；32位reference accumulator不足真实FP16时拒绝，不降低模型位宽；当前planner额外min(32,profile位宽)，单改profile不足 |
| dual_dram_cim_shared | 同上，共享PHY/NoC | TSV/NoC真实资源有流量才算覆盖 |

HBF例子不假装已知物理规格：沿用项目存储测试的64Gb/s读、32Gb/s写、20ns链路假设。active统一4096B物理事务粒度、32 outstanding，读写延迟独立；profile保守4GB/s公共带宽，端点仍分别用读写带宽。控制器与端点的开销归属依核心，不能重复记成独立“实测”。

remote_flash短case会显式将batch至少设2以制造压力，实际workload写入清单。设置了offload但无交换不能记为通过。较长请求若超过设置容量可能被拒绝，检查将失败而不是给出“更快”的虚假结论。

每场景只扫 read latency=50/100/200ns 三个短形状；可用 `--include-long` 额外加一个512/128形状，不做全笛卡尔组合。形状、随机种子、模型、硬件、placement约束和源码均有身份。源稳定不等于硬件已校准。

## P7 验证

纯函数位于 `tools/memory_tier_sweep.py`：

- `zero_traffic_invariance`：相同模型、workload、placement，完整资源ledger中端点bytes/service均0，TTFT/TPOT/E2E必须不变。未知资源完整性不能按0通过。
- `monotonicity`：固定placement的弱单调性，延迟增大时耗时不降低；允许平坦瓶颈区但额外标记是否实际敏感。发生重映射不自动判错，记NOT_COVERED，避免把优化器切换和硬件单调性混为一谈。
- `validate_observation`：非负有限计数、请求完成、KV/state分开、每个层级/权重放置约束对应实际映射、权重shard归属与容量和；执行demands字节总和=批次资源字节和=报告资源字节。
- `scenario_coverage`：Flash必须真实双向KV交换；active HBF必须有实际访问；双DRAM必须controller/TSV资源真实访问；CIM必须实际array service。拓扑上有设备不代表已验证。

资源字节是逐跳/缓存累计，**不得等同唯一权重负载大小**；副本/填充/暂存分开。controller observed bytes也可能与bulk重复，仅作为诊断不能另加总。

最小检查：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_memory_tier_validation.py -q
```

## 后续硬件校准

要宣称预测效果，还需用户提供目标硬件拓扑、controller/PHY共享关系、读写带宽与尾延迟、事务粒度/并发、容量、CIM精度/累加位宽/加载路径，以及独立微基准与留出工作负载。训练/校准与验收数据必须分开。现在仅能证明指定分析假设下的机制一致性，不能保证未知硬件绝对误差。


### 覆盖门补充

正式场景结果只有所有内部检查为PASS且所要求机制确实被执行才能记SIMULATED；NOT_COVERED不是成功。资源检查包含每批原始demand非负、有限值、完整批次计数，全请求和全部输出token均完成。每个cell保留实际batch、完整硬件/profile/placement/workload authoring配置及模型配置哈希（模型用sidecar恢复），不重复保存巨大模型图与任务轨迹。任一cell检测源码漂移立即停止剩余cell并标INVALIDATED。

线性状态交换接口已接入 `PlacementPolicy.linear_state_offload_target`，不通过手工tensor映射绕过CPU控制面。某些旧版核心的swap cohort不提供逐资源bytes ledger，此时资源守恒保持失败，不能通过把缺失当0或仅复制summary总数消除错误。


### CIM 算术边界与双DRAM对照

核实真实GGUF侧文件：**F16二维矩阵数量为0**；F32二维矩阵仅48个SSM conv1d权重，不是可直接映射的FP16 GEMM。默认数字CIM整数位切片不能把IQ3/IQ4当整数位平面，也不能将32位累加器改48位就宣称支持浮点。

新增 `dual_dram_only` / `dual_dram_only_shared` 是**明确不同的无CIM执行对照**：保留原GGUF与双DRAM拓扑，未转码或修改模型。即便拓扑中保留CIM器件，报告 `cim_execution=NOT_COVERED`，不替代原 `dual_dram_cim` 请求成功。原CIM请求因缺少真实packed权重到dense FP16转换契约应继续fail-closed。

测试中独立合成的小FP16算子仅验证 `fp16_fp32_analytical` 契约：FP16操作数、FP32舍入部分和、显式float tile周期与输出速率，均为UNVALIDATED假设，非Qwen权重转换、非数值等价证明、非真实CIM硬件测量。

Flash压力场景显式 `placement.metadata.linear_state_offload_mode=pressure`：只在serving压力触发交换，不每个算子镜像状态。HBF物理总流量仍可能同时含KV和linear-state交换，应看分别列出的KV swap in/out与linear-state counters，不能称全部HBF字节为KV字节。


### 最终矩阵判定口径

`--scenario all` 现在包含12种显式场景×3个短延迟点，共36点；原双DRAM+CIM仍单独保留BLOCKED，不被双DRAM-only对照替代。每点都通过V4配置往返一致性检查。单调性必须三个点全部机制验证成功且placement不变；自动优化改变placement时记NOT_COVERED，不从成功子集挑点。

TSV资源可能是无方向后缀的 `link.vertical_dram0/1`，共享归属从核心 `declared_resource_owners` 读取（包含hardware.physical_resource_owners）；共享资源在报告中可仍保留logical名称，不要求出现虚构的owner计数器。物理owner流量的聚合只描述归属，不新增传输字节。

原 `dual_dram_cim` 显式绑定 `fp16_fp32_analytical`、7 cycles/eval、2 outputs/cycle的合成假设profile（UNVALIDATED），以便真实拒绝点落在packed转换缺口，仍不允许IQ/Q直接位切片。当前GGUF没有F16矩阵，不能据独立FP16单元测试声称其端到端CIM成功。


## 本轮正式结果（v4）

源码稳定，36点：30点SIMULATED且内部机制检查PASS，6点CIM因packed转换缺口BLOCKED，0个FAILED_CHECKS。全部36点配置往返PASS。整体状态INCOMPLETE，不冒充所有需求已完成；硬件精度统一UNVALIDATED。结果见新目录 `p8_short_formal_v4.json` 与 `summary.json`。

100ns设计点如下（8/4，Flash压力batch2，其余batch1；不同拓扑/放置不作公平硬件排名）：

|场景|状态|TTFT ms|TPOT ms|资源累计GB（逐跳）|
|---|---|---:|---:|---:|
|baseline|SIMULATED|205.272|191.160|226.443|
|hbf_idle|SIMULATED|53.236|52.669|226.443|
|hbf_remote_flash|SIMULATED|75.376|127.863|453.830|
|hbf_remote_weights|SIMULATED|5290.173|3205.492|258.171|
|hbf_active_memory|SIMULATED|4202.739|3762.785|254.954|
|hbf_active_weights|SIMULATED|3018.455|3017.889|258.171|
|hbf_active_kv|SIMULATED|53.211|52.644|226.448|
|hbf_active_split|SIMULATED|116.705|116.137|227.072|
|dual_dram_cim|BLOCKED|—|—|—|
|dual_dram_cim_shared|BLOCKED|—|—|—|
|dual_dram_only|SIMULATED|222.122|200.766|267.315|
|dual_dram_only_shared|SIMULATED|222.122|200.766|267.315|

较长512/128基线独立稳定运行成功：TTFT823.223ms，TPOT52.782ms，E2E7526.534ms；KV峰值41,943,040B，linear state156,893,184B。其源身份单独保留，不宣称与v4同源或实测准确。趋势检查详见summary；发生placement切换的组保持NOT_COVERED。共享拓扑在此短顺序workload未必产生可见争用差值，不能只凭连接共享就保证减速。


## 后续增量：显式冷转换与固定放置（2026-09-22）

新增 `dual_dram_cim_converted`、`dual_dram_cim_converted_shared`，与原始、不带转换的 CIM 场景分开；旧场景仍拒绝压缩权重直接执行。新路径不修改 GGUF，将 IQ3_S/IQ4_XS 压缩权重按调用转换成 dense FP16，计入原始读取、FP16 padding 写入、阵列编程、必要的 FP32 激活转换。转换吞吐与浮点硬件均为显式未验证假设，非实测或数值等价证明。

当前转换仅支持单请求、单序列、TP/PP/EP=1；调用内保守资源锁保护暂存区，返回后无跨调用复用。SRAM 数组容量与专用转换空间分开占用，组件容量必须容纳两者。不支持的格式、容量不足、暖驻留及并发组合拒绝执行，不提供免费转换。下一步要扩展多请求，必须先实现入站暂存的动态预留生命周期。

示例仅将首层 FFN 分配给 CIM：数组512MiB、转换空间512MiB，组件1GiB；decoder及激活转换16元素/ns，是敏感性坐标，不是产品规格。真实27B短运行每批转换两块矩阵：压缩读取123,944,960B、dense padded合计534,773,760B、单矩阵转换空间峰值433,111,040B。合计物化量不是同时驻留峰值。

`--placement-mode fixed` 在100ns求解后，通过合法的 `operator_targets`、权重和状态政策约束冻结放置；所有点重新正常求解并核对最终mapping，不改签名或忽略差异。默认adaptive保留重放置分析。各组源身份独立保留；不得将不同源文件版本自动当同源严格配对。

已完成：baseline、双DRAM独立/共享、带冷转换CIM独立/共享，每组50/100/200ns共15点，均稳定执行，固定放置单调性通过。结果文件分别是 `fixed_baseline_side_task_20260922.json`、`dram-fixed-v1.json`、`dram-shared-fixed-v1.json`、`cim-converted-fixed-v1.json`、`cim-converted-shared-fixed-v1.json`（实验目录同上）。短串行负载下共享/独立结果相同，不足以证明带宽争用收益。新观测字段 `storage_distribution` 给出各介质权重分片、张量/状态归属和资源流量，不把逐资源累计字节误作唯一payload或峰值容量。

本轮针对性集成回归257项通过。新硬件依然UNVALIDATED；不能把内部机制检查通过写成25%实测精度已达标。


## 多请求暂存生命周期增量

上一节单请求限制已被有界单槽实现替代：TP/PP/EP仍为1，允许多请求。静态与在线共同将入站激活/权重、转换、阵列计算和输出搬离最后消费者封装成原子资源调用；一个scratch槽及涉及的物理资源保守占用整个区间。调度器等待槽释放后才允许后续调用，避免只锁计算而在入站时提前覆盖。原task IDs与逻辑访问标记保留为零资源事件，实际流量只计一次。无跨调用缓存复用；更细粒度流水、多槽/双缓冲仍未实现。

暂存容量现在包含输入/输出缓冲，不能沿用早期只计权重转换的峰值。小容量在执行前拒绝，失败不会残留预留状态。真实静态双请求测试确认scratch占用区间不重叠，并观察到queue wait和资源前驱；失败后重新编译一致。

真实27B p8/o4 的batch2、batch4分别完成2/4请求及8/16输出token，资源检查通过。batch4 prefill报告8次转换，压缩读取495,779,840B、dense累计2,139,095,040B，scratch单槽峰值433,750,016B（累计转换不是同时占用）。结果 `cim-b4-lifecycle-v1.json` 运行期间源码稳定；仍UNVALIDATED。当前保守整调用锁可能高估等待，不当作最高吞吐预测。本轮相关集成122项通过。

本机校准证据见 `docs/LOCAL_CALIBRATION_20260922.md`；实测PC算子不可直接替代CIM内置decoder或阵列参数。


## 两个应用场景的第一版答案（2026-09-22）

### 交付边界：现在能回答什么

现在可交付**参数条件下的趋势与放置策略**；不能交付未经目标器件独立测量验证的绝对性能保证。P0–P8 的功能/机制检查不等于 P8 的真实硬件精度验收。新硬件无需先制造才能进行设计探索，但没有目标硬件、可信器件模型或独立测量时，只能给假设敏感性范围，不能将其称为误差或置信区间。

本节不重新发明验收框架，复用现有场景执行器、资源账本、固定放置与测试。固定 R0、native 原始数据和 optimization loop 不变。

### 模型和容量口径

针对项目现有 GGUF，而非仅凭“27B”估算：64 层主干，其中 16 层 full attention 和 48 层 linear attention；MTP 关闭，保留逐张量混合量化。

- 当前无 CIM 对照中，模型内持久权重分片约 **13.565 GiB**；这不是 GGUF 文件大小，也不包含所有运行时工作区。
- FP16 KV 的逻辑增长为 **64 KiB / token / request**，包括本模型 16 个 full-attention 层的 K 和 V；不再把线性状态算作随上下文无限增长的 KV。
- linear/recurrent state 约 **149.625 MiB / request**，在该模型配置下不按已见 token 数线性增加。
- 仅作容量推算：32,768 个有效 token 对应约 2 GiB KV/请求；8 个这样的请求，KV 加线性状态约 **17.169 GiB**，还没加权重、页对齐、激活、转换区和运行时开销。这不是已跑完的 32K 性能实验。

### 场景一：HBM + HBF

#### 先区分两种角色

1. **远端 Flash / 后备存储**：数据要先搬到计算端可用内存，计入存储读写、页粒度、排队、链路与目标端写入。权重源文件和暂停请求状态可放这里；如果模型每个 token 都要读取某份权重，不能因其不可修改就将其当作“冷数据”。
2. **活动内存语义**：允许权重/KV/状态在 HBF 上持久驻留并服务调用，但不会自动消除 Flash 的访问粒度、延迟、写入约束和互连限制。当前跨介质搬运不是经过数值/硬件验证的透明 load/store 系统。

#### 已完成的同源配对

`application-pairs-v2.json` 两组 KV 配对的实际算子、权重与线性状态映射相同，仅 KV 目标介质改变。单请求、输出 8 token；下表是 TPOT，即首个输出之后的平均每 token 时间：

| 输入 token | KV 在 HBM（ms） | KV 在 HBF（ms） | 变化 |
|---:|---:|---:|---:|
| 128 | 52.669 | 53.399 | +1.39% |
| 2048 | 53.033 | 101.292 | +91.00% |

**这组是 4 GB/s HBF 通路压力案例，不是高带宽 HBF 产品预测。** HBF 端点读 8 GB/s、写 4 GB/s，通路 4 GB/s，活动 profile 4 GB/s；页 4096 B，32 outstanding，读 100 ns / 写 200 ns 均是示例假设。不能将上表的比例推广到其他 HBF 带宽、延迟、上下文、批量或调度器。

两组 TTFT 恰好不变，不代表没有写回：prefill 分别写入 8,388,608 和 134,217,728 B，HBF 物理写流量包含这些写入与 7 次 decode 追加。源码允许写回与后续计算重叠；仅凭总指标不进一步声称已测得每条关键路径。

新增 `application-weight-pairs-v2.json` 固定算子、工作负载与同一模式的驻留设置，只切换 64 个 MLP 权重组的源介质。输入 128 / 输出 8 / batch1：

| 同模式比较 | MLP 权重在 HBM，TPOT ms | MLP 权重在 HBF，TPOT ms |
|---|---:|---:|
| active-memory 路径 | 52.669 | 3017.913 |
| remote-flash 冷取路径 | 52.669 | 3205.516 |

它说明在**这条低速通路、逐调用取权重、无跨调用预取复用**的设定下，整批 MLP 权重放在远端很昂贵，不说明所有 HBF 方案都差。两种模式之间不能直接排名，尤其其冷加载/驻留开关不同。

远端 KV/state 压力案例已有真实 swap-in/out：旧 v4 的 100 ns 点 KV 交换读写各 1,048,576 B，线性状态交换读写各 156,893,184 B。HBF 总流量不能全部叫 KV。此点是机制覆盖证据，不是与上表同源的公平性能对照。

#### 补充：高读带宽也必须有足够访问并发

SanDisk 2025 年 7 月 HBF Fact Sheet 给出的首代设计点包括 1.6 TB/s 读取和 512 GB 每栈；这里只把这两个数作为公开规格坐标，不将宣传性能比例迁移给本模型。该资料“接近无限 HBM”的脚注使用 Llama 3.1 405B、8-bit 预训练权重的内部测试/仿真，与本项目混合量化 27B、KV 与调度条件不同。

来源：`https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf`（读取于 2026-09-22）。

`hbf-high-read-sensitivity-v1.json` 采用相同 p128/o8/b1、固定算子与状态放置，MLP 权重通过远端 HBF 冷取。HBF 读取峰值和假设的连接通路均设为 1600 GB/s，页 4096 B；**下列读延迟与并发数都是假设，不是厂商数据**：

| 权重介质 | 读延迟 us | 最大未完成访问数 | TPOT ms |
|---|---:|---:|---:|
| 同组 HBM 对照 | — | — | 52.669 |
| 高带宽 HBF 假设 | 1 | 32 | 107.697 |
| 高带宽 HBF 假设 | 10 | 32 | 653.169 |
| 高带宽 HBF 假设 | 100 | 32 | 6107.889 |
| 高带宽 HBF 假设 | 10 | 4096 | 52.209 |

这组没有 HBF 写流量，不验证 KV 写入或非对称 rank-local active-memory 性能。采用当前保守存储端点语义：带宽服务加延迟批次，后续还要计入链路与目标内存阶段；不是把峰值带宽直接当持续带宽。

粗略预算 `有效速率 <= min(物理带宽, 并发数 × 每次访问字节 / 延迟)`：4 KiB、10 us、32 outstanding 对应约 13.1 GB/s 的并发上限，远低于 1600 GB/s；4096 outstanding 才可能不再被这个单项上限卡住。有限事务量、串行阶段及其他瓶颈仍使端到端收益低于理想化预算。

结论是**HBF 权重方案可以接近 HBM，也可以慢很多，关键在路径实际带宽、访问延迟和足够并发**。52.209 对 52.669 的小优势仅是模型内结果，不宣称真实 HBF 更快或已达到厂商宣传值。这四个 HBF 点实际放置相同、HBF 写流量为零；固定并发的延迟单调性与增加并发的不恶化检查均通过。

#### 可直接采用的策略起点（尚非最优策略搜索结果）

- HBM 优先保证正在执行请求的线性状态、需要频繁读取的活动 KV、激活和搬运/转换工作区；剩余容量再与高复用权重做收益权衡。
- 普通 full attention 每步需要读历史 KV；“旧 token”不自动等于“不会被访问的冷 KV”。将其搬远端并不会免费减少读取；按时间冷却的自动分页策略当前未实现。
- HBF 远端模式更适合持久权重源、暂停/换出的请求状态或经过流量预算的权重流式读取。容量压力下应比較减少并发与 swap 的吞吐/SLO 代价，而不是只比较能不能放下。
- 真正高读取带宽的 HBF 可以优先探索**读密集权重放 HBF、活动 KV/线性状态放 HBM**，但仍要检查有效带宽是否被随机访问延迟、并发深度或链路限制。KV 放置同时涉及写入性能，不能只用 HBF 读取峰值判断。

### 场景二：双层 DRAM + SoC + SRAM-CIM

#### 双 DRAM 本身不保证两倍性能

已有独立/共享 PHY-NoC 成对拓扑，两层 DRAM、控制器、TSV 和片上路径均计入资源。固定短顺序负载下，两种拓扑指标相同：这类负载没有触发足够重叠竞争，不能宣称仿真错了，也不能据此说共享没有影响。

已有定向传输检验实际触发竞争：每层各传 1 MiB，单层时两拓扑均 8.262 us；两层同时传时，独立路径 8.262 us、共享路径 12.368 us。它证明模型有共享争用机制，**不是 27B 的端到端提速比例，更不保证双层 DRAM 永远恰好两倍**。

#### SRAM-CIM 要算完整路径

成本至少为：压缩权重读取 → 解码/FP16 物化 → 阵列装载 → 激活转换 → CIM 计算 → 输出搬离。只有节省的传统计算/数据搬运足够覆盖这些成本时才加速；是否复用已经装载的权重是关键，不能省略加载成本来制造优势。

现有转换场景仅将 `layer-000.mlp` 这个粗粒度组交给 CIM，包含不止一次底层 GEMM。没有把整个 27B 放入 SRAM。样例还给了 **512 MiB 阵列 + 512 MiB 转换区**；这是分析坐标，不是已经证明面积/功耗可行的 SoC 设计。p128/b2 的单槽 scratch 峰值约 422.8 MiB。如果目标片上 SRAM 只有数 MiB，必须先实现分块转换/流水/复用，不能只改小容量后仍宣称同一模型能运行。

修复后的严格配对（`application-pairs-v2.json`，p128/o8/b2，除首层 MLP 执行及其必需存储路径外其余配置相同）为：

| 首层 MLP 执行方式 | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|
| SoC 执行，不使用 CIM | 3439.624 | 365.693 | 5999.474 |
| CIM 每调用冷转换/加载 | 3736.425 | 404.590 | 6568.554 |

TPOT **+10.64%**，而非加速。这支持“冷转换/加载不能忽略”，不证明暖驻留或流水 CIM 没有价值，也不是全模型 CIM 的效果。

本轮修正容量准入：首层 MLP 的压缩源按实际 payload 与量化元数据计 **123,944,960 B**，作为 `layer-000.mlp_weights#cold_backing` 常驻来源 DRAM；CIM 中的永久 dense 权重仍为 0，scratch/阵列另行计量。开启/关闭冷 CIM 的 packed 持久容量均为 14,565,150,720 B，不再因改变执行器而消失。普通存储分片也按现有投影描述器计实际物理量，而不使用该组 135,450,366 B 的 nominal 位宽估值。旧 v1 的容量统计已由 v2 取代，原文件未覆盖。

多请求暂存生命周期已经支持 TP/PP/EP=1 下的 batch2/batch4：从入站到最后输出消费者完成才释放，后续请求会等待。当前单槽整调用占用是保守实现，不能把它作为双缓冲、流水重叠后的最高吞吐。此前 batch4 p8/o4 已完成 4 请求、16 输出 token，源稳定且检查通过；本轮 `cim-b4-lifecycle-v2.json` 重跑仍完成 4 请求、16 输出 token；prefill 8 次转换，scratch 峰值 433,750,016 B，源稳定且资源检查通过。

### 什么结果才算“对”

1. **实现正确**：资源/字节归属明确；KV 与线性状态分开；完成所有请求和输出 token；容量不足必须拒绝；暂存不重叠覆盖、不泄漏；不能靠丢请求使延迟更好看。
2. **趋势正确（固定放置、相同负载与其余参数）**：未使用器件参数变化不影响结果；提高必要路径的延迟不应让端到端更快，但可因重叠不变；增加并发能力在该瓶颈解除后应趋于饱和；共享资源只有在时间重叠时才产生额外等待。自适应放置变化必须单列，不和固定映射单调性混合。
3. **预测精度正确**：必须对未参与参数拟合的独立目标测量验收。建议先协商工程目标为：TTFT/TPOT 相对误差中位数 ≤15%、90 分位 ≤25%；对大于实测噪声的配置差异，趋势方向一致率 ≥90%。这是**建议目标，非业界统一标准，也不是目前已经通过**。若无实测则不计算误差；参数上下界扫描也不称统计置信区间。8-token 小样本不能验证在线 p95/p99。

### 继续做可信设计预测所需的最少输入

不要求先购买 SRAM-CIM。规格未知也可给范围；不要假装默认值就是目标器件。

| 项目 | 至少需要 |
|---|---|
| 工作负载 | 输入/输出长度分布、并发/到达率、KV 精度、首 token/逐 token 时延目标 |
| HBM/HBF | 各自可用容量，读/写**有效**带宽与延迟范围，访问粒度，并发深度，链路带宽，HBF 两种角色各自允许的访问/写缓冲语义 |
| 双 DRAM/SoC | 每层容量/带宽/延迟，控制器和 PHY/NoC 是否共享，TSV 及片上互连速率，SoC 算力/频率 |
| SRAM-CIM | 阵列容量与暂存容量分别多少，支持的精度/累加方式，装载带宽，tile 形状/计算周期，是否跨 token/请求保留权重，解码位置与速率 |
| 可信度依据 | 厂商规格、独立器件模型或公开/用户微基准；没有则保留分析假设并扫范围 |

### 不能被现有检查隐藏的剩余限制

- **活动 HBF 的读写带宽非对称**：端点路径支持不同读写速率，但 typed HostMemoryProfile 仍是单带宽，当前校验要求不超过读写较小值。它不足以精确表示“高读取峰值、低写入速率”的 rank-local 活动 HBF；不能把读带宽复制给写带宽规避约束。
- 按 KV 页自动冷热分级、主动预取距离、跨调用 CIM 常驻权重复用、MB 级分块转换及多槽流水仍不属于已验证能力。不要把它们的理想收益算入现有结果。
- 热模型目前是给定工作点降额，不是已校准的瞬态 3D 热耦合；SoC 算子模型仍是 GPU 代理，不能当真实 SoC 核函数实测。
- 本机补测只是 PC 证据且未启用拟合 profile；不能将本机 GDDR7/NVMe/CPU/GPU 的结果重命名为 HBM/HBF/3D DRAM/CIM 校准。详见 `LOCAL_CALIBRATION_20260922.md`。


### 本轮结果索引与复核

结果位于 `artifacts/memory_tier_scenarios_20260922/`，不覆盖旧文件：

- `application-pairs-v2.json`：4 个 KV 点 + 2 个 CIM 点。
- `application-weight-pairs-v2.json`：4 个固定算子权重介质对照点。
- `hbf-high-read-sensitivity-v1.json`：1 个 HBM 对照 + 4 个高读带宽 HBF 假设点。
- `cim-b4-lifecycle-v2.json`：修复容量后的多请求暂存重跑。

这 16 个执行点均完成全部请求，合计 112 项现有内部检查通过，运行中源码稳定且与复核时当前源码一致；这些是同一套内部机制检查的重复应用，**不是 112 份独立硬件证据**。两个 KV 配对、一个 CIM 配对额外核对 packed 持久容量不变；高读带宽 HBF 核对固定放置与参数单调性。定向代码回归 **135/135 通过**，覆盖冷转换、packed backing 容量、投影格式、分层放置、存储报告、双 DRAM 争用与并行分片。同步修正旧测试中“packed 权重可直接位切片”的过期预期，保留 GPU/CPU 实际格式字节断言，改为验证未经显式转换必须拒绝；没有放宽运行时约束。差异格式检查通过。


## 公开标准与论文参考参数（2026-09-22）

**决定：不再把用户提供四组参数作为开展设计预测的前置条件。** 可由公开标准、器件资料、论文组织建立参考点，未知实现参数由明确的工程范围扫描；只有用户特定的面积/功耗/成本约束、既定拓扑和业务SLO需要用户修正。以下为研究输入，不是已启用的生产profile；本轮未改变仿真器算法、未重跑论文配置，也未获得目标硬件测量。

### 证据标签

- `SPEC`：公开标准或厂商明确给出的接口、容量、组织与限制；标准上限不是持续实测吞吐。
- `PAPER`：论文明确采用或报告的参数；再区分实芯片宏测量与论文系统仿真。
- `ASSUMPTION`：本项目设计点或扫描范围，不伪装成标准要求或目标器件规格。
- 整机预测仍为 `UNVALIDATED`。不能把其他设备/模型的实测误差、论文加速比或本机PC标定直接迁移过来。

### A. HBM与HBF：公开约束和选用范围

#### 已核实的参考点

1. **Micron HBM3E**（厂商公开页）：24GB/8-high、36GB/12-high，每stack带宽大于1.2TB/s。这里GB沿用厂商命名；可用GiB预算与实体stack容量分开，绝不把单stack带宽乘两次。
2. **OCP HBF High-Level Base Die Specification v0.7.0，2026-08-03**：
   - 表4给出host接口用户带宽档位0.384/1.536/3.072TB/s，对应8/16/32GT/s，最大stack高度分别8/16/16。
   - 表2的3072GB/s已经包含75%的AXI链路效率，不能再重复乘0.75，也不能当成NAND介质持续写带宽。
   - 表3提供16-die、512GiB、4KiB NAND页的组织示例。
   - 主机读按64B对齐，长度为64B至4KiB且不跨页；写入聚合为4KiB页。主机访问粒度和NAND物理访问粒度必须分开。
   - §5.9.2.1的MOCS是Base Die UCIe Channel能力，允许的命令深度含256/512/1024/2048/4096/8192/16384；实际取值由实现公布，不能把4096命令直接叫4096个NAND阵列并行读。
   - 同节默认每bank两个4KiB缓存。可选scratchpad、缓存命中、写合并、程序完成和共享读写bank等语义不能被一个带宽/延迟数替代。
   - 本文单位采用表2/4的十进制GB/s与TB/s。规范正文有“3.072TiB/s”的写法，与表内量纲不一致，未将其混用。
3. **FlashAccel，2026-07-11，表2/§7.1**：论文采用tR=4us、tProg=75us、4KB页、每die96 planes、8个Flash die/stack，报告192GB/stack、768GB/s读带宽。这是基于已有Flash参数构造的系统仿真，不是已经量产的OCP HBF实测卡。它是独立的论文复现配置，不与OCP 16-die/512GiB示例冒充同一器件。论文4KB/GB命名与物理2幂组织换算须在复现时显式统一。

#### 第一轮采用的输入范围

| 参数 | 参考/默认 | 扫描 | 标签与边界 |
|---|---|---|---|
| HBM实体容量 | 单stack厂商36GB档 | 24/36GB档 | SPEC；采用厂商容量映射，不能把可用预算当新品规格 |
| HBM可用预算 | 16GiB | 4/8/16/24/32GiB | ASSUMPTION；需小于实体容量，用于触发不同分层需求 |
| HBM带宽 | 以1200GB/s作保守峰值锚点，效率0.7 | 效率0.5/0.7/0.85 | 厂商为>1200GB/s；效率是ASSUMPTION，不是测量 |
| HBM首数据服务延迟 | 100ns | 50/100/200ns | PAPER量级+ASSUMPTION；不是JEDEC单条时序，也不含再次叠加的同一排队 |
| HBF标准组织参考 | 512GiB，grade2接口1536GB/s | grade2/3；grade1单列8-high配置 | SPEC参考；禁止把8-high限制与16-die容量组合 |
| HBF论文组织参考 | 原文8-die/96-planes、768GB/s | 先保持论文组织复现，再改组织 | PAPER；不得将论文结果标为OCP合规产品 |
| HBF页读取tR | 4us | 4/10/20us；1us仅乐观探索 | 4us为论文参数，其余ASSUMPTION；不是之前示例100ns |
| HBF页编程tProg | 75us | 75/150/300us | 75us为论文参数，其余ASSUMPTION；不把读速率复制给写 |
| HBF主机命令深度 | 1024/通道 | 256/1024/4096/16384 | OCP允许值中的设计选择；不能代替NAND介质并行数 |
| HBF实际读/写吞吐 | 分别由组织和服务路径计算 | 顺序/离散布局、只读/混合写 | 不直接赋值为host接口峰值 |

硬约束预算：`吞吐 <= min(host接口, TSV/NoC, 独立介质单元数×每次物理访问字节/tR或tProg)`，再考虑bank冲突、读写互斥/流水、缓存、GC与布局。命令数足够只是必要条件之一。新增有效带宽因子时必须明确其覆盖内容，不能重复扣除已经计入的协议损耗或排队。

**对原结论的修订：**4GB/s、100ns的旧HBF案例只保留为机制/压力测试，不再作为公开资料支持的HBF典型配置。“远端Flash”和“活动内存”是访问路径差异，不是把同一个物理器件改名后赋予两套无关物理能力。OCP §13.3还直接描述KV读写及权重/KV分通道，因此应在“状态留HBM”之外比较“KV部分进HBF、写合并/分通道”的方案，而非预先锁定KV一定不能进HBF。

### B. 两片DRAM垂直堆叠SoC

JEDEC JESD229-2官方公开范围覆盖8–32Gb/片、4或8个64-bit通道、1–4片memory与controller直接连接，适合作为接线与容量/带宽不必同步翻倍的参考。官方完整PDF需注册/登录，本轮未取得其全文；不把镜像里的tRCD/tRP细表当已核实官方数据。

原标准范围内两片最多约8GiB，本项目当前模型持久权重约13.565GiB，已经放不下。不能为了宣称场景可跑而隐去容量失败。用于容纳模型的2×8/16GiB应称为**定制3D DRAM设计假设**，不是原WideIO2现成器件。

DreamRAM（arXiv:2512.12106v1）提供按die/channel/bank/MAT、总线复用、TSV和面积约束生成3D DRAM设计的方法。其表I中61.1/64.2ns是模型行未命中延迟，实测该栏为NA；不能把它们叫本SoC实测。其跨层数设计空间极值也不证明恰好两层能实现同样容量/带宽。

首轮定制系统扫描（以下全部ASSUMPTION，不声称标准保证）：

- 物理DRAM片数固定2；每片容量4/8/16GiB，默认8GiB。2×4GiB作为容量不够的负例，不偷偷回退。
- 每片有效读写带宽32/64/128/256GB/s，默认64GB/s；高档需另经组织/引脚/面积检查，不能与最小面积和最大容量任意拼接。
- 单片无队列首数据服务延迟40/60/100ns，默认60ns。分别记录行状态与端到端链路/控制器队列，不把tRCD当总延迟。
- 独立路径：每片一条同速链路/owner；共享路径：两片链路映射同一个同速owner。固定单链路服务能力比较共享争用，聚合上限分别2B与B；CIM通信也必须占用其实际经过的共享NoC。
- SoC有效FP16矩阵算力先扫4/16/64TFLOP/s，默认16；这是未知SoC的设计变量，不借本机GPU数值声称实测。

### C. SRAM-CIM

ISSCC 2025 B-A-N2CMAC：28nm数字SRAM CIM，64kb（8KiB）宏；BF16输入/权重、FP32输出，BF16两种模式access time为5.4/6.8ns，面积0.136mm²。它不是64kB宏，也不能将TFLOPS/W直接当TFLOPS。正文描述双bit串行及近似对齐流程：输出为FP32不等于严格FP32逐步累加，BF16更不能自动替换现有FP16契约。

可核实来源为作者机构PDF的检索正文和芯片summary；该全文直接抓取超时，不将未取得的完整权重装载带宽补成实测值。这里只登记芯片宏数据，不声称27B数值等价或整个SoC已校准。

第一轮系统设计范围（ASSUMPTION）：

- 总CIM阵列0.5/2/8MiB，默认2MiB；独立暂存0.25/1/4MiB，默认1MiB。
- 权重装载有效带宽16/32/64GB/s，默认32；不是该论文测得的加载率。
- 无跨调用复用、跨8次、跨32次使用三个对照；多槽/双缓冲作为独立策略，不能免费启用重叠。
- 保留论文单宏的精度和计算周期定义；FP16、BF16近似宏分开，不换标签套吞吐。
- 宏数扩大时同时核查面积、供数带宽、输出归约和功率。8MiB不是8KiB宏的“免费1024倍性能”。

**当前执行边界：**现有冷转换要求全矩阵暂存，不能直接运行MB级目标。应先补分块转换、分块装载与复用，并进行数值/精度契约检查；旧512MiB阵列+512MiB暂存只保留为功能验证样例，不当片上SRAM推荐配置。

### D. 工作负载

JEDEC不规定聊天/长上下文业务长度。模型身份仍使用本项目GGUF主干、混合权重格式、FP16 KV和独立linear state，不借用论文其他模型的KV bytes/token。

首轮自主选择的可复现设计集（ASSUMPTION，不声称来自用户线上日志）：

| 项目 | 默认 | 范围 |
|---|---:|---|
| 输入tokens | 2048 | 128/2048/8192/32768 |
| 输出tokens | 128 | 32/128/512 |
| 同时请求数 | 4 | 1/4/16/32 |
| 到达模式 | 同时到达便于机制比较 | 再扫固定服务率基线的50%/80%/95%到达负载 |
| TPOT预算 | 同时报告50ms和100ms | 这是研究比较预算，不是用户已确认SLO |
| KV布局 | 当前真实FP16 KV + 独立linear state | HBM全驻留、HBF部分驻留、HBF分通道/写合并；未实现策略必须明确标记 |

先做单因素与少量配对，不把全部维度笛卡尔积一次跑完；容量不够的点保留失败，不能通过丢请求、截断上下文或暗改精度制造通过。

### 参数落地与验证顺序

1. 公开值保留版本、单位、per-die/per-stack/per-system范围；先复现完整论文配置，再做本项目设计变化，不拼接多个论文的最优值。
2. 将HBF主机64B读、4KiB介质页、缓存与写完成、命令队列和bank/plane并行拆开。先补真正影响KV/权重访问的机制，再跑标准参考配置。
3. 对CIM采用MB级容量目标，补分块和跨调用复用；阵列、暂存与后备权重分别计容量，数值契约另验。
4. 有效带宽、容量和单次访问用手算/独立模型交叉检查；协议符合性、对论文模型复现、一手器件校准三者分开报告。论文系统仿真之间一致不是实物精度证明。
5. 输出“在多组公开合理参数下稳定成立的策略”与“依赖乐观参数才成立的策略”，不强行得到HBF/CIM一定加速的结论。

### 一手来源

- Micron HBM3E：`https://www.micron.com/products/memory/hbm/hbm3e`
- OCP HBF v0.7.0：`https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf`，重点§4.2表2–4、§5.3–5.4、§5.9.2.1、§13.3。
- FlashAccel：`https://arxiv.org/html/2607.10186v1`，重点表2、§3.1、§7.1、§7.4；论文为系统仿真。
- JEDEC WideIO2官方范围：`https://www.jedec.org/standards-documents/docs/jesd229-2`，未读取注册后完整标准。
- DreamRAM：`https://arxiv.org/html/2512.12106v1`，重点§III、表I；模型latency不是该栏实测。
- B-A-N2CMAC作者机构稿：`https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_A%2028-nm%2017.83-62.84%20TFLOPSW%20Broadcast-Alignment%20Floating-Point%20CIM%20Macro%20with%20Non-2s-Complement%20MAC%20for%20CNNs%20and%20Transformers.pdf`，本轮使用可核实检索正文/芯片summary。


## 2026-09-22 增量实现：公开参考参数已进入可运行路径

本轮没有把公开资料的数字直接写成“已校准器件参数”，而是把其中可验证的**访问机制**接入仿真器：

- 新增 `src/heterollm_sim/hbf_media.py` 的 `cold_page_v1`：主机 64 B 事务、最大 4 KiB 主机请求、4 KiB 介质页、命令队列深度、物理并行 plane、页读/编程延迟、对齐/未知对齐、部分写 RMW 和写完成语义均单独记账。默认无缓存命中、无免费 early-ack、无跨请求持久页缓存。
- `TopologyRouter` 在 HBF contract opt-in 时，链路仍传主机可见字节，HBF endpoint 另计介质物理页流量；部分写的 RMW 读流量保留在 endpoint metadata。没有 contract 的旧 SSD/HBF 路径保持旧行为。
- 新增 `hbf_media_remote_weights`、`hbf_media_remote_flash` 场景。4 个短参考点位于 `artifacts/memory_tier_scenarios_20260922/public-reference-parameter-smoke-v1.json`，全部请求完成、内部检查和覆盖检查通过；硬件准确度仍为 `UNVALIDATED`。
- `HBMProfile`/`HostMemoryProfile` 新增可选 `read_bandwidth_gb_s`、`write_bandwidth_gb_s`。旧 `bandwidth_gb_s` 仍是回退值；纯读不再被慢写方向强行限速，shape override 按旧参考带宽比例作用到两个方向。
- 新增 `packed_to_fp16_tiled_cold`：`tile_m/tile_k/tile_n` 明确指定，`tile_k` 必须是 256 的倍数；每个 M/N/K tile 重新完成 packed 读取、解码、dense 阵列编程和 partial 累加，统计 tile count、packed payload/metadata、partial backing 流量、阵列峰值和 scratch 峰值。不支持暖驻留、M replication、免费双缓冲或数值等价声称。
- Qwen3.8-27B 的 tiled CIM 参考点使用 8×256×256 tile、2 MiB array、1 MiB scratch 的分析坐标；首层 MLP 的单 tile scratch 峰值约 174 KiB，完整调用的 tile 转换次数和累计流量仍可能很大。`dual_dram_cim_tiled` 与 shared 版本已实际跑通，结果也在上述 smoke artifact 中。

本轮定向回归：**264 项通过**（包括 HBF cold-page、方向带宽、tiled CIM、容量/生命周期、双 DRAM/共享 PHY 与旧转换路径）。公开参数只决定扫描坐标；绝对性能、热行为和 CIM 数值精度仍需独立目标硬件证据。
