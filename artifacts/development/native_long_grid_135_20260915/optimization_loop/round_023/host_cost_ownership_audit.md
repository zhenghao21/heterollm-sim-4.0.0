# R23 host 成本归属审计

日期：2026-09-16。结论：**生产路径已有 host_prefix、CPU command build、driver submit、GPU command processor 和逐 kernel 的 1000 ns frontend；不能再把“host submission”整体补一遍。** 明确缺口是 CPU 图复用/重建的条件与工作量没有单独表达，以及同步完成端的 CPU 服务显式为零；这不等于可直接增加一个未知常数。现有命令构建汇总与 kernel launch 都须先划清成本归属，再替换或细分。

范围：仅静态读取源码及 R22 scope 审计；未运行 native、完整预测或 cohort 重编译，未读取目标 native 时延/预测误差。逐文件 SHA256 比较确认主区 planner.py、cost_models.py、reference.py、serving.py、predict_stable_native_dataset.py、native_llama_compare.py 与 R23 current/source 和 retained/source 均相同。只新增本文件。

## 1. 实际调用与来源

`tools/predict_stable_native_dataset.py:2306–2342` 构造 SamplingPolicy → `grid.build_matching_scenario` → 五项静态 contract → GPU clock/MMVQ binding → `replan_final_static_scenario` → `reporting.run_scenario(... aggregate)`。replan (`:1938`) 调 `apply_llama_runtime_config(... materialize_placement=True)`，是仿真放置规则刷新，Python 花掉的真实时间并非被测 CPU graph build 的入账。

`tools/native_grid_predict.py:32` 导入 `native_llama_compare.build_matching_scenario`；后者 `:1121` 从 `build_reference_scenario()` 建场景，`:1316` 的 replace 保留 host profile。生产 worker 没有调用 native calibration 加载/应用，也没有读取 `host_runtime_microbench_v1.json`。reference (`src/heterollm_sim/reference.py:557`) 明写固定值；GPU launch 同文件 `:421` 为 1000 ns、`:428` 指向 gpu0.frontend。它们是示例/分析 profile 的继承值，**不是该格 native residual，也不是通用 host microbench 的测量结果**。

通用 `tools/host_runtime_microbench.py:1–6` 仅覆盖 tokenizer、loopback SSE、JSON。那份证据不证明 CUDA submit、CGraph build 或 sync，更不能接进本轮 Engine 边界。

连续服务实际链为 `reporting.py:144` → control-plane bootstrap → continuous `TopologyAwareBatchCostProvider` → `planner.py:21849` `_estimate_serving_cohort_cost` → `_lower_serving_cohort` (`:23632`) → `_add_host_orchestration` (`:23831`) → `_add_physical_invocation_frontend` (`:23915`) → 每个 `_compile_or_replay_serving_invocation` → `_compile_parallel_iteration` / lm_head / logits CPU tail (`:20575–20616`)。这里的 compile/replay 是仿真内部计划复用，不是 native CUDA Graph replay。

## 2. 收费次数和归属

记每次 realized cohort 的请求项数 R、token rows T、物理 ubatch invocation group 数 G、实际被下沉的 GPU kernel 数 K；CPU 单控制线程速率 W=min(decode,issue,retire width)×CPU GHz。R/T/G/K 不是一个计数。

| 成本 | 实际收费范围 | 明确来源/边界 |
|---|---|---|
| capacity + operator schedule | 每 cohort `(96+64R)/W`、`(192+48R+8T)/W`，另有 control cache demand | planner:6717–6789；不是逐层/逐kernel |
| host prepare | 每 cohort `350+180R` ns；admission/input_decode 默认零 | planner:6691–6702、6795–6826；request_parse 在每 cohort 按项收，并非只在首请求收 |
| payload pack | payload=`96R+4T` bytes，读写的CPU memory模型；pack floor=`12T` ns | planner:6829–6905；同phase资源取max，不能把pack与CPU demand再简单相加 |
| IOMMU/DMA/H2D | 每 cohort、按payload页/queue wave/物理路径收费 | planner:6907–7046；已在host_prefix内，DMA descriptor 与bulk transfer分开 |
| CPU invocation command build | 每 cohort一次聚合 `(128+12G)/W` | planner:7100–7142；metadata称graph_and_command_build，明确排除driver/CP/kernel_launch |
| driver submit | 每 cohort聚合 `250G` ns，gpu0.command_queue | planner:7144–7174；一physical invocation group一次，不是每layer一次 |
| GPU command processor | 每 cohort `ceil(G/16)×5000` ns | planner:7178–7221；schema_v4.py:159–165默认profile，经config.py:327/ schema_v4.py:288继承；与frontend不同resource |
| GPU kernel launch | 每被下沉的launch phase 1000 ns，gpu0.frontend；总服务1000K | cost_models.py:2743–2760、2940–2945、2970–2972；在相应compute前单独phase，不是HBM demand内的一部分 |
| logits D2H + completion | 有logits的group按实际rows搬运；equal-length stateful ubatch在cohort尾合并；completion CPU服务0 | planner:19020、19220–19248、20607、23966；等GPU的经过时间已由依赖承担 |
| sampling + token commit | 每logit row处理V个词；candidate读4V写12V bytes，串行rows；top-k扫描条件成本；每committed row写4 bytes | planner:19268–19535；不按prompt每token或layer采样；top-p/min-p/temperature/RNG/accept仍partial |

G由 `planner.py:20080` 的lane/状态批处理能力和physical_ubatch_rows分组决定：普通dense可合批，stateful按能力隔离或equal-length合批，超过ubatch行数继续切。decode的T通常为活跃请求数，prefill的T是本轮实际prompt块行数。故不能统一写成“每token G=1”或把250 ns乘层数。K也只覆盖当前physical projection/conversion已下沉的kernel，`native_dispatch_proven=false`时不可宣称是native全部API调用精确计数。

## 3. 看似没算、其实已算的项目

`_compact_host_prefix_stage` (`planner.py:22470`) 明确仅作为因果包络；其 `execution_tasks` 保留CPU/cache/DRAM/IOMMU/DMA/PCIe/CP的原始resource demands (`:22440–22455`)。**只查看主GEMM/HBM行会遗漏host_prefix和独立kernel_launch的展示，但不会从执行DAG删掉它们。** stage service、task demand总和、critical path elapsed三者不能混为一个可累加总量。

R22 `mmvq_application_diagnostic.json:302–307` 已确认conversion有自己的前驱kernel/launch，GEMM也保留launch；不能再给conversion补一次1000 ns，也不能将所有算子launch隐藏进新的host总额。

历史校准旁路真实存在：`planner.py:7226–7302` 仅当`native_calibration_apply_phase_boundary=true`且model/hardware/runtime/coverage均匹配，才加入native CUDA API aggregate及first-decode residual。当前worker未启用它。未来启用时其`aggregate_launch_plus_sync_once_per_phase_invocation`与已有launch的物理边界必须重新审查，不能凭不同resource ID就声称无重叠。

## 4. 有源码支持的问题与最小接线

1. **图构建的条件未表达，而不是完全没有build费用。** native `source/llama.cpp-annotation-control/src/llama-context.cpp:1359–1403` 每ubatch检查can_reuse；命中跳过build/sched_alloc，失败才reset/build/alloc，随后两者都set_inputs。当前汇总`128+12G`不含node数量、reuse hit/miss或split/alloc事实，无法区分两条路径。最小方案应把现有command_build owner拆成明确的复用检查/set_inputs/命令装配及条件rebuild owner，并从原汇总中移除对应部分。先接source-bound事件计数/条件，未知时保留partial；禁止无证据每decode重建，禁止额外叠加原汇总覆盖的工作。
2. **无CUDA Graph的host逐kernel调用缺乏独立CPU占用/流水线表示；总时延是否缺少不能由此判定。** 固定DLL图宏关闭，native `ggml-cuda.cu:4404` 仍逐有效node调用compute_forward，kernel helper实际提交可能多于一node一launch。当前driver的250G只是ubatch group聚合；但1000K的frontend已经给每kernel一个顺序launch服务。最小方案是给既有launch标注host API/device frontend的明确owner、source计数与非重叠定义，再在证据具备时**重分配/替换**1000 ns的组成及resource，保留与GPU执行可重叠的host提交依赖。不能据250G很小而另加K×host_launch常数。
3. **同步等待已算，完成通知CPU服务明确未算。** `completion_interrupt_service_ns=0`、`host_wait_service_ns=0`、`wait_accounting=dependency_elapsed_time`是源码显式声明。应只在有独立证据时给完成/唤醒/检查的CPU服务定价，不得将cudaStreamSynchronize wall time整体加上，因为GPU等待区间会重复。此处也不是每layer/kernel无条件同步；native有具体调用点和条件。
4. **采样不能整体补一遍。** candidate materialization/top-k已入CPU tail，余下链条partial。只对明确尚未覆盖的过滤/RNG/accept操作建立独立source-bound成本，并保留已有logits读取，避免第二次host-memory传输。

一级指标仍为Engine TTFT/TPOT/E2E。`serving.py:10485–10509` 在首prompt处理资格确定后、准备/分配前记录engine_start；后续cohort host工作会进入Engine时间，结束用tokens_committed（`native_llama_compare.py:624`附近）。admission/tokenizer/JSON/SSE属于另一个边界；当前三个扩展字段默认零，应保持不以HTTP微基准补Engine差额。`request_parse_ns=180`的历史命名/每cohort收费值得改成清晰的engine cohort setup归属，但缺少来源时不能武断删掉或认作HTTP真实解析。
## 5. 下个候选：只校正建图复用语义的最小设计（追加只读审计）

**不能把现有 `128+12G` 在 reuse 命中时关掉。** `HostOrchestrationProfile.command_build_instruction_count` 只有 invocation_count 参数；`planner.py:7100–7142` 把它称为 `cpu_invocation_command_build` / `host_cpu_command_build`，唯有instruction_class用了含糊的 `graph_and_command_build`。没有证据把这128条固定指令或12G条指令识别为 `model.build_graph`、scheduler split 或 gallocr allocation。因此当前可证实的最小修正是**纠正归属与新增独立的条件事件定义**，不是通过reuse删除已有时间；这项修正本身不保证TTFT/TPOT下降。

### 5.1 必须复刻的锁定源码条件

来源继续采用annotation-control的context、其receipt继承的semantic graph/backend对象；不是把另一版上游代码当已运行事实。R22图scope审计已记录继承链和源码hash。

- `llama-context.cpp:1348–1417`：每个`process_ubatch`先`mctx->apply()`，取**当前context唯一的**`gf_res_prev`，生成gparams。只有`!graph_reuse_disable && res->can_reuse(gparams)`才命中；pipeline_parallel时命中路径仍同步后再更新输入。miss执行`res->reset()`、`ggml_backend_sched_reset`、`model.build_graph`、`ggml_backend_sched_alloc_graph`。两个分支最后都`res->set_inputs(&ubatch)`并`graph_compute`。禁用环境在`:291–295`解析；不能把compiled_cuda_graphs=false变成CPU复用禁用。
- `llama-graph.h:815–883` `allow_reuse`：比较equal_seqs、n_tokens、n_seq_tokens、n_seqs、n_seqs_unq及token/embd输入模式的源码布尔式；equal_seqs还要求旧ubatch持有data、逐序比较seq_id_unq。另比较n_outputs、**backend graph sampler map的对象身份**；map非空还校验两边ubatch.data、每token output标志和首seq_id。其余包括nextn_layer_offset、embeddings/embeddings_nextn/embeddings_nextn_masked、causal_attn、arch/gtype，以及cvec/loras/cross对象身份。不能只用(batch,ubatch,phase)作为复用key；CPU SamplingPolicy本身也不能代替该sampler map。
- `llama-graph.cpp:1406–1437`：参数比较通过仍逐个`input->can_reuse`。普通KV输入 `:489–501` 校验k-index长度并更新mctx绑定；`:48–64` 校验mask四维`[n_kv,n_tokens/n_stream,1,n_stream]`，其中`n_stream=kv_unified?1:n_seqs_unq`。故固定kvu=true、b=ub=64仍可能因真实物理n_kv变化而miss；不能用每步context_tokens递增直接判miss，也不能用KV容量始终相同直接判hit。
- recurrent `llama-graph.cpp:345–361` / hybrid `:1116–1137` 还比较n_rs、n_seqs、head、rs_z，以及相关tensor形状；token/embedding、pos、out_ids的输入检查分别在`:85`、`:149`、`:226`。输入类型未覆盖时结果应是**unknown**，不能乐观命中，也不能伪称source-proven miss。
- `ggml-backend.cpp:1989–2005` 的alloc_graph才调用split_graph、alloc_splits并设is_alloc；`:2014–2027` compute_async只有is_alloc=false才走alloc_graph，否则直接compute_splits。CPU图复用不免掉compute_splits、必要copy/更新、backend dispatch和kernel执行。

### 5.2 生命周期与最小状态模型

使用**每native context一个上一张执行图状态**，在真实physical ubatch执行次序推进；不能按request、layer各自缓存，也不能因“见过某shape”就从多图缓存恢复。`llama-context.h:367–368` 区分gf_res_prev和gf_res_reserve；前者不是每请求新建。

`llama-context.cpp:594–618` 的sched_reserve会同步、重建两个graph result及scheduler；`:2440–2461` reserve/split路径reset scheduler并显式reset gf_res_prev，另在gf_res_reserve上build。`:831–838` 内存更新路径也reset gf_res_prev。`llama-graph.cpp:1323–1343` reset清空params/inputs；`:1447–1449`、`:1497` build期间保存params。因此需要有`previous_graph_valid`、已建graph参数/输入shape签名、scheduler allocation generation，以及明确的reserve/reset invalidation事件。

初始有效性不能由R23 retained-KV的“旧KV存在”推断：KV状态与gf_res_prev是两套状态。若未覆盖warmup→清理→首请求的图生命周期，初始graph状态保持unknown；不要偷设“冷启动必miss”或“warmup必hit”。相关本轮Engine外reserve不得计入首请求Engine，Engine内触发的reserve才属于本次路径。

建议状态输出为`reuse_hit / rebuild_required / unknown`及reason、source refs。只有所有适用predicate与前序生命周期都有证据才出hit；禁用或已知invalidate/形状不兼容可出required；缺字段出unknown。该状态属于运行时实例，不得写入跨cohort纯shape成本缓存，使前一个请求或ubatch的状态丢失。构图计数应按每个physical ubatch，不能仅一次cohort聚合G而抹掉组间前后关系。

### 5.3 新定义与原owner保留清单

| 项 | 最小语义处理 |
|---|---|
| 现有128+12G command build | 保留原数值/收费次数；将含糊instruction_class注释明确为通用invocation command编码假设，记录`native_cgraph_build_coverage=unproven`。这是归属澄清，不应宣称源码证明了该数值 |
| 独立`native_cgraph_reuse_check` | 每physical ubatch做判定，记录实际适用predicate/check工作。unknown时明确未定价，不能把missing服务视为测得0 |
| 独立`native_cgraph_rebuild` | 仅rebuild_required记录reset/build的结构工作；按实际tensor/node/edge/view等source ledger驱动。reuse_hit的该事件计数为0；unknown保留unknown，不造hit率 |
| 独立`native_sched_split_alloc` | 绑定实际scheduler is_alloc/reset及同次alloc路径，区分split、已有arena复用、扩容；不得把alloc_graph调用等同真实malloc/全buffer重分配 |
| 独立`native_graph_input_update` | hit和miss都存在：`set_inputs`每个input执行，指针/mctx更新、token/pos/out_ids/KV indices/mask更新。只能计当前pack/transfer/输入准备未覆盖的部分；不能再完整叠加mask写入和H2D |
| 原capacity/schedule/prepare/pack/IOMMU/DMA/H2D | 继续按原cohort/payload收取，graph reuse不能证明这些工作消失；已有payload抽象不等于已完整覆盖native全部set_inputs |
| driver submit / GPU CP / 1000K frontend | graph宏关闭时都保留原计费；CPU CGraph命中不消除逐kernel提交。独立host CPU launch拆分仍须先排除与1000K的重叠 |
| logits同步/D2H/sampling/commit | 保留；与CGraph构造复用无免除关系。pipeline_parallel命中同步也只能计不重叠CPU服务，GPU等待仍由依赖承担 |

**定价门槛：** source能证明事件和次数，不会自动证明ns/指令系数。先作零新增系数的事件/归属候选；新事件`unpriced`不能被汇总解释为成本不存在。若将来需要让预测数值改变，须独立证明build/split/alloc服务及原command-build的重叠份额；有证明才替换/扣除该份额，并保留通用编码余额。当前不能把128或12G任意拆出一个“图构建比例”，也不能把新build成本直接叠加后声称无重复。

### 5.4 只针对语义的后续验收设计（本次未执行）

使用源码谓词的静态fixture/轻量单元断言，不运行LLM预测：同shape、稳定n_kv/seq且有效previous应hit；尾ubatch/n_outputs/seq集合顺序/n_kv跨维度/recurrent head变化各自产生对应miss；token/pos**值**变而shape不变时仍要执行input updates，不能仅因数值变化强判miss；A→B→A必须与上一张B比较，不能命中历史A；scheduler reserve/memory reset使旧graph失效；未知warmup图状态和未覆盖input类必须unknown。对已存在所有成本owner的计数及服务做前后恒等断言，确保“语义候选”不暗中删除128+12G、250G或1000K。

这一候选的可交付结论是正确区分“每次输入更新/提交”与“条件CGraph重建/split/alloc”，并显式暴露未定价工作。它不是以native误差选择省略哪些成本的优化。本追加未修改任何核心、司机、冻结输入或任务书，未测native、未读目标时延。