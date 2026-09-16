# R22 CPU 图构建与 L1 边界 scope 审计

审计日期：2026-09-16。**固定 fa=off、kvu=true、b=ub=64 不意味着每次 llama_decode 都重建，也不意味着始终复用。每个 ubatch 都检查“上一张图能否复用”；只有复用失败才重建并重新 scheduler split/alloc。这些逐请求路径位于 L1 engine 的时间范围内；R22 long-graph 探针把 graph build、首次 split/alloc 和输入上传提前一次完成，正式测量不覆盖它们。** 因而不能把长图 probe 的 host wall/kernel 差额当成 LLM 图构建成本或统一 CPU 延迟常数。

本审计仅读取源码、build receipt 和 R21 freeze 的静态配置字段；未读取任何 prediction/errors 或目标 native 时延，未运行 native/LLM 重测，未修改 core 或已准备探针。

## 1. 锁定配置与来源

R21 pure/physical freeze 的131格静态配置核对一致：batch=ubatch=64、flash_attn=false；physical source-bound 配置的 kv_unified 全部为true。冻结环境 `LLAMA_GRAPH_REUSE_DISABLE=null`，而 class 默认 `graph_reuse_disable=false`（semantic `llama-context.h:379`）；[llama-context.cpp:291](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:291) 只在环境存在且 atoi非0时禁用复用。此处证明**允许进入复用判定**，没有证明实际复用命中率。`compiled_cuda_graphs=false` 与这里的 CPU CGraph 复用是两个层次，不能互相替代。

运行来源采用 annotation-control 的实际 context/server 源，而非把语义快照当作新编译运行。其 `evidence/build_receipt.json` 为complete，`input_sha256` 绑定 annotation context/server 源，并明确继承 semantic 的 `llama-graph.cpp.obj`（`c49d27a1c02ff5cdf58b1f70a46265d061ff7cbf52a61dbc6886533811e4f6bf`）。native-thread-control 的complete receipt保留 llama.dll、llama-server.exe、ggml-base.dll未变，分别为 `bd03c8dd9e5944f4b89ccd535f57f108f098c44814c2af64efe8f5f0ee32168f`、`9882ee2a07ff0649d9dfd22e5fd4ed2a796c92eae46c53acde8017bf0ed79dae`、`ebb357e640e217aef6a33e646b5d9a8e715635f72e478d85f2aa112b5362e1ca`。这是 overlay build 的既有证据链，不宣称重新 clean-build。

本次逐函数比较确认 semantic 与 annotation context 的 `process_ubatch`、`decode`、`sched_reserve` 函数体一致；annotation 的 phase NVTX 受 trace_annotations 开关控制，相关行号应以 annotation 源为准。

## 2. 重建与 reuse 的精确条件

- **循环单位是 ubatch，不是请求，也未必等于一次 llama_decode。** [llama-context.cpp:1754](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:1754) 先 sched_reserve、memory_update/init_batch，[llama-context.cpp:1841](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:1841) 在每个 ubatch 计完 n_outputs 后调用 process_ubatch；semantic `llama-context.cpp:1994` 以 `mctx->next()` 继续。b=ub=64 是容量约束，序列分组/内存上下文仍参与 ubatch 拆分，不能据此把执行次数固定成1。
- **复用入口：** [llama-context.cpp:1348](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:1348) 的 process_ubatch 先 `mctx->apply()`，取 `gf_res_prev`，生成新 gparams；[llama-context.cpp:1362](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:1362) 当 `!graph_reuse_disable && res->can_reuse(gparams)` 才复用。若 pipeline_parallel，还在 set_inputs 前同步，随后 n_reused++。不兼容分支在1374–1396执行 `res->reset()`、`ggml_backend_sched_reset()`、set_eval_callback、`model.build_graph(gparams)`、`ggml_backend_sched_alloc_graph()`。这是上一图复用，不是任意历史shape缓存。
- **参数资格：** [llama-graph.h:815](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.h:815) 检查 equal_seqs、n_tokens、n_seq_tokens、n_seqs、n_seqs_unq，以及 token/embd存在模式兼容；equal_seqs还要求旧ubatch持有data并且 seq_id_unq逐项相同。还检查 n_outputs、backend samplers映射及必要的每token output/seq绑定、nextn_layer_offset、embedding/causal设置、arch/gtype、cvec/loras/cross身份（815–884）。
- **输入资格：** [llama-graph.cpp:1406](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.cpp:1406) 在参数通过后对全部输入逐项 `can_reuse`；input embedding/token长度（85–91）、位置长度（149–154）、output IDs数量（226–231）等仍需满足。[llama-graph.cpp:489](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.cpp:489) 的普通attn KV要求 K索引长度与n_tokens一致，并检查KQ mask；[llama-graph.cpp:48](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.cpp:48) 要求mask的 `n_kv`、`n_tokens/n_stream`、1、n_stream四维一致，kvu=true令n_stream=1。Qwen35等hybrid还检查 recurrent state拷贝尺寸、head、rs_z（[llama-graph.cpp:1116](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.cpp:1116)），不能只按K/V宽度判断。
- **KV宽度为何不是每token变化一次：** [llama-kv-cache.cpp:1250](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1250) 用 `n_pad_cur=max(n_pad,256)`，取各流 `min(cache_cells,max(n_pad_cur,pad(used_max_p1,n_pad_cur)))` 的最大值；源码注释明确以padding保持图稳定、利于reuse。因此，同形状decode在同一256宽度区间内且其余条件不变时**可能**复用；跨宽度台阶、slot/sequence改变、输出数改变、尾部prefill不足64等都可能失败。池holes、inactive slots和真实high-water未被本次审计观测，不能由可见context推导精确命中次数。

**prefill/decode的具体推论（非测得命中率）：** 相邻64-token prefill块若所有复用条件都相同，可以复用；末块logits输出数/尺寸变化、KV宽度跨界可能触发重建。prefill到decode通常n_tokens/n_outputs变动，必须重新资格检查；decode的固定c1/c4/c8 cohort不保证参与序列、hybrid状态与KV高水位恒定。[llama-context.cpp:1408](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:1408) 的 `ubatch.n_tokens>1` 还会选择线程配置和phase标签；它不能把多请求decode的M>1变成语义prefill，也不能代替上述结构判断。

**reserve不等于每次设备重新分配：** [llama-context.cpp:594](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:594) 的sched_reserve在sched_need_reserve=false时立即返回。context构造期已经调用一次（[llama-context.cpp:473](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:473)），那部分通常在请求engine边界之外。配置/causal、sampler、LoRA等变动可重新置位（1199–1343）；若在decode内触发则成本落入engine。[llama-context.cpp:2430](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:2430) 的graph_reserve会reset旧图、build代表图并reserve/split。[ggml-backend.cpp:1591](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-backend.cpp:1591) 的alloc_splits先尝试已有gallocr分配；backend buffer类型改变或分配失败才可能同步并reserve_n，再重试。因此“每次重建必做新的cudaMalloc”同样不成立。

## 3. L1范围、scheduler split 与长图 probe 的差别

[server-context.cpp:3357](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:3357) 在首次prompt batch填充点 `update_prompt_start()` 后发request-begin。[server-common.h:377](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-common.h:377) 的现有 ggml_time_us 计数器是engine时间来源。普通采样路径在 [server-context.cpp:4103](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:4103) 才进入token-begin/sample/accept，4127更新gen计数；request-end在4145，response serialization和slot release之后工作不应追计到engine。

因此，请求开始到首token以及后续token间隔里的 `llama_decode` 包含CPU graph兼容检查、重建、split/alloc、set_inputs等工作。[server-context.cpp:3905](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:3905) 的compute marker包住 `queue_tasks.yield_to_queue` 内的llama_decode（3913）及有输出时的llama_synchronize（3915），3920结束；该marker只为符合普通采样slot条件发出，不保证每个中间prompt块都有独立compute marker。**token-begin/end包的是采样计数部分，不是整次decode；context内部phase NVTX从 [llama-context.cpp:2506](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/src/llama-context.cpp:2506) 才开始，已晚于process_ubatch里的build/alloc/set_inputs。** 冻结 `LLAMA_TRACE_ANNOTATIONS=0` 时这些annotation可能不发出，边界的源码位置和现有engine计数仍有效；本报告没有虚构已捕获的marker。

[ggml-backend.cpp:1989](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-backend.cpp:1989) 的sched_alloc_graph实际调用split_graph和alloc_splits；[ggml-backend.cpp:2014](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-backend.cpp:2014) 的sched_graph_compute_async仅在未分配时再做这些准备，否则直接compute_splits。后者从 [ggml-backend.cpp:1643](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-backend.cpp:1643) 按实际 `sched->n_splits` 迭代，每split到1796/1818才调用backend graph_compute_async。split_graph有多轮backend/placement判定（1066起），split数由实际图/支持算子/内存/放置决定。**faoff、kvu、b=ub=64和“单GPU”均不构成LLM split=1证据；本审计没有证明split=1。** 就连CPU-only/单backend合成split数量也应通过计数记录，而不是预填。

R22 [long_graph_main.cpp:412](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_022/long_graph_probe/long_graph_main.cpp:412) 在任何记录call之前完成GGML元数据context、tensor和SCALE链构造，432完成build_forward_expand，438–443分配持久buffer并scheduler alloc_graph，451上传输入，461才运行buffered的1+5+30。故即便first-call也不包含这些setup QPC区间。正式36call研究的是**已分配图上的scheduler dispatch/原生kernel/API及graph-end sync**；不含model.build_graph、每ubatch reset/split/alloc、真实LLM set_inputs/mask生成、reuse判定、模型多backend放置或LLM融合拓扑。setup有单次QPC记录，也不能冒充独立、稳定且可迁移的CPU建图微基准。

## 4. planner host_prefix 已付费与尚未证实覆盖的内容

| 现有路径 | 已表达的成本/因果 | 尚未证明 |
|---|---|---|
| [planner.py:6659](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:6659) `_add_host_orchestration` | capacity会计、batch/operator schedule指令、batch/request/admission/token decode准备、payload pack的CPU/cache/DRAM、IOMMU、DMA队列及H2D | 没有以实际GGML tensor/node/edge、mask shape、reuse miss、gallocr或backend split数驱动的重建计数 |
| [planner.py:7048](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:7048) `_add_physical_invocation_frontend` | command_build、driver_submit、GPU command processor各自cost owner；普通target invocation归host_prefix（[planner.py:23852](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:23852)） | command_build虽标`graph_and_command_build`，仍只是按physical invocation数量的聚合近似，不是已验证的model.build_graph/split/alloc CPU服务 |
| [cost_models.py:1344](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/cost_models.py:1344) HostOrchestrationProfile | 默认command build指令为128+12×invocation_count；schedule默认192+48×request_count+8×token_count（1438–1455），指令服务由单serial控制线程issue rate换算 | 不随N_tensors/N_edges/N_views/reuse predicate/实际allocation路径变化；不能凭名称称建图完整覆盖。默认值不等于本次有目标拟合证据 |
| [planner.py:22407](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:22407) compact_host_prefix_stage | 保留live DAG里已有CPU/cache/DRAM/IOMMU/DMA/PCIe/CP资源需求的因果envelope | 只打包既有任务，不会自动补入缺失机制 |
| [cost_models.py:2703](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/cost_models.py:2703) 等GPU kernel_launch demand | 原有kernel launch已经计费；frontend也明确排除重复的driver/CP/operator-launch owner | 不能把“所有launch未计费”当作新图构建修复的依据 |

planner 7220–7294另有必须显式metadata/identity/coverage启用的phase-boundary与first-decode residual hook；这不是通用图构建模型，也不能在本轮无校准的开发评估中默认为已补齐。

缺口应描述为“**通用command-build已收费，但缺少来源绑定、结构计数和复用状态控制的GGML建图/准备模型**”，而不是“host_prefix为零”。未来新成本应替换或拆解其重叠的command-build份额；capacity/pack、输入数据生成、driver-submit、CP和kernel launch须维持独立owner，避免double count。

## 5. 允许的独立验证方案（仅建议，未执行）

1. **CPU-only元数据建图微基准：** 不读LLM目标时延、不加载目标权重、不调用llama_decode/graph_compute。用锁定GGML API构造参数化链/扇入/视图/多输出DAG，预注册N_tensors、N_nodes、N_edges、N_views、metadata bytes；分开测context/reset、node/tensor构造、build_forward_expand、compatibility检查。图形状跨64/256及独立中间/外推规模，检查拓扑/别名/输出标志，记录线程与QPC；CPU数据准备开销单独记账。
2. **CPU scheduler split/alloc微基准：** CPU backend或明确标为mock的多backend supports_op/buffer策略，仅调用split_graph/alloc_graph与释放，不运行算子。记录实际n_splits、graph copy node/leaf数、backend指派探测、copy edges、gallocr复用/重分配、分配元数据字节；分开已有arena与必须扩容两路径。CPU/mock结果只证明算法与CPU元数据服务；不能外推成实际CUDA分配/同步成本或LLM split=1。若以后研究真实GPU分配，必须另立受控实验和同步边界。
3. **条件计数模型：** 每ubatch先用上述源码条件形成 reuse predicate；`Tprepare = Tcompat + Tset_inputs + I(reuse_miss)×(Treset+Tbuild+Tsplit+Talloc)`，再按实际reserve事件加独立项。候选结构特征是tensor/node/edge/view计数、metadata字节、backend数/实际split数、gallocr重分配标志，而非模型名称/目标误差/LLM wall residual。服务率仅可由独立CPU微基准训练，在保留的合成形状上验证。set_inputs的mask元素写入与H2D不能再重复计入payload pack。
4. **未知必须保留：** 未知KV池high-water、hybrid状态、真实reuse命中与fusion/split ledger时，给conditional/unknown与可证明上下界，不填0，不将长图wall−kernel拟合成固定延迟，不用目标LLM总时延选择系数。

本报告新增系数0；未安排或执行上述微基准，R22已准备测量包保持不动。

## 本次读取的源码哈希

| 文件 | SHA-256 |
|---|---|
| `source/llama.cpp-annotation-control/src/llama-context.cpp` | `e677c1e6e56fc08561fa56d9861501740405190e578d690e9efcad26fa48622e` |
| `source/llama.cpp-semantic/src/llama-graph.cpp` | `a6a8241c2d149961801d0fdeaa68f1bb176297b4d5156b6138db0b3d169abb04` |
| `source/llama.cpp-semantic/src/llama-graph.h` | `bfc769f1902da5e5f005a6ab6779ba7aaccab9ec942f0f06b1d17072d58be0ad` |
| `source/llama.cpp-semantic/src/llama-kv-cache.cpp` | `e4d2aa977c8aa048ddd79c878683e94909f5f24ac109b0fcd1c0b59ef70f3899` |
| `source/llama.cpp-semantic/ggml/src/ggml-backend.cpp` | `803298af097545b4b611df53a2e9ac39f43c18c095c0d3b4c879d2ae911a2d79` |
| `source/llama.cpp-annotation-control/tools/server/server-context.cpp` | `99f7aead4dd6b190292db2a14b2586d4076871a3710a49a94c71424b6f05501e` |
| `source/llama.cpp-semantic/tools/server/server-common.h` | `063b32a7c0b4927e8325b553e837605b991d5f752e34cb726b2683e8768d47c0` |
| `src/heterollm_sim/planner.py` | `15ecc454075febf632fd4cbd9303ee6f7fa0293aa2e8b413670b0aba03b8b2f6` |
| `src/heterollm_sim/cost_models.py` | `a591b36370930d30f6e2b4b2b166ad9c5e4075b0facc063db3fbc79ccf0a9ae5` |
| `artifacts/development/native_long_grid_135_20260915/optimization_loop/round_022/long_graph_probe/long_graph_main.cpp` | `4858c8dd2117c6b33670e07b5dbeddcd476d1287416941a6d08d730c436e759e` |

静态配置来源：R21 pure/physical 的 freeze.json（仅静态字段），以及 annotation-control/native-thread-control 的 build_receipt.json。没有打开目标评分文件。
