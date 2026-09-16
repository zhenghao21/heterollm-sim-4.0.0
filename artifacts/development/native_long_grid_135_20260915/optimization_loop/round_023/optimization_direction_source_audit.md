# R23 优化方向独立源码审计

日期：2026-09-16。仅只读源码、R23 current/retained freeze静态字段及已有scope审计；未读取R23 full预测/时延/误差。只新增本文件。审计委派列出了重点主题，但未附用户原始12项逐字文本，因此下表按主题逐项反馈，**不冒充原清单编号对应**；可供主代理准确映射1–8、12。

先给结论：可以继续做机制优化，但不能把131格当作qwen25、把所有尾ubatch当不满64、把并发c直接当所有GEMM的M、把已有nonflash/sampling/launch都当空白。明确可行动的源码问题包括末层行选择实现未被生产GGUF构造接线、MMVQ带宽沿用MMA波次代理，以及graph-build命名与实际收费归属不清；收益大小不由本审计推断。

## 逐项判定与最小证据

| 主题 | 判定 | 源码/冻结依据与最小下一步 |
|---|---|---|
| 样本适用范围、模型泛化 | **需纠正：131非仅qwen25。** | 只聚合`current/freeze.json.cells[].model_key`：qwen25=17、smollm2=23、tinyllama=22、qwen35=22、qwen38=20、qwen38_gpu=27。gpu_layers分别-1=84、0=20、66=27；CPU-only weight组不应套全GPU launch/MMVQ公式。下一步按architecture/physical tensor format/backend分层列适用域，不按模型名或全局误差调倍数。 |
| prefill chunk、尾块、ubatch | **分块方向可用，“P128/512/1536天然尾块不足64”错误。** | 131格batch=ubatch=64；prompt_token_ids长度逐格等于expected_prompt_tokens，128有40格、512有49格、1536有42格；全都整除64，单请求普通连续prefill分别2/8/24个整块。`planner.py:20165–20221`按physical_ubatch_rows切；`:20228–20260` stateful equal-length重组可改变组shape，混合decode/prefill和剩余budget也会产生实际小组。最小证据是source-bound realized group lane/phase/token rows表，不能用P%64制造尾部惩罚。 |
| 并发c、GEMM M与MMVQ/MMQ分派 | **可用，但须按真实physical invocation及量化type。** | c=1/2/4是请求并发；decode全活跃普通dense可M=c，prefill通常M=64；最后lm_head M是logit rows；mixed/stateful组另见`planner.py:20080–20300`。锁定annotation `ggml-cuda.cu:1881–1913`依次MMVF/MMF/MMVQ/MMQ/BLAS；semantic `mmvq.cu:318–345`对Blackwell：Q2_K/Q3_K/Q4_K阈值5、Q5_K6、Q6_K7、其它量化一般8。`mmq.cu:266–323`还要求type支持、非FORCE_CUBLAS、每block shared≥48KiB，Turing MMA可用后通过。仿真`mmq_work.py:12–20`和`planner.py:9183–9205`已用Q4_K5/Q5_K6/Q6_K7/IQ等8。c从1到4**不会因这些阈值切到MMQ**，但kernel多列几何可变。最小下一步核实物理weight type/activation layout/M/K/N、融合与convert ledger；native_dispatch_proven=false时不得说已观测native kernel身份。 |
| MMVQ自身性能、内存/计算并行 | **有明确语义错配，可采用；不能直接提高带宽。** | 已有`mmvq_memory_geometry_audit.md`：MMVQ CTA/warp/K-loop路径仍用MMA输出tile wave proxy调HBM。现有issue-bound是实际资源替换但可能被HBM resource-max遮蔽，R22 diagnostic已说明。最小证据是独立MMVQ geometry→HBM服务限定，区分unique bytes/issued bytes/L2、resident CTA、访存合并，不能由CTA数直接认定饱和率。保留legacy值时必须标fallback；无证据不引入按c拟合带宽。 |
| nonflash prefill计算/物理KV维度 | **已有核心路径，细化覆盖可用；不是只算triangular logical FLOP。** | `planner.py:19676–19759` source-bound nonflash视图应用于普通materialized prefill/decode；`:20537–20584`把physical_k送入iteration；`:14344`仅全attention层替换context；`:15524`起非flash分支显式QK→softmax→PV。`tools/predict_stable_native_dataset.py:800–813`绑定mask=`F32[n_kv,ubatch.n_tokens,1,1]`和256 padding。`serving.py:13511`的可选prefill scan flag未启用，不等于source nonflash合同没生效：两者是不同入口。最小证据逐group核对physical_k、score rows/heads、QK/PV实际kernel布局以及mask生成/上传是否有未覆盖CPU工作；不能再整体添加一遍attention FLOP/bytes。 |
| sampling / CPU output tail | **已覆盖一部分，余项只可精确补缺。** | 全131 static sampling_binding存在，示例是`llama_cpp_cpu_chain`, temperature0、top_k1；不是无采样。`planner.py:19268–19348`每logit row读4V写12V candidate，`:19351–19452` top_k≤128时扫描V−K个logit字段、串行rows；`:19503` commits×token_bytes写回。logits D2H/完成依赖在`:19020–19248`。heap root replacement/repair/sort、其它filters/RNG/accept和completion CPU服务仍partial；K1不等于可跳过candidate/mandatory scan，也不等于generic greedy argmax shortcut。最小证据审计锁定temperature0实际sampler chain，先删选不执行分支而非对全部候选添统一采样固定费。 |
| output row选择、最后FFN/final norm/lm_head | **lm_head/logits选行已覆盖；末层更早选行有现成实现但生产接线缺失，优先核实。** | `planner.py:19988–20017`requires_logits按lane，`:20595–20616` lm_head/sample只跑logit_token_batch。更精细的`_final_output_selection`在`:14089`必须读model.metadata的`llama_cpp_final_layer_output_selection`，缺失立即None。`final_layer_output_selection.py:8–13,17–27`有qwen2/llama before_last_ffn、qwen35 after_final_norm合同；但`gguf_parity.py:464–471`metadata生产列表没有该字段，全src/tools非测试检索只有定义/消费者，无调用source_declaration的生产者。当前冻结源码也同样缺接线。锁定native qwen2.cpp:106–119、llama.cpp:174–189先GET_ROWS再最后FFN；qwen35.cpp:211–212则在最终norm后取行。最小下一步是静态证明普通completion的输出标志/embeddings/MTP排除与build对象身份，并做小shape成本图fixture验证后从GGUF builder的受控合同入口接线；不能全层M改成logit rows，也不能qwen35照搬qwen2最后FFN裁剪。 |
| graph reuse / host launch固定费随图与c变化 | **区分计数与单次系数；无证据按c缩费不采用。** | 已有`host_cost_ownership_audit.md`第2/5节。每cohort command build=(128+12G)/W、submit250G、CP ceil(G/16)×5000；每下沉kernel独立launch1000ns。CPU CGraph复用允许skip真实build/alloc，固定DLL CUDA Graph编译关闭，不能省去全部launch或当graph replay。1000K总额可随真实kernel数量/融合/输出行/后端变化，时间线可随resource占用重叠变化；但1000ns单次服务没有证据随c自动折扣。最小证据先对齐owner与source kernel调用计数，再拆host API与device frontend，禁止250G和1000K之外再完整补host launch。 |
| retained全局KV物理视图 | **可用且R23正在条件实现；不等于(P+t)c，也不是精确地址。** | native `llama-kv-cache.cpp:1250–1263`用各物理stream的used_max_p1、padding256、capacity截断，非logical context简单求和。`retained_kv_state.py:144–185`只在有singleton-owner证据时维护slots：begin_prompt清该slot旧rows，未接新请求的slot仍保留warmup rows，finish保留P+O−1 materialized rows；`serving.py:13501`读取全slot占用，`planner.py:19745–19757`按256上取整形成下界。空洞、实际地址高水位仍未知，shared prefix无singleton证据不能sum。retained freeze状态：qwen25/smollm2/tinyllama共62格conditional，qwen35/qwen38/qwen38_gpu共69格uncovered；不允许把这69格也宣传修好。最小证据是准确warmup→prompt清理→组执行→finish生命周期和allocator高水位/holes，不能单纯在每个请求长度上乘c。 |

## 优先级：按已证实缺口和成本归属，而非未揭盲误差

1. **先核实并准备末层output-selection接线。** 有明确native branch和已存在但未绑定的实现，scope有限、可静态/小fixture验证；保留architecture差异。它通常减少prefill最后FFN工作，不能承诺全模型/TPOT收益。
2. **完成本轮retained状态原定评估；机制侧补allocator/生命周期证据。** 不读full结果预选修改，不变更正在运行的冻结；混合架构69格维持uncovered，下一候选必须有新增源码证据。
3. **MMVQ带宽所有权/几何单独立项。** 是明确model-class mismatch，但需要独立微证据；不能因为issue-bound被HBM遮蔽就选择更高带宽。MMVQ和MMQ阈值目前已有源码接线，不应再做一轮“按c切MMQ”补丁。
4. **nonflash剩余具体kernel/输入更新、采样缺项，按真实executed branch补全。** 两者已有主要服务，应先owner ledger再补。
5. **host graph reuse条件事件和launch时间线。** 先语义与状态，保留现有128+12G；没有独立服务证据时仅标unpriced，不声称必有数值收益。不要将“graph复用”用作按c降低固定launch的依据。

后续每个候选应使用冻结静态形状/源级微fixture或独立合成kernel证据判断正确性；严格保留Engine TTFT/TPOT/E2E边界，HTTP/SSE/JSON微基准不进入Engine固定开销。本次没有对上表建议做代码修改或性能验证。