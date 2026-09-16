# R22 retained KV occupied-count 下界审计

2026-09-16，只读。**可以形成比 R21 `max(single retained context)` 更强的、有明确资格的 distinct-occupied-count 下界；不能把固定 warmup 总量与当前所有context盲目相加。** 对冻结采集器发出的普通、单completion、单序列 token 请求路径，已找到完整的排除共享条件；`cache_prompt=false` 本身不足，但它与 `n_cmpl=1`、`--cache-ram 0`、单slot标签batch及无slot restore/外部状态操作组合后，能够排除此路径上的跨slot `seq_cp`/多owner prefix。初始warmed池与每slot生命周期仍必须作为状态，而非每个fresh cohort重置为0。

本审计没有打开目标 prediction/errors、native responses或时延数值，没有运行native/LLM重测、拟合或修改R22测量包/core。仅读取冻结protocol/freeze静态plan、冻结执行器源码、server/KV源码与既有构建身份。以下“完成的warmup”依赖采集器/选择器的结构完成资格，不是本次重新核对原始响应的声明。

## 1. 真正固定的HTTP与block协议

核对主批、SSE补批、GPU正式/只读正式及clock exception补批的native protocol/freeze：所有plan均是 **2 warmup batches → 3 measured batches → 同block进程结束**，一个process block，`preconnect_http`，`cache_ram_mib=0`；无job/condition覆盖改变这些项。slot context=2048、总ctx=parallel×2048；所有静态plan的P+O≤2048。

冻结runner有两个版本：主批/SSE补批SHA `d14e77582047676f3bd9336fd5c2ac460e1eb654777b65b73027e9295a51fb6f`；GPU/clock补批SHA `44797d425e71fb6f2fccf011802f606b5d67a38b153aa472c69b048250c26e54`。本次已按各native freeze的source_refs重新核对其文件SHA。以下主批行号与GPU版本差约1–2行，关键逻辑相同。

- [native_repeatability_experiment.py:455](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/execution/tools/native_repeatability_experiment.py:455) 的固定payload：prompt、`n_predict=c.output`、ignore_eos=true、cache_prompt=false、temperature=0、top_k=1、seed、stream=true。**warmup使用同一payload和完整O，不是另一工具native_llama_compare中的warmup_predict=2。** 没有n/n_cmpl、id_slot、slot load、LoRA或spec请求参数。
- [native_repeatability_experiment.py:423](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/execution/tools/native_repeatability_experiment.py:423) 的server命令包含np=parallel、b=ub=64、faoff、kvu、cache-ram=0、spec-type=none。[native_repeatability_experiment.py:435](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/execution/tools/native_repeatability_experiment.py:435) 每block新起一个server；457开始的两阶段loop重用同一server与client，warmup和repeat之间不发送KV clear/slot erase，不重启server。
- [native_repeatability_experiment.py:350](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/execution/tools/native_repeatability_experiment.py:350) 以barrier提交parallel份独立HTTP请求；357收齐全部future后才开始下一batch。没有把同prompt转成一个n>1 parent请求。`inspect_batch` 在267附近拒绝slot缺失或重复；完整batch必须恰好parallel个不同slot。结合server slot总数np，**被认可的完整warmup batch覆盖所有slot**，即使到达/完成不完全同步。仅看到“发了parallel个请求”本身不够，要保留这条distinct-slot完成资格。
- 选择器 [native_162_dataset.py:419](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tools/native_162_dataset.py:419) 要求2 warmup/3 measure/1process，426要求complete block，441–459同时审核warmup和runs的batch/request覆盖。此审计未重新打开那些响应，所以不能替换实际completion evidence或补造slot map。

## 2. 逐项排查共享、clone和cache路径

| 路径 | 锁定源码条件 | 对当前闭合协议的结论 |
|---|---|---|
| parent/child共享prompt | [server-context.cpp:4598](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:4598) 仅n_cmpl>1创建children；[server-task.h:63](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-task.h:63) 默认n_cmpl=1，[server-schema.cpp:62](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-schema.cpp:62) 将n作为alias；[server-task.h:253](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-task.h:253) is_parent取child_tasks非空 | 当前payload未覆盖默认1，没有parent/child |
| `copy_state_to`的真正seq_cp | [server-context.cpp:910](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:910) 先清child seq再`mem.seq_cp(parent,child)`；[server-context.cpp:4008](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:4008) 仅DONE_PROMPT且task.is_parent时调用 | 上述门关闭。此处是server context中找到的唯一直接seq_cp调用 |
| 同一KV cell多owner | [llama-kv-cache.cpp:451](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:451) 同stream seq_cp只向既有cell添加dst seq_id，不复制物理数据；[llama-kv-cache.cpp:1154](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1154) apply_ubatch会按输入n_seq_id给cell添加owner | **所以cache_prompt=false单独不能证明不共享。** 但[server-context.cpp:205](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:205) render每token调用common_batch_add(..., `{t.id_slot}`, ...)，当前普通请求每token恰有一个owner；没有按token内容自动去重的路径 |
| LCP相似度选slot | [server-context.cpp:1735](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:1735) 仅在可用slot间选一个相似/LRU目标，不调用seq_cp | 相同prompt/seed可影响slot选择，但不等于跨slot共用KV；此选择不受cache_prompt=false直接关闭 |
| RAM prompt cache | [server-context.cpp:1539](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:1539) 只有cache_ram_mib!=0才创建；[server-context.cpp:1825](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:1825) get_available_slot的save/load不以task.cache_prompt为唯一门 | 当前明确cache-ram=0关闭。不能把这个结论归因于cache_prompt=false |
| idle-slot自动save/clear | params默认cache_idle_slots=true（[common.h:628](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/common/common.h:628)），但[server-context.cpp:1608](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:1608) 在cache_ram_mib=0时明确禁用；2633–2645否则可清idle | 当前自动idle清理分支关闭，不能只读默认true就认定warmup残留被清空 |
| task/tokens/prompt clone | [server-task.h:228](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-task.h:228) add_child clone在n>1分支；[server-context.cpp:920](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:920) prompt.clone紧跟state copy；RAM cache的prompt clone是CPU容器快照 | C++ tokens.clone本身不建立KV多owner。须区分CPU容器复制与seq_cp |
| RAM/slot状态restore | [server-task.cpp:1793](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/tools/server/server-task.cpp:1793) cache.load通过state_seq_set_data_ext加载指定slot；server 2800–2845为显式slot restore API；[llama-kv-cache.cpp:2335](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:2335) single-dest restore先seq_rm目标、要求每cell n_seq_id=1并重写为dest ID | RAM关闭且runner不发restore。指定单seq restore也不是凭内容给现存cell添另一owner；全context外部restore/额外API不在本资格内 |
| checkpoints/speculative | cache_prompt分支中的checkpoint恢复及spec rollback、sampler clone仍存在；如server 4210/4260的seq_rm | 固定无MTP/spec，n_past=0绕过正前缀checkpoint恢复；不能据此给非固定请求同样结论 |

**限定的归纳证明：** 启动能力测试/模型warmup结束会清memory；之后每个插入token只带一个slot seq_id，普通KV find_slot只用空cell（或明确SWA回收分支），插入后只添该单owner；seq_rm、位置shift/defrag改变/移动现有单owner，当前路径又没有seq_cp、多seq token或任意状态恢复。因此在ordinary、无共享历史的attention KV池中，不同slot的保留集合两两不交。这是源码加冻结工作负载的条件证明，不是对任意外部HTTP操作的无条件保证。Qwen35等hybrid只能对其符合条件的attention cache使用此证明，不能把recurrent state数当成token KV数，也不能跨layer/cache重复相加。

## 3. warmup残留、slot复用与清理时点

1. **区分两种warmup。** [common.cpp:1511](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/common/common.cpp:1511) 的server `--warmup` 是启动空跑；1541显式memory_clear并同步，不能把它算作每slot保留前缀。[common.cpp:1591](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/common/common.cpp:1591) 至1619的seq-rm能力检查也清memory。真正可能留下N个slot状态的是随后采集器发的两轮HTTP warmup。
2. **普通release保留KV。** [server-context.cpp:730](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:730) 设置IDLE、更新时间、callback/reset；只有child才prompt_clear。reset清task/统计/sampler状态（575起），不清普通prompt/KV。**成功完成普通请求并不等于释放其KV占用。**
3. **cache_prompt=false在“新请求开始填prompt”时清旧内容。** [server-context.cpp:2011](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:2011) 分配请求仅变SLOT_STATE_STARTED；[server-context.cpp:3322](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:3322) 若batch已满直接跳过剩余slots，尚未进入3349的STARTED分支。实际进入后3502将n_past=0，3630的keep_first(0)，3654–3660按p0=0 seq_rm目标slot旧tokens，再填本次prompt。一个已经接到请求、is_processing=true但尚未获得prompt预算的slot，仍可保留上轮KV，不能仅看active/idle布尔值就归类为“已换成新context”。
4. **最后一个生成token尚未进入KV。** [server-context.cpp:692](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:692) 的handle_last_sampled_token只在下一轮decode准备时把上次sampled token插入batch并push到prompt；当前sample结束达到O预算后停止并release（2087、4145）。在普通无shift/noSWA/无中断且P已含实际tokenizer/BOS的资格内，完成长度O的请求保留 **P+O−1** 个KV位置，而不是P+O，也不是O。warmup与正式payload相同，所以上一批完成后的每slot候选W为该值。
5. **有清理反例，不能隐去。** [server-context.cpp:1855](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-annotation-control/tools/server/server-context.cpp:1855) 的try_clear_idle_slots在kvu模式清一个非processing且有prompt的slot；3963在llama_decode空间失败时会调用它再重试，最终请求仍可能成功。context shift在3117–3160删除/平移区间；SWA find_slot在[llama-kv-cache.cpp:1042](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1042)附近可回收被mask的旧cell；错误、取消、restore/erase、model sleep/reload也必须进入状态机。cache-ram=0只排除RAM/自动idle-cache清理，**不排除压力清理**。

普通非SWA KV有一条进一步可验证的容量论证：冻结所有plan满足P+O≤2048且C=np×2048；每slot在新prompt前删旧、每slot保留≤P+O−1，故全池占用不超过np×(P+O−1)<C。[llama-kv-cache.cpp:750](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:750) 的prepare使用find_slot(ubatch,false)，允许跨holes找空cell，地址碎片本身不强制连续分配；若无额外owner/状态操作，则这一普通KV路径有足够总空位。这可以用于排除该限定路径的KV容量型purge，而不能自动外推到hybrid recurrent allocator、SWA、上下文变更或失败恢复。要把W初始化成为全131统一事实，必须逐模型验证这些附加资格，或保留非计时的清理/slot状态证据；“block complete”本身不证明从未清过idle。

**跨repeat状态：** block里第1次正式repeat从最后一轮HTTP warmup的池接续；第2、3次从上一正式repeat池接续。新block进程才重新初始化。若每个slot都完成同P/O且所有资格成立，batch边界可用每slotW重建状态，而不需要把5个历史batch的context累加。若部分slot未被新请求开始处理、已清理、已释放但保留、已复用，必须保留各自阶段，不能按请求完成数猜slot身份。

## 4. 更强下界的数学形式与防止重复计数

对**同一个unified attention cache**定义每slot在时刻t真实仍存的集合S_s(t)。只有上述singleton-owner契约成立，才有

`U(t)=|union_s S_s(t)|=sum_s |S_s(t)|`。

[llama-kv-cells.h:85](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cells.h:85) get_used是used集合大小；97的used_max_p1是最大占用地址+1，所以不论head或holes如何，`U(t)≤H(t)=used_max_p1(t)≤C`。[llama-kv-cache.cpp:1250](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1250) 的源规则是 `n_kv=min(C,max(256,pad256(H)))`（统一流），因此若对各slot持有可信下界L_s(t)，则

`n_kv(t) ≥ min(C,max(256,pad256(sum_s L_s(t))))`。

它通常强于只取max单slot，因为singleton-owner下sum≥max。**这里得到的是物理宽度下界，不是exact；head、holes和任何未计入的occupied单元只会令H更大。** 超出容量的sum不是应该强行截断掩盖的情况，应先拒绝不一致的状态输入。kvu=false时必须按实际stream分别算，不能先把不同stream的slot求和。

应按互斥状态取每slot一次：

- `old_retained_not_cleared`：包括IDLE，也包括已assignment但尚未开始prompt的SLOT_STATE_STARTED；取最后成功已证明保留W_s。
- `cleared_for_current_request`：旧W_s失效，改取本次已经进入/将进入当前成功ubatch的真实retained下界R_s(t)。不是所有新请求的完整P，更不是尚未处理的prompt总长。
- `current_completed_retained`：普通完成后改存本次P_s+O_s−1（在相同资格下）。
- `cleared/purged/unknown`：已清为0；未知不能继续保留旧W作下界，应降至独立可证的活动集合下界，并报uncovered原因。

**仅为源码算例，不是观测或拟合：** np=4、P=512、O=32、C=8192、全部上一HTTP batch成功且ordinary资格成立，则各slot W=543。新repeat的第一个64-token prompt ubatch若只清/填了一个slot，其余3个尚未开始prompt，U下界=64+3×543=1693，padding后n_kv下界1792，而max单个新context规则只给256。此算例依赖清理/调度阶段明确，不能套到所有图。4个slot全部清旧后，只能用各自新R_s求和，不能再加4×543。

## 5. 不能证明的部分与具体反例

- **仅cache_prompt=false不足：** 改n=2仍可走parent/child copy_state_to，same-stream seq_cp把两个seq挂到同一cell；两context之和会双算prefix。这个反例被当前n=1协议排除，但说明契约必须包含n条件。
- **盲目W+active在当前n=1/cache-ram=0下也错：** np=1旧W=543，新请求开始已seq_rm旧池，只插入64个prompt token；W+64=607不是占用下界。旧W只可在同slot尚未清除时存在一次。
- **warmup完成不自动保证之后的W存活：** 压力purge可清idle slot后成功重试；默认cache_idle_slots在RAM开启时也会清。当前RAM=0排除了后者，前者需ordinary容量证明或状态证据。hybrid/非普通attention不能凭P+O容量护栏就宣称整个memory准备永不失败。
- **P/context不是普遍的KV cell计数：** SWA回收、共享prefix、多序列token、trim/shift、MTP draft/rollback、外部restore都破坏朴素计数。不能把recurrent缓存尺寸或每layer重复KV视图加到单池U。
- **本审计不输出实际每时刻U/H。** 源码证明能下沉资格和状态转移，不提供native slot运行ledger、实际head地址、清理时间或131格数值效果。

## 6. 后续状态输入与最小安全下沉边界（只建议）

需要block级来源契约：新进程/启动clear、2 warmup+3repeat、np及总C、每请求n=1/noMTP、RAM=0及cache_idle实际禁用、P/O/BOS语义、slot容量、ordinary/noSWA attention-cache资格和无其他state API；还需warmup distinct-slot完成资格，以及上一batch到本batch的slot assignment、prompt-start-clear、成功ubatch提交/完成、release-retain、purge/erase/shift/rollback等非计时状态。

模拟应持有跨HTTP batch/repeat的**per-slot retained pool**；scheduler进入STARTED前后的KV转移与现有engine起点保持一致；当前invocation只能统计已进入池的query tokens，不能提前算完整prompt。若同构完整batch可证明每slot最终W一致，可用这一静态初始模板简化，但必须包含“旧池尚未清/已清”的slot掩码。对无共享且生命周期已知的active slots，可先用sum active lower bounds；idle未知部分不给正的假下界。这样既不需要精确head/holes，也不需要任何目标时延系数。

结论分级：**无共享路径资格和U≤H公式可成立；普通已证明warmup-retained池可增强下界；全131的warmup池初始化/跨repeat生命周期尚不能仅凭fresh-cohort输入自动补齐。** 新增系数0，未实施模型改动或重测。

## 源码/静态证据哈希

| 文件 | SHA-256 |
|---|---|
| `source/llama.cpp-annotation-control/tools/server/server-context.cpp` | `99f7aead4dd6b190292db2a14b2586d4076871a3710a49a94c71424b6f05501e` |
| `source/llama.cpp-semantic/tools/server/server-task.h` | `8bed70ca9a719c82a1e83b5ea8f7423d2e863c3ce79a154df6d0ed5ca67d849e` |
| `source/llama.cpp-semantic/tools/server/server-task.cpp` | `4ef9e2c0a78ee4480b795d5056a4f979f10529c89e2c3526c728e6d7f6107613` |
| `source/llama.cpp-semantic/tools/server/server-schema.cpp` | `4b419ef42556cf8dda5c0471ed9294388da613b62eeee56efa2e13ed34252238` |
| `source/llama.cpp-semantic/common/common.cpp` | `5e80bdf7336324a55942bd7a46aba652bf438ceebd2d14133fe5e8be455b530c` |
| `source/llama.cpp-semantic/common/common.h` | `6e678df90075831853a3b509a7a3fe01e5aa4cbeb1b3ae4364c0ac51442b36c3` |
| `source/llama.cpp-semantic/src/llama-kv-cache.cpp` | `e4d2aa977c8aa048ddd79c878683e94909f5f24ac109b0fcd1c0b59ef70f3899` |
| `source/llama.cpp-semantic/src/llama-kv-cells.h` | `5e8cb5115dab535346d4ab895ca6e25d2b85a376a510fa16faaf9a64539e9206` |
| `artifacts/development/native_long_grid_135_20260915/execution/tools/native_repeatability_experiment.py` | `d14e77582047676f3bd9336fd5c2ac460e1eb654777b65b73027e9295a51fb6f` |
| `artifacts/development/native_long_grid_135_20260915/gpu_extension/execution/tools/native_repeatability_experiment.py` | `44797d425e71fb6f2fccf011802f606b5d67a38b153aa472c69b048250c26e54` |
| `tools/native_162_dataset.py` | `87487a8864e2d860abb2be74cf2fe97422bd033dc7ce59c9012708d5466a47a2` |

静态协议审计覆盖主批、supplement_sse_v2、gpu_extension/formal、gpu_extension/formal_readonly、clock_exception_supplement 的 native_protocol.json/native freeze.json；未读取这些目录的响应、runs时延或评分。
