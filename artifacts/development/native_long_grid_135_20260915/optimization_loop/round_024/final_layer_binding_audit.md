# R24 末层 output-selection 生产绑定深审

日期：2026-09-16。结论：**生产确实没有启用，但缺口不是只少一行metadata：还有GGUF架构名和测试架构名不一致、metadata嵌套层级不同，以及source declaration没有历史编译内容证明。** 下个候选需要一个source/runtime受控绑定器，不能在通用GGUF读取器按模型名默认打开。

本次只写本文件和`final_layer_binding_audit.json`。未改core、tools、测试、冻结、任务书；未commit；未运行native、LLM预测或读取新轮目标时延。执行的验证仅为三个小型合成GGUF metadata对象的真实production builder调用，以及9组小fixture的baseline/declared静态cohort图（18张静态图、无服务调度模拟）。读取6个冻结模型的头部general.architecture只需每文件69–70字节，没有重新扫描大型模型权重。证据及当前源码SHA均在JSON。

## 1. 排除别名和延迟注入：三重证据

1. **真实构造器探针。** 对qwen2、llama、qwen35分别调用原`gguf_parity.build_model_from_gguf`，输出model.architecture依次为`qwen2`、`llama`、`qwen3_5_hybrid_transformer`。三者model.metadata顶层和内层metadata均无SOURCE_KEY。随后仅把合法`source_declaration()`交给当前resolver：qwen2和llama均报`requires a supported exact graph architecture`，只有hybrid可解析。没有自动`*_decoder`别名。
2. **嵌套层级实证。** `gguf_parity.py:456–471`调用`build_model_graph_from_layer_specs(metadata=...)`后`ModelSpec(metadata=graph.attributes)`；`ir.py:836–843`把用户metadata放到`graph.attributes['metadata']`。而`planner.py:14089`只查`scenario.model.metadata[SOURCE_KEY]`，不查嵌套。测试的`tests/model_helpers.py:30–49`故意把传入metadata直接放ModelSpec.metadata顶层。即使仅往GGUF builder的metadata字典加入字段，当前消费者仍可能看不到。
3. **完整生产数据流。** `predict_stable_native_dataset.predict_cell`真实build_model→build_matching_scenario→host_offload→tensor_storage→gpu_invocation→nonflash→retained→clock/MMVQ→replan→run_scenario。tensor_storage只改workload metadata (`llama_tensor_storage.py:159–184`)，nonflash/retained也只改workload；GPU invocation虽重建graph，`llama_gpu_invocations.py:413–421`显式恢复原graph.attributes并`replace(model,graph=...)`，保留原model.metadata，不添source selection字段；`llama_scenario.py:280–305`replan改workload/placement/profiles等，未替换model。MMVQ binding是component/profile工作量合同；没有迟到的选择器注入。冻结R23版本也含同一缺口；本结论不是仅凭rg找不到字面量。

当前tests共分两层：`test_final_layer_output_selection.py`验证纯声明/精确架构/shape；`test_final_layer_output_selection_planner.py:27–56`手工构造已声明fixture，`:245–256`验证无声明保持legacy，`:273`验证长prompt lm_head只1行。**没有生产GGUF builder→所有static binding→replan→planner入口的启用测试。** 特别纯测试`:39–44`还刻意将`qwen2`/`llama`作为不接受的别名；下一修改必须显式更新受支持的“真实GGUF canonical architecture”，而不是让测试继续绕开production形式。

## 2. 架构与锁定native分支

头部读取与冻结SHA引用得到：qwen25=`qwen2`；smollm2/tinyllama=`llama`；qwen35、qwen38、qwen38_gpu均=`qwen35`。Qwen3.8是本项目模型标签，**不存在需要另造的qwen38选行规则**。`gguf_parity.py:328–332,455`依据真实general.architecture将qwen35映射为hybrid graph。native `llama-model.cpp:321–324`实例化qwen35类，`models/qwen35.cpp:126`覆盖build_arch_graph；qwen2和llama对应文件`:49`/`:94`。

| 真实架构与固定分支 | 锁定源码位置 | 正确选行范围 |
|---|---|---|
| qwen2 ordinary completion | `models/qwen2.cpp:106–119` | 最后一层attention输出和残差各做GET_ROWS，再residual/norm/FFN；最后FFN按R行 |
| llama dense ordinary completion | `models/llama.cpp:174–189`，MoE另有分支 | 同样最后FFN前双GET_ROWS；现有仿真只覆盖dense tail，MoE fail closed |
| qwen35（含本项目qwen38标签）且embeddings_nextn_masked=false | `models/qwen35.cpp:174–176,180–223` | 最后FFN仍B行；全B行final norm之后GET_ROWS，lm_head才R行 |
| qwen35且embeddings_nextn_masked=true | 同上`:174–176` | 会提前双GET_ROWS；**不在现有声明范围**，不能按普通false分支误绑定 |

B=当前physical ubatch token rows，R=其中`requires_logits`行数。`llama-graph.cpp:199–223`先写共享host output indices；全输出B=R仍写identity indices。`:2475–2494`明确**即使所有tokens都输出也保留GET_ROWS输入与拓扑**，不能在decode B=R时把gather自动优化为零。所有attention/QKV/KV append仍处理B行；不是整模型只处理R行。

## 3. source_declaration能证明什么

`final_layer_output_selection.py:17–27`是纯常量factory，返回schema、固定backend_commit、completion/explicit_logits/embeddings=false/embeddings_nextn_masked=false/speculative_type=none。没有source SHA、DLL SHA、build receipt、model SHA、request/config hash、源解析或content_sha256。resolver只比较字段与factory完全一致；**此举证明声明格式和意图，不证明声明是真的。** 本次计算的canonical JSON SHA=`531fc6328c9076ebd614cac2516d51c03f78411b08ee1abaeb5d7fa797fbab74`仅能绑定这些字段，不能把它当native代码编译证明。

当前源码的模型文件SHA、llama-graph.cpp/llama-model.cpp和Unity include文件SHA记录在JSON。qwen2/qwen35由`Unity/unity_7_cxx.cxx:31,43` include；llama由`unity_4_cxx.cxx:49` include。annotation receipt保存这两个继承OBJ的hash，并经llama.link.rsp进入已固定llama.dll；native-thread-control继承模块identity链已存在。

**证据缺口必须保留：** base receipt的source_sha256_before/after和annotation input_sha256未记录这三个architecture include文件的历史内容hash。`predict_stable_native_dataset.py`既有`gpu_invocation_source_linkage`返回`architecture_specific_model_source_content: conditional; historical unity include body hashes unavailable`。当前include文件的内容和Unity对象的历史hash不能合成“该内容必定编译进了该对象”的完整证明。本轮不用重编译或换native来填洞；新合同应明确`historical_include_content_proven=false`、`source_runtime_binding=conditional`，若治理要求完全编译来源证明则保留uncovered，不偷偷升为verified。

## 4. 可以实施的最小绑定方案（本次未修改）

**接口位置：** 新的optional frozen static contract + 每格binding，与tensor_storage/gpu_invocation合同同级；在production worker应用F32与physical projection合同后、最后replan前调用`apply_final_layer_output_selection_binding(scenario, inputs, gguf)`。这样可校验真实模型/后端/工作负载并纳入最终placement fingerprint；不要在无runtime上下文的通用`build_model_from_gguf`里无条件启用。

**最小合同内容：** 固定schema与canonical declaration、当前模型分支源文件和graph input实现的SHA/引用、固定runtime module SHA、继承build refs及上述conditional限制、真实GGUF SHA和general.architecture、simulator graph architecture、ordinary completion有效参数（explicit logits、embeddings、embeddings_nextn_masked、speculative/MTP）、single-rank/F32隐藏数据资格、output row index的source语义。对合同canonical payload计算content_sha256；worker重新核验hash与输入一致性。以GGUF metadata与tensor/backend资格匹配，不能由model_key含qwen、显示名称或文件名选分支。

**架构修正：** 明确支持真实production的`qwen2`/`llama`和已有`qwen2_decoder`/`llama_decoder`的固定映射；这两类是graph canonical形式，不是任意显示名称别名。可在policy模块增加严格映射，或在已验证的binding resolver作有证据的canonical投影，但audit同时保留original_graph_architecture和backend_architecture。不建议为通过resolver而全局改模型architecture，避免影响其它source合同/缓存/placement。

**metadata修正：** 在worker拿到已验证证据后向planner实际读取的ModelSpec.metadata顶层写SOURCE_KEY；同时按既定serialization规则在graph metadata保存同一声明及独立binding proof，使后续graph rebuild/replan/序列化不丢失。须有唯一canonical所有者，若顶层/嵌套同时存在且冲突应报错，不接受“谁先查到用谁”。现有resolver精确限制字段集合，证据proof应放独立sibling key或新typed binding，而不是往旧declaration随意塞字段。

**有效参数证明：** 不能把`source_declaration()`输出false当作现场参数证明。冻结config未显式捕获的embedding/nextn_masked/speculative设置，需要锁定server request构造+runtime有效默认的source证据，或已有request/static capture明确字段。GGUF存在auxiliary nextn权重不自动意味着运行MTP；需检查actual workload.mtp/执行描述、server speculative配置和被建graph类型。任一不符合或缺证据应为uncovered并保留legacy，不猜默认。

**成本归属：** 复用现有row-index输入、GET_ROWS和tail shape实现，不再另建第二套gather成本。host index输入更新计入新增专门owner；现有host_prefix为通用聚合而非精确该scan，仍需标清未覆盖/可能重叠边界，不可用图选行删除128+12G command费用。

## 5. 静态plan验证：实际变化不是“所有成本都下降”

使用已有小fixture，仅编译任务图，未执行simulation。JSON保留每架构B/R下FFN、norm、head行数和gather计数。结果：

| 架构与shape | 最后FFN（legacy→declared） | final norm（legacy→declared） | lm_head | 新增selection工作 |
|---|---|---|---|---|
| qwen2/llama B64,R0 | 64→0 | 0→0 | 0→0 | host index scan；无非空gather |
| qwen2/llama B64,R1 | 64→1 | 1→1 | 1→1 | 一份host indices及上传、2个GET_ROWS |
| qwen2/llama B4,R4 decode | 4→4 | 4→4 | 4→4 | identity indices仍在，2个GET_ROWS |
| hybrid B64,R0 | 64→64 | 0→64 | 0→0 | host index工作；仍有全行norm |
| hybrid B64,R1 | 64→64 | 1→64 | 1→1 | host indices及上传、1个GET_ROWS |

hybrid的B4/R4测试fixture没有声明batched stateful capability，真实拆成4个M1group；每group数值shape不变，新增1个GET_ROWS。该fixture只证明接口按group保留工作，不冒充R23具体batching结果。

qwen2/llama变化还包括最后attention residual、FFN norm/up+gate/activation/down/FFN residual等B→R；QKV、QK/PV、attention输出投影及KV append仍B。last FFN真实weight GEMM可能因M从64变R而改变MMQ/MMVQ/转换子图，必须通过现有physical dispatch合同重新下沉，不把旧M64成本简单乘R/64。R0时不应留虚构的正尺寸FFN GEMM。当前qwen2/llama legacy final norm已经按R行，**不会再次节省norm**；hybrid原来把norm跟随R，接线会纠正为B行，可能增加prefill时间。B=R decode新增真实gather/indices，也可能增加TPOT。

## 6. 必要而有限的验收

已有单元测试涵盖纯声明、B/R/零行、多请求indices、DAG facts保留、template replay和无声明legacy。新候选最少还要增加：

1. 用**真实GGUF builder构造的小metadata fixture**，验证qwen2/llama生产架构名与metadata nesting；不要继续只用`tests.model_helpers`规避问题。
2. 验证source/runtime/model/config绑定失败、缺失、冲突均不静默开启；合法绑定在所有static转换与最后replan后仍存在。可拦截`run_scenario`入口断言，无需跑LLM预测。
3. 验证三真实architecture的B64/R0、B64/R1、B=R和多输出row索引；至少明确qwen35/qwen38 final norm增加为B行、QKV/KV账本不变、lm_head本来就是R。
4. 未绑定时任务图与baseline一致；未知runtime与embedding/MTP/MoE/tensor precision失败闭合。
5. 指标评估分两栏：source语义正确性与Engine精度。若既有TTFT已偏低，qwen2/llama最后FFN裁剪可能使误差更大，必须如实报告**语义修正伴随精度退化**，不能保留错误FFN行数凑目标；也不能因误差方向只开启对自己有利的架构。新轮结果只能按预注册完整比较揭盲，当前审计不预测收益幅度。

限制：本次只验证source分支、生产数据流和小静态图；没有native physical placement/GET_ROWS内核dispatch证明，没有通过新参数消解历史Unity include来源缺口，也没有重测完整目标。上述conditional证据等级必须随候选结果保留。