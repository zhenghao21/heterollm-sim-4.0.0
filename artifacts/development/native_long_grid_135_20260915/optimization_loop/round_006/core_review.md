R6 核心只读评审已完成。共发现四项待处理边界：两项在特定路径改变成本/任务图，两项只影响计数与覆盖标签。本次没有修改实现、冻结文件或预测结果，没有运行原生LLM、原生构建、完整事件模拟或拟合时间。

当前已冻结路径的相关证据：真实CPU27B在M8、M64单次R6处理前后，任务ID、依赖及资源成本逐项一致；当前在线microbatch命名带组编号，分批汇总正确；五个保留GPU模型的完整注意力QKV目录均为三个物理段。由本次证据未发现必须改写已经冻结的预测结果的理由。

| 编号 | 问题 | 时间预测影响 | 当前冻结路径 |
|---|---|---|---|
| R6-CR-01 | 重复 apply 清空已生成 GPU 别名来源，旧 CPU 能力随后开启 QKV 拆分 | 特定触发条件下会改变 | 单次apply未触发 |
| R6-CR-02 | 真正 packed QKV、flash-off 的 V 视图需要连续化，但缓存路径直接 SET_ROWS | 特定触发条件下会改变 | 完整注意力非packed，未触发 |
| R6-CR-03 | 场景级 GPU 标签让未通过投影组验证的通用 GEMM 也被计为 applied | 无，仅统计/标签 | 当前有效组未复现 |
| R6-CR-04 | MMQ conversion/fixup 仅以 op_name 去重，会合并不同请求的同名调用 | 无，仅统计/标签 | 在线组名不同，未触发 |

**R6-CR-01（P2） 重复 apply 清空已生成 GPU 别名来源，旧 CPU 能力随后开启 QKV 拆分**

同一含 legacy llama_cpp_physical_projection_invocations=True 的场景连续两次启用 apply；首次调用生成 attention.q/k/v 别名。

首次 generated_gpu_aliases 有三个值，第二次变为空。CPU QKV split 从 False 变 True，CPU GEMM 从5变7，总任务56变61；依赖/资源成本签名不同。F32 在两个输入中均已启用，排除 F32 混杂。

_layer_qualification 每次从 generated=[] 开始，只将本次新建别名加入；再次 apply 时现有别名相等，因此来源标签被覆盖为空，CPU 拦截不再生效。

未在本次当前冻结调用路径观察到：wrapper 每场景单次 apply；真实 CPU27B R5 M8/M64 单次 apply 的依赖与成本签名完全相同。

后续建议：保留并校验此前生成别名的来源，或提供语义完整的重复 apply 处理；增加同一 contract 重复应用的 CPU 隔离回归。当前未修改。

证据：[llama_gpu_invocations.py:236](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/llama_gpu_invocations.py:236)；[llama_gpu_invocations.py:286](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/llama_gpu_invocations.py:286)；[planner.py:8596](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:8596)。

**R6-CR-02（P2） 真正 packed QKV、flash-off 的 V 视图需要连续化，但缓存路径直接 SET_ROWS**

attention.qkv 恰为一个物理合并矩阵，reshape V 为原矩阵的带跨度视图，flash_attn=False，M>1。

源码 build_qkv 的 Vcur 使用 qkv->nb[1] 作为 token stride；cpy_v 在该 stride 与 V 连续行宽不等时发出 ggml_cont_2d。受支持的合成 M8/K256/N512 packed case 被标记 cache status=applied，但图中只有 RoPE 与 V SET_ROWS，没有连续化阶段。V token stride=2048B，连续行宽=512B；缺少至少4096B读+4096B写的逻辑复制工作。

_gpu_invocation_kv_contract 对 packed 情况只验证总宽度；_add_native_local_kv_writeback 未区分 packed V 视图的布局。

保留的五个 GPU 模型完整注意力 attention.qkv 目录描述均为三个物理段，因此当前这些分开存储的 Q/K/V 路径未触发。线性注意力的 packed QKV 不属于此缓存写入分支。

后续建议：在 packed V stride 已证明时建模连续化；否则缓存子合同应明确 uncovered。复制的具体 CUDA 内核/服务需沿复制派发证据继续核对，不在本评审编造延迟。当前未修改。

证据：[llama-graph.cpp:1649](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-graph.cpp:1649)；[llama-kv-cache.cpp:1388](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1388)；[planner.py:12141](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:12141)；[planner.py:12277](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:12277)。

**R6-CR-03（P2） 场景级 GPU 标签让未通过投影组验证的通用 GEMM 也被计为 applied**

至少一个投影组通过，但 attention.qkv 等其他组因物理张量形状/字节证据不符而未通过。

将 Q 权重绑定 n_bytes 加1，attention.qkv 组为 applied=False/source_tensor_storage_mismatch，图保持三段合并的通用 QKV；它仍获得 gpu_native_invocation 标签，GPU 汇总为5/5 applied、uncovered_tasks=0。MMQ 子报告则正确将该任务列为 one_physical_projection_not_proven。

_add_rank_gemm 仅检查场景合同已应用及 GPU 目标，给所有权重 GEMM 打标签；summarize_gpu_invocations 只检查标签存在。该赋值也覆盖更具体的 FFN group/physical_weight_matrices 标签，尽管 projection_segments 仍保留两个物理权重段。

只影响覆盖披露，不改变调度或成本。在当前有效的完整注意力输入未复现失败组；不应将 applied_tasks 解释为逐组成功证明或原生派发数。

后续建议：保留逐组资格与失败原因；区分“场景 contract 标签”“投影组通过”和“实际原生执行证明”。不要仅据标签存在计 applied。当前未修改。

证据：[planner.py:9265](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:9265)；[planner.py:8955](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:8955)；[planner.py:16850](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:16850)。

**R6-CR-04（P2） MMQ conversion/fixup 仅以 op_name 去重，会合并不同请求的同名调用**

对多请求离线 TaskSpec 序列汇总；不同 request_id 的 operation 名称均形如 prefill.full0.rank000....

两请求 M64：14个 matrix 主任务正确计数；实际14个 conversion launch 和14个 fixup launch，汇总分别只有7。按(request_id, op_name)可区分14个。真实在线130行/ubatch64的三组路径带 group0000/0001/0002，21个 matrix 中14个MMQ，conversion/fixup均正确为14。

stages[stage].add(op_name) 缺少调用身份；TaskBuilder.task_id 本身具有 request/counter，但 metadata.op_name 不自动包含这些字段。

当前 _serving_lowering_from_builder 按单个 cohort 汇总，分组 phase 已包含 cohort/group 标识；本次分批探针未触发，因此无需因该计数问题改变当前预测。公共汇总函数对离线/合并任务输入存在真实少计。

后续建议：使用稳定物理调用身份而非仅名称去重，包含 request/cohort/invocation group。实际设备阶段名是 gpu_elementwise，不能改为只数 gpu_tensor_kernel；无数据工作的 fixup 只有 kernel_launch 也须保留。当前未修改。

证据：[planner.py:8980](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:8980)；[planner.py:3394](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:3394)；[planner.py:23478](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:23478)；[planner.py:10263](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:10263)。

**机制与记账核对**

M1 gate/up 融合记账：一个融合向量内核处理两个物理权重矩阵；输入 Q8_1 转换一次。模拟中一个 MxKx(2N) 通用 GEMM 保留2个 projection_segments、两矩阵权重字节以及压缩后的 N 元素输出。M1合成例总权重147456B、输出2048B=4*512、转换一次；MMQ明确 uncovered/one_physical_projection_not_proven。2N只是总算术工作表示，不构成原生 MMVQ 主吞吐证明。

证据：[mmvq.cu:1431](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:1431)；[mmvq.cu:1484](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:1484)；[planner.py:16892](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:16892)；[planner.py:9057](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:9057)。

M>1 独立 MUL_MAT 的输入转换：固定源码的 ggml_cuda_mul_mat_q 每次调用内部申请转换临时区并调用 quantize_mmq_q8_1_cuda；MMVQ函数也在每次调用内部转换。未找到跨独立 MUL_MAT 复用已量化输入的分支。因此拆分 gate/up 后分别转换有源码依据，不能因输入地址相同便只计一次。融合 gate/up 在同一次 MMVQ 调用中共用转换。

证据：[mmq.cu:135](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmq.cu:135)；[mmvq.cu:1484](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:1484)；[planner.py:9690](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:9690)。

MMQ conversion/main/fixup 启动与流量：每个支持的矩阵恰有一 conversion、一 matrix 主调用，以及源码条件要求时的一 fixup；main partial writes 加在同一 GEMM 的内存阶段。转换消费者只读转换临时区，未再次计原始 F32 作为 GEMM 输入。fixup P=0仍保留launch-only；M64合成例7个MMQ对应7 conversion+7 main+7 fixup。缓存/roofline成本仍是分析模型，不能等同测量的CUDA时间。

证据：[quantize.cu:575](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/quantize.cu:575)；[mmq.cuh:1441](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmq.cuh:1441)；[planner.py:9690](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:9690)；[planner.py:9983](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:9983)；[cost_models.py:2594](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/cost_models.py:2594)；[cost_models.py:2801](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/cost_models.py:2801)。

alpha/beta F32 与 flash-off KV 几何：真实冻结 Qwen3.5 在M8/M64各有36个alpha/beta GPU矩阵；1024->16控制输出分别512B/4096B，均为F32。runtime flash_attn=False、统一F16缓存：每个V写入源值4*M*kv_width、持久2*M*kv_width、索引8*M*kv_width；K索引8*M。SET_ROWS内部完成F32->F16，没有另增转换launch。精确转换指令与重复索引事务未定价；packed V 连续化缺口见R6-CR-02。

证据：[planner.py:16125](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:16125)；[planner.py:12157](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:12157)；[planner.py:12303](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:12303)；[llama-kv-cache.cpp:1409](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/src/llama-kv-cache.cpp:1409)；[set-rows.cu:240](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/set-rows.cu:240)；[set-rows.cu:380](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/set-rows.cu:380)。

CPU当前R5单次应用隔离：真实CPU27B M8和M64在R5基线与单次R6处理之后，任务数、任务ID、依赖及每项资源demand完全一致。M8：3522任务/497 CPU GEMM；M64：6562任务/33 CPU GEMM+464由R4派发的GPU GEMM；两边均相同。未给CPU GEMM打GPU标签。合成CPU场景F32事先开启也保持一致；重复apply例外见R6-CR-01。

证据：[planner.py:8596](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:8596)；[llama_gpu_invocations.py:391](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/llama_gpu_invocations.py:391)。

GPU处理全局启用F32的范围：apply在整个scenario打开 llama_cpp_f32_hidden_storage。若CPU场景此前未开启F32，CPU与相关primitive的流量/成本会变化，即使投影调用数相同。当前R5 f32_and_gather_r2已开启该标志，因此本次真实CPU探针对照不受影响；不能把任意旧CPU场景应用R6宣称为零影响。

证据：[llama_gpu_invocations.py:370](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/llama_gpu_invocations.py:370)；[planner.py:13376](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:13376)。

本次只读验证使用现有测试夹具进行针对性图编译，并读取两份真实GGUF的头部/张量目录，编译四个R5冻结配置的M8/M64图；未读取或哈希张量payload。没有重跑已经通过的完整测试批次。机器可读报告保留全部探针观测、图签名、头部读取范围和已评审源码SHA。

细节边界：MMQ汇总只记录MMQ conversion/fixup，M1 MMVQ的Q8_1转换不应计入该MMQ计数；GPU GEMM汇总也不是全图CUDA kernel总数。当前设备阶段名为`gpu_elementwise`，不是`gpu_tensor_kernel`，P=0修正阶段仅有`kernel_launch`。

所有原生派发与性能仍保持conditional/unproven。MMVQ主吞吐、转换指令、索引事务、图入口/启动/同步服务没有被本评审补出经验常数或残差。后续源码修复由父代理决定，当前保持冻结结果不变。
