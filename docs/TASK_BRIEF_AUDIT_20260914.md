# 仿真器机制建模、可信度与泛化优化任务书

> **文档状态（动态）**：当前任务仍未达到第一阶段验收阈值；验收主口径已更正为 engine-first，client 仅作二级诊断。本文件是唯一的推进规范；稳定章节定义不可变的规则，动态章节在每轮修改后原位替换。历史实验只保留索引，不在此重复展开。
>
> **适用项目**：`F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0`
>
> **当前版本**：仿真器成本模型已包含 prefill chunk 适用域修正、host/frontend 成本接口和 native-evidence replay 门禁；最新完整回归为 `892 passed, 1 skipped`。当前 L1 Engine TTFT/TPOT/E2E 尚未同时达到 `<10%`；最近一次 V4-L9-r3 native 批次在 `67/135` 条后停止，不能作为验收结果。

## 1. 总目标与优化优先级（稳定）

根据模型结构、权重格式、输入输出长度、硬件、运行时和调度配置，生成可解释的算子图与执行时间预测。仿真对象的一级范围是 engine 内部的 GPU、HBM/HBF 与其他内存访问、kernel、KV cache、CPU/GPU 协作、同步、采样和调度执行；这些部分共同决定研究对象的 engine wall。HTTP、JSON、tokenizer、admission、队列、SSE、socket 和操作系统线程调度属于 server/client 服务边界，只能作为分解后的二级诊断。对同一请求，服务侧只可在边界证据证明不存在重叠时写成 `T_server = T_frontend + T_queue + T_engine + T_response`，再将请求/接收传输单独计入 `T_client`；存在并发或包含关系时必须保留时间线，不能把各段简单相加。HBF 参数扫描只能改变 engine 内受其影响的内存、KV、kernel 和调度成本，不能把不受 HBF 影响的 frontend/response 固定开销吸收到 engine 成本。预测不得读取或拟合待预测场景的原生端到端时延。

优化顺序固定为：

1. 先保证执行语义和测量口径正确；
2. 再改善未见场景的预测误差；
3. 最后降低校准成本和模型复杂度。

任何降低平均误差但引入数据泄漏、错误语义、证据缺失或覆盖率下降的方案都不能保留。

## 2. 迁移能力与允许信息（稳定）

| 迁移类型 | 允许使用的信息 | 明确禁止 |
|---|---|---|
| 同硬件与运行时，换新模型 | 新模型结构、权重格式、静态 shape、已冻结的通用算子性能模型 | 用新模型完整推理时延重新拟合；补测专属场景后宣称“零校准迁移” |
| 换硬件或软件运行时 | 设备规格、预先规定的通用算子/内存/传输/调度微基准 | 用目标 LLM 的 TTFT、TPOT、E2E 拟合硬件参数 |
| 全新硬件，只有规格 | 规格与已有机制模型 | 与经过硬件标定的预测混用同一可信度等级；必须单独标记“规格驱动、未验证” |

profile 的 **provenance**（采集时实际 binary、DLL、硬件、运行时和配置的来源真实性）与 **compatibility**（执行路径、构建配置和性能曲面能否迁移）必须分开判定。EXE/DLL SHA 和依赖集合是 provenance 门禁条件，不是性能等价证明；`profile_gate=blocked` 只禁止该 profile 进入正式预测，不改变原始 native actual 在其采集环境下的证据状态。跨 binary 只有在源码路径、构建参数和独立通用微基准证明等价，并写明 allowlist 适用范围后，才可按兼容性范围复用。

## 3. 三层建模约束（稳定）

### 3.1 执行语义层

确定实际执行的算子、输入输出 shape、dtype、layout、量化路径、输出行选择、KV cache 行为、设备放置、依赖顺序和重复计费。按锁定版本 llama.cpp/ggml 的真实实现选择模块；不得按模型名称加入时延补丁。`lm-head` 的行数由后端实际 logits 输出行决定，不能固定成 prompt 长度或一行。

### 3.2 算子成本层

成本写成

\[
T_{op}=f(op,shape,dtype,layout,kernel,hardware,runtime)
\]

允许使用独立微基准得到的有效吞吐、带宽、启动开销、缓存效应和局部性能曲面，也允许使用有物理解释且独立验证的残差模型。每个参数必须记录来源、单位、适用范围和失效条件。禁止以“模型 × prompt 长度 → 最终时延倍率”为主要机制；模型 SHA 和 prompt fingerprint 只用于溯源。

Roofline 只能作为基线和约束。GEMM 效率必须考虑矩阵尺寸、分块、并行利用率和 kernel 选择，不能只用峰值算力或峰值带宽。NVTX CPU range 或 API wall 不能直接等同于 GPU kernel wall；异步提交、多 stream 和同步造成的时间必须按关键路径分析，不能把 phase wall 减去 kernel wall 的差值机械地追加到算子成本。

### 3.3 系统调度层

根据依赖图、资源竞争、请求到达、batch 策略、CPU/GPU 协作和传输重叠生成时间线。可重叠的区间不能直接相加；stage wall、API wall、kernel wall 的包含关系不能重复计入。连续 batching、并发 prefill、decode 顺序和共享启动边界必须有 trace 或明确推导支持。

## 4. 统一计时契约（稳定）

一级验收采用 llama.cpp **engine 口径**；二级诊断为 server 边界分解，三级诊断为 client 完整链路。三套口径必须同时保存，不能互相改名、混用或用二级结果抵销一级失败。

engine 一级指标固定为：

\[
TTFT_{engine}=t_{first\ engine\ token}-t_{engine\ request\ begin}
\]
\[
TPOT_{engine}=\frac{t_{last\ engine\ token}-t_{first\ engine\ token}}{N_{out}-1},\quad N_{out}>1
\]
\[
E2E_{engine}=t_{last\ engine\ token}-t_{engine\ request\ begin}
\]

`engine request begin` 必须是请求已进入 llama.cpp engine、完成 server admission 的边界；`first engine token` 必须是 engine 完成首个真实 token 的采样/生成边界；`last engine token` 必须是 engine 完成最后一个真实输出 token 的边界。`engine request end` 若被保存，必须等于该最后 token 边界，不能包含 response queue 或 socket 写入。`prefill_begin`/`prefill_end` 只表示阶段范围，不能单独替代 `TTFT_engine`。原生 `prompt_eval_ms`、`eval_ms`、`total_ms` 只有在 extractor 明确证明与上述 marker/counter 同边界时才可作为一级字段，否则只能作为阶段诊断。simulator 必须输出相同边界的 prefill、decode/TPOT 和 engine E2E；一级误差只比较同边界的 engine 字段。若原生缺少 engine marker、源码证明的 counter 或逐 token 时间戳，一级该指标记为证据不足，不得拿 client 值或 `prompt_eval_ms` 回填。

server 二级分解用于解释服务开销，不改变一级边界：

\[
T_{server}=T_{frontend}+T_{queue}+T_{engine}+T_{response}
\]
\[
T_{client}=T_{request\ transport}+T_{server}+T_{receive\ transport}
\]
\[
TTFT_{client}=T_{frontend}+T_{queue}+TTFT_{engine}+T_{response,first}+T_{transport}
\]

其中 frontend 包括 HTTP/JSON/tokenizer，queue 包括 admission/排队，response 包括采样后编码、SSE/JSON 和 socket 写入；只有 marker 或独立微基准证明边界与重叠关系时才可估计各项。并发或传输重叠时必须按时间线计算，不能把包含的 stage/API/kernel wall 重复相加。HBF 扫描只作用于 engine 内 memory/HBF、KV、kernel 和调度项，server/client 固定开销保持独立。

客户端三级指标固定为（用于完整在线服务诊断，不作为 HBF/GPU 机制准确性的替代标准）：

\[
TTFT_{client}=t_{first\ real\ token}-t_{request\ POST}
\]
\[
E2E_{client}=t_{last\ real\ token}-t_{request\ POST}
\]
\[
TPOT_{client}=\frac{t_{last\ real\ token}-t_{first\ real\ token}}{N_{out}-1},\quad N_{out}>1
\]

首 token 必须是真实非空响应；`[DONE]` 等控制消息不计入 headline E2E，尾部单独记录。客户端排队、HTTP、序列化和流式发送开销不得吸收到 engine 算子成本；只有具备独立 server boundary evidence 时，才进入二级 client 解释。engine 边界内实际发生的采样、CPU/GPU 同步与调度执行属于一级范围，但必须由对应模块解释，不得混入 GEMM 或带宽倍率。

输入 token 数必须由锁定 tokenizer 确认。输出长度、EOS 策略、warmup、缓存状态、并发、batch/ubatch 和线程数写入证据。固定输出（`ignore_eos=true`）与自然 EOS 分开统计；自然 EOS 提前结束不能静默回填 requested length。

并发场景保存每个请求的 request/first-token/last-token boundary、token 时间戳、请求延迟和整个 batch 的起止及 makespan。增加 NVTX/CUPTI/NSYS marker 后须检查观测开销，profiling 运行不能直接冒充无扰动真值。

## 5. 数据集隔离（稳定）

| 数据集 | 可执行操作 | 可证明内容 |
|---|---|---|
| 开发集 | 查看 trace、分析误差、修改机制、拟合允许的底层参数 | 问题定位和机制开发 |
| 验证集 | 比较候选方案、选择版本、决定保留或回滚 | 开发过程中的泛化表现 |
| 最终盲测集 | 版本冻结后只读评分 | 冻结版本的独立验收结果 |

版本选择后，揭盲数据转为开发/回归数据，不能继续作为最终盲测。盲测预测必须先落盘并带时间戳、身份和 SHA，再揭示 native actual。冻结期间不得修改影响结果的代码、profile、微基准库、运行配置、测试清单、指标提取器或聚合脚本；修改即终止当前批次并建立新版本。历史 native actual 是其原始采集环境的独立测量记录；当前工作区变化或 profile 被阻断，不得覆盖、改写或删除该记录，只能生成带新 run_id 和来源关系的派生解析/回放结果。

## 6. 泛化分组（稳定）

正式报告至少按模型、硬件和迁移类型分组，并分别覆盖：

- **Shape 泛化**：未参与开发的 prompt/output/并发/batch 组合，含边界附近和区间外样本；
- **跨模型泛化**：完整留出模型，规模变化与架构变化分开；
- **跨硬件泛化**：先完成规定的通用微基准并冻结硬件模型，再测目标 LLM；
- **组合泛化**：新模型与新硬件或未见 shape 的组合。

相关维度必须联合覆盖，且 kernel 路径一致才可称为插值；域外样本显式标记回退、外推或不支持。随机拆分相关请求不能替代按完整模型、设备或实验组留出。

## 7. 验收阈值与统计（稳定）

对每个场景先计算 native 与 simulator 重复运行的中位数，再计算。一级先分别验收 `TTFT_engine`、`TPOT_engine`、`E2E_engine`；只有一级三项都达到目标后，才进入 server/client 二级、三级结果的机制解释与优化：

\[
e_i=\frac{|median(T_{sim,i})-median(T_{native,i})|}{median(T_{native,i})}\times100\%
\]

一级验收只使用同一 engine boundary 的 `TTFT_engine`、`TPOT_engine`、`E2E_engine`；二级按 server 分解证据报告 frontend/queue/response，三级再报告 `TTFT_client`、`TPOT_client`、`E2E_client`。缺少对应边界证据时只能标记为诊断，不能回填或折算。各层均报告有符号误差、绝对毫秒误差、每次运行误差和测量波动。场景误差 P90 与请求延迟 P90 必须分别命名。

`E2E_engine = TTFT_engine + (N_out-1) * TPOT_engine` 在固定输出策略下具有确定的代数关系，因此同一短输出请求的三项同时达标不能视为三个独立机制已经验证。任何一次性启动或边界残差都必须在未见输出长度上检查其对首 token、首个间隔、后续逐 token 间隔和 E2E 的影响；正式验收至少保留一个不同输出长度的留出场景，并报告每个 token 间隔的误差分布。一次性残差的来源只能写成待验证假设，除非源码和独立 trace 证明其触发条件、生命周期和重复次数。

| 验收项目 | 第一级：engine 目标 | 第二级：server 诊断 | 第三级：client 诊断 |
|---|---:|---:|---:|
| 每个主要分组绝对相对误差中位数 | <10%（TTFT_engine、TPOT_engine、E2E_engine 各自必须满足） | 单独报告各边界，不设抵销一级的阈值 | 单独报告，不设抵销一级的阈值 |
| 每个主要分组绝对相对误差 P90 | ≤20% | 诊断，单独报告 | 诊断，单独报告 |
| 每个主要分组最大场景误差 | ≤30% | 诊断，单独报告 | 诊断，单独报告 |
| 声明支持域内有效预测覆盖率 | ≥95% | 不降低一级覆盖率 | 不降低一级覆盖率 |
| 纳入正式结论的证据完整率 | 100% | 证据存在时报告 | 证据存在时报告 |
| 跨模型目标模型专属时延拟合次数 | 0 | 0 | 0 |
| 跨硬件目标 LLM 端到端时延拟合次数 | 0 | 0 | 0 |

TTFT、TPOT、E2E 在一级 engine 口径下必须分别达标，不能相互抵消。失败、超时、身份不一致、低可信度和证据不足样本必须进入覆盖率与失败统计，不能静默删除或通过更换测试集达标。样本不足或波动过大时结论为“证据不足”。

## 8. 自动优化循环（稳定）

每轮按以下顺序执行，并保留三种对照：纯分析基线、现有校准模型、新机制模型。

1. 从开发集定位问题，明确属于执行图、shape、量化、访存、kernel、调度、计时或测量；
2. 提出单一、可证伪的机制假设；
3. 收集源码、trace、微基准或推导证据；
4. 进行最小修改并跑结构回归；
5. 做消融比较，检查未参与定位的样本和副作用；
6. 在验证集评估，保留或回滚；
7. 若要验收，建立新冻结版本并在揭盲前锁定全部输入。

不得追加场景常数或全局倍率来掩盖系统性误差。

每轮还必须执行因果归因门禁：如果 profile gate、native 重采、extractor、simulator 或成本 profile 在同一轮同时变化，不能把误差变化归因于其中某一个开关。必须固定同一份 native payload 和同一组场景，分别运行纯分析基线、旧 profile（仅诊断）和身份/适用域合格的新 profile 三路 simulator-only 消融，并单独报告 native 分母是否变化。只有在 native identity、计时契约和 extractor 均固定时，才可把差异归因于 simulator 机制；旧 identity 的低误差与当前 identity 的失败只能作为关联事实，不能直接写成“门禁导致误差变大”或“旧 profile 掩盖了缺陷”。

## 9. 交付与停止条件（稳定）

交付必须包括：模型结构和假设、参数来源/单位/有效域/失效条件、冻结清单、数据集划分、原始测量与预测、分组误差、消融、失败案例、回归测试和一键复现入口。每条预测标记来源类型（机制分析、微基准插值、域外外推、条件回放）及是否在验证域内。

缺少第二种硬件或独立验收数据时，交付可执行协议并明确“未验证”。预算耗尽时交付当前最好版本和未完成项，不能放宽阈值、隐藏失败或把规格参数冒充实测验证。最终结论只允许写“通过、失败、尚未验证”三类与证据相符的状态。

## 10. 当前版本与证据等级（动态，原位更新）

| 项目 | 当前状态 | 证据等级/限制 |
|---|---|---|
| simulator 成本模型 | 已完成 prefill chunk 证据 tuple 限制；host/frontend 参数接口默认保持 0；保留 QKV/FFN/KV/lm-head/launch/sync 等阶段建模；B-01/B-04/B-06/B-07 候选全局校准未启用；B-07 已完成 M/T shape 梯度审计；B-08 修复 stage/memory/phase/request 校准开关与 launch 独立门控；B-09 增加 exact 六维 key fail-closed 门禁与 coverage 报告，未命中保持 analytical_fallback | 开发机制版本；B-07 exact 联合键 21/91、B-09 profile exact entries=0，不能将平均 rate 下沉为全域成本 profile，尚未通过独立盲测 |
| native 计时与 extractor | engine/server/client 三层口径已下沉：锁定 semantic binary 的 `server_slot_stats` 源码证明 `prompt_ms=t_prompt_last-t_start`（首 token sample/accept 后）、`predicted_ms=t_gen_last-t_prompt_last`（后续 decode）；payload 保存 `engine_ttft_ms/engine_tpot_ms/engine_e2e_ms` 与 client TTFT/TPOT/E2E | 新 payload 使用 `engine-stage+client-real-token/v3`，engine 字段标记 `counter_proven`；L1 三项当前均未通过 `<10%`；hosttrace 已完成显式逐 token engine marker 的结构验证，但它与 direct binary 分属不同 identity，跨 binary 泛化仍待兼容性证据；旧 `client-real-token/v1` payload 仅 legacy/client/阶段诊断；契约 v3 统一首个/最后真实 engine token 与 counter 语义；fixed 可 replay，自然 EOS 仅条件回放 |
| native evidence | 支持 binary、模型、硬件、CLI、timing contract、extractor、raw log 及 supplemental NSYS/CUPTI/NVTX 文件 SHA 门禁；v3 engine evidence 额外绑定完整 `native` 测量对象 digest；明确区分 provenance（采集来源真实性）与 compatibility（执行路径/性能可迁移性） | 新清单 fail-closed；旧 payload 标记 `legacy_development`；v3 缺少 `engine_timing` proven section、native measurements digest 或字段/source 不完整时不得进入 L1。SHA 相同只说明来源/身份一致，SHA 不同默认阻断正式 profile 应用但不证明性能必然不同；profile 被阻断时，若原始证据未被篡改，native evidence 仍保持其采集环境下的有效性 |
| simulator-only replay | `tools/replay_simulator_from_native.py` 可复用已保存 native actual、token 时间戳、boundary 和 extractor 输出，并输出 engine/server/client 三层口径 | 锁定 semantic binary 的 `counter_proven` server-slot 计时或完整 engine marker 且 contract/identity 一致时才可比较一级 engine TTFT/TPOT/E2E；旧 payload 没有该证据时只能做 client/阶段诊断；profile gate 只控制 profile 应用，不使原始 native actual 失效 |
| 硬件迁移 | 当前只有一套实际硬件 | 跨硬件与规格驱动均未验证 |

## 11. 本轮完成工作（动态，原位更新）

| change_id | 修改或审计 | 证据与回归 | 影响域 | 状态 |
|---|---|---|---|---|
| A-01 | 执行语义守恒回归：lm-head 输出行、prefill chunk、KV owner/read/append、FFN split、token parity | 语义专项 60 passed；无新的语义缺口 | 执行语义 | 保留 |
| B-01 | 修复 NVTX wall `_operator_id` 解析，并审计 CPU/GPU MMQ、lm-head、linear-attention 联合覆盖 | direct-equal train/holdout 各 2,590 unknown/uncovered、3,320 memcpy 排除；成本 coverage blocked | 算子成本 | 阻断；不启用校准 |
| B-02 | phase extractor 支持 textId→StringIds，跳过 instant marker，输出排除原因 | API phase 专项 2 passed；不改变已有 native actual | launch/sync 证据 | 部分完成 |
| B-03 | 在 7 个 attention build_attn* 的 wo 投影后恢复 `kqv_wo` marker，并扩展 attention_output 规则 | 新 binary + Qwen2.5-0.5B train/holdout：kernel matched 2166/2166、3138/3138；stage_unknown=0 | native semantic instrumentation | 保留；作为 B-06 owner 证据基础 |
| B-04 | 修复 `_phase_scopes` 不应把 prefill_begin/decode_begin 生命周期 marker 当作 phase range；用新 trace 重新生成 calibration | `b03_qwen25_semantic_calibration_v3.json`：unknown=0、missing_phase=0、status=covered；candidate replay L6 的 TTFT median 73.0%、TPOT 22.5%、E2E 25.1%，长长并发场景最坏 TTFT 308.6%，故拒绝启用 | phase coverage / 成本 profile | 部分完成；语义证据闭合但 profile 跨 shape 不可泛化 |
| B-05 | 运行独立 llama-bench/ggml CUDA 微基准，覆盖 CPU/GPU、M=1/4/16/64、T=1/8/32、Q4_K_M 与 f16 KV | [kernel_microbench_b05_v1.json](../artifacts/development/kernel_microbench_b05_v1.json)：24/24 组合完成、每格 3 次；固定 binary/model/hardware/threads/batch/ubatch；未读取目标 LLM TTFT/TPOT/E2E | MMQ 形状趋势证据 | 完成采集但阻断启用；llama-bench 只给 aggregate prompt/decode 吞吐，无 semantic owner/kernel wall，不能生成通用算子成本面 |
| B-06 | 在 semantic direct binary 上采集 kernel-level MMQ/QKV/FFN/KV/lm-head owner/correlation train/holdout trace，并建立 operator 成本留出 | [b06_qwen25_calibration_v1.json](../artifacts/development/b06_qwen25_calibration_v1.json)、[b06_qwen25_kernel_operator_coverage_v1.json](../artifacts/development/b06_qwen25_kernel_operator_coverage_v1.json)：4122/4122 与 4384/4384 kernel matched；owner_unknown=0、stage_unknown=0、missing phase/shape=0；operator-wall 留出误差 QKV -2.61%、FFN -11.98%、KV -2.01%、lm-head -1.47%、attention_output -13.55% | MMQ/目标阶段语义与 operator wall | 语义证据完成；shape×dtype×kernel 联合键 462/2048（22.56%），layout 字段缺失，通用成本面阻断，不启用全局 rate |
| B-07 | 按 shape×dtype×layout×kernel 缺口补采 Qwen2.5-0.5B M/T 梯度（M=16/T=4、M=32/T=8、M=64/T=16），合并 B-06 M=8 train，留出 M=32 | [B07_SHAPE_GRADIENT_20260914.md](B07_SHAPE_GRADIENT_20260914.md)、[b07_qwen25_shape_gradient_manifest_v1.json](../artifacts/development/b07_qwen25_shape_gradient_manifest_v1.json)、v2 coverage/calibration；新 trace matched 2320/4384/8272，owner_unknown=0、stage_unknown=0；联合键 train 91、holdout 47、交集 21/47；v2 保留 NVTX phase 后 missing_phase=0；operator-wall 留出 QKV -17.88%、FFN +9.59%、KV -2.47%、lm-head +35.93%、attention_output -16.39%、linear_attention_aux +30.66% | 算子 shape 性能曲面 | 完成采集；通用成本面仍阻断，不启用 profile |
| B-08 | 在 semantic CUDA NVTX marker 增加基于 tensor stride 的 `layout=` 分类，并让 extractor 输出 `semantic_layout`；用新 DLL 采两条 Qwen2.5-0.5B shape trace | [B08_LAYOUT_MARKER_20260914.md](B08_LAYOUT_MARKER_20260914.md)、[b08_qwen25_layout_evidence_v1.json](../artifacts/development/b08_qwen25_layout_evidence_v1.json)；m16/t4 与 m32/t8 共 2320/4384 kernels 全 matched，layout missing=0；contiguous 100%；extractor 回归 6 passed；ggml-cuda.dll SHA 已记录 | 执行语义/算子证据 | marker→extractor 完成；跨 shape 联合 key 462/2166（21.33%），全局成本 profile 仍阻断；新增 token span/gap 摘要仅作调度诊断 |
| C-01 | 审计同到达并发 cohort、host prepare、target submit 和 request-local streaming 图 | prepare=1、submit=1、request_count=2；调度专项 2 passed | 系统调度 | 阻断；无证据支持改边界 |
| B-08-GATE | 修复 `apply_native_calibration` 的校准开关语义：stage/memory/phase/request 不再被 `apply_launch=False` 提前返回丢弃，launch 仍独立显式门控 | `tests/test_native_evidence.py` 47 passed；全量回归 892 passed、1 skipped；Qwen2.5/Qwen3.5 小模型 probe 已生成 prediction-before-native | 校准下沉门控 | 保留；不改变 native identity |
| GATE-IND-REGRESSION | 增加 profile gate 与 native evidence 独立性回归：使用完整 v3 payload 配合不匹配 profile，验证 gate 只阻断 profile 应用，不改写 native evidence 状态 | `tests/test_replay_profile_gate.py` 4 passed；`_validate_native_evidence` 保持 complete，`profile_gate=blocked`，native execution count 仍为0 | provenance/profile 资格边界 | 保留；为后续三路 simulator-only 消融提供回归约束 |
| E-ENGINE-MARKER-GAP-V1 | 扩展 NSYS extractor 的 Engine marker 摘要，分离 token 执行区间、token 间空窗和 request 区间；禁止把空窗机械归因于 kernel | `tests/test_nsys_trace_extract.py` 与 `tests/test_replay_profile_gate.py` 共 14 passed；新增 `engine_marker_summary`，含 `token_intervals_ms`、`inter_token_gaps_ms`、`inter_token_begin_gaps_ms`、`engine_request_interval_ms` 和 `cost_inference=forbidden_without_boundary_and_nonprofiling_validation` | Engine 边界/调度诊断 | 保留；仅作机制证据，不启用成本 profile |
| E-ENGINE-COMPUTE-MARKER-V1 | 在 llama.cpp `decode()` 内增加围住 `llama_decode+sync` 的 `engine_compute_begin/end`，与 sampling token marker 分离；extractor 保留两类边界 | `source/llama.cpp-semantic-patches/tools/server/server-context.cpp` 与实际 semantic checkout 同步；`tests/test_nsys_trace_extract.py`、`tests/test_replay_profile_gate.py`、`tests/test_native_evidence.py`、`tests/test_unified_evidence_manifest.py` 共 76 passed；尚未重建 binary | Engine compute/sampling 边界语义 | 完成源码与结构回归；等待同 binary 重建和小模型 trace 复核，不进入 L1 校准 |
| B-09 | 对 exact `(stage,phase,shape,dtype,layout,kernel_family)` 建立 canonical key 与 fail-closed coverage 审计；缺字段/未命中均 analytical_fallback | `tests/test_native_evidence.py` 47 passed；`b09_exact_operator_gate_v1.json`、`b09_exact_operator_profile_v1.json`、`b09_exact_ablation_prompt8_output8.json`：profile exact entries=0，B-07 holdout 21/47 仅作稀疏证据 | 算子成本适用域 | 完成实现；不启用全局 rate |
| DOC-01-TASK-BRIEF-STRUCTURE | 任务书改为稳定规范、动态看板和历史短索引 | 17 个二级章节；动态章节原位更新 | 推进流程 | 保留 |
| B-11 | 将 engine-first 计时契约下沉到 native comparator、simulator replay 与聚合输出；锁定 binary 的 prompt/eval counter 作为有源码依据的 engine 边界，client 字段独立保留并作为 L3 诊断；server frontend/queue/response 只在边界证据充分时分解；修复 serving replay 对不存在 `sim.trace` 的引用 | `engine_contract_probe_engine_v3.json`：旧 native engine 33.250/18.301/51.551 ms 已失效；当前 identity 重测 native=24.564/13.024/37.588 ms，simulator=34.182/18.820/53.002 ms，L1=+39.15%/+44.50%/+41.01%；旧结果仅保留为历史测量波动记录；`tests/test_native_evidence.py` 47 passed；engine/native/trace 专项 95 passed；契约迁移 replay 校验 valid；全量 892 passed、1 skipped | native/replay 数据契约与计时聚合 | 字段接线完成；误差失败，继续 E-ENGINE-01/E-01 |
| B-12 | 修复 CPU 无 ISA 专用 quantized-dot capability 时丢失 Q4/Q5/Q6 packed-weight dequant/unpack 工作：generic 路径保留 primitive count，按保守 SIMD issue grouping 纳入共享 vector-ALU execution envelope（analytical fallback，待独立 ISA microbench 验证） | `engine_contract_probe_engine_v3.json` simulator engine TTFT/TPOT/E2E 从 7.043/7.017/14.060 ms 更新为 34.182/18.820/53.002 ms；旧 native 33.250/18.301/51.551 ms 与 +2.80%/+2.84%/+2.81% 结果因 identity 重测已失效；当前 native=24.564/13.024/37.588 ms、simulator=34.182/18.820/53.002 ms，L1=+39.15%/+44.50%/+41.01%；无目标场景 native 拟合；`tests/test_cost_models.py` 与 `tests/test_native_evidence.py` 95 passed | CPU quantized GEMM 执行语义/成本 | 旧 native identity 下曾三项达标，但当前 identity 重测三项失败；机制实现保留，准确性结论回滚，仍需独立 shape/model 验证 |
| D-02 | 修复 v3 replay/evidence 门禁：要求 proven engine section 的 fields/source，绑定 `native_measurements_sha256` 到完整 native 实测对象，并在 unified manifest 校验 source/manifest digest 一致；明确 replay 输出 `native_execution_count=0` 与 payload 扫描计数分离 | `engine_contract_probe_engine_v3.json` 与两个 unified manifest 已刷新 digest `e6798efbcd4bfd4aae01e2c6a44d7c86107c76b4f2ce6f7cd3fa745a117dd64a`；native actual 未重测；新增 digest mutation/proven-section 测试；相关专项 62 passed，全量回归 892 passed、1 skipped | L1 证据完整性/防止旧 payload 或伪 native replay | completed-dev；当前 v3 probe replay valid，旧/篡改测量在 fail-closed 条件下拒绝 |

## 12. 尚未解决的差距（动态，原位更新）

当前差距优先按 L1 engine 三项误差与证据覆盖排序；L2 server/L3 client 仅作边界诊断。任何 client 重尾都不能回填到 HBF、KV 或 kernel 成本，任何缺少 engine contract 的记录都不能计入 L1 通过率。

本节只维护 gap 的当前证据、状态和根因假设；执行顺序、输入证据与退出条件唯一维护在第 13 节，避免同一 step_id 在两个动态表中产生冲突。

| gap_id | 指标或分组 | 当前证据 | 根因假设 | 状态 | 下一动作 |
|---|---|---|---|---|---|
| G-01 | TTFT_engine：多模型跨场景 replay | 当前身份一致无校准基线：Qwen3.5 -44.64%、TinyLlama -44.53%、SmolLM2 -30.39%；Qwen3.5/Tiny历史phase低误差因源binary=dd8b...与当前=a2836...不符，全部降级诊断；SmolLM2当前exe/DLL identity已保存但未建立Engine phase校准 | phase API 只能提供独立 launch/sync 语义证据，不能直接替代 Engine wall；需要源码/marker证明边界后再应用 | active / holdout-required | 保留Smol output2 profile作为诊断，等待engine token marker设计；当前binary phase成本不得直接进入L1 |
| G-02 | TPOT_engine/decode：多模型 | 当前无校准：Qwen3.5 +0.85%、TinyLlama -39.49%、SmolLM2 -44.72%；SmolLM2 output2 fixed phase profile v2/v3 的API phase已采集但仅诊断，不直接计入Engine；profile kernel trace 1111/1111 matched | legacy8us launch与跨binary首decode残差禁用；output2 phase API不能证明output4 Engine wall或token边界；当前三条 output=2 trace（warmup=1）TPOT 区间为 16.400/15.204/15.398 ms，最大事件 gap 为 8.552/8.139/8.441 ms；对照 output=4（warmup=2）decode 区间为 4.338/5.287/5.363 ms、最大 gap ≤0.389 ms；强关联于 formal shape/graph cache miss，但仍需 graph lifecycle 与非 profiling 证据区分 graph 更新、同步和 profiling 空窗 | active / engine-boundary-required | 保持Smol phase profile application disabled；先利用 `engine_marker_summary.inter_token_gaps_ms` 区分 token 间调度/同步空窗与 token 执行区间，再按独立证据重评估（当前两条 marker holdout 的 end-to-begin gap 约 15.049/15.269 ms；另有一条独立 output=2 trace，待统一归档） |
| G-03 | 并发 prefill/continuous batching | cohort prepare 仍是诊断字段 | 共享 prefill 依赖、重叠和 launch/sync 临界路径未完整下沉 | active | 用并发 trace 建时间线；不能直接相加重叠 stage wall |
| G-04 | client/server 二级边界 | B-08 probe 的 client TTFT 28.075ms、client E2E 63.489ms；同一 probe 的 simulator engine E2E 37.935ms，但没有 native engine marker 可比；1,829 条 request 的 client TTFT−prompt_eval 中位数约1.50ms、P90约57.54ms，≤10ms仅68.7% | `_emit_parallel_token` 为零推进 marker；admission、input decode、output encode/SSE framing 没有独立 server 证据；不能把 client 开销吸收到 engine 算子 | blocked / secondary-only | 一级 engine 三项未达标前不以 client 优化为主线；后续仅采同 binary 的 server accepted/tokenize/slot/first-sample/socket marker、in-process tokenizer 和空载 admission/排队微基准 |
| G-05 | 原始 GPU trace 与模型专属成本证据 | D-01 manifest v2 已绑定 B-06/B-08/B-10 sidecar、raw trace、extractor、binary 与 SHA；server boundary 仍明确 unavailable | engine 证据清单已闭合，但正式盲测仍需按冻结版本绑定全部输入 | completed-dev / acceptance-pending | 以同一 manifest 生成新冻结版本；缺少 server marker 的记录仍不得扩展 L2/L3 结论 |
| G-06 | 正式独立 engine 验收 | Qwen2.5、Qwen3.5、TinyLlama、SmolLM2均有少量counter-proven开发probe；当前身份无校准L1：Qwen3.5 -44.64%/+0.85%/-16.44%、Tiny -44.53%/-39.49%/-40.86%、SmolLM2 -30.39%/-44.72%/-41.77%；所有历史phase低误差因binary不匹配或边界不足均不计入 | 仍缺五模型、独立冻结、shape/并发留出和完整当前身份Engine marker；证据完整不等于准确性通过 | active / independent-acceptance-required | 保留Smol output2 fixed phase API作为诊断，等待token marker设计完成；不得以phase API直接宣称L1通过 |

## 13. 下一轮计划（动态，原位更新）

当前顺序已从“补 marker”推进到“先固定历史证据与身份语义，再用 marker/counter 证明完整 Engine 边界，最后做同 native 的机制消融”。L1 Engine TTFT、TPOT、E2E 仍是唯一主验收目标；L2 server 与 L3 client 不得替代 L1。固定输出下三项指标存在代数关系，因此下一轮必须同时检查不同输出长度和逐 token 间隔，不能仅以一个短请求的三项误差判断机制成立。binary gate 的结果只改变 profile 资格，不单独解释性能差异；任何误差变化必须先完成同 native 三路对照。

| step_id | 优先级 | 目标 gap | 动作 | 输入证据 | 退出条件 | 状态 |
|---|---:|---|---|---|---|---|
| PROV-01 | 1 | G-05/G-06 | 对历史 native payload、capture manifest、runtime artifact 清单和派生 extractor 结果做不可变来源审计；区分采集时身份、当前文件身份和派生解析身份，禁止用当前 SHA 回写历史采集声明 | 历史 payload、immutable capture manifest、artifact SHA、extractor SHA、run_id 关系 | 每条记录都有 `declared_at_capture`、`capture_manifest_sha256`、native digest 和派生 run_id；身份无法证明的记录降为 provenance unknown，不进入 L1 | active；需先完成证据链审计 |
| E-ENGINE-01 | 2 | G-01/G-02/G-06 | 保持 `server_slot_stats` counter 为 L1 数值来源，并用 `engine_request_begin`、`engine_token_begin/end`、`engine_request_end` marker 证明边界位置；逐 token 对照首 token、首间隔、后续间隔和 response queue 排除；使用 extractor 的 `engine_marker_summary` 单独记录 token span 与 inter-token gap | hosttrace binary、server-context 源码、[`eengine_smollm2_marker_structure_holdout_v1.json`](../artifacts/development/eengine_smollm2_marker_structure_holdout_v1.json)、两条不同 prompt 的 SmolLM2 trace | marker 与 counter 边界关系可复核；marker 不改变 `ggml_time_us`；EXE/DLL capture-time SHA 完整；跨 binary 仅在兼容性证据充分时 allowlist | completed-dev；结构证据通过，边界等价性与性能兼容性仍待验证 |
| E-01 | 3 | G-01/G-02 | 在同一份 native actual 上做纯分析、旧 profile（仅诊断）和身份/适用域合格新 profile 三路 simulator-only 消融；分别报告首 token、首个间隔、后续逐 token 间隔、TTFT/TPOT/E2E，并加入不同 output 长度留出 | 当前 native actual、hosttrace marker trace、profile provenance/compatibility gate、输出长度 holdout | 每个主要模型分组三项 L1 中位数<10%，P90/最坏/覆盖率/证据门槛同时满足；一次性残差必须先证明触发条件、生命周期和重复次数；优先交叉 warmup_output=1/2 与 formal output=2/4，并加入 graph lifecycle/ggml graph plan marker，并将后续 token marker 放到 `decode()` 的 `llama_decode+sync` 前后以覆盖完整 token 关键路径；不得直接当稳定每步成本 | queued；等待 PROV-01 与 E-ENGINE-01 |
| B-12 | 4 | G-01/G-02 | 保留 CPU generic quantized GEMM dequant/unpack 机制，但用当前 direct binary 的重复 native probe、未见 shape 和线程预算重新验证；旧 24.564/13.024/37.588 与 30.369/15.473/45.842 仅作为不同运行的历史波动，不合并成单一真值 | 当前 direct binary payload、B-12 结构验证、CPU shape holdout | 重复测量的中位数与波动先稳定；机制改善必须在未参与定位的 shape 上成立，否则保留失败状态 | active；结构机制保留，准确性未达标 |
| C-02 | 4 | G-03 | 用 marker 和并发 raw trace 重建 cohort lowering、prefill/decode overlap、launch/sync 关键路径和 batch makespan；包含请求级 token 时间线，不把包含关系重复相加 | 并发 marker trace、依赖图、Engine request boundary | 输出每请求 boundary、token timestamp 和批次 makespan；无证据不改调度成本 | queued |
| B-10 | 5 | G-04 | 在 L1 之后继续诊断 server/client 边界；保留 host marker 与 Engine marker 分离，不能把 response queue 或 socket 开销吸收到 Engine | hosttrace handler/slot/response marker、空载微基准 | server/client 仅作为二级诊断，边界与重叠证据完整后再报告 | secondary-only |

## 14. 冻结与 replay 规则（动态，原位更新）

### 14.1 冻结清单

冻结必须同时锁定：模型/GGUF、native binary 及采集时实际加载的运行时依赖（`runtime_artifacts`，必须保存采集时实际加载的全部 EXE 与全部 DLL 的路径、大小和 SHA；如果运行时确实只加载一个 DLL，则集合自然只包含一个 DLL，不得以任意目录扫描结果替代实际加载集合）、tokenizer、硬件指纹、CLI、prompt/output policy、batch/ubatch/threads/GPU layers、**engine timing contract（一级）**、server boundary evidence contract（二级分解）、client timing contract（三级诊断）、planner、cost profile、微基准数据库、extractor、聚合脚本、测试清单及各自 SHA。执行前、恢复执行时和结束后都要 fail-closed 校验；文件不存在、SHA 为空、身份字段为 `None`、实际加载依赖集合缺失/不一致或硬件指纹非实际环境时不能通过。manifest 必须明确 `validation_scope`：`live_capture` 需要现场文件逐项复核，`historical_capture` 只保留采集时声明的 SHA 与 native digest，并明确当前文件漂移不代表当前冻结通过。manifest 内部字段自洽只能证明 `validation=valid`，不等于 `provenance=verified`；后者还需要不可变外部归档、父 raw digest、签名或其他可信来源来证明采集声明未被事后改写。历史 capture manifest、`declared_at_capture`、`capture_manifest_sha256` 和 native digest 一旦发布不可原位改写；需要修正时只能创建新的派生 manifest/run_id，并保留来源关系。`capture_manifest_sha256` 应同时引用不可变的父 raw digest 或签名；仅重新计算当前文件不能证明历史声明未被事后改写。没有 engine contract 或其 SHA 的冻结版本不得进入一级验收。

### 14.2 Native evidence 保存

每条 native 记录至少保存：engine `request_begin/first_token/last_token/request_end` boundary 或源码证明的 `server_slot_stats` counter、每 token 原始 engine 时间戳（counter 版本保存其计数语义）、server frontend/queue/response boundary（若有）、客户端 boundary、客户端与引擎指标、实际输出 token 数、stop/truncated/finish reason、原始 llama.cpp log、采集时的 `runtime_artifacts`（EXE 与实际加载 DLL 的路径、大小和 SHA，且在采集前后复核未变化）；如采集 NVTX/CUPTI/NSYS，则保存原始文件路径、大小、SHA，以及对应 extractor 输出和脚本 SHA。只有 extractor 能从 marker 或已锁定 counter 重建三项 engine 指标时，记录才具备一级证据；当前 server log 不能被称为 NVTX/CUPTI 原始 trace。`declared_at_capture`、`capture_manifest_sha256` 和 native measurements digest 必须来自采集时或不可变归档；不得以当前工作区文件重新定义历史来源。若原始证据完整但 profile provenance/compatibility 不合格，native 记录仍可作为其采集环境下的有效实际测量，只禁用该 profile。

### 14.3 何时允许 simulator-only replay

当相对于被预测 native payload 的 binary、模型、硬件、CLI、prompt/output policy、**engine、server、client 三套计时契约**和 extractor 全部不变，且 native evidence 完整一致（包括 `native_measurements_sha256` 对完整 native 实测对象校验）时，只修改 simulator 成本模型可以复用 native actual，直接重跑 simulator。预测 shape 使用冻结的 requested output policy，不能使用 native 实际输出长度反向构造。算子校准仅在完整 `(stage,phase,shape,dtype,layout,kernel_family)` exact key 命中时适用；缺字段或未命中保持 `analytical_fallback`，不跨 shape 使用平均 rate。没有 proven engine section、native measurement digest 或 digest 不一致的旧 evidence 可以复用作 client/阶段诊断，不能生成一级 engine 误差。binary/DLL SHA 不同首先表示来源或兼容性尚未证明，默认阻断正式 profile 应用，但不证明性能必然不同，也不使原始 native actual 失效；只有源码路径、构建参数和独立微基准证明等价，并在报告中标记兼容性范围后，才允许跨 binary 复用 profile。replay 工具的 `native_execution_count` 永远为 0，payload 扫描数与 simulator replay 数独立统计。

## 15. 最近验证结果（动态，原位更新）

本表的 headline 只记录 L1 engine 口径；server/client 数值保留为二级、三级诊断，不能抵销 L1 失败。当前最新 GPU Engine probe 已升级为 v4，观察到首个 decode invocation 的候选一次性启动残差；根因、触发条件、重复次数和可迁移性仍待源码、非 profiling native 与不同 shape/output 验证，结果仍属于开发证据。

| run_id | dataset | scenarios | 结果 | 覆盖率/证据完整率 | 结论 |
|---|---|---:|---|---|---|
| B03-SEMANTIC-WO-TRACE | 新 semantic binary 的 Qwen2.5-0.5B train/holdout 开发 trace | 2 次采集 + 2 次 extractor | kernel 全部 matched；stage_unknown=0；`kqv_wo` -> attention_output；成本 profile 仍因 missing_phase blocked | owner 语义 100%；phase coverage 不完整 | 部分完成；不得用于正式时延校准 |
| B04-PHASE-COVERAGE-REPLAY | 新 trace 的 phase scope 修复与候选成本 profile 回放 | 2 条小模型 trace；L6 candidate replay | calibration coverage=covered；但候选 profile 在留出场景 TTFT/E2E 误差仍高，拒绝保留 | owner/phase 证据有效；成本泛化失败 | 失败；不启用该 profile |
| C01-SCHEDULER-AUDIT | 同到达并发调度开发审计 | 2 tests + 2-request check | cohort prepare=1、target submit=1、request_count=2；request-local streaming 图保持原语义 | 无新的边界证据 | 阻断/证据不足 |
| B01-OPERATOR-COVERAGE-V1 | 既有 semantic train/holdout trace 开发审计 | 4 对 trace | 196 个匹配联合键；完整 coverage blocked，unknown owner 和 memcpy 排除仍存在 | 证据完整率不足 | 失败/证据不足，不得启用 MMQ |
| B05-KERNEL-MICROBENCH-V1 | llama-bench/ggml 独立微基准 | 24 组合（CPU/GPU×M/T），每格3次 | 所有组合完成；GPU/CPU aggregate prompt/decode tokens/s 已保存；不含目标LLM端到端时延 | 不能映射到单算子 kernel wall/owner | 完成采集，blocked；仅供趋势诊断 |
| B06-KERNEL-OWNER-TRACE-V1 | semantic direct binary 的 Qwen2.5-0.5B kernel-level train/holdout | 2 条 profile、2 次 extractor、8/58 prompt tokens、固定 output=8；4122/4122 与 4384/4384 kernel matched | owner_unknown=0、stage_unknown=0、phase/shape 缺失=0；operator-wall 留出误差 QKV -2.61%、FFN -11.98%、KV -2.01%、lm-head -1.47%、attention_output -13.55%；含 shape 联合键交集 22.56%，layout 字段缺失 | 语义证据完整；联合性能曲面覆盖不足 | 完成采集，blocked；仅供开发期 operator diagnostics |
| B08-GATE-PROBE | 修复校准开关后，对 Qwen2.5-0.5B/Qwen3.5-0.8B 各 1 个开发 probe 运行 prediction-before-native；已报告的 TTFT/TPOT/E2E 是 client/阶段混合诊断，不能作为 engine 一级误差；native engine marker 尚未绑定到 payload | 小模型开发验证；不纳入盲测，不重测27B | 一级 engine 证据不足；继续 E-ENGINE-01 |
| B07-SHAPE-GRADIENT-V2 | Qwen2.5-0.5B M/T shape 梯度与独立 M=32 留出 | train 合并 prompt/output 8/8、16/4、64/16，holdout 32/8；14714/4384 kernel 全 matched；修复 merged trace 丢失 NVTX phase | exact 联合键 21/91（holdout 21/47）；按 holdout kernel event 加权覆盖 3437/4384=78.40%；event-rate 误差多数 ±5%，但 operator-wall lm-head +35.93%、linear_attention_aux +30.66% | shape 性能面和布局证据不足 | 完成采集，blocked；不启用候选 profile |
| B08-LAYOUT-MARKER-V1 | semantic CUDA marker 显式 layout 与 extractor 验证 | `b08_qwen25_layout_m16_t4_v1`、`m32_t8_v1`：2320/4384 kernels 全 matched，layout missing=0，contiguous=100%；extractor `semantic_layout` 回归 6 passed；新 ggml-cuda.dll SHA 已固定 | layout 语义链路闭合；跨 shape exact key 462/2166=21.33%，成本 profile 仍阻断 | 完成但阻断 |
| B09-EXACT-KEY-GATE | B-07/B-08 operator coverage 的 exact 六维适用域审计 | [b09_exact_operator_gate_v1.json](../artifacts/development/b09_exact_operator_gate_v1.json)、[b09_exact_operator_profile_v1.json](../artifacts/development/b09_exact_operator_profile_v1.json)、[b09_exact_ablation_prompt8_output8.json](../artifacts/development/b09_exact_ablation_prompt8_output8.json)；profile exact entries=0，train/holdout 联合键 91/47，交集 21；native evidence 47 passed | 缺字段/未命中均 analytical_fallback；禁止跨 shape 平均 rate；未启用全局校准 | 完成门禁；等待 E-01 验证 |
| E01-B09-ABLATION | simulator-only 对 Qwen2.5-0.5B 8/8、16/4、32/8 三个开发 shape 做纯分析/旧 profile/exact gate 三路消融 | [e01_b09_ablation_v1.json](../artifacts/development/e01_b09_ablation_v1.json)、[b09_exact_ablation_prompt8_output8.json](../artifacts/development/b09_exact_ablation_prompt8_output8.json)；native_execution_count=0；当前 B09 exact entries=0，三路保持分析回退，未宣称误差改善 | 验证集机制消融 | 证据不足；继续 B-10 host boundary 与后续独立验证 |
| B-11-ENGINE-CONTRACT-PROBE | 将 L1 engine、L2 server、L3 client 三层口径下沉到 comparator、replay 与聚合，并修复 serving replay 的边界读取 | [engine_contract_probe_engine_v3.json](../artifacts/development/engine_contract_probe_engine_v3.json)：锁定 semantic binary `server_slot_stats` counter_proven；旧 native engine TTFT/TPOT/E2E=33.250/18.301/51.551ms（identity重测后失效）；当前 native=24.564/13.024/37.588ms，simulator=34.182/18.820/53.002ms；契约迁移 replay 校验 valid；client 字段单独保留；`tests/test_native_evidence.py` 47 passed，engine/native/trace 专项 95 passed | engine 计时契约/回放 | 初始成本模型一级三项失败，转入 B-12 机制修复 |
| B-12-CPU-GENERIC-DEQUANT | 修复无 ISA 专用 quantized-dot capability 时 Q4/Q5/Q6 packed-weight dequant/unpack 工作被丢弃的问题；generic path 按 primitive count 与保守 SIMD issue grouping 纳入共享 vector-ALU execution envelope（analytical fallback，待独立 ISA microbench 验证） | [engine_contract_probe_engine_v3.json](../artifacts/development/engine_contract_probe_engine_v3.json) simulator engine TTFT/TPOT/E2E=34.182/18.820/53.002ms；旧 native=33.250/18.301/51.551ms 与 +2.81%/+2.84%/+2.81% 仅为历史结果，identity重测后失效；当前 native=24.564/13.024/37.588ms，L1=+39.15%/+44.50%/+41.01%；未读取待预测场景时延拟合；成本与证据专项 95 passed；全量 892 passed、1 skipped | CPU quantized GEMM 成本 | 旧 identity 下曾三项达标；当前 identity 重测三项失败（+39.15/+44.50/+41.01%），仅保留执行语义机制，等待当前证据与未见 shape/GPU/模型留出 |
| D-01-EVIDENCE-MANIFEST-V2 | 将 engine/server/client 边界、B-06/B-08/B-10 sidecar、raw trace、extractor、binary 与各 SHA 绑定到统一清单；对历史声明做重新审计并生成派生 manifest；原始 capture 声明不可原位改写，无法证明来源时降级 provenance unknown | [`engine_contract_probe_evidence_manifest_v2.json`](../artifacts/development/engine_contract_probe_evidence_manifest_v2.json)：schema v2，15/15 artifact SHA 重新计算，timing contract v3，validation=valid；engine counter_proven，server boundary unavailable 明确保留 | 证据完整性/冻结前置 | 开发清单完成；正式盲测仍需独立冻结版本与完整 server evidence |
| B-12-CPU-SHAPE-HOLDOUT | 对 generic/ISA CPU quantized GEMM 做未见 `(M,K,N)`、prompt/output 和 threads 结构验证，检查 dequant primitive、issue envelope、capability gate 与 segment evidence | [`b12_cpu_shape_holdout_v1.json`](../artifacts/development/b12_cpu_shape_holdout_v1.json)：4 个未见 shape，另含 prompt/decode/thread sweep；`dequant_scales_with_m_k_n`、`generic_fallback_retains_primitive_count`、`generic_issue_envelope_counted_once`、`generic_instruction_accounting_exact`、`prompt_shape_monotonic`、`output_length_linear_decode_steps`、`thread_budget_non_increasing_service` 均 true；native_execution_count=0 | CPU 执行语义/成本结构 | simulator-only structural pass；不构成 native accuracy 或跨模型通过，ISA primitive timing 仍待独立 microbench |
| E01-B12-ABLATION | 对 native identity 重测后的当前 probe 保留纯分析与 B-12 generic dequant 机制对照 | [`e01_b12_ablation_engine_v1.json`](../artifacts/development/e01_b12_ablation_engine_v1.json) 的旧 +2.80%/+2.84%/+2.81% 仅对应已失效 native measurement；当前重测 native=24.564/13.024/37.588ms、simulator=34.182/18.820/53.002ms，L1=+39.15%/+44.50%/+41.01%，native_execution_count=0 | 旧误差改善不能用于当前identity结论；B-12仍是执行语义修复但成本/波动需当前binary独立证据 | 当前开发probe失败，继续校准/波动分析，禁止保留旧+2.8%作为通过依据 |
| E-ENGINE-CANDIDATE-AUDIT | 对 development payload 做 Engine v3 可复用性筛选，按完整 identity、counter/marker、timing contract 和 extractor SHA fail-closed 分类 | [`gpu_engine_holdout_candidate_audit_v2.json`](../artifacts/development/gpu_engine_holdout_candidate_audit_v2.json)：扫描 318 个 artifact；Qwen2.5 88 条中仅 2 条 replayable_engine_v3；Qwen3.5 82、Qwen3.8 40、SmolLM2 54、TinyLlama 54 条均为 partial_engine_tpot_only 或 stale/invalid；未读取 native 时延拟合 | Engine evidence 覆盖审计 | 只有 Qwen2.5 两条可进入 L1 replay；其他模型必须补齐 engine evidence，不能用 client/阶段值替代 |
| E-ENGINE-GPU-AUDIT | 对候选 payload 进行 v3 replay 筛选并复用身份一致 native actual | [`gpu_engine_holdout_replay_audit_v4.json`](../artifacts/development/gpu_engine_holdout_replay_audit_v4.json)：8 个候选 payload 中 2 条完整 v3 replay 有效（CPU 与 GPU 各 1 条）；`native_execution_count=0`，其余因旧契约、旧 extractor 或 SHA 不一致 fail-closed，不能进入 L1 | GPU Engine evidence/回放 | GPU 首次 decode 残差机制接入前基线为 -2.54%/-32.15%/-25.74%，接入后见 E-ENGINE-GPU-STARTUP-V1 为 -2.54%/+8.95%/+6.47%；未建立通用校准 |
| E-ENGINE-GPU-TRACE-V1 | 对 Qwen2.5-0.5B GPU prompt=2/output=4 采集 semantic CUDA/NSYS operator trace 与 API/memcpy evidence | [`gpu_semantic_qwen25_p2_o4_v1.json`](../artifacts/development/gpu_semantic_qwen25_p2_o4_v1.json)、[`gpu_semantic_qwen25_p2_o4_v1.trace.json`](../artifacts/development/gpu_semantic_qwen25_p2_o4_v1.trace.json)：4590 events，NVTX 2342，kernel owner 覆盖 QKV/FFN/KV/lm-head/attention_output；CUDA API 含 launch/synchronize，raw nsys/sqlite/kernel/api/memcpy 均保留；该 trace 不读取或拟合 Engine 总时延 | GPU operator/API 语义证据 | 作为 decode gap 诊断输入；尚未生成跨 shape 通用 profile，未知 kernel 77 条和 phase marker 缺失保持显式 |
| E-ENGINE-GPU-STARTUP-V1 | 从独立 Qwen2.5 RTX5080 semantic trace 观察首个 decode invocation 的一次性启动残差候选，身份门控后暂作为 phase-boundary 诊断；profile 应用前新增 EXE+依赖 DLL capture-time gate | [`gpu_decode_startup_audit_v1.json`](../artifacts/development/gpu_decode_startup_audit_v1.json)、[`gpu_decode_startup_profile_qwen25_v1.json`](../artifacts/development/gpu_decode_startup_profile_qwen25_v1.json)、[`profile_binary_provenance_audit_v1.json`](../artifacts/development/profile_binary_provenance_audit_v1.json)：B06/B07 四条独立 trace 的首 phase−后续 phase 中位差分别 10.973/8.407/9.027/11.205ms；8.407471ms 只是四条 trace 的最小值估计，不是保守上界或通用参数；NVTX/API wall 可能包含异步等待和 profiling 开销，尚未证明 CUDA Graph 根因、每请求触发次数或跨 binary 可迁移性 | GPU decode 首次 invocation 启动空窗与binary依赖身份 | 旧 profile 在当前 a2836... binary 下不再自动应用；此前 TTFT -2.54%/+8.95%/+6.47% 仅保留为旧身份开发诊断，待非 profiling native、不同 output/warmup/shape 和当前 binary+DLL 证据验证后再评估；不向后续每步 decode 自动收费 | identity gate completed-dev；当前候选残差仅诊断，Qwen2.5 GPU 单probe不作为当前版本通过依据 |
| D-02-REPLAY-COUNT-CORRECTION | 审计历史 `simulator-replay-from-native` 报告中的计数语义；将 `engine_first_replay_probe.json`（81）、`gpu_engine_holdout_replay_audit_v1.json`（62）及 `artifacts/multimodel_next/simulator_replay_prefill_fix_v1.json`（135）的误标 `native_execution_count` 归零，保留旧值为 `legacy_native_execution_count_claim`，并补 `payload_scan_count/native_actual_reused_count/simulator_execution_count` | 三份历史 replay artifact 现均声明 `native_execution_count=0`；新增全目录回归断言，simulator replay schema 不得报告 native 执行 | replay 计数真实性 | completed-dev；不改变 native actual 或各行 fail-closed 结论 |
| GATE-IND-REGRESSION | profile gate 与 native evidence 独立性回归 | 完整 v3 payload + 不匹配 profile：`_validate_native_evidence` 为 `complete`，`profile_gate=blocked`；`tests/test_replay_profile_gate.py` 4 passed；native execution count=0 | 验证 gate 不会把原始 native actual 标成 invalid | completed-dev；支持后续同 native 三路消融 |
| E-ENGINE-MARKER-GAP-V1 | Engine marker 摘要的 token span/gap 分离回归 | [`eengine_smollm2_marker_gap_diagnosis_v1.json`](../artifacts/development/eengine_smollm2_marker_gap_diagnosis_v1.json)：p2/o2 与 p4/o2 的 end-to-begin gap 为 15.049/15.269 ms、begin-to-begin gap 为 15.204/15.399 ms，token span 为 0.154/0.086 与 0.130/0.061 ms；`engine_marker_summary` 显式禁止直接成本推断；相关专项 13 passed | 为 SmolLM2 -72% TPOT 的调度/同步空窗假设提供结构化证据 | completed-dev；等待同 identity、不同 warmup/output shape 的非 profiling 验证；当前三条 output=2 trace 均复现 8.139–8.552 ms 最大事件 gap，output=4 对照 ≤0.389 ms |
| E-ENGINE-COMPUTE-MARKER-V1 | compute marker 与 sampling marker 分离回归 | `engine_compute_begin/end` 已纳入 extractor 的 request marker 输出；相关专项 76 passed；全量回归 892 passed、1 skipped；尚未有新 binary trace | 证明 decode graph+sync 可被单独观测，避免把 token marker body 当完整 token wall | completed-dev；等待 binary 重建与 trace 复核 |
| E-ENGINE-SMOLLM2-V1 | 当前binary下采集SmolLM2 fixed output=2 phase API微基准，capture-settle-s=0.5，保存exe及全部15 DLL前后SHA；profile/compare identity一致但分开capture | [`eengine_smollm2_p2_o2_profile_v2.json`](../artifacts/development/eengine_smollm2_p2_o2_profile_v2.json)、[`eengine_smollm2_p2_o2_profile_v3.json`](../artifacts/development/eengine_smollm2_p2_o2_profile_v3.json)、[`eengine_smollm2_p2_o2_api_phase_v2.json`](../artifacts/development/eengine_smollm2_p2_o2_api_phase_v2.json)、[`eengine_smollm2_p2_o2_api_phase_v3.json`](../artifacts/development/eengine_smollm2_p2_o2_api_phase_v3.json)、[`eengine_smollm2_p2_o2_trace_v2.json`](../artifacts/development/eengine_smollm2_p2_o2_trace_v2.json)、[`eengine_smollm2_p2_o2_trace_v3.json`](../artifacts/development/eengine_smollm2_p2_o2_trace_v3.json)：两次phase均prefill1/decode1、1111/1111 kernel matched、unknown=0；API总墙均值prefill=2,324,457ns/decode=1,971,225.5ns（两次独立fixed-output=2 capture）；profile [`smollm2_semantic_calibration_phase_v2.json`](../artifacts/development/smollm2_semantic_calibration_phase_v2.json)含两份API SHA、当前exe/DLL SHA、`calibration_basis=phase_api_fixed_output2_train_holdout`，manifest strict v3 valid（26 artifacts、5 sidecars） | output2 API launch/sync phase值只作语义诊断，不能直接当Engine wall或校准output4；SmolLM2 fixed output4无校准L1=-30.39%/-44.72%/-41.77%，profile仅用于等待engine token marker的证据准备 | 当前identity phase采集完成；在token marker/源码边界完成前保持profile disabled、L1 accuracy blocked |
| E-ENGINE-MULTIMODEL-QWEN35-TINY-V1 | Qwen3.5/Tiny phase证据身份审计与profile settle修复 | 旧phase源binary=dd8b...与当前native=a2836...不一致，Qwen3.5 -9.39/+0.85/-3.05、Tiny +3.37/-3.68/-1.77均降级cross-binary diagnostic；[`profile_binary_provenance_audit_v1.json`](../artifacts/development/profile_binary_provenance_audit_v1.json)进一步确认旧profile缺capture-time EXE+DLL manifest；当前无校准基线仍分别 -44.64/+0.85/-16.44 与 -44.53/-39.49/-40.86；Tiny v2 current trace 1911/1911 matched、prefill1/decode3、unknown=0，仅作语义诊断 | 已撤销跨binary低误差通过表述；replay profile gate 已 fail-closed，默认phase禁用，等待当前身份独立phase train/holdout |
| E-ENGINE-SMOLLM2-MARKER-V2 | 重建带 engine boundary marker 的 hosttrace binary（最终 EXE SHA `0e747bd3...`，GGML_CUDA_NVTX=ON、LLAMA_SERVER_HOST_TRACE=ON）并在同一 hosttrace identity 下采集 SmolLM2-1.7B prompt=2/output=2 与 prompt=4/output=2 两条小型语义 trace；不重测27B | [`eengine_smollm2_marker_p2_o2_profile_v2.json`](../artifacts/development/eengine_smollm2_marker_p2_o2_profile_v2.json)、[`eengine_smollm2_marker_p2_o2_trace_v2.json`](../artifacts/development/eengine_smollm2_marker_p2_o2_trace_v2.json)、[`eengine_smollm2_marker_p4_o2_profile_v1.json`](../artifacts/development/eengine_smollm2_marker_p4_o2_profile_v1.json)、[`eengine_smollm2_marker_p4_o2_trace_v1.json`](../artifacts/development/eengine_smollm2_marker_p4_o2_trace_v1.json)、[`eengine_smollm2_marker_structure_holdout_v1.json`](../artifacts/development/eengine_smollm2_marker_structure_holdout_v1.json)、[`eengine_smollm2_marker_p2_o2_compare_v2.json`](../artifacts/development/eengine_smollm2_marker_p2_o2_compare_v2.json)、[`eengine_smollm2_marker_p2_o2_manifest_v2.json`](../artifacts/development/eengine_smollm2_marker_p2_o2_manifest_v2.json)：两条 trace 均输出 `engine_request_begin=1`、`engine_token_begin/end=2/2`、`engine_request_end=1`，token_index=1,2，顺序与 begin/end 平衡通过；两条 trace identity（EXE+15 DLL、模型、runtime/hardware）一致；manifest strict v3 valid，绑定22个artifact（含16个capture-time runtime artifacts）；`engine_request_end` 已移至 `process_token` 返回 false 且 `send_final_response` 之前，不包含 response queue | 新 marker 仅用于覆盖 `server_slot_stats` 相邻的 engine request/token 边界，不改变 `ggml_time_us` 计时逻辑；prompt=2 与 prompt=4 只作为边界结构 train/holdout 证据，不读取或拟合 output4 目标总时延，也不作正式泛化准确性结论 | marker/extractor/manifest 结构完成；与 direct binary 身份分离，当前两条 hosttrace 结构证据通过，需继续 phase成本与准确性验证 |
| B10-HOST-TRACE-V1 | `LLAMA_SERVER_HOST_TRACE` 开发 binary 的 Qwen2.5-0.5B NSYS 采集 | [b10_qwen25_host_boundary_evidence_v1.json](../artifacts/development/b10_qwen25_host_boundary_evidence_v1.json)：http.handler.begin/end=2/2、slot.launch=1、response.queue=3、json.tokenize=0、sink.write=0；无 CUDA kernel/memcpy 表，不能生成 host 成本 profile | 部分完成；host 参数仍为0 |
| HOST-BOUNDARY-AUDIT | 1,829 条已有 native request 的 client/阶段边界诊断（二级） | Δhost=client TTFT−prompt_eval 中位数约1.50ms、P90约57.54ms；≤10ms仅68.7%；B-08 两条 gap 24.2/21.1ms；没有 native engine marker | 重尾主机调度/排队/交接未建模；禁止吸收到 GEMM，也不能用来否定 engine 机制准确性 | 二级证据不足；待一级 engine 三项完成后进入 B-10 |
| REGRESSION-20260914 | 全量代码回归与 profiler/phase/identity gate 机制专项 | 最近全量 **892 passed，1 skipped**（150.39s）；此前 profiler settle/phase provenance/semantic calibration 专项 71 passed，identity gate/compare/manifest 专项历史记录 65/57 passed；py_compile通过 | 运行中 pandas/pyarrow/ortools 出现 Windows `0xc0000139` 导入噪声，但 pytest 最终退出码为 0；新增 identity gate 后 manifest 重建与 compare 仍通过 | 最新全量与相关专项通过；不能替代跨模型/shape Engine 泛化验收 | 不能替代泛化验收 |

## 16. 历史实验索引（稳定格式，短表）

| 版本/文件 | 用途 | 结果状态 | 当前用途 |
|---|---|---|---|
| `blind_generalization_freeze_v3.json` | 5 模型完整 135-cell 初筛 | 结构有效记录 402/405；存在身份异常与 TPOT 缺失 | 历史诊断 |
| `blind_generalization_freeze_v4.json` | Qwen3.8-27B 9 组合开发筛查 | 27/27 有效；TTFT/TPOT/E2E 目标均失败 | 开发证据 |
| `blind_generalization_freeze_v4_l9.json` | 五模型统一 9 组合首版 | 四模型 GPU layers 身份错误 | invalid 历史证据 |
| `blind_generalization_freeze_v4_l9_r2.json` | 修复 GPU layers 后重测 | 135/135 有效；误差目标仍失败 | 开发/验证对照 |
| `blind_generalization_freeze_v4_l9_r3.json` | prefill fix 后新批次 | 67/135 后停止 | 未完成实验，不纳入统计 |
| `simulator_replay_prefill_fix_v1.json` | 复用旧 native evidence，只重跑 simulator | 135/135 replay；误差改善但仍失败 | 开发机制对照 |
| `b06_qwen25_kernel_operator_coverage_v1.json` | semantic direct kernel owner/correlation train/holdout 开发证据 | 4122/4122、4384/4384 matched；shape 联合键 22.56%，通用成本面阻断 | B-07 shape 梯度补采输入 |
| `b07_qwen25_operator_coverage_v2.json` | M/T shape 梯度联合覆盖审计 | 21/91 exact key 相交；holdout event 加权覆盖 78.40%；候选 profile 阻断 | B-08 exact-key 适用域门禁输入 |
| `b08_qwen25_layout_evidence_v1.json` | 显式 layout marker/extractor 小模型验证 | layout 缺失 0；contiguous 100%；跨 shape exact key 21.33%，全局 profile 阻断 | B-09 exact-key 门禁输入 |

## 17. 任务书维护规则（稳定）

- 稳定章节只在规则本身改变时修改；不得把每轮实验记录追加到稳定章节。
- 动态章节（第 10–15 节）每轮**原位替换**为当前事实：版本、差距、下一轮顺序、冻结/replay 状态和最近验证结果必须反映最新代码与证据。
- 历史实验只新增或修正第 16 节短表中的一行，不复制完整报告内容；详细原始数据仍放在 `artifacts/`，分析报告放在独立文档。
- 每次代码、profile、冻结清单、微基准、extractor 或测试口径修改后，先更新动态章节，再运行验证并回填结果；未达标必须明确写“失败”或“未验证”。
