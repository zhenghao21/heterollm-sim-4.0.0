# 仿真器机制建模、可信度与泛化优化任务书

> **文档状态（动态）**：当前任务仍未达到第一阶段验收阈值；验收主口径已更正为 engine-first，client 仅作二级诊断。本文件是唯一的推进规范；稳定章节定义不可变的规则，动态章节在每轮修改后原位替换。历史实验只保留索引，不在此重复展开。
>
> **适用项目**：`F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0`
>
> **当前版本**：固定131格native的自动优化循环已启动；v2为冻结失败基线。第一轮只修源码资格约束下的hybrid混合批次调度，第二轮独立检查CPU IQ panel解包复用。当前尚未取得新候选误差结果。

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
7. 原位更新任务书的结构性结论、验证结果、退化与下一轮顺序；
8. 每轮优化完成后立即建立本地Git提交，并推送到当前GitHub上游分支，记录提交SHA与推送结果；不得把多轮已完成修改长期积压为一个最终提交；
9. 若要验收，建立新冻结版本并在揭盲前锁定全部输入，关联对应代码提交和内容哈希。

不得追加场景常数或全局倍率来掩盖系统性误差。

版本控制规则自用户本次要求起执行：提交包含该轮代码、相关测试、任务书和必要的小型证据；在途子代理的未完成修改及其他独立工作保持隔离，不整体暂存工作树。GGUF、DLL/可执行文件、海量raw trace和native原始数据继续保留本机，不加入常规推送。失败或回滚也保留可追溯的结论；推送失败时保留本地提交和失败原因，恢复网络后重试，禁止强制推送或覆盖远端历史。

每轮还必须执行因果归因门禁：如果 profile gate、native 重采、extractor、simulator 或成本 profile 在同一轮同时变化，不能把误差变化归因于其中某一个开关。必须固定同一份 native payload 和同一组场景，分别运行纯分析基线、旧 profile（仅诊断）和身份/适用域合格的新 profile 三路 simulator-only 消融，并单独报告 native 分母是否变化。只有在 native identity、计时契约和 extractor 均固定时，才可把差异归因于 simulator 机制；旧 identity 的低误差与当前 identity 的失败只能作为关联事实，不能直接写成“门禁导致误差变大”或“旧 profile 掩盖了缺陷”。

## 9. 交付与停止条件（稳定）

交付必须包括：模型结构和假设、参数来源/单位/有效域/失效条件、冻结清单、数据集划分、原始测量与预测、分组误差、消融、失败案例、回归测试和一键复现入口。每条预测标记来源类型（机制分析、微基准插值、域外外推、条件回放）及是否在验证域内。

缺少第二种硬件或独立验收数据时，交付可执行协议并明确“未验证”。预算耗尽时交付当前最好版本和未完成项，不能放宽阈值、隐藏失败或把规格参数冒充实测验证。最终结论只允许写“通过、失败、尚未验证”三类与证据相符的状态。

## 10. 当前版本与证据等级（动态，原位更新）

当前native仍固定为162/162格有效中的131个稳定格（80.86%），131 raw、894正式请求以及选择SHA均保持原样。第五轮完整F32存储和GET_ROWS行访问候选已完成全131格，0失败/超时；这是本循环第2/4次完整评估。6/131格三项Engine误差同时<10%，全部主要分组验收仍失败。

当前语义候选为round_005/f32_and_gather_r2；完整候选未获准确性晋级。27B显著改善，但小模型GPU低估加重，必须同时保留。当前固定数据已经揭示，只作为开发/回归，不能称独立盲测、跨模型或跨硬件验收。31格未选择、14条历史失败和四格Smol用户批准频率例外保留在原始162分母及证据中。

## 11. 本轮修改与验证（动态，原位更新）

- 第五轮修正CPU embedding把整词表当作每次访存的错误。GET_ROWS只读实际索引对应的packed行和I32索引，输出F32；整表容量和真实跨设备暂存仍保留，不重复累计路由流量。27B M64逻辑读取/写入总量为1659136B，权重整表容量为1350860800B；解量化工作量已记录但吞吐未标定。
- 原资格判断漏读graph.attributes.metadata中的GGUF身份，导致首版14锚点零应用。首版结果独立保留；修正读取嵌套身份、拒绝冲突后，六组实际图资格和行访问均生效。14锚点三路消融完成，独立审核126个请求和2399条embedding记录，token、时间、字节守恒通过。
- 完整131格采用源码要求的F32+行访问。gather-only在部分GPU场景误差较低，但保留错误hidden位宽不能作为优化。与v2/首轮的完整比较含多轮修正；第五轮的独立因果证据来自14锚点三路对照。
- 本机133项关联回归通过；本轮精确暂存版本在隔离树128 passed、21 skipped，跳过项明确依赖未上传的本地native档案，不能计为通过。本机来源/身份集成检查另已完成。
- 第六轮物理GPU投影、融合和可选MMQ成本三路已冻结，147项结构检查通过。三个关闭开关的控制格逐请求和聚合时延与第五轮完全一致；批次只增加诊断字段，旧字段完全一致。三路12锚点已完成；关闭开关的全部12场景逐请求与第五轮精确一致。四项边界/披露问题正在修复，新修订独立冻结，不覆盖已完成预测。历史模型专用include哈希缺失，合同维持conditional和native_dispatch_proven=false。
- 合成算子trace工具已完成1次profile和1次无profile对照，24 NVTX调用对应72个kernel；Nsight与driver兼容性门失败且没有锁频，不能用于性能标定。独立CUDA启动/同步诊断已在空闲窗口完成160 warmup+480 formal、0错误，16种配置均未通过全部预定稳定性门，没有产出或应用系数。QPC连续对读中位为0 tick不代表观测开销为零。没有运行新的LLM native。
- 每轮commit+GitHub push规则已推送（de26d58）；后端检查点715a083、第四轮69273ca、诊断工具ff9ca8d已推送。第五轮216fb8b已提交推送，远端SHA已核验；第六轮在途代码和历史前端修改独立保留。

## 12. 当前测量与预测差距（动态，原位更新）

第五轮完整131格Engine APE如下；每格先计算模拟请求中位数，与三个native run中位数的中位数比较。表中均为“中位 / P90 / 最坏”，单位%。

| 模型/部署 | 固定格数 | TTFT APE | TPOT APE | E2E APE |
|---|---:|---:|---:|---:|
| qwen25 | 17 | 82.65 / 85.01 / 86.74 | 62.45 / 74.23 / 74.95 | 65.38 / 77.66 / 79.93 |
| qwen35 | 22 | 45.01 / 67.72 / 68.04 | 58.37 / 63.38 / 64.24 | 56.24 / 63.94 / 64.36 |
| qwen38 | 20 | 7.18 / 37.31 / 42.27 | 20.42 / 26.64 / 32.18 | 10.99 / 20.43 / 28.54 |
| smollm2 | 23 | 53.26 / 70.50 / 75.63 | 17.93 / 40.42 / 68.32 | 22.70 / 50.77 / 71.78 |
| tinyllama | 22 | 63.18 / 72.00 / 78.18 | 52.72 / 61.29 / 76.71 | 53.58 / 66.48 / 77.90 |
| qwen38_gpu | 27 | 24.81 / 60.93 / 69.53 | 7.65 / 16.71 / 33.55 | 11.27 / 27.96 / 46.98 |

全部主要分组均未同时通过。三项逐格同时<10%为6/131；相对v2，格×指标改善148、退化245、不变0。相对首轮改善121、退化272、不变0。源语义正确与误差降低必须分别判断。

完整绝对毫秒、有符号、逐run和失败覆盖见round_005/f32_and_gather_r2/report.html及errors.0002.json。CPU权重27B TTFT中位误差由v2的802.89%降至7.18%，但TPOT为20.42%；GPU27B TPOT为7.65%，TTFT为24.81%。小模型GPU普遍低估，下一步检查实际物理算子数、量化路径、kernel及同步成本，不能用场景倍率补齐。

## 13. 下一轮优化顺序（动态，原位更新）

1. 持续核验固定native选择和131 raw，冻结、恢复、评分前后均校验；原生不重测，不改变场景成员和计时口径。
2. 第五轮全131结果、本地提交及GitHub推送已完成；保留首版零应用、gather-only消融及所有退化。
3. 第六轮12锚点三路对照已完成，先修复独立审核指出的重复apply、packed V连续化、逐投影资格和跨请求计数边界，再独立冻结。核查Q/K/V、gate/up、alpha/beta物理调用、M1融合和M>1拆分、F32到F16 KV写入，以及MMQ主矩阵/conversion/fixup的独立计数。
4. 来源与结构资格通过后，按源语义选择候选做第3/4次全131评估。MMVQ、融合分支和未定价项显式保留；不因误差较低启用身份不合格的旧profile。
5. 独立CUDA launch/同步诊断已完成，16种配置均未通过全部稳定性门，当前不生成参数。仅在现有预算内、另有预定独立传递验证证据时考虑成本子消融；不使用不兼容CUPTI或不稳定数据标定，不拟合目标LLM时延。
6. 每轮原位更新任务书，结构回归、消融和分组评估后提交推送。6轮机制、4次全量、8小时预算不扩大；未达标就保留失败范围与下一项所需证据，不宣称验收通过。

## 14. 冻结、复用与循环预算（动态，原位更新）

- native选择SHA固定cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5，选择文件只读；模型、binary、实际argv/env、prompt/output policy、extractor和原始时间戳均不变。
- 预测输入使用同SHA的27B只读权重副本，映射保存在prediction_model_snapshot_map.json。各轮独立source/freeze/predictions；首版错误或零应用也不覆盖。
- 自动循环状态在optimization_loop/state.json。预定上限6轮机制、4次全131评估、8小时，截止本机2026-09-16 05:27:44。全量已完成2/4次，第六轮12锚点已完成，边界修复后待新冻结；每格600秒、总4个预测worker。
- 固定数据仅作开发回归；完整预测覆盖100%不等于准确性通过。没有独立盲测或真实跨硬件验收；不以配置中的算力/带宽修改冒充换硬件验证。
- 每轮提交上传规则已授权；只上传本轮代码、测试、任务书、精简证据，不上传权重、运行库、大型raw/trace或整份冻结源码。

## 15. 最近结果与交付位置（动态，原位更新）

路径根为artifacts/development/native_long_grid_135_20260915/。

- report_162/report.json、report.html与native_variability.svg：完整162格、稳定性、排除和失败历史；stable_native_dataset.json保存131格固定选择及逐请求原始时间戳引用。
- stable_simulation_v2与optimization_loop/round_001/candidate：既有两路完整131对照，预测和评分保留。
- optimization_loop/round_005/full_evaluation.json、decision.json：第五轮完整分组统计、退化、保留决定；repaired_paired_anchor_comparison.json、repaired_output_audit.json为14锚点消融与独立审核。
- optimization_loop/round_005/f32_and_gather_r2/freeze.json、predictions/、errors.0002.json、report.html、report.md及三热图：第五轮完整冻结、逐请求预测、逐格/逐run误差。
- optimization_loop/round_006/baseline、physical_mapping、physical_mapping_mmq：第六轮三路冻结；baseline_control_verification.json记录关闭新开关的精确控制核验。
- optimization_loop/operator_microbench_v2：合成算子工具、静态设备属性和带兼容性失败标记的trace诊断；launch_probe保留独立CUDA启动/同步工具及160 warmup、480 formal原始结果，当前不生成系数。
- optimization_loop/report_checkpoint_r5/report.html、report.md、summary.json：v2/R1/R5完整131与R6两路12锚点独立列示；评分文件显式固定，来源校验无失败；不混合覆盖或宣称准确性晋级。
- tools/verify_fixed_native.py --state <optimization_loop/state.json>为固定native核验入口；各候选source/tools/predict_stable_native_dataset.py --output <候选目录> --resume/--score执行复用预测/评分；tools/render_stable_native_evaluation.py生成报告。

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
