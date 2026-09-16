# 仿真器机制建模、可信度与泛化优化任务书

> **文档状态（动态）**：当前任务仍未达到第一阶段验收阈值；验收主口径已更正为 engine-first，client 仅作二级诊断。本文件是唯一的推进规范；稳定章节定义不可变的规则，动态章节在每轮修改后原位替换。历史实验只保留索引，不在此重复展开。
>
> **适用项目**：`F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0`
>
> **当前目标**：A门为固定131格每格三项Engine误差严格<10%的开发回归目标；B门为独立数据、计时证据和统计不确定性合格后的正式准确性验收，仍采用逐格三项<10%。A门通过不得称总体完成。阶段轮数仅为复盘点；当前A、B均未通过。

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

`E2E_engine = TTFT_engine + (N_out-1) * TPOT_engine` 在固定输出策略下具有确定的代数关系，因此同一短输出请求的三项同时达标不能视为三个独立机制已经验证。任何一次性启动或边界残差都必须在未见输出长度上检查其对首 token、首个间隔、后续逐 token 间隔和 E2E 的影响；恒等式在每个请求上核验，三项跨重复中位数一般不满足同一代数等式，必须分别汇总实测与仿真，不能由两个中位数推导第三项。正式验收至少保留一个不同输出长度的留出场景，并报告每个 token 间隔的误差分布。一次性残差的来源只能写成待验证假设，除非源码和独立 trace 证明其触发条件、生命周期和重复次数。

| 验收项目 | 第一级：engine 目标 | 第二级：server 诊断 | 第三级：client 诊断 |
|---|---:|---:|---:|
| 每格三项绝对相对误差（A开发门与B正式门分别判断） | 每格TTFT_engine、TPOT_engine、E2E_engine均严格<10%，等于10%不通过 | 单独报告各边界，不设抵销一级的阈值 | 单独报告，不设抵销一级的阈值 |
| 每个主要分组绝对相对误差中位数/P90 | 保留诊断统计，不替代逐格验收 | 诊断，单独报告 | 诊断，单独报告 |
| 每个主要分组最大场景误差 | 三项均严格<10% | 诊断，单独报告 | 诊断，单独报告 |
| 固定验收范围有效预测覆盖率 | 100%；缺失或拒收格不得排除后宣称达标 | 不降低一级覆盖率 | 不降低一级覆盖率 |
| 纳入正式结论的证据完整率 | 100% | 证据存在时报告 | 证据存在时报告 |
| 跨模型目标模型专属时延拟合次数 | 0 | 0 | 0 |
| 跨硬件目标 LLM 端到端时延拟合次数 | 0 | 0 | 0 |

当前固定范围为stable_native_dataset.json锁定的131格，不能因预测困难再缩小；原始162格和31格稳定性排除仍单独完整报告。逐格误差按各场景重复运行中位数计算，使用未舍入数值判定；不要求每次重复的瞬时时延误差都小于10%。A开发门必须在同一冻结版本获得131/131格、393/393项有效且严格<10%的结果，失败、超时、身份异常、非有限值和证据不足均不能通过。固定集通过只代表开发回归目标达到，不代表独立盲测或跨硬件通过。

**A/B双门与可证明范围。** 不降低逐格10%要求，但将“开发目标”和“正式结论”分开：
- A门：固定131格点估计全部<10%，用于机制优化与回归，不能反复针对这些答案优化后称独立泛化。三次重复时的低波动只描述已观测样本；当前131格均只有3批次、formal_repeatability_accepted=false，不能据此证明总体测量误差<5%。
- B门：在A门之后冻结机制和所有输入，在预先登记的独立场景/完整留出模型上按相同逐格三项<10%判定；模型、硬件及支持域在揭盲前确定。根据实测独立实验批次建立并报告统计不确定性。若置信范围跨越10%边界，结论为“证据不足”，不能硬判通过；不将噪声从误差中减去，也不设置事后容差。
- 正式采样数量、独立实验单位、最多加样次数、联合置信水平（默认95%）及多重比较控制方法必须在新native采集前冻结。并发请求与token不能冒充独立重复；普通单项95%区间不能宣称393项同时95%可信。不能靠反复查看结果直到区间过线停止采样。样本不足时保留证据不足，不用3批次bootstrap制造虚假精度。
- A门通过后停止针对固定答案调参并冻结候选，转入B门；若独立证据缺失，标记“开发回归达标、独立验收未完成”，不能反复拟合A门替代B门。当前用户锁定native继续复用，本次审计不授权或启动重测；若B门所需新证据缺失，继续能独立完成的机制工作，并明确剩余采集需求，不重新拟合目标LLM。
- 固定131格的100%预测覆盖与原始162格的native可评价覆盖分别报告：后者目前131/162=80.86%。31格不稳定样本不进入现有误差优化，但仍属于未验证范围；不得写成整个162格或所有新输入已支持。
- B门通过仅证明预先声明且实际验证的范围；缺少第二硬件时跨硬件保持未验证，不作为当前单机目标永远无法完成的隐性条件。HBF参数扫描另外报告敏感性与未验证范围，不能从本机拟合精度直接推出新硬件准确性。


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
10. 每轮提交推送后复核第7节逐格三项目标；未达到且仍有可执行方向时自动进入下一轮，无需用户再次发送“继续”。阶段轮数和计划时间窗仅为复盘检查点，记录阶段效果和下一阶段预算后自动续轮，不作为停止理由。连续多轮仅完善工具而未改善预测时，必须复盘机制假设和证据缺口，避免无效重复；只有A、B两门均在声明范围通过、用户明确暂停或现有授权与资源下无法解决的外部阻塞才停止，仍能独立完成的工作继续。不得突破用户明确的费用、资源或时间硬上限。连续3轮若既未改善同一评估范围的最坏误差/达标格数，也未关闭预先声明的必要证据缺口，则自动进行路线复盘并切换有证据的假设；没有可执行假设时标记研究阻塞，不靠重复同一测量无限续轮。A门执行器tools/evaluate_strict_engine_goal.py已实现逐请求时间戳重算、全范围与冻结身份核验、严格阈值及缺失拒收；B门尚无独立验收实现，固定返回未验证。它输出下一步行动，不启动脱离当前任务的无限后台循环。

不得追加场景常数或全局倍率来掩盖系统性误差。

版本控制规则：提交包含该轮代码、相关测试、任务书和必要的小型证据；在途子代理的未完成修改及其他独立工作保持隔离，不整体暂存工作树。GGUF、DLL/可执行文件、海量raw trace和native原始数据继续保留本机，不加入常规推送。失败或回滚也保留可追溯的结论；推送失败时保留本地提交和失败原因，恢复网络后重试，禁止强制推送或覆盖远端历史。

每轮还必须执行因果归因门禁：如果 profile gate、native 重采、extractor、simulator 或成本 profile 在同一轮同时变化，不能把误差变化归因于其中某一个开关。必须固定同一份 native payload 和同一组场景，分别运行纯分析基线、旧 profile（仅诊断）和身份/适用域合格的新 profile 三路 simulator-only 消融，并单独报告 native 分母是否变化。只有在 native identity、计时契约和 extractor 均固定时，才可把差异归因于 simulator 机制；旧 identity 的低误差与当前 identity 的失败只能作为关联事实，不能直接写成“门禁导致误差变大”或“旧 profile 掩盖了缺陷”。

## 9. 交付与停止条件（稳定）

交付必须包括：模型结构和假设、参数来源/单位/有效域/失效条件、冻结清单、数据集划分、原始测量与预测、分组误差、消融、失败案例、回归测试和一键复现入口。每条预测标记来源类型（机制分析、微基准插值、域外外推、条件回放）及是否在验证域内。

缺少第二种硬件或独立验收数据时，交付可执行协议并明确“未验证”。阶段轮数或计划时间窗到达时保存检查点并自动登记下一阶段，未达标继续。只有用户明确的硬预算/资源限制或不可解决的外部阻塞才暂停依赖工作并交付可恢复状态；不能放宽阈值、隐藏失败或把规格参数冒充实测验证。最终结论统一为“通过、精度失败、证据不足、尚未验证”四类；身份或重复证据失败不能冒充精度结论，但仍保留在分母中。

## 10. 当前版本与证据等级（动态，原位更新）

当前语义候选为第14轮 `tail_candidate_r2`：20格（17个Qwen2.5固定格及3个其他模型控制）成功，0/20逐格三项同时<10%，其余111格未在此冻结中重跑。不得与第六轮结果拼成一份新131格验收。第六轮仍是最近完整对照：129/131评分、9/131逐格三项达标，2格因模型身份失败拒收。

当前第16轮正在收集独立合成算子证据和修复新成本解析器的审查问题，默认预测未启用新校准。固定native选择和原始时间戳继续复用；已有数据只属开发回归，A门未通过、B门未验证，跨硬件未验证。

## 11. 本轮修改与验证（动态，原位更新）

第14轮逻辑K尾段修复与120项相关回归已完成，Qwen2.5 TTFT中位误差为68.502%、E2E为51.919%，三个其他模型控制不变。第15轮已取得驱动兼容的CUPTI trace，区间拆分17项测试通过；GPU kernel并集和包含它的同步wall不能相加，单组profile对照仍不作校准。

第16轮采用3个子代理并行：独立算子探针、原始事件采集及准入、新成本解析器语义审核。探针r2首pair在计时前因未导入ggml.dll触发严格身份门而失败，2个失败进程和空目标trace保留；r3已增加绝对路径显式加载、handle身份检查及生命周期保持，19项Python和8项loader检查通过，376项依赖冻结；首个新pair已完成：36次调用的数值、真实MMVQ链和2400±30MHz读回均通过，GPU并集中位2.7845微秒、P90/P10=1.108；direct host P90/P10=1.630超过1.5，因此暂拒收为校准。余下冻结组合正在同一锁频会话中串行执行。

交叉审查发现解析器仅验SHA不能保证accepted声明与原始结果相符、设备与硬件事实绑定不足；采集器存在冻结集合省略、身份失败续跑、额外GPU活动未拒收和异常退出归属四项缺口。采集器新版本四项P1及频率门已修复，75项主机测试通过、1669项冻结核验通过；解析器语义绑定仍在修复，尚不启用实际成本。26shape分析预测已先于有效采集保存，80项主机测试通过。完成后原位回填实际准入、消融和误差结果，不把合成测试通过写成准确性达标。

## 12. 当前测量与预测差距（动态，原位更新）

第六轮完整批次条件统计；各指标为APE“中位/P90/最坏”，单位%。两格身份失败不补零，覆盖与误差分开报告。

| 模型/部署 | 已评分/固定 | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|
| qwen25 | 17/17 | 75.66 / 79.18 / 81.88 | 48.87 / 65.84 / 66.66 | 54.68 / 70.09 / 72.77 |
| qwen35 | 22/22 | 31.72 / 59.79 / 60.15 | 47.91 / 54.12 / 55.35 | 45.25 / 54.92 / 55.49 |
| qwen38 | 20/20 | 8.18 / 38.25 / 43.13 | 20.26 / 26.64 / 32.18 | 11.87 / 20.54 / 28.89 |
| smollm2 | 23/23 | 49.12 / 68.04 / 73.67 | 8.85 / 34.80 / 65.79 | 14.89 / 46.29 / 69.44 |
| tinyllama | 22/22 | 55.09 / 65.68 / 73.63 | 35.00 / 50.29 / 71.54 | 36.40 / 57.84 / 73.08 |
| qwen38_gpu | 25/27 | 19.79 / 36.65 / 37.36 | 11.07 / 17.91 / 23.52 | 12.08 / 21.11 / 29.82 |

三项同时<10%为9/131。与第五轮共同有效格的指标相比，改善312、退化67、不变8。GPU27B TPOT中位有所退化，不能用TTFT改善抵消。完整逐格、有符号、毫秒和逐run统计见round_006/physical_mapping_mmq_r2/report.html及errors.0001.json。

## 13. 下一轮优化顺序（动态，原位更新）

1. 完成第16轮采集器和成本解析器的交叉审查修复：原始数值/区间/样本量与准入重算绑定，设备资源来自执行上下文，冻结清单完整且锚定；失败身份立即停在当前批次，旧记录不覆盖。
2. 使用r3探针、新采集冻结和2400MHz控制记录，串行执行26shape、每shape三对profile/direct交错对照。先验证首pair的真实kernel链和数值路径，再接续矩阵；GPU计时期间不运行重仿真或编译。硬件读回不符、外部GPU活动重叠和高扰动记录拒收且保留分母。
3. 对照预先保存的分析预测，分别核对conversion/main/fixup；测量不直接变成最终时延倍率。新键包含完整M/N/K、量化/布局、kernel变体、cache和硬件/runtime身份。缺缓存迁移、单stream资源归属或源码等价证据时保持条件状态，不强接入。
4. 合格的机制修复先做纯分析/当前机制/新机制消融，复用固定native比较逐格Engine三项及最差分组；结构正确、未参与修改样本有收益且无未解释退化时冻结全131重评。未达标自动续轮，不补模型名称常数或放宽10%。
5. 每轮原位更新任务书、commit并push后核验远端SHA。阶段轮数只是复盘点；连续3轮无精度或必要证据进展则改换机制路线。A门通过后才进入新的独立B门，不将开发固定集或已揭示算子留出组称为最终盲测。

## 14. 冻结、复用与循环预算（动态，原位更新）

- native选择SHA固定 `cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5`；本轮不重测目标LLM，不修改其binary、模型和计时原始记录。
- 第一、第二阶段已归档；当前第三阶段从第13轮起推进，6轮/8小时为复盘点而非自动停止条件。未达标继续，所有新freeze和失败均保留。
- 第14轮20格候选、r1及旧R6完整对照独立保存。两次27B身份异常不可由事后SHA一致改成通过。
- 第16轮旧collection首pair计时前失败；后续采集器与probe采用新版本和新冻结，不能把跨freeze进程凑成同一三对采样。重测仅限独立合成算子。
- 每轮只提交本轮代码、任务书和精简证据；不上传模型权重、运行库、大raw或完整冻结源码，不混入前端/原生历史WIP。

## 15. 最近结果与交付位置（动态，原位更新）

路径根为artifacts/development/native_long_grid_135_20260915/。

- report_162/report.json、report.html与native_variability.svg：完整162格、稳定性、排除和失败历史；stable_native_dataset.json保存131格固定选择及逐请求原始时间戳引用。
- stable_simulation_v2与optimization_loop/round_001/candidate：既有两路完整131对照，预测和评分保留。
- optimization_loop/round_005/full_evaluation.json、decision.json：第五轮完整分组统计、退化、保留决定；repaired_paired_anchor_comparison.json、repaired_output_audit.json为14锚点消融与独立审核。
- optimization_loop/round_005/f32_and_gather_r2/freeze.json、predictions/、errors.0002.json、report.html、report.md及三热图：第五轮完整冻结、逐请求预测、逐格/逐run误差。
- optimization_loop/round_006/physical_mapping_mmq_r2/report.html、report.md、errors.0001.json：第六轮129评分、2身份失败与完整分组结果。
- optimization_loop/round_006/baseline、physical_mapping、physical_mapping_mmq：第六轮三路12锚点；paired_anchor_comparison.json记录同版本对照。physical_mapping_mmq_r2为边界修订冻结；core_repairs/validation.json记录72图和315个冻结SHA核验。
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
