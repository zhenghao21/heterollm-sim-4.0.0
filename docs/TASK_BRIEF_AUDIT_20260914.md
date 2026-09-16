# 仿真器机制建模、可信度与泛化优化任务书

> **文档状态（动态）**：当前任务仍未达到第一阶段验收阈值；验收主口径已更正为 engine-first，client 仅作二级诊断。本文件是唯一的推进规范；稳定章节定义不可变的规则，动态章节在每轮修改后原位替换。历史实验只保留索引，不在此重复展开。
>
> **适用项目**：`F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0`
>
> **当前目标**：A门为固定131格每格三项Engine误差严格<10%的开发回归目标；B门为独立数据、计时证据和统计不确定性合格后的正式准确性验收，仍采用逐格三项<10%。A门通过不得称总体完成。阶段轮数仅为复盘点；当前A、B均未通过。

## 1. 总目标与优化优先级（稳定）

根据模型结构、权重格式、输入输出长度、硬件、运行时和调度配置，生成可解释的算子图与执行时间预测。仿真对象的一级范围是 engine 内部的 GPU、HBM/HBF 与其他内存访问、kernel、KV cache、CPU/GPU 协作、同步、采样和调度执行；这些部分共同决定研究对象的 engine wall。发生在 engine 计时边界之外的 HTTP、JSON、tokenizer、admission、队列、SSE、socket 及其线程调度属于 server/client 服务边界，作为分解后的二级诊断。边界内的 host 建图、CPU 采样、驱动提交、同步及其调度等待仍属于 engine wall；按源码计时点、实际资源和重叠关系确定归属，不能因其发生在 CPU/OS 就一概移出 engine，也不能把所有宿主等待加成 GPU kernel 常数。对同一请求，服务侧只可在边界证据证明不存在重叠时写成 `T_server = T_frontend + T_queue + T_engine + T_response`，再将请求/接收传输单独计入 `T_client`；存在并发或包含关系时必须保留时间线，不能把各段简单相加。HBF 参数扫描只能改变 engine 内受其影响的内存、KV、kernel 和调度成本，不能把不受 HBF 影响的 frontend/response 固定开销吸收到 engine 成本。预测不得读取或拟合待预测场景的原生端到端时延。

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

1. 从开发集定位执行图、shape、量化、访存、kernel、调度、计时或测量问题；先按模型/部署、误差符号、P/O/C切片、逐请求闭合关系及重复波动分层。趋势比较固定其余因素，只比较完整匹配组，缺格和失败另计，不能把观察相关性当成因果证明；
2. 提出单一、可证伪的机制假设；在评估前由源码与静态shape规定预计受影响集合、变化方向及应不变集合，不按看过的误差选择容易格；
3. 收集锁定源码、trace、独立微基准或推导证据，先核对现有成本所有权，避免重复收取nonflash、KV写入、logits传输、采样、launch和sync；
4. 进行最小修改并跑结构回归；实际microbatch的M、量化分派与物理KV视图均由真实语义决定，不能以并发数、模型名称或prompt档位替代；
5. 做同输入开/关成对消融，报告signed error、APE、绝对毫秒、共同有效格变化、覆盖率与失败；区分机制正确性、因果假设和精度收益，正确语义被其它资源瓶颈遮蔽不等于实现无效，不能仅因平均误差下降就准入；
6. 在独立于本轮定位的验证样本评估后决定候选准入、保留诊断或回滚；低波动三重复不证明噪声下限，native最大偏离、CV和预测APE分开报告，固定131格不得事后剔除或用噪声抵扣误差；
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

## 10. 当前版本与证据状态（动态，原位更新）

固定native为131格、894请求，选择文件SHA为cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5。覆盖五个模型、六种部署：qwen25 17、smollm2 23、tinyllama 22、qwen35 22、qwen38 CPU 20、qwen38 GPU 27。原162格及31格既定稳定性排除保留，本阶段目标LLM重测0、专属时延拟合0。

最新已评分版本为R24：131终态中130预测、1个完整GGUF SHA校验失败，9/131格三项Engine误差严格<10%。A未通过；B与跨硬件未验证。R25为同源末层选行开关对照，off131格全部预测成功，但on在启动校验时因配置证明错位被拒绝，0格预测、未评分；进程已自然退出，execution_closed.json已封存。其语义资格为conditional，不能升级为原生分派已验证。分组/配对报告入口已准备，4项测试验证严格阈值、失败分母、两路native一致和评分前拒收；尚未对R25运行报告。

R23的27个GPU proof表示失败已作为历史保留。R24将其中26格恢复为预测；另1格qwen38_gpu_p512_o128_c1完整SHA不匹配，未重试覆盖。预测前后全文校验正常不解释这次瞬时异常的根因，也不能追认该失败通过。

## 11. 已完成修改与验证（动态，原位更新）

- R24身份修复统一path/sha256/bytes，拒绝矛盾长度和非法类型，缓存命中复核header；补齐未映射模型的freeze/resume全文SHA。原R23 104个数值预测全部精确不变，证明本轮没有成本收益；恢复覆盖导致总体分布变化不能当作精度改善。
- R25修复真实GGUF架构名及metadata owner错位，在最后replan之前绑定选行。qwen2/llama于最后FFN前选行，qwen35于全行final norm后选行；indices/GET_ROWS保留，不重复增加lm-head。M64/R1静态探针保留attention M64，FFN重新分派M1，down进入MMVQ，融合up/gate仍明确保留单矩阵资格未覆盖。新增成本系数0。
- R25主区联合回归114通过/1跳过；独立语义复审61通过；两路冻结及运行器保护9通过，计数有重叠不相加。两路复制后的extractor与硬件资料仅在已核对字节、别名和派生摘要的明确字段位置归一化。首次硬件资料路径比较失败保留，有单独qualification凭据，未改变冻结内容。
- MMVQ R6共享ABI已修正conversion的type/k/m/padded_k四个int声明，定义与caller共用头；旧少参数声明负向编译被C2733拒绝，2项检查通过。R6重新提取的主MMVQ和Q8_1 cubin分别5450408B、222816B，与目标DLL逐字节相同。旧R4证据不改；转换正确性、运行时分派/参数/线程布局/stream/cache及计时扰动仍未验证，0参数准入。

## 12. 最新误差及主要缺口（动态，原位更新）

下表为R24的130个可评分场景；1个失败仍在固定131分母，不能将R23的104格总体分布与此表直接比较后宣称改善。

| Engine指标 | APE中位数 | 场景误差P90 | 最坏APE | 绝对ms中位数 | 最坏ms |
|---|---:|---:|---:|---:|---:|
| TTFT | 42.651% | 66.967% | 72.496% | 88.720 | 2667.094 |
| TPOT | 25.143% | 51.746% | 64.773% | 2.574 | 125.907 |
| E2E | 29.877% | 54.474% | 66.189% | 364.967 | 25473.633 |

普通attention62格仍无三项全过；多数系统性偏低不能用已筛选native波动解释。既有894 native请求及292条R22预测请求的逐请求E2E闭合通过，最大残差2.91e−11ms；独立聚合中位数不保证加法恒等式。三次batch中位数CV最大TTFT2.5313%、TPOT1.9039%、E2E1.7986%，仅为现有样本诊断，不是正式稳定性保证。

目前有证据的成本缺口包括MMVQ访存仍耦合MMA输出tile利用率、host图构建/提交与GPU重叠边界尚未独立定价。权重字节无重复、已有nonflash/logits/sampling/launch不能整段再加。P128/512/1536均整除ubatch64，不能默认长prompt存在尾块；c不等于实际kernel M。

## 13. 当前执行及下一轮优化顺序（动态，原位更新）

1. 保留R25启动失败，转入新冻结版本修复：原proof从raw rows读取flash_attn/op_offload为None，而实际static_inputs经显式alias及verified host合同得到False/True。已改为先规范化static_inputs再派生proof，未添默认值/未弱化比较；联合176通过、1跳过，真实131静态重导全部通过。新轮必须先对两路冻结进行实际worker门禁预检，再运行完整对照并评分；不改旧R25、不复用旧输出冒充新冻结。
2. 依据选行源码与冻结开关评估结构修正；普通架构FFN缩行可能降低原本偏低的TTFT，混合架构补全final norm及gather可能增时。数值效果必须实测，结构正确不等于准确性验收通过。
3. 推进R26合成MMVQ运行路径资格。R6共享ABI和设备二进制身份已完成；下一步用固定DLL合成MUL_MAT与同源探针对照conversion/main的完整模板symbol、参数、grid/block/shared、stream和pointer关联。固定Q5_0 M1 K4096 N3072参数记录器已编译，38项纯主机解码测试通过，0 GPU执行；wrapper正确性程序已编译、5项CPU参考/拒绝测试通过。执行门禁已独立复审，7项测试覆盖所有已知仿真/native/recorder入口、关闭凭据、环境隔离与异常终态；源和298头文件身份核验通过。run.0001已在R25退出后执行一次不计时合成图：固定DLL/GPU前后身份通过，2个预期kernel符号出现，无runtime memcpy/memset；实际使用cudaLaunchKernelExC，记录器未解码该API而拒收，失败终态完整保留，0参数准入。独立ExC版已补齐config/attrs深拷贝、ID6/PDL=1来源检查、未知属性拒收和未观测几何null输出，63项主机测试（保留原38）及7项执行保护通过；新构建及299头文件身份核验通过，准备新的不计时路径验证。规范化修复的新冻结R27入口正在并行准备，尚未预测。M4作为预先留出的形状。无需目标LLM权重和时延，不重复设备代码比较。
4. 路径与转换数值资格合格后，串行采独立CUDA-event主kernel/转换耗时，warm复用与超过L2的旋转工作集分开。CUPTI重profiling只证明路径，不用于性能系数；不以CTA数直接换经验带宽，不用131格误差选择参数。
5. 并行审计可继续针对真实microbatch、nonflash物理KV、retained高水位、slot及host图生命周期收集静态证据。现有成本owner逐项查重；native token时间戳只供诊断，不驱动预测后称泛化。对未知成本显式标未定价，不随p/c添加残差。
6. 每轮原位更新任务书、提交、推送并核验远端。A固定131/131、393/393均严格<10%；不达标继续有证据的机制假设；通过A后另用独立B，当前B尚未验证。

## 14. 冻结、复用与循环预算（动态，原位更新）

- 固定native、历史冻结/预测/评分不可变；复用actual，只重跑simulator。完整文件SHA不能由payload摘要、header或stat代替。临时身份失败保留，不能悄悄重跑至成功。
- R24来自R23冻结113个源码字节继承及1个已提交adapter替换，完整预测/评分已结束，139项详细证据逐字节归档并分3卷上传。
- R25两路相同源码、相同成本开关和配置，仅末层选行处理不同；来源为R24冻结加已提交候选文件，不混主区历史WIP。protocol、freeze receipt、controls已保存；评分由262终态屏障约束。R25现已因on启动校验失败封存，133项冻结/预测明细已逐字节归档并分3卷，driver/helper/冻结源码继续不可变。采集合成证据必须核对关闭凭据及实际进程退出；存在任何新轮仿真或native/probe进程时仍禁止并行执行。
- 第4阶段6轮/8小时为复盘点而非停止条件；出现连续无效优化按第8节换假设。新探针编译与静态审阅可并行，CPU/GPU性能采集与重仿真串行，子进程自然退出。
- 提交相关源码、测试、任务书及小证据；大JSON/trace逐字节分卷。不上传GGUF/DLL/EXE/OBJ/cubin或整棵复制源码，构建脚本和身份凭据可复现。

## 15. 最近结果与交付位置（动态，原位更新）

以下路径根为artifacts/development/native_long_grid_135_20260915/。

- stable_native_dataset.json、optimization_loop/state.json：固定native与循环状态。
- optimization_loop/round_024/REPORT.md、identity_repair_report.json、repaired/errors.0001.json：最新完整评分、104格精确复现及失败。
- optimization_loop/round_024/full_engine_error_heatmap.png/svg：最新131格热图；X为失败，短横线为固定集之外。
- optimization_loop/round_024/full_predictions.json、full_scores.json、detailed_evidence.parts.json：先预测后评分与3卷恢复入口。
- optimization_loop/round_025/protocol.json、freeze_receipt.json、controls.json、execution_closed.json：已拒绝批次的冻结、131配置错位重导和退出记录，未评分。
- optimization_loop/round_024/mmvq_device_probe/r5_device_code_compare/、r6_shared_abi/：设备代码及共享ABI证据，均无性能系数。
- optimization_loop/round_023/optimization_direction_metric_audit.md、optimization_direction_source_audit.md、host_cost_ownership_audit.md、mmvq_memory_geometry_audit.md：已纳入任务书的参考方向审计。
- optimization_loop/round_022/REPORT.md、round_023/REPORT.md及各轮归档索引：历史误差、退化和失败保留。
