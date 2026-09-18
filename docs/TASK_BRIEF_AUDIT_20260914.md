# 仿真器机制建模、可信度与泛化优化任务书

> **文档状态（动态）**：继续自动优化循环；哈希校验失败按第8节仅完整复核一次，并按失败阶段决定恢复范围。R37预测已完整执行，原评分失败保留；用户要求的五模型完整复核已通过，恢复评分与配对报告已生成，但独立严格验收因GGUF元数据头不一致返回证据不足；恢复链不再重试。R39同次读取绑定修复已独立审查并新冻结，131格预测、评分和证据完整性已通过；严格准确性仅6/131格达标，有条件保留继续机制优化，暂不提交。尚未通过误差验收。R36/R38结构检查不代替精度验收。验收主口径为engine-first，client仅作二级诊断；稳定章节定义规则，动态章节原位替换。
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

只保留三种数据角色：开发集用于机制分析和允许的底层参数拟合，验证集用于候选比较，最终盲测集用于冻结后的独立评分。一次运行只产生一个不可变的 `run_id`，记录测试清单、源码/profile/运行时身份、计时契约、预测和评分的引用；不再为同一事实建立平行 control、bundle、receipt 或报告副本。预测必须先写入，再读取 native actual；揭盲后的数据不再作为最终盲测。

冻结只验证一次当前运行实际引用的源、配置、固定 native selection、提取器和评分器，并保存其 SHA。预测阶段只记录入口/出口是否仍绑定这份冻结；评分阶段只验证冻结、预测、评分引用和覆盖，不重复全文验证已经由冻结绑定的模型/运行库。只有文件 SHA 不一致、阶段失败或用户明确要求复核时，才启动一次受限复核；原 attempt 不覆盖，复核失败保留原失败和原因。源码、配置或 profile 改变即建立新的 `run_id`，不能拼接不同运行。

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

每轮只保留一个候选、一个基线、一个成对消融和一个评分入口。诊断仍采用自顶向下顺序：先检查阶段、依赖、资源所有权、状态生命周期、关键路径和重复计费是否表达正确；确认结构语义后再修局部成本。每轮流程为：

1. 从开发/验证集提出一个可证伪的确定性缺口，声明预计受影响格、方向和不变格；
2. 以同一输入运行基线和候选，先完成结构回归，再完成两者的逐格三项误差比较；
3. 任一基线或候选格失败，立即定位失败阶段并只对该格或该阶段补测；哈希失败最多完整复核一次；
4. 只有确定性缺口已修复，且候选在未参与假设形成的比较组上改善或保持、没有不可接受退化，才接受候选并提交；否则保留失败记录并回滚/继续下一候选；
5. 评分结束后检查 A 门是否达到固定131格每格三项 Engine 误差严格<10%。未达到则自动进入下一轮；达到后冻结候选并转入独立 B 门。用户明确暂停、预算耗尽或外部阻塞时停止。

运行框架只保留两个外部入口：`predict → score`。`predict` 内部完成静态场景投影、必要的输入身份绑定和 simulator 运行；`score` 内部完成固定 native 绑定、逐请求 Engine 时间戳重算、覆盖率、失败分类和每格三项 `<10%` 判定。没有独立 freeze、strict、failure_recheck、report 或热图验收步骤；热图、HTML、JUnit 和历史回执只属于可选诊断。仓库中保留的 native 采集、微基准和历史解析工具不属于运行入口，不得被自动优化循环调用。

## 9. 交付与停止条件（稳定）

交付必须包括：模型结构和假设、参数来源/单位/有效域/失效条件、冻结清单、数据集划分、原始测量与预测、分组误差、消融、失败案例、回归测试和一键复现入口。每条预测标记来源类型（机制分析、微基准插值、域外外推、条件回放）及是否在验证域内。

缺少第二种硬件或独立验收数据时，交付可执行协议并明确“未验证”。阶段轮数或计划时间窗到达时保存检查点并自动登记下一阶段，未达标继续。只有用户明确的硬预算/资源限制或不可解决的外部阻塞才暂停依赖工作并交付可恢复状态；不能放宽阈值、隐藏失败或把规格参数冒充实测验证。最终结论统一为“通过、精度失败、证据不足、尚未验证”四类；身份或重复证据失败不能冒充精度结论，但仍保留在分母中。

## 10. 当前版本与证据状态（动态，原位更新）

固定 native 是 `stable_native_dataset.json` 锁定的 131 格、894 请求，选择文件 SHA-256 为 `cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5`。覆盖五个模型和六种部署；原始 162 格及稳定性排除仍按历史记录保留。本轮不重测 native，不使用目标场景时延拟合成本。

当前开发基线已切换为 R35/on 的完整 131 格评分结果：Engine TTFT/TPOT/E2E APE 中位数分别为 28.434%/25.950%/24.468%，相对 R34/on 的 40.010%/30.599%/29.851% 有确定性改善；R35 曾发生身份门禁失败，因此仅作为“准确性开发基线”，不作为最终验收通过。R34/on 保留为历史对照。R40 已补齐为 131/131 格：qwen38_gpu 的 TPOT/E2E 局部改善，但 CPU qwen25、qwen35、smollm2、tinyllama 等分组退化，因此只保留为 GPU 局部候选，不替换完整基线；R42 全量 relaxed 评分明显退化，不接纳为基线。

## 11. 已完成修改与验证（动态，原位更新）

- R42 修复 GPU controller 服务归属：每个 GPU 数据阶段承担自己的 MMU/L2/VRAM 服务，串行阶段不再把全部访存提前归到首个 root；CPU、H2D、DMA、PCIe 和 command processor 字节不会制造 GPU 访存回退，显式零字节声明不会继承其他资源字节。
- R42 局部结构测试 6 项通过，既有 GPU controller、consumer、runtime 集成回归 35 项通过；真实 lowerer 多阶段探针确认两个阶段分别为 32,036B 与 47,440B，并保留串行依赖。
- 验收框架进一步精简为 `predict`/`score`：`evaluation_contract.py` 是 Engine 指标公式的唯一来源；score 同时完成固定 native 绑定、逐请求时间戳重算、覆盖率和严格门判定，不再调用独立 strict/report/recheck 入口。
- 以上结构结果不等同于准确性改善。没有通过 A 门前不得宣称支持域已达标。

## 12. 当前误差与流程缺口（动态，原位更新）

当前开发基线为 R35/on；R39 与 R35 的完整评分结果等价，R40 覆盖不完整，R42 已完成全量 `predict`/`score` 但候选退化。这些准确性结论与运行框架身份问题分开记录，身份失败不能被当作仿真误差。

审计确认原流程的主要阻塞来自重复 provenance、runtime/module、worker attempt 和 coordinator lock 校验，而不是 native actual 缺失。默认优化路径已移除这些重复阻断，只保留 native selection 的 schema、cell ID、实际字段和逐格结果检查。完整身份复核保留为显式 `--strict-identity` 选项，不属于正常优化循环。

任何 simulator 格失败都必须保留该格状态和原因并计入覆盖率；成功格可以继续评分。默认流程不重新采集 native，也不使用 native actual 拟合成本模型。

## 13. 当前执行计划（动态，原位更新）

R35 已被选为当前开发基线（仅用于成对比较，不代表验收通过）。R40 已完成 131 格补全评分，证据完整率恢复到 100%，但只作为局部 GPU 候选；R42 已完成 131 格 relaxed predict/score，结果退化：Engine TTFT APE 中位数 58.957%、P90 1037.090%、最坏 3100.686%；TPOT 中位数 26.860%、P90 62.112%、最坏 84.429%；E2E 中位数 40.547%、P90 156.615%、最坏 1054.580%；三项均严格 <10% 的格数为 0/131。R42 不接纳、不合并、不提交；保留其评分作为开发证据。下一轮以 R35 为父基线，先审计 qwen38_gpu 的 controller 资源归属、重复计费和串行化，并用 R40 的 GPU 局部改善作对照；随后检查 qwen25/qwen35/tinyllama 共同偏低的 prefill、首 token、host-submit 与同步阶段是否完整建模。先确认执行语义，再决定局部成本修正；不改变 native 数据、模型、硬件、命令行配置、prompt/output policy、计时契约或 simulator 成本模型。

默认 `predict`/`score` 流程只保留必要的输入结构检查、逐格结果状态和误差计算：不在 worker、收尾和 score 前重复验证完整 provenance、源码快照、runtime/module SHA、attempt seal 或 coordinator lock。`freeze.json` 和 worker 文件仍可作为内部场景包与失败记录，但不再作为默认评分阻断条件。需要严格身份复核时才显式启用 `--strict-identity`。

单格 simulator 失败继续写入该格结果并计入覆盖率；已完成的格可以独立进入 `score`。默认流程不重新采集 native，也不使用 native actual 拟合成本模型。native selection 仍需通过最小 schema、cell ID 和实际字段检查。

本轮已完成底层 JSON 单次读取、稳定快照和进程内缓存；主流程已关闭重复身份门禁、历史 attempt 门禁和默认全局锁。下一步审计 R42 引入的 GPU controller 关键路径；只有确定性缺口修复并且候选组相对基线改善后才提交，未达到目标则进入下一轮，不把身份失败误写成仿真误差。

## 14. 冻结、复用与循环预算（动态，原位更新）

固定 native、历史预测和评分结果保持不可变；当 simulator 成本模型、binary、模型、硬件、命令行配置、prompt/output policy、计时契约和 extractor 均未改变时，可复用已保存的 native raw/逐 token 时间戳，只重跑 `predict`，再运行 `score`。每个运行只需要 prediction 结果集合和一个 score；失败直接记录在两者中，不自动重测、不覆盖原结果。

R35 已作为当前开发基线；R40 的准确性状态为“局部 GPU 候选/全局失败”，R42 的准确性状态为“候选失败/退化”；当前固定 native 仍为 `stable_native_dataset.json`（SHA-256 `cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5`）。本轮完成 R42 全量开发评分，但未宣称 A/B 门通过；R42 已判定为候选退化。

## 15. 最近结果与交付位置（动态，原位更新）

- 固定 native：`artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json`。
- 循环状态：`artifacts/development/native_long_grid_135_20260915/optimization_loop/state.json`。
- 当前开发基线评分：`artifacts/development/native_long_grid_135_20260915/optimization_loop/round_035/on/errors.0001.json`（身份门禁曾失败，仅作准确性基线）。
- R42 候选源码：`artifacts/development/native_long_grid_135_20260915/optimization_loop/round_042/candidate_source`。
- 仿真器关键实现：`src/heterollm_sim/serving.py`、`src/heterollm_sim/planner.py`、`tools/predict_stable_native_dataset.py`、`tools/evaluation_contract.py`。
- 历史 round 目录保留原始失败和必要证据，但不再要求为同一事实复制新的 report、control、bundle、receipt 或审计文档。
