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

第六轮physical_mapping_mmq_r2已结束，129/131格完成评分，2格因模型副本SHA校验不一致拒收（均在建模前失败，并非超时）；有效预测覆盖98.47%，原始162格分母保留。逐格三项Engine误差同时<10%为9/131，六个主要分组均未同时通过。当前是有来源约束的条件机制研究候选，未获准确性晋级。

固定native选择、131 raw和894正式请求SHA核验仍通过。运行后完整读取27B副本SHA与冻结值一致，但不能据此删除两次历史不一致或宣称根因已解决。全量预测于02:13:33完成，早于预算截止；本次恢复后补做评分与提交，没有到期中断。当前固定集已揭示，仅属开发回归；没有独立盲测或跨硬件验收通过。

## 11. 本轮修改与验证（动态，原位更新）

第7轮已提交1f0c765：同一文件句柄解析GGUF并计算SHA，检查读取字节数与前后身份；6项可移植及3项独立合成检查通过。两格补预测成功但三项误差仍未全部小于10%，历史SHA异常未复现、原因未知。

第8轮已提交edf9b41：统一等待、64次连续图提交；12组中2组诊断准入、10组拒收。独立补充审核绑定107文件，检查257条实际事件包络均合理，9项篡改/异常回归通过；这些不等于432条预定事件完整完成，也不构成成本标定资格。

第9轮只读源码参考诊断：Q5_0六组重建权重/输入SHA全部一致；源码Q8_1激活量化与半精度原输入和修正后的参考最大绝对差1.15–1.34e-6，原F32数学参考差约0.067–0.070。支持参考路径遗漏的机制解释，非GPU准确度或仿真时延达标证明；旧失败不改写。第10轮已实施双参考并完成全部24进程，解决数值提前退出；时延标定仍被8组计时拒收阻断，第11轮已分解全部12组×30批次：提交占主机中位数23%–40%，与设备包络相关性多数组较弱，不能归因单一CPU提交。第12轮预设1024次提交扩展测量窗口，保持其余配置和所有门限，不删除慢样本；这是本窗口第6轮，完成后按预算交付。

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

1. 第7轮读取身份加固和两格补预测已提交推送1f0c765，远端SHA一致；两格成功但误差仍失败，历史2次SHA失败原因未知。旧预测和失败不覆盖。
2. 第二个有界窗口最多6轮机制修改、4次完整评估、8小时；第7轮完成一次局部评估，不新增目标LLM实测。固定131格继续作为开发回归。
3. 第8轮统一等待、64次连续提交的12组已运行：2组诊断准入、10组拒收，不生成成本参数。补充审核检查GPU包络与主机时钟合理性及raw摘要；额外r2冻结仅为审核工具修订，未重复实测。第9轮已验证同路径参考并提交3b6b148；第10轮双参考r2的12组24进程已完成，数值和完整性全部通过，4组诊断准入、8组因计时门拒收，未生成系数；17项审核回归通过，73项身份通过，同路径C++参考与Python重建在6组24576点完全一致，保留原F32数学误差，新增限定shape的严格路径一致性检查，重新冻结后才采计时，不能回填旧失败。
4. 旧12配置探针0通过，Q4_K/M64数值失败必须独立诊断；不放宽旧阈值，不选择性补测覆盖失败。Windows桌面负载如实记录，不能将设备进程列表非空一律视为LLM抢占，也不能宣称独占设备。
5. 只有底层证据通过后才接入联合shape与kernel路径受约束的成本候选，并做纯分析/既有模型/新模型消融。GET_ROWS/KV转换和backend split复制/同步仍需独立证据，不拟合LLM端到端时延。
6. 每轮原位更新动态章节、验证后commit和push。新的正式验收集必须独立，当前没有独立盲测或跨硬件通过结论。

## 14. 冻结、复用与循环预算（动态，原位更新）

- native选择SHA固定cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5；模型/原生binary/配置/计时契约未修改，没有新增LLM测量。
- 第一窗口6轮机制、3次完整评估已结束并归档。第二窗口于2026-09-15T22:22:41.555440+00:00开启，截止2026-09-16T06:22:41.555440+00:00；以第7轮和累计3次完整评估为计数起点，不改写旧冻结。
- 第六轮独立r2冻结保留原合同、三路12锚点消融、72静态图核验与三个控制逐请求精确一致证据。2格SHA失败记录不可改写或算通过。
- 独立launch诊断480正式样本原始记录有效，但16组均未过全部稳定性门，未生成系数。stream事件探针第二窗口完成12配置、0通过；第8轮批次探针24进程完成，2/12诊断准入、其余拒收；未改变成本模型。
- 每轮commit+push继续生效；不上传权重、运行库、大型原始trace或整份冻结源码，不混入前端/原生历史WIP。

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
