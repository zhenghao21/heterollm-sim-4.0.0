# CUDA Graph 独立调度成本测量：尚不能作为通用预测校准

2026-10-08，RTX 5080 / Windows 11 / CUDA 12.8 / 驱动 617.14。

**结论：这些实验解释了为什么不能直接把既有 CUDA event 整段耗时加入仿真，但还没有得到覆盖普通提交、Graph、PDL 和 D2D 拷贝的通用独立成本模型。所有结果保持 `prediction_qualified=false`，没有将这些数值写入预测配置或硬件预设。本文不代表 Graph 预测精度已经修复。**

实验不加载模型权重、不运行 native 模型推理、不读取或拟合模型总耗时。此处的校准样本全部是可控合成操作。

测量前未发现其他模型 / CUDA 计算进程，但桌面、WebView 和 Edge 仍在运行；连续五次 GPU 利用率为 12%–29%，功耗约 19–21 W。没有关闭用户应用。这是当前 Windows 桌面环境中的独立测量，不宣称完全无图形负载干扰；尤其普通提交的重复波动应连同原始记录一起解读。

## 证据与测量边界

- [紧凑结论及全部分类统计](independent_dispatch_summary.json)
- [初始 host callback 门控原始记录](launch_gap_suite.json.gz)
- [host callback + 尾事件查询、预上传对照](launch_gap_suite_flush.json.gz)
- [真正驻留 GPU 的门控原始记录](launch_gap_suite_gpu_gate.json.gz)
- [CUPTI 开关配对的完整 kernel / D2D activity](typed_activity_suite.json.gz)

`.json.gz` 是原始 JSON 的无损压缩，保留每次重复、每个节点的时间戳。42 个 globaltimer 实验配置和 11 个 CUPTI 配对配置均为每个条件 31 次正式重复、2 次局部预热。初始实验测普通提交 / 首次 Graph / replay，后两批额外测预先调用 `cudaGraphUpload` 的首次执行。实际测量时间在 JSON 中记录。此前十模型 native 对照报告及数据未被覆盖。

globaltimer 实验在每个合成 kernel 中记录“有效 body”的开始与结束，按所有 block 的最早开始、最晚结束得到节点包络。对普通串行链：

`间隙[i] = body_begin[i] − body_end[i−1]`

只观察节点之间的 `N−1` 条边。kernel body、DRAM 执行时间与第一个节点的启动延迟都不能再次由这些间隙计费。探针入口 / 出口及 PDL 完成通知的尾部仍在间隙中，因此也不能把它包装为完全纯净的硬件调度常数。存在重叠时记录有符号间隙及区间并集；不将负值作为任务时长加入仿真。

## 普通 CUDA 提交不能用单一间隙常数

128 个节点、1 block、32 threads，使用真正驻留 GPU 的门控后，逐次平均边间隙的中位数如下，单位为微秒：

| 合成有效 body 目标时长 | 普通提交，default 边 | 普通提交，PDL 边 | Graph replay，default 边 | Graph replay，PDL 边 |
|---|---:|---:|---:|---:|
| 0 μs | 3.052 | 3.032 | 0.512 | 0.373 |
| 1 μs | 3.899 | 3.966 | 0.510 | 0.372 |
| 10 μs | 3.195 | 3.056 | 0.507 | 0.375 |

同一普通提交方式的间隙会随 body 时长变化。早期 host callback 门控记录还显示每 11 条边一组的明显脉冲，其中每 22 条边常出现更大的脉冲；普通边的开始间距有 2.048 μs 的量化现象。尾事件查询没有消除该行为。GPU 门控复测仍没有使普通提交间隙变为与 body 无关的常数。

host callback 门控只能证明 CUDA API 已返回，不能证明 WDDM 驱动缓冲区已全部送到设备。后续 GPU 门控先等待 GPU 的 ready 标记，确认门已在 GPU 执行，再提交链、查询尾事件并释放门；这排除了“门只是 CPU 回调”的歧义，但仍不能据此宣称所有 WDDM 命令缓冲区已驻留设备。

GPU 门控的 Windows 可见性问题也被实际排查：仅 volatile / `ld.global.cv`，包括写合并映射页，都未可靠观察到 CPU 释放。最终使用 CUDA **系统作用域 acquire / release 原子操作**。每次有 500 ms 超时，超时样本立即拒绝，未进入结果文件。CPU 观察到 ready、提交命令的时间均远短于超时，失败不能误归因于 CPU 提交等待填满队列。

## 纯 kernel replay 只支持很窄的观察结论

在已测纯 kernel 家族中，replay 的有效 body 间隙相对稳定：default 边约 0.51 μs，PDL 边约 0.37 μs。独立检查包括 16 / 32 / 128 / 256 节点、0 / 1 / 10 μs body，以及 32 blocks、128 threads 的几何变化；这些不是所有组合的笛卡尔积。

这只能描述该合成家族。它没有覆盖真实 kernel 的全部参数、资源占用、copy 边界或 native 执行，也没有测第一个节点的启动成本。首次 Graph 的间隙依赖节点数；预上传没有使首次执行自动等价于 replay。不能把少数纯 kernel 间隙直接用于所有模型的混合 Graph。

## CUPTI 可以看见 copy 边界，但观测改变了 Graph 执行

第二个工具使用本机 CUPTI Kernel9 / Memcpy6 activity，测完整 kernel 和 D2D 拷贝的开始 / 结束。覆盖 default、PDL、交替边，以及来源目录中实际出现的 3584、8192、14336、16384、20480、24576 字节 D2D 拷贝。源节点类型、拷贝字节数、Graph 节点身份和完整记录数均严格核对。每个配置还在关闭 CUPTI 时重复测 CUDA event 整段时间，用来检查测量工具自身的影响。

**零 body 的纯 kernel 与混合链中，CUPTI 开启后 replay 整段时间比关闭时增加 40.64%–53.57%。** 在 1 μs 和 10 μs body 配置中，增幅分别约 19.17% 和 2.98%。两种配置由独立进程顺序运行，保留全部重复波动，没有把单个小值当作无扰动结果。

CUPTI 记录中的 replay kernel→copy 间隙约 97 ns、copy→kernel 约 301 ns；但它们属于已经被观察工具改变的执行条件，不能用于未启用 CUPTI 的预测。PDL 全 kernel activity 还存在约 −42.62 ns 的重叠；交替 default / PDL 链的 PDL 间隙又不同。这说明“每种边固定一个正数”缺少足够依据，也不能简单把负值截成零。

## 后续可执行的边界

1. **普通提交：显式建模驱动供给与设备执行两个队列。** 用独立的受控 body 时长、命令批次、提交速率和节点数实验识别 WDDM 的送入时刻及批次行为；API 返回时刻与设备可执行时刻分别记录。对保留样本验证后再形成该 Windows / 驱动版本的模型。这样才能避免把 CPU / 驱动饥饿再叠加到已有 host submit 成本。
2. **混合 Graph：取得能通过扰动检查的设备边界。** 需要确认驱动或硬件时间戳接口可观测 kernel / copy 的真实就绪与完成边界，并对开关观测的配对结果做定量验证。本次 CUPTI 路径未通过这项检查，不能假设换一个调用方式就已解决。
3. **PDL：从源结构表达可重叠阶段。** PDL 允许后继 kernel 提前进入、等待前驱完成通知；需把“进入 / 等待 / 有效执行 / 完成通知”明确表示，再与物理执行成本结合。若没有这种排程表达，拒绝把负间隙或依赖上下文的间隙强行压成固定任务时长。
4. **验收：先独立留出测试，再进行模型误差对照。** 保留实际 GGUF 和来源构图路径；新成本只能由独立实验决定。native 模型耗时只用于最终验证误差，不用于选择拟合参数。

本次没有继续用新的任意常数尝试逼近 native 总耗时，也没有启用聚合成本回退。

## 队列模型可行性复核

随后仅对已保存的独立数据做 CPU 分析，没有新增 GPU 实验，也没有引入模型实测耗时。结果见 [候选队列模型留出检验](queue_candidate_holdout.json)。候选表达式是：

`普通边间隙 = ceil((前驱 body 时长 + c) / 2048 ns) × 2048 ns − 前驱 body 时长`

每 11 / 22 条边再叠加由训练样本得到的两类脉冲。只用 16 / 128 节点、零 body 训练组；节点数、body 和几何变化保持为留出组。每种门控家族分别估计脉冲相位。这项探索不是预先登记的盲测：2.048 μs / 11 / 22 的结构来自对现有数据的观察，不能据此宣称独立泛化已经通过。

在 GPU 门控家族中，该候选模型的留出整段 span 误差为 default **−1.65%～+4.48%**、PDL **−5.95%～+1.85%**，说明显式队列模型比固定间隙值得继续推进。但目前仍有无法唯一确定的变量：

- `c` 不是已测硬件参数。零 body 训练不能唯一确定它；即使再利用已经看过的 1 / 10 μs 点，约 0.90–1.86 μs 的阈值范围仍能解释同样的量化步长。在未覆盖的中间 body 上，这些等价参数可给出相差一个 2.048 μs 步长的结果。本次候选仅用训练可行区间的中点展示可行性，未写入预测配置。
- 11 节点脉冲相位随 gate 类型翻转，不能假设真实源图的第一个节点总是处于相同驱动命令缓冲区相位。普通提交的每条 API 开始 / 返回时刻、WDDM buffer 提交 / 可执行时刻、起始队列状态都没有被现有记录直接观察。
- 11 节点批次可能还取决于合成 kernel 的参数大小和驱动编码。当前合成 kernel 参数布局固定，真实混合 kernel 的参数、资源占用及 D2D 命令没有得到该队列模型的独立覆盖。

因此后续不必只等待一个“完美常数”：可以继续记录独立探针逐条 CPU 提交时间，使用已有 GPU globaltimer，增加跨阈值的 body、参数字节数和提交速率留出实验，检验一个明确的供给 / 消费队列模型。ETW 能帮助直接验证其中的 WDDM 队列状态，但本次权限探针实际失败，见下一节。

只启用纯 kernel replay 的实测子集也有严格边界。当前数据限定了合成 kernel 的有效 body、退出尾部和 PDL 通知协议；并没有证明真实 GGML kernel 的同名“kernel→kernel 边”就属于相同测量域，且未在无 CUPTI 的混合链中证明上下文不变性。因此不能仅按节点类型将 0.51 / 0.37 μs 应用到真实模型，再把其余边自动落到原分析成本。未来若子集通过独立验证，可以用明确的逐边来源合同划分“已覆盖实测”与“显式分析模型”，显示各自覆盖范围，拒绝所有未声明路径；当前数据尚不支持把任何真实模型边标成已校准。

## Windows ETW 观测路径与实际权限结果

本机已安装 WPR、xperf、logman，WPR 存在 GPU profile，`Microsoft-Windows-DxgKrnl` provider 已注册。只读 manifest 中有 Device（PID→device）、Context（device→context）、HwQueue，以及 QueuePacket / DmaPacket 的提交、开始、结束等数值事件。这是一条可实现的后续观测路径，但不能直接用事件 header 的 PID 丢弃其他记录：一些排程事件由系统线程代办，需要先建立探针 PID 的 device→context→queue 映射。

拟采用的范围是窄事件 ID、短时内存实时 collector，只保存属于独立 probe context 的数值时间戳，其他应用事件即时丢弃；不使用宽泛的 WPR GPU profile，不收集屏幕、文本、网络或文件内容。是否低扰动仍需 observer off / on 配对实测，不能因为使用 ETW 就宣称无开销。

实际权限探针 [etw_access_check.json](etw_access_check.json) 中，`StartTraceW` 返回 **Win32 5：拒绝访问**。专用 session 未启动，provider 未启用，没有 consumer、ETL 或应用事件收集。没有请求 UAC、修改用户组或绕过权限。当前令牌不属于管理员 / Performance Log Users / Performance Monitor Users，也没有 `SeSystemProfilePrivilege`；实际 API 拒绝结果使该 ETW 分支在当前执行权限下受阻。工具为 `tools/check_cuda_dispatch_etw_access.py`。这不阻止继续已有纯 CPU 分析、源图逻辑修复，或不依赖 ETW 的独立探针设计。

## 新增 31 组：事前划分的队列识别与留出

在后续任何新 GPU 测量前，先保存了 [实验清单](queue_model_probe/experiment_plan.json)，随后完成 31 个配置、各 21 次正式重复。训练包含未见过的中间 body 阈值和 32 / 96 / 288 字节实际 CUDA 参数范围；留出包含 0.5 / 1.5 / 3 / 7 μs body、33 / 255 节点、不同几何、交替 / 打乱的混合 body，以及新的 1056 字节参数范围。额外保留逐次 CPU 时间戳观察开关对照。

新工具逐条记录 CPU enqueue 开始 / 返回时刻及 gate 释放时刻，GPU 继续独立记录有效 body 包络；两种时钟未被伪装为同一时基。实际 CUDA 参数范围由 `cudaFuncGetParamInfo` 读取，完整性检查拒绝错误参数字节、缺 CPU 时间戳和提前释放 gate。训练与留出没有因为结果失败而更名或互换。

[测量统计](queue_model_probe/measurement_summary.json) 和 [训练模型 / 留出预测](queue_model_probe/queue_model_identification.json) 显示了进一步可识别的结构：

- 只使用训练数据，量化尾部参数范围缩小为 **(960, 992] ns**，处于 32 ns 探针计时粒度附近。这个参数描述当前合成 body 的有效边界，仍不是已分离的纯硬件启动延迟。
- 参数字节确实会改变批次：32 / 96 字节时识别为 **11 个节点**，288 字节时变为 **10 个节点**。现有训练不足以推出任意参数字节数的 batch 规则；两个 1056 字节留出配置明确记录为不支持，未用近邻值代替。
- 对已覆盖参数范围，四个 body 阈值留出的整段 span 误差为 **−2.34%～+1.42%**，两种混合 body 链为 **−2.07% / +2.50%**，几何留出为 **−3.96%**。33 节点误差 **−5.65%**，255 节点则失败到 **−12.86%**，对应间隙总成本低估 **16.63%**。
- 非批次边的量化步进在 ±64 ns 内解释约 **91%–95%** 的留出边。剩余误差集中在驱动供给、较长链及尚未识别的参数变化，不能以整体均值看起来接近为由判定模型通用有效。

测量时十模型前端矩阵正在使用 CPU。各 case 平均 CPU 占用为 **66.05%–76.73%**，全部采样范围 **57.1%–84.6%**；内存充足。CPU 时间戳观察开关的成对 event 中位数差异为 **+5.34% / −0.66%**，尚不能从不同竞争程度中分离观察本身的影响。因此这批证据用于继续识别模型，**仍然未通过部署资格，不修改成本参数或预设**。

## 10 月 9 日：低 CPU 负载的原清单固定模型复测

前端矩阵结束、临时仿真服务关闭后，于北京时间 **2026-10-09 00:04:21–00:04:32** 完成相同 31 个配置、各 21 次重复。原清单逐字节相同，使用分析工具 `--fixed-model` 读取原模型；参数对象完全一致，**没有重新训练、调整阈值或使用留出反馈调参**。资料仍放在本轮 10 月 8 日起始目录中。

- [两轮逐配置比较](queue_model_probe_quiet/quiet_comparison.json)
- [低 CPU 测量及完整原始文件索引](queue_model_probe_quiet/measurement_summary.json)
- [固定原模型的留出结果](queue_model_probe_quiet/queue_model_identification.json)
- [运行前负载检查](queue_model_probe_quiet/preflight.json)

运行前 CPU 占用为 2.2%–3.9%；各 case 的平均 CPU 占用为 **2.85%–14.25%**，全部采样范围为 0%–18.5%。但 GPU 预检利用率仍约 **25%**、功耗约 19 W。进程列表中未见已知 native / 模型 / 探针计算进程，**尚不能完整归属这些 GPU 活动**；因此这组只能称低 CPU 负载复测，不能称 GPU 完全空闲或完全无竞争。

| 独立留出 | 首轮有 CPU 竞争时的 span 误差 | 低 CPU 负载、固定原模型的 span 误差 |
|---|---:|---:|
| 4 个 body 阈值 | −2.34%～+1.42% | −0.38%～+6.99% |
| 33 节点 | −5.65% | −0.08% |
| 255 节点 | −12.86% | −0.57% |
| 不同几何 | −3.96% | +0.04% |
| 交替 body | +2.50% | +10.52% |
| 打乱顺序 body | −2.07% | +5.80% |

255 节点的明显低估在这次复测中大幅缩小，说明此前长链失败对运行环境敏感；不能再把它全部归因于固定的长度建模错误。与此同时，混合 body 留出的误差仍明显，其中交替链的间隙成本高估 **28.90%**。低 CPU 占用没有使当前供给模型成为通用模型。

CPU 时间戳观察开关的 event 中位数差异仍有 **+7.23% / +10.07%**。这些开关对照由不同进程按原计划执行，GPU 活动也未完全排除，因此不能证明差异全部由时间戳记录本身导致；它们同样不能证明记录过程没有显著扰动。1056 字节参数范围的两个留出配置继续明确标为不支持，没有套用近邻配置。

**复测后的决定不变：`prediction_qualified=false`，不部署、不修改预设，不依据这些留出重新拟合。** 待解决的是依赖 body / 参数 / 运行环境的驱动供给与观测边界，而不是继续选择一个能让平均误差更小的常数。本轮独立测量完成，后续若继续识别，需另外声明训练数据与新的留出验证。

## 工具与回归

- `tools/cuda_graph_launch_gap_microbench.cu`
- `tools/run_cuda_graph_launch_gap_microbench.py`
- `tools/cuda_graph_typed_activity_microbench.cu`
- `tools/run_cuda_graph_typed_activity_microbench.py`
- `tools/cuda_dispatch_queue_microbench.cu`
- `tools/run_cuda_dispatch_queue_microbench.py`
- `tools/analyze_cuda_dispatch_queue_microbench.py`
- `tools/check_cuda_dispatch_etw_access.py`
- 三个对应的解析回归测试文件，共 34 项通过。

工具使用 CUDA 12.8、Windows Visual Studio 2022 工具链，独立编译为 `sm_120`。旧 microbench 未被修改。解析测试拒绝缺节点、错误字节数、重复 Graph 节点、缺测量模式、非有限时间和错误观测边界；body / copy 执行时间与边间隙明确区分。整个测量期间没有停止用户原有 8788 / 8792 服务，GPU 工作与结构捕获按约定互斥。
