# R22 MMVQ integer-warp issue bound 独立审阅

2026-09-16。审阅对象：worktree `r22_mmvq`，HEAD `728ca98d1fdcfef36965179ee009edb79d09ab99`，基线 `2d4c6c6`。**未发现P1/P2级必修代码错误；建议仅按“默认关闭、显式contract、条件性dot issue下界”整合，不接受其为已验证DP4A吞吐率、完整kernel时延或LLM校准模型。** 这是R22后续候选审查，不属于R21全131结果。没有修改worktree或主区core/R22测量包；只生成本报告。

## 1. 四分区来源与单位：条件性上界有来源，实际执行率没有证明

独立从NVIDIA官方地址读取RTX Blackwell白皮书，文件8,283,392字节，SHA **`906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa`**，与代码常量一致；在内存渲染检查第11页Figure5，确实为四个分区，每分区标示Warp Scheduler/Dispatch、32 thread/clk，旁边为FP32/INT32执行区。本图可以支持每SM至多4个32-lane普通warp issue slots/cycle的**分派上界解释**；不能据图宣称DP4A每SM的实际执行吞吐、latency、依赖链隐藏或硬件occupancy。

官方PTX ISA §9.7.1.24说明dp4a消费两个含四个byte的32-bit输入并累加到32-bit结果；锁定CUDA [common.cuh:743](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/common.cuh:743) 在对应架构分支调用`__dp4a`。但是本审计没有SASS/编译原生指令映射证据。[mmvq_issue_bound.py:122](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/mmvq_issue_bound.py:122) 保留 `native_instruction_mapping_proven=false`、`dp4a_execution_throughput_known=false` 和source mapping条件，属于正确限制。

代码 [mmvq_issue_bound.py:109](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/mmvq_issue_bound.py:109) 的换算是：

- `active_sms_upper=min(sm_count,CTA_count)`；每CTA不能跨多个SM，因此它是最大活跃SM数量上界，不是occupancy利用率。
- 每分区 `32 threads/cycle ÷ 32 threads/warp = 1 warp issue slot/cycle`；容量 `active_sms_upper×4`。
- 总量floor=`warp_issue_slots/capacity`，单warp串行issue floor=`longest_warp_slots/1`，取两者max；**这没有给DP4A依赖latency定价**。
- `service_ns=cycles/frequency_ghz` 的量纲正确：GHz为cycles/ns。代码取声明的SM频率字段，明确标注它不是实际wall-time频率上界；如果实际时钟更高，不能将这个ns无条件当观测wall-time下界。
- `dp4a_weight_dependent_thread_calls`、`dp4a_warp_issue_slots`、逻辑`2MNK`分列；资源work_units仍是既有算术量，未把warp slots加入FLOP/MAC总数。部分warp也按一次issue计数，没有按活跃lane比例虚构分派能力。

**限制（非本次发现的实现错误）：** 此floor成立还需要source DP4A计算在编译后仍对应普通integer warp issue、单warp不能获得多个同类分派slot/clock等前提。白皮书加PTX说明的是结构/语义，不构成当前native kernel的性能测量。对应前提必须随contract和结果保留；整体CostEstimate仍含legacy HBM fallback，也不能把整个返回时延称为严格物理下界。

官方资料地址（本次只读，不保存额外文件）：
`https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf`
`https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a`

## 2. Q4_K/Q6_K计数与边界

本次重新核对`mmvq_work.SOURCE_SHA256`的全部5个锁定源码，实际SHA均匹配。[common.cuh:60](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/common.cuh:60) 将GGML_CUDA_CC_BLACKWELL定义为1200，避免把DGX Spark1210或其他Blackwell代号混入。

| 项目 | Q4_K | Q6_K | 来源/判断 |
|---|---:|---:|---|
| source dispatch M范围 | 1..5 | 1..7 | [mmvq.cu:319](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:319) 的CC1200should_use_mmvq分支明确如此 |
| QK / QI / VDR / block bytes | 256 / 32 / 2 / 144 | 256 / 32 / 1 / 210 | ggml-common及vecdotq宏、锁定block布局；新表值吻合 |
| 每vecdot权重相关DP4A | 4 | 2 | Q4_K QR循环每轮两嵌套DP4A；Q6_K每轮一个，QR=2 |
| Q4常数修正表达式 | 另4个，未价 | 无此项 | Q4 `dot2`与activation有关、可跨output rows复用；不把它作为不可省的权重相关dot量 |
| M1 small-K严格边界 | K<2048 | K<1024 | qblocks `< warps×VDR×32/QI`，等号不算small |
| tail | K必须256整除；N必须rows/CTA整除 | 同左 | 未证明partial output weight-tail allocation时fail closed，不靠padding猜读范围 |

[mmvq_work.py:177](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/mmvq_work.py:177) 默认表仍只有Q5_0/Q8_0；K格式只由显式boolean opt-in打开。warp/row规则沿用generic CC1200：M≤4四warp、较大M两warp，M1小K扩4row、M1大K一row、其他两row。源码[mmvq.cu:699](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:699) 的kbx循环加j(M)/i(rows)嵌套与派生规则一致。

[vecdotq.cuh:508](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/vecdotq.cuh:508) Q4_K数据dot两层嵌套×QR2=4；[vecdotq.cuh:627](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/vecdotq.cuh:627) Q6_K QR2=2。两者权重相关thread calls都闭合为MNK/4，与每dp4a四个乘加、逻辑2MNK吻合。[mmvq_issue_bound.py:82](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/mmvq_issue_bound.py:82) 对每warp使用所属32线程的最大循环次数，保留partial-warp的issue占用，不假定整个block每lane同活跃度。

资格仍限定ordinary contiguous 2D、单channel/sample、无ids/fusion、force_cublas=false和已声明MMVQ dispatch；不外推到Q5_K、IQ、MoE IDs、fusion或row-tail。未找到Q4/Q6的M、QI/VDR、DP4A数据项或严格边界错误。

## 3. MMA替换、重复billing与未价工作

[cost_models.py:2454](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/cost_models.py:2454) 仅当typed issue contract存在时派生vector_bound。GemmWorkload在289–296拒绝mismatched runtime、fused epilogue、其他source partial service等竞争项；planner也拒绝同时启用PRMT partial treatment（[planner.py:9330](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/planner.py:9330)）。

[cost_models.py:2765](F:/codex_project/37_LLMsim/_worktrees/r22_mmvq/src/heterollm_sim/cost_models.py:2765) 将主compute demand从tensor resource替换为shared scalar/integer resource；旧MMA compute demand不再添加。旧packed-weight-transform的scalar service在2647附近明确归零，2776起第二个scalar demand分支对vector_bound关闭。input Q8_1 conversion task、logical input/output/weight字节、原kernel_launch task数量保持。现有测试确认没有同时收MMA+issue、没有另收legacy unpack。

**这不是免费修复全部低估：** 被移除的legacy packed_transform不等于已证明其真实成本为零。代码将 `unpack_cost_priced=false`、`compute_energy_priced=false`、ignored_legacy_packed_transform_operations和unpriced_work列表写出；unpack、Q4常数修正、float scale、warp reduction、barrier、register/spill、指令latency仍未价。启用后即便预测更快或误差变小，也不能宣布这些成本已覆盖。

HBM部分仍使用旧MMA output-tile wave/occupancy fallback（cost_models:2676–2731）；对应metadata注明source MMVQ HBM concurrency未应用、overall_timing_completeness为partial_with_legacy_HBM_fallback。这里是保留已有模型的一项明显限制，不是已验证的MMVQ带宽模型；后续不要将compute条件floor与旧HBM拼接结果整体宣传为下界。

## 4. 默认off：做了真实前后提交数值不变检查

不仅运行新代码里false对missing的测试，还将基线提交的`mmvq_work.py`、`cost_models.py`、`planner.py`用只读git show加载至隔离Python进程，在内存中与HEAD执行相同合成fixture；未checkout、写worktree或加载目标模型。

- **288个Q5_0/Q8_0 source几何case：0差异。** M=1..8；K=32/96/896/1024/2048/4096；N=4/64/256；比较全部MMVQWork字段。
- **30个default-off合成planner场景：0数值差异。** Q5_0/Q8_0/Q4_K/Q6_K/IQ4_XS × M=1/2/4/5/7/8；比较全部任务name与ResourceDemand字段（service_ns、bytes、work_units、energy等）。此检查覆盖了K格式扩展是否意外落入原默认分支的担忧。
- HEAD额外比较四支持格式的`enabled=true但缺contract`与false：资源数值一致，返回uncovered而不猜rate。
- `python -B -m pytest -p no:cacheprovider tests/test_mmvq_issue_bound.py tests/test_mmvq_work.py`：**66 passed，2 skipped，1.12s**。两个skip为worktree缺少可选本地source/历史synthetic trace；锁定5-source SHA已在主区独立只读补核对，仍未用历史trace替代SASS证据。
- 审阅前后worktree git status为空，HEAD始终上述提交。

因此默认off在上述数值域内保持基线；新dataclass字段/新source hash导致的序列化或证据身份变化不应谎称字节完全不变。这里验证的是几何字段和资源需求数值，不是所有可能输入或完整目标LLM结果。

## 5. 整合建议与剩余审查边界

可在保持默认off、显式源/runtime/profile contract、失败回退和全部partial/unknown标签的前提下作为R22实验机制整合；本次没有必须通知原worker立即停止的严重错误。建议将“前一提交vsHEAD”的default-off数值不变矩阵作为持久回归补充，而非只保留同版本missing/false比较。

后续若要扩大声明范围，依次需要：原生指令映射/issue资源资格、unpack/scale/reduction的独立计数和服务模型、MMVQ专属HBM/occupancy资格、实际时钟域验证。未取得前仍须B=unvalidated，不应默认启用，不得由R21目标误差反推系数，也不能重用本审计为R21通过门槛的证据。

本次无native/GPU/目标sim执行、无目标时延读取、无系数拟合或core改动。官方PDF/图仅在内存读取与渲染；唯一新产物为本报告。

## 被审代码哈希

| 文件 | SHA-256 |
|---|---|
| `src/heterollm_sim/mmvq_issue_bound.py` | `3d01065ad62e97a6c47b48cb7d856e2fd32f236da5ec83b0bd0bd75a51a20bd5` |
| `src/heterollm_sim/mmvq_work.py` | `514110d0fe76d9ca39614e44e233bf8a90e6108d7f437b74c6d9054f59ca79d1` |
| `src/heterollm_sim/cost_models.py` | `e029522a3603c1c1d477c0faefa918697cc509e87dad229147b2237108e1211e` |
| `src/heterollm_sim/planner.py` | `a3f156ba4bb7eaee8e41a53896f85029276c960c6cae5d806ea5a660fa36994a` |
| `tests/test_mmvq_issue_bound.py` | `3c396c5251bc4f45d00fc1238ada99b7139450ccb6e08190a663a62cd34b86a8` |
