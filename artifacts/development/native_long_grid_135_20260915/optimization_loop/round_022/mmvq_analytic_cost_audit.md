# R22 MMVQ 纯分析成本下界审计

日期：2026-09-16。状态：只读诊断／可实施方案，未实现。未修改 src/tools/tests；未读取本项目 native_actual、目标延迟、预测误差或目标运行结果；未启动 native/GPU/重模拟。唯一新文件为本报告。

## 判断

可以在不使用目标时延系数的前提下，建立比当前 MMA 回退更符合执行类别的 **MMVQ 源码／PTX 发射成本下界**。已有源码足够确定 Q5_0/Q8_0 的 DP4A 工作量和 CTA/warp/K-loop 几何；官方 RTX Blackwell SM 图可支持一个非常乐观的 warp-dispatch 容量上界。该下界不需要先测 occupancy。

但现有 `GPUProfile.elementwise_gops` **不是已验证的 DP4A 执行吞吐**。将其直接用于整数 dot，或将 `scalar_lanes_per_sm=128` 称为“128 DP4A/SM/cycle”，均缺少证明。当前资料也不足以给出完整 MMVQ kernel 延迟的严格下界：选中运行时的 intrinsic→SASS 映射、每条专门整数指令的吞吐/延迟、实际寄存器/驻留率、缓存状态和时钟上界尚不完整。

建议先提供来源限定的独立诊断下界；需要数值消融时，使用显式 opt-in，将 MMVQ 的 **dot 计算资源由 tensor_core 改为 vector/scalar 发射下界，替换而非叠加**。保留其它旧成本时，应继续声明整项成本为 mixed-source/partial，不能把总延迟改标为完整 source-qualified lower bound。

## 1. 当前实际数值路径

| 位置／函数 | 已发生的工作 | 未发生的工作／影响 |
|---|---|---|
| `planner.py:9218 _declared_mmvq_work` | 校验单一物理 model-weight、非 expert、非 fused-epilogue、固定转换/调度/source refs，生成 MMVQSourceContract | 不从模型名猜 MMVQ；只覆盖 Q5_0/Q8_0、CC1200、M<=8、K%32=0、完整输出 tile |
| `planner.py:9856 _add_rank_gemm` 的 GPU 分支 | MMQ 不适用且 MMVQ 优先时接入 `mmvq_work`，改用 Q8_1 consumer bytes | 没有独立 MMVQ 主 kernel 成本估算器 |
| `planner.py:8642 _mmvq_activation_conversion_workload` | 已有单独 F32→Q8_1 转换任务；写入 padded K，主矩阵只读 logical K 对应的 consumer 区域 | 不能再在新 dot 成本内重复加转换 kernel/转换写入 |
| `cost_models.py:2441 estimate_gpu_gemm` | 写入 `kernel_family=cuda_mmvq_vector_dp4a`、`source_geometry_priced=false` 等标记 | 标记不会改变下面的 MMA 数值计算 |
| `cost_models.py:2540..2600` | 仍计算 MMA tiles、`issued_operations`、tensor-core peak，并乘 inherited efficiency/occupancy | 实际 MMVQ CTA 数、warp 数、K-loop 不参与计算服务时间 |
| `cost_models.py:2644..2695` | HBM bandwidth 仍乘 `mma_output_tile_wave_proxy` 的输出 tile wave utilization | `source_geometry_hbm_concurrency_applied=false` 是诚实说明；它没有取消旧 MMA 带宽回退 |
| `cost_models.py:2719..` | `roofline_ns=max(compute,scalar,sfu,memory)`；dot 的 ResourceDemand 仍占用 tensor_core | 因此即使总数值被 memory 掩盖，资源竞争语义仍然是 MMA |

现有算式明确为：

```text
T_MMA_compute_ns = issued_MMA_operations / (attainable_tensor_TOPS * 1000)
attainable_tensor_TOPS = structural_tensor_TOPS * attainable_efficiency * occupancy
T_scalar_ns = epilogue_ops / elementwise_gops
            + packed_weight_transform_operations / elementwise_gops
            + source_partial_service_ns
shape_HBM_GBps = HBM_effective_GBps * MMA_output_tile_wave_utilization
```

`mma_output_tile_wave_proxy`（`cost_models.py:2374..2425`）使用 ceil(M/mma_m)、ceil(N/mma_n)、ceil(K/mma_k)，独立输出 tile 容量来自 SM×tensor_cores_per_sm×occupancy；它不是 MMVQ 的 CUDA block 分配。`mmvq_work` 和 `mmq_work` 在 GemmWorkload 中互斥，但只有 mmq_work 获得 source-priced 主计算分支。

## 2. 已有 profile 参数与物理单位

`reference.py:363..430` 的 inherited GPU profile 与 `tools/native_llama_compare.py:1280..1287` 的 RTX5080 映射：

| 参数 | 当前值／来源 | 可安全解释的单位与边界 |
|---|---|---|
| `tensor_core.sm_count` | 映射为84 | SM 数；不代表84个SM都在某个kernel瞬间活跃 |
| `tensor_core.frequency_ghz` | 原始映射2.617；固定预测器随后用冻结时钟替换 | cycles/ns；冻结中位时钟不是实际执行期间的严格频率上界 |
| `scalar_lanes_per_sm` | inherited 128 | generic scalar lane 参数；没有 DP4A 绑定 |
| `scalar_ops_per_cycle` | inherited 1.0 | generic scalar op/lane/cycle；不是 packed-dot MAC 或 warp instruction |
| `reduction_ops_per_cycle_per_sm` | inherited64 | generic reduction op/SM/cycle；不能等同源码五轮 float shuffle+add |
| `special_function_units_per_sm` / `special_function_ops_per_cycle` | 16 / 1.0 | SFU 类别，与DP4A无关 |
| `occupancy` / `attainable_efficiency` | inherited0.85 / 0.65 | 既有分析假设；没有本kernel寄存器／驻留率证明，不能拿来声称严格下界 |
| Tensor MMA geometry | inherited16×16×16、4 tensor cores/SM | 不适用于MMVQ整数向量dot的指令计数 |
| `cycles_per_mma` | 根据注释中的112.6 dense BF16 TFLOP/s反推 | tensor专用参数，不可迁移为DP4A周期 |
| HBM profile | 映射960 GB/s，另有 inherited efficiency0.75 | 物理峰值与有效假设须分开；MMA wave折扣也不是MMVQ事实 |
| `kernel_launch_ns` | inherited1000 ns | 既有分析固定值，不是从本次source推导的严格下界；本任务不建议新添启动常数 |

`GPUProfile.elementwise_gops`（`cost_models.py:1876`）等于 `S*lanes*ops_per_cycle*f_GHz*efficiency*occupancy`。该数值单位是 GOP/s，也等于 op/ns。它没有指明 op 是 INT add、IMAD、DP4A、PRMT、LOP3、float MUL 还是 float FMA。不同指令不得直接套同一个“每操作”定义。

`GemmWorkload.operations=2MNK` 按通常的两操作/MAC计数；DP4A 一条线程指令完成四个8-bit乘累加，即8个这种算术操作。不能把2MNK直接除以“DP4A thread-instructions/s”，否则单位错8倍；也不能把 INT32 IMAD 广告TOPS自动当成 INT8 DP4A TOPS。[N1]

## 3. 锁定 kernel 能确定的源工作量

本次对 `source/llama.cpp-semantic/ggml/src` 下五份文件重新计算 SHA256，全部与 `mmvq_work.py:14..20` 的不可变字典吻合：

| 文件（相对 ggml/src） | SHA256 |
|---|---|
| ggml-cuda/mmvq.cu | 14026871030393662628abdbd4937d5cab72031e20ddf582c9de1d7b424bb368 |
| ggml-cuda/mmvq.cuh | 93ef0ea631585f93c35b00087d265e4ca45f82aaca8e7c363d382dc561d85fbd |
| ggml-cuda/vecdotq.cuh | 9d165e8e36db9bdfb69bfc2301e550a58c4a3057907bbf4d075b5f3eb72347a2 |
| ggml-common.h | 3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea |
| ggml-cuda/common.cuh | 1cc3186a56426d5b929e9c29f9fae2d61d583f5f7abf0764d363c21cbbcbe932 |

在现有 narrow contract 内定义：M=token列数，N=输出行数，K=逻辑reduction长度，W=warps/CTA，R=rows/CTA，C=N/R=CTA数；warp=32，qk=32，vdr=2。

`mmvq_work.py:162 derive_mmvq_work` 已确定 W（M1..4为4，M5..8为2）、strict small-K 分支、R、grid/block、每条线程 `I_t=loop_iterations_by_thread[t]`。源码 `mmvq.cu:699..739` 的循环中，每个有效线程K步，对每个M列、每个R行调用一次 vec_dot。

```text
F_thread = C * M * R * sum_t(I_t)      # 已有 source_vector_dot_calls
Q5_0: D_thread = 4 * F_thread          # 每次vec_dot有vdr=2，每轮2个DP4A
Q8_0: D_thread = 2 * F_thread          # 每轮1个DP4A
两者在当前合格范围内：D_thread = M*N*K/4
```

依据：`vecdotq.cuh:173..199`（Q5_0）和`:243..258`（Q8_0），以及 `common.cuh:743..749` 的 NVIDIA `__dp4a` 路径。做了144组纯整数源计数等式核对（M1..8、K若干32倍数、完整N tile），均成立；没有执行模拟或计时。

更合适的发射工作单位是 warp 指令槽。尾部只有部分lane有效时，一条warp指令仍占发射槽，所以不能仅用 D_thread/32 当成精确槽数：

```text
I_warp_DP4A = C*M*R*d * sum_{warp w}(max_{t in w}(I_t))
d = 4 for Q5_0; d = 2 for Q8_0
```

这是源SIMT路径对应的DP4A warp槽计数；把它升级为选中binary的已执行SASS指令数，仍须证明编译后映射。`MMVQWork.to_metadata()` 目前明确保留 `native_dispatch_proven=false`、`binary_source_equivalence_proven=false`，本审计不将其改写成true。

还可新增**计数元数据**，但不急于定价：

- Q5_0 bit unpack 的 shifts/AND/OR、Q5 offset校正；源码表达式可以被LOP3/PRMT等合并，不能按每个C++运算符收一条指令。
- 每次vec_dot的浮点scale和外层float累加；编译器可能做FMA、公共子表达式提升，不能固定“每call几条FFMA”而无SASS。
- `mmvq.cu:742..788` 的跨warp shared写/读、一条CTA barrier、warp0合并；`common.cuh:466..471` 的最终五轮float shuffle+add。当前无fusion且完整tile时，源层跨warpfloat加法槽为 `32*M*N*(W-1)`，末尾float加法和shuffle各 `5*32*M*N`。这是源线程表达式计数，并非编译后最终执行条数。
- `reduction_shared_array_bytes=4*max(1,W-1)*M*R*32` 是源码主数组大小；unused gate数组是否被消除、实际静态shared和寄存器数仍未知。

不要把既有 Q4_K/IQ4_XS 的 `_declared_mmvq_prmt_partial_work`（`planner.py:9320`）套到上述Q5_0/Q8_0规则。它有独立driver/device/source以及M2/M4限制，并明确排除了未通过稳定性门槛的DP4A计价；本任务没有重读或利用它的实测时延。

## 4. 官方资料足以支持哪一种容量上界

[N2] 的RTX Blackwell SM图明确画出4个scheduler/dispatch分区，每个标32 thread/clk；同文v1.1说明FP32和INT32共用执行路径，且并非所有整数指令都达到相同的翻倍吞吐。这支持一个乐观的 **4 warp issue slots/SM/cycle** 上界，而不是“DP4A实际每SM吞吐=128”。

[N3] 的CC12.0章节也记录4个warp scheduler。其通用INT32-core描述与whitepaper新版表达不完全一致；Table7还区分不同整数操作的吞吐，且没有一个可直接搬作本kernel CC1200 DP4A实测速率的明确行。故不从“64 INT32”或“128 lanes”任选一个数做默认速率，也不把Blackwell数据中心CC10.x当成RTX CC12.0。

基于[N2]的发射上界，可采用：

```text
S_active_upper = min(profile.sm_count, MMVQWork.cta_count)
T_issue_lower_cycles = I_warp_DP4A / (4 * S_active_upper)
T_issue_lower_ns = T_issue_lower_cycles / f_GHz
```

推导条件：每个源DP4A warp槽至少需要一条相应native warp指令；一个CTA同一时刻只在一个SM；4取自公开dispatch图并需写入显式硬件来源合同。无需猜实际occupancy，因为使用最大的可能容量只会使下界更低。寄存器压力、数据依赖和少量ready warps会把真实时间推高，不能凭经验再乘某个惩罚系数。

`f_GHz` 若只是冻结的中位值，结果须标为 **给定scenario时钟条件下的下界**。对真实wall time的严格断言还需要执行期间频率上界；GPU Boost标称值也未必是各种OC状态的硬上界。最安全的初始交付是周期下界＋条件化ns，两者分开。

若进一步取得已来源限定的 `dp4a_thread_instructions_per_sm_cycle_upper_bound=r_dot`，可加入 `D_thread/(S_active_upper*r_dot*f_GHz)` 并与发射下界取max。必须明确r_dot单位是线程指令，不是MAC、INT32 ops或tensor TOPS；它必须有硬件型号/CC/指令/编译或公开文档依据，默认值应为缺失而非随意64/128。

## 5. 哪些证据缺失、是否阻断

| 缺口 | 对粗发射下界 | 对可信的MMVQ完整分析成本 |
|---|---|---|
| 选中native binary的DP4A/IDP4A与源码路径映射、compiler flags、实际模板instantiation | 必须保留source/PTX conditional；不可冒充binary exact | 需要离线编译／反汇编／历史对象链接证据闭环；不需要目标LLM时延 |
| CC1200具体DP4A、PRMT、LOP3、IADD/IMAD、FMA、shuffle吞吐及依赖延迟 | DP4A执行吞吐缺失不妨碍更松的dispatch上界 | 阻断指令混合与依赖链定价；不同类别不可共用未验证scalar_gops |
| 每模板寄存器/thread、spill/local bytes、实际static shared | 不需要用未知occupancy缩小理想容量 | 阻断CTA residency、active-warp与spill成本；源码数组大小并非cubin资源表 |
| shared carveout、分配粒度、每SM CTA/warp资源上限 | 可维持最宽的S_active_upper，不声称驻留率 | 需实际CC12.0/device属性及binary资源；通用上限不能当实测occupancy |
| 内存transaction/coalescing/L1/L2命中及跨kernel权重驻留 | 计算下界可独立存在 | 阻断把全部logical bytes称为必需DRAM流量；不能由CTA数推HBM利用率 |
| 时钟严格上界 | 可报告cycles；ns是declared-clock conditional | 阻断真实wall-time严格下界；冻结中位不是频率上界 |
| 能量/指令 | 不影响时间下界 | 不能沿用tensor_energy_pj_per_op冒充DP4A能耗 |

更精确的occupancy可以在未来拿到寄存器/SMEM之后纯分析推导上限，但 occupancy上限也不等于ready-warp利用率。不得将 inherited0.85/0.65重新包装为来源证据。

## 6. 重复计费与资源语义

1. **dot必须替换MMA主计算。** 在现有tensor demand之外再添DP4A demand会同时为同一次数学dot收两类资源，即使roofline取max也会污染多任务并发竞争。新分支应互斥选择MMA/MMQ/MMVQ，MMVQ不再占tensor_core。
2. **不能无条件制造独立FP与INT并行资源。** GB20x统一路径见[N2]。最小方案把MMVQ vector工作落入既有scalar共享资源，或新增共享issue资源并保留与float/unpack的竞争；不要简单将INT、FP各自满峰并行，然后只取max。
3. **转换已有独立任务。** F32读4MK bytes；K按512对齐；Q8_1转换写36*M*Kpad/32；主kernel读36MK/32。生产者写与消费者读是两次不同访问，应保留；不能在MMVQ主kernel再次写一遍转换输出或增加一次转换launch。
4. **packed weight只计一次主访问。** Q5_0=22*N*K/32 bytes；Q8_0=34*N*K/32 bytes。不能乘M或CTA数重收费，intra-kernel重读／缓存命中未知。output为4MN bytes。
5. **旧packed transform proxy不是新指令计数。** `projection_descriptors.py`默认每weight一个dequant操作。新增精确unpack/scale计数后必须替换该proxy，不能叠加；若仅定价DP4A，旧proxy是否保留可作为单独分析legacy项，但整项不是严格下界。
6. **同kernel内部使用正确的重叠关系。** 同一issue瓶颈中的必要指令工作可按已证明共享容量合并；不同执行管线的独立下界取max。不要把全部内存、整数、浮点、shared、barrier时间简单相加。
7. **启动不拆分。** 保留现有一个MMVQ主kernel launch；vecdot次数和warp数只是内部工作，不是新的kernel调用数。转换launch与主launch继续分开。
8. **整个估算器不能轻易叫下界。** 旧HBM折扣、cache复用假设、generic unpack/efficiency及固定launch都可能产生非下界的预测；只能称新compute部分为下界。若做完整lower-bound模式，应另列经过证明的compulsory-memory下界；无缓存状态时不能断言每个weight都必须读DRAM。

## 7. 最小可实施方案（R22以后，当前未改代码）

**A. 实现内部的来源计数／诊断准备；不单独开启一次metadata-only优化轮次。**

- 扩展 `mmvq_work.py` 元数据：dp4a_thread_calls、dp4a_warp_issue_slots、source-reduction-expression counts；不扩展到尚未覆盖的quant formats/odd N/fusion/expert。
- 明确加入来源限定的 hardware-cap contract：CC1200、warp32、scheduler dispatch上界及[N2]位置；`dp4a_execution_throughput`保持unknown，`capacity_kind=dispatch_upper_bound`。
- `estimate_gpu_gemm` 可先只输出 `mmvq_issue_lower_bound_cycles/ns`、时钟条件与剩余未知项；不要将 `source_geometry_priced`误设为full coverage。这一步无需目标native/GPU，也不改变数值预测；它只是B的内部验证准备，不能算一次已完成的成本机制修复。

**B. 建议下一轮直接完成的最小数值／资源消融。**

- 新增默认关闭、能进入freeze和resume校验的 `mmvq_compute_lower_bound` opt-in。只有现有MMVQSourceContract及hardware-cap contract同时合格才进入。
- 在 `estimate_gpu_gemm` 中，MMVQ主dot用issue-bound vector/scalar需求替换tensor demand；原MMQ与generic路径不动。不使用occupancy0.85/efficiency0.65计算新bound；需要这些折扣时必须单独声明为非严格下界的legacy行为。
- launch、F32→Q8_1转换、logical主矩阵读写保留原有次数。把 legacy-memory 模式与 source-compute-bound 模式分别记录，明确 `overall_timing_completeness=partial`。
- `analytical_ops`/work_units若继续代表2MNK，须保留“useful arithmetic ops”标签；指令数放独立字段，不能把warp instruction和FLOP无单位相加。能耗另列uncovered。
- planner对传入GemmWorkload的MMVQ shape及hardware-cap身份作最终校验；动态shape/cache key需包括工作量与cap identity，避免重放旧的指令数或旧容量。

**C. 提高可信度所需的后续独立证据。**

优先获取已绑定编译对象的离线SASS/资源元数据（每模板寄存器、shared、spill、DP4A/整数解包/float指令）。随后才考虑独立硬件指令吞吐/依赖微基准；微基准须独立于目标LLM误差并有自己的稳定性验证，不能借当前LLM偏差反推r_dot、occupancy、HBM利用率或启动常数。若不取得这些证据，停留在A/B的partial状态是合理结果。

建议回归只验证：D=MNK/4、partial-warp issue槽、small-K严格边界、M1/M4/M5/M8几何、source/CC不合格回退；opt-in时tensor dot需求消失而conversion/主kernel数不变；bytes不乘M/CTA；compute bound量纲和f缩放正确；memory legacy标签明确。无需为该前置审计重跑目标网格。

## 8. NVIDIA原始来源及适用界限

以下资料只用于指令语义／硬件上界，kernel算法始终以本报告第三节的锁定源码为准；未采用第三方性能帖子或目标推理benchmark。

- [N1] NVIDIA PTX ISA，Integer Arithmetic Instructions: dp4a；当前页面及archive11.7的同一语义：`https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a`。定义四路byte dot并累加32-bit结果，未给出本卡每SM执行吞吐。
- [N2] NVIDIA RTX Blackwell GPU Architecture whitepaper v1.1，印刷页11 Figure5、页12 INT operation update／Figure6，已视觉核对这两页：`https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf`。用于4×32 thread/clk dispatch和FP/INT共享资源，不将“many integer ops doubled”扩大成所有DP4A/PRMT等都翻倍。
- [N3] CUDA C++ Programming Guide 12.9.1，Table7 与 §20.10.1 CC12.0：`https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#arithmetic-instructions` 及同页Compute Capability12.0章节。用于区别具体指令吞吐、scheduler和编译实现；表格合并列必须核对，不能靠文本抓取后的列位置猜CC12.0值。

结论仅指机制可行性：当前profile没有独立DP4A吞吐上界；官方dispatch资料可另行来源绑定，建立条件化发射下界。尚无证据将MMVQ实际执行吞吐、occupancy或整项延迟标为已知；没有数值改善承诺。成稿后收到根任务主动推送的R21摘要，本报告未用这些数值推导工作量、容量、系数或实施选择。
