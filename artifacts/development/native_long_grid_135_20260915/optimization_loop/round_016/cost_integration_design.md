# R16 独立 kernel 成本接入设计（只读审查）

日期：2026-09-16。范围：审查现有成本接口、物理调用链及 R15/R16 合成算子证据；本文件未运行 GPU、未修改成本代码、native、任务书或既有冻结结果。

## 结论

建议增加一个与 LLM 阶段校准分离的、默认关闭的独立 kernel 证据入口。第一步只接纳通过质量门的**同一硬件/软件、同一缓存条件、精确联合 shape、同一实际 kernel 路径**。先接转换、主计算和 fixup 的设备区间，主机图调用包络继续独立报告。

不要把 R15 的整次图调用 wall 写入 GEMM，也不要直接把新证据放进现有 `NativeCalibrationProfile` 的阶段均值表。现有严格键可以借鉴其拒绝缺字段的行为，但其调用方 shape 是输出 `N×M×1×1`，没有 K，不能作为通用 GEMM 性能键。现有校准写入某个 compute demand 的机制，也不能证明测到的完整 kernel 时长就是纯计算资源占用。

R15 证明了新 Nsight/CUPTI 可以提取当前驱动的调用结构；它自己仍标记 `calibration_eligible=false`。R16 的 26 组只是第一批低层支持域，不是五模型全域标定，更不是独立最终验收。

## 1. 已核对的实际证据

来源：`round_015/trace_2026_5/result_summary.json`、`mapped_calls.json`、`README.md`。

- Q4_K、F32 输入输出、M=64/N=4096/K=1024；24 次单图调用对应 72 个图内 kernel，24 个清扫 kernel 均在测量 NVTX 外。
- 图内次序是 `quantize_mmq_q8_1` → `mul_mat_q` → `mul_mat_q_stream_k_fixup`。同一次调用通过线程/API correlation/process 身份归属，不能仅按 kernel 名或时间包含关系聚合。
- 正式 20 次的设备 kernel 区间并集中位数为 15,648.5 ns；profile 主机 wall 中位数 56,500 ns；无注入对照 wall 中位数 48,600 ns。三者不是可以相加的成本。
- 分 kernel 正式中位数：conversion 1,568 ns、main 10,096 ns、fixup 3,936 ns。**分项中位数之和不必等于逐调用总量中位数**，报告和校验应先逐调用求区间并集再统计。
- profile/direct 主机中位数差 +16.255%；协议明确它只有一次未锁频顺序对照，混有状态漂移，不能据此推导纯 profiler 开销或标定系数。
- 清扫输入实际为 268,435,456 B，L2 为 67,108,864 B，满足协议的 4×L2 容量；这只是清扫条件，不是 HBM 事务已测量的证明。
- 旧 probe 缺完整原始每调用 QPC tick、完整构建依赖闭包及逐调用充分数值检查。后补哈希不能修复旧运行的证据缺口。

因此：这些数值可定位成本归属和设计提取器，**不可作为本轮直接启用的 profile**。

## 2. 当前代码的精确接入位置

| 位置 | 已有能力 | 本轮应采用的方式 |
|---|---|---|
| `calibration.py:180` `_profile_identity_matches` | 对已填写的模型/硬件/运行时身份比较；未填写的字段不比较 | 新 kernel profile 必须要求硬件、有效硬件参数、运行库/驱动/构建身份非空；不能借空字段绕过匹配 |
| `calibration.py:251` `canonical_exact_operator_key`、`:398` `resolve_exact_operator_calibration` | 六维严格匹配，缺键不会查阶段均值 | 借鉴拒绝逻辑；不要复用原输出 shape 或模型阶段作为物理算子键 |
| `calibration.py:634` `calibrate_cost_phase` | 替换某个计算 demand，可另开内存/launch 校准 | 不把完整设备时长塞成纯 tensor/scalar 吞吐，不让旧阶段 profile 与新 kernel profile 同时拥有同一时间区间 |
| `planner.py:9082` `_declared_mmq_work` | 验证物理投影、格式、F32 输入输出、固定后端及 MMVQ 优先条件；导出 MMQ 工作量 | 建立本次物理调用的共同 `kernel_query`；未通过这些语义条件时拒绝新 profile |
| `planner.py:9716–9799` GPU GEMM 分支 | 独立发出 MMVQ/MMQ activation conversion，再估计 main | 先完成物理 shape/临时数据量和 dispatch 判定，再匹配成本；不能在模型级逻辑投影拆分前匹配 |
| `planner.py:10050` MMQ fixup | 主计算后独立转换成 tensor kernel，P=0 也可保留 launch-only | 带同一个物理调用身份和 `role=fixup`；测得的非零控制 kernel 不得因算术量为零而消失 |
| `planner.py:10207` `_add_rank_tensor_kernel` | 转换/fixup 等独立 launch 和设备资源成本 | 补传完整 query；当前它给旧校准重组的 `cal_meta` 没有完整 dtype/layout/kernel/MNK，不能指望自动命中 |
| `cost_models.py:2412` `estimate_gpu_gemm` | launch 与 main 分相；main 内 tensor/scalar/SFU/memory 取最大 | 精确保留分析基线；新证据只能拥有 main 设备包络，不拥有 conversion/fixup 或 host wall |
| `cost_models.py:2814` `estimate_gpu_tensor_kernel` | tensor kernel launch、scalar/memory 与 launch-only | 同一接口处理 conversion/fixup；一个角色一个计时所有者 |
| `cost_models.py:905` `CostPhase`；`contracts.py:184` `ResourceDemand` | 一阶段并发资源需求取最大；各资源独占服务且可不同时间释放 | 这是时间边界与资源占用的约束，不能把完整 kernel elapsed 分别解释成已测 tensor 与 HBM 服务量 |

`_declared_mmvq_prmt_partial_work` 当前明确只替换已获证据的 PRMT 部分，DP4A 主体及既有 tensor/memory 仍分析回退。不能因 main 有了一次设备时长，就把之前的 PRMT 部分额外串行加到它后面。

## 3. 最小新增数据契约

推荐新模块 `kernel_calibration.py`（名称可调整），而不是让 `NativeCalibrationProfile` 继续承担两类不同证据。可以共享已有数值合法性与文件身份工具；旧 native/阶段 profile 原样保留。

建议 profile 包含：

- `schema`、`profile_id`、完整 profile SHA；生成器、extractor、probe、协议、源码/构建依赖及原始证据索引 SHA。
- `evidence_kind=independent_synthetic_operator`；明确 `target_llm_latency_used=false`。模型 SHA 和 prompt fingerprint 最多作为外部溯源，不参与查表或拟合。
- 实际观测的 GPU UUID/SM 架构/SM 数/L2、驱动及 CUDA runtime/ggml 动态库 SHA、编译选项、影响 kernel 选择的环境变量。虚拟 HBF/HBM 参数还须绑定**有效硬件 profile SHA**，否则同一 GPU UUID 下改带宽仍会误用旧端点。
- `measurement_boundary=cuda_device_kernel_interval`；`cache_protocol=cold_sweep_ge_4_l2`、清扫大小和顺序。不得将其命名为“已证明冷 HBM”；不得与热权重循环或 1024 次 burst 的每调用值混合。
- 预先冻结的质量状态、计时扰动判据、数值判据、重复样本量、失败样本、适用域与拒绝原因。未通过的 entry 仍保留，不能靠删除失败组提高覆盖率。

每条键必须是结构化值，而非不透明模型阶段字符串：

`(op, role, M, N, K_logical, K_executed, activation_dtype, weight_format, output_dtype, accumulator_dtype, contiguous/strides/layout, kernel_family, kernel_variant, dispatch_signature, cache_protocol, hardware_profile_id, runtime_profile_id)`。

- `role` 为 conversion/main/fixup；保留原始 demangled/mangled kernel 符号、模板参数及实际 grid/block/shared-memory。
- `dispatch_signature` 应由源码选择规则和 shape 决定，并由对应合成 trace 验证；不是通过待预测 LLM 的实测时延选择最快分支。
- conversion 的源 F32 shape、MMVQ 的 Q8_1 padding 与 MMQ 的内部布局必须区分；Q5_0/Q8_0 的 MMVQ 与 MMQ 参考数值路径不同，不能只按“q8 激活转换”合并。
- 保存完整父调用 shape 可避免先错误合并。只有源码证明某个角色与 N 等维度无关，且独立测试证实后，才允许按该角色的输入布局消去这一维；消维规则也进入冻结。
- 同时保留原始观察、统计值和选用证据项，支持逐条解释预测用了哪次底层测量。

## 4. 时间线所有权与最小安全成本策略

一条物理 MMQ 调用的所有权应是：

`launch_conversion → conversion_device → launch_main → main_device → launch_fixup → fixup_device → boundary_wait`。

这是依赖与成本所有权示意，不主张 CPU 提交必然与设备完全串行。后续重叠由调度层的依赖与资源约束决定；不能通过删掉已知依赖来压误差。

- 单个 kernel 的 `end-start` 只覆盖该设备 kernel；转换、fixup 和主机 API/等待不含在 main 中。
- NVTX graph wall、同步 API wall、首末 kernel span 包含重叠、主机供给空隙或等待。仅用于独立诊断/闭合检查，不能把它们再加到各 kernel 和上。
- 不用 `host_wall - kernel_sum` 得到“launch 常数”：区间可能重叠，设备 clock 与 QPC epoch 未对齐，差值不唯一对应一个资源。
- 一组调用缺少任何预期 kernel、归属冲突、重复关联、图外清扫重叠时，整个调用不可生成可用完整链条；保留失败证据。不同角色允许独立可用，但必须有明确 partial 覆盖统计。

**建议首个候选只做精确点的设备时长下界接入，不声称完成资源分解。**

在已经从锁定源码/调度契约证明为串行设备 stream 的范围内，为对应物理角色增加一个 `device_kernel_envelope` 时长需求，和原有分析资源需求放在**同一 `CostPhase`** 中取最大：

`T_role = max(T_role_analytical, T_role_device_measured)`。

该 envelope 绑定经证明的执行 stream/service owner，不伪装成 tensor 吞吐或 HBM 带宽。原有字节量、分析访存需求、PRMT 等分析资源需求保留，不再额外串行添加一次测得时长。调度契约不能证明 stream/资源关系时，拒绝激活这一候选，保留只读诊断。

这个方案的适用范围和不足必须直接写入预测：

1. 它是 `microbench_exact_device_floor_plus_analysis`，不是“精确实测替代”或“已标定纯算力”。分析值大于测量时不会被强行调小；报告该冲突并定位原因。
2. 它能避免把一个含访存的 kernel elapsed 全部写成 compute service，也能保留原来的分析交通量；但各资源真实占用时间尚未因而得到测量。并发资源竞争改善必须单独验证。
3. 只有同一有效硬件/软件/缓存条件可以使用。更改 HBF/HBM 带宽、硬件、driver 或内核时回退分析，并标注域外；否则固定设备时长下界会错误地压住 HBF 带宽收益。
4. 源码证明单 stream 足以确定排队顺序，但不等于真实 LLM 权重已被证明处于清扫后的状态。未证明缓存可迁移时，该成本只能称为条件预测，不能提升为已验证的通用模型。
5. 若后续希望对 HBF 扫描也保持响应，应另做独立算力/访存微基准和可验证的资源分解，建立有物理含义的局部成本曲面。不能简单把此处完整 elapsed 除以 FLOPs 生成全局 TFLOPS。

若团队选择完整设备 duration 替代而非上述下界，必须先补齐资源占用/stream 范围契约，并明确其仅保证已测硬件上的隔离算子时长；不能悄悄把其他 memory demand 清零来获得更好的 LLM 指标。

## 5. 缓存、dispatch 与插值规则

R16 计划训练 18 组：Q5_0/Q8_0 × M{1,4,64} × N{128,896,4864}，K=896；验证 4 组为 M{2,32}/N1792/K896 × 两格式；对齐控制 4 组为 K1024/N896/M{4,64} × 两格式。

- M1/4 与 M64 分属 MMVQ/MMQ，绝不能跨分支插值。
- M2 可能使用与 M1/4 不同的模板/列块配置；M32 与 M64 也未必具备相同 tiling/stream-K 分区。必须看实际 kernel 和源码 dispatch，不因数值位于范围内就通过。
- K896 和 K1024 不是“相近 K”：逻辑 stride、padding、完整 K iteration 与尾部读取条件不同。先作为边界验证；不得将 K1024 的吞吐直接乘 896/1024。
- N1792 是未见联合 shape。只有 N 轴相邻点拥有同一路径，且离散块数/波次机制得到验证，才能候选插值；仅有 N 最小/最大包围不够。
- 第一版精确键表在 M2/32、N1792 应返回 `analytical_fallback/unseen_joint_shape`。这是正确的覆盖行为，但不算已完成微基准插值泛化。
- 如训练一个 source-aware 曲面，冻结后在上述完整留出组上比较；不能看过这四组误差再调参数，随后仍称它们独立验证。每次揭示后的用途需要登记。
- cold sweep 与 hot-cache 重复是两套 profile，不做平均。计划只做 cold 条件时，hot 或未知 cache 的 query 明确拒绝该表。

## 6. 实施顺序与可复用接口

1. **证据解析先行**：复用 R15 的 NVTX/API/correlation 严格归属思路，新建 extractor 版本；输出逐调用 conversion/main/fixup 设备区间、并集、span、host submit/wait/total；保留 raw QPC tick。不要更新旧 R15 结果。
2. **schema/loader/resolver**：只读创建独立 profile，身份与数值状态 fail-closed；严格联合键返回 hit/fallback/blocked 及理由。先不改默认预测输出。
3. **物理调用 query 下沉**：在 `_declared_mmq_work` 之后建立 query 并传给两个转换分支、main 和 fixup；涉及每调用/request 的身份，与模型名称无关。每个角色记录实测与分析基线。
4. **有边界的成本 adapter**：在 `estimate_gpu_gemm`/`estimate_gpu_tensor_kernel` 的设备阶段或独立的 CostEstimate adapter 接入；launch phase 原样保留。新 profile 与旧 native stage/memory ownership 不可同时写同一角色，冲突拒绝而非按优先级偷偷覆盖。
5. **缓存键与冻结**：成本 memoization key 加 profile SHA、精确 query、边界策略和生效硬件 profile SHA。profile 变了不能沿用同一进程的旧 CostEstimate。预测记录这些身份并冻结前/中/后验证。
6. **先结构与合成留出，再 LLM 回归**：静态验证每角色只收费一次、shape/data量不变；在合成留出上看候选是否值得启用，再复用现有固定 native 实施 LLM 对照。不会新测目标 LLM 来拟合。

若现阶段没有可靠的设备服务 owner，先交付 resolver 和预测覆盖审计，不应把未解决的资源归属包装成成本优化已完成。

## 7. 必需验证与放行条件

- **身份负例**：缺硬件身份、空 DLL SHA 表、不同 driver、同 UUID 但 HBF/HBM 有效参数改变、不同 cache policy、未冻结 extractor，全部拒绝，不能因为模型 SHA 为空而放行。
- **键区分**：同 M/N 不同 K；同 nominal bits 不同 Q5_0/Q5_K；同 shape 不同 stride；MMVQ↔MMQ；不同模板/fixup形状；CPU目标；融合 epilogue；专家/scatter 等未覆盖路径，不能误命中。
- **数值门**：MMVQ 的 half d/half 原和与 MMQ D4 的 F32 scale/有符号量化路径分别验证；保留相对原 F32 数学结果误差和源码路径误差，不能仅放宽原参考容差。
- **原始计时门**：首轮、warmup、formal 全部检查状态与数字；QPC ticks 正序、换算可重建；完整设备 kernel 归属；清扫无测量重叠；profile/direct 使用同份输入、参数、DLL、缓存协议。
- **扰动门**：重复进程级交叉顺序对照及固定前置判据应在采集前确定。30 次同进程重复不能独自证明注入低扰动。若 wall 对照仍不能分辨 drift，kernel 证据保持 diagnostic；不得只挑较稳定部分自动转为 calibration eligible。
- **不重计费**：conversion→main→fixup 每次一个 role；main activation 临时输入字节替换 F32 原输入；零工作 fixup 仍有真实 launch/设备控制区间；launch、memcpy、图同步分清；总体 device 成本不叠加 graph wall。
- **调度验证**：同 stream 不发生非法 overlap；有独立 stream/拷贝时不得引入未经证明的全 GPU 锁。保留分析资源与添加 envelope 的策略必须有针对性的反例测试。
- **纯分析回归**：flag off、缺 profile、uncovered shape 时预测除审计 metadata 外数值完全不变。
- **冻结与渗漏检查**：证据中没有目标模型完整延迟、prompt fingerprint 查表、模型名分支或场景倍率；冻结 hash 覆盖 profile、query/adapter、extractor 与聚合脚本。
- **比较分层**：至少纯分析、现有已冻结候选、新 kernel 候选三路。先报告合成训练/留出、命中角色数和未覆盖原因，再报告固定131格 Engine 三项逐格变化、最差组、绝对毫秒与失败分母。现有 native 是开发/回归，不重新称盲测。

## 8. 本轮不应作出的承诺

26 组合成测量不会自动覆盖 Q4_K/Q6_K/IQ 格式、所有维度、KV、GET_ROWS、attention/SSM、CPU、host 调度或异步传输。图包络仍比 kernel 并集大很多时，应定位已发生的主机提交、分配、等待和供给间隙，不能把差额揉入 GEMM 或显存系数。

这份方案的可验收交付是：独立证据能被严格归属、精确命中、拒绝域外并不重复计费，同时明确它对资源竞争与 HBF 扫描尚未提供的证据。只有新数据过门、结构测试通过、未参与修改的样本改善且其他组没有失控，才应在新的冻结版本中启用成本候选。
