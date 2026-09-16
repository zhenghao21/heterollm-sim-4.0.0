# R19 CPU sampling probe 独立静态审查

审查日期：2026-09-16。仅阅读 `PREPARATION_READY.json`、`protocol.json`、`entry.py`、`cpu_sampling_probe.cpp` 和来源身份文件；没有编译、加载 DLL、运行探针、调用 GPU 或执行模型推理。

## 结论

**可以保留当前准备材料，但不建议在修复下列两个测量接受门之前启动正式测量。** 两项都不阻止 root 在之后单独进行“仅编译”步骤；它们阻止把测量系列判为可用时序证据。

### P1：频率漂移没有被完整地归入“仅诊断”，且质量门仍会给出通过

协议明确说 `ProcessorInformation` 的前后报告发生变化时，该 stage 只能诊断使用。代码却只比较 `current_mhz` 和 `limit_mhz`，遗漏 `max_mhz`：`cpu_sampling_probe.cpp:193`。随后 `entry.py:114-149` 只检查结构、精确数值和重复次数，未拒绝或标记不可用于时序的 `frequency_changed_diagnostic_only=true` stage；`quality.json` 仍写入 `exact_numeric_and_structure_pass=true`。

**测量前要求：** stage 稳定性比较必须覆盖 `max_mhz`、`current_mhz`、`limit_mhz`；`validate_result` 和最终 `quality.json` 应明确将任何变化标为 `timing_usable=false` / `diagnostic_only=true`。数值正确可保留，但不可与时序可用混为同一个通过状态。

### P1：实际 CPU 硬件身份只记录，未被运行冻结或允许列表约束

协议和 C++ 会记录 CPUID brand、逻辑 CPU、group、affinity mask 与 OS MHz；线程也会在每个 stage 前后验证固定在选择的 CPU 上，见 `protocol.json.identity`、`cpu_sampling_probe.cpp:139-168`。但 `entry.py:131-149` 的 run freeze 只有 `cpu_index`，没有允许的 CPU brand、可用 group/processor count、进程 affinity mask 或选定硬件身份的冻结值；`validate_result` 也不验证这些实测字段。当前任意满足 `0..63` 的逻辑 CPU 都能产生通过的质量文件。

**测量前要求：** root 选择 CPU 后，在 `run_freeze.json` 写入并在三份结果中验证允许的 CPU brand、group、active processor count、logical CPU、实际 thread affinity mask；至少要把同一系列三进程的完整硬件身份一致性作为质量门。若有既定主机身份，应由 root 预先允许该品牌/拓扑，而不是只在原始结果中记录。

## 通过的静态检查

- **K=1 数值语义：通过。** 3 个 V（32768、131072、262144）均为 2 的幂且小于 2^24；单调和确定性 Fisher-Yates 模式都由有限、位级可表示且唯一最大值的 logits 构成。candidate 检查每个 `id/logit/p`，原 DLL top-k 检查唯一 argmax、`size=1`、`sorted=true`、`selected=-1` 和同一 backing pointer。见 `cpu_sampling_probe.cpp:55-76,179-185,202-237`。
- **原 DLL 生命周期与路径守卫：通过。** 只使用绝对 `llama.dll`；加载后检查主 DLL 与 `ggml.dll` / `ggml-base.dll` 实际来自同一锁定目录；导出地址必须由该模块拥有；析构时先释放 sampler，再释放 DLL。见 `cpu_sampling_probe.cpp:85-124`。入口还在每个进程前后验证冻结的 DLL、依赖、来源、头文件和构建输入身份，见 `entry.py:58-78,106-112,138-146`。
- **阶段边界、冷暖区分与循环计数：通过。** candidate 首次调用和 top-k 首次 apply 独立记录，不与 steady 混合；candidate 稳态复用容量，top-k 每次在时窗外 memcpy/reset；两个计时窗顺序且不嵌套。每 stage/case/process 是 16 warmups + 64 原始 steady ticks；3 process × 6 cases × 2 stages × 64 = 2304 steady ticks。见 `protocol.json.stages/first_use/clock` 与 `cpu_sampling_probe.cpp:202-231`。
- **迁移与线程固定：通过。** 单 group / 0..63 限制明确；进程允许 affinity、`SetThreadGroupAffinity`、开始/结束的实际 processor 与 mask 都检查。见 `cpu_sampling_probe.cpp:139-168`。这能阻止已检测到的迁移，但不能替代上面的硬件身份冻结。
- **无 LLM 拟合和范围控制：通过。** 接口没有模型或 LLM 输入；只输出原始 QPC ticks，不做 observer subtraction；质量输出明确 `fit_performed=false`，131072 为预先声明的 holdout。bias/suppress、RNG、accept/history、同步和完整 chain 都保持未计价。见 `protocol.json.train_holdout/unpriced_scope`、`entry.py:149`。

## 证据限制

candidate loop 仍是外部编译的 source-equivalent wrapper，不是 `common.dll` 的二进制等价测量；源文件和协议已正确注明这一点。原 DLL stage 是 `llama_sampler_apply` 的整体 top-k apply，不应描述为纯比较指令成本。即使修复两个门，结果也只能作为所定义复用热缓冲区的低层 CPU 证据，不能相加为完整 native sampling cost。
