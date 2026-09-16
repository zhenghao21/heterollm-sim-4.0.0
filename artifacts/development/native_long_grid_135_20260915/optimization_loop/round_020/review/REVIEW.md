# R20 独立 closeout 与测量设计审查

审查日期：2026-09-16。范围为 R19 CPU sampling probe 的 r4/r5/r6 冻结与质量汇总、R19 graph-gap 协议/源码、R19 GPU 公式诊断及 R20 任务文件。未读取目标 LLM 的原生时延；没有运行计时、原生探针、GPU 或 LLM，也未修改 review 目录以外的文件。

## 可直接推进的结论

### P1：graph buffered 臂的“失败保留”承诺需要先收紧并补预启动收据

`protocol.json` 要求“retain every failed/partial/missing/unstable artifact”，并要求串行流程保留 partial buffer。`graph_gap_probe.cpp` 的 buffered 臂却在完成全部 36 次调用和三个检查块之后，才首次写入 arm metadata 与调用记录；只有 C++ 可捕获异常才会进入 catch 补刷。见 `round_019/graph_gap_probe/protocol.json:quality.failure_policy,execution.sequencing` 与 `round_019/graph_gap_probe/graph_gap_probe.cpp:347-384`。外层 `graph_gap_main.cpp` 可以为可展开的异常保存 error record/receipt，但进程崩溃、驱动终止或不可展开异常不会执行该路径。见 `graph_gap_main.cpp:512-529`。

这会产生一个可避免的不一致：control 臂每次调用都有原始记录，buffered 臂在硬失败时可能只留下文件头或外部 trace，无法保留已完成调用的完整 raw JSONL。这不是性能问题，也不需要任何数值拟合。

**可立即实施的修复：** 在 launcher 启动子进程前，以排他创建的 `launch_receipt` 写入 arm、config、mode、pair、输出路径和 freeze 哈希；在 C++ 第一次 timed call 前写入并 durable-flush `buffered_started` arm metadata。协议应明确：硬终止时该臂保留 prelaunch/started 证据与任何外部 trace，但不声称保留所有内存中的 per-call record；所有此类记录仍为 non-eligible。若后续要增加 phase checkpoint，也必须在 timed phase 外完成，并将这一写入节奏加入 arm 语义冻结。

### P1：R6 CPU 结果已经正确拒绝，不能被解释为可用系数或主成本证据

R6 的 quality 汇总报告数值/结构、身份和频率门均通过，但 36 个 stage 中只有 34 个通过 steady dispersion；它明确写入 `timing_usable=false`、`diagnostic_only=true`、`accepted_for_timing_evidence=false`，失败理由为 `steady_dispersion`。见 `round_019/cpu_sampling_probe_r6/sampling_series_0001/quality.json`。这与 r6 修订说明中的“单次 series、阈值不变、不拟合”一致。见 `cpu_sampling_probe_r6/revision_reason.json`。

因此当前 CPU 输出可作为“数值语义、实际 CPU identity、频率记录和门控流程都被执行”的证据；它**不能**用于更改 sampling 成本、倍率或任何 planner 系数。无需再加一个准备型微基准来确认这个结论；下一次 CPU 收集应先由 R20 reset 工作决定是否能隔离 reset/allocator 的变化，再按照原阈值重新获得独立合格 series。

R6 的硬件冻结本身没有发现漂移：identity freeze 和 run freeze/quality 的哈希绑定一致，且 identity-only 结果声明无 GPU context、无 model load、无 timing values。这个部分不应回退。见 `cpu_sampling_probe_r6/cpu_identity_r6_cpu0/actual_cpu_identity_freeze.json`、`sampling_series_0001/run_freeze.json`、`sampling_series_0001/quality.json`。

### P2：MMVQ 可先做“语义覆盖修复”，不能先做数值率修复

R19 的源码审计已确认：普通 Q5_0 MMVQ 是量化 vector DP4A、浮点 scale 和 CTA/warp reduction 路径；它不是 tensor-core MMA kernel。见 `source/llama.cpp-semantic/ggml/src/ggml-cuda/vecdotq.cuh:175-200,811-828` 与 `mmvq.cu:692-819`。现有 generic GPU GEMM 仍按 MMA output-tile wave 把固定 profile occupancy 投影到 HBM 带宽。见 `src/heterollm_sim/cost_models.py:2347-2404,2496-2513,2589-2639`。

与此同时，`MMVQWork` 已能在固定 CC1200/Q5_0/Q8_0 小 batch 域派生 grid、block、warp、K-loop、Q8_1 consumer bytes 和归约 shared array，但其 metadata 明确写有 `cost_model_applied=false`；当前 planner 的 MMVQ 分支只将 M≤阈值绕过 MMQ，并没有把该工作对象接入成本模型。见 `src/heterollm_sim/mmvq_work.py:79-193` 与 `src/heterollm_sim/planner.py:9750-9798`。

**可立即实施的修复：** 在固定 source/runtime contract 命中时，将 MMVQ 记录为 `cuda_mmvq_vector_dp4a` / `source_geometry_unpriced`，挂接 `MMVQWork` 的 geometry metadata，并让 formal eligibility/unsupported-dimension 明确说明 generic MMA wave 不代表该 source kernel。保留旧的 generic 数值只能作为 legacy diagnostic，不得把它标记为 source-qualified MMVQ 成本。该修复是分类、证据和可解释性修复，不需要把 CTA 数量转换为 HBM 利用率，也不需要读取或拟合 R16 的 rejected device 时间。

不建议现在删除 MMA-wave HBM throttle、改用峰值 HBM，或以 CTA 数量直接生成新 bandwidth/occupancy。三者都会引入尚未获得独立验证的新数值率。R19 已经提出一条可证伪的 CUDA-event/CUPTI 方案；它应服务于将来的 rate model，而不是阻塞上述语义覆盖修复。

## 已核对的正向控制

- R6 在运行前执行了 compiled emitter schema gate；r5 的 schema prerequisite 在 series 后才执行，因此旧 r5 被保留但明确拒绝，r6 没有追认它。见 `cpu_sampling_probe_r5/schema_repair.json` 与 `cpu_sampling_probe_r6/revision_reason.json`。
- R6 build manifest 实际绑定了 r6 的 C++、entry、protocol 与 source identity，未发现 r5 控制文件被静默复用。见 `cpu_sampling_probe_r6/build_manifest.json`。
- graph pilot 的计数自洽：两臂 × 三 pair × direct/profile = 12 native processes；profile export 为两臂 × 三 pair = 6。buffered 协议已经诚实说明它只在三处 phase boundary 做 all-stage numerical block，不能证明每次调用的数值。见 `round_019/graph_gap_probe/protocol.json:arms.buffered,execution.pilot`。

## 静态契约验证

新增只读审查测试 [test_evidence_contracts.py](test_evidence_contracts.py)，验证：

1. R6 不合格时序不会被提升为 timing evidence；
2. R6 run freeze、quality 与 CPU identity freeze 指向并哈希绑定同一实际身份；
3. build manifest 绑定 r6 最终控制文件；
4. graph pilot 的 process/export 算术和 buffered 覆盖限制保持明确。

执行：

```text
.\.venv\Scripts\python.exe -m pytest -q artifacts/development/native_long_grid_135_20260915/optimization_loop/round_020/review/test_evidence_contracts.py
4 passed in 0.10s
```

## 未发现的事项

没有发现需要撤销 R6 identity freeze、重写其频率门或放宽 graph/CPU 原有质量阈值的问题。没有依据将 R19 GPU 诊断中的 rejected operator 数据转化为成本系数。
