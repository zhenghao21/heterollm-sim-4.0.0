# TinyLlama 组合策略校准报告（2026-09-13）

本报告在同一 semantic llama.cpp binary、GGUF、硬件与运行时配置下，比较四种形状绑定策略。request marker 的 `first_token` holdout 不稳定，始终未应用。

## 结果（相对于三个已有 merged replay 的 native 中位数）

|场景|策略|TTFT|TPOT|E2E|平均绝对误差|
|---|---|---:|---:|---:|---:|
|short|launch_only|-2.91%|+3.92%|-2.24%|3.02%|
|short|launch_request_begin|-2.91%|+3.92%|-2.24%|3.02%|
|short|stage_only|-30.99%|-29.95%|-33.65%|31.53%|
|short|stage_launch_request|+11.79%|+16.49%|+9.96%|12.75%|
|long|launch_only|-33.13%|+1.77%|-4.66%|13.19%|
|long|launch_request_begin|-33.13%|+1.77%|-4.66%|13.19%|
|long|stage_only|-53.56%|-31.32%|-35.41%|40.10%|
|long|stage_launch_request|-25.26%|+14.05%|+6.82%|15.38%|

## 推荐

- `prompt=3`（Hi.）：使用 `launch_request_begin`。与 launch-only 的模拟时间相同，因为 request_begin 是 16.56 μs 的零推进边界任务；三项相对于 native 中位数约 −2.91%、+3.92%、−2.24%。
- `prompt=29`：使用 `stage_launch_request`。它让 TPOT/E2E 明显优于 stage-only，但 TTFT 仍受长 prompt 请求边界差异影响；不能把该策略外推到短 prompt。
- 不推荐 `stage_only`：短、长场景三项均约 30–54% 低估。

## 证据与限制

- 短/长场景 native 参考分别来自 `tiny_marker_short_merged_replay_r1..r3_v1.json` 与 `tiny_marker_long_merged_replay_r1..r3_v1.json`。
- `launch_request_begin` 已在当前源码以短、长各三次新回放验证，产物前缀为 `tiny_combo_*_launch_request_r*.json`，三次 identity mismatch 均为 false。
- 当前 planner 将 request_begin 作为一次性、零推进边界任务；因此它提供语义与审计证据，但不会重复计入 host/operator wall。
- `first_token` train/holdout 偏差：短 −63.48%，长 −38.07%，按 fail-closed 规则 blocked，不进入任何策略。

完整机器可读结果见 [`tiny_combination_calibration_report_v1.json`](../artifacts/multimodel_next/tiny_combination_calibration_report_v1.json)。