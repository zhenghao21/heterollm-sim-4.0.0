# strict_raw 合成回归审阅

2026-09-16：只运行合成 JSONL/sidecar 夹具，无 GPU、原生 emitter 或成本拟合。

结果：11 passed。

覆盖 control 和 buffered 的真实 schema 通过例，以及 pair/argv/PID、QPC 区间、数值 stage、footer count/math、模块稳定性、sidecar raw SHA、buffered start marker 的拒绝路径。buffered arm 只接受其三处声明的 numeric cadence；`per_call_final_validated` 仍只由 control 返回 true。

未发现 strict_raw 实现需要 root 修复的问题。测试中的模块篡改首先触发 module lifetime drift，这是 validator 的预期先后顺序；sidecar SHA 反例显式篡改 receipt hash 后被拒绝。
