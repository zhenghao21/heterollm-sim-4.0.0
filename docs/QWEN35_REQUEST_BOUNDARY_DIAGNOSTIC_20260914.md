# Qwen3.5 请求边界校准诊断（2026-09-14）

本次审计只检查 `qwen35_launch_phase_audit_v1.json` 中 phase 外同步是否可以作为请求级 TTFT/E2E 边界项。结论为 **暂不校准**，没有修改 planner 或 calibration profile。

## 证据

| 项目 | train | holdout |
|---|---:|---:|
| outside-phase sync 次数 | 14 | 14 |
| outside-phase sync 总时长 | 4.296877 ms | 3.766938 ms |
| prefill phase span | 4.132940 ms | 4.652735 ms |
| decode phase span | 17.062415 ms | 17.214436 ms |
| 最大 post-prefill sync | 2.751110 ms | 1.808682 ms |
| formal prompt tokens | 2 | 2 |
| formal predicted tokens | 2 | 2 |

outside-phase sync 中位数为 `4.031908 ms`，两次样本相对离散度为 `13.14%`。它位于 phase marker 之外，但并不等于一个请求边界：区间包含多个 CUDA stream synchronize，既有 prefill 完成后的长同步，也有 decode 前后的短同步。

## 阻止直接下沉的原因

1. nsys train/holdout trace 的 formal 请求是 `prompt=2, predict=2`，而当前仿真对比回放使用 `prompt=2, predict=8`。输出长度和 decode 调度不同，不能把 p2 的边界值转移到 p8。
2. trace 只有 `phase:prefill` 和 `phase:decode` NVTX 范围，没有 `request_begin/request_end`、`first_token/last_token` 或 HTTP 到达/响应标记，无法把 phase 外同步唯一归属到请求级 TTFT/E2E。
3. 当前 operator-wall profile 已包含算子执行期间的 launch/同步影响；把 4.03 ms 直接加到请求节点可能重复计入同步或 CUDA API launch。
4. 阶段范围与请求级 wall time 不一致：train 的 prefill span 加最大 post-prefill sync 已约 6.88 ms，holdout 约 6.46 ms，而外部 formal prompt timing 还受请求调度、采样和服务端边界影响。

因此报告标记为 `diagnostic_only`，`apply_request_boundary_calibration=false`。不添加未经验证的全局 TTFT/E2E 偏移。

## 下一步取证

在与正式回放完全一致的 llama.cpp 命令下，使用 `predict=8` 重新采集 train/holdout，并在 server request lifecycle 增加成对的 `request_begin/request_end`（至少 `first_token/last_token`）NVTX 标记。只有当两个同配置 trace 的边界项稳定、identity 一致，并且从 operator-wall/launch/sync 计费中明确扣除后，才建立 request-level additive calibration。

机器可读证据：[qwen35_request_boundary_diagnostic_v1.json](../artifacts/multimodel_next/qwen35_request_boundary_diagnostic_v1.json)。
