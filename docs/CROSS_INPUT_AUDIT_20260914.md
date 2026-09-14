# 跨输入场景证据审计（2026-09-14）

本次审计只检查已经存在的 native/仿真回放产物，没有把一个输入形状的 profile 外推到另一个形状，也没有修改仿真核心代码。

## 可作为当前正式输入场景的证据

### Qwen2.5-0.5B，prompt=8/output=8

证据链：

- profile：`artifacts/multimodel_next/qwen25_medium_prompt8_boundary_profile_v3.json`
- 三次当前源码回放：`qwen25_medium_prompt8_boundary_current_median_r1/r2/r3.json`
- median：`qwen25_medium_prompt8_boundary_current_median_summary_v1.json`
- 当前源码/模型/GGUF/硬件/runtime 锁：`qwen25_medium_prompt8_boundary_current_identity_lock_v1.json`

三次回放均使用同一 prompt 文本 `Explain why deterministic benchmarking matters.`，prompt fingerprint 为 `1d77ea96d081577d79c8502b5221e0c76262b13a70ec62a9d6a1edcceaefdd02`。配置固定为 `ctx=512, parallel=1, batch=64, ubatch=64, threads=16, gpu_layers=-1, FA=off, seed=42, temperature=0, top_k=1, warmup_predict=2`。GGUF 几何/量化检查三次均通过，`identity_mismatch=false`。

三次 native 中位数与仿真结果为：

| 指标 | native 中位数 | 仿真器 | 相对误差 |
|---|---:|---:|---:|
| TTFT | 5.255 ms | 5.6413 ms | +7.35% |
| TPOT | 4.3550 ms/token | 4.6459 ms/token | +6.68% |
| E2E | 36.397 ms | 38.1629 ms | +4.85% |

该场景满足当前 profile 的输入形状和身份约束，可以纳入正式多输入矩阵。

### Qwen3.5-0.8B，prompt=2/output=8

证据链：

- train/holdout semantic trace：`qwen35_hi_prompt8_train_trace_v1.json`、`qwen35_hi_prompt8_holdout_trace_v1.json`
- prefill-scoped profile：`qwen35_hi_prompt8_prefill_scoped_calibration_v1.json`
- 三次回放及 median：`qwen35_hi_prompt8_prefill_scoped_replay_1/2/3.json`、`qwen35_hi_prompt8_prefill_scoped_replay_summary_v1.json`

该 profile 只将 prefill API boundary 应用于一个物理 prefill invocation；decode API aggregate 被排除，以避免与 operator-wall 重复计时。三次结果的 median 误差为 TTFT +8.98%、TPOT −8.45%、E2E −4.29%，且 train/holdout 覆盖状态为 `covered`、缺失 phase/token-shape 数为 0。

## 不能作为严格跨输入复现的产物

| 场景 | 原因 | 处理 |
|---|---|---|
| TinyLlama prompt=29/output=8 | 旧 profile 没有 authored prompt fingerprint；只有单次锁定回放，且没有三次当前源码 median | provisional；不得作为当前正式输入矩阵 |
| TinyLlama prompt=11/output=1 | output 只有 1，TPOT 无定义；现有 profile 与 prompt=3 的 profile 不同 | 仅可做 TTFT/E2E 边界探针，不作为三项误差场景 |
| Qwen2.5 prompt=9/13 | 现有回放曾使用 prompt=8 或 prompt=13 的不同 profile，且 prompt fingerprint/shape 不一致 | 禁止把 prompt=8 boundary 或 kernel-shape profile 外推到 9/13 |
| Qwen3.5 旧 stage/launch 回放 | 使用旧 phase/stage-only 或全局 launch 假设，不能代表新的 prefill-scoped 语义 | 历史诊断，不能进入正式矩阵 |

## 结论

当前可以严格纳入统一误差矩阵的输入证据至少包括 Qwen2.5 prompt=8/output=8 与 Qwen3.5 prompt=2/output=8；TinyLlama prompt=3/output=8 另有 current source lock。现有数据中没有另一个已完成“不同 prompt 长度 + 同模型 + 同形状 profile + 三次当前源码 median + 完整身份锁”的场景，因此不应声称已经验证了同一模型跨 prompt 长度的泛化能力。下一步应针对一个新 prompt 长度重新采集 train/holdout trace、生成对应 shape/boundary profile，再做三次 median。
