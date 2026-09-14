# Qwen3.5 `Hi.` / predict=8 精确 phase 取证与 prefill 边界回放

采集和回放均使用同一项目 37 GGUF、semantic direct `llama-server.exe`、同一硬件和固定配置：`prompt=Hi.`、真实 `prompt_n=2`、`predict=8`、`ctx=512`、`parallel=1`、`batch/ubatch=64`、`threads/threads_batch=16`、`gpu_layers=-1`、`temperature=0`、`top_k=1`、`seed=42`、warmup predict=2。

## 新采集的 train/holdout 证据

两份 nsys trace 均通过 GGUF parity，且 formal timing 都是 `prompt_n=2,predicted_n=8`。phase extractor 识别到 1 个 prefill range 和 7 个 decode ranges：

| 阶段 | train launch | train sync | holdout launch | holdout sync |
|---|---:|---:|---:|---:|
| prefill | 892 次 / 3.003390 ms | 18 次 / 0.191325 ms | 892 次 / 2.715763 ms | 18 次 / 0.201133 ms |
| decode | 5110 次 / 14.646650 ms | 126 次 / 1.441222 ms | 5110 次 / 14.156736 ms | 126 次 / 1.430086 ms |

每份 API trace 排除了 56 个跨 phase 边界调用。prefill 的 launch+sync 合计为 `3.194715 / 2.916896 ms`，按 train/holdout 中位数得到 `3.0558055 ms`，只绑定到一个 prefill invocation。

## prefill-scoped profile

`qwen35_hi_prompt8_prefill_scoped_calibration_v1.json` 仅包含：

- semantic operator-wall stage calibration；
- `phase_boundary_policy=one_task_per_phase_invocation`；
- `phase_boundary_ns_per_invocation.prefill=3055805.5`；
- decode boundary 刻意不启用，避免将 decode API aggregate 与 operator-wall/decode runtime 重复计费。

## 三次同配置回放

| 指标 | native 中位数 | simulator 中位数 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| TTFT | 8.072 ms | 8.796592 ms | +0.724592 ms | +8.98% |
| TPOT | 6.003143 ms/token | 5.495850 ms/token | -0.507293 ms/token | -8.45% |
| E2E | 49.386 ms | 47.267543 ms | -2.118457 ms | -4.29% |

Native 三次 CV 为 TTFT `9.87%`、TPOT `6.88%`、E2E `6.05%`；三项误差均低于 10%。完整逐次结果和锁定 SHA 见 [qwen35_hi_prompt8_prefill_scoped_replay_summary_v1.json](../artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_replay_summary_v1.json)。

## 文件

- [train nsys profile](../artifacts/multimodel_next/qwen35_hi_prompt8_train_nsys_v1.json)
- [holdout nsys profile](../artifacts/multimodel_next/qwen35_hi_prompt8_holdout_nsys_v1.json)
- [train semantic trace](../artifacts/multimodel_next/qwen35_hi_prompt8_train_trace_v1.json)
- [holdout semantic trace](../artifacts/multimodel_next/qwen35_hi_prompt8_holdout_trace_v1.json)
- [train CUDA API phase](../artifacts/multimodel_next/qwen35_hi_prompt8_train_cuda_api_phase_v1.json)
- [holdout CUDA API phase](../artifacts/multimodel_next/qwen35_hi_prompt8_holdout_cuda_api_phase_v1.json)
- [semantic calibration](../artifacts/multimodel_next/qwen35_hi_prompt8_semantic_calibration_v1.json)
- [prefill-scoped profile](../artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_calibration_v1.json)
- [replay summary and SHA lock](../artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_replay_summary_v1.json)
