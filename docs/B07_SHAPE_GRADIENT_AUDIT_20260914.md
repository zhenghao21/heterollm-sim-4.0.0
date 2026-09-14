# B-07 shape 梯度与联合覆盖审计（2026-09-14）

## 目的

使用已采集的 semantic direct Qwen2.5-0.5B trace 检查 M/T shape 梯度能否形成可留出的 operator 成本面。没有重新采集 27B，也没有使用目标模型端到端时延拟合成本。

## 输入与修复

- train 合并：`b06_qwen25_train_v1`（prompt 8/output 8）+ `b07_qwen25_m16_t4_v1`（prompt 16/output 4）+ `b07_qwen25_m64_t16_v1`（prompt 64/output 16）。
- holdout：`b07_qwen25_m32_t8_v1`（prompt 32/output 8）。
- 原始 b07 merged 文件丢弃了 `nvtx_events`，导致 phase 全部缺失；本轮生成 v2，保留 kernel 与 phase scopes，并把源 trace 时间线隔离 1 秒，避免跨文件包含关系。

## 结果

- train kernel=14714，holdout kernel=4384；两侧 semantic_status 均全 matched，unknown owner=0。
- 基于 `(stage, phase, shape, dtype, layout, kernel_family)`：train keys=84，holdout keys=43，交集=19。
- 按 holdout kernel event 加权覆盖：3437/4384 = 78.40%。
- holdout 仍有 26 个联合 key 不在 train；其中 attention_qkv/attention_output/FFN 的 32-token prefill kernel 路径占主要缺口。
- layout 是从 kernel family 识别的 `ds_layout/matrix_q/vector_q/unspecified`，不是 trace 显式字段；应继续标记为 derived，不可当作独立硬件 layout 观测。

| stage | holdout event coverage |
|---|---:|
| attention_output | 672/936 (71.79%) |
| attention_qkv | 840/1104 (76.09%) |
| ffn | 346/530 (65.28%) |
| kv | 192/192 (100.00%) |
| linear_attention_aux | 0/23 (0.00%) |
| lm_head | 24/24 (100.00%) |
| normalization | 345/392 (88.01%) |
| quantize | 1018/1183 (86.05%) |

## calibration 留出

| stage | event-rate error | operator-wall error |
|---|---:|---:|
| attention_qkv | -6.75443766705819% | -17.883786060399242% |
| ffn | 0.39437707587226056% | 9.588215696004708% |
| kv | 4.7918036559647454% | -2.470057814740859% |
| lm_head | 0.4627470374861646% | 35.92728829971844% |
| attention_output | 0.587197827235606% | -16.39127710937919% |
| normalization | 0.020563982934907736% | 0.5475188333720492% |
| linear_attention_aux | -0.8129737053817165% | 30.657230581275712% |

## 判定

phase/shape 修复后 calibration coverage=covered，event-rate 阶段误差大多在 ±5% 内，但联合 key 仅 21/91，按 holdout kernel 加权覆盖 78.40%，且 linear_attention_aux 无 train 对应 key。由于任务书要求联合 shape×dtype×layout×kernel 证据达到支持域覆盖，B-07 不能启用全局 operator profile；未覆盖组合必须走分析回退或显式不支持。下一步应增加有独立语义 layout/correlation 的 M/T 梯度或采用仅对 exact key 生效的校准，并在 simulator-only 验证集上做三路消融。

## 产物

- `artifacts/development/b07_qwen25_train_merged_v2.trace.json`
- `artifacts/development/b07_qwen25_holdout_v2.trace.json`
- `artifacts/development/b07_qwen25_operator_calibration_candidate_v2.json`
- `artifacts/development/b07_qwen25_operator_coverage_v2.json`

结论：B-07 完成 shape 梯度审计，覆盖率不足，候选成本 profile 保持阻断。