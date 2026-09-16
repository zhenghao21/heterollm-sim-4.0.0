# R19 CPU sampling probe R3 时序质量门审查

日期：2026-09-16。本次在 R2 CPU identity 与三项频率门之上补齐预先声明的原始 tick 质量标准。只修改未冻结 probe 源码、协议和审查回归；没有编译、加载 DLL、运行探针、使用 GPU 或输出时序值。R1 准备证据继续保留原样。

## 新增的时序接受门

任意门失败都会保留原始结果，但最终 `quality.json` 必须标记 `timing_usable=false`、`diagnostic_only=true` 和 `accepted_for_timing_evidence=false`；入口随后拒绝把该系列作为时序证据继续接受。

- **Stage 内稳定性：** 每个 stage 保留 64 个 steady 原始 QPC ticks。按线性插值计算 P10/P90，要求 `P90 / P10 <= 1.5`；P10 不为正直接仅诊断。
- **跨进程稳定性：** 每一个 `(V, pattern, stage)` 组必须收集三个独立进程的 steady median。相对该三者中位数的最大绝对偏差必须不超过 5%。
- **时钟观察开销：** 每个 stage 都将本进程 64 个空 QPC bracket 的中位数与该 stage steady median 比较，要求不超过 1%；观察值只保存，绝不扣除。
- **频率可用性：** 每个 stage 的 `os_max_mhz`、`os_reported_current_mhz` 和 `os_limit_mhz` 在前后都必须为正并保持相等。0 是不可用，不会被当成稳定。
- **固定分母：** 3 processes × 6 cases = 18 个 case-process 记录；3 × 6 × 2 = 36 个 stage 记录；6 × 2 = 12 个跨进程 case/stage 组；steady 样本总数为 2304。

仍不做任何核系数拟合、LLM 拟合、时序外推或空 bracket 扣除。

## 验证

- R3 只读来源/PE 复核通过：[STATIC_SOURCE_VALIDATION_R3.json](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_review\STATIC_SOURCE_VALIDATION_R3.json)。它确认 20 个冻结输入引用和所需非转发 sampler 导出；未加载 DLL。
- [test_cpu_sampling_probe_r2.py](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_review\test_cpu_sampling_probe_r2.py) 通过：`11 passed`。包含 max/current/limit 三类漂移、P90/P10 抖动、跨进程离群、observer 开销、0 分母、CPU identity/affinity 和错误状态标记的拒绝或仅诊断路径。
- `entry.py` 的 Python 语法检查通过。C++ 仍未编译；root 的独占编译步骤仍是下一步。
