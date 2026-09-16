# R18 采样集成独立审查

审查日期：2026-09-16  
范围：采样核心语义、静态预测接线、冻结期契约验证器与无原生/无仿真的回归检查。未修改核心代码；本目录以外的改动均未触碰。

## 结论

当前 R18 接线通过本次审查，没有发现会阻止继续集成的功能错误。

- `min_keep=0` 已仅对 `llama_cpp_cpu_chain` 放宽到非负整数；`top_k` 仍要求正整数且保持 `min_keep <= top_k`。见 `src/heterollm_sim/config.py:269-312`。
- `top_k=1` 不会走旧的 greedy reduction 分支。采样器首先物化整行 F32 logits 为 12-byte `llama_token_data` 记录，然后仅为 K=1 建立 `V-1` 的 tail-scan 表示；每个行的服务量以 `serial_sum` 聚合。见 `src/heterollm_sim/planner.py:19094-19110` 与 `19172-19223`。源端的 `std::partial_sort(..., npartial=1, ...)` 和 K=1 截断见 `source/llama.cpp-semantic/src/llama-sampler.cpp:193-201`、`321-337`。
- 静态预测器将已验证单元的 `typed_policy` 构造成 `SamplingPolicy`，传给 `build_matching_scenario`，并在最终控制面重规划之前保留在场景对象中。见 `tools/predict_stable_native_dataset.py:1563-1593` 与 `tools/native_llama_compare.py:1105-1113,1316-1317`。
- 本次独立的控制面静态重规划检查确认 K=1 / `min_keep=0` 的完整策略在 `replan_final_static_scenario` 后不变，普通校验通过，且没有调用 `run_scenario`。结果见 `control_plane_replan_static_check.json`。
- 验证器对选中单元、原始块、原始文件 SHA/大小、运行时四模块、构建收据、历史头文件和每个 warmup/run 请求的 resolved settings 进行重验；真实 131 单元契约独立重跑成功。结果见 `independent_static_verification.json`。

## 静态验证证据

- 使用 Python 3.12（项目要求 `==3.12.*`）重跑 `verify_sampling_contract`：131/131 单元、131/131 原始块、warmup 596 份与 runs 894 份 resolved settings、175 个证据引用全部通过。
- 仅使用夹具的回归测试通过：`71 passed, 126 deselected`。覆盖契约的 SHA、缺失/重复单元、原始证据、运行时模块、来源和历史头文件拒绝路径，以及 K=1、12-byte 物化、串行行聚合和静态接线。
- 未启动 GPU、原生推理或仿真；未读取原生延迟值，也未拟合采样成本系数。

## 限制仍然明确

采样链仍是部分建模。logit bias / 额外抑制、过滤尾部、RNG、sampler accept 和历史 ring commit 没有被宣称为完整或计时完成。验证器的限制文本与预测器的 `cpu_sampling_chain` unsupported dimension 一致，见 `tools/native_sampling_contract.py:25-34` 和 `tools/predict_stable_native_dataset.py:1137-1141`。

`STATIC_KEYS` 不含原生延迟、时间统计或成本系数字段；采样契约输出也明确标记 `latency_fields_projected_or_used=false`。见 `tools/predict_stable_native_dataset.py:43-52` 与 `independent_static_verification.json`。

## 建议补充的非阻塞回归

1. **P2：将控制面保留检查固化为仓库测试。** 现有 `test_sampling_policy_static_binding_reaches_builder_without_native_answers` 在 `tests/test_predict_stable_native_dataset.py:1751-1768` 替换了 `replan_final_static_scenario`，所以它验证进入 builder 和 metadata，却不验证真实重规划函数不会在未来丢失 `sampling_policy`。本次独立静态检查已证明当前实现正确；建议添加一个不调用 `run_scenario` 的测试，直接断言真实 `replan_final_static_scenario` 前后策略完全相等。

2. **P2：补充生产历史头文件 SHA 的工件级拒绝测试。** 当前 `tests/test_native_sampling_contract.py:53-73,258-272` 通过临时夹具和 monkeypatch 的哈希集合验证来源拒绝，并能拒绝错误 header snapshot；它没有直接用 R18 生产固定 header/SHA 构造“旧 header 与生产哈希不一致”的回归。真实契约已在本次重验中通过，故这不是当前阻塞项；增加该测试会防止未来测试退化为只修改夹具源码的路径。
