# 37_LLMsim 全链路审计（2026-09-12）

本次审计由三个独立子代理分别检查数据流与调度、原生 llama.cpp/Nsight 对照、以及测试/文档/新增脚本；主进程复核并修复了确定性缺陷。审计结论按结果风险分级，不把“测试通过”解释成 native kernel 已完成校准。

## 已修复

- `RuntimeExecutionPlan.to_schedule()` 现在保留显式 `resource_capacities`，adapter 路径不会回退到单 lane。
- serving 的 `n_batch/n_ubatch` 不再被模型 context 错误裁剪；单请求 context admission 与批次 token budget 分离。
- `split_mode` 只接受当前 llama.cpp 支持的 `none/layer/row`。
- `op_offload` 在传入完整 config 时纳入互斥参数检查。
- K/V dtype 不一致时 fail-closed，避免把 `ctk=q8_0, ctv=f16` 静默压成一个 dtype。
- native launch 校准默认为 evidence-only；只有显式 `--apply-launch-calibration` 才会注入，避免 CUDA graph API 均值按每个算子重复计时。
- `task_stage()` 处理 qkv/ffn/kv/output 的优先级和通用 `kernel_launch` 回退；非 kernel 事件从 matcher 中拆出为 `unsupported`。
- Nsight graph-node creation 记录保存为 metadata，不再当作执行区间参与 task mapping。
- profile builder 记录 D2H `ns/byte`（存在逐事件 trace 时），并保存模型 SHA、硬件 fingerprint、runtime fingerprint。
- 校准数值加载增加有限、非负和顶层 schema 校验。
- join 工具从 profile command 读取 ctx/parallel/batch/ubatch/threads/gpu_layers，并输出真正的 scenario hash 和 GGUF SHA gate。

## 当前仍是证据缺口

- 本机 profile 中约 73.8% kernel 时间仍为通用 `mul_mat*` 的 `unknown`，没有 layer/operator 语义。
- Nsight WDDM 轨迹因当前 shell 未提升到管理员完整性级别而被禁用；CUDA activity 仍可采集。
- `--parallel>1` 的 `ctx-size` 是全局还是每 slot 口径尚未用同一 llama.cpp build 完成 admission 验证。
- native compare 仍以单请求为主，continuous batching 的多请求竞争尚未形成矩阵化实测。
- `split_mode`、`main_gpu`、`tensor_split`、NUMA/CPU affinity、mmap/mlock 等部分 runtime 字段仍主要作为身份 metadata，未全部转化为资源行为。
- canonical IR 已保存 llama.cpp runtime profile，但旧 schema/恢复路径仍需补充 round-trip 回归。

## 原生证据

- [native_profile_audit.json](../artifacts/native_profile_audit.json)
- [native_profile_audit.trace.json](../artifacts/native_profile_audit.trace.json)
- [native_calibration_audit.json](../artifacts/native_calibration_audit.json)
- [native_task_mapping.json](../artifacts/native_task_mapping.json)

最新 profile 共提取 4,134 个活动事件：2,178 kernel、1,856 runtime、100 memcpy；graph node creation 单独保存为 metadata。当前 task mapping 的 `matched` 只接受显式 task id，阶段唯一候选保留为 candidate/ambiguous，不把低置信度候选当作已校准任务。

## 验证

本轮专项回归：`65 passed`；本轮修复后的全量验证：`769 passed, 1 skipped`。Windows 测试中偶发的 pyarrow/pandas/OR-Tools `0xc0000139` 堆栈仍属于环境依赖噪声，pytest 最终完成并报告通过。
