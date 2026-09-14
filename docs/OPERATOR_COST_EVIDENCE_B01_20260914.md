# B-01 算子成本证据审计（2026-09-14）

本轮只读取已有 semantic train/holdout trace，未启动 native，也未读取目标模型端到端时延来拟合任何参数。审计对象是 `stage × phase × shape × dtype × layout × kernel_family` 的联合覆盖，重点包括 CPU MMQ 的 M=1/T、GPU MMQ projection、lm-head 和 linear-attention auxiliary。

## 结果

审计输出保存在 [operator_cost_evidence_b01_v1.json](../artifacts/development/operator_cost_evidence_b01_v1.json)，它针对 direct-equal 的一对 train/holdout trace；联合键统计保存在 [operator_cost_coverage_b01_v1.json](../artifacts/development/operator_cost_coverage_b01_v1.json)，覆盖 4 对既有 train/holdout trace。后者的匹配事件共形成 196 个联合键，且每一对的 holdout 键都能在对应 train 中找到。这个结果只能说明已匹配事件的 shape、dtype 和 kernel 信息可被提取，不能说明完整 trace 已覆盖。

完整 `build_semantic_calibration` 门禁结果为 `blocked`：train 和 holdout 各有 2,590 个 unknown/uncovered 事件，unknown 时间分别约 6.50 ms 和 6.79 ms；各有 3,320 条 memcpy 被排除。unknown owner 没有稳定的 operator 语义时，不能把其时间归入 MMQ、lm-head 或 linear-attention，也不能据此生成可启用的吞吐曲面。该审计因此标记为“证据不足”，没有修改任何成本参数。

## 发现的代码问题

`tools/build_semantic_calibration.py` 的 NVTX wall 路径调用了不存在的 `_operator_id`，含 operator scope 的 trace 会直接抛出 `NameError`。已补上只解析稳定 label 前缀的函数，并用回归测试锁定：shape/type 字段不会进入 operator id，空 label 返回 `None`。测试文件为 [test_semantic_calibration.py](../tests/test_semantic_calibration.py)，专项结果为 `11 passed`。

## 结论与后续

现有 `estimate_gpu_gemm`/`MMQWork` 可以保留其物理 work accounting；由于 unknown owner 覆盖未闭合，不能把分析 roofline 替换成按 kernel/shape 的实测成本，也不能启用阶段校准。下一步是 B-02：在原始 NVTX/CUPTI/NSYS 与 extractor sidecar 中闭合 unknown owner，要求 train/holdout 的 unknown=0、联合字段完整、未见 shape 明确 fail-closed。只有满足这些条件，才允许用独立 MMQ 微基准建立参数；不得使用目标 LLM 的 TTFT、TPOT 或 E2E 做拟合。
