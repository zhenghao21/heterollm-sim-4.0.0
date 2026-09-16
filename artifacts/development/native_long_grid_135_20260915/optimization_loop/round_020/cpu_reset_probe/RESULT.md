# R20 reset A/B frozen result audit

日期：2026-09-16。分析仅读取 `reset_ab_series_0001` 的 run freeze、quality、六份 process JSON 与 receipts；未读取 LLM actual，未进行拟合或新测量。

## 完整性与身份

- six-process 顺序和 arm 与 run freeze 一致：`0/control, 3/treatment, 1/control, 4/treatment, 2/control, 5/treatment`。
- 六个 PID 均不同；receipts 均为成功返回，并且 receipt 的 reset policy、result SHA 与 raw process 文件一致。
- 两 arm 各 18 个 timed stage，总计 36；12 个 `(arm, V, pattern, top-k stage)` 跨进程组均有 3 个 median，quality 的 cross-process group 全部通过。

## 质量结果

冻结 `quality.json` 与重算结果都显示：**31/36** 个 stage 通过 P90/P10 阈值，不是 34/36。该差异需要在对外汇报中使用冻结 quality 和本审计的 31/36 数字。

失败的 5 个 stage 全部是 `V=262144`、`deterministic_random_permutation`、`original_dll_topk_apply`：

- control / memcpy：process 1，ratio 1.7364；process 2，ratio 1.7338。
- treatment / source candidate loop：process 3，ratio 1.7256；process 4，ratio 1.8349；process 5，ratio 1.7588。

所有 ratio 均超过固定门槛 1.5；因此总体仍为 `timing_usable=false`、`accepted_for_timing_evidence=false`。没有调整门槛、筛选通过 arm、输出 cost coefficient 或进行任何成本/LLM 拟合。

## 对 source reset 的证据边界

control 为 16/18 通过，treatment 为 15/18 通过。两 arm 都在相同高 V 随机排列条件下失败，treatment 还多一个失败 stage。因此这次 A/B **没有证据表明 source candidate-loop reset 降低波动**；最多只能说两种 reset 的失败位置重合于同一输入条件。源码语义差异不能据此被归因为 cache 或 write-buffer 因果。

可复现计算位于 [analyze_result.py](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_020\cpu_reset_probe\analyze_result.py)，结果 JSON 位于 [result_summary.json](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_020\cpu_reset_probe\result_summary.json)。
