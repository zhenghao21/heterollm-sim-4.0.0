# Native / R0 / 当前仿真对比

这份表用于明早演示时说明仿真的可信边界。数据来自同一组冻结的 Qwen2.5 Native 负载场景：Native 每个场景取 3 次运行的中位数，R0 和当前结果取仿真请求指标的中位数。误差定义为：

```text
误差百分比 = (当前仿真 - Native) / Native × 100%
```

负值表示当前仿真低估 Native；正值表示高估。R0 只用于显示本次修复前后的变化，不把 R0 当作真实硬件基准。

| 场景 | 指标 | R0 仿真 | 当前仿真 | Native 中位数 | 当前相对 Native | 当前相对 R0 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 / prompt 512 / output 256 / 并发 1 | TTFT | 21.204 ms | 12.261 ms | 53.772 ms | -77.20% | -42.18% |
|  | TPOT | 2.078 ms | 2.916 ms | 3.188 ms | -8.54% | +40.33% |
|  | E2E | 551.070 ms | 755.800 ms | 866.736 ms | -12.80% | +37.15% |
| Qwen2.5 / prompt 1536 / output 128 / 并发 1 | TTFT | 68.509 ms | 35.585 ms | 168.359 ms | -78.86% | -48.06% |
|  | TPOT | 2.134 ms | 2.963 ms | 3.208 ms | -7.65% | +38.86% |
|  | E2E | 339.506 ms | 411.891 ms | 576.084 ms | -28.50% | +21.32% |
| Qwen2.5 / prompt 1536 / output 256 / 并发 4 | TTFT | 143.796 ms | 141.086 ms | 270.928 ms | -47.92% | -1.88% |
|  | TPOT | 3.389 ms | 3.354 ms | 5.670 ms | -40.85% | -1.03% |
|  | E2E | 1009.455 ms | 996.300 ms | 1726.808 ms | -42.30% | -1.30% |

## 讲解结论

- 当前结果相对 R0 的结构变化已经稳定在三个代表场景上：长 prompt 单并发的 TTFT 进一步降低；并发 4 的结果与 R0 接近，说明本次修改没有破坏原有并发趋势。
- 当前模型仍系统性低估 Native 的 TTFT，约低估 48%～79%。这说明首 token 前的真实调度、采样、服务端排队或 runtime 固定开销尚未完整建模。
- 单并发 TPOT 的误差约 8%，可以用来展示生成阶段的趋势；并发 4 的 TPOT 误差约 41%，说明并发竞争还不能当作精确绝对值。
- 当前 E2E 误差约 13%～42%。它适合比较不同 KV 驻留策略的相对变化，不能宣称已经达到 Native 级绝对预测精度。

## 证据位置

- R0：`artifacts/development/native_long_grid_135_20260915/optimization_loop/round_000/on/errors.0001.json`
- 当前重跑：`.tmp/native_compare_20260928_v2/errors.0001.json`
- Native 数据：`artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json`
- 离线包对应证据：`demo/evidence/r0_errors.0001.json`、`demo/evidence/current_errors.0001.json`、`demo/evidence/stable_native_dataset.json`

当前重跑只对这 3 个代表场景有完整仿真预测，因此表格是演示用样本对比，不是 131 个网格单元的全量精度结论。完整冻结结果中的严格门限判定仍为 `accuracy_failed`，原因正是 TTFT 和并发场景的误差超过 25%。
