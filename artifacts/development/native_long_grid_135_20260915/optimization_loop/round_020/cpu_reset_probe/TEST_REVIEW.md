# R20 CPU reset entry synthetic test review

日期：2026-09-16。仅运行 `test_entry.py` 的 Python 合成夹具；没有编译、DLL 加载、GPU、模型或测量。

结果：`7 passed`。

覆盖内容：

- six-process 交替计划 `[0,3,1,4,2,5]` 与 arm 映射；
- 每 case 仅一个 `original_dll_topk_apply` stage；
- 36 stage、36 case-process、12 arm/case cross-process group 和 2304 steady sample 分母；
- 错误 arm、candidate-stage 泄漏和 measurement schema 错配；
- 重复 PID、缺少 process、混合 arm；
- 频率漂移、P90/P10 抖动、跨 process median 离群、observer 开销；
- one-series guard 与全部 size 仅作为 quality diagnosis 的披露。

未发现需要 root 修复的 entry/protocol 测试失败。R20 仍未编译或测量。
