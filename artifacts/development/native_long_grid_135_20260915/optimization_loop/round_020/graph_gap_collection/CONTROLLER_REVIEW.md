# Graph-gap controller read-only review

日期：2026-09-16。仅运行 `test_controller.py` 的纯 Python mock；没有调用 prepare/run、Popen、NVML、smi、原生程序或 GPU。

结果：`5 passed`。

已核对：18 stage 计划为 12 native + 6 export；每个 arm/pair 绑定 direct/profile/export、相同 pair_id 与原生 argv；`wait_same` 在 bookkeeping interrupt 后继续等待自然退出；环境只显式设置 collector runtime keys；controller 源码在 finally 中执行 `-rgc` reset，不调用 kill/terminate，且 stage 释放前要求 child 已退出。

## Root 修复项：first-pair 人工检查门过早

当前 plan 的前 3 个 stage 都是 `pair 1 / control`：direct、profile、export。`run()` 在 `completed_stages == 3` 时调用 `summarize_pairs()`，但该函数会为两个 arm 都读取 pair 1 的 direct/profile raw audit；buffered pair 1 尚未运行。因此该 pause 不能形成跨 arm first-pair 检查，并且很可能在缺失 buffered raw audit 时直接抛错、进入 finally reset。

应将人工 first-pair 门移到 buffered pair 1 的 direct/profile/export 自然完成之后，即 6 个 completed stages，或将 first-pair summary 明确限制为已完成 arm 并停止声称 cross-arm 比较。前者符合当前 `summarize_pairs()` 的双 arm 结构。

未发现其他会导致错误真值、误杀或 clock finally 不恢复的问题。注意 environment() 会写 PATH，这是 collector 的既定 native/CUDA 解析策略，不是 test 的无执行行为。
