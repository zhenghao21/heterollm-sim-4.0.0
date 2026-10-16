# 仿真器执行性能优化结果

日期：2026-10-15。结论：最终实现的 12 组完整推理结果与优化前逐项一致，实际运行耗时下降约 23%–26%。预测结果没有因本轮优化而变化。

## 完整推理前后对照

Small，输入 128 token、批量 1、输出 4 token；512 GiB HBM，CK=.625 ns，读写方向各 1600 GB/s，物理接口 6553.6 GB/s，REFab 开启。每组重复 3 次，以下取中位数。双方排除一次性 JIT 预热，保留正常运行时状态保存、请求处理与汇总开销。

| 参数组 / 放置 | 优化前耗时 s | 优化后耗时 s | 耗时减少 | 加速倍数 | CPU 时间减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference / A_hbm_only | 17.936 | 13.309 | 25.80% | 1.348x | 25.94% |
| reference / B_all_kv_hbf | 18.964 | 14.492 | 23.58% | 1.309x | 23.63% |
| ramulator2 / A_hbm_only | 18.190 | 13.938 | 23.38% | 1.305x | 23.21% |
| ramulator2 / B_all_kv_hbf | 19.385 | 14.855 | 23.37% | 1.305x | 23.26% |

A 为全 HBM；B 为 KV 放在 HBF。运行时八个原始大模型实验继续占用 CPU，因此这些数值是本机带背景负载的测量，不是机器无其他任务时的性能保证。CPU 时间和墙钟时间给出的收益接近。

## 等价与回归

- 12/12 组完整返回数据精确相等，仅去掉顶层 wall_seconds 字段；TTFT、TPOT、吞吐、命令计数、实际硬件配置和服务执行检查均参与比较。
- 562 项回归及 33 项子测试通过，涵盖命令时序、刷新、tFAW、时钟算术、控制器与调度、运行时恢复、物理调用方和异常路径。
- 数值核心旧/新对照覆盖 899 次成功续接、61 次相同拒绝和 24 对溢出边界；成功请求及全部 14 个数组无容差比较一致。
- 复核中发现跳过未变化 PC 的写回会保留旧实现原本断开的扩展引用，最终实现已恢复所有已出现 PC 的原有写回方式，并通过专门回归。

## 局部性能与原因

128 MiB 连续访存需处理 4,194,304 个 32 字节 burst。诊断中约 97% 的单次大请求耗时位于已编译的数值核心。只删除重复时序计算后，该核心的连续读写加速 1.15–1.19 倍，同 bank 两行交替加速约 1.10 倍，详见 [数值核心测量](../hbm_numeric_performance_20261015/README.md)。

完整流程还受状态整理与复制影响：缓存静态资源索引，只写回有变化的 bank/channel，保留 PC 原有历史重建方式，减少普通历史容器的通用深复制调用。最终历史复制单项中位数为 0.02704 → 0.02469 秒（约 1.10 倍）；该短测量波动较大，以完整推理表为主要收益依据。

实现说明见 [HBM 仿真执行速度优化](../../docs/hbm_simulation_performance.md)。未修改模型、时序参数、命令顺序、浮点精度、刷新、调度及异常恢复规则。

## 数据与复现

- serving-before.json / serving-after.json：各 12 次完整推理的预测数据及实际运行耗时。
- serving-summary.json：逐组汇总、精确一致检查和回归结果。
- baseline-small-serving.prof / baseline-small-serving.json：优化前热点诊断；该诊断含分析器开销，不能与本表耗时混用。
- history-copy.json：最终历史复制微测量。

优化前对照使用本轮保存的临时源码，不使用 Git HEAD 作为基线，避免混入此前已经完成的协议修复。旧实现没有成为产品回退分支。

```powershell
.venv/Scripts/python.exe scripts/benchmark_hbm_serving.py --output artifacts/hbm_execution_performance_20261015/serving-before.json --baseline-numeric C:/Users/A/AppData/Local/Temp/heterollm_numeric_performance_20261015/_hbm_numeric_baseline.py --baseline-packing tmp/_compiled_hbm_before_packing_optimization.py --baseline-history-copy tmp/hbm_control_copy_before.py
.venv/Scripts/python.exe scripts/benchmark_hbm_serving.py --output artifacts/hbm_execution_performance_20261015/serving-after.json
```

70B/235B 的八个原始进程已在优化前启动，继续使用旧实现。本轮没有把局部或 Small 测得的倍数当作大模型完整流程已测得的加速倍数；新启动的仿真进程会使用优化代码。
