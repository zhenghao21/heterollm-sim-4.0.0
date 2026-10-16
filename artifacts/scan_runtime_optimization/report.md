# B200 扫描仿真速度优化

日期：2026-10-16。正式的 0.5B～7B、72 点扫描已暂停，完成点数为 **0/72**；本轮没有自动恢复扫描。

本轮优化已完成。完整请求基准中，HBM 提速 **13.26 倍**，HBF 提速 **22.39 倍**。以下耗时是运行仿真程序所需的真实时间，和模型推理的预测 TTFT、TPOT 是不同指标。

## 耗时对比

使用 Qwen2.5-0.5B、单张 B200 计算预设、单请求、BF16，保留全部模型层和物理访存执行。HBM/HBF 的硬件参数、CPU、HOST DRAM 和数据放置沿用正式扫描配置。表中的输入、输出单位是 token。

| 部署与请求长度 | 优化前耗时 | 优化后耗时 | 加速倍数 |
| --- | ---: | ---: | ---: |
| 全 HBM，输入 32、输出 3 | 215.33 秒 | 16.24 秒 | **13.26×** |
| 全 HBF，输入 32、输出 3 | 119.18 秒 | 5.32 秒 | **22.39×** |
| 全 HBF，输入 2048、输出 3 | 110.89 秒¹ | 8.71 秒 | **12.74×** |
| 全 HBM，输入 2048、输出 3 | 未测同配置旧版基线 | 28.57 秒 | 不计算 |

¹ 2048-token HBF 对比起点已经包含部分常驻读取优化，是较快的中间版本；12.74 倍表示在该中间版本上继续获得的提升。

HBM 两项均采用最终修复后的重测结果。之前一次 HBM 32-token 测得 12.48 秒，最终报告采用重测的 16.24 秒，不取更快的一次。2048-token HBM 结果用于确认正式扫描使用的 Prefill 分块能完整执行。

数据文件：[HBM 优化前](before_HBM.json)、[HBM 最终](final_HBM_32.json)、[HBF 优化前](before_HBF.json)、[HBF 最终](final_HBF_32.json)、[HBF 2048 中间基线](benchmark_2048_HBF.json)、[HBF 2048 最终](final_HBF_2048.json)、[HBM 2048 验证](final_HBM_2048.json)。

这些是完整执行的短请求性能基准，不属于正式的 64K/256K/512K 输入、1K/5K/10K 输出扫描点。**尚未测量全部模型和长输出上的加速倍数，不能据此保证每个正式配置都达到同样倍数。**

## 改动内容

- HBM 控制器状态保留在数值数组中，减少每个物理任务反复整理、复制大量 bank 状态。只有公开读取、完整快照等确有需要时才生成字典状态。
- 独立 HBM 通道最多使用 8 个 CPU 线程并行计算；同一通道内部仍按原顺序执行每条命令，保留两个 pseudo-channel 之间的共享约束、刷新和时钟对齐。
- 严格 OCP HBF 读写页流使用编译执行，保留逐页接纳、阵列操作、缓存占用、链路和编程时序。HOST 暂存只记录事务中实际改变的页，回收时使用页索引。
- 移除正式物理执行前的重复预演；合并满足顺序和依赖约束的同类 DRAM 任务，缓存不变的物理配置和资源名称。
- 仅在独立扫描/基准进程执行场景时提高 Python 垃圾回收阈值，结束或异常时恢复原值。
- 并行提交、执行、等待报错时，先阻止新任务进入并等已运行任务结束，再允许状态回滚；极端地址交错配置在进入数值内核前检查整数范围。

没有缩小模型、抽样 token、跳过物理命令或改用带宽公式估算。HBF 方案仍不使用 HBM；激活、工作区和 OCP 对齐所需的 HOST 暂存语义保留。

## 结果等价与测试

三组有前后结果的完整请求，其 **TTFT、TPOT、请求指标、物理执行范围和存储流量一致**。HBM 32-token 请求的资源忙碌时间也完全一致；新增的 `dram_traffic.scheduler_batches` 是实际调度合并次数，由 0 变为 654，不是新增访存量。

HBF 的链路忙碌时间因浮点累加顺序发生末位差异：32-token 对比最大约 9.31×10⁻¹⁰ ns，2048-token 对比最大约 5.96×10⁻⁸ ns。HBM 并行定向测试中，全部 14 个状态数组逐位一致，仅两个跨通道汇总忙时计数允许浮点末位差异。

相关检查已通过，以下测试组有重叠，不累加成总数：

- HBM 控制器、事务、并行和异常回滚：174 项测试、8 项子测试通过。
- HBF OCP、HOST 暂存和流量记账联合检查：68 项通过；资源前缀缓存检查通过。
- 服务执行、swap、RoPE 与 final norm 数据流：43 项通过。
- 物理配置解析相关回归：120 项测试、68 项子测试通过。
- 扫描进程垃圾回收设置在正常/异常退出后的恢复检查通过。

## 测量条件和复现

前后基准均由项目 `.venv/Scripts/python.exe` 启动；最初两份 before 文件没有记录解释器字段，已从本次会话的原始启动命令确认。后续结果文件直接记录解释器路径。正式比较使用未开启性能分析器的墙钟时间；初始 HBM/HBF 基准同时启动，后续测量在迭代开发期间分别执行，因此属于本机现场单次对比，耗时存在波动。

最终测量使用已有 Numba 编译缓存；首次运行或修改数值内核后，会增加一次编译开销（本机观察约 8 秒）。HBM 通道并行会提高 CPU 占用，其 CPU 累计时间可能大于墙钟时间。未进行整轮 72 点耗时外推。

```powershell
.venv/Scripts/python.exe scripts/benchmark_scan_runtime.py --architecture HBM --prompt 32 --output 3 --result artifacts/scan_runtime_optimization/recheck_HBM_32.json
.venv/Scripts/python.exe scripts/benchmark_scan_runtime.py --architecture HBF --prompt 32 --output 3 --result artifacts/scan_runtime_optimization/recheck_HBF_32.json
.venv/Scripts/python.exe scripts/benchmark_scan_runtime.py --architecture HBF --prompt 2048 --output 3 --result artifacts/scan_runtime_optimization/recheck_HBF_2048.json
```

[正式扫描状态与实验参数](../b200_hbm_hbf_0_5b_7b_20261016/report.md)。扫描报告已支持运行期间每 60 秒刷新及 Prefill/Decode 阶段进度；暂停期间不会产生新的测量结果。
