# B200 HBM/HBF 单请求扫描

报告刷新时间：**2026-10-16 22:57（北京时间 UTC+08:00）**。

已完成 **0/72** 个配置点；失败 0 个；容量不满足 0 个；尚未完成 72 个。

队列状态：**已暂停**；队列记录已完成 0/72；等待运行 72 个。队列更新时间：2026-10-16 17:13:40（北京时间）。

队列说明：按要求暂停，未自动恢复扫描。

Decode 优化状态：按要求完成最后三种尝试并结束优化，保留当前最佳实现；最终 0.5B/64K 对照 HBM 约 6.40×、HBF 约 5.63×，十倍目标未达到。完整结果与基线一致，正式扫描继续暂停。[最终报告和测量范围](../decode_runtime_optimization/report.md)。

| 参数 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-0.5B (0.5B)、Qwen2.5-1.5B (1.5B)、Qwen2.5-3B (3B)、Qwen2.5-7B (7B) |
| 存储策略 | HBM、HBF |
| 单请求输入 | 64K, 256K, 512K token |
| 单请求输出 | 1K, 5K, 10K token |
| K 单位 | 1024 token |
| GPU 预设 | nvidia-b200-sxm-gpu |
| GPU BF16 峰值（TFLOPS） | 2250.0 |
| 请求数 | 1 |
| Batch size | 1 |
| 权重精度 | bf16 |
| KV 精度 | bf16 |
| 激活精度 | bf16 |
| Prefill 分块（token） | 2048 |
| 权重预驻留 | 是 |
| HBM 物理预设 | hbm3e_reference |
| HBM 容量（byte） | 180000000000 |
| HBM 接口带宽（GB/s） | 8000.0 |
| HBF 物理预设 | hbf_ocp_0_7 |
| HOST 暂存容量（byte） | 67108864 |
| HBF 容量（byte） | 549755813888 |
| HBF 阵列单元数 | 64 |
| HBF 读取带宽（GB/s） | 1600.0 |
| HBF 写入带宽（GB/s） | 16.0 |
| HBF 编程页（byte） | 4096 |
| HBF 读取粒度（byte） | 64 |
| HBF 读页延迟（ns） | 4000.0 |
| HBF 编程延迟（ns） | 75000.0 |
| HBF 擦除延迟（ns） | 3000000.0 |
| CPU 核数 | 72 |
| CPU 频率（GHz） | 3.0 |
| HOST DRAM 容量（byte） | 32000000000 |
| HOST DRAM 带宽（GB/s） | 44.8 |
| PCIe 代际 | 5.0 |
| PCIe 通道数 | 16 |

完整硬件参数、点位清单和指标定义见 [manifest.json](manifest.json)。
HBF 的器件延迟采用仿真预设值，不代表 OCP 规定的器件性能。

HBM3E 使用标准参考模型；当前 24Gb × 8H 组织未套用只支持 32Gb × 16H 的额外时序包。
独立 RoPE 每次生成并写入按 Q/K 广播展开的 sin/cos 表，两种部署均计入这项开销。
超过模型原生上下文的长度仅用于性能外推，不验证输出质量。

数据单位：原始 TTFT/TPOT 为 ns；图中 TTFT 为秒，TPOT 为毫秒/token。

当前没有已完成的测量点，尚未生成曲线图。

文件：[summary.csv](summary.csv) 仅含已完成点的主指标和其他标量；[results.json](results.json) 保留全部结果、详细数据和参数。
