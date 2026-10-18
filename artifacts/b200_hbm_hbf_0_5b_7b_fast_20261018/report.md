# B200 HBM/HBF 单请求扫描

报告刷新时间：**2026-10-18 14:20:57（北京时间 UTC+08:00）**。

已成功 **23/72** 个配置点（详细完成 0，预测完成 23）；预测安全检查拒绝 4 个；失败 0 个；容量不满足 0 个；尚未完成 45 个。

队列状态：**已暂停**；队列记录已完成 23/72；等待运行 45 个。队列更新时间：2026-10-18 14:18:17（北京时间）。

队列说明：修正配对预算及未来擦除上界后，优先重跑原拒绝点。

配对预算修复后优先重跑：**4 个点**。旧拒绝尝试保留在历史目录，只有 active points 的最新结果计入本报告；重跑不执行完整详细模式。

优先点位：qwen2_5-0_5b__HBF__P65536__D10240、qwen2_5-1_5b__HBF__P65536__D10240、qwen2_5-3b__HBF__P65536__D10240、qwen2_5-7b__HBF__P65536__D10240。

本轮修复重跑已处理 **0/4** 个；验收状态：**尚未全部处理**。该组全部接受后继续原队列；仍有拒绝则暂停处理。

| 参数 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-0.5B (0.5B)、Qwen2.5-1.5B (1.5B)、Qwen2.5-3B (3B)、Qwen2.5-7B (7B) |
| 存储策略 | HBM、HBF |
| 单请求输入 | 64K, 256K, 512K token |
| 单请求输出 | 1K, 5K, 10K token |
| K 单位 | 1024 token |
| GPU 预设 | nvidia-b200-sxm-gpu |
| GPU BF16 峰值（TFLOPS） | 2250.0 |
| 仿真模式 | fast |
| Decode 采样策略 | adaptive |
| 误差目标（非逐点实测保证） | 10.0% |
| Decode 真实采样上限（步） | 128 |
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
| HBF 独立 bank 数 | 256 |
| HBF 主机通道数 | 16 |
| HBF 读接口预算（GB/s） | 1536.0 |
| HBF 写接口预算（GB/s） | 1536.0 |
| HBF 每内部通道数据带宽（GB/s） | 96.0 |
| HBF 每主机通道在途总上限（研究假设） | 256 |
| HBF 每主机通道在途写上限（包含在总额内） | 64 |
| HBF 几何依据 | FlashLabAE 16 channels * 1 die/channel * 16 banks/die; target and LUN are singleton mapping levels |
| HBF 时序假设 | 4 us read, 75 us program and 2 ms erase are upstream exploratory assumptions, not OCP mandated latencies |
| HBF 队列假设 | 256 combined requests and 64 writes per host channel are local research assumptions; not a measured MOCS or an exact copy of upstream read/write queues |
| HBF 编程页（byte） | 4096 |
| HBF 读取粒度（byte） | 64 |
| HBF 读页延迟（ns） | 4000.0 |
| HBF 编程延迟（ns） | 75000.0 |
| HBF 擦除延迟（ns） | 2000000.0 |
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

**快速模式口径：Prefill 和 Decode 采样前缀为详细物理执行，尾段为预测。** TTFT 为详细结果，平均 TPOT 包含估算；资源占用、事务及物理状态只覆盖已执行前缀，不能当作完整逐 Token、逐事务结果。

10% 是误差目标；条件误差预算依赖模型假设，不能替代每个配置与完整详细模式的实测对照。未做完整详细对照的点保留原始 validation_status，不标为全请求误差验证通过。预测拒绝和容量拒绝是已处理终态，保留原因并继续后续点；代码执行失败需要处理。

![TTFT 和 TPOT 扫描结果](comparison.png)

每个面板用颜色区分 9 种输入/输出长度组合，用实线表示 HBM、虚线表示 HBF。横轴为模型参数量，纵轴为对数刻度；未完成、预测拒绝、容量拒绝和失败点留空。快速完成点的 TPOT 曲线为预测结果。

## 已成功的配置

| 模型 | 存储 | 输入/输出 | 模式 | TTFT（s） | TPOT（ms/token） | 真实/预测 Decode 步数 | 条件预算 | 完整对照状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2_5-0_5b | HBM | 64K/1K | fast（TPOT 估算） | 0.403659 | 1.2329 | 16/1007 | 0.390% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBM | 64K/5K | fast（TPOT 估算） | 0.40013 | 1.25646 | 16/5103 | 1.929% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBM | 64K/10K | fast（TPOT 估算） | 0.401046 | 1.2857 | 16/10223 | 3.803% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBM | 256K/1K | fast（TPOT 估算） | 5.37557 | 2.15715 | 16/1007 | 0.098% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBF | 64K/1K | fast（TPOT 估算） | 10.5285 | 122.96 | 80/943 | 1.687% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBF | 64K/5K | fast（TPOT 估算） | 10.5285 | 124.722 | 80/5039 | 3.392% | not_validated_against_full_detailed |
| qwen2_5-0_5b | HBF | 256K/1K | fast（TPOT 估算） | 45.8033 | 183.141 | 52/971 | 1.236% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBM | 64K/1K | fast（TPOT 估算） | 0.54326 | 2.55436 | 16/1007 | 0.392% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBM | 64K/5K | fast（TPOT 估算） | 0.540074 | 2.60265 | 16/5103 | 1.929% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBM | 64K/10K | fast（TPOT 估算） | 0.533552 | 2.63767 | 16/10223 | 3.803% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBM | 256K/1K | fast（TPOT 估算） | 5.91751 | 4.70356 | 16/1007 | 0.098% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBF | 64K/1K | fast（TPOT 估算） | 19.0112 | 188.075 | 28/995 | 1.993% | not_validated_against_full_detailed |
| qwen2_5-1_5b | HBF | 64K/5K | fast（TPOT 估算） | 19.0112 | 190.914 | 28/5091 | 3.522% | not_validated_against_full_detailed |
| qwen2_5-3b | HBM | 64K/1K | fast（TPOT 估算） | 0.963634 | 4.12324 | 16/1007 | 0.391% | not_validated_against_full_detailed |
| qwen2_5-3b | HBM | 64K/5K | fast（TPOT 估算） | 0.960526 | 4.20423 | 16/5103 | 1.929% | not_validated_against_full_detailed |
| qwen2_5-3b | HBM | 64K/10K | fast（TPOT 估算） | 0.95017 | 4.26106 | 16/10223 | 3.803% | not_validated_against_full_detailed |
| qwen2_5-3b | HBF | 64K/1K | fast（TPOT 估算） | 29.8801 | 295.043 | 28/995 | 2.592% | not_validated_against_full_detailed |
| qwen2_5-3b | HBF | 64K/5K | fast（TPOT 估算） | 29.8801 | 299.498 | 28/5091 | 4.109% | not_validated_against_full_detailed |
| qwen2_5-7b | HBM | 64K/1K | fast（TPOT 估算） | 1.6025 | 7.55381 | 16/1007 | 0.390% | not_validated_against_full_detailed |
| qwen2_5-7b | HBM | 64K/5K | fast（TPOT 估算） | 1.6025 | 7.68395 | 16/5103 | 1.927% | not_validated_against_full_detailed |
| qwen2_5-7b | HBM | 64K/10K | fast（TPOT 估算） | 1.6025 | 7.82967 | 16/10223 | 3.802% | not_validated_against_full_detailed |
| qwen2_5-7b | HBF | 64K/1K | fast（TPOT 估算） | 40.7268 | 485.124 | 16/1007 | 1.926% | not_validated_against_full_detailed |
| qwen2_5-7b | HBF | 64K/5K | fast（TPOT 估算） | 40.7268 | 492.446 | 16/5103 | 3.413% | not_validated_against_full_detailed |

## 未成功的配置

| 模型 | 存储 | 输入/输出 | 状态 | 原因 |
| --- | --- | --- | --- | --- |
| qwen2_5-0_5b | HBF | 64K/10K | forecast_rejected | prefix_calibration_exceeds_error_budget |
| qwen2_5-1_5b | HBF | 64K/10K | forecast_rejected | prefix_calibration_exceeds_error_budget |
| qwen2_5-3b | HBF | 64K/10K | forecast_rejected | prefix_calibration_exceeds_error_budget |
| qwen2_5-7b | HBF | 64K/10K | forecast_rejected | prefix_calibration_exceeds_error_budget |

文件：[summary.csv](summary.csv) 保留全部终态的主指标、精度口径、采样记录、单点/配对预算、参考点及拒绝原因（无指标时留空）；[results.json](results.json) 保留全部结果、详细数据和参数。
