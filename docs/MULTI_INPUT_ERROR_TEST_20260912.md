# 多输入场景误差测试（重新审计后 v3）

本次结果文件为 [native_error_matrix_v3.json](../artifacts/native_error_matrix_v3.json)，运行器为 [native_error_matrix.py](../tools/native_error_matrix.py)。10 个 case 每次独立启动本机 llama.cpp server，使用项目 37 内同一个 Qwen2.5-0.5B Q4_K_M GGUF，并把同一份本机硬件快照绑定到仿真场景。旧的 v2 数值不再作为结论。

## 场景与结果

| case | prompt token | output token | native prompt ms | native TPOT ms | sim TTFT ms | sim TPOT ms | sim E2E ms | TTFT 相对误差 | TPOT 相对误差 | E2E 相对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_output1 | 2 | 1 | 4.522 | — | 1.559 | — | 1.559 | -65.52% | — | -65.53% |
| short_output8 | 2 | 8 | 4.558 | 2.982 | 1.559 | 1.541 | 12.349 | -65.80% | -48.31% | -51.44% |
| medium_output8 | 13 | 8 | 6.160 | 2.629 | 1.787 | 1.551 | 12.645 | -71.00% | -40.99% | -48.52% |
| long_output8 | 193 | 8 | 26.985 | 2.401 | 4.986 | 1.556 | 15.880 | -81.52% | -35.17% | -63.74% |
| medium_output32 | 9 | 32 | 5.844 | 2.131 | 1.703 | 1.551 | 49.777 | -70.86% | -27.22% | -30.77% |
| batch32_ub16 | 9 | 8 | 6.780 | 2.905 | 1.703 | 1.559 | 12.612 | -74.89% | -46.35% | -53.49% |
| batch128_ub64 | 9 | 8 | 5.954 | 2.436 | 1.703 | 1.559 | 12.612 | -71.40% | -36.02% | -45.18% |
| cpu_only | 5 | 2 | 14.846 | 8.815 | 10.087 | 10.010 | 20.097 | -32.06% | +13.56% | -15.06% |
| gpu_tail12 | 7 | 4 | 10.308 | 5.162 | 2.900 | 2.783 | 11.250 | -71.86% | -46.08% | -56.38% |
| threads8 | 5 | 4 | 4.521 | 3.569 | 1.620 | 1.544 | 6.252 | -64.16% | -56.74% | -58.94% |

聚合结果（有定义的样本）：

- TTFT：中位数 **-70.93%**，p95 **-64.16%**；
- TPOT：中位数 **-40.99%**，p95 **-27.22%**（output=1 没有 TPOT）；
- E2E：中位数 **-52.46%**，p95 **-30.77%**。

10/10 个 case 的 `geometry_gate=pass`、`config_gate=pass`，token 数量 gate 为 `structural_only`，总体状态为 `valid_for_structural_analysis`。这表示场景结构、GGUF 几何和运行参数一致，不表示模型独立生成了相同 token 序列。

## 重新审计得到的确定性问题与修复

1. PCIe fallback 表记录的是每 lane 的有效 GB/s，而内部链路字段约定为十进制 Gb/s。Gen5×8 已修正为 `3.938 GB/s × 8 × 8 = 252.032 Gb/s`。此前 31.504 被当成 Gb/s，导致 144,643,072 B 的 output.weight 每次搬运被虚增约 32 ms。
2. llama.cpp 的 `-ngl` 计数包含 output layer：`-ngl 0` 为 CPU-only，`-ngl 12` 为 output 加最后 11 个 repeating layers，`-ngl -1` 为 output 加全部 24 层。planner 已统一这个语义，并保持 Qwen GGUF 的 `token_embd.weight` 在 host、`output.weight` 在 HBM 的 tied-weight 物理布局。
3. 数据流账本复核没有发现普遍重复计费：`model_weight_access` 是零服务时间的驻留标记，GEMM 已计一次权重读取；本地 KV 使用 kernel 内含访问；host logits D2H 只有一条真实传输；control-plane makespan 只作为一次起始偏移。
4. 矩阵报告已拆分 geometry、token count、configuration、timing、overall 五类状态，避免把“token gate 通过”误报为“时延通过”。

## 时间口径与限制

native 的 `prompt_eval_ms` 不包含仿真器的 host/control-plane 起始路径；native 的 `total_ms=prompt_eval_ms+predicted_ms` 不包含 HTTP/排队，而 simulator E2E 是 arrival 到 finish。因此 TTFT 和 E2E 只作边界不一致的诊断比较；TPOT 使用 native `predicted_ms/(predicted_n-1)` 与 simulator committed token 间隔，才具有相对可比性。

本机硬件快照为 AMD Ryzen 9 9950X3D、RTX 5080、Gen5×8 PCIe、约 134.9 GB 主存；GPU 峰值、HBM/DDR 带宽仍是解析 profile，不是用这次误差反推的校准值。矩阵是单请求（`parallel=1`），尚未证明 continuous batching、多请求竞争或逐 CUDA kernel 映射的准确性。

当前判定：数据流与放置建模可用于结构和趋势分析；绝对时延预测仍不可用。已完成一轮 CUPTI/Nsight Systems 采集，但当前二进制没有 NVTX operator 语义，因此只得到部分名称启发式证据。要继续缩小倍率误差，需要重新编译 llama.cpp 插入 NVTX range，再对 GPU kernel 吞吐、launch、同步和 cache 行为做真正的留出场景校准。

本轮已经完成的 CUPTI/Nsight Systems 阶段证据和留出结果见 [STAGE_CALIBRATION_20260912.md](STAGE_CALIBRATION_20260912.md)。
