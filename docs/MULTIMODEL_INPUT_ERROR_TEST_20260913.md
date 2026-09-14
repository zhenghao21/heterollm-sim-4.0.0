# 多模型多输入场景仿真—实测误差报告（2026-09-13）

本轮共 18 个 cell，结构有效 18 个。每个 cell 使用相同 ctx=512、batch/ubatch=64、threads=16、parallel=1、f16 KV、mmap、KQV/offload、seed=42；native 侧为本机 llama.cpp server，仿真器侧绑定同一 GGUF 几何。

## 结果概览

| 模型 | 输入 | 放置 | TTFT 有符号误差 | TPOT 有符号误差 | E2E 有符号误差 | 备注 |
|---|---|---:|---:|---:|---:|---|
| qwen25_0p5b | short | cpu_only | -25.6% | 8.18188410861078 | 2.4% |  |
| qwen25_0p5b | short | partial | -88.1% | -69.01898379215754 | -74.2% |  |
| qwen25_0p5b | short | full | -63.6% | -41.541657044292556 | -45.7% |  |
| qwen25_0p5b | medium | cpu_only | -59.2% | -7.310245315185497 | -20.2% |  |
| qwen25_0p5b | medium | partial | -82.5% | -52.91589762565167 | -61.4% |  |
| qwen25_0p5b | medium | full | -75.9% | -36.76603860583712 | -48.2% |  |
| qwen25_0p5b | long | cpu_only | -69.8% | 18.96312822726622 | -13.9% |  |
| qwen25_0p5b | long | partial | -86.8% | -34.095544446109336 | -56.6% |  |
| qwen25_0p5b | long | full | -80.8% | -36.578570998920775 | -49.6% |  |
| tinyllama_1p1b | short | cpu_only | -46.0% | -21.144670215789485 | -25.5% |  |
| tinyllama_1p1b | short | partial | -51.6% | -23.04520779531048 | -28.4% |  |
| tinyllama_1p1b | short | full | -54.2% | -39.36150856606074 | -41.8% |  |
| tinyllama_1p1b | medium | cpu_only | -61.7% | — | -61.7% | TPOT 不适用（native output=1） |
| tinyllama_1p1b | medium | partial | -67.3% | — | -67.3% | TPOT 不适用（native output=1） |
| tinyllama_1p1b | medium | full | -64.4% | — | -64.4% | TPOT 不适用（native output=1） |
| tinyllama_1p1b | long | cpu_only | -80.9% | -15.475423751495988 | -42.6% |  |
| tinyllama_1p1b | long | partial | -83.4% | -37.64587674938128 | -55.0% |  |
| tinyllama_1p1b | long | full | -69.4% | -36.04782508908339 | -44.4% |  |

## qwen25_0p5b 分层中位绝对误差

| 指标 | 全部输入/runtime | CPU-only | partial | full |
|---|---:|---:|---:|---:|
| ttft_ms | 75.9% | 59.2% | 86.8% | 75.9% |
| tpot_ms | 36.6% | 8.2% | 52.9% | 36.8% |
| e2e_ms | 48.2% | 13.9% | 61.4% | 48.2% |

## tinyllama_1p1b 分层中位绝对误差

| 指标 | 全部输入/runtime | CPU-only | partial | full |
|---|---:|---:|---:|---:|
| ttft_ms | 64.4% | 61.7% | 67.3% | 64.4% |
| tpot_ms | 29.5% | 18.3% | 30.3% | 37.7% |
| e2e_ms | 44.4% | 42.6% | 55.0% | 44.4% |

## 解读与限制

- 两个模型、三档输入、三种层放置均完成结构有效运行；TinyLlama 的 medium 输入提前 EOS，实际输出 1 token，因此该 cell 的 TPOT 被排除。
- 误差为 `(simulator-native)/native`。本轮 TTFT/E2E 仍是边界不同的诊断值；TPOT 仅在实际输出 token>1 时比较。
- 仿真器当前将 native 实际 token 数回填到场景构建，因此 token 数 gate 仍属于 bound/native-derived，不能作为独立 tokenizer 验证。
- 结果显示仿真器在多数 GPU 放置场景低估耗时（负误差）；Qwen2.5 CPU-only 的短输入 E2E 接近实测（+2.4%），但其余场景仍有 20–80% 量级偏差，下一步应继续做按模型×放置×阶段的校准。

原始数据：[native_multimodel_matrix_v1.json](../artifacts/multimodel_20260913/native_multimodel_matrix_v1.json)

## Qwen3.8-27B（Qwen 27B）追加测试

Qwen3.8-27B-IQ3_S-FFN-IQ4_XS 已从项目 33 复制到项目 37 后测试。GGUF 几何为 65 层、hidden=5120、heads=24、KV heads=4、vocab=248320，SHA256 为 `157479b0083662c7c9cfd8754cae24ab9cc1abbd4def91d49758512c90f1e406`。该模型是 Qwen3.5 hybrid transformer，仿真器已加入 linear-attention descriptors 与 IQ 量化格式支持；首轮使用 CPU-only 三档输入，并增加 partial1（`gpu_layers=1`）探针。

| 输入/放置 | TTFT 误差 | TPOT 误差 | E2E 误差 | native 输出 token |
|---|---:|---:|---:|---:|
| short / CPU-only | -45.4% | -36.9% | -41.5% | 2 |
| medium / CPU-only | -78.0% | -22.4% | -66.8% | 2 |
| long / CPU-only | -75.5% | -18.9% | -65.7% | 2 |
| short / partial1 | -44.7% | -32.9% | -39.4% | 2 |

Qwen 27B CPU-only 的 E2E 绝对误差中位数为 54.6%，partial1 短输入为 39.4%。RTX 5080 可用显存约 14.1 GiB，而该 GGUF 文件约 13.84 GiB；完整 `gpu_layers=-1` 还需额外 KV/cache 与运行时显存，因此没有把 full-GPU OOM 风险场景伪装成有效误差样本。

## 热力图

热力图按模型、输入和 runtime 分层；颜色越深表示绝对误差越大，灰色格表示该指标没有有效样本（例如提前 EOS 导致 TPOT 不适用）。

- [TTFT 绝对误差热力图](../artifacts/multimodel_20260913/heatmaps/ttft_ms_abs_heatmap.png)
- [TPOT 绝对误差热力图](../artifacts/multimodel_20260913/heatmaps/tpot_ms_abs_heatmap.png)
- [E2E 绝对误差热力图](../artifacts/multimodel_20260913/heatmaps/e2e_ms_abs_heatmap.png)
- [TTFT 有符号误差热力图](../artifacts/multimodel_20260913/heatmaps/ttft_ms_signed_heatmap.png)
- [TPOT 有符号误差热力图](../artifacts/multimodel_20260913/heatmaps/tpot_ms_signed_heatmap.png)
- [E2E 有符号误差热力图](../artifacts/multimodel_20260913/heatmaps/e2e_ms_signed_heatmap.png)

合并后的矩阵数据：[native_multimodel_matrix_v3.json](../artifacts/multimodel_20260913/native_multimodel_matrix_v3.json)。
