# R22 r2 长图诊断结果

冻结 SHA：`43fd1af86de220b2748dfa61689237e3d1dd5947f0fdbcc63152e665951e6e5e`。控制器完成 18/18 stages，GPU 时钟 reset 返回 0；所有 18 个 stage receipt 和 validation 都是已执行/结构验证。

两种配置都**未通过** measurement-cost quality gate：`accepted 0/2`。这不是“数据无用”：device-side kernel 分布稳定、数值/时钟/CUPTI 链完整；失败的是完整 host whole-graph 合同，唯一失败原因是每个 pair 的 direct host `P90/P10` 超过冻结上限 1.5。

## 分配置结果

### scale_f32_e262144_g64

状态：`diagnostic_quality_rejected`；跨 profile process kernel-median 最大相对偏差：`0.0274%`。

| Pair | Kernel P90/P10 | Direct host P90/P10 | Profile host P90/P10 | Profile/direct median 扰动 | Clock | Trace / CPU fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.0097 | 2.2671 | 1.5321 | 4.488% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |
| 2 | 1.0118 | 2.7393 | 1.9277 | 4.196% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |
| 3 | 1.0098 | 2.5894 | 1.9750 | 10.034% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |

每个 pair 的数值、时钟、trace chain、source path 和 trace warning 门均通过。仅 direct host 离散度使该配置拒收。

### scale_f32_e262144_g256

状态：`diagnostic_quality_rejected`；跨 profile process kernel-median 最大相对偏差：`0.0803%`。

| Pair | Kernel P90/P10 | Direct host P90/P10 | Profile host P90/P10 | Profile/direct median 扰动 | Clock | Trace / CPU fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.0028 | 2.1568 | 1.7417 | 7.405% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |
| 2 | 1.0020 | 2.0930 | 1.7021 | 12.096% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |
| 3 | 1.0026 | 2.0470 | 1.7248 | 19.609% | pass | device 0 / stream 13；CUDA pin；CPU buffer 0 |

每个 pair 的数值、时钟、trace chain、source path 和 trace warning 门均通过。仅 direct host 离散度使该配置拒收。

## CUPTI 与 CPU 回退语义

- G64：每个 formal call 观察到 64 个 `scale_f32` kernel、64 个 `cudaLaunchKernelExC_v11060` launch API 和 1 个 `cudaStreamSynchronize_v3020` 图末同步；每个 profile 36-call trace 共映射 2,304 kernels。
- G256：对应为 256 / 256 / 1；每个 profile 36-call trace 共映射 9,216 kernels。
- 映射的 launch/sync 事件中未出现 CUDA Graph API；这条 synthetic SCALE 路径的观察结果是逐 node `cudaLaunchKernelExC` 提交。不能据此推断其他 LLM 图是否 replay CUDA Graph。
- 所有 profile trace 的 kernel 均在 device 0、stream 13；all-36 chain complete、source path match，且无 trace warnings。
- Raw setup 同时证明 scheduler `[CUDA, CPU]`、所有 graph tensor 显式 CUDA pin、CPU scheduler buffer 为 0、parallel 与 op-offload 均为 false。它支持“没有 CPU graph-compute fallback”，不量化 host scheduler 或同步开销。

12 个 direct/profile formal clock gate 均通过，360 个 formal window 的最大前/后 read-window 间隙分别为 `5.8680 ms` / `5.8905 ms`，低于 25 ms 门限。

## 可以与不可以使用

可以保留为同 DLL、同硬件、synthetic SCALE CUDA dispatch 的诊断证据：device 侧 kernel-only 分布稳定，且本次 raw、时钟和 CUPTI 语义完整。后续机制分析必须把 kernel-only 证据与 whole-graph host wall 分开。

不能据此增加 simulator 成本系数。尤其不能把 host wall 减去 kernel 时间解释成 CPU 常数：该差额仍含 launch、scheduler、图末同步、profiling 扰动和主机抖动。也不能把该合成 SCALE 路径外推成 LLM 的 fusion、CUDA graph、多 stream、CPU worker pool 或端到端时延模型。
