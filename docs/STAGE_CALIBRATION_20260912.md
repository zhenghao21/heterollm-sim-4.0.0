# CUDA 阶段留出校准

本轮使用项目 37 内同一个 Qwen2.5-0.5B Q4_K_M GGUF、同一 llama.cpp server 参数和本机 RTX 5080，分别采集了一条训练 profile 和一条留出 profile。采集命令由 [native_llama_profile.py](../tools/native_llama_profile.py) 生成，Nsight Systems 同时启用 CUDA、NVTX、WDDM 和 CUDA graph node trace。

产物如下：

- [native_profile_operator_v1.json](../artifacts/native_profile_operator_v1.json)：训练场景，6 prompt tokens、4 decode tokens；
- [native_profile_operator_v1.sqlite](../artifacts/native_profile_operator_v1.sqlite)：CUPTI/graph node 原始导出；
- [native_profile_operator_v1.trace.json](../artifacts/native_profile_operator_v1.trace.json)：4,110 个规范化事件；
- [native_profile_operator_holdout_v1.json](../artifacts/native_profile_operator_holdout_v1.json)：留出场景，23 prompt tokens、8 decode tokens；
- [native_profile_operator_holdout_v1.trace.json](../artifacts/native_profile_operator_holdout_v1.trace.json)：6,886 个规范化事件；
- [native_stage_holdout_calibration_v1.json](../artifacts/native_stage_holdout_calibration_v1.json)：阶段训练/留出误差报告。
- [build_semantic_calibration.py](../tools/build_semantic_calibration.py)：按 phase 和 tensor shape 分组的显式 NVTX 校准工具。

训练场景按“阶段总 kernel 时间 / 名称启发式归类的 kernel 实例数”拟合，每个速率只在留出场景验证。由于没有 NVTX range，这些实例仍是候选证据，不是真实 operator 边界：

| 阶段 | 训练速率 | 留出实测 | 留出预测 | 留出误差 | 结论 |
|---|---:|---:|---:|---:|---|
| QKV/rope（名称启发式） | 1,216.70 ns/实例 | 473,618 ns | 467,212 ns | -1.35% | 仅 evidence-only |
| FFN（目前只有 SiLU 激活核） | 926.61 ns/实例 | 28,000 ns | 21,312 ns | -23.89% | 需更多 shape |
| KV | — | — | — | — | 无语义证据，阻塞 |
| lm-head（名称启发式） | 1,256.00 ns/实例 | 21,121 ns | 20,096 ns | -4.85% | 仅 evidence-only |

launch 和同步单独从 CUDA API 统计，不折算为某个 GEMM：

| 阶段 | 训练速率 | 留出实测 | 留出预测 | 留出误差 |
|---|---:|---:|---:|---:|
| kernel launch | 3,225.48 ns/call | 4,362,714 ns | 6,149,258 ns | +40.92% |
| stream synchronize | 90,732.66 ns/call | 17,969,512 ns | 14,517,226 ns | -19.21% |

当前二进制没有 NVTX activity 表。虽然 CUPTI kernel、runtime、memcpy 和 CUDA graph node 均已采集，但训练 profile 中 73.45%、留出 profile 中 77.53% 的 kernel 时间仍被归类为 `unknown`。因此：

- QKV/rope 和 lm-head 的结果只能作为名称启发式的 evidence-only 阶段系数；
- FFN 目前只覆盖激活核，不能代表 FFN GEMM 的完整成本；
- KV 没有可验证实例，保持 blocked；
- launch/synchronize 的总 API 均值受 CUDA graph、后台同步和 shape 混合影响，不能直接作为每个仿真 task 的固定服务时间；
- 本次两份硬件快照的动态时钟可能不同，留出误差只能解释为同机趋势证据，不能视为硅片绝对校准。

Nsight WDDM trace 在当前权限下被禁用，但 CUDA activity 仍然成功。若要完成真正的 operator 级校准，需要重新编译 llama.cpp，在 ggml-cuda 的 QKV、FFN、KV、lm-head dispatch 周围插入 NVTX range，并将 range 与 request、prefill/decode、CUDA graph node correlation 一起导出。获得这些标记后，再把阶段系数映射到仿真器对应 task；在此之前，未知阶段和 API aggregate 只保留为证据，不自动改变绝对时延模型。

## Graph 与 direct dispatch 分离

后续检查发现该 llama.cpp 版本没有运行时关闭 CUDA graph 的环境变量，`GGML_CUDA_DISABLE_GRAPHS=1` 只是无效提示。为得到真正的 direct 证据，项目 37 又用相同源码和 NVTX patch 构建了 `GGML_CUDA_GRAPHS=OFF` 的 [v4 direct Release 包](../artifacts/llama-semantic-build-v4-direct-release)。其 direct trace 不含任何 `graph_node_id`，训练/留出分别有 4,612/9,424 个规范化事件和 905/1,316 个 NVTX ranges。

direct 版本的名称启发式阶段留出结果为：KV +1.68%、lm-head −1.25%、QKV −36.35%、FFN −19.80%。QKV/FFN 的偏差说明即使剥离 graph replay，也需要按 phase、token shape、矩阵维度和 quantization shape 建模；它们不能用单一每实例常数替换仿真器成本。

在 shape 标签加入后，又以相同 prompt（6 tokens）固定 `predict=8`、`warmup_predict=8` 做了一次 direct 留出。此时 prefill shape 完全一致，QKV 总误差降为 −2.91%，FFN 为 −0.20%；decode 阶段 QKV −3.03%、KV −9.12%、lm-head −4.69%。这组结果支持按 phase/shape 分桶，但 native host timing 本身仍有较大波动（同配置 formal predicted 时间约 28.2 ms 与 136.7 ms），所以系数仍只保存为 evidence-only，不能据此宣称绝对时延已校准。
