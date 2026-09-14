# 带 NVTX/CUPTI 语义的 llama.cpp 构建

项目 37 已在本机 Windows + CUDA 12.8 环境中，从项目 37 内的 `source/llama.cpp-semantic` 源码构建 Release 版本。源码提交为 `3057bb66c86c46d5781e50e85462a760ba7d1feb`（llama.cpp `0.4.0-dev`），它与 LMStudio 随附的 `2.33.0 / 0f3a71b` 二进制不是同一提交，因此性能数值不能直接替代原生基线；本构建用于取得 operator 语义和验证插桩路径。

构建参数固定为：MSVC 19.44、CUDA 12.8.93、`CMAKE_CUDA_ARCHITECTURES=120a-real`、`GGML_CUDA=ON`、`GGML_CUDA_GRAPHS=ON`、`GGML_CUDA_FA=OFF`、`GGML_CUDA_NVTX=ON`。新增的 CMake 选项默认关闭，开启时通过 `find_path` 查找 `nvtx3/nvToolsExt.h`，缺少头文件会直接失败，避免生成没有语义标记的伪语义构建。

插桩位于 `ggml/src/ggml-cuda/ggml-cuda.cu` 的 `ggml_cuda_compute_forward` 外层，为每个 CUDA backend node 产生：

```text
operator:<GGML_OP>:<ggml_tensor_name>
```

tensor name 保留 llama.cpp graph builder 生成的 layer/operator 信息，例如 `MUL_MAT:Qcur-0`、`MUL_MAT:ffn_gate-12`、`SET_ROWS:cache_v_l0`、`MUL_MAT:result_output`。CUPTI 仍由 Nsight Systems 采集，不在 llama.cpp 中链接 CUPTI；提取器使用 kernel/memcpy 的 runtime correlation、同线程和时间包含关系关联 NVTX，graph replay 没有可信 correlation 时保持 unknown。

Release 运行包位于 [llama-semantic-build-v3-phase-release](../artifacts/llama-semantic-build-v3-phase-release)，构建指纹见其 [build_manifest.json](../artifacts/llama-semantic-build-v3-phase-release/build_manifest.json)，版本自检输出为：

```text
version: 0.4.0-dev (build 1, commit 3057bb6)
built with MSVC 19.44.35228.0 for Windows AMD64
```

为拆分 CUDA graph replay，又构建了独立的 [llama-semantic-build-v4-direct-release](../artifacts/llama-semantic-build-v4-direct-release)，其 [build_manifest.json](../artifacts/llama-semantic-build-v4-direct-release/build_manifest.json) 明确记录 `ggml_cuda_graphs=false`。direct 构建与 graph 构建使用相同源码、MSVC、CUDA、`120a-real`、NVTX 和 GGUF；两者的 kernel 时间不能混合解释。此前仅设置 `GGML_CUDA_DISABLE_GRAPHS=1` 的 profile 不作为 direct 证据。

语义采集结果：

- 训练 trace：752 个 NVTX ranges，758 个 kernel 通过 correlation 关联；
- 留出 trace：752 个 NVTX ranges，1,032 个 kernel 通过 correlation 关联；
- 显式 operator 可覆盖 `MUL_MAT`、`ADD`、`ROPE`、`SET_ROWS`、`SOFT_MAX`、`GLU`、`GET_ROWS`；
- [native_semantic_calibration_phase_v1.json](../artifacts/native_semantic_calibration_phase_v1.json) 保存训练/留出阶段系数和 unknown 覆盖率；
- 训练 unknown/uncovered kernel 时间约 71.5%，留出约 80.0%，主要来自 runtime 没有唯一 correlation 或 graph replay 无法做可信 node 对应。

phase 版本还验证了 `phase:prefill` 和 `phase:decode` 外层 range：训练 trace 中分别出现 1/3 个可见 phase range，留出 trace 中分别出现 1/7 个。它们标记的是 host graph dispatch 阶段，不能直接当作 GPU 完成时间；GPU operator 仍以同一线程 runtime correlation 关联。

direct-dispatch 采集结果为 4,612/9,424 个规范化事件，NVTX range 为 905/1,316，kernel 均没有 `graph_node_id`。加入 shape 标签后，最新的 [direct train trace](../artifacts/native_profile_semantic_direct_equal_v1.trace.json) 和 [same-shape holdout trace](../artifacts/native_profile_semantic_direct_equal_sameprompt_holdout_v1.trace.json) 可以按 phase/shape 分桶；相同 prompt 的留出结果为 prefill QKV −2.91%、FFN −0.20%，decode QKV −3.03%、KV −9.12%、lm-head −4.69%。完整结果见 [native_semantic_calibration_direct_sameprompt_v1.json](../artifacts/native_semantic_calibration_direct_sameprompt_v1.json)。

名称、layer、tensor 的 NVTX 语义已经进入 SQLite 和规范化 trace，但还没有把阶段系数自动注入仿真器。原因是 `MUL_MAT` 的显式范围虽然知道 tensor 名，却仍需要确认它在 prefill/decode、graph capture/replay 和 fused dispatch 中的生命周期。要采集 direct dispatch，必须使用单独的 `GGML_CUDA_GRAPHS=OFF` 编译目录；`GGML_CUDA_DISABLE_GRAPHS=1` 对此版本只是无效的环境变量提示。graph-enabled 运行单独作为 replay 证据，不能与 direct kernel 时间混合。这样才可以把 Q/K/V、FFN gate/up/down、KV append/read、lm-head 与仿真器 task 一一对应，并对留出场景做安全绝对时延校准。
