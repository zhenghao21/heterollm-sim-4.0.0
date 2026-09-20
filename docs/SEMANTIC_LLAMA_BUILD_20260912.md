# 带 NVTX/CUPTI 语义的 llama.cpp 构建

项目 37 的 `source/llama.cpp-semantic` 提供 Windows + CUDA 12.8 下的 Release semantic 构建。源码提交为 `3057bb66c86c46d5781e50e85462a760ba7d1feb`（llama.cpp `0.4.0-dev`），它与 LMStudio 随附的 `2.33.0 / 0f3a71b` 二进制不是同一提交，因此性能数值不能直接替代原生基线；本构建用于取得 operator 语义和验证插桩路径。

构建配置为：MSVC 19.44、CUDA 12.8.93、`CMAKE_CUDA_ARCHITECTURES=120a-real`、`GGML_CUDA=ON`、`GGML_CUDA_GRAPHS=ON`、`GGML_CUDA_FA=OFF`、`GGML_CUDA_NVTX=ON`。新增的 CMake 选项默认关闭，开启时通过 `find_path` 查找 `nvtx3/nvToolsExt.h`，缺少头文件会直接失败，避免生成没有语义标记的伪语义构建。

插桩位于 `ggml/src/ggml-cuda/ggml-cuda.cu` 的 `ggml_cuda_compute_forward` 外层，为每个 CUDA backend node 产生：

```text
operator:<GGML_OP>:<ggml_tensor_name>
```

tensor name 保留 llama.cpp graph builder 生成的 layer/operator 信息，例如 `MUL_MAT:Qcur-0`、`MUL_MAT:ffn_gate-12`、`SET_ROWS:cache_v_l0`、`MUL_MAT:result_output`。CUPTI 仍由 Nsight Systems 采集，不在 llama.cpp 中链接 CUPTI；提取器使用 kernel/memcpy 的 runtime correlation、同线程和时间包含关系关联 NVTX，graph replay 没有可信 correlation 时保持 unknown。

## Graph 与 direct dispatch

Graph 与 direct 构建使用独立目录。要观测 direct dispatch，须设置 `GGML_CUDA_GRAPHS=OFF`；`GGML_CUDA_DISABLE_GRAPHS=1` 对此源码版本不能替代编译配置。比较时应锁定源码、编译器、CUDA、目标架构、NVTX及GGUF身份，不能混合graph replay与direct kernel时间。

`phase:prefill` 和 `phase:decode` 外层range标记host graph dispatch阶段，不能直接当作GPU完成时间；GPU operator仍需通过同线程runtime correlation归属。NVTX名称、layer、tensor和shape标签提供语义信息，但不能单独证明prefill/decode、graph capture/replay和fused dispatch的生命周期或校准有效性。

本文说明构建与插桩机制，不提供当前验收结果或采集授权。采集权限和证据门槛以任务书为准。
