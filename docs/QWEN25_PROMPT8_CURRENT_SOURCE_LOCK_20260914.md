# Qwen2.5 prompt=8 当前源码锁定证据

本记录把 Qwen2.5-0.5B-Instruct 的 `prompt=8 / predict=8` boundary 校准三次回放绑定到当前项目源码、同一 GGUF、同一 llama.cpp semantic runtime 与同一硬件身份。机器可验证的完整清单见 [identity manifest](../artifacts/multimodel_next/qwen25_medium_prompt8_boundary_current_identity_lock_v1.json)。

## 锁定结论

- 三次回放均使用同一 prompt：`Explain why deterministic benchmarking matters.`，prompt fingerprint 为 `1d77ea96d081577d79c8502b5221e0c76262b13a70ec62a9d6a1edcceaefdd02`。
- 三次回放的 native/simulator 配置一致：`ctx=512`、`batch=ubatch=64`、`threads=threads_batch=16`、`gpu_layers=-1`、`FA=off`、`seed=42`、continuous batching、pipelined coherent DMA。
- GGUF parity 三次均 pass，无 geometry、tensor format 或 physical bytes mismatch；磁盘 GGUF SHA-256 为 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`。
- llama.cpp semantic binary SHA-256 为 `dd8b158cef066071e133ca8ebcc49a672d50daaf3fccd36676bce76e174d54c0`。
- hardware fingerprint 为 `f320170cd468255451f9f1fb1586ff1be4ddde9303f4a4c85be8d5542dc50977`；runtime fingerprint 为 `90617f319d66c9f002eaedb765665224c97af898c630d19ff1e5abdc2da153d6`。硬件指纹算法排除会变化的时钟与显存 free/used 字段。
- 三次 `identity_mismatch=false`，所有 identity consistency 检查为 true。

## 当前源码范围

Manifest 对以下文件做逐字节 SHA-256 锁定：

- `src/heterollm_sim/planner.py`
- `src/heterollm_sim/calibration.py`
- `src/heterollm_sim/gguf_parity.py`
- `tools/native_llama_compare.py`
- `tools/native_llama_profile.py`
- `tools/build_calibration_profile.py`
- `tools/extract_cuda_api_phase.py`
- `tools/build_semantic_calibration.py`

同时锁定 boundary profile、train/holdout profile、CUDA API phase evidence、三次回放、GGUF 和 binary 的 SHA。

## 三次中位数

| 指标 | native 中位数 | simulator | 相对绝对误差 |
|---|---:|---:|---:|
| TTFT | 5.255 ms | 5.6413 ms | 7.35% |
| TPOT | 4.3550 ms/token | 4.6459 ms/token | 6.68% |
| E2E | 36.397 ms | 38.1629 ms | 4.85% |

该记录只证明这组 prompt=8 场景的当前源码身份与回放证据完整；其他 prompt 或配置必须使用对应的 profile，不能外推本 profile。
