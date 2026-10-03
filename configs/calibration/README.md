# 校准配置

`qwen25_p4_native_phase_v1.json` 是一个可执行的、身份绑定的开发校准配置。它从项目已有的 Qwen2.5 CUDA 训练 profile 复用 phase-boundary 系数，并补齐了当前 Native payload 的 `binary_sha256`、DLL `runtime_artifacts`、模型/硬件指纹，以及 Native runtime 到仿真器 runtime 的显式映射。

这不是形状等价的盲测配置：源 profile 是 prompt=8/output=8/parallel=1，目标回放是 prompt=186/output=49/parallel=4。文件显式保留 `source_capture_shape`、`target_replay_shape` 和 `shape_identity_match=false`，所以它只能用于 Round 068 开发回放，不支持形状泛化或正式发布验收。

它只校准 engine phase boundary；DRAM row/burst/refresh、NAND page/program/erase/queue 仍由各自硬件模型拥有。严格 replay gate 要求 profile 的可执行文件与依赖模块哈希集合和 Native payload 完全一致。

Round 068 的 A/B 与逐指标误差表见：

- `outputs/round_068_joint_calibration_dram_nand.md`
- `outputs/round_068_error_comparison.csv`

Round 069 增加了当前 binary 绑定的 operator/kernel profile：

- `qwen25_current_kernel_v1.json`
- `qwen35_current_kernel_v1.json`
- `smollm2_current_kernel_v1.json`
- `tinyllama_current_identity_kernel_v2.json`
- `qwen38_current_identity_kernel_v2.json`

这些 profile 使用六维 exact key：`stage`、`phase`、`shape`、`dtype`、`layout`、`kernel_family`。kernel-only A/B 只允许修改 GPU/CPU operator compute demand，`apply_memory=false`、`apply_phase_boundary=false`，所以不会接管 DRAM/NAND 物理服务时间。Qwen3.8 的 current CPU trace 缺少 `layout` 和 `kernel_family`，配置保持 `blocked_exact_key_coverage`；缺失字段不会从 kernel 名或 shape 推断。

Round 069 的身份清单、逐指标误差表和覆盖表见：

- `outputs/round_069_current_identity_manifest.json`
- `outputs/round_069_kernel_error_comparison.md`
- `outputs/round_069_kernel_calibration_matrix.json`
