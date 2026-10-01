# 项目约定

## 默认误差对比配置（用户于 2026-09-29 指定）

后续用户要求做误差对比时，默认使用本次 native 同配置基线，不得换成前端自动部署默认值。除非用户明确指定改变比较条件，否则固定模型、负载、硬件与运行参数，只切换待评估的仿真器版本/成本模型。

- 配置与审计依据：`artifacts/development/ui_native_matched_20260929/configuration_audit.json`。
- 六个完整仿真输入：同目录 `*.scenario.json`；运行及构建输入的方法见 `run_comparison.py`。复用脚本时将结果写入新目录，不覆盖本次基线及历史 native/R5。
- Native 基准：`artifacts/development/native_long_grid_135_20260915/stable_native_dataset.json` 中对应六个 cell；R5 对照：其 `optimization_loop/round_005/r5_llama_native_preset_results.json`。
- 场景：Qwen2.5、SmolLM2、TinyLlama、Qwen3.5、Qwen3.8 CPU、Qwen3.8 GPU；使用对应 GGUF SHA 和原始 prompt 身份。
- 负载：P512/O128/C1；batch=64、ubatch=64、context=2048、threads=16、threads_batch=16、seed=42。
- llama.cpp 调度映射；KV K/V F16；FlashAttention 关闭；保留 native 的硬件快照、时钟、源码运行契约及采样配置；不重置硬件 profile，不启用自动内存分层（device_memory_tiering=false）。完整序列化输入优先于本摘要。
- 前四个场景 gpu_layers=-1；Qwen3.8 CPU=0；Qwen3.8 GPU native -ngl=66，按有 GGUF MTP 依据的主干映射为仿真 65，不得把两个场景都改成 -1。
- 本次启用了当前 Blackwell analytical kernel_model；后续保留该建模路径，版本升级须明确记录，不得默默退回旧模型。不得为降低误差用目标 LLM 耗时拟合参数或安装未经验证的实测曲面。
- 指标采用 engine 边界 TTFT/TPOT/E2E，与历史 native 对应 cell 的中位数比较；TPOT=(末 token 时间-首 token 时间)/(输出 token 数-1)，排除模型加载及客户端网络耗时。
- 如配置无法复现或模型身份不符，明确报告，不能静默替换。替代部署实验须单独标注，不能冒充同配置预测精度。


## Kernel surface acceptance state (2026-09-30)

- Synthetic cold-cache holdouts are recorded under `artifacts/development/cold_surface_*_20260929` and `cold_surface_q4_repeat_20260930`; they are evidence only and are not installed as the default LLM profile.
- Accepted narrow operator domains: N interpolation at M=1/2/4,K=4096 for Q4_K/Q6_K/IQ4_XS; K interpolation at M=1,N=4096 for all three; K interpolation at M=2/4,N=4096 for Q6_K/IQ4_XS.
- Rejected domain: Q4_K K interpolation at M=2/4,N=4096. The K=2560 anomaly is repeatable; do not use separable or blanket Q4_K interpolation there.
- Acceptance matrix: `artifacts/development/cold_surface_acceptance_20260929.json`.

## Level-2 calibrated analytical surface (2026-09-30)

- Opt-in preset: `blackwell_calibrated_analytical_v1`; the ordinary frontend `blackwell_analytical_v1` remains the uncalibrated analytical path.
- Surface source/build: `tools/collect_level2_shape_grid.py`, `tools/build_level2_surfaces.py`, `artifacts/development/level2_shape_grid_20260930/` and `artifacts/development/kernel_level2_surface_manifest.json`.
- Surface uses independent synthetic CUDA main-kernel measurements only; no native LLM latency fitting. It binds runtime DLL SHA, source MMVQ signature, cache protocol, dtype/output and complete shape cells. Unsupported or rejected cells fall back to analytical.
- Holdout acceptance: `artifacts/development/level2_shape_grid_20260930/holdout_evaluation.json`; four of seven declared holdouts pass the current `<10% APE` and `<10% CV` gates. Q5_K/Q5_0 exact observations and Q8_0 observations remain evidence only because no complete validated joint cell/holdout surface was accepted; rejected correctness and high-CV cases are listed in the protocol/manifest.
- Native-matched Level-2 comparison: latest rerun is `artifacts/development/ui_native_matched_level2_mmq_20260930/`; `ui_native_matched_level2_release2_20260930/`, `ui_native_matched_level2_grid_20260930/` and `ui_native_matched_level2_grid2_20260930/` are retained as prior attempts. All remain separate from the analytical baseline.

## Level-2 prefill MMQ extension (2026-09-30)

- Independent protocol and measurements: `artifacts/development/level2_mmq_prefill_exact_20260930/`; collector `tools/collect_level2_mmq_grid.py`, exact-Q8 probe `tools/probe_synthetic_mmq.py`, holdout report `holdout_evaluation.json`.
- The installed surface now includes only MMQ prefill M=512 Q4_K/Q6_K/IQ4_XS cells whose six holdouts pass CV<10% and APE<10%. MMQ stream-K and no-fixup dispatch signatures are separate; no interpolation crosses them.
- The measured boundary is the `mul_mat_q` main kernel. Activation quantization/repacking and `mul_mat_q_stream_k_fixup` remain separately priced and are not included in the surface wall.
- MMQ resources are bound from Nsight Compute for the exact specialization (255 registers/thread, 58880 B shared allocation, 8 warps/CTA) and remain guarded by hardware/runtime/source signature.
- Latest native-matched rerun is `artifacts/development/ui_native_matched_level2_mmq_20260930/`; its mean APE is TTFT 24.5985%, TPOT 36.0785%, E2E 31.2146%, overall 30.6306%. This is an opt-in experiment, not a claim that every LLM path is validated.
