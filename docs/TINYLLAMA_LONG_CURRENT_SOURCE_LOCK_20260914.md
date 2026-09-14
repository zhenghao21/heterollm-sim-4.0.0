# TinyLlama 长输入当前源码锁定（2026-09-14）

## 场景

- 模型：`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- 输入：`Benchmarking long-context inference requires measuring prompt evaluation separately from steady-state decode, while preserving identical model and runtime configuration.`
- llama.cpp 实际 prompt token 数：29；生成 token 数：8
- 配置：`ctx=512, parallel=1, batch=64, ubatch=64, threads=16, gpu_layers=-1, FA=off, seed=42, temperature=0, top_k=1, warmup_predict=2`
- launch profile：`tinyllama_semantic_calibration_launch8us_locked_v1.json`，`launch_ns_per_call=8000`

## 证据锁定

三次回放文件：

- `artifacts/multimodel_next/tinyllama_long_current_r1.json`
- `artifacts/multimodel_next/tinyllama_long_current_r2.json`
- `artifacts/multimodel_next/tinyllama_long_current_r3.json`

三次检查均满足：prompt fingerprint `5e7b6198dd2ebd8faa9b1c7079ce5b1352ce8ec2e5387e5ad104a512fc31c9c2`、模型/GGUF SHA `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`、runtime fingerprint `90617f319d66c9f002eaedb765665224c97af898c630d19ff1e5abdc2da153d6`、稳定硬件 fingerprint `f320170cd468255451f9f1fb1586ff1be4ddde9303f4a4c85be8d5542dc50977`，且 GGUF parity pass、`identity_mismatch=false`。

完整身份与当前源文件 SHA 见 `artifacts/multimodel_next/tinyllama_long_current_lock_manifest_v1.json`。

## 中位数结果

| 指标 | native 中位数 | simulator | 相对误差 |
|---|---:|---:|---:|
| TTFT | 7.964 ms | 4.4158 ms | −44.55% |
| TPOT | 4.1377 ms/token | 4.1916 ms/token | +1.30% |
| E2E | 36.939 ms | 33.7572 ms | −8.61% |

## 结论与阻塞项

该场景的 decode 和端到端误差已经较小，但 TTFT 仍显著低估。三次采样中 native TTFT 为 7.964/7.856/8.391 ms，差异不足以解释 3.55 ms 缺口；现有 trace 只提供 prefill/decode phase，没有 request-begin、first-token 或 prefill 阶段边界 marker。将旧 operator-wall 或 prefill aggregate 再叠加会重复计费，已验证会使结果进一步偏离。因此该场景保留为 `provisional_long_input`，暂不纳入“所有三项误差均小”的正式矩阵；待 llama.cpp request marker 分支完成后，再按同一 prompt/config 重采样并做边界校准。
