# Qwen3.8 CPU-only 新 marker binary 回放（2026-09-14）

本报告记录当前源码与请求级 NVTX marker binary 下的 Qwen3.8-27B CPU-only 严格匹配回放。请求边界校准保持关闭；复用已验证的 prompt=8/output=8 operator-wall exact semantic profile，仅应用 stage 和 memory 校准。

## 锁定条件

- 模型：`Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf`
- prompt：`Explain why deterministic benchmarking matters.`（8 tokens）
- output：8 tokens；warmup output：2 tokens
- `ctx=512`、`parallel=1`、`batch=64`、`ubatch=64`、`threads=16`、`threads_batch=16`
- `gpu_layers=0`、`FA=off`、`mmap=true`、`mlock=false`、`offload_kqv=true`、`op_offload=true`、`KV=f16/f16`、`continuous batching=true`、`seed=42`
- profile：`qwen38_cpu_semantic_calibration_prompt8_f9_v2.json`（operator-wall；stage+memory apply；phase/request boundary off）
- marker binary：`llama-server.exe` SHA-256 `4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337`
- `llama.dll` SHA-256 `dba8eb57633ffa2e2d7ad35b1e856376503f5208937a3b9cd6416cbba8a7d55a`
- GGUF SHA-256：`157479b0083662c7c9cfd8754cae24ab9cc1abbd4def91d49758512c90f1e406`
- hardware fingerprint：`f320170cd468255451f9f1fb1586ff1be4ddde9303f4a4c85be8d5542dc50977`
- runtime fingerprint：`21198f0dd7da88f03bd5b8800345ec0eaf95894c15ec440a101750b97a1bd247`
- 当前代码关键 SHA：`planner.py=493f8bc98eaff84f480093782691b7e54780439790c2d75ea122a1aa9a48ab99`；`calibration.py=6b5845e72b9278870c077cd7209533f83e300104aaa78eb65d516b58a13ff86f`

## 三次结果

| 回放 | Native TTFT (ms) | Native TPOT (ms/token) | Native E2E (ms) | TTFT 误差 | TPOT 误差 | E2E 误差 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 1047.001 | 290.913 | 3083.390 | +0.295% | +3.318% | +2.291% |
| r2 | 1112.745 | 283.974 | 3100.560 | −5.631% | +5.842% | +1.725% |
| r3 | 1117.494 | 335.623 | 3466.855 | −6.032% | −10.446% | −9.023% |
| **中位数** | **1112.745** | **290.913** | **3100.560** | **−5.631%** | **+3.318%** | **+1.725%** |

仿真器固定输出为 TTFT `1050.085 ms`、TPOT `300.565 ms/token`、E2E `3154.037 ms`。三次回放 `identity_mismatch=false`，GGUF geometry/tensor parity 均通过，三次 binary hash 均与锁定值一致。

## 证据文件

- [三次 r1 回放](../artifacts/multimodel_next/qwen38_cpu_marker_current_r1.json)
- [三次 r2 回放](../artifacts/multimodel_next/qwen38_cpu_marker_current_r2.json)
- [三次 r3 回放](../artifacts/multimodel_next/qwen38_cpu_marker_current_r3.json)
- [中位数汇总](../artifacts/multimodel_next/qwen38_cpu_marker_current_median_summary_v1.json)
- [当前源码与身份锁定 manifest](../artifacts/multimodel_next/qwen38_cpu_marker_current_source_lock_v1.json)
