# Semantic request-marker binary 刷新回放报告（2026-09-14）

本报告使用新的 semantic request-marker `llama-server.exe`，并在当前仿真器源码上对既有 formal profile 做同配置回放。request boundary 校准保持关闭；没有向 TTFT/TPOT/E2E 叠加 request_begin/first_token/request_end 项。

固定条件：`ctx=512`、`parallel=1`、`batch=ubatch=64`、`threads=threads_batch=16`、`gpu_layers=-1`、`FA=off`、`seed=42`、warmup output=2、mmap、K/V unified、continuous batching、layer split、pipelined DMA。

| 场景 | 新 native 中位数 TTFT | 新 sim TTFT | TTFT 误差 | 新 native TPOT | 新 sim TPOT | TPOT 误差 | 新 native E2E | 新 sim E2E | E2E 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B prompt=8/output=8 | 6.309 | 5.641 | -10.58% | 4.219 | 4.646 | +10.12% | 36.283 | 38.163 | +5.18% |
| Qwen3.5-0.8B prompt=2/output=8 | 8.200 | 8.797 | +7.28% | 5.628 | 5.496 | -2.34% | 48.356 | 47.268 | -2.25% |

旧 formal 中位数与刷新中位数的差异来自原生服务实测运行时抖动；模拟器侧 profile 未改变。Qwen2.5 的 TTFT 受原生启动边界抖动影响较大，刷新三次 native TTFT 为 4.901/6.309/7.277 ms；Qwen3.5 为 7.346/8.200/9.328 ms。

身份锁定：

- 新 binary SHA-256：`4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337`。
- 两个场景的 model/GGUF SHA、硬件 fingerprint `f320170cd468255451f9f1fb1586ff1be4ddde9303f4a4c85be8d5542dc50977`、runtime fingerprint 与 source SHA 均写入对应 manifest。
- 所有回放 `identity_mismatch=false`，token count 与 prompt fingerprint 均匹配。

产物：

- [Qwen2.5-0.5B prompt=8/output=8 刷新中位数 summary](../artifacts/multimodel_next/qwen25_refresh_requestmarker_median_summary_v1.json)
- [Qwen2.5-0.5B prompt=8/output=8 新 binary identity manifest](../artifacts/multimodel_next/qwen25_refresh_requestmarker_identity_manifest_v1.json)
- [Qwen3.5-0.8B prompt=2/output=8 刷新中位数 summary](../artifacts/multimodel_next/qwen35_refresh_requestmarker_median_summary_v1.json)
- [Qwen3.5-0.8B prompt=2/output=8 新 binary identity manifest](../artifacts/multimodel_next/qwen35_refresh_requestmarker_identity_manifest_v1.json)
- [回放 r1-r3（Qwen2.5-0.5B prompt=8/output=8）](../artifacts/multimodel_next/qwen25_refresh_requestmarker_r1.json)
- [回放 r1-r3（Qwen3.5-0.8B prompt=2/output=8）](../artifacts/multimodel_next/qwen35_refresh_requestmarker_r1.json)

