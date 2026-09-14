# TinyLlama 请求级 marker 稳定性审计

semantic-request-markers-v1 binary（SHA256 `4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337`）在同一 prompt/output/hardware/runtime 下采集了 prompt=3/output=8 与 prompt=29/output=8 的 train/holdout trace。Nsight 点事件使用 `eventType=34`，range 事件不参与 marker 间隔计算。

| 场景 | marker | train (ns) | holdout (ns) | holdout 相对误差 |
|---|---|---:|---:|---:|
| prompt=3 | request_begin→prefill_begin | 16,560 | 17,260 | −4.06% |
| prompt=3 | prefill_end→first_token | 106,723 | 292,227 | −63.48% |
| prompt=29 | request_begin→prefill_begin | 25,500 | 17,971 | +41.90% |
| prompt=29 | prefill_end→first_token | 98,473 | 159,014 | −38.07% |

request_end 没有可归属的尾部区间，因此保持零下界。只有 prompt=3 的 request_begin 间隔在一次留出采样中接近稳定；first_token 间隔和长 prompt 的 request_begin 间隔都明显受 host scheduling / response handoff 噪声影响。当前证据不足以把它们作为正式 additive request-marker calibration，应保持 `unstable`/fail-closed，仅把 marker 用于边界可观测性诊断。

来源：

- `artifacts/multimodel_next/tiny_marker_short_train_v1.trace.json`
- `artifacts/multimodel_next/tiny_marker_short_holdout_v1.trace.json`
- `artifacts/multimodel_next/tiny_marker_long_train_v1.trace.json`
- `artifacts/multimodel_next/tiny_marker_long_holdout_v1.trace.json`
- `artifacts/multimodel_next/tiny_marker_short_request_boundary_profile_v1.json`
- `artifacts/multimodel_next/tiny_marker_long_request_boundary_profile_v1.json`
