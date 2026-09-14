# E-ENGINE GPU 首个 decode 启动残差（2026-09-14）

## 结论

Qwen2.5-0.5B、RTX 5080、semantic llama.cpp 的 GPU trace 显示，首个 `M=1` decode invocation 的 `kq-0` operator range 含一个没有对应 GPU kernel/memcpy 事件的 runtime/图启动空窗。该空窗在后续 decode invocation 中消失，因而不能用固定每 token 倍率表示，也不能吸收到 HBM 带宽。

## 证据

目标 trace [`gpu_semantic_qwen25_p2_o4_v1.trace.json`](../artifacts/development/gpu_semantic_qwen25_p2_o4_v1.trace.json) 的 decode phase wall 为 `15.065527/2.584323/2.378197 ms`。首 phase 中 `MUL_MAT:kq-0` 的 NVTX wall 为 `11.243776 ms`，但捕获 kernel/runtime/memcpy 事件仅有 `1.101539/2.078479/0.024288 ms`，并存在 `9.945725 ms` 与 `1.246022 ms` 的事件空窗。

独立的 B06/B07 Qwen2.5 trace（不包含目标 p2/o4 的 native 总时延）给出首 decode phase 减去后续 phase 中位数的差值：`10.973093、8.407471、9.027459、11.205085 ms`。profile 选择保守最小值 `8.407471 ms`，并绑定模型、硬件和 runtime identity。

## 下沉规则

`decode_first_invocation_extra_ns` 只在显式首个 GPU decode invocation 上应用一次：静态 lowering 的 `decode0001`，或 continuous cohort 中每个 item 的 `context_tokens == prompt_tokens`。CPU-only (`gpu_layers=0`) 和后续 decode invocation 不应用；稳定 decode 的资源成本不被覆盖。profile 缺失、coverage 非 covered 或 identity 不匹配时 fail-closed。

## simulator-only 回放

native actual 保持原值，未重新启动 native binary：

| 指标 | Native | Simulator（基线） | Simulator（残差） | 残差版本误差 |
|---|---:|---:|---:|---:|
| Engine TTFT | 5.650 | 5.507 | 5.507 | -2.54% |
| Engine TPOT | 6.818 | 4.626 | 7.429 | +8.95% |
| Engine E2E | 26.105 | 19.386 | 27.793 | +6.47% |

结果见 [`e_engine_probe_qwen25_gpu_v4.json`](../artifacts/development/e_engine_probe_qwen25_gpu_v4.json) 和 [`gpu_decode_startup_audit_v1.json`](../artifacts/development/gpu_decode_startup_audit_v1.json)。`native_execution_count=0`。这只是当前 Qwen2.5 GPU identity 的开发验证；没有据此宣称跨模型、跨 shape 或正式冻结验收通过。

## 可证伪条件

审计必须在后续 trace 中检查：首个 decode invocation 是否仍有同类 kq/图启动空窗；额外成本是否只出现一次；后续 invocation 和 CPU 路径是否保持未增加。若独立 shape 或模型显示启动空窗消失或大小不同，应拒绝复用该 profile 并回退到 analytical fallback。
