# GPU 首个 decode 启动残差机制

`decode_first_invocation_extra_ns` 表示可选的首个GPU decode启动成本，不是固定每token倍率，也不能吸收到HBM带宽。是否适用及参数值必须由独立证据支持。

## 下沉规则

`decode_first_invocation_extra_ns` 只在显式首个 GPU decode invocation 上应用一次：静态 lowering 的 `decode0001`，或 continuous cohort 中每个 item 的 `context_tokens == prompt_tokens`。CPU-only (`gpu_layers=0`) 和后续 decode invocation 不应用；稳定 decode 的资源成本不被覆盖。profile 缺失、coverage 非 covered 或 identity 不匹配时 fail-closed。

## 可证伪条件

审计必须在后续 trace 中检查：首个 decode invocation 是否仍有同类 kq/图启动空窗；额外成本是否只出现一次；后续 invocation 和 CPU 路径是否保持未增加。若独立 shape 或模型显示启动空窗消失或大小不同，应拒绝复用该 profile 并回退到 analytical fallback。
