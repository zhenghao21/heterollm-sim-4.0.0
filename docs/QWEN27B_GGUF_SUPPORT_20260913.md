# Qwen3.8-27B GGUF 支持与 smoke 证据

项目 37 已可将 `Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf` 绑定为可执行
`ModelSpec`。GGUF 文件级几何为 65 blocks、hidden 5120、attention heads 24、KV
heads 4、vocabulary 248320；其中 1 个 block 是 `nextn` 辅助头，主干执行图为
64 layers，包括 48 个 linear attention 和 16 个 full attention。线性块从 GGUF 的 `ssm.*` 元数据构造为
`key_heads=16,value_heads=48,key/value_head_dim=128,conv_kernel=4,
state_dtype=fp32,output_gate=true`。

Qwen3.8 的 block 64 同时带有 `nextn.*` 辅助预测张量。解析器保留
`n_layer_all=65`、`n_layer_nextn=1`，只把 64 层主干下沉到执行图；`nextn`
辅助头尚未单独 materialize 成 MTP 分支，矩阵结果保留这一边界说明。

验证：

* `tests/test_gguf_parity.py`：6 passed；
* CPU-only llama.cpp smoke（ctx=512、batch/ubatch=64、threads=16、
  `-ngl 0`、predict=2、prompt=`Hi.`）的 geometry/token gate 通过；
* 修复后结果：[audit_qwen27_cpu_short_v4.json](../artifacts/audit_qwen27_cpu_short_v4.json)；
  TTFT −43.59%、TPOT −31.12%、E2E −38.03%。

该 GGUF 文件约 13.84 GiB；RTX 5080 可用显存约 14.1 GiB，full offload 还需
为运行时和 KV cache 预留空间，可能 OOM。首轮多模型矩阵应优先测试 CPU-only
或小比例 partial offload，并将 OOM 单元标记为 excluded。
