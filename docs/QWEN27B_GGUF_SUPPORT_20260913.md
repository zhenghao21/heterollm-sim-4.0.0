# Qwen3.8-27B GGUF 支持边界

项目 37 已可将 `Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf` 绑定为可执行
`ModelSpec`。GGUF 文件级几何为 65 blocks、hidden 5120、attention heads 24、KV
heads 4、vocabulary 248320；其中 1 个 block 是 `nextn` 辅助头，主干执行图为
64 layers，包括 48 个 linear attention 和 16 个 full attention。线性块从 GGUF 的 `ssm.*` 元数据构造为
`key_heads=16,value_heads=48,key/value_head_dim=128,conv_kernel=4,
state_dtype=fp32,output_gate=true`。

Qwen3.8 的 block 64 同时带有 `nextn.*` 辅助预测张量。解析器保留
`n_layer_all=65`、`n_layer_nextn=1`，只把 64 层主干下沉到执行图；`nextn`
辅助头尚未单独 materialize 成 MTP 分支，使用时须保留这一边界说明。

GGUF约13.84 GiB；完整GPU offload还需为运行时与KV cache预留空间。容量判断应使用运行时设备可用内存，不能仅以权重文件大小断言可运行。固定验证范围内的OOM或缺失结果按任务书计入覆盖率，不以删除单元解决。
