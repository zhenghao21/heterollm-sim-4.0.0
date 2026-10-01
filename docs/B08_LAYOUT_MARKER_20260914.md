# 显式 layout marker 与 extractor

> `source/llama.cpp-semantic` 是本机忽略的外部 llama.cpp checkout，不随仓库提交。没有恢复该 checkout 时，本文只作为插桩契约和路径说明，不能直接执行构建命令。

## 格式

在 `source/llama.cpp-semantic/ggml/src/ggml-cuda/ggml-cuda.cu` 的 `ggml_cuda_nvtx_scope` 中使用 `layout=` 字段。分类只依赖 tensor stride 语义：`contiguous`、`transposed`、`permuted`、`strided`；不写入地址或其他场景答案。`tools/extract_nsys_trace.py` 将该字段保存为 `semantic_layout`。

## 解释限制

显式layout字段可用于 `(stage,phase,shape,dtype,layout,kernel_family)` 联合覆盖检查。字段可提取不等于不同shape或kernel路径已覆盖，不能据此启用全局stage/operator速率。未命中的适用域保持分析回退或证据不足；任何新采集须遵守任务书。
