# R21 原生 CUDA 提交、启动与同步静态审计

审计结论：当前规划器已经建模了主机命令构建、驱动提交、GPU 命令处理器和每个已降低 GPU GEMM/张量内核的抽象 `kernel_launch`。对单一、全 GPU、单 split、没有融合且没有临时缓冲区尾部填充的调用，这一结构可以作为分析性近似。

但它**不能证明普通量化 CUDA 推理的原生提交数、同步边界或融合后的启动数正确**。以下三个问题不依赖任何 LLM 时长，也不建议在本轮填入系数。

本审计没有启动原生程序、GPU、微基准或完整仿真，也没有修改运行时代码。

## 已确认的现有建模

- `planner.py:7048-7294` 的 `_add_physical_invocation_frontend()` 按 `invocation_count` 创建 CPU 命令构建、主机提交和 GPU 命令处理器任务。`planner.py:7143-7175` 明确以 `invocation_count * submission_ns` 计费。
- 普通请求路径将 prefill 和每个 decode 固定传入 `invocation_count=1`：`planner.py:12960-12971`、`planner.py:13040-13052`。
- GPU GEMM 成本模型创建一个 `kernel_launch` 阶段：`cost_models.py:2702-2720`；规划器随后把每一成本阶段变成串行依赖任务：`planner.py:10062-10175`。通用张量内核也逐阶段建模启动：`planner.py:10373-10535`、`cost_models.py:2877-2942`。
- 因而当前不是“完全没有启动/提交建模”，问题是它们与 ggml 的原生图 split、融合和同步拓扑没有受同一份运行时证据约束。

## F1：图提交固定为一次，未绑定 `n_splits`

**严重度：高；性质：结构性计数/依赖遗漏；不构成可填入的服务速率。**

原生来源中，`llama_context::graph_compute()` 调用一次 `ggml_backend_sched_graph_compute_async()`（注解编译源 `src/llama-context.cpp:2505-2539`）。但该调度函数进入 `ggml_backend_sched_compute_splits()`（固定语义源 `ggml-backend.cpp:2014-2027`），后者以 `split_id < sched->n_splits` 循环（`ggml-backend.cpp:1643-1656`），并在每个 split 上调用一次 `ggml_backend_graph_compute_async(split_backend, &split->graph)`（`ggml-backend.cpp:1795-1799`）。

因此，原生 backend 图提交次数是运行时 `n_splits`（回调路径还会进一步按 graph view 分段），不是逻辑“目标算子调用数”。跨 backend 的前一 split 结束后，源码还会在下一 split 前执行事件同步或 backend 同步（`ggml-backend.cpp:1658-1666`），并对输入使用事件等待/同步（`ggml-backend.cpp:1674-1688`）。

当前普通 prefill/decode 都把前端 `invocation_count` 固定为 1，且前端没有 `n_splits`、split backend 序列、split 输入拷贝或 split 间事件依赖字段。因此：

- 当实际 `n_splits == 1` 且没有回调分段时，当前提交数可能碰巧匹配；
- 当实际 `n_splits > 1` 时，`host_cohort_submit`、命令构建和 GPU 命令处理器的计数均少计，且 split 间同步/输入就绪依赖不可表达；
- 现有 `llama_runtime_source_binding.py` 只把 `ggml_backend_sched_compute_splits` 记录为源符号（`tools/llama_runtime_source_binding.py:201-210`），没有冻结调用时 `n_splits` 或每一 split 的 backend/输入清单，故不能把“1”宣称为原生已证实计数。

**窄修复：**将每个原生 graph invocation 的静态/记录契约扩展为 `split_count`、有序 `split_backend_ids`、每个 split 的输入拷贝/事件依赖和 callback-view 分段状态。仅在该契约完整且 `split_count == 1` 时保留一条聚合提交任务；否则按 split 创建提交/命令处理任务和显式依赖。服务时间保持“未定价”，不得从 host wall 或 LLM 端到端数据回填。

**所需回归测试：**构造纯 IR fixture，`n_splits=2`，验证有两条提交/CP 任务，第二条依赖第一条的完成事件或同步边界；`n_splits=1` 保持当前一条任务。缺少该契约时，报告覆盖状态为 `uncovered`，而不是输出“native count verified”。

## F2：服务端同步有因果边，但没有主机同步占用

**严重度：中高；性质：结构性主机工作遗漏；同步等待的服务率未知。**

固定服务端源在普通有输出 batch 中，在 `llama_decode()` 返回成功后立即调用 `llama_synchronize(ctx_tgt)`（注解编译源 `tools/server/server-context.cpp:3889-3917`）。`llama_context::synchronize()` 调用 `ggml_backend_sched_synchronize()`（注解编译源 `src/llama-context.cpp:728-749`）；后者遍历所有 backend 并调用 `ggml_backend_synchronize()`（固定语义源 `ggml-backend.cpp:2029-2040`）。CUDA backend 的同步实现是 `cudaStreamSynchronize(cuda_ctx->stream())`（注解编译源 `ggml-cuda.cu:2584-2590`）。

当前计划器在 logits D2H 后放置 `completion_interrupt`，但其资源服务时间明确为 0：`planner.py:19161-19191`，并标注 `host_wait_service_ns: 0.0` 与 `completion_interrupt_latency_not_declared`。这能保留“输出 D2H 完成后才能采样”的一条 DAG 因果边，却没有让执行 `llama_synchronize` 的主机线程/调度资源在等待期间被占用，也没有表达“同步全部 scheduler backend”的边界。

结果是：

- 对单请求、严格串行的输出链，GPU 完成时间会因依赖而进入总路径，故不应把它称为完全丢失的设备等待；
- 对并发 cohort，零资源完成标记允许主机控制资源在原生同步仍阻塞调用线程时被过早复用；
- 同步的 CPU API 开销与实际阻塞时长尚无合格服务率。它们是未知量，不能用本轮 57-98 us host wall 或任何 LLM 时长推导。

**窄修复：**在源契约确认“有输出后调用 `llama_synchronize`”的普通 server 路径上，新增一个显式 `host_backend_synchronize` 任务：依赖所有该 graph invocation 的 split 完成事件/后端完成；占用专用 host scheduler/wait 资源；服务字段只允许为 `unknown_unpriced` 或经独立 API 微测量合格后才为非零。将现有 D2H/采样依赖接到该同步边界之后。不要把未知主机等待并入每个 kernel launch。

**所需回归测试：**IR fixture 中令两个 cohort 共用 host scheduler 资源、一个 invocation 的 GPU 完成晚于下一 cohort 的 host 控制任务。启用同步契约时，下一份使用同一同步调用线程的 host 控制任务不得穿过同步边界；没有契约时必须显示 `uncovered`，不得凭默认值阻塞或填时延。

## F3：MMVQ 临时权重填充清零的额外异步命令未表达

**严重度：中；性质：条件性原生启动/队列计数遗漏；服务率未知。**

规划器对满足 MMVQ 条件的普通物理投影，创建一个 F32->Q8_1 conversion 张量内核（`planner.py:9870-9918`）和一个 GEMM 启动阶段（`cost_models.py:2702-2720`）。这个两启动结构与没有尾部填充时的“量化 + 向量点积”基本形式相符。

但是固定 MMVQ 实现中，若 `src0` 是 `GGML_BACKEND_BUFFER_USAGE_COMPUTE` 且分配字节大于数据字节，先执行 `cudaMemsetAsync` 清除尾部填充（`source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:1473-1481`），随后执行 `quantize_row_q8_1_cuda`（`:1484-1491`）和 `mul_mat_vec_q_switch_type`（`:1515-1519`）。该分支至少多一个异步 CUDA 内存命令，并且它必须先于量化和向量点积。

当前 `_mmvq_activation_conversion_workload()` 的资格只使用 `M/K/格式`（`planner.py:8642-8668`），其元数据没有 `compute-buffer usage`、`allocation bytes` 或 `padding bytes`。因此在上述已满足的源条件下，模型把原生三命令拓扑降低为两条启动任务；该差异是可证明的条件性少计。不能假设所有普通调用都触发，也不能把它变成固定第三个启动。

**窄修复：**把 `src0` 临时缓冲区使用类型、`size_alloc`、`size_data` 和 `padding_bytes` 纳入逐投影调用契约。当且仅当 `usage == COMPUTE && size_alloc > size_data` 时，在 conversion 前创建一条 `temporary_padding_clear` 异步内存命令，依赖顺序为 clear -> q8_1 conversion -> MMVQ dot；缺少这些字段时保持未覆盖，不增添猜测性启动。

**所需回归测试：**同一 `M/K/format` fixture 分别给出 `padding_bytes=0` 与正值。前者应为 conversion + dot 两个原生提交单元；后者应多出清零任务，且量化任务依赖清零。测试应仅验证任务数和 DAG，不断言任何 ns。

## 融合与并发流：当前不能宣称计数正确

R20 已冻结 `compiled_cuda_graphs=false`，这排除了 CUDA Graph replay 的单一 `cudaGraphLaunch` 解释；但它**不等于关闭 ggml CUDA 算子融合或并发流**。已编译 CUDA 源在直接执行路径中先尝试 `ggml_cuda_try_fuse()`，成功时跳过若干后续 graph node（注解源 `ggml-cuda.cu:4365-4385`），常规 node 才进入 `ggml_cuda_compute_forward()`（`:4404-4412`）。同一源还以 `cudaEventRecord`/`cudaStreamWaitEvent` 实现并发流的 join（`:4332-4335`）。

规划器含有若干手工融合（如 fused attention）和分析性 `fused_dequant`，但没有冻结“实际 CGraph 节点序列 -> `try_fuse` 决策 -> 启动单元/stream”映射。故它不能证明每个语义 tensor task 都对应一个原生启动，也不能证明其串行依赖没有把原生可并行流强行串行化。这里不能反向地把所有节点合并：是否融合取决于源码谓词和当次图形状、布局、dtype、环境开关。

**窄修复：**为固定原生输入收集只读图编译/融合 ledger：CGraph 节点序列、被跳过的 node 数、fusion family、stream id、fork/join event、CUDA Graph enabled 状态。规划器仅把 ledger 中同一已证实 launch unit 合并；没有 ledger 时保持分析性节点并标记 `native_launch_topology=uncovered`。不要由 kernel union 时长、host wall 或误差拟合选择融合数。

## 来源链与适用界限

以下来源均由 R20 `graph_gap_collection_r2/execution_freeze.json` 的 `native_and_tool_refs` 和/或 `source/llama.cpp-annotation-control/evidence/build_receipt.json` 绑定：

| 源文件 | SHA-256 | 用途 |
| --- | --- | --- |
| `source/llama.cpp-annotation-control/ggml/src/ggml-cuda/ggml-cuda.cu` | `38b7fbbaa33cd7ca4f3b59cd1b19dcd9d2fdff3109e05eeeadc274dd69f708e6` | 直接 graph loop、融合尝试、CUDA stream 同步、CUDA Graph 分支 |
| `source/llama.cpp-semantic/ggml/src/ggml-backend.cpp` | `803298af097545b4b611df53a2e9ac39f43c18c095c0d3b4c879d2ae911a2d79` | scheduler split 循环、每 split 异步提交、scheduler 同步 |
| `source/llama.cpp-annotation-control/src/llama-context.cpp` | `e677c1e6e56fc08561fa56d9861501740405190e578d690e9efcad26fa48622e` | `graph_compute_async` 与 context synchronize 调用点 |
| `source/llama.cpp-annotation-control/tools/server/server-context.cpp` | `99f7aead4dd6b190292db2a14b2586d4076871a3710a49a94c71424b6f05501e` | 普通有输出 server batch 的 `llama_synchronize` 调用点 |
| `source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu` | `14026871030393662628abdbd4937d5cab72031e20ddf582c9de1d7b424bb368` | MMVQ 填充清零、Q8_1 转换和 dot 提交顺序 |

MMVQ helper 的 hash 已由 `src/heterollm_sim/mmvq_work.py:13-19` 的固定源契约锁定。R20 的直接 execution freeze 显式列出编译的 CUDA wrapper、scheduler、context 和 server 源；它没有单独列出 helper translation unit 的原始编译输入。因此 F3 的 helper 分支是“固定源契约支持的静态机制”，还应在后续 build receipt/调用 ledger 中补一条直接 helper 编译/链接来源，才可升级为该运行的原生计数已证实。

所有三项的未知都是服务率未知，不是零成本：没有独立、身份绑定且通过稳定性门槛的 CPU submit、host synchronize 或 async memset 服务测量前，任何 `ns` 值都不得写入成本模型。R20 的 kernel union 与 host wall 仅可作为未来独立测量设计的动机，不能作为本审计的系数来源。
