# R21 固定 server 调用链与同步边界：下一项机制修复

## 结论

在固定原生配置 `-np {1,2,4} -b 64 -ub 64 -fa off -kvu --kv-unified-per-slot 2048 -cb` 下，若一个 server `batch_view` 不超过 64 行，且路径正常成功，则可以从锁定源码确定一条逻辑调用链：一次 `llama_decode`、一次 unified-KV ubatch、一次 `process_ubatch`、一次 `ggml_backend_sched_graph_compute_async` 入口。不能据此把原生 backend 图提交数确定为一次；它仍取决于 scheduler 的 `n_splits` 与 callback graph-view 分段。

`llama_synchronize` 也不是每 token 或每输出行调用一次。它在每个**成功且含至少一条 output row** 的 server `batch_view` 中调用一次，并且发生在服务端取得 host 可见输出之前。当前规划器的 `frontend submit=1` 和 D2H 后的零成本 `completion_interrupt` 因而只能算分析性占位，不能标为已由该原生运行验证的提交/同步拓扑。

本报告不启动 native 程序、GPU、目标 LLM 或微基准；不读取目标 LLM 延迟数组；不修改核心代码或冻结文件。

## 源码可确定的正常成功调用计数

### server view 到 decode

固定语义源 `source/llama.cpp-semantic/tools/server/server-context.cpp`：

- server 以 `llama_n_batch(ctx_tgt)` 将工作 batch 切为 `batch_view`；固定 `n_batch=64`，所以 server view `<=64` 时是一个常规单 view；
- 调用 `llama_decode(ctx_tgt, batch_view)` 一次；
- 先扫描该 view 内 `batch.tokens[i].output` 形成 `has_output`；只有 `ret == 0 && has_output` 时才调用一次 `llama_synchronize(ctx_tgt)`。

因此每个正常成功的 server view 的 source-level count 是：

| 事件 | 可静态确认的计数 | 条件 |
| --- | ---: | --- |
| `llama_decode` | 1 | 本 view 已进入 decode |
| `llama_synchronize` | 0 或 1 | `ret == 0` 且 `has_output=true` |

`has_output=true` 的覆盖情形包括：任意 decode row、完成某请求的末尾 prompt row、以及含 decode 或末尾 prompt 的 mixed view。纯粹的非末尾 prefill view 不同步。

### decode 到 scheduler graph entry

固定语义源 `source/llama.cpp-semantic/src/llama-context.cpp`：

- `llama_context::decode()` 遍历 `mctx->get_ubatch()`；
- 每一个 ubatch 调用一次 `process_ubatch()`；
- `process_ubatch()` 在完成图复用/构图、图分配与输入设置后，调用一次 `graph_compute()`；
- `graph_compute()` 对该 ubatch 调用一次 `ggml_backend_sched_graph_compute_async(sched.get(), gf)`。

固定语义源 `source/llama.cpp-semantic/src/llama-kv-cache.cpp` 对 unified KV 的 `n_stream==1` 路径使用 `balloc.split_simple(n_ubatch)`。当 `ubatch=64`、server view `<=64` 且分配成功时，正常路径正好有一个 ubatch。由此得到：

| 调用层级 | 每个正常成功 server view 的计数 |
| --- | ---: |
| `process_ubatch` | 1 |
| `ggml_backend_sched_graph_compute_async` 入口 | 1 |

这个结论的覆盖条件是：因果普通模型路径、server view `<=64`、`n_batch=n_ubatch=64`、unified 分配成功、没有 decode 错误后的缩小 batch retry，并且没有改变该循环的特殊 MTP/回调路径。它不是“任意请求永远一次”的全域声明。

## backend submit 仍不可由静态配置定为一次

固定语义源 `source/llama.cpp-semantic/ggml/src/ggml-backend.cpp` 中，`ggml_backend_sched_graph_compute_async()` 进入 `ggml_backend_sched_compute_splits()`，后者按 `split_id < sched->n_splits` 处理调度 split。

无 callback 的分支中，每个 split 产生一次 `ggml_backend_graph_compute_async(split_backend, &split->graph)`。所以 backend graph submit 数是 `n_splits`，而不是前述的一次 scheduler graph entry。跨 split/backend 的输入就绪还可能引入 event sync、backend sync、event wait 或输入拷贝依赖。

callback 分支可以将一个 split 再按 graph view 提交，并在 view 间执行后端同步。此时 submit 数不能简化为 `n_splits`，必须以 graph-view 级记录为准。

因此，当前 planner 的普通路径 `invocation_count=1` 仅在下列结构证据同时成立时，才可以对应“一条聚合原生提交”：

1. 该 physical invocation 的 runtime scheduler ledger 明确 `split_count=1`；
2. 没有 callback graph-view 分段；
3. 没有未表达的 split 输入拷贝或跨 backend 等待；
4. ledger 与这一次实际图形状、后端配置和执行路径绑定。

现有固定 server 配置并未冻结这些动态图事实。不能通过猜测一个 split 数来倍增前端提交，也不能将 `frontend submit=1` 标为 native count verified。

## 同步边界和当前 completion marker 的对应关系

`llama_synchronize()` 进入 context scheduler synchronize，后者遍历 scheduler 的所有 backend 并调用其 backend synchronize。对已绑定的 CUDA 实现，这最终包含 stream synchronize。因此它是跨已声明 scheduler backend 的主机可见性栅栏，不是 response completion 的零时长标签。

当前规划器的 `_add_physical_invocation_frontend()` 为普通 physical invocation 建模 `invocation_count=1` 的命令构建、主机提交与 GPU command processor。输出路径则在 logits D2H 后放置 `completion_interrupt`，其 metadata 明确标注：

- `completion_interrupt_service_ns=0`；
- `host_wait_service_ns=0`；
- `completion_interrupt_status=partial`；
- `completion_interrupt_latency_not_declared`；
- `opaque_device_fence=true`。

该 marker 不能承担 source 等价同步语义，原因有二：

1. 原生 server 的同步条件是 `ret == 0 && has_output`，并非每一调用均发生；
2. 原生同步在 host 读取输出前，当前 marker 却在 logits D2H 后，因果顺序相反。

应将“最终响应完成”与“server 为读取 output 而执行的 backend synchronize”保持为两个不同语义事件。后者没有独立服务率时可以是**未定价的依赖/资源占用节点**，但不得被表述为零成本且已完整建模。

## 可以立刻从静态源码修复的部分

这部分是依赖、资源归属和覆盖标签修复，不需要时间常数或端到端时延拟合。

建议增加可选契约 `llama_cpp_scheduler_submit_sync_contract`。每个 physical invocation 至少记录：

```text
ubatch_count
split_count
ordered_split_backend_ids
split_edges: event_wait | event_sync | backend_sync | input_copy
callback_view_count
has_output
```

在契约覆盖的 invocation 上：

1. 用 `ubatch_count` 降低 scheduler graph entry；本固定正常路径应记录为 1；
2. 仅按受约束的 `split_count` 和 callback view 数创建前端 submit/command-processor 任务；
3. 按 `split_edges` 连接 split 间输入就绪和完成依赖；
4. 当且仅当 `has_output=true` 时，在最后一条 split/view 完成之后、source 等价 logits D2H/host 输出提取之前，创建 `host_backend_synchronize` fence；
5. 给该节点清晰的 host 资源 ID、`call_count=1` 与 `service_unknown/unpriced` metadata；不填入 `ns` 常数；
6. 将旧的 post-D2H `completion_interrupt` 收窄为最终完成标记，或替换为不冒充原生同步的语义。

如果当前还没有 scheduler ledger，最小的立即改动应只增加上述 source-qualified output sync 依赖节点，并将其标记为 `conditional`、`service_unknown`。没有 ledger 时，前端 submit 仍应报告 `conditional/uncovered`，而非声称 native `submit=1` 正确。

## 仍需要结构采集或独立微基准的部分

### 需要结构采集，不是时延拟合

以下事实不能从 server 固定命令行或静态源码推导为具体数值：

- 当次 `n_splits`；
- 每一 split 的 backend 次序；
- split 间输入拷贝、event wait/event sync/backend sync 的实际边；
- callback graph-view 的次数与边界；
- 融合和多 stream 对 submit/view 拓扑的影响。

所需证据是与固定原生构建和图形状绑定的只读 scheduler ledger 或结构 capture，而不是用目标 LLM wall time、TTFT 或 kernel union 反推。

### 需要独立微基准率

下列服务占用不能由源码给出数值：

- CPU 命令构建与 driver/API enqueue；
- `ggml_backend_synchronize` 的主机调用占用；
- CUDA stream synchronize/API 调用的主机服务；
- 必要时的 event、copy 与 callback-view 调度开销。

这些项目在获得身份绑定、稳定性门槛合格的独立微基准前应保持未定价。不得从完整 LLM 推理时长、延迟数组或误差残差回填常数。

## 最小回归覆盖

测试只验证 IR 节点计数、依赖和 coverage metadata，不断言任何纳秒值：

| Fixture | 必须验证 |
| --- | --- |
| `n_batch=ubatch=64`、`split_count=1`、`has_output=false` | 一个 scheduler graph entry；无 `host_backend_synchronize` |
| 同上但 `has_output=true` | 一个未定价 sync fence；其位于最后 split 后、D2H/host 输出前；不引入时间常数 |
| `split_count=2` | 两个 submit/command-processor 单元；第二个带 ledger 指定的 event 或 blocking 依赖；输出时仅一个最终 sync |
| callback views `>1` | submit 单元按 view 降低，view 间依赖和同步边被保留 |
| 无 scheduler contract | 输出为 `conditional/uncovered`；不得给出“native submit=1 verified” |
| retry/allocator failure | 不把正常成功静态计数推广到该路径；显式排除或由独立 runtime ledger 覆盖 |

## 来源与范围

本轮固定语义证据来自：

- `source/llama.cpp-semantic/tools/server/server-context.cpp`：server batch view、`has_output`、`llama_decode` 成功后的同步条件；
- `source/llama.cpp-semantic/src/llama-context.cpp`：decode→ubatch→`process_ubatch`→scheduler graph entry；
- `source/llama.cpp-semantic/src/llama-kv-cache.cpp`：unified KV `split_simple`；
- `source/llama.cpp-semantic/ggml/src/ggml-backend.cpp`：scheduler split、backend graph submit、跨 split 依赖、scheduler synchronize；
- `source/llama.cpp-annotation-control/src/llama-context.cpp` 与 `ggml-cuda/ggml-cuda.cu`：context synchronize 到 backend/CUDA stream sync 的链路；
- 既有 `native_launch_audit.md`：现有 planner frontend 和 completion marker 的定位。

本机制说明只适用于上述锁定来源和固定配置的正常成功覆盖范围。它修正的是提交计数、同步依赖与资源所有权的声明边界；不产生或暗示目标时延预测。

## 追加核对：每 kernel 启动收费与 host driver 资源不是同一项

当前模型已经在每个已降低的 GPU 计算估计中创建显式 `kernel_launch` phase。`cost_models.py` 的通用 typed-roofline 路径在 `gpu.kernel_launch_ns > 0` 时向 `gpu.launch_resource_id` 添加该 phase；GEMM、elementwise、reduction 等估计都把 `dispatch_name="kernel_launch"` 与同一 `kernel_launch_ns` 传入。`planner.py` 会把每个 phase 降为一个 DAG task；有通过门槛的独立 launch calibration 时，也只替换该显式 `kernel_launch` phase 的 launch/frontend demand，且源码注释明确禁止向每个语义子算子重复收取全局 launch latency。

同时，`_add_physical_invocation_frontend()` 是另一层成本：一次 physical invocation group 的 CPU command build、`host_cohort_submit` 和 GPU command processor。它的 metadata 把 `operator_kernel_launch` 列在 excludes 中。因此普通 `frontend submit=1` 本身并不表示“本图只有一次 CUDA kernel launch”；它是当前聚合的 scheduler/driver 事务计数。

由此可作如下区分：

| 项目 | 当前状态 | 不能做的事 |
| --- | --- | --- |
| 每个分析性 lowered GPU phase 的 `kernel_launch` | 已有独立 launch resource demand；是否等于 native CUDA call 仍取决于 fusion、view/no-op、一个 node 内多次 launch 等真实拓扑 | 不应因为 frontend 只有 1 次而再对所有语义 phase机械叠加一份 host submit 服务 |
| 每 physical invocation group 的 `host_cohort_submit=1` | 只在单 split、无 callback/view 分段及无未表达跨 backend 边时有受限结构解释 | 不应把它当作已覆盖的每 kernel host driver/API 调用资源 |
| host CPU 端实际的 per-kernel launch/API 工作 | 当前没有与 `kernel_launch` 一一绑定的 CPU resource ledger | 不能用 GPU launch `ns` 代替 CPU 占用，也不能把缺失 CPU 占用转写为重复的 GPU kernel launch 计费 |

锁定 CUDA 源支持这个边界：`ggml_backend_cuda_graph_compute()` 遍历 CGraph nodes；view/no-op 与无 COMPUTE 标记的 node 被跳过；其余 node 先尝试 `ggml_cuda_try_fuse()`，融合成功会跳过随后 node，常规 node 才进入 `ggml_cuda_compute_forward()`。所以“CGraph 有 N 个节点”既不等于 N 次原生 kernel launch，也不等于 N 次 host driver submit。反过来，一个常规 forward 路径也可能包含多个 CUDA API launch。没有 API/correlation ledger 时，不能把这部分差异归到任何一个既有时间项。

### 推荐的长合成图观测单位

此前短图探针受 profile 扰动和时钟波动影响，下一次可使用源码可解释的长合成图来降低单次 graph 的固定观测噪声，但不改变已有阈值，也不从目标 LLM 延迟反推参数。

建议固定两个 graph-size fixture，例如 64 node 与 256 node，并把每个样本定义为：

```text
一次 ggml_backend_sched_graph_compute_async(graph)
→ 图内已解析的 CUDA/fusion/API activity
→ 一次 ggml_backend_sched_synchronize(sched)
```

理由是锁定调用链中 `llama_context::graph_compute()` 正是一次 scheduler async graph entry；`ggml_backend_sched_graph_compute_async()` 进入 split 计算；`ggml_backend_sched_synchronize()` 只在 graph 末尾遍历 backend 完成同步。这个单位可把一次观测中的 host/API activity扩大到许多已知 graph node，同时保持一个明确的 per-graph sync 边界，而不会将每个 node 强制同步并污染提交批量化。

fixture 必须记录而非假设以下结构字段：`n_splits`、每 split backend、callback view 数、输入 copy/event 边、CGraph node 总数、view/no-op 跳过数、fusion 尝试/跳过数、实际 CUDA API launch 数以及 API 所属 stream。只有这些字段才能将观测拆分为：

- 已有 per-kernel `kernel_launch` 覆盖了哪些实际 launch unit；
- host driver/API 工作是否需新增独立 CPU resource；
- `frontend submit=1` 是否仍可作为图级聚合事务，还是要按 split/view 而非按 kernel 分解。

长图适合比较同一、冻结结构下 64 与 256 node 的增量，检查 API 调用数和 CPU 调用工作是否随已解析 launch unit 增长。它不应直接产生 `kernel_launch_ns`、`submission_ns` 或 host synchronize 常数；任何候选率仍须满足既有独立微基准和稳定性要求。
