# R21 non-Flash KV physical-view 修复记录（已验证）

日期：2026-09-16。范围：源码与构建链复核、planner 物理形状、固定预测器静态 opt-in 和纯内存聚焦测试。未读取目标 native 延迟或实际误差；未启动 native、GPU 或目标 LLM；未进行大规模模拟。

## 已实现的机制

- 新静态 opt-in：`--nonflash-kv-view-source-contract <round_021/nonflash_kv_view_source_contract.json>`，默认关闭。不改变旧冻结。仅初次 freeze 接受，resume 禁止改开关。
- 每个 cell 保存来源 contract 和原生配置；worker 将其安装为 `workload.metadata.llama_cpp_nonflash_kv_view`。逻辑 per-slot `context=2048` 与原生统一池 `context=parallel*2048` 分别保留。
- 对 source/build/runtime/config 合格、普通 f16 unified 非 Flash 路径，物理 attention K 下界是 `min(native_cache_cells, pad(max_current_lane_retained_prefix,256))`，至少一个 256 padding block。按最长当前保留序列证明占用下界，不相加请求上下文。
- 64 行 first prefill 的逻辑均值仍为33，物理 K 下界256；A decode@128 加 B prefill 0..62 的逻辑均值仍为34，物理 K 下界256。A decode@1024 混合新预填充时逻辑均值48，物理 K 下界1280，绝不将长前缀截成256。
- 每条 lane 的 request、position、causal context、requires_logits、KV append/materialized 信息保留；QK/PV/softmax 使用矩形物理长度，普通 GEMM 和物理 batch 不拆行。linear-attention 的逻辑 context 参数保留。
- KV 物理访问由现有 attention kernel 计费；本地 `kv_read` 仍是 included-in-attention 的零需求生命周期节点，不额外添加64个读或计算 kernel。
- 动态 attention replay 接收当前物理长度/读长度，现有完整 attention 叶缓存以实际物理 context 为 shape；逻辑 cohort 缓存仍区分逐 item context。

## 来源链和证明边界

`verify_llama_runtime_source_binding` 重新验证 R4 保存的 build binding。KV cache、graph、model 的原始编译对象由 annotation llama.dll 链接并由 native-thread-control 继承。`llama-context.cpp` 是例外：annotation 实际重新编译 overlay 对象。因此新增校验使用 annotation 的 source manifest、实际编译 argv、对象输出 hash、link response 和 module hash，不能谎称沿用原始 context 对象。

锁定源码没有名为 `get_padding` 的函数。真实规则是：普通/混合 KV cache 构造传 `n_pad=1`；`llama_kv_cache::get_n_kv` 内部取 `max(n_pad,256u)`。context 构造将总 n_ctx 向256对齐；unified 时 n_ctx_seq 等于总 n_ctx。cache 为唯一 stream 分配 kv_size cells。`apply_ubatch` 之后才调用 `get_n_kv`。

`llama-graph.cpp` 建立 F32 `[n_kv,ubatch.n_tokens,1,1]` mask，并执行 KQ、masked softmax、PV。KV mask 先用 seq_id 过滤非本序列 cells，再根据位置屏蔽未来 tokens；逻辑 mask 不缩小矩阵宽度。

每个无滑窗、无丢弃、未发生 context shift 的当前序列需要至少其保留前缀数量的不同 cells，因此任意 allocator 的 used_max_p1 至少是该数。多个请求可共享 prefix cells，不能把这些长度求和。即使新 cohort，也不能据此宣称完整 pool 等价：warmup、idle slot 内容、holes、历史复用和共享前缀 union 都可能进一步抬高实际 high-water。合格结果逐 group 标记 `extent_completeness=lower_bound` 和该未知原因。GGUF 声明 sliding window、未知架构、context 越过每槽容量、非普通 materialization、MTP、配置/源码绑定不符等返回明确 uncovered。

## 聚焦验证结果

2026-09-16 执行：

```text
E:/anaconda/python.exe -m pytest tests/test_nonflash_kv_view.py tests/test_nonflash_kv_view_adapter.py tests/test_final_layer_output_selection_planner.py tests/test_predict_stable_native_dataset.py -q --disable-warnings --maxfail=2
168 passed in 44.17s
```

新增29条测试和相邻回归共同通过，无跳过。验证覆盖：

- 固定 b=ub=64、fa=off、kvu、f16 与并发1/2/4。
- first64 保留逻辑33与64条lane，mixed64保留逻辑34、A一条logits、B63条不取logits；每层仍一个QK、一个PV，投影任务的资源需求保持原样。
- 直接拦截传入估算器的类型化 workload，证明 QK 的 N=256、PV 的 K=256、softmax score elements=64*256，并非只改 audit 标签。
- 256边界与上下文增长到2048；mixed长前缀1025导致物理下界1280；不求和不同请求的前缀，也不把总容量视为扫描长度。
- 本地 KV-read 生命周期节点无额外 demands，物理K/V字节由现有attention kernel计费。first prefill原来的no-prior-KV节点变为物理view的零需求说明节点，未增加kernel或任务总数。
- 真正的动态回放：使用已有 Qwen attention execution descriptor（包含其实际 QK scale），在同一个 CompilationContext 中从逻辑256跨到257，`_compile_parallel_iteration`调用数保持1；物理长度从256变512。回放的完整 ScheduleIR 与独立重新编译结果相等，包含kernel launch/GEMM维度、query geometry、token_shape和K/V字节。
- 初版极小fixture缺少QK-scale执行描述，既有回放保护正确拒绝capture。没有削弱或删除“必须真正复用”的断言；改用已有完整执行描述后，又发现并修正了回放操作层的旧几何审计字段。该审计刷新只在nonflash opt-in下启用。
- 原始GGUF导入架构名 llama/qwen2 和 qwen3_5_hybrid_transformer均覆盖；sliding-window声明、非合格绑定、Flash/non-unified/generic、配置不符和越过slot容量均显式uncovered。
- 初始freeze保留contract，worker从frozen inputs应用；baseline为None；resume不能变更CLI contract。源码/对象/link/module历史链的重新派生测试通过，覆盖新增加的hybrid转接层和annotation context对象。

## 根任务接线与冻结说明

```python
from tools import predict_stable_native_dataset as adapter
binding = adapter.verified_host_offload_source_contract(
    round_004 / "runtime_source_binding_structural_audit.json", rows, project_root)
contract = adapter.derive_nonflash_kv_view_contract(binding, project_root)
# 本次已经生成下面的JSON，无需再次写入。
freeze = adapter.freeze_selection(...,
    host_offload_source_contract_path=round_004 / "runtime_source_binding_structural_audit.json",
    nonflash_kv_view_source_contract_path=round_021 / "nonflash_kv_view_source_contract.json")
```

静态合同位置：`round_021/nonflash_kv_view_source_contract.json`。
CLI：`--nonflash-kv-view-source-contract <该JSON>`。省略即关闭，可用同一源码做pure/current/physical三路；无须改写source或旧freeze。

`verified_nonflash_kv_view_contract`逐cell验证选中runtime，固定保存native_context_tokens、simulator_slot_context_tokens、parallel、batch/ubatch、Flash/unified/cache types。`freeze.nonflash_kv_view.evidence_refs`保留合同本身、继承的runtime证据和新增源/构建/link证据，`verify_freeze_references`在恢复时重核这些引用。最终code closure涉及planner、predictor和两份测试；没有更改typed IR/serving状态机。根任务开始冻结后，本实现不再修改这些代码。

本报告仅证明物理形状与代码行为修正；尚未执行任何目标LLM预测或误差对照，不能推断具体延迟/误差改善。
