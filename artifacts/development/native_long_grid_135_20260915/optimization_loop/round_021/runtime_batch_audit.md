# R21 固定 native batch 形状审计（复核版）

- 日期：2026-09-16
- 范围：固定 131 格的静态协议、预测器/规划器、锁定 llama.cpp 父源码与构建链。
- 限制：未读取 native actual、目标延迟数组或预测误差；未运行 llama-server、GPU 或目标 LLM；未修改核心代码。

## 复核后的结论

固定预测器的实际静态路径**确实触发**一个 attention 物理 shape 缺口，但原因不是“ragged 行应该按平均三角形计算”。锁定 llama.cpp 在本协议的 `-fa off -kvu -b 64 -ub 64` 配置下，对 unified KV 的普通 attention 使用**至少 256 行 padded KV 视图**，并执行非 Flash 的 `K × Q` / softmax / `V × P` 矩阵链；每行的 `pos`/`seq_id` 只写入 mask，不会把矩阵计算缩小为逻辑可见 token 数。

当前固定预测器没有启用 KV-scan / Q4 materialization 覆盖：它把同一个物理 64 行 batch 传为 `token_batch=64, context_tokens=34`（或其他按 lane context ceiling-mean 得到的标量）。因此其 attention score geometry 是 `64 × 34`，而锁定原生源码在相同条件下的可证明下界是 `64 × 256`。该差异在 `parallel=1/2/4` 都可发生；`parallel=2/4` 的 mixed batch 给出最直观反例。

这是一项高置信的**执行/形状语义缺口**，不是对任何数值误差的直接归因，也不应以目标延迟常数或新 target LLM 运行来处理。

## 静态运行时配置已确认

`native_protocol.json` 固定每槽 `kv_unified_per_slot=2048`，并发取 `1/2/4`。原生 runner 对每一 cell 强制 `ctx=parallel*2048`，并启动：

```text
-np parallel -b batch -ub ubatch -fa off -kvu --kv-unified-per-slot 2048 -cb
```

其中 protocol job 未覆盖 `batch/ubatch/flash_attention` 时，runner defaults 分别为 `64/64/False`。

- [native_protocol.json](../../native_protocol.json) 第 24–40、65–69 行。
- [native_repeatability_experiment.py](../../../../../../tools/native_repeatability_experiment.py) 第 47–51、194–201、420–430 行。
- [predict_stable_native_dataset.py](../../../../../../tools/predict_stable_native_dataset.py) 第 1066–1079 行也拒绝非 `flash_attn=False` 的固定预测输入。

锁定构建链来自 `source/llama.cpp-native-thread-control/evidence/build_receipt.json` 的 verified annotation parent；annotation receipt 的 base 是 `source/llama.cpp-semantic/build-semantic-direct`。以下源码结论使用 `source/llama.cpp-semantic` 的审计时 SHA-256 `1C8D964CE27AF8E3D32964C75234E1D4C11B7E3E8D49FDEE4FEBE044B5F36B8D`（`tools/server/server-context.cpp`），不以工作区中未提交的 `llama.cpp-semantic-patches` server patch 作为依据。

## 可复现的固定配置反例

使用协议允许的 `parallel=2, prompt=128, output>=2, batch=ubatch=64`，且采用预测器的 `stable_admission` 槽顺序：

1. A prefill：位置 `0..63`。
2. A prefill：位置 `64..127`，最后一行请求 logits。
3. A decode 一行（位置 128、请求 logits）与 B prefill 63 行（位置 `0..62`）进入一个 64 行 native server batch。

server 先收集 generating 行，再在连续 batching 下填 prompt，直到 `n_batch`；batch 行由 `common_batch_add(token, pos, {slot_id}, output)` 传入 context。

- [server-context.cpp](../../../../../../source/llama.cpp-semantic/tools/server/server-context.cpp) 第 2996–3017、3232–3245、3662–3739 行。

我用现有的纯内存小 fixture（非 native、非 GPU、非 target LLM）构造该 cohort，得到当前 planner 的实际元数据：

```json
{
  "items": [["A", "decode", 1, 128, 1], ["B", "prefill", 63, 0, 0]],
  "group_context_tokens": 34,
  "group_kv_read_tokens": 128,
  "group_kv_scan_tokens": 0,
  "group_q4_view_tokens": 0,
  "flash_attention": false
}
```

这里 `34 = ceil((129 + sum(1..63))/64)`。该 fixture 只验证当前 in-process scheduler/plan 元数据；不包含或推断任何 native timing。

## 为什么 KV scan 没有覆盖该反例

planner 确实在 `group.kv_scan_tokens` 存在时优先传递它：

```python
attention_context_tokens = group.kv_scan_tokens or group.context_tokens
```

但固定预测路径没有此值。`_kv_scan_enabled` 要求 `flash_attention=True`，而固定协议与预测器都固定 `flash_attention=False`。因此既没有 `llama_cpp_kv_scan_tokens`，也没有 Q4 materialization view；最终会走 `group.context_tokens=34`。

- [serving.py](../../../../../../src/heterollm_sim/serving.py) 第 4662–4703、13469–13506 行。
- [planner.py](../../../../../../src/heterollm_sim/planner.py) 第 20365–20421 行。

## 原生物理 attention shape

锁定源码给出了不依赖 timing 的下界：

1. unified KV 时，attention mask 的 stream 数为 1，形状为 `[n_kv, ubatch.n_tokens, 1, 1]`；本反例第二维为 64。
2. `llama_kv_cache::get_n_kv()` 使用 `max(n_pad, 256)` 作为最小 padding；因此 `n_kv >= 256`，即使真实已使用 KV 行更少。
3. KV context 把这个 `n_kv` 传给 K/V view。
4. flash 关闭时，attention 走 `ggml_mul_mat(k, q)`，随后在这个全矩阵上做 masked softmax，再做 `ggml_mul_mat(v, kq)`。

- [llama-graph.cpp](../../../../../../source/llama.cpp-semantic/src/llama-graph.cpp) 第 29–45、2604–2710、2818–2825 行。
- [llama-kv-cache.cpp](../../../../../../source/llama.cpp-semantic/src/llama-kv-cache.cpp) 第 1250–1264、2725–2758 行。

所以原生普通 attention 的 score matrix 物理下界是 `n_heads × 64 × 256`。mask 中的 A/B sequence 与 causal position 决定数值可见性，但不把非 Flash `ggml_mul_mat` 的 K dimension 变为 34，也不把该矩阵缩成逻辑三角形。

## 当前 planner 的实际压缩点

planner 会保留 lane 身份和每 lane context：A 为 129，B 为 `1..63`；也会保留准确 logits lane。问题发生在 group 层：

- `_ServingInvocationGroup.context_tokens` 计算 lane context 的 ceiling mean；
- mixed group 将全部 64 lane 作为一个 `explicit_mixed_phase_physical_batch`；
- `attention_context_tokens` 采用 scan 值或这个 mean；
- `FusedAttentionWorkload` 与非融合 QK/softmax/PV 路径均按 `token_batch * context_tokens` 创建 score elements。

- [planner.py](../../../../../../src/heterollm_sim/planner.py) 第 19629–19643、19806–19884、20085–20094、20365–20421、15341–15416 行。

因此当前 lane metadata 本身尚在，但没有进入 native 需要的 padded `n_kv` 物理 K dimension。`kv_read_tokens=128` 是另一条持久 KV 流量统计，不能替代 QK/PV score matrix 的 `n_kv=256`。

## 与 1/2/4 并发语义的对应

| 项目 | 结论 |
|---|---|
| `n_batch` / `n_ubatch` | 固定协议均为 64，scheduler 的 outer cohort 与 physical ubatch 行数映射正确。 |
| `n_parallel` | 正确投影为 `max_num_seqs`；并发 1/2/4 的新鲜 cohort 槽上限正确。 |
| engine request begin | source 的首次 prompt-batch 填充点与 simulator `_record_llama_engine_start()` 一致。 |
| logits 输出行 | final prompt row 与 decode row 均被 lane `requires_logits` 保留；不是完整 prompt lm-head 的问题。 |
| logical per-slot context | 2048 每槽与 `ctx=parallel*2048` 已校验。 |
| physical unified KV / attention K width | 未忠实建模：在 `-fa off` 路径遗漏了 source 至少 256 的 padded `n_kv`。 |

`parallel=1` 没有跨请求 decode+prefill 混合，但同样满足 source 的 `n_kv>=256` padding；例如 64 行 first prefill 的 planner mean context 约为 33，而 source 的 non-Flash K dimension 下界仍为 256。因此该 gap 的适用面不限于并发 2/4。

## 最小机制修复与回归

不引入每模型常数，不读取目标延迟。应在 source-bound `-fa off + kv_unified` 路径把 native K dimension 作为独立物理 shape：

1. 从已绑定 runtime/source contract 取得 `n_kv = min(cache_cells, max(n_pad, 256, pad(used_max_p1,n_pad)))` 的可证明下界；固定协议可先安全建模最小值 256，并将高于下界的动态 allocator extent保留为 conditional。
2. 将该值传给 attention QK/PV 和 mask/softmax 工作量，独立于 lane context mean 与 `kv_read_tokens`。
3. 保留 lane `pos`/`seq_id`/logits 作为 mask 与输出选择元数据；它们不应被误当成物理 K matrix 宽度。

新增纯单元回归：构造 A `decode@128` + B `prefill@0..62` 的 64 行 cohort，`flash_attention=False`、unified KV，断言最终 attention workload 的 physical K width 至少为 256，且 `group_kv_scan_tokens=0` 时不得退回到 `context_tokens=34` 作为唯一 QK/PV width。

## KV allocation 说明

预测器把每槽 2048 写入 `serving_runtime.kv_slot_context_tokens`，但 unified physical pool 的完整分配/竞争仍被自身标为 conditional。该事实不削弱上述结论：`get_n_kv()` 的 256 minimum padding 已由锁定源码决定，不依赖完整物理 pool 等价，也不需要 native timing。

## 新鲜 cohort、warmup 与 unified KV 高水位：可严格推出的范围

上文的 `n_kv>=256` 是对任一相关 ubatch 都成立的源码下界；不能从“当前 64 个 lane 的 context 之和”推出 exact `n_kv`。锁定源码的 exact 规则是：

```text
n_kv = min(cells.size(), max(max(n_pad, 256), PAD(cells.used_max_p1(), max(n_pad,256))))
```

其中 unified KV 只有一个 stream，所有 sequence id 映射到该同一 cell array；`find_slot()` 从可变 head 开始寻找空 cell、可环绕并留下洞，`seq_rm()` 释放 cell 后只将 head 前移到最早释放处，并不把后部 occupied cell 压缩。因此 `used_max_p1()` 是 allocator 的物理 high-water，不是当前请求的 token 数、逻辑 position，也不一定等于活动 lane context 之和。

- [llama-context.cpp](../../../../../../source/llama.cpp-semantic/src/llama-context.cpp) 第 255–258、298–324 行：本协议 `n_batch=n_ubatch=64`，`kv_unified=true`，configured context 会按 256 对齐。
- [llama-kv-cache.cpp](../../../../../../source/llama.cpp-semantic/src/llama-kv-cache.cpp) 第 65–155、898–1094、1250–1264 行：unified 时 `n_stream=1`、所有 seq 共用 cell array、ring/head allocation 与 exact `n_kv` high-water 公式。

固定 runner 对每个 block 只启动一个 server 进程；同一进程内先执行 warmup batches，再执行 measured batches，HTTP payload 固定 `cache_prompt=false`，并且 protocol 的 `cache_ram_mib=0` 传为 `--cache-ram 0`。

- [native_repeatability_experiment.py](../../../../../../tools/native_repeatability_experiment.py) 第 437–471 行。

这并不意味着每个 repeat 从完全空的 unified pool 开始。普通 completion `release()` 只令 slot idle 并 `reset()`，不会调用 `prompt_clear()`；`reset()` 本身也不清 KV。对下一次分配到某 slot 的 `cache_prompt=false` 请求，source 在该 slot 首次 STARTED prompt path 中让 `n_past` 保持 0，执行 `prompt.tokens.keep_first(0)`，再 `seq_rm(slot.id, 0, -1)`。因此旧 KV 会在**该 slot 第一次重新开始 prompt 时**被清除，但其他仍 idle、尚未进入 STARTED 的 slot 仍可保留上一 warmup/repeat 的 KV，并参与 shared-stream `used_max_p1()`。

- [server-context.cpp](../../../../../../source/llama.cpp-semantic/tools/server/server-context.cpp) 第 472–515、652–674、3355–3364、3547–3582 行。
- [llama-kv-cache.cpp](../../../../../../source/llama.cpp-semantic/src/llama-kv-cache.cpp) 第 382–448 行：`seq_rm()` 释放 cell 并只调整 search head。

所以 source-bound 修复应严格分层：

1. **可安全用于所有固定 cell 的物理下界**：`n_kv >= 256`，且 fixed non-Flash ubatch 的 QK/PV 不能低于 `64 × 256` 的 K width。
2. **仅在 fresh, first-process batch、且 allocator head/occupied set 已由可验证 runtime snapshot 绑定时可精确化的量**：`used_max_p1()` 对应的 padded high-water。
3. **不能仅凭协议或当前 simulator lane 推出的量**：warmup 后/同 server repeated request 的 exact high-water、fragmentation、wraparound 与其它 idle slot 的残留占用。没有这种 state evidence 时，不得用 `sum(current lane contexts)`、`max logical context` 或完整 pool capacity 伪装成 exact native `n_kv`。

这不会否定 `n_kv>=256` 的已证实形状下界；它只限制任何把该下界提升为 warmup/repeat exact physical width 的声明。

源码哈希（本审计时）：`llama-context.cpp=6a9d17f10caa586c1bf299e8bb217bcd14c1d1f62485020cd41e78c3b70b022b`，`llama-kv-cache.cpp=e4d2aa977c8aa048ddd79c878683e94909f5f24ac109b0fcd1c0b59ef70f3899`，`llama-graph.cpp=a6a8241c2d149961801d0fdeaa68f1bb176297b4d5156b6138db0b3d169abb04`，`server-context.cpp=1c8d964ce27af8e3d32964c75234e1d4c11b7e3e8d49fdee4febe044b5f36b8d`。

## WIP 独立源码审阅：`nonflash_kv_view_fix`（2026-09-16）

审阅对象为当前工作树中的 `planner.py`、`predict_stable_native_dataset.py` 与新增纯单元测试；未读取 native/LLM timing，未改动核心代码。

### 已通过的源码审阅点

1. **lower/upper 声明边界正确。** 新 opt-in 只在 runtime binding verified、`-fa off`、`kv_unified=true`、F16 K/V、ordinary prefill/decode、scheduler 与 frozen config 一致时启用。`physical_k_tokens` 使用 `max(largest_current_sequence_retained_context, 256)` 的 256 对齐值，并用 native total context 夹住；metadata 标为 `extent_completeness=lower_bound`，同时写明 allocator holes、inactive slot、cache/shared-prefix union unknown。它没有将 total context 当作 exact high-water。

2. **unified/nonunified 没有混用。** config gate 同时要求 `kv_unified`，并绑定 `native_context_tokens = simulator_slot_context_tokens * parallel`；不满足时以 uncovered reason 退出。non-unified 不会错误复用该 single-stream lower bound。

3. **QK/PV/softmax 的物理维度已实际进入工作负载。** 新代码将 `physical_k_tokens` 传给 full-attention layer 的 `context_tokens`，并将相同宽度传入 `kv_read_tokens`。在 non-Flash 路径，QK GEMM 的 `N`、PV GEMM 的 `K`、softmax reduction/normalize score elements 都由这一宽度驱动；不是只写 audit metadata。新增纯测试覆盖 64×256 与跨 256 边界的 64×1280 case。

4. **KV 读流量未见明显重复记账。** `_kv_tensor_storage_metadata_bytes()` 明确返回单个 K 或 V operand 的 bytes；non-Flash QK 与 PV 分别使用一次，正好对应 source 的 KQ 与 PV 两个矩阵操作。`_add_kv_access()` 对本地读取标记 `included_in_attention_kernel`，不另加 resource demand；新增测试也断言该 local bookkeeping event 没有 demand。这个结论仅限当前 source-bound local GPU path。

5. **coverage/validity 不过度承诺。** static report 保持 `conditional`，group metadata 同时记录 applied lower bound 与 unknown high-water 原因；sliding/nonordinary retained prefix、MTP、线性 attention、config mismatch 都 fail closed。`gguf.architecture` 不在 adapter allowlist 时会走 uncovered，而不会误启用；这是保守覆盖收缩，不是语义越权。

### 阻断项：动态任务重放回归未通过

本轮运行的纯测试（无 native/GPU/目标 LLM）：

```text
E:\anaconda\python.exe -m pytest -q tests/test_nonflash_kv_view.py tests/test_nonflash_kv_view_adapter.py
结果：28 passed, 1 failed
```

失败用例：`tests/test_nonflash_kv_view.py::test_dynamic_replay_recomputes_physical_width_and_reads_at_padding_boundary`。

该用例先编译 `decode context=255`（physical K=256），再在同一 `CompilationContext` 编译 `decode context=256`（physical K=512）。fixture 已确认 group audit 从 256 正确变为 512，但 `_compile_parallel_iteration` 调用数从 1 变为 2，未按测试承诺走 dynamic replay。

```text
cold: kind=decode, batching=serial_stateful_position,
      logical context=256, physical_k=256
warm: kind=decode, batching=serial_stateful_position,
      logical context=257, physical_k=512
actual _compile_parallel_iteration calls: 2; expected after cached replay: 1
```

这不是 native timing 问题，而是 WIP 自己声明的 replay contract 没有成立。合并前应二选一：

- 修复 serving invocation segment template/replay，使 256→512 的动态 physical K width 通过 replay payload 重新生成 shape 与 KV read；或
- 若此类边界本来必须重新编译，明确缩小/修改 replay contract、缓存 key 与测试预期，并证明不复用旧 256-shape template。

不应直接删除或放宽该失败断言来得到绿灯。

### 审阅参考

- WIP lower-bound gate：[planner.py](../../../../../../src/heterollm_sim/planner.py) 第 19620–19692。
- WIP physical propagation：[planner.py](../../../../../../src/heterollm_sim/planner.py) 第 20453–20512、14239–14293。
- non-Flash K/V 单 operand accounting：[planner.py](../../../../../../src/heterollm_sim/planner.py) 第 11754–11771、11818–11859、15468–15524。
- frozen source contract：[predict_stable_native_dataset.py](../../../../../../tools/predict_stable_native_dataset.py) 第 770–845、848–859、1215–1223。
- failing test：[test_nonflash_kv_view.py](../../../../../../tests/test_nonflash_kv_view.py) 第 145–159。
