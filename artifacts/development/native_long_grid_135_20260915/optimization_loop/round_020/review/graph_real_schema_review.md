# R20 Graph gap collector：R18 真实 Nsight SQLite 兼容性复核

复核日期：2026-09-16。此文只读取 R18 已有产物和源码；未编译、未加载 DLL、未访问 GPU、未运行测量。

## 复核样本与结论

实际样本为 R18 `collection/runs/scale_f32_e1024_g1/pair_01/export/trace.sqlite`。其真实表包含 `NVTX_EVENTS`、`CUPTI_ACTIVITY_KIND_RUNTIME`、`CUPTI_ACTIVITY_KIND_KERNEL` 与 `StringIds`；该样本**没有** `CUPTI_ACTIVITY_KIND_DRIVER`。

R20 的 `graph_gap_collection/extract.py` 是 synthetic fixture 可用的 toy extractor，不能直接用于实测。它需要由 root 接管的成熟 R18 提取路径来替换/适配；不得把 toy 输出解释为 GPU dispatch、API、clock 或质量证据。

## 真实 schema 与 R20 toy extractor 的不兼容点

| 范围 | R18 真实 schema/成熟行为 | R20 toy 行为 | 后果与最小修复 |
| --- | --- | --- | --- |
| Runtime/driver API 列 | Runtime 是 `globalTid`、`correlationId`、`nameId`、`returnValue`；driver 表可缺失，存在时也按 API 表处理。 | `_rows()` 假定每个表均有 `name` 或 `text`、`globalPid`。 | 对 runtime/driver 分别定义 required columns；driver 必须可选。使用 `globalTid` 与 `nameId`，不能用 `globalPid/name`。 |
| NVTX 文本 | `NVTX_EVENTS` 同时有 `text`、`textId`；R18 由 `StringIds.id/value` 解析为 `resolved_text`。 | 只选 `text`，不解析 `textId`。 | 复制 `read_trace()` 的 `StringIds` 解析；标签匹配必须基于 `resolved_text`。 |
| Kernel 名称 | Kernel 的 `demangledName`、`shortName` 是 StringIds 引用。 | toy 只计 interval，不读取或解析 kernel 名称/几何。 | 复用 `read_trace()` 名称解析和 `probe_adapter.kernel_role()`、`validate_kernel_chain()`。 |
| PID/TID 命名空间 | NVTX 和 runtime 以 `globalTid` 关联。R18 用 `PROCESS_MASK` 从 `globalTid` 提取 trace process namespace，并同时核验原始 native PID、caller thread ID。 | toy 用 `globalPid == pid` 或低 24 位相等。 | 复用 `correlate()` 的 native PID/TID 绑定逻辑；profile 必须验证 target PID，而不是 Nsight wrapper PID。 |
| API→kernel 归因 | 普通 launch 通过同 process + `correlationId` 关联 runtime/driver API 和 kernel；CUDA Graph child kernel 可能只落在 NVTX interval，需要显式标为 graph-child correlation 状态。 | toy 只用 kernel 与 NVTX 区间重叠。 | 复用 `correlate()` 中 ordinary-launch / graph-child 两条分支，保留 unresolved 诊断，不能把 overlap 当作依赖证明。 |
| 计时域 | GPU、runtime、driver、host-submit/sync 的分区均在 Nsight ns 时钟中完成；R18 `mutually_exclusive_partition()` 要求区间分区闭合。QPC 只保留绝对主机边界。 | toy 只检查 QPC 局部顺序，未建立 NVTX/API/kernel 同域分区。 | 复用 `mutually_exclusive_partition()`；禁止把 QPC 与 Nsight 时间相减或相加。 |
| 实际 dispatch 数 | R18 从 trace 中得到实际 kernel/API records，保留 capture/replay 未解析状态。 | toy 仅要求每个 label 至少一个 interval overlap。 | 输出每 call 的实际 kernel/API records、correlation method 与 unresolved diagnostics；不得从 N=8 计划节点填充。 |
| 原始 raw / process 绑定 | R18 `stage_complete()`、`probe_adapter.process_origin()` 检查 freeze、launch receipt、实际 argv、pair、输出路径、child PID、父启动 QPC 区间和 artifact hash。 | toy 只读 raw JSONL，不绑定 launch receipt、build/freeze、实际 child/target PID 或 artifact hashes。 | 复用 `stage_complete()` 与 `process_origin()`；R20 adapter 只变 arm/pair/path 命名。 |
| footer 与 numeric audit | R18 `numeric_document()` 校验 header/setup/footer、36 call、raw source hash、数学签名和 audit。 | toy 仅检查 36 labels 与 numeric block 名称，不读取 footer counts/quality 状态。 | adapter 必须核验 footer `graph_calls`、`numeric_blocks`、`checked_values`、mismatch/nonfinite、module stability、raw receipt hash。 |
| control / buffered 语义 | control 是逐 call final 加首尾 intermediate；buffered 是 first/post_warmup/post_formal 的 all-stage partial coverage。 | toy 区分名称，但对 bitwise failure 以前直接异常，且没有把 footer/coverage 与完整分母连结。 | 保留 36-call 分母，数值失败标记不合格但不能删除 stage/call；buffered 永远 `per_call_validated=false`。 |
| hard termination | R20 修订后 buffered 在 first timed call 前 durable `buffered_started`；硬终止只承诺 launch/freeze/trace 证据。 | toy 没有 launch receipt 或 buffered_started 处理。 | adapter 必须接受“无 flush call records”的 hard-stop 状态并保留为 incomplete，而不能假定 buffer 可恢复。 |
| clock/telemetry | R18 `verify_clock_receipt()`、`clock_readback_gate()` 使用外部 clock receipt 与每个 formal 的完整 telemetry bracket；finally reset 由控制器生命周期保存。 | toy 没有 clock receipt、telemetry、30 formal readback 或 reset evidence。 | 复用两项 clock 函数并接 R20 root 的控制器/telemetry artifact；不能由 collector 自行锁频。 |
| matrix/paired quality | R18 `extract_pair()` 保留 direct/profile/export 各 stage status，之后才计算 paired denominator、trace chain、warnings 与质量。 | toy 是单 raw / 单 sqlite 函数，没有 12 进程与 6 export 的 pair/arm 汇总。 | R20 最小 adapter 需要 `arm × pair` stage mapping，保留每个失败 status，之后才计算每臂三对质量门。 |

## 可直接复用的 R18 函数

以下函数适合作为 R20 的基础，避免再实现 SQLite 和 evidence binding：

1. `round_018/collection/extract.py::read_trace`：只读 SQLite、schema freeze、artifact hash 前后核验、StringIds 解析、API/kernel 名称解析。
2. `extract.py::correlate`：NVTX→API→kernel correlation、PID/TID namespace、普通 launch 与 graph-child 分支、36-call 完整链和 Nsight 同域 partition。
3. `extract.py::mutually_exclusive_partition`：每 call GPU/API/host/unattributed 分区闭合检查。
4. `extract.py::numeric_document`：raw header/setup/footer、numeric audit、raw hash 和 source binding。
5. `extract.py::stage_complete`：freeze、launch receipt、artifact hash、argv、pair、PID、QPC 生命周期和 export-input binding。
6. `extract.py::extract_pair`：先保留 direct/profile/export stage statuses，再消费完整 raw/trace 证据。
7. `round_018/collection/probe_adapter.py::process_origin`、`expected_labels`、`kernel_role`、`validate_kernel_chain`：真实 probe identity 与 graph-chain 语义。
8. `round_018/collection/common.py::verify_clock_receipt`、`clock_readback_gate`、`union_ns`、`distribution`：clock、trace 和质量汇总基础。
9. `round_018/collection/quality.py::quality`：保留门限应用位置；R20 不新增或拟合任何常数。

## R20 最小 adapter 需求

不应复制 R18 整个 collector。最小安全 adapter 应：

1. 使用 R18 `read_trace/correlate/stage_complete/numeric_document` 的实现，参数化 R20 的 probe root、pilot config、`control|buffered`、pair 1..3 和 R20 launch receipt 名称。
2. 将 expected labels 固定为 `scale_f32_e262144_g8` 的 36 个 R20 NVTX labels；保持 R18 的实际 PID/TID/correlation 逻辑。
3. 将 numeric validator 拆为 control 与 buffered 两种 coverage policy；失败仍进入 36-call 分母，带 `eligible=false`，不删记录。
4. 以 root 实际产出的 clock receipt、formal telemetry 和 finally reset artifact 调用 R18 clock gates；缺失时 stage/pair 仅为 incomplete，不做补算。
5. profile export 只能消费其绑定的实际 profile target process；direct 不要求 trace。每个 arm/pair 的 direct、profile、export 三个 stage status 都必须保存。
6. 输出诊断结果，不输出 calibration、LLM-time fit、成本系数或“节点数等于实际 kernel 数”的结论。

R20 toy extractor 可继续作为 synthetic fixture/接口单测，不得进入实测执行路径。