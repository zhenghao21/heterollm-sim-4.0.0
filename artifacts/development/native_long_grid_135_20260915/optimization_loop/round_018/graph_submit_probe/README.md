# R18 合成图提交边界探针

**当前状态：source_prepared_not_compiled_not_measured。** 本目录已准备源码、协议、独立 CPU 数学参考、编译冻结与单进程运行入口；没有编译、启动 GPU、生成原生时延、输出校准系数或修改核心成本模型。测量仍由父任务与采集子代理协调。

## 要解决的机制问题

R16 的设备/主机间隔只能提出 H4 假设：现有主机图提交成本可能缺少实际控制流、图重用检查和驱动提交工作。不能把包络与 kernel 并集的差值全部归为 launch，也不能将带重度 profiling 的时间强加为 device 下界。

源协议核对发现：固定 long-grid native 的 `GGML_CUDA_DISABLE_GRAPHS` **缺省**，旧 R16/r3 单算子探针固定为 `1`。因此本探针必须保持 native 的 CUDA Graph 缺省策略。锁定控制源码中首轮直接执行，连续稳定属性后进入捕获与重放；实际发生哪种路径以 trace 为准，不能只凭环境值声明 Graph 已启用。

使用独立变化的图节点数与 shape，检查 `ggml_backend_graph_compute_async` 主机时间是否随节点数变化、变化是否发生在首次/捕获/稳定重放，以及最终同步中有多少等待能由设备活动解释。源码和原始事件先给出证据，不自动拟合当前固定的 LLM 答案。

## 最小实验

| F32 元素数 | 图节点数 | 对照方式 |
| --- | --- | --- |
| 1024 | 1、8、32 | 每组合 3 个独立进程对 |
| 262144 | 1、8、32 | 每组合 3 个独立进程对 |

总计 6 组合、18 对、36 进程。每进程首次 1 次、预热 5 次、正式 30 次，合计 1296 次整图调用；同一配对的 direct/profile 使用相同应用二进制、配置、输入、循环、NVTX、数值校验，应用 argv 仅输出路径不同。配对顺序按 `(config_index + pair_index)` 奇偶交替，三次过程配对由采集器顺序执行，禁止并行占用 GPU。

初版没有 probe CUDA event，没有 graph-disabled 对照，不扩成四倍矩阵。profile 使用 `--cuda-graph-trace=node`：本机 Nsight 帮助明确说明默认 graph 粒度不收集子节点活动；node 粒度可有明显观察扰动。不能省掉 node trace 后用计划节点数冒充实际 kernel 数，也不能把配对差异直接称为纯 CUPTI 开销。

## 图、数值和缓存语义

每个节点都是非 inplace `ggml_scale`，输入连接上一节点、分配独立输出，整个图一次构建、一次 backend buffer 分配。检查实际 GGML 节点列表、顺序、src0、连续 layout、设备支持与地址非别名；它们是**执行图结构证据，不是 CUDA kernel 计数**。

输入为 `(((i*73+19)%255)-127)/256`，每节点乘 0.5。CPU 参考独立计算整数乘 `2^(-8-stage)`。所有值精确可表示；32 层后最小非零值为 `2^-40`，远离下溢。每次完整读取最终输出并按 F32 位模式检查，首次和最后正式调用额外核查所有中间输出。这样既避免数值放大，也避免长链抵消后仅看末端“碰巧正确”。

初始输入只上传一次；每次整图只在全部节点提交后调用一次显式 `ggml_backend_synchronize`。数值读取在计时边界外。原生 `tensor_set/get` 内部确有其自身的流同步，属于输入上传/输出检查，不是插入图内的每算子同步。backend 内部是否还有同步由 trace 记录，不由探针掩盖。

缓存策略是重复使用相同图 buffer，正式调用之间含最终输出 D2H 和 JSONL 写入。没有 cache sweep，也没有“冷 HBM”保证。图 payload 为 `(nodes+1)*elements*4`，逻辑读写各为 `nodes*elements*4`；allocator padding 单独记录。不同图大小可能处于不同缓存状态，不可把总时间斜率直接当统一内存带宽。

## 时间边界与输出

逐次输出绝对 QPC tick 与频率，并保存以下三层 NVTX 范围：

- `graph_submit/<config>/<first|warmup|formal>/<index>`：整次观测范围。
- `.../host_submit`：仅括住 backend graph compute 调用。
- `.../final_sync`：仅括住最后的 backend 同步调用。

`qpc_submit_end-start` 是主机提交调用返回时间，`qpc_sync_end-start` 是同步调用时间。二者之间的 NVTX/QPC 插桩间隔单独可计算；`sync_end-submit_start` 是带已记录间隙的提交到完成包络。初始化、构图、分配、上传、数值校验和写盘均在外部。所有 NVTX push/pop 自身的 QPC 边界也保存。

GPU、driver/runtime API 与 idle 区间只能在 Nsight 同一时钟内分区求并集。**不能把 QPC 与 Nsight 纳秒直接相减，不能将 GPU 与包含/重叠它的主机等待相加。** JSONL 的 `observed_device_kernel_count`/`observed_cuda_graph_launch_count` 留 `null`，等待采集侧按标签、Graph ID 与 CUPTI 活动核实；禁止用 `requested_nodes` 或 `actual_ggml_nodes` 代替。

输出文件用排他创建。每次调用和校验单独成行，出现异常追加错误行、保留已产生内容；只有完整 footer 才表示进程完成。完成也不等于可校准。运行 receipt 保存 launch argv、进程 PID、原始硬件指纹、退出码及前后身份检查。超时标为仍在运行，停止调度后续进程，不终止 child/profiler。

## 原生环境一致性及范围限制

四个原生 GGML DLL 与固定 native 数据同源同 SHA；复制了 R16/r3 的绝对路径 DLL 预加载 guard，引用保持至所有 backend/buffer 销毁之后。编译使用现有 import libs，不重编 CUDA、不重连原生 DLL。具体源文件、构建回执、原生协议及 SHA 见 `source_provenance.json`。

原生运行环境完整继承 long-grid `runtime_environment`，包含 Graph/fusion 开关缺省、`LLAMA_TRACE_ANNOTATIONS=0`。额外清空能够意外覆盖 kernel/同步路径的遗留变量。GPU UUID、名称、计算能力来自 CUDA 实际查询；driver 与时钟来自运行前后 GPU 工具实际查询。期望值只用于比对，不写入 actual 字段。

该合成图只有一个 CUDA backend、一个主机提交者，无 CPU 节点，也没有创建 CPU worker pool。原生的 16 CPU 线程和 `0x55555555` worker mask 因而在此**不适用**；本探针不能证明 CPU 混合卸载、llama graph builder、server 调度、多请求 batching 或跨设备传输的等价性。它只为单 backend 图控制成本补充通用证据，不能凭一次 SCALE 链覆盖量化 GEMM、FFN 或整个 LLM 的成本。

## 入口（仅供父任务审查后执行）

源文件检查可以直接运行，不加载任何 CUDA DLL：

```powershell
& 'E:\anaconda\python.exe' '.\reference_check.py'
```

父任务审查后，以下入口编译当前独立 probe 和 CPU 参考、运行 CPU 参考、捕获实际 compiler include 闭包并发布 `build_manifest.json`。它不会启动 GPU：

```powershell
& 'E:\anaconda\python.exe' '.\build_probe.py' --compile --reviewed
```

父任务或采集子代理准备新输出目录并确认空闲窗口后，单个配对 arm 示例：

```powershell
.\invoke.ps1 -Config scale_f32_e1024_g8 -Pair 1 -Arm direct -Output '<新建原始输出目录>\pair1.direct.jsonl' -Run -RootReviewed -IdleWindowConfirmed
.\invoke.ps1 -Config scale_f32_e1024_g8 -Pair 1 -Arm profile -Output '<新建原始输出目录>\pair1.profile.jsonl' -Run -RootReviewed -IdleWindowConfirmed
```

`invoke.ps1` 不负责重测矩阵、不负责拟合、不自动重试失败进程。`-Run` 缺省时只检查准备/冻结状态。源码准备目前未生成 `prepared_identity.h`、编译日志、EXE、build manifest 或 READY 文件。第一次构建若失败，保留全部证据，由父任务建立下一版本再修复，不能覆盖一个已冻结目录。

## 审查完成后仍需的证据

1. C++ 实际编译、动态链接与设备数值校验；目前只有源码与主机数学/语法检查。
2. 每个正式 label 的实际 kernel 数、名字、grid/block、Graph 捕获/更新/重放、CUDA API 类型和次数；特别检查连续 SCALE 是否被某条路径合并。
3. 18 对的 profile/direct 比较、进程间波动和实际频率；任一失败、超时、缺 trace 都保留。候选质量门为 profile/direct 正式中位数偏差不超过 20%、正式 P90/P10 不超过 1.5；不达标只报告证据不足，不能据此修成本系数。
4. 每次调用的 GPU/API/同步/idle 区间分区闭合，以及 Graph 节点归因不跨调用；插桩间隙必须单独解释。
5. 新证据能否降低未参与修改场景的 Engine TTFT/TPOT/E2E 误差，仍由父任务冻结预测后验证；本探针准备完成不代表目标已经达成。
