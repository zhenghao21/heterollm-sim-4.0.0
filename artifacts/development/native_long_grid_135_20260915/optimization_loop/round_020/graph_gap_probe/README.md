# R19 SCALE 图间隙 A/B 准备包

**状态：`source_prepared_not_compiled_not_measured`。** 这里没有编译、GPU 初始化、基准、推理、成本系数或核心模型改动。父任务审阅并完成冻结/启动器接线前，不得执行。

R18 的诊断记录表明，大张量配置在一次图调用之后通常伴随毫秒级的 `tensor_get` 数值核验和相近量级的图外主机间隔。它只是待证伪的来源：不能把间隔直接归因于驱动、缓存或 GGML，也不能把 QPC 主机差值当成 GPU 时间。

本包构造同一原始 DLL、同一 SCALE 图、同一输入、同一 1+5+30 调用序列的 A/B 对照。两臂均保留每次整图调用的一次最终 `ggml_backend_synchronize`，保持原有 QPC 位置、NVTX 标签语法、Graph 默认策略和运行时/硬件策略。探针源码重建后只能说明两个探针臂相互对照；它不宣称与原生应用源码或二进制完全等价。

| 臂 | 每次图调用后的动作 | 数值覆盖 | 原始输出节奏 |
| --- | --- | --- | --- |
| `control` | 完全沿用 R18：逐次 final D2H、逐次精确校验、逐次 JSONL；首次与最后正式调用再检查全部 stage。 | 36 个 final；首次与 postformal 的所有 stage | 每调用 |
| `buffered` | 调用记录预先放入固定内存；每个阶段内部仅含图调用和最终同步。 | `first`、`post_warmup`、`post_formal` 三个边界块中所有 stage 精确校验，保留每块每 stage 的 FNV-1a 原始字节哈希。 | postformal 后统一写出 |

`buffered` 臂刻意不证明每次调用都数值正确；它也改变了阶段间 D2H/写盘上下文。因此它如果改变抖动，只能支持“逐次核验/写盘可能参与间隙”的诊断假设，不能证明 QPC 本身被改变，也不能证明驱动或 GPU 工作量的因果来源。

第一阶段只有一次调用，随后的 `first` 数值块发生在它完成之后；五次预热和三十次正式调用各自在内部没有 D2H/JSON。这样既保留 `first + post_warmup + post_formal` 三个强制数值块，也使处理臂不具备逐次核验的时序环境。该差异已经写入原始 `arm_metadata.write_cadence`，不得在分析时淡化。

`graph_gap_probe.cpp` 是可接入 R18 启动器/冻结器的聚焦执行单元，并非复制完整采集器。父任务接线时必须在 `arm_metadata` 提供每臂的实际 argv、冻结收据，以及缓存、分配、上传和 reset/no-reset 的实际语义。两个臂的输入/图/缓存语义必须相同；不得为处理臂额外重置、扫缓存、改节点、改 backend、改输入或移除最终同步。

完整矩阵预登记为 `2 arms × 6 configs × 3 sequential pair records = 36` 个 arm/config/pair 记录；每条记录需要配套 direct/profile 执行。父任务可先用一个明确记录的根试点配置验证端到端，但该试点不能无声替代后续 6×3 覆盖。

- direct：不使用 CUPTI/Nsight；不得用主机 QPC 差值反推 GPU 时间。
- profile：使用 NVTX + CUPTI/Nsight `--cuda-graph-trace=node`，实际核数、图 launch/capture/update、runtime/driver API、依赖与重叠必须来自 trace。
- 质量门：保留 R18 的 `2400 ± 30 MHz` 记录策略、正式段 p90/p10 ≤ 1.5、3 进程中位数离散度 ≤ 5%、profile/direct 正式墙钟中位数差 ≤ 20%。未通过也必须保留所有原始材料，并标为不合格；不得删数据、拟合全局 LLM 时间或输出成本系数。

## 文件

- `graph_gap_probe.cpp`：两臂的局部执行与数值块逻辑；无 `main`、无编译脚本。
- `math_reference.h`：独立闭式 F32 参考。
- `protocol.json`：矩阵、控制差异、运行时/硬件/trace/失败保留约束。
- `reference_check.py`：仅主机静态数学、协议和源码结构检查；不加载原生库、不调用 CUDA。

建议在父任务解除短时采样窗口后运行：`python reference_check.py --output preparation_checks.json`。该检查不构建或运行探针，结果也只表示准备材料的静态一致性。
## R19 集成入口

`graph_gap_main.cpp` 是新探针的独立入口：`--identity-check` 仅校验冻结文件、运行时环境和已加载的原始 DLL 身份，不调用 CUDA 设备 API；`--run` 才会创建 backend、读取硬件并执行图。输出 JSONL 使用排他创建；正常或可捕获失败结束后，入口关闭原始文件、计算完整 SHA-256，并以新的 `*.receipt.json` 保存哈希、长度和状态。

`build_probe.py` 默认只校验依赖。只有父任务明确执行 `--compile --reviewed` 时才会生成新的 `prepared_identity.h`、编译命令、`/showIncludes` 日志、host 数学/模拟 DLL 加载器检查和新 build manifest；它拒绝覆盖任何既有编译痕迹，且不会运行 GPU 工作。

`invoke.ps1 -IdentityCheck` 是构建后无 GPU 的 DLL 身份入口。实际运行必须显式传入 `-Run`，并且本修订只允许预登记 pilot：`scale_f32_e262144_g8`、`control|buffered`、pair 1..3、`direct|profile`，共 12 个 native 进程和 6 个 profile 导出。其余 6 配置域必须在 pilot 审阅后由新的冻结修订授权。
