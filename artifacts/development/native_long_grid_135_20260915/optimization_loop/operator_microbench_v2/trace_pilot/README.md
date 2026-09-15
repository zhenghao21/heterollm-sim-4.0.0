# 合成 GGML GPU GEMM trace 试点

结果：一次 profiling、一次相同设置的直接对照和一次 SQLite 导出全部返回 0。结构提取通过，3 项解析器回归测试通过。**Nsight 驱动兼容性门未通过；本试点只作诊断，不可校准，不可宣称时延误差 <5%。**

## 运行前固定的协议

- 算子：独立合成 `GGML_OP_MUL_MAT`，Q4_K 权重，F32 输入/输出，M=64、N=4096、K=1024。
- seed=20260914；每进程首次调用 1 次、预热 3 次、正式调用 20 次。
- 首次和末次各检查 256 个均匀输出位置，绝对容差 0.05、相对容差 0.03；判据是 `abs(actual-reference) <= atol + rtol*abs(reference)`，另检查所有输出有限。并非每个调用都逐元素做数值比较。
- 外层 NVTX 显式开启；`LLAMA_TRACE_ANNOTATIONS=0`、`GGML_CUDA_DISABLE_GRAPHS=1`。MMQ/cuBLAS 强制选择开关均显式未设置。完整环境白名单记录于 protocol 和应用输出。
- 请求 128 MiB 缓存清扫，设备 L2=64 MiB，工具实际按 4 倍 L2 使用 **256 MiB 输入加独立输出**的读写清扫。清扫、上传、量化、数值比较不进入外层 NVTX 或逐调用 host 计时。
- GPU：RTX 5080，UUID `GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b`。未锁频；每 50ms 记录一次 NVML 时钟/温度/功率和原始 runner QPC tick。该采样不能保证覆盖每个很短的 kernel。
- 固定协议于任何应用执行前写入；每个子进程最多 180 秒，最多一次 profile 加一次对照，无自动重试。

## 观察到的算子链

24 个外层 NVTX 区间各对应：`quantize_mmq_q8_1`、`mul_mat_q`、`mul_mat_q_stream_k_fixup` 三个 kernel，共 72 个。24 个 `scale_f32` 缓存清扫 kernel 均位于区间外。共保留 96 个 CUDA kernel 事件和 224 个 CUDA runtime API 事件。

解析器先用相同线程的 CUDA API 区间匹配 NVTX，再用 correlationId 加 globalPid 关联 kernel；globalPid 掩码来自本机 NVIDIA 自带 `nvtx_gpu_proj_trace.py`。并发 kernel 按时间区间并集计算，避免叠加重叠时间。

| 正式 20 次调用 | 中位数 | p95（最近秩） | 最大值 |
|---|---:|---:|---:|
| profiling host graph_compute+sync | 52.3 µs | 140.4 µs | 181.7 µs |
| 无 profiling host graph_compute+sync | 56.2 µs | 64.3 µs | 64.4 µs |
| profiling CUDA kernel 时间并集 | 15.552 µs | 15.905 µs | 15.937 µs |

**host wall 已含 launch 与同步，CUDA kernel 并集是其 GPU 视角，二者不能相加。** 单次顺序对照中 profiling 的中位数低约 6.94%，同时有明显 host 尾部；未控制频率和顺序，不能据此宣称 profiling 加速或获得可靠开销修正。

## 可信度限制

Nsight 2024.6.2 在 trace 中明确报告：安装的 CUDA driver version 13.4 不受该版本支持，改用 driver version 12.8 的采集库。CUPTI 实际加载 `cupti64_128.dll`。虽然事件生成且结构关联完整，兼容性门仍为失败；未重试。

现有已测 C++ 程序使用 MSVC `steady_clock`，其实现封装 QPC。逐调用 `start_ns/end_ns/elapsed_ns` 均原样保存；该程序没有输出绝对 QPC tick 或 process_start 原始 epoch。本试点不反推绝对 tick，也不对应用相对时钟与 Nsight 时钟猜测对齐。runner launch/exit 和遥测中的原始 QPC tick 不能替代应用内原始计时。

原 build manifest 缺少 CUDA/NVTX/MSVC/Windows SDK 的完整传递头文件闭合清单。新 protocol 补录当前直接相关头文件 SHA，并明确不能据此追认旧构建的完整闭合。原 CPP、EXE、build manifest、smoke 文件均未修改；已有 manifest 中各项 SHA 在执行前后重新核对一致。实际加载 DLL 的路径和 SHA 在两进程前后均稳定且与已冻结模块一致。

没有读取 GGUF、没有运行 LLM、没有使用目标 LLM 时延、没有修改成本模型或做参数拟合。

## 文件与后续使用

- `protocol.json`：不可追改的运行前协议及工具、源码、GPU、补充头文件标识。
- `pilot.py`：有一次性目录/协议哈希保护的独立 runner。该目录的实验已完成，重新运行会拒绝。
- `analyze.py` / `test_analyze.py`：只读 SQLite 提取及 3 项归因/重叠回归测试。
- `execution_receipts.json`：三次有界进程的 argv、返回码、QPC/UTC 边界与环境。
- `result_summary.json` / `extraction_validation.json`：结果、质量门、时钟范围、模块身份和结构验证。
- `per_call_summary.json`：24 次调用的应用相对时间戳、外层 NVTX 区间、kernel 并集及关联信息。
- `profile/` 与 `direct_control/`：应用原始 JSON、日志、遥测。profile 中另保留 `.nsys-rep`、`.sqlite`、`raw_events.json`。临时 QDSTRM 已由 Nsight 自身清理，未伪造或重采；NSYS report 和原始 SQLite 仍完整保留。
- `mapped_calls.json`：包含全部关联事件的详细提取。

建议 Git 仅收录 README、三个 Python 文件、protocol、execution_receipts、result_summary、extraction_validation、per_call_summary。`.nsys-rep`、SQLite、原始事件、日志、二进制、遥测全量和大型关联提取留本地，由 root 统一选择提交。
