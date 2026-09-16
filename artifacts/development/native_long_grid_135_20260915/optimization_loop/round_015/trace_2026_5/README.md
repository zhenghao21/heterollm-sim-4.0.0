# Nsight 2026.5.1 合成算子 trace（准备阶段）

本目录是独立的新协议，保留旧 `operator_microbench_v2/trace_pilot` 的全部证据。准备阶段只核验文件、版本、签名和源码，不运行 GPU 工作负载。采集需由 root 在硬件空闲时显式启动。

## 冻结的对照

复用原 `generic-gemm-microbench.exe` 和 `build_manifest.json`，保持 Q4_K、M=64/N=4096/K=1024、F32 输入/输出、seed=20260914、16 线程、首轮1次＋warmup3次＋formal20次、128 MiB 请求的冷缓存清扫及原容差。CUDA Graphs 关闭，NVTX 外层标记开启，内部 LLM 注释关闭；不读取模型或目标 LLM 时延。

变更为已签名的 Nsight Systems 2026.5.1.161，工具目录含 `cupti64_134.dll`。仅凭该库存在不能宣称运行时兼容性通过；必须检查新报告的实际诊断和模块。新命令额外显式使用 `--kill=false`，不让60秒采集期限终止目标程序。若外层等待180秒仍未退出，记录 `still_running` 和 PID，停止后续阶段并保留该进程。检查点不是完成时间，不能推断退出码或自行重跑；命令回执指出继续观察的对象。

本协议记录原实验 GPU UUID 作为待核验的设备条件。准备阶段不把历史设备身份冒充当前观测。正式执行时从 NVML 获取实际身份及遥测，校验 UUID 和驱动，保存前后证据。

## 身份和提取门

`prepare_tool_identity.ps1` 保存完整分发工具文件树哈希（忽略生成的 Python 缓存）及所有 PE 文件签名，关键 NVIDIA 主程序、注入库、服务和 CUDA 13.4 CUPTI 必须为有效 NVIDIA 签名。完整工具树不是完整 OS 动态依赖闭包，实际加载仍需运行后核查。

`pilot.py prepare` 冻结协议、runner、extractor、测试、schema 检查器、说明、版本/帮助、工具清单、现有构建身份和 Python 可执行文件。每次执行前、每阶段完成或检查点及执行结束都验证冻结；提取前后也验证。缺字段、缺文件、空身份表和哈希变化均拒绝。旧构建缺失传递头文件的局限继续保留。

解析器只支持原先明确验证过的列名，不猜新版 SQLite 字段。执行后先保存 `observed_sqlite_schema.json`；不兼容时保留失败，再由 root 建立新 extractor 版本，不能覆写本冻结。独立 `inspect_schema.py` 可只读列出实际列名。每个 NVTX 调用必须通过相同线程的 API 与 correlationId/process ID 关联 kernel，并按事件区间并集计算设备时间。host wall 与其内部设备时间不得相加。

## root 的操作入口

在本目录执行 `E:\anaconda\python.exe pilot.py verify` 可只读检查；正式启动为 `E:\anaconda\python.exe pilot.py run`。该命令只允许一组新 profile/direct/export，不自动重试。`pilot.py status` 查看检查点。确认三阶段退出成功后执行 `E:\anaconda\python.exe analyze.py`；遇到 schema 错误查看 `observed_sqlite_schema.json`，不得放宽规则后覆盖结果。

采集与无注入对照都保留 24 次调用。单一未锁频顺序对照仍仅用于诊断，不能生成 kernel 校准系数、不能证明测量扰动<5%，不能用负开销推导 profiler 加速。若需校准必须另行冻结含扰动控制的通用微基准协议。

提交仅包含新源码、协议、说明及精简证据。EXE/DLL、raw、SQLite、`.nsys-rep`、完整工具清单和大量 telemetry 留本地，由 root 统一选择。
