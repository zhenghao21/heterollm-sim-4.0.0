# R16 Nsight 长等待只读诊断

结论：原报告中的挂起在本次审查开始前已经自行恢复。profile 进程真实返回 0，完整退出回执及 trace.nsys-rep 均存在；无需杀进程、停止 trace 或重跑此 profile。本审查未做生命周期操作，未改冻结/探针/时钟控制器，未运行 GPU、native、测试或安装依赖。

## 确认事实

目标：collection_r2/runs/train_Q5_0_m1_n4864_k896/pair_03/profile。

- 11:07:29.880（UTC+8）supervisor 开始；11:07:30.398 启动 Nsight PID 36548，supervisor PID 47056。
- microbench.json 于 11:07:39 创建、11:07:40 完整写入，共 46,274,491 bytes；可完整解析为 single-operator-surface-probe/v2，status=measured，graph_compute_calls=36，1 first+5 warmup+30 formal 全部保留。
- 全 36 行 ggml_status/cuda_wait_status=0，correctness.passed=true，模块前后稳定。这里只检查原输出的结构和已记录状态，不重跑全数值验收。
- 11:10:29.807 产生 180 秒非终止检查点。checkpoint 的 still_running 是历史状态，不能在已有 complete.json 后继续当现状。
- 11:15:55 stdout 出现完整 Nsight report 生成至100%，trace.nsys-rep 成功落盘 79,695 bytes，stderr 为空。
- complete.json：returncode=0、status=completed、child_process_exited=true、wait_observation_errors=[]，utc_finished=11:15:56.053；冻结末尾校验通过。
- 25/234 个阶段回执现已存在，而非先前24个。检查时 PID 5548/33068/36548/47056/25692 均已不存在；没有试图绕过权限或读取其他进程私有内存。
- 同一个绝对 QPC 域内，最后 formal 数值验证结束至 Nsight wrapper 返回经过 **498.4347515 秒**。这不是 GPU kernel 耗时或主机开销估计，只是进程退出/报告完成阶段的观察时间。
- clock_readback_gate=true，30/30 formal 均有前后读回包围，共60个读回，全部2392MHz。每个短 formal 区间内直接采样数为0，故仍只能声称邻近包围读回；不能声称连续内核频率证明。
- clock_control_finish.json 记录主任务控制器已在11:15:57执行 nvidia-smi -rgc，returncode=0，状态 finished。

## 可以定位到哪里，不能确定什么

microbench.json 在应用 main 的 run 返回后写入，因此完整文件证明36次图调用、显式等待及数值输出已经结束。之后进程可能仍在 CRT/DLL/CUDA 清理、Nsight 注入回调、跟踪导出或落盘。没有线程堆栈/ETW/时序化 profiler flush 日志，不能把498秒确定归因于某一种清理或落盘死锁。无永久死锁证据，因为最终正常退出并生成报告。

实际冻结命令带 --kill=false、--wait=primary、--stop-on-exit=true，没有 --duration 固定截断。保存的 Nsight 2026.5 CLI help 明确：wait=primary 等待应用进程终止；stop-on-exit 在应用退出时停止分析；kill=false 不终止目标。故应用数据已写完而 wrapper 仍等待，与当前选项允许的退出/跟踪完成阶段一致。尚无必要执行额外 Nsight status 命令；进程已经结束，且只读文件已给出更强的真实退出证据。

## 主任务安全后续

1. 保留 checkpoint、完整应用JSON、trace report、telemetry、complete 原样；不是重跑条件。
2. 可独立做既有 nsys-rep 的只读导出/提取，不需要再次运行 probe。该 profile 的数值/频率结构正常不等于整个 config 已通过三对质量门。
3. **不要直接使用已结束的旧锁频回执恢复后续 GPU 阶段。** 外部控制器已恢复频率，新锁频会话必须与旧数据严格分开，由主任务决定新 revision/新绑定。当前原冻结只允许一个不可变锁频会话；不能通过把旧回执当仍有效来跳过真实控制状态。
4. 控制器把 remaining_matrix 的 checkpoint 返回码0记为 finished；它表示控制进程生命周期结束，不等于234阶段矩阵完成。应以实际25/234回执及其状态说明覆盖范围，不以 clock_control_finish.status 判全矩阵成功。
5. 若后续需要缩短退出尾部，必须先独立定位线程/驱动/profiler行为并另起协议版本；不修改本次冻结的等待/超时策略，也不从本次样本推断可扣除的纯 profiler 开销。

## Bridge 与 resolver 审查提醒

profile_bridge/bridge.py 当前已显式检查 collector protocol v2 与 clock-domain 兼容性；代码保留 resolver_collection_protocol_v2_unsupported、resolver_clock_domain_evidence_unbound 两种 blocker。只有 wrapper/schema接受v2不足以晋级，resolver必须独立验证 clock receipt/telemetry与正式QPC包围域，同时保留源/运行时、缓存、K-domain条件。此处仅静态读取，没有生成或放行成本profile。请主任务转交正在处理 RESOLVER_V2_CLOCK_TASK 的审查代理核对，避免丢弃v2时钟字段。

## 证据摘要

- complete.json SHA256: c1d34cd9c2d5338b2db0a3fc249c3deb9b61af0a33e1df0331ea3a66ec17bae3
- microbench.json SHA256: 34d6ccbefba32a6c70735a65dd80e1f5b36826d9e7083590421836330e60767a
- trace.nsys-rep SHA256: 0e912c4a62d06f7bb5ed419c36ada433907099528e3b46574791327dbae3bd8b
- checkpoint.json SHA256: d27a6b48dbf9692222bbd7cb62c23262272d9fe0ca4152f57a61b4c4073ee231

只读来源还包括 profile/stdout.txt、stderr.txt、spec.json、launched.json、clock_control_finish.json、clock_control_run.py、冻结Nsight profile help。没有读取任何 native LLM actual。
