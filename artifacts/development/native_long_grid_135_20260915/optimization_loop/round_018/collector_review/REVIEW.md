# R18 采集器独立审查

结论：**先修两类 P1 证据绑定问题，再冻结采集器。** 本次未启动 GPU、未修改采集器/探针/核心源码。审查对应的编译探针 manifest SHA 为 `15fc2cf1c449968d4cc756d4c43cafb319c2764aa49e9a69c197269eb07fb617`，EXE 为 `86d268a3ceb9a223f1caa578ba1a68ac57ad8f971eac2b738aef81047f406a0b`。采集器尚未冻结；精确被审查源码 SHA 已保存在 `review_identity_v1.json`，独立回归期间源码没有变化。

## P1-1：完成记录没有强制完整、就地的原始证据集合

位置：`collection/extract.py:169–181` 的 `stage_complete`，尤其 177–180 行。

该函数只遍历 `r.get('artifacts', [])` 中已有的条目；若列表缺失或为空，循环直接结束，仍将阶段视作完整。即便列表非空，也只按文件 basename 决定是否验 SHA，没有要求被验证的路径正是当前 stage 内稍后会被读取的文件。`microbench.json`、`telemetry.json`、trace 和 stdout/stderr 不能仅因为“现在能读到”就被当作本次原生进程产生、且自完成后未变化的证据。

独立回归 `test_completed_stage_rejects_absent_required_artifacts` 提供其他已通过的时钟与退出绑定、但 `artifacts=[]` 的完成记录：当前函数**没有拒绝**。这不是已有 GPU 数据损坏的结论，而是尚未关闭的入口。

修复要求：

- 根据 direct/profile/export 阶段明确必要集合。测量阶段至少包含本阶段 raw JSONL、telemetry、stdout/stderr；profile 加入 report，export 必须包含本阶段 SQLite。必要 spec/启动记录也需绑定。
- 对集合检查缺失、重复、None、非法 SHA、非绝对路径；实际文件路径必须与当前阶段期望路径一致。stdout/stderr 可以合法为 0 字节，不能一概拒绝空日志。
- 校验 spec_ref/启动 argv 与当前配置、pair、mode 和固定协议一致；extract 后续读取的必须是刚校验过的同一个路径及 SHA。
- 失败维持原有 6 配置/18 对分母，不通过补造引用来“修复”缺失原始证据。

## P1-2：原生进程、配对、trace 尚未形成一条身份链

位置：`collection/probe_adapter.py:136–191` (`audit_raw`)、195–207 (`raw_signature`)；`collection/extract.py:73–95` (`correlate`)；相关生命周期在 `collection/worker.py:27–43` 与 `collection/clock_controller.py:97–106`。

探针实际保存 `header.pid/argv/qpc_start/utc_start` 与 `setup.pair_id`，但原始审核没有检查这些字段。trace 归因只由“配置相同的36个label + trace内部PID + correlationId”构成，**未将 trace PID 与原生 header PID 连接**。不同pair的label本来就完全相同；只凭label，无法排除同一配置的另一个进程/配对被误配。

两项独立回归均复现接受缺口：

1. 删除原始记录的进程、argv、起始时间和pair身份后，`audit_raw` 仍返回 `valid_raw=true`。
2. 原生 app 声明 PID=22222，而合成 trace 的 globalPid 属于另一进程，`correlate` 仍返回 `all36_chain_complete=true`，并完整统计288条kernel。

修复要求：

- 必填并检查 header PID、argv、起始 QPC/UTC、setup pair_id；将它们与实际 stage spec、启动记录及父进程记录的 QPC 生命周期绑定。输出路径必须属于当前阶段。
- 使用本版本真实 Nsight 的进程身份编码或目标进程元数据，把NVTX/kernel目标进程绑定至探针PID；不是只让NVTX和kernel在trace内部彼此相同。
- 单配置三次配对的独立性应由进程启动身份与时间区间证明，不能仅用数组下标 `{0,1,2}`。PID可能被系统复用，应联合启动时间/启动记录检查，而不是盲目要求所有PID永不相同。
- profile 下监督器直接等待的是 Nsight。已核对本机 Nsight CLI 的 `--wait` 默认是 `all`，正常退出路径会等待目标；异常路径仍建议用原生header PID纳入明确的退出核查，避免仅凭工具包装进程退出就恢复频率或开启下一阶段。

这些修复都属于证据与执行生命周期门禁，不需要改阈值、探针数学、native binary或成本系数。

## 已通过的审查项

- 6配置 × 3配对 × (direct/profile/export) = 54阶段，其中36个测量进程、18次导出；首次1、预热5、正式30的分母一致，配对次序交错。
- direct/profile使用同一个event-free探针与native-default Graph/fusion策略；profile明确开启node粒度。实际kernel缺失不会用计划节点数回填。
- 独立正向回归：一次GraphLaunch的8个child kernel成功归因；runtime/driver嵌套记录不会重复计device event。36次调用共288个实际kernel；每次互斥GPU/API/sync/idle分区闭合。
- 独立缺失回归：移除一条child kernel后，实际数为7，trace整体拒绝；不会改写成计划8条。
- 原始QPC与Nsight时间分别运算，没有发现跨时钟相减。API区间保留层级，GPU与host等待通过互斥分区避免重复相加；API记录总数不被称为唯一Graph提交数。
- SM采样来自实际NVML读数，保存读取起止窗，以5ms目标周期做单字段采样；正式区间前后括号各按冻结25ms规则判断，2400±30MHz阈值未改。它是抽样频率证据，不能解释为连续逐kernel时钟波形。
- 非daemon采样线程拥有NVML；停止后真实补一次最终读数，join后关闭timer。检查点不kill；直接子进程 wait 异常仍保持所有权。
- 固定策略保留三个配对及全部失败；独立失败回归确认拒收时 `measurement_cost_eligible=false`、`calibration_eligible=false`、`fit_performed=false`。没有自动生成验收系数。

## 可复现结果与下一步

`test_evidence_binding.py` 的6项独立回归目前为 **3失败、3通过**，详细输出在 `host_regression_v2.log`；失败恰对应上述两类门禁的三个反例。测试只使用结构化合成事件，不是实际Nsight性能验证，也不重复探针的数学证明。

```powershell
$review = 'F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_018\collector_review'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'E:\anaconda\python.exe' -m pytest "$review\test_evidence_binding.py" -q -p no:cacheprovider --basetemp "$review\pytest_scratch_after_fix"
```

由采集代理修复并更新其结构化准备状态后，重跑这6项及相关既有回归，再由根代理冻结。随后只运行首个真实pair，核对实际Nsight graph-child correlation、源路径、SM读窗和观察扰动，再决定继续其余17对。真实trace尚未出现，因此此次静态/合成回归不能宣称设备实测或Engine误差已经达标。
