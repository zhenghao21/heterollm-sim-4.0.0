# Round018 图提交采集器

本目录是新图探针的独立采集器副本，尚未冻结或运行。根代理已完成 `graph_submit_probe` 编译；采集器按实际源码、协议和编译清单对接，不修改探针、既有冻结版本或旧实测。当前状态见 `preparation_status.json`。独立复核已修复必需产物缺失、原始PID/pair/argv/QPC与trace进程身份未闭合的问题，根代理重跑39项主机测试全部通过。通过主机检查只说明采集器接口和拒收行为符合预期，不代表真实 GPU trace 已通过。

## 固定矩阵与运行边界

- 6 配置：F32 连续一维 SCALE 链，elements 为1024、262144，节点数为1、8、32。
- 每配置3对独立进程；每对含direct、profile和SQLite导出，共18对、54阶段。
- `(config_index+pair_index)` 偶数时direct在前，奇数时profile在前；导出始终最后。
- 每进程1次first、5次warmup、30次formal。每次图只做一次异步提交及一次最终backend同步。
- 两个测量进程使用相同探针参数，仅输出文件不同。实际入口为 `--run --config ID --pair-id ID/pair_01 --output NEW_FILE`。`microbench.json` 的内容是JSONL，不能按单个JSON对象解释。
- 两臂均含NVTX语义标签，均不插入CUDA Event；保持锁定原生协议的默认CUDA Graph与fusion设置。没有event臂、graph-disabled臂或目标LLM实测。
- Nsight使用 `--kill=false --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=false`。节点trace相对于direct的差异仅称为该协议下的观测扰动，不能当成纯CUPTI开销。

主机耗时来自探针同一QPC域的submit开始至最终同步结束，保留submit、sync与NVTX嵌套间隙。Nsight时间线仅在自己的时钟域内做kernel/API/sync/unattributed互斥分区；不跨时钟相减，不把kernel、API与包含它们的wall直接相加。setup、输出验证及文件写入在正式边界外。

## 身份、频率与进程证据

`runner prepare` 要求实际编译清单及328项来源/二进制身份闭合检查；不要求探针不存在的READY文件。后续冻结覆盖采集器、适配器、计时助手、频率控制器、测试、说明、准备报告、复制来源、探针清单、工具身份与协议。执行、恢复和收尾均检查调用方传入的同一字面SHA；必要字段或文件缺失直接失败。

SM频率门槛保持2400±30MHz，每个formal区间的真实SM读数括号必须≤25ms。高频只采SM时钟，使用本目录复制的高分辨率一次性Windows计时器和绝对QPC期限，不改全局系统timer resolution。每个读数保留独立起止QPC，跳过期限不补造采样。NVML所属线程在stop后作最后一次SM读取，再关闭NVML；主线程确认join后才能关闭timer句柄。

此处**改变了观测方法**：旧轮次的多字段NVML定时查询已换成SM-only高频读窗，必须新冻结，不能归因成仅改变定时器。GPU UUID、driver、实际读数与实际CUDA设备属性都来自运行时并与冻结身份比较，不把期望值复制为实际值。模型、原生LLM actual、成本参数均不参与这里的校准。

180秒检查点只返回“仍在运行”，不会打断子进程。worker在真正退出后才写completion receipt；控制器保持子进程所有权，等同一进程收尾后再恢复下一个阶段，所有已启动进程退出前不恢复GPU频率。失败、缺失和不稳定数据保留在完整6配置/18对分母中。

## Trace解释与限制

- 必须精确捕获36个外层语义标签，核对原始图的shape、dtype、src0链、分配与访问字节、运行环境、硬件、模块及所有数值验证摘要。
- 通过实际launch correlation关联kernel。一个GraphLaunch可归属多个图内kernel；runtime wrapper与driver记录重复映射同一device事件时只数一次kernel，API原始记录都保留。
- 无法关联的graph child必须拒收，不能仅凭时间区间或计划节点数推断归属。直接进程没有CUPTI的实际kernel数保持null。未知融合或kernel路径不自动套用SCALE成本。
- API计数是API记录数，可能同时包含runtime与driver两层；不能把两层记录总数解释成去重后的图提交数。源码图依赖与时序区间分开验证，允许PDL导致外层kernel区间重叠，不把起止排序当作依赖证据。
- 所有调用的最终输出摘要以及first/最后formal的中间节点摘要必须通过。探针保存的是checked/mismatch/bitwise等摘要，并未保存全部输出数组；采集器不能独立重算未保存的数组。参考实现及编译身份保留在探针清单中。
- 首个真实trace仍需确认该版本Nsight的graph-child correlation。无法证明时保持拒收，不补写成已完成的关联。
- diagnostic通过不等于LLM迁移通过，不产生校准系数或宣称Engine三项误差<10%。默认CUDA Graph、单提交线程的CUDA-only图也不构成完整LLM/CPU线程池调度等价性。

## 根代理操作入口

先完成独立审查，再冻结；以下命令均从实际编译结果读取身份，不修改冻结探针。`--root-reviewed`表示根代理审查此版本完成；不要在审查前执行。

```powershell
$collector = 'F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_018\collection'
& 'E:\anaconda\python.exe' "$collector\runner.py" prepare --root-reviewed
$approved = (Get-FileHash -Algorithm SHA256 -LiteralPath "$collector\freeze.json").Hash.ToLowerInvariant()
& 'E:\anaconda\python.exe' "$collector\runner.py" verify --expected-freeze-sha256 $approved
```

人工审查实际freeze SHA后，在协调空闲窗口且根代理允许时启动控制器。控制器先只跑第一对的3阶段，保存 `graph_clock_control_first_pair_ready.json` 并等根代理检查真实trace，不自动放行剩余矩阵。

```powershell
& 'E:\anaconda\python.exe' "$collector\clock_controller.py" --run --root-reviewed --idle-window-confirmed --expected-freeze-sha256 $approved
```

首对通过审查后，根代理创建新的 `graph_clock_control_continue.json`，内容必须包含 `approved_freeze_sha256`、`first_pair_reviewed:true` 及first-pair-ready里clock receipt的实际 `clock_receipt_sha256`。停止路径为创建 `graph_clock_control_stop.json`。这两个文件是控制器的根代理协调信号，不是用户再次授权请求。默认review期限600秒；到期保留数据，等待已启动进程退出后恢复频率，不杀进程。

矩阵结束后使用新输出目录提取，不能覆盖旧提取结果：

```powershell
& 'E:\anaconda\python.exe' "$collector\extract.py" --output "$collector\extracted_v1" --expected-freeze-sha256 $approved
```

本次准备未执行这些冻结、频率修改或GPU运行命令。若冻结后需更改代码，必须新建完整副本和新冻结，不能修改当前冻结文件后继续同批数据。
