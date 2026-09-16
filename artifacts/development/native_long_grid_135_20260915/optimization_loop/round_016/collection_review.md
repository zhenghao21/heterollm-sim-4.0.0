# R16 collection 启动前独立只读审查

审查时间：2026-09-16 10:26 +08:00。审查者只读取 collector/probe 源码并执行内存内合成反例；未启动 GPU、native 或 runner，未编辑采集器或冻结文件。

**结论：当前冻结版本不宜直接用于无人值守的正式全矩阵验收。存在实质 P1 门禁缺口，需 collector 建立新 revision 后再冻结和复核；不能直接修改已经冻结的版本。** 这不是认定当前1671个文件身份已经失效，而是当前验证程序允许关键身份被省略且仍返回成功，无法兑现任务书的“缺身份不得通过”。

Root随后明确决定：已经手动核验当前ready→freeze锚点和1671项引用，先执行首个完整pair作为实际schema诊断，原版证据保留，防省略门在新revision修复。这个窄范围诊断决定不等于本审查已放行正式全矩阵；目前没有发现能证明当前已核验首pair必然无效的P0。诊断结果仍不可直接生成正式成本profile。

当前实际 `freeze.json` SHA：`8f9dca14c964da35dc56a4469240ce858c915020ec51a815c4080034afe6bcec`。
当前实际 `ready.json` SHA：`89b663406aed9b90e4004cc38f77a659f25ce55d5a90c478594b56f4701bec71`。

## P1：冻结门未验证被批准的冻结身份，也未强制必要集合完整

位置：`collection/common.py:74–81` `verify_freeze`；`runner.py:104` 及 worker 的调用。

目前只要求 schema 正确、`files` 非空，然后拼接调用者提供的 `files/probe_files/tool_files`，逐个核验。`probe_files`、`tool_files`、`critical_tool_files` 可以为空；`protocol_ref` 只取 path 读取，没有独立校验其 SHA/bytes；`python` 字段也没有必需身份核验。验证返回的 `freeze_ref` 是**当下再算的新 SHA**，没有与 root 批准值或 `ready.freeze_ref` 比较。

已执行一个纯内存反例，替换文件读取/哈希动作，不访问 GPU：manifest仅含一个无关 `files` 项，probe/tool/critical列表均空，`protocol_ref.sha256="WRONG"`，python为空；`verify_freeze(..., full_tools=True)` 仍返回：

```json
{"passed": true, "checked_files": 1, "full_tool_inventory": true}
```

它只核验了那一个无关文件。现有1671项通过不能覆盖这个反例。

修复要求：

- 运行、恢复、每个阶段和结束都接收并核验一个从外部批准入口传入的 expected freeze SHA；不能从正在被验证的 manifest 自己现算后当成批准身份。ready与freeze的关联也应一致。
- 强制所有必要身份集合非空且字段合法；对 protocol_ref、python和关键工具显式核验。
- 从已绑定的 probe build manifest/tool inventory/signature evidence重建必要集合，验证集合相等或规定的完整包含关系，不仅对当前manifest里“恰好还剩下的项”逐个检查。
- 加缺集合、空集合、删关键文件、错 protocol_ref SHA、换 manifest本身、改ready绑定的负例测试。

## P1：身份失败的阶段会被跳过后继续采集

位置：`runner.py:112`、`:136–137`；`worker.py:49`。

worker能把阶段记为`failed_identity`。但runner只要看到`complete.json`便计入完成并继续；新结束阶段同样只记录status，未阻止下一次启动。全部receipt存在时还返回`matrix_complete`，不区分成功完成、普通失败和冻结失效。

普通数值/波动失败可以保留并继续预定矩阵；**冻结身份失败必须停止本冻结后续测量**。否则文件在某阶段改变后又恢复，下一次当前文件检查通过，流程就会把失效期间前后的数据归入同一冻结矩阵。

修复要求：身份/冻结失败立即写不可忽略的终止检查点，并禁止后续阶段；恢复时同样检查已有失败receipt。若需修复代码/工具，使用新revision和新冻结，旧失败保留。最终状态区分“全部阶段已有回执”和“全部阶段成功且冻结有效”。

## P1：额外、未归属 GPU kernel 与测量区间重叠仍会通过完整链门

位置：`extract.py:70–117`，尤其`:116`。

当前正确地用 PID/TID/correlation关联主调用，也记录所有`unmapped_kernel_rowids`，但未把这些未归属事件与测量NVTX/kernel区间重叠作为失败条件。清扫或其他GPU工作如果来自不同API/线程/上下文，会落入未归属列表，同时目标conversion/main链仍可被判完整。

已用现有测试的36次MMVQ合成trace加入一个额外kernel：`rowid=999`、不同PID、区间`[100,170]`，与首个被测调用内目标kernel重叠。结果仍是：

```json
{"all36_chain_complete": true, "source_path_matches_observed_family": true,
 "unmapped_kernel_rowids": [999]}
```

修复要求：在同设备上对所有捕获kernel、memcpy/memset逐一检查与被测区间的重叠；预期清扫只能在测量之外。区分“已捕获且重叠”与“工具未覆盖外进程活动”，后者仍保留观测限制。不能把额外工作并入主kernel成本，也不能只列rowid而允许它进入measurement_cost_eligible。

## P2：动态库离线归属检查需要补全；已有probe防线应保留

位置：`extract.py:126–138`；`operator_probe/r2/full_raw_audit.py:23`。

collector未命中的模块只进入`unbound_loaded_os_runtime_modules`，不改变valid_raw。但补查probe源码发现它在`stream_event_probe.cpp:131–132`、`:170`、`:214`、`:249`会核验`identity_lock.h`中四个GGML库的实际path及SHA；因此**不能把关键GGML影子DLL称为当前可绕过的P1**。这条防线降低了离线检查不完整的实际风险。

建议collector仍显式区分四个必须绑定的GGML库、CUDA运行库以及允许只观察的OS模块，核对原始记录能独立复验编译侧结论。CUDA运行库不在上述四项frozen_modules中；当前保留其实际观测身份和条件迁移限制，不应把所有未绑定模块都笼统称为OS。这个补强可以进入新revision，不构成已经证明当前首pair无效。

## P1：Popen之后写启动回执失败，仍可能发布结束回执而子进程未结束

位置：`worker.py:33–50`。

`Popen`之后，`write_new(launched.json)`和QPC读取仍在到达`proc.wait()`之前。任一步异常会进入except/finally，写`complete.json`；当前代码没有在该异常路径等待或确认已经启动的进程退出。随后runner会把这个receipt当作可继续的阶段，存在重叠GPU负载的风险。

修复要求：一旦成功启动进程，先在内存保存其身份；所有退出路径都必须确认同一子进程终止，才可以发布“完成”receipt。若记录失败但进程仍运行，保留`unresolved_running`状态并阻止下一阶段，不杀进程。增加模拟`Popen成功→写launched失败`的主机测试，断言不会发布可继续receipt或启动下一项。

## P2：观测family仍由expected_source_path分支选择，错误时丢失真实已知family

位置：`extract.py:24–42`。

当前校验角色链能拒绝大部分错误路径，不会仅凭expected_path就通过；这是已有防线。但`family`先从expected_source_path决定，期望MMVQ而实际捕获明确MMQ时，`observed_family`变为`unsupported`，而不是`MMQ + expected_mismatch`。这会把“已观察到且可识别的实际路径”与“未知路径”混淆。

纯合成反例：实际roles=`conversion_mmq,main_mmq`，期望MMVQ；输出`observed_family=unsupported`。

建议先仅由实际symbol/role/量化类型/布局/网格识别observed_family，再单独比较expected/source-path reference。两者不符则拒绝数值语义迁移，但保留真实观测，便于诊断。

## 已通过静态审查的边界

- 主要JSON用独占创建；run目录`exist_ok=False`；提取输出必须是新目录；Nsight `--force-overwrite=false`。既有协议/冻结存在时prepare拒绝覆盖。
- 26×3组配对顺序固定为profile/direct、direct/profile、profile/direct；应用参数除输出路径一致，均为单图调用、1+5+30次。
- `--kill=false`保持；180秒checkpoint不杀进程；未收到完成receipt时不启动下一阶段。
- 主关联采用同线程API完整落入NVTX，再由process/correlation匹配kernel；重复kernel归属会报错；同调用多stream/device目前拒绝。
- kernel区间使用并集，host API、sync API、kernel span分别报告，不直接相加；QPC独立保留，未做跨时钟减法。
- 固定质量阈值与任务一致：每profile过程p90/p10≤1.5、三个profile过程中位数相对中心偏差≤5%、每pair host差≤20%、direct过程p90/p10≤1.5，数值/链/路径/警告均须过门。
- 验证组保留、不拟合、不输出profile系数；结果仍`calibration_eligible=false`。event模式与event-free控制差异也有明确局限说明。

## 审查版本身份与边界

| 文件 | SHA-256 |
|---|---|
| common.py | b82e8720025f466edfc46679bb550df26a2b3dc155138953ed54c68d965fef34 |
| runner.py | f72720ecfde47572c7f0da16b0e9b8917398e02213d78147adbe31282905416c |
| worker.py | 7f8a80dcc09dfe4a919ba73904ade0739350f7a7b6172b44fb3a4a431283070b |
| extract.py | 48ee7e470aa17d030b0b2b7f92cbaf5dbe90489535e7b428dd41ff1f5c2c8791 |
| quality.py | 5aaf8b2d1268e0bda920cf0ae2c0dd8ce8325a76abed27b50d7a8ba9602a42ea |

已执行的反例仅是纯内存冻结逻辑和合成trace，没有正式GPU样本。本报告不表示实际GPU已经出现串流、DLL替换或manifest损坏；它记录的是在正式证据采集前需要封闭的可到达失败路径。修复后的新revision应重新运行这些负例及已有测试，再由root决定启动。
