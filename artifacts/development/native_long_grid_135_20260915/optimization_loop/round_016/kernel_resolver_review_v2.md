# Kernel calibration v2 独立审查

2026-09-16。只读审查 `kernel_calibration.py` 与测试、collector_r2冻结代码；不读取正在采集的结果，不改实现，不编译，不运行整套134测试。使用既有小型合成fixture执行三种最小反例，每次三pair、M4/N2，单次进程约1秒，无GPU。

**结论：v2修复了原先的三类重要漏洞，但尚不能放行真实profile导入。存在4项P1：3个可接受错误证据的反例，以及正式原始JSON的大小兼容问题。** 当前collector_r2有额外重叠和clock门，其自身可继续收集；问题在于resolver声称独立重验但遗漏这些门，不能靠top-level passed代替。

## 已确认修复

1. 不再只信任correct SHA + passed：accepted entry需要measurement bundle、原始SQLite、保留事件表、六份原始应用数据、quality/numeric报告、日志和过程身份。device_ns和90个正式样本会从实际kernel区间重算，失败报告不能仅靠entry gate=true覆盖。
2. 有效硬件hash来自结构化内容，含compute/memory/cache/scheduling/placement；调用adapter必须有factory建立的实际phase/key/resource上下文。缺上下文、CPU需求冒充GPU、修改phase后复用旧上下文会拒绝。
3. 仅训练MMVQ main允许accepted，MMQ/conversion/fixup缺独立executed-work规则会拒绝；物理K、格式、stride、kernel符号和launch geometry都参与验证。没有打开planner默认应用。

## P1-1：未归属GPU重叠没有独立重算

`_correlate_kernel_samples`（约577–651行）只检查主链内部不重叠，以及同一调用API correlation归属的memcpy/memset。它不检查同设备、其他PID/TID/correlation的kernel或内存事件与被测NVTX区间重叠。collector_r2 `extract.py:98–105` 已增加这条规则，resolver未跟进。

最小反例：在既有fixture第一份SQLite追加一个同device、不同PID/correlation的kernel，区间[10205,10305]，与已归属主kernel相交。重新生成对应raw tables和文件SHA，保留原本quality报告的通过声明。`load_kernel_calibration`仍接受1条成本entry。

实际输出：`extra_overlapping_kernel_accepted=true`。

修复：逐NVTX调用扫描所有同设备捕获kernel/memcpy/memset，排除已归属主链后检查交集；清扫必须在区间之外。保留额外事件证据并拒绝cost eligibility。不能仅检查主链内部或只信任quality布尔值。

## P1-2：clock失败和freeze失败未进入独立门

`_validate_measurement_entry`约752–757行只检查过程`status=completed`和`returncode=0`，没有检查`freeze_before/freeze_after`。约772–775行只检查pair的numerics/chain/source/warning，忽略collector_r2新增`clock_domain_validated`和clock_domain_gates。bundle目前也无原始telemetry/clock条件关联。

最小反例：把所有pair `clock_domain_validated=false`，把六份receipt `freeze_after={passed:false}`，更新这些文件SHA；保持top-level quality通过。loader仍接受。

实际输出：`clock_false_and_freeze_after_false_accepted=true`。

修复：先强制每个pair全部必要gate明确为true、过程冻结明确通过；再绑定冻结expected SHA及其前后核验。为锁频参数迁移，bundle必须包含已冻结clock协议和对应telemetry原始证据，在loader独立重算正式区间两侧clock bracket、最大间隔和2400±30MHz条件，或复用一个冻结且哈希绑定的纯函数validator。至少不能允许已有明确false被忽略。

这不是说当前实际时钟失败，而是loader允许与报告内部条件矛盾的profile成为可用成本。

## P1-3：一对进程数据可复制为三对独立重复

约700–703行只验证pair编号集合{0,1,2}。每项pair可以指向完全相同的profile/direct app、SQLite、raw、receipt和日志。它会把相同30条kernel数据累计三次，得到90条样本并“通过”跨进程稳定性。

最小反例：将第一pair引用复制三份，只把pair编号改成0/1/2；quality对应列表也复制；numeric source列表重复三次。没有新增任何测量。loader仍接受。

实际输出：`one_pair_replayed_as_three_accepted=true`。

修复：三个pair需要不同的实际运行身份、路径、进程启动时间/创建标识、spec/receipt绑定；原始SQLite/app角色不能在不同pair重复使用。profile与direct也不能复用同一次app/receipt。PID可能合法复用，不能只用PID distinct判断，应结合启动时间和spec身份。样本唯一键至少绑定原SQLite SHA/路径/rowid，重复来源拒绝统计。保持相同输入SHA是正确的，不能混淆输入复用和运行复用。

## P1-4：统一16MiB上限会拒绝正式大N/M原始JSON

`_verified_bytes(..., keep=True)`约203行超过16MiB即抛出“profile or execution-contract document is too large”，但v2 loader约832行对`raw_application/raw_events/quality_report`也统一使用keep=True。

R16最大4096个数值检查点，每进程36次调用，另有first/final correctness，共38份numeric record。即使用所有数值都是单字符1的紧凑JSON，单行基础字段×4096×38也达到**28,327,936字节**；真实浮点长数字与完整字段只会更大。因此N4864/M1和多个M4训练配置会被拒绝，当前fixture仅M4/N2所以134测试没有发现。

修复：按证据kind设置明确且适合协议最大规模的上限，或流式解析并设总预算；不要取消所有上限。profile/owner可保留16MiB，raw app的上限应由冻结sample_count/calls和最大字段长度预算决定；同时避免同一config数据被多个entry无限次加载。新增不需巨量CPU的文件大小策略单元测试，边界检查可用稀疏/小测试阈值注入。

## P2与剩余适用限制

- `_trace_tables`使用`with sqlite3.connect(...)`不会自动close连接，Windows临时SQLite可能直到GC才释放。我第一次反例清理遇到WinError32；建议`contextlib.closing`或finally close，避免多entry长期持有大量句柄。这不是计时采集问题。
- `_validate_measurement_entry`的MMVQ源码资格只检查filename为mmvq.cu及文本中出现`ncols_x`、`mul_mat_vec_q`。这不足以独立证明源码归约规则；当前fixture甚至用仅含这两个词的注释通过。实际锁定源码的规则确为blocks_per_row_x=ncols_x/qk，再按逻辑块数循环；应验证具体源码anchor或绑定经过审查的规则版本，继续保留source/binary equivalence未证实。
- K_executed=K对本批Q5_0/Q8_0、K为32倍数的MMVQ main可表达逻辑有效归约，但不要称所有线程实际发射work均无padding。MMQ/conversion保持拒绝是正确的。
- factory能防止“旧context＋新query”误用，但不能自动知道调用方遗漏的硬件字段；接planner时必须交付完整实际硬件配置，不能手造四个非空字典当“完整配置”。此限制已有文档，不列为新漏洞。
- `quality.pairs`用dict覆盖重复编号，建议精确三项、无重复，并验证所有列表/统计分母。bundle pair独立身份修复应同时覆盖。

## collector_r2兼容性判断

基本schema（raw v2、Nsight表、protocol v1、现有quality）可衔接；不能据134个小fixture宣称真实bundle已通过：需要先修16MiB限制、collector clock门和独立重复门。bridge不得伪造时钟、执行K、硬件参数或将验证组写成training。若实际quality没有可用训练MMVQ main，输出“无可用标定”并保留全部失败分母。

## 本次反例总结果

```json
{"baseline":1,"extra_overlapping_kernel_accepted":true,
 "clock_false_and_freeze_after_false_accepted":true,
 "one_pair_replayed_as_three_accepted":true,
 "4096_rows_38_numeric_records_min_json_bytes":28327936}
```

未编辑`kernel_calibration.py`、测试或任何正在执行的collector文件。上述反例只改变临时目录中的合成证据；不会污染正式native或GPU矩阵。Root应通知实现代理优先修P1，再新冻结resolver/bridge后导入真实数据。
