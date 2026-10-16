# HBM 与 HBF 建模审查（c864890，含当前工作区）

后续状态：本报告保留**整改前**的审计事实和测量值。2026-10-15第一轮整改为执行版本9；随后按用户要求继续完善普通时序与刷新、参数来源与敏感性、映射与队列及调度，当前执行版本为10。修改范围、已解决项及保留边界见[整改方案与执行记录](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/docs/hbm_hbf_remediation_plan.md)，当前结果见[HBM参数与控制器敏感性报告](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbm_sensitivity_20261015/report.md)，[版本9验证报告](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbm_hbf_remediation_20261015/report.md)保留为历史。下文不代表整改后的当前行为。

审计日期：2026-10-15。对比基线：样例报告中的 `5a13823`。当前提交：`c8648904ea7c071e14759a9071c7f8cabd141f44`。

本次补订：加入 **JEDEC 与 OCP v0.7.0 原文逐项对照**，并纠正 GC、访问粒度和协议责任边界的表述。代码版本与工作区范围未变；下文的测试和微测数据来自首次审计，本次补订没有重新运行实验。

本次按用户提供的审查结构，重新检查配置入口、地址映射、HBM/NAND 执行路径及保存的实验结果。附件是审计范围与写法参考，附件中的结论不直接视为事实。当前工作区另有 `planner.py` 和 `test_current_lowering_callers.py` 的未提交修改，已纳入检查；本次没有修改模拟器实现。

## 结论速览

**当前版本仍不能直接用仓库中的正式实验结果证明新 HBM 模型的性能。另确认一个 HBM 时钟对齐缺陷：内部浮点误差会被放大成多等待一个完整时钟周期。建议先修复它，再确定模型参数，最后重跑正式实验。**

HBM 的命令、阵列和数据通路已经分开建模，但主实验仍使用由容量和带宽预算反推的等效时钟，刷新关闭，多项器件时序未配置。HBF 主实验仍受介质并行度限制：接口读上限写成 1600 GB/s，介质读理论上限却只有 262.144 GB/s；这不是同等有效带宽的 HBM/HBF 对照。

本报告区分四类结论：**代码缺陷**是实现没有准确执行自身规则；**结果有效性问题**是保存结果不能支持当前版本结论；**建模假设**是代码明确采用、但尚未标定的选择；**未覆盖行为**是当前模型没有实现的器件机制。后两类不自动等同于代码错误。

| 项目 | 上次报告（5a13823） | 当前核查 | 判断 |
| --- | --- | --- | --- |
| HBM 几何 | 标准组织，32 B burst、1 KiB 有效页 | 主实验仍为 8 stacks、256 PC、每 PC 64 bank | 未发现新的容量或几何错误 |
| 命令、行时序、固定 RL/WL | 已实现 | 实现仍在；74 项 HBM 定向测试通过 | 有对应实现，测试覆盖部分行为；不是完整协议验证 |
| 时钟对齐 | 未指出缺陷 | 17.92 + 10.24 ns 被向上对齐至 30.72 ns，而非 28.16 ns | **本次新确认的代码缺陷** |
| 容量与时钟耦合 | 512 GiB：2.56 ns；1 TiB：5.12 ns | 通过实际配置入口复算一致 | 假设仍在；容量实验同时改变频率和并行度 |
| HBM 顺序读 | 按代码推算约 55% | 少量冷顺序读实测约 48%–52% | 55% 只能保留为局部理论估算 |
| HBM 刷新 | 32 Gb × 16H 的 tRFC 未知，关闭 | 仍关闭；tRASmax 仍单独执行 | 仍未覆盖数据保持成本 |
| tRRD / tFAW / tRTP / tWTR / tRTW | 主实验为 0 | 仍为 0，来源明确标为未知、未建模 | 未补齐 |
| 调度与队列 | 按提交顺序、开页、设备总并发上限 | 仍如此 | 控制器策略简化 |
| 正式实验结果 | 旧模型结果 | 三个主要目录 379 个 JSON，未找到当前版本正式结果 | **不能直接引用为当前性能** |
| HBF 读带宽 | 介质上限约 262 GB/s | 理论 262.144，64 MiB 微测 261.474 GB/s | 旧结论成立 |
| HBF 写带宽 | 未突出 | 理论 13.981，64 MiB 微测 13.793 GB/s | 补充：也低于接口配置的 16 GB/s |
| HBF 缓存 | “没有 cache buffer 命中” | 无持久读页缓存；有脏写暂存命中 | 旧表述需要限定 |
| HBF 缓冲写 | 上次称行为未变 | 正缓冲容量统一启用页合并和主机早确认 | 有行为变化；主实验缓冲为 0，不受此项影响 |
| HBF 写入与接口 | 无自动擦除、单个共享主机资源 | 仍成立 | 协议覆盖不足，影响取决于工作负载 |
| HBF 页与主机访问 | 未逐项对照 OCP | 主实验 16 KiB 页、主机按 4 KiB 计费；OCP 页为 4 KiB，主机读基本粒度为 64 B | 必须分层比较，当前不是 OCP 页/事务模型 |
| HBF GC 与磨损均衡 | 原报告未充分区分 | OCP 不提供设备内 GC；现有擦除计数也不等于分区重映射 | 缺少设备内 GC **不属于 OCP 缺项** |

## 协议基线与判断口径

| 对象 | 本次直接使用的协议文件 | 对照范围与页码口径 |
| --- | --- | --- |
| HBM 主实验 | [JEDEC JESD238，January 2022](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/.refs/jedec/JESD238-2022.pdf) | 用 HBM3 规则核查当前标为 HBM3E 的参考组织；以下 J 表全部采用**标准印刷页码**，PDF 页序比印刷页多 14 |
| HBF | [OCP High-Level Base Die Specification v0.7.0，2026-08-03，官方文件](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)；[本地原文](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/.refs/ocp/OCP_HBF_v0.7.0_20260803.pdf) | 以下 O 表采用**原文印刷页码**，与 PDF 页序相同；核查普通闪存访问、维护和接口抽象 |

“语义一致”只表示该行所列规则的计时起点、作用域或结构对应；“数值未标定”表示使用分析假设或参考编码；“未启用”表示有实现但主实验没有执行；“未覆盖”表示缺少相应模型；“偏离”表示当前行为与所列协议不同。**任何单项一致都不等于整个器件符合性认证。** 可选功能、厂商定义的参数和主机职责会单独标出。

本项目还包含 HBM2/E、HBM4 组织。本次细表围绕主实验 HBM3E 使用 JESD238，不把其命令长度、RL/WL 编码或 tRFC 表套用到其他代际，也不把“HBM3E”标签视为已核验某个厂商速度档。

## 实验结果仍不能代表当前 HBM 模型

核查范围是 `artifacts/hbf_hbm`、`artifacts/hbf_hbm_rerun_20261014`、`artifacts/hbf_hbm_unified_20261014` 三个主要实验目录。它们分别包含 220、17、142 个 JSON，共 379 个；并非每个 JSON 都是一个独立实验点。

| 证据 | 当前检查结果 |
| --- | --- |
| 保存的 point 版本 | 递归出现的 `point_result_version` 只有 2、3、4，没有当前的 8 |
| unified 中的 HBM 执行配置 | 仍是旧的 8 KiB 行、64 B burst；代表主配置为 8 个物理通道 × 2 PC、每 PC 1 BG × 8 bank |
| 当前运行入口 | serving 使用版本 8；decode 合约为 `physical_core_unified_hardware_catalog_v3/serving-points-v8` |
| 旧缓存是否会静默复用 | 当前入口检查版本或执行合约，不满足时不会当作合格缓存复用 |
| 主结果文件自身说明 | `artifact_note` 明确说明保留历史时间、吞吐和字节结果，仅清理字段、重算费用、重绘报告，没有重新仿真 |

直接证据可见 [unified 代表点的实际 HBM 配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm_unified_20261014/decode_points/llama3_1-8b-131072-A_hbm_only-64.json:3652)、[当前 serving 版本](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_serving_study.py:24)、[缓存接受条件](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_serving_study.py:368) 和 [decode 执行合约](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_decode_study.py:27)。主结果文件为 [results.json](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm/results.json)，应查看其顶层 `artifact_note`。

旧报告拿两个相近的 TPOT 数字辅助判断模型没变，这个论据不够强，两个点的输出长度不同：

| 8B、128K 上下文 | 输出 token 数 | A：全 HBM 的 TPOT p50 | B：KV 放 HBF 的 TPOT p50 |
| --- | ---: | ---: | ---: |
| rerun / study_points | 4 | 43.561209 ms | 78.573625 ms |
| unified / decode_points | 64 | 43.567186 ms | 78.577141 ms |

**本次采用保存的配置、版本和执行合约作为依据，而不是靠数值相似下结论。** 上表只能描述历史运行，不能据此认定当前 A/B 比值保持不变。两个方案都可能使用 HBM 存放权重，HBM 时序变化会同时影响它们，比例如何变化需要重跑。

这里的“没有当前结果”限定为上述正式实验结果集。项目已有当前模型的单元测试，本次也做了小规模复现；不能写成“新模型从未运行过”。

独立的 HBF 机制实验也不会仅因 HBM 换了模型就自动失效。需要重点重跑的是用于说明当前完整服务、吞吐和 A/B 比值的结果；其余历史实验应按各自使用的配置判断适用性。

## HBM 数据通路分层

| 层次 | 当前实现 | 尚存边界 |
| --- | --- | --- |
| 上层请求 | 按 burst 拆分；按整个设备控制在途请求数 | 不是每通道的真实事务队列 |
| 地址映射 | stack / channel / PC / BG / bank / row | 固定线性映射，未标定真实控制器散列 |
| 行命令口 | 每物理通道一个，两个 PC 共用；ACT、PRE、REF | 不含完整初始化、训练、可选命令 |
| 列命令口 | 每物理通道一个，两个 PC 共用；RD、WR | 按提交顺序预约，不重排回填 |
| 阵列状态 | 开行、tRCD、tRAS、tRP、tRC、组时序、tRASmax | 部分时序在主实验为 0 |
| 数据总线 | 每 PC 独立数据位段；固定 RL/WL；2 CK burst | 缺少完整 PHY 换向极值 |
| 维护 | 支持周期性 REFab 和最大开行时间 | 主实验 REFab 关闭；REFpb/RFM 未实现 |
| 存储单元 | 容量与地址组织 | 不模拟电荷、实际错误或保持失效 |

命令口占用、阵列依赖和数据总线占用是不同的量。不能把所有资源 busy time 相加当作请求延迟，也不能因总线不重叠就认定所有器件时序已经满足。

## HBM：主实验配置对比

以下数值来自当前 `_hbm_physical_config` 返回的对象，采用读写均为 1600 GB/s、未单独指定物理接口速率的主实验输入。

| 参数 | 512 GiB 配置 | 1 TiB 配置 | 说明 |
| --- | ---: | ---: | --- |
| stack 数 | 8 | 16 | 按容量选择覆盖组织 |
| 每 stack 层数 × die 密度 | 16H × 32 Gb | 16H × 32 Gb | 参考组织，非指定厂商料号 |
| 每 stack 物理通道 / PC | 16 / 32 | 16 / 32 | 每物理通道 2 PC |
| 总 PC | 256 | 512 | 独立数据通路数 |
| 每 PC 的 BG × bank | 16 × 4 | 16 × 4 | 每 PC 64 bank |
| 有效行数 / bank | 32768 | 32768 | 没有靠任意加行数凑容量 |
| 有效页 / 激活页 | 1 KiB / 2 KiB | 1 KiB / 2 KiB | 半页由行地址位选择 |
| burst | 32 B | 32 B | 每次占 2 CK |
| 聚合接口带宽 | 1600 GB/s | 1600 GB/s | 不随 stack 数自动增加 |
| 每 PC 带宽 | 6.25 GB/s | 3.125 GB/s | 聚合值均分 |
| 等效引脚速率 | 1.5625 Gb/s | 0.78125 Gb/s | 由预算推导，非器件测量 |
| tCK | 2.56 ns | 5.12 ns | 容量变化同时改变时钟 |
| RL / WL | 81.92 / 20.48 ns | 163.84 / 40.96 ns | 32 / 8 CK |
| burst 占用 | 5.12 ns | 10.24 ns | 2 CK |
| tRP / tRAS / tRC | 15.36 / 33.28 / 48.64 ns | 15.36 / 35.84 / 51.2 ns | IDD 参考量化值 |
| 自动刷新 | 关闭 | 关闭 | tREFI、tRFC 实际字段均为 0 |
| 最大开行时间 | 35100 ns | 35100 ns | 独立于自动刷新执行 |

入口与来源：[实验配置入口](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/calibrated_hbf_profiles.py:407)、[容量与组织选择](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/physical_presets.py:212)、[带宽、时钟与 RL/WL 推导](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:139)。

关键判断有三点：

1. **这是容量、并行度和频率一起变化的实验。** 512 GiB 增加到 1 TiB 时，bank/PC 数增加而时钟减半；不能把差异完全归为容量收益。
2. **慢时钟不等于已经证明违反 JEDEC。** 当前元数据明确声明这是等效调度速率、不是厂商速度等级。问题在于它没有被标定为目标硬件工作点，不能把名称 HBM3E 当作真实时延保证。见 [参数来源标签](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:333)。
3. **物理接口和方向限速已经分开。** 只有输入缺少接口信息时，实验入口才用读写预算最大值充当接口假设；不能泛化为“修改任何读写限速都会修改时钟”。

## HBM：一次访问的实际流程
1. 请求按 32 B 切成 burst；非对齐访问也按完整 burst 计数据传输。
2. 地址先轮转 PC，再换 bank，再换列、行；stack 由 lane 所属范围确定。
3. 发令前估算完成相关恢复的时间。如果会越过已开行的 tRASmax，先安排 PRE；刷新启用时检查 REFab 到期。
4. 行命中直接走列命令。未开行发 ACT；行冲突先 PRE、等 tRP，再 ACT，同时遵守 tRAS 和 tRC。
5. ACT 从上升沿开始，占行命令口 1.5 CK；tRCD 等阵列时序从第二个上升沿计算。
6. RD/WR 等待 tCCD、跨 SID 读间隔及已配置的读写依赖；同时推迟发令，使固定 RL/WL 后的数据恰好能进入数据总线。
7. 32 B 数据占该 PC 的 2 CK。方向限速调节发射节拍，不拉长物理 burst。
8. 读后 PRE 的下限从 RD 起算；写后 PRE 的下限从写数据结束加 tWR 起算。
9. 请求在最后一个 burst 数据结束时完成，写请求完成时间不额外加 tWR。

对应实现：[地址映射](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:26)、[命令与阵列时序](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:103)、[主执行循环](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:394)。

“先来先服务”在这里指按提交顺序预约资源，而不是全设备只有一个请求串行执行。不同资源仍可重叠，后提交的独立请求也可能先完成；但控制器没有 FR-FCFS、读写批处理或空闲时段回填。

## HBM：本次确认的时钟对齐缺陷

**类型：代码缺陷。处理优先级：正式性能重跑之前修复。**

位置：[明细路径 align](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:80)、[编译路径 align](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_numeric.py:28)。两处都先判断值是否恰好等于一个浮点表示的时钟边沿，否则使用向上取整。

主实验 tCK=2.56 ns。同一行连续读的同组间隔是 4 CK，即 10.24 ns。第二次列命令的计算出现如下结果：

```text
第一次 RD：             17.92 ns = 7 CK
应满足的下一次 RD：      (7 + 4) CK = 28.16 ns
浮点相加实际值：         17.92 + 10.24 = 28.160000000000004
当前向上对齐：           30.72 ns = 12 CK
多等：                   2.56 ns = 1 CK
```

最小复现：用上节的 512 GiB 配置创建新的 DramCore，依次提交三个地址为 0、长度 32 B、到达时间为 0 的读请求。

| 请求 | 实际 RD 时刻 | 实际完成时刻 | 判断 |
| --- | ---: | ---: | --- |
| 第一次 | 17.92 ns | 104.96 ns | 建立开行与数据流水线 |
| 第二次 | 30.72 ns | 117.76 ns | RD 本可在 28.16 ns 发出，完成约 115.20 ns |
| 第三次 | 40.96 ns | 128.00 ns | 延续已经发生的偏移 |

第二次请求没有别的约束要求等到 30.72 ns：列命令口只忙到 20.48 ns；数据总线反推的最早 RD 约为 23.04 ns；主约束就是 28.16 ns 的同组间隔。明细和编译路径得到相同的错误时刻。

最近的修改已经增加“恰好处于网格点时不再上取整”的保护，但未覆盖合法整拍时间**相加**后产生的尾差。本次是发现尚未处理完的情况，不据此断言该错误由最近三个提交首次引入。

修复建议：内部时钟事件尽量使用整数拍或半拍运算，或显式区分内部整拍推导值与外部到达时间。不要简单给所有时间加宽容差，因为真实晚于边沿到达的请求不能被提前发出。现有测试验证了两路径一致和外部边界行为，但两路径使用同样的算法，一致性检查不能排除共同错误。

可复现输入：

```python
import sys
sys.path[:0] = ['src', 'scripts']
from calibrated_hbf_profiles import _hbm_physical_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest

c = _hbm_physical_config(capacity_bytes=512 * 1024**3,
                         read_gb_s=1600., write_gb_s=1600.)
for details in (True, False):
    core = DramCore(c, capture_details=details)
    for i in range(3):
        result = core.submit(AccessRequest(str(i), 'read', 0, 32, 0.))
        print(details, i, result.completion_ns,
              [(s.name, s.start_ns) for s in result.stages
               if s.name == 'READ_COMMAND'])
```

## HBM：地址映射与实际顺序读带宽

同一 PC 内，线性地址先访问同一 BG 的四个 bank，再进入下一个 BG。这个映射使同组 tCCD_L 经常成为约束，结论仍成立。见 [bank/BG 分解](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:57) 和 [列命令历史约束](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:153)。

旧报告约 55% 的来源可以复算：一个 64-bank 循环含 48 次同 BG 的 4 CK 间隔、12 次同 SID 异 BG 的 2 CK 间隔、4 次跨 SID 的 4 CK 间隔，局部平均间隔为 3.625 CK；每 burst 数据占 2 CK，故为 `2 / 3.625 = 55.17%`。

但这忽略行操作、启动、维护与数值对齐等开销。当前实际程序的少量微测如下：

| 可见容量 | 单次冷顺序读 | 完成时间 | 有效带宽 | 占 1600 GB/s |
| --- | ---: | ---: | ---: | ---: |
| 512 GiB | 1 MiB | 1364.48 ns | 768.480 GB/s | 48.03% |
| 512 GiB | 16 MiB | 20349.44 ns | 824.456 GB/s | 51.53% |
| 512 GiB | 64 MiB | 86405.12 ns | 776.677 GB/s | 48.54% |
| 1 TiB | 64 MiB | 85688.32 ns | 783.174 GB/s | 48.95% |

每个点使用新建 DramCore，地址 0、一个 READ、到达时间 0。配置保留入口默认值；汇总执行完成全部 burst，1 MiB 另用明细路径检查，完成时间一致。64 MiB 的 burst 数超过明细展开上限，但不会因此停止计时或少算数据。

这些是**未修复对齐缺陷的当前程序**测量值，不是硬件实测，也不是修复后的预测。不能把 55% 当作已测吞吐，也不能把表中所有差额都归给地址映射。真实控制器的映射和重排策略需要具体依据，不应笼统声称所有真实控制器都采用同一种 BG 低位映射。

## HBM：JEDEC 原文逐项对照

以下 J 表直接对照 [JESD238 原文](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/.refs/jedec/JESD238-2022.pdf)。当前值指 512 GiB、1600 GB/s、tCK=2.56 ns 主实验；nCK 表示周期数。所有“已实现”判断仍受前述浮点对齐缺陷的限制，不能解读为逐次发令都已正确。

### J1. 几何、接口与命令

| 项目 | JEDEC 要求或允许值 | 原文章节 / 表 / 印刷页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| channel / PC | 每 stack 最多 16 channel；每 channel 两个 32-bit PC | §3 p.2；§3.1.2 p.4 | 16 channel/stack、32 PC/stack；8 stacks 共 256 PC。[配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:267) | 组织可对应标准；8 stacks 是实验选择 |
| 32 Gb × 16H 组织 | 每 PC 16 Gb；RA[14:0]、CA[4:0]、SID[1:0] 与 BA[3:0] | §3.2，表4 p.6 | 每 PC 64 bank、每 bank 32768 有效行、1 KiB/行；单 stack 64 GiB。[组织表](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_dependencies.py:53) | 容量与寻址相符；没有重复乘 die 数 |
| bank group | 64 bank 分为 16 BG，每 BG 4 bank；SID 扩展 bank 地址 | §3.2.1，表5/6 p.7 | 16×4，保存 BG 与 SID 历史。[列时序](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:153) | 组织、作用域对应 |
| 有效页 / 激活页 | 有效页 1 KiB；RA 最高位选择已激活 2 KiB 页的一半 | 表4注1–3 p.6 | 清除该行地址位形成开行身份。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:405) | 语义一致；不是合并相邻两个行号 |
| burst / prefetch | 256-bit prefetch；32 DQ × BL8 = 32 B | §2 p.1；§6.3.3.2 p.57、§6.3.3.3 p.67 | 每段收取完整 32 B 传输。[拆分](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:118) | 粒度对应 |
| PC 共享与独立状态 | 同 channel 的两个 PC 共享命令口、CK，阵列时序分别计量 | §3.1.2、表3 **p.4** | 命令资源按 stack/channel，状态与历史按 PC。[资源及状态](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:49) | 作用域对应 |
| 行 / 列命令口 | 两套接口可并行工作 | §3.1.3 p.5 | 分别预约 row 与 column 资源。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:75) | 结构对应，不是一个串行总线 |
| CK / DQ 关系 | DQS 为 CK 两倍频、双边沿传数；4 UI/CK，BL8 占 2 CK | §6.1 p.26 | 每 PC 16 B/CK；burst 5.12 ns。[推导](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:139) | 比例对应，未建完整电气波形 |
| 时钟范围 | 表92列 4.8–6.4 Gb/s 档；6.4 档 tCK=0.625–20 ns，其余档上限也为20 ns | §10，表92 p.159；表93注2 p.164 | 当前 tCK=2.56 ns；1 TiB 时5.12 ns。[时钟公式](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_dependencies.py:83) | 未超该表时钟范围；不能因慢于0.625 ns判违规，也未证明器件支持整套参数 |
| ACT | 上升沿开始，R/F/R 占1.5 CK；阵列时序从第二个上升沿起算 | §6.3 p.34；表30 p.35；§6.3.2.2 p.38–39 | ACT占1.5 CK，anchor=开始+1 CK。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:118) | 命令与阵列起点对应 |
| PREpb / PREab | 占0.5 CK，可在上升/下降沿发令；tRP 从该边沿起算 | 表30 p.35；§6.3.2.3 p.41–43 | 按半拍网格发PRE，恢复从命令开始计算。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:103) | 语义对应；网格数值运算有已确认缺陷 |
| RD / WR | 上升沿开始，R/F 占1 CK，BL8 不可截断 | 表31 p.36；§6.3.3.2 p.57、§6.3.3.3 p.67 | 列命令1 CK，完整32 B数据段。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:449) | 命令骨架对应，不含全部PHY细节 |
| 主机地址交织 | 定义器件地址字段和BG，未在这些条款固定系统交织算法 | §3.2 p.6；§3.2.1 p.7 | PC→bank→列→行。[映射](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:26) | 控制器选择；性能限制不等于JEDEC违规 |

### J2. 阵列、列访问与数据时序

| 参数 | JEDEC 要求或允许值 | 原文章节 / 表 / 印刷页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| RL | MR2 可编码4–63 nCK；厂商支持的连续子区间另定 | §5，表13 p.19 | RL32，81.92 ns。[配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:151) | 编码范围内；器件支持值未标定 |
| WL | MR1 可编码4–16 nCK；厂商支持子区间另定 | §5，表12 p.18 | WL8，20.48 ns。[配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:152) | 编码范围内；器件支持值未标定 |
| 首个有效数据 | 读为 RL×tCK 加 tDQSS、tWDQS2DQ_O、tDQSQ；写为 WL×tCK 加 tDQSS | §6.3.3.2 p.57；§6.3.3.3 p.67 | 直接用命令+RL/WL；未逐项建PHY偏移。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:456) | **只有固定周期流水线，未覆盖完整首数据时序** |
| tRCDRD / tRCDWR | ACT阵列起点到RD/WR；AC最小数值留空，单位ns | §6.3.2.2 p.38；表93 p.161 | 从第二上升沿分别加14 ns。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:145) | 起点对应；14 ns为分析假设 |
| tRAS(min) | ACT→PRE最短间隔；AC最小值留空；IDD **示例**为33 ns | §6.3.2.2 p.39；表93 p.161；表84 p.152 | 参考值量化为33.28 ns。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:147) | 有门；**IDD示例不是AC保证** |
| tRP | PRE到下一行访问；AC最小值留空；IDD示例15 ns | §6.3.2.3 p.41；表93 p.161；表84 p.152 | 量化为15.36 ns。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:114) | 起点对应；数值未标定 |
| tRC | 同bank连续ACT下限；AC最小值留空；IDD示例48 ns | §6.3.2.2 p.39；表93 p.161；表84 p.152 | 独立门48.64 ns，并满足tRAS/tRP。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:123) | 有门；数值未标定 |
| tRRD_S / tRRD_L | 同PC、不同bank的ACT按异/同BG限间隔；AC数值留空、单位ns | 表3 p.4；表6 p.7；§6.3.2.2 p.39；表93 p.161 | 两者0；非0时有对应逻辑。[执行](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/dram_core.py:185) | 主实验未启用；留空不代表可用0 |
| tFAW | 滚动窗口最多4次激活；计入规定的REFpb；数值留空、单位ns | §6.3.2.2.1 p.41；表93 p.161、注10 p.164 | 0；非0时限制ACT，REFpb未实现。[窗口](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/dram_core.py:191) | 主实验未启用，REFpb部分未覆盖 |
| tRTP | 同bank RD→PRE，AC单位**nCK**、最小值留空 | 表33 p.44；§6.3.3.2.4 p.65；表93 p.161、注11 p.164 | 内部字段用ns，当前0；从RD起计。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:465) | 计时起点对应，恢复门未启用 |
| tWR | 从写burst结束起算；WR→PRE包括WL、BL/4、tWR；AC数值留空、单位ns | 表33 p.44；§6.3.3.3.3 p.73；表93 p.161 | 写数据结束+15 ns才可PRE；数据结束即可完成请求。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:467) | 恢复起点对应；15 ns为分析假设 |
| tCCD_S | 同方向、异BG列命令至少2 nCK | 表6 p.7；表93 p.161、注15–16 p.164 | 5.12 ns。[列门](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:159) | 已配置；参数满足所列下限 |
| tCCD_L | 同方向、同BG至少max(4×tCK,2.5 ns) | 表93 p.161、注13–14 p.164 | 10.24 ns。[列门](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:159) | 下限配置正确；实际对齐可能额外加拍 |
| tCCDR | 跨SID连续读替代tCCD_S；厂商值，支持tCCD_S+1 CK至tCCD_S+2 CK；跨SID写仍用tCCD_S | 表6注2 p.7；表93注17 p.164 | 参考选4 CK=10.24 ns，仅跨SID读采用。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:162) | 支持范围内的参考选择；非统一厂商值 |
| tWTR_S / tWTR_L | 按异/同BG；AC单位**nCK**、数值留空；WR→RD为WL+2+tWTR周期 | 表6 p.7；表33 p.44；§6.3.3.3.3 p.72；表93 p.161 | 字段用ns，均0；RD仍等待写数据结束。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:165) | 额外恢复门未启用，并非允许数据重叠 |
| tRTW | 系统换向约束，由RL、BL/4、WL及PHY参数推导，不是一个统一器件常数 | §6.3.3.2.4 p.64；表93注18 p.164 | 命令额外门0，数据资源防重叠；无完整PHY公式。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:170) | 完整换向约束未覆盖 |
| tPPD | 同PC连续PRE至少2 nCK | 表33 p.44；表93 p.161 | 5.12 ns，按PC保存历史。[实现](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:99) | 参数与作用域对应 |
| 行时序取整 | tRAS/tRTP/tWR/tRP允许0.5×ceil(2t/tCK)半拍量化；后继ACT仍对齐上升沿 | **§6.3.2.4 p.45** | PRE用半拍；参考tRAS/tRP先整拍量化；存在浮点尾差加整拍。[对齐函数](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:80) | 整拍等待可保守合法；额外数值误差属于实现缺陷 |

完整的 JESD238 注18换向表达式为：`(RL + BL/4 − WL + tDQSS(min) + 0.5)×tCK + tWDQS2DQ_O(max) + tDQSQtra(max)`，结果向上取到可发令周期。括号中的`tDQSS(min)`按CK归一化；若输入为ps/ns，须先除以tCK，不能直接把时间与周期数相加。当前不能因“DQ不重叠”就宣称该式已满足。

另须区分：表16 p.20 的 MR5 RTP 编码2–15 CK服务于**自动预充电设置**，不能拿它直接替代显式PRE所需、尚未取得的器件tRTP。表84 p.152的33/15/48 ns标为IDD测量示例；本次已经核对其表头，未当作表93的AC保证值。

### J3. 刷新与维护

| 项目 | JEDEC 要求或允许值 | 原文章节 / 表 / 印刷页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| REFab | 上升沿半周期命令；选中PC全部bank先PRE+tRP，之后阻塞tRFCab；另一PC仍有独立状态 | **§6.3.2.5 p.46–47**；表30 p.35 | 支持PREab、tRP、REFab、PC阻塞；主实验关闭。[执行](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:265) | 命令骨架存在，主实验无刷新成本 |
| tREFI | 平均REFab间隔上限3.9 µs | §6.3.2.5 p.47；表93 **p.163** | 已知tRFC组织默认3900 ns；主实验字段0、disabled。[配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/hbm_geometry.py:295) | **关闭刷新是未建模，不是标准允许免刷新** |
| 刷新延期 / 提前 | 最多欠8次，绝对间隔≤9tREFI；任意tREFI内至多9次 | §6.3.2.5 p.47；表93注23 p.164 | 周期性策略，不主动优化延期/提前；过due+8tREFI报错。[执行](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:300) | 简化调度；主实验未启用 |
| tRFCab | 按密度/层数查表；32 Gb×16H为TBD，16 Gb×16H为450 ns | 表93 **p.162** | 对主实验返回未知，再禁用刷新；未借用HBM4数字。[查表](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_dependencies.py:69) | 来源处理透明，但保持成本未覆盖 |
| tRASmax | ACT→PRE最大9tREFI | 表93 p.161 MAX列 | 使用**参考**tREFI=3900 ns得35100 ns，独立于刷新开关执行。[关闭计划](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:206) | 有最大开行门；不是当前0字段的九倍，也不能替代刷新 |
| REFpb | 逐bank刷新，需PRE+tRP和覆盖轮转；其他bank可工作 | **§6.3.2.6 p.48–50** | 支持命令子集没有REFpb。[范围](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:1) | 未覆盖该维护路径 |
| tREFIpb / tRFCpb / tRREFD | tREFIpb=tREFI/N；32 Gb的tRFCpb为TBD；tRREFD至少max(3tCK,8 ns) | §6.3.2.6 p.48–50；表93 p.163 | 无这些执行门；主实验N=64时公式平均间隔为60.9375 ns | REFpb配套机制未覆盖，不能把TBD当0 |
| RFMab / RFMpb | 由DEVICE_ID.RFM决定是否需要；需要时跟踪RAA与厂商阈值；不能替代普通REF | **§6.3.2.7 p.52–53**；表93注29 p.165 | 无RFM/RAA执行。[支持范围](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_hbm_control.py:1) | 未覆盖；不能说所有器件都强制需要 |
| ARFM | 可选，MR8选择厂商公布的RFM阈值等级 | **§6.3.2.8 p.54–55**，表36/37 | 未实现 | 可选机制未覆盖，非一概强制缺项 |

表93 p.161的空白最小值、p.162的TBD均已对原表核查。**空白/TBD不等于零。** 此处页码和章节按原文纠正：PC独立计时表3在p.4；§6.3.2.4是取整规则，REFab是§6.3.2.5。它们不能沿用项目说明中的错位引用。

## HBM：控制器层行为与影响

| 行为 | 当前实现 | 对结果的影响 |
| --- | --- | --- |
| 页策略 | 开页，受到 tRASmax 与已启用刷新约束 | 行局部性影响显著 |
| 调度 | 按提交顺序预约资源，无重排和回填 | 混合流量或单请求很大时可能低估可利用并行度 |
| 队列 | 设备统一在途请求上限，按上层请求计 | 请求大小改变队列含义；不能直接等同每通道事务深度 |
| 地址映射 | PC → bank → 列 → 行 | 同 BG 连续访问受到长列间隔限制 |
| 方向限速 | 限制各方向起始节奏，物理 DQ 半双工 | 方向预算与接口峰值不能混为一个数 |
| 汇总加速 | 单独的 HBM 编译实现；不支持的加速情况走同一 HBM 参考控制器 | 不是切换到旧分析式 DRAM；仍可能与参考路径共同犯错 |

队列依据：[DramCore 准入控制](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/dram_core.py:270)。汇总路径边界：[编译入口](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/_compiled_hbm.py:17)。

流式读同时受映射、列时序、行维护和对齐缺陷影响。随机访问增加了 bank/行命令约束，但 ACT 频率门仍缺失。混合读写缺少真实 PHY 参数。上述偏差方向不同，不能简单归纳为整个 HBM 模型“一定偏乐观”或“一定偏悲观”。

## HBF：当前配置与实际瓶颈

主实验由 [synthetic_profile](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_experiments.py:61) 给出输入，再经过 [统一 HBF 预设](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/physical_presets.py:344) 和 [HBF Core 构造](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/calibrated_hbf_profiles.py:447)。

| 参数 | 当前主实验 |
| --- | ---: |
| 等效阵列单元 / 内部通道 | 64 / 64 |
| NAND 页 | 16 KiB |
| 主机计费 / 向外取整粒度 | 4 KiB（不是 OCP 主机事务粒度） |
| 页读 tR / 编程 tPROG | 4 / 75 µs |
| 显式擦除耗时 | 3 ms |
| 主机接口读 / 写上限 | 1600 / 16 GB/s |
| 单内部通道带宽 | 1,600,000 GB/s，刻意弱化内部传输瓶颈 |
| 主机写缓冲 | 0，写穿模式 |
| 最大在途主机请求 | 32 |
| 部分页写 | RMW：读旧页再写整页 |
| 磨损模型 | 主实验关闭 |

这里的 64 个内部等效单元不等于 64 个真实主机通道。默认主机接口另用一个共享资源建模。

介质上限由并行度、页大小和阵列延迟共同决定：

```text
读：64 × 16384 B / 4000 ns  = 262.144 GB/s
写：64 × 16384 B / 75000 ns = 13.981013 GB/s
```

只按当前页大小和读延迟估算，支撑 1600 GB/s 至少需要 391 个等效读单元，还没有计其他开销。主实验显式选 64，因而接口上限远不是可兑现的介质吞吐。

新建 NandCore、512 GiB 容量、地址 0、一个 64 MiB 整页顺序请求的结果为：

| 操作 | 完成时间 | 有效带宽 | 计数 |
| --- | ---: | ---: | --- |
| 64 MiB 读 | 256656.01536 ns | 261.473957 GB/s | 4096 次页读 |
| 64 MiB 写 | 4865536.65536 ns | 13.792695 GB/s | 4096 次编程、0 次擦除 |

这两个点分别运行明细和汇总路径，结果仅有浮点末位差异。它们支持“当前配置被阵列限制”的判断；不能用它们代替完整 LLM 吞吐或持续写实验。

## HBF：一次访问的实际流程

1. 整个上层请求先经过设备级在途数量限制。
2. 地址按 NAND 页轮转内部单元，拆成页操作，不经过 FTL。
3. 读先执行 PAGE_READ，再整页内部传输，最后经共享主机资源传回请求数据。
4. 整页写先完成主机与内部传输，再 PAGE_PROGRAM，写穿模式返回编程完成时刻。
5. 部分页写执行 RMW，再编程完整物理页；没有检查单元内容、顺序写入是否合法或是否需要先擦除。
6. 只有显式 ERASE 请求触发擦除。普通写和 RMW 不自动插入擦除。
7. 若 `write_buffer_bytes > 0`，先进入主机暂存、提前确认，重叠页可合并；容量压力或显式 flush 再安排介质写入。
8. 读已暂存的脏数据可命中；部分覆盖时读取介质并合并暂存内容。这与保存已读 NAND 页的读缓存不同。

依据：[NandCore 模型范围](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:33)、[擦除与写暂存路径](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:615)、[读与编程路径](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:669)。

## HBF：OCP v0.7.0 原文逐项对照

以下 O 表直接对照 [OCP 官方原文](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf) 与[本地 PDF](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/.refs/ocp/OCP_HBF_v0.7.0_20260803.pdf)。p. 均为原文页码；比较对象是普通 HBF 闪存路径。上层 `AccessRequest` 可以聚合多个事务，**请求大于 4 KiB 本身不证明报文违规**；需要检查模型有没有拆分、计费及维护对应协议状态。

### O1. 通道、访问粒度与带宽

| 项目 | OCP 规则或定义 | 原文章节 / 表 / 页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| 主机通道 | 每 stack 最多16；独立、全双工 | §4、§4.1 p.15 | 每设备一个默认host资源，读写共用；没有channel/direction维度。[资源ID](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:120)、[默认单lane](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_transfer.py:109) | 未覆盖接口拓扑与双工性；64内部单元不是64主机通道 |
| NAND 页 | 4 KiB | §4.1 p.15；§4.2 表3 p.16 | 主实验16 KiB；HBF-2参数组4 KiB。[主预设](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/physical_presets.py:61) | 主实验是不同的等效页配置，偏离本版页几何 |
| 主机读 | 64 B对齐，64 B–4 KiB，不跨4 KiB | §4.1 p.15；§5.3.1 第1、4项 p.56 | 按16 KiB介质页拆分；主机字节按4 KiB向外取整；无64 B子事务和4 KiB协议边界。[映射](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:177) | 聚合抽象；小读计费不同，报文合法性未覆盖 |
| 主机写单元 | 完整单元4 KiB、4 KiB对齐；可由64 B片段累积 | §4.1 p.15；§5.4.1 第3、6项 p.58 | 主实验小写立即对16 KiB页RMW；配置可选择拒绝部分物理页，但不是4 KiB协议校验。[校验](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_mapping.py:136)、[RMW](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:650) | 缺少独立的协议写单元；介质页与主机粒度混用 |
| 接口带宽口径 | 表2：32 GT/s、64-bit/channel、16通道、75%效率→3072 GB/s；表4有不同速率档 | §4.2 表2、表4 p.16 | 读1600 / 写16 GB/s来自实验输入，tR=4 µs、tPROG=75 µs另作介质假设。[输入](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_experiments.py:61) | 协议接口表不能为这些介质参数或持续吞吐背书 |
| outstanding | 每通道、产品定义 | §5.4.1 第7–8项 p.58；BUCCAP.MOCS p.70；BUCC.MOCS p.71–72 | 整设备最多32个上层请求；不按通道、AXI事务或4 KiB待写单元计数。[准入](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:493) | 队列粒度不同；不能将32与规范举例64/128直接比较 |
| AXI 顺序与批处理 | 同ID保序；异ID可乱序；batch hint | §5.3.1 第5–6项、§5.3.2 p.57；§5.4.3 p.59 | 无AXI ID、协议顺序状态或batch hint；已有页/资源依赖及暂存范围依赖，按上层请求预约资源。[请求类型](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_types.py:135) | 未覆盖协议顺序与提示调度；提交顺序不能代替AXI顺序模型 |

粒度差异有直接影响：以页内、64 B对齐的一次64 B读为例，当前映射会计4096 B主机传输，是请求数据的64倍；这是**主机传输字节的放大**，不是总延迟必然放大64倍。介质侧还会读16 KiB页。三层数字应分别报告：64 B主机基本访问、4 KiB协议页/写单元、当前16 KiB等效介质页。

若把主实验改成4 KiB页，同时保留64单元、4 µs读和75 µs编程，按前述同一公式得到的介质读/写上限将降为65.536 / 3.495253 GB/s。因此不能只改页大小再沿用原吞吐结论；需要重新确定页大小、并行度和延迟的联合工作点。这是配置公式推导，本次没有新增性能运行。

原文带宽还存在口径差异：§4.1正文写TiB/s，而表2/4使用GB/s、TB/s；此处明确采用表2的十进制口径，且不把接口估算当成应用持续吞吐或介质阵列吞吐。

### O2. 写入、完成与缓存

| 项目 | OCP 规则或定义 | 原文章节 / 表 / 页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| 写累积与超时 | 64 B片段收齐4 KiB；有累积时限 | §5.4.1 第3、6项 p.58；表13 p.61 | buffer=0时立即RMW/编程16 KiB；正buffer按配置物理页合并，没有4 KiB累积计时器。[暂存](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:241) | 未覆盖OCP写聚合；可选暂存不能等同协议累积器 |
| 普通写完成 | Core die写完后应答 | §5.4.1 第1–2项 p.58 | 主实验buffer=0返回PAGE_PROGRAM结束；正buffer在主机接收后早确认，介质完成另计。[写穿完成](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:735)、[早确认](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:628) | 主实验此项边界对应；可选早确认语义偏离普通写，不能混称 |
| 页序 | 本地页连续，不跳写 | §5.4.1 第4、9项 p.58；表13 p.61 | 没有前序页与内容有效性状态；允许容量范围内任意地址编程和重写。[编程](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:720) | 未覆盖写入合法性及写顺序错误 |
| 自动擦除 | page 0写触发块擦除 | §5.4.1 第5项 p.58 | 只有显式ERASE触发；普通WRITE没有page-0分支。[擦除](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:615) | 协议覆盖缺口；当前普通写没有对应擦除成本 |
| 重复片段与队列错误 | overlap 0x2；待写单元满0x4；超时0x5；页序0x6 | §5.6 表13 p.61 | 正buffer合并重叠区间，无协议错误码；设备请求准入也不是4 KiB累积器容量管理。[合并](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:245) | 未覆盖；主实验buffer=0没有该累积状态 |
| 读页缓存 | 每bank两个4 KiB缓冲；命中直返 | §5.3.1 第5a、7–8项 p.57；BUCCAP.NCBB p.70 | 普通读总执行PAGE_READ；`_buffer_ready`只记占用时间，没有页号。HBF-2“两页cache”仅在说明字段。[读路径](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:687) | 未覆盖介质读缓存；配置文案不是已执行能力 |
| 暂存已到部分的读 | 允许转发 | §5.4.3 p.59 | 正buffer对已覆盖范围走STAGING_READ；主实验不启用。[命中](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:669) | 该行为方向一致；不是持久读页缓存 |
| 暂存尚缺部分的读 | 错误0xA | §5.4.3 p.59；表12 p.60 | 正buffer未完全覆盖时，读旧介质后合并暂存内容，未返回该错误。[未命中](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:687) | 可选模式偏离；不能说主实验当前发生此错误 |
| scratchpad SRAM | 可选、产品定义 | §5.3.3 p.57–58；§5.4.2 p.58；§11.3 p.116；SMEMS p.71 | 无HBF独立SRAM地址区、优先级和控制状态；通用SRAM profile或写暂存不等于此功能 | 可选能力未覆盖，不单凭此判定不合规 |

首次审计的定向复现支持上述路径判断：同页两次4 KiB写后再读，buffer=0时两次分别RMW/编程，擦除为0；buffer=64 KiB时先确认、读命中暂存、flush后只做一次合并RMW/编程。**主实验buffer=0，不能把可选早确认模式的问题写成主实验已经早确认。**

无自动擦除可能低估持续写成本；每次小写立即RMW又可能高估协议累积后的编程成本。二者影响相反，不能概括为HBF模型整体只会偏乐观或只会偏悲观。

### O3. 可靠性、维护与责任边界

| 项目 | OCP 规则或定义 | 原文章节 / 表 / 页 | 当前实现与主实验 | 判断 |
| --- | --- | --- | --- | --- |
| ECC 与读重试 | ECC状态；主机/Base die重试配置 | §5.3.1 第9项 p.57；表12 p.59–60；§11.5.1.6 p.120 | 配置无纠错/重试参数；读路径不生成相关阶段或错误。[配置](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/memory_types.py:566) | 未覆盖错误、重试与尾延迟；不应凭空补一个固定ECC耗时 |
| 数据刷新 | 设备计数/提示，主机安排 | §4.1 p.15；§11.5 p.118 | 无保持时间、读扰动计数和相应主机刷新策略 | 设备与主机配合机制未覆盖；不是要求设备自主后台刷新 |
| 温控 | 节流、过温关闭 | §9.2 p.106–108 | 当前HBF执行无温度状态和反馈限速 | 未覆盖；长期持续负载不能据此预测热稳定吞吐 |
| 设备内 GC | 本版HBF不提供 | §11.4 p.117，第2段 | NandCore明确不模拟GC。[模型范围](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:33) | **不是OCP缺项**；不能套用SSD设备内GC要求 |
| 磨损均衡 | Base die或主机重映射 | §11.4 p.117；BUCCAP.WLS p.71；BUCC.WLS p.72 | 只有初始P/E+显式擦除次数驱动编程延迟放大，并封顶；主实验wear关闭。[公式](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/nand_core.py:158) | 磨损延迟敏感性不是磨损均衡；职责随能力/配置区分 |
| zone / PEC 管理 | PEC查询；无有效数据后重映射 | §11.4.1 p.117–118 | 无分区映射、PEC查询接口或主机数据失效/重写流程 | 未覆盖对应管理闭环；不能宣称完整寿命预测 |

持续写审计应检查**自动擦除、页序、主机有效数据管理与重写、PEC查询、分区重映射和刷新**。不含设备GC可以作为功能范围描述，但不能作为本协议的缺口。数据保持和读扰动阈值、重试代价等仍需目标产品依据，OCP架构文件不能替代器件标定。

本次没有仅凭当前实现自身契约确认新的HBF功能性代码缺陷。O表指出的是它作为OCP v0.7.0模型时的覆盖差异和解释边界；“遵守简化模型规则”不等于“完整实现OCP协议”。

## HBF-2 参数组与研究结论适用范围

论文参数组仍使用 128 个等效单元、4 KiB 页、20 µs 读、200 µs 编程，主机读写接口各 1200 GB/s。由当前执行参数得到读介质上限 26.2144 GB/s，写介质上限 2.62144 GB/s。

代码将并行单元数标为敏感性实验选取的锚点，而非论文披露的物理通道数。因此可以说使用了论文的部分参数，不能说已经复现论文器件的有效 1200 GB/s 服务。配置说明中出现“两页 cache buffer”也不代表执行路径已经实现读页缓存。见 [HBF-2 配置与来源说明](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/calibrated_hbf_profiles.py:276)。

同样，第 14 组磨损实验研究的是显式擦除或预老化参数下的延迟敏感性，不是完整寿命预测。见 [磨损实验限制](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/scripts/hbf_hbm_wear_study.py:64)。

## 本次版本变化与未提交修改

| 提交 | 时间（北京时间） | 本次审计相关变化 |
| --- | --- | --- |
| 17befa5 | 2026-10-15 13:54 | 清理旧接口、迁移调用方；旧配置入口和缓冲兼容分支不再保留 |
| 799dd40 | 2026-10-15 14:11 | 清理 phase、scan、replay 兼容适配 |
| c864890 | 2026-10-15 14:28 | 清理其余运行时和浏览器兼容路径 |

与 `5a13823` 比较，HBM 的主要几何、等效时钟和未知参数选择没有重新标定；两条 HBM 路径增加了浮点网格点保护，但本报告的相加尾差仍未解决。HBF 删除了正缓冲容量却仍走旧直写语义的兼容模式，因此不能笼统写成“HBF 全部未变”。

本地 `planner.py` 修改使并行 embedding 的权重缓冲按控制面提供的物理分片容量分配，相关调用测试已通过。它属于容量/分片处理，并没有补上 HBM 刷新、HBF 擦除或当前正式实验数据。见 [本地 embedding 修改](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/src/heterollm_sim/planner.py:18051)。

## 待人工判断与处理清单

### 先处理，关系到正式结果可信度

- [ ] 修复两条 HBM 路径的内部整拍对齐缺陷，同时保留“外部晚到请求不能提前发令”的约束。
- [ ] 明确当前主实验是等效预算模型还是目标器件模型；若目标是实际器件，补齐独立接口速度、厂商阵列时序、RL/WL 支持范围和刷新依据。
- [ ] 在以上选择确定后，用当前代码重跑正式 A/B、decode 和容量对照；旧图表保留历史标识，不当作当前性能。
- [ ] 报告分别列出 HBF 接口上限和介质吞吐；若要比较同等有效读带宽，需要重新选择并行度/页延迟并验证。

### HBM 建模选择

- [ ] 是否接受容量增加同时降低等效时钟、增加 PC/bank 数；如接受，应明确这不是纯容量实验。
- [ ] 是否接受当前 BG 地址映射与按提交顺序预约的控制器策略；需要归因时再做对应敏感性实验。
- [ ] 是否接受主实验无刷新，以及 tRRD/tFAW/tRTP/tWTR/tRTW 未配置；不能将未知值描述为真实零值。
- [ ] 是否需要每通道队列、重排、读写批处理、REFpb/RFM 等机制；由研究问题决定，不要求为审计形式全部补齐。

### HBF 建模选择

- [ ] 明确16 KiB等效介质页与OCP的4 KiB页、64 B主机读之间的差异；若改成协议页，联动重标并行度与延迟，不能只替换字段。
- [ ] 持续写需要覆盖哪些页序、page 0自动擦除、主机有效数据管理/重写、PEC查询和分区重映射；**设备内GC不是OCP v0.7.0要求**。
- [ ] 是否需要按主机通道区分读写全双工资源，以评估混合流量。
- [ ] 是否需要介质读页cache、ECC、重试、设备读计数与主机数据刷新；当前写暂存不能替代这些功能。
- [ ] 是否需要4 KiB写累积、AXI顺序与错误状态、每通道队列和温控；明确哪些研究结论依赖这些机制。
- [ ] 缓冲实验分别报告主机确认与介质完成时间，避免混用为普通写完成。

## 方法、验证与限制

本次完成了源码与配置调用链检查、正式结果目录扫描、关键协议表格核对及少量定向执行。没有运行完整的 LLM 研究实验，也没有修改实现以生成“修复后”数字。

- HBM 既有测试：[命令控制](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_hbm_jedec_control.py)、[编译路径](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_hbm_compiled_control.py)、[协议计量](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_hbm_protocol_accounting.py)，**74 项通过**。
- 当前 [lowering 调用](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_current_lowering_callers.py)、[脚本调用](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_current_script_callers.py)、[报告调用](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_current_report_callers.py) 和 [报告输入](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/tests/test_current_report_inputs.py) 测试：**31 项通过**，包含本地 embedding 修改的覆盖。
- HBM 定向执行：同址三次 32 B 读，明细/编译各一组；512 GiB 的 1/16/64 MiB 冷顺序读、1 TiB 的 64 MiB 冷顺序读，1 MiB 另核对明细路径。
- HBF 定向执行：64 MiB 读/写各做明细与汇总，共四组；缓冲 0/64 KiB 的同页写读行为再做两组。
- 另启动的 serving/decode/bank-group 三个扩展测试文件未完整跑完，已停止本次启动的测试进程；不计入通过总数，也不作为“完整服务测试通过”的依据。

测试通过说明相应检查未失败，不代表所有时序、全部工作负载或真实硬件已验证。本次发现的对齐缺陷正说明“两个执行路径一致”不能替代独立的规则判断。微测带宽采用 `请求字节数 / 从 0 开始的完成时间`，GB/s 为十进制，GiB/MiB 为二进制。

本次补订对照项目已有的 [JESD238 原文](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/.refs/jedec/JESD238-2022.pdf)，核查J表所列几何、MR编码、命令和维护章节，以及表84、92、93及相关注释；其中印刷p.6和p.161–162的关键表格另以渲染图核对，避免丢失表头、单位和空白/TBD含义。HBM2/HBM4完整原文未逐项重审；主实验为HBM3E参考组织，不能把本报告扩展为所有代际的合规认证。项目说明仅作定位线索，原文章节优先；配置来源另见 [项目HBM文档](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/docs/hbm_geometry.md)。

HBF补订使用官方 [OCP High-Level Base Die Specification v0.7.0](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)，日期2026-08-03，130页；已取得本地原文。核查O表所列§4、§5、§9、§11相关条款，p.58普通写与p.117磨损/GC另以渲染图核对。协议文件用于判断明确的规则和职责，不替代目标器件参数。**本次补订只扩充原文与代码对照，没有新增或重跑测试；上述105项通过及微测均属于首次审计记录。**
