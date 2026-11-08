# HBM 控制逻辑 JEDEC 审计

审计日期：2026-11-07

结论：当前实现不能称为“完整 JEDEC HBM 控制器”。它可以作为按代际区分的 HBM 正常读写时序子集：在伪通道模式下，对 ACT、PREpb、PREab、RD、WR、REFab，以及 HBM4 的有限 REFpb 路径建模，并对命令总线、DQ、bank-group、tFAW、tRAS、刷新期限和 FR-FCFS 调度执行约束。初始化、模式寄存器、低功耗、自刷新、训练、奇偶/ECC/RAS、RFM/ARFM/DRFM、自动预充电和厂商地址散列仍未实现，因此器件级或全规范合规结论不成立。

## 规范基线与代际范围

| 代际标签 | 代码当前基线 | 审计范围 | 结论 |
|---|---|---|---|
| HBM1 | 无 | JESD235 Legacy channel | 不支持。不能用 HBM2 伪通道结果外推。 |
| HBM2 / HBM2E | JESD235B-2018，`.refs/jedec/JESD235B-2018.pdf` | 仅 PC mode；有限 HBM2 组织 | 正常命令子集可核对；HBM1 Legacy、更多合法密度/层数、完整 HBM2E 修订族未覆盖。 |
| HBM3 / HBM3E | JESD238-2022，`.refs/jedec/JESD238-2022.pdf` | HBM3 PC 几何和正常命令子集 | 不能宣称最新修订合规；HBM3E 是 JESD238 系列的厂商扩展/速度等级，不是独立 JEDEC 文件。REFpb、RFM 和 RAS 功能未完整实现。 |
| HBM4 | JESD270-4A-2025，`.refs/jedec/JESD270-4A-2025.pdf` | HBM4 PC 几何、有限 REFpb 和正常读写 | 规范版本基线较新，但仍只覆盖正常操作子集。 |

`.refs/jedec` 中已有 JESD238B.01-2025 preview；代码仍绑定 JESD238-2022，不能把旧基线描述成当前修订的完整验证。JESD235B 也不能代表 HBM2E 可能采用的 JESD235C/D 修订族。公开镜像用于追踪来源，不能替代 JEDEC 受控发布文本。

## 已核对的实现

- 几何入口按代际区分通道、伪通道、位宽、突发长度、bank-group 和有效行数。HBM2/HBM2E 只导入有限组织；HBM3/HBM3E 与 HBM4 的组织范围也只是表中已导入的密度和层数。
- `jedec_hbm` 配置校验命令占用：HBM2/HBM2E 为 ACT=2 CK、PRE=1 CK、列命令=1 CK；HBM3/HBM3E/HBM4 为 ACT=1.5 CK、PRE=0.5 CK、列命令=1 CK。
- 数值控制器维护每 bank 的行状态、tRCD/tRAS/tRP/tRC、tRRD、tFAW、读写恢复、列命令间隔、DQ 方向转换和 tRASmax。未提供值的字段不会被自动包装成厂商保证。
- HBM4 REFpb 路径使用 `tRFCpb`、`tRREFD`、每 bank 的刷新步长和 SID 分组限制；这是控制器策略模型，不等于已证明覆盖所有规范调度排列。

## 已修复的实质缺陷

JESD238 §6.3.2.5（印刷第46–47页）和 JESD270-4A §6.3.2.5.1（印刷第55–56页）要求 REFab 在列命令输入上有同周期 CNOP，除非另一伪通道正在发配对列命令。原实现只在 HBM2/HBM2E 发 `REFRESH_CNOP`，导致 HBM3/HBM3E/HBM4 的 REFab CA 序列缺项；长空闲快进和编译内核也漏计列总线占用。

本次修复已完成：

1. `src/heterollm_sim/_hbm_control.py` 对所有支持代际的 REFab 等待列总线并发出同周期 CNOP；空闲刷新聚合也计入每条 CNOP。
2. `src/heterollm_sim/_hbm_fr_core.c` 将编译路径的列等待、CNOP 发射和空闲刷新聚合从 HBM2 标志改为 all-bank refresh 标志，同时保留 HBM2 的行命令量化规则。
3. `tests/test_hbm_jedec_control.py` 增加 HBM3E/HBM4 的 REFab-CNOP 回归；`tests/test_hbm_fr_fcfs.py` 的合法性检查将同周期 CNOP 与 REFab 配对，不把 CNOP 错误地套用 tRFC 访问禁止窗口。

## 未覆盖项与判定边界

- **初始化和状态转换**：RESET、tINIT、MRS、MRS 依赖、IEEE 1500、PDE/PDX、SRE/SRX、时钟停止/恢复无状态机。
- **命令族**：RDA/WRA 自动预充电、HBM3 REFpb、HBM2 单 bank refresh、RFMab/RFMpb、ARFM/DRFM 未形成完整路径；HBM4 REFpb 只覆盖当前模型的顺序和节流策略。
- **信号完整性与可靠性**：CA parity、data parity、on-die ECC、ECS/SEV、错误注入/报告、修复和训练未建模。
- **电气/器件参数**：RL/WL、阵列时序、tRFC/TBD、换向和带宽多为显式分析输入、IDD 示例或工程假设，不是已标定料号的 AC 保证。HBM3 24/32Gb 等组织的 TBD 刷新时序会被禁用或要求显式假设。
- **地址映射**：`sid_aware` 是按 JEDEC 表组织的可复现模型映射；厂商控制器的地址散列、交织和物理保留区未验证。
- **HBM2E/HBM3E 版本与速度**：技术代标签不能自动选择规范修订或厂商 speed bin；必须由具体料号、修订文本和 AC 表补齐。

## 验证证据

本次定向回归包括：

- HBM JEDEC 控制、配置、时序约束、几何来源、依赖和 FR-FCFS：修复后相关测试通过；其中 `tests/test_hbm_jedec_control.py` 34 项通过。
- `tests/test_hbm_fr_fcfs.py::test_refresh_stream_stays_legal`：通过，确认 REFRESH_CNOP 与 REFab 同周期。
- `tests/test_hbm_compiled_control.py -k compiled_matches_mixed_commands_and_refresh`：4 项通过，确认重建后的编译路径在 HBM2E/HBM3E/HBM4 刷新场景与参考路径一致。
- 长空闲刷新聚合回归：确认 `capture_details=False` 的列命令 busy 与逐条参考执行一致，HBM3E 示例为 200 条刷新对应 201 ns 的累计列总线占用（包含初始命令）。

这些结果证明修复后的正常操作子集和两条执行路径一致；它们不证明未实现命令族、PHY、RAS 或具体厂商器件的完整 JEDEC 合规。
