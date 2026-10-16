# Ramulator2 来源参数对照

本功能在 HeteroLLM 仿真器中使用一组有明确开源出处的 HBM 时序，比较参数选择对结果的影响。默认 `reference` 参数组不变。所有参数组均未完成厂商器件标定，`vendor_calibrated=false`。

## 来源与适用范围

- 仓库：CMU-SAFARI/ramulator2。
- 版本：`54e90367191062067e9be74840439b0daca40343`，该文件最后修改日期为2026-09-02；本次核对日期为2026-10-15。
- [固定版本的原始文件](https://github.com/CMU-SAFARI/ramulator2/blob/54e90367191062067e9be74840439b0daca40343/python/ramulator/dram/hbm3.py)。原始速度档在第287–300行，刷新推导在第219–235行，读写切换公式在第204–210行。
- 速度档 `HBM3_6400Mbps`，组织 `HBM3_32Gb_16hi`；对应32 Gbit/die、16H、每stack原始64 GiB、tCK=0.625 ns。
- 上游把多项核心参数和该组织的550 ns刷新时间明确标为 `Ramulator Guesstimate`。这提供研究对照的来源，不提供厂商AC保证。

公共参数组名为 `ramulator2`，公共物理预设为 `hbm3e_timing_ramulator2`。本项目将同一HBM3命令体系映射到HBM3/E分析工作点；不据此声称某一真实HBM3E器件在此工作点的参数已被确认。改为其他时钟或组织会被拒绝，避免将这一速度档静默外推。

## 逐项参数

下表为归一化配置值，单位ns；它们是命令约束输入，不是某一请求的最终端到端延迟。执行时仍需考虑命令接收锚点、合法发令边沿、其他约束以及资源占用。原始周期值及来源类型保存在参数组元数据中。

| 参数 | 现有reference | Ramulator2来源组 | 上游输入或推导 |
| --- | ---: | ---: | --- |
| RL / nCL | 20 | 12.5 | 20 CK |
| WL / nCWL | 5 | 6.25 | 10 CK |
| tRCDRD | 14 | 19.375 | 31 CK |
| tRCDWR | 14 | 9.375 | 15 CK |
| tRAS | 33.125 | 28.125 | 45 CK |
| tRP | 15 | 16.25 | 26 CK |
| tRC | 48.125 | 44.375 | tRAS+tRP，71 CK |
| tWR | 15 | 20.625 | 33 CK |
| tRTP | 3.75 | 5.625 | 9 CK |
| tRRD_S | 4 | 2.5 | 4 CK |
| tRRD_L | 6 | 3.125 | 5 CK |
| tFAW | 20 | 15 | 24 CK |
| tWTR_S | 3.75 | 4.375 | 7 CK |
| tWTR_L | 5 | 6.25 | 10 CK |
| tRTW | 18.125 | 10.625 | 上游公式，17 CK |
| tCCD_S | 1.25 | 1.25 | 2 CK |
| tCCD_L | 2.5 | 2.5 | max(4,ceil(2500/tCK_ps)) CK |
| tCCDR | 2.5 | 1.875 | 4 SID时为3 CK，仅跨SID连续读 |
| tPPD | 1.25 | 1.25 | 2 CK |
| tRFCab | 500 | 550 | 550 ns转为880 CK |
| tREFI | 3900 | 3900 | floor(3900000/tCK_ps) CK |
| tRASmax | 35100 | 35100 | 本项目保留9×tREFI，上游无对应字段 |
| 刷新策略 | 周期REFab | 周期REFab | 沿用本项目控制器 |

其中RL、tRCDRD、tRCDWR、tWR、tRTP、tRRD_L、tRTW这7项超出现有fast/slow两档所覆盖的数值范围。这说明原有三档不能代表全部公开研究模型的取值范围，更不能当作器件上下界。本次保留独立Ramulator2组，不把开源估算重新包装为新的厂商上下限。

Ramulator2组的读写切换使用上游公式：

```text
nRTW = nCL + nBL - nCWL
       + ceil((max(-2000, -2*tCK_ps) + 5*tCK_ps + 10*(2500+20)) / (10*tCK_ps))
```

在625 ps和上游RL/WL下，结果为17 CK。这里使用上游2.5 ns和20 ps的PHY参数，不加入现有reference的额外系统裕量。RL只取nCL，数据突发时长另计；不能把上游用于请求完成的nCL+nBL一起填入RL。

读写切换的单项实验保持reference的RL=20 ns、WL=5 ns，仅改用上游公式，因此结果是31 CK=19.375 ns。不能在reference的RL/WL不变时直接填入全组的10.625 ns并把结果解释为公式差异。RL/WL自身的单项实验仍使用reference公式重算tRTW，报告单独列出这些联动。

## 执行边界

这是一项“同一仿真器中的参数对照”。没有运行Ramulator2本体，也没有导入它的控制器、地址映射、刷新调度或命令锚点处理。

上游会按照命令最后接收tick调整约束；本项目按既有JEDEC命令解释执行。例如本项目从ACT第二上升沿计算tRCD，从RD首上升沿计算tRTP，并在PRE之后等待完整tRP。ACT到RD在上升沿对齐后可得到相同间隔；RD/WR到PRE、PRE/REF到ACT则仍可能不同。因此相同标量输入不保证两个模拟器产生相同命令时间线。

tWTR_S/L和tWR从写数据结束后计时，WL和突发时间已由本项目另行计算。tCCDR仅约束跨SID连续读，不能扩展为所有跨SID读写都使用3 CK。tRASmax以及周期REFab策略属于本项目保留的约束，来源字段与上游参数分开记录。

上游nRFCpb、nRFMab、nRFMpb、nRREFD、nREFIpb只保留原始值及未映射说明，不启用REFpb、RFM等此前暂不实施的功能。

## 使用方法

直接取得公开配置：

```python
from heterollm_sim.physical_presets import resolve_physical_config

config = resolve_physical_config("hbm3e_timing_ramulator2", overrides={
    "capacity_bytes": 512 * 1024**3,
    "read_bandwidth_gb_s": 1600.,
    "write_bandwidth_gb_s": 1600.,
})
```

或对已有、组织和时钟符合条件的配置应用 `apply_hbm_timing_profile(config, "ramulator2")`。完整服务研究入口接受 `hbm_parameter_profile="ramulator2"`；单项实验通过 `hbm_timing_overrides={字段名: ns值}` 调用同一配置入口，来源会标记为研究者覆盖，依赖项会重新计算。

全组对照与逐项对照的实验入口为 `scripts/hbm_ramulator2_study.py`，结果写入 `artifacts/hbm_ramulator2_20261015/`。实验保持几何、时钟、读写带宽预算和控制器一致。独立访存探针使用相同请求地址和到达时间；完整服务使用相同模型、输入、批量和外部请求到达时间，内部访存时刻随执行进度自然变化。RAS/RP的修改联动RC，RL/WL的修改联动RTW；这些结果不能按独立百分比相加，也不能把某个探针的带宽变化直接称为LLM加速。

## 已完成的比较

首轮252个微基准和40个完整服务点全部完成，实际Core配置、参数覆盖和服务执行等8项汇总检查全部通过。逐项数据见 [实验报告](../artifacts/hbm_ramulator2_20261015/report.md)，完整数据见 [results.json](../artifacts/hbm_ramulator2_20261015/results.json)。

| 全组替换后的每token延迟变化 | A：全HBM | B：KV使用HBF |
| --- | ---: | ---: |
| Small，输入1024/批量4/输出8 | +0.0065% | -0.0031% |
| Qwen2.5-0.5B，输入128/批量1/输出4 | +1.5544% | +0.3416% |

变化为正表示更慢。两组完整服务中方案排序均未改变，A更快；不能外推到未测模型或长上下文。微负载变化明显依赖访问模式，例如顺序读-1.04%、随机读+6.02%、同PC混合读写+22.40%，不能把这些带宽变化称为同幅度的LLM加速。

因此继续保留reference作为主默认，将ramulator2作为独立来源对照。tCCDR和tFAW单项在这批固定条件下未改变结果，仍保留约束；该零变化不足以证明它们没有作用。取得真实器件AC表后再做厂商标定，本次结果不改变未标定状态。

## 大模型扩展

增加Llama 3.1-70B和Qwen3-235B-A22B，分别执行reference/ramulator2 × A全HBM/B的KV放HBF，共8个完整服务点。两模型均使用输入128 token、单请求、输出4 token，HBM仍为512 GiB；其他硬件、控制器和时序来源与首轮相同。

| 模型 | 结构 | 层数 | BF16权重存储量 | 模型支持范围 |
| --- | --- | ---: | ---: | --- |
| Llama 3.1-70B | 稠密 | 80 | 131.414 GiB | 项目内置架构配置；官方仓库配置需授权访问 |
| Qwen3-235B-A22B | MoE，128个专家中每token选择8个 | 94 | 437.895 GiB | 官方配置确定结构，专家路由与访问量使用分析模型 |

Qwen3的235B是总参数量，22B是标称每token激活参数量，不能当作235B稠密模型。其结构已与[官方固定版本配置](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json)核对；本实验不下载权重，也不执行真实数值推理。Llama使用项目保存的[固定版本架构来源](https://huggingface.co/meta-llama/Llama-3.1-70B/blob/349b2ddb53ce8f2849a6c168a81980ab25258dac/config.json)，不声称本次匿名重新读取了受限文件。

当前MoE路由按确定性均衡分配生成访问量：128-token Prefill会涉及全部128个专家，单token Decode选择8个专家，未覆盖真实输入造成的专家负载偏斜。所有算子仍放在同一假设GPU设备上，TP/PP/EP均为1，未模拟多GPU的专家通信。因此大模型结果是在本实验硬件假设下的存储时序敏感性，不能当作现成GPU部署的速度预测。

单独运行这8点：

```text
.venv/Scripts/python.exe scripts/hbm_ramulator2_study.py --stage large --workers 8 --resume
```

首token由实际Prefill产生，之后测量3次Decode间隔；同时记录首token等待时间（TTFT）、每token平均延迟（TPOT）、独立Decode吞吐和包含Prefill的完整请求吞吐。该短输入、单请求实验用于检查更大模型下参数选择是否改变结论，不代表长上下文、高并发或长期稳态。模型之间还存在结构差异，不能把跨模型差异全部归因为参数总量。
