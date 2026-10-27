<!-- ppt-master-schema: design-spec/v1 -->
# HBM HBF Modeling - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | HBM HBF Modeling |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 18 |
| Primary Language | zh-CN |
| Target Audience | 需要理解当前仿真器的项目使用者与研发人员 |
| Communication Intent | 解释当前工作区中 HBM 与 HBF 的完整建模流程，优先流程和资源因果 |
| Desired Audience Outcome | 能解释请求怎样由张量访问变成物理完成时间，并区分两种介质的瓶颈和假设 |
| Core Message / Ask / Action | 地址、介质状态与共享资源时间线共同决定访存完成时间 |
| Delivery Context | 中文技术讲解，约 15–20 分钟 |
| Artifact Afterlife | 可编辑 PPTX，附逐页说明和源码定位 |
| Reading Mode | balanced |
| Content Strategy | 当前工作区代码为事实依据，源码中研究参数明确标注；采用概念图，不制造性能实测 |
| Design Style | 自由设计，白底工程图，HBM 蓝色、HBF 棕橙色、公共流程深灰色 |
| AI Image Acquisition Path | not applicable；所有图为可编辑的流程、层级与时序示意 |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — workflow default，逐页解释与源码定位 |
| Custom Animations | disabled — workflow default |
| Narration Audio | disabled — workflow default |
| Created Date | 2026-10-26 |

未向用户增加确认步骤；页数、字体与配色属于本次制作中可逆的普通设计选择。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 左右 64，上 52，下 40 |
| Content Area | x64–1216, y180–640 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: 先解释公共链路，再依次拆开 HBM 与 HBF，最后解释指标与建模边界。
- **Visual style**: custom
- **Visual Style Behavior**: 工程说明图；字与线保持可编辑，颜色标记介质，层级、分流和时间轴承担主要视觉信息。
- **Theme**: 双介质汇入一条资源时间线
- **Tone**: 简明、精确，避免宣传式表达

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #F6F8FB | 内容页底色 |
| Secondary background | #FFFFFF | 图内实体与浅色区域 |
| Primary | #007FBA | HBM |
| Accent | #B96628 | HBF |
| Secondary accent | #526479 | 公共资源与说明 |
| Body text | #11243A | 主要文字 |
| Dark | #0C1B2B | 封面与结束页 |
| Line | #CCD6DF | 结构连线 |
| HBM tint | #E4F2FA | HBM 区域 |
| HBF tint | #FBECDD | HBF 区域 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 清晰黑体 | Microsoft YaHei | Arial | sans-serif |
| Body | 清晰黑体 | Microsoft YaHei | Arial | sans-serif |

- **Title stack**: Microsoft YaHei, Arial, sans-serif
- **Body stack**: Microsoft YaHei, Arial, sans-serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 26 |
| Title | 42 |
| Subtitle | 24 |
| Annotation | 18 |
| Label | 22 |
| Emphasis | 32 |
| Hero | 64 |
| Footnote | 16 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 先标题与一句解释，再读主体图，最后读边界与出处。
- **Composition tendency**: 总图、层级展开、对比、三行时序与单页结论交替。
- **Cross-page continuity**: 蓝色对应 HBM，棕橙对应 HBF，文字解释不只依赖颜色。
- **Spacing posture**: 图内保留读图间隔，代码路径放备注与页脚。

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## IX. Content Outline

#### Slide 01 - HBM / HBF 建模流程
- **Audience move**: 从主题名称进入两条物理建模路径与一条公共时间线。
- **Layout**: 左侧主题，右侧分流汇合图。
- **Title**: HBM / HBF 建模流程
- **Core message**: 地址、介质状态和资源竞争共同产生完成时间。
- **Content**: 张量访问进入 HBM DramCore 或 HBF NandCore，汇入共享时间线与完成事件。
- **Cover impact**: 同一条请求，经过两套物理规则；右侧双路径图表达这个钩子。

#### Slide 02 - 仿真器的完整链路
- **Audience move**: 能按顺序复述六步建模链路。
- **Layout**: 上下两排连贯流程。
- **Title**: 仿真器的完整链路
- **Core message**: 任务实际到达后才由有状态 Core 给出最终时间。
- **Content**: 工作负载、任务与放置、地址绑定、Core执行、事件推进、结果统计。

#### Slide 03 - 两种介质的建模对象
- **Audience move**: 理解 HBM burst 与 HBF page 的差异。
- **Layout**: 双侧对比图，共同出口。
- **Title**: 两种介质的建模对象
- **Core message**: 公共访问契约之下，介质的服务粒度与状态不同。
- **Content**: HBM 用 DramConfig/DramCore，JEDEC burst 32B；HBF 用 NandConfig/NandCore，论文原生页4KiB。共同输入为操作、地址、大小、到达。

#### Slide 04 - 配置先定义物理约束
- **Audience move**: 区分工作负载、放置与硬件输入。
- **Layout**: 三组输入汇入访问请求。
- **Title**: 配置先定义物理约束
- **Core message**: 容量、拓扑、带宽、时序和协议分别影响模型。
- **Content**: 模型/Batch/上下文/精度/并行配置，权重/KV/激活端点，DramConfig/NandConfig；峰值只约束传输上限。

#### Slide 05 - 张量访问到物理地址
- **Audience move**: 能说明访问描述怎样解析成地址范围。
- **Layout**: 权重、KV、激活汇入访问契约，再拆分。
- **Title**: 张量访问到物理地址
- **Core message**: 分配器、owner 与字节范围把逻辑任务接到物理资源。
- **Content**: operation/owner/byte_count，address 或 buffer_id+offset，容量与生命周期校验，HBM burst 与 HBF page 拆分，逻辑与物理字节分开。

#### Slide 06 - HBM 地址映射与并行度
- **Audience move**: 认识 stack/channel/PC/bank/row 和 group 轮转。
- **Layout**: 层级展开加 bank 序列。
- **Title**: HBM 地址映射与并行度
- **Core message**: 地址分布决定落入哪些可并行或共享的资源。
- **Content**: HBM4每堆32 channel、每channel2 PC；SID已包含在bank地址中；bank_interleaved先lane后bank/column/row，同SID内group轮换0,8,1,9。

#### Slide 07 - HBM 的两种控制模式
- **Audience move**: 能区分简化与 JEDEC 命令路径。
- **Layout**: 两条并列深浅不同的流程。
- **Title**: HBM 的两种控制模式
- **Core message**: analytical 与 jedec_hbm 是不同建模精度选择。
- **Content**: analytical简化DRAM，不模拟刷新/tRAS；jedec_hbm显式ACT/PRE/RD/WR/REF、行状态和CA/DQ；加速路径不会改变所选模式。

#### Slide 08 - HBM 行状态与命令链
- **Audience move**: 理解同样读字节数为何会产生不同延迟。
- **Layout**: 行命中、关闭、冲突三行概念时序。
- **Title**: HBM 行状态与命令链
- **Core message**: 行状态决定需要执行的命令和等待。
- **Content**: hit RD/RL/DQ，closed ACT/tRCD/RD/RL/DQ，conflict PRE/tRP/ACT/tRCD/RD/RL/DQ；概念顺序非比例。

#### Slide 09 - HBM 共享资源时间线
- **Audience move**: 分清共享CA与独立DQ，理解发射约束。
- **Layout**: Row CA/Column CA/PC0 DQ/PC1 DQ四泳道。
- **Title**: HBM 共享资源时间线
- **Core message**: 不同PC数据可重叠，但共享命令与时序仍限制发射。
- **Content**: 请求/银行/总线/约束最大值后时钟对齐，tCCD/tRRD/tFAW/切换、前瞻PRE和all_bank刷新均入时间线；请求级调度不是逐命令FR-FCFS。

#### Slide 10 - HBF 几何与资源映射
- **Audience move**: 理解论文原生配置的真实plane并行度。
- **Layout**: stack到die到plane到page展开与资源数量。
- **Title**: HBF 几何与资源映射
- **Core message**: 资源数由几何产生，不能由目标带宽反推。
- **Content**: 每堆16die×25plane，4KiB页；每die映一内部channel，每plane阵列；连续页先base_units后plane再block内page；7堆112通道2800阵列。

#### Slide 11 - HBF 读请求的三段服务
- **Audience move**: 能复述未命中与命中两条路径。
- **Layout**: 主线array/internal/host，缓存命中旁路。
- **Title**: HBF 读请求的三段服务
- **Core message**: miss需要页读取与内部搬运，hit仍等待slot就绪和host资源。
- **Content**: 队列准入、按页拆分映射、缓存查找，miss PAGE_READ→INTERNAL_TRANSFER→HOST_TRANSFER，hit跳过前两段，所有页完成取最大值。

#### Slide 12 - 每个 plane 的双缓冲
- **Audience move**: 理解双slot提高重叠同时保留覆盖等待。
- **Layout**: 两slot与上下游连接，标示ready和reader寿命。
- **Title**: 每个 plane 的双缓冲
- **Core message**: 填充完成前不可读，旧读者完成前不可覆盖。
- **Content**: 每plane2页；3,276,800B=3.125MiB每堆；请求触发填充；与每堆40MB中间SRAM分开。

#### Slide 13 - HBF 写入与协议边界
- **Audience move**: 理解写入与擦除成本和generic范围。
- **Layout**: 写路径与RMW分支，下方协议边界。
- **Title**: HBF 写入与协议边界
- **Core message**: 页写必须付program成本，generic基线不自动加入FTL/GC。
- **Content**: HOST→INTERNAL→PROGRAM；部分写reject或付费RMW；显式block erase；论文原生generic_nand，OCP为另一配置。

#### Slide 14 - 跨端点数据搬运与依赖
- **Audience move**: 了解copy路径的三段成本如何进入LLM调度。
- **Layout**: 源Core读、链路、目标Core写与依赖箭头。
- **Title**: 跨端点数据搬运与依赖
- **Core message**: 搬运完成需要目标写完成，独立资源才能重叠。
- **Content**: 源读→链路→目标写，owner共享资源，RAW/WAR/WAW，任务实际dispatch提交，后继等待完成；物理memory envelope保留计算模型抽象。

#### Slide 15 - 输出指标与瓶颈判断
- **Audience move**: 按完成时间、流量与资源忙碌定位瓶颈。
- **Layout**: 中央输出向三组观测分支。
- **Title**: 输出指标与瓶颈判断
- **Core message**: 总延迟、服务跨度、资源busy必须分开阅读。
- **Content**: completion/arrival、logical/physical bytes、HBM命令与row计数、HBFcache/array/internal/host、等待、Token throughput/TPOT；service_ns跨度可重叠，不能当wall time。

#### Slide 16 - 快速仿真与物理语义
- **Audience move**: 理解执行加速与降低模型精度的不同。
- **Layout**: 参考与数值执行汇向同一物理约束。
- **Title**: 快速仿真与物理语义
- **Core message**: 编译循环和压缩记录降低模拟器开销，保留所选时序规则。
- **Content**: HBM Numba数值路径/批量提交；NAND逐页编译/缓存状态/资源预约；适用条件与参考实现；不代表硬件实测。

#### Slide 17 - 一组配置输入示例
- **Audience move**: 区分论文输入与本地研究假设。
- **Layout**: HBM4与HBF两侧参数；底部假设说明。
- **Title**: 一组配置输入示例
- **Core message**: 这些是项目中的配置输入，不是本次跑出的性能结果。
- **Content**: HBM4接口12.8TB/s每GPU、8堆；HBF配置7堆，读11.2TB/s写112GB/s，每页读1μs/program100μs；HBM tRRD/tFAW借HBM3E，NAND256页block/2ms erase/100GB/s每die为本地假设。

#### Slide 18 - 理解模型的四个抓手
- **Audience move**: 保留完整因果链并能定位源码。
- **Layout**: 四关键词串联与简短源码索引。
- **Title**: 理解模型的四个抓手
- **Core message**: 请求描述、地址映射、介质状态、资源时间线共同解释仿真结果。
- **Content**: 请求→地址→状态→时间；HBM看行与命令，HBF看页/缓存/三段传输；代码入口memory_types/memory_mapping/_hbm_control/nand_core/data_motion/reporting。
- **Closing impact**: 每个性能变化都应沿请求、地址、状态和时间线追溯原因。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 中文逐页解释，覆盖信息图，追加精确项目相对源码路径和行号，不宣称本次运行仿真或做了芯片校准。
- **Total duration**: 15–20 minutes
- **Notes style**: conversational technical explanation
- **Presentation purpose**: instruct
