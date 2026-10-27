<!-- ppt-master-schema: design-spec/v1 -->
# HBM HBF Executive - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | HBM HBF Executive |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 5 |
| Primary Language | zh-CN |
| Target Audience | 老板与研发管理者 |
| Communication Intent | 用仓库比喻解释模型，同时保留地址、命令、页缓冲与资源排队等技术细节 |
| Desired Audience Outcome | 能理解访存如何影响Token性能，并知道模型能比较什么以及结果的适用范围 |
| Core Message / Ask / Action | 把访存需求映射到存储资源，计算并行与等待，再得到推理延迟和吞吐 |
| Delivery Context | 老板口头汇报，约5–7分钟 |
| Artifact Afterlife | 可编辑PPTX与PDF预览；备注提供讲稿与源码定位 |
| Reading Mode | presentation |
| Content Strategy | 复用前轮已核实源码事实；删去参数表与代码细节，不添加未测量性能结果 |
| Design Style | 自由设计；浅底工程信息图；蓝色HBM与棕橙色HBF保持语义一致 |
| AI Image Acquisition Path | built-in imagegen；首张概念配图，其余技术图为原生可编辑对象 |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — 延续上轮讲解备注，按用户新受众压缩 |
| Custom Animations | disabled — workflow default |
| Narration Audio | disabled — workflow default |
| Created Date | 2026-10-26 |

用户最新要求保留技术、可用比喻并允许加页；采用五页、无独立封面。其余普通版式选择由制作过程确定，不伪造用户确认。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 左右56，上36，下34 |
| Content Area | x56–1224, y192–650 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: 总览、HBM命令、HBF页服务、资源时间线、指标与边界各一页。
- **Visual style**: custom
- **Visual Style Behavior**: 大字、扁平工程图、低文字密度，关系由箭头和分流汇合表达，色彩只辅助标记介质。
- **Theme**: 访存到性能的因果链
- **Tone**: 清楚、克制、适合管理层

使用内置imagegen生成三页构图参考，保存在analysis/visual_reference.png；借鉴分流汇合、介质结构与低密度层级。参考图不作为事实来源或最终页面位图；最终逻辑和文字均以可编辑对象绘制。

### AI Image Strategy

- **Image Rendering**: custom
- **Image Rendering Behavior**: 浅底等距仓库比喻，蓝色抽屉柜对应HBM bank/row，棕橙整箱与两个装卸位对应HBF页与读缓冲；不表示真实芯片结构或性能比例。
- **Visual**: 简洁立体配图，无嵌入文字；原生标签与技术图承担准确内容。
- **Mood**: 工程师用仓库模型向管理者讲解访存。

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #F6F8FB | 页面底色 |
| Secondary background | #FFFFFF | 实体与留白 |
| Primary | #007FBA | HBM |
| Accent | #B96628 | HBF |
| Secondary accent | #526479 | 公共说明 |
| Body text | #11243A | 主体文字 |
| Dark | #0C1B2B | 结论横带 |
| Line | #B5C3CF | 连线 |
| HBM tint | #E4F2FA | HBM资源 |
| HBF tint | #FBECDD | HBF资源 |

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
| Body | 28 |
| Title | 46 |
| Subtitle | 26 |
| Annotation | 18 |
| Label | 24 |
| Emphasis | 34 |
| Small | 22 |
| Footnote | 16 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 结论型标题、主图、底部一句讲解。
- **Composition tendency**: 大幅主图与技术顺序图配合；五页均直接承载内容。
- **Cross-page continuity**: HBM蓝、HBF棕橙，统一的输入输出与箭头语言。
- **Spacing posture**: 投影可读，说明靠近对应机制，代码只进备注。

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warehouse_metaphor.png | 1774 × 887 | 2.00 | P01两种取货方式的比喻 | Concept illustration | 首张中央双场景帮助理解两种服务机制，四周原生标签 | no-crop | ai | Generated | 蓝色抽屉柜与棕橙整箱货架和两个装卸位，非芯片实物 | none | local |

## IX. Content Outline

#### Slide 01 - 仓库比喻与总流程
- **Audience move**: 由抽屉取货与整箱搬运理解两种介质，再连到实际建模流程。
- **Layout**: 中央概念配图，两侧HBM/HBF原生标签，底部完整流程。
- **Title**: 像管理两种仓库一样，建模 HBM / HBF
- **Core message**: 访问先落实到字节地址，再计算介质服务与排队。
- **Content**: bank像柜子、row像抽屉；page像整箱、每plane两槽读缓冲像装卸位；负载与配置→字节地址→介质与排队→Token指标。
- **Images**: warehouse_metaphor.png完整展示，只解释概念。
- **Cover impact**: 不设独立封面，两种取货机制的差异直接引出技术因果链。

#### Slide 02 - HBM命令链
- **Audience move**: 能解释相同字节数为何因行状态不同产生不同延迟。
- **Layout**: 地址映射条与命中/关闭/冲突三行命令链。
- **Title**: HBM：抽屉是否已打开，决定命令成本
- **Core message**: 行命中省去开关行成本，所有命令仍受资源与时序约束。
- **Content**: jedec_hbm按32B burst拆分并映射channel/PC/bank/row；每bank一行打开；hit RD→RL→DQ，closed ACT→tRCD→RD→RL→DQ，conflict PRE→tRP→ACT→tRCD→RD→RL→DQ；另计命令间隔、读写切换和刷新。
- **Visualization**: 原生命令顺序图，无数值时间比例。

#### Slide 03 - HBF页服务
- **Audience move**: 理解页拆分、三段服务、命中旁路和缓冲寿命。
- **Layout**: 三个编号阶段；阵列页堆与介质暂存、内部搬运与双读槽、请求方；灰色虚线为命中旁路，读槽规则和写入单列。
- **Title**: HBF：整页取货，用双缓冲衔接三段服务
- **Core message**: 介质暂存与两个读缓存槽不同；内部搬完才能读，旧读者结束才能覆盖。
- **Content**: 论文原生例16die×25plane/堆、4KiB页，连续页先die通道再plane；miss阵列→内部→slot→host，hit跳过array/internal但等ready和host；写host→internal→program结束，部分写可另付RMW。
- **Visualization**: 原生页堆、介质暂存托盘、两个读槽和请求方；明确阵列→介质暂存→内部通道→读槽→host，命中绕过前两段；1μs读和100μs program仅为示例输入。

#### Slide 04 - 时间线与排队
- **Audience move**: 能判断哪些阶段可重叠，哪些必须等待。
- **Layout**: 三个资源泳道，页A灰色、页B棕橙；实线同页依赖，虚线明确内部填A结束→读B开始，下面分别解释重叠与资源/槽复用等待。
- **Title**: 同一条路要排队，不同资源可重叠
- **Core message**: 单阶段必须等依赖和资源同时就绪，双缓冲不保证理想流水。
- **Content**: 示例前提为同plane、A/B均未命中且已准入、两个初始空槽、无写覆盖；读B≥填A结束，读B/填B可与输出A重叠；同内部/host读通道互斥；覆盖槽A等待旧读者结束，host读完不清空缓存页。
- **Visualization**: 概念泳道，无时间数值，不按时长比例。

#### Slide 05 - 输出与边界
- **Audience move**: 由物理事件追溯Token性能，同时认识参数假设范围。
- **Layout**: 两介质观测汇入统一统计，再接Token指标和配置比较。
- **Title**: 算得出性能，也追得回原因
- **Core message**: 同负载同口径比较配置，以物理事件解释性能变化。
- **Content**: HBM行命中/冲突及命令、HBF缓存命中/读页/program；逻辑与物理字节、排队、busy、任务完成；Token吞吐/TTFT/TPOT；服务跨度可重叠，不等于busy或makespan；标准参考、论文输入和研究假设分开，尚未目标芯片校准。
- **Visualization**: 介质观测到推理指标的因果图，无虚构测量值。
- **Closing impact**: 解释差异原因，并用配置输入和假设限定结论。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 每页一段中文口头讲稿，代码来源用简短末句注明；说明具体HBM模式、HBF基线与非实测边界。
- **Total duration**: 5–7 minutes
- **Notes style**: conversational
- **Presentation purpose**: explain
