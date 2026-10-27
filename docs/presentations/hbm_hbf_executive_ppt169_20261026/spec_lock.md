<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 老板与研发管理者
- objective: 用五页图解和仓库比喻讲清HBM/HBF地址、命令、页缓冲、资源排队及性能归因。
- core_message: 访存需求映射到资源，计算并行与等待，得到推理延迟和吞吐。
- consumption_mode: presentation

## mode
- mode: custom
- mode_behavior: 总流程、HBM命令、HBF页服务、资源时间线、输出与边界各一页，无独立封面。

## visual_style
- visual_style: custom
- visual_style_behavior: 大字浅底工程信息图，HBM蓝、HBF棕橙，主图承担因果逻辑。

## colors
- background: #F6F8FB
- secondary_background: #FFFFFF
- primary: #007FBA
- accent: #B96628
- secondary_accent: #526479
- text: #11243A
- dark: #0C1B2B
- line: #B5C3CF
- hbm_tint: #E4F2FA
- hbf_tint: #FBECDD
- image_rendering: custom
- image_rendering_behavior: 浅底等距仓库比喻，蓝色抽屉柜与棕橙整箱和两个装卸位，无嵌入文字。

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- body: 28
- title: 46
- subtitle: 26
- annotation: 18
- label: 24
- emphasis: 34
- small: 22
- footnote: 16

## icons
- library: none
- inventory: none

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: anchor

## images
- warehouse: images/warehouse_metaphor.png | source=ai | pattern=首张中央双场景帮助理解两种服务机制，四周原生标签 | crop=no-crop

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
