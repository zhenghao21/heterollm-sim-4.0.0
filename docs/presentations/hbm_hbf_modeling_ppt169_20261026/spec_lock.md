<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 项目使用者与研发人员
- objective: 用中文图解当前HBM/HBF建模流程，使观众能解释物理请求如何产生完成时间并分清假设。
- core_message: 地址、介质状态与共享资源时间线共同决定访存完成时间。
- consumption_mode: balanced

## mode
- mode: custom
- mode_behavior: 公共链路后分别解释HBM与HBF，再汇总指标与建模边界。

## visual_style
- visual_style: custom
- visual_style_behavior: 白底工程图，蓝色HBM、棕橙HBF、灰色公共流程，所有文字和图形可编辑。

## colors
- background: #F6F8FB
- secondary_background: #FFFFFF
- primary: #007FBA
- accent: #B96628
- secondary_accent: #526479
- text: #11243A
- dark: #0C1B2B
- line: #CCD6DF
- hbm_tint: #E4F2FA
- hbf_tint: #FBECDD

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- body: 26
- title: 42
- subtitle: 24
- annotation: 18
- label: 22
- emphasis: 32
- hero: 64
- footnote: 16

## icons
- library: none
- inventory: none

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: dense
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: dense
- P13: dense
- P14: dense
- P15: dense
- P16: breathing
- P17: dense
- P18: anchor

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
