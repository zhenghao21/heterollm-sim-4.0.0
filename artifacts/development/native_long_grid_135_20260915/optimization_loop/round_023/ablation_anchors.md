# R23 retained KV 两路开发消融

阶段：anchors。retained 全三项严格小于10%：0/131；A=not_passed，B=unvalidated。新增系数0。

current 与 retained 同为 R22 CTA+issue 组合，只改变 retained warmup-state 开关。固定62格 ordinary 候选，69格 hybrid 保留原机制回退；model_key 只用于报告分组。

| 20-anchor 对照 | 全三项<10% | TTFT APE中位/P90/最坏 | TPOT APE中位/P90/最坏 | E2E APE中位/P90/最坏 |
|---|---:|---|---|---|
| current | 0/20 | 67.626% / 72.349% / 76.170% | 47.908% / 61.187% / 65.332% | 49.837% / 65.869% / 68.147% |
| retained | 0/20 | 67.362% / 70.580% / 72.496% | 47.696% / 59.979% / 61.493% | 49.256% / 64.035% / 65.882% |

APE为绝对百分比误差；P90是场景误差分位数，不是请求延迟P90。

| 独立对照 | 指标 | 改善/恶化/不变/未评分 | 数值改变格数 | APE差中位（百分点） | 模拟时延差中位（ms） |
|---|---|---|---:|---:|---:|
| retained_minus_current | TTFT | 10 / 1 / 9 / 0 | 11 | -0.0036360442610785526 | 0.000919482872743238 |
| retained_minus_current | TPOT | 11 / 0 / 9 / 0 | 11 | -0.27765843301402526 | 0.013310873842592441 |
| retained_minus_current | E2E | 11 / 0 / 9 / 0 | 11 | -0.2489783989779646 | 0.689787413164801 |

差值均为后者减前者；APE差为负表示误差下降。未评分保留在每指标20格分母内。
retained 已保存预测 20/131；缺失 111；失败或未评分 111/131。
终态结果（含failure）20/131；三项可评分 20/131。
新增机制uncovered只降低retained覆盖，不删去原数值预测。原R22的SHA失败仍属原实验失败；本轮新源码预测不得改写旧结论。

| 分组 | 固定分母 | 三项严格<10% | 失败/未评分 |
|---|---:|---:|---:|
| model:qwen25 | 17 | 0 | 0 |
| model:qwen35 | 22 | 0 | 21 |
| model:qwen38 | 20 | 0 | 20 |
| model:qwen38_gpu | 27 | 0 | 27 |
| model:smollm2 | 23 | 0 | 22 |
| model:tinyllama | 22 | 0 | 21 |
| scope:fallback | 69 | 0 | 68 |
| scope:retained_covered | 62 | 0 | 43 |

未评分原因（按指标计数）：
- prediction unavailable or incomplete: 333

逐格误差、失败、四路引用和前后哈希保留于JSON。缺失/失败不从131分母移除；A通过也不代表B独立验证通过。
