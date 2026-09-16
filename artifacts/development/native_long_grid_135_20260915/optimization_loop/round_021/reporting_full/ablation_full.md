# R21 nonflash KV 三路开发消融

阶段：full。physical 全三项严格小于10%：9/131；A=not_passed，B=unvalidated。新增系数0。

pure：无 sampling/MMQ/nonflash；current：sampling+MMQ；physical：current 加 source-bound nonflash KV 下界，池状态仍未知。

| 20-anchor 对照 | 全三项<10% | TTFT APE中位/P90/最坏 | TPOT APE中位/P90/最坏 | E2E APE中位/P90/最坏 |
|---|---:|---|---|---|
| pure | 0/20 | 77.726% / 81.201% / 83.782% | 51.401% / 68.391% / 68.861% | 54.136% / 72.266% / 75.354% |
| current | 0/20 | 66.716% / 72.506% / 76.477% | 46.544% / 61.139% / 62.772% | 48.357% / 64.857% / 67.298% |
| physical | 0/20 | 67.626% / 72.349% / 76.170% | 47.908% / 61.187% / 65.332% | 49.837% / 65.869% / 68.147% |

APE为绝对百分比误差；P90是场景误差分位数，不是请求延迟P90。
- pure_to_current：improved=60；指标分母60。
- current_to_physical：regressed=32, improved=28；指标分母60。
- pure_to_physical：improved=60；指标分母60。

physical 已保存预测 131/131；未保存 0；失败或未评分 0/131。

physical 全131 APE：
- TTFT：42.476% / 68.193% / 77.568%
- TPOT：26.099% / 53.545% / 70.847%
- E2E：30.001% / 56.263% / 72.829%

逐格完整误差、失败状态、来源引用、预测前后哈希与评分顺序检查保留于同名JSON。缺失/失败不从131分母移除；不得将开发结果当独立验证或校准。
