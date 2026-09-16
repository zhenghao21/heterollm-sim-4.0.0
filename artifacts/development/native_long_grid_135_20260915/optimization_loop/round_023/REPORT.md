# R23 retained-KV 完整结果与参考方向审计

R23完成固定131格终态。**104格可评分，27格证据重派生失败；5/131格三项Engine误差严格<10%。A失败，B未验证。** native仍为原131格894请求，未重测或拟合目标LLM。

## 结果与可比较范围

| 指标 | APE中位数 | P90 | 最坏 | 绝对ms中位数 | 最坏ms |
|---|---:|---:|---:|---:|---:|
| TTFT | 51.281% | 68.045% | 72.496% | 60.648 | 2667.094 |
| TPOT | 35.277% | 53.228% | 64.773% | 2.175 | 125.907 |
| E2E | 36.456% | 54.676% | 66.189% | 332.814 | 25473.633 |

此表只描述104个可评分格，不能与R22的129格总体统计直接比较。共同104格312项：117改善、3退化、192不变。62普通attention格仍0格三项通过；5个通过格均来自未改变数值的27B CPU部署。3项退化均为TTFT，最大APE增加0.030951个百分点（约−0.00536ms的预测变化），详见closeout.json。

## 失败与输入身份

27个qwen38_gpu格均报 retained warmup proof differs from raw/source re-derivation。CPU与GPU共享同一GGUF路径，冻结时路径缓存保留CPU snapshot引用的size_bytes字段，GPU独立worker使用bytes字段。静态重派生逐项比较发现唯一差异为这两个长度别名，不能通过忽略矛盾字段或重试R23来修复。

原R23失败、freeze、预测和分数保持不变。规范引用修复进入R24新版本，必须严格拒绝冲突别名和错误类型，核对CPU/GPU两种遍历顺序及独立worker，再以新冻结运行。此原因与R22两次未归因SHA故障不同，不能将它用来解释或抹掉R22失败。

## 对用户参考清单的采用

采用误差符号、固定其余变量后的P/O/C趋势、逐请求指标闭合、native波动分层及成对开/关消融；这些是诊断证据，不是按误差调参依据。方向和预计受影响集合必须在候选评估前由机制规定。

已纠正：131格不是仅Qwen2.5；三个P值均整除ubatch64；c不等于实际kernel M；当前MMVQ分派、nonflash主链、logits输出行和部分采样已有实现，不能整段重复收费。每步总开销可以随真实节点/提交次数变化，不能凭并发直接改单次系数。native CV、最大相对中位偏离与预测APE不能混用，不能事后删固定格。

894条native与292条R22预测请求逐条计时闭合通过，残差至多2.91e−11ms；分别取中位数并不保留该恒等式。三次重复不证明总体测量稳定性，也不足以给出正式联合置信保证。

真实后续缺口：末层FFN前选行的生产绑定需要确认；MMVQ HBM仍借用MMA输出tile利用率，须独立微基准而非直接调高带宽；host提交与GPU执行重叠，CPU cgraph与CUDA Graph不可混同。详细证据见optimization_direction_*、host_cost_ownership_audit.md、mmvq_memory_geometry_audit.md。

## 冻结与复现

同源码两路current/retained，各131静态输入、114共同源码文件。先保存40锚点预测再评分，再保存主候选全131终态凭据并评分；前/恢复/结束校验成功。汇总器文案中“四路引用”是历史标签残留，实际严格校验两路，不修改已冻结脚本。

复核入口：在仓库根运行 E:/anaconda/python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_023/evaluate_candidate.py summary --scope full。已有预测不得覆盖；从Git恢复较大JSON时使用本目录restore_evidence.py及detailed_evidence.parts.json，源码按代码提交dc776e2与freeze中的manifest恢复校验。

完整热图为full_engine_error_heatmap.png/svg；X是失败，短横线是原始162格中未入固定集的格。ablation_*图仅20锚点，不能替代全量图。所有结果为已揭盲开发回归，不是独立泛化验收。
