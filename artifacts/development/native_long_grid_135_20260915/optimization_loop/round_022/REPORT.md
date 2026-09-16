# R22 机制优化结果

**结论：A门失败，B门未验证。131格保留完整终态，其中129格可评分、2格身份校验失败，9格三项严格小于10%。**

与R21 physical的129格共同可评分结果相比，0格模拟时延发生变化。20锚点中CTA-only与CTA+issue各三项均完全不变；不把结构修复称为精度改善。

| Engine指标 | APE中位数 | P90 | 最坏 | 绝对ms中位数 | 最坏ms |
|---|---:|---:|---:|---:|---:|
| TTFT | 42.854% | 68.249% | 77.568% | 80.672 | 2667.094 |
| TPOT | 27.517% | 53.684% | 70.847% | 2.585 | 125.907 |
| E2E | 31.320% | 56.400% | 72.829% | 382.285 | 25473.633 |

误差分布只在129个可评分格统计；失败2格保留在131格完成率、覆盖率和A门分母中。未评分不能按0误差算入，也不能静默删除。

## 机制与证据

- MMVQ指令下界实际进入生产解码图：所检查1450任务图中120个主GEMM应用，但120个均由HBM demand主导。权重、Q8_1输入及F32输出字节可精确复算，没有发现这两个代表算子的物理字节重复计费。shape带宽仍为未标定分析模型。
- 独立长图完整18阶段结束，数值/时钟/CUPTI链通过；设备kernel跨进程中位数偏差0.0274%/0.0803%。host P90/P10失败，0组性能系数准入。固定CUDA DLL编译未启用CUDA Graph，本probe逐node launch。
- 两次worker文件SHA失败原样保留；随后.NET/OpenSSL/HACL复核一致，512次PDF双算法检查零异常、全模型双算法一致。历史失败原因尚未确定，不能追认成通过。
- R23 retained-slot生命周期与适配器已在隔离工作树实现并独立审阅，普通attention有62格候选，hybrid69格继续旧机制数值回退。下一冻结不修改本轮工件。

## 复现与交付

- evaluate_candidate.py / evaluation_protocol.json / evaluation_controls.json 固定四路源码与输入；anchors_predictions.json、anchors_scores.json、full_predictions.json、full_scores.json证明先预测后评分。
- ablation_full.json/md保存分组、逐格、两种独立消融；full_engine_error_heatmap.png/svg为全131热力图。
- closeout.json保存全部失败原因及R21共同格对照；hash_integrity_protocol.json/result.json保存仅用于诊断的校验检查。
- 原始/详细证据包保留本地并按字节验证归档；不包含模型、DLL、EXE、OBJ或整个源码副本。
