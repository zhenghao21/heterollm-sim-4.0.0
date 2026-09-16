# R20：缓冲记录改善微基准波动，误差目标仍未达成

本轮没有重测目标 LLM，没有拟合任何新成本系数。固定 131 格、894 正式请求及全部原始文件 SHA 一致。最新推理误差仍是 R18 的同版本 20 锚点，0 格三项同时小于 10%；没有把不同冻结结果合并。

## 独立测量结果

CPU reset A/B：6 进程、36 stage、2304 个 steady 样本；12 个跨进程中位数组均在 ±5% 内。单次波动门 control 16/18、source-loop 15/18，合计 31/36。源 reset 未显示降低波动的证据，整组拒收；没有选取较快 arm 来拟合。

GPU E262144/G8 两臂对照完成 12 原生进程和 6 导出，共 432 次图调用、1728 个真实 trace kernel。全部数值、依赖、模块、PID/TID、trace 关联和采样时钟门通过；两组最终仍因计时质量拒收，0 系数。控制臂逐 call 核验，buffered 仅 first/post-warmup/post-formal 三个边界全节点核验，不能称为逐 call 数值证明。

| 记录方式 | 3 对 direct P90/P10 | 超过 profile/direct 20% 扰动门的对数 | 组结论 |
|---|---|---:|---|
| 每调用核验并写出 | 2.055 / 2.141 / 2.324 | 2/3 | 拒收 |
| 固定缓冲、阶段边界核验 | 1.303 / 1.114 / 1.313 | 1/3 | 拒收 |

三个 profile 进程的 kernel 中位数变化：control 0%，buffered 约 0.096%。这是同一合成图的诊断证据，不能据此声称目标 LLM native 测量也已改善，或将主机差额作为统一 launch 常数。

首次 collector 只运行了一次 direct，因清单漏绑定 3 个 CUDA DLL 与 2 个驱动 DLL 而拒收，频率恢复成功。补齐的五个库逐个与 R18 实际加载记录及当前文件 SHA 一致；新建 r2 冻结重新运行完整两臂，原批次不合并。r2 的 18 阶段全部自然完成，finally 恢复 GPU 频率返回 0。最初的简化解析器没有进入正式采集；最终解析复用 R18 的真实 StringIds、PID/TID namespace、correlation、geometry、同步及其他 GPU 活动检查。

## Simulator 修改与验证

MMVQWork 接入 planner/cost metadata，仅在锁定源/运行时合同满足时标注 vector DP4A 的 CTA、warp、K-loop 和物理格式信息；没有合格数值率时明确 `source_geometry_priced=false`。旧 MMA 数值仅保留为分析回退，不再声称它代表真实 MMVQ 的并行度。

根代理 132 项成本/转换/原生调用回归及 18 项最终采集器测试通过。基于前一提交的隔离源码，对 Q5_0/Q8_0、M=1/2/4/8/64 共 10 个合成输入，702 个任务的非 metadata 字段完全一致，包含依赖与资源成本。故本轮不重复运行数值相同的 LLM 预测，也不宣称误差下降。

原始6份SQLite、6份Nsight报告和较大的逐事件/遥测JSON均保留原字节，33份详细证据由13.14MiB压缩为1.05MiB的detailed_evidence.tar.gz；索引逐个记录SHA。执行restore_evidence.py可恢复，已有文件只核验、不覆盖。全部33份已做解压字节核对。

## 自动续轮

R21 检查微图 5 次 warmup 是否过短：新冻结比较 buffered/0ms 与 buffered/1000ms 额外工作预热，保留原始 first、5 warmup、30 formal 的记录语义，阈值不变。预热同时影响主机、设备与缓存，不能宣称 CPU-only 因果。并行重审实际 request/microbatch/Engine 边界和内核启动/同步覆盖，优先寻找可验证的结构修正，避免持续用准备文件替代预测改进。
