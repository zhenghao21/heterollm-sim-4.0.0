# R16 合成算子设备与主机间隙诊断

2026-09-16。纯诊断，不拟合、不生成profile、不改planner、不读LLM actual、不运行GPU。使用**在03:00:15 UTC保存的26项预测**，SHA `66da88e6b8d24a1728945ddb17454ba42d028ec524f7b118fb3b175f1acb3c20`；对照collection_r2已完成的8个pair及collection_r3已完成首个MMQ pair，共270次formal调用。未读取r3在途矩阵。

入口 `device_gap_analysis.py`，不可覆盖结果 `device_gap_analysis.json`。逐输入SHA、逐调用区间分解和逐角色统计均在JSON。

## 结论

**不能把统一 `max(analytic, measured_device_floor)` 当作当前误差修复。** M1 MMVQ main当前高估38%–170%，下界不会降低它；conversion低估约90%，若仅补conversion，原本部分抵消的总误差反而扩大。M64 MMQ则conversion/main/fixup全低估，方向相反。应首先修正MMVQ主成本的并行度/访存机制，再针对MMQ工作路径和非GEMM固定设备服务收集独立证据。

所有以下数字均为不合格或证据不足样本的诊断统计。R16没有accepted标定；r3仅一pair且profile/direct差及direct波动失败，不能因device较稳定就绕过既定质量门。

## 设备角色结果

单位µs；区间为不同已完成pair的30次formal中位数范围，非置信区间。误差为(分析−设备)/设备。

| 路径与shape | 完整pair数 | conversion分析 / 观察 | main分析 / 观察 | fixup分析 / 观察 | device并集总误差 |
|---|---:|---:|---:|---:|---:|
| Q5_0 M1/N128/K896 MMVQ | 3 | 0.120 / 1.136–1.184（−90%左右） | 3.977 / 1.472–1.601（+148%–170%） | 无 | +47.1%–57.1% |
| Q5_0 M1/N896/K896 MMVQ | 3 | 0.120 / 1.216–1.280（−90%左右） | 3.934 / 2.145–2.496（+57.6%–83.4%） | 无 | +8.7%–17.3% |
| Q5_0 M1/N4864/K896 MMVQ | 2 | 0.120 / 1.184–1.376（−90%左右） | 7.856 / 5.696–5.713（+37.5%–37.9%） | 无 | +12.5%–15.9% |
| Q5_0 M64/N896/K896 MMQ | 1 | 0.421 / 1.729（−75.6%） | 2.727 / 7.248（−62.4%） | 1.593 / 5.440（−70.7%） | −67.2% |

每角色中位数之和不等于逐调用区间并集总量的中位数。计算脚本逐调用验证并集，未把分项中位数之和伪装成实测总时间。以上8个r2 pair全部是M1/Q5_0，不能推及Q8_0、M2/4或其他模型。

### 下界适配器对方向的影响

- MMVQ main-only下界：分析已大于观察，数值完全不改变，所以不能解决当前main高估。
- 若未来扩展到conversion，下界替换conversion低估会把M1/N128设备预测从约4.10µs抬到5.11–5.16µs，观察总量仅2.61–2.78µs；M1/N896会从约4.05µs抬到5.15–5.21µs，观察3.46–3.73µs；M1/N4864会抬到9.04–9.23µs，观察6.88–7.09µs。这里的“抬到”是分角色中位数代入的**方向诊断**，不是新冻结预测。
- MMQ三角色下界的方向是增加设备成本，符合低估。但当前resolver有意只支持MMVQ main；即便支持MMQ，也不能用这个失败pair的角色中位数生成系数，不能将“用实测替换后接近实测”当验证。

## 同一Nsight时钟下的主机/设备包络

每调用先按GPU活动，再按GPU之外的launch API、sync API、other API、剩余未归属区间做互斥分区。**逐调用所有分区和严格等于NVTX时长**。这只是观测归属优先序，不能解释成互斥的物理资源工作量。API与GPU重叠部分归GPU，不再次相加。

| shape | kernel并集中位数 | NVTX包络中位数 | 首kernel前 | kernel间隙 | 末kernel后 | launch API并集 |
|---|---:|---:|---:|---:|---:|---:|
| M1/N128（3pair范围） | 2.61–2.78 | 50.09–56.94 | 32.47–42.42 | 2.13–2.37 | 7.79–13.02 | 6.92–8.15 |
| M1/N896（3pair范围） | 3.46–3.73 | 59.55–74.90 | 46.13–59.80 | 2.30–2.43 | 7.55–9.17 | 10.50–12.75 |
| M1/N4864（2pair范围） | 6.88–7.09 | 78.48–90.19 | 59.51–70.47 | 2.11–2.22 | 9.46–10.10 | 13.06–16.10 |
| M64/N896（1pair） | 14.43 | 101.69 | 73.21 | 2.61 | 8.07 | 18.11 |

NVTX范围内包含event记录/查询、submit和等待。主机QPC wall从其他边界测得，其median与Nsight NVTX可分别展示，但**没有做跨QPC/Nsight时钟减法**。

互斥分区诊断中，MMQ首pair的GPU活动中位数14.43µs，GPU之外launch18.11µs、sync20.76µs、other API16.55µs、未归属26.55µs。分项各自的median不能相加当101.69µs；精确闭合仅逐调用成立。

r2 profile相对direct host中位数差+8.9%到+41.5%；r3首pair+28.4%。它们含注入、进程顺序、调度、事件instrumentation和状态漂移，**不能从该差值推导纯profiler开销**，更不能用负/正差校正kernel系数。direct也包含event instrumentation；还需event-free对照才能探究这部分扰动。

## 与现有planner成本归属对照

`reference.py:421`保留每kernel `kernel_launch_ns=1000`。本分析基线有MMVQ两次launch=2µs、MMQ三次launch=3µs，它们在单独launch_total字段中，未加入device成本。

`planner._add_physical_invocation_frontend`（7043起）另外有三类显式成本：

1. CPU command build：按声明指令数/CPU发射宽度与频率。
2. driver submit：按物理invocation_count×submission_ns。
3. GPU command processor：按launch_batch_size分批后的command_submission_latency_ns。

源码metadata明确三者不包含operator kernel launch，并将phase boundary校准保持独立。这里的“一个物理invocation group”与合成probe的“一次独立图调用”不能未经账本证明直接等同。

观察到的6.9–18.1µs launch API并集**不等于**每kernel设备launch latency；其中可能包含driver CPU工作和Nsight开销。更不应把50–102µs NVTX包络全部放入kernel_launch_ns，否则会与已有CPU build/driver submit/GPU CP、转换/main/fixup和同步重复计费。

## 机制假设与下一步允许证据

### H1：MMVQ错误继承了MMA输出tile波次的HBM效率

冻结预测四个代表main均是memory bound。M1/N128与N896分别预测3.977、3.934µs，远高于实际1.47–2.50µs；M1/N4864仍高38%左右。当前 `estimate_gpu_gemm` 对generic MMVQ使用fp16 MMA tile输出波次来缩减HBM带宽，而实际是量化向量/DP4A kernel，CTA/warp按N行及行列块组织。MMA几何不一定对应MMVQ并行访存。

建议：从锁定mmvq.cu和实际grid/block导出独立CTA、warp、每CTA处理行数、逻辑K循环；将其与访存并行度模型对照。补允许的通用MMVQ shape扫描、HBM/L2实际事务或稳健设备计时，先做结构/量纲推导与留出N，再决定是否替换MMA-wave proxy。不能按这三个N写倍率表。

### H2：conversion短kernel有未建模设备控制/归约延迟

M1 conversion同shape、不同N均约1.1–1.4µs，而纯scalar+bytes估计0.120µs。已知11×padded元素工作只覆盖abs与部分归约，shuffle、round、转换、CTA控制、最小设备执行延迟未完整表达。M64 conversion分析0.421µs也低于1.729µs。

建议：独立conversion-only微基准，明确MMVQ half scale/original sum与MMQ D4 F32布局分开；短/长M及padded K边界留出。不能将约1µs直接定义为CUDA API启动成本，因为这里测的是kernel start/end。

### H3：MMQ主计算和fixup的tile/stream-K服务仍不完整

M64主计算低62%、fixup低71%，与MMVQ主计算高估方向相反。应核对actual grid/block/shared-memory与source partial writer数量/有效元素/完整K iteration；fixed tensor/scalar吞吐可能缺数据交换、归约和控制服务。

建议：在不增加LLM测量的前提下完成合成MMQ主/尾部/对齐组，按角色独立验证源码工作量，再对比不同K、N、stream-K分区。不能用host graph wall补main差额。

### H4：图调用主机供给与同步边界尚需独立建模

首kernel前32–73µs和sync-only约18–25µs明显大于设备main，但不证明LLM每投影都产生同等开销。probe每图有同步、event记录和数值检查外部间隔，运行时路径与长LLM graph不同。

建议独立采集不含目标LLM时延的控制微基准：一图多kernel、多个依赖图连续提交、event-free和event配对、明确一次最终同步；用QPC/实际CUDA API相关关系定位CPU build/driver submit/device wait。时间线按真实依赖建模，不把图wall、APIwall、kernelwall直接相加。新协议需预先冻结和重新检查扰动，不修改当前失败样本质量结论。

## 接续优先级

先把H1的MMVQ物理并行度与H2的转换设备服务分离，避免通过局部“补高”破坏已高估main；继续收集MMQ独立留出证据核对H3；最后以H4控制实验解释host gap。R16当前结果仅有诊断价值，没有达到可以激活kernel profile的质量条件。
