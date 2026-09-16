# Conversion机制审计：H2/H3

结论：现有conversion的字节语义与源代码的普通连续二维路径基本相符；主要缺口是**把少数CTA的短依赖核按全GPU标量吞吐估时，且只计部分源表达式**。先下沉真实CTA/warp/归约依赖与运算类别，再取得独立微基准参数。不能把已拒绝测量中约1微秒差额写成常数，也不能把host launch/API同步再加到device conversion中。

只读锁定源码与已有完成trace；没有读取LLM actual，没有GPU运行、重测、改core或拟合参数。可复现轻量入口`conversion_mechanism_derivation.py`，输出`conversion_mechanism_derivation.json`，含源码SHA与逐配置单位。9份已完成mapped trace（8份r2 MMVQ、1份r3首MMQ）中全部36调用的conversion grid/block与源码推导一致；这仅验证物理形状，未读取其duration字段用于推导。

## 1. 源码工作量与当前proxy

来源：`quantize.cu:54–100,456–555,558–603`；`quantize.cuh:8–12`；`common.cuh:466–472,526–533`。普通连续2D、非scatter、F32输入；K必须按格式约束对齐。定义`Kp=ceil(K/512)*512`、`E=M*Kp`。

| 路径 | 实际grid/block | 源级lane表达式 | 当前operations含义 |
|---|---|---|---|
| MMVQ Q8_1 | grid=(Kp/256,M,1)，block=(256,1,1)，线程E | E次abs，5E次浮点max，5E次浮点add，10E次shuffle；非零块至多2E除法表达式、E次roundf及int8转换；每32元素2次half元数据转换 | 11E仅包括abs/max/add；**不是完整CUDA指令数** |
| MMQ D4 | grid=(M,Kp/512,1)，block=(128,1,1)，线程E/4 | 每线程4abs＋6max＋3shuffle＋4multiply＋2division＋4roundf＋4int8转换；每8线程一次F32尺度存储 | 14*(E/4)包含abs/max/multiply；漏掉shuffle/division/round/cast/控制等 |
| MMQ DS4 | 同D4 | D4基础另每线程6add＋3shuffle，每32值half d和half原和 | 20*(E/4)，附加6add；仍为部分proxy |

D4/DS4按`mmq_get_q8_1_ds_layout`选，不能仅按“8bit激活”合并。Q5_0与Q8_0为D4，Q4_0/1、Q5_1为DS4；K格式另外遵守源码映射。D2S6、scatter、MXFP4/NVFP4、专家多写路径不属于本次推导。

注意：以上是源级表达式数，**不是编译后的SASS指令计数**。`roundf`与int8 cast可能合并；除法可能降成倒数、多指令或近似路径；零padding块会改变分支/转换行为。不能给每一表达式统一“1cycle”。MMVQ和MMQ归约树有依赖，SIMT的warp指令与lane工作单位要分开。

## 2. IO语义与依赖

- MMVQ读取逻辑F32：`4*M*K` bytes；写Q8_1：`36*M*Kp/32` bytes。每32值32字节q＋4字节half2元数据。
- MMQ读取同上；写D4/DS4：`144*M*Kp/128` bytes。每128值128字节q＋16字节尺度/和。
- 两者本域都等于`9/8*M*Kp`写字节，但布局、数值含义、每线程工作和消费方式不同。
- 非零逻辑输入以外的padding仍执行部分计算和输出写入；**不应该增加F32读取字节到4*M*Kp**。
- conversion生产的临时激活必须在main消费前完成。现有planner先lower独立conversion，main activation_storage_bytes替换为Q8布局，方向正确；MMQ consumer extra/window不是conversion额外写入，不能混算。
- 这里没有权重反量化工作。Q5/Q8 decode/整数点积在main，不能作为conversion的dequant成本补入。

## 3. 为何全GPU吞吐不合适

K896→Kp1024时：

| 路径/输入 | CTA | 总warp | 最多可同时占用的SM数上界（84SM） |
|---|---:|---:|---:|
| MMVQ M1 | 4 | 32 | 4 |
| MMVQ M2 | 8 | 64 | 8 |
| MMVQ M4 | 16 | 128 | 16 |
| MMQ M64 | 128 | 512 | 84 |

`cost_models.py:1847–1857`的elementwise_gops乘全部84SM、全scalar lanes与occupancy；`estimate_gpu_tensor_kernel:2814–2874`直接把partial operations除这个值。M1实际仅4CTA，没有机会同时使用84SM。这是**结构性资源可用性缺口**；当前统一occupancy不是CTA分布与尾波的替代。

但CTA数量也不能机械变成HBM带宽利用率。4个SM仍可产生多个未完成内存事务，L2/DRAM控制器路径独立，短核可能受依赖和固定设备调度延迟限制。**禁止把带宽直接乘4/84、禁止用CTA波数同时扩大compute和memory。** MMQ128CTA的SM上界已够覆盖，不表示每SM一定满occupancy；寄存器、每CTA warp、驻留上限、指令issue和访存延迟仍影响它。

## 4. 关键路径与资源归属

MMVQ有两棵32lane浮点树，各5层shuffle＋max/add。它们的数据依赖链彼此部分独立，但最终都汇聚到量化或half元数据；不能简单把两树总工作量当10cycle关键路径，也不能把所有lane操作串行累加。MMQ D4每线程4值先做3个串行max，再3层跨8线程max，接着division/scale→4值round/cast；DS4另有原和分支。

当前TensorKernelWorkload支持dependency_depth，typed roofline能取`max(throughput_ns,dependency_depth/frequency)`；但dependency_depth的cycle定义需要真实编译/标定，不能把“归约层数”当cycle直接注入。先输出有单位的source DAG和unpriced节点，再为shuffle/fmax/fadd/div/round/pack取得同架构、同runtime参数。

`common.cuh:133–143`还有Hopper以上PDL grid dependency同步/完成通知条件；是否实际生成/启用受编译及launch属性影响。**不能把存在源码调用等同于一次host同步**。未知时保留metadata，不凭空追加固定CPU等待。

现有`_estimate_typed_roofline`单独建立kernel_launch phase，然后scalar/SFU/memory在同device phase取max。launch已有收费；真实device区间缺口不是再次添加host API墙钟的理由。若后续证明短核最低设备服务延迟，应在同device资源阶段作为经过独立验证的条件包络，并标明该包络不代表HBM占用。sync/API wall与kernel elapsed包含/重叠关系不能相加。

## 5. 建议最小改动（尚未修改core）

第一步增加`ConversionWork`源映射，仅输出路径、Kp、grid/block、CTA/warp、逻辑IO、源表达式类别、归约树结构、PDL条件与未定价项；保持默认成本数值不变，按锁定源码和实际launch geometry做结构回归。

第二步仅在受证明配置中，将scalar throughput的可并行资源上界限制到`min(CTA_count,SM_count)`，并对warp/驻留/尾波显式建模。这个上界是必要约束，**不是足够的准确成本公式**；应保留old baseline与新机制比较，不能先按观察差额推“efficiency”。memory保持原字节服务，不共用scalar并行度罚因子。

第三步在独立质量合格的通用benchmark上标定或验证指令issue/依赖与短核设备服务；若质量门仍失败，第二步只能称分析候选，不能宣布达到实测精度。

## 6. 最小独立证据需求

1. conversion-only F32→Q8_1 / MMQ D4 / DS4，固定ordinary contiguous；真实原生DLL、同device/clock，实际kernel路径逐条核验。
2. M短/中/长和K512边界附近（K480/512/544/896/1024），必须区分逻辑K、padding与CTA台阶；训练/留出预冻结。
3. 单独验证shuffle树、division/round/pack的依赖延迟和可吞吐，并保存编译SASS/资源用量；source expression不能代替opcode计数。
4. event-free/event/profile配对、多独立进程，保持已有质量阈值；CPU供给gap与device区间分别保存；cold/hot缓存作为不同域。
5. 检验大小矩阵中CTA饱和前后趋势；不强迫线性或单调。所有失败、未测和未知仍保留。

本轮可以确定“operations不完整且scalar资源上界未体现有限CTA”，不能据不合格r2/r3时间确定缺口分别有多少纳秒。H2不能单独补高来掩盖H1 main高估，H3也不能沿用同一个常数。应先完成源工作量结构与资源约束，再用独立benchmark决定成本模型。
