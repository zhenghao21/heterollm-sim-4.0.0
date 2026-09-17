# R31 Q4_K固定形状条件计时准备

唯一开发形状Q4_K/M1/K2048/N2048，源于已有静态预测命中。Q5零命中不再测；M4及K1024/4096留出。复用冻结R26 all-type连续布局host shim对象、Q8转换和support对象，只以type12、strides8/64/2048调用，不重建完整CUDA模板。

先运行独立wrapper资格：一次不计时warmup和一次捕获conversion→main，匹配R29的完整ExC430参数/PDL/几何，拒收捕获内alloc/free/copy/set；只允许末尾同步。使用同一已冻结R29 packed dyadic fixture，Q8逐字节和全部2048输出由独立Python参考重新核算。R29成功不自动授予新wrapper资格。资格未通过，计时入口硬阻断。

后继仅单批5 blocks×5 states×4 U/A/B/AB模式，最多100正式进程、1 HES能力进程；每进程64正式calls，至少32 warmup且0.5秒、最多5秒，观察器priming32；总墙钟预算1200秒，单进程软预算90秒，超过只停止新slot并等待已有进程自然退出，无kill、失败覆盖、自动重试或区间扩展。

权重状态预登记为单地址warm或独立地址轮换>4×L2；轮换不是已证明冷缓存，最多128 slots/1GiB。实际kernel key、PDL、连续布局、DLL/cubin身份在protocol中锁定；预测缓存/布局未知的cell继续未知，不能自动迁移到所有命中。

沿用R28的合法HES调用顺序、U/A/B/AB envelope门、STATE检查和自然等待。HES body仅条件经验观察值；envelope等价不证明单kernel服务成本无偏。保持service_cost_qualification=unvalidated，不产生全局带宽/issue系数或LLM拟合参数。

所有GPU/计时仅由root串行启动。构建及主机测试不调用GPU。
