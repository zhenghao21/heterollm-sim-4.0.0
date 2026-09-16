# 遥测采样与host波动独立审计

2026-09-16。只读已结束的collection_r3原始telemetry/QPC和采集源码，未运行GPU、未读取LLM actual、未修改冻结协议或质量判据。入口`telemetry_stability_audit.py`，不可覆盖输出`telemetry_stability_audit.json`。处理97个有完整应用/telemetry输出的进程（49profile、48direct），保留全部失败状态。

## 结论与实际失败定位

当前程序请求每5ms采样，但**各进程实际采样起点间隔的中位数约16.54ms**；不能把配置字段`period_seconds=0.005`当成实际达到5ms。25ms门本身不是这个失败的根因，应修采样器并新冻结，不能放宽门。

`train_Q8_0_m4_n4864_k896 / pair01 / profile`：

- 原进程exit0，但clock门失败，保留`failed_identity`。
- 669条有效QPC采样，0条采样异常记录；间隔中位16.382ms、P90=17.550ms、P99=19.557ms、最大33.206ms。
- 82.19%间隔在14–20ms，7.78%在4–8ms。不是稳定5ms采样。
- **唯一超25ms采样间隔**包围第7个formal调用：前样本QPC819018381510，后样本819018713570；频率10MHz，区间长33.206ms。
- 调用起点距前样本7.2703ms，调用终点距后样本**25.8618ms**，后侧超过原门0.8618ms；两样本之间没有内部采样。这是缺少及时观测，不能证明其间频率如何变化。
- 此过程可读取的SM数值均2392MHz，原正式门使用的60个读数也全2392MHz。它仍然不能补出漏采的33.206ms中间波形；失败应保留。

全部97进程没有显式telemetry错误行，合计31个sample-start间隔>25ms。只有目标过程的正式区间因此触发25ms夹取失败；其它长间隔位于非正式区间或没有导致两侧超过25ms。

| 统计（各进程先取中位数，再跨进程） | profile49进程 | direct48进程 |
|---|---:|---:|
| 实际sample-start间隔median | 16.5417ms | 16.5399ms |
| 数值验证段median | 32.8687ms | 33.0278ms |
| host wall P90/P10分布median | 1.9685 | 2.2087 |
| host wall P90/P10最坏 | 4.9271 | 5.4958 |

频率覆盖问题与direct host波动是两个指标：只有1个过程触发clock门，并不意味着其余direct wall已稳定。采样器修好之后仍须重新检验原host扰动和波动门。

## 源码可确认和不能确认的原因

`collection_r3/worker.py`的采样线程依次执行`gpu.sample()`，然后`stop.wait(0.005)`。R15 `pilot.py`的`sample`在起点记一次QPC，随后顺序读取graphics/SM/memory频率、P-state、温度、功耗和throttle等多个NVML字段。故相邻起点间隔包括**本轮所有NVML调用时间＋等待实际唤醒时间＋OS调度时间**，不是纯等待时间。

本机`E:/anaconda/Lib/threading.py`确认`Event.wait`进入`Condition.wait`的带timeout锁等待。当前源码未请求Windows高分辨率waitable timer，未显式管理timer resolution。真实16.5ms量级与Windows常见粗粒度timeout现象一致，**这里只能列为主要假设，不能宣布已经量出15.625ms timer quantum**：原始记录没有wait begin/end、NVML单字段begin/end、线程调度trace或当前timer-resolution值，无法把三部分精确分解。

目标过程每次formal的数值验证中位32.655ms、最长40.945ms，调用间隔中位34.235ms；目标host wall中位77.55µs、最大225.70µs。验证远比kernel/host计时段长，可能影响下一轮CPU缓存、设备空闲间隔及系统调度；但验证发生在**native子进程**，采样器在Python supervisor，不能误称它们争抢同一个Python GIL。现有证据不证明验证CPU工作就是direct host jitter的唯一原因。

不移除每次数字检查，不人为等待到固定时长“熨平”曲线，也不延长被测操作凑大分母。

## 下一版本的最小采样器改法（未实施）

1. 将采样调度改为**QPC绝对目标时刻**：`deadline=origin+i*period`，本次工作完成后等待到下个deadline，不能“工作结束后再等5ms”累积漂移；错过deadline时记录missed deadlines，不补发多条伪造相同时刻的采样。
2. 首选独立高分辨率waitable timer；本机Windows SDK 10.0.19041.0的`um/synchapi.h`确有`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`、`CreateWaitableTimerExW`、`SetWaitableTimerEx`声明。设计上用单次timer和stop event一起等待，stop能及时唤醒；不采用CPU忙等。记录创建flags与实际返回码，若不支持，不静默退化成“已5ms”。
3. 可选兼容后备是**同采样进程范围**的`timeBeginPeriod(1)`，并在所有退出/异常路径`timeEndPeriod(1)`对称恢复。本机SDK `timeapi.h`中API声明已核对。该后备须显式记录选择，不能在未验证时称其保证5ms；更改timer策略也可能扰动native运行，需新冻结并做开销对照。
4. 为每次SM读取记录`sample_begin_qpc`、`sm_read_begin_qpc`、`sm_read_end_qpc`、`sample_end_qpc`、`wait_begin/end_qpc`、`target_deadline_qpc`。现有单起点QPC只标记整组读取开始，SM读取实际在其后；新门应把**SM读取窗口**绑定到正式区间，而非把所有字段都假设在同一纳秒采到。
5. 高频关键通道可只做SM frequency及QPC，其余完整状态字段放在预先规定的较低频率通道；不能偷偷丢掉配置要求的数据。要保留完整身份、时钟控制回执、错误和缺测统计，用独立协议比较“原完整采样、关键采样、无采样控制”对native host/device的扰动。
6. 启动时等采样线程ready后再启动目标；退出时等待**确认采样线程结束**再final sample/close NVML/write。当前`join(timeout=2)`后未查`is_alive()`，极端NVML阻塞可能让close与采样并发，虽本轮未见异常，也应在新生命周期实现中消除。
7. **25ms、2400±30MHz和全部formal夹取要求保持原值**。先用不含GPU工作的定时器诊断确认实际采样间隔和尾部，再跑最小原生通用算子控制，观察timer分辨率变更是否增加扰动。新probe重复仍依原预设采样规则，失败不可删掉。

## 需要补的独立证据

- 同一进程策略下，Event.wait、high-resolution waitable timer、可选timeBeginPeriod后备的QPC唤醒分布；采样任务为空与真实NVML读取两组，分离等待与NVML成本。
- SM读取窗口与formal边界，原始线程/进程identity、无漏deadline声明只能由这些记录重算。
- 原数值检查完整保留条件下，记录单次验证前后耗时、CPU时间及下一调用host jitter，判断是否存在相关性；有相关性也不等于因果。
- 若下一版把参考数学预计算到timed循环前，仍须每次读回并验证全部规定样本，保持相同输入/检查逻辑，明确观察开销变化并新冻结；不能让一次检查替代36次。

本轮的结果只证明当前采样实现未达到请求的5ms节奏，并定位25ms门的单个缺测区间。没有重新把该格记为有效数据，也没有生成任何成本参数。
