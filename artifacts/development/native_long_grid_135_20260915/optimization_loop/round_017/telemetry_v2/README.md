# Round017 高精度SM遥测准备

已经完成独立helper、16项hostmock测试，以及一次授权的2秒只读NVML采样。未启动推理、未写GPU时钟/功耗/优先级、未修改旧collector或任何旧测量。此目录准备给root集成到**新的**collector版本。

## 实际2秒结果

运行目录：`runs/20260916T043918410245Z`。

| 指标 | 结果 |
|---|---:|
| 请求/实际采样条数 | 400 / 400 |
| 漏deadline | 0 |
| 采样起点间隔median | 5.0146ms |
| 采样起点间隔P90/P99 | 5.2052 / 5.3846ms |
| 最小/最大间隔 | 4.0072 / 5.5250ms |
| 超25ms间隔 | 0 |
| SM读取耗时median/P99 | 8.7 / 20.1µs |
| 最长单次SM读取 | 511.4µs |
| wake lateness median/最大 | 0.3069 / 1.0380ms |

400条采样的NVML调用均成功，全部399个间隔在4–6ms内。非daemon线程正常结束，NVML在最后一次读取返回后关闭，timer/stop句柄在join确认后关闭，生命周期错误为0。运行前后源码身份一致。

实际SM频率2707、2715、2722、2730MHz，**不在2400±30MHz域**。本次未运行正式kernel，`formal_clock_gate=not_evaluated`、`cost_calibration_eligible=false`。不能把采样节奏结果当作锁频验证或仿真误差验收。

此前R16采样间隔约16.54ms；本次改善到约5.015ms，但同时将高频NVML读取从多个状态字段缩减为单个SM字段，因此**不能把全部改善仅归因于timer API**。两个实验系统负载也非受控相同。下一步需要保持完整身份/低频状态采集，另做采样开销对照。

## 机制和边界

`sampler.py`：

- 建立匿名`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`和stop event，均不继承，不调用`timeBeginPeriod`，不更改系统或进程timer resolution。
- 目标调度使用`origin+i*period`的绝对QPC deadline；Win32 timer接收的是每次重算的**负数相对100ns due time**，不把QPC值误当UTC absolute timer。
- 采用单次timer和stop event共同等待。超过deadline后跳过遗漏槽并记录，禁止连续补发假采样。每条保留deadline、arm、wake、wait、sample及独立SM读取起止QPC。
- NVML reader只在初始化时读设备UUID/name/driver和DLL身份，高频通道每次一次SM clock调用。没有任何NVML setter。
- worker独占NVML生命周期；join超时保留活线程和句柄所有权，不标完成、不在活线程下关闭NVML。stop事件用于协作结束，不杀线程/进程。
- 高分辨率timer创建失败时明确失败，没有静默降级。兼容后备timer resolution方案未实现，因此没有隐含全局修改。

`assess.py`：重算采样间隔、读取耗时、wake lateness、缺槽和读数错误。额外提供SM读取**窗口**的formal夹取门：前样本完整读取结束≤formal起点，后样本读取开始≥formal终点；以最保守的前读开始/后读结束距离检查25ms。原2400±30MHz和25ms阈值不变。该函数检验调用方传入的区间；collector集成时仍须用冻结清单强制全部30个formal身份、顺序与计数完整，不能传子集后宣布整配置通过。

## hostmock测试

`python test_sampler.py`：16项通过（0.015秒），不访问Windows timer或GPU。覆盖绝对deadline不累积read cost、分数QPC tick向上取整、超时跳槽、窗口外late wake、stop先于过期工作、读取失败保留、独立读取窗口、无formal不宣布验收、join超时禁止关闭、factory失败清理、2392MHz有效、2000MHz拒绝、25ms不得放宽和缺读窗口拒绝。

`python run_diagnostic.py --real-nvml --seconds 2`是显式只读诊断入口，输出新的UTC目录，不覆盖任何结果。无`--real-nvml`会拒绝执行真实采样。运行目录保留manifest、原始样本与总结。这里不负责启动LLM或改变clock控制生命周期。

## API核对

本机Windows SDK 10.0.19041.0 `um/synchapi.h`声明了`CreateWaitableTimerExW`、`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`和`SetWaitableTimerEx`。实现用ctypes显式指定参数类型、检查创建/设置/等待/清理返回值。实际2秒运行已成功创建高分辨率timer并取得上述QPC证据。没有将“请求5ms”写成硬实时保证。

尚未完成：新collector集成、真实formal频率域验证、采样扰动实验和新的算子矩阵。旧R16失败记录和固定native数据完全未改。
