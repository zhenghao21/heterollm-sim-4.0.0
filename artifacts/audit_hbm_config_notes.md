# HBM 配置与地址映射审计笔记

审计日期：2026-10-15。审计当前工作区；父任务记录基线为 5a13823、当前 HEAD 为 c864890，另有 planner/tests 未提交修改。本分项只读取配置、映射与相关控制器代码，未修改产品代码。以下标准名、标准数值来源说明以代码和项目已存来源为依据，本分项没有独立进行 JEDEC 原文认证。

## 可确认的代码缺陷：浮点相加后对齐会凭空增加一个 CK（建议 P1，重跑正式性能结论前修复）

位置：`src/heterollm_sim/_hbm_control.py:83-85`；同一算法也在 `src/heterollm_sim/_hbm_numeric.py:31-33`。列命令时序相加的位置是 `_hbm_control.py:159-165`，对齐调用为 `:184`。

`align` 只对恰好等于 `round(value / quantum) * quantum` 的值免除上取整。由两个合法整数拍时间相加产生的浮点尾差不一定满足这个相等判断，因此不是仅有亚皮秒舍入，而是被放大为一个完整命令周期。

真实入口复现，使用项目 `.venv/Scripts/python.exe`，工作目录为仓库根：

```python
import sys
sys.path[:0] = ['src', 'scripts']
from calibrated_hbf_profiles import _hbm_physical_config
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest

c = _hbm_physical_config(capacity_bytes=512 * 1024**3,
                         read_gb_s=1600., write_gb_s=1600.)
for details in (True, False):
    core = DramCore(c, capture_details=details)
    for i in range(3):
        r = core.submit(AccessRequest(str(i), 'read', 0, 32, 0.))
        print(details, i, r.completion_ns,
              [(s.name, s.start_ns) for s in r.stages if s.name == 'READ_COMMAND'],
              r.counters.get('compiled_hbm_float64'))
```

| 请求 | 正确的整拍列命令时刻（第二次） | 实际明细路径 RD | 实际两路径完成时间 |
| --- | ---: | ---: | ---: |
| 首次 32 B 读 | — | 17.92 ns（7 CK） | 104.96000000000001 ns |
| 第二次同址读 | 28.16 ns（11 CK） | 30.72 ns（12 CK） | 117.76 ns |
| 第三次同址读 | — | 40.96 ns | 128.0 ns |

第二次读的主约束是同 BG 间隔 4 CK：`17.92 + 10.24` 在浮点中为 `28.160000000000004`，而第 11 个边沿表示为 `28.16`。当前 `align` 将其提升至第 12 拍。此时列命令总线只忙到 20.48 ns；数据总线只要求第二次 RD 晚于 `104.96 - 81.92 = 23.04 ns`，无其它资源迫使它等到 30.72 ns。按正确列命令时刻，第二次读应约在 115.2 ns 完成。

`capture_details=False` 的三个结果完全一致，且 `compiled_hbm_float64=True`；用完成时间减去 RL=81.92 ns 和 burst=5.12 ns 可以反推相同的列命令时刻。因此两个路径一致不能排除共同的时序错误。修复应区分内部整拍运算与真实外部到达时间，避免把任意晚到的请求一律用宽容差提前。

## 主实验配置：旧报告的数值仍成立，属于未标定的建模选择

调用链：`scripts/hbf_hbm_serving_study.py:109-118` → `scripts/calibrated_hbf_profiles.py:407-434` → `src/heterollm_sim/physical_presets.py:212-239` → `src/heterollm_sim/hbm_geometry.py:59-162`。

| 参数 | 512 GiB，读写各 1600 GB/s | 1 TiB，读写各 1600 GB/s |
| --- | ---: | ---: |
| stack 数 | 8 | 16 |
| 每 stack 层数 × die 密度 | 16H × 32 Gb | 16H × 32 Gb |
| 每 stack 物理通道 / PC 数 | 16 / 32 | 16 / 32 |
| 总 PC | 256 | 512 |
| 每 PC BG × bank | 16 × 4 | 16 × 4 |
| 每 bank 行数 / 有效行大小 | 32768 / 1024 B | 32768 / 1024 B |
| 访问粒度 | 32 B | 32 B |
| 聚合接口带宽 | 1600 GB/s | 1600 GB/s |
| 等效每引脚速率 | 1.5625 Gb/s | 0.78125 Gb/s |
| tCK | 2.56 ns | 5.12 ns |
| RL / WL | 81.92 / 20.48 ns | 163.84 / 40.96 ns |
| 每个 burst | 5.12 ns | 10.24 ns |
| tRCD | 14 ns | 14 ns |
| tRP / tRAS / tRC | 15.36 / 33.28 / 48.64 ns | 15.36 / 35.84 / 51.2 ns |
| tCCD_S / tCCD_L / tCCDR | 5.12 / 10.24 / 10.24 ns | 10.24 / 20.48 / 20.48 ns |
| tRASmax | 35100 ns | 35100 ns |
| tRRD_S / tRRD_L / tFAW / tRTP / tWTR_S / tWTR_L / tRTW | 全部 0，关闭这些额外时序门 | 全部 0 |
| REFab 自动刷新 | disabled，tREFI=0、tRFC=0 | 同左 |

数值均由实际 `_hbm_physical_config` 解析后的对象读取，不是手工替换字典猜测。

关键依据：

- stack 数按容量选择，聚合接口不随 stack 数自动增长：`physical_presets.py:215-219`；`_hbm_dependencies.py:83-84` 的公式由接口反推 CK。
- 配置本身已明确标注“等效调度速率，不是器件速度等级”：`hbm_geometry.py:333-351`。接口与读写方向限速已分离，降低方向 cap 本身不应被误写为必然降低 CK，见 `docs/hbm_geometry.md:70`。
- `hbm_geometry.py:373-377` 将未配置门明确标成 `unknown_disabled`，不是声称真实器件允许零时延。
- `refresh_cycle_ns_for` 在 `_hbm_dependencies.py:69-80` 对 HBM3 只收录 8 / 16 Gb 表项；32 Gb 返回 None。`hbm_geometry.py:295-297` 明确选择 disabled。tRASmax 仍独立配置为 9×3900 ns（`:288`、`:371-372`），不可将“没有刷新”写成“没有最大行激活时间限制”。

判定建议：将慢时钟、零时序和刷新关闭放入“仍需确认的模型假设 / 正式实验适用范围”，不能仅凭低速工作点断言 JEDEC 不合法。需要真实器件速度等级与厂商时序，才能判断实际偏差。当前实现把容量扩大同时改变 bank 并行度和 CK，因此 512 GiB→1 TiB 的比较也不是纯容量实验。

## 地址映射及“55%”旧结论

`memory_mapping.py:26-43` 以 32 B 为步长先轮转 PC，再 bank，再列 / 行；`:58-61` 用 `bank_group = bank_in_rank // banks_per_group`，所以同一 PC 依次访问 BG0 的 bank0/1/2/3、BG1 的 bank0/1/2/3……。该文件相对 5a13823 没有变化。

事实成立：同 BG 的连续列访问更多受到 tCCD_L 限制。单看一个 64-bank 循环，48 次同 BG 间隔、12 次同 SID 异 BG 间隔、4 次跨 SID 间隔，理想平均为 `(48×4 + 12×2 + 4×4)/64 = 3.625 CK`；与每 burst 的 2 CK 相比是 `2/3.625 = 55.17%`。这只是忽略行操作、启动、维护、浮点对齐等因素的局部推算，并非当前程序完整顺序读的确定吞吐。

最小复核实验：每个点创建新的 `DramCore`，地址 0，单个 READ 请求、arrival=0，配置同上；`max_outstanding_requests=64`、`max_expanded_segments=1_000_000` 均保持入口默认值。编译路径使用 `capture_details=False`。1 MiB 另用 `capture_details=True` 复核，完成时间与汇总路径一致。没有运行 serving/decode 全量实验。

| 容量 | 单次冷读长度 | 完成时间 ns | 吞吐 GB/s | /1600 |
| --- | ---: | ---: | ---: | ---: |
| 512 GiB | 1 MiB | 1364.48 | 768.4803 | 48.03% |
| 512 GiB | 16 MiB | 20349.44 | 824.4559 | 51.53% |
| 512 GiB | 64 MiB | 86405.12 | 776.6769 | 48.54% |
| 1 TiB | 64 MiB | 85688.32 | 783.1740 | 48.95% |

64 MiB 有 2,097,152 个 burst，大于明细展开上限，但汇总路径仍执行全部 burst，不能将该上限误当成只计算前 1,000,000 个 burst。这组微测既不能替代完整 TPOT / TPS，也不能把所有差额归因于 BG 映射；已确认的额外整拍缺陷也在其中。旧报告的约 880 GB/s 应改为“局部理论估算；当前少量冷顺序读实测约 768–824 GB/s，需先修复对齐缺陷再评价”。

不要把“真实控制器一定把 BG 放在低位”当成已证实事实；地址散列和调度是具体控制器策略。当前映射是清楚标注的线性模拟策略（`hbm_geometry.py:243-246`），需要敏感性实验才能归因到某种真实设备。

## 本分项未发现新问题的覆盖项

- 512 GiB / 1 TiB 的 raw geometry 容量与组织一致，未发现重复乘以 die 数或通过随意扩大 rows 凑容量。
- 32 B burst、每 PC 64 bank、1 KiB 有效页、2 KiB 激活页的配置沿用已有组织；2 KiB 激活页通过清除行地址最高位合并（`_hbm_control.py:405-406`），不是把相邻 1 KiB 行强行合并。
- 当前入口的速度、RL/WL、列间隔、IDD 参考行时序和 tRFC 有可追踪来源；不应将 `known_jedec_constraints_checked` 解释成通过真实器件全参数校准。
- 地址映射相对旧审查基线未改变，所以同 BG 先访问的现象不是新回归。
- 配置没有隐瞒未启用时序和刷新；问题是模型适用范围，不能用标签代替真实器件参数。
