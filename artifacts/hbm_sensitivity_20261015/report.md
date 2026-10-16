# HBM 基本时序、参数范围与控制器敏感性

执行版本：10；范围完成状态：passed。
计划/已执行：微基准 112/112，完整服务 34/34。只有全部计划点执行并通过Core/Decode检查才标为passed。

三档参数是公开资料不足时的工程假设范围，未标定厂商器件；不代表统计置信区间。
主参考保持 HBM3E 32 Gb × 16H、512 GiB、0.625 ns、读写方向预算各1600 GB/s。所有微基准使用相同主机地址与到达时间。
none 对照明确省略若干时序与刷新；它用于解释修正的影响，不作为真实硬件基准。

## 已观测的影响

- 关闭刷新，顺序读：相对参考组的有效带宽变化 +14.34%。
- 关闭激活频率约束，密集开行：相对参考组的有效带宽变化 +8.87%。
- 关闭读写恢复门，同PC混合读写：相对参考组的有效带宽变化 +27.50%。
- 优先已开行，热点行冲突：相对参考组的有效带宽变化 +712.55%。
- 读写分组，同PC混合读写：相对参考组的有效带宽变化 +132.61%。
这些探针用于分离机制影响，不能直接当作LLM加速百分比；完整服务结果单独列出。

## 参数与来源

以下为配置中的时间下限，单位ns；发令仍按CK/半CK边界对齐。nCK输入随实际时钟转换，不把周期数当作ns。完整输入单位、公式、依据、人工覆盖和来源均保存在results.json的configurations元数据中。

| 参数 | 快档 | 参考档 | 慢档 |
| --- | ---: | ---: | ---: |
| tRCDRD | 12 | 14 | 18 |
| tRAS | 28.125 | 33.125 | 40 |
| tRP | 12.1875 | 15 | 18.125 |
| tRC | 40.3125 | 48.125 | 58.125 |
| tWR | 12 | 15 | 20 |
| tRTP | 2.5 | 3.75 | 5 |
| tRRD_S | 2.5 | 4 | 6 |
| tRRD_L | 4 | 6 | 8 |
| tFAW | 12 | 20 | 32 |
| tWTR_S | 2.5 | 3.75 | 5 |
| tWTR_L | 3.75 | 5 | 7.5 |
| tRTW | 16.25 | 18.125 | 23.125 |
| RL | 17.5 | 20 | 25 |
| WL | 3.75 | 5 | 6.25 |
| tRFCab | 350 | 500 | 650 |
| tREFI | 3900 | 3900 | 3900 |

JEDEC JESD238提供规则、单位和部分编码范围；主组织32Gb×16H的tRFC仍为TBD，多项AC最小值没有给出。参考档不是JEDEC保证值，快/慢档也不是厂商上下限。公开Micron、Samsung、SK hynix产品资料未提供与此组织对应的完整AC表，因此vendor_calibrated=false。
时序预设为hbm3e_timing_fast、hbm3e_timing_reference、hbm3e_timing_slow；研究入口另外保持1600GB/s方向预算。默认控制器仍为bank_interleaved、fcfs、device队列深度64；候选策略不因局部探针加速就替换默认。

## 完整服务比较

先执行真实 Prefill，再从提交事件计算独立 Decode；A 为全 HBM，B 为 KV 使用 HBF。

| 模型/输入/批量/容量 | 参数或策略 | A TPOT µs | B TPOT µs | B/A | 更快 |
| --- | --- | ---: | ---: | ---: | --- |
| qwen2_5-0_5b/128/1/512 GiB | fast | 1216.8000 | 3197.3475 | 2.6277 | A |
| qwen2_5-0_5b/128/1/512 GiB | reference | 1286.5031 | 3260.9615 | 2.5347 | A |
| qwen2_5-0_5b/128/1/512 GiB | slow | 1380.6000 | 3350.1000 | 2.4266 | A |
| small/1024/4/1024 GiB | fast | 26.9365 | 124.8826 | 4.6362 | A |
| small/1024/4/1024 GiB | reference | 27.2350 | 124.8832 | 4.5854 | A |
| small/1024/4/1024 GiB | slow | 27.4600 | 125.0933 | 4.5555 | A |
| small/1024/4/512 GiB | channel_depth1 | 27.4566 | 125.6361 | 4.5758 | A |
| small/1024/4/512 GiB | channel_depth8 | 27.4497 | 125.0843 | 4.5568 | A |
| small/1024/4/512 GiB | device_depth128 | 27.4497 | 125.0843 | 4.5568 | A |
| small/1024/4/512 GiB | device_depth8 | 27.4497 | 125.0843 | 4.5568 | A |
| small/1024/4/512 GiB | fast | 26.8514 | 124.8833 | 4.6509 | A |
| small/1024/4/512 GiB | read_write_batch | 27.4475 | 125.0815 | 4.5571 | A |
| small/1024/4/512 GiB | reference | 27.4497 | 125.0843 | 4.5568 | A |
| small/1024/4/512 GiB | row_hit_first | 27.2897 | 125.0843 | 4.5836 | A |
| small/1024/4/512 GiB | row_local | 159.8558 | 245.4595 | 1.5355 | A |
| small/1024/4/512 GiB | slow | 27.6095 | 125.7648 | 4.5551 | A |
| small/1024/4/512 GiB | unconfigured | 26.5016 | 124.6744 | 4.7044 | A |

## 独立访存

延迟为各请求从到达到完成的时间，包含排队；GB/s 为全部逻辑字节除以整个完成窗口。

| 负载 | 参数或策略 | GB/s | P99 ns | ACT | REF | 重排请求 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| activation_pressure | channel_depth1 | 0.727 | 11111.250 | 256 | 4 | 0 |
| hot_row_conflicts | channel_depth1 | 0.559 | 14506.250 | 256 | 6 | 0 |
| mixed_rw | channel_depth1 | 22.880 | 2819.375 | 2016 | 0 | 0 |
| mixed_same_pc | channel_depth1 | 1.317 | 6140.000 | 127 | 2 | 0 |
| random_read | channel_depth1 | 20.809 | 3119.375 | 2016 | 0 | 0 |
| row_locality | channel_depth1 | 11.203 | 2903.125 | 64 | 0 | 0 |
| sequential_read | channel_depth1 | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | channel_depth1 | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | channel_depth8 | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | channel_depth8 | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | channel_depth8 | 275.217 | 203.750 | 2016 | 0 | 0 |
| mixed_same_pc | channel_depth8 | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | channel_depth8 | 267.494 | 190.000 | 2016 | 0 | 0 |
| row_locality | channel_depth8 | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | channel_depth8 | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | channel_depth8 | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | device_depth128 | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | device_depth128 | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | device_depth128 | 175.936 | 321.875 | 2016 | 0 | 0 |
| mixed_same_pc | device_depth128 | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | device_depth128 | 163.840 | 345.625 | 2016 | 0 | 0 |
| row_locality | device_depth128 | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | device_depth128 | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | device_depth128 | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | device_depth8 | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | device_depth8 | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | device_depth8 | 13.613 | 4766.875 | 2022 | 128 | 0 |
| mixed_same_pc | device_depth8 | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | device_depth8 | 12.043 | 5372.500 | 2026 | 198 | 0 |
| row_locality | device_depth8 | 21.408 | 1525.000 | 64 | 0 | 0 |
| sequential_read | device_depth8 | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | device_depth8 | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | fast | 2.926 | 2760.000 | 256 | 0 | 0 |
| hot_row_conflicts | fast | 0.728 | 11126.875 | 256 | 4 | 0 |
| mixed_rw | fast | 123.945 | 504.375 | 2016 | 0 | 0 |
| mixed_same_pc | fast | 2.617 | 3103.125 | 64 | 0 | 0 |
| random_read | fast | 111.078 | 568.750 | 2016 | 0 | 0 |
| row_locality | fast | 43.837 | 741.875 | 64 | 0 | 0 |
| sequential_read | fast | 1320.131 | 12708.750 | 65536 | 768 | 0 |
| sequential_write | fast | 1317.604 | 12733.125 | 65536 | 768 | 0 |
| activation_pressure | no_activation_gates | 2.532 | 3186.250 | 256 | 0 | 0 |
| hot_row_conflicts | no_activation_gates | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | no_activation_gates | 108.212 | 573.750 | 2016 | 0 | 0 |
| mixed_same_pc | no_activation_gates | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | no_activation_gates | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | no_activation_gates | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | no_activation_gates | 1272.507 | 13184.375 | 65536 | 768 | 0 |
| sequential_write | no_activation_gates | 1269.438 | 13216.250 | 65536 | 768 | 0 |
| activation_pressure | no_refresh | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | no_refresh | 0.657 | 12321.250 | 256 | 0 | 0 |
| mixed_rw | no_refresh | 108.212 | 573.750 | 2016 | 0 | 0 |
| mixed_same_pc | no_refresh | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | no_refresh | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | no_refresh | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | no_refresh | 1450.924 | 11563.125 | 16384 | 0 | 0 |
| sequential_write | no_refresh | 1452.809 | 11548.125 | 16384 | 0 | 0 |
| activation_pressure | no_rw_recovery | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | no_rw_recovery | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | no_rw_recovery | 108.548 | 569.375 | 2016 | 0 | 0 |
| mixed_same_pc | no_rw_recovery | 2.830 | 2871.250 | 64 | 0 | 0 |
| random_read | no_rw_recovery | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | no_rw_recovery | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | no_rw_recovery | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | no_rw_recovery | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | read_write_batch | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | read_write_batch | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | read_write_batch | 108.212 | 578.125 | 2016 | 0 | 299 |
| mixed_same_pc | read_write_batch | 5.162 | 1570.000 | 64 | 0 | 120 |
| random_read | read_write_batch | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | read_write_batch | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | read_write_batch | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | read_write_batch | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | reference | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | reference | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | reference | 108.212 | 573.750 | 2016 | 0 | 0 |
| mixed_same_pc | reference | 2.219 | 3658.750 | 64 | 0 | 0 |
| random_read | reference | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | reference | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | reference | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | reference | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | row_hit_first | 2.326 | 3473.125 | 256 | 0 | 0 |
| hot_row_conflicts | row_hit_first | 4.752 | 1706.875 | 10 | 0 | 123 |
| mixed_rw | row_hit_first | 108.212 | 575.000 | 2016 | 0 | 16 |
| mixed_same_pc | row_hit_first | 2.229 | 3641.875 | 64 | 0 | 16 |
| random_read | row_hit_first | 97.542 | 643.125 | 2016 | 0 | 16 |
| row_locality | row_hit_first | 43.582 | 746.250 | 64 | 0 | 35 |
| sequential_read | row_hit_first | 1268.958 | 13221.250 | 65536 | 768 | 0 |
| sequential_write | row_hit_first | 1265.071 | 13261.875 | 65536 | 768 | 0 |
| activation_pressure | row_local | 8.965 | 883.750 | 256 | 0 | 0 |
| hot_row_conflicts | row_local | 0.585 | 13860.000 | 256 | 6 | 0 |
| mixed_rw | row_local | 102.101 | 630.625 | 1014 | 0 | 0 |
| mixed_same_pc | row_local | 13.416 | 579.375 | 256 | 0 | 0 |
| random_read | row_local | 91.819 | 703.750 | 1014 | 0 | 0 |
| row_locality | row_local | 2.007 | 16156.875 | 287 | 8 | 0 |
| sequential_read | row_local | 639.071 | 26252.500 | 17152 | 1536 | 0 |
| sequential_write | row_local | 637.326 | 26324.375 | 17152 | 1536 | 0 |
| activation_pressure | slow | 1.720 | 4737.500 | 256 | 2 | 0 |
| hot_row_conflicts | slow | 0.464 | 17488.125 | 256 | 8 | 0 |
| mixed_rw | slow | 88.042 | 705.000 | 2016 | 0 | 0 |
| mixed_same_pc | slow | 1.514 | 5367.500 | 109 | 2 | 0 |
| random_read | slow | 79.438 | 792.500 | 2016 | 0 | 0 |
| row_locality | slow | 43.080 | 755.000 | 64 | 0 | 0 |
| sequential_read | slow | 1173.437 | 14297.500 | 65536 | 768 | 0 |
| sequential_write | slow | 1169.501 | 14345.625 | 65536 | 768 | 0 |
| activation_pressure | unconfigured | 2.532 | 3186.250 | 256 | 0 | 0 |
| hot_row_conflicts | unconfigured | 0.657 | 12321.250 | 256 | 0 | 0 |
| mixed_rw | unconfigured | 108.548 | 569.375 | 2016 | 0 | 0 |
| mixed_same_pc | unconfigured | 2.830 | 2871.250 | 64 | 0 | 0 |
| random_read | unconfigured | 97.633 | 643.125 | 2016 | 0 | 0 |
| row_locality | unconfigured | 43.582 | 746.250 | 64 | 0 | 0 |
| sequential_read | unconfigured | 1452.337 | 11551.875 | 16384 | 0 | 0 |
| sequential_write | unconfigured | 1454.225 | 11536.875 | 16384 | 0 | 0 |

## 解释边界

- 不同队列范围的预算不同：device 是全设备请求数；channel 是每物理通道请求数。512 GiB配置有128个物理通道，因此 channel_depth1 的总名额上限为128，须与 device_depth128 一起比较。跨通道请求同时占用所有涉及通道的名额。
- 调度作用于已有的聚合请求和有限窗口，不是完整逐命令 FR-FCFS 控制器；不跨尚未生成的算子请求重排。
- 快/慢档同时改变多个未知时序；单项消融用于分离刷新、激活和恢复约束的作用。
- HBF仍为既有逻辑页服务模型；本次未增加REFpb、RFM、低功耗、初始化或错误注入。
- 方案排序仅适用于实际列出的模型、上下文、批量和参数范围；不替代完整20组研究矩阵。
