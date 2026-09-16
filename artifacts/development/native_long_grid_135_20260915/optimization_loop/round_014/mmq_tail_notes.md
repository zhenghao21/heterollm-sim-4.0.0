# R14 MMQ 尾块修正

仅修改 `src/heterollm_sim/mmq_work.py` 与 `tests/test_mmq_work.py`。没有改成本模型、planner、native、冻结清单、任务书或状态。

## 接口供父代理集成

- `k`：实际逻辑 K；`k_padded`：转换缓冲按512补齐；`k_execution`：矩阵 K 迭代按256补齐。
- `execution_arithmetic_operations=2*M*N*k_execution`：用于保持当前输出维度成本映射、仅补强制 K 尾迭代。
- `source_nominal_arithmetic_operations=2*U*I*J*k_execution`：完整源模板的名义算术量；包含 I/J 输出尾块，若成本端已有输出 tile-wave 损耗，不能重复乘这部分。没有新增拟合系数。
- `source_nonempty_block_count`、`stream_k_boundaries_qblocks`：真实量化块分区；K896/U14/G84为42非空、28部分写回，K1024为56/42。
- `consumer_unique_bytes`：两次128元素源读取窗口的联合高水位；`source_repeated_bytes`：逐tile、逐迭代源读取，不自动提升为DRAM流量。
- `logical_weight_bytes` 由源量化块大小推得；`weight_tail_read_bytes_per_row` 是每行尾读宽度；`weight_read_high_water_bytes=logical_weight_bytes+weight_tail_read_bytes_per_row`，因为中间行尾读与下一行数据重合。不能按 N 倍尾读创建虚假的逐行权重分配。
- `weight_tail_range_bytes` 是最后一行越过逻辑张量末端的半开区间；非对齐K的 `weight_tail_allocation_proven=false`。只有源形状不会证明实际张量分配充足，调用侧仍需独立验证。

## 源证据

锁定源码提交：`0f3a71be15af836d277c9f918adfafb45732677e`。

- `ggml-cuda/mmq.cuh:898–935`：256步长与两次128元素 activation 装载。
- `ggml-cuda/mmq.cuh:1063–1075` 与 `1251–1266`：真实B=K/qk，raw=floor(b*U*B/G)，boundary=raw-(raw%B)%R，R=256/qk。
- `ggml-cuda/mmq.cuh:1076–1164`：完整tile与最后部分写回的划分。
- `ggml-cuda/mmq-load-tiles.cuh:313` / `471`：Q5_0、Q8_0按真实行stride装载完整尾迭代。
- `ggml-cuda/mmq.cu:120–143`：按512补齐的转换分配与J_max附加空间。
- `ggml-common.h:234–256,337–368,421–460`：7种量化块结构大小。

## 验证

12项MMQ工作量测试、6项planner测试、22项GPU调用映射测试通过（共40项）；包括七种格式黄金值、尾K、子128转换块、最终张量尾读与分区循环覆盖。对896个已对齐控制，逐一比较所有旧metadata字段，0变化。精简结果在 `mmq_tail_validation.json`；1.5MB完整改前快照仅供本地核验，不建议上传。

本轮没有GPU或目标LLM计时，不能据此宣称误差达标；成本接入与新冻结评估由父代理继续。
