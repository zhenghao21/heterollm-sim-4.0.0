# R18 采集器证据门禁修复交付

修复已完成，**39 项主机回归通过**，实际 pytest 返回码为 0。原始红色回归、审查报告和四个修改前源文件均已保留。没有 GPU 运行，没有改动冻结探针、native actual、成本模型或误差阈值；采集器尚未冻结。

## 修改范围

只修改主任务允许的四个文件：

- `collection/probe_adapter.py`：统一 stage argv 构造与核验；强制原始 PID/TID、pair_id、argv/输出路径、UTC与QPC生命周期字段；核查调用区间属于原始进程；增加重复进程证据检查，并允许不同启动区间的合法PID复用。
- `collection/extract.py`：按 direct/profile/export 强制完整的原始文件集合；拒绝缺失、重复、None SHA、外部路径与内容变化；将 spec、实际启动记录、监督器和完成记录的 argv/PID/QPC绑定；direct必须与实际子PID一致，profile目标身份独立于Nsight包装进程。trace通过官方24位PID/TID编码与原始进程绑定，同时保留非零VM/硬件高位，不把globalPid整个当作操作系统PID。后续读取必须与阶段已经校验的原始文件SHA一致。
- `collection/runner.py`：使用同一 argv 构造入口；SQLite导出在启动spec中绑定输入report的实际SHA；正常推进和恢复时都执行完成阶段的证据门禁，再开启后续阶段。错误沿用终止当前身份批次的路径，不补造数据。
- `collection/test_collection.py`：有效测试夹具补全明确的合成PID、TID、pair、argv、UTC/QPC，不减少原检查；增加完整阶段正向校验及原始证据缺失/错配变异检查。

## 验证

原审查的三个拒收反例均转为通过。新增检查覆盖：direct/profile/export完整集合；合法空stdout/stderr；缺raw/telemetry；外部同名文件；重复文件；缺SHA；错误PID、pair和输出argv；原始时间早于实际父进程启动；非零Nsight VM命名空间；错误TID；重复进程证据与合法PID复用。原有Graph一对多映射、kernel去重、缺失不回填和失败分母检查继续通过。

`host_regression_after_fix_recorded.log` 记录了完整39项结果；`repair_identity_v1.json` 保存修复前后源码、红色基线、测试与编译探针的SHA。

## 交给主任务的下一步

1. 审阅 `collector_evidence_binding.patch`，并独立运行现有与审查回归。
2. 更新采集器 `preparation_status.json` 及相应说明中的当前源码/测试状态后再冻结。这两份当前状态文件不在本代理本次获准编辑范围内，因此未改写；旧报告的SHA列表不能被当成修复后状态。
3. 编译探针manifest仍为 `15fc2cf1c449968d4cc756d4c43cafb319c2764aa49e9a69c197269eb07fb617`。无需重编探针即可创建新的采集器冻结。
4. 首个真实pair仍需核对Nsight实际PID/TID及Graph-child correlation。合成回归通过不是GPU数值/计时或LLM误差验收通过；未改变采样、频率控制与子进程生命周期的实现。
