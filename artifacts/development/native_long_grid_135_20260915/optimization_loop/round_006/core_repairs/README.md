四项 P2 已在主树局部修复，未改动在途冻结文件。

- R6-CR-01：重复 apply 保留 generated_gpu_aliases 来源，CPU legacy 标志不能继承这些别名。
- R6-CR-02：packed QKV flash-off 只有在新增源码布局证据成立时才建模 V 连续化；M>1 使用有跨度的 F32 复制内核，M1 的连续数据复制不额外计 CUDA kernel launch，复制提交服务仍明确未定价。旧合同缺少布局证据时缓存子合同 uncovered。
- R6-CR-03：逐投影核对源张量、所属组、调用形状后标记 applied；失败组保留原因，融合 gate/up 保留两个物理权重矩阵的统计。
- R6-CR-04：MMQ 转换/修正以真实 task ID、request 和依赖链辨认调用，覆盖跨请求同名、同请求重复同名、无 launch 设备阶段和 launch-only fixup。

验证：GPU/KV/MMQ 38项、CPU/物理投影/offload 41项，共79项不重叠测试通过。12个锚点、physical_mapping 与 physical_mapping_mmq 两分支，各取 prefill_start、prefill_final、decode，共72个静态图；修复前后所有非metadata TaskSpec字段精确一致，MMQ计数一致。40图仅GPU覆盖标签口径变化。315个冻结文件SHA未变。没有完整事件模拟、原生调用、原生测量或模型payload读取。

派生合同新增 cpy.cu 与 packed-view 证据，因此今后新一轮 wrapper freeze 需要重新 derive 保存合同；本次旧冻结合同未改，旧合同仍可用于已验证的非packed路径。测试与静态图不是整轮延迟评估，不将其宣传为完整性能验证。

文件：validation.json 保存逐图比较；before_static_anchors.json/after_static_anchors.json 保存摘要；local_repairs.patch 是相对此次修复前快照的局部差异。父任务负责合并进后续轮次和提交推送。
