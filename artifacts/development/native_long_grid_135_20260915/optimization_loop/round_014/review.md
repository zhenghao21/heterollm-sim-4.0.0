# R14 MMQ 尾块独立审计

日期：2026-09-16。范围限于 MMQ 尾块派生、成本接入、planner 门禁和本轮冻结；未修改生产代码，未运行 GPU/native 或大规模评估。以下行号对应审计时工作树。

## 结论

没有发现 K896 被误当成逻辑 K1024、stream-K 分区错误、普通 MMVQ 被改道或新增重复 launch 的阻断问题。可以继续条件性开发评估；但发现一处尾部权重读字节漏计，以及一处需要明确保留的分配证据边界。不能据此宣称完整数据流或准确性已验证。

## 可执行问题

1. **[P2] 已派生的最后一行尾读没有进入主 kernel 内存需求。**
   - 位置：`src/heterollm_sim/cost_models.py:2604–2607`；对应派生量在 `src/heterollm_sim/mmq_work.py:165–178`。
   - 主内存读取仍为 `activation_bytes + weight_bytes`。保留逻辑权重存储字节是正确的，但实际 source 读取会越过最后一行的逻辑末端；已派生的 `weight_read_high_water_bytes - logical_weight_bytes` 没有单独进入资源需求。不是要求把每行 K 补齐，也不是把重复 tile 读取全部提升到 DRAM。
   - 小型纯 CPU 构造：M64/K896/N896，Q5_0 的逻辑权重551936B、读取高水位552024B，相差88B；Q8_0分别852992B/853128B，相差136B。当前成本只计前者。
   - 建议：逻辑 `weight_bytes` 保持不变，主内存读需求和 working set 单独加入经过分配门禁的最终唯一尾读字节，并附带明确 metadata。补一个资源需求差分测试，断言只增加一次88B/136B，不能增加 N 倍尾读，也不增加 launch。
   - 影响很小，不能把它当成当前大误差的根因；它属于本轮新增尾块覆盖的语义完整性问题。

2. **[P2，正式支持声明前需补齐] 源码中的分配规则已冻结，但每次调用的尾部可读范围尚未被证明。**
   - 位置：`src/heterollm_sim/planner.py:9172–9182`，`src/heterollm_sim/llama_gpu_invocations.py:434–437`；`mmq_work.py:212`仍明确输出`weight_tail_allocation_proven=false`。
   - 当前尾K的额外门禁只验证合同版本字符串，随后将工作标记`applied`。新合同正确固定了 CUDA allocator 的 padding 实现，但该实现是“采用该 buffer 类型时”的规则；每个 physical projection 的实际 buffer/非view/可用末尾容量证据并未在此检查。因此不能把`applied`解释为实际分配也已验证。
   - 当前整条预测仍标为`conditional`，所以这不阻断条件性开发评估。建议把 allocator 契约和实际调用适用前提绑定到 audit：或者由已冻结放置/张量来源证明标准 CUDA weight buffer、逻辑 stride 与所需 padding，或者显式保留分配未证明原因并拒绝无条件支持声明。直接根据 K 形状把 false 改 true 不成立。

## 已核对通过的部分

- 源码`mmq.cuh:1067–1075`以 B=K/qk 分区，并将边界按当前输出tile内的 R=256/qk 向下对齐；新`mmq_work.py:335–350`保留逻辑 B，未使用补齐后的 K 参与 stream-K 分区。
- 源码`mmq.cuh:908–938`执行完整256值迭代并加载两段128值 activation；成本仅将执行 K 用于发射运算量，有效工作量、权重逻辑存储和原矩阵形状保持K896。
- 转换、主矩阵、fixup在planner中按依赖顺序分开。主阶段只写一次部分结果，fixup另读部分结果并读写有效输出，未发现包含关系重复收费。
- `planner.py:9169–9173`先处理MMVQ优先级，再处理MMQ尾K门禁；普通M1/2/4向量派发不会因尾块扩展而强行进入MMQ。
- 固定launch=1000ns的独立小构造只有一个主矩阵launch；有效操作102760448，发射操作117440512，与逻辑K896/执行K1024一致。GPU测试辅助函数默认launch=0会省略launch phase，不应把这当成重复/漏启动故障。
- 本轮`tail_candidate/freeze.json`中的GPU合同与`tail_source_contract.json`完全一致，包含`quantize.cu`、`mmq.cuh`、`mmq-load-tiles.cuh`新来源引用。wrapper在`predict_stable_native_dataset.py:774–778`重新派生后做完整合同比较，没有静默复用R6旧合同。
- 冻结仍记录131格conditional；现有未验证的MMA输出tile-wave成本和未实测缓存行为没有被包装为已验证物理吞吐。

本审计只做源码核对和两个极小合成算子构造；没有复跑父代理已完成的51项测试。若修改上述问题，应创建新冻结，保留当前冻结与预测，不能覆盖旧证据。
