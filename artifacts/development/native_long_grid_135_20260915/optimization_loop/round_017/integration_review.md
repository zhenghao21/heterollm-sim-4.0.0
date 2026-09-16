# R17 转换CTA成本集成审查

2026-09-16。只读当前core和三路新冻结，未改源码/冻结文件，未运行GPU或simulator，不读取native时延或评分。Root已完成123+43+4项测试，本审查不重复全套。

**结论：当前限定Q5_0/Q8_0、固定CC1200构建与普通转换路径的R17候选，没有发现需要停止20锚点的P0/P1。可以继续当前冻结的开发/回归对照，但不能把metadata显示applied或三路准备成功当成误差改善/目标达成。**

## 1. 三路冻结输入一致性

检查`pure/current/conversion_cta/freeze.json`：

| 条件 | pure | current | conversion_cta |
|---|---|---|---|
| selection denominator | 131 | 131 | 131 |
| selection SHA | 同一cab8f3a4…df9c5 | 相同 | 相同 |
| MMQ source flag | false | true | true |
| conversion CTA flag | false | false | true |
| per-cell conversion flag | 全false | 全false | 全true |
| preparation_error | 0 | 0 | 0 |

三路131个cell id顺序与model_snapshot_map相同。按snapshot根的相对路径比较，**110份冻结源码文件集合和SHA全相同**。source总SHA因不同冻结目录产生不同值，不应误判为代码不同；逐相对文件已证明一致。

抽查冻结副本中planner、cost_models、llama_gpu_invocations、conversion_work、predict_stable_native_dataset五个关键文件，其实际内容SHA与清单一致。此次没有遍历GGUF或重新核验全部原始native文件；继续运行仍由冻结执行器完成全量前后门。

三路均声明`blind_evaluation=false`、`evaluation_type=development_post_selection`，没有将已揭示固定集伪装成盲测。目录名pure应解释为“本次MMQ/CTA成本处理关闭的分析对照”，不代表删除已有执行语义、F32存储与host/source机制。

## 2. 编译架构1200是否只是从设备CC猜测

新apply代码构造`highest_compiled_arch=cc`表面上缺直接证据，但上游`derive_llama_gpu_invocation_contract`在生成payload之前已验证：

- runtime build binding schema为`llama-recorded-runtime-source-binding/v1`、status=`verified_build_chain`，content SHA正确；
- `cuda_compute_capability//10`必须属于`compiled_cuda_architectures`；
- 当前三路引用的实际build_binding记录为 **compiled_cuda_architectures=[120]**；
- `source_compilation.cuda.module`绑定到当前`ggml-cuda.dll` SHA `8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297`；源码注册/编译链接收据仍保留。

因此对**当前只有120的窄构建**，CC1200对应compiled1200是有上游证据的，不是仅凭设备代填。mmq_device_evidence也记录warp32、SM84、shared optin101376，与硬件profile的SM数核对。

局限：这些是历史构建/链接收据，未新增cubin反汇编或完整源码/二进制等价证明。现有conditional标记必须保留。未来若支持多compiled arch或其他GPU，建议payload显式携带`selected_compiled_arch`与证据来源，apply直接消费它，而不是长期保留`highest_compiled_arch=cc`这一隐含推导；当前不构成阻断。

## 3. 来源绑定与窄格式域

`apply_llama_gpu_invocation_contract`新增路径：

- conversion flag要求MMQ/MMVQ source costs已请求且设备契约实际applied；
- conversion_work中的每个固定源SHA都必须在payload.source_refs唯一匹配；缺失/错SHA直接拒绝；
- runtime binary SHA来自相同payload的ggml-cuda.dll，不读取待预测场景原生时延；
- ConversionSourceContract仅接收CC1200/compiled1200/warp32、普通F32二维、单通道/样本、无ids/scatter/fp4。

planner只在既有conversion实际发出的位置调用helper。物理格式必须恰好一个；Q5_0/Q8_0的MMVQ、MMQ D4进入支持域。其他格式/path导出applied=false和原因，conversion本身保持分析路径，不扩张既定支持范围。DS4源描述存在，但当前MMQ工作支持的K类量化不能被它错误套用，格式不匹配会回退。

helper还严格比较源派生`partial_scalar_operations/read_bytes/write_bytes`与既有lowered workload。如果不一致，会失败而非把原工作量静默替换。唯一返回的工作量变化是`source_grid_ctas=work.cta_count`。

## 4. 成本数学与重复计费

cost_models已采用上轮建议的峰值cap：

`scalar_gops=min(既有有效scalar吞吐, min(grid_CTA,SM)×单SM scalar峰值)`

SFU同理。旧occupancy/efficiency不会再次乘到active-SM上限；metadata明示`occupancy_efficiency_applied_twice=false`。当旧有效吞吐本来更低时新cap不改变结果，当grid饱和时也不追加波次数惩罚。

没有改变HBM带宽、read/write字节、cache需求、energy、launch phase和dependency_depth。conversion仍然一组明确kernel阶段；MMVQ主计算随后消费Q8_1临时输入字节、MMQ消费consumer_unique_bytes，未同时计入原F32和临时输入。main/fixup的原计价没有被conversion cap重复覆盖。

这仍是普通CTA执行资源的**必要峰值上界＋旧分析机制**，不是实测占用率。warp/寄存器/依赖与整数/转换指令服务尚未因此完整建模；HBM/SSM/CPU等不在此次机制覆盖内。

## 5. 冻结开关及执行边界

`freeze_selection`把conversion_cta_costs_requested写入全局GPU证据和每cell证据。load/apply入口检查inputs flag为真实bool且与cell proof完全一致；不得在读取目标结果后改输入flag。强制`conversion flag -> MMQ/source flag`，避免仅设置一个孤立字段绕过物理派发前提。

source snapshot包含新增conversion_work与成本代码，三路都使用同一源码；不是在冻结后修改current并称旧版本。Root应继续确保20anchor使用预先固定相同id顺序，错误/超时继续保留，不从131分母删去困难格。

## 6. 非阻断事项与后续证据

1. `tests/test_conversion_planner.py`完整planner用例创建了`changed`计数但不assert>0；它证明“若改变则只发生在声明转换”，不能证明所选fixture实际上发生时延变化。是否有收益应看已冻结预测数值，不能用applied metadata替代。后续可以增加明确scalar-bound小grid fixture，但无需为此改动当前冻结。
2. `highest_compiled_arch`建议将上游[120]推导显式写入下一版本审计metadata，降低未来误用风险。
3. 缺layout/shape/source证据路径的回退统计应汇总到每锚点报告；sourcecap有绑定不等于所有token/投影都覆盖。
4. 当前机制只可能增加小grid conversion服务，而MMVQ main历史诊断存在高估。20格结果可能改善、持平或退化，都要按角色因果解释；不得先验宣称有收益。
5. 没有经过质量门的独立kernel profile被接入，这一点正确。R16失败样本仍只用于诊断，不能被这次分析cap“洗成”合格标定。

## 建议完成本轮的交付

保持三个已冻结版本不动，结束共同20锚点后保存每格Engine TTFT/TPOT/E2E的有符号/绝对相对/毫秒误差，以及source conversion覆盖、真实数值变化量、失败原因和固定native身份。若结构正确但目标误差未改善，应记录消融失败并继续下一机制；不能只因完成20次预测或metadata完整而宣布达到三项<10%。
