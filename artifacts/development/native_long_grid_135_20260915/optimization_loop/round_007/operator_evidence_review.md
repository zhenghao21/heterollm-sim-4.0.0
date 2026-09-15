# R7 算子证据独立审计

审计对象：operator_microbench_v2/stream_event_probe 的源码、冻结协议、身份门、计时与质量脚本。只读审计；没有运行 GPU、native LLM 或 simulator，没有改成本模型/冻结文件/任务书。

## 结论

当前探针可以执行为诊断试点，但任何结果仍须保持 device-event envelope（设备事件包络）口径，不能直接提供每 kernel launch、纯 kernel 时间、HBM 带宽或全 LLM 成本系数。完成本批诊断后，才有证据决定下一版测量结构。没有发现其事件记录到错误 CUDA stream 的问题。

## 源码确认

- C++ compute() 按 begin_event -> graph_compute_async -> end_event -> cudaEventSynchronize 排序；begin/end 使用锁定内部 event ABI，把事件交给 ggml_backend_event_record。
- source/llama.cpp-annotation-control/ggml/src/ggml-cuda/ggml-cuda.cu:4529–4532 的 event_record 委托 cuda_ctx->stream()；同文件2584–2587的 backend synchronize 等待相同stream。
- 单连续二维MUL_MAT、一个CUDA backend、graphs OFF限制和逐次同步足以避免本试点未知的多流分支。但事件区间包括主机提交间隔；不能称作纯GPU算术/访存服务时间。
- 全部first/warmup/formal都在计时后validate()，状态错误非零退出，raw不覆盖。fixed synthetic seed、解量化权重double参考是合法独立算子证据，不使用LLM答案。
- 同缓冲区重复热缓存，矩阵只有512×512或768×1024；无法覆盖实际LLM矩阵、冷数据/HBM路径、K896非256对齐、Q5_0/Q8_0、M2、backend split、多并发。

## 需要限制或修复的缺口

1. event/control的差值同时改变了计时事件及等待API：event用cudaEventSynchronize，control用cudaStreamSynchronize。现有20%门可视为整个测量方式的扰动筛查，不能解释为纯event instrumentation overhead。不要从两者差值推launch或同步常数。下一版可增加无计时event但保持同一event等待的control以隔离此混淆，仍保留原版结果。
2. p90/p10<=1.50远松于目标native±5%稳定性。本门是预先规定的诊断准入门，不能将accepted写成已满足最终准确度/精度验收；不应事后放宽或收紧原门来选择数据。
3. build_manifest锁定assess.py/协议/probe/invoke，但没有锁定run_frozen_matrix.ps1和full_raw_audit.py。这两个脚本决定调度与补充完整性检查。运行前单独冻结二者SHA，运行后再次验证；保持原build_manifest不可变。它们不进入旧binary签名也不能写成已受旧冻结保护。
4. assess.py只核formal；full_raw_audit补first/warmup、QPC顺序、配置及所有状态，二者必须联合通过。full_raw_audit自身不核所有设备/DLL身份，因此仍必须连同invoke门、loaded_modules原始记录及实际设备快照读取。
5. assess.py对pair属性使用get(key)==get(key)，两边同时缺字段会通过；full_raw_audit不覆盖这些设备属性。当前C++正常输出这些字段，构建门降低风险，但若将该extractor复用为通用证据导入器，应先强制必需身份字段存在且非空，不能仅成对相等。
6. runner只检查predict_stable_native_dataset.py进程；IdleWindowConfirmed是操作者声明，不证明无hash诊断、其他GPU进程、服务负载。应等hash结束并记录实际进程/利用率后测，不抢占，不停止无关进程。before/after设备快照只能发现端点状态，无法证明每个样本内时钟稳定。
7. runner末尾nvidia-smi未检查退出码；execution只记invoke异常，assessment失败保留日志但不结构化聚合。必须人工/新增只读汇总核验每个raw/assessment存在、退出码、完整24配置模式记录。缺失证据不得计为成功。

## 建议采用次序

A. 先以当前冻结协议完成诊断，提前补独立runner/audit SHA回执。待hash任务结束后安排空闲窗口。
B. 汇总应并列raw结构完整性、数值正确、身份、波动、测量方式扰动，每个失败保留，不产生系数。
C. 下一版合成协议补Q5_0/Q8_0 K896与合法对齐控制、M1/2/4/64以及实际N代表范围；先查源码实际路径和数值，不以模型名决定参数。不要把本版512/1024 K的通过称作覆盖新联合shape域。
D. 若事件包络仍主要受主机供给/等待影响，使用批量链与同等待方式control区分调度，避免再次尝试从单次总包络反推一个全局launch常数。

本审计不把SHA一致等同准确度，不对当前尚未测量的probe宣称通过。
