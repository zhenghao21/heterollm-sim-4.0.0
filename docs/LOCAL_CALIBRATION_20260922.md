# 本机校准证据与下一步（2026-09-22）

本轮是证据审查与最小补测，不是已校准profile发布。固定R0、原native和验收阈值均未改变。

## 已有证据

- `kernel_microbench_b05_v1.json`：24/24成功，但实际为llama-bench整段prompt/decode，文件本身明确blocked；仅可诊断端到端趋势，不可用于单kernel成本拟合。
- `generic_gemm_microbench_v1/smoke/summary.json`：12个CPU/CUDA、6格式正确性/时延入口成功摘要；所引逐项原始文件在当前目录缺失。README仍说未计时，与summary不一致，以可读summary为已记录事实，但不能独立审核原始样本。
- `host_runtime_microbench_v1.json`：进程/tokenizer/JSON/SSE开销；不能并入engine GEMM。
- Qwen38 CPU semantic profile：operator_wall、模型/形状范围有限、train_identity为空；不自动复用为通用校准。

## 本轮有限补测

事先记录protocol.json：8个独立GGML GEMM，CPU/CUDA × IQ3_S/IQ4_XS × M=1/8，N=K=256；warmup3，正式20次，reference抽样32，atol=.05/rtol=.03固定，90秒/格，不重试；M8预留holdout，但没有已拟合模型，不宣称holdout已通过。无LLM推理、无修改功率/频率设置。计时含dispatch、转换及内部同步，不能重复添加launch成本。

结果8/8正确性通过、模块前后稳定；当前EXE/4个GGML DLL哈希前后相同。正式样本CV约4%—60%，CPU亦有13%—22%波动。缓存热态小矩阵不代表大工作集主存/GDDR带宽。GPU桌面活动存在，未锁频/线程亲和性；原EXE构建源码/manifest尚缺。故结果EVIDENCE_ONLY，profile_enabled=false，不将噪声拟合为参数。

原始样本、逐次timing、命令、协议、硬件前后快照和汇总位于 `artifacts/local_calibration_20260922/`。本机GPU是RTX5080 GDDR7，不能标注为实测HBM，更不能作为CIM器件或HBF微基准。

## 最小后续顺序

1. 恢复可审核的基准source/build/loaded DLL链；固定实际运行库与本机driver。
2. 针对上述通用形状独立会话重复，记录亲和性、时钟、热状态和背景活动；不通过改误差阈值掩盖不稳定。
3. 独立大工作集主存/GDDR顺序带宽与小随机访问延迟，区分缓存、读写方向与传输粒度。
4. PCIe pinned/pageable双向传输大小矩阵，测持续服务与固定开销，不混成同一个参数。
5. 单独解码路径测量；PC CPU/GPU解码率只对应实际执行设备，不能自动校准CIM内置decoder。
6. 冻结通用成本模型后，再用未参与标定的Qwen负载验收TTFT/TPOT/E2E；不使用目标答案拟合。
