# B-07 shape-gradient semantic trace 开发证据（2026-09-14）

## 范围

本轮使用现有 semantic direct `llama-server.exe`（B-06 binary）和 Qwen2.5-0.5B Q4_K_M，在同一 RTX 5080/CUDA/runtime 下补采不重叠的 prompt/output shape：`M=16,T=4`、`M=32,T=8`、`M=64,T=16`。所有请求均为单并发，`ctx=512,batch=64,ubatch=64,threads=16,gpu_layers=-1,KV=f16,FA=off,CUDA graphs=OFF`，warmup=1。未使用任何目标模型 TTFT/TPOT/E2E 拟合成本参数，未修改正式 profile。

## 证据产物

- `artifacts/development/b07_qwen25_m16_t4_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b07_qwen25_m32_t8_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b07_qwen25_m64_t16_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b07_qwen25_train_merged_v2.trace.json`
- `artifacts/development/b07_qwen25_holdout_v2.trace.json`
- `artifacts/development/b07_qwen25_operator_coverage_v2.json`
- `artifacts/development/b07_qwen25_operator_calibration_candidate_v2.json`
- `artifacts/development/b07_qwen25_shape_gradient_manifest_v1.json`（binary/model/nsys 与每个 sidecar 的 SHA-256）

三条新 trace 的 extractor 事件数分别为 4,920、9,424、17,856；semantic matched kernel 分别为 2,320、4,384、8,272。三条 profile 均为 `profiled_stage_evidence`，且原始 nsys-rep、SQLite、kernel/API/memcpy CSV 均已保留。

## 联合覆盖与留出

把 B-06 `M=8,T=8` train 与 B-07 `M=16,T=4`、`M=64,T=16` 合并为 train，把 B-07 `M=32,T=8` 作为 holdout：

- train matched kernels：14,714；holdout matched kernels：4,384；
- owner_unknown=0、stage_unknown=0、missing token shape=0；
- train 联合键 91，holdout 联合键 47，精确 `stage×phase×shape×dtype×layout×kernel_family` 交集 21/47（44.68% holdout key coverage）；26 个 holdout 联合键属于未见 M=32 shape，保持域外回退；
- 忽略 shape 后的 owner/stage/type 覆盖仍闭合，但这不等于可进行 shape 插值；layout 字段仍未由 trace 显式提供。

候选 `build_semantic_calibration` 结果明确为 `status=blocked`：虽然 owner_unknown/stage_unknown 为 0，但合并 trace 没有可归属 phase scope，`train_missing_phase_count=14714`、`holdout_missing_phase_count=4384`，所有 wall calibration 均 blocked。候选结果仅用于开发诊断，未写入正式 simulator profile。

## 启用判定

B-07 成功补齐 M=16/M=64 的物理事件和 M=32 的独立留出证据，但 exact shape 联合键交集仍不足，且 phase scope/layout 证据缺失。不能把三条 trace 的平均 rate 或 aggregate benchmark 下沉为通用算子成本；未见 shape 必须显式标记回退/不支持。下一步需要在同一 binary 下恢复 phase marker 与显式 layout，或补充可独立验证的 kernel-level shape 性能模型。

## 合并修正

初版 v1 合并文件只拼接 `events`，未保留 NVTX phase 列表，导致候选校准误报 `missing_phase`。v2 合并文件保留 `events`、`nvtx_events`、request markers 和 graph metadata，并重新生成 coverage/calibration；v1 保留为过程证据，不用于结论。
