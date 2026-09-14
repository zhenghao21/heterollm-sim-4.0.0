# B-06 kernel-level owner/correlation 开发证据（2026-09-14）

## 范围

本轮只使用 semantic direct binary + Qwen2.5-0.5B Q4_K_M，在同一 RTX 5080/CUDA/运行时配置下采集一条 8-token prompt train 和一条 58-token prompt holdout。没有读取或拟合任何目标模型 TTFT/TPOT/E2E；证据仅用于算子成本机制开发。

## 固定身份与配置

- binary SHA-256：`4bd697950897b396664917652e8f80f7b8883f736efa4427ccdb9d5ebb829337`
- GGUF SHA-256：`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- Nsight Systems SHA-256：`9e4d325628a774f358be8ff068b8d1ff284416479b5f72a44fcfe632d046ff6b`
- `ctx=512, batch=64, ubatch=64, threads=16, gpu_layers=-1, np=1, KV=f16, FA=off, CUDA graphs=OFF, output=8, warmup=1`
- train prompt tokens=8；holdout prompt tokens=58；两者均固定输出 8 token。

## owner/correlation 与联合覆盖

- train kernels：4122；holdout kernels：4384。
- 两侧 semantic_status=matched：4122/4122、4384/4384。
- owner_unknown=0；stage_unknown=0；missing phase=0；missing shape=0。
- `(stage, owner, dtype, shape, kernel)` 联合键交集/并集：462/2048（22.56%）。trace 没有显式 layout 字段；因此 shape×dtype×layout×kernel 的 layout 覆盖为未知，不能宣称已覆盖。
- 忽略 shape 后，train/holdout 的 673 个 owner/stage/type 联合键全部相交；差异来自实际 shape/kernel 路径，而不是 owner 丢失。
- memcpy 与 CUDA API 事件单独保存，未折入算子 wall rate。

## 留出结果

| stage | operator event holdout error | operator-wall holdout error | train instances | holdout instances |
|---|---:|---:|---:|---:|
| attention_qkv | -23.77277771968268% | -2.605862040726389% | 1608 | 1032 |
| ffn | -8.732850245733454% | -11.983117755770836% | 891 | 461 |
| kv | 5.458051723944392% | -2.005295006592497% | 384 | 384 |
| lm_head | -0.5518523570785928% | -1.4654957307090737% | 32 | 24 |
| attention_output | -9.849377837819457% | -13.545423511258587% | 792 | 768 |
| normalization | -0.08876373542285207% | 7.575471160963389% | 392 | 392 |
| linear_attention_aux | -3.9905126498002668% | -16.197423469636536% | 23 | 23 |

## 启用判定

语义 owner/correlation 证据通过，且可用于开发期 operator-level 诊断。由于 shape 联合键只有 22.56% 交集且 layout 字段缺失，不能把单一 train 平均 rate 直接下沉为覆盖所有 shape 的正式 simulator 成本 profile；应按 shape×dtype×layout×kernel 做留出插值，未覆盖组合保持回退/不支持。该证据也没有提供并发 cohort 或客户端边界时间，因此不能用于 TTFT/TPOT/E2E 校准。

## 产物

- `artifacts/development/b06_qwen25_train_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b06_qwen25_holdout_v1.{json,sqlite,nsys-rep,trace.json,kernel.csv,api.csv,memcpy.csv}`
- `artifacts/development/b06_qwen25_calibration_v1.json`
- `artifacts/development/b06_qwen25_kernel_operator_coverage_v1.json`

结论：B-06 的 owner/correlation 与基础语义覆盖完成；通用 operator 成本面因 shape 联合覆盖不足而保持阻断，不能启用未经留出验证的全局校准。
