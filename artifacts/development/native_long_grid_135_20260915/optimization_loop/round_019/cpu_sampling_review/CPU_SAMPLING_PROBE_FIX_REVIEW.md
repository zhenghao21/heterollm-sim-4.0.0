# R19 CPU sampling probe R2 修复审查

日期：2026-09-16。R1 准备证据保持原样；本次只更新未冻结的 `cpu_sampling_probe.cpp`、`entry.py`、`protocol.json`，新增 `source_identity_r2.json` 与本目录回归测试。没有编译、加载 DLL、运行探针、执行 GPU 或产生任何时序值。

## 已修复的测量接受门

- **实际 CPU 身份冻结。** 新增编译后、测量前的 `--identity-only` 路径。它只固定一个显式逻辑 CPU 并读取 CPUID brand/signature、processor group 数、group-0 active CPU 数、逻辑 CPU 和实际 thread affinity mask；不加载 DLL、不创建模型/GPU 上下文，也不读取 QPC 时序。`capture-identity` 将该输出与当前 build/protocol 绑定为不可覆盖的 identity freeze。
- **三进程精确一致性。** `measure` 必须传入 identity freeze；每份进程结果的顶层 CPU identity 和所有 stage 前/后的 group、logical CPU、affinity 都与冻结值精确比较。identity freeze 若属于不同 build 或 protocol，会在运行前拒绝。
- **三项频率字段。** stage 稳定性现在同时比较 `os_max_mhz`、`os_reported_current_mhz`、`os_limit_mhz`。任何一项变化都会写为 `diagnostic_only=true`、`timing_usable=false`、`frequency_changed_diagnostic_only=true`；系列 `quality.json` 同样写 `accepted_for_timing_evidence=false`，入口随后以错误结束，避免漂移系列被接受为时序证据。
- **范围仍保持。** 保留 K=1、12-byte 记录、6 synthetic cases、16 warmups、64 steady raw ticks、3 独立 PID、2304 steady samples、holdout V=131072；没有增加 LLM、模型名、GPU、内核系数、延迟外推或拟合逻辑。

## 验证

- `E:/anaconda/python.exe -m py_compile entry.py` 通过。`entry.inputs()` 的 R2 只读来源/PE 复核也通过，结果见 `STATIC_SOURCE_VALIDATION_R2.json`；确认 20 个冻结输入引用与所需非转发 sampler 导出，且 DLL 没有加载。
- [test_cpu_sampling_probe_r2.py](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_review\test_cpu_sampling_probe_r2.py) 通过：`7 passed`。
  - 覆盖匹配 CPU identity；CPU affinity 篡改拒绝；max/current/limit 三类漂移各自变为不可用；错误的“稳定/可用”标记拒绝；以及源码/协议是否包含 identity-only 与三字段门。
- 未执行 C++ 编译或运行，所以 static asserts、实际 DLL 生命周期和身份捕获命令仍等待 root 的下一阶段授权。

## Root 后续命令

在空闲窗口且认可编译后：

```powershell
& E:/anaconda/python.exe "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_probe\entry.py" compile --root-idle-confirmed --root-compile-authorized
```

仅当 CPU 0 由 identity-only 路径验证为可固定、允许的逻辑 CPU 时，创建冻结身份：

```powershell
& E:/anaconda/python.exe "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_probe\entry.py" capture-identity --root-idle-confirmed --root-identity-authorized --cpu-index 0 --identity-freeze cpu_identity_r2_cpu0
```

后续测量必须引用上述目录中的 `actual_cpu_identity_freeze.json`。任何频率漂移会保留 raw 记录，但不会成为可用时序证据。

