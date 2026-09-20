# HeteroLLM Simulator clean package

这是当前 V4 仿真器的最小可运行包：输入硬件、模型、负载和运行策略，统一进入 `ScenarioConfig`、
`TopologyAwareBatchCostProvider` 和 `UnifiedEventKernel`。前端是 `src/heterollm_sim/webui` 下的原生
HTML/CSS/JavaScript，不需要 npm。

## 安装与启动

```powershell
py -3.12 -m pip install .
py -3.12 -m heterollm_sim.cli demo
py -3.12 -m heterollm_sim.cli ui
```

开发环境先安装测试与进程检查依赖：`py -3.12 -m pip install -e ".[dev]"`。运行测试：`py -3.12 -m pytest -q`。Node.js 运行 `node --test tests/webui_*.test.cjs` 可检查静态前端。

历史测量、Bionic 适配器、原生批测和实验脚本不在此包中；它们仍保留在开发仓库。

本包现在附带一个不依赖 Python 推理库的原生 llama.cpp 对照入口：

```powershell
py -3.12 tools/native_llama_compare.py --output artifacts/native_compare.json
```

入口直接启动本机可用的 `llama-server.exe`，先读取项目37内的 GGUF header、tensor directory、量化块布局和 SHA256，再把实际 token 数、`--perf` timing、`/metrics` 差分、`/slots` 快照、完整命令和硬件快照写入 JSON。需要 CUDA kernel/API/memcpy 证据时运行 `py -3.12 tools/native_llama_profile.py --output artifacts/native_profile.json`，再用 `tools/build_calibration_profile.py` 生成带来源指纹和 `native-kernel-mapping/v1` 阶段聚合的校准配置；`native_llama_compare.py --calibration-profile ...` 默认只保存证据，只有额外传入 `--apply-launch-calibration` 才会显式启用 launch 参数。仿真器通过 `build_model_from_gguf` 使用相同模型张量，并通过 `apply_llama_runtime_config` 下沉 batch、ubatch、context、slot、GPU layers、KV 类型、Flash Attention、warmup 和 seed。其余 prompt/decode、同步、cache 和 host output 仍按阶段证据保留，直到获得可映射的 NVTX/CUPTI 数据。
多输入误差矩阵入口：`py -3.12 tools/native_error_matrix.py --exe <llama-server.exe> --model <project37-gguf> --output artifacts/native_error_matrix.json`。
## 运行框架

正式运行只保留两个入口：

```powershell
py -3.12 tools/predict_stable_native_dataset.py predict --selection <stable-native-dataset.json> --output <prediction-dir>
py -3.12 tools/predict_stable_native_dataset.py score --output <prediction-dir> --native-report <stable-native-dataset.json>
```

本任务当前禁止重新采集 native/微基准；上面的采集工具说明不是采集授权。执行约束以 `docs/TASK_BRIEF_AUDIT_20260914.md` 为准。

`predict` 内部完成静态场景投影和仿真；`score` 内部完成固定 native 绑定、Engine 时间戳重算、覆盖率和每格三项误差判定。独立 freeze、strict、failure-recheck、report 和 heatmap 不再是运行步骤。

当前评分按任务书A/B共用数值规则：每格 TTFT、TPOT、E2E 的未舍入 APE 均须严格小于25%，等于25%不通过。输出采用 `stable-native-simulation-errors/v2`，显式记录阈值与严格比较符；逐格字段为 `all3_below_threshold`。`strict_gate.gate=A` 表示当前固定集开发回归，不证明B门独立性或统计不确定性要求。canonical R0 已按此评分器完成 131 格评分；当前结果和 SHA 以 `optimization_loop/state.json` 及 `round_000/on/errors.0001.json` 为准。

## 当前优化起点

本次优化从固定R0基线开始，基线位于 `artifacts/development/native_long_grid_135_20260915/optimization_loop/round_000/on`。`artifacts/development/native_long_grid_135_20260915/optimization_loop/state.json` 以R0为基线、`rounds=[]`；首次优化轮次R1尚未执行。固定native范围仍为131格，循环外备份仅用于恢复，不参与本次优化计数。

R0是用户指定的比较起点。canonical R0 的 131 格 predict→score、worker seal、sidecar 身份和 strict identity 已闭合；A 门仍因数值精度失败而未通过，B 门尚未验证。后续候选须从 R0 冻结源码独立派生并按任务书核实身份及配对条件。完整约束与当前状态见 `docs/TASK_BRIEF_AUDIT_20260914.md`。
