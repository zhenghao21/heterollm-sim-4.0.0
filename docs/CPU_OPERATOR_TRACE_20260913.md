# CPU operator wall-time trace

项目37的 semantic direct binary 现在支持一个默认关闭的 CPU graph operator trace。设置
`GGML_CPU_OPERATOR_TRACE=1` 后，`ggml-cpu.c` 在每个可执行 graph node 的现有
node barrier 范围内，由 worker 0 输出一条 `CPU_PERF|...` 记录。记录包含操作名、节点名、
输出及两个输入的形状和类型、线程数、graph/node 编号、融合标志，以及微秒级起止时间。
未启用该变量时不增加额外 barrier，也不改变普通计算路径。

## 构建

```powershell
cmd /S /C '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Users\A\AppData\Roaming\Python\Python312\site-packages\cmake\data\bin\cmake.exe" --build "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\source\llama.cpp-semantic\build-semantic-direct" --config Release --parallel 16'
```

## 采集与转换

运行 native harness 时设置环境变量：

```powershell
$env:GGML_CPU_OPERATOR_TRACE='1'
py -3.12 tools/native_llama_compare.py --exe source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe `
  --model artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf `
  --prompt 'Hi.' --predict 2 --ctx 512 --parallel 1 --batch 64 --ubatch 64 --threads 16 --gpu-layers 0 `
  --output artifacts/multimodel_next/qwen38_cpu_operator_trace_direct_p2_v2.json
```

将 harness 日志转换为 formal prompt + decode graph：

```powershell
py -3.12 tools/extract_cpu_perf_trace.py input.llama.log --formal-graphs 3 --output cpu_trace.json
py -3.12 tools/build_semantic_calibration.py train_cpu_trace.json holdout_cpu_trace.json `
  --schema native-semantic-calibration/v2 --output cpu_semantic_calibration_v2.json
```

解析器默认保留最后三个 graph；阶段依据输出 tensor 的 `ne[1]`（M>1 为 prefill，M=1
为 decode），并把 CPU 事件标为 `kind=cpu_operator`、`backend=cpu`。校准器只对明确的
operator owner 应用 wall-time 系数，量化或无法稳定归属的事件继续保持 blocked/excluded。

## 解释限制

CPU operator trace 的 wall 时间包含该节点 worker 0 的计算和现有 barrier 等待，但不新增同步，
最后一个节点沿用 graph 末尾的既有 barrier。Qwen3.8 CPU-only 回放显示 `cpu0.memory` 服务
占总 makespan 主导，单独的 operator wall 校准无法替代内存流量和带宽测量；后续应按 quantized
权重实际读取字节核对 CPU memory demand。
