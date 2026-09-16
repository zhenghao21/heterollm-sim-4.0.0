# R19 CPU sampling loader R4 修复审查

日期：2026-09-16。旧 `cpu_sampling_probe` 的 build manifest、CPU identity 和 `sampling_series_0001` 没有修改。该系列在 `process_0`、任何计时开始之前，因 `LoadLibraryExW` 无法加载 `llama.dll` 失败，`completed_cases=[]`；R4 因此位于新的 `cpu_sampling_probe_r4`，且没有复制旧 binary、build 或 run 产物。

## 修复内容

- 新版本冻结五个 runtime 模块：`llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-cpu.dll`、`ggml-cuda.dll`，以及 `E:\cuda\bin` 的 `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`。入口在编译前、每个进程前后按 SHA/bytes 重验。
- R18 `microbench.json` 的 `setup.loaded_modules_before` 被作为 ggml/CUDA 目录与 SHA 的来源绑定；`llama.dll` 继续由 native runtime source identity 绑定。
- C++ 只对两个冻结绝对目录调用 `AddDllDirectory`，绝不修改 `PATH`、复制 DLL 或放宽任意目录搜索。它按冻结顺序预加载 CUDA、ggml，再加载绝对 `llama.dll`；所有 dependency handle 与目录 cookie 会持续到 sampler 释放、llama unload 后才逆序释放。
- 每个 `AddDllDirectory` / `LoadLibraryExW` 失败立即记录 `GetLastError` 数字和 `FormatMessageW` 文本。成功时记录 8 个实际加载绝对路径；入口逐一与冻结路径精确比较。
- child 环境只显式写入 `LLAMA_TRACE_ANNOTATIONS=0`；不传入或变更任意 PATH 值。CUDA dependency DLL 被加载仅为解析依赖，输出明确标为非 GPU 测量、无 probe GPU API/context 调用；并不对 DLL 初始化副作用作无法证明的声明。
- R3 CPU identity、36 stage / 2304 sample、P90/P10、跨进程、observer 和频率门完整保留。

## 验证

- [STATIC_LOADER_BINDING_VALIDATION_R4.json](F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_review\loader_r4\STATIC_LOADER_BINDING_VALIDATION_R4.json) 只读复核通过：26 个输入引用、5 个 runtime DLL、3 个 CUDA DLL 与 R18 setup 证据均匹配当前 SHA；未加载 DLL。
- loader 与原有 timing quality 回归合计 `16 passed`。覆盖模块集合/路径拒绝、固定加载顺序、handle/cookie 释放顺序、错误上下文、无 PATH 修改，以及原有 CPU/timing gate。
- Python 入口语法通过。没有编译 R4 C++，没有加载原 DLL，没有执行 GPU、模型或时序测量。

## Root 后续步骤

先在空闲窗口编译 R4，再使用 R4 executable 执行 identity-only 冻结，最后才允许一个新的 R4 series。旧失败系列不重用，也不覆盖。
