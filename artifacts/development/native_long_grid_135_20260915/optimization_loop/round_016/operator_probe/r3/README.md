# Round016 r3：固定原生 DLL 显式加载修复

r3 从 r2 复制源码和协议后独立构建。r1、r2、原始 collection 失败记录、原生 DLL、resolver 和任务书均未修改。当前仅完成编译及主机测试，**尚未执行 GPU probe，也尚未生成 build_manifest.json**。由 root 审查后冻结，再决定首个 GPU pair。

## 根因和修复

首个 profile/direct 在 GPU 初始化前都退出：`expected native DLL not loaded: ggml.dll`。r2 的 PE 导入表实际包含 `ggml-base.dll`、`ggml-cpu.dll`、`ggml-cuda.dll`，没有 `ggml.dll`；链接代码未调用后者接口，不能依赖它自动加载。身份门却要求四库都已在进程中，形成了初始化顺序错误。

r3 保留四库身份和所有原有检查：

1. 先验证冻结文件的 SHA。
2. 从冻结的驱动器绝对路径逐个 `LoadLibraryExA`，使用目标 DLL 目录和系统默认安全目录寻找传递依赖。
3. 每次加载马上通过真实模块 handle 查询实际路径，并核验路径和 SHA。
4. 持有四个 handle 到整个运行结束；图资源和 CUDA 事件先销毁，然后逆序释放本 probe 增加的引用。
5. 首次 CUDA API 调用前、测量前和测量后仍执行原有 `verify_loaded_modules`。

没有删除 `ggml.dll` 条目，没有使用模型专属参数或改计时容差，没有修改任何原生 DLL。失败加载、错误路径或错误哈希都会拒绝执行。

`frozen_module_guard.h` 封装引用生命周期。主机测试使用假的加载API和虚拟路径，**不会加载真实 GGML/CUDA库**。编译得到的 `module-guard-host.exe` 导入表也没有 GGML、cudart 或 nvcuda。

## 保持不变的测量契约

仍使用 raw schema `single-operator-surface-probe/v2` 和 timing contract `actual-backend-single-graph-envelope/v2`，完整单图 NVTX、原始 QPC tick、36次调用、缓存清扫、全部数值检查和双参考语义不变。26组合、所有阈值、direct/profile应用参数策略也与r2相同。

身份加载发生在首次 CUDA 初始化与所有计时之外。实际模块列表中应包含四个冻结GGML库；这由root的首次实际运行核验。本次主机模拟测试不能代替真实DLL加载和GPU运行成功的证据。

## 验证

- 新加载器7个测试与既有quality12个测试：**19 passed**。
- C++ fake loader：8项生命周期/负例检查，包括第四个未导入库被加载和验证、部分加载失败、身份不符、空handle、相对路径、重复调用及不可复制。
- 双参考 C++/NumPy：1,536个样本，最大差0。
- 旧MMVQ原始输出只读参考核验：49,152个样本，最大差 `1.3373792171478271e-06`。
- PowerShell语法、单图/NVTX边界静态检查及46项锁定依赖验证通过。
- r2复制来源文件SHA复核通过。

详细身份和限制见 `module_preload_validation.json`、`revision_provenance.json`。

## Root下一步

审查r3后执行 `E:\anaconda\python.exe freeze_build.py`，它只生成新构建清单，不运行GPU。随后collector的新revision应指向 **operator_probe/r3**，接受 `round016-operator-probe-ready/v3`，保持相同的v2 raw/timing契约，并建立新冻结。不得把r2旧pair的失败改成成功，也不得将两版本当作同一次冻结采样。

collector首个新pair仍应先验证真实加载库、actual kernel dispatch、数值参考以及计时归属，再扩大矩阵。此目录没有自动开始GPU的后台任务。
