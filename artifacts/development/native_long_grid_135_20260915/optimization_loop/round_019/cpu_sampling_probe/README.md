CPU 采样探针的源码和冻结协议已经准备完成；当前没有编译、加载 DLL、运行探针、运行模型或访问 GPU。只读检查结果见 `readonly_inspection.json`。本轮没有新增任何时延测量值，也不影响 R18 冻结结果。

范围分为两个相互独立的操作：

- `candidate_loop`：逐字复制锁定 `common/sampling.cpp` 的 resize、id/logit/p 构造及 view 赋值语句，在外部编译的 noinline 包装中运行。它是源码等价的外部编译操作，**不是原 common.dll 测量，也未证明二进制等价**。编译阶段会保留汇编清单，供后续核对实际循环。
- `original_dll_topk_apply`：绝对路径加载锁定原 `llama.dll`，通过 `GetProcAddress` 获取 `llama_sampler_init_top_k(1)`、`llama_sampler_apply`、`llama_sampler_free`，只计时 apply 调用。导出指针所属模块、已加载 llama/ggml/ggml-base 实际路径均检查一致；DLL 释放前先释放 sampler。无 llama_backend_init、模型加载或上下文创建。

原 DLL 的 PE 表已只读解析，三个 API 均存在且非转发导出。top-k 的源码路径是 `llama_sampler_apply -> top_k_apply -> top_k_impl -> partial_sort_inplace -> std::partial_sort(k=1)`。这个 probe 测量原实现整体 apply 操作；不能把它直接命名为纯比较指令开销。

固定域为3个词表大小（32768、131072、262144），2个有限且唯一最大值的 logits 模式（单调升序、固定种子的随机排列），3个独立顺序进程。每个 stage/case/process 有1次单独记录的首次调用、16次热身、64次稳态原始样本，共2304个稳态 stage 样本。尺寸131072在采集前声明为保留验证集，另外两尺寸为训练集；本探针不执行拟合。三进程分别从不同尺寸开始循环测试，顺序已固定，不按结果选择。

两段时间窗口相互独立且不嵌套。candidate 首次调用包含空vector的首次分配和first touch；稳态使用相同data/size/capacity。top-k 的每次调用前都在计时外 memcpy 恢复全部候选并重置size=V、selected=-1、sorted=false，绝不在已经缩短/排序的数组上连续测量。初始化、reset、质量检查和频率查询均在对应时窗外，首次调用与稳态不混合，也不宣称实际cache-cold。

质量检查采用精确值，不使用容差：候选循环检查每个id/logit位模式/p=+0；top-k检查唯一最大值id/logit、size1、sorted=true、selected=-1与原buffer指针。检查、baseline复制都会影响缓存，因此结果限定为协议定义的复用热缓冲区状态。它不是一般缓存模型或通用engine采样总开销。

使用 QueryPerformanceCounter，保存整数原始tick及其frequency。每进程另记录64个空时钟括号的观察开销样本，不做噪声相减。程序钉住一个显式选择的逻辑CPU并验证线程实际位置；保存CPUID品牌、线程/进程id、group和mask。每stage前后记录Windows ProcessorInformation报告的current/max/limit MHz，报告变化则该stage只能诊断使用；稳定报告也不证明实际turbo周期恒定，禁止把该MHz直接当作周期换算依据。

不覆盖的部分：bias/model suppress、top-p/min-p/temp/dist及RNG、accept/history/commit、logits传输/同步，以及整个sampler chain。因此，不应把两段结果相加成完整native sampling cost。后续是否使用这些证据，由root结合来源和覆盖范围决定。

已经完成的只读检查命令如下。其JSON输出已独占写入，直接再次执行会拒绝覆盖原检查文件。

```powershell
& E:/anaconda/python.exe "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_probe\entry.py" inspect
```

以下为未来root的命令，本轮没有执行。编译必须在root确认无测量干扰后执行；它只生成独立probe，不重建原DLL。失败尝试保留，成功后创建build_manifest并锁定源码、protocol、可执行文件、对象、汇编、compiler及实际showIncludes头文件清单。

```powershell
& E:/anaconda/python.exe "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_probe\entry.py" compile --root-idle-confirmed --root-compile-authorized
```

测量仍由root单独决定并授权。先根据机器拓扑选择实际允许的一个逻辑CPU；以下以0号为明确候选，若root选择其他CPU，应在首次series冻结之前替换此数字。程序会验证CPU属于唯一支持的Windows processor group且位于进程允许affinity内。

```powershell
& E:/anaconda/python.exe "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_019\cpu_sampling_probe\entry.py" measure --root-idle-confirmed --root-measure-authorized --cpu-index 0 --series sampling_series_0001
```

该命令先创建独占series目录和run_freeze，再顺序启动3个独立进程。不得和root的LLM、GPU、CPU锚点或静态仿真并发测量。每个进程运行前后核对冻结文件，任何失败原样保留并停止，不自动重试；若确需重做，要建立新series并保留旧失败记录。源码或协议若变化，成功构建后须新建版本，不能覆盖build_manifest。

交付文件保持紧凑：一个C++、一个入口脚本、protocol、source_identity、只读检查、README和readiness。实际raw repeats将集中于每进程单个JSON；其stdout/stderr、receipt以及总质量JSON只有在未来root授权运行后才产生。若未来编译发现工具链或SDK差异，应修复新revision并重新冻结；当前“准备完成”不代表已经编译验证。
