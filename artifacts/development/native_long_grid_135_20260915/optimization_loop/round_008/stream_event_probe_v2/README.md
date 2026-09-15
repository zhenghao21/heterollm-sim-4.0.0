# 第8轮：批量实际stream事件探针 v2

已编译并完成静态验证，尚未执行GPU测量。此版本只写本目录，旧探针、旧测量和native DLL保持不变。`copied_source_provenance.json`保存从v1复制源码与工具时的逐文件摘要。

12个预设配置：Q5_0/Q8_0 × M=1/2/4 × K=896/1024；N固定4096。K896为开发诊断组，K1024为预声明的合法对齐对照验证组。两种量化均按实际GGML块大小检查K，不填充到1024；不能把K1024的结果宣称覆盖896。所有数据来自固定seed的合成矩阵，不读取GGUF或LLM时间。

每种配置各用一个event进程与一个control进程；顺序固定交替。每次先执行1个first_call批、5个warmup批，再执行30个formal批。每批在同一个backend stream上连续提交同一MUL_MAT图64次，无逐图读回或人为休眠。event模式在64次提交前后记录计时事件，control不记录事件；**两者都用ggml_backend_synchronize等待**，消除v1的不同等待API混淆。主机wall与设备event不可相加，除以64只是摊销图时间，不能当作单kernel延迟、同步常数或launch系数。

每批结束且计时结束后检查最后一次图的完整输出有限性及最多4096个参考点。中间63次输出写入同一缓冲区，未逐个保存或核验；每次图提交的状态均检查。参考用解量化权重与原F32输入进行double累加。阈值仍为abs(error)<=0.05+0.03*abs(reference)，没有放宽；输入量化、上传、读回、正确性计算和身份哈希不计时。重复热缓冲区不能证明真实LLM权重的HBM/冷缓存行为。

`protocol.json`在执行前固定波动门p90/p10<=1.50及event/control主机中位数差<=20%，与v1相同。这些仅为诊断准入门，**不等于native±5%或仿真误差验收**。`assess.py`复用`full_raw_audit.py`逐批检查first/warmup/formal、64次提交计数、数值样本、QPC、环境、必需设备身份和实际加载DLL。缺失身份不能通过None==None。`run_frozen_matrix.ps1`和全部extractor纳入同一新manifest；运行前后再次核对。

身份检查，无GPU访问：

```powershell
.\invoke.ps1
```

主任务安排空闲窗口后运行固定矩阵：

```powershell
.\run_frozen_matrix.ps1 -IdleWindowConfirmed
```

只运行单个已声明配置：

```powershell
.\invoke.ps1 -Config dev_Q5_0_m1_k896 -Mode event -Output <new-output.json> -Run
```

所有输出必须新建，失败raw/log保留，矩阵汇总保留12组固定分母。runner只检查已知负载进程和运行前compute进程快照，WDDM查询中出现桌面图形进程仅记为background_graphics_present，独占状态明确为未验证；已知LLM、探针、hash及sim进程仍拒绝启动。操作者仍须安排低干扰窗口；端点频率记录不证明每个样本内时钟恒定。不停止任何其他进程。

旧版12组0准入事实保留于round_007/stream_probe_decision.json。其中validation_Q4_K_m64两模式first-call数值失败未修复、未删除；本版只测M<=4与Q5_0/Q8_0，不能声称覆盖或解决旧失败。已核对ggml_mul_mat输出为[N,M]、输入为[K,M]，旧参考ni=i%N、mi=i/N及按ni解量化权重行符合源码，未发现简单的转置/索引错误。旧失败4个点约0.056–0.059的差距，后续仍需区分算子精度路径和正确性问题；本版不调整其阈值。

构建：执行本目录compile.cmd，再在没有build_manifest.json时运行freeze_build.py。完成冻结后更改任何输入需要另建版本，不覆盖manifest。本目录不产生或拟合成本模型。
