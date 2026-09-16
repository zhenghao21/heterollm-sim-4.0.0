# R27 大文件身份诊断（仅准备，未读取真实模型）

目的：保留两个不同错误 SHA 的旧失败，检查新独立读取中的算法一致性、块内容一致性和串行/并发差异。它不会修复、重试或追认 R27/R24 结果，也不能单独证明硬件损坏。

当前没有可复用的完整双算法诊断。既有 gguf_parity 是单路 Python 哈希；R28 的 identity_support.h 有用于 DLL 身份的 BCrypt 调用，但没有同一读取块的双路对照。本工具用 CPython _hashlib.openssl_sha256 与 Windows BCrypt SHA256（明确 Microsoft Primitive Provider），BCrypt 不可用时直接拒绝，无同一 OpenSSL 包装层替代。

每次独立打开同一文件，使用最多1MiB的不可变 bytes 块，把同一个块喂入两路完整哈希，并分别保存每块的两路摘要、偏移和长度。保存实际 bytes、路径/句柄前后 stat、错误、全文件摘要。任何块摘要/完整摘要分歧、预期身份不匹配、文件状态变化、短读或覆盖不足都拒收；不会选择匹配预期的那一路。文件内容不写入诊断日志。

执行顺序固定：先1个串行 pass，再4个独立 Windows spawned Python worker，并发各读1遍。四worker用同一释放事件启动；并发重叠程度取决于OS调度，不伪称时间完全重合。每个 pass 在读文件之前，仅运行一次无磁盘的100万ASCII a控制，每个哈希实现使用一次1,000,000-byte update；预期SHA为 cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0。控制失败也保存两路结果并停止该worker的模型读取。小向量自测不重复此大buffer控制。

**最大预算**：同一14,865,116,128-byte文件最多打开5次，模型逻辑读取总量最多74,325,580,640 bytes（74.326 GB，约69.22 GiB）；各pass使用同一块同时做两算法，不额外重读模型。辅助metadata、已加载crypto程序/DLL和诊断日志读取共用另一个2GiB硬预算，因此本工具显式文件内容读取上限76,473,064,288 bytes（约76.473 GB）。大buffer控制另外只有5×1,000,000 bytes的内存输入、0磁盘输入。操作系统缓存不被清除或绕过，物理磁盘流量未知。文件尺寸不同则读前拒绝。不会为了查EOF再额外读1byte。无自动重试。

真实执行须R27的predictions_complete.json证明off/on各131条均已终态（允许失败/不完整，但不允许pending），并核对control/freeze/262条身份引用；只检查状态与身份，不计算模型误差。启动前、串行后和结束后检查项目native/simulator/probe进程空闲。三次快照不是持续互斥证明，root仍负责串行调度。所有输出仅在本目录 runs/read_comparison_0001，目录已存在就拒绝再次执行；旧失败不改。程序不改变文件内容、BIOS、CPU频率、驱动、OPENSSL全局flags或缓存策略，不测模型时延、不生成cost参数。

准备与小fixture验证（已经允许）：

~~~powershell
E:\anaconda\python.exe -m pytest -q test_identity_diagnostic.py --basetemp .test_tmp
E:\anaconda\python.exe prepare.py
E:\anaconda\python.exe diagnostic.py self-test
~~~

R27全262终态且项目idle后，由root审阅并执行：

~~~powershell
E:\anaconda\python.exe diagnostic.py run --manifest preparation_manifest.0003.json --authorize-large-file-read
~~~

解释约束：

- 同一块两算法不同，说明差异发生于进入两实现之后的算法/状态/内存处理路径，不能直接断言磁盘或硬件坏。
- 两算法一致但不同pass对应块不同，只能定位不同读取/并发状态下观测到的字节差异；仍需考虑缓存、调度、内存、存储及文件变化证据。
- 两路在某pass同时给出错误全文件SHA，仍然拒收；一致不是正确性的替代。
- 五pass全匹配仅说明这次有限诊断一致，不能撤销旧失败或证明间歇性问题不存在。
- 无磁盘百万a控制在并发时也失败，会提高hash/update/内存路径的调查优先级；控制通过而模型不一致，也不能自动确定存储故障。

上限约束的是工具显式读取的文件内容，不包含Python/操作系统隐式加载、页缓存或后台系统IO；不把逻辑字节数当物理磁盘流量。OPENSSL相关进程环境只记录和核对，不修改。

当前准备清单为 preparation_manifest.0003.json；初版 preparation_manifest.json 保留为历史。初版交付前核验发现 Windows 系统目录大小写（WINDOWS/Windows）的路径比较误报，后继版本仅规范路径比较，仍逐文件核对SHA。

0003修复：路径stat与句柄fstat跨API比较dev/ino/size/mtime/birthtime；原始ctime/birthtime全部保留，并分别在path前后、handle前后比较ctime。进程清单包含ParentProcessId，明确阻止gpu-operator-timing.exe、相对runner.py run/resume/extend及其spawn子进程；父归属缺失/循环/不可判定时拒绝，只有完整明确的host自测命令及其子进程可放行。BCryptDestroyHash与BCryptCloseAlgorithmProvider返回值均检查；失败拒收，即使digest已匹配；若先前已有异常，原异常与所有cleanup错误共同保存。0002清单与交付前源码副本保留在prepared_revision.0002。
