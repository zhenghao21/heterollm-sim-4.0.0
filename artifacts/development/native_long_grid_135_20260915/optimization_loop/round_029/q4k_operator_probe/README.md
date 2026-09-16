# R29 Q4_K target路径与独立数值资格（准备态）

范围只限真实锁定DLL上的GGML_OP_MUL_MAT：Q4_K/M1/K2048/N2048。没有计时器、活动时间戳、HES、LLM调用或成本参数。静态候选45格的cache/layout均未验证，本结果不会自动迁移到这些格子。

复用R26 ExC参数记录器的本地副本，明确改为Q4_K type12、row stride8、Q8 stride64。预计main grid2048×1×1、block32×4×1；源码generic warps4，8个superblock不满足8<8，smallK=false。conversion预计grid8×1×1/block256×1×1。两者均须实际ExC430及唯一PDL attr6=1；符号/几何/参数/指针/stream/EXIT返回任一不符就拒收。静态目标DLL resource dump仅证明符号存在，实际路径尚未执行或验证。

目标DLL不重编：使用GGML backend为权重/输入/输出分配真实CUDA buffer；一次不计时warmup后捕获一次replay。捕获期间不允许任何分配/释放（包括async、pool分配和host分配）/memcpy/memset；仅允许两次kernel提交均返回成功后的末尾同步，callback只复制host参数，绝不调用CUDA或读device memory。通过路径门后，在禁用callback并同步之后回读actual.q8_1.bin与全部actual.f32.bin。

权重由Python直接生成144-byte Q4_K block（d,dmin,12-byte scales/min,128-byte nibbles），不调用GGML量化/反量化；scales/min覆盖1..63和高位编码。输入是有正负且非零的dyadic，每32项max=127*d，d与原始sum均能被half精确表示，每块sum非零。独立解释Q8_1的36-byte块、六位scale/min后，以整数dot+double计算全部2048输出。min修正严格非零；回读Q8必须逐字节等于独立生成值。

误差界：参考构造均为小幅dyadic，double累加精确；DP4A及scale/min整型中间值小于2^24。Q4_K每256块16个partial，本shape共128 partial；保守以2K+64=4160覆盖局部FP32运算和全部归约路径。逐行界=gamma4160乘unsigned positive/min分量绝对幅度总和，避免抵消产生虚假紧界。无测后调容差，无拟合native时延。主机测试用独立逐元素反量化dot对照分组参考，另验证忽略min修正会被界拒绝。

所有合成packed/input/Q8/reference/bounds/amplitude/min-correction文件先冻结。数值结果保留每个输出actual/reference/bound/error与全量raw。target raw仅标path_observed_pending_independent_numeric，只有Python重算独立参考并检查全部输出后才资格成功。该成功仍是单fixture/单shape正确性，不是数值泛化、缓存状态或成本资格。M4以及K1024/4096仅协议留出，未执行。

本轮允许命令：
~~~powershell
E:\anaconda\python.exe build.py prepare
E:\anaconda\python.exe -m pytest -q test_probe.py
E:\anaconda\python.exe build.py build
E:\anaconda\python.exe build.py host-test --manifest build_manifest.0003.json
~~~

未来仅root串行，所有R26/identity/native/simulator/GPU任务自然完成、project idle后：
~~~powershell
E:\anaconda\python.exe run_probe.py --manifest build_manifest.0003.json --execute-untimed-qualification
~~~

runner核前后源码/fixture/DLL/EXE/header身份，固定runs/capture.0001独占输出。自然等待，无timeout kill；失败保留并禁止自动再试。主机decoder selftest不导入CUDA/CUPTI/GGML DLL。CUPTI采用R26已选回调库与冻结的参数ABI，不使用ActivityKernel结构或HES模式；新Q4实际callback兼容性必须在未来这次capture中确认。

本交付使用0003构建；0001/0002源码、清单和产物均保留。0002已绑定link.exe和Python解释器身份，必需原始文件缺失或CUPTI捕获清理失败会最终拒收。0003在R29本地冻结进程检查器，除冻结R28排他规则外，直接/孤立启动的q4k-target.*.exe及本目录其他非host-test可执行文件也会阻断；不改R28 guard。捕获期分配和释放一律拒收，辅助调用只允许两次launch均完成提交后的成功同步。当前不应执行旧0001/0002。
