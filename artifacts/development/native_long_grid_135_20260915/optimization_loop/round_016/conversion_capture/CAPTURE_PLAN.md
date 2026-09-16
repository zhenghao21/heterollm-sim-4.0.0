# 原 DLL MMQ 转换中间值捕获：实现前可行性与证据路径

状态：仅完成只读可行性检查，未编译、未运行GPU、未修改原生DLL、探针、collection或冻结数据。本方案提交主任务后再实施；当前采集进程不附加、不注入。

## 已确认

1. PE export table实际只有10个ggml_backend_cuda_*入口，没有quantize_mmq_q8_1_cuda或内核直接导出。公共ggml-cuda.h也没有暴露内部MMQ临时转换缓冲。不能直接GetProcAddress调用原内核或把目标tensor当中间q8缓冲。
2. 锁定mmq.cu:136–169在ggml_cuda_pool_alloc<char> src1_q8_1中分配转换空间，调用原quantize_mmq_q8_1_cuda后将同指针交给MMQ。内部pool对象不是公共后端张量，调用结束后的可读生命周期必须独立验证。
3. 锁定quantize.cu:458的原内核参数声明为x、ids、vy、ne00、s01、s02、s03、ne0、ne1、ne2、n_expert_used。CUPTI generated_cuda_runtime_api_meta.h提供cudaLaunchKernel_v7000_params，包含func/gridDim/blockDim/args/sharedMem/stream。cupti_callbacks.h明确提供symbolName和functionParams；参数内容只能在callback内读取并复制，不能把functionParams指针留到回调外。
4. 当前成功Nsight记录已经捕获cudaLaunchKernel_v7000；这支持优先订阅该实际ABI，不猜未观察到的launchEx ABI。若独立进程实际调用不同API，则停止并按官方元数据结构补支持。
5. CUPTI头文件明确只有单订阅者，与Nsight/Nsight Compute等同时运行可能返回MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED。必须由root在矩阵完成后的串行时窗启动新的非profile诊断，不附加现有进程。
6. 本机已有CUPTI头文件/import lib和2025.1.1 DLL；Nsight目录另含cupti64_134.dll。旧本地CUPTI对当前驱动/runtime的兼容性未证明，不允许拿新DLL配旧头文件就假称ABI验证通过。实现应记录实际库版本、订阅返回码并失败关闭。

## 优先设计：观察原 DLL，不改写转换内核

新增独立诊断可复用r3合成图构造与已锁定DLL，仅执行一个Q8_0 M64/N896/K1024图。先在进程初始化时注册CUPTI runtime launch callback（不启动Nsight）。callback本身只做无CUDA调用的固定大小记录：完整symbol、correlation/context、stream、grid/block/shared-memory、按官方launch元数据结构复制args中的参数值。只接受原quantize_mmq_q8_1 D4/non-scatter与主MMQ符号；输入shape、ids=null、步长、输出pointer及消费者pointer必须完整吻合，缺一即不导出结论。

不要在callback中调用同步或Memcpy，不修改launch参数，不替换函数，不解析猜测的DLL内存偏移。GPU原kernel仍由原DLL正常launch；callback只是观察。

图完成后，通过现有ggml_backend_synchronize确认完成，且在数值验证、cache sweep或下一图分配之前，尝试独立的受界限D2H复制。但**实现前必须封闭生命周期**：记录该临时buffer分配/释放/复用的行为，证明主MMQ和fixup不会覆写q8输入，并且pool返回后底层allocation仍存在。若公开API/可靠callback不能证明有效，则该复制结果只能标为不可信，不得解引用悬空设备地址或作原DLL数值证明。可以改为在独立编译的host wrapper中建立一个可证明的复制钩子，但必须明确wrapper变化和不可转移边界。

锁定block_q8_1_mmq布局：4个F32 d4占16 bytes，随后128个int8，sizeof=144。non-scatter ib=k_block*ne1+row（当前z=0）；此处从锁定结构和写入公式导出，不是经验偏移。目标M_index=11,K_index=33对应ib=11、qs[33]、d4[1]；候选偏移11*144+16+33、11*144+4必须通过sizeof/offsetof静态断言、实际参数与完整块dump相互校验。建议复制完整已界定的M*K_padded/128个块，并只报告关心位置及其邻居，避免凭单字节碰巧吻合。

输出包括原DLL、CUPTI、driver、probe SHA；callback原始launch与消费者链；原始buffer字节SHA及q/scales；QPC只用于顺序、不输出成本；重建input/packed-weight SHA；原值63/64的直接观测。保留原拒收，不改原容差或结果。

## 若原DLL捕获不能证明生命周期

可另建source-built微型转换程序，编译精确冻结quantize.cu及其依赖，分别显式精确/快速数学flag，只作边界诊断。但这条路线必须标注“源构建复现”，不能称原DLL执行证据；原DLL真实转换q仍未知。不得手写相似kernel当作证明，不得把CPU参考改到观测输出以通过门。当前没有启动这条备选编译。

## 仍缺的关键证明

- 当前驱动下可用CUPTI版本及callback结构匹配。
- 同进程捕获的原始kernel符号/参数和输出缓冲生命周期。
- 原DLL设备指令究竟是否在x*d_inv=63.5边界生成63。
- 即使源码复现某选项得到63，也不能反推旧DLL采用该选项。

主任务可先批准最小callback记录器的源码准备；所有编译与GPU执行由root串行安排。本目录仅存方案和身份清单，未创建可被误认为完成捕获的数据。
