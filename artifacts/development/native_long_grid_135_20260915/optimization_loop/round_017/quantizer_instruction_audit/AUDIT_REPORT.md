原 DLL 的近似倒数与舍入路径已得到静态机器指令证据。本轮仅使用 CPU 读取和反汇编，原 DLL 及第16轮冻结文件哈希均未变；没有执行 GPU、性能测试或修改参考/质量门。

第16轮根任务重放的事实已重新核对并汇总在 `replay_proof_summary.json`：q(row11,k33)=63，scale=0.003926435019820929，原始位模式 0x3b80a953。完整 D4 输出73,728字节，SHA256为313be32ba3637a22a3dbceaecab836caf450da899eccb32ae2005bd2966b9112。由真实q/d4及原DLL打包权重计算的64个第11行样本，与同进程原graph和归档算子输出的最大差为7.140915840864182e-7；64个原graph值与归档值逐项完全相同。原source-path仍有60项拒收，未覆盖。

反汇编严格过滤目标 mangled symbol 和 sm_120 架构族。原 DLL 为26,091,520字节，SHA256为8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297。84,760字节过滤输出中只有一个匹配函数；另142条“not found”来自其他fatbin，已明确计数，未当成匹配失败。匹配的是第48个ELF `ggml-cuda.48.sm_120a.cubin`，仅提取222,816字节，SHA256为1000c076b15dab0dbf817985fd8741b30c557dda9f46ac6251e536d3ffc98140。静态REG=22、SHARED=0、LOCAL=0与运行时属性相符；runtime binaryVersion/ptxVersion均为120，但这两个整数本身不编码架构的a后缀。

关键数值链：

- 0x0420–0x0510：FTZ绝对值最大值及4/2/1的shuffle归约，得到32元素amax。
- 0x0540：`MUFU.RCP R3, R15`，取amax近似倒数。
- 0x0590：`FMUL.FTZ R3, R3, 127`；随后0x05a0/05b0/05d0/05e0乘四个输入。
- 0x05f0–0x0760：带符号0.5、FADD.FTZ.RZ及F2I.TRUNC.NTZ，实现正常量化范围内的最近整数、平局远离零。
- 0x07b0：写q；0x07d0再次`MUFU.RCP R3, R3`，0x0810写D4 scale。

详细逐指令与source对应关系在 `instruction_chain.json`，可读精简汇编在 `quantizer_instructions.txt`。这证明了原部署二进制含有该指令链；未捕获逐指令运行轨迹或中间寄存器，不能把推导的inverse候选称为寄存器实测值。

进一步发现不能只修一个码：虽然65,536个码中仅1个与CPU source math不同，2,048个scale中有825个位模式不同，差为-3到+3个位步。条件性正常数值域审计覆盖了本输入全部2,048块，均存在同一inverse同时解释32码与scale的路径；其中4个输入位置存在多个合法候选码。文档误差范围不足以选出跨GPU一致的唯一近似倒数算法；原可执行fatbin也没有PTX可将具体PTX误差契约直接绑定到此SASS。该审计是条件证明，不是转换gate通过。

下一验证器变更应增加独立的D4操作数主计算参考和明确的验证范围字段；保留原FP32数学门、旧source-math结果与拒收记录。当前重放先作为一致性证据；后续主计算重放实际消费同一份自持D4字节后，才可验证“给定该操作数时main是否正确”。转换本身仍需独立验证。完整推导和具体下一步分别见 `NUMERICAL_CONTRACT_AND_VALIDATOR.md`、`next_validator_change.json`。

限制：当前读到的是同进程同原注册转换函数的自持重放输出，不是原pool字节；64样本一致不等于所有形状、所有数据和完整转换均正确。未读取LLM actual、未进行成本拟合、未创建或修改CUDA kernel。
