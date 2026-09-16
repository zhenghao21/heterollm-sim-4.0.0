# Q8_0 MMQ 对齐配置数值拒收：只读诊断

**结论：误差强烈指向单个激活量化码的半整数舍入分歧，不能用放宽全局容差处理，也尚不能据此宣布原GPU计算有bug或新参考已被证明。** 输入行 M_index=11、K_index=33 的主机参考乘积恰为63.5，roundf得到64；假设设备该处得到63，能解释该行全部64个已保存抽样输出，最大剩余误差仅8.025672286748886e-7。GPU实际转换缓冲/指令没有捕获，因此“设备倒数偏低一个ULP”目前是有定量支持的候选机制，尚不是直接观测。

## 输入与拒收状态

只读 collection_r3/runs/aligned_Q8_0_m64_n896_k1024/pair_01/profile 和 direct 的 microbench.json。两进程均保留36次调用、status=failed_compute_or_correctness；每次60/4096个抽样输出超出冻结path门，全部集中于M_index=11。每次path最大误差均为0.0017982302233576775。profile/direct全36轮抽样记录逐项相同，没有时间扰动引起的随机输出变化迹象。

最大误差样本：M=11,N=772，actual=-1.7001020908355713，path_reference=-1.701900321058929，差+0.0017982302233576775；原F32数学参考=-1.7060345661011524，math_pass=true。存在正负交错差异，不是统一加性偏移或乘性比例漂移。

## 轻量独立复现

使用探针冻结xorshift32 seed=20260914，在CPU内存重建不到100万个F32随机值；没有启动probe、DLL、GPU、编译、测试套件或新依赖。按固定quantize_row_q8_0_ref语义重建packed权重，得到两个与raw完全一致的SHA：

- input：853d9cdaa9c6c57b2761588349a9c0513d7915510ea1b43c92610f7338619a76
- packed Q8_0：4a38682d8cb2b39eb983c93fb401b5af4b59ed8b79ca07ae64af5f40261c6e58

因此不是随机数、矩阵索引、权重反量化或输入重建错误。只对输入行11的64个已保存样本重做块整数dot及F32 scale，主机参考与已记录path_reference **逐项误差0**。

该行第1个32元素块中的索引1（全行K=33）：

| 项 | 值 |
|---|---:|
| x | 0.24932861328125 |
| amax | 0.4986572265625 |
| 主机F32 d_inv=127/amax | 254.6839599609375 |
| F32 x*d_inv | 63.5 |
| roundf码 | 64 |
| d_inv向下1 ULP | 254.68394470214844 |
| 对应F32乘积 | 63.499996185302734 |
| 对应roundf码 | 63 |

只替换这个激活码64→63，其余权重、scale和运算均保持原冻结参考，64个row11样本全部落回**原始**1e-4+1e-5×abs(reference)门内；最大残差8.03e-7。误差符号随对应权重码变动，符合单输入量化码造成的列相关响应。没有用输出拟合连续参数；这是显式单离散码差异的机制反例，不能作为新采集结果或正式验收。

## 源码一致与因果边界

probe/r3/source_reference.h 中MMQ D4使用F32 inverse=127/amax、round(values*inverse)、scale=1/inverse；Q8_0使用signed q8 dot与half权重scale，按块结果累加到double。冻结quantize.cu:515–521也声明float d_inv=127/amax、roundf(xi*d_inv)、float d=1/d_inv；D4的32元素最大值语义一致。相关quantize.cu、mmq.cuh、mmq-load-tiles.cuh当前SHA与probe identity_lock全部一致。

文字公式一致不意味着CPU与当前DLL设备指令在半整数边界位级一致。没有实际GPU转换buffer、PTX/SASS或该DLL对应完整编译选项证据；不能擅自断言use_fast_math已开启，也不能把FMA当成已证实原因。这里xi*d_inv没有加数，直接FMA归约更不适合解释“只一行、单码变化即可闭合”的模式。普通块归约/FMA顺序误差只能解释替换后约1e-6量级残差，无法自然解释稳定1.8e-3列相关差异。

## 不改变门槛的后续建议

1. 当前配置与所有失败轮次继续保持拒收、保留全26分母，不能因数学宽门通过就覆盖path失败。
2. 主任务若准备新参考版本，先独立确认当前DLL quantize kernel在这一输入位置实际写出的q8值及scale，或取得已绑定设备指令/编译选项证明。该诊断未新增GPU测量；应另立明确受控合成证据任务。
3. 新参考须明确其目标：精确复制设备量化语义，或预注册、仅在严格识别的半整数敏感位置允许离散量化不确定性。两种路线都需要独立边界用例和输出传播界，不得依据本次误差 blanket放宽atol/rtol。
4. 保留原数学误差与源路径误差双轨；Q5/Q8和MMVQ/MMQ分别验证，不把单Q8配置发现推广为所有量化算子已正确。

复现脚本：numeric_q8_mmq_diagnosis.py；数值摘要及64行逐样本比较：numeric_q8_mmq_diagnosis.json。无LLM actual、模型时延或拟合输入；旧冻结参考与门槛完全未改。
