# Kernel-aware GPU 成本模型

## 使用边界

`GPUProfile.kernel_model` 为显式启用入口；不配置时保留原有解析成本及冻结实验兼容性。
本次没有采集新的 GPU 微基准，也没有把目标 LLM 的 TTFT/TPOT/E2E 当作训练数据。
**新增模型能力不代表已验证达到 10% 误差。** `caller_declared` 表示外部声明，不能当作已经核验的硬件实测。

## 配置

在对应 GPU profile 内加入 `kernel_model`。例如下面是**示意配置，不是实测参数，也不是所有 llama.cpp 版本通用的分派规则**：

```json
{
  "hardware_id": "your-exact-device-and-clock-profile",
  "runtime_id": "your-runtime-build-driver-fingerprint",
  "architecture": "your-gpu-architecture",
  "stateful_l2": false,
  "kernels": [
    {
      "kernel_family": "cuda_mmvq_q4_k",
      "weight_formats": ["q4_k"],
      "activation_dtype": "fp16",
      "phase": "decode",
      "compute_primitive": "dp4a",
      "evidence": "replace-with-dispatch-source-reference",
      "min_shape": [1, 1, 32],
      "max_shape": [4, 65536, 32768],
      "cta_geometry": [1, 128, 32],
      "warps_per_cta": 4,
      "registers_per_thread": 32,
      "shared_memory_per_cta": 0,
      "surface_model": "calibrated_analytical",
      "samples": []
    }
  ]
}
```

`min_shape/max_shape/cta_geometry` 顺序均为 **M,N,K**；时间 ns、带宽十进制 GB/s、容量 bytes。
Q4_0、Q4_K、IQ4_XS 必须单独配置和采样，不能拿同一“4-bit”速率互换。
FP16 GEMV/GEMM 同样用 descriptor 表达，`weight_formats=["fp16"]`、`internal_dtype="fp16"`。
`activation_dtype`、输出位宽、累加位宽、layout、epilogue 均须匹配；不匹配时明确回退。
正式 planner 使用带样本的 profile 时，必须同时设置 `workload.metadata.kernel_runtime_id` 和 GPU 组件的 `metadata.kernel_hardware_id`，与 profile 身份完全相同；无绑定或不匹配会报错。
同一有效域匹配多个 kernel 会报错，避免依赖配置排序偷偷改变 dispatch。
phase 由 planner 的执行阶段绑定，不通过 M 猜测；MTP 未单独建模时回退。

硬件提供寄存器数、shared memory、线程/warp/CTA 上限，驻留 CTA 取这些约束的最小值。
参数不合法或连一个 CTA 都无法驻留时拒绝执行。静态旧 occupancy 不再支配匹配到的新 kernel。

## 性能曲面及证据

每个 `samples` 行包含 M/N/K、device_ns、analytical_ns、stddev_ns、sample_count、evidence，可选 achieved_bandwidth_gb_s。
- 精确点：实测 device wall 可以替换偏高或偏低的解析 wall，不仅作为下限。
- 完整联合网格单元：log-shape 多线性插值。可选解析基线乘修正系数，或直接插值实测 wall。
- 超域或缺少联合角点：返回解析值及原因、距离，**不外推**。
- stddev 是测量离散度，不是预测误差置信区间；端到端不确定性不伪造，不简单累加相关误差。
- 单个 device wall 无法识别 tensor/HBM 等资源各自占用；缩放解析资源比例会明确标记为未验证假设。
- 已有 source MMQ/MMVQ 特化带额外 launch/stride/stream-K 条件，普通 M/N/K 曲面不能冒充相同联合域，因此保留解析特化并输出排除原因。

`load_microbenchmark_surface(path, expected_sha256=..., profile=..., kernel=..., analytical_cost=...)`
可读取独立算子重复测量，核验文件 SHA、设备/runtime/格式/阶段/缓存协议，并重新计算中位数和标准差。
原始 schema 为 `heterollm.kernel-microbenchmark/v1`；必须声明
`source_kind=independent_synthetic_operator`、`target_llm_latency_used=false`、
`measurement_boundary=cuda_device_kernel_interval`。行包含 `m,n,k,device_durations_ns,hbm_bytes,hbm_durations_ns`。
HBM 数据缺失时后两项为 null；有数据时至少两次独立 timing。
该轻量导入核验数据一致性，**不替代**原有 `kernel_calibration.py` 的 CUPTI/SQLite、构建及数值正确性证据审核。

## 资源及 Attention

kernel 内部 load/unpack/scale/dot/reduce/SFU/dependency 分项输出。
不同资源并发取最大值；同一 scalar issue 资源上的工作相加，避免把共享执行单元错误当成独立流水。
HBM 使用 CTA 数、饱和 CTA 参数、事务大小、尾部利用率、kernel transaction efficiency 和独立读写服务；有同域带宽样本时可替换。解析带宽不是“实测 achieved bandwidth”。
MMQ conversion/main/fixup 继续由原有 lowering 分别计费，不将整个调用当成一个 CUDA kernel。

Attention descriptor 的 `operator="attention"`，prefill/decode 单独注册。
曲面 shape 对应 query rows/context tokens/query hidden width；实测 Attention 还必须绑定 KV width 和 score heads。
保留物理 KV 格式和元数据字节、GQA 共享存储以及 QK/softmax/PV 工作；Q4 MMA materialization 特例不被泛化模型覆盖。
当前仍不是精确的 FlashAttention CUDA 分支、causal tile 或 paged KV 地址映射模拟；未知布局不能声称已验证。

## Stateful L2

`stateful_l2=true` 时，planner 输出访问契约，`UnifiedEventKernel` 在 dispatch 时调用已有 line-LRU 状态机。
缓存属于一次实际 event-kernel run，跨 token/cohort 保留，不共享给其他仿真。
compiled 和 online 快速执行遇到动态 L2 会回退到同一事件实现，避免提前计算后重复使用旧缓存答案。
读写、partial-line、write allocate、dirty eviction 均沿用 `ExplicitCacheState`。

`buffer_accesses` 可提供完整 `buffer_id,offset_bytes,size_bytes,operation` 列表；必须精确覆盖该算子的读写字节。
weights 有物理 projection 身份时自动保持稳定 ID；其余显式 `input_buffer_id/output_buffer_id` 及 offsets 可保留别名。
KV、activation、RoPE table、norm weight、logits、sampling 都支持同一访问契约。
**没有可靠物理 buffer/page 身份的现有 lowering 仍以唯一流量污染 L2，不猜测重用命中。**
因此当前自动缓存覆盖不是全量地址重建；这比“矩阵小于 L2 就命中”保守，但不会造出虚假命中。

此第一版按整 kernel 保守串行 L2 访问，避免其他并发 kernel 看到尚未完成的 cache fill；不是 cycle-accurate cache。
line-level 成本随访问行数增长，大模型 trace 可能较慢。测量曲面与动态 L2 不能混用，除非以后有相同 cache protocol 的资源级标定。

## Launch、fusion 与多 GPU

`launch_ns/graph_launch_ns/launch_evidence` 区分普通提交与 graph replay。
开启 graph 标志不自动删除 launch：只有 runtime 给出单次 replay 的 `cuda_graph_id` 和
`cuda_graph_captured=true` 才能把多节点提交合并成一次；零成本节点显式依赖该提交。
`graph_launch_phases` 和 `apply_captured_graph_launch` 提供相同低层接口。
原有 RMSNorm/QKV/activation/FlashAttention fusion 继续负责实际 kernel 数，模型不凭名称额外合并。

`src/heterollm_sim/runtime_residual.py` 提供独立的 runtime residual 协议。
它要求直接 launch 与 CUDA Graph replay 的重复测量同时存在，并分别按
`N_kernel * ordinary_launch_ns` 与 `N_replay * graph_replay_ns` 计费；融合只
通过实际 kernel 节点数体现，不能从 native LLM TTFT/TPOT/E2E 反推。当前仓库
`tools/probe_runtime_residual.cu` 已在 RTX 5080/CC12.0 上完成独立 direct-vs-graph
探针（N=1/8/32 训练，N=16 保留 holdout，CUDA correctness 6840/6840）。导出的
ordinary 1162.5 ns/kernel 与 graph 1100 ns/replay 在 holdout 上分别达到
18.42%/24.14% APE，且 host CV 超过 10%；CUDA module SHA 也未完成验证。因此
证据状态为 `measured_rejected`，数值没有装入 production profile。
`load_runtime_residual_calibration` 对缺失、身份不符、不稳定或含 LLM 目标的输入
fail-closed；保留完整原始测量于 `artifacts/development/runtime_residual_20260930/`。

Attention 的 Blackwell profile 现在分别注册 `flash_attention_prefill_l2` 与
`paged_attention_decode_l2` analytical descriptors。它们按 phase、KV 物理格式、
因果 mask、GQA/head geometry 分派，保留 KV compulsory traffic；uniform
stream-K/fixup 仍由显式 descriptor 才会启用，非 uniform、paged/量化 KV 或未知
source specialization 回到 analytical fallback。`attention_causal_20260929`
和 `attention_streamk_20260929` 的主/fixup holdout 仅为诊断（M=64 causal fixup
APE 16.03%，且 probe runtime SHA 与 calibrated MMQ 不同），因此没有把 device
wall 或错误的资源列作为生产 Level-2 correction surface。
decode 的 paged/page-table Level-2 surface 尚无同运行时 trace，因此该校准面
保留解析回退；这不等价于禁止显式 dense contiguous descriptor 使用 Stream-K。

继续复用已有 topology/DES 中 compute、内存方向、NVLink 方向、PCIe、NIC、DMA 和 CPU submission 的独立资源及 owner alias。
有依赖时串行、无依赖且不共享资源才可重叠；没有增加 blanket sum/max。
新增资源必须绑定真实设备命名空间。`kernel_predictions` 汇总分派/模型/域外情况，不把服务时长之和冒充总 wall。

## 验证

`python -m pytest -q tests/test_kernel_model.py` 覆盖 dispatch、阶段、格式、occupancy、完整网格、域外回退、实测替换、Attention context、L2 reuse distance/dirty eviction、compiled parity、online 入口、graph launch 与多资源依赖。
合成样本只验证机制，不能用于产品精度宣传。真实准确率仍需独立微基准及未见 LLM 场景验证。

## 前端 llama 模式的解析预设

前端在 RTX5080 预设上选择 llama.cpp 模式会写入
`llama_cpp_kernel_model_preset=blackwell_analytical_v1`。Level 2 通过显式的
`blackwell_calibrated_analytical_v1` 预设启用，不覆盖未校准的前端默认路径。

## Level 2 calibrated analytical surface (2026-09-30)

`tools/build_level2_surfaces.py` 将独立的 synthetic CUDA main-kernel
测量生成 `src/heterollm_sim/kernel_level2_surfaces.py`。当前安装的是
`calibrated_analytical` correction surface：保留资源模型的 occupancy、HBM
需求和 overlap 语义，只用 `device_ns / analytical_ns` 修正同一个 kernel
phase；它不是 Level 3 measured surrogate，也没有使用 native LLM TTFT/TPOT/E2E。

校准绑定同时检查：RTX5080 hardware id、native CUDA DLL SHA、MMVQ source
dispatch signature、M/M-N-K shape、cold-streaming cache protocol、activation
logical dtype 和 F32 output。未覆盖 shape、缺少完整 joint cell、small-K
specialization 不匹配或 runtime/binary 不匹配时保持 analytical fallback。

最新宽 M=1 网格证据在
`artifacts/development/level2_shape_grid_20260930/`，holdout 判定在
`holdout_evaluation.json`。通过的 holdout 是 Q4_K、Q6_K 的中等 cell 与
IQ4_XS 的两个 cell；Q4/Q6 的大-N/K holdout 因完整 cell 未被测量而回退，
Q8_0 因重复性/holdout 门槛未接纳。correctness gate 失败的 shape 记录在
protocol 的 `excluded_cases` 中，未重试、未安装。该证据只证明窄的
synthetic operator domain，不证明整条 LLM execution path 已达到 10% 误差。
运行时据此创建显式 KernelModelProfile；用户已配置的 profile 不覆盖，其他 GPU 不套用。
量化 MMVQ/MMQ 的 M 阈值引用锁定源码。新增的独立 prefill MMQ 主 kernel 网格在
`artifacts/development/level2_mmq_prefill_exact_20260930/`，覆盖 M=512、Q4_K/Q6_K/IQ4_XS，
并分别覆盖 stream-K/fixup 与无 fixup 两个 dispatch signature；6 个留出点均通过
`CV<10%` 和 `APE<10%` 门槛。该网格只校准 `mul_mat_q` 主 kernel，activation
repack/quantize 与 `mul_mat_q_stream_k_fixup` 仍由独立 planner phase 计费，不把它们
混进主 kernel wall。Nsight Compute 还记录了该固定 specialization 的
255 registers/thread、58880 B shared allocation、8 warps/CTA；这些资源只绑定到
RTX5080 + 当前 `ggml-cuda.dll` SHA + MMQ signature，其他格式/运行时不套用。
MMVQ 旧域仍使用其原有资源/签名证据；未取得对应证据的参数继续是解析假设。该预设
只用于可观察的分析实验，不构成整条 LLM execution path 的精度保证。

资源校准后的 Level-2 service envelope 会保留 achieved bandwidth 作为审计字段，
但不会把独立 trace 的 DRAM transaction bytes 直接重算成模拟 HBM floor；否则 source
staging/padding/partial buffer 的逻辑字节差异会把已校准的 device wall 再次放大。
最终仍以 `device_ns / analytical_ns` 的同 phase correction 为主，资源 demand 只用于
解释与 DES 资源占用。
mixed-phase batching 仍保留 recurrent 资格门槛；单 slot 下报告 `single_slot_no_cross_request_phase_overlap`，避免把关闭标志误认为此次单请求耗时差异的原因。

### Level-2 prefill MMQ 验证结果（2026-09-30）

使用固定的 native-matched 六场景配置重新运行后，最新记录为
`artifacts/development/ui_native_matched_level2_mmq_20260930/comparison.json`。
相对原 analytical 基线，平均 APE 为：TTFT 24.5985%（基本不变）、TPOT
36.0785%、E2E 31.2146%，三项总体 30.6306%；原基线总体为 30.9068%。
这是当前候选版本的观测结果，不应解释为所有负载均改善；decode MMVQ、CPU
阶段和未覆盖 kernel 仍可能回退或受其他资源模型主导。

## 2026-09-29 独立测量与补缺

- 使用已有独立 GGML 合成 GEMM 可执行文件及 Nsight Systems/CUPTI，采集真实 device-kernel 区间、symbol、grid/block、registers/thread 和 shared memory。协议先于运行写入；不读取 GGUF/LLM 时延作为输入。
- 原始文件在 `artifacts/development/kernel_gap_20260929`，固定 M 的完整网格实验在 `kernel_gap_20260929_fixed_m`。训练/留出分离；不把 graph wall 当单 kernel，不把固定缓冲热缓存数据当 HBM 带宽。
- 观察到 Q4_K 寄存器随 M/K 变化，32 registers/thread 的预设不是设备事实。新增 `dispatch_signature` 可阻止跨实测 specialization 插值。未通过留出门槛的数据不安装为正式校准。
- 新增 `paged_buffer_accesses` 物理页表转换：offset、部分页、物理别名、allocation generation 均显式，复用页 ID 不得误命中旧 allocation。planner 支持该契约，但旧 KV allocator 到所有消费者的完整身份传递仍未闭合。
- Attention 的 KV width/score-head 现在参与 dispatch 筛选；新增显式 causal query positions 与 tile 工作量（区分有用 score 和 tile padding）。仅单请求单 slot 普通路径自动绑定位置，多请求与 MTP 不推断。
- Stateful L2 当前采用内存阶段排序，允许独立计算尾部重叠；仍未实现逐行并发 fill。Attention 已有下述 CUDA 微基准，但不代表全硬件/运行时覆盖。

## 2026-09-29 后续完成项与验收结果

- 新局部协议 `kernel_gap_local_20260929` 在读取留出答案前指定线性 N/K 插值，M=1，训练四角 N/K=2048/4096，独立留出 N=K=3072。Q4_K 主 kernel APE=4.34%，Q6_K=6.19%；IQ4_XS=14.39% 且 specialization 不同，拒绝。
- `local_surfaces.json` 已生成两份有限域诊断性能曲面，使用实测 registers/shared-memory、原始 SQLite hash 和训练点；没有擅自安装到 LLM 热/冷混合缓存场景。设备时钟/运行库等价性及 activation conversion 独立计费仍需闭合。
- 新增 `tools/probe_synthetic_attention.py`：直接调用既有 GGML DLL，无 GGUF 输入，以 NumPy 双精度 attention 为独立数值参考。首次 direct DLL 探测发现编译配置 `GGML_CUDA_FA=OFF` 并失败；改用已经存在的 hosttrace FA=ON 构建并记录不同库身份，没有覆盖原 DLL。
- Attention 两个场景均通过数值检查并取得 CUPTI：Q=1 的 vector 主 kernel 2.112 us + combine 1.408 us；Q=64 的 MMA 主 kernel 6.592 us + fixup 1.696 us。context=1024，D=64，Hq=4，Hkv=2，无 mask。不能把该无 mask/hot-cache 测量当成 causal/paged Attention 全域校准。
- L2 从整 kernel 排序改为内存阶段排序：填充在内存服务结束后可供后续访问，独立计算尾部可以重叠。并行 bank/逐行填充仍未建模，不声称 cycle-accurate。
- `DynamicKVPool.cache_accesses` 可从实际 live page IDs/owner 导出物理访问范围；prefix 使用原 page ID，释放重分配不会复用旧身份，迁移 owner 改变身份。多层非均匀 token 行布局拒绝推断。全量 serving consumer 的自动调用仍未闭合。

当前验收结论：有真实有限域校准和 Attention kernel 证据，不能宣称所有剩余项全部完成或所有硬件/运行时已验证。上述保留项不得改写为已完成。

### Level-2 剩余 surface 的最终判定（2026-09-30）

- MMQ activation repack 与 stream-K fixup 已从 `mul_mat_q` 主 kernel 中拆成独立
  stage contract。每个 stage 都带 source/runtime SHA、格式、J/fallback/stream-K
  dispatch signature、grid/block/register/shared 资源和独立 device-kernel 区间。
  证据和判定在 `artifacts/development/level2_mmq_stage_20260930/`；转换阶段因
  部分 M 点 CV≥10% 或没有完整联合 cell 继续 fail-closed；通过门槛的窄
  source signatures（包括 activation repack 与 stream-K fixup）单独记录，
  未把其余 `J=128` 或未覆盖形状外推到生产。
  planner 仍分别生成 conversion/main/fixup task，缺少精确 stage surface 时保留
  可解释 analytical service，不把主 kernel wall 偷算给 conversion/fixup。
- Prefill MMQ M sweep 已用独立 exact-Q8 输入和 NCU 完成 60 个 case：
  Q4_K/Q6_K/IQ4_XS 各覆盖 M=64/128/256/1024，N/K=2048/4096 四个训练角点，
  并对每个 M 测量 N=K=3072 holdout；M=512 既有主 kernel 网格保留为独立证据。
  `prefill_m_sweep/holdout_evaluation.json` 记录每个格式、M、dispatch/resource
  identity 和 APE/CV。M 轴没有完整角点或 signature/resource 改变时禁止插值，
  不满足 gate 的点回到 analytical fallback。
- Q5_K、Q8_0 使用独立四角 + 3072 holdout，不复用旧 evidence：Q5_K holdout
  CV=3.7673%、APE=1.6387%，Q8_0 CV=2.5280%、APE=3.4487%，数值正确性、
  runtime SHA、source/resource signature 均通过，因此两种格式已安装到窄的 decode
  MMVQ surface。Q5_0 没有完整合格 joint surface，仍在 manifest 中明确
  `not_installed_formats` 并 fail-closed；未知格式不会回退成无依据 INT8。
- 最终 generated surface 和 acceptance manifest 是由
  `tools/build_level2_surfaces.py` 重建，当前 manifest 的 `sample_count=139`，
  `not_installed_formats=["q5_0"]`，格式 holdout、M-sweep holdout 与 MMQ stage
  holdout 的路径均被写入 `artifacts/development/kernel_level2_surface_manifest.json`。
  M-sweep 主 kernel holdout 的最大 APE 为 Q4_K 3.46%、Q6_K 4.43%、IQ4_XS
  3.14%；这些误差来自独立 analytical-ratio surface，而不是 native LLM 时延。


### 独立 causal Attention 留出与身份修复

`attention_holdout_20260929/protocol.json` 在运行前固定：训练 L=1024/4096，
留出 L=2048，M=1/64，线性插值，4 次预热及 20 次正式测量。六个场景均通过独立
数值参考；每次 trace 恰为 24 组主 kernel 加 combine/fixup。采集进程现在枚举实际
加载的 GGML/CUDA DLL 路径及 SHA256，并检查测量前后稳定性。

- M=1：逐阶段 APE 6.035%/3.086%，服务之和 APE 2.359%，通过有限诊断域门槛。
- M=64：逐阶段 APE 0.495%/16.034%，服务之和 APE 4.119%，**未通过**逐阶段 10% 门槛。
- 服务之和不是端到端 wall，未包含 kernel 间空隙、host submission。以上均为
  hot-same-buffer 合成实验，未获正式 LLM 冷/热混合场景迁移资格。
- `KernelCapability.attention_mask` 可区分 `none` 和 `causal` 分派；未指定保持通用
  解析兼容。不会因为新增分派字段就启用 causal 性能曲面。
- KV 容量预检现在在独立副本执行同一放置算法，不再改变真实页/前缀身份；实际迁移
  更新分配代次（包括迁回原设备），独立 pool 缓存命名空间隔离，共享 prefix 保留别名。

### Stream-K 后处理误差的源码解释与新留出

上一实验 M=64 fixup 的 16.03% 误差不能用总时长误差掩盖。锁定源码
`fattn-common.cuh` 的 stream-K 调度先将主 block 数限制在 occupancy×SM 数以内，
再按输出 tile 的整数倍取整（最多允许 5% 效率损失）。uniform fixup 每个输出
循环合并 `blocks_per_tile - 1` 个结果，故其工作量并不始终正比于 context。

已有 trace 的 L=1024/2048/4096 主 gridX 为 64/128/168；后一段已饱和。
新协议 `attention_streamk_20260929/protocol.json` 在执行前声明只测新 L=1536，
以原 L=1024/4096 为训练端点，fixup 按源码合并次数线性插值，主阶段仍按 context。
原 L=2048 不再作为新方法的独立留出。

新测量数值、DLL 稳定性、kernel 次序、完整 24 对调用均通过；源码预测主 gridX=96
与实测相同。新独立留出主阶段 APE=4.402%，fixup APE=1.049%。评估器
`tools/evaluate_attention_streamk_holdout.py` 可重放该判定，并拒绝 shape、mask、cache、
module、specialization、grid 或校准域不匹配。5 个调度算术测试通过。

168 active blocks 目前取自该 GPU/特化的既有 trace 饱和值，而非跨设备常数。
该结果仍是诊断级证据；尚未将这套多 kernel 分阶段模型接入正式 Attention costing，
也未覆盖非均匀 stream-K、paged KV、运行时迁移及端到端精度验证。

### Stream-K 已接入解析 costing（显式启用）

`KernelCapability.attention_stream_k=true` 现在通过公开
`estimate_gpu_fused_attention` 产生主阶段及独立 uniform fixup 阶段，分别计启动开销。
现有 planner 的 phase 依赖链负责顺序，DES 测试验证最终耗时等于这些阶段之和。
主 CTA 数由共享的源码调度函数确定；独立评估器也调用该函数，避免两份算法漂移。
fixup 使用独立 register/occupancy，scalar、SFU、memory 需求；临时缓冲流量暂用
分配大小上界而非冒充精确 DRAM 事务。phase 级 metadata 进入 L2 计费与覆盖统计，
fixup 不再被当作读取持久 KV 的阶段重定向到远端内存。

当前仅显式配置的 dense FP16 KV、FP32 输出、head_dim=64、uniform partition
采用此解析路径；其他情况回退。整 kernel 实测样本不能与该分解混用。
没有自动安装到前端预设，也未把独立测量的 1.05% fixup 插值误差当作此解析模型误差。
后续仍需证明临时流量、阶段校准、runtime 适用性及实际前端工作负载覆盖。

### Uniform fixup 临时流量修正

已用 `fattn-mma-f16.cuh` 的 `needs_fixup/is_fixup` 写入分支与
`fattn-common.cuh` uniform 合并循环替换上一版分配容量上界：设主 block 数 B、
输出 tile 数 T、每 tile 列数 C、head_dim D，则部分结果写入为
`(B-T)*C*4D`，metadata 写入为 `B*C*8`；分配容量 `B*C*(16+4D)`
单独保留，不作为访问字节。fixup 仅访问有效 query/head 行，读取每行
`(B/T-1)*4D + (B/T)*8` 的 scratch，并读写最终输出。

这些是唯一逻辑地址字节，既不是所有 lane 重复 load 的指令字节，也不是已测 HBM
事务量；后两者仍依赖 cache/transaction 模型。后处理 achieved bandwidth 现在由自身
CTA 数及读写量估算，不再沿用主阶段带宽。回归覆盖完整 tile、padding、无拆分、
非法非均匀分割，以及公开 costing 中实际 memory demand 的字节数。

### 主阶段到 fixup 的自动缓冲身份

本地 KV、完整 query tile 的 uniform Stream-K 路径在 planner 中自动生成调用级
scratch/output 身份，分别定位 CUDA 的两个 metadata bank 和部分结果区。
主阶段写入与 fixup 读取共享相同 buffer/offset；不同调用使用不同身份。
访问范围总量必须等于阶段成本中的 read/write，错误不静默退化。

测试分别验证：真实 planner 自动产生契约、阶段 kernel_prediction 指向正确 kernel，
以及 DES+LRU 下缓存足够时 fixup 无 HBM 读、改成另一次调用身份后产生冷读。
43 项相关回归通过。此绑定尚不覆盖 padding、远端 KV 或所有在线模板重绑定路径；
这些场景继续保守处理，不将缺少物理布局的输入推断为命中。KV allocator 到通用
Attention consumer 的完整 page 身份传递仍未完成。

### 尾部 query tile 的物理范围

uniform Stream-K 本地缓冲绑定现已覆盖不足一个 query tile 的尾部。调度元数据携带
query_tokens/query_tile/heads_per_tile，地址生成沿 CUDA 的 query tile 优先顺序。
主阶段仍写 padded stride，fixup 对每个 partial block 单独读取有效列，不能把多个
partial 的尾部行错误压成一个连续区间。覆盖 M=1/31/33/63/65 的地址包含性与字节
核对，并通过实际 planner 验证 M=33 自动绑定。非均匀 stream-K 分割仍明确回退，
没有因 padding 支持而放开尚未实现的 general fixup。47 项相关测试通过。

### 在线 replay 临时身份隔离

在线任务此前会更新 task_id 命名空间但保留 L2 临时 buffer_id，重复使用同一模板
存在虚假跨批命中的风险。编译器临时身份现在用 `@invocation:` 标记，并在 L2 契约
显式声明 invocation_buffers。普通 stage、缓存 layout 和 trusted layout 三条实例化
路径均按在线 replay namespace 重绑定这些临时身份；持久 weights/KV 不重命名。
只复制需要改动的契约，不修改模板。主阶段与 fixup 在同次 replay 内继续共享身份。

三条在线实例化路径均通过持久 DES 测试：同批 reader 命中，下一批 writer 冷访问，
原模板保持不变。模型测试 41 项通过，MMQ runtime/KV layout 回归 6 项通过。
这闭合已声明临时缓冲的在线实例化隔离，不代表通用 KV page 自动绑定已完成。

### 校准域诊断与导入边界

微基准原始样本导入现在保留可选 `dispatch_signature`，避免实测 block/register
特化标识在导入时丢失，从而错误允许跨 kernel 特化插值。旧无标识 schema 仍可读取，
但不会因此获得已验证的来源声明。回归通过导入两个不同特化样本，确认中间 shape
被 `kernel_specialization_boundary` 拒绝。

`distance_to_calibration_domain` 与 `nearest_sample_distance` 现分别报告：前者为
log2 shape 空间到校准包围盒的距离，后者为到最近实测点的距离。缺角或跨特化时
联合有效域未被证明，域距离返回 null 并保留具体拒绝原因，不伪装成域内可插值。

### 非法 L2 访问批次不再污染执行状态

`ExplicitCacheState.access_many` 现在先完整验证类型、已有/新建 buffer 大小一致性和
所有范围，再修改缓存行、LRU 和统计。生成器先物化，后续非法访问不能让前序合法
访问提前驱逐脏行。L2 首次状态仅在访问批次验证成功后注册。

DES 在 L2 契约验证失败时恢复被取出的 global ready entry，任务不会在仍标记 active
的情况下丢失调度入口。回归覆盖脏行保留、大小冲突、生成器/错误类型及重复失败后
ready queue 可见性。59 项缓存/成本模型测试通过。此保证针对访问契约验证失败，
不宣称任意系统错误或全执行流程均具有事务回滚能力。

### Stream-K 计算量使用实际 descriptor tile

修复主 CTA/scratch 已按 descriptor 分块，但 tensor/scalar 工作量仍按通用 workload
query_tile 计算的不一致。Stream-K 现在用所选 kernel 的 query/KV tile 计算 causal
执行对数，无 mask 路径同样计入尾部 padding。generic workload 的默认 tile 不再
偷偷改变该 kernel 的 service。测试用 M=1、generic tile=128/16、kernel tile=32
验证有/无 causal mask 均使用 32×L 执行对数（有用对数仍是 L）。50 项相关测试通过。

### MMVQ 源码几何参与占用率

携带 `MMVQWork` 的 GEMM 现在使用其源码 warps_per_cta 计算 occupancy，而非继续
采用通用 descriptor 的 warp 数。shared-memory 取 descriptor 声明与源码 reduction
数组下界的较大值；不会把源码数组大小冒充最终 cubin 静态分配，也不覆盖更大的
显式测量值。registers/thread 仍来自 descriptor，并在 residency_source 中明确说明。
回归以故意不一致的 8-warp descriptor 对比 M=8 实际 2-warp 路径，验证公开 costing
使用源码值；同时检查较大 shared 声明保留。114 项 kernel/MMVQ 相关测试通过。

### 全量回归与真实模型报告恢复

全量运行完成：2945 passed、12 failed、4 skipped（822.56 s）；失败集合与历史记录
同名，不将此结果宣称为全绿。该运行启动后新增的修改由各自局部回归验证。

修复 `tools/qwen38_memory_scenario.load_model`：旧 sidecar 不合格时用严格 GGUF
解析及 payload hash，保留 sidecar 原文件，并在身份中记录 metadata_source、
sidecar_rejection、gguf_payload_rehashed。真实 Qwen 文件重读成功后，报告仍因旧
资源名断言失败；已验证现在 HBF 控制器为 hbf0.controller，写向链路有 720896 bytes，
因而回归改查实际 controller 及 GPU→HBF 方向，保留原容量/权重/KV/state 断言。
该真实模型报告及 sidecar 不覆盖回归共 2 项通过。其余全量失败尚未全部解决。

### 热工作点与校准适用域

GPU 频率、缓存带宽或延迟工作点改变后，不再沿用原 kernel device-wall samples；
保留 descriptor 并清除曲面，evidence 标注失效原因，退回解析。未改变工作点时保留
原曲面，原 profile 不被修改。尚未覆盖独立 HBM 域变化自动失效 GPU 曲面的跨组件情形。

修正两项旧 thermal 回归的测试前提：参考 HBM 现在是 aggregate banks，而旧断言
假设每个 bank 独立。独立域测试显式声明单 bank 容量/带宽；另加 aggregate 测试
检查一个 bank 降频后，其他 bank 的本地字段不变、共享物理上限按成员变化重绑定。
不是删除原隔离要求。thermal 与 kernel 回归合计 50 项通过。

### 跨组件热工作点失效传播

`apply_thermal_operating_point` 现在比较重绑定后的 memory_service 与链路；仅内存域
或链路域变化也会撤销 GPU kernel 实测曲面，而不要求 GPU 频率变化。当前曲面尚无
完整的内存/路由依赖集合，因此采用明确记录的 all_gpu_profiles 保守失效范围，不能
声称只影响精确关联 GPU。模型描述与原始基线保留，不缩放样本制造新测量。

测试覆盖 memory-only、link-only、GPU 时钟不变、公开 costing 回退及无实际参数
变化时曲面保留。thermal/kernel 联合 52 项通过；补强 identity 断言后 thermal 6 项通过。
这只覆盖热工作点 API；任意用户直接编辑其他配置后的全局适用性证明仍未闭合。

### 已知回归收敛与前端重新对比

最后一项 UI 静态失败来自旧 HBF 24000 Gb/s、official-reference 来源和仅 UCIe 的
断言；测试已对齐当前用户配置的读3904/写217.6 Gb/s、analytical_user_configured、
HBF/UCIe 连接规则，未改变前端默认值。UI/规格/API/算子映射/thermal 57 项通过，
llama 前端 Node 测试 3 项通过；缓存/kernel/分页/MMQ/真实报告补充组 75 项通过。
原全量12项失败均已在各自定向运行通过，但没有重跑整套，不能宣称全量绿。

`ui_llama_kernel_r5_20260929` 新启动完整六场景前端重跑，协议和源码 hash 单独保存，
不覆盖 R5/native/前轮数据。复用已验证的完整负载一致性检查与实际 HTTP
normalize/simulate-score 入口。Qwen2.5 首个完成结果与前轮相同，说明新增修复不代表
默认场景已改善；其余结果以该目录的实时 comparison.json 为准。

### llama 前端解析预设接入 GET_ROWS

`blackwell_analytical_v1` runtime 预设现在读取本仓库锁定 llama 源码，派生并应用已有
GET_ROWS 存储契约（仅当用户未给 SOURCE_KEY）。GGUF 形状/类型/容量/架构和任务
资格继续由原 qualification 检查；缺源码或不认识的源码规则明确 unsupported，不
偷偷启用。此处只启用 selected-row gather，未启用全局 F32 hidden dtype 开关；
整表分配和跨设备 staging 仍保留，row conversion 时间仍标为 partial/unpriced。

117 项 kernel/GET_ROWS/mixed batching/hardware 回归通过。真实六场景 baseline
重跑完成，结果与前一轮一致；新目录 `ui_llama_gather_r5_20260929` 正在通过相同
前端/API/完整负载检查运行接入后的六场景，不覆盖旧数据，不按 Native 总延迟调参。
结果必须结合 conversion 未定价、放置不相同的限制解释，不能因某个 APE 下降便
宣称高精度目标已完成。

### CPU GET_ROWS 单任务与独立微基准

锁定 `ggml-cpu.c::ggml_get_n_tasks` 将 GET_ROWS 固定为一个任务；现在派生源码契约
时读取该规则（忽略注释），将 cpu_get_rows_tasks=1 经 gather、MemoryWorkload 传入
CPU instruction schedule。缺少调度源的旧契约不推断单任务，改变后的源码规则拒绝。
这只修正已有内存 load/store 的核心并行度，量化转换运算仍未完整计费。

`tools/probe_synthetic_get_rows.py` 用直接 GGML CPU backend 执行合成行，不读取 GGUF。
Q4_K/Q6_K 直接构造有效 packed block，NumPy 独立解码验证选中行、乱序和重复索引。
预声明15场景（两行数×两宽度训练、另8×2048留出），8次预热40次正式测量，全部数值
检查通过。实际加载 GGML 模块哈希前后稳定。计时为同步 backend graph wall，包含
调度，不能作为单独反量化吞吐率。Q4_K/Q6_K/F16 线性插值留出误差分别2.413%、
28.229%、27.298%，最大CV分别约40.5%、21.2%、23.0%，全部不具正式部署资格。
证据在 cpu_get_rows_20260929；未安装到默认计费。探针解码手算样例与GET_ROWS/kernel
联合84项通过，GET_ROWS/cost_models 87项通过（两组重叠，非相加）。

六场景 GET_ROWS 前端重跑及 comparison.md 已完成；改善和恶化均保留。它是在本次
单任务修正前完成，不可把其结果归因于本次修改，也不能宣称端到端精度达标。


## RTX 5080 parameter evidence

Checked 2026-09-29. Do not substitute datacenter Blackwell (SM100) limits for RTX 5080 (SM120).

Sources:
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html (Occupancy): currently states CC12.0 48 warps, 64K registers, 32 blocks, 128KB shared memory per SM.
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__TYPES.html documents device attribute IDs.
- Local NVIDIA driver `cuDeviceGetAttribute`, device 0, CC12.0: attribute 39=1536 resident threads, 81=102400 shared bytes, 82=65536 registers, 106=24 resident blocks; all calls returned CUDA_SUCCESS. Query is hardware inspection, not a timing benchmark.

The online guide conflicts with the local device for block/shared limits. This local RTX preset uses the device-reported limits (24 blocks, 100KiB), not the generic guide's values and not the former inherited defaults (16 blocks, 64KiB). Shared capacity is an upper limit, not proof of a particular kernel's carveout or allocation granularity.

Registers/thread=32, descriptor shared/CTA=0, efficiency=.65, transaction efficiency=.8, DP4A scalar-op rate=128 and reduction work=5 are STILL UNVERIFIED legacy assumptions. They are not NVIDIA specifications. `unverified_parameters` now exposes these and other unbound performance inputs in kernel audit output. Existing values remain for compatibility; this change does NOT claim to have eliminated assumptions or improved accuracy. A public architecture table cannot supply per-specialization register allocation or achieved bandwidth. Obtain registers/static shared from the exact native binary (cuobjdump / function attributes), dynamic shared and launch geometry from matching source/callsite, and effective rates/overlap from independent benchmarks. No cuobjdump executable was available at the expected local CUDA location during this check. Do not substitute another build's register count or replace efficiencies with 1.0 and call them measured.

The frozen native-matched comparison artifacts remain unchanged. New runs must retain their deployment/workload and explicitly record these model-limit changes.


## Exact native CUDA resource extraction (2026-09-29)

The native-matched binary `source/llama.cpp-native-thread-control/build-native-thread-control/bin/ggml-cuda.dll` was inspected with `E:/cuda/bin/cuobjdump.exe --dump-resource-usage`. The binary contains `sm_120a` fatbin code. Artifact: `artifacts/development/cuda_resource_usage_sm120.json`; tool output has 6,717 function records and binds the DLL SHA256. For templated quantized `mul_mat_vec_q`, type 12 (Q4_K in this source) has 46–192 registers/thread and 1,408–4,096 shared bytes across specializations; the variation is real and depends on c_ncols/flags. Other type IDs likewise vary. This replaces the old blanket claim that the native kernel uses 32 registers/thread, but it is not yet safe to install a single number in `KernelCapability`: runtime dispatch must bind the mangled specialization (format, output columns, fusion, small_k/halve_iters) to the matching workload.

Resource usage is not achieved throughput. No efficiency or bandwidth parameter was changed from this extraction. `cuobjdump` cannot provide wall-clock throughput, cache behavior, or effective HBM bandwidth.


## Native binding correction and independent measurements

The previous column-only Q4_K resource table was erroneous (some entries came from a different type) and did not bind fusion/small-K/halve-iters or the binary. It has been removed from the preset and nonempty column-only tables are rejected. No production calibrated surface has been installed.

`artifacts/development/native_kernel_binding_20260929` contains frozen protocols and 18 synthetic cases on the exact historical native CUDA DLL: Q4_K/Q6_K/IQ4_XS, M=1/2/4/8/64 at N=K=4096 plus held-out M=1,N=K=3072. All numerical/module checks passed. Nsight Systems records actual symbols, launch geometry, registers and shared fields with 20 formal samples. Held-out measurements are reserved observations, not yet an accuracy-validation pass.

Independent Nsight Compute cold-cache (`cache-control=all`, clocks unmodified) DRAM counters, three main-kernel invocations each, give median bytes/device-time: Q4_K 738.28 GB/s, Q6_K 791.28 GB/s, IQ4_XS 768.22 GB/s at M=1,N=K=4096. These are profiler-conditioned device DRAM counters, not target LLM timing fits and not automatically transferable rates. Hot Nsight Systems times are respectively 7.136/8.448/6.112 us; cold profiler times 12.800/17.408/11.616 us. Never divide hot-cache time into cold DRAM bytes.

A critical unresolved tool-field discrepancy: Q4_K M1 cuobjdump SHARED=1408 whereas CUPTI staticSharedMemory=384 and dynamicSharedMemory=0 (registers agree at 53). Do not install either field as equivalent total occupancy allocation without resolving the accounting difference. Runtime resources vary with specialization: Q4_K M2 registers=80 and M4=120; prior reported 70/91 were incorrect. No claim of improved LLM prediction accuracy is made by this collection.


## Resolved shared accounting and narrow resource binding

`native_kernel_binding_20260929/query_occupancy.py` loads the exact extracted native MMVQ cubin and calls CUDA driver function attributes and occupancy APIs. Device attribute 111 reports 1024 driver-reserved shared bytes per block. All 13 observed MMVQ cases satisfy cuobjdump SHARED = cuFuncGetAttribute static shared + 1024; CUPTI agrees with function static attributes. Thus the prior discrepancy is resolved for these observations, not guessed. Registers, block size and symbols also agree. Q4_K M1/M2/M4 resident blocks are 9/6/4, Q6_K 9/8/6, IQ4_XS 10/6/4 (4096-square). Official API occupancy includes allocation constraints that the simple division may miss.

`native_kernel_resources.py` now provides narrow exact-observation bindings, requiring native DLL SHA, RTX5080 hardware ID/limits, format, M/N/K, small-K flag and source-derived block/grid. Unknown shapes/binaries retain the analytical path; no interpolation or use of target-model timing. This is not broad resource coverage. API occupancy evidence overrides the simple analytical resident count only on a match. Existing native-matched scenario files are not modified.

Cold-cache transfer check frozen before collection: predict 3072-square M1 main-kernel time from the prior 4096-square time using N*K scaling; APE<=10% gate, no refit. Q4_K 13.46% (fail), Q6_K 8.38% (pass for this check only), IQ4_XS 15.97% (fail). These results reject a blanket constant-bandwidth replacement. Clocks were unmodified, only three profiled main-kernel samples each; no production performance curves or claimed global 10% accuracy.


## Specialization coverage repair

Resource lookup now re-derives and validates the source work contract, then matches DLL SHA, format, template M, small-K/unfused/halve-iters flags and block dimensions. Runtime N/K no longer falsely restrict a compiled function's resource allocation. Ambiguous resource rows fail closed. No timing-surface extrapolation is introduced.

IQ4_XS source geometry is explicitly opt-in for the Blackwell kernel preset: QK=256, QI=32, VDR=4, block bytes=136, max M=8. Existing source/runtime gates remain. The unrelated DP4A issue-bound model excludes IQ4_XS rather than inventing an instruction rate.

Full six-cell diagnostic replay: native_resource_coverage_fixed_20260929. Phase-tagged cost-call native resource bindings: Qwen2.5=3, SmolLM2=8, TinyLlama=8, Qwen3.5=4, Qwen3.8 CPU=0, Qwen3.8 GPU=8. Counts include planning/cache costing and are NOT executed kernel counts. All 18 latency outputs remain identical to the prior run despite nonzero resource binding. Further bottleneck attribution is required; no accuracy improvement claimed.


## Bound-kernel bottleneck diagnostics (2026-09-30)

Kernel audit output now includes `resource_service_ns`, `resource_bottleneck`, and `analytical_service_ns`. This distinguishes a real native-resource occupancy change from a no-op when HBM remains the critical path. It is diagnostic only and does not alter timing formulas or fit target LLM results. Full six-cell replay already showed nonzero exact bindings but unchanged E2E; the next comparison should aggregate these fields by phase/format before any further model change.


## Cold-cache shape-surface holdout (2026-09-30)

A new bounded synthetic protocol measured M=1, K=4096 at N=2048 and N=4096 as training points and N=3072 as an untouched holdout for Q4_K, Q6_K and IQ4_XS. Nsight Compute used cache-control=all and clock-control=none; 6 main launches were captured per case, 3 stable formal samples retained. The original Q4_K K=8192 probe failed the harness correctness gate and was explicitly excluded in `protocol.json`.

Linear N interpolation passed the predeclared 10% holdout gate for all three formats: Q4_K 0.314%, Q6_K 2.644%, IQ4_XS 0.342%. Formal CVs were below 10%. Predicted vs observed cold main-kernel time: Q4_K 10144 vs 10176 ns, Q6_K 13664 vs 13312 ns, IQ4_XS 9312 vs 9344 ns. Predicted/observed DRAM rates remain separate (e.g. Q4_K 683.9 vs 696.8 GB/s); no bandwidth was silently substituted.

This is an accepted *synthetic operator-level holdout* for this narrow M=1,K=4096,N-domain and cold-cache protocol, not an LLM accuracy proof. It is not installed in the production profile because the native LLM path has different conversion ownership, cache reuse, launch batching, and runtime state. The surface artifact is `artifacts/development/cold_surface_20260930/holdout_result.json`.


## Interaction-axis acceptance matrix (2026-09-29)

The M=2/4 N-axis holdouts and M=1 K-axis holdouts all passed the 10% error gate for Q4_K, Q6_K and IQ4_XS. The interaction check fixed N=4096 and tested K interpolation at M=2/4: Q6_K and IQ4_XS passed, but Q4_K failed at K=3072 (M2 14.31%, M4 11.22%). This rejects separable or blanket Q4_K M>1 K interpolation. The machine-readable policy is `artifacts/development/cold_surface_acceptance_20260929.json`; no synthetic surface is installed in the LLM production profile.


## Q4_K M×K interaction repeatability

The Q4_K M=2/4 K=2560 and K=3072 anomaly was repeated in two independent Nsight Compute processes. Between-process median variation stayed below 1.08% for all four shapes, while the K=2560 point remains much slower than a straight interpolation from K=2048 to K=3072. This is a stable shape/transaction effect under the frozen cold-cache protocol, not a one-run outlier. The exact kernel symbol and block geometry remain unchanged, so the current evidence points to memory transaction/latency behavior inside one specialization. Q4_K M>1 K interpolation remains rejected; no corrective multiplier has been fitted.


## Explicit fallback behavior

A kernel descriptor miss returns to `_estimate_gpu_gemm_analytical` (or the attention analytical path), which retains declared physical bytes and any existing source/capability work but has no measured accuracy guarantee for unseen formats. Fallback audit now identifies `legacy_analytical`, the format coverage, the formats absent from the current kernel profile, supported profile formats, and the dispatch candidate. It distinguishes unsupported quantization, mixed-format unresolved dispatch, known formats outside the descriptor domain, and an unavailable kernel profile. No timing equation changes were made by this audit metadata update. A format that cannot be parsed by the GGUF importer or represented in the typed workload may fail earlier; fallback cannot make an unsupported artifact executable.
\n### Level-2 grid acceptance note (2026-09-30)\n\nThe later Level-2 grid contains Q5_K/Q5_0 exact-shape observations. Only rows with formal CV below 10% are installed; Q8_0 remains measurement evidence only because its holdout repeatability gate failed. Correctness failures and high-CV rows are retained in the protocol/manifest and are not retried or silently used for prediction.\n
