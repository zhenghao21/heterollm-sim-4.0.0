# R19 GPU 成本机制诊断

日期：2026-09-16  
范围：仅限锁定 `llama.cpp-semantic` 源码、现有静态公式和 R16 的已拒绝合成算子诊断。未读取 LLM actual，未运行 GPU、仿真、原生推理或基准；没有修改核心代码。

## 结论

R16 的两个方向相反的 main 偏差有不同的、可由源码和现有公式直接指出的候选机制。

1. **MMVQ M=1 的 main 高估存在明确的算子类别和并行度错配。** 锁定源码是每线程量化向量点积：Q5_0 路径先做位域重构与 `ggml_cuda_dp4a`，再做浮点尺度/偏置组合；它不是 tensor-core MMA GEMM。见 `source/llama.cpp-semantic/ggml/src/ggml-cuda/vecdotq.cuh:175-200`、`811-828` 与 `mmvq.cu:38-65,692-739,741-819`。但 `estimate_gpu_gemm` 对没有 `mmq_work` 的量化 GEMM 仍按 `mma_output_tile_wave_proxy` 的 16×16×16 MMA tile，使用 `SM × tensor_cores_per_sm × 固定 occupancy` 作为并行槽，并把该利用率直接乘到 HBM 带宽。见 `src/heterollm_sim/cost_models.py:2347-2404,2496-2513,2589-2639`。这正是 R16 所称的 MMVQ M=1 memory-bound main 高估的结构性候选解释，而不是可直接写入系数的证据。

2. **MMVQ 的源码几何已经被单独派生，但尚未接入成本模型。** `mmvq_work.py` 只产生源代码网格、block、K-loop、逻辑字节和共享归约数组；其 metadata 明确写入 `cost_model_applied=false`，并要求在独立带宽/占用率验证后才使用。见 `src/heterollm_sim/mmvq_work.py:1-6,79-159,162-193`。仓库中没有该 `MMVQWork` 的 planner/cost-model 消费点。因此目前模型无法把 MMVQ 的实际 CTA/warp/K-loop 替换进 generic MMA wave 代理。

3. **MMQ M=64 的 main/fixup 低估不能由“把 MMQ 改成 float MMA”解决。** planner 只有在 CC1200、固定后端、`compiled_int8_mma=true` 等条件下才启用 MMQ，并且成本模型将其 internal dtype 显式设为 `int8`。见 `src/heterollm_sim/planner.py:9177-9208` 与 `src/heterollm_sim/cost_models.py:2440-2508`。锁定 Q5_0 MMQ 的 MMA 分支也选择 `..._mma` vector-dot/write-back 函数；Blackwell 配置文件对该格式回退到 Ampere MMQ 配置。见 `source/llama.cpp-semantic/ggml/src/ggml-cuda/mmq.cuh:736-740`、`source/llama.cpp-semantic/ggml/src/ggml-cuda/mmq-config-blackwell.cuh:36`、`mmq-config-ampere.cuh:70-72`。因此当前 int8 peak 的大类别是合理起点；缺口是同一 CTA 内的量化解包、整数/尺度运算、共享内存交换、同步、写回和实际 launch/驻留配置没有获得独立资源速率。

4. **MMQ 已有正确方向的部分源码工作，但仍非完整 kernel 资源模型。** `MMQWork` 计算完整 256-value K iteration、Q8_1 consumer 高水位、source allocation、stream-K 边界、partial writer 和 fixup 元素；成本路径把 consumer bytes 与 main partial write 加进静态内存映射。见 `src/heterollm_sim/mmq_work.py:120-173,299-374`、`src/heterollm_sim/planner.py:9795-9808`、`src/heterollm_sim/cost_models.py:2421-2449,2618-2643`。但主 kernel 实际有 `__launch_bounds__(nthreads, occupancy)`、动态 shared memory、源码配置选择、stream-K 以 tile 效率选择 block 数，并按需另启 fixup kernel。见 `source/llama.cpp-semantic/ggml/src/ggml-cuda/mmq.cuh:951-991,1388-1474`。当前成本仍把 CTA 可并发性简化为通用 MMA wave 和固定 profile occupancy，且没有对 Q5 解包、整数 dot 旁路、float scale/write-back、barrier 或 CTA 调度建立来源限定的服务率；这些均可解释 R16 M64 main/fixup 同向低估，但不足以给出数值修正。

R16 的 rejected operator observations 只用于方向诊断：MMVQ M1/Q5_0 main 高估约 38%–170%，而 M64/Q5_0 MMQ main 与 fixup 低估约 62% 与 71%。它们不能用于拟合或绕过质量门。见 `round_016/DEVICE_GAP_ANALYSIS.md:65-81`。

## 具体形状：仅由静态源码公式得到

### Q5_0 MMVQ，M=1、K=896

`MMVQWork` 的固定 CC1200/Q5_0 域使用 `qk=32`、`qi=4`、`vdr=2`、四个 warp。K 有 28 个量化 block；M=1 的 small-K 判定为真，因此每 CTA 有四行，网格为 `(N/4, 1, 1)`，block 为 `(32, 4, 1)`。见 `src/heterollm_sim/mmvq_work.py:166-193`。

| N | 源码派生 CTA | launched warps | Q5 logical weight bytes | Q8_1 consumer bytes | F32 output bytes |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 128 | 78,848 | 1,008 | 512 |
| 896 | 224 | 896 | 551,936 | 1,008 | 3,584 |
| 4,864 | 1,216 | 4,864 | 2,996,224 | 1,008 | 19,456 |

最后一行的权重字节为 `4864 × (896/32) × 22 = 2,996,224`；公式见 `MMVQWork.logical_weight_bytes`（`mmvq_work.py:119-132`）。三行展示的关键是：源码同时发射 128 或 896 个 warp，而通用 MMA 代理在 M=1 时只数 `ceil(1/16) × ceil(N/16)` 个输出 tile，并把这与 tensor-core slot 直接等同。该代理的固定 profile 值是 84 SM、4 tensor cores/SM、occupancy 0.85。见 `src/heterollm_sim/reference.py:364-421`。这两种并行单位没有源码等价关系。

MMVQ main 还包含 `tmp_shared` 写入、CTA barrier、跨 warp 合并以及 warp shuffle reduction；普通 unfused 路径不含 GLU SFU，但含浮点尺度和 reduction。见 `mmvq.cu:741-819`。所以不能用“纯 int8 peak”或“纯 float MMA peak”覆盖该路径。

### Q5_0 MMQ，M=64、K=896、N=896

给定已绑定的 84 SM 与充足 shared-memory 合同，`MMQWork` 的固定公式将 K 扩展为 4 个 256-value iteration，即 `k_execution=1024`；它按 logical Q5 block 而非 `ceil(K/256)` 来做 stream-K 边界。见 `src/heterollm_sim/mmq_work.py:313-346`。该形状有七个 128-wide output tile；当 tile 利用率低于 90% 时，源码派生模型取 SM count 作为 stream-K block count，因此会进入 partial/fixup 路径。实际 kernel 也以 `ntiles_dst`、SM 数和 90% 条件选择 tile/stream-K block 数，并在余数存在时分配 partial buffer、发射独立 fixup kernel。见 `mmq.cuh:1439-1474`。

这是主成本和 fixup 均不应只靠通用 16×16×16 MMA 波次的原因。当前代码保存了 partial writer/valid element 字节，但没有从源码或 cubin 获得该形状的实际 `J`、`nthreads`、`__launch_bounds__` 驻留数、共享内存银行/事务或整数/scale 指令速率。

## 已确认的字节与未确认的字节

- MMVQ 主 kernel 读取 Q5_0 packed weight：每 32 元素 22 bytes；消费 Q8_1 为 `36 × M × K / 32`。来源限定公式见 `mmvq_work.py:119-132`。对于 M=1、K=896，Q8_1 consumer 为 1,008 bytes；conversion 是独立 kernel，读取 3,584 bytes F32 输入并按其自身 padded-K 规则写入，不能把两者混为主 kernel HBM 需求。planner 在 conversion 后将 main 的 activation storage 替换为 consumer bytes。见 `planner.py:9756-9798`。
- MMQ 转换读 `4MK`，写 `144 × M × K_padded / 128`；main consumer 使用由 source high-water 推出的 `consumer_unique_bytes`。见 `mmq_work.py:86-131,319-331`。M64/K896 的 conversion 需要 K padded 到 1024，main 的 source 仍报告 repeated load work；但 metadata 明确禁止把 repeated source loads自动升级为 backing-memory/DRAM traffic。见 `mmq_work.py:225-263`。因此 repeated bytes 是待测 cache/transaction 假设，不能直接作为补价。

## 允许的唯一下一步：一个可证伪的配对合成微基准

不修改现有 gate，也不读取或拟合 LLM 时延。用锁定 binary/source hash、固定 GPU UUID/driver/clock、固定输入与输出 checksum，拟定一个包含两个operator family的合成工具；以下是待冻结方案，尚非可执行预注册清单：

1. **MMVQ family：** Q5_0、M=1、K=896，开发对照 N=128、896、4864（这些形状已在R16用于诊断，不能再称独立未见形状）；新形状验证候选 N=1536，执行前仍需冻结完整形状/内核分支与验收协议；记录 source activity 的 kernel name、grid、block、dynamic shared bytes 和 device event 时长。拟冻结断言为 source activity 必须显示 MMVQ vector path，grid/block 必须等于已验证派生几何；否则 MMA-wave 替换假设直接被拒绝。
2. **MMQ family：** Q5_0、M=64、N=896，K=768、896、1024；记录 main/fixup kernel 序列、grid、block、dynamic shared bytes、device event 时长。K=896 必须作为 stream-K/fixup case，K=1024 是 full-iteration 对照；若 activity 不显示预期 main/fixup 序列，则现有 MMQ source-work dispatch 假设被拒绝。

每个 family 使用同一输入、同一 stream、相同预热与随机化顺序的两种配对条件：

- **device estimand：** CUDA event 包围单个目标 kernel 序列；只在批末同步，报告设备流包络。该包络可能包含事件开销、launch提交不足形成的GPU空闲和kernel间隙，不能直接命名为纯kernel执行时长。需用同条件CUPTI并集/空隙和独立事件控制解释；存在未解释空隙时拒绝用作kernel成本。
- **CUPTI perturbation estimand：** 同一条件启用 CUPTI Activity kernel records，验证 kernel identity/geometry，并单独报告 `event_with_CUPTI - event_without_CUPTI`；不得把差值并入 kernel 系数。
- **CPU host estimand：** 另用 QPC 记录 submit-only 和 final-sync-only 分布，绝不从 event duration 相减，也不把 QPC wall 加到 device duration。

只有在 source identity、输出 checksum、event/CUPTI kernel 序列、留出形状和稳定性门全部通过后，才可讨论新增 MMVQ CTA/warp 模型或 MMQ scalar/shared/occupancy 资源模型。该实验可推翻具体机制；它不预设任何性能倍率。

## 需要决定前补齐的数据

当前源码足以确认 MMVQ/MMA 的类别错配和 MMQ 的未建模资源类别，但不足以确定任何数值率。若要进行实现选择，需要上述微基准保存的：每个 target kernel 的 CUPTI activity identity/geometry、CUDA-event 分布、event/CUPTI 扰动、submit/sync QPC 分布，以及完整 source/binary/device/driver/clock/output-checksum 身份。


父任务复核：已用于机制定位的N=4864被改为开发对照；新N=1536仅为待冻结验证候选。CUDA event覆盖范围已明确为设备流包络，未建立活动与观察扰动闭合前，不允许从该值拟合纯kernel成本。
