# llama.cpp 与 HeteroLLM Simulator 的 DRAM/NAND 执行验证文档

## 1. 文档目的

本文件规定一套可重复的执行流程，用于比较：

1. 前端配置的 HeteroLLM Simulator 仿真结果；
2. 同一模型、同一负载、同一调度和映射条件下，本机 llama.cpp 的实测结果。

验证分成两条证据链：

- **DRAM/GDDR 直接 parity**：使用本机 RTX 5080、GDDR7、DDR5 和真实 llama.cpp 推理结果，比较逐请求 TTFT、TPOT、E2E 以及仿真器的 DRAM 物理访问 ledger。
- **NAND/SSD 条件验证**：先通过正常前端 planner/event path 验证 NAND 页读、编程、队列和 KV offload 的仿真行为；只有本机 llama.cpp 能显式导出 KV 到 SSD 的证据时，才进行 native NAND parity。

本文件不把权重 `mmap`、操作系统文件缓存或模型加载过程误认为 KV Cache SSD 卸载。

## 2. 当前项目和本机状态

### 2.1 项目内已有能力

- 模型目录包含 `qwen3_8-27b`。该条目是 `analytical_approximation`，内部使用 Qwen3.5/3.8 混合线性注意力图，只建模文本主干。
- 本机硬件预设为 `local-native-rtx5080-9950x3d-gddr7-ddr5`，包含 GPU、GDDR7、CPU 和 DDR5 主存。
- llama.cpp runtime adapter 会把 `gpu_layers`、batch、KV、continuous batching、slot order 和映射规则投影到统一 scenario。
- `storage_probe` 会通过正常 planner、事件内核和报告链路访问 SSD/HBF/NAND，不应改用绕过调度器的 direct-core fixture。
- 前端评分接口要求显式 native reference，并按 request ID 对比 TTFT、TPOT 和 E2E。

相关代码和文档：

- `src/heterollm_sim/model_presets.py`：Qwen3.8-27B 模型条目。
- `src/heterollm_sim/architecture_presets.py`：本机 RTX 5080 + GDDR7 + DDR5 硬件预设。
- `src/heterollm_sim/llama_scenario.py`：llama.cpp runtime、GPU layer 和 KV owner 映射。
- `src/heterollm_sim/runtime_adapters.py`：`LlamaCppRuntimeConfig` 定义。
- `src/heterollm_sim/web.py`：native reference 和仿真结果的评分逻辑。
- `docs/PHYSICAL_MEMORY_MODEL.md`：DRAM/NAND 物理事务、`storage_probe` 和 ledger 约束。

### 2.2 已观测的本机硬件

正式执行前仍要重新采集并保存快照。当前只读检查得到：

- GPU：NVIDIA GeForce RTX 5080，显存约 16 GB，驱动 617.14；
- CUDA：12.8；
- CPU：AMD Ryzen 9 9950X3D，16 核 32 线程；
- 内存：4 条 32 GB DDR5，配置频率 5600 MT/s；
- SSD：3 块约 1 TB NVMe。

### 2.3 当前阻塞

当前 PATH、项目目录和 `Downloads` 中没有发现可直接运行的 `llama-cli` 或 `llama-server`，也没有发现 Qwen3.8-27B GGUF。工作区有 llama.cpp 源码副本：

```text
F:\\codex_project\\_runtime_sources\\llama.cpp
commit: d3146f2b56c2db4711ac8391871c9e529d1946d7
tag: b10919
```

真正采集 native 数据前，需要：

1. 固定一个 Qwen3.8-27B GGUF 文件及其 SHA-256；
2. 从固定源码构建带 CUDA backend 的 llama.cpp；
3. 记录二进制 SHA-256、编译选项和运行时版本；
4. 用实际 GGUF 做模型加载和架构 parity 预检。

## 3. 验证原则

### 3.1 先锁定身份，再比较时间

每一组结果必须绑定以下身份：

- 模型文件路径、GGUF SHA-256、量化格式和 GGUF 几何；
- GPU UUID、显存、驱动、CUDA、CPU、内存和 SSD；
- llama.cpp 源码 commit、二进制 SHA-256、编译选项；
- runtime 配置、scheduler 配置和 placement/mapping；
- workload request ID、prompt token 数、output token 数和 seed。

身份或结构不一致时，不进入 timing parity，结果标记为 `invalid_identity` 或 `partial_reference`。

### 3.2 仿真器不负责证明模型输出质量

本轮目标是性能和存储路径验证。仿真器不会产生与 native 完全相同的 logits，因此不把文本内容相同作为仿真器验收条件。native 必须固定 seed、temperature 和输出 token 上限，保证 timing 样本的 token 数可对齐。

### 3.3 不允许自引用校准

如果后续要根据 native 数据调整 kernel 或内存成本：

- calibration workload 和 holdout workload 必须分开；
- 不能用某个 workload 的 native 结果调参后，再用同一个 workload 宣称验证通过；
- 当前已有的 Qwen3.8 GPU kernel 历史校准只能作为起点，不能代替本轮 DRAM/NAND 端到端验证。

## 4. 模型选择和 parity 预检

### 4.1 主模型

首选模型：

```text
Qwen3.8-27B
```

选择理由：

- 64 层，规模足够大，能够产生明显的权重、KV 和主存压力；
- 使用混合线性注意力/全注意力结构，可以同时覆盖普通 KV 和 linear state；
- 能检验复杂模型在 llama.cpp runtime adapter、调度和存储层上的完整路径。

模型限制：

- 项目中的 Qwen3.8-27B 是分析近似，不是已证明的逐算子精确模型；
- 当前 llama.cpp 源码显式使用 `qwen35`/`qwen35moe` 架构标签，Qwen3.8 GGUF 是否能直接使用必须由实际文件验证；
- GGUF 可能包含一个 nextn/MTP 加载单元，native 的 `-ngl` 计数和仿真器主执行图的层数需要单独记录。

### 4.2 降级和控制模型

如果 Qwen3.8 GGUF 无法被固定的 llama.cpp 二进制加载，按以下顺序降级：

1. Qwen3.5-27B；
2. Qwen2.5-32B 作为稠密模型控制组。

降级后必须让仿真器和 native 使用同一模型文件，不允许跨模型比较。

### 4.3 GGUF parity 门槛

至少检查以下字段：

```text
n_layer             = 64
n_embd              = 5120
n_head              = 24
n_head_kv           = 4
intermediate_size   = 17408
vocab_size          = 248320
context_length      = GGUF 声明值
architecture        = 实际 GGUF 标签
quantization        = 实际 GGUF 量化格式
nextn/MTP layers    = 实际 GGUF 声明值
```

仿真器应优先使用 GGUF metadata 和 tensor directory 生成模型输入，不要只手工填入模型名称和参数规模。

## 5. 硬件输入

### 5.1 直接 parity 硬件

直接 parity 使用：

```text
local-native-rtx5080-9950x3d-gddr7-ddr5
```

硬件输入应包含：

- RTX 5080 GPU UUID、显存容量、PCIe 协商链路、驱动、CUDA；
- GDDR7 组件、容量、带宽和物理内存配置；
- Ryzen 9 9950X3D CPU 以及实际线程配置；
- DDR5 容量、通道、频率、带宽和服务 owner；
- 选定 SSD 的型号、容量、PCIe 链路、顺序/随机读写测量值和延迟。

GDDR7、DDR5 的 bank、row、burst、刷新和命令参数如果来自组件预设，必须在报告中标记为 `analytical_parameter`。不能把协议峰值带宽直接写成实测持续带宽。

### 5.2 NAND 仿真硬件

NAND 仿真场景需要在硬件 JSON 中增加一个明确的 SSD 组件和 PCIe 链路，并为组件设置：

- `physical_memory_config.kind = SSD`；
- 容量；
- page、block、channel、die、plane 几何；
- page read/program 延迟；
- 主机传输带宽和延迟；
- 内部介质带宽和队列深度；
- 唯一的 `resource_id` 和 memory service owner。

未知的 FTL、GC、磨损均不应补造为确定时序。真实 SSD 只做非破坏性文件读写测量；物理 NAND erase 只在仿真器中执行。

## 6. Runtime、调度和映射锁定

### 6.1 初始 runtime 配置

第一轮建议从以下配置开始，然后用 native 启动日志中的实际值覆盖：

```text
policy=llama_cpp
threads=16
threads_batch=16
batch=512
ubatch=512
context=4096
parallel=1
gpu_layers=以 native 实际加载值为准
flash_attn=以 native 实际值为准
kv_unified=true
cont_batching=true
warmup=true
seed=0
mmap=true
mlock=false
offload_kqv=true
op_offload=true
split_mode=layer
main_gpu=0
preemption_enabled=false
```

第一轮不启用 speculative decoding 和 MTP 推测执行。GGUF 中的 nextn/MTP loading unit 仍需记录，因为它可能影响 `-ngl` 计数，但不能自动当成主执行图的一部分。

### 6.2 映射规则

仿真器必须通过 llama.cpp runtime adapter 生成 placement，不手工复制一份近似映射。重点记录：

- native `-ngl`；
- GPU layer 的尾部映射边界；
- output layer 是否计入 loading units；
- 每层 KV owner；
- linear state owner；
- `split_mode`、`main_gpu` 和 `tensor_split`；
- GDDR/DDR/SSD 对应的物理服务 owner。

如果前端 llama 模式自动打开 `device_memory_tiering` 或 `blackwell_analytical_v1`，直接 parity 场景应关闭未经 native 证明的 HBF/分层扩展，并把它作为单独的分析场景。

## 7. 负载矩阵

第一轮先执行 D0-D3；D4 作为长上下文和 NAND 压力场景。

| 编号 | 目的 | 请求形状 | runtime 形状 | 优先级 |
|---|---|---|---|---|
| D0 | llama 基线 | B=1，输入 512，输出 128 | `context=4096, batch=512, ubatch=512, parallel=1` | 必做 |
| D1 | Prefill 压力 | B=1，输入 4096，输出 128 | `context=8192, batch=512, ubatch=512` | 必做 |
| D2 | Decode 压力 | B=1，输入 512，输出 512 | `context=4096, batch=512, ubatch=512` | 必做 |
| D3 | Continuous batching | 4 个请求，每个输入 512、输出 256 | `parallel=4, batch=2048, ubatch=512, context=4096` | 必做 |
| D4 | 长上下文/KV 压力 | B=1，输入 8192，输出 256 | `context=16384` | 可选，NAND 前置 |

所有请求必须使用：

- 固定 request ID；
- 固定 prompt token 数；
- 固定 output token 上限；
- 相同到达时间；
- `temperature=0`；
- 固定 seed；
- 不抢占；
- 不复用已有 slot；
- 不在批次中途插入新请求。

prompt 应使用固定文本或固定 token ID 文件，不能使用“约 512 token”这类不精确描述。

## 8. Native 采集流程

### 8.1 采集次数

每个负载：

1. 启动一次固定版本的 llama.cpp server；
2. warmup 2-3 次；
3. 正式测量 5 次，资源允许时测量 7 次；
4. 保存每次的原始日志和逐请求结果；
5. 模型加载时间单独统计，不计入 TTFT。

运行期间应避免其他 GPU 任务，并记录 GPU 温度、功耗、时钟和显存占用。发现明显降频或系统干扰时，该轮标记为 `unstable_native_run`。

### 8.2 Native reference 最小格式

前端评分至少需要逐请求的以下字段：

```json
{
  "schema": "heterollm.native-reference/v1",
  "identity": {
    "model_sha256": "...",
    "hardware_fingerprint": "...",
    "runtime_fingerprint": "..."
  },
  "requests": {
    "request-0000": {
      "prompt_tokens": 512,
      "visible_output_tokens": 128,
      "ttft_ns": 0,
      "tpot_ns": 0,
      "e2e_ns": 0
    }
  },
  "scheduler": {},
  "mapping": {},
  "repetitions": []
}
```

主比较使用 native 多次运行的中位数；`repetitions` 保留原始样本，用于计算 p50、p90 和变异系数。

TTFT、TPOT、E2E 的边界必须和仿真器相同：

- TTFT：从 engine request begin 到首个可见输出 token；
- TPOT：首 token 到末 token 的时间除以可见输出 token 数减一；
- E2E：从 engine request begin 到末个可见输出 token。

## 9. 前端仿真流程

每个 D0-D4 场景都生成独立 scenario JSON：

1. 导入本机架构预设；
2. 导入同一 GGUF 的模型几何和量化信息；
3. 填入相同 request IDs、prompt/output token 数；
4. 填入与 native 一致的 llama.cpp runtime；
5. 运行场景校验；
6. 重新生成 runtime placement；
7. 执行事件仿真；
8. 导出完整 JSON 报告；
9. 将 native reference 注入前端评分接口；
10. 保存 score、fingerprint 和差异分类。

连续 batching 使用 aggregate retention policy。静态单请求可以保留更细的执行轨迹，但 aggregate trace 不得当作逐事件精确 trace。

## 10. NAND/SSD 执行分支

### 10.1 N0：NAND 机制验证

通过 `workload.metadata.storage_probe` 走完整 planner/event path，至少覆盖：

- page read；
- page program；
- 跨页访问；
- 多请求队列；
- 主机 PCIe 传输；
- page buffer 或内部通道竞争；
- 必要时的仿真 erase。

验收内容包括：

- logical bytes、host transfer bytes、internal transfer bytes 守恒；
- page read/program/erase 计数正确；
- 队列和资源区间不重叠；
- owner、地址、offset、generation 一致；
- 重试不会重复提交已完成事务。

### 10.2 N1：仿真器 KV/SSD 敏感性

构造一个 D4 长上下文场景：

- active KV 保留在 GDDR/DRAM；
- `kv_policy.offload_component` 指向 SSD；
- 触发 KV page migration 和 restore；
- 观察 SSD 页读、主机传输、内部 NAND 传输、队列等待、KV 峰值驻留和总延迟。

该结果必须标记为：

```text
simulation_only_nand_sensitivity
```

### 10.3 N2：Native SSD KV parity

只有在 llama.cpp 或定制 runtime 能提供显式 KV 页写盘和恢复证据时才执行。至少需要：

- 每个请求的 KV offload/restore 字节数；
- 页或块粒度；
- SSD I/O 时间戳；
- host transfer 时间戳；
- KV owner 和页生命周期；
- 与仿真器相同的 request ID 和调度顺序。

如果本机 llama.cpp 只支持主存 KV、GPU KV 或 `mmap` 权重，不做 N2 timing parity，并明确报告：

```text
native_nand_comparison = not_supported
```

## 11. 指标和验收门槛

### 11.1 结构门槛

以下条件必须全部满足：

- GGUF 文件和 SHA-256 一致；
- 模型几何和量化一致；
- GPU/CPU/内存身份一致；
- runtime、scheduler 和 mapping fingerprint 一致；
- request ID、prompt/output token 数一致；
- 没有请求被拒绝或提前终止；
- 没有未声明的 fallback；
- native 和仿真输出 token 数可对齐。

### 11.2 DRAM/NAND 机制门槛

- DRAM read/write 方向不被静默改成全读；
- physical bytes、logical bytes 和 host transfer 分类守恒；
- Row/Bank/Refresh 状态更新符合请求顺序；
- NAND page read/program/erase 计数正确；
- 共享 PCIe 资源日历没有重叠预约；
- 失败任务不留下部分提交状态；
- 同一输入重复运行时 ledger 稳定。

### 11.3 Timing parity 建议门槛

对于有明确 native/kernel 覆盖的场景，第一轮建议目标为：

- TTFT 中位数 APE <= 10%；
- E2E 中位数 APE <= 10%；
- TPOT 中位数 APE <= 15%；
- p90 APE <= 20%-25%。

对于 Qwen3.8 混合结构的分析近似，初始工程门槛可放宽为：

- TTFT/E2E 中位数 APE <= 15%；
- TPOT 中位数 APE <= 25%。

超出门槛时，先分类为 `needs_calibration`、`unsupported_mapping`、`native_instability` 或 `coverage_gap`，不要直接归因于 DRAM 或 NAND 模型错误。

所有 timing 结果同时报告：

- signed error；
- absolute error；
- APE；
- native p50/p90；
- 仿真值；
- native 样本数和变异系数。

## 12. 产物目录和命名

建议在项目外的实验目录保存原始数据，在项目内只保存最终可复用的 manifest 和汇总结果：

```text
artifacts/validation/llama_dram_nand/<run_id>/
  hardware_snapshot.json
  model_gguf_metadata.json
  runtime_identity.json
  workload_manifest.json
  scenarios/
    D0.json
    D1.json
    D2.json
    D3.json
    D4.json
    N1.json
  native/
    D0-rep01.json
    D0-rep01.log
    ...
  simulator/
    D0-report.json
    ...
  native_reference/
    D0-reference.json
    ...
  comparison/
    summary.csv
    summary.json
    report.md
```

建议每次 run 使用独立 `run_id`，并将以下 fingerprint 写入所有汇总文件：

- model fingerprint；
- hardware fingerprint；
- runtime fingerprint；
- scheduler fingerprint；
- mapping fingerprint；
- native reference fingerprint。

## 13. 最终报告必须回答的问题

每次实验结束后，报告至少回答：

1. 仿真器和 native 是否运行了同一模型文件？
2. 模型架构和量化是否通过 parity？
3. GPU layer、KV owner、linear state owner 是否一致？
4. scheduler 是否一致？
5. 每个 request 的 TTFT、TPOT、E2E 误差是多少？
6. native 运行是否稳定？
7. DRAM physical ledger 是否守恒并且可重复？
8. NAND 是否只是仿真敏感性，还是有 native SSD KV 证据？
9. 误差属于 timing calibration、mapping、scheduler、model coverage 还是 native 测量不稳定？
10. 结果能否推广到其他模型、上下文或硬件，还是只对当前 pair 有效？

## 14. 执行顺序

建议按以下顺序推进：

1. 构建并固定 llama.cpp 二进制；
2. 获取并固定 Qwen3.8-27B GGUF；
3. 采集硬件快照；
4. 完成 GGUF parity 和 native 加载预检；
5. 先执行 D0，确认单请求链路；
6. 执行 D1、D2，确认 Prefill/Decode；
7. 执行 D3，确认 continuous batching；
8. 导出 native reference 并通过前端评分接口比较；
9. 检查 DRAM physical ledger 和误差分类；
10. 构造 D4/N1，执行仿真器 SSD/KV 敏感性；
11. 只有发现真实 native SSD KV 证据后，才继续 N2；
12. 汇总 `summary.csv`、`summary.json` 和最终报告。

本轮的完成条件是：D0-D3 具有完整 native reference 和仿真报告，身份与结构门槛全部通过，DRAM ledger 通过机制检查，并且每个 timing 误差都有明确分类。NAND 如果没有 llama.cpp 的显式 SSD KV 路径，仍可完成 N0/N1，但必须把 N2 标记为不适用，不能把仿真结果写成 native 实测结论。
