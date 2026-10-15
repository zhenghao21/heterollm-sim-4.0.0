# 现有 18 组实验的内存执行路径审计

**结论：旧结果并非全部使用 DramCore 和 NandCore。** 多数完整服务点使用的是 **HBM 聚合服务 + HBF/SSD 的 NandCore + 主机 DDR 的 DramCore**。主机 DDR 的物理访问不能证明 GPU 的 HBM 也走了 DramCore。

本审计针对 2026-10-13 15:57 的 `results.json`，只读核对产物和对应源码，没有重跑矩阵。逐子项的源码版本、路径、行号、资源 owner、迁移记录和限制见 [结构化审计](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm/core_path_audit.json)。共分类 **18 组、58 个子项**。

| 组 | 旧产物实际路径及例外 |
|---|---|
| 01 理想不变量 | HBM 直接 DramCore，HBF 直接 NandCore。名为 aggregate 的一致性检查是“一次大请求 vs 拆分 burst”，两边都用 DramCore。 |
| 02 固定负载 | 完整服务的 HBM 聚合、主机 DDR 物理；B 的 HBF KV 用 NandCore。70B 需求表是字节/带宽公式。 |
| 03 容量与 SLO | 24 个可执行点同上，专用 `hbm_kv` 也继承旧聚合标记；12 个容量拒绝点未执行服务。 |
| 04 冷热会话 | 独立页池轨迹：A 用 HBM DramCore；B 用 HBF NandCore；C 两者及迁移链路均执行。无 GPU 算子。 |
| 05 写缓冲 | 180 个基础写入/排空轨迹用 NandCore，展开成 720 行；额外 GC 放大是公式覆盖层，没有新的 Core 请求。 |
| 06 Direct/relay | HBF 源读用 NandCore，目标及中继 HBM 用 DramCore；链路实际排队。3 个主点、15 个迁移量扫描点。 |
| 07 封装预算 | 可运行槽位按介质使用独立 Core；纯 HBF 行没有 HBM 访问；全 HBM 容量失败行未执行。旧记账回归另列。 |
| 08 权重轨迹 | A 的权重/KV 直接 DramCore；D 的权重 NandCore、KV DramCore。仅内存访问轨迹。 |
| 09 完整服务研究 | 公共 HBM 仍聚合；B/D 的 HBF 直接 NandCore；C 的 HBF 迁移端及 F 的 SSD 迁移端走 NandCore，HBM 端仍聚合；E/G 的远端 KV 是主机 DDR DramCore。 |
| 10 热度/读写比 | 7 个冷热/容量可运行点使用两种 Core；0% HBM 策略拒绝；4 个读写比点只使用 NandCore。 |
| 11 预算配置 | 4 个完整服务点仍混合后端，5 个拒绝点没有服务时间线；购置成本和面积是条件公式。 |
| 12 服务消融/路径 | 专用理想 `hbm_kv` 确实用 DramCore，但公共 `hbm0` 仍聚合。HBF 用 NandCore；有迁移的 relay/dual 中继用 DramCore。FIFO 无迁移点不访问中继。 |
| 13 Agent 生命周期 | 活跃 HBM 及 HBM 迁移端仍聚合；实际下沉/恢复/取消已提交的 HBF 阶段用 NandCore。写预算阻止下沉的点没有 HBF 访问。 |
| 15 多 GPU/共享池 | GPU rank 和通信真实调度；各 rank 的 HBM 复制了旧聚合标记。HBF 为 NandCore，CXL 远端介质为 DDR DramCore。 |
| 16 持续 SLO | 77 个服务点仍是混合后端；观察窗口和边界分类是结果统计，不会改变 HBM 执行路径。 |
| 17 扩展模型 | 12 个完整服务点继承旧 HBM 聚合。24 个结构探针则直接提交 HBM DramCore/HBF NandCore；计算时间用 FLOPs 公式，TPS 为 null。 |
| 18 有效读带宽 | 13 点仅 NandCore + 链路队列；FIFO 是容量/回压账本，不执行 DRAM 读写，也不计算 GPU TPS。 |

旧 18 组共有 **258 个完整服务结果行**。其算子 `dram_traffic` 中，只有组 12 的专用 `hbm_kv.memory` 一行出现 HBM Core owner；**没有一行出现公共 `hbm0` 的 DramCore owner**。多数记录出现的 `cpu0.memory` 属于主机 DDR。组 12 的迁移原始记录还明确保存了 `source_read: aggregate_service_v1`，因此不能仅凭 `PhysicalKVTransferExecutor` 的名字断定搬移两端都走 Core。

另附的 `economics.json` 保留购置成本与均匀磨损寿命的条件公式。它使用历史服务流量，不提交 Core 请求，也不代表本轮重新执行了服务实验。

**当前代码已作强制修正：HBM 聚合执行开关和运行时旁路已移除，新 HBM 仿真必须使用 DramCore。** `serving_row` 新增 `core_execution_audit`，按实际介质种类和 owner 记录持久在线服务中的 `submit/submit_many`；预览、孤立重放及服务前初始化不计入这份审计。旧检查点版本已更新，避免作为新物理结果复用。

这项修正不会自动重算历史数据。新的 Decode 对比应以新运行结果及其 Core 审计为准；旧 18 组在这里保持原始执行口径。Core 本身仍是事件模拟器，可能使用内部标量加速或裁剪明细，不代表逐周期硬件仿真或厂商校准。

本次修正后的复用复核：**组 01、04、05、06、07、08、10、18 可直接保留原结果**。它们直接调用 Core，或使用原本就没有聚合标记的物理迁移配置；删除旁路没有改变这些实际执行分支。原有公式覆盖层、未执行的容量拒绝及不含 GPU 的微基准范围均保留，不能把复用写成重新执行。

配套文件已按新代码重新生成：[acceptance.json](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm/acceptance.json) 的 17 项检查通过；[profiles.json](F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/hbf_hbm/profiles.json) 的四组 HBF/HBM 完整服务对照均记录了实际 HBM DramCore 资源。本段描述配套文件的刷新；旧路径审计保持历史口径。**GPU 计算及启动开销仍使用既有解析算子与资源模型；本轮移除的是 HBM 访存旁路。**

本轮受影响的完整服务实验现已全部重跑并合并到[最新总报告](report.html#core)：19组检查通过，243个完整服务点均有实际HBM DramCore记录，其中151点还有NAND Core记录。16点独立Decode对照与9张图见[Decode结果](report.html#decode)。不受此次代码修改影响的直接Core微基准按上文说明复用。
