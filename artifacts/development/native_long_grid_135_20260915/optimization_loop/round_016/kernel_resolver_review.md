# Round016 kernel resolver 独立审查

日期：2026-09-16T10:25:49。只读实现和测试，额外只读 CostPhase 与 event kernel 的相关定义以核对实际计费；未改实现、未回退他人编辑、未运行 GPU。

审查源码 SHA256：
- `src/heterollm_sim/kernel_calibration.py`：`e2b1d98a30ddd6f8e3d4fbe4b55d84cb5cac08ebc34b09d64a717d2b6caf5fae`
- `tests/test_kernel_calibration.py`：`d0d981886437ca33207645c18511d079482fb09f8c0d68e8c8a63803de0c000f`

结论：**同 CostPhase 的 max 计费成立，联合 key 严格匹配与基本字节身份门有效；但目前是证据完整性和声明门框架，尚不适合接入真实性能系数后宣称“数值/观测/硬件/执行归属已自动验证”。有3项实质缺口需在生产接入前处理。**

## 已复现的问题

### P1：raw/质量报告只是哈希存在，accepted 的时延和通过声明没有绑定原始内容

位置：`kernel_calibration.py:289–292, 325–348, 385–388`。只有 execution_contract 会保留并解析文档，raw_events、numerical_validation、quality_report、hardware_runtime_observation、protocol 等其余材料只读字节计算 SHA。数值、稳定性、profiling 门只取 profile 内 `passed=true`，未核对引用文档内容。`device_ns` 和 `sample_count` 也没有重算或引用到具体 raw 样本集合。

主机复现使用现有 pytest fixture，保持所有文件 SHA 正确，故不存在“文件被篡改但没更新hash”的问题：

- raw_events 内容为 `sample_count=1, device_ns=[7], stream_count=2`；
- numerical_validation 内容为 `passed=false, failure_count=99`；
- quality_report 为稳定性与 profiling 全失败，扰动为12；
- profile 仍写 `device_ns=100, sample_count=30, gates全部true`。

loader 成功，resolver 返回 **mode=exact、100ns、30样本**。这意味着未来提取器若引用错报告、漏同步状态或写入错误聚合时延，loader 不能阻止错误证据生效。源码255–259行已诚实声明“不重新运行提取/数值检查”，但调用方不能把 evidence_files_verified 理解成这些门已经验证。

建议：明确分开 verified-bytes 与 measurement-validated。接入真实成本前，由独立、版本化的验证器读取质量报告严格schema，并检查每条报告的 key、原始文件SHA、样本身份/索引、计数、聚合方法、duration、阈值和失败状态。至少可从受约束raw导出内容重算中位device_ns、stability、profile扰动；数值验证报告应有受约束结果结构和其读取的raw/参考实现SHA。未完成语义验证只能 analytical_fallback。无需每次重复跑GPU，但不能仅增加另一个 `validated=true` 声明。

### P1：GPU/单流/owner来自合同与phase metadata声明，未约束phase的实际资源

位置：`kernel_calibration.py:408–434, 462–490`。

主机复现：保持 phase category=COMPUTE、metadata target_component=gpu0/stream=main/role=main，同时将其实际 demands 改为 `cpu0.compute=30ns`、`cpu0.memory=40ns`。适配器仍添加 `gpu0.kernel_stream.main=100ns`，`kernel_calibration_applied=true`。当前“CPU拒收”测试只改合同的device_kind，不能覆盖实际phase资源与metadata错位。

另一复现：将经过字节核验的合同owner_resource_id改为 `gpu0.frontend.calibration`，适配器也接受。现有launch拒收只检查进入时已有demands和phase.name；新增owner本身只检查 `gpu0.` 前缀，不核对资源类型和实际资源目录。

建议：适配器需接收或引用实际执行图/资源目录中的不可变执行归属，核验phase确实是对应GPU物理kernel、目标组件/stream和所有资源归属一致，owner资源存在且类型是专用kernel包络，CPU/host/frontend/通信/混合设备phase拒收。单流合同需关联原始trace或已经受约束的planner执行契约，不能由可随手填写的verified/stream_count代替。图中实际多流而合同仍写1时，目前没有足够输入验证。

### P1：硬件观测字段未绑定effective profile内容；HBF拒收依赖调用方正确重算query哈希

位置：`kernel_calibration.py:212–251, 296–302, 372–379`。

已有优点：query effective_hardware_sha改变时会fallback，现有测试201–204对此有覆盖。

但主机复现：profile中的 `gpu_uuid` 改成 `A_DIFFERENT_GPU`、sm_count改成9999，保留同一个effective_profile文件SHA及entry key，loader/resolver仍返回exact；hardware_runtime_observation甚至可写observed=false/不同GPU，只要文件SHA匹配就照样通过。effective_profile文件也未解析，无法证明“它包含实际HBF/HBM带宽、容量、延迟及其他影响成本的配置”。

适配器没有当前硬件对象或其可信派生身份输入。如果caller复用旧query哈希，实际HBF配置变化本模块无法知晓。因此不能把当前测试表述成“任意HBF变化均已自动拒收”。

建议：用当前实际硬件描述的规范化、完整有效配置生成身份，明确列出纳入哈希的HBM/HBF、计算、缓存、资源/设备放置等字段；把GPU观测报告和有效配置在验证器中关联。profile读取与apply时，都比较来自实际执行上下文的身份，避免允许自由传入陈旧key。跨设备/硬件参数扫描必须自然失配，而不是要求上层手工记得修改一个SHA字符串。

## 已核对正确的部分

1. **max而非相加成立。** CostPhase.service_ns在cost_models.py:923–926直接取所有demand的max；适配器在同一个phase追加duration，未新建顺序phase。主机fixture原30/40ns追加100ns得到100ns。event_kernel.py:2111–2136同样从相同start计算每个end并取最大，没有把100加到40上。
2. **联合匹配包含关键shape和K。** key含m、n、k_logical、k_executed、类型、格式、layout、三种stride、role、kernel_family/variant/dispatch_signature、cache与hardware/runtime身份。只接受完整精确key，不做独立维度插值；K、stride或kernel路径字段变化会fallback；禁止额外模型名/模型SHA/prompt键。注意“key字面完备”不等于它已从实际物理调用正确派生，未来接入时仍需绑定phase实际shape。
3. **基础文件门fail-closed。** 必要schema/字段、空identity maps、非有限时延、缺文件、错误SHA/长度、文件读取时改变、重复JSON字段和key、显式failed gate、未知evidence id均拒绝。loaded映射冻结后不可修改。这是良好的证据字节完整性门。
4. **已声明多流拒绝。** owner stream_count=2或concurrent_kernels=true会fallback；CPU owner、phase设备/stream/role/boundary直接不匹配也拒收。但不覆盖前述“声明与实际图不一致”。
5. **同名重复包络拒绝。** 再调用已应用phase不追加第二次，reason=kernel_envelope_already_applied；已有相同resource_id也拒收。实际物理owner别名需要现有资源映射参与；event kernel在提供resource_owners时会拒绝同任务重复物理owner（event_kernel.py:947–953），适配器自身尚不掌握这份映射。
6. **保留分析基线与限制标签。** 不改原bytes/energy；max下界不会缩短原分析时延，明确conditional、validated_llm_scope=false、hardware/cache transfer未验证。不能把这种下界机制当作已恢复真实compute/memory资源占用，代码没有如此宣称。

## 验证与测试缺口

- 已运行 `E:\anaconda\python.exe -m pytest tests/test_kernel_calibration.py -q`：**107 passed（1.75秒）**。
- 上述额外反例仅在 TemporaryDirectory 内构造fixture，不修改测试或实现、不引入性能常数。
- 现有fixture的raw、quality、numerics均为占位JSON，却可用于accepted exact测试，适合测试传输/适配框架，但不能证明真实证据语义验证。
- 建议新增回归：正确hash但失败质量报告；raw计数/中位数与entry不同；raw语义key与entry不同；报告引用另一raw；CPU实际demands配GPU metadata；frontend owner；实际执行上下文硬件变化而query旧哈希；已观测多流与单流声明不一致。
- 尚未运行真实profile、CUDA kernel、LLM预测或新硬件验收。当前模块没有默认启用，因此以上是接入前阻断项，不表示现有默认预测已受到这些反例影响。
