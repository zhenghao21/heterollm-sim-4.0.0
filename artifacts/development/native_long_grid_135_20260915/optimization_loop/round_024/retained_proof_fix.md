# R24 retained proof 引用一致性修复

2026-09-16。确定性适配器修复；没有改变成本、retained KV物理语义、native记录或R23冻结。未提交、未push。

## 根因

R23的27个qwen38_gpu格全部在worker重派生时失败。CPU qwen38通过snapshot map引用readonly模型，字段为`size_bytes`；GPU直接引用同一路径，字段为`bytes`。冻结端按模型路径缓存完整scope，CPU先入缓存的原始引用表示被用于GPU proof。独立GPU worker没有该缓存，重新派生后整体字典不相等。实际重派生一个失败格的唯一差异是`/model_scope/model_ref/bytes`与`size_bytes`；静态检查确认恰好27格出现该表示差异，路径、SHA及14865116128字节长度均相等。协调器顺序缓存会重现冻结表示，原单行测试未覆盖共享路径异构引用。

冻结源码定位：`tools/predict_stable_native_dataset.py:1360,1511–1521,1544–1552,2628–2648`。属于proof表示/缓存问题，不是架构资格、GPU物理模型或硬件哈希故障。

## 修复

新增`canonical_retained_model_ref`：统一为path/sha256/bytes；绝对非空路径严格解析为实际文件；SHA须64位小写十六进制；长度须非布尔的正exact int；bytes与size_bytes并存必须一致；与实际及打开文件长度一致。没有降低整模型SHA校验要求。

新增`cached_retained_gguf_scope`：缓存包含规范引用及文件身份；每次命中重新读取并哈希已知header长度、验证文件前后身份，防止首调用字段表示与陈旧header被复用。freeze及coordinator resume共用此逻辑。直接worker的scope同样规范化；header短读立即拒绝。

## 验证

- 新增针对性22项通过：CPU→GPU及反序，共享同一模型路径但不同长度别名；逐行独立Python进程重派生；冲突别名/错误长度类型/路径SHA/实际长度检查；相同mtime与长度下改变header也拒绝缓存。
- 三组结构回归：`122 passed, 1 skipped`；跳过的是原环境开关控制的真实131测试，下述独立完整静态检查已覆盖实际数据。
- `retained_identity_static_probe/result.json`：131个全新独立worker全部proof一致，62 conditional、69 uncovered；原27个GPU均可正常静态重派生。每worker只读固定selection、非计时warmup证据、实际GGUF头部及来源身份。没有跑完整模型哈希、模型推理、场景模拟或native重测。
- 原R23 freeze SHA保持`4b6abf6a93e5e9db90059f03a820e73d192abecf621a1188de912082a5d05606`。27个原失败没有被覆盖或改为成功。

完整根因、源码SHA及probe引用见`retained_proof_fix.json`。此验证只证明proof重派生的一致性，不说明预测精度、A131通过或独立B通过。正常预测worker仍须完成整模型SHA验证。

## 独立review补充：完整模型freeze/resume门禁

核实外层`verified_model_snapshot_map`只全哈希显式映射副本，未映射模型原本没有完整freeze/resume核验。补充`model_identity_refs`独立清单及`verify_retained_model_identities`：每阶段每个唯一规范模型身份全文件哈希一次，检查长度与文件前后身份；清单覆盖及规范表示严格校验。该清单与一般evidence_refs分离，避免单格worker重新全哈希场景中所有模型；其已有read_gguf_metadata整模型验证保留。

新增测试证实：同header、同长度、恢复mtime的主体篡改及假声明SHA均拒绝；别名去重只读一次；resume主体篡改拒绝；worker不重复场景全哈希。最终相关联合结构回归为**141 passed, 1 skipped**。既有131独立worker静态probe依然只证明旧版header/proof一致性，不宣称完整模型身份已在该probe验证；本补充没有创建真实新freeze或运行预测。

旧R23冻结不改，仍由自带源码只读审计。新R24 source对缺model_identity_refs的新retained冻结fail-closed。R24执行器按阶段启动/恢复与结束每唯一模型检查并打印进度，不按每cell重复全模型集合。
