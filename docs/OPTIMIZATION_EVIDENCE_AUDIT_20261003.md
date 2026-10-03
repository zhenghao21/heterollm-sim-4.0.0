# 仿真器历史优化证据审计（2026-10-03）

本审计覆盖 `round_001` 至 `round_065` 的四份机器记录、方向台账、Git 提交、`artifacts/optimization/round_*` 证据和已有测试记录。初始审计基线源码为 `bae6f9c8f11280feeb0db0eb05079827bde7b5fd`；补证后 H65 候选提交为 `4efc351c77e189b18a11680eb615c443bd8a92e9`，记录收口提交为 `dfb0dd9bca4eb7b2edc30689eedb9dd8bf2e4320`，均已推送并核对 `origin/main`。`pytest-of-A` 目录权限警告单独记录。用户要求已暂停定时推进，`llm` heartbeat automation 已删除；H65 已收口，补证状态详见 [EVIDENCE_SUPPLEMENT_STATUS_20261003.md](EVIDENCE_SUPPLEMENT_STATUS_20261003.md)。

这次审计得到的结论很明确：历史轮次大多证明了局部机制或错误状态，少数证明了真实 API/report 链路，几乎没有轮次证明了预测精度，也没有足够轮次证明仿真器自身的稳定速度或内存改善。全量 pytest 通过只能证明工程回归，不能证明性能、Native 误差或用户可见收益。

## 证据等级

- **E0：记录或静态扫描。** 只能证明记录、源码路径、公式或候选存在，不能证明真实执行。
- **E1：局部机制。** 有 focused 测试、解析矩阵、服务或 planner/event 不变量；可以支持“机制修复/机制负结论”，不能外推到端到端有用性。
- **E2：真实前端链路。** 通过 `run_scenario`、`run_jobs`、CLI、HTTP/API、report/trace/visualization 触发，并逐字段比较任务、事件、资源、状态和序列化结果；可以支持“端到端机制行为”。
- **E3：独立外部观测。** 在 E2 基础上有同模型、硬件、shape、运行时的 Native 或真实存储设备成对数据，逐场景记录 TTFT/TPOT/E2E 或设备服务误差、APE、留出和退化；才可以讨论预测精度或硬件准确性。

## 必须优先补证据的轮次

### P0：存储硬件必须走真实前端链路

`H24、H26、H30、H39、H40、H48、H49、H55、H62` 都需要统一的存储前端矩阵。H62 已补 `plan_runtime_placement → run_scenario → simulate_online → report_dict` 的两 profile 机制证据，但仍未覆盖 HTTP、known-offset 跨页前端激活或父/候选 A/B；其它轮次仍不能因局部 `PhysicalService`、`nand_media_service`、地址/擦除/队列控制而视为完整请求闭合。

最小补测应把 DRAM/HBM 视为一个家族、SSD/NVMe/HBF/NAND 视为一个家族，分别选至少两个 profile，使用同一 `run_scenario`/`run_jobs` 输入覆盖页内、跨页、跨 plane、read、program/write、erase、queue conflict 和非法参数。对 baseline/candidate 或 profile A/B 保存 TaskResult、makespan、resource bytes、energy、owner、queue、metadata、`report_dict` 和 HTTP JSON。没有真实设备/Native 时，结论只能写成“前端机制已验证”；若要写硬件准确性，另做 E3 设备成对误差。

`H62` 已修复 direct HBF/NAND billing，并补到 `plan_runtime_placement → run_scenario → simulate_online → report_dict`；当前结果只能接纳为机制/守恒证据，因 known-offset 跨页未在 `run_scenario` 激活、HTTP 未启动、父/候选 A/B 未重跑，不能写成修复收益或硬件准确率。`H48/H49/H55` 仍有跨请求 queue、report/API projection 或 DRAM/HBM 传播边界，不能因为负向矩阵通过就视为整个存储方向已闭合。

### P0：评估、API、可视化和误差口径

`H19、H22、H28、H42、H43、H50、H57、H60、H64、H65` 需要真实 report/API/serialization roundtrip。当前 H64 已修复缺失 TTFT/TPOT/E2E 指标仍 `compared/passed=true` 的门控缺陷；H65 已修复 online `report_dict` 中 `hardware_phase_ledger.guardrails` tuple → JSON list 的直接 payload 漂移，并以 focused/full 回归和独立 report/API 矩阵收口。其余轮次的状态/误差边界仍不能外推为精度改善。

最小补测需要构造 complete、partial、empty、missing metric、unknown、failure、incomplete、analytical fallback、OOD 和负值 guardrail 输入，实际调用 score/report/API/trace/visualization，执行 JSON encode/decode，再比较 HTTP status、`status`、`passed`、coverage、missing/limits、report hash 和 UI/trace payload。H65 已完成其中的 report/API primitive roundtrip；浏览器可视化只有在要宣称页面显示正确或无障碍时才需要启动浏览器，其余状态矩阵仍是待补证据。

这些轮次不能把 fail-closed 语义改善写成 accuracy gain。若要证明预测质量，必须再做 E3 Native 成对 TTFT/TPOT/E2E 误差，不能复用 synthetic reference。

### P0：预测精度与泛化只在有 Native 时成立

`H28、H34、H43、H51、H60、H64` 当前主要证明 request coverage、fingerprint、fallback/OOD、provenance 和不误报通过。Native readiness 补证已核对当前可执行文件/模型身份，但没有同一当前源码、模型、硬件、shape、runtime 的成对数据；这些轮次仍没有独立 Native 精度证据。

若用户需要“准确率改善”结论，必须冻结父版本和修复版本，使用同模型、同硬件、同 shape、同 runtime、同请求批次成对运行 simulator 与 Native，逐场景保存 TTFT、TPOT、E2E 的 signed error、absolute error、APE，中位数/P90/最大值、失败/回退，并留出一个模型或硬件配置。当前 readiness 记录确认历史 development freeze 不是当前盲测，不能复用；H64 的 `incomplete_metric_reference` 只能证明评估门控正确，不能证明误差变小。

### P1：运行速度和内存必须做重复的前端 A/B

`H29、H36、H41、H46、H54、H61` 不能把单次局部 profile 当作仿真器性能改善。

- `H29` 已补 120 个独立 CLI 子进程、60 对 AB/BA、每规模 20 对，并记录 wall/RSS/stdout 等价：small/medium/large wall 中位改善约 5.82%/10.79%/12.34%；medium/large RSS 中位改善约 12.27%/18.34%，small RSS 不接纳为改善。结论仍限定为 CLI 文本前端，不外推 web/run_jobs 或目标系统 latency。
- `H36` 已补 20 对独立 run_scenario 冷/暖 provider 进程 AB/BA，并发现 `.venv` launcher RSS 误采样；校正进程树后，三规模 wall 中位数暖路径下降约 57.98%/75.56%/74.85%，report/makespan/task 结果 20/20 等价，但 RSS 分别上升约 1.98%/1.43%/5.25%，撤销内存改善结论。暖路径仍只是显式 provider 控制，没有安全共享生产生命周期、并发、eviction 或场景身份契约，不晋升生产候选。
- `H41` 已补 small/medium、exact/streaming/aggregate 的独立 run_scenario 前端 AB/BA：同策略 report/metadata 重复稳定，跨策略 task_count/makespan 与 retention contract 一致，但不同 retention 的 report/trace 差异是有意契约成本；没有安全源码候选。`H46` 已补 leaf-cache 配置级独立 A/B，完整 report/trace 等价但没有新的源码候选，RSS 仅为 before/after delta。
- `H54` fresh/reused 有 245.4ms → 32.35ms、3.68MB → 1.58MB，但仍缺随机批次、淘汰、并发和生产入口。
- `H61` 已补真实 `/api/run-jobs` 独立进程 AB/BA，并校正 launcher-only RSS：cached wall 在 n=8/32/64 约 -0.58%/-0.54%/-0.01%，tree RSS 约 -2.12%/+1.69%/+5.23%，输出完全等价但方向不稳定且大规模内存退化；保留有限负结论，不晋升缓存候选。

最小 A/B：baseline/candidate 独立进程，冷/暖缓存分层，至少 20 个独立批次，覆盖 small/medium/large；采集 wall time、P50/P90/置信区间、峰值 RSS/tracemalloc、CPU、任务数、makespan、report/trace/energy/coverage 全量等价，并保留逐批次退化。若没有安全候选，保留负结论即可。

### P1：成本模型必须进入 planner/event/resource/report

`H27、H38、H44、H52、H58` 的公式、单位、owner、queue 和解析守恒证据不能替代真实计费链路。H27 已补 `compile_scenario → simulate_schedule → run_scenario → report_dict` 并有 HTTP A/B；H38 已补真实 `planner → event → resource → report_dict` 与 HTTP `/api/run`，加上 shape、方向时延、单队列/双 lane 控制，但无候选 A/B、Native 或校准性能结论；H44 已补 MMA/full-kernel 的真实 planner/event/report 激活证据，但 collective 仍被单 compute topology 阻塞；H52 已补正式 topology/HTTP 激活性扫描并保留同一 blocker；H58 已补 routed LinkService 的同级 compile→event→run→report 机制 A/B，但 parent/candidate 无差异且无 Native。

补测应通过真实 planner/event/run_scenario 产生 GEMM、reduction、MMA、collective、link/transfer workload，比较 logical/physical bytes、energy、owner capacity、queue/phase、makespan 和最终 report。覆盖 shape、并发、zero/invalid dtype/size、duplicate owner。Native 带宽/延迟只有有独立硬件观测才可报告。

### P1：机制语义修复的端到端边界

`H1、H2、H3、H4、H5、H10、H18、H20、H21、H27、H37、H45、H52、H53、H58、H59` 可以在没有 Native 的情况下成立为机制修复，但若要说“对用户路径有用”，仍需真实入口。

按根因选择 `run_scenario`/`run_jobs`、HTTP 或 planner → event → report，分别执行成功、触发、排除、错误/回滚控制，保存 before/after 状态快照和最终输出。单元测试不能替代写入、读取、并发、错误、API 和回归表面的闭合。

## 可以暂时保留原结论的轮次

在声明范围内，`H6、H7、H8、H9、H11–H17、H23、H25、H31–H35、H47、H51、H56、H63` 的主要目标是环境、记录身份、局部负向扫描或工具链闭合；它们不需要为了“证明精度”强行补 Native。若重新打开这些轮次，必须先有新的实际触发面，不能仅因为证据不是前端就制造工作。

但“暂时保留”不等于“全项目已证实”：`H51` 的 caller-owned mismatch、`H56/H63` 的工具链结论和早期 H6–H17 的记录身份都应在引用为发布证据时明确其局部范围。

## 早期轮次记录本身的缺口

`H1–H20` 多数没有显式 `source_commit`、review artifact hash 或统一 sidecar；`H21–H30` 开始有提交和更完整的证据，但仍有轮次缺少独立 review hash。早期 accepted 轮次主要是 focused/mechanism evidence，建议在重新作为当前开发基线使用前补：当前源码身份、当前远端 SHA、受影响全量回归或明确继承来源、真实入口触发记录。不要回溯重跑所有历史轮次；只补仍被当前候选、发布或回归基线引用的轮次。

当前轮次记录中，继承全量回归是合法的，但必须写清父版本源码身份并确认当前源码未受影响。继承回归不能被写成“本轮新实测”。

## 建议的补证据顺序

1. **H42/H50/H57/H64 的剩余状态边界**：H65 的 primitive serialization 已闭合，继续补缺失/失败/coverage 的 report/API 投影，不重做 H65。
2. **H62/H39/H40/H48/H49/H55/H24/H26/H30**：H62 已有当前版本前端机制证据，下一步补 known-offset 跨页激活、HTTP/report projection 和父/候选 A/B，随后扩展其它存储轮次。
3. **回看 H37/H45/H53/H59**：H27/H38/H44/H52/H58 的当前成本前端边界已记录；后续只在新增 topology、跨请求或失败回滚触发面实际存在时补测，不能重复已通过的单 compute/单链路路径。
4. **H36/H41/H46/H61**：独立前端性能证据已补齐当前登记边界，均保留 no-safe-production-callsite/无稳定候选边界；后续只有出现新的生产调用方或已登记触发面才重开，严格区分仿真器自身 wall/RSS 和被模拟系统延迟。
5. **H28/H34/H43/H51/H60/H64**：如果确实要发布预测质量结论，再运行 Native 成对误差和留出泛化；否则保持“门控/证据边界修复”。

三份机器审计产物已保存到忽略的 artifacts 目录：

- `audit_records_summary.json` SHA-256 `a153b81b9bababc1491d9592b594a06211e642d0e5802e6453d5278a15f44cf4`
- `audit_frontend_eval.json` SHA-256 `e0171ea7cd8978a536760c165bb607adb2111294d4ad79df96dc8cfff30cbfaa`
- `audit_perf_accuracy.json` SHA-256 `6b9641846f82475e0c8e42178f476ca97b1f15ebb6394284defeaf56b2040b34`

初始审计本身没有执行补测；补证阶段实际执行的 H65/H29/H62/H27/H36 命令和退出码已写入对应 artifacts 与 [EVIDENCE_SUPPLEMENT_STATUS_20261003.md](EVIDENCE_SUPPLEMENT_STATUS_20261003.md)。H65 的自动推进已暂停，后续应由用户逐项批准或继续手动选择。

## 第二批补证结果

- H62 已通过 parent/candidate 隔离的真实 `run_scenario` 和 HTTP `/api/run` 路径补测，但六个同输入场景的 report/API/makespan/NAND task hashes 完全相同；H62 修复点没有在该正式 workload 激活，因此保留为路径健康证据，拒绝“修复有用”结论。
- H27 已通过 parent/candidate 隔离的真实计费前端和 HTTP `/api/run` 补测。serialized/overlapped 中 candidate 的 physical backing 与独立 ceil oracle 一致，`ResourceDemand` 匹配行数提升（GEMM `52→54`、memory `3→6`、reduction `0→15`），但 makespan 不变；接纳为计费机制因果证据，不接纳为速度或精度改善。
- H64 评估/API 门控已用两套隔离源码各执行 8 个真实 HTTP `/api/simulate-score` 场景。parent 对缺失 TTFT/TPOT/E2E 仍错误通过，candidate 全部 fail-closed；这证明 H64 修复进入真实 API，仍不构成 Native 精度证据。H42/H43/H60 状态因正式输入没有触发面，继续标记 `local/not-frontend`。
- 第二批机器摘要：H62 `c043c22e1a6f8e890ab01590db82afd396f474c271c7af8689de1ccb45b9c170`；H27 `9778ad14893442dda20f96ee174fbdf38ab6f47d2c431b395736efde6add99e0`；H64 `03a0b06bc1727eb824f9f932984e4f45572ecc49353b4ed19b15b1ce213467ab`。所有结果均没有加载真实 Native。

- 第三批 H54 已完成 120 个独立进程和 60 对 AB/BA；context-local reuse 在 small/medium/large 的 wall time 分别下降 86.82%/86.24%/86.22%，输出和 report/trace 完全等价，RSS 与 tracemalloc 也下降。由于没有跨请求、Web、run_jobs 或全局缓存的安全生产调用方，结论限定为局部工程收益，不晋升新源码候选。

