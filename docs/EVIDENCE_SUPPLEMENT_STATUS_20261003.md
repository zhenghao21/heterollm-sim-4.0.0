# 证据补充状态（2026-10-03）

本文件记录自动补证任务实际执行的结果，不把协议、预飞或单元测试写成最终收益证据。旧定时任务 `llm` 已删除；当前由 `evidence-supplement` 每十分钟唤醒，按本文件和工作流继续选择方向。

## 已完成

### H65：在线报告 primitive serialization

- 候选提交：`4efc351c77e189b18a11680eb615c443bd8a92e9`；记录收口提交：`dfb0dd9bca4eb7b2edc30689eedb9dd8bf2e4320`；`origin/main` 已核对为同一 SHA。
- 改动：`src/heterollm_sim/reporting.py` 将 `hardware_phase_ledger.guardrails` 从 tuple 改为 list，并在 `tests/test_webui_runtime_contract.py` 增加在线 aggregate 报告严格 JSON roundtrip 回归。
- 实测：focused `56 passed`；全量 `3116 passed, 4 skipped`。
- 独立 post-fix 矩阵：`artifacts/optimization/round_065/serialization_matrix_postfix_independent.json`，SHA-256 `be700a18a4bf2fc9c093b085b7a93f6e3e9223495444a9bd478537fe7a61b535`。在线/静态 report、HTTP `/api/run`、HTTP `/api/simulate-score` 均 roundtrip 相等；缺 Native 仍为 HTTP 400。
- 结论：评估/API 序列化契约修复已接纳。没有 Native、预测精度、吞吐或内存改善结论。

### H29：CLI 文本报告复用的独立性能补证

- 补证协议：`artifacts/optimization/evidence_supplement_20261003/performance/protocol.json`，SHA-256 `1b5a4e680d31644114f9582ee87cae0f78922bcf18a889bedc426ed3a291ce08`。
- 实测：120/120 个独立 CLI 子进程退出码为 0，3 个规模（请求数 1/8/16）各 20 对，AB/BA 交替；每对 stdout bytes/hash 完全相同，120 个 provenance JSON 与 sidecar 均存在。
- 仿真器自身 wall time 中位数：small `150.014→141.290 ms`（`-5.82%`，bootstrap 95% CI `[-12.121,-7.923] ms`）；medium `587.179→523.819 ms`（`-10.79%`，CI `[-68.167,-46.099] ms`）；large `1110.739→973.699 ms`（`-12.34%`，CI `[-145.529,-123.653] ms`）。
- 子进程峰值 RSS：small `46.66→46.43 MB`（`-0.49%`，P90 配对 delta 为正，因此不称小规模内存改善）；medium `64.65→56.71 MB`（`-12.27%`，20/20 更低）；large `85.56→69.87 MB`（`-18.34%`，20/20 更低）。
- 输出等价：三种规模所有配对 stdout 完全相同。结果文件 `h29_cli_supplement_results.json` SHA-256 `fb9b58bef8f8c8a3e4a2917ae95e2c456d8a52adb2a44bdbbc9e75ecf7fafccd`；分析 `h29_cli_supplement_analysis.json` SHA-256 `b9e3b2ca893fbe6d37c67c3205d77604eb9c26635b5d318471bfe8b9842adedd`。
- 结论：可接纳为 H29 **CLI 文本前端**的仿真器 wall-time 稳定改善，中/大规模子进程 RSS 改善。未覆盖 web、run_jobs、全局 report cache；未清理 OS page cache；不代表目标系统 latency 或预测精度。

### H62：NAND/SSD/HBF 共享介质模型的真实前端机制补证

- 产物目录：`artifacts/optimization/evidence_supplement_20261003/storage/`；摘要 `README_summary.json` 及 sidecar；主结果 `frontend_nand_supplement.json` SHA-256 `525ae67816dafb478b30f21bcef177b22bcf57e8e488f820d0292c9650c21947`。
- 实际入口：`plan_runtime_placement → run_scenario(aggregate) → simulate_online → report_dict`，并检查 `compile_scenario` 的任务流。两个 profile：page 4 KiB / 1 plane / Q4 与 page 8 KiB / 4 plane / Q16；每个 profile 18 个真实 `nand_media` 任务。
- 实测：每个 profile 的 `physical_bytes/pages_touched/media_waves/service_ns` 与独立 `nand_media_service` 真值逐任务相等；owner、resource bytes、energy、makespan、完整 report JSON 均已保存并有 sidecar。当前源码身份为 `dfb0dd9bca4eb7b2edc30689eedb9dd8bf2e4320`，功能模块与 H62 修复提交身份一致。
- 限制：`run_scenario` 生成的是页对齐/unknown-offset 访问，已知跨页 offset 只作为独立 oracle；HTTP API 未启动；没有 H62 修复前后 A/B；没有 Native/设备数据。
- 结论：接纳为“当前 NAND/HBF 前端机制与守恒已验证”，不接纳为 H62 修复收益、硬件准确率或 API 覆盖。

### H27：计算成本模型真实计费链路

- 协议：`artifacts/optimization/evidence_supplement_20261003/cost_model/protocol.json`，SHA-256 `4b6f5745d2421791a2578d8b9c775413440494ecf0c8b776a4ab54b9d392e0bb`。
- 实测命令通过 `tools/command_provenance.py` 执行，returncode=0；正式入口为 `build_reference_scenario → compile_scenario → simulate_schedule → run_scenario(aggregate) → report_dict`。主结果 `frontend_cost_billing.json` 的 provenance stdout 给出 SHA-256 `0f721283404bf0b89991f15dde18a2c0ecbc34fa254e7c71bd78d8f00b3553d9`，源代码身份为 `dfb0dd9bca4eb7b2edc30689eedb9dd8bf2e4320`。
- 覆盖：memory/GEMM/reduction 的 serialized/overlapped cost rows；logical/physical bytes、energy、service、owner/resource、queue/makespan、最终 report；zero bytes 通过、duplicate owner demand 与 conflicting capacity 均显式拒绝。
- 限制：HTTP `/api/run` 未启动；没有 Native 或物理设备数据；解析/前端守恒不能作为时延精度或预测精度证据。
- 结论：接纳为 H27 真实 planner→event/resource→report_dict 计费链机制证据；没有 Native 精度、部署吞吐或硬件时延准确性结论。

### H38：计算成本模型边界补证（当前源码，无候选）

- 结果：[h38_frontend_cost_coverage.json](../artifacts/optimization/evidence_supplement_20261003/cost_model/h38_frontend_cost_coverage.json)，SHA-256 `eeeb53febb365258c941a76ed61ac764c7e872c9f42859c896c55e33c1cacb2b`；命令 provenance [h38_frontend_cost_coverage.provenance.json](../artifacts/optimization/evidence_supplement_20261003/cost_model/h38_frontend_cost_coverage.provenance.json)，SHA-256 `e601ce1a75a60e458a9423eaf468b0e25f5af8c0a1546527842b8ba93738caa7`，returncode=0。
- 当前源码为执行命令时的 `3a23a4e2b8cc89cad9422f6a8c205186be74870f`（后续 H44/H58 仅提交记录/docs，成本模块 content/blob SHA 未变）；结果记录写入 `experiments/round_038/supplement_20261003.json`，SHA-256 `dc111e55be5473996e2204750b7d250ccd810cc19a3bc123ae16a2dc03395865`。
- 真实入口：`build_reference_scenario → compile_scenario → simulate_schedule → run_scenario(aggregate) → report_dict`，并启动本地 `build_server(port=0)` 执行 HTTP `POST /api/run`；HTTP 返回 200。
- 独立同输入控制：GEMM 四组 shape、reduction 四组 shape、非法 shape/dtype 拒绝、读写方向带宽/时延（serialized/overlapped）、单资源排队与双 lane 有限流水线；全部控制通过，planner/event/report 与 HTTP 摘要一致。
- 决策：没有发现 H38 新源码缺陷，不修改/不晋升候选；本批只支持成本模型机制、守恒、队列和 API 前端证据，不支持 Native 精度或真实性能结论。
- 未覆盖：MMA/collective/full-kernel 微架构、跨请求生产缓存、真实硬件校准和 Native 成对误差。

## 第二批补证（同一批次，未改源码）

### H62 parent/candidate 前端 A/B

- [storage/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch2/storage/summary.json) SHA-256 `c043c22e1a6f8e890ab01590db82afd396f474c271c7af8689de1ccb45b9c170`。
- 用 H62 parent `6cea26e4d554543306fdd98dbc5857b2045e80ad` 与修复提交 `26d040303b719ef029ef3306341353b5c2d31110` 的隔离 Git archive，2 个 profile、generic no-NAND 控制和双请求 owner/queue，共 6 个同输入场景。
- Python `run_scenario` 和 HTTP `/api/run` 全部成功；输入、report、HTTP payload、makespan 和 NAND task count 在 parent/candidate 之间完全相同。
- 结论：正式前端路径健康，但没有激活 H62 的 `_direct_memory_phase` NAND-bearing direct GPU GEMM；不能接纳 H62 修复收益。known-offset 跨页仍是独立 oracle，未伪造为前端证据。

### H27 parent/candidate 成本计费 A/B

- [cost_model/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch2/cost_model/summary.json) SHA-256 `9778ad14893442dda20f96ee174fbdf38ab6f47d2c431b395736efde6add99e0`。
- parent `9c2b21a6811fca1517df1922eee45a3eaaf417f4` 与 candidate `f6457f07f8f44b1d8c70d4a12ee637a65d1cbdfd` 在 analytical/serialized/overlapped 三种模式下各跑一次静态链和一次独立 HTTP `/api/run`，六次 HTTP 均为 200。
- serialized/overlapped 中，candidate 的 `ResourceDemand` 与物理 backing bytes 匹配行数相对 parent：GEMM `52→54`、memory `3→6`、reduction `0→15`；candidate `resource_accounted_bytes` 增加 `11034`，energy 增加 `56040`，makespan 保持 `442041.55833333335 ns`。
- 结论：H27 计费守恒和修复因果差异已由真实前端 A/B 支持；没有速度、Native 精度或设备时延结论。前两次 runner 失败及最终成功 provenance 均保留。

### H64 评估指标完整性 HTTP A/B

- [evaluation/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch2/evaluation/summary.json) SHA-256 `03a0b06bc1727eb824f9f932984e4f45572ecc49353b4ed19b15b1ce213467ab`。
- parent `23b56f67e4f44f1663a4044c9be14055372f32d5` 与 candidate `fb71035e81be1273741c795d3d7805065fa0bb77` 各执行真实 HTTP `/api/simulate-score` 8 个场景。
- parent 对缺失 TTFT/TPOT/E2E 都错误返回 `compared/passed=true`；candidate 返回 `incomplete_metric_reference/passed=false` 并给出 `missing_metrics`。complete、partial、empty、N=1 TPOT-NA、missing-native 400 控制均正确。
- H42/H43/H60 的 OOD/coverage/support 状态没有正式输入触发面，保留为 `local/not-frontend`；self-reference 只证明 HTTP 契约，不构成 Native 误差。

## 仍未补足的关键边界

## 第三批补证（当前自动任务）

### H62 direct billing 激活性扫描

- 摘要：[batch3/storage/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch3/storage/summary.json)，SHA-256 `9ffc0825f421b8f7961dd963e9d5c0c1798698bc06b809e5ad38cfc2a59b5ec9`。
- 对 parent `6cea26e4d554543306fdd98dbc5857b2045e80ad` 与 candidate `26d040303b719ef029ef3306341353b5c2d31110` 做 source scan 和同输入隔离执行。
- 正式触发条件已核对：direct storage/device 匹配、local backing demand/resource、读写字节非零、GPU rank GEMM 路径。
- 两版本 control-plane placement 和 compile 均成功，任务数均为 377，但 `direct_memory_access_count=0`、`nand_task_count=0`，状态均为 `not_activated`。
- 结论：保留 H62 direct billing 的正式触发边界；没有伪造激活，也没有新增因果收益结论。

### H52 collective 成本模型激活性扫描

- 摘要：[batch3/cost_model/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch3/cost_model/summary.json)，SHA-256 `613d5a6184a9327f202960825b1ee1d71d464f24a4752f5176d56d03c9daecd0`。
- parent `b00523f9` 与 candidate `6cdf5fc4` 的正式 reference planner/event/report 两侧均完成 326 tasks，HTTP `/api/run` 均返回 200；但 28 个 collective rows 都是 local zero-demand。
- 尝试 `tp_degree=4` 时两版本均因 reference topology 只有一个 compute component 而 `ScenarioValidationError`，这是明确的拓扑阻塞。
- 独立 `plan_collective` route oracle：ring/tree/auto、4 个 participant、4096 B all-reduce 均为 24576 B，等于 `4096×2×(4−1)`；非法算法、负 bytes、重复 participant、owner capacity 冲突和 duplicate demand 均 fail-closed。
- 结论：保留 collective 本地 oracle 和 topology blocker；不把 local direct matrix 写成正式端到端 collective 证据，不声称 Native 带宽或性能改善。

### H54 context-local reuse 运行速度/内存补证

- 摘要：[batch3/performance/summary.json](../artifacts/optimization/evidence_supplement_20261003_batch3/performance/summary.json)，SHA-256 `89309d19594ee8b61c87cd422b15a78ddf70e744ebb1f1a4b81ac0ec3ec35bc7`。
- 使用 H54 源码 `982e539167bd89b85cba620ef95a70c61ecea1de`，fresh/reused 两种模式，small/medium/large 三种规模，各 20 对 AB/BA，共 120 个独立子进程；全部退出码为 0，120 个 provenance JSON 和 sidecar 全部匹配。
- wall time：small `263.565→34.730 ms`（-86.82%）、medium `2110.427→290.366 ms`（-86.24%）、large `4239.655→584.133 ms`（-86.22%）。RSS：-2.54%/-4.26%/-3.58%；tracemalloc：-57.39%/-37.12%/-36.94%。
- 每对 task signature、metadata、demands、report、trace、makespan 和 resource busy 完全相等。
- 结论：确认 `CompilationContext` 场景内复用有稳定工程收益；由于没有跨请求/Web/run_jobs/global cache 的安全生产调用方，本批不晋升新源码候选，也不写成生产级优化或 Native 精度改善。

### H36 run_scenario provider reuse 独立前端 A/B

- 初始记录 [supplement_20261003_frontend.json](../experiments/round_036/supplement_20261003_frontend.json) 的 RSS 采样只读到了 `.venv` launcher，已保留但其内存结论作废；校正记录 [supplement_20261003_frontend_pidfix.json](../experiments/round_036/supplement_20261003_frontend_pidfix.json)，SHA-256 `f72c1eb7835cf86fb65469d813e67a036f39ca7333a83fede047fec7f54bf21b`，校正矩阵 SHA-256 `6a89836140fb1bdcbc6c5227119eaf8b9b365155066032b243d55ccfa7079a61`。
- 校正执行源码 `8e6aa2b3307da25a6eabc1b0611de5183ef0429b` 未修改；冷路径是实际 `run_scenario` 默认 provider，暖路径是在独立子进程内显式 `TopologyAwareBatchCostProvider` 预热后第二次 `run_scenario`。3 个规模、20 对 AB/BA，共 40 个独立子进程，全部退出码 0。
- 每对 `report_dict` JSON hash、makespan、batch_count、request_count 完全相等（20/20）；校正后的仿真器 wall 中位数暖/冷：1×64×4 `0.0377/0.0897 s`（`-57.98%`），8×128×16 `0.0420/0.1720 s`（`-75.56%`），16×128×16 `0.0465/0.1847 s`（`-74.85%`）。
- 进程树峰值 RSS 中位数暖相对冷分别 `+1.98%`、`+1.43%`、`+5.25%`，只有 2/20 对暖路径更低；因此撤销所有 RSS 改善结论。
- 结论：接纳为同一显式 provider 生命周期内的局部工程 wall-time/输出等价证据；没有共享 provider 的生产生命周期、场景身份和淘汰契约，拒绝新增生产缓存候选、Web/run_jobs/global cache 或 Native 精度结论。

### H46 leaf-cache 配置独立前端 A/B

- 补证记录：[experiments/round_046/supplement_20261003_frontend.json](../experiments/round_046/supplement_20261003_frontend.json)，记录 SHA-256 `142e0fbcfba07df6b2b8b9f32db853a8faa37a1bb94c8cdbd537d5ef947e02be`；矩阵 [h46_frontend_cache_ab.json](../artifacts/optimization/round_046/h46_frontend_cache_ab.json)，SHA-256 `bfa9cd2b999f75a43f81a2bb7f32220752b35b21f1ad6dfcebcd5a91fffc89a0`。
- 同一 8-request `run_scenario` workload 下，cached_plan（leaf cache 4096）与人为 bounded cache（1）各 6 个独立 child，共 12 个进程；子进程 returncode 12/12 为 0，report/trace/batch/schedule hash、task_count、makespan、bytes、energy 全部相等。
- cached_plan 相对 bounded cache 的 subprocess wall 中位数 `11.151→13.476 s`（约 `-17.17%`），tracemalloc 峰值 `35.53→36.16 MB`；RSS 采集为 before/after delta（`82.25→87.52 MB`），不是独立系统 peak working set。
- 结论：只保留 leaf-cache 配置的工程机制证据；当前生产默认已是大 cache，本批没有新增源码候选，也不把配置差异外推为稳定生产优化、Native 延迟或预测精度。分析脚本的 trailing-literal 失败（exit 1）已保留，原始矩阵未受影响。

### H61 run-jobs HTTP 独立前端 A/B 与 RSS 校正

- 首版记录 [supplement_20261003_runjobs.json](../experiments/round_061/supplement_20261003_runjobs.json) 保留；其 launcher-only RSS 结果不作为内存结论。校正记录 [supplement_20261003_runjobs_tree_rss.json](../experiments/round_061/supplement_20261003_runjobs_tree_rss.json)，SHA-256 `c5dfbd05e67f7d25faec64fb2406aa97fc38894e1de3b5309fe09f704fd04986`。
- 真实入口为 `POST /api/run-jobs → GET /api/run-jobs/{job_id} → RunJobManager → run_scenario → report_dict`；8/32/64 requests 各 3 对 AB/BA，每 child 3 次 job，共 18 个独立 child，全部 rc0，canonical report SHA/bytes 全部相等。
- 进程树 RSS 校正后，cached wall 相对 baseline 在 8/32/64 requests 为 `-0.58%/-0.54%/-0.01%`，近似中性；tree RSS 为 `-2.12%/+1.69%/+5.23%`，方向随规模变化且大规模退化。结论为有限负结论，不晋升缓存源码候选或稳定速度/内存改善。
- 仍未覆盖 `/api/run` 直接路径、多客户端并发扇出、跨请求/global cache 安全、Native/目标系统 latency；旧 launcher-only 产物和失败边界均保留。

### H41 retention/metadata 独立前端 A/B

- 补证记录：[experiments/round_041/supplement_20261003_retention.json](../experiments/round_041/supplement_20261003_retention.json)，SHA-256 `83b18ddada5287f3f41670bcae592362bf54de5df46e5a633ac2e1cbd34364b4`；矩阵 [h41_retention_metadata_ab.json](../artifacts/optimization/round_041/h41_retention_metadata_ab.json)，SHA-256 `b61e2e8362fa8c4013e92f116c8ca472bd2407908b75e5b04aa747fddfbfee4b`。
- 真实入口 `build_reference_scenario → run_scenario(retention_policy) → compile_streaming_scenario → execute_incremental_schedule → report_dict`；small/medium 两种规模、exact/streaming/aggregate 三种策略，AB/BA 独立 child 共 12 次，全部退出码 0。
- 同策略 report/metadata hash 在重复进程中完全相等；跨策略 task_count 与 makespan 完全相等，保留任务数严格符合 exact=全部、streaming=min(total,2000)、aggregate=0。wall/RSS 只作仿真器工程观测，未形成稳定源码候选。
- 结论：H41 的 retention 语义与真实前端可复现性已补足为有限负结论；不同 retention 的 report/trace 差异是输出契约，不能强行归一成全报告等价，也没有 Web/run_jobs/global cache 或 Native 精度结论。

### H28/H34/H43/H51/H60/H64 Native 成对精度 readiness 阻塞

- 补证记录：[experiments/round_064/supplement_20261003_native_readiness.json](../experiments/round_064/supplement_20261003_native_readiness.json)，SHA-256 `c45c3379d689098759b3aa9e321dffbd354f2603531d0f797d70d44d93cd4e74`；readiness artifact `native_readiness_scan.json` SHA-256 `232d44383ebda29371e4f552913670238c4651df57c2b8c5587fb1541374b173`。
- 已核对当前源码 `0dd393025a78952499474364f17f53bf03f179e5`、Native 可执行文件与模型文件的 SHA；真实入口 `tools/native_llama_compare.py` / `tools/native_error_matrix.py` 仅执行 `--help`，命令退出码均为 0，未运行 Native 计时。
- 当前没有同一模型/硬件/shape/runtime/timer 边界下的 simulator↔Native 成对 TTFT/TPOT/E2E、APE、覆盖、回退和留出数据；历史 freeze 为 `development_post_selection` 且 `blind_evaluation=false`，源码身份早于当前 HEAD，不能复用为盲测或当前精度证据。
- 结论：H28/H34/H43/H51/H60/H64 均为 `blocked_no_current_same_source_native_pair`。本补证只证明数据与入口阻塞，不证明精度改善；下一步必须先冻结当前 workload/runtime 合约，再独立运行 simulator 与 Native 成对数据。

- H62 以及 H24/H26/H30/H39/H40/H48/H49/H55：需要跨页/跨请求 queue 和 HTTP/report projection；若声称修复带来收益，还需要父版本与候选版本同一正式前端 A/B。
- H62 当前已完成 parent/candidate 前端 A/B，但修复点未激活；仍需设计能实际进入 `_direct_memory_phase` 的正式 workload，不能把现有六场景写成收益。
- H62 新 workload2 尝试已保留：[experiments/round_062/supplement_20261003_workload2.json](../experiments/round_062/supplement_20261003_workload2.json)，SHA-256 `900ffe7795d30c621ded5a4eee52712782bc16f75ef6afabfdd08ae0d7f7581f`。该输入采用 `_nand_case` 的 HBF/NAND direct exposure 与 GPU GEMM，但在 `compile_scenario` 前置校验被合法拓扑和 `tensor_bytes[model_weights]` 声明阻塞，未进入 event/run/report；初始 import、syntax 和最终 validation 失败均保留，未改源码。
- H29 的收益目前只属于 CLI 文本前端；H36 校正后只保留 wall/output 等价证据且无安全生产共享 provider 候选；H46 已完成 leaf-cache 配置级独立 A/B 但无源码候选；H61 已完成真实 `/api/run-jobs` A/B 与进程树 RSS 校正，仍无稳定收益；H41 已补 retention 前端边界并保留有限负结论。
- H27、H38、H44 已完成真实 planner/event/resource/report 前端补证；H52 已完成正式 topology/HTTP 激活性扫描但 collective 仍被单计算组件拓扑阻塞；H58 已完成 routed LinkService 前端机制补证但无候选差异。解析矩阵不能替代前端计费证据。
- H28/H34/H43/H51/H60/H64：若要声称预测精度或泛化改善，必须补同模型/硬件/shape/runtime 的 Native 成对 TTFT/TPOT/E2E 误差、留出和退化统计；当前记录均不构成精度改善。

### H58：LinkService/transfer pipeline 真实前端补证（本批）

- 补证记录：[experiments/round_058/supplement_20261003_frontend.json](../experiments/round_058/supplement_20261003_frontend.json)，记录 SHA-256 `cd1273e063318f6824a0a0b2c8cdbdca7fd6de1fc188457f66e29808869d236a`，sidecar 同目录。
- 同输入隔离执行 parent `d859ae52a1b0f9d3e37346ca1eb3c4b6144afbe5` 与 candidate `3a23a4e2b8cc89cad9422f6a8c205186be74870f`：`build_reference_scenario → resident_access_path=topology → compile_scenario → simulate_schedule → run_scenario(aggregate) → report_dict`，两命令均退出码 0。
- 真实路由激活：每侧 326 tasks、96 个 routed resident-memory tasks；`link.gpu-hbm0.gpu0->hbm0`/反向资源、`bytes_moved=444096520`、`energy_pj=1731888883.1`，compiled makespan `429064.203125 ns`。task/report/result SHA 及 resource busy 全部同输入相等。
- 结论：H58 链路 finite/owner/capacity 成本路径已进入真实 compile→event→run→report 前端；parent/candidate 无差异，没有源码候选或因果收益，不能外推 Native 带宽/延迟、仿真器速度或预测精度。未覆盖多链路/多请求竞争、HTTP `/api/run` 和真实硬件观测。
- 机器产物（忽略目录）：`artifacts/optimization/round_058/frontend_link_billing_supplement.json` SHA-256 `5fdd88cb07a0caf80010226df43114d448a7c733e10ca61172b9d5952437d84a`；父/候选 raw JSON SHA 均 `b8585e0b31a8ea65636f7b8478905225d9a222d1fd8307529d52aaf2867e1de7`。

### H44：MMA/full-kernel 真实前端补证（本批）

- 补证记录：[experiments/round_044/supplement_20261003.json](../experiments/round_044/supplement_20261003.json)，记录 SHA-256 `c084487041e42ae6f01cc0b5b081e4c4bbc8e1ec927fc50d4ff9ec1c11ff5c3d`。
- 独立子进程入口 `build_reference_scenario → compile_scenario → simulate_schedule → run_scenario(aggregate) → report_dict`，命令退出码为 0；12/12 fused-attention 与 96/96 MMA GEMM 任务进入 event/resource/report，schedule/trace 均为 326 tasks，makespan `223915.48190045252 ns`。
- 28 个 collective task 在当前单 compute reference topology 下全部 local zero-demand（active=0）；因此只接纳 MMA/full-kernel 机制激活证据，保留 collective topology blocker，不作 Native 带宽、时延、性能或精度结论。

