# 证据补充状态（2026-10-03）

本文件记录暂停自动推进后实际执行的补证，不把协议、预飞或单元测试写成最终收益证据。定时任务 `llm` 已删除；后续补证需要人工继续选择方向。

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

## 仍未补足的关键边界

- H62 以及 H24/H26/H30/H39/H40/H48/H49/H55：需要跨页/跨请求 queue 和 HTTP/report projection；若声称修复带来收益，还需要父版本与候选版本同一正式前端 A/B。
- H29 的收益目前只属于 CLI 文本前端；H36/H41/H46/H54/H61 仍缺同等级独立进程、重复批次、输出等价和 RSS 证据。
- H27/H38/H44/H52/H58：仍需真实 planner/event/resource/report 计费证据；解析矩阵不能替代前端计费链路。
- H28/H34/H43/H51/H60/H64：若要声称预测精度或泛化改善，必须补同模型/硬件/shape/runtime 的 Native 成对 TTFT/TPOT/E2E 误差、留出和退化统计；当前记录均不构成精度改善。
