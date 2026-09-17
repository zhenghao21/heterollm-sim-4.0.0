# 前端—后端适配审计与实机验收（2026-09-15）

## 结论

已修复六类前端适配/交互问题；完成真实浏览器 → 本地 HTTP API → V4 事件内核 → 报告 → 回放的完整运行。
最终 364 项前端测试、116 项后端接口/运行/报告回归测试通过。最后一轮浏览器控制台无错误/警告，运行请求全部返回 HTTP 200/202。
保留了用户原来的模型、硬件、Profiles、负载和策略参数；验收新增的第二条请求与静态模式已撤回，最终页面显示原始单请求配置的已完成结果。

## 已修复项

| 优先级 | 问题与影响 | 修复 |
|---|---|---|
| P1 | 编辑硬件后旧“映射过期”门禁阻止运行，界面却没有手动刷新放置入口 | 与 V4 对齐：每次运行由后端动态计算放置，提交前仍执行后端拓扑、容量、模型校验；不再把历史放置结果作为新运行前提 |
| P1 | 成功运行后映射页仍显示 0 算子、0 张量；历史输出回传还会触发过期检查 | 报告增加只读 `runtime_placement`，前端从当前报告读取实际 Rank/算子/张量数据；请求中移除旧 decision/evidence，保留 policy 和其他元数据；绝不把运行输出写回人工 op/tensor placement |
| P2 | 本地场景序列化/Profiles 引用错误发生在 try 外，产生未处理 Promise 拒绝 | 将校验、估算、启动任务的序列化放进统一异常处理；错误展示后按钮可重试 |
| P2 | GPU 比较可能覆盖仍在执行的任务或编辑后的新场景 | 活动任务期间阻止竞争比较；比较响应按场景代次检查，过期响应丢弃；旧任务完成不会更新新场景的映射 |
| P2 | 事件明细折叠时，单步或播放后的“上一步/下一步”状态不刷新 | 将导航按钮状态绑定到全局事件选择，不再依赖可选事件表格是否渲染 |
| P2 | 静态 exact 轨迹在侧栏和回放说明中被标成聚合或无详细轨迹 | 按后端 fidelity 分别标注精确/代表性/聚合事件；静态报告内事件显示实际载入数，不冒充聚合模式 |

## 后端兼容合同

- `/api/validate` 与 `/api/run-estimate` 接收纯 V4 authoring 场景。
- `/api/run-jobs` 仍使用 `{scenario, retention_policy}`；continuous 使用 aggregate，static 小场景使用 exact。
- `/api/run-jobs/{job_id}` 完成报告中的 `runtime_placement` 使用 `schema_version: runtime-placement/v1`、`read_only: true`，含 `parallel` 与 `control_plane`。
- 该字段仅是结果视图，不是 V4 authoring 新输入；不放宽人工 placement、拓扑、容量、模型或参数合法性限制。
- 未改变仿真成本模型、任务调度定义、校准参数或用户已有的后端实验改动。

## 真实浏览器运行结果

| 场景 | 实际输入 | 完成结果 | 仿真总历时 | 首 Token p50 |
|---|---|---|---|---|
| 原始连续批处理 + MTP | 1 请求，64 输入 Token，4 输出 Token | 651 个任务，4 个批次，4 个可见输出 Token | 240.845873 µs | 110.279978 µs |
| 静态异长双请求 + MTP | 请求一 64/4；请求二 32/8；合计 96 输入、12 输出 Token | 1,184 个任务/精确事件，2 请求全部完成 | 338.231179 µs | 76.449900 µs |

两种模式均显示实际 36 个算子组、8 个权重张量组。记录中的 run ID、job ID、保留策略和精确数值见 `verification_summary.json`。

实机额外检查：
- 通过界面键盘编辑显式请求并核对后端估算收到的 96/12 Token。
- 连续批次按需详细回放：prefill 153 个事件，MTP 批次 193 个事件，加载接口 HTTP 200。
- 播放/暂停、前后单步、跨批次切换正常；默认折叠事件明细时前后按钮能正确更新。
- 静态轨迹显示 1,184/1,184 条精确事件及正确说明。
- GPU 基线比较与标准 IR 导出经前端操作完成。
- 修改配置后旧报告作废；回归测试覆盖后台旧结果、比较旧响应和本地输入错误。

## 测试与证据

前端命令：`node --test --test-reporter=spec tests/webui_*.test.cjs`（364 passed）。

后端使用本项目 `.venv/Scripts/python.exe`：
`-m pytest -q tests/test_web.py tests/test_v12_web_errors.py tests/test_ui_v03.py tests/test_v12_architecture_api.py tests/test_run_jobs.py tests/test_webui_runtime_contract.py tests/test_end_to_end.py tests/test_engine_metrics.py tests/test_component_timeseries.py`（116 passed）。

全部证据目录：`F:/codex_project/37_LLMsim/artifacts/frontend_audit_20260915/`。

- `frontend_verified.log` / `backend_verified.log`：最终回归结果。
- `verification_summary.json`：运行断言、指标和本次源码 SHA-256。
- `continuous_job.json` / `static_job.json`：真实 HTTP 完成报告。
- `prefill_trace.json`：连续批次按需详细事件响应。
- `http_trace.json` / `browser_errors.json`：最后一轮真实请求记录与控制台检查。
- `final_results_dom.txt`：最终结果页的可访问性树文本。
- `final-results.png` / `static-playback.png` / `trace-playback.png`：实机页面证据。
- `original_browser_scenario.json` / `final_authoring.json`：测试前后参数，已检查仿真语义一致（忽略纯界面布局/状态元数据）。

## 边界与未覆盖项

- 这是前端接入当前后端的功能验收，不是全仓库穷尽测试、全部硬件组合验收或真实 GPU 预测精度认证。
- 结果始终为 ANALYTICAL；exact 只表示保留完整事件，不代表周期级或硅后测量精度。
- 运行保留两条既有 CIM 提示：按显式详细权重校验容量、采用权重常驻热启动语义；它们不是运行错误。
- 当前参考场景未声明 host-output contract，报告保留 partial 标识，不据此宣称覆盖真实客户端完整输出链路。
- 外部 Hugging Face 在线目录/下载与 native llama.cpp 校准矩阵不在本次前端验收范围；本次未重新跑这些实验。
- 测试工具会话曾在验收后断开，本地页面重新连接后已恢复原始参数并再跑完整流程；已保存的独立证据不依赖旧会话。
