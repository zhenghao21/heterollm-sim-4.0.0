# 原生后端论文对照实验

更新：2026-10-26T10:12:23.984687+00:00；状态：失败。
配置 30 组，成功 0，全候选拒绝 0，失败 0。

每配置固定 8 GPU；部署按论文式搜索：旁路模型筛选布局并给出批量区间，原生在旁路排名前2且原生可组合的布局上各测2–3个批量点，把TPOT上界线性插值到100ms，再在最佳布局的插值批量上做一次原生验收；最终数字只取原生候选，旁路数值只用于放置批量点。够不到100ms的布局停在已测最大点并标注受限，不外推；验收未通过时退回已测通过点。DP 副本是互相独立的拷贝，只仿真一个副本，集群吞吐为单GPU吞吐乘GPU总数。每配置选择通过 100ms 请求均值TPOT约束的最高预算接受的预测单GPU吞吐，不表示全局最优部署或最大批量。
完整原生批次样本与预测窗口分开保存；10%为明示条件模型预算，不是未执行全窗口的实测保证。预测尾段没有物理轨迹，预算拒绝时指标为空，不回退详细后缀。
预测吞吐使用相同测量窗口及可见Token定义，TPOT使用窗口内完整新请求的首末Token预测区间；预测完成不计为物理完成。
TTFT = Prefill通路 + 入口TTFT。Prefill副本为HBM4、与Decode副本相同的TP/PP/EP布局，请求到达即单独一次前向计算前P-1个提示Token（不排队、不分块），到最后一层完成为止（此时开始导出KV）；入口TTFT从该请求实付KV导入开始到首个Token，Decode侧计算最后一个提示Token并给出首Token。Prefill通路为分层采样区间，与入口TTFT区间相加，仍满足同一独立预算。
DRAM/NAND/SRAM与PD真实账本只覆盖采样前缀；全窗口物理流量未知。预测尾段不生成虚构的流量、利用率或物理状态。

累计Token、批次及阶段壁钟见[进度分析](progress_analysis.md)；真实慢路径调查见[原生后端对照说明](../../docs/exploring_hbf_native_comparison.md)。局部事务提速不能代替完整候选耗时验收。

| 模型 | 存储 | 输入/输出Token | 配置状态 | 选中布局 | 选中批量/副本 | tokens/s/GPU | 集群tokens/s | 请求均值TPOT(ms) | 请求均值TTFT(ms，含Prefill) | 搜索结果 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| Llama 3.1 405B | HBM4 | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBM4 | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBM4 | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV+ | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV+ | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | CONV+ | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF+ | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF+ | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 3.1 405B | HBF+ | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBM4 | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBM4 | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBM4 | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV+ | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV+ | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | CONV+ | 103500/1100 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF+ | 1660/373 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF+ | 5900/499 | 运行中 | — | — | — | — | — | — | — |
| Llama 4 Maverick | HBF+ | 103500/1100 | 运行中 | — | — | — | — | — | — | — |

失败原因：{'type': 'PermissionError', 'message': "[Errno 13] Permission denied: 'artifacts\\\\exploring_hbf_native_20261026_v21\\\\progress\\\\native__llama3_405b__CONV__SHORT__G8__DP1_TP8_PP1_EP1__B180.json'", 'traceback': 'Traceback (most recent call last):\n  File "F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\scripts\\run_exploring_hbf_native.py", line 1759, in main\n    return run_search_manager(directory, workers=args.workers, case_ids=args.cases)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\scripts\\run_exploring_hbf_native.py", line 1660, in run_search_manager\n    code = _poll_candidate(directory, identity, job, mode)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\scripts\\run_exploring_hbf_native.py", line 1389, in _poll_candidate\n    observed = read_json(progress_path, {})\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\scripts\\run_exploring_hbf_native.py", line 134, in read_json\n    return json.loads(Path(path).read_text(encoding="utf-8"))\n                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "C:\\Users\\A\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\Lib\\pathlib.py", line 1027, in read_text\n    with self.open(mode=\'r\', encoding=encoding, errors=errors) as f:\n         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "C:\\Users\\A\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\Lib\\pathlib.py", line 1013, in open\n    return io.open(self, mode, buffering, encoding, errors, newline)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nPermissionError: [Errno 13] Permission denied: \'artifacts\\\\exploring_hbf_native_20261026_v21\\\\progress\\\\native__llama3_405b__CONV__SHORT__G8__DP1_TP8_PP1_EP1__B180.json\'\n'}
