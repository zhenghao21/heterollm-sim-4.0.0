# Round 069 kernel/operator calibration error comparison

APE = `|simulator - Native| / Native × 100%`; positive improvement pp means B is closer. Kernel holdout rows are separate from engine TTFT/TPOT/E2E rows.

| 模型 | scope | 状态 | 指标 | Native ms | A ms | B kernel ms | A APE | B APE | 改善 pp | exact 命中 | changed demand | ledger |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen35 | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_partial | ttft | 131.306500 | 273.416590 | 285.719968 | 108.227765% | 117.597734% | -9.369969 | 48 | 220 | not_applicable (GPU replay no authored DRAM/NAND) |
| qwen35 | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_partial | tpot | 8.034292 | 6.262536 | 6.323983 | 22.052419% | 21.287609% | 0.764810 | 48 | 220 | not_applicable (GPU replay no authored DRAM/NAND) |
| qwen35 | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_partial | e2e | 519.616500 | 648.510222 | 654.378391 | 24.805548% | 25.934875% | -1.129327 | 48 | 220 | not_applicable (GPU replay no authored DRAM/NAND) |
| qwen38 | target_long_p4_r1_engine | fresh_cpu_operator_blocked_exact | ttft | 11904.233000 | 13022.257097 | 13022.257097 | 9.391820% | 9.391820% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| qwen38 | target_long_p4_r1_engine | fresh_cpu_operator_blocked_exact | tpot | 1258.817429 | 331.049415 | 331.049415 | 73.701555% | 73.701555% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| qwen38 | target_long_p4_r1_engine | fresh_cpu_operator_blocked_exact | e2e | 21300.966000 | 17198.816166 | 17198.816166 | 19.258046% | 19.258046% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| tinyllama | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_shape_domain_miss | ttft | 31.041000 | 31.057108 | 31.057108 | 0.051894% | 0.051894% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| tinyllama | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_shape_domain_miss | tpot | 4.862802 | 2.606033 | 2.606033 | 46.408822% | 46.408822% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| tinyllama | target_long_p4_r1_engine | fresh_current_binary_kernel_exact_shape_domain_miss | e2e | 266.495000 | 156.146689 | 156.146689 | 41.407272% | 41.407272% | 0.000000 | 0 | 0 | passed (A/B invariant) |
| qwen25 | kernel_holdout | fresh_current_binary_kernel_exact_covered | kernel_holdout_summary | n/a | n/a | n/a | n/a% | n/a% | n/a | 31 | 31 | passed (kernel-only zero ledger + storage companion) |
| qwen35 | kernel_holdout | fresh_current_binary_kernel_exact_partial | kernel_holdout_summary | n/a | n/a | n/a | n/a% | n/a% | n/a | 48 | 220 | not_applicable (GPU replay) |
| smollm2 | kernel_holdout | fresh_current_binary_kernel_exact_covered | kernel_holdout_summary | n/a | n/a | n/a | n/a% | n/a% | n/a | 28 | 28 | passed (kernel-only zero ledger + storage companion) |
| tinyllama | kernel_holdout | fresh_current_binary_kernel_exact_covered_target_domain_miss | kernel_holdout_summary | n/a | n/a | n/a | n/a% | n/a% | n/a | 0 | 0 | passed (A/B invariant) |
| qwen38 | kernel_holdout | fresh_cpu_operator_blocked_exact | kernel_holdout_summary | n/a | n/a | n/a | n/a% | n/a% | n/a | 0 | 0 | passed (A/B invariant) |

## Coverage table

| 模型 | train/holdout events | profile entries | common exact | exact hits | changed demand | median/P90/max kernel APE |
|---|---:|---:|---:|---:|---:|---|
| qwen25 | 16048 / 8272 | 75 | 31 | 31 | 31 | 0.1276% / 0.7176% / 2.2384% |
| qwen35 | 9534 / 11000 | 94 valid / 9564 blocked | n/a | 48 | 220 | 0.3227% / 1.2551% / n/a |
| smollm2 | 15905 / 8129 | 58 | 28 | 28 | 28 | 0.1792% / 1.8488% / 12.0590% |
| tinyllama | 13078 / 6678 | 38 | 38 | 0 target / 38 profile | 0 target | 0.1712% / 0.9611% / 1.9393% |
| qwen38 | 6684 / 6668 CPU operators | 0 | 0 | 0 | 0 | n/a (layout/kernel_family missing) |

## 结论边界

- Qwen2.5 与 SmolLM2 的 fresh current-binary kernel exact holdout 已成立；这不是模型级校准。
- Qwen3.5 的 kernel holdout 误差很小，但目标 long/p4 仅局部命中，TPOT APE 改善 0.765 pp，TTFT/E2E 退化，不能称整体改善。
- TinyLlama 的 fresh profile 有 38 个共同 exact key，但目标 long/p4 replay 没有匹配 demand；这是真实 shape-domain miss，不是 profile 缺失。
- Qwen3.8 的当前 CPU direct capture 明确缺少 layout 与 kernel_family，严格 exact gate 阻止启用；没有把 CPU trace 当成 GPU kernel profile。
- DRAM/NAND ledger 由硬件前端负责；kernel-only A/B 只改变 compute demand，traffic/bytes/pages/queue invariants 保持不变。
