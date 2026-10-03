# Round 069 全量误差对比表

计算定义：signed error = simulator − Native；APE = |signed error| / |Native| × 100%；改善 pp = A APE − B APE，正数表示 B 更接近 Native。

## Engine TTFT / TPOT / E2E（主表）

|模型|scope|状态|指标|Native ms|A baseline ms|B kernel ms|A signed ms|B signed ms|A abs ms|B abs ms|A APE|B APE|改善 pp|exact hits|changed demands|binary|native runtime|sim runtime|hardware|ledger|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
|qwen25|target_long_p4_r1_engine|fresh_current_binary_exact|ttft|31.631000|15.201549|15.201549|-16.429451|-16.429451|16.429451|16.429451|51.940977%|51.940977%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen25|target_long_p4_r1_engine|fresh_current_binary_exact|tpot|4.939625|3.057221|3.057221|-1.882404|-1.882404|1.882404|1.882404|38.108247%|38.108247%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen25|target_long_p4_r1_engine|fresh_current_binary_exact|e2e|265.912500|161.896427|161.896427|-104.016073|-104.016073|104.016073|104.016073|39.116654%|39.116654%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen35|target_long_p4_r1_engine|fresh_current_binary_exact|ttft|131.306500|273.416590|276.366036|142.110090|145.059536|142.110090|145.059536|108.227765%|110.473995%|-2.246230|220|220|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen35|target_long_p4_r1_engine|fresh_current_binary_exact|tpot|8.034292|6.262536|6.323983|-1.771755|-1.710309|1.771755|1.710309|22.052416%|21.287609%|0.764807|220|220|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen35|target_long_p4_r1_engine|fresh_current_binary_exact|e2e|519.616500|648.510222|654.378391|128.893722|134.761891|128.893722|134.761891|24.805548%|25.934875%|-1.129327|220|220|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|smollm2|target_long_p4_r1_engine|fresh_current_binary_exact|ttft|38.460500|41.813951|41.813951|3.353451|3.353451|3.353451|3.353451|8.719209%|8.719209%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|smollm2|target_long_p4_r1_engine|fresh_current_binary_exact|tpot|6.236010|6.372445|6.372445|0.136434|0.136434|0.136434|0.136434|2.187844%|2.187844%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|smollm2|target_long_p4_r1_engine|fresh_current_binary_exact|e2e|334.314500|347.515725|347.515725|13.201225|13.201225|13.201225|13.201225|3.948744%|3.948744%|0.000000|26|26|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|tinyllama|target_long_p4_r1_engine|fresh_current_binary_exact_shape_miss|ttft|31.041000|31.057108|31.057108|0.016108|0.016108|0.016108|0.016108|0.051894%|0.051894%|0.000000|0|0|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|tinyllama|target_long_p4_r1_engine|fresh_current_binary_exact_shape_miss|tpot|4.862802|2.606033|2.606033|-2.256769|-2.256769|2.256769|2.256769|46.408822%|46.408822%|0.000000|0|0|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|tinyllama|target_long_p4_r1_engine|fresh_current_binary_exact_shape_miss|e2e|266.495000|156.146689|156.146689|-110.348311|-110.348311|110.348311|110.348311|41.407272%|41.407272%|0.000000|0|0|True|True|True|True|not_applicable (GPU replay; DRAM/NAND ledger not authored)|
|qwen38_gpu|fresh_current_gpu_train_p14_o8|fresh_current_binary_gpu_exact_diagnostic|ttft|91.149000|51.124973|51.124973|-40.024027|-40.024027|40.024027|40.024027|43.910550%|43.910550%|0.000000|72|72|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|fresh_current_gpu_train_p14_o8|fresh_current_binary_gpu_exact_diagnostic|tpot|26.469857|53.257067|53.257067|26.787210|26.787210|26.787210|26.787210|101.198921%|101.198921%|0.000000|72|72|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|fresh_current_gpu_train_p14_o8|fresh_current_binary_gpu_exact_diagnostic|e2e|276.438000|423.924442|423.924442|147.486442|147.486442|147.486442|147.486442|53.352449%|53.352449%|0.000000|72|72|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|fresh_current_gpu_holdout_p18_o4|fresh_current_binary_gpu_exact_diagnostic|ttft|92.843000|57.365168|57.365168|-35.477832|-35.477832|35.477832|35.477832|38.212716%|38.212716%|0.000000|68|68|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|fresh_current_gpu_holdout_p18_o4|fresh_current_binary_gpu_exact_diagnostic|tpot|26.515667|53.231683|53.231683|26.716016|26.716016|26.716016|26.716016|100.755590%|100.755590%|0.000000|68|68|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|fresh_current_gpu_holdout_p18_o4|fresh_current_binary_gpu_exact_diagnostic|e2e|172.390000|217.060217|217.060217|44.670217|44.670217|44.670217|44.670217|25.912302%|25.912302%|0.000000|68|68|True|True|True|True|not_applicable (GPU kernel-only replay; DRAM/NAND ledger separate)|
|qwen38_gpu|historical_gpu_p512_o128_c1|historical_gpu_native_screening_analytical_fallback|ttft|513.083000|803.054152|803.054152|289.971152|289.971152|289.971152|289.971152|56.515447%|56.515447%|0.000000|0|0|False|False|False|True|not_applicable (historical GPU screening; no authored DRAM/NAND ledger)|
|qwen38_gpu|historical_gpu_p512_o128_c1|historical_gpu_native_screening_analytical_fallback|tpot|24.810496|21.464680|21.464680|-3.345816|-3.345816|3.345816|3.345816|13.485487%|13.485487%|0.000000|0|0|False|False|False|True|not_applicable (historical GPU screening; no authored DRAM/NAND ledger)|
|qwen38_gpu|historical_gpu_p512_o128_c1|historical_gpu_native_screening_analytical_fallback|e2e|3665.769000|3529.068489|3529.068489|-136.700511|-136.700511|136.700511|136.700511|3.729109%|3.729109%|0.000000|0|0|False|False|False|True|not_applicable (historical GPU screening; no authored DRAM/NAND ledger)|

## Kernel train / holdout（设备 kernel 层，不是模型总延迟）

|模型|train events|eligible holdout events|profile entries|common exact keys|holdout exact hits|APE median|APE P90|APE max|状态|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|qwen25|16048|8272|75|31|31|0.127553%|0.717626%|2.238413%|fresh_current_binary_kernel_exact|
|qwen35|9534|11000|94 valid / 9564 blocked|n/a (partial)|n/a|0.322708%|1.255110%|—%|fresh_current_binary_kernel_exact|
|smollm2|15905|8129|58|28|28|0.179188%|1.848764%|12.058992%|fresh_current_binary_kernel_exact|
|tinyllama|13078|6678|38|38|38|0.171161%|0.961082%|1.939309%|fresh_current_binary_kernel_exact|
|qwen38_gpu|18673|10913|140|78|6272|0.302384%|1.970443%|28.673481%|fresh_current_binary_gpu_exact_covered|

## Qwen3.8 CPU 路线（保留作阻塞诊断，不计入 GPU 校准）

|指标|Native ms|A/B ms|APE|原因|
|---|---:|---:|---:|---|
|ttft|11904.233000|13022.257097|9.391820|没有 CUDA kernel；stage/layout/kernel_family 缺失，严格 gate 保持 A=B|
|tpot|1258.817429|331.049415|73.701555|没有 CUDA kernel；stage/layout/kernel_family 缺失，严格 gate 保持 A=B|
|e2e|21300.966000|17198.816166|19.258046|没有 CUDA kernel；stage/layout/kernel_family 缺失，严格 gate 保持 A=B|

## 证据与解释

- Qwen3.8 GPU 当前 profile 使用 705ea 当前 binary、GPU layers=66、显式 NVTX phase containment 和 CUDA kernel interval；profile 不含 DRAM/NAND service time。
- 当前 GPU train/holdout 是 prompt/output=14/8 与 18/4 的独立形状，不能合并成 p512/o128 的泛化结论。
- Qwen3.8 GPU 当前 engine A/B 中 exact demand 确实改变（train 72、holdout 68），但 TTFT/TPOT/E2E critical path 未改变，因此改善 pp 为 0；这不等于 profile 没生效。
- historical p512/o128/c1 GPU Native 单独保留在 JSON/CSV，binary/runtime 不匹配且 kernel samples 为空，不能冒充当前六维校准。
- DRAM/NAND ledger 与 kernel 校准并行：kernel-only A/B 不改变 traffic/bytes/pages/queue ledger。
