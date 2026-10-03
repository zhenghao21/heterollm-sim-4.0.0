# Round 068 共同改善与误差对比

A=新 DRAM/NAND 路径、校准关闭；B=同一路径、校准开启。误差百分比为 `(仿真器-Native)/Native`，APE 为绝对值；`改善百分点=A APE-B APE`。

## qwen25 同形状三次 replay A/B

| 场景 | 指标 | Native ms | A ms | B ms | A signed ms | B signed ms | A APE | B APE | 改善 pp | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen25__long__long__p4__r1 | ttft_ms | 31.631000 | 15.201549 | 34.223951 | -16.429451 | +2.592951 | 51.941% | 8.198% | +43.743 | improved |
| qwen25__long__long__p4__r1 | tpot_ms | 4.939625 | 3.057221 | 3.462309 | -1.882404 | -1.477316 | 38.108% | 29.907% | +8.201 | improved |
| qwen25__long__long__p4__r1 | e2e_ms | 265.912500 | 161.896427 | 200.426105 | -104.016073 | -65.486395 | 39.117% | 24.627% | +14.490 | improved |
| qwen25__long__long__p4__r2 | ttft_ms | 31.266500 | 15.201549 | 34.223951 | -16.064951 | +2.957451 | 51.381% | 9.459% | +41.922 | improved |
| qwen25__long__long__p4__r2 | tpot_ms | 4.986917 | 3.057221 | 3.462309 | -1.929696 | -1.524607 | 38.695% | 30.572% | +8.123 | improved |
| qwen25__long__long__p4__r2 | e2e_ms | 268.205500 | 161.896427 | 200.426105 | -106.309073 | -67.779395 | 39.637% | 25.271% | +14.366 | improved |
| qwen25__long__long__p4__r3 | ttft_ms | 32.415000 | 15.201549 | 34.223951 | -17.213451 | +1.808951 | 53.103% | 5.581% | +47.523 | improved |
| qwen25__long__long__p4__r3 | tpot_ms | 5.079542 | 3.057221 | 3.462309 | -2.022321 | -1.617232 | 39.813% | 31.838% | +7.975 | improved |
| qwen25__long__long__p4__r3 | e2e_ms | 273.681000 | 161.896427 | 200.426105 | -111.784573 | -73.254895 | 40.845% | 26.767% | +14.078 | improved |

| 指标 | A 中位 APE | B 中位 APE | 中位改善 | 3 个场景是否全部改善 |
|---|---:|---:|---:|---|
| ttft_ms | 51.941% | 8.198% | +43.743 pp | 是 |
| tpot_ms | 38.695% | 30.572% | +8.123 pp | 是 |
| e2e_ms | 39.637% | 25.271% | +14.366 pp | 是 |

## 既有五模型 Native replay 基线

| 场景 | 指标 | Native ms | 当前模拟 ms | signed error | APE | 状态 |
|---|---:|---:|---:|---:|---:|---|
| qwen25__long__long__p4__r1 | ttft_ms | 31.631000 | 15.201549 | -16.429451 (-51.941%) | 51.941% | historical baseline |
| qwen25__long__long__p4__r1 | tpot_ms | 4.939625 | 3.057221 | -1.882404 (-38.108%) | 38.108% | historical baseline |
| qwen25__long__long__p4__r1 | e2e_ms | 265.912500 | 161.896427 | -104.016073 (-39.117%) | 39.117% | historical baseline |
| qwen35__long__long__p4__r1 | ttft_ms | 131.306500 | 273.416590 | +142.110090 (+108.228%) | 108.228% | historical baseline |
| qwen35__long__long__p4__r1 | tpot_ms | 8.034292 | 6.262536 | -1.771755 (-22.052%) | 22.052% | historical baseline |
| qwen35__long__long__p4__r1 | e2e_ms | 519.616500 | 648.510222 | +128.893722 (+24.806%) | 24.806% | historical baseline |
| qwen38__long__long__p4__r1 | ttft_ms | 11904.233000 | 13022.257097 | +1118.024097 (+9.392%) | 9.392% | historical baseline |
| qwen38__long__long__p4__r1 | tpot_ms | 1258.817429 | 331.049415 | -927.768014 (-73.702%) | 73.702% | historical baseline |
| qwen38__long__long__p4__r1 | e2e_ms | 21300.966000 | 17198.816166 | -4102.149834 (-19.258%) | 19.258% | historical baseline |
| smollm2__long__long__p4__r1 | ttft_ms | 38.460500 | 41.813951 | +3.353451 (+8.719%) | 8.719% | historical baseline |
| smollm2__long__long__p4__r1 | tpot_ms | 6.236010 | 6.372445 | +0.136434 (+2.188%) | 2.188% | historical baseline |
| smollm2__long__long__p4__r1 | e2e_ms | 334.314500 | 347.515725 | +13.201225 (+3.949%) | 3.949% | historical baseline |
| tinyllama__long__long__p4__r1 | ttft_ms | 31.041000 | 31.057108 | +0.016108 (+0.052%) | 0.052% | historical baseline |
| tinyllama__long__long__p4__r1 | tpot_ms | 4.862802 | 2.606033 | -2.256769 (-46.409%) | 46.409% | historical baseline |
| tinyllama__long__long__p4__r1 | e2e_ms | 266.495000 | 156.146689 | -110.348311 (-41.407%) | 41.407% | historical baseline |

## storage ledger composition control

| 字段 | A | B | B-A | invariant |
|---|---:|---:|---:|---|
| DRAM.task_count | 10 | 10 | 0 | true |
| DRAM.logical_bytes | 332800 | 332800 | 0 | true |
| DRAM.physical_bytes | 332800 | 332800 | 0 | true |
| DRAM.physical_read_bytes | 264192 | 264192 | 0 | true |
| DRAM.physical_write_bytes | 68608 | 68608 | 0 | true |
| DRAM.queue_wait_ns | 288.0 | 288.0 | 0.0 | true |
| NAND.task_count | 58 | 58 | 0 | true |
| NAND.logical_bytes | 108976936 | 108976936 | 0 | true |
| NAND.physical_bytes | 109019136 | 109019136 | 0 | true |
| NAND.physical_read_bytes | 108748800 | 108748800 | 0 | true |
| NAND.physical_write_bytes | 270336 | 270336 | 0 | true |
| NAND.pages_touched | 12026 | 12026 | 0 | true |
| NAND.queue_wait_ns | 0.0 | 0.0 | 0.0 | true |
| NAND.media_queue_wait_ns | 11992000.0 | 11992000.0 | 0.0 | true |
| NAND.host_queue_wait_ns | 9794355.2 | 9794355.2 | 0.0 | true |

该 profile 的源 shape 是 prompt=8/output=8/parallel=1，目标 replay 是 prompt=186/output=49/parallel=4，因此这里只能视为开发回放适配。校准 profile 只增加 engine phase boundary task；DRAM/NAND resolver 继续拥有各自物理服务时间、字节数、页数和队列字段。该组合证明同时生效与资源不重计费；它不等于用校准去拟合 DRAM/NAND 器件延迟。
