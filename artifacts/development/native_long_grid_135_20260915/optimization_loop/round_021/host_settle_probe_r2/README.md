# R21 Host settle probe

仅源码准备。两条件都是 buffered E262144/G8：`--settle-ms 0` 与 `--settle-ms 1000`。每个进程先记录原 first graph 与全 stage 精确核验；1000 ms 条件随后在记录图范围外执行未记录 graph+final-sync settle，再保留原 5 warmup 与 30 formal 记录路径。settle metadata durable 保存迭代数、QPC 起止、状态、条件和 argv，且明确不进入估计器。

此设计检验主机/driver 稳态假设，不声称 CPU-only 因果隔离；device、cache 与 runtime 状态也会变化。R21 是单一预登记系列，不能反复运行 R20 直至质量门通过。