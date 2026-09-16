# R20 图间隙 pilot 采集器

状态：仅源码、命令和提取器准备；不编译、不加载 DLL、不锁频、不启动进程。

`collector.py plan` 生成固定顺序：E262144/G8、control/buffered、每臂 3 个 direct/profile 配对，共 12 个自然完成的 native 进程与 6 个 profile SQLite 导出。任何已启动进程必须自然结束；超时只保留证据并等待，禁止 timed kill。时钟控制完全由 root 持有，采集计划明确要求 finally reset。

`extract.py` 要求完整 36 次调用分母、同一 pair/arm/PID/QPC 身份。control 只能按逐次 final + 首尾 intermediate 解释；buffered 只能按 first/post_warmup/post_formal 三个全 stage 边界块解释。profile 的 kernel/API 计数来自实际 SQLite trace；绝不由计划 N=8 填充，也不产生拟合常数或 calibration。