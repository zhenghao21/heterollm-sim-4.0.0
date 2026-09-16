# R22 r2 后端修复准备结果

R22 r1 在任何 timing 前已被拒收：锁定版 `ggml_backend_sched_new` 要求 scheduler 后端列表的最后一项为 CPU。R2 使用独立的 probe、protocol 和 collection 目录，未覆盖 r1 的失败证据，也没有执行 18-stage 正式计划。

- 构建成功：`long-graph-probe.exe` SHA-256 `8cb92bfb4d7056d5636f81980b33ee9931b566b789e204af57b991596b9bd336`，编译闭包 296 个头文件，构建清单冻结 332 个输入。
- 身份检查成功：所有冻结源、工具和 DLL 闭包匹配；该步骤没有 GPU work。
- 非计时数值冒烟成功：仅 `scale_f32_e262144_g64` 一张图、一次 async graph compute 和一次 scheduler synchronize；`16777216` 个值全逐位相等，0 mismatch / 0 nonfinite。无 first/warmup/formal QPC 窗口、无 NVTX 时间窗、无 CUPTI profile。
- Scheduler 是 `[CUDA, CPU]`：CUDA 优先。r2 在 allocation 前将输入和全部 SCALE 节点显式 pin 到 CUDA，并在 allocation 后拒绝非 CUDA tensor、非单 split 或非零 CPU scheduler allocation。此次 smoke 记录为 CPU scheduler buffer 0。CPU 仅为上游的 terminal backend 要求；真实 CUDA kernel/stream 身份仍待正式 profile trace 验证。

收集器 r2 已绑定 r2 binary/protocol，保留原始完整 18 stage 计划（12 native + 6 export）、失败分母、自然退出、无 300 秒审核超时。尚未创建 execution freeze，尚未运行正式 native 或 export。

关键证据：

- `build_manifest.json`: `3c34bfb88b6f9cf0bb8071ca2a782fa30b7b130f244297973aff372499564c80`
- `protocol.json`: `d80e8e1fb16b5311560ad064227d8d50be7b8cc586f87d496e17c3f5ad3d57a1`
- `source_provenance.json`: `953c80fc78f6a83e93b51d17574395c2f1959d0a30eadc9b5ec27fc7738b0a4d`
- `identity_only_result.json`: `aea41e4a2d65dfd08ff3ff14de46f8dd7a6d69b966338c36b1f8eaf78320824b`
- `smoke_result.json`: 将在写入后由文件自身链外引用；其输入引用已列明。
