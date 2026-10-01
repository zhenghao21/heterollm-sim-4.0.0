# HeteroLLM Simulator 离线演示包

这个目录可以直接复制到另一台 Windows x64 电脑运行，不需要安装 Python、Node.js，也不需要联网。包内的 `runtime` 是私有的 Python 3.12 运行时，`app` 是仿真器源码和前端资源。

## 启动

双击 `Start-Simulator.cmd`。脚本会启动本地服务并打开浏览器：

```text
http://127.0.0.1:8765/
```

如果浏览器没有自动打开，手动访问上面的地址即可。演示结束后关闭启动窗口，或在窗口中按 Ctrl+C。

也可以在 PowerShell 中指定端口，或者不自动打开浏览器：

```powershell
.\Start-Simulator.ps1 -Port 8766
.\Start-Simulator.ps1 -Port 8766 -NoBrowser
```

## 现场演示建议

1. 点击“载入参考”，先展示完整硬件拓扑和模型/负载默认值。
2. 在“硬件预设库”中载入 `nvidia-b200-1gpu-2hbf-2hbm`。当前演示预设已登记 HBF 读 488 GB/s、写 27.2 GB/s、读延迟 4 μs、写延迟 75 μs，链路逻辑协议为 `HBF`，物理承载在 metadata 中记录为 UCIe；这些值是可替换的分析坐标，不是量产芯片实测。
3. 在“映射”视图设置 KV 驻留策略：HBM-only、HBM 主缓存 + HBF 卸载或按 layer 混合；在“负载”视图分别设置短上下文、长上下文、单并发和多并发。建议先演示 `prompt=512, output=256, concurrency=1`，再演示长上下文多并发。
4. 点击“校验”，确认没有红色错误后点击“运行仿真”。
5. 点击顶部“KV Cache 分层扫描”，运行当前策略、HBM-only、HBF-only 和 HBM+HBF 混合候选；默认 B200 演示 HBF 已满足 active-memory 字段，可以直接比较三类候选。若导入自定义只读 HBF，扫描会将其标为不可行，不会静默把它当成 HBM。
6. 重点查看 TTFT、TPOT、E2E、KV 峰值/容量、物理读写流量、Offload/Prefetch/Migration 和 Swap Transfer；结果页与回放页分别用于汇总和解释路径。
7. 导出当前 JSON、硬件参数和标准 IR，保存每种 KV 策略的结果截图；跨模型、负载和并发重复扫描，按 SLO 与容量约束寻找甜点区间。

`demo/evidence` 中保存了冻结的 Native 数据、R0 结果和本次结果，便于离线讲解误差来源。Native 数据是既有实测快照，演示电脑不会重新采集硬件数据。

本次 DeepSeek-V3 专项证据也已随包提供：`deepseek_v3_edge_short_scan.json` 是 128/32 有界端侧演示的 HBM-only、HBF-only、Hybrid 三方案结果；`deepseek_v3_edge_evidence.json` 是完整 `edge_personal_assistant` 1024/512 负载的容量、放置、估算和校验证据；`deepseek_frontend_api_flow.json` 记录前端对应的校验、估算和架构扫描接口摘要。它们用于现场复核，不包含模型权重，也不会冒充完整 1024/512 serving 实测。

## 健康检查

服务运行时执行：

```powershell
.\Check-Offline.ps1
```

该检查会验证私有 Python、NumPy、OR-Tools 和 `/api/health`。脚本在服务未启动时也会给出明确提示。

## 运行边界

- 包面向 Windows x64；不依赖网络。
- 包含运行 UI 所需的源码、前端静态资源、文档和小型演示证据；开发测试、采集工具、数十 GB 的原生 trace、模型权重和临时文件没有复制。
- 不能在此包中重新执行需要 `llama-server.exe`、CUDA 或真实 GPU 的 Native 采集；离线演示使用已冻结的实测结果和仿真器本身。
- 如果现场电脑禁止 PowerShell 脚本，使用 `Start-Simulator.cmd`，它会以当前用户权限调用脚本，不需要管理员权限。
