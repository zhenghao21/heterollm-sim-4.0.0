# GDDR 审计三轮回归记录

本轮回归使用真实 `ScenarioConfig`、`compile_scenario` 和 Web API 传输边界，并覆盖了
stateful L2 到物理 GDDR 的事件提交路径；物理地址由 run-local allocator 解析。

本轮进一步统一了 planner、L2、allocator 和物理核心之间的事务契约：访问记录保留
`offset_bytes`、`allocation_generation` 和 alias 字段；L2 recost 保留既有分配声明；
独立物理请求共享到达时间，真实写读重叠才建立完成依赖；根分配存在存活 alias 时不能释放。

编译完成后会在整张任务图上汇总访问范围：只有访问推导出的 extent 会被提升为同一
buffer/generation 的最大完整范围；显式容量仍保持固定并冲突即报错。这样较短后续切片
不会被当成重新分配，同时不会允许物理 allocator 静默迁移已使用的分配。

## 运行时编译

测试将参考场景中的 `hbm0` 替换为已连接的 GDDR7 组件，并保留原模型图、请求和 GPU 算子 lowering。实际编译得到 326 个任务，其中包含 GPU GEMM、融合 Attention 和显式 GDDR 物理访问描述。

每个物理任务的方向字节来自任务的正式 `memory_accesses` 描述。GEMM 的读取覆盖 activation 与 weight，输出写入保持为 `write`；编译阶段的地址会做容量检查，运行时则由 owner 级 allocator 按 `buffer_id / offset / generation` 重新解析，避免任务长度变化导致同一缓冲区漂移或独立缓冲区碰撞。

方向信息的来源层次如下：

1. `cost_model.read_bytes` / `write_bytes` 在与物理 demand 守恒时可直接使用；
2. GEMM 使用 `activation_bytes + weight_bytes` 与 `output_bytes`，前提是它们与物理 demand 守恒；
3. cache/backing 阶段的 `physical_read_bytes` / `physical_write_bytes` 描述缓存后的外存流量，并作为分片或裁剪后的正式方向来源；
4. 只有总量而没有精确方向时现在明确报错，不再按原始读写比例猜测。

## 前端传输往返

测试通过本地 HTTP 服务执行前端使用的 `/api/normalize` 与 `/api/validate` 请求。保存后的 JSON 重新解析为 `ScenarioConfig`，确认 GDDR 类型、GDDR7 代际、16,000,000,000 B（16 GB）容量及物理配置保持一致，并通过场景校验。

当前环境没有可用浏览器实例，因此没有把浏览器点击流程伪装成已验证；HTTP 往返覆盖了页面保存、规范化和重新加载所使用的真实 JSON 边界。

## 小型真实模型闭环

回归构造了一个真实的 tiny dense transformer 配置，经过 JSON 重载、场景校验、planner、
UnifiedEventKernel 和实际 DramCore 完成 Prefill 及两个 Decode 阶段。Prompt 长度分别覆盖
2 和 5；stateful L2 分别关闭和开启。开启 L2 时观察到冷填充、热命中、脏行淘汰和 GDDR
写回；所有正式请求都检查了 owner、地址、offset、generation、方向和容量边界。

## 可重复命令

```text
python -m pytest -q
```

结果：项目全量回归为 `48 passed`，其中包含 HTTP 往返、小型真实 Prefill/Decode 闭环、动态 L2、运行时分配器、固定 offset/alias、独立物理请求和物理核心测试。
