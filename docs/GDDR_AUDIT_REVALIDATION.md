# GDDR 审计二轮回归记录

本轮回归使用真实 `ScenarioConfig`、`compile_scenario` 和 Web API 传输边界，并覆盖了
stateful L2 到物理 GDDR 的事件提交路径；物理地址由 run-local allocator 解析。

## 运行时编译

测试将参考场景中的 `hbm0` 替换为已连接的 GDDR7 组件，并保留原模型图、请求和 GPU 算子 lowering。实际编译得到 326 个任务，其中包含 GPU GEMM、融合 Attention 和显式 GDDR 物理访问描述。

每个物理任务的方向字节来自任务的正式 `memory_accesses` 描述。GEMM 的读取覆盖 activation 与 weight，输出写入保持为 `write`；编译阶段的地址会做容量检查，运行时则由 owner 级 allocator 按 `buffer_id / offset / generation` 重新解析，避免任务长度变化导致同一缓冲区漂移或独立缓冲区碰撞。

方向信息的来源层次如下：

1. `cost_model.read_bytes` / `write_bytes` 是显式分方向字节；
2. GEMM 使用 `activation_bytes + weight_bytes` 与 `output_bytes`；
3. cache 阶段的 `physical_read_bytes` / `physical_write_bytes` 和 `backing_read_bytes` / `backing_write_bytes` 描述缓存后的外存流量。

## 前端传输往返

测试通过本地 HTTP 服务执行前端使用的 `/api/normalize` 与 `/api/validate` 请求。保存后的 JSON 重新解析为 `ScenarioConfig`，确认 GDDR 类型、GDDR7 代际、16,000,000,000 B（16 GB）容量及物理配置保持一致，并通过场景校验。

当前环境没有可用浏览器实例，因此没有把浏览器点击流程伪装成已验证；HTTP 往返覆盖了页面保存、规范化和重新加载所使用的真实 JSON 边界。

## 可重复命令

```text
python -m pytest -q
```

结果：项目全量回归为 `29 passed`，其中包含 HTTP 往返、动态 L2、运行时分配器和物理核心测试。
