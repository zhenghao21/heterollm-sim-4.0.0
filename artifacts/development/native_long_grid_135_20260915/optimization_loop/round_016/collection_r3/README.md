# collection_r3 新会话准备说明

本目录仅从已冻结 collection_r2 复制源码和 README；未复制 protocol/freeze/runs/readiness，未冻结，未执行GPU。保持相同26配置、全部原顺序和质量门，没有排除两项旧失败配置，没有将首pair改为MMQ。

新控制器为上一级 clock_control_run_r2.py。主任务冻结本目录后，以 --expected-freeze-sha256 传入真实批准SHA；控制器所有输出使用 clock_control_r3_* 新前缀。仍等待首个完整pair的人工 review：写 clock_control_r3_continue.json，字段 approved_freeze_sha256、first_pair_reviewed=true、clock_receipt_sha256，必须与本次会话一致。

180秒检查点后，控制器等待同一 supervisor 真退出，核验 complete/身份/clock binding，再用完全相同启动回执和SHA重新调用runner。只有 matrix_success 或 matrix_receipts_complete_with_failures 才结束矩阵循环；failed_identity 终止。finally 先等待所有已拥有进程结束才恢复频率，不kill。

首pair是否更换为MMQ或排除旧失败配置会改变预注册计划，当前未实施，需主任务单独批准新顺序后再冻结。

以下保留上一修订详细机制说明；目录名和启动入口以本节新版本为准：

# R16 collection_r2：仅准备，待主任务批准冻结

新目录保留原 collection/freeze 及其失败回执。只修改本目录采集器、提取器、质量检查和主机测试；探针绑定 `../operator_probe/r3`，raw/timing 使用 v2 完整调用语义。不会修改 probe、native DLL、planner、resolver 或任何旧冻结数据；不读取 LLM actual、不拟合成本。

## 已修复的严重路径

1. **外部批准 SHA 与完整身份集合。** run、resume、每阶段、worker 与 extractor 都必须接收明确 `--expected-freeze-sha256`。不能从当前 manifest 现算后当批准值。ready.freeze_ref 必须一致；protocol/python/ref 格式单独校验；从绑定 probe manifest、Nsight inventory 和签名清单重建集合，相等校验所有 probe/tool/critical 引用，强制必要 collector/probe 文件存在。运行前后全量核验，阶段间核验完整 probe 闭包与关键 Nsight/CUPTI 身份。
2. **身份失败不可恢复继续。** `identity-stop.json` 一旦出现，该 revision 永久停止；新旧 `failed_identity` 回执和 freeze_after_error 都阻断下一阶段。修复须新 revision。最终区分 matrix_success、matrix_receipts_complete_with_failures、bounded_stages_complete。
3. **额外 GPU 重叠拒收。** 逐调用检查同设备全部捕获 kernel、memcpy、memset 与完整 NVTX 是否重叠，不限 PID/线程/上下文。未归属或外 PID 重叠保留原事件并拒收；区间外 eviction 保留但不并入成本。未捕获的外部设备活动仍是观测限制。
4. **Popen 后必须等真实退出。** 一旦创建进程，即刻保存其内存身份。写 launched.json、QPC、等待观察抛错等路径仍持有同一子进程直到真实 exit code；不 kill，不发布可继续的 complete.json。写盘失效或 supervisor 意外消失时，没有 complete 回执便不得继续。

同时把已知 observed MMVQ/MMQ 与 expected/source-path 匹配分开记录；实际 MMQ 而预期 MMVQ 时保留 `observed_family=MMQ` 并拒收，不抹成未知。关键实际加载 GGML 模块有离线闭包检查，CUDA/OS 未绑定模块单独留证，不能笼统称 OS。

## 主任务审核后准备与启动

r3 已编译并主机测试；只有主任务批准并给出 r3 manifest SHA 后才能冻结。准备命令（只读文件及哈希，不访问 GPU）：

```text
E:/anaconda/python.exe runner.py prepare --root-reviewed --probe-root ../operator_probe/r3
```

prepare 生成协议、freeze、ready 与 prepare_verification。主任务应独立批准 ready 给出的 freeze SHA，随后在空闲时窗使用如下入口（示例参数必须换成真实批准值）：

```text
E:/anaconda/python.exe runner.py run --root-reviewed --idle-window-confirmed --expected-freeze-sha256 APPROVED_SHA --clock-control-receipt ROOT_RECEIPT.json --clock-control-sha256 ROOT_RECEIPT_SHA --max-stages 1
```

移除 `--max-stages` 继续既有计划，绝不重跑已有阶段。每阶段最多等待 180 秒，超时记录 checkpoint 和子/监督进程 PID，后台 worker 继续收集原进程退出；没有完整回执就不启动后续阶段。不杀进程、不推断退出码。

## 2400 MHz 频率域（与原 collection 独立冻结）

协议显式固定 `target_sm_clock_mhz=2400`、`sm_clock_tolerance_mhz=30`。锁频由主任务在外部执行 `nvidia-smi -lgc 2400,2400` 并 finally 恢复，本采集器绝不执行锁频、恢复或伪造 locked=true。

主任务写一个**新且保持不变**的 JSON 锁频回执，采用以下字段（路径/输出/时间/UUID 必须是真实记录；这是格式契约而非已执行回执）：

```json
{
  "schema": "operator-clock-control-receipt/v1",
  "created_utc": "ACTUAL_UTC",
  "gpu_uuid": "ACTUAL_GPU_UUID",
  "target_sm_clock_mhz": 2400,
  "sm_clock_tolerance_mhz": 30,
  "requested_lock_min_mhz": 2400,
  "requested_lock_max_mhz": 2400,
  "lock_command_returncode": 0,
  "command": ["ACTUAL_NVIDIA_SMI_PATH", "-lgc", "2400,2400"],
  "stdout_ref": {"path": "ABSOLUTE_CAPTURED_STDOUT_PATH", "sha256": "ACTUAL_SHA256", "bytes": 0},
  "stderr_ref": {"path": "ABSOLUTE_CAPTURED_STDERR_PATH", "sha256": "ACTUAL_SHA256", "bytes": 0},
  "restore_on_exit_planned": true
}
```

`stdout_ref`/`stderr_ref` 大小为实际字节数，允许真实空文件。恢复操作写**另一个**新回执，不能更新这个已绑定锁频回执。首次 run 保存 clock-control-binding.json；此 revision 不能换一个锁频会话继续混入。每阶段在启动前验证回执 SHA/内容和 GPU UUID，并实际读回 SM 频率。每次测量过程保存 5 ms 目标周期 NVML 遥测及绝对 QPC，频率门对 30 个 formal 的 qpc_start/qpc_end 各取前后包围样本，每侧距离最多 25 ms，所有包围区间内读回均须 2400±30 MHz；缺失、过稀或越界即失败。该策略在采集前冻结，不随结果改宽。

遥测是采样观测而非连续每个内核的频率波形。报告始终保留 `clock_locked_inferred=false` / locked 未推断；外部锁频命令成功、实际读回、采样覆盖限制三者分开。正式频率门失败记为 failed_identity 并停止本 revision，防止跨硬件域混样。

## 固定矩阵与质量门

26 个配置、每配置 3 对独立进程，顺序 profile/direct、direct/profile、profile/direct，每对随后导出 SQLite。共 78 profile、78 direct、78 export；每测量进程 1 first + 5 warmup + 30 formal，单图调用，显式冷缓存扫读。profile/direct 都是相同 event 模式；未增 event-free control 组，所以不宣称 CUDA event 本身扰动已消除。保留 --kill=false、--force-overwrite=false；不做定长截断。

所有三对进程必须满足：全数值原始行及模块身份正确；实际链/量化类型/D4 源参考一致；每 profile 正式内核并集 p90/p10≤1.5；三个 profile 进程中位数相对其中位数最大偏差≤5%；每对主机 profile/direct 中位数相对 direct 偏差≤20%；每 direct 主机 p90/p10≤1.5；无未关闭工具警告；每阶段 formal 频率域验证通过。拒绝额外重叠设备工作，不删除异常行。

只读提取命令：

```text
E:/anaconda/python.exe extract.py --output analysis_0001 --expected-freeze-sha256 APPROVED_SHA
```

输出必须在本目录内且此前不存在。实际 SQLite schema 全检查；保存全部 NVTX/CUPTI/kernel/API/sync/memory/diagnostic/target 表原始行。内核通过 correlation+PID 与同 TID NVTX 内 API 关联，完整调用范围必须包含目标内核。记录 exact name、grid/block/shared memory、角色与并集。QPC 和 Nsight 不跨域相减，CPU/GPU 重叠时间不相加。

报告保留全部 26 配置、78 对及所有阶段失败/缺失分母。仅输出诊断标签，calibration_eligible=false，不拟合、不输出 profile 系数；validation/aligned_control 不用于拟合。历史源/运行时等价性仍未证明，同 DLL 实际观测与有限条件迁移分开表述。

Root premeasurement scheduling revision: execute registered train_Q5_0_m64_n896_k896, validation_Q5_0_m32_n1792_k896 and aligned_Q8_0_m64_n896_k1024 first, then remaining original order. All26 configs, all3pairs and fixed gates retained. This probes MMQ+holdout+aligned path early without filtering prior failures. New controller is included in freeze file closure.
