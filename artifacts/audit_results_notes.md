# 当前结果与代码对应性审计笔记

审计工作树：`c864890`，包含用户已有的 `planner.py` / `test_current_lowering_callers.py` 未提交修改。比较起点：`5a13823`。日期：2026-10-15。

## 结论

正式保存的 HBM/HBF 服务与 Decode 实验仍不能代表当前 JEDEC HBM 模型。这个结论来自保存的实际配置、执行契约和版本，不能仅凭结果相近推断。当前代码已有 JEDEC 微测与单元测试，不应写成“新模型完全没有数据/没有验证”。独立 HBF 机制实验也不能仅因 HBM 换模型就一概宣布失效；需要重跑的是引用当前完整服务、吞吐及 A/B 比值的研究结果。

## 保存结果统计

递归读取三目录全部 JSON（不包括 HTML、图片、日志）；共 379 个文件。下表按“文件内是否出现 point_result_version 及其值”统计，结果总包与逐点文件有重复，所以不是独立实验次数。

| 目录 | JSON 文件数 | 含 v2 的文件 | 含 v3 的文件 | 含 v4 的文件 | 无该字段 |
| --- | ---: | ---: | ---: | ---: | ---: |
| artifacts/hbf_hbm | 220 | 2 | 175 | 0 | 43 |
| artifacts/hbf_hbm_rerun_20261014 | 17 | 0 | 16 | 0 | 1 |
| artifacts/hbf_hbm_unified_20261014 | 142 | 0 | 0 | 85 | 57 |

三个目录均无 `point_result_version=8`，也无 `jedec_hbm` 或 `serving-points-v8` 文本。不是所有结果格式都要求 point_result_version：Decode 有独立 memory_execution_contract，不能把字段缺失本身当唯一证据。

- 当前服务点要求：`scripts/hbf_hbm_serving_study.py:24`，`POINT_RESULT_VERSION = 8`。
- 当前 Decode 要求：`scripts/hbf_hbm_decode_study.py:27`，`physical_core_unified_hardware_catalog_v3/serving-points-v8`。
- unified 的代表服务点 v4：`artifacts/hbf_hbm_unified_20261014/study_points/model__llama3_1-8b__131072__A_hbm_only.json:3226`。
- rerun 的代表服务点 v3：`artifacts/hbf_hbm_rerun_20261014/study_points/model__llama3_1-8b__131072__A_hbm_only.json:3074`，B 点见同目录对应文件第 4174 行。
- unified 的代表 Decode 旧契约：`artifacts/hbf_hbm_unified_20261014/decode_points/llama3_1-8b-131072-A_hbm_only-64.json:6641`，值为 `physical_core_unified_hardware_catalog_v2`。

### 直接配置证据

`artifacts/hbf_hbm_unified_20261014/decode_points/llama3_1-8b-131072-A_hbm_only-64.json`：

- 第 3652 行 `channels=8`；第 3661 行 `row_bytes=8192`；第 3662 行 `burst_bytes=64`。同一执行配置还有 `stacks=1`、`pseudo_channels_per_channel=2`、`bank_groups_per_rank=1`、`banks_per_group=8`。
- 与当前 16 channel/stack、2 PC/channel、1024 B 有效页、32 B burst 的 JEDEC 组织不同。
- 第 20 组仍是旧组织：`artifacts/hbf_hbm_unified_20261014/20_bank_group_decode.json:2289` 起的 `executed_hbm_parameters`，第 2293 行 channels=8、第 2302 行 row_bytes=8192、第 2303 行 burst_bytes=64；分组对照在旧总 bank 数上重分组。

### 代表 TPOT，注意工作负载不同

| 来源 | 模型与输入 | 输出 token | A TPOT ms | B TPOT ms |
| --- | --- | ---: | ---: | ---: |
| unified/decode_points | Llama 3.1 8B，131072 token | 64 | 43.5671861043 | 78.5771406585 |
| rerun/study_points | Llama 3.1 8B，131072 token | 4 | 43.5612093036 | 78.5736246015 |

JSON 键均为 `tpot_ns.p50`，除以 1e6 得 ms。unified A 第 4363 行；rerun A 第 851 行，B 第 951 行。输出 token 不同，不能把两个近似值视为严格同工作负载比较。

## 最近变更并未补齐正式实验

三个新增提交为：

1. `17befa5`，删除兼容路径，迁移运行时/内存/报表调用方及旧工件字段；提交说明明确 “Large simulation experiments were not fully rerun.”。
2. `799dd40`，删除 phase/scan/replay 兼容适配，更新当前生产者和消费者。
3. `c864890`，删除运行时及浏览器残留兼容路径，分离控制流量、容量、传输字节。

`artifacts/hbf_hbm/results.json` 是单行 JSON，键 `artifact_note` 明确记录：保留历史运行时间、吞吐、字节；仅清理废弃字段、重新计算购置费用和剩余预算、重绘报告，没有重新执行仿真。不能用 artifacts 的大量文本 diff 证明有新实验。

`artifacts/experiment_catalog_20261014.json:4` 仍写明本轮未启动仿真、unified 报告使用旧几何、JEDEC 最终时序待确认。这是历史状态提示，不能替代当前代码检查。

报告中的 `memory_core_coverage`（hbf_hbm 为 237/237，unified 为 247/247）只证明历史服务走过 DramCore，不证明走过当前 JEDEC 控制器。历史报告自身设有 `external_prediction_accuracy_validated=false` 和 `internal_experiment_matrix_complete=false`，不能把“检查通过”当当前完整实验完成。

## 复用入口检查

- 服务逐点复用：`scripts/hbf_hbm_serving_study.py:368` 起要求版本、ID、placement、parameters 相等，以及实际服务或有证据的物理容量拒绝；旧 v2/v3/v4 会重算。
- 分组复用：`scripts/hbf_hbm_experiments.py:578` 起要求 v8，否则报错。
- Decode 复用：`scripts/hbf_hbm_decode_study.py:240` 起要求当前契约、实际物理配置、参数、Core checks；旧 v1/v2 契约不会被复用。
- `--render-only` 在 `scripts/hbf_hbm_experiments.py:560` 只加载旧 results.json 并重绘；这条路径不代表重跑。

因此未发现当前缓存入口会把仓库这些旧实验自动当成当前 v8 实验的缺陷。报告引用时仍须按其保存版本理解。

## 工作树未提交修改的影响

`src/heterollm_sim/planner.py:18050` 起从控制面的 `rank_weight_shards` 选择 embedding 的本 rank 物理容量，第 18143 行将完整分片容量写入 `weight_buffer_size_bytes`。实际索引读取量仍随 cohort 改变；驻留表分配容量固定。该修改影响权重放在远端 HBF 时的 embedding 分配，也处理 TP 分片容量，不改变 HBM/NAND 基础介质时序。

`tests/test_current_lowering_callers.py:84` 新增普通/原生 gather 的 prefill→decode 共用 allocation 检查；第 148 行新增偶数/奇数大小 TP=2 容量检查。

最小相关回归已执行并通过：

```text
.venv/Scripts/python.exe -m pytest -q tests/test_current_lowering_callers.py tests/test_current_script_callers.py tests/test_current_report_callers.py tests/test_current_report_inputs.py
31 passed in 4.83s
```

这确认当前 embedding 修复及当前脚本/报告接口调用；不等于全量服务实验重跑，也不能据此断言所有兼容删除均无回归。
