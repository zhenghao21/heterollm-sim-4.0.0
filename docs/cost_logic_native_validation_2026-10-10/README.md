# 成本逻辑修复后的 native 误差复跑

2026-10-10，从实际前端提交 10 个原始 GGUF 模型的 CUDA Graph 开/关共 20 组，全部完成。正式比较请求为 512 输入 / 128 输出；native 复用北京时间 2026-10-08 19:32:40–19:36:17 的已核实测量，本轮没有重新测 native，也没有用模型实测耗时拟合成本。

- [完整报告](report.html)：逐模型首 token、后续每 token、总耗时误差，以及 Graph 收益。
- [机器可读比较结果](comparison_summary.json)：20 组有效配对、输入一致性及物理成本参与检查。
- [前端提交记录](ui_runs.json)：实际提交、任务身份、进度和最终状态。记录中的绝对路径是本机运行位置。
- [代表性结果页检查](render_checks/result_page_checks.json)：4 组真实结果在桌面和窄屏中的显示检查，不代表全部 20 组的持续前端轮询检查。

整体总耗时平均绝对误差由 15.597086% 变为 15.596511%，变化极小。Graph 收益预测仍有明显缺口，不能据此宣称成本模型已通过精度或泛化验证。20 个临时服务及仿真任务已结束。

## 完整结果的保存与读取

20 份 `ui_*_result.json.gz` 是原始结果 JSON 的无损压缩，包含完整报告，不是摘要。单份原始文件超过 GitHub 文件大小限制，因此 Git 只跟踪压缩副本；本机原始 `.json` 保留并由根目录 `.gitignore` 排除。所有压缩文件提交前均已流式解压，与原始文件逐字节比较一致。

直接打开 `report.html` 无需解压或启动服务。在仓库根目录用项目 Python 重新计算报告前，先解压：

```python
from pathlib import Path
import gzip
import shutil

directory = Path("docs/cost_logic_native_validation_2026-10-10")
for compressed in directory.glob("ui_*_result.json.gz"):
    target = compressed.with_suffix("")
    if target.exists():
        continue
    with gzip.open(compressed, "rb") as source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
```

随后运行 `python tools/render_cost_logic_rerun_report.py` 可从结果重新计算比较；`python tools/render_cost_logic_rerun_report.py --render-only` 仅用已保存的比较数据刷新 HTML。两者均不会提交新的仿真或执行 native 推理。这里的运行场景保留了本机结构构建器与独立成本文件路径；重新执行仿真需要恢复对应运行环境，完整结果本身不受这些外部路径影响。
