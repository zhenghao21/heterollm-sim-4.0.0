# R30 完成后展示与原始证据归档

仅供已完成的固定131格、两组393指标使用。准备代码没有执行真实评分、绘图、预测或归档。旧R27和R30已经冻结的源码、helpers、protocol均不修改。

`postprocess.py` 复用R27的完整结果核验与热图实现。`predictions_complete.json`、两组评分回执及分组报告全部存在后，才调用R30屏障验证和读取预测/评分。严格保留6个模型/部署组、131个选中格和31个域外格；失败显示X，域外显示-；delta为APE_on减APE_off（百分点）。不依据误差选择格子，不运行新的评分器。

`archive_verified.py` 复用R27的显式引用归档及固定R24分卷/恢复接口。它从262终态屏障逐条定位真实预测，加入对应worker的start、child、raw、execution、seal、日志、软时限和观察中断记录，核验自然退出和冻结身份。非零退出保留失败终态及其可能存在的raw。raw和正式预测各保留原始字节，不进行内容重写或去重。只选显式引用及规范run记录；不递归选择测试、identity_diagnostic、source、execution_source、control_source或旧归档目录。模型路径只作为JSON文字保留，不读取或打包GGUF。

中断后停止新提交的run可以保留尚未启动的scheduled格；其已完成终态必须由对应attempt和seal证明，并与后续run合起来恰好覆盖131格。存活锁、未解决或额外attempt、缺seal、改变过的raw/预测都会拒绝归档。

待root确认R30两组及分组报告完成、其他实验自然退出后，才执行：

```powershell
$round = 'F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_030'
E:\anaconda\python.exe "$round\postprocess\postprocess.py" heatmaps
E:\anaconda\python.exe "$round\postprocess\archive_verified.py" --execute --heatmaps heatmaps.0001
```

所有输出独占创建。热图产生off/on/delta三幅PNG和三幅SVG及provenance。归档默认位于`postprocess/archive_verified.0001`，每卷不超过40MiB；从分卷重新组装、解包并逐字节比较源文件。只有finish的status=verified、parts_reassembled_verified=true、restored_bytes_compared=true才构成发布候选。发布文件只取finish中的publication_files；restored/reassembled和测试临时文件不发布。失败中间物保留，不覆盖或自动重试。

测试只用合成131格数据、小文件及1KiB分卷，不访问真实R30结果、不生成真实图表、不运行GPU或模型。

准备验证：初版23项纯合成测试全部通过（154.56秒），包括多卷恢复逐字节一致。静态收尾已将R27遗留的blind字段检查适配为R30实际提供的gate_B=unvalidated，并添加专用回归用例。root确认没有GPU采集或完整仿真后，仅该新增用例补跑通过：1 passed、6 deselected、0.04秒；没有重跑154秒归档测试。当前没有真实R30热图、归档或发布结果。
