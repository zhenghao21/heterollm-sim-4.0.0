# R27 双路执行器准备说明

当前仅完成代码与结构测试；没有创建protocol/freeze/controls、没有预测、没有native执行、没有提交。

来源：R25/off/freeze.json的115个源码文件，113个精确继承冻结字节；仅从根代理明确reviewed commit替换tools/predict_stable_native_dataset.py与tools/native_final_output_binding.py。两路源代码字节完全相同，无新增/删除源码文件；off默认关闭、on开启末层选行，其他R25机制flags按既有reviewed recipe保持。驱动、验证helper、聚合器与测试也须与该提交字节一致。

freeze与lock分别对off/on启动全新Python进程，从各arm/source真实导入并执行verify_freeze_references。每格配置proof必须与已经规范化的实际static_inputs一致，全部131格没有preparation_error才通过。所有预检结果含实际API/helper、freeze、protocol身份并只写一次；任何一路失败不生成预测许可controls，保留失败预检记录。

双路静态对比只规范三项：验证过的extractor副本位置、从完整retained evidence重算并验证的派生digest、验证过且与campaign及evidence别名一致的硬件PDF复制路径。没有递归忽略字段。未知路径/配置/source变化全部拒绝。

full开始前必须已有两路freeze与lock预检通过证据；每arm前后复核controls/source/contract/helper/native身份。终态predicted/failed/incomplete不重跑；评分存在后禁止resume。两路完整262终态全部落盘后写predictions_complete.json，scorer必须先验证该屏障。报告每model/deployment及全131的三指标误差分布、strict<10格数、逐格逐指标paired变化；失败留固定分母。B保持unvalidated，旧R25失败不追认。没有归档新脚本或盲测/校准声称。

## 根代理review并提交后的命令

```powershell
E:\anaconda\python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_027/freeze_candidate.py --reviewed-commit <FULL_REVIEWED_COMMIT_SHA>
E:\anaconda\python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_027/run_candidate.py lock
E:\anaconda\python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_027/run_candidate.py full --workers 4 --timeout-seconds 600
E:\anaconda\python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_027/run_candidate.py score
E:\anaconda\python.exe artifacts/development/native_long_grid_135_20260915/optimization_loop/round_027/summarize_groups.py
```

性能采集与模拟须由根代理串行调度。本执行器不会创建native进程；full只运行模拟器。

结构验证：32 passed。涵盖115文件继承/两文件替换、三个精确路径差异、未知差异拒绝、新进程真正调用arm verifier、旧None proof拒绝、双预检失败阻断off提前运行、完整262屏障、终态失败保留、strict10与分组分母。准确命令和文件SHA见driver_preparation.json。

补充baseline门禁：freeze/lock及后续guard均逐格比较新off与R25/off的完整static_inputs，除三项明确副本差异外必须相等；model/deployment和preparation状态也保持一致。不能以新off/on彼此一致替代baseline一致。结构测试覆盖双路共同漂移也必须拒绝。
