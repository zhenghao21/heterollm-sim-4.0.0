# R27 引用闭合归档（入口已准备，真实归档未执行）

旧 R27 detailed_evidence.tar.gz、其6分卷及 detailed_evidence_index.json 逐字节验证通过，但递归选择混入了 identity_diagnostic 测试夹具；其发布状态为 **not_for_publication**。旧输出只留本机，不删除、不修改、不上传。postprocess.py 与原 archive.0001 不修改。

新 archive_verified.py 不递归走访 R27，不递归追踪任意 JSON ref。只收以下来源：

- predictions_complete.json 指定的真实 off/on 共262个终态预测；每个引用必须精确落在 arm/predictions/{frozen_cell_id}.prediction.json，核对引用SHA、大小及源/freeze身份。
- 两个真实 freeze、report引用的两个 errors.NNNN.json，以及controls/protocol/freeze receipt/grouped report。
- controls/freeze receipt引用的4份真实preflight及精确同名日志。
- arm/runs根目录的规范run.NNNN.finish/start成对记录；其预测引用必须等于barrier引用、覆盖冻结131，worker日志只能根据这些已验证scheduled_cell_ids及run_id构造。
- 已完成heatmaps.NNNN的provenance、绑定的barrier/score/grouped来源和精确6个图文件。图不重画，score不重算。

identity_diagnostic整个目录、test*与.test*路径、源码副本目录、旧archive目录以及源代码/二进制扩展均禁止进入。未引用的prediction/freeze，即使名字很像真结果也不选。所有允许路径拒绝symlink/junction/hardlink。

执行先确认进程70249及其他旧postprocess/campaign已自然退出；没有kill。新输出只能是本目录 archive_verified.NNNN，已存在即拒绝。root应在当前R26动态核验结束后串行运行，不与GPU/性能任务并行。

~~~powershell
Set-Location "F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_027\postprocess"
E:\anaconda\python.exe archive_verified.py --execute --output archive_verified.0001 --heatmaps heatmaps.0001
~~~

打包使用原字节tar+gzip，仅规范tar元数据。复用并校验固定SHA的R24 split/restore接口，每卷40MiB以内；随后从分卷在新目录重组packet、还原全部文件，再逐字节对比真实源文件。保留失效中间物和rejected finish，不覆盖/重试。脚本不运行score、预测、模型推理或GGUF哈希，freeze中的模型路径只作为JSON原文保留。

只有 finish.json.status=verified 且两项parts_reassembled_verified/restored_bytes_compared为true的新版本可以作为发布候选。发布集合是finish中的publication_files：成员index、分卷index、finish和所有分卷。不要上传旧根目录包、旧6卷、restored/、reassembled/或本目录任何合成测试产物。恢复/重组副本保留本机，磁盘需容纳原始数据、压缩包/卷和一份解压副本。

合成验证：11项pytest通过，测试用小文件和1KiB分卷覆盖多卷重组；生产固定40MiB，无真实R27归档执行。未删除临时夹具目录。
