# 编译与主机验证结果

已完成授权编译及host-api-test，未运行GPU捕获。最终exe由build_manifest.json唯一指向，原先失败/未生成清单的编译记录全部保留。

- 最终文件：launch-recorder.attempt0004.exe，43,008 bytes。
- exe SHA256：27f2075db5beebf4845b87ef4f860020cdc4133b3452ad449968da2d41c982e1。
- 构建清单SHA256：3e5d24a8a16e8d3b57994911f9fc070a18b789fdd64e057f250c72f98a1ba648。
- 289个实际编译头文件及所有输入SHA已入清单；编译错误仅来自重复包含生成API表，修正后链接通过。中文include日志解析已修正，所有尝试日志保留。
- 主机测试返回0：参数值复制、完整合成callback、未知符号不得解码均通过；stderr为空；无CUDA context/订阅/graph执行。

本机CUDA12.8头文件给出CUPTI_API_VERSION=26；旧库版本2025.1.1.0，cuptiGetVersion=26。选择的Nsight随附cupti64_134.dll文件版本2026.3.1.0，cuptiGetVersion=130401，两库数字签名均为Valid/NVIDIA。现代库通过绝对路径LoadLibraryEx加载且校验实际模块路径、所需4个legacy回调导出和运行库版本。没有重命名DLL、没有把新DLL冒充旧import library；PE import table确认不存在CUPTI静态导入。

头文件和运行库API版本确实不同，报告没有将其标为相同版本，也没有因为主机测试通过就宣称真实GPU回调兼容。实际GPU订阅/符号/参数仍由root独立非计时运行验证。记录器只接受已经在既有Q8 MMQ trace中观察到的精确conversion symbol及参数shape；没有或多个匹配、溢出、畸形回调均失败。

下一条root命令在compile_ready.json的root_run_entry中。仅root批准串行GPU时窗后运行。输出仍为recorder-only：不读vy、不做中间缓冲copy，quantized_code_observed=false，不能用它证明q=63/64。
