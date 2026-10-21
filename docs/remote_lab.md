# 远程仿真工作台

- 网站与历史结果：<https://zh.gripe/>。
- 现有仿真前端：<https://zh.gripe/simulator/>，唯一维护源为 `src/heterollm_sim/webui/`。
- HTTPS API：`https://sim-api.zh.gripe`，只转发到本机 `127.0.0.1:8765` 的 `/api` 路径。
- 所有 API 都要求 Bearer 令牌；跨域只允许正式网站及其 www 域名。

## 开机启动

管理员运行 `scripts/install_remote_lab_startup.ps1`，注册唯一任务 `HeteroLLM Remote Lab`。
任务在系统启动 30 秒后运行，不要求用户登录，不保存 Windows 密码。
主体是原用户 A，S4U 登录、普通权限，不以 SYSTEM 身份运行仿真后端。
操作使用 Windows 自带 PowerShell 5.1 和明确的绝对路径，不依赖 Codex 是否打开。

任务仅调用 `scripts/start_remote_lab.ps1 -KeepAlive`，启动鉴权 API 和现有的独立 HTTPS 隧道。
守护每 15 秒检查服务，退出后重新拉起；守护自身失败由任务计划程序重试。
网络尚未就绪时，隧道自行重连。跨会话私有文件锁与真实监听 PID 检查防止重复启动。
手动运行同一启动脚本会复用已有服务。

**开机任务不调用扫描入口，不恢复仿真作业，不修改队列、历史结果或取消标记。**
HTTP 后端启动时作业队列为空；只有用户之后在网页明确运行任务，才会执行新仿真。
网站原有 40 个完成结果、32 个取消点继续独立保留。

## 口令和状态

原连接令牌保存在 `%USERPROFILE%\.heterollm-sim\remote-access\api-token.txt`，不会因重启而轮换。
隧道继续使用 `C:\Users\A\.cloudflared\42954194-8db2-47fd-97ca-84950ed672cb.json`，其内容不得写入代码或日志。
私有目录权限仅授予 A 和 SYSTEM；任务命令行仅包含路径，不含令牌。
用户模型和硬件预设也固定在 `%USERPROFILE%\.heterollm-sim\` 下，保留已有预设。
不使用桌面应用的 AppData 合并视图，避免开机任务和桌面应用读取到不同口令或配置。

同一私有目录的 `status.json` 保存真实 API、隧道及守护 PID；
`autostart.log` 记录启动和恢复错误，`backend.err.log`、`tunnel.err.log` 用于定位服务故障。
任务安装失败时查看 `autostart.install-error.log`。任务状态可以在 Windows 任务计划程序中查看。

电脑必须保持开机和联网；自动开机任务不负责 BIOS 通电启动，也不会从关机状态唤醒电脑。
验收需实际检查任务启动后的公网鉴权响应，不能只凭任务显示“正在运行”判断成功。


## 原生论文对照的实时结果

网站默认展示独立的原生30点对照；旧B200的24个1K结果保留在 `/history`。原生快照由 `/api/native-comparison` 读取，不覆盖旧 `/api/scan` 数据。

本机只读发布器 `scripts/publish_exploring_hbf_native.py --background` 每15秒生成 `artifacts/exploring_hbf_native_20261019/progress_analysis.json/.md`，再通过官方Netlify Blobs SDK写入独立的 `native-comparison/latest.json`。它复用本机既有Netlify CLI登录，凭据只留在Node内存；不写前端、URL、日志或上传参数，不改变公开写接口的鉴权和账号权限。

真实发布PID、成功时间与脱敏错误保存在 `tmp/native_comparison_publisher.json`。网站中的任务进度只属于当前批次，不能当整个候选或整轮的完成比例。只有真实已完成配置才显示吞吐、TPOT和同模型同负载的存储/HBM4比值；缺少基线时留空。

发布器只读扫描数据并更新分析报告，不启动、暂停、恢复或重启仿真，也没有加入开机扫描任务。原管理器仍独立维护 `report.md`；分析报告另存，避免被每分钟刷新覆盖。没有足够完整候选时，整轮ETA保持未知，局部批次耗时估计注明观察时间并在20分钟后失效。
