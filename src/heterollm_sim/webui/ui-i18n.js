"use strict";

(function initUiI18n(root, factory) {
  const api = factory(root);
  if (typeof module === "object" && module.exports) module.exports = api;
  root.UiI18n = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createUiI18n(globalRoot) {
  const LANGUAGES = Object.freeze(["zh-CN", "en"]);
  const textPairs = new Map();
  const textSources = new WeakMap();
  const textRendered = new WeakMap();
  const attributeSources = new WeakMap();
  const attributeRendered = new WeakMap();
  let currentLanguage = "zh-CN";
  let observer = null;
  let observedRoot = null;

  const ENGLISH_INTERFACE_TEXT = Object.freeze({
    "Overview": "结构总览",
    "Explicit Requests": "显式请求",
    "Runtime & Solver Diagnostics": "运行环境诊断",
    "operator · typed port · tensor edge": "算子 · 带类型端口 · 张量边",
    "Prediction Layer": "预测层",
    "Auxiliary Head": "辅助头",
    "Proposal Logits": "提议 Logits",
    "Proposal Output": "提议输出",
    "DECODER-ONLY TRANSFORMER": "仅解码器 Transformer",
    "DECODER HIDDEN BYPASS": "解码器隐藏旁路",
    "PACKAGE TOPOLOGY / SCHEMA 4.0.0": "封装拓扑 / SCHEMA 4.0.0",
  });
  const CANONICAL_PARENTHETICALS = new Set([
    "BF16", "FP16", "FP8", "INT8", "GPU", "HBM", "HBF", "SSD", "CIM", "CXL", "PCIe", "UCIe",
    "KV", "MoE", "MTP", "TP", "PP", "EP", "GQA", "MQA", "RMS", "MLP", "Q/K/V", "I/O",
    "dtype", "shape", "layout", "repeat", "layer_template", "overrides", "experts", "top-k", "auto",
  ]);

  const DEFAULT_TEXT_PAIRS = Object.freeze({
    "产品信息": "Product information", "场景操作": "Scenario actions", "工作台步骤": "Workbench steps",
    "诊断信息": "Diagnostic information", "关闭诊断": "Close diagnostics", "组件库": "Component library",
    "拓扑画布工具": "Topology canvas tools", "通信协议手动覆盖": "Manual communication-protocol override", "可留空": "Optional",
    "连接状态": "Connection status", "画布缩放比例": "Canvas zoom level",
    "硬件拓扑画布。方向键移动所选组件，按住 Shift 加速；Ctrl 或 Command 点击多选，空白拖动框选。": "Hardware topology canvas. Arrow keys move selected components; hold Shift to move faster, use Ctrl or Command to multi-select, and drag empty space for marquee selection.",
    "颜色图例": "Color legend", "属性检查器": "Property inspector", "只读模型摘要": "Read-only model summary",
    "模型结构总览与端口连接工具": "Model overview and port connection tools", "当前显示模式": "Current display mode",
    "缩小模型图": "Zoom out model graph", "放大模型图": "Zoom in model graph",
    "模型语义组件图。空白拖动平移，按住 Ctrl 或 Command 滚轮缩放，选择端口建立带维度连接。": "Model semantic component graph. Drag empty space to pan, hold Ctrl or Command while scrolling to zoom, and select ports to create dimensioned connections.",
    "Rank 映射摘要": "Rank mapping summary", "Rank 映射筛选": "Rank mapping filters", "名称、组件或分片语义": "Name, component, or shard semantics",
    "算子 Rank 映射分页": "Operator Rank mapping pagination", "张量 Rank 分片分页": "Tensor Rank shard pagination",
    "显式请求表，可横向滚动": "Explicit requests table; scrolls horizontally", "运行时回放控制": "Runtime playback controls",
    "上一个事件": "Previous event", "下一个事件": "Next event", "数据流拓扑视图工具": "Data-flow topology view tools",
    "缩小数据流拓扑": "Zoom out data-flow topology", "放大数据流拓扑": "Zoom in data-flow topology",
    "可滚动的 Trace 数据流拓扑；仅高亮当前时间实际活动的组件和链路": "Scrollable trace data-flow topology; highlights only components and links active at the current time",
    "聚焦拓扑画布后，使用方向键滚动画布，PageUp / PageDown 翻页滚动；聚焦组件节点后，方向键浏览相邻组件，Home / End 跳到首个或末个组件。键盘操作不会改变仿真或布局。": "When the topology canvas has focus, use arrow keys to scroll and PageUp / PageDown to scroll by a page. When a component node has focus, use arrow keys to browse nearby components and Home / End to jump to the first or last component. Keyboard navigation does not change the simulation or layout.",
    "硬件组件": "Hardware components", "收起事件详情": "Collapse event details", "语义事件流本地筛选": "Local semantic event-stream filters",
    "组件利用率图表管理": "Component utilization chart management", "请求指标：窄屏显示为响应式卡片，桌面表格可横向滚动": "Request metrics: responsive cards on narrow screens; desktop table scrolls horizontally",
    "关闭 JSON 对话框": "Close JSON dialog", "关闭模型预设": "Close model presets", "模型目录来源": "Model catalog source",
    "模型预设筛选": "Model preset filters", "名称、系列、来源": "Name, family, or source", "本地模型预设列表": "Local model preset list",
    "模型预设分页": "Model preset pagination", "例如 Qwen2.5 或 repo id": "For example, Qwen2.5 or a repo ID", "在线模型候选": "Online model candidates",
    "关闭硬件预设库": "Close hardware preset library", "硬件预设类型": "Hardware preset type", "组件预设筛选": "Component preset filters",
    "名称、厂商、型号或组件类型": "Name, vendor, model, or component type", "组件预设列表": "Component preset list",
    "架构预设筛选": "Architecture preset filters", "中英文名称、厂商、类别或来源": "Name, vendor, category, or source",
    "架构拓扑预设列表": "Architecture topology preset list", "关闭通信协议目录": "Close communication protocol catalog",
    "通信协议预设筛选": "Communication protocol preset filters", "协议、版本、组织或传输单元": "Protocol, version, organization, or transfer unit",
    "通信协议预设列表": "Communication protocol preset list", "关闭设置": "Close settings",
    "运行环境与求解器诊断": "Runtime and solver diagnostics", "关闭架构候选扫描": "Close architecture candidate scan",
    "关闭后台仿真窗口": "Close background simulation dialog", "通知": "Notifications",
    "通用计算（GPU）": "GPU Compute",
    "高带宽内存（HBM）": "High-Bandwidth Memory (HBM)",
    "高带宽闪存（HBF）": "High-Bandwidth Flash (HBF)",
    "固态硬盘（SSD）": "Solid-State Drive (SSD)",
    "首 Token 延迟（TTFT）": "Time to First Token (TTFT)",
    "每输出 Token 时间（TPOT）": "Time per Output Token (TPOT)",
    "端到端延迟（E2E）": "End-to-End Latency (E2E)",
    "当前场景": "Current scenario", "修改状态": "Modification status", "未修改": "Unmodified", "已修改": "Modified",
    "连接检查中": "Checking connection",
    "后端服务": "Backend service",
    "新建场景": "New scenario",
    "打开参考场景": "Open reference scenario",
    "导入 JSON": "Import JSON",
    "导出 JSON": "Export JSON",
    "设置": "Settings",
    "关于": "About",
    "保存设置": "Save settings",
    "恢复默认": "Restore defaults",
    "取消": "Cancel",
    "关闭": "Close",
    "确认": "Confirm",
    "应用": "Apply",
    "删除": "Delete",
    "复制": "Copy",
    "复制诊断信息": "Copy diagnostics",
    "重试": "Retry",
    "刷新": "Refresh",
    "运行": "Run",
    "停止": "Stop",
    "校验": "Validate",
    "上一步": "Previous",
    "下一步": "Next",
    "播放": "Play",
    "暂停": "Pause",
    "整理": "Arrange",
    "适配": "Fit",
    "缩放": "Zoom",
    "全屏": "Fullscreen",
    "退出全屏": "Exit fullscreen",
    "搜索": "Search",
    "筛选": "Filter",
    "全部": "All",
    "错误": "Errors",
    "警告": "Warnings",
    "成功": "Success",
    "详情": "Details",
    "结果": "Results",
    "摘要": "Summary",
    "状态": "Status",
    "名称": "Name",
    "类型": "Type",
    "类别": "Category",
    "阶段": "Phase",
    "来源": "Source",
    "开始时间": "Start time",
    "结束时间": "End time",
    "持续时间": "Duration",
    "仿真时间": "Simulation time",
    "当前事件": "Current event",
    "选中事件": "Selected event",
    "语义事件流": "Semantic event stream",
    "数据流拓扑": "Data-flow topology",
    "运行回放": "Run playback",
    "模型": "Model",
    "模型图": "Model graph",
    "硬件架构": "Hardware architecture",
    "部署映射": "Deployment mapping",
    "负载": "Workload",
    "运行结果": "Run results",
    "显示语言": "Display language",
    "界面主题": "Interface theme",
    "字体缩放": "Font scale",
    "简体中文": "Simplified Chinese",
    "此刻没有活动事件": "No event is active at this time",
    "未选择事件": "No event selected",
    "暂无数据": "No data",
    "暂无结果": "No results",
    "无活动诊断": "No active diagnostics",
    "正在加载": "Loading",
    "正在校验场景": "Validating scenario",
    "正在运行仿真": "Running simulation",
    "参考场景已载入": "Reference scenario loaded",
    "设置已保存": "Settings saved",
    "诊断信息已复制": "Diagnostics copied",
  });

  const STATIC_PAGE_TEXT_PAIRS = Object.freeze({
    "分析型": "ANALYTICAL",
    "已修改": "Modified",
    "载入参考": "Load reference",
    "导出标准 IR": "Export canonical IR",
    "查看 JSON": "View JSON",
    "运行仿真": "Run simulation",
    "架构": "Architecture",
    "待载入": "Waiting to load",
    "诊断面板": "Diagnostics panel",
    "尚无诊断信息。运行校验可检查拓扑、映射和当前 lowering 支持范围。": "No diagnostics yet. Run validation to check topology, mapping, and current lowering support.",
    "异构封装拓扑": "Heterogeneous package topology",
    "加速器计算": "Accelerator compute",
    "专用 HBM 链路": "Dedicated HBM links",
    "512 GiB · 约 3 TB/s · 分析估算": "512 GiB · about 3 TB/s · analytical estimate",
    "PCIe / CXL 存储": "PCIe / CXL storage",
    "高并发 PCIe / CXL": "High-concurrency PCIe / CXL",
    "数字 SRAM-CIM": "Digital SRAM-CIM",
    "存算一体": "Compute-in-memory",
    "允许冷 CIM 权重流式加载": "Allow cold CIM weight streaming",
    "会改变候选集合与映射成本": "Changes the candidate set and mapping cost",
    "拖动节点仅调整画布坐标，不会改变 scenario JSON。": "Dragging nodes changes canvas coordinates only; it does not change scenario JSON.",
    "版本": "Version",
    "通道 / lane / link 数": "Channel / lane / link count",
    "时延（ns）": "Latency (ns)",
    "内置默认值 · 可手动覆盖": "Built-in default · can be overridden",
    "选择节点或链路查看属性": "Select a node or link to inspect its properties",
    "未选择": "Nothing selected",
    "拓扑为空": "The topology is empty",
    "从左侧添加 GPU、HBM、HBF、SSD 或数字 SRAM-CIM。": "Add a GPU, HBM, HBF, SSD, or digital SRAM-CIM from the left.",
    "分层存储 / IO": "Tiered storage / I/O",
    "协议链路": "Protocol link",
    "空白框选 · Ctrl/Cmd 多选 · Space/中键平移 · Ctrl/Cmd C/V · Delete 删除": "Drag empty space to select · Ctrl/Cmd multi-select · Space/middle button to pan · Ctrl/Cmd C/V · Delete",
    "选择组件或链路以编辑常用字段。其余 Schema / metadata 字段保留在 JSON 中。": "Select a component or link to edit common fields. Other Schema / metadata fields remain in JSON.",
    "以带类型端口、张量形状和显式连接定义模型；重复 Block 可折叠，维度不匹配会在连线前阻止。": "Define the model with typed ports, tensor shapes, and explicit connections. Repeated Blocks can collapse, and incompatible dimensions are blocked before connection.",
    "结构总览": "Structure overview",
    "← 返回结构总览": "← Back to structure overview",
    "未选择组件 · 点击组件在右侧查看详情 · 端口可用于建立连接": "No component selected · select a component to inspect it on the right · ports remain available for connections",
    "连接端口": "Connect ports",
    "撤销": "Undo",
    "重做": "Redo",
    "点击组件钻取结构；从输出端点拖选或点击至输入端点建立连接": "Click a component to inspect its structure; drag or click from an output endpoint to an input endpoint to connect.",
    "模型语义组件图画布": "Model semantic component graph canvas",
    "输入端口": "Input ports",
    "输出端口": "Output ports",
    "权重端口": "Weight ports",
    "提示格式：DType [Shape] / Layout": "Format: DType [Shape] / Layout",
    "形状符号：B 批次 · T 序列 · H 隐藏维度 · I 中间维度 · V 词表": "Shape symbols: B batch · T sequence · H hidden dimension · I intermediate dimension · V vocabulary",
    "未选择组件": "No component selected",
    "选择叶子算子后查看说明、实例数、参数和具体端口；总览框只显示名称。": "Select a leaf operator to inspect its description, instance count, parameters, and concrete ports; overview boxes show names only.",
    "端口连接会修改权威 model.graph 并使映射过期；钻取、平移、缩放和整理只改变显示状态。": "Port connections modify authoritative model.graph and make mapping stale; drill-down, pan, zoom, and arrange only change view state.",
    "端口连接会修改权威 model.graph 并使映射过期；选择、平移、缩放和整理只改变显示状态。": "Port connections modify authoritative model.graph and make mapping stale; selection, pan, zoom, and arrange only change view state.",
    "新增层": "Add layer",
    "算子与张量映射": "Operator and tensor mapping",
    "映射键会直接写回场景；组件候选来自当前拓扑。": "Mapping keys are written directly to the scenario; component candidates come from the current topology.",
    "运行时放置状态": "Runtime placement status",
    "容量、放置、分配与调度由 CPU 控制平面动态决定；本页仅展示已物化结果。": "Capacity, placement, allocation, and scheduling are decided dynamically by the CPU control plane; this page only shows materialized results.",
    "运行时控制平面（Runtime Control Plane）": "Runtime Control Plane",
    "尚未物化（Not Materialized）": "Not Materialized",
    "运行时放置由内部控制平面物化；此处仅显示当前只读状态。": "Runtime placement is materialized by the internal control plane; this view is read-only.",
    "以此处为 TP/PP/EP 执行依据": "Use this as the TP/PP/EP execution authority",
    "搜索算子 / 张量 / 组件": "Search operators / tensors / components",
    "逻辑 Rank": "Logical Rank",
    "全部 Rank": "All Ranks",
    "执行 / 存储组件": "Execution / storage component",
    "全部组件": "All components",
    "清除筛选": "Clear filters",
    "操作": "Actions",
    "并行与驻留": "Parallelism and residency",
    "并行与 KV 策略输入": "Parallel and KV policy inputs",
    "每一行表示内部控制平面已物化的实际执行或驻留逻辑 Rank；此视图只读。": "Each row represents a materialized execution or residency logical Rank from the internal control plane; this view is read-only.",
    "请求负载": "Request workload",
    "支持显式 requests 或有效 synthetic workload；arrival 以纳秒表达，界面同时显示友好单位。": "Supports explicit requests or a valid synthetic workload; arrival uses nanoseconds and is also shown in readable units.",
    "显式请求": "Explicit requests",
    "新增请求": "Add request",
    "常用调度与 MTP 参数写入嵌套策略；SLO、抢占细节与 acceptance trace 可在高级 JSON 中继续编辑。": "Common scheduling and MTP parameters are stored in nested policies; edit SLO, preemption details, and acceptance trace in advanced JSON.",
    "推理运行时回放": "Inference runtime replay",
    "按仿真事件时间展示张量分片、逻辑内存区间、协议路径、Rank 算子计算与集合通信；不伪装为 JEDEC bank/row 级仿真。": "Shows tensor shards, logical memory intervals, protocol paths, Rank operator compute, and collectives by simulation event time; it is not JEDEC bank/row simulation.",
    "尚无 Trace": "No Trace yet",
    "运行仿真后生成确定性回放": "Run the simulation to generate deterministic replay",
    "详细 DES 返回精确事件；Scalable Serving 会明确标记代表性或聚合语义。": "Detailed DES returns exact events; Scalable Serving explicitly marks representative or aggregate semantics.",
    "运行当前场景": "Run current scenario",
    "回到起点": "Return to start",
    "全部请求": "All requests",
    "全部批次": "All batches",
    "聚合回放：当前报告没有可按批次读取的任务级 Trace。": "Aggregate replay: this report has no task-level Trace readable by batch.",
    "此刻发生了什么": "What is happening now",
    "时间空档": "Idle time",
    "移动时间游标后，这里会用中文概括当前真正处于执行区间内的工作。": "Move the time cursor to see a natural-language summary of work actually executing at that moment.",
    "选中事件与当前活动事件会分开说明。": "Selected and currently active events are described separately.",
    "张量与算子数据流": "Tensor and operator data flow",
    "整理并适配": "Arrange and fit",
    "拓扑全屏": "Topology fullscreen",
    "定位当前事件": "Locate active event",
    "选择事件后显示计算主体、数据流、时间并发与精确标识。": "Select an event to see the compute subject, data flow, temporal overlap, and exact identifiers.",
    "打开事件详情": "Open event details",
    "默认折叠 · 展开后每页最多 50 条": "Collapsed by default · at most 50 items per expanded page",
    "关键词": "Keyword",
    "全部类别": "All categories",
    "全部阶段": "All phases",
    "相对时间": "Relative time",
    "全部状态": "All states",
    "事件流本地筛选不改变全局回放顺序；选中状态与已发生、正在发生、尚未发生状态分别标记。": "Local event-stream filters do not change global replay order; selection and past/current/future states are marked independently.",
    "时序与持续条": "Timeline and duration bars",
    "语义事件": "Semantic event",
    "计算主体 / 数据流": "Compute subject / data flow",
    "请求范围": "Request scope",
    "状态 / 详情": "Status / details",
    "仿真结果": "Simulation results",
    "分析模型输出，不代表已校准硅后测量。架构决策前应使用已校准 profiles。": "Analytical model output, not calibrated post-silicon measurement. Use calibrated profiles before architecture decisions.",
    "与 GPU 基线比较": "Compare with GPU baseline",
    "重新运行": "Run again",
    "尚无仿真报告": "No simulation report yet",
    "校验当前场景后运行，以查看延迟、吞吐、能耗、瓶颈与关键路径。": "Validate and run the current scenario to inspect latency, throughput, energy, bottlenecks, and the critical path.",
    "组件与 Rank 性能时序": "Component and Rank performance series",
    "按保留策略 fidelity 原样展示，不把聚合数据伪装为精确硬件采样。": "Preserves retention-policy fidelity and never presents aggregate data as exact hardware samples.",
    "默认显示 1 张图，最多 4 张": "Shows 1 chart by default, up to 4",
    "添加图表": "Add chart",
    "推理运行时": "Inference runtime",
    "资源利用率": "Resource utilization",
    "类别时间 / 关键路径": "Category time / critical path",
    "请求明细": "Request details",
    "Token 间延迟（TBT）p": "Inter-Token latency (TBT) p",
    "当前 JSON": "Current JSON",
    "应用会替换当前场景；Schema 字段与 metadata 会保留，其他扩展字段仍需通过内核解析。": "Apply replaces the current scenario; Schema fields and metadata are preserved, while other extensions still require core parsing.",
    "JSON 可编辑": "JSON is editable",
    "应用 JSON": "Apply JSON",
    "选择模型后会保留并行与通用存储策略；运行时放置由内部控制平面按需重新物化，超出适用域（Out of Domain）的预设仅供查看。": "Selecting a model preserves parallel and general storage policies; runtime placement is rematerialized on demand by the internal control plane, while Out of Domain presets are view-only.",
    "应用预设只替换当前模型；所有以任一旧层 ID（Layer ID）为键或前缀的算子、张量及字节逐层映射都会清除，即使新模型复用同名层 ID。模型专属的层到阶段映射（Layer-to-Stage Mapping）也始终清空；TP/PP/EP 度数、Rank 映射（Rank Mapping）、集合通信算法（Collective Algorithm）等并行策略与模型权重（Model Weights）、KV 缓存（KV Cache）等通用映射保持不变。新的运行时放置会在后续内部控制平面物化时生成。": "Applying a preset replaces only the model. Operator, tensor, and byte-per-layer mappings keyed by any old Layer ID are cleared even if the new model reuses an ID. Model-specific Layer-to-Stage Mapping is always cleared; TP/PP/EP degrees, Rank Mapping, Collective Algorithm, Model Weights, KV Cache, and other general policies remain. New runtime placement is generated when the internal control plane materializes it later.",
    "全部等级": "All levels",
    "尚未载入预设。": "No presets loaded.",
    "只有提交搜索后才会请求在线目录；打开此对话框不会触发远程搜索。": "The online catalog is requested only after search submission; opening this dialog does not trigger remote search.",
    "修订（Revision，可选）": "Revision (optional)",
    "输入查询并主动搜索，或直接导入 Repo ID。": "Enter a query and search, or import a Repo ID directly.",
    "组件预设追加到当前拓扑；架构预设经确认后替换硬件。两套目录继续独立筛选、缓存与滚动，后端 API 保持分离。": "Component presets append to the topology; architecture presets replace hardware after confirmation. Their catalogs, filters, caches, scrolling, and backend APIs remain separate.",
    "组件预设 · 追加": "Component presets · append",
    "架构预设 · 替换": "Architecture presets · replace",
    "单组件和组合拓扑均追加，不覆盖已有硬件；只产生一次撤销记录。": "Single components and composite topologies append without replacing hardware and create one undo entry.",
    "全部类型": "All types",
    "全部厂商": "All vendors",
    "尚未载入组件预设。": "No component presets loaded.",
    "载入前需确认；替换组件、端口、链路、分组与布局，保留模型和负载。": "Confirmation is required; components, ports, links, groups, and layout are replaced while model and workload remain.",
    "全部层级": "All tiers",
    "全部支持状态": "All support states",
    "尚未载入架构预设。": "No architecture presets loaded.",
    "目录区分原始线路、单向有效与双向聚合带宽；选用后写入连接工具的单向仿真默认值，仍可手动覆盖。": "The catalog distinguishes raw lane, effective one-way, and aggregate bidirectional bandwidth; selection writes an overridable one-way simulation default.",
    "全部协议": "All protocols",
    "全部组织": "All organizations",
    "尚未载入通信协议目录。": "No communication protocol catalog loaded.",
    "工作台设置": "Workbench settings",
    "界面偏好仅保存在此浏览器；运行时控制平面状态为只读，不提供客户端规划设置。": "Interface preferences are stored only in this browser. Runtime control-plane status is read-only and exposes no client-side planning settings.",
    "界面语言": "Interface language",
    "界面一次只显示一种语言；模型名、组件标识和标准缩写保持原样。": "The interface shows one language at a time; model names, component identifiers, and standard acronyms remain unchanged.",
    "连续调整字体大小": "Continuously adjust font size",
    "恢复 100%": "Restore 100%",
    "实时调整全部文字，并同步扩展控件、侧栏、拓扑节点和响应式布局。": "Adjust all text immediately together with controls, sidebars, topology nodes, and responsive layout.",
    "确定性启发式（可解释）": "Deterministic heuristic (explainable)",
    "全局搜索（求最优）": "Global search (optimize)",
    "运行环境诊断": "Runtime diagnostics",
    "正在检查": "Checking",
    "正在读取 Python、程序版本和 OR-Tools CP-SAT 状态…": "Reading Python, application version, and OR-Tools CP-SAT status…",
    "重新检查运行环境": "Check runtime again",
    "运行环境（Runtime）": "Runtime",
    "显示本地仿真器、Python 与可选运行时依赖的只读诊断信息。": "Shows read-only diagnostics for the local simulator, Python, and optional runtime dependencies.",
    "TTFT、TPOT 与吞吐（throughput）是分析映射代理目标（analytical placement surrogate），不是完整事件仿真指标（not full event-simulation metrics）。全局搜索达到时间上限时会保留可行解；只有证明完成才标记“全局最优”。": "TTFT, TPOT, and throughput are analytical placement surrogate objectives, not full event-simulation metrics. A feasible solution is retained at the global-search time limit; only a completed proof is labeled globally optimal.",
    "仅覆盖中央工作区，不影响图表语义色": "Affects only the central workspace, not semantic chart colors",
    "完成": "Done",
    "基于当前真实拓扑、Rank、模型代表性 GEMM 与现有映射批量评估候选；这是 Roofline 分析扫描，不是事件仿真或完整部署计划。": "Batch-evaluates candidates using the current real topology, Ranks, representative model GEMM, and mapping. This is a Roofline scan, not event simulation or a full deployment plan.",
    "CuPy CUDA（可选）": "CuPy CUDA (optional)",
    "尚未运行候选扫描。": "Candidate scan has not run.",
    "局部形状（M × K × N）": "Local shape (M × K × N)",
    "扫描诊断与限制（Diagnostics & Limits）": "Scan diagnostics and limits",
    "先展示逻辑规模与推荐保留策略；运行期间可关闭此窗口继续查看场景，也可随时返回查看进度或安全取消。": "Review logical scale and the recommended retention policy first. You may close this window during execution, return for progress, or cancel safely.",
    "正在估算": "Estimating",
    "已排队": "Queued",
    "准备启动": "Preparing to start",
    "等待后台工作线程接收任务。": "Waiting for a background worker to accept the task.",
    "正在运行分析模型": "Running analytical model",
    "正在编译 ScheduleIR 与事件轨迹…": "Compiling ScheduleIR and event trace…",
    "前向计算": "Forward compute",
    "反向计算": "Backward compute",
    "算子": "Operator",
    "运行标记": "Runtime marker",
    "输出一个词元": "Emit one Token",
    "词元": "Token",
    "精确记录": "Exact record",
    "代表性记录": "Representative record",
    "聚合记录": "Aggregate record",
    "聚合批次，仅保留部分代表工作": "Aggregate batch retaining representative work only",
    "聚合路径，不含逐跳精确时间": "Aggregate path without exact per-hop timing",
    "批次时间范围内的聚合忙碌区间": "Aggregate busy interval within the batch envelope",
    "精确任务事件": "Exact task event",
    "精确事件": "Exact event",
    "代表性任务事件": "Representative task event",
    "未说明": "Not specified",
    "已选中": "Selected",
    "简要说明": "Summary",
    "记录方式": "Recording method",
    "模型位置": "Model location",
  });

  // Standalone headings need title-case English even when their Chinese copy
  // is also used by compact dynamic labels elsewhere in the UI.
  const TITLE_TEXT_PAIRS = Object.freeze({
    "诊断": "Diagnostics",
    "组件": "Components",
    "映射": "Mapping",
  });

  // Longest first: this also translates dynamic labels assembled in app.js.
  const PHRASE_PAIRS = Object.freeze(Object.entries({
    "运行校验可检查当前场景": "Run validation to check the current scenario",
    "成功状态只代表结构与当前 lowering 约束通过": "Success only means the structure and current lowering constraints passed",
    "操作失败，未提供中文说明": "The operation failed without a localized explanation",
    "未提供中文诊断说明，请根据错误代码检查输入": "No localized diagnostic was provided; inspect the input using the error code",
    "没有符合当前筛选条件的事件": "No events match the current filters",
    "每秒推进一个筛选后的事件": "Advance one filtered event per second",
    "逻辑地址": "Logical address",
    "活动分片": "Active shard",
    "尚未完全放置": "Placement incomplete",
    "运行时放置仍有": "Runtime placement still has",
    "不能运行": "and cannot run",
    "未放置": "unplaced",
    "已过期": "Stale",
    "报告就绪": "Report ready",
    "结果已过期": "Results stale",
    "另外": "other",
    "并发工作": "concurrent work",
    "同时进行的工作": "Concurrent work",
    "当前显式": "Current explicit",
    "保持契约校验": "preserve contract validation",
    "语义组": "semantic groups",
    "内部链路": "internal links",
    "关联链路": "related links",
    "逻辑工作单元": "logical workers",
    "容量": "Capacity",
    "已发生": "Past",
    "正在发生": "Current",
    "尚未发生": "Future",
    "提示词处理": "Prefill",
    "逐词生成": "Decode",
    "数据传输": "Data transfer",
    "内存访问": "Memory access",
    "计算": "Compute",
    "通信": "Communication",
    "调度器": "Scheduler",
    "调度": "Scheduling",
    "任务": "Task",
    "批次": "Batch",
    "事件": "Event",
    "组件": "components",
    "链路": "links",
    "节点": "nodes",
    "端口": "ports",
    "映射": "mapping",
    "请求": "requests",
    "算子": "operators",
    "张量": "tensors",
    "分组": "groups",
    "组": "groups",
    "层": "layers",
    "项": "items",
    "条": "items",
    "个": "",
    "时间": "Time",
    "数量": "Count",
    "利用率": "Utilization",
    "吞吐量": "Throughput",
    "延迟": "Latency",
    "能耗": "Energy",
    "证据": "Evidence",
    "限制": "Limitations",
    "诊断": "Diagnostics",
    "场景": "Scenario",
    "正在": "In progress",
    "已完成": "Completed",
    "未运行": "Not run",
    "未提供": "Not provided",
    "默认": "Default",
    "启用": "Enabled",
    "禁用": "Disabled",
    "展开": "Expand",
    "折叠": "Collapse",
    "上一页": "Previous page",
    "下一页": "Next page",
    "第": "Page ",
    "页": "",
    "共": "of",
  }).sort((a, b) => b[0].length - a[0].length));

  [DEFAULT_TEXT_PAIRS, STATIC_PAGE_TEXT_PAIRS, TITLE_TEXT_PAIRS].forEach((bundle) => {
    Object.entries(bundle).forEach(([zh, en]) => textPairs.set(zh, en));
  });

  function normalizeLanguage(value) {
    return String(value || "").trim().toLowerCase().startsWith("en") ? "en" : "zh-CN";
  }

  function interpolate(value, parameters = {}) {
    return String(value ?? "").replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => (
      Object.hasOwn(parameters, key) ? String(parameters[key]) : match
    ));
  }

  function pair(zh, en, parameters = {}) {
    return interpolate(currentLanguage === "en" ? en : zh, parameters);
  }

  function directBilingualPair(value) {
    const text = String(value ?? "").trim();
    let match = text.match(/^(.+?[\u3400-\u9fff].*?)\s*[·｜|]\s*([A-Za-z][A-Za-z0-9 /&+,:.'()_·-]*)$/u);
    if (!match) match = text.match(/^(.+?[\u3400-\u9fff].*?)\s*\/\s*(Overview)$/u);
    if (!match) {
      match = text.match(/^(.+?[\u3400-\u9fff].*?)[（(]([A-Za-z][A-Za-z0-9 /&+,:.'·_-]{2,})[）)]$/u);
      if (match && canonicalParenthetical(match[2])) match = null;
    }
    return match ? { zh: match[1].trim(), en: match[2].trim() } : null;
  }

  function canonicalParenthetical(value) {
    const text = String(value || "").trim();
    if (CANONICAL_PARENTHETICALS.has(text)) return true;
    if (/^[a-z][a-z0-9_]*\.[a-z0-9_.[\]-]+$/u.test(text)) return true;
    return /^(?:[A-Z][A-Z0-9]*)(?:\s*[/+·]\s*[A-Z][A-Z0-9]*)*$/u.test(text);
  }

  function chineseInterfaceText(value) {
    let text = String(value || "");
    text = text.replace(/[（(]([^（）()]*)[）)]/gu, (match, inner) => {
      if (!/[A-Za-z]/u.test(inner)) return match;
      if (canonicalParenthetical(inner)) return match;
      if (/[\u3400-\u9fff]/u.test(inner)) {
        const chinese = inner.replace(/[A-Za-z][A-Za-z0-9 /&+,:.'·_-]*/gu, "").replace(/^[，、\s]+|[，、\s]+$/gu, "");
        return chinese ? `（${chinese}）` : "";
      }
      return "";
    });
    return text.replace(/\s{2,}/gu, " ").replace(/\s+([，。；：])/gu, "$1").trim();
  }

  function localizeText(value, language = currentLanguage) {
    const source = String(value ?? "");
    const leading = source.match(/^\s*/u)?.[0] || "";
    const trailing = source.match(/\s*$/u)?.[0] || "";
    const text = source.trim();
    const canonicalWithChineseDefinition = text.match(/^([A-Z][A-Za-z0-9/+.-]*)[（(][\u3400-\u9fff\s]+[）)]$/u);
    if (canonicalWithChineseDefinition) {
      return `${leading}${language === "en" ? canonicalWithChineseDefinition[1] : text}${trailing}`;
    }
    if (language !== "en" && ENGLISH_INTERFACE_TEXT[text]) return `${leading}${ENGLISH_INTERFACE_TEXT[text]}${trailing}`;
    if (!/[\u3400-\u9fff]/u.test(source)) return source;
    const direct = directBilingualPair(text);
    if (direct) return `${leading}${language === "en" ? direct.en : chineseInterfaceText(direct.zh)}${trailing}`;
    if (language !== "en") return `${leading}${chineseInterfaceText(text)}${trailing}`;
    if (textPairs.has(text)) return `${leading}${textPairs.get(text)}${trailing}`;
    let translated = text;
    PHRASE_PAIRS.forEach(([zh, en]) => { translated = translated.split(zh).join(en); });
    // Phrase replacement is only a safe fallback when it resolves the complete
    // phrase.  Returning a half-translated string turns words such as “时间” or
    // “算子” into corrupt hybrids and can also rewrite backend-owned prose.
    // Front-owned dynamic copy belongs in an explicit pair()/uiText() template;
    // unknown text stays intact so its provenance remains visible and testable.
    if (/[\u3400-\u9fff]/u.test(translated)) return source;
    return `${leading}${translated}${trailing}`;
  }

  function shouldSkipTextNode(node) {
    const parent = node?.parentElement;
    if (!parent) return true;
    if (["SCRIPT", "STYLE", "CODE", "PRE", "TEXTAREA"].includes(parent.tagName)) return true;
    return Boolean(parent.closest?.("[data-i18n-skip], [data-i18n-en]"));
  }

  function localizeTextNode(node) {
    if (shouldSkipTextNode(node)) return;
    const current = String(node.nodeValue ?? "");
    const last = textRendered.get(node);
    if (!textSources.has(node) || current !== last) textSources.set(node, current);
    const localized = localizeText(textSources.get(node), currentLanguage);
    if (localized !== current) node.nodeValue = localized;
    textRendered.set(node, localized);
  }

  function localizeAttribute(node, attribute) {
    if (!node?.hasAttribute?.(attribute) || node.closest?.("[data-i18n-skip]")) return;
    let sources = attributeSources.get(node);
    let rendered = attributeRendered.get(node);
    if (!sources) { sources = {}; attributeSources.set(node, sources); }
    if (!rendered) { rendered = {}; attributeRendered.set(node, rendered); }
    const current = node.getAttribute(attribute) || "";
    if (!(attribute in sources) || current !== rendered[attribute]) sources[attribute] = current;
    const localized = localizeText(sources[attribute], currentLanguage);
    if (localized !== current) node.setAttribute(attribute, localized);
    rendered[attribute] = localized;
  }

  function walkText(rootNode) {
    if (!rootNode) return;
    if (rootNode.nodeType === 3) return localizeTextNode(rootNode);
    const documentNode = rootNode.ownerDocument || rootNode;
    const walker = documentNode.createTreeWalker?.(rootNode, globalRoot.NodeFilter?.SHOW_TEXT || 4);
    if (walker) while (walker.nextNode()) localizeTextNode(walker.currentNode);
    const scope = rootNode.querySelectorAll ? rootNode : null;
    if (!scope) return;
    [rootNode, ...scope.querySelectorAll("[placeholder], [aria-label], [title]")]
      .forEach((node) => ["placeholder", "aria-label", "title"].forEach((attribute) => localizeAttribute(node, attribute)));
  }

  function localize(rootNode) {
    if (rootNode?.nodeType === 3) {
      localizeTextNode(rootNode);
      return;
    }
    const scope = rootNode?.querySelectorAll ? rootNode : null;
    if (!scope) return;
    const pairedNodes = [];
    if (scope.matches?.("[data-i18n-en]")) pairedNodes.push(scope);
    pairedNodes.push(...scope.querySelectorAll("[data-i18n-en]"));
    pairedNodes.forEach((node) => {
      if (!node.hasAttribute("data-i18n-zh")) node.setAttribute("data-i18n-zh", node.textContent || "");
      node.textContent = interpolate(node.getAttribute(currentLanguage === "en" ? "data-i18n-en" : "data-i18n-zh") || "", node.dataset);
    });

    ["placeholder", "aria-label", "title"].forEach((attribute) => {
      const englishAttribute = `data-i18n-${attribute}-en`;
      const chineseAttribute = `data-i18n-${attribute}-zh`;
      const targets = [];
      if (scope.matches?.(`[${englishAttribute}]`)) targets.push(scope);
      targets.push(...scope.querySelectorAll(`[${englishAttribute}]`));
      targets.forEach((node) => {
        if (!node.hasAttribute(chineseAttribute)) node.setAttribute(chineseAttribute, node.getAttribute(attribute) || "");
        node.setAttribute(attribute, node.getAttribute(currentLanguage === "en" ? englishAttribute : chineseAttribute) || "");
      });
    });
    walkText(rootNode);
  }

  function observe(rootNode) {
    if (typeof globalRoot.MutationObserver !== "function" || !rootNode) return;
    const target = rootNode.documentElement || rootNode;
    if (observedRoot === target) return;
    observer?.disconnect();
    observedRoot = target;
    observer = new globalRoot.MutationObserver((records) => {
      records.forEach((record) => {
        if (record.type === "characterData") localizeTextNode(record.target);
        else if (record.type === "attributes") localizeAttribute(record.target, record.attributeName);
        else record.addedNodes.forEach((node) => localize(node));
      });
    });
    observer.observe(target, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["placeholder", "aria-label", "title"] });
  }

  function setLanguage(language, rootNode = typeof document !== "undefined" ? document : null) {
    currentLanguage = normalizeLanguage(language);
    const documentElement = rootNode?.documentElement || rootNode?.ownerDocument?.documentElement;
    if (documentElement) {
      documentElement.lang = currentLanguage;
      documentElement.dataset.language = currentLanguage;
    }
    localize(rootNode);
    observe(rootNode);
    const eventTarget = rootNode?.dispatchEvent ? rootNode : rootNode?.ownerDocument;
    if (eventTarget && typeof globalRoot.CustomEvent === "function") {
      eventTarget.dispatchEvent(new globalRoot.CustomEvent("ui-languagechange", { detail: { language: currentLanguage } }));
    }
    return currentLanguage;
  }

  function language() { return currentLanguage; }

  const api = Object.freeze({
    LANGUAGES, interpolate, language, localize, localizeText, normalizeLanguage,
    pair, setLanguage,
  });
  return api;
});
