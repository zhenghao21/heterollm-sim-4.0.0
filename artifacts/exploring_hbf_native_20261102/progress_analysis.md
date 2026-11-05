# 原生30点对照进度与已完成结果

观察时间：2026-11-04T16:08:17.699050+08:00。状态：完成。
已验收完成配置：30/30；已验收完成候选：326/591。

## 正在运行

|候选|批量/副本|实际worker PID|壁钟小时|当前阶段|当前cohort任务|累计CPU小时|RSS GiB|
|---|---:|---:|---:|---|---:|---:|---:|

## 已暂停尝试

- native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186：批量1186，已暂停；运行0.53小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136：批量1136，已暂停；运行0.33小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512：批量512，已暂停；运行0.03小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164：批量1164，已暂停；运行0.02小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792：批量2792，已暂停；运行0.08小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856：批量856，已暂停；运行0.50小时，保留该次实际进度，不计为完成候选。
- native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890：批量890，已暂停；运行0.26小时，保留该次实际进度，不计为完成候选。

## 累计Token与批次

|候选|已完成批次|实际Prefill Token|实际可见输出Token|已完成Decode批次/窗口目标|预热/测量Decode批次|
|---|---:|---:|---:|---:|---:|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|0|0|0|0/1494|0/0|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|0|0|0|0/1494|0/0|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|0|0|0|0/1494|0/0|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|0|0|0|0/1494|0/0|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|0|0|0|0/1116|0/0|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|0|0|0|0/1116|0/0|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|0|0|0|0/1116|0/0|

## 真实壁钟耗时分项

|候选|原生阶段|累计秒数|
|---|---|---:|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|启动|1.656|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|原生内部阶段|1.101|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|校验|3.565|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|批次完成|2.224|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|原生准入及PD入站|20.764|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|批次状态准备|0.288|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|批次任务图生成|26.098|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|调度前沿预演|3.672|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|物理任务执行|6.889|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|资源调度准备|27.490|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|原生完整层窗口采样|1815.276|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2__B1186|原生内部阶段|0.101|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|启动|1.660|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|原生内部阶段|1.111|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|校验|3.439|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|批次完成|2.234|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|原生准入及PD入站|20.076|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|批次状态准备|0.197|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|批次任务图生成|35.871|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|调度前沿预演|6.842|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|物理任务执行|15.121|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|资源调度准备|44.519|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|原生完整层窗口采样|1060.496|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP2_EP2_M2__B1136|原生内部阶段|0.064|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|启动|1.613|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|原生内部阶段|0.938|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|校验|1.746|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|批次完成|0.994|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|原生准入及PD入站|9.652|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|批次状态准备|0.151|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|批次任务图生成|16.258|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|调度前沿预演|4.158|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|物理任务执行|4.975|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|资源调度准备|18.964|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP2_PP4_EP1__B512|原生完整层窗口采样|54.301|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|启动|1.635|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|原生内部阶段|1.462|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|校验|2.694|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|批次完成|1.661|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|原生准入及PD入站|24.902|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|批次状态准备|0.244|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|批次任务图生成|42.129|
|native__llama4_maverick__HBM4__MID__G8__DP1_TP4_PP1_EP2__B1164|调度前沿预演|0.797|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|启动|1.831|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|原生内部阶段|0.921|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|校验|5.033|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|批次完成|3.490|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|原生准入及PD入站|18.058|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|批次状态准备|0.332|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|批次任务图生成|19.293|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|调度前沿预演|1.874|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|物理任务执行|4.868|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|资源调度准备|21.721|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|原生完整层窗口采样|217.865|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP1_PP4_EP2_M2__B2792|原生内部阶段|0.012|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|启动|1.619|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|原生内部阶段|0.994|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|校验|1.769|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|批次完成|1.149|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|原生准入及PD入站|9.127|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|批次状态准备|0.078|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|批次任务图生成|23.368|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|调度前沿预演|6.695|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|物理任务执行|8.813|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|资源调度准备|26.023|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|原生完整层窗口采样|1715.635|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP2_PP4_EP1_M2__B856|原生内部阶段|0.268|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|启动|1.629|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|原生内部阶段|1.143|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|校验|3.020|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|批次完成|1.966|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|原生准入及PD入站|20.892|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|批次状态准备|0.130|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|批次任务图生成|40.033|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|调度前沿预演|5.834|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|物理任务执行|15.070|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|资源调度准备|47.647|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|原生完整层窗口采样|791.252|
|native__llama4_maverick__HBM4__SHORT__G8__DP1_TP4_PP2_EP1__B890|原生内部阶段|0.063|

上述任务计数仅属于当前cohort，不代表整个候选的完成比例。累计Token和已完成批次不随下一批任务归零。

## 已完成配置

|模型|存储|负载|输入/输出Token|选定布局DP/TP/PP/EP|选定批量/副本|tokens/s/GPU|集群tokens/s|TPOT ms|吞吐/HBM4|TPOT/HBM4|SLO内最大批量/GPU（布局）|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|llama3_405b|HBM4|SHORT|1660/373|2/4/1/1|335|2074.636|16597.090|40.366|1.000|1.000|117.625（1/8/1/1）|
|llama3_405b|HBM4|MID|5900/499|1/8/1/1|458|938.257|7506.054|61.008|1.000|1.000|57.250（1/8/1/1）|
|llama3_405b|HBM4|LONG|103500/1100|1/8/1/1|23|90.285|722.282|31.843|1.000|1.000|2.875（1/8/1/1）|
|llama3_405b|CONV|SHORT|1660/373|1/8/1/1|319|422.317|3378.537|94.418|0.204|2.339|39.875（1/8/1/1）|
|llama3_405b|CONV|MID|5900/499|1/8/1/1|117|165.360|1322.880|88.442|0.176|1.450|14.625（1/8/1/1）|
|llama3_405b|CONV|LONG|103500/1100|1/8/1/1|12|17.669|141.356|84.892|0.196|2.666|1.500（1/8/1/1）|
|llama3_405b|HBF|SHORT|1660/373|4/2/1/1|230|1491.148|11929.180|77.122|0.719|1.911|115.000（4/2/1/1）|
|llama3_405b|HBF|MID|5900/499|4/2/1/1|161|892.657|7141.257|90.175|0.951|1.478|80.500（4/2/1/1）|
|llama3_405b|HBF|LONG|103500/1100|1/8/1/1|90|115.231|921.848|97.630|1.276|3.066|11.250（1/8/1/1）|
|llama3_405b|CONV+|SHORT|1660/373|1/8/1/1|377|567.769|4542.153|82.929|0.274|2.054|47.125（1/8/1/1）|
|llama3_405b|CONV+|MID|5900/499|1/8/1/1|165|238.287|1906.293|86.555|0.254|1.419|20.625（1/8/1/1）|
|llama3_405b|CONV+|LONG|103500/1100|1/8/1/1|11|19.384|155.072|70.935|0.215|2.228|1.375（1/8/1/1）|
|llama3_405b|HBF+|SHORT|1660/373|4/2/1/1|373|2405.935|19247.479|77.512|1.160|1.920|186.500（4/2/1/1）|
|llama3_405b|HBF+|MID|5900/499|4/2/1/1|170|1019.919|8159.355|83.339|1.087|1.366|85.000（4/2/1/1）|
|llama3_405b|HBF+|LONG|103500/1100|1/8/1/1|99|127.473|1019.785|97.078|1.412|3.049|12.375（1/8/1/1）|
|llama4_maverick|HBM4|SHORT|1660/373|1/1/1/8|3360|12062.106|96496.849|34.820|1.000|1.000|476.250（1/4/1/2）|
|llama4_maverick|HBM4|MID|5900/499|1/2/1/4|1140|3867.461|30939.688|36.845|1.000|1.000|149.125（1/8/1/1）|
|llama4_maverick|HBM4|LONG|103500/1100|1/8/1/1|205|800.180|6401.442|32.024|1.000|1.000|28.500（1/4/1/2）|
|llama4_maverick|CONV|SHORT|1660/373|1/2/1/4|1280|1812.435|14499.481|88.279|0.150|2.535|170.000（1/1/1/8）|
|llama4_maverick|CONV|MID|5900/499|1/4/1/2|612|804.464|6435.713|95.094|0.208|2.581|76.500（1/4/1/2）|
|llama4_maverick|CONV|LONG|103500/1100|2/4/1/1|66|173.485|1387.878|95.109|0.217|2.970|16.500（4/2/1/1）|
|llama4_maverick|HBF|SHORT|1660/373|2/1/1/4|2980|7991.930|63935.438|93.218|0.663|2.677|745.000（2/1/1/4）|
|llama4_maverick|HBF|MID|5900/499|1/1/1/8|1648|3143.489|25147.915|65.532|0.813|1.779|206.000（1/1/1/8）|
|llama4_maverick|HBF|LONG|103500/1100|2/2/1/2|328|888.271|7106.170|92.313|1.110|2.883|82.000（2/2/1/2）|
|llama4_maverick|CONV+|SHORT|1660/373|1/2/1/4|1700|2336.201|18689.604|90.956|0.194|2.612|212.500（1/2/1/4）|
|llama4_maverick|CONV+|MID|5900/499|1/4/1/2|512|692.959|5543.670|92.339|0.179|2.506|64.000（1/2/1/4）|
|llama4_maverick|CONV+|LONG|103500/1100|1/4/1/2|150|199.234|1593.869|94.110|0.249|2.939|18.750（1/4/1/2）|
|llama4_maverick|HBF+|SHORT|1660/373|1/1/1/8|3088|6957.798|55662.382|55.477|0.577|1.593|512.000（1/1/1/8）|
|llama4_maverick|HBF+|MID|5900/499|1/1/1/8|1504|3764.357|30114.853|49.942|0.973|1.355|236.250（2/4/1/1）|
|llama4_maverick|HBF+|LONG|103500/1100|2/4/1/1|359|1057.958|8463.666|84.832|1.322|2.649|99.500（2/2/1/2）|

## 剩余时间

本轮已结束，无剩余扫描时间。

以管理器真实终态为准。


## 统计口径

- 本轮仅使用原生后端，按9＋6＋15顺序执行；每个候选（布局＋批量）在独立子进程中运行，按预测峰值内存自动增开并行，k个并行时单候选壁钟上限放宽为k倍。
- 每组在有限的(DP, TP, PP, EP)布局和每副本批量候选中，选择通过约束者里单GPU吞吐最高的一项，不代表全局最优。DP副本互相独立，只仿真一个副本，集群吞吐为单GPU吞吐乘GPU总数。快速预测与完整物理结果分别标记。
- 比值方向固定为当前存储/HBM4；只比较同模型、同负载、同GPU数、精度及采样合同的已完成结果，各存储取各自选中的布局与批量。
- 当前cohort的completed/total是任务数，不是Token数、请求数或整轮完成百分比。累计Token与批次数独立记录，仅在原生提交完成后增加。
- 累计Prefill Token只计实际执行的查询Token；PD导入历史KV不冒充Prefill计算。累计输出Token来自真实可见Token提交事件，包含真实首Token。
- 壁钟分项按原生阶段切换时刻独占累计，包含该阶段未完成的当前区间；它是仿真器运行耗时，不是硬件延迟。旧尝试未记录的累计Token和分项保持未知。
- 物理层或批次样本与条件预测窗口分别记录。层样本不计完整批次或Token。PP Decode、PP=1 Decode和Prefill均使用10%条件模型预算；未执行窗口没有物理轨迹，不是全窗口实测误差保证；预测拒绝指标为空，不详细回退。
- 旧详细参考和已经取消的5K、10K配置均不重跑。
