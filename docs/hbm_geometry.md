# HBM 几何、控制与成本计量的 JEDEC 依据

`hbm_geometry.py` 直接采用已取得的 JEDEC 协议原文件中的寻址表。文件通过公开镜像取得；镜像站不是标准作者，镜像中的介绍文章和厂商白皮书不作为几何依据。`standard_organization` 只表示采用了标准允许的组织，不表示某张 GPU 或某个厂商料号已被验证。

## 已核对的协议

| 模型名称 | 原文件与直接依据 | 本地文件 | 原文件公开镜像 |
|---|---|---|---|
| HBM2 / HBM2E | JESD235B，2018-11；§3.2.1.2，印刷 p.4；表 4，p.6–7；表 5，p.8 | `.refs/jedec/JESD235B-2018.pdf`，207 PDF 页 | [GitCode 中的 JESD235B-HBM.pdf](https://gitcode.com/Open-source-documentation-tutorial/a1451) |
| HBM3 / HBM3E | JESD238，2022-01；§3.1.2，印刷 p.4–5；表 4/5，p.6–7 | `.refs/jedec/JESD238-2022.pdf`，268 PDF 页 | [GitCode 中的 JESD238-HBM3.pdf](https://gitcode.com/Open-source-documentation-tutorial/69c72) |
| HBM4 | JESD270-4A，2025；§3.1.2；表 4/5，印刷 p.6–7 | `.refs/jedec/JESD270-4A-2025.pdf`，286 PDF 页 | [GitHub 中的 JEDECJESD270-4A-2025.pdf](https://github.com/Adilohit/HBM4_verification/blob/main/resources/JEDECJESD270-4A-2025.pdf) |

另外取得了 [JESD235A 原文件的 Internet Archive 镜像](https://web.archive.org/web/20220514151205id_/https://composter.com.ua/documents/JESD235A.pdf)，存于 `.refs/jedec/JESD235A_archive_composter.pdf`（172 PDF 页），用于历史对照。JESD235C 全文网页镜像、JESD235D 的引用材料、JESD238B.01 和 JESD270-4 的三页预览，不代表取得或核准了这些版本的完整协议；当前实现不使用它们作为几何来源。HBM2E / HBM3E 是模型的技术代名称，不能据此推断其速度等级或厂商实现。

原协议全文只保存在本地 `.refs/jedec`，不随源代码提交。

## 字段口径

| 字段 | HBM2 / HBM2E | HBM3 / HBM3E | HBM4 |
|---|---:|---:|---:|
| 一个完整 stack 的物理通道数 | 8 | 16 | 32 |
| 每物理通道的伪通道数（PC） | 2 | 2 | 2 |
| 每 PC 数据位宽 | 64 bit | 32 bit | 32 bit |
| 每 stack 外部数据总位宽 | 1024 bit | 1024 bit | 2048 bit |
| 突发长度 | BL4 | BL8 | BL8 |
| 每 PC 每次突发数据 | 32 B | 32 B | 32 B |
| 每 PC 有效寻址页 `row_bytes` | 1024 B | 1024 B | 1024 B |
| 每个已激活页 | 当前不作物理大小断言 | 2048 B | 2048 B |
| 每 bank group 的 bank 数 | 4（已导入组织） | 4 | 8 |

PC 是伪通道。每个 PC 有独立可寻址 bank 空间，两个 PC 仍共享所在物理通道的命令接口。`channels` 表示每 stack 的物理通道数；执行模型的 lane 数是 `stacks × channels × 2`。SID 是扩展 bank 地址的 stack ID，不是另一个应再次乘入容量的完整地址空间。因此采用 `bank_geometry_scope="per_stack"`，物理 `dies_per_stack` 只作组织描述。

当前 HBM2/E 目录明确采用 Pseudo Channel mode。JESD235B §3.2.1 同时定义 Legacy mode（称为 HBM1）与 PC mode（称为 HBM2），因此每通道 2 PC 不是该协议覆盖的全部组织。模式支持由器件设计固定：DEVICE_ID[17:16] 的 01 表示只支持 PC，10 表示只支持 Legacy，00/11 非法（表 99，印刷 p.184）；不能推断一个 HBM2 器件可在两者间运行时切换。当前不导入 Legacy 组织。JESD238 §3.1.2 与 JESD270-4A §3.1.2 则直接规定每通道两个 PC，其通道定义没有 Legacy 可选模式；HBM3/E、HBM4 的 2 PC 来自各自协议。

JESD238 和 JESD270-4A 表 4 注 3 明确区分 1 KiB 有效寻址页与已打开的 2 KiB 页：RA 最高位选择其一半。`rows_per_bank` 在本模型中是有效行地址数；`activated_page_bytes=2048` 单独记录激活页大小。协议控制中的已打开行身份须清除该 RA 最高位，两个半页可以共享一次激活。24 Gbit 组织不是 2 的幂：HBM3 清除 RA14，HBM4 清除 RA13；不能使用 `row % (rows_per_bank / 2)` 代替实际地址位掩码。

## 已导入组织与容量

令 D 为单 die 密度（Gbit），H 为每 stack 的物理 die 层数，S 为 stack 数：

| 模型 | 已导入 D | 已导入 H | 每 PC 的 bank 数 | 每 PC 的 bank group 数 | 每 bank 的有效行地址数 |
|---|---|---|---:|---:|---:|
| HBM2 / HBM2E | 16 | 4、8 | 4H | H | 32768 |
| HBM3 / HBM3E | 8、16、24、32 | 4、8、12、16 | 4H | H | D × 1024 |
| HBM4 | 24、32 | 4、8、12、16 | 4H | H / 2 | D × 512 |

HBM2/E 只导入 JESD235B 中的一个有限子集；未导入不等于 JEDEC 禁止。HBM2/E 与 HBM3/E 的无 SID 基础组织，其物理 stack 高度由厂商决定；4H 是本模型对基础组织的具体实现选择。HBM4 表 4 直接列出 4H，注 8 排除 SID[0]=1，不沿用前两代的这一限制。

HBM3 的 24 Gbit 组织排除 RA[14:13]=11，故有效行数为 24576；HBM4 的 24 Gbit 组织排除 RA[13:12]=11，故有效行数为 12288。12H 排除 SID[1:0]=11，得到 3 个 SID × 16 bank = 48 bank/PC。JESD238 原版表 4 的部分脚注上标编号错位；这里按脚注文字、寻址位及表中每 PC 容量核对，未照抄错误上标。

原始可寻址容量使用二进制密度：

```text
raw_capacity_bytes = S × D × H × 2^30 / 8
                   = lane_count × banks_per_PC × rows_per_bank × row_bytes
```

例如 24 Gbit × 12H 的单 stack 是 36 GiB，即 38,654,705,664 B。这个数是寻址组织导出的原始容量，不是对厂商营销标注的推断。`capacity_bytes` 是另行给定的可见容量预算；它可以小于原始容量，但不会通过增大行数去凑容量。多 stack 的带宽输入也按整体预算处理，不再次乘 stack 数。

缺少 D/H 时，选覆盖可见预算的最小原始容量；容量相同时优先 8H、12H、16H、4H。选出的 D/H 标成 `selected_reference`，目标器件的实际 D/H 仍列在 `unknown_parameters`。

## 配置修改的依赖规则

HBM 预设、实验脚本、配置解析和 `reconfigure_physical_config` 共用依赖规则。修改 `generation`、`dies_per_stack` 或合法 `rows_per_bank` 时，同步核对通道、bank、有效行地址与刷新组织。标准允许的行数可以确定本模型导入的 die 密度；自定义行数不能沿用旧密度的标准标签或 tRFC。自定义几何保留为 `explicit_assumption`，仍须通过地址映射及容量约束校验。修改高度或行数后若原始容量不足，会报错，不会自动改变可见容量。

底层 `DramConfig` 构造、`dataclasses.replace`、字典解析和硬件导入均执行相同规则。界面的物理配置与 bank-group 编辑也调用后端公共修改入口，并展示返回的配置。完整 JSON 中保留自动来源标签的字段仍是派生值；需要固定实验时序时，应使用修改入口，或明确将该字段来源改为 `explicit_assumption`。直接修改带标准来源的派生数值，不会将它变成新的器件规格。

`stacks` 是独立 stack 数，`dies_per_stack` 是每 stack 的高度，不能互相替代。普通参考预设可以按容量预算选择覆盖的标准组织；直接重配已有配置不隐式增添 stack。微型实验默认 HBM3E 单 stack，但显式几何输入与普通参考使用相同规则。`expand_geometry=True` 只用于明确的等效几何扩展；标准研究配置通过重新选择合法组织满足容量预算。

接口工作点和方向限速分别配置。`interface_bandwidth_gb_s` 是所有 stack 的物理接口总带宽；`command_clock_ns` 是命令 CK 周期；`data_rate_mt_s` 是每根 DQ 的传输速率。它们须遵循本代 CK/DQ 比例。方向字段 `read_bandwidth_gb_s` / `write_bandwidth_gb_s` 仅约束请求发射节拍，不能据此重新计算 CK。主服务实验独立指定 `command_clock_ns=0.625`，不再由方向预算推算接口工作点。文献探针可另行显式输入接口总带宽；该输入是探针假设，不代表目标器件实际时钟已知。微基准保留各自明确的接口与时序条件。

来源标为自动推导或参考选择的字段随依赖更新，包括 CK、突发占用、tCCD、PRE 间隔、RL/WL 的 ns 值、IDD 参考 tRAS/tRP/tRC、按密度和高度查表的 tRFC，以及 `9 × tREFI` 的最大激活时间。RL/WL 保留所选周期编码，再按新 CK 换算。显式实验时序保留并标注假设，但必须满足已知硬约束；未知器件时序不会因为几何确定就变成已知。冲突的接口/时钟输入、非法模式寄存器编码和不够覆盖可见容量的组织应明确报错。

所有 `jedec_hbm` 配置入口共用时序校验，包括直接构造、序列化读取、`dataclasses.replace`、预设、实验覆盖、硬件配置及网页编辑。读取已有配置时重算旧输入对应的公式，以区分未改变的派生值和手动改值，再按新的工作条件全量检查；不能靠保留或改写来源标签把非法输入悄悄修成合法值。改变主输入频率时，有效的自动派生值仍会随频率更新；显式固定值若在新频率下低于下限，则直接报错。同时修改几何、频率和时序时，使用新条件判断合法性。

检查内容包括分代列命令/PRE 间隔下限、命令占用、RL/WL 编码、已知组织的 tRFC 下限、当前实现支持的 tREFI 范围，以及 tRASmax 与刷新间隔、tRASmin、读写恢复的可排程关系。较长的最小等待值可以保守合法，最大期限不能任意延长。周期性全 bank 刷新还要求 `tRFC <= tREFI`，否则该策略无法持续执行；这是本模型策略的可排程条件。标记 `timing_constraint_validation=known_jedec_constraints_checked` 只表示这些已知约束通过，仍保留 `timing_validation=not_a_vendor_speed_bin`、厂商时序未知和 PHY 换向参数未知。它不表示完整协议或真实器件已经验证。

合成时序预设的 tCCDL/S 保留 `max(当前标准最小值, analytical_floor_ns)` 依赖，改变时钟后重新求值，而不是固定保存上一次的 ns 结果。用户显式覆盖同一时序字段时清除该规则，按用户给出的固定实验值执行并标为假设。没有已知标准最小值的合成 tRRD、tFAW、tWTR 仍保留为固定分析假设，不将未知值当作标准零值。

## 命令接口与时钟

下列依据均为上表中的完整协议原文件；页码是印刷页码。控制语义、标准允许的编码范围、参考工作点和目标器件保证值是不同层次，不能因几何来自 JEDEC 就把全部模拟时序标成已验证。

| 内容 | JESD235B，HBM2 PC mode | JESD238，HBM3 | JESD270-4A，HBM4 |
|---|---|---|---|
| PC 与命令接口 | §3.2.1.2，p.4；§3.2.2，p.5 | §3.1.2，p.4–5；§3.1.3，p.5 | §3.1.2，p.5；§3.1.3，p.6 |
| CK 与数据速率 | §6.1，p.22；表 67，p.107：每 CK 2 次数据传输 | §6.1，p.26：DQS 为 CK 的 2 倍频，每 CK 4 次数据传输 | §6.1，p.33：每 CK 4 次数据传输 |
| ACT 行命令占用 | 2 CK，表 30，p.30 | 1.5 CK，表 30，p.35 | 1.5 CK，表 33，p.42 |
| PRE / 全 bank REF 行命令占用 | 1 CK | 0.5 CK | 0.5 CK |
| READ / WRITE 列命令占用 | 1 CK，表 31，p.31 | 1 CK，表 31，p.36 | 1 CK，表 34，p.43 |
| 每 PC 一个 32 B 突发的数据占用 | BL4 / 2 = 2 CK | BL8 / 4 = 2 CK | BL8 / 4 = 2 CK |

每个物理通道有分别工作的行命令口与列命令口，两者可以并行；两个 PC 共享这两个物理命令口，各自拥有数据位段与阵列状态。因此不能为每个 PC 配一个完全独立的命令口，也不能用一条串行命令队列把所有行、列操作互相阻塞。数据突发间隔为 2 CK，不是一个命令时钟周期。由带宽推导 CK 时，须注明采用理想满载数据率的假设；读写预算不同、协议开销或降频时，单凭预算不能查得真实 CK。

ACT 必须在上升沿开始，阵列时序从 ACT 的第二个上升沿起算（B §6.3.2.2，p.33；238 同节 p.38；4A 同节 p.46）。HBM3/4 的 ACT 结束后的那个下降沿不是通用发令位置：只允许 RNOP、其他 bank 的 PREpb，或另一 PC 的 PREab；不能把 1.5 CK 占用理解为下一 ACT 可立即从下降沿开始。普通列命令也须满足自己的发令边沿与时序。

## 阵列与数据时序的计量

三代的 PC 时序表均把 tRC、tRAS、tRCDRD/WR、tRRDL/S、tFAW、tRTP、tRP、tWR、tCCDL/S/R、tWTRL/S、tRTW 和刷新时序列为每个 PC 的约束（B 表 3，p.5；238 表 3，p.4；4A 表 3，p.6）。共享命令口的占用仍在物理通道层计量。SID 扩展 bank 地址，不能不加区分地把全部 PC 的 bank 时序合在一起，也不能把每个 SID 当成独立的物理通道。

| 约束 | 起点、终点与作用范围 |
|---|---|
| tRCDRD / tRCDWR | ACT 阵列起点到同 bank READ / WRITE |
| tRAS / tRP / tRC | 激活到预充电、预充电到下一激活、同 bank 两次激活；时序须同时满足 |
| tRRDS / tRRDL | 同 PC 不同 bank 的连续 ACT，分别对应不同 / 相同 bank group |
| tFAW | 同 PC 滚动窗口最多 4 次激活；支持 REFpb 时，该操作也参与相应限制。不能只限制相邻两次 ACT |
| tCCDS / tCCDL | 同 PC 连续同方向列操作，分别对应不同 / 相同 bank group |
| tCCDR | 同 PC 跨 SID 的连续 READ；跨 SID WRITE 仍使用通常的 tCCDS |
| tRTP | READ 命令到同 bank PRE；不是从读数据传输结束起算 |
| tWR | 写数据结束后的恢复时间，不能直接当成 WRITE 命令到 PRE 的全部间隔 |
| tWTRS / tWTRL | 内部写入到 READ 的间隔，分别对应不同 / 相同 bank group；须加上命令到写数据的延迟与突发长度 |

发令间隔依据 B 表 33（p.38）、238 表 33（p.44）、4A 表 36（p.52）。令无量纲 `B = 2` 为本模型各代 PC 突发占用的 CK 周期数，则不含额外 PHY 偏移的基本关系为：

```text
WRITE -> PRE（同 bank） >= WL × tCK + B × tCK + tWR
WRITE -> READ          >= WL × tCK + B × tCK + tWTR
READ  -> PRE（同 bank） >= tRTP
```

公式中的参数必须先按标准表的单位换算：例如 AC 表中的 tRTP、tWTR 是 nCK，而 tWR 是 ns。若实现以 ns 储存，须在配置导入时转换，不能把数字直接相加。READ / WRITE 与另一 bank 的 PRE 可以同时发出，完成时间应遵循资源与依赖约束，而不是把所有开销顺序相加。

HBM3/4 允许 tRAS、tRTP、tWR、tRP 按 `0.5 × ceil(2 × tXX / tCK)` 量化；这一规则只适用于列出的四个参数。PRE 可在下降沿发出，但其后的 ACT 等行访问仍须对齐上升沿（238 §6.3.2.4，p.45；4A p.53）。B 的时序不能直接套用这条半周期规则。

正常 READ -> WRITE 的 tRTW 还包含 PHY 收发偏移，是系统总线换向要求。B 表 68 注 23（p.112）、238 表 93 注 18（p.164）、4A 表 108 注 18（p.198）给出的公式不同：

```text
B:   (RL + BL/2 - WL + tDQSS(min) + 0.5) × tCK
     + tDQSCK(max) + tDQSQ(max)
238: (RL + BL/4 - WL + tDQSS(min) + 0.5) × tCK
     + tWDQS2DQ_O(max) + tDQSQtra(max)
4A:  (RL + BL/4 - WL + 0.5) × tCK
     + tWDQS2DQ_O(max) - tWDQS2DQ_I(min)
```

结果须向上取整为可发令周期。缺少这些 PHY 参数时，可使用明确标注的换向参考值，但不能称为完整执行了上式。

## 数值的来源层次

正式 AC 表是 B 表 68（p.109–112）、238 表 93（p.161–165）、4A 表 108（p.194–199）。其中 tRC、tRAS 最小值、tRCDRD/WR、tRRD、tFAW、tRP、tWR、tRTP、tWTR 等多项最小值在这些原文件中留空；已通过渲染检查表格，不是文本抽取丢失。留空不表示零，不支持声称一个任意 ns 默认值是 JEDEC 器件保证值。

有明确数值的约束包括：三代 PC 模式 tCCDS 最小 2 CK；B 的 tCCDL 最小 `max(4 CK, 2.8 ns)`，238/4A 最小 `max(4 CK, 2.5 ns)`；tRAS 最大值均为 `9 × tREFI`。B 表 33 p.38 的 PRE 命令间隔为 1 CK，B 并没有 tPPD 符号；238/4A 的 tPPD 最小为 2 CK，不应将其反推为 B 的硬下限。跨 SID READ 的 tCCDR 是厂商参数：B 给出的支持范围为 2–4 CK（p.112 注 22）；238/4A 为 `tCCDS + 1` 至 `tCCDS + 2` CK（p.164 / p.198 注 17），不能把这个范围误读为总共 1–2 CK。校验拒绝低于各代公共下界的等待值；大于 4 CK 可以是保守等待，并不证明厂商 tCCDR(min) 就是该值。

可以明确命名为 **JEDEC IDD 测量参考工作点，非器件 AC 保证** 的一组参数为 tRC = 48 ns、tRAS = 33 ns、tRP = 15 ns。来源是 B 表 58（p.102）、238 表 84（p.152）、4A 表 99（p.180）；后两代的表标题直接标为 Example。三代都允许厂家使用其他 tRAS / tRP，并在这组测量条件中要求 `nRAS + nRP = nRC`。自动生成这组参考值时按 CK 量化后保持该关系；它不是所有手动控制器等待值必须相等的一般约束。普通命令分别满足 tRC、tRAS、tRP，因此实际重复激活间隔至少为 `max(tRC, tRAS+tRP)` 并遵守命令边沿。这些表没有提供缺失的 tRCD、tRRD、tFAW、tWTR、tWR 默认值。原文也未给出通用的 `tFAW >= 4*tRRD`、`tRRDL >= tRRDS` 等数值公式，不能凭名称添加。

RL / WL 是模式寄存器中的时钟周期编码，不是任意的 ns 延迟：

| 协议与出处 | RL 可编码范围 | WL 可编码范围 |
|---|---:|---:|
| B，表 11，p.17（含扩展位） | 3–48 CK | 1–16 CK |
| 238，表 12/13，p.18–19 | 4–63 CK | 4–16 CK |
| 4A，表 11/12，p.21–22 | 17–90 CK | 4–19 CK |

器件不必支持全部编码，实际支持连续的子区间且与速度等级相关。已有 35 ns 分析延迟不能直接声称为 JEDEC RL/WL，高速下转换的周期数甚至可能超出可编码范围。未模拟奇偶校验延迟、训练与 PHY 偏移时，也须保留这些限制。

未应用工程参数包的几何参考构造器默认选择 RL = 32 CK、WL = 8 CK，作为三代编码范围内的 `selected_reference`。配置中的读写延迟据此换算为 ns，并校验为整 CK 且处于本代编码范围；这不证明目标器件支持这些编码。tRAS / tRP 使用上述 33 / 15 ns IDD 参考，tRC 保持量化后的 `nRAS + nRP`。tRCD = 14 ns、tWR = 15 ns 仍是分析假设；未取得数值的 tRRD、tFAW、tWTR 为 0 时只表示该约束未建模，不表示标准允许零间隔。tCCDR 选择范围上限 4 CK，也属于未验证厂商参数的参考选择。数据总线读写避冲突可以防止传输重叠，但没有完整 PHY 参数时仍不保证满足实际器件的 tRTW。下述工程参数包另行提供非零阵列、换向和刷新假设。

## 刷新

全 bank REF 命令选择一个 PC，不选择 SID；一次作用于该 PC 的所有 SID、所有 bank。刷新前须使相应 bank 空闲并满足 tRP，刷新后该 PC 在 tRFC 内不可进行阵列访问；另一 PC 仍有自身状态。依据 B 表 30 / §6.3.2.4（p.30 / p.39）、238 表 30 / §6.3.2.5（p.35 / p.46–47）、4A 表 33 / §6.3.2.4（p.42 / p.55–56）。HBM3/4 刷新后的半周期仍有限定命令组合，不能将其当成任意空槽。

tREFI 是平均刷新间隔。当前周期刷新实现支持 `0 < tREFI <= 3900 ns`，默认采用 3900 ns；超过上限会报 `model_limit`，规则名为 `refresh_interval_scope`。这是本模型支持的静态间隔范围，不是 JEDEC 所有刷新模式的统一上限，也不保证适用于具体器件。HBM3/4 允许受限延期与提前刷新（最多延期 8 次、绝对间隔不超过 9 × tREFI），周期性不延期策略是其一个简单子集。仅记录刷新命令次数而不计预充电及 tRFC 阻塞，会低估时间成本。

以下 tRFC 为原文件 AC 表的全 bank 刷新值，单位 ns；按 die 密度与堆叠层数查表，不能把每通道密度错当成单 die 密度：

| 组织 | 4H | 8H | 12H | 16H | 原文件位置 |
|---|---:|---:|---:|---:|---|
| B，16 Gbit/die（当前导入子集） | 350 | 450 | 未导入 | 未导入 | 表 68，p.110；对应 8 / 16 Gbit 每通道 |
| 238，8 Gbit/die | TBD | 260 | 310 | 350 | 表 93，p.162 |
| 238，16 Gbit/die | 260 | 350 | 410 | 450 | 表 93，p.162 |
| 238，24 / 32 Gbit/die | TBD | TBD | TBD | TBD | 表 93，p.162 |
| 4A，24 Gbit/die | 360 | 410 | 450 | 490 | 表 108，p.195 |
| 4A，32 Gbit/die | 400 | 450 | 490 | 530 | 表 108，p.195 |

这里的 TBD 是原刷新表实际印出的标记，与前述 AC 最小值单元格留空分别记录；两者都不等于零。缺少有效 tRFC 时，刷新模型应明确禁用并列为未模拟，或者要求显式参考值；不能把其他技术代的数字标成这一代的 JEDEC 数值。REFpb 还需要逐 bank / SID 的刷新覆盖计数与 tRREFD、tRFCpb、tFAW 等约束，不能用全 bank REF 简单拆分来声称完整支持。

未应用工程参数包的几何参考策略仅对上表有确定 tRFC 的已导入组织启用每 3900 ns 一次的周期性全 bank 刷新；HBM3 24 / 32 Gbit 及其他 TBD 组织明确禁用。HBM3E 工程时序预设则使用显式假设的 tRFC 启用刷新，原表中的 TBD 状态不因此改变。`metadata.refresh_interval_scope` 标明实现支持的间隔范围；这两种策略均不代表完整器件验证。

最大激活时间与刷新开关分别配置。默认 `maximum_active_ns = 9 × 3900 = 35100 ns`，包括因 tRFC 未知而禁用刷新的组织。该字段不能为 0，也不能超过当前 `9*tREFI`；未知刷新周期时使用 3900 ns 参考条件计算上限。导入时检查孤立读写能否在期限内恢复并预充电，发令时继续检查共享资源竞争下的关闭位置，必要时先预充电再激活；排不出合法位置时报错。已知组织的 tRFC 不能以 0 或 `assumption` 标签绕过表中下限；TBD 组织的 0 仍只表示刷新未建模，数据保持约束未覆盖。

## 配置被拒绝时的诊断

HBM 的已知时序约束报错使用 `HBMConfigError`（仍是 `ValueError`），包含中英文说明和可读取的 `details`：具体字段、输入值及单位、技术代、命令时钟、违反的规则、相关上下限、修改建议及适用的标准出处。`category` 区分 `standard_limit`（已知标准约束）、`model_limit`（当前模型限制）与 `invalid_input`（无效或互相冲突的输入）。组件加载保留组件标识；网页保留 `invalid_physical_memory_config` 错误码并显示中文诊断，编辑错误会留在诊断栏中。

例如 HBM3 在 `command_clock_ns=1` 时填写 `column_same_group_ns=3`，错误会给出当前值 3 ns、下限 `max(4×1, 2.5)=4 ns`、建议设为至少 4 ns，以及 JESD238 表 93 的出处。修改后仍须重新检查整套配置，系统不会为了通过校验而自动压低手动时序，也不会把未知的器件参数补成真实规格。一次报错指出首先遇到的问题；修正后重新校验可发现后续问题。

## 成本口径与尚未确定的内容

协议能够确定命令数、命令口占用、传输字节、阵列依赖与等待时间。32 B 是每 PC 一个突发的数据量，不能按请求字节数无条件计一个 READ / WRITE，也不能将有效页 1 KiB 计成一次数据突发。并行 bank、PC、物理通道的操作可以重叠：延迟取依赖路径与共享资源排程的完成时间；各资源 busy time 的和是另一项统计量，不应当作墙钟耗时。

汇总执行使用独立的 HBM float64 编译路径，仍执行这套命令、SID、行状态和刷新逻辑，并回写与明细路径相同的状态与资源计数。非零 tFAW 已支持数值编译执行：每个 PC 保留有效激活记录及已有未来预约，第五次 ACT 必须等待滚动窗口允许的位置，对齐命令边沿后还会重新检查；不因启用 tFAW 自动回到明细路径。激活窗口的存储不随请求突发数增长。只在 bank 全部关闭、命令口与 PC 完全空闲、且浮点运算可证明精确的时间网格及数值范围内折叠空闲刷新周期；被折叠的刷新次数和占用仍全部计入。需要明细、实际合并或重命名本 HBM 资源，以及超出加速器支持的组织或数值范围等情形，仍使用同一套 HBM 明细控制，不切换为分析式 DRAM 控制模型。内核中无关资源的别名不影响 HBM 加速。

`jedec_hbm` 只接受命令锚定时序：`activate_to_read_ns` / `activate_to_write_ns`、`precharge_to_activate_ns`、`write_to_precharge_ns`。修改配置时必须使用这些字段；不再把 `open_ns` / `close_ns` / `write_recovery_ns` 自动转换为新时序。`open_ns` / `close_ns` / `read_recovery_ns` / `write_recovery_ns` 属于 `analytical` 分析式模型，不能用来调整 JEDEC HBM。`burst_interval_ns` 在 JEDEC HBM 中由接口速率推导。RL/WL 的 ns 配置必须对应整 CK，校验不会自动取整改变输入。

厂商料号、厂商地址映射、实际速度等级、支持的 RL/WL 子区间、未填阵列时序及 PHY 换向极值仍未由这些寻址表和协议参考工作点确定。初始化、训练、奇偶校验、故障恢复及可选刷新机制也不因支持普通读写而自动得到实现。配置的逐字段来源和未知项保存在 `metadata.parameter_provenance` 与 `metadata.unknown_parameters` 中；协议结构正确与某个真实器件的时序已验证必须分别表达。

## 当前物理配置接口

序列化的 DRAM 配置必须显式声明 bank-group 时序、控制模式与几何作用域；不再为旧配置补零。NAND 配置必须显式声明共享、读取、写入三项主机带宽，以及 `protocol_mode`、`host_channels`、`host_full_duplex`、`read_cache_pages_per_bank`、`queue_scope` 和 `max_outstanding_write_requests`；最后一项可以显式为 `null`，表示写请求共用总在途额度，不为旧文件补语义。`generic_nand` 保留 SSD 等通用 NAND 的层级和全设备队列；`write_buffer_bytes > 0` 启用页合并写缓冲，返回主机确认后通过刷新完成介质写入，设为 `0` 使用写穿模式。`ocp_hbf_0_7` 要求零写缓冲、4 KiB完整页写和64 B读粒度，并在介质完成后应答；队列范围必须为 `host_channel`，bank 由独立 plane 几何派生，不接受 `parallel_units` 数值覆盖。原 `metadata.buffered_write_mode` 开关已删除。HBM预设使用 `hbm3e_reference`、`hbm3e_reference_flat`、`hbm3e_reference_synthetic_timing` 及对应 HBM4 名称；分析式零 bank-group 控制使用 `hbm_analytical_zero`。

## HBM3E 工程时序预设

统一物理预设目录新增 `hbm3e_timing_fast`、`hbm3e_timing_reference`、`hbm3e_timing_slow`，可由 `physical_config_profile_ids()` 查询，脚本与配置 API 均通过 `resolve_physical_config` 使用。三者默认采用 HBM3E、32 Gbit/die、16H、512 GiB 可见容量、8 stacks、tCK=0.625 ns；物理接口为每 stack 819.2 GB/s，默认总计 6553.6 GB/s。默认方向上限等于该物理接口；1600 GB/s 是主实验另行给出的方向预算。

```python
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.physical_contract import reconfigure_physical_config

config = resolve_physical_config("hbm3e_timing_reference", overrides={
    "capacity_bytes": 512 * 2**30,
    "read_bandwidth_gb_s": 1600.0,
    "write_bandwidth_gb_s": 1600.0,
    "activate_to_read_ns": 20.0,  # 单项显式实验输入，覆盖本档的 14 ns
    "scheduler": "row_hit_first",
    "queue_scope": "channel",
})
changed = reconfigure_physical_config(config, overrides={"command_clock_ns": 1.25})
# RL/WL 等周期输入随 CK 更新；显式 activate_to_read_ns 仍保持 20 ns。
```

未给定 stacks 时沿用普通预设的 `max(8, ceil(capacity_bytes / 64 GiB))` 覆盖规则，例如 1 TiB 使用 16 stacks；仅改变容量不会改变默认 CK。显式 stacks 必须覆盖可见容量。参数包先应用，显式时钟、接口、单项时序、调度器和队列覆盖随后通过公共重配入口执行；合法的零值消融也会保留。完整配置导出后再导入，仍保存这些来源和依赖。三档只适用于这里明确的 HBM3/E 32 Gbit、16H 组织，不套用到 HBM4 或其他密度。已有 `hbm3e_reference`、`flat` 和 `synthetic_timing` 等几何控制预设保持原有含义。

以下是参数包的输入，周期项在执行前转换为 ns；tRAS/tRP 另按命令半周期向上量化，tRC 随执行的 tRAS+tRP 推导：

| 参数 | fast | reference | slow | 输入单位 |
|---|---:|---:|---:|---|
| tRCDRD / tRCDWR | 12 / 12 | 14 / 14 | 18 / 18 | ns |
| tRAS / tRP / tWR | 28 / 12 / 12 | 33 / 15 / 15 | 40 / 18 / 20 | ns |
| tRRD_S / tRRD_L / tFAW | 2.5 / 4 / 12 | 4 / 6 / 20 | 6 / 8 / 32 | ns |
| tRTP / tWTR_S / tWTR_L | 4 / 4 / 6 | 6 / 6 / 8 | 8 / 8 / 12 | CK |
| RL / WL / tCCDR | 28 / 6 / 3 | 32 / 8 / 4 | 40 / 10 / 4 | CK |
| tRFCab / tREFI | 350 / 3900 | 500 / 3900 | 650 / 3900 | ns |
| tRTW，tCK=0.625 ns 时的推导结果 | 16.25 | 18.125 | 23.125 | ns |

截至 2026-10-15，已查阅 [Micron HBM3E 目录](https://www.micron.com/products/memory/hbm/hbm3e/part-catalog)、[Samsung HBM3E 产品页](https://semiconductor.samsung.com/dram/hbm/hbm3e/) 和 [SK hynix 16 层公告](https://news.skhynix.com/en/sk-hynix-announces-16-layer-hbm3e-at-sk-ai-summit-2024/)，未取得适用于该 32 Gbit/16H 组织的完整 AC 时序表。因此上述档位是工程敏感性场景，既不是厂商速度档，也不是真实器件上下界或统计置信区间。reference 的 33/15 ns 沿用 IDD 示例作选择依据，不能升级成 AC 保证；32 Gbit/16H 的 tRFCab 在 JESD238 表 93 p.162 仍为 TBD。三档统一 tREFI=3900 ns，未借 slow 档暗中加入高温或加倍刷新假设。

tRTW 采用 JESD238 表 93 注 18（p.164）的公式，并加入单独声明的系统裕量。tDQSS(min) 使用 p.163 的 `max(-0.2 ns, -0.2*tCK)` 包络；tWDQS2DQ_O(max) 分别假设为 1/1.5/2.5 ns；tDQSQtra(max) 选用 p.160 示例最大 20 ps（0.020 ns），另加 30/80/180 ps 系统裕量，最后对齐到上升命令边沿。额外裕量不是 JEDEC 器件参数，所选 PHY 极值也没有目标封装校准。关闭刷新且将 tREFI/tRFC 清零时，独立最大开行时间仍按参考间隔保留 35100 ns。

`metadata.hbm_parameter_profile` 记录档名、各字段输入单位与范围、公式、PHY 假设和资料检索结果；`metadata.parameter_provenance` 记录各字段实际输出值及手工覆盖，`physical_config_profile_id` 对应实际选择的公开预设。`vendor_calibrated`、`vendor_verified` 与 `target_hardware_calibrated` 保持为 `False`。也可直接调用 `apply_hbm_timing_profile(config, "reference")`；精确支持 `fast/reference/slow/ramulator2/none`，其中 `none` 保留传入配置，不会撤销已经套用的参数包。它是参数包 API 的选择，不是另一个名为 `hbm3e_timing_none` 的预设。

另提供 `hbm3e_timing_ramulator2` 公开预设，移植固定版本 Ramulator2 的 `HBM3_6400Mbps` / `HBM3_32Gb_16hi` 标量时序和相关公式。仅支持 tCK=0.625 ns、32 Gbit/die、16H，不将该速度档外推至其他时钟。执行继续使用本项目的命令锚点、控制器和刷新排程，不等同于运行 Ramulator2 本体；默认 `reference` 不变。来源、逐项映射、调用方法及对照实验见 [Ramulator2 参数对照说明](hbm_ramulator2_comparison.md)。

## HBF 参数研究与配置修正（2026-10-17）

`hbf_ocp_0_7` 已改用下述 FlashLabAE 256-bank 研究参考几何，并声明每主机通道的在途限制；**这不是厂商校准，实验扫描仍暂停**。旧预设使用 `channels=64`、`parallel_units=64`，512 GiB 来自 `64 × 8192 blocks × 256 pages × 4096 B`；这个未校准的等效阵列选择已从严格 HBF 预设废弃，不能用于真实器件性能结论。旧值可追到 `d4a9fee` 的实验脚本，`b5f76bbe1a00c4c7a571d7dbb4635c96d5462dda` 添加严格协议子集预设时仍保留该假设。支持部分协议事务不等于几何和时序已获验证。

下表按来源分别记录，不把几套组织拼成同一个器件。OCP 页码均为印刷页码；开源配置使用固定提交，论文使用固定版本。

| 来源与证据类型 | 可采用或核对的参数 | 不能由此确定的内容 |
|---|---|---|
| [OCP HBF v0.7.0，2026-08-03][hbf-ocp-v070]：协议要求，p.15、p.69–70 | 最多 16 个独立全双工主机通道；4 KiB NAND 页；写入为 4 KiB 对齐完整页；读取以 64 B 为粒度且不跨页；每 bank 两个 4 KiB cache buffers | 完整厂商物理组织及实测介质时序；两个 buffer 也不等于两个独立感测阵列 |
| 同一 OCP 规范：实现可配，BUCCAP，§5.9.2.1，p.69–70 | 每 UCIe 通道 core die 数 NCDU 可为 1/2/4/8/16；MOCS 是该通道的命令队列能力，列出 1 以及 256–16384 的 2 的幂档位 | NCDU、实际队列档位须由具体实现确定；不能用全设备队列深度替代每通道 MOCS |
| 同一 OCP 规范：容量计算与架构示例，Table 3 p.16、Figure 4 p.23 | Table 3 列 16 dies/cube、16 banks/channel、4 KiB 页和 512 GiB；Figure 4 示例为每 host channel 下 2 core channels × 2 dies × 16 banks，即 64 bank 实例/host channel | 两处不是一套已披露的产品几何；不能强行把 Figure 4 外推到 16 个 host channels 后的 1024 个通道局部 bank 实例解释为已确认的 16-die 产品 |
| [FlashLabAE/HBFSim，提交 `7a6cb439` 的默认配置][hbf-flashlab-config]：OCP 0.7 研究参考 | 每栈 16 channels × 1 die/channel × 16 planes/die，模型按 256 banks 调度；每 bank 2048 blocks × 256 pages × 4 KiB，合计 512 GiB；读/编程/擦除为 4 μs / 75 μs / 2 ms；每 bank 两页缓存 | [来源表][hbf-flashlab-provenance]明确标为 `exploratory_assumption`；一 channel 一 die、block 几何和时序不是厂商披露或 OCP 强制值 |
| [FLINT，arXiv:2608.25062v1，§2、§5 Table 2][hbf-flint]：论文设计点 | 每栈 16 dies × 32 planes/die = 512 planes；每 plane 256 blocks × 1024 pages × 4 KiB，按这些整数得 512 GiB；读/编程/擦除为 2 μs / 50 μs / 3 ms；接口 1 TB/s/stack | 论文表中容量写作 GB，本表仅把几何乘积换算成 GiB；未确认公开的 FLINT 仿真器及对应配置。论文使用 HBF 权重层，KV 和激活在 HBM，不能直接套用其性能结论到全 HBF |
| [Sandisk HBF Fact Sheet，2025-07，p.2、p.3 脚注 2][hbf-sandisk-factsheet]：厂商公开参数 | 第一代列 16 dies/stack、256 Gb/die、512 GB/stack、读带宽 1.6 TB/s；脚注明确 GB/TB 为十进制，实际用户容量更小 | 512 GB = 512,000,000,000 B，不能改写为 512 GiB；该资料没有给出完整 bank/plane/block/page 几何和读写时序表 |

`bank` 与 `plane` 不应默认一一对应。OCP 使用 host channel、core channel/die、NAND bank 等概念；论文和模拟器中的 plane 可能是不同层次的独立操作资源。FlashLabAE 将其 16 planes/die 用作 bank 的调度实例，是该模型的明确映射。其 `16 × 1 × 16` 能与 Table 3 的数量在数值上自洽，但不能据此断言标准要求每个物理 die 独占一个 host channel。

当前主参考采用 FlashLabAE 的 **256 banks 可复现研究点**；FLINT 的 512 planes 仍是独立文献参考，未混入该预设。字段映射为 `host_channels=16`、`channels=16`、`targets_per_channel=dies_per_target=luns_per_die=1`、`planes_per_lun=16`、`planes_independent=True`、`parallel_units=None`；即每主机通道一个内部通道和一个 core die，target/LUN 为单例映射层，每 plane 在本模型中代表一个独立 bank。每 bank 的 2048 blocks × 256 pages × 4 KiB 导出总容量 512 GiB，`array_units` 从几何得到 256。严格模式拒绝独立并行数覆盖和共享 plane；通用 NAND/SSD 仍保留自己的资源组织。

严格预设不再继承通用 NAND 的全设备 32 请求限制，使用 `queue_scope="host_channel"`、`max_outstanding_requests=256` 和 `max_outstanding_write_requests=64`：**每主机通道读写合计最多 256 个在途请求，其中写请求最多 64 个**，写额度包含在总额内。256 来自将上游 4096 个每栈读页 credits 均分给 16 个通道；把它解释为读写合计额度是本项目的研究选择，不是上游队列实现的逐字复制，也不是 OCP 规定的器件 MOCS。字段作用域、通道归属与独立 bank 排程共同限制可用并行度。`generic_nand` 仅允许 `queue_scope="device"`，按完整请求占用全设备额度；严格 OCP 模式才把跨页请求拆为各主机通道的页命令。不能在未拆页的通用 NAND 路径启用每通道额度。

介质读/编程/擦除采用 4 μs / 75 μs / 2 ms。主机接口按 OCP p.16 Table 2/4 的 grade 2：16 GT/s、16 channels、75% 有效载荷效率，得到双向各 1536 GB/s；不再使用旧的 16 GB/s 写接口上限重复限制 NAND 编程吞吐。上游[速度等级实现][hbf-flashlab-grade]保留标准 p.32/35 的 94/188 与 p.16 的 96/192 GB/s 差异，本项目采用后者作为研究参考。`internal_bandwidth_gb_s=96` 按每内部通道的 payload 计，由上游 raw-channel 参考值按 `101.25 GB/s × 4096/(4096+224 OOB) = 96 GB/s` 转换；本项目没有复刻上游的 OOB、ECC、TSV 等独立阶段，`front_ns=0` 也是本项目简化，因此不能称为 FlashLabAE 完整复现。几何、时序、队列与传输假设及固定提交链接分别记录在预设 `metadata.parameter_provenance` 中；显式修改字段后记录为新的研究假设。

仍缺少目标产品的真实层级映射、可见容量与保留空间、读/编程/擦除时序、实际 MOCS、共享内部链路及 ECC 吞吐校准。公开研究配置能替换无来源的参数选择，不能使这些未知项自动变成真实器件规格。

[hbf-ocp-v070]: https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf
[hbf-flashlab-config]: https://github.com/FlashLabAE/HBFSim/blob/7a6cb439235d721137f8c5c3a952e2ecc520a47b/src/physical/hbf/hbf_config.hpp#L12
[hbf-flashlab-provenance]: https://github.com/FlashLabAE/HBFSim/blob/7a6cb439235d721137f8c5c3a952e2ecc520a47b/configs/parameter-provenance.json#L509
[hbf-flashlab-grade]: https://github.com/FlashLabAE/HBFSim/blob/7a6cb439235d721137f8c5c3a952e2ecc520a47b/src/physical/hbf/hbf_standard.hpp#L11
[hbf-flint]: https://arxiv.org/html/2608.25062v1#S5
[hbf-sandisk-factsheet]: https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf

## 版本 10 主实验工作点（历史控制）

`scripts/hbf_hbm_experiments.py` 与 `hbf_hbm_serving_study.py` 的主服务比较固定 HBM3E 16H × 32 Gbit，即每 stack 原始64 GiB，tCK=0.625 ns（6.4 Gb/s/pin分析工作点）。512 GiB / 1 TiB 分别用8 / 16 stacks，物理接口总速率为6553.6 / 13107.2 GB/s，读写方向预算均为1600 GB/s。容量不再改变频率；更多stack会改变并行度，不能把这个对照称为仅改变容量且所有物理资源不变。版本10的普通主实验套用 `hbm_timing_profiles` 的 `reference` 工程参数包：tRRD_S/L=4/6 ns、tFAW=20 ns、tRTP=6 CK、tWTR_S/L=6/8 CK、tRTW=18.125 ns，周期REFab使用tREFI=3900 ns、tRFC=500 ns。对应厂商tRFC仍未知，500 ns是明确的敏感性参考假设，不是原标准TBD已经得到解决，也没有借用HBM4数值。

参数包支持 `fast/reference/slow` 三档、来源于开源模型的 `ramulator2` 对照组和明确不套包的 `none`；RL/WL及周期单位时序随CK换算，RTW按对应参数组的公式向上取整，逐字段保存输入单位、来源、范围和手工覆盖。`ramulator2` 固定在0.625 ns，其他三档可随CK换算。三档不构成真实器件上下界或统计置信区间；所有参数组均为 `vendor_calibrated=False`。已有纯几何/分组消融预设维持自身研究语义，第20组分组对照显式使用 `hbm_parameter_profile="none"`，不混入普通主实验的新时序包。

控制器可选择 `address_mapping=bank_interleaved/row_local`、`scheduler=fcfs/row_hit_first/read_write_batch`、`scheduler_window` 和 `queue_scope=device/channel`。窗口只选择已到达且没有更早重叠读写依赖的请求，加入老化防饥饿；结果保持输入顺序。它重排聚合请求，行命中评分看首个突发，不声称逐命令FR-FCFS。单独 `submit` 无可供重排的请求窗口；服务内同一物理任务的连续同owner访问可通过 `submit_many` 实际使用策略。每通道队列额度按整个请求涉及的全部物理通道占用，直到该请求整体完成；这个保守聚合定义不等于每通道突发队列。

版本10历史控制中的HBF使用逻辑页服务分析：4 KiB介质页、64 B主机传输计费、64个有效阵列单元、16个全双工主机通道和每bank两页读缓存。读/编程延迟4 / 75 µs是分析假设；接口1600 / 16 GB/s与理想冷阵列65.536 / 3.495253 GB/s上限分别输出。`published_hbf` 的未给定并行度采用固定128单元假设，主实验显式覆盖为64，均不按接口吞吐反推单元数。逻辑模型可执行RMW，但未包含主机重映射、物理页序和擦除成本；严格完整页事务另用 `hbf_ocp_0_7` 预设。这些记录仅保留为历史控制；64单元及其推导吞吐均未获得器件校准，不能据此判断真实HBF相对HBM的性能，`hbf_ocp_0_7` 的名称也不解除上述限制。

上述默认值只适用于主实验。文献参数化探针保留显式输入的文献工作点和未知项，微基准保留其独立敏感性参数；第11组封装候选明确使用0.32 ns分析时钟和每栈1600 GB/s方向预算，不混入主容量比较。其36十进制GB可见容量预算小于64 GiB原始参考stack容量，不代表已识别器件。

执行版本为10，Decode契约为 `physical_core_unified_hardware_catalog_v4/serving-points-v10`。旧版本缓存不可复用；报告仅给旧数据历史入口，绘图入口拒绝旧或混合版本，图表清单也记录执行版本。版本9结果保留在 `artifacts/hbm_hbf_remediation_20261015/`，不可作为当前默认时序的结果。版本10参数/刷新消融、控制器比较和完整服务的入口为 `scripts/hbm_sensitivity_study.py`，结果保存在 `artifacts/hbm_sensitivity_20261015/`；它不代表全部20组或8B/70B长上下文矩阵已重跑。
