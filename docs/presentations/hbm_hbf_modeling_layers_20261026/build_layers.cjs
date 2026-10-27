const pptxgen = require('pptxgenjs');
const path = require('path');

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'HeteroLLM-Sim';
pres.subject = '当前代码中 HBM/HBF 已实现的组织、状态、资源及管理层级';
pres.title = 'HBM / HBF 已建模层级';
pres.company = 'HeteroLLM-Sim';
pres.lang = 'zh-CN';
pres.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };
const W = 13.333333, H = 7.5;
const C = { navy:'0D1C2F', ink:'1B2E46', muted:'627287', line:'C8D3DF', white:'FFFFFF', grey:'F2F5F8', teal:'0B7A9D', tealLight:'E8F4F8', amber:'AC6225', amberLight:'FCEDDF', darkPanel:'1B3049', paleText:'B9C9DA', brightTeal:'66D0E5', brightAmber:'F2B46E' };
let slideIndex = 0;

function rect(s,x,y,w,h,fill,line=fill,r=0){
  s.addShape(r ? pres.ShapeType.roundRect : pres.ShapeType.rect, {x,y,w,h,rectRadius:r,fill:{color:fill},line:{color:line,width:0.8},radius:r});
}
function text(s,txt,x,y,w,h,size=18,color=C.ink,bold=false,align='left',extra={}){
  s.addText(txt,{x,y,w,h,fontFace:'Microsoft YaHei',fontSize:size,color,bold,margin:0,breakLine:false,paraSpaceAfterPt:0,valign:'middle',align,...extra});
}
function line(s,x1,y1,x2,y2,color=C.line,width=1.2,arrow=false,dash=false){
  const opt={x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1),line:{color,width,...(dash?{dashType:'dash'}:{}),...(arrow?{endArrowType:'triangle'}:{})}};
  if(x2<x1) opt.flipH=true;
  if(y2<y1) opt.flipV=true;
  s.addShape(pres.ShapeType.line,opt);
}
function circle(s,txt,x,y,fill,color=C.white){
  s.addShape(pres.ShapeType.ellipse,{x,y,w:.4,h:.4,fill:{color:fill},line:{color:fill,width:0}});
  text(s,txt,x,y,.4,.4,16,color,true,'center');
}
function base(title,sub,{dark=false,eyebrow='已实现建模层级'}={}){
  const s=pres.addSlide(); slideIndex++;
  s.background={color:dark?C.navy:C.white};
  text(s,`HeteroLLM-Sim  /  ${eyebrow}`,.65,.29,11,.28,10.5,dark?C.paleText:C.muted,true);
  text(s,title,.65,.77,12.05,.67,36,dark?C.white:C.ink,true);
  if(sub) text(s,sub,.65,1.48,12.05,.43,16,dark?C.paleText:C.muted);
  text(s,'当前工作区实现  ·  几何示例与模式范围均明确标注',.65,7.15,11,.18,9.5,dark?C.paleText:C.muted);
  text(s,`${String(slideIndex).padStart(2,'0')} / 05`,12.02,7.10,.68,.3,10.5,dark?C.paleText:C.muted,false,'right');
  return s;
}

// 01: The five abstraction layers, not the order of one read transaction.
{
  const s=base('我们已完成哪些建模层级？','HBM / HBF 已连接硬件组织、数据地址、介质时序、资源竞争与推理指标',{dark:true,eyebrow:'覆盖总览'});
  const levels=[
    ['硬件组织与参数','几何、容量、方向带宽、时序、模式'],
    ['数据与物理地址','张量放置、地址与缓冲生命周期'],
    ['介质状态与时序','HBM 行与命令；HBF 页、块、读缓存'],
    ['共享资源与事件','有状态 Core、准入、排队、任务依赖'],
    ['观测与性能输出','流量、busy、延迟与 Token 吞吐'],
  ];
  levels.forEach(([a,b],i)=>{
    const y=2.15+i*.82;
    rect(s,.65,y,7.45,.68,C.darkPanel);
    circle(s,String(i+1),.83,y+.14,C.brightTeal,C.navy);
    text(s,a,1.4,y+.12,2.75,.43,20,C.white,true);
    text(s,b,4.24,y+.12,3.6,.43,14.4,C.paleText);
  });
  text(s,'物理组织也已进入模型',8.62,2.14,4.05,.43,22,C.white,true);
  rect(s,8.62,2.87,4.08,1.38,C.darkPanel);
  text(s,'HBM',8.87,3.04,3.5,.36,24,C.brightTeal,true);
  text(s,'Stack → Channel → PC\n→ SID / BG → Bank / Row',8.87,3.52,3.55,.51,16.2,C.white);
  rect(s,8.62,4.61,4.08,1.38,C.darkPanel);
  text(s,'HBF',8.87,4.78,3.5,.36,24,C.brightAmber,true);
  text(s,'Stack / 实际 Die → Plane\n→ Block → Page → 页内字节',8.87,5.25,3.55,.51,16.2,C.white);
  text(s,'“有层级”与“有独立执行资源”需要分开',8.62,6.36,4.08,.43,13.5,C.brightAmber,true);
  s.addNotes('本页描述五个已实现的抽象层级，不是一次访存的五个时间步骤。硬件组织和参数层提供显式几何与物理预算；数据地址层管理张量端点、owner、分配代次和物理范围；介质状态层维护HBM命令与HBF媒体/缓存；共享时间线与事件内核表达资源竞争和任务依赖；性能层再结合计算与链路模型输出系统指标。层级可被寻址，不意味着每一级都有独立总线或队列。已有机制模型不等于已校准目标芯片。来源：src/heterollm_sim/physical_contract.py:154–200、data_motion.py:314,1151,1250、memory_transfer.py:263–335、event_kernel.py:685,1036、memory_service.py:26–84、reporting.py:632,780。');
}

// 02: Containment represents the HBM address organization. Side notes identify resources.
{
  const s=base('HBM：已覆盖到 Bank / Row / Column','16H HBM4 参考组织；以下状态与时序以 jedec_hbm 模式说明',{eyebrow:'物理组织 ①'});
  rect(s,.65,2.12,7.55,4.47,C.grey,C.line);
  text(s,'Device / 内存端点（一个 DramCore）',.87,2.25,6.95,.34,19,C.ink,true);
  rect(s,.98,2.84,6.89,3.43,C.tealLight,C.line);
  text(s,'Stack',1.19,2.99,2.6,.34,20,C.teal,true);
  text(s,'32 Channel / Stack',4.72,3.01,2.91,.28,15,C.teal,false,'right');
  rect(s,1.31,3.48,6.23,2.48,C.white,C.line);
  text(s,'Channel',1.52,3.60,1.72,.31,18,C.teal,true);
  text(s,'共享 Row CA + Column CA',3.16,3.61,4.11,.3,14.5,C.muted,false,'right');
  rect(s,1.63,4.05,3.27,1.65,C.tealLight,C.line);
  text(s,'PC0',1.82,4.13,.8,.27,18,C.teal,true);
  text(s,'独立 DQ',3.44,4.15,1.25,.24,13.5,C.teal,false,'right');
  const x0=1.85;
  ['SID × 4','Bank Group × 2 / SID','Bank × 8 / Group','Row → Column'].forEach((t,i)=>{
    text(s,t,x0+i*.12,4.47+i*.265,2.7-i*.12,.23,i===0?15:13.5,C.ink,i===0);
    if(i<3){line(s,x0-.06+i*.12,4.64+i*.265,x0-.06+i*.12,4.75+i*.265,C.teal,1);line(s,x0-.06+i*.12,4.75+i*.265,x0+.08+i*.12,4.75+i*.265,C.teal,1);}
  });
  rect(s,5.16,4.05,2.06,1.65,C.tealLight,C.line);
  text(s,'PC1',5.36,4.16,1.63,.28,18,C.teal,true);
  text(s,'同样的\nSID / BG / Bank 组织',5.36,4.68,1.62,.55,13.5,C.ink);
  text(s,'另有独立 DQ',5.36,5.33,1.63,.23,13.5,C.teal,true);
  text(s,'组织层级 ≠ 总线数量',8.66,2.16,4.02,.41,22,C.ink,true);
  const notes=[
    ['Channel','两个 PC 共享行命令与列命令总线'],
    ['PC','每个 PC 独立数据通道 DQ 及其时序历史'],
    ['SID / Bank Group','地址与时序区分维度；没有额外 DQ'],
    ['Bank / Row','独立开行状态；行命中 / 冲突\n以及 PRE / ACT 就绪时刻'],
  ];
  notes.forEach(([a,b],i)=>{const y=2.96+i*.84;text(s,a,8.66,y,4,.29,18,C.teal,true);text(s,b,8.66,y+.34,4,i===3?.56:.44,15.3,C.ink);});
  text(s,'32 B burst 是传输粒度；堆叠 die 已通过 SID 展开，容量不再重复乘一次 die 数',.65,6.76,12.05,.26,12.8,C.muted);
  s.addNotes('HBM正确组织是Device/内存端点→Stack→Physical Channel→PC→SID→Bank Group→Bank→Row→Column。图中只展开PC0，PC1拥有同样的组织。以16H HBM4参考为例：每Stack 32个物理Channel，每Channel两个PC；每PC四个SID，每SID两组，每组八个Bank，即每PC 64个Bank。bank_group在PC中连续编号，SID从bank/group地址推导，AddressMapping没有单独SID字段。SID/BG不提供额外DQ；Channel共享Row CA与Column CA，各PC有独立DQ，Bank有独立打开行与就绪状态。32B burst是服务粒度，不是新增硬件容器。当前per_stack几何中die映射为0，不能在PC与SID间再画一层独立Die执行资源或重复乘容量。HBM3/4的两个有效1KiB寻址页可共享2KiB激活页；Column是burst位置。analytical为另一简化模式，不能套用本图全部正常工作命令时序。来源：src/heterollm_sim/hbm_geometry.py:132,225,250,267、memory_types.py:577,589,593,629、memory_mapping.py:47,78,90,173、_hbm_control.py:70,79,95,206,309,318,364,486。');
}

// 03: Physical HBF organization and its explicit mapping into NandCore coordinates.
{
  const s=base('HBF：已覆盖到 Plane / Block / Page','论文原生配置：实际 Die 映射为内部 Channel；Stack 通过通道分组表达',{eyebrow:'物理组织 ②'});
  rect(s,.65,2.11,12.05,.65,C.amberLight,C.line);
  text(s,'HBF 设备 / 一个 NandCore',.9,2.26,7,.33,21,C.amber,true);
  text(s,'host：1 个聚合接口',8.8,2.28,3.65,.28,17,C.ink,false,'right');
  text(s,'下面是组成与地址映射展开，不是读写步骤',.65,2.92,12,.26,12.5,C.muted);
  const cols=[{x:.65,w:2.02},{x:3.14,w:2.43},{x:6.07,w:2.66},{x:9.22,w:3.48}];
  cols.forEach(p=>rect(s,p.x,3.4,p.w,2.65,C.white,C.line));
  [2.72,5.64,8.80].forEach(x=>line(s,x,4.63,x+.32,4.63,C.amber,1.7,true));
  text(s,'Stack 分组',.84,3.58,1.66,.36,20,C.amber,true);
  for(let row=0;row<4;row++)for(let col=0;col<4;col++)rect(s,.94+col*.35,4.21+row*.24,.25,.15,C.amberLight,C.amber);
  text(s,'16 个实际 Die / 堆',.84,5.36,1.67,.31,13.3,C.ink);
  text(s,'实际 Die',3.36,3.57,2,.32,20,C.amber,true);
  text(s,'⇔ Core Channel',3.36,4.00,2,.31,17,C.ink,true);
  rect(s,3.36,4.55,1.99,.71,C.amberLight,C.amberLight);
  text(s,'内部搬运总线',3.52,4.69,1.69,.34,16,C.amber,true);
  text(s,'同 Die 的 Plane 共享',3.36,5.37,1.99,.3,12.3,C.muted);
  text(s,'Plane × 25',6.28,3.57,2.24,.33,20,C.amber,true);
  text(s,'独立阵列执行资源\n介质暂存就绪状态',6.28,4.10,2.24,.65,16,C.ink);
  text(s,'每 Plane 两个读槽',6.28,4.96,2.24,.28,14.5,C.amber,true);
  rect(s,6.28,5.37,.98,.26,C.amberLight,C.amber);rect(s,7.53,5.37,.98,.26,C.amberLight,C.amber);
  text(s,'Block：擦除单位',9.45,3.62,2.99,.34,20,C.amber,true);
  rect(s,9.45,4.16,2.99,1.39,C.amberLight,C.line);
  text(s,'Page：Read / Program',9.66,4.35,2.56,.29,15.8,C.ink,true);
  text(s,'4 KiB 页 → 页内字节',9.66,4.86,2.56,.28,15.5,C.ink);
  text(s,'Block / Page 可寻址与计费，\n不各自新增一条阵列总线',9.45,5.60,3.01,.37,11.5,C.muted);
  text(s,'通用 NAND 坐标：Channel → Target → Core die → LUN → Plane → Block → Page → offset',.65,6.35,12.05,.3,14,C.ink);
  text(s,'本配置 Target / Core die / LUN 每级数量均为 1；7 堆 = 112 个内部通道、2,800 个 Plane 阵列',.65,6.79,12.05,.25,12.4,C.muted);
  s.addNotes('本页区分物理HBF层级与通用NAND核心坐标。HBF设备由一个NandCore绑定owner并拥有聚合host接口。NandConfig没有独立stack几何字段；论文原生builder用连续16个Core channel表示一堆，每个实际Die映为一个Core channel并拥有内部搬运总线。Target、Core die、LUN作为通用几何坐标已经实现，但该配置都取1，不能再乘实际die数。每实际Die有25个Plane，各Plane独立阵列日历、媒体暂存就绪状态与两个读槽。同Die的Plane共享内部搬运总线，流量最后汇入设备host。Block是擦除、顺序写前沿和可选管理单位，Page是读/program、缓存标签和字节计费单位；Block/Page不各自拥有独立媒体执行日历。本例Page为4KiB，7堆形成112内部通道和2800个Plane。source：scripts/exploring_hbf_paper_config.py:272–342、src/heterollm_sim/memory_types.py:659,677,684,813、memory_mapping.py:124、nand_core.py:380–409,537,549,553,641,647,680。');
}

// 04: A native table makes the cross-layer resource ownership directly comparable.
{
  const s=base('每一层，都维护了哪些状态与资源？','有地址层级，不代表每一层都有独立执行器；共享关系决定并行与等待',{eyebrow:'建模深度'});
  const header=['作用层级','HBM：jedec_hbm','HBF：论文原生页服务'];
  const rows=[
    ['数据 / 地址','张量 → owner 与缓冲地址\n32 B burst 拆分与物理映射','相同访问契约，按 Page 拆分\n页内偏移、对齐、部分写 RMW'],
    ['介质状态','Bank 开行；ACT / PRE / RD / WR / REF\n命令间隔、方向切换、刷新','Plane 读 / Program / Erase、介质暂存\n双槽标签 / ready / reader_end / LRU'],
    ['共享资源','Channel 共享 Row CA / Column CA\nPC 各有 DQ；Bank 各有行状态','每实际 Die 的内部通道由 Plane 共享\n每 Plane 独立阵列，设备 host 汇聚'],
    ['事件 / 观测','有状态 Core → 资源日历 → 任务依赖\n完成时间、流量、命令、等待、busy','再结合计算与链路模型 → 系统指标\nmakespan、TTFT、TPOT、Token 吞吐'],
  ];
  const tableRows=[header.map(t=>({text:t,options:{bold:true,color:C.white,fill:{color:C.navy},fontSize:17}})),...rows.map((r,i)=>r.map((t,j)=>({text:t,options:{fontSize:15.6,bold:j===0,color:C.ink,fill:{color:j===0?C.grey:(j===1?C.tealLight:C.amberLight)}}})))];
  s.addTable(tableRows,{x:.65,y:2.13,w:12.05,colW:[1.65,5.12,5.28],rowH:[.58,1.0,1.0,1.0,1.0],fontFace:'Microsoft YaHei',fontSize:15.6,color:C.ink,margin:[.12,.17,.12,.17],border:{type:'solid',pt:1,color:C.white},valign:'middle',autoPage:false,verbose:false});
  text(s,'统计口径分开：逻辑 / 物理字节；服务跨度 service_ns / 资源 busy / 总耗时 makespan',.65,6.91,12.05,.22,12.0,C.muted);
  s.addNotes('这是一张已实现范围的映射表，不是新增功能计划。数据地址层对权重、KV、激活和缓冲代次、owner、范围进行绑定和解析；两种介质按burst或page拆分。HBM的jedec_hbm维护Bank开行和正常工作命令、时序间隔、方向切换与刷新。HBF维护Plane媒体日历、媒体暂存和两个读缓存槽，槽标签、ready、最晚reader_end和LRU均有状态。资源层体现共享：Channel的两PC共用Row/Column CA但各有DQ；同实际Die的Plane共用内部通道，而Plane阵列资源独立，最后汇聚host。有状态Core结果在实际任务dispatch时回写事件内核，结合计算与链路模型产出系统指标。任务依赖来自任务图和访问描述，不应宣传为自动发现所有跨任务依赖。service_ns是可能重叠的完成跨度，不能当资源busy或整体makespan。来源：src/heterollm_sim/data_motion.py:314,1151,1250,1330–1496、event_kernel.py:685、memory_transfer.py:263–335、memory_service.py:26–84、_hbm_control.py:70–206、nand_core.py:380–409,647,1901–1956、reporting.py:632,780。');
}

// 05: Separate core capability, optional management, and unmodeled detail.
{
  const s=base('已实现范围与模型边界','完成的是可配置机制模型；不同控制与协议模式启用不同能力',{dark:true,eyebrow:'范围与边界'});
  rect(s,.65,2.17,5.81,2.61,C.darkPanel);
  text(s,'媒体核心：已实现',.92,2.42,5.27,.4,23,C.brightTeal,true);
  text(s,'HBM：组织、地址、行状态、命令与资源',.92,3.04,5.27,.35,17,C.white);
  text(s,'HBF：组织、块页操作、缓存与总线',.92,3.56,5.27,.35,17,C.white);
  text(s,'论文 generic 基线不启用上层 FTL / GC',.92,4.16,5.27,.29,14,C.paleText);
  rect(s,6.88,2.17,5.82,2.61,C.darkPanel);
  text(s,'可选控制 / 管理：已实现',7.15,2.42,5.27,.4,23,C.brightAmber,true);
  text(s,'strict OCP：页命令准入、顺序写、自动擦除',7.15,3.04,5.27,.35,16.5,C.white);
  text(s,'OCP HOST 暂存：页合并 + 日志式 FTL + GC',7.15,3.56,5.27,.35,16.5,C.white);
  text(s,'另有可选磨损反馈；均按配置启用',7.15,4.16,5.27,.29,14,C.paleText);
  text(s,'当前边界',.65,5.21,12,.36,22,C.white,true);
  const scopes=[
    ['电路与内容','不模拟逐 bit / cell 内容、\nECC 与 retention 等行为'],
    ['完整协议','HBM 正常工作命令子集；\n不宣称完整 JEDEC / AXI 复现'],
    ['参数与校准','标准参考、论文输入与研究假设\n不等于目标芯片实测校准'],
  ];
  scopes.forEach(([a,b],i)=>{const x=.65+i*4.18;circle(s,String(i+1),x,5.82,C.brightAmber,C.navy);text(s,a,x+.58,5.83,3.16,.31,17.5,C.white,true);text(s,b,x,6.35,3.79,.5,14,C.paleText);});
  s.addNotes('范围需要分为媒体核心、可选管理和未覆盖细节。HBM有analytical与jedec_hbm，后者覆盖所示正常工作命令与时序，但不覆盖训练、自动预充电和完整刷新推迟策略。NandCore媒体核心本体不内置FTL/GC；论文generic_nand基线不启用上层管理。系统上层已存在strict ocp_hbf_0_7专用的HbfStagingController，提供有限HOST DRAM页合并、小型日志式FTL、异地写映射、free-block管理、victim选择、有效页复制及GC；strict OCP还约束页命令准入、完整对齐写、block顺序program和page-0自动erase。可选磨损反馈不等同于硬件级wear-level remapping。不要把模型已实现说成所有模式默认全部启用。仍不模拟逐bit/cell内容、电气电路、ECC/retention、完整AXI IDs与错误响应，也没有独立OOB/TSV阶段。参数有标准参考、论文输入和本地假设之分，尤其HBM4论文对照的tRRD/tFAW借用HBM3E参考；尚未目标芯片实测校准。来源：src/heterollm_sim/_hbm_control.py:1、dram_core.py:90、nand_core.py:353,608,680,716、memory_types.py:740–771、hbf_staging.py:476,491,503,864,908、scripts/exploring_hbf_paper_config.py:207–258,314。');
}

const out=path.join(__dirname,'HBM_HBF_已建模层级.pptx');
pres.writeFile({fileName:out}).then(()=>console.log(out));
