const pptxgen = require('pptxgenjs');
const path = require('path');
const fs = require('fs');
const p = new pptxgen();
p.layout='LAYOUT_WIDE';
p.author='HeteroLLM-Sim'; p.subject='当前工作区 HBM / HBF 建模层级、资源与精度边界';
p.title='HBM 与 HBF：仿真器究竟模拟到哪一层'; p.company='HeteroLLM-Sim'; p.lang='zh-CN';
p.theme={headFontFace:'Microsoft YaHei',bodyFontFace:'Microsoft YaHei',lang:'zh-CN'};
const C={ink:'192A2B',muted:'5E7071',dark:'132626',panel:'203A39',paper:'FFFFFF',grey:'F1F5F4',line:'CDDBD7',green:'087F6B',mint:'DDF2EB',lime:'88E0BF',orange:'B15D22',peach:'FBEBDC',gold:'F7BE80',white:'FFFFFF',red:'A63E3C'};
const S=p.ShapeType, W=13.333333,H=7.5; let n=0; const all=[];
function box(s,x,y,w,h,fill,line=fill,r=false){s.addShape(r?S.roundRect:S.rect,{x,y,w,h,rectRadius:.12,fill:{color:fill},line:{color:line,width:.8},radius:.1});}
function txt(s,t,x,y,w,h,size=18,color=C.ink,bold=false,align='left',extra={}){s.addText(t,{x,y,w,h,fontFace:'Microsoft YaHei',fontSize:size,color,bold,margin:0,align,valign:'mid',breakLine:false,paraSpaceAfterPt:0,...extra});}
function ln(s,x1,y1,x2,y2,color=C.line,width=1.2,arrow=false,dash=false){s.addShape(S.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1),flipH:x2<x1,flipV:y2<y1,line:{color,width,...(arrow?{endArrowType:'triangle'}:{}),...(dash?{dashType:'dash'}:{})}});}
function dot(s,t,x,y,fill=C.green,color=C.white,size=.42){s.addShape(S.ellipse,{x,y,w:size,h:size,fill:{color:fill},line:{color:fill,width:0}});txt(s,t,x,y,size,size,14,color,true,'center');}
function tag(s,t,x,y,w,fill=C.mint,color=C.green){box(s,x,y,w,.35,fill,fill,true);txt(s,t,x+.08,y+.025,w-.16,.3,11.5,color,true,'center');}
function base(title,sub='',dark=false,section=''){const s=p.addSlide();n++;s.background={color:dark?C.dark:C.paper};txt(s,section||'HeteroLLM-Sim · 建模层级',.62,.50,11.7,.20,10.5,dark?C.lime:C.muted,true);txt(s,title,.62,.79,12.05,title.includes('\n')?1.5:.72,36,dark?C.white:C.ink,true);if(sub)txt(s,sub,.62,1.57,12.05,.45,16,dark?'BDD2CA':C.muted);txt(s,String(n).padStart(2,'0'),12.02,6.80,.68,.2,10,dark?'BDD2CA':C.muted,false,'right');all.push({slide:n,title,notes:''});return s;}
function source(s,t,dark=false){txt(s,t,.62,6.80,11.1,.18,9.2,dark?'AFC8BF':C.muted);}
function notes(s,t){s.addNotes(t);all[all.length-1].notes=t;}
function node(s,t,x,y,w,h= .72,fill=C.grey,color=C.ink,size=18){box(s,x,y,w,h,fill,fill,true);txt(s,t,x+.13,y+.07,w-.26,h-.14,size,color,true,'center');}
function arrow(s,x,y,len=.4,color=C.muted){ln(s,x,y,x+len,y,color,1.8,true);}
function chip(s,x,y,w,h,label,color=C.green){box(s,x,y,w,h,color,color,true);for(let i=0;i<4;i++){ln(s,x-.12,y+.17+i*(h-.34)/3,x,y+.17+i*(h-.34)/3,color,2.5);ln(s,x+w,y+.17+i*(h-.34)/3,x+w+.12,y+.17+i*(h-.34)/3,color,2.5);}txt(s,label,x+.12,y+.14,w-.24,h-.28,22,C.white,true,'center');}
function drawer(s,x,y,w,h,active=1){box(s,x,y,w,h,C.grey,C.line,true);for(let i=0;i<3;i++){box(s,x+.13,y+.12+i*(h-.24)/3,w-.26,(h-.43)/3,i===active?C.mint:C.white,i===active?C.green:C.line,true);ln(s,x+w/2-.18,y+.22+i*(h-.24)/3,x+w/2+.18,y+.22+i*(h-.24)/3,i===active?C.green:C.muted,2);}}
function carton(s,x,y,w,h,color=C.orange){box(s,x,y,w,h,C.peach,color);ln(s,x,y,x+w/2,y+.17,color,1);ln(s,x+w,y,x+w/2,y+.17,color,1);ln(s,x+w/2,y+.17,x+w/2,y+h*.55,color,1);}

// 01 — establish one clear subject, then read left to right.
{
 const s=base('HBM 与 HBF\n仿真器究竟模拟到哪一层','',true,'HeteroLLM-Sim 4.0.0 · 当前工作区实现');
 txt(s,'从地址、介质动作与共享资源，\n解释一次访问如何影响 Token 时间',.66,2.55,5.45,1.05,24,'D5E7E0');
 tag(s,'代码驱动的机制讲解',.66,4.02,2.52,C.panel,C.lime);
 txt(s,'技术细节 × 日常比喻 × 可编辑示意图',.66,4.68,5.5,.4,16,'BDD2CA');
 txt(s,'基于 2026-10-26 工作区代码；参数示例不代表实测芯片',.66,6.19,5.6,.43,11.5,'AFC8BF');
 drawer(s,7.01,2.40,2.12,2.35);chip(s,7.22,5.10,1.7,.68,'HBM');
 carton(s,10.22,2.55,1.83,1.15);carton(s,10.22,3.94,1.83,1.15);chip(s,10.25,5.42,1.77,.68,'HBF',C.orange);
 txt(s,'打开抽屉，连续取物',6.75,1.93,2.7,.34,15,C.lime,true,'center');
 txt(s,'整箱取回，再按需拆零',9.64,1.93,3.02,.34,15,C.gold,true,'center');
 ln(s,9.43,2.18,9.43,6.07,'49615A',1,false,true);
 notes(s,'本演示解释当前工作区已实现的抽象层级，不是芯片规格介绍，也不宣称完整标准或电路级仿真。HBM用抽屉比喻开行与行命中；HBF用整箱拆零比喻整页阵列读取和主机访问粒度。所有比喻只是帮助理解，技术行为以当前代码为准。内容依次覆盖组织、地址、介质状态、资源竞争、系统指标和精度边界。主要来源：src/heterollm_sim/memory_types.py、memory_mapping.py、dram_core.py、_hbm_control.py、nand_core.py、data_motion.py。');
}
// 02 — the abstraction ladder is not a chronological read sequence.
{
 const s=base('五个观察层级','沿层级向下看细节，再沿执行结果向上解释性能');
 const levels=[['组织与参数','有哪些单元？容量、几何、带宽与时序'],['数据与地址','数据放在哪里？owner、allocation、offset'],['介质状态与动作','要做什么？开行、读页、编程、缓存替换'],['共享资源与事件','何时能做？准入、排队、依赖与资源日历'],['推理性能输出','造成什么影响？算子、Token、TTFT / TPOT']];
 levels.forEach((r,i)=>{let y=2.21+i*.76;box(s,.68,y,7.85,.61,i%2?C.grey:C.mint);dot(s,String(i+1),.87,y+.095,C.green);txt(s,r[0],1.48,y+.10,2.27,.38,18,C.ink,true);txt(s,r[1],3.95,y+.11,4.33,.37,13.5,C.muted);});
 chip(s,9.67,2.45,1.78,1.1,'Core');
 ln(s,10.56,3.81,10.56,4.29,C.green,2,true);
 node(s,'资源日历',9.41,4.43,2.29,.71,C.grey,C.green);
 ln(s,10.56,5.30,10.56,5.76,C.green,2,true);
 txt(s,'任务完成 → Token 时间',8.94,5.94,3.38,.38,18,C.ink,true,'center');
 source(s,'代码：physical_contract.py · data_motion.py · event_kernel.py · reporting.py');
 notes(s,'此页的五层是观察视角，不是一次读操作依次经过的五个时段。配置和组织决定地址可表达范围；allocation和owner把任务数据绑定到实际内存端点；Core维护介质状态并在共享日历上预定资源；完成时间回写事件内核，再由serving形成token指标。来源：src/heterollm_sim/physical_contract.py:154-200；planner.py:5074-5084；data_motion.py:1209-1235,1376-1485；event_kernel.py:685-705；serving.py:16736-16754,17035-17039。');
}
// 03 — ontology before geometry.
{
 const s=base('组织坐标、状态、执行资源','可被寻址的一层，不一定拥有独立总线或独立队列');
 const data=[{x:.68,num:'1',title:'组织坐标',sub:'它在哪里',fill:C.grey,t:'Bank / Row\nPlane / Block / Page',desc:'像楼栋、房间与抽屉编号',type:0},{x:4.91,num:'2',title:'介质状态',sub:'它现在怎样',fill:C.mint,t:'打开哪一行\n缓存哪一页、何时就绪',desc:'像抽屉是否打开、箱子是否取回',type:1},{x:9.14,num:'3',title:'执行资源',sub:'谁能同时工作',fill:C.peach,t:'命令总线 / DQ\n阵列 / 内部通道 / Host',desc:'像共用柜台、走廊与出口',type:2}];
 data.forEach(a=>{dot(s,a.num,a.x,2.27,C.ink);txt(s,a.title,a.x+.59,2.27,3.13,.42,24,C.ink,true);txt(s,a.sub,a.x,2.99,3.54,.37,17,C.muted);box(s,a.x,3.65,3.52,1.53,a.fill,a.fill,true);txt(s,a.t,a.x+.20,3.91,3.12,.99,18,C.ink,true);txt(s,a.desc,a.x,5.63,3.58,.72,15,C.muted);});
 source(s,'代码：memory_mapping.py · _hbm_control.py · nand_core.py');
 notes(s,'组织坐标回答位置，状态回答历史，执行资源回答共享和并行。比如HBF Block、Page都有物理坐标和计费意义，但不会各自增加独立阵列总线；HBM PC的数据通路独立，但同一物理Channel的两个PC共享行命令与列命令总线。资源数量不能仅从地址层级的节点数量推断。来源：src/heterollm_sim/memory_mapping.py:49-98,124-161；dram_core.py:98-102；nand_core.py:537-567。');
}
// 04 — common physical access contract.
{
 const s=base('共同入口：张量怎样变成一次访问','数据身份先绑定分配区间，真正派发任务时再提交给内存 Core');
 const xs=[.68,3.13,5.58,8.03,10.48];const ns=[['张量 / Buffer','权重、KV、激活'],['Allocation','owner 与 generation'],['物理字节范围','base + offset'],['AccessRequest','操作、地址、字节、到达'],['介质拆分','HBM burst / HBF page']];
 ns.forEach((r,i)=>{node(s,r[0],xs[i],2.57,2.14,.79,i===4?C.mint:C.grey,C.ink,17);txt(s,r[1],xs[i],3.65,2.14,.65,13,C.muted,false,'center');if(i<4)arrow(s,xs[i]+2.18,2.96,.23);});
 box(s,.68,4.80,7.34,1.46,C.grey,C.grey,true);txt(s,'仓位比喻',.92,5.02,1.7,.32,18,C.green,true);txt(s,'货物编号 → 仓位起点 → 第几格 → 本次取多少',.92,5.55,6.8,.36,20,C.ink,true);
 txt(s,'访问范围与地址校验',8.49,4.87,3.99,.42,21,C.ink,true);txt(s,'访问不能越出 allocation\n显式地址须与分配绑定一致，\n并落在设备容量内',8.49,5.43,3.99,1.02,16,C.muted);
 source(s,'代码：planner.py · memory_allocator.py · data_motion.py · memory_mapping.py');
 notes(s,'Planner生成memory_accesses并保持稳定buffer/tensor身份；allocator在运行内分配对齐且不搬移的地址，以buffer identity加generation区分生命周期。派发时重新解析allocation、offset和显式地址，检查范围与设备容量。之后构造AccessRequest(operation,address,byte_count,arrival)，调用DramCore或NandCore。纯读读不凭空增加依赖，重叠读写依赖会保留。不能把所有planner路径都声称为实测硬件trace，部分aggregate缓存路径另有显式抽象标记。来源：planner.py:4930-4974,5074-5084；memory_allocator.py:25-47,91-100；data_motion.py:1209-1235,1261-1282,1376-1406；_token_program.py:338-359。');
}
// 05 — flattening and geometry, one concrete authored example.
{
 const s=base('HBM：从 Stack 寻址到 Burst Column','示例：16H、32 Gb Die 的 HBM4 标准参考组织；不是已知厂商器件配置');
 const a=[['Stack','1 堆'],['Channel','32 / Stack'],['PC','2 / Channel'],['Bank Group','8 / PC'],['Bank','8 / Group']];
 a.forEach((r,i)=>{const x=.68+i*2.47;node(s,r[0],x,2.35,2.11,.75,i<3?C.mint:C.grey,C.green,18);txt(s,r[1],x,3.25,2.11,.40,14,C.muted,false,'center');if(i<4)arrow(s,x+2.15,2.72,.26,C.green);});
 box(s,.68,4.12,7.42,2.12,C.grey,C.grey,true);txt(s,'Bank 内部继续寻址',.92,4.34,6.94,.35,21,C.ink,true);
 for(let r=0;r<4;r++){box(s,.99,4.98+r*.23,3.20,.16,r===1?C.green:C.line);for(let q=0;q<8;q++)ln(s,1.39+q*.4,4.98+r*.23,1.39+q*.4,5.14+r*.23,C.white,.8);}
 txt(s,'Row → 32 B Burst Column',4.56,5.14,3.19,.65,18,C.green,true);
 txt(s,'容量不能重复乘 Die',8.63,4.21,3.99,.38,23,C.ink,true);txt(s,'64 PC × 64 Bank × 16,384 Row\n× 1,024 B = 64 GiB / Stack',8.63,4.89,3.99,.88,17,C.green,true);txt(s,'per_stack 几何已经包含 16H；\nSID 是从 Bank 序号派生的标签',8.63,5.98,3.99,.51,12.5,C.muted);
 source(s,'代码：hbm_geometry.py · memory_mapping.py · 测试 test_hbm_stack_geometry.py');
 notes(s,'HBM标准构造器对HBM4选择32物理Channel、每Channel两个PC。16H组织为每PC 8个Bank Group×每组8 Bank，共64 Bank。32Gb密度时每Bank 16384个有效1KiB Row，得到64GiB/Stack。bank_geometry_scope=per_stack，因此addressable_die_count=1，物理16 Die是描述字段，不可再次乘容量。通用DramConfig虽有die/subchannel/rank坐标，但标准构造器subchannel=1且rank=1。SID不是单独资源树节点，而由每16 Bank的编号派生。全局lane展平Stack、Channel与PC，lane是权威数据资源ID。HBM3/4有效寻址Row为1KiB，而开行激活页为2KiB，active_row会屏蔽相关高位；图中Row是有效寻址概念。来源：hbm_geometry.py:132-151,244-285；memory_types.py:593-599；memory_mapping.py:77-98；_hbm_dependencies.py:60-66；tests/test_hbm_stack_geometry.py:275-290。');
}
// 06 — two shared command windows, two independent data paths.
{
 const s=base('HBM：两个 PC，共享命令、各走数据','以下按 jedec_hbm 模式说明；命令能否发出与数据通路是否空闲是两件事');
 box(s,.70,2.22,7.41,4.18,C.grey,C.grey,true);txt(s,'一个物理 Channel',.98,2.41,6.84,.36,22,C.ink,true);
 node(s,'共享 Row CA\nACT / PRE / REF',1.04,3.12,3.02,1.02,C.mint,C.green,18);node(s,'共享 Column CA\nRD / WR',4.52,3.12,3.02,1.02,C.mint,C.green,18);
 txt(s,'两类命令窗口都服务 PC0 与 PC1',1.04,4.38,6.50,.30,14,C.muted,true,'center');
 [1.04,4.52].forEach((x,i)=>{box(s,x,4.96,3.02,1.07,C.white,C.line,true);txt(s,'PC'+i+' · 各 Bank 状态',x+.14,5.09,2.74,.28,16,C.ink,true,'center');node(s,'独立 DQ 数据通路',x+.16,5.53,2.70,.43,C.green,C.white,13.5);});
 txt(s,'Bank 开行与时序状态、命令和 DQ 分开维护',1.04,6.19,6.5,.25,12.5,C.muted,false,'center');
 txt(s,'两个柜台窗口',8.66,2.39,3.99,.42,24,C.ink,true);txt(s,'同一类命令共用窗口；\n不能因为 PC 不同就同时发出',8.66,3.08,3.99,.8,17,C.muted);
 txt(s,'两条传送带',8.66,4.35,3.99,.42,24,C.green,true);txt(s,'PC0 与 PC1 各有 DQ；\n数据传输可以独立排期',8.66,5.04,3.99,.8,17,C.muted);
 source(s,'代码：dram_core.py:98–102 · _hbm_control.py');
 notes(s,'资源归属是本页重点。同一物理Channel的两个PC共同使用Row CA和Column CA，两条CA本身是独立的命令资源，都可服务两个PC；DQ按PC分开，命令窗口不会直接等同DQ传输。Bank状态控制开行、预充电和下次操作的合法时间；PC/Channel还维护方向与组间约束。不要把PC独立DQ误画成CA也独立。来源：src/heterollm_sim/dram_core.py:98-102；_hbm_control.py:37-77,95-130。');
}
// 07 — state-driven sequences; not a proportional timing chart.
{
 const s=base('HBM：同样是读，行状态会改变动作','比喻：已开的抽屉直接取；关着的先开；开错的先关再开');
 const rr=[{name:'行命中',y:2.39,steps:[['RD',C.green],['RL',C.grey],['DQ',C.mint]],desc:'目标激活页已打开'}, {name:'未开行',y:3.75,steps:[['ACT',C.green],['tRCD',C.grey],['RD',C.green],['RL',C.grey],['DQ',C.mint]],desc:'先打开目标激活页'}, {name:'行冲突',y:5.11,steps:[['PRE',C.orange],['tRP',C.peach],['ACT',C.green],['tRCD',C.grey],['RD',C.green],['RL',C.grey],['DQ',C.mint]],desc:'先关闭旧页再打开新页'}];
 rr.forEach(r=>{txt(s,r.name,.68,r.y,1.50,.38,22,C.ink,true);txt(s,r.desc,.68,r.y+.49,2.10,.5,12.2,C.muted);r.steps.forEach((a,i)=>{const x=3.05+i*1.37;node(s,a[0],x,r.y+.14,1.10,.67,a[1],[C.green,C.orange].includes(a[1])?C.white:C.ink,14.5);if(i<r.steps.length-1)arrow(s,x+1.12,r.y+.48,.22);});});
 source(s,'示意不按时间比例；还受 tRAS / tRC、组间间隔、方向切换与可选刷新等约束');
 notes(s,'这是动作顺序图，不能根据形状宽度推断实际时延。jedec_hbm维护ACT/PREpb/PREab/RD/WR/REFab正常操作子集。命令占用与阵列时序约束分开，read latency与DQ占用也分开。行命中按active_row判断，HBM3/4的两个1KiB有效半页可能对应同一2KiB激活页。写恢复时间限制下次PRE，不额外添加到本次写的DQ完成尾部。周期all-bank refresh由配置启用，analytical模式不包含这一组完整约束。来源：_hbm_control.py:1-13,486-488,569-584；hbm_geometry.py:249-253,285-297,379-391。');
}
// 08 — a genuinely sourced numeric example, drawn natively.
{
 const s=base('HBM：三次同地址读的时钟例子','代码测试：HBM3E 参考、512 GiB 可见容量、1,600 GB/s 接口预算；非实测时延');
 tag(s,'tCK = 2.56 ns',.70,2.21,2.25);tag(s,'RL = 32 CK',3.26,2.21,2.25);tag(s,'tCCD = 4 CK',5.82,2.21,2.25);tag(s,'Burst = 2 CK',8.38,2.21,2.25);
 const x0=2.36,scale=.198,y0=3.16; const xx=t=>x0+t*scale;
 [0,10,20,30,40,50].forEach(t=>{ln(s,xx(t),3.10,xx(t),5.58,C.line,.8,false,true);txt(s,String(t),xx(t)-.17,2.79,.38,.24,11,C.muted,false,'center');});txt(s,'CK',12.39,2.79,.28,.24,11,C.muted);
 const lanes=[['Row CA',3.34],['Column CA',4.16],['DQ',4.98]];lanes.forEach(a=>{txt(s,a[0],.73,a[1]-.03,1.35,.33,16,C.ink,true);ln(s,x0,a[1]+.35,xx(50),a[1]+.35,C.line,1);});
 box(s,xx(0),3.35,.30,.30,C.green);txt(s,'ACT @ 0',2.40,3.76,1.45,.23,11.5,C.green,true);
 [7,11,15].forEach((t,i)=>{box(s,xx(t),4.15,.20,.34,C.green);txt(s,'RD'+(i+1),xx(t)-.14,4.60,.68,.23,11.5,C.green,true);box(s,xx(t+32),4.98,2*scale,.34,C.orange);txt(s,String(i+1),xx(t+32),5.02,2*scale,.22,10.5,C.white,true,'center');});
 txt(s,'RD：CK 7 / 11 / 15',.72,5.95,4.18,.38,18,C.green,true);txt(s,'DQ：39–41 / 43–45 / 47–49',4.73,5.95,5.7,.38,18,C.orange,true);
 txt(s,'完成：104.96 / 115.20 / 125.44 ns',.72,6.40,11.75,.26,14,C.muted);
 source(s,'来源：tests/test_hbm_clock_arithmetic.py:16–38；ACT 后阵列时序锚点为 CK1');
 notes(s,'测试明确规定tCK=2.56ns、read latency=81.92ns、same-group column spacing=10.24ns。ACT在CK0，阵列时序锚点在CK1；tRCD=14ns，合法首个RD对齐到CK7。之后RD为CK11、CK15。RL固定32CK，burst占2CK，故DQ分别为39-41、43-45、47-49CK；完成为41×2.56=104.96ns、45×2.56=115.20ns、49×2.56=125.44ns。三次请求均为32B，同地址，到达时间默认0。横轴为CK，ACT形状显示1.5CK占用但文字不声称完整波形。参考配置另覆盖512GiB，仅用于测试时钟计算。来源：tests/test_hbm_clock_arithmetic.py:16-38。');
}
// 09 — keep logical NAND coordinates distinct from paper die mapping.
{
 const s=base('HBF：地址到页，资源分到阵列与通道','通用 NAND 坐标完整保留；论文原生配置把实际 Die 映成 Core Channel');
 txt(s,'通用坐标',.70,2.23,1.77,.34,19,C.ink,true);txt(s,'Channel → Target → Die → LUN → Plane → Block → Page → offset',2.48,2.23,10.04,.37,16,C.orange,true);
 const cells=[['Stack 分组','16 个实际 Die / Stack'],['实际 Die','1 个 Core Channel'],['Plane','25 个独立阵列 / Die'],['Block → Page','擦除单位 → 4 KiB 页']];
 cells.forEach((a,i)=>{const x=.70+i*3.13;node(s,a[0],x,3.23,2.69,.74,C.peach,C.orange,22);txt(s,a[1],x,4.23,2.69,.65,15,C.muted,false,'center');if(i<3)arrow(s,x+2.75,3.60,.3,C.orange);});
 box(s,.70,5.29,7.05,1.18,C.grey,C.grey,true);txt(s,'同一实际 Die 的 Plane 共享内部传输通道',.94,5.48,6.57,.33,19,C.ink,true);txt(s,'Plane 阵列独立 ≠ 内部通道与 Host 出口独立',.94,6.00,6.57,.25,14,C.muted);
 txt(s,'论文 HBF 示例',8.26,5.34,4.38,.38,20,C.orange,true);txt(s,'每堆 16 × 25 = 400 个阵列单元\nTarget / Core Die / LUN 各取 1',8.26,5.94,4.38,.55,14.5,C.muted);
 source(s,'代码：memory_mapping.py:124–161 · scripts/exploring_hbf_paper_config.py:272–342');
 notes(s,'本页主图采用论文原生builder的HBF示例，不是全部HBF模式统一的组织。每个实际Die映射一个Core Channel，连续16个Channel构成一个Stack分组；NandConfig没有独立stack字段。targets_per_channel、dies_per_target、luns_per_die均为1，planes_per_lun=25且planes_independent=True。不得把实际Die再乘到Core die字段。每Plane拥有独立阵列日历，同一实际Die的Plane共享内部数据通道，最后汇聚到聚合Host接口。通用坐标无独立名为package的字段；target不擅自改称package。planes_independent=False时同LUN的Plane共享服务单元；parallel_units可折叠有效执行资源但不改变地址坐标。来源：memory_types.py:659-712；memory_mapping.py:124-161；nand_core.py:537-567；scripts/exploring_hbf_paper_config.py:272-342。');
}
// 10 — a read is multiple separately contended stages.
{
 const s=base('HBF：冷读与命中走不同的路径','整箱拆零：冷读先取回一整页，命中后可复用已取回的页');
 txt(s,'冷读 / Miss',.70,2.40,2.13,.4,22,C.orange,true);
 const a=[['准入','有限 outstanding'],['PAGE_READ','Plane 阵列'],['INTERNAL','共享内部通道'],['HOST','主机传输日历']];
 a.forEach((r,i)=>{const x=3.00+i*2.45;node(s,r[0],x,2.35,2.06,.75,i===1?C.orange:C.peach,i===1?C.white:C.orange,17);txt(s,r[1],x,3.34,2.06,.42,13,C.muted,false,'center');if(i<3)arrow(s,x+2.1,2.72,.27,C.orange);});
 txt(s,'缓存命中 / Hit',.70,4.39,2.13,.75,22,C.green,true);node(s,'缓存页已就绪',3.00,4.48,4.51,.75,C.mint,C.green,20);arrow(s,7.68,4.86,.42,C.green);node(s,'HOST',8.32,4.48,2.06,.75,C.green,C.white,18);
 txt(s,'命中仍需等待缓存页就绪与 Host 资源；\n缓存替换另受未完成读者约束，避免过早覆盖',3.00,5.60,9.64,.77,15,C.muted);
 source(s,'代码：nand_core.py:1800–1955 · memory_mapping.py:240–257');
 notes(s,'冷读先受准入限制，再PAGE_READ、INTERNAL_TRANSFER、HOST_TRANSFER，各阶段由不同资源日历计时，不能简单压成数据量除总带宽。跨页请求逐页拆分。阵列和内部搬运以完整媒体页计费；Host按host_granularity_bytes取整。读缓存命中跳过PAGE_READ与INTERNAL_TRANSFER，但要等ready与Host资源；reader_end约束缓存替换与重新填充时机，不是命中读取的额外前置条件。每bank两槽与LRU在所讲配置中启用，不代表所有通用NAND配置默认如此。来源：nand_core.py:641-676,1800-1955；memory_mapping.py:240-257。');
}
// 11 — distinguish cache lifetimes from an explicit SRAM core.
{
 const s=base('HBF：双页缓存与中间 SRAM 是两种模型','一个管理媒体页的复用与覆盖时机；另一个承接显式工作区访问');
 box(s,.70,2.26,6.29,4.22,C.peach,C.peach,true);txt(s,'每阵列的两个读缓存槽',.98,2.51,5.72,.4,23,C.orange,true);
 ['Slot A','Slot B'].forEach((a,i)=>{node(s,a,1.03+i*2.89,3.32,2.60,.67,C.white,C.orange,20);txt(s,'page tag · ready\nreader_end · LRU',1.03+i*2.89,4.33,2.60,.88,17,C.ink,false,'center');});
 txt(s,'覆盖前必须保护仍在读的页；\n两槽支持流水，但不会消除共享通道瓶颈',1.03,5.63,5.52,.57,15,C.muted);
 chip(s,8.66,2.80,2.60,1.14,'SramCore',C.green);txt(s,'独立中间 SRAM',7.73,4.33,4.49,.4,24,C.green,true,'center');txt(s,'显式容量、端口与带宽访问\n由 allocator 管理工作区生命周期',7.73,5.10,4.49,.86,17,C.muted,false,'center');
 source(s,'代码：nand_core.py:641–676 · sram_core.py · scripts/exploring_hbf_paper_config.py');
 notes(s,'NandCore页缓存是标签、ready、reader_end和LRU元数据，以及真实资源预订的抽象。它不是完整SRAM bank/port时序模型。论文HBF每Stack有16Die×25Plane×2槽×4096B=3,276,800B=3.125MiB原生读缓存；论文3.13MB标签在本地builder明确按约MiB解释，不能写成3,130,000B。中间SRAM是另建的SramCore与普通allocator工作区，本页不把两者容量相加或重复收费。来源：nand_core.py:641-676,1898-1922；scripts/exploring_hbf_paper_config.py:272-291,326-340；sram_core.py；scripts/exploring_hbf_native_config.py。');
}
// 12 — mode-specific write completion and upper-layer management.
{
 const s=base('HBF：媒体写入与可选管理层','写完成的含义，取决于协议模式和是否启用 HOST 暂存');
 txt(s,'通用 NAND 核心',.70,2.28,5.83,.42,23,C.orange,true);txt(s,'Strict OCP + HOST 暂存',7.01,2.28,5.60,.42,23,C.green,true);
 node(s,'小写入',.70,3.03,1.63,.67,C.grey,C.ink,18);arrow(s,2.44,3.37,.32);node(s,'RMW 或拒绝',2.88,3.03,2.30,.67,C.peach,C.orange,17);
 txt(s,'未启用 write_buffer：\n部分页写按配置补读旧页再 Program，\n或直接拒绝',.70,4.13,5.71,1.14,17,C.muted);
 txt(s,'可选 write_buffer：有限容量、页合并；\nHost 到达可先确认，媒体随后 drain',.70,5.61,5.70,.65,15,C.muted);
 const x=7.01;for(let j=0;j<16;j++)box(s,x+(j%8)*.38,3.09+Math.floor(j/8)*.28,.30,.20,j<12?C.green:C.mint);
 txt(s,'16 × 256 B = 4 KiB',10.39,3.12,2.21,.62,16,C.green,true);
 node(s,'HOST 页合并',7.01,4.15,2.25,.72,C.mint,C.green,18);arrow(s,9.40,4.51,.31,C.green);node(s,'整页 Program',9.87,4.15,2.73,.72,C.green,C.white,18);
 txt(s,'上层小型日志式 FTL：异地更新、映射、GC\n核心普通写必须整页，完成到媒体；\n不能借用 generic 的提前确认语义',7.01,5.28,5.60,1.12,15.5,C.muted);
 source(s,'代码：nand_core.py · hbf_staging.py · hbf_staging_runtime.py · 测试 test_hbf_staging.py:303–319');
 notes(s,'NandCore本体不含FTL/GC；但项目上层HbfStagingController已实现有限HOST DRAM暂存、小型日志式FTL和live-page GC，不能笼统说项目无FTL/GC。通用模式write_buffer_bytes>0是核心有限host-side budget，有页合并和host-ack/drain语义；strict ocp_hbf_0_7禁止该核心write_buffer，要求完整对齐4KiB普通写，顺序Program，page-0自动擦除，完成点为媒体完成。独立HOST DRAM staging接小KV片段，绑定真实HOST DRAM与fabric，不能拿HBM冒充。测试16次256B写在凑足4KiB后才Program。FTL异地更新旧页失效，GC搬存活页后复用块；不是原地单页擦除。来源：nand_core.py:353-362,784-899,1840-1884,1958-1996；memory_mapping.py:199-208；hbf_staging.py:476-526,834-986；hbf_staging_runtime.py:1-38；tests/test_hbf_staging.py:303-319。');
}
// 13 — preserve owner/resource competition and kernel envelope semantics.
{
 const s=base('跨设备搬运与算子完成时间','媒体服务结果回到任务图，计算、链路与内存的重叠由模型和依赖决定');
 node(s,'源端访问',.70,2.35,2.33,.74,C.peach,C.orange,20);arrow(s,3.21,2.72,.59);node(s,'链路 / Fabric',4.01,2.35,2.55,.74,C.grey,C.ink,20);arrow(s,6.76,2.72,.59);node(s,'目的端访问',7.56,2.35,2.56,.74,C.mint,C.green,20);
 txt(s,'具体任务路径决定串行、分块或流水关系；每个 owner 保留自己的日历与计费',.70,3.45,11.97,.4,15,C.muted);
 txt(s,'Native kernel 的服务包络',.70,4.25,4.89,.4,23,C.ink,true);
 txt(s,'其他资源',.70,5.05,1.63,.3,15,C.muted);box(s,2.61,5.04,4.11,.31,C.line);
 txt(s,'内存 Core',.70,5.70,1.63,.3,15,C.green,true);box(s,2.61,5.69,5.39,.31,C.green);
 ln(s,8.00,4.98,8.00,6.13,C.green,1.4,false,true);node(s,'+ dependency',8.29,5.30,2.31,.56,C.grey,C.ink,15);arrow(s,10.83,5.58,.45);txt(s,'完成',11.46,5.38,1.18,.4,21,C.ink,true);
 txt(s,'Tkernel = max（其他资源跨度，内存 Core 跨度）+ dependency',.70,6.34,11.95,.32,17,C.ink,true);
 source(s,'公式适用于代码中的 native kernel envelope；不是所有搬运阶段的通用公式');
 notes(s,'资源日历为每个owner记录真实资源占用；单lane预订开始时间=max(earliest,ready+方向切换代价)，多lane选最早可用lane。图的上半部分是搬运要计入的端点与链路示意，不声称所有路径均完整串行相加；具体直接、staged或分块流水关系需查看实际任务。下半部分对应data_motion中的native kernel envelope：原backing memory估算由实际Core span替换，避免重复加一遍；kernel duration=max(other resource span,memory span)+dependency。模型是Core memory加已有compute resource model，非实测whole-kernel replay。来源：memory_transfer.py:263-328；data_motion.py:1444-1485,1730-1769；event_kernel.py:685-705；kv_transfer.py、memory_transfer.py、data_motion.py相关搬运实现。');
}
// 14 — stage-specific amplification; don't sum traffic on different buses.
{
 const s=base('64 B 读请求，为什么可能触发 4 KiB 页读','固定示例：HBM 为 32 B burst；HBF 为 strict OCP 4 KiB 页、64 B Host 粒度');
 const rows=[['HBM · 对齐 64 B','2 × 32 B DQ','64 B','开行与命令时序另计',C.mint,C.green],['HBF · 冷缓存','4 KiB 阵列页 + 内部搬运','64 B Host','媒体页 / 有效载荷 = 64×',C.peach,C.orange],['HBF · 已就绪命中','跳过阵列读与内部搬运','64 B Host','仍受 Host 日历与依赖约束',C.grey,C.ink]];
 txt(s,'访问条件',.87,2.27,2.44,.3,16,C.muted,true);txt(s,'介质 / 内部动作',3.71,2.27,4.04,.3,16,C.muted,true);txt(s,'接口有效传输',8.18,2.27,3.93,.3,16,C.muted,true);
 rows.forEach((r,i)=>{let y=2.90+i*1.08;box(s,.70,y,11.93,.83,r[4],r[4],true);txt(s,r[0],.93,y+.15,2.48,.49,17,r[5],true);txt(s,r[1],3.72,y+.15,4.00,.49,17,C.ink,true);txt(s,r[2],8.21,y+.15,1.74,.49,17,r[5],true);});
 txt(s,'64×',.78,6.00,1.67,.65,34,C.orange,true);txt(s,'只比较冷读的媒体页容量与有效载荷；不是延迟倍数，也不是 Host 传输倍数',2.66,6.16,9.65,.48,15,C.muted);
 source(s,'代码：memory_mapping.py:173–188,240–257 · nand_core.py:1898–1955');
 notes(s,'本页假设地址64B对齐且不跨HBF页，HBM burst为32B，strict OCP HBF Host粒度为64B。HBM逻辑64B触发2个burst，总DQ字节64B，但ACT/PRE/CA等时序不能因字节放大为1而忽略。HBF冷读的阵列和内部搬运是4096B，Host仍为64B；媒体页容量相对有效载荷4096/64=64倍。命中且页ready时跳过阵列和内部传输。不同资源的字节不能相加成一个含糊的总流量，再拿它去除某一带宽。论文generic配置常把Host粒度设为4KiB，因此同一小请求在那一配置也可能有Host放大；不能把本页OCP数值套过去。来源：memory_mapping.py:173-188,240-257；nand_core.py:1898-1955；physical_presets.py:130-163；tests/test_hbf_protocol_contract.py:28-59。');
}
// 15 — three time/traffic scopes.
{
 const s=base('怎样把访存影响读成系统性能','一次 transaction 的延迟先影响关键路径，再影响 Token；它不直接等于 TPOT');
 const r=[{x:.70,num:'1',title:'字节账',big:'要多少 / 搬多少',a:'Logical bytes：应用有效载荷',b:'Physical：按 burst / page 计费',fill:C.grey},{x:4.90,num:'2',title:'时间账',big:'跨度 / 忙时 / 总耗时',a:'service_ns：任务 / owner\n完成跨度之和（可重叠）',b:'busy：资源实际占用\nmakespan：总跨度',fill:C.mint},{x:9.10,num:'3',title:'Token 账',big:'TTFT / TPOT',a:'首 Token 时间与后续提交时刻',b:'TPOT 由可见 Token 间隔计算',fill:C.peach}];
 r.forEach(a=>{dot(s,a.num,a.x,2.32,C.ink);txt(s,a.title,a.x+.60,2.31,2.92,.42,24,C.ink,true);box(s,a.x,3.19,3.54,2.45,a.fill,a.fill,true);txt(s,a.big,a.x+.20,3.48,3.14,.78,22,C.ink,true);txt(s,a.a,a.x+.20,4.40,3.14,.58,13,C.muted);txt(s,a.b,a.x+.20,5.03,3.14,.54,12.5,C.muted);});
 txt(s,'TPOT =（末个 Token 时间 − 首个 Token 时间）/（可见 Token 数 − 1）',.72,6.12,11.9,.46,20,C.ink,true);
 source(s,'代码：memory_service.py:26–84 · memory_types.py:890–916 · serving.py:17035–17039');
 notes(s,'service_ns累加per-task/per-owner完成跨度，这些跨度可重叠，不等于elapsed wall time，也不等于DQ busy time。实际每资源busy和bytes在resource_totals中。transaction achieved bandwidth通常logical bytes/latency；物理页和burst放大、Host/internal字节另看。Token提交时间取terminal item完成；TPOT=(last-first)/(committed_tokens-1)，不足两个可见Token时未定义。系统性能须由任务依赖和资源竞争推导，不能拿一次HBF冷页读取延迟代替TPOT。来源：memory_service.py:26-84；memory_types.py:890-916；memory_transfer.py:478-510；serving.py:16736-16754,16801-16809,17035-17039。');
}
// 16 — preserve model choice, implementation acceleration and sampling distinction.
{
 const s=base('模型精度、执行加速、采样预测','这三种开关改变的东西不同，解读结果时必须分开');
 const rows=[['模型选择','HBM analytical / jedec_hbm\nHBF generic / strict OCP','改变状态、命令、约束与完成点',C.grey],['同模型执行加速','少记明细、数值 / 编译路径\n可证明分支的 max-plus 跳步','保留该模型的资源竞争与状态转移',C.mint],['采样与预测','选中层独立副本执行\nfast window / stationary 预测','预测窗口不推进全部\nlive 物理日历',C.peach]];
 rows.forEach((r,i)=>{let y=2.31+i*1.35;dot(s,String(i+1),.70,y+.17,i===2?C.orange:C.green);txt(s,r[0],1.30,y+.10,2.42,.56,22,C.ink,true);box(s,4.02,y,4.11,1.04,r[3],r[3],true);txt(s,r[1],4.22,y+.15,3.71,.73,16,C.ink,true);txt(s,r[2],8.65,y+.15,3.94,.73,17,i===2?C.orange:C.muted);});
 source(s,'代码：dram_core.py · _compiled_nand.py · _nand_acceleration.py · native_*sampling / fast_window');
 notes(s,'HBM标准构造器control_mode=jedec_hbm，但DramCore也支持显式analytical，后者有简化Bank开行、PRE/ACT与传输，不是纯bytes/BW，却没有jedec_hbm完整正常操作约束。capture_details=False不自动改变控制模式。编译NAND仍逐页更新缓存；max-plus acceleration仅在可证明max分支和日历转移后跳步，不能说跳过了物理模型。采样则不同：native_layer_sampling在独立副本执行完整选中层，遗漏前驱作为boundary inputs，不提交live可见Token；native_fast_window不推进live请求/物理日历，forecast_has_physical_trace=False；stationary假设同类未采样层地址跨row差异可忽略，尚未严格界定。来源：dram_core.py:90-105,527-556；_compiled_nand.py:11-25；_nand_acceleration.py:1-8；native_layer_sampling.py:1-7；native_fast_window.py:1-7,29-35；native_stationary.py:34-40,70-74。');
}
// 17 — implementation coverage vs absent details, no overclaim.
{
 const s=base('已实现的建模深度与边界','准确名称：有状态、物理地址驱动的事务 / 命令与资源时序仿真');
 const rows=[['地址与组织','Stack / Channel / PC / Bank / Row','Channel / LUN / Plane / Block / Page'],['状态与时序','正常操作命令子集、开行、方向与刷新约束','页读 / Program / Erase；可配置读缓存'],['可选管理','简化 analytical 或 jedec_hbm；有限请求调度','generic write buffer；strict OCP HOST FTL / GC'],['未覆盖 / 不宣称','完整 JEDEC、厂商映射、训练与电路行为','完整 AXI、ECC / retention 与 cell 内容行为']];
 const headers=['范围','HBM','HBF'];
 const data=[headers.map((t,j)=>({text:t,options:{bold:true,color:C.white,fill:{color:C.dark},fontSize:18}})),...rows.map((r,i)=>r.map((t,j)=>({text:t,options:{bold:j===0,color:i===3?C.muted:C.ink,fill:{color:j===0?C.grey:(j===1?C.mint:C.peach)},fontSize:j===0?16:14.5}})))];
 s.addTable(data,{x:.70,y:2.27,w:11.93,colW:[1.66,5.03,5.24],rowH:[.55,.75,.83,.83,.86],margin:[.12,.16,.12,.16],fontFace:'Microsoft YaHei',border:{type:'solid',pt:1.5,color:C.white},valign:'middle',autoPage:false});
 txt(s,'机制已经实现 ≠ 参数已对目标芯片校准；研究假设、标准参考与实测证据应分别说明',.72,6.40,11.9,.28,15,C.orange,true);
 source(s,'代码：hbm_geometry.py:379–391 · nand_core.py:353–362 · hbf_staging.py:476–526');
 notes(s,'这张表是实现范围，不是所有模式同时启用。HBM刷新all_bank需要参数启用，analytical不具备相同约束；jedec_hbm仅正常命令子集，缺训练、parity/ECC传输、auto-precharge和per-bank refresh。NandCore不含FTL/GC，但上层strict OCP staging有小型日志式FTL和GC。HBF可选wear model计擦除P/E并按用户比例增长Program时延，不等同寿命物理预测或wear-level remapping。参考预设有明确假设来源；论文HBM4借用HBM3E的tRRD/tFAW不能包装成目标厂商已校准。strict OCP参考4us读/75us写/2ms擦也是探索假设，不是OCP规范强制值。来源：hbm_geometry.py:379-391；nand_core.py:353-362,724-768；physical_presets.py:137-163；scripts/exploring_hbf_paper_config.py:231-258；hbf_staging.py:476-526,834-986。');
}
// 18 — a clean end, centered on the useful mental model.
{
 const s=base('理解这个仿真器的三个抓手','先找到数据位置，再看介质动作，最后看谁在等待谁',true);
 const items=[['位置','地址层级告诉你\n数据落在哪个物理单元',C.lime],['动作','HBM 看开行与命令\nHBF 看页、缓存与编程',C.gold],['竞争','共享总线与任务依赖\n共同决定完成时间',C.lime]];
 items.forEach((a,i)=>{let x=.72+i*4.20;dot(s,String(i+1),x,2.45,a[2],C.dark,.52);txt(s,a[0],x,3.38,3.49,.61,34,C.white,true);txt(s,a[1],x,4.35,3.49,1.02,21,'CEE0D8');if(i<2)ln(s,x+3.64,2.43,x+3.64,5.60,'49615A',1,false,true);});
 txt(s,'HBM 像按抽屉取物；HBF 像整箱拆零。\n仿真器真正计算的是：这些动作如何占用资源，并推迟后续任务。',.72,5.98,11.93,.65,19,C.white,true);
 source(s,'阅读每次实验结果时，同时确认：物理配置、协议模式、是否采用采样预测',true);
 notes(s,'结论回到用户可操作的理解：地址不只是一串数字，几何决定映射；HBM的状态与命令决定何时能占DQ；HBF的页/缓存/Program决定何时能走共享通道；任务图把这些完成点连成关键路径。不要仅根据峰值带宽判断性能，也不要仅根据地址层级推导独立资源数。最后读实验结果必须同时检查配置、protocol/control mode和采样预测标记。所有解释基于2026-10-26当前工作区而非单一提交快照，现有未提交改动亦在分析范围中。');
}
// 19 — reference map is an optional appendix, with actionable reading order.
{
 const s=base('附录：按问题阅读代码','每页讲稿备注还包含对应文件和行号，可用于继续核对细节');
 const groups=[['定义哪些层级？','memory_types.py\nhbm_geometry.py / _hbm_dependencies.py','配置与标准组织'],['地址怎样落到资源？','memory_mapping.py\nmemory_allocator.py','映射、拆分与分配'],['访问怎样完成？','dram_core.py / _hbm_control.py\nnand_core.py / hbf_staging.py','状态、时序与管理'],['怎样影响 Token？','data_motion.py / event_kernel.py\nmemory_service.py / serving.py','回写、依赖与指标']];
 groups.forEach((a,i)=>{const y=2.28+i*1.04;dot(s,String(i+1),.72,y+.07,C.green);txt(s,a[0],1.34,y+.03,3.03,.52,20,C.ink,true);txt(s,a[1],4.56,y+.02,5.38,.67,14,C.green);txt(s,a[2],10.22,y+.11,2.38,.49,14,C.muted);});
 txt(s,'数值例来源：test_hbm_clock_arithmetic.py · test_hbm_stack_geometry.py · test_hbf_staging.py',.72,6.43,11.92,.25,11.8,C.muted);
 source(s,'以上文件位于 src/heterollm_sim；测试位于 tests；论文配置入口位于 scripts');
 notes(s,'完整源码阅读顺序：1 src/heterollm_sim/memory_types.py:243-304,659-773，hbm_geometry.py:266-297，_hbm_dependencies.py:60-66。2 memory_mapping.py:49-98,124-188,199-257，memory_allocator.py:25-47,91-100。3 dram_core.py:90-105,411-420,527-556，_hbm_control.py:1-13,37-130,486-584，nand_core.py:353-362,537-690,1800-1996，hbf_staging.py:476-526,834-986。4 data_motion.py:1209-1235,1376-1485,1730-1769，event_kernel.py:685-705，memory_service.py:26-84，serving.py:17035-17039。数值例：tests/test_hbm_clock_arithmetic.py:16-38；tests/test_hbm_stack_geometry.py:275-290；tests/test_hbf_staging.py:303-319。论文映射：scripts/exploring_hbf_paper_config.py:231-342。本PPT未使用外部图片；示意图均为可编辑PowerPoint原生形状，图中比喻并非硬件结构实拍。');
}
if(n!==19)throw new Error('Unexpected slide count');
fs.writeFileSync(path.join(__dirname,'.build','storyboard.json'),JSON.stringify(all,null,2));
p.writeFile({fileName:path.join(__dirname,'HBM_HBF_建模层级讲解.pptx')}).then(()=>console.log('Created 19-slide deck'));
