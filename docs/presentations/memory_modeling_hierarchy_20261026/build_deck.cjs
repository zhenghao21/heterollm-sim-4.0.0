// Build: NODE_PATH=<dir with pptxgenjs> node build_deck.cjs
// Numbers come from probe_results.json (simulator cores) unless a slide footer
// names another source.
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');

const R = JSON.parse(fs.readFileSync(path.join(__dirname, 'probe_results.json'), 'utf8'));
const p = new pptxgen();
p.layout = 'LAYOUT_WIDE';
p.title = '大模型推理访存与 HBM / HBF / SRAM 建模层级';
p.author = 'HeteroLLM-Sim';
p.lang = 'zh-CN';
const F = 'Microsoft YaHei';
p.theme = { headFontFace: F, bodyFontFace: F, lang: 'zh-CN' };
const S = p.ShapeType;

const C = {
  ink: '1A1D23', sub: '52514E', muted: '8A8A86', line: 'D9D9D4', panel: 'F4F4F1', white: 'FFFFFF',
  navy: '13213A', navySub: 'B9C4D6',
  hbm: '2A78D6', hbmDark: '1F5FAE', hbmTint: 'E4EEFA', hbmMid: 'BBD3F2',
  hbf: 'EB6834', hbfDark: 'C4501F', hbfTint: 'FCE8DF', hbfMid: 'F6C2AA',
  sram: '1BAF7A', sramDark: '13845C', sramTint: 'DDF3EA', sramMid: 'A9E0CA',
  warn: 'B3261E',
};
const W = 13.333;
let page = 0;

// ---- helpers --------------------------------------------------------------
function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, {
    x, y, w, h, fontFace: F, fontSize: o.size || 14, color: o.color || C.ink, bold: !!o.bold,
    align: o.align || 'left', valign: o.valign || 'top', margin: o.margin ?? 0,
    paraSpaceAfter: o.para ?? 0, lineSpacingMultiple: o.lsm || 1.12, lang: 'zh-CN', ...(o.extra || {}),
  });
}
function rect(s, x, y, w, h, fill, line) {
  s.addShape(S.rect, { x, y, w, h, fill: { color: fill }, line: { color: line || fill, width: line ? 0.75 : 0 } });
}
function hline(s, x1, y, x2, color = C.line, width = 1, arrow = false, dash = false) {
  s.addShape(S.line, { x: x1, y, w: x2 - x1, h: 0, line: { color, width, ...(arrow ? { endArrowType: 'triangle' } : {}), ...(dash ? { dashType: 'dash' } : {}) } });
}
function vline(s, x, y1, y2, color = C.line, width = 1, arrow = false) {
  s.addShape(S.line, { x, y: Math.min(y1, y2), w: 0, h: Math.abs(y2 - y1), flipV: y2 < y1,
    line: { color, width, ...(arrow ? { endArrowType: 'triangle' } : {}) } });
}
function arrow(s, x1, y, x2, color = C.muted) { hline(s, x1, y, x2, color, 1.5, true); }
function circleNum(s, n, x, y, fill = C.navy, size = 0.36) {
  s.addShape(S.ellipse, { x, y, w: size, h: size, fill: { color: fill }, line: { color: fill, width: 0 } });
  text(s, String(n), x, y, size, size, { size: 12, bold: true, color: C.white, align: 'center', valign: 'middle' });
}
function base(kicker, title, footer) {
  const s = p.addSlide();
  page += 1;
  s.background = { color: C.white };
  text(s, kicker, 0.6, 0.34, 9, 0.26, { size: 11, color: C.muted, bold: true });
  text(s, title, 0.6, 0.6, 12.13, 0.9, { size: 25, bold: true, valign: 'top', lsm: 1.05 });
  if (footer) text(s, footer, 0.6, 7.02, 11.3, 0.3, { size: 9, color: C.muted, valign: 'bottom' });
  text(s, String(page).padStart(2, '0'), 12.13, 7.02, 0.6, 0.3, { size: 9, color: C.muted, align: 'right', valign: 'bottom' });
  return s;
}
function chip(s, label, x, y, w, h, fill, size = 13) {
  rect(s, x, y, w, h, fill);
  text(s, label, x, y, w, h, { size, bold: true, color: C.white, align: 'center', valign: 'middle' });
}
// Speaker notes: the spoken script from 演讲稿.md, then the technical backup.
const SPEECH = {};
fs.readFileSync(path.join(__dirname, '演讲稿.md'), 'utf8').split(/^## /m).slice(1).forEach((part) => {
  const m = part.match(/^第 (\d+) 页[^\n]*\n([\s\S]*)$/);
  if (m) SPEECH[+m[1]] = m[2].replace(/^---[\s\S]*$/m, '').replace(/\*\*/g, '').trim();
});
function note(s, backup) {
  const spoken = SPEECH[page];
  if (!spoken) throw new Error('演讲稿缺少第 ' + page + ' 页');
  s.addNotes(spoken + '\n\n—— 备查（数据与代码出处）——\n' + backup);
}
const fmt = (v, d = 0) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const sw = R.size_sweep, hp = R.hbm_placement, fp = R.hbf_placement;

// ---- 01 cover -------------------------------------------------------------
{
  const s = p.addSlide(); page += 1;
  s.background = { color: C.navy };
  text(s, 'HeteroLLM-Sim 4.0.0 · 存储建模汇报', 0.8, 0.8, 8, 0.3, { size: 13, color: C.navySub, bold: true });
  text(s, '大模型推理的访存需求极其复杂\n仿真器如何对 HBM / HBF / SRAM 精细建模', 0.8, 1.45, 11.8, 1.6, { size: 34, bold: true, color: C.white, lsm: 1.15 });
  text(s, '从每一次访存请求出发：拆成 burst / 页 / 端口，记住行与页的状态，\n在共享的物理资源上排队计时，最后汇总成 Token 延迟与吞吐。', 0.8, 3.3, 11.5, 0.9, { size: 17, color: C.navySub, lsm: 1.3 });
  const cards = [['HBM', '32 B burst · bank 行状态 · 命令时序', C.hbmDark], ['HBF', '4 KiB 页 · plane 并行 · 页缓冲', C.hbfDark], ['SRAM', '64 B burst · 端口流水线', C.sramDark]];
  cards.forEach(([name, desc, color], i) => {
    const x = 0.8 + i * 3.95;
    rect(s, x, 4.75, 3.7, 1.05, '1C2D4B');
    rect(s, x, 4.75, 0.09, 1.05, color);
    text(s, name, x + 0.3, 4.87, 3.2, 0.4, { size: 20, bold: true, color: C.white });
    text(s, desc, x + 0.3, 5.3, 3.3, 0.35, { size: 12.5, color: C.navySub });
  });
  text(s, '2026-10-26　｜　文中数值来自仿真器自身的 HBM / HBF / SRAM 核心实测；配置为研究参考，未经厂商器件标定', 0.8, 6.65, 11.8, 0.35, { size: 11, color: '8796AE' });
  note(s, '这份材料回答一个问题：为什么仿真器要把 HBM、HBF、SRAM 建模到 burst、页、端口这一级，而不是用一个带宽数折算。'
    + '核心观点：大模型推理里单次访存的大小、存放位置、并发度、读写方向差别很大，同样的字节数耗时可以差几十到几万倍；只有按介质的物理层级逐请求计算，仿真才能反映这些情况。'
    + '所有数字都是用仿真器自己的介质核心（DramCore、NandCore、SramCore）跑出来的，脚本 probe_memory_patterns.py，结果 probe_results.json。');
}

// ---- 02 conclusion ---------------------------------------------------------
{
  const s = base('01 · 结论先行', '访存耗时取决于“怎么访问”，不是一个带宽数能折算的',
    '探针：论文 Table I 的 HBM4（8 堆，12.8 TB/s）与 HBF（7 堆，读 11.2 TB/s）配置，冷启动单请求；数据 probe_results.json');
  const one = sw.hbm['1 token'].span_ns, oneF = sw.hbf['1 token'].span_ns;
  const adj = hp['256 x 4KiB adjacent'].gb_s, sca = hp['256 x 4KiB scattered'].gb_s;
  const wH = R.hbm_write['1 token 256B'].span_ns, wF = R.hbf_write['1 token 256B'].span_ns;
  const cards = [
    { c: C.hbf, label: '读 1 个 token 的 KV（256 B）', big: (oneF / one).toFixed(1) + '×', cmp: `HBM ${one.toFixed(1)} ns　｜　HBF ${fmt(oneF)} ns`,
      why: 'HBF 必须把整页 4 KiB 读出，1 µs 的阵列读页时间被这 256 B 独自承担。' },
    { c: C.hbm, label: '同样读 1 MiB：相邻存放 vs 分散存放', big: (adj / sca).toFixed(1) + '×', cmp: `相邻 ${fmt(adj / 1000, 2)} TB/s　｜　分散 ${fmt(sca / 1000, 2)} TB/s`,
      why: `分散后出现 ${fmt(hp['256 x 4KiB scattered'].row_conflicts)} 次行冲突：bank 要先关掉旧行，再打开新行。` },
    { c: C.hbf, label: '写 1 个 token 的 KV（256 B）', big: '约 ' + fmt(Math.round(wF / wH / 100) * 100) + '×', cmp: `HBM ${wH.toFixed(1)} ns　｜　HBF ${fmt(wF / 1000, 1)} µs`,
      why: '不满一页的写要先读出旧页、合并后整页编程，编程一次 100 µs。' },
  ];
  cards.forEach((k, i) => {
    const x = 0.6 + i * 4.14;
    rect(s, x, 1.75, 3.85, 3.25, C.panel);
    rect(s, x, 1.75, 3.85, 0.07, k.c);
    text(s, k.label, x + 0.25, 1.98, 3.4, 0.35, { size: 14, bold: true });
    text(s, k.big, x + 0.25, 2.4, 3.4, 0.85, { size: 44, bold: true });
    text(s, k.cmp, x + 0.25, 3.32, 3.45, 0.35, { size: 13.5, color: C.ink });
    text(s, k.why, x + 0.25, 3.8, 3.4, 1.0, { size: 12.5, color: C.sub, lsm: 1.25 });
  });
  text(s, '决定一次访存耗时的因素', 0.6, 5.28, 4, 0.32, { size: 13, bold: true, color: C.sub });
  const f = ['单次访问多大', '数据存放在哪', '同时在途多少', '读还是写', '介质当前状态（开着哪一行、缓存着哪一页）'];
  let x = 0.6;
  f.forEach((t, i) => {
    const w = i === 4 ? 4.3 : 1.72;
    rect(s, x, 5.68, w, 0.48, C.panel, C.line);
    text(s, t, x, 5.68, w, 0.48, { size: 12.5, align: 'center', valign: 'middle' });
    x += w + 0.2;
  });
  text(s, '→ 仿真器逐请求、按介质的物理层级计算完成时间，而不是“字节数 ÷ 峰值带宽”。', 0.6, 6.38, 12.1, 0.4, { size: 15, bold: true });
  note(s, '三张卡片对应汇报要回答的三类典型情况。'
    + `卡片一：读 1 个 token 的 KV。405B、TP8 时每张卡每层每 token 的 K 只有 256 B。HBM 冷启动读完 ${one.toFixed(2)} ns，HBF ${oneF.toFixed(1)} ns，因为 HBF 要先做 1 µs 的阵列读页，再把整页 4 KiB 搬出来。`
    + `卡片二：同样是 256 次 4 KiB 读、总共 1 MiB，相邻存放 ${fmt(adj)} GB/s，随机分散存放 ${fmt(sca)} GB/s；分散后行冲突 ${hp['256 x 4KiB scattered'].row_conflicts} 次，相邻存放为 0。`
    + `卡片三：写 1 个 token 的 KV，HBM ${wH.toFixed(1)} ns，HBF ${(wF / 1000).toFixed(2)} µs（读改写：旧页读 1 µs + 整页编程 100 µs）。`
    + '这些差异来自访问形态，不是来自介质标称带宽，所以必须精细建模。');
}

// ---- 03 access patterns -----------------------------------------------------
{
  const s = base('02 · 推理访存的真实形态', '同一个 Decode 步里，几类截然不同的访存同时发生、争用同一组资源',
    '尺寸按 Llama 3.1 405B、BF16、TP8（每卡 1 个 KV 头 × 128 维）计算；SHORT / LONG 为论文负载 1,660 / 103,500 输入 token');
  const hd = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 13 } });
  const c0 = (t) => ({ text: t, options: { bold: true, fontSize: 13.5, fill: { color: C.panel } } });
  const c = (t) => ({ text: t, options: { fontSize: 12.5 } });
  const rows = [
    [hd('数据'), hd('单次访问有多大（每卡、每层）'), hd('访问特点'), hd('对存储的考验')],
    [c0('权重'), c('约 0.8 GB，大块连续'), c('每生成一个 token 完整读一遍；只读'), c('持续带宽；已开的行能否连续命中')],
    [c0('KV 读'), c('上下文长度 × 256 B（K、V 各一份）\nSHORT 1,660 token → 425 KB\nLONG 103.5K token → 26.5 MB'), c('每个请求、每一层各读一段；长度随请求不同'), c('短读：HBF 读页时间摊不薄\n分散存放：HBM 行冲突')],
    [c0('KV 写'), c('每 token 每层 256 B（K、V 各一次）'), c('极小、追加式、每一步都有'), c('HBF 不满一页的写；读写方向切换')],
    [c0('激活 / 中间结果'), c('KB ~ MB 级'), c('读写交替，生命周期短'), c('容量放不放得下；端口争用')],
    [c0('PD 导入的 KV'), c('按 1 MiB 分段的大块写'), c('与 Decode 同时进行'), c('写带宽；与读争用同一批资源')],
  ];
  s.addTable(rows, { x: 0.6, y: 1.72, w: 12.13, colW: [1.75, 3.7, 3.33, 3.35], fontFace: F, color: C.ink,
    border: { type: 'solid', pt: 0.75, color: C.line }, valign: 'middle', margin: [0.06, 0.12, 0.06, 0.12],
    rowH: [0.42, 0.55, 1.05, 0.55, 0.55, 0.55] });
  rect(s, 0.6, 5.85, 12.13, 0.85, C.hbfTint);
  text(s, '关键：同一个数据，单次访问可以从 256 B 到 GB 级；同一时刻还有成百上千个请求在途。存储表现取决于这些访问的组合，而不是其中任何一种。',
    0.85, 5.85, 11.7, 0.85, { size: 14, valign: 'middle', lsm: 1.25 });
  note(s, '这一页把领导关心的“访存很复杂”具体化。数字按 405B、TP8 计算：每卡每层权重约 811.7 GB ÷ 126 层 ÷ 8 卡 ≈ 0.8 GB；KV 每 token 每层每卡 K、V 各 256 B（8 个 KV 头分到 8 张卡，每张 1 个头 × 128 维 × 2 B）。'
    + '仿真器中 KV 的真实布局：每个请求、每一层各有独立的 K 缓冲区和 V 缓冲区，按 token 顺序连续存放（planner.py 的 _generic_kv_owner_contract，layout=separate_token_major_k_and_v）。'
    + '所以 Decode 一步里既有权重的大块连续读，也有每个请求各一段的 KV 读，还有每 token 256 B 的 KV 追加写，以及 PD 导入的大块写。');
}

// ---- 04 pipeline --------------------------------------------------------------
{
  const s = base('03 · 仿真器的统一做法', '每一次访存都走完同一条物理链路，时间由资源排队算出来',
    '代码：planner.py（算子与缓冲区）· memory_allocator.py · memory_types.AccessRequest · memory_mapping.py · dram_core.py / nand_core.py / sram_core.py · event_kernel.py');
  const steps = [
    ['算子', 'Attention、GEMM 等声明要读写哪些缓冲区'],
    ['物理地址', '分配器给出地址：权重、每请求每层的 K/V、激活'],
    ['访存请求', '读/写 · 起始地址 · 字节数 · 到达时间'],
    ['介质核心', '拆成最小单位 → 映射到物理单元 → 查当前状态'],
    ['资源排队', '命令总线、数据线、plane、内部通道、链路、端口按时间表占用'],
    ['回写结果', '完成时间回到事件内核 → 算子结束 → TPOT / TTFT / 吞吐'],
  ];
  steps.forEach(([h, d], i) => {
    const x = 0.6 + i * 2.05;
    const hot = i === 3 || i === 4;
    rect(s, x, 1.95, 1.85, 2.15, hot ? C.hbmTint : C.panel);
    circleNum(s, i + 1, x + 0.15, 2.1, hot ? C.hbmDark : C.navy);
    text(s, h, x + 0.15, 2.58, 1.6, 0.4, { size: 16, bold: true });
    text(s, d, x + 0.15, 3.02, 1.6, 1.0, { size: 11.5, color: C.sub, lsm: 1.2 });
    if (i < 5) arrow(s, x + 1.86, 3.02, x + 2.04);
  });
  text(s, '④⑤ 是本次汇报的重点：HBM、HBF、SRAM 在这两步上建模的层级不同', 0.6, 4.22, 12, 0.3, { size: 12, color: C.hbmDark, bold: true });
  rect(s, 0.6, 4.75, 5.95, 1.6, C.panel);
  text(s, '常见简化', 0.85, 4.92, 5.5, 0.35, { size: 15, bold: true, color: C.sub });
  text(s, '时间 = 字节数 ÷ 峰值带宽（＋固定延迟）\n与地址、单次大小、介质状态、同时在途数都无关', 0.85, 5.35, 5.5, 1.2, { size: 14, color: C.sub, lsm: 1.35 });
  rect(s, 6.78, 4.75, 5.95, 1.6, C.hbmTint);
  text(s, '本仿真器', 7.03, 4.92, 5.5, 0.35, { size: 15, bold: true });
  text(s, '时间 = 沿依赖链、在共享资源时间表上排队得到的完成时刻\n同样的字节数，因访问形态不同可以差几个数量级', 7.03, 5.35, 5.5, 1.2, { size: 14, lsm: 1.35 });
  note(s, '这一页说明仿真器的总体结构，三种介质共用这条链路。'
    + '第 1～3 步：算子声明缓冲区，分配器给出不重叠的物理地址，派发时生成 AccessRequest（操作、地址、字节数、到达时间）。'
    + '第 4 步：介质核心按自己的物理组织拆分请求——HBM 拆成 32 B burst，HBF 拆成 4 KiB 页，SRAM 拆成 64 B burst——再把地址映射到具体的物理单元，查询当前状态（开着哪一行、页缓冲里有哪一页）。'
    + '第 5 步：每一段操作都要在共享资源的时间表上预约，资源忙就排队。第 6 步：完成时间回到事件内核，决定算子何时结束，最终汇总成 Token 指标。'
    + '与“字节数除以带宽”的区别：仿真器得到的是排队与依赖后的完成时刻，所以能反映访问形态的影响。');
}

// ---- 05 hierarchy -----------------------------------------------------------
{
  const s = base('04 · 三种介质的建模层级', '建模层级：HBM 到 32 B burst，HBF 到 4 KiB 页，SRAM 到端口',
    'HBM 组织来自 JEDEC HBM4（JESD270-4A）；HBF 堆/die/plane 数与时延来自论文 Table I；SRAM 为论文 HBF+ 配置的研究参考；代码 hbm_geometry.py · memory_mapping.py');
  const cols = [
    { name: 'HBM', cfg: '论文 HBM4 配置 · 8 堆', fill: C.hbmDark, tint: C.hbmTint, mid: C.hbm, levels: [
      ['堆 Stack', '× 8'], ['通道 Channel', '× 32 / 堆'], ['伪通道 PC', '× 2 / 通道，共 512 个'], ['Bank Group', '× 8 / PC'], ['Bank', '× 8 / 组'], ['行 Row', '1 KiB × 16,384 / bank'], ['Burst（最小搬运单位）', '32 B']],
      attrs: [['状态', '每个 bank 开着哪一行'], ['排队', '命令总线（2 个 PC 共用）、PC 数据线'], ['时序', 'tRCD、tRP、RL、tRRD/tFAW、tCCD、刷新']] },
    { name: 'HBF', cfg: '论文 HBF 配置 · 7 堆', fill: C.hbfDark, tint: C.hbfTint, mid: C.hbf, levels: [
      ['堆 Stack', '× 7'], ['Die', '× 16 / 堆，共 112 个'], ['Plane（独立阵列）', '× 25 / die（共 2,800 个）'], ['Block', '256 页'], ['页 Page（最小读出单位）', '4 KiB']],
      attrs: [['状态', '每 plane 2 个页缓冲（LRU）'], ['排队', 'plane、die 内部通道、主机链路'], ['时序', '读页 1 µs、编程 100 µs、擦除 2 ms']] },
    { name: 'SRAM', cfg: '论文 HBF+ 配置 · 8 × 40 MB', fill: C.sramDark, tint: C.sramTint, mid: C.sram, levels: [
      ['存储池', '320 MB'], ['端口', '× 8，按 64 B 轮流条带'], ['Burst（最小搬运单位）', '64 B']],
      attrs: [['状态', '无行 / 页状态'], ['排队', '8 个端口（共 80 TB/s）、在途额度'], ['时序', '1 ns 流水线时延']] },
  ];
  cols.forEach((col, i) => {
    const x = 0.6 + i * 4.14, w = 3.85;
    rect(s, x, 1.72, w, 0.62, col.fill);
    text(s, col.name, x + 0.2, 1.72, 1.2, 0.62, { size: 20, bold: true, color: C.white, valign: 'middle' });
    text(s, col.cfg, x + 1.25, 1.72, w - 1.4, 0.62, { size: 12, color: C.white, valign: 'middle', align: 'right' });
    col.levels.forEach(([lv, cnt], j) => {
      const y = 2.48 + j * 0.42, last = j === col.levels.length - 1, ind = j * 0.07;
      rect(s, x + ind, y, w - ind, 0.36, last ? col.mid : col.tint);
      text(s, lv, x + ind + 0.12, y, 1.9, 0.36, { size: 11.5, bold: true, valign: 'middle', color: last ? C.white : C.ink });
      text(s, cnt, x + ind + 1.6, y, w - ind - 1.72, 0.36, { size: 11, valign: 'middle', align: 'right', color: last ? C.white : C.sub });
    });
    col.attrs.forEach(([k, v], j) => {
      const y = 5.5 + j * 0.47;
      text(s, k, x, y, 0.55, 0.42, { size: 11.5, bold: true, color: C.sub });
      text(s, v, x + 0.58, y, w - 0.58, 0.42, { size: 11.5 });
    });
  });
  hline(s, 0.6, 5.38, 12.73, C.line, 0.75);
  note(s, '三列分别是三种介质在仿真器里的建模层级，从上到下是地址被逐级分解的顺序，最下面一格是最小搬运单位。'
    + 'HBM：8 堆 × 32 通道 × 2 个伪通道 = 512 条独立数据通路；每个伪通道 8 个 bank group × 8 个 bank；每个 bank 16,384 行、每行 1 KiB；一次 burst 32 B。仿真器记住每个 bank 当前开着哪一行，并模拟 ACT/PRE/RD/WR/刷新命令在共享命令总线上的发令与时序约束。'
    + 'HBF：7 堆 × 16 die × 25 plane = 2,800 个可独立读页的阵列；每个 die 一条 100 GB/s 内部通道，同一 die 的 25 个 plane 共用；最小读出单位是 4 KiB 页。每个 plane 有 2 个页缓冲，命中时不用重新读页。'
    + 'SRAM：没有行和页的概念，只按 64 B burst 轮流分到 8 个端口，端口带宽与在途额度是唯一会排队的资源。'
    + '层级深度不是统一的，而是由介质的物理特性决定。');
}

// ---- 06 KV landing ------------------------------------------------------------
{
  const s = base('05 · 同一段 KV 落到硬件上', '同一段 KV，在 HBM 里被切成 32 B 分散并行，在 HBF 里被整页读出',
    '映射规则：memory_mapping.map_dram_address（32 B 交织到 512 条通路）· map_nand_address（页先轮转 112 个 die，再换 plane）· SramCore（64 B 轮转 8 个端口）');
  text(s, '一个请求、一层的 K 缓冲区：按 token 顺序连续存放，每 token 256 B', 2.1, 1.6, 10, 0.3, { size: 12.5, color: C.sub });
  for (let i = 0; i < 16; i++) {
    rect(s, 2.1 + i * 0.62, 1.95, 0.58, 0.42, i === 0 ? C.navy : C.panel, i === 0 ? C.navy : C.line);
    text(s, 't' + i, 2.1 + i * 0.62, 1.95, 0.58, 0.42, { size: 11, align: 'center', valign: 'middle', color: i === 0 ? C.white : C.sub, bold: i === 0 });
  }
  text(s, '…', 12.05, 1.95, 0.5, 0.42, { size: 14, color: C.sub, valign: 'middle' });
  const rows = [
    { y: 2.75, name: 'HBM', fill: C.hbmDark, cell: C.hbm, n: 8, cw: 0.5, unit: '32 B',
      sub: (k) => 'C' + Math.floor(k / 2) + '·PC' + (k % 2),
      msg: [['t0 的 256 B = 8 个 32 B burst', true], ['按 32 B 交织落到 4 个通道 × 2 个 PC，8 条数据通路同时传输。', false],
        [`冷启动读 t0：${sw.hbm['1 token'].span_ns.toFixed(1)} ns（8 个 bank 各开一次行）`, true]] },
    { y: 3.85, name: 'HBF', fill: C.hbfDark, cell: C.hbf, page: true,
      msg: [['t0 ~ t15 共用一个 4 KiB 页', true], [`读 t0 也要整页读出：${fmt(R.hbf_cache.first_ns)} ns；下一步读 t1 时，该页还在 plane 页缓冲里：${R.hbf_cache.second_same_page_ns.toFixed(2)} ns。`, false],
        ['相邻页依次落到不同 die，再换 plane', true]] },
    { y: 5.65, name: 'SRAM', fill: C.sramDark, cell: C.sram, n: 4, cw: 1.0, unit: '64 B', sub: (k) => '端口 ' + k,
      msg: [['t0 的 256 B = 4 个 64 B burst，轮流分到 4 个端口', true], [`无行/页状态：${sw.sram['1 token'].span_ns.toFixed(1)} ns（1 ns 流水线 + 传输）`, false]] },
  ];
  rows.forEach((r) => {
    chip(s, r.name, 0.6, r.y, 1.25, 0.42, r.fill, 14);
    if (r.page) {
      rect(s, 2.1, r.y, 4.0, 0.42, r.cell);
      text(s, '页 0（4 KiB）= t0 … t15', 2.1, r.y, 4.0, 0.42, { size: 12, bold: true, color: C.white, align: 'center', valign: 'middle' });
      text(s, '→ die 0 · plane 0', 2.1, r.y + 0.47, 4.0, 0.3, { size: 10.5, color: C.sub, align: 'center' });
      const nxt = [['页 1', 'die 1 · plane 0'], ['页 2', 'die 2 · plane 0'], ['页 112', 'die 0 · plane 1']];
      nxt.forEach(([a, b], k) => {
        const x = 2.1 + k * 1.37;
        rect(s, x, r.y + 0.85, 1.27, 0.5, C.hbfTint, C.hbfMid);
        text(s, a + '\n' + b, x, r.y + 0.85, 1.27, 0.5, { size: 9.5, align: 'center', valign: 'middle', lsm: 1.0 });
      });
    } else {
      for (let k = 0; k < r.n; k++) {
        const x = 2.1 + k * r.cw;
        rect(s, x, r.y, r.cw - 0.04, 0.42, r.cell);
        text(s, r.unit, x, r.y, r.cw - 0.04, 0.42, { size: 10, bold: true, color: C.white, align: 'center', valign: 'middle' });
        text(s, r.sub(k), x - 0.05, r.y + 0.47, r.cw + 0.06, 0.28, { size: 9, color: C.sub, align: 'center' });
      }
    }
    const runs = r.msg.map(([t, b], k) => ({ text: t, options: { bold: b, color: b ? C.ink : C.sub, breakLine: k < r.msg.length - 1 } }));
    text(s, runs, 6.55, r.y - 0.05, 6.18, 1.3, { size: 12.5, lsm: 1.3 });
  });
  note(s, '这一页把“KV 访存需求”和“介质最小单位”对上号，直接对应汇报里提到的两个例子。'
    + 'HBM：地址按 32 B 交织到 512 条数据通路（每条是一个伪通道）。一个 token 的 256 B 落到 8 条通路上，可以同时传输；但每条通路上的 bank 都要先开行，所以冷启动是 37.8 ns 而不是几个纳秒。'
    + '如果数据存放得很分散，每次 burst 都可能落到新的行上，开行/关行时间就成了主要开销（下一页细讲）。'
    + 'HBF：16 个 token 才凑满一个 4 KiB 页。只读 1 个 token 时，阵列仍要整页读出（1 µs），这是短读的代价；但读完后该页留在 plane 的页缓冲里，紧接着读同页的下一个 token 只需主机传输。'
    + '相邻的页先轮转到 112 个 die 上，再换下一个 plane，所以大块连续读能同时动用很多 plane。'
    + 'SRAM：只按 64 B 轮流分到端口，没有状态，耗时基本就是 1 ns 流水线加传输。');
}

// ---- 07 HBM timing ------------------------------------------------------------
{
  const s = base('06 · HBM：一次读怎么计时', 'HBM：同样读 32 B，耗时取决于目标行是否已经打开',
    '仿真器实测（DramCore，jedec_hbm 模式）：HBM4 研究参考时序 tCK 0.64 ns、tRCD 14 ns、tRP 15.36 ns、读延迟 20.48 ns；tRRD/tFAW 借用 HBM3E 参考包，非厂商标定');
  const x0 = 2.45, k = 0.0935; // inch per ns
  const X = (ns) => x0 + ns * k;
  const rows = [
    { name: '行命中', sub: '目标行已打开', y: 2.0, total: 22.24, seg: [
      ['cmd', 0.48, 1.12, 'RD'], ['pipe', 1.12, 20.96, '读延迟 RL'], ['data', 20.96, 22.24, '']] },
    { name: '未开行', sub: 'bank 空闲', y: 3.15, total: 36.48, seg: [
      ['cmd', 0, 0.96, 'ACT'], ['wait', 0.96, 14.72, 'tRCD 开行'], ['cmd', 14.72, 15.36, 'RD'], ['pipe', 15.36, 35.2, '读延迟 RL'], ['data', 35.2, 36.48, '']] },
    { name: '行冲突', sub: '开着别的行', y: 4.3, total: 52.32, seg: [
      ['cmd', 0.16, 0.48, 'PRE'], ['wait', 0.48, 15.52, 'tRP 关行'], ['cmd', 15.84, 16.8, 'ACT'], ['wait', 16.8, 30.56, 'tRCD 开行'], ['cmd', 30.56, 31.2, 'RD'], ['pipe', 31.2, 51.04, '读延迟 RL'], ['data', 51.04, 52.32, '']] },
  ];
  const col = { cmd: C.navy, wait: 'D9D9D4', pipe: C.hbmMid, data: C.hbm };
  rows.forEach((r) => {
    text(s, r.name, 0.6, r.y - 0.02, 1.7, 0.32, { size: 15, bold: true });
    text(s, r.sub, 0.6, r.y + 0.3, 1.7, 0.28, { size: 11, color: C.sub });
    rect(s, x0, r.y, X(60) - x0, 0.5, 'FAFAF8');
    r.seg.forEach(([kind, a, b, lab]) => {
      const w = Math.max(X(b) - X(a), 0.05);
      rect(s, X(a), r.y, w, 0.5, col[kind]);
      if (kind === 'cmd') text(s, lab, X(a) - 0.25, r.y - 0.27, w + 0.5, 0.24, { size: 9.5, bold: true, align: 'center', color: C.ink });
      else if (lab) text(s, lab, X(a), r.y, X(b) - X(a), 0.5, { size: 10.5, align: 'center', valign: 'middle', color: C.ink });
    });
    text(s, r.total.toFixed(1) + ' ns', X(r.total) + 0.1, r.y, 1.0, 0.5, { size: 15, bold: true, valign: 'middle' });
  });
  hline(s, x0, 5.0, X(60), C.muted, 0.75);
  for (let t = 0; t <= 60; t += 10) {
    vline(s, X(t), 5.0, 5.07, C.muted, 0.75);
    text(s, t + (t === 60 ? ' ns' : ''), X(t) - 0.3, 5.1, 0.6, 0.25, { size: 9.5, color: C.sub, align: 'center' });
  }
  const leg = [['cmd', '命令（ACT / PRE / RD）'], ['wait', '等待时序约束'], ['pipe', '读流水线'], ['data', '32 B 数据 burst']];
  leg.forEach(([kd, lb], i) => { const x = 0.6 + i * 2.15; rect(s, x, 5.6, 0.22, 0.22, col[kd]); text(s, lb, x + 0.3, 5.56, 1.85, 0.3, { size: 10.5, color: C.sub }); });
  text(s, '仿真器逐个 bank 记住“开着哪一行”：上一个访问留下的行状态，决定下一个访问走哪条路径。', 0.6, 6.15, 7.9, 0.6, { size: 13, bold: true, lsm: 1.25 });
  rect(s, 8.95, 1.72, 3.78, 4.98, C.panel);
  text(s, '此外还要同时满足', 9.18, 1.88, 3.4, 0.35, { size: 14, bold: true });
  const items = [['开行速率', '同一 PC 相邻 ACT ≥ 4 / 6 ns；任意 20 ns 内最多 4 次 ACT'],
    ['列命令间隔', '同 bank group 2.56 ns，跨组 1.28 ns；地址映射让连续 burst 轮换 bank group'],
    ['共享命令总线', '同一物理通道的 2 个 PC 共用行 / 列命令总线'],
    ['刷新', '每 3.9 µs 全 bank 刷新一次，期间该 PC 不可访问'],
    ['其他', '读写换向、写恢复、最长开行时间']];
  let yy = 2.33;
  items.forEach(([a, b]) => {
    text(s, a, 9.18, yy, 3.4, 0.28, { size: 12, bold: true, color: C.hbmDark });
    text(s, b, 9.18, yy + 0.28, 3.4, 0.6, { size: 11, color: C.sub, lsm: 1.15 });
    yy += 0.86;
  });
  note(s, '三条时间轴来自仿真器实际输出的命令阶段（capture_details 模式），按同一比例绘制。'
    + '行命中：只发一条 RD，经过 20.48 ns 读延迟后传 32 B，总 22.2 ns。未开行：先 ACT，等 tRCD 14 ns 才能发 RD，总 36.5 ns。行冲突：先 PRE 关掉旧行、等 tRP，再 ACT、等 tRCD，再 RD，总 52.3 ns，是命中的 2.4 倍。'
    + '这就是“HBM 的访存时间与数据存储位置强相关”的物理来源：位置决定了落在哪个 bank、哪一行，而 bank 当前开着哪一行决定走哪条路径。'
    + '右侧是同时生效的其他约束：开行速率 tRRD/tFAW 限制了“到处开新行”的速度；同组列命令间隔更长，所以地址映射让连续 burst 轮换 bank group；同一通道两个伪通道共用命令总线；周期刷新会阻塞访问。'
    + '代码：_hbm_control.py（命令控制器）、memory_mapping.rotated_bank（bank group 轮换）。');
}

// ---- 08 HBM placement charts ---------------------------------------------------
{
  const s = base('07 · HBM 实测：位置与大小', 'HBM：数据放在哪里、一次读多大，直接决定带宽',
    '左：仿真器探针（冷启动、1 MiB）；右：docs/exploring_hbf_native_comparison.md 记录的 405B 真实地址实测（2026-10-22，提交 add4a5d）');
  const chartOpts = (title) => ({ barDir: 'bar', chartColors: [C.hbm], showValue: true, dataLabelFormatCode: '#,##0',
    dataLabelFontSize: 12, dataLabelColor: C.ink, dataLabelFontFace: F, dataLabelPosition: 'outEnd',
    catAxisLabelFontFace: F, catAxisLabelFontSize: 12, catAxisLabelColor: C.ink, catAxisLineShow: false,
    valAxisHidden: true, valGridLine: { style: 'none' }, catGridLine: { style: 'none' }, barGapWidthPct: 55,
    showTitle: false, showLegend: false, catAxisOrientation: 'maxMin' });
  text(s, '同样读 1 MiB（256 次 × 4 KiB），有效带宽 GB/s', 0.6, 1.7, 5.9, 0.3, { size: 13, bold: true });
  s.addChart(p.ChartType.bar, [{ name: 'HBM', labels: ['相邻存放', '随机分散存放'], values: [hp['256 x 4KiB adjacent'].gb_s, hp['256 x 4KiB scattered'].gb_s] }],
    { x: 0.5, y: 2.05, w: 5.9, h: 2.0, ...chartOpts(), valAxisMaxVal: 3600 });
  text(s, `相邻：${fmt(hp['256 x 4KiB adjacent'].row_misses)} 次开行，0 次冲突\n分散：${fmt(hp['256 x 4KiB scattered'].row_misses)} 次开行 + ${fmt(hp['256 x 4KiB scattered'].row_conflicts)} 次行冲突（先关再开）`, 0.6, 4.1, 5.9, 0.6, { size: 11.5, color: C.sub, lsm: 1.25 });
  text(s, '405B 真实 KV 读（B=64，第 63 层，128 条 × 470 KB），GB/s', 6.83, 1.7, 5.9, 0.3, { size: 13, bold: true });
  s.addChart(p.ChartType.bar, [{ name: 'HBM', labels: ['地址连续', '4 KiB 对齐分散', '实际 KV 地址'], values: [9575, 2494, 2484] }],
    { x: 6.73, y: 2.05, w: 5.9, h: 2.4, ...chartOpts(), valAxisMaxVal: 11500 });
  text(s, '同样的字节数，真实 KV 地址只有连续存放的 26%', 6.83, 4.5, 5.9, 0.3, { size: 11.5, color: C.sub });
  const cards = [
    ['一个 32 B burst 只用到 8 B', `搬运字节不变，有效带宽 ${hp['32768 x 32B scattered'].gb_s.toFixed(1)} → ${hp['32768 x 8B scattered'].gb_s.toFixed(1)} GB/s，只剩 1/4。仿真器按 burst 计搬运量，不按请求字节数。`],
    ['单次读越大，已开的行被复用越多', `1 MiB 冷读每个 bank 只用一次行：${fmt(hp['contiguous 1MiB'].gb_s / 1000, 1)} TB/s；26.5 MB 读 ${fmt(sw.hbm['LONG 103500'].gb_s / 1000, 1)} TB/s，${Math.round(100 * sw.hbm['LONG 103500'].row_hits / sw.hbm['LONG 103500'].burst_count)}% 的 burst 命中已开的行。`],
  ];
  cards.forEach(([h, b], i) => {
    const x = 0.6 + i * 6.23;
    rect(s, x, 5.1, 5.9, 1.6, C.hbmTint);
    text(s, h, x + 0.22, 5.24, 5.5, 0.35, { size: 14, bold: true });
    text(s, b, x + 0.22, 5.65, 5.5, 0.95, { size: 12, color: C.sub, lsm: 1.25 });
  });
  note(s, '左图：仿真器探针。同样 256 次 4 KiB 读，地址相邻时 2.93 TB/s，随机分散时 1.17 TB/s；分散后有 12,288 次行冲突，每次都要先关行再开行，并受开行速率 tRRD/tFAW 限制。'
    + '右图：项目文档记录的真实负载数据，405B、HBM4、SHORT、B=64，第 63 层 rank0 的 128 条 KV 读（每条 470,016 B）。把同样大小的读放在连续地址是 9,575 GB/s，4 KiB 对齐分散是 2,494 GB/s，真实 KV 地址 2,484 GB/s。说明真实推理中 KV 的分散存放确实会把带宽拉低到约四分之一。'
    + '下方两点：其一，burst 是最小搬运单位，若一次 32 B burst 里只有 8 B 有用，搬运量不变，有用带宽就降为四分之一（汇报中“只有 32b 能用”的情况更极端，32 bit 只占 32 B burst 的 1/8）。其二，单次读越大，行被复用越多：26.5 MB 读时 96% 的 burst 命中已开行，达到 10.7 TB/s。');
}

// ---- 09 HBF read path ----------------------------------------------------------
{
  const s = base('08 · HBF：一次读怎么计时', 'HBF：一次读要占用三段资源，最小单位是 4 KiB 页',
    '仿真器 NandCore（generic_nand）：页读 1 µs；内部通道每 die 100 GB/s；主机读 11.2 TB/s；本配置主机传输粒度 4 KiB（OCP HBF 规范允许 64 B，可配置）');
  rect(s, 0.6, 2.0, 1.6, 1.2, C.navy);
  text(s, '读请求', 0.6, 2.0, 1.6, 1.2, { size: 15, bold: true, color: C.white, align: 'center', valign: 'middle' });
  const st = [['① 阵列读页', 'tR = 1 µs', '每个 plane 独立，共 2,800 个可同时工作'],
    ['② 内部通道', '搬整页 4 KiB', '每 die 一条，100 GB/s；同 die 的 25 个 plane 共用'],
    ['③ 主机链路', '搬 4 KiB', '全部堆共 11.2 TB/s']];
  st.forEach(([h, v, d], i) => {
    const x = 2.6 + i * 2.95;
    rect(s, x, 2.0, 2.6, 1.2, C.hbfTint);
    rect(s, x, 2.0, 0.07, 1.2, C.hbf);
    text(s, h, x + 0.2, 2.07, 2.3, 0.32, { size: 13.5, bold: true });
    text(s, v, x + 0.2, 2.38, 2.3, 0.32, { size: 13, bold: true, color: C.hbfDark });
    text(s, d, x + 0.2, 2.7, 2.35, 0.48, { size: 10.5, color: C.sub, lsm: 1.1 });
    arrow(s, x - 0.38, 2.6, x - 0.02);
  });
  arrow(s, 11.45, 2.6, 11.83);
  rect(s, 11.85, 2.0, 0.88, 1.2, C.navy);
  text(s, '完成', 11.85, 2.0, 0.88, 1.2, { size: 14, bold: true, color: C.white, align: 'center', valign: 'middle' });
  // cache-hit bypass
  vline(s, 1.4, 3.2, 3.75, C.hbfDark, 1.25);
  hline(s, 1.4, 3.75, 8.45, C.hbfDark, 1.25, false, true);
  vline(s, 8.45, 3.75, 3.22, C.hbfDark, 1.25, true);
  rect(s, 3.0, 3.5, 4.6, 0.5, C.white, C.hbfMid);
  text(s, `页缓冲命中：跳过 ①②，只走 ③（${R.hbf_cache.second_same_page_ns.toFixed(2)} ns vs ${fmt(R.hbf_cache.first_ns)} ns）`, 3.0, 3.5, 4.6, 0.5, { size: 11.5, align: 'center', valign: 'middle', bold: true });
  const pan = [
    ['读放大', '只要 256 B，也要读出并搬运整页 4 KiB', `16×`, '单个 token 的读：搬运 4,096 B，有用 256 B'],
    ['并行靠“铺页”', '连续的页依次落到 112 个 die，再换 plane', `${(fp['contiguous 1MiB'].span_ns / 1000).toFixed(2)} µs`, '1 MiB 连续读：256 页同时在不同 plane 上读'],
    ['位置也有影响', '页落到同一个 plane，读页只能排队', `${(fp['256 x 4KiB scattered'].span_ns / 1000).toFixed(2)} µs`, '256 个随机页：有的 plane 分到 3 页，只能串行读 3 次'],
  ];
  pan.forEach(([h, d, big, note], i) => {
    const x = 0.6 + i * 4.14;
    rect(s, x, 4.5, 3.85, 2.2, C.panel);
    text(s, h, x + 0.22, 4.63, 3.4, 0.33, { size: 14, bold: true });
    text(s, d, x + 0.22, 4.98, 3.45, 0.5, { size: 11.5, color: C.sub, lsm: 1.15 });
    text(s, big, x + 0.22, 5.5, 3.4, 0.62, { size: 30, bold: true });
    text(s, note, x + 0.22, 6.15, 3.45, 0.45, { size: 11, color: C.sub, lsm: 1.15 });
  });
  note(s, 'HBF 的一次读在仿真器里拆成三段，每段占用不同的共享资源：阵列读页（每个 plane 一个资源，tR = 1 µs）、内部通道搬整页（每个 die 一条 100 GB/s 通道，同 die 的 25 个 plane 共用）、主机链路（所有堆合计 11.2 TB/s）。'
    + '每个 plane 有 2 个页缓冲（LRU 替换），命中时跳过读页与内部传输。实测：读完 t0 后紧接着读同页的 t1，只要 0.37 ns。'
    + '下面三点：一是读放大，只要 256 B 也要整页读出搬运，16 倍；二是并行，连续页依次铺到 112 个 die 再换 plane，所以 1 MiB 连续读 256 页几乎同时完成，约 1.13 µs；三是位置，随机分散的 256 个页里有的 plane 被分到 3 个页，只能串行读 3 次，整体 3.25 µs。'
    + '注意：本配置把主机传输粒度也设成 4 KiB（研究假设）；OCP HBF 规范允许 64 B 读粒度，仿真器可配置。代码：nand_core.py _execute_detailed_accepted。');
}

// ---- 10 HBF size chart -----------------------------------------------------------
{
  const s = base('09 · HBF 实测：读多大', 'HBF：KV 读得越大、越并发，越接近 HBM；短读时劣势最大',
    '仿真器探针：单个冷请求，读一个请求一层的 K（405B、TP8 每 token 256 B）；批量行为 64 个 SHORT 请求同时读各自的 K 缓冲区');
  const cats = ['1 token\n256 B', '16 token\n4 KiB', 'SHORT\n425 KB', 'MID\n1.5 MB', 'LONG\n26.5 MB'];
  const keys = ['1 token', '16 token (4KiB)', 'SHORT 1660', 'MID 5900', 'LONG 103500'];
  text(s, '单次读的有效带宽（GB/s，对数刻度）', 0.6, 1.68, 7.8, 0.3, { size: 13, bold: true });
  s.addChart(p.ChartType.bar, [
    { name: 'HBM', labels: cats, values: keys.map((k) => sw.hbm[k].gb_s) },
    { name: 'HBF', labels: cats, values: keys.map((k) => sw.hbf[k].gb_s) },
  ], { x: 0.5, y: 1.98, w: 7.9, h: 4.7, barDir: 'col', barGrouping: 'clustered', chartColors: [C.hbm, C.hbf],
    valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisMaxVal: 100000, showValue: true,
    dataLabelFormatCode: '[<1]0.00;[<100]0.0;#,##0', dataLabelFontSize: 10, dataLabelColor: C.ink, dataLabelFontFace: F, dataLabelPosition: 'outEnd',
    catAxisLabelFontFace: F, catAxisLabelFontSize: 11, catAxisLabelColor: C.ink, valAxisLabelFontSize: 10, valAxisLabelColor: C.sub,
    valAxisLabelFormatCode: '[<1]0.0;#,##0', valAxisCrossesAt: 0.1, valGridLine: { color: 'E8E8E4', size: 0.5 }, catGridLine: { style: 'none' },
    barGapWidthPct: 70, showLegend: true, legendPos: 't', legendFontFace: F, legendFontSize: 12 });
  rect(s, 8.75, 1.72, 3.98, 2.95, C.panel);
  text(s, 'HBF 耗时是 HBM 的几倍', 8.97, 1.85, 3.6, 0.33, { size: 13.5, bold: true });
  const names = ['1 token（256 B）', '16 token（4 KiB）', 'SHORT（425 KB）', 'MID（1.5 MB）', 'LONG（26.5 MB）'];
  keys.forEach((k, i) => {
    const y = 2.27 + i * 0.46, r = sw.hbf[k].span_ns / sw.hbm[k].span_ns;
    text(s, names[i], 8.97, y, 2.3, 0.4, { size: 12, valign: 'middle', color: C.sub });
    text(s, (r >= 10 ? r.toFixed(1) : r.toFixed(2)) + '×', 11.0, y, 1.5, 0.4, { size: 16, bold: true, align: 'right', valign: 'middle' });
  });
  const b = R.batch64_short;
  rect(s, 8.75, 4.82, 3.98, 1.88, C.hbfTint);
  text(s, '批量也能把读“凑大”', 8.97, 4.94, 3.6, 0.33, { size: 13.5, bold: true });
  text(s, `64 个 SHORT 请求同时读：HBF ${fmt(b.hbf.gb_s / 1000, 2)} TB/s，HBM ${fmt(b.hbm.gb_s / 1000, 2)} TB/s，只差 ${(b.hbf.span_ns / b.hbm.span_ns).toFixed(2)}×。`, 8.97, 5.3, 3.6, 0.7, { size: 12, lsm: 1.25 });
  text(s, '劣势的本质是“同一时刻在途的页太少，1 µs 读页时间摊不薄”。', 8.97, 6.02, 3.6, 0.6, { size: 12, bold: true, lsm: 1.2 });
  note(s, '这一页回答“短上下文 HBF 一次读 4 KiB 弊端大、长上下文收益大”。'
    + `单个请求时：读 1 个 token，HBF 比 HBM 慢 ${(sw.hbf['1 token'].span_ns / sw.hbm['1 token'].span_ns).toFixed(1)} 倍；SHORT 上下文（425 KB）慢 ${(sw.hbf['SHORT 1660'].span_ns / sw.hbm['SHORT 1660'].span_ns).toFixed(1)} 倍；LONG 上下文（26.5 MB）只慢 ${(sw.hbf['LONG 103500'].span_ns / sw.hbm['LONG 103500'].span_ns).toFixed(2)} 倍，HBF 达到 ${fmt(sw.hbf['LONG 103500'].gb_s)} GB/s。`
    + '原因：大读铺到上千个 plane 上同时读页，1 µs 的读页时间被摊薄；小读只用到一两个 plane，1 µs 无法摊薄。'
    + `但这并不只由上下文长度决定：64 个 SHORT 请求同时读时，HBF 也能达到 ${fmt(b.hbf.gb_s)} GB/s，与 HBM 只差 ${(b.hbf.span_ns / b.hbm.span_ns).toFixed(2)} 倍。`
    + '所以判断 HBF 的优劣要看“同一时刻在途的页数”，它由上下文长度、批量、层间依赖（前一层算完才读下一层的 KV）共同决定。简单公式做不到，只能逐请求、逐 plane 仿真。');
}

// ---- 11 HBF write ---------------------------------------------------------------
{
  const s = base('10 · HBF 写入', 'HBF：每 token 的小写入代价最大，“写合并”作为显式策略建模',
    '仿真器探针（单个冷请求）：页编程 100 µs、读改写策略；主机写暂存见 docs/hbf_host_staging.md（严格 OCP HBF 预设，可选）');
  const w = R.hbf_write;
  const bars = [
    ['HBM 写 256 B（1 个 token）', '', R.hbm_write['1 token 256B'].span_ns, C.hbm, `${R.hbm_write['1 token 256B'].span_ns.toFixed(1)} ns`],
    ['HBF 写 256 B（1 个 token）', '读旧页 1 µs + 整页编程 100 µs', w['1 token 256B'].span_ns, C.hbf, `${(w['1 token 256B'].span_ns / 1000).toFixed(1)} µs`],
    ['HBF 写 4 KiB（整页）', '直接整页编程', w['full 4KiB page'].span_ns, C.hbf, `${(w['full 4KiB page'].span_ns / 1000).toFixed(1)} µs`],
    ['HBF 写 1 MiB（256 页）', '256 页分到不同 plane 并行编程', w['1 MiB'].span_ns, C.hbf, `${(w['1 MiB'].span_ns / 1000).toFixed(1)} µs`],
  ];
  const x0 = 3.7, kx = 3.4 / 110000;
  text(s, '完成时间（线性刻度，0 ~ 110 µs）', 0.6, 1.7, 7, 0.3, { size: 13, bold: true });
  bars.forEach(([lab, sub, ns, color, val], i) => {
    const y = 2.2 + i * 0.78, bw = Math.max(ns * kx, 0.02);
    text(s, lab, 0.6, y, 3.0, 0.3, { size: 12.5, bold: true });
    if (sub) text(s, sub, 0.6, y + 0.3, 3.0, 0.26, { size: 10.5, color: C.sub });
    rect(s, x0, y + 0.08, bw, 0.38, color);
    text(s, val, x0 + bw + 0.12, y + 0.02, 1.2, 0.5, { size: 13, bold: true, valign: 'middle' });
  });
  vline(s, x0, 2.15, 5.3, C.muted, 0.75);
  rect(s, 0.6, 5.55, 7.3, 1.15, C.hbfTint);
  text(s, `Decode 每生成 1 个 token，每层要给 K、V 各追加 256 B。如果直接写入 HBF，每次都要付一次约 100 µs 的整页编程，比 HBM 慢约 ${fmt(Math.round(w['1 token 256B'].span_ns / R.hbm_write['1 token 256B'].span_ns / 100) * 100)} 倍。`,
    0.82, 5.55, 6.9, 1.15, { size: 12.5, valign: 'middle', lsm: 1.3 });
  rect(s, 8.2, 1.72, 4.53, 4.98, C.panel);
  text(s, '仿真器提供的写合并策略', 8.42, 1.88, 4.1, 0.35, { size: 14, bold: true });
  const steps = [['不满一页先暂存', 'KV 追加先写进主机 DRAM，经真实链路传输并占用暂存容量'],
    ['写满再整页写入', '凑满 4 KiB 后读出暂存页，整页编程到 HBF'],
    ['读时两处合并', 'Attention 同时读 HBF 中已写入的部分和主机 DRAM 里的尾页'],
    ['成本全部计入', '暂存容量、链路、读改写、空间回收与擦除都按物理操作计费']];
  steps.forEach(([h, d], i) => {
    const y = 2.4 + i * 1.03;
    circleNum(s, i + 1, 8.42, y, C.hbfDark, 0.32);
    text(s, h, 8.86, y - 0.02, 3.7, 0.32, { size: 12.5, bold: true });
    text(s, d, 8.86, y + 0.3, 3.7, 0.65, { size: 11, color: C.sub, lsm: 1.2 });
  });
  note(s, 'HBF 的写比读更受访问形态影响。仿真器实测：写 256 B（一个 token 的 K 或 V）要 101 µs，因为不满一页的写需要读出旧页（1 µs）、合并后整页编程（100 µs）；写一整页 4 KiB 也是约 100 µs；写 1 MiB 的 256 页分到不同 plane 并行编程，只要 109 µs。HBM 写 256 B 只要 22 ns。'
    + '所以如果把每 token 的 KV 追加直接写到 HBF，会严重拖慢 Decode。仿真器把“主机侧写合并”作为显式、可选的策略来建模（严格 OCP HBF 预设下的 host_write_staging）：'
    + '不满一页的写先经链路写入主机 DRAM，凑满一页再整页写入 HBF；读的时候同时读 HBF 和主机 DRAM 的尾页；暂存容量、链路、读改写、空间回收与擦除全部按物理操作计费，不会把“主机收到”当成“介质写完”。'
    + '这类策略是否划算，取决于批量、上下文长度和主机带宽，也只能靠仿真回答。');
}

// ---- 12 SRAM --------------------------------------------------------------------
{
  const s = base('11 · SRAM', 'SRAM：物理上没有行和页，模型只保留真正起作用的容量与端口',
    '仿真器 SramCore；参数为论文 HBF+ 配置的研究参考（8 堆 × 40 MB，总 80 TB/s，1 ns 流水线，256 在途额度）；GPU 片上 L1/L2 属于算子模型，不在本页范围');
  rect(s, 0.6, 1.72, 5.6, 4.98, C.panel);
  text(s, '模型保留的要素', 0.82, 1.88, 5.2, 0.35, { size: 14, bold: true });
  const it = [['容量', '320 MB，存放中间数据（激活、工作区）；放不下就要退回别的介质'],
    ['端口', '8 个端口，地址按 64 B 轮流条带；每端口 10 TB/s，读写共用'],
    ['流水线时延', '1 ns，先于数据传输；不按字节串行累加'],
    ['在途额度', '最多 256 个请求同时在途，超出就排队'],
    ['不建的部分', '无行/页状态，不因地址位置产生额外惩罚']];
  it.forEach(([h, d], i) => {
    const y = 2.38 + i * 0.84;
    rect(s, 0.82, y + 0.05, 0.07, 0.62, i === 4 ? C.muted : C.sram);
    text(s, h, 1.03, y, 4.9, 0.32, { size: 12.5, bold: true });
    text(s, d, 1.03, y + 0.32, 4.95, 0.45, { size: 11, color: C.sub, lsm: 1.15 });
  });
  const tiles = [
    ['读 1 个 token（256 B）', sw.sram['1 token'].span_ns.toFixed(1) + ' ns', '基本只有 1 ns 流水线时延'],
    ['读 26.5 MB（LONG 上下文的 K）', fmt(sw.sram['LONG 103500'].gb_s / 1000, 1) + ' TB/s', '已接近 80 TB/s 峰值，大读不受位置影响'],
    ['32,768 次分散的 32 B 读', '搬运 2×', '每次至少搬一个 64 B burst；小请求还受 256 在途额度限制'],
  ];
  tiles.forEach(([h, big, d], i) => {
    const y = 1.72 + i * 1.7;
    rect(s, 6.5, y, 6.23, 1.5, C.sramTint);
    text(s, h, 6.75, y + 0.13, 5.8, 0.32, { size: 13, bold: true });
    text(s, big, 6.75, y + 0.48, 2.6, 0.7, { size: 28, bold: true, valign: 'middle' });
    text(s, d, 9.35, y + 0.5, 3.2, 0.75, { size: 11.5, color: C.sub, valign: 'middle', lsm: 1.2 });
  });
  note(s, 'SRAM 是三种介质里建模最简单的，这是有意为之：SRAM 没有 DRAM 的行激活、也没有 NAND 的整页读出，访问时间几乎只由端口带宽和流水线时延决定。'
    + '仿真器保留的是真正影响结果的部分：容量（中间数据放不放得下）、8 个端口的条带与争用、64 B 的最小搬运单位、1 ns 时延、256 个在途额度。'
    + `实测：读 256 B 只要 ${sw.sram['1 token'].span_ns.toFixed(2)} ns；读 26.5 MB 达到 ${fmt(sw.sram['LONG 103500'].gb_s)} GB/s，接近 80 TB/s 峰值；分散的 32 B 小读因为 64 B burst，搬运量翻倍。`
    + '这一页说明建模深度的原则：按介质的物理决定，SRAM 加行/页状态不会提高准确性，只会增加仿真开销。'
    + '这里的 SRAM 指论文 HBF+ 配置中逻辑 die 上的中间数据 SRAM，GPU 自己的 L1/L2 在算子模型里处理。代码：sram_core.py，配置 scripts/exploring_hbf_native_config.py 的 sram_reference。');
}

// ---- 13 naive vs simulated -----------------------------------------------------------
{
  const s = base('12 · 为什么必须精细建模', '按峰值带宽折算，小读写与分散访问会被低估 10 倍到数万倍',
    '“按峰值折算” = 字节数 ÷ 标称峰值（HBM 12.8 TB/s；HBF 读 11.2 TB/s、写 112 GB/s；SRAM 80 TB/s）；“仿真器” = 同一场景的探针实测');
  const rowsData = [
    ['SRAM 读 26.5 MB', 26496000 / 80000, sw.sram['LONG 103500'].span_ns],
    ['HBM 连续读 26.5 MB', 26496000 / 12800, sw.hbm['LONG 103500'].span_ns],
    ['HBF 连续读 26.5 MB', 26496000 / 11200, sw.hbf['LONG 103500'].span_ns],
    ['HBM 读 1 MiB（256 × 4 KiB 分散）', 1048576 / 12800, hp['256 x 4KiB scattered'].span_ns],
    ['HBF 读 1 个 token（256 B）', 256 / 11200, sw.hbf['1 token'].span_ns],
    ['HBF 写 1 个 token（256 B）', 256 / 112, R.hbf_write['1 token 256B'].span_ns],
  ];
  const t = (ns) => ns < 1 ? ns.toFixed(2) + ' ns' : ns < 10000 ? fmt(ns, 0) + ' ns' : fmt(ns / 1000, 1) + ' µs';
  const hd = (x) => ({ text: x, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 13, align: 'center' } });
  const rows = [[hd('场景'), hd('按峰值折算'), hd('仿真器结果'), hd('相差')]];
  rowsData.forEach(([n, a, b], i) => {
    const r = b / a, big = r >= 5;
    const ratio = r >= 1000 ? '约 ' + fmt(r / 10000, 1) + ' 万×' : r.toFixed(1) + '×';
    const f = big ? { fill: { color: C.hbfTint } } : {};
    rows.push([{ text: n, options: { fontSize: 13, ...f } }, { text: t(a), options: { fontSize: 13, align: 'right', ...f } },
      { text: t(b), options: { fontSize: 13, align: 'right', ...f } }, { text: ratio, options: { fontSize: 14, bold: true, align: 'right', color: big ? C.warn : C.ink, ...f } }]);
  });
  s.addTable(rows, { x: 0.6, y: 1.72, w: 12.13, colW: [4.9, 2.4, 2.4, 2.43], fontFace: F, color: C.ink,
    border: { type: 'solid', pt: 0.75, color: C.line }, valign: 'middle', margin: [0.05, 0.18, 0.05, 0.18], rowH: 0.47 });
  rect(s, 0.6, 5.3, 12.13, 1.4, C.hbmTint);
  rect(s, 0.6, 5.3, 0.08, 1.4, C.hbm);
  text(s, '细节会改变端到端结论', 0.9, 5.42, 11.6, 0.35, { size: 14.5, bold: true });
  text(s, '405B、HBM4、SHORT、B=512 的 Decode：仅修正 HBM 地址映射（连续 burst 轮换 bank group）和行命令调度（提前关行），每层由 1.275 ms 降到 0.437 ms，TPOT 约 160 ms → 55 ms。同一份负载，层级细节建错，结论可以差 3 倍。',
    0.9, 5.8, 11.6, 0.85, { size: 12.5, color: C.sub, lsm: 1.3 });
  note(s, '这一页是“为什么要精细建模”的量化回答。左两列分别是按峰值带宽折算和仿真器实测。'
    + '大块连续访问时简单模型误差不大：SRAM 几乎一致，HBM 1.2 倍、HBF 1.5 倍。但大模型推理里充满小读、小写和分散访问：HBM 分散读差 10.9 倍；HBF 读 1 个 token 差 4 万多倍（峰值折算 0.02 ns，实际要 1 µs 读页）；HBF 写 1 个 token 差 4 万多倍。'
    + '底部是项目中的真实例子（docs/exploring_hbf_native_comparison.md，2026-10-22，提交 add4a5d）：405B HBM4 B=512，原地址映射让连续 burst 落在同一个 bank group、行命令不能提前关行，修正后每层 1.275 → 0.437 ms，TPOT 约 160 → 55 ms。'
    + '说明这些层级细节是一阶因素，会直接改变 HBM 与 HBF 的对比结论。');
}

// ---- 14 boundaries -------------------------------------------------------------------
{
  const s = base('13 · 建模深度与边界', '当前建模深度与边界：机制已经实现，参数尚未经器件标定',
    '依据：hbm_geometry.py / docs/hbm_geometry.md · nand_core.py 类说明 · sram_core.py · scripts/exploring_hbf_paper_config.py；所有配置 vendor_calibrated = False');
  const hd = (x) => ({ text: x, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 13 } });
  const m = (x, fill) => ({ text: x, options: { bold: true, color: C.white, fill: { color: fill }, fontSize: 15, align: 'center' } });
  const c = (x) => ({ text: x, options: { fontSize: 11.5 } });
  const rows = [
    [hd('介质'), hd('已经建模'), hd('参数来源'), hd('未建模 / 使用时注意')],
    [m('HBM', C.hbmDark), c('地址映射到堆 / 通道 / PC / bank / 行 / 列；ACT、PRE、RD、WR、全 bank 刷新命令；每 bank 行状态；共享命令总线与数据线；tRRD / tFAW / tCCD / 换向'),
      c('组织来自 JEDEC HBM4 协议；时序为参考工程值与研究假设'), c('厂商地址映射与速度档、PHY 训练、按 bank 刷新；调度按请求而非逐命令重排')],
    [m('HBF', C.hbfDark), c('页级读 / 编程 / 擦除；plane 并行、die 内部通道、主机链路；每 plane 2 页缓冲；不满页读改写；在途额度；可选主机写暂存'),
      c('堆 / die / plane 数与读写时延来自论文 Table I；块几何、擦除、内部带宽为研究假设'), c('论文配置（generic NAND）无 FTL / 垃圾回收；ECC、数据保持未建；磨损为可选简化模型')],
    [m('SRAM', C.sramDark), c('容量、8 端口条带、64 B burst、1 ns 流水线、在途额度'),
      c('容量来自论文；端口数、时延为研究假设'), c('按池化处理，未区分各堆私有空间与路由')],
  ];
  s.addTable(rows, { x: 0.6, y: 1.72, w: 12.13, colW: [1.25, 4.6, 3.0, 3.28], fontFace: F, color: C.ink,
    border: { type: 'solid', pt: 0.75, color: C.line }, valign: 'middle', margin: [0.08, 0.14, 0.08, 0.14], rowH: [0.45, 1.15, 1.15, 0.75] });
  rect(s, 0.6, 5.55, 12.13, 1.15, C.panel);
  text(s, [{ text: '结果应这样读：', options: { bold: true, breakLine: true } },
    { text: '同一套物理机制下比较不同介质与部署是可靠的；绝对数值取决于未标定参数。拿到目标器件的时序表后，替换参数即可，不需要改模型结构。', options: { color: C.sub } }],
    0.85, 5.55, 11.7, 1.15, { size: 13, valign: 'middle', lsm: 1.3 });
  note(s, '最后说明建模边界，避免过度解读。'
    + 'HBM：已实现 JEDEC 正常操作命令子集（ACT、PRE、RD、WR、全 bank 刷新）和 bank 行状态，几何来自 JEDEC 协议原文；时序是参考工程值和研究假设，不是厂商速度档。未实现训练、PHY 细节、按 bank 刷新，调度器按请求窗口重排而不是逐命令 FR-FCFS。'
    + 'HBF：页级介质操作、plane 并行、内部通道、主机链路和页缓冲都已建模；论文配置使用通用 NAND 模式，没有 FTL 和垃圾回收；ECC、数据保持未建。'
    + 'SRAM：按池化的端口模型，没有区分各堆私有空间。'
    + '因此：在同一套物理机制下比较 HBM、HBF、SRAM 的相对优劣是可靠的；绝对数值取决于参数。拿到目标器件的时序表后，只需替换参数。');
}

const out = path.join(__dirname, '大模型推理访存与存储建模层级.pptx');
p.writeFile({ fileName: out }).then(() => console.log('written', out, page, 'slides'));
