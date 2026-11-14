const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');

// Cross-check ratings against docs/llm-simulator-comparison-20261110.html at build time
const htmlPath = path.resolve(__dirname, '../../llm-simulator-comparison-20261110.html');
const htmlSrc = fs.readFileSync(htmlPath, 'utf8');

const p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; // 13.3333" x 7.5"
p.author = 'HeteroLLM-Sim';
p.company = 'HeteroLLM-Sim';
p.subject = '14 个 LLM 与内存仿真器能力矩阵与实测对齐总览';
p.title = '14 个 LLM 与内存仿真器能力矩阵对比';
p.lang = 'zh-CN';
p.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };

const C = {
  paper: 'FFFFFF',
  navy: '1B263B',
  navySub: '263552',
  ink: '0F172A',
  ink2: '334155',
  muted: '64748B',
  na: '94A3B8',
  line: 'CBD5E1',
  lineStrong: '94A3B8',
  zebra: 'F8FAFC',
  grpBg: 'F1F5F9',
  mineBg: 'FEF9EC',
  mineCatBg: 'FDEFD2',
  mineInk: '9A5B00',
  good: '1D63C8',
  goodBg: 'EFF6FF',
  part: '2563EB',
  partCal: 'B45309',
  partCalBg: 'FFFBEB',
  bad: 'C83232',
  badBg: 'FEF2F2',
  bannerBg: 'EEF4FB',
  bannerLine: 'BFDBFE'
};

const SYM = { S: '●', P: '◐', N: '✕', NA: '—' };
const COL_KEYS = ['batch', 'kv', 'pd', 'tp', 'pp', 'ep', 'link', 'hbm', 'nand', 'ftl'];

// 5 groups with concise PPT display labels
const GROUPS = [
  { g: 0, label: '跨层\n系统', span: 1 },
  { g: 1, label: '服务与\n集群仿真', span: 4 },
  { g: 2, label: 'LLM +\n内存 / HBF', span: 3 },
  { g: 3, label: '硬件\n设计空间', span: 2 },
  { g: 4, label: '单一组件\n(外接前端)', span: 4 }
];

const ROWS = [
  {
    g: 0, mine: true, name: 'HeteroLLM-sim 4.0.0', org: ['本项目', '独立自研'],
    st: { batch: 'S', kv: 'S', pd: 'P', tp: 'S', pp: 'S', ep: 'S', link: 'S', hbm: 'S', nand: 'S', ftl: 'N' },
    cal: { st: 'N', scope: '无实机对照', hw: '—（未对照真实硬件）', mdl: '仅内部守恒与时序敏感性', err1: '无实机误差', err2: '10% 为采样预测预算' }
  },
  {
    g: 1, name: 'AISimulate', org: ['NVIDIA', '企业 · ai-dynamo'],
    st: { batch: 'S', kv: 'S', pd: 'P', tp: 'S', pp: 'P', ep: 'P', link: 'P', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'S', scope: '单机端到端', hw: '8×H100–B300 / 4×GB200', mdl: 'vLLM / SGLang · 14 族模型', err1: 'TPOT 14.9% MAPE', err2: '甜区 3.35% · 吞吐 25.5%' }
  },
  {
    g: 1, name: 'Vidur', org: ['Microsoft Research', '企业+高校 · 佐治亚理工'],
    st: { batch: 'S', kv: 'S', pd: 'N', tp: 'S', pp: 'S', ep: 'N', link: 'P', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'S', scope: '端到端服务', hw: 'A100-80G / H100 (1–4卡)', mdl: 'LLaMA2-7/70B · Qwen-72B', err1: '动态延迟 < 9%', err2: '静态 P95 < 3.5%' }
  },
  {
    g: 1, name: 'LLMServingSim 2.0', org: ['KAIST', '高校 · 韩国科学技术院'],
    st: { batch: 'S', kv: 'S', pd: 'S', tp: 'S', pp: 'S', ep: 'S', link: 'S', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'S', scope: 'vLLM 端到端', hw: 'RTX 4090 / PRO 6000', mdl: 'vLLM · Llama-8B / Qwen3', err1: '4 部署均 ≤ 1.2%', err2: '60 项 TTFT / TPOT 指标' }
  },
  {
    g: 1, name: 'SimAI 1.7', org: ['Alibaba Cloud', '企业+高校 · 联合清华等'],
    st: { batch: 'S', kv: 'S', pd: 'P', tp: 'S', pp: 'S', ep: 'P', link: 'S', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'P', scope: '仅训练集群', hw: 'A100 / H100 大规模集群', mdl: 'Megatron GPT-3 / LLaMA', err1: '训练耗时 ~1.9%', err2: '1.5+ 推理模块未公布' }
  },
  {
    g: 2, name: 'HBFSim · FlashLabAE', org: ['MBZUAI', '高校 · 阿布扎比 AI 大学'],
    st: { batch: 'S', kv: 'S', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'P', hbm: 'N', nand: 'S', ftl: 'S' },
    cal: { st: 'P', scope: '计算与卸载', hw: 'A100 / H200 + NVMe', mdl: 'Llama-8B/70B · 16M DMA', err1: '整层 5.0%–5.8%', err2: '整批 5.8% · 卸载 1.34%' }
  },
  {
    g: 2, name: 'HBFSim · SlugLab', org: ['UC Santa Cruz 等', '高校 · 联合上科大/中科大'],
    st: { batch: 'P', kv: 'P', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'P', hbm: 'NA', nand: 'P', ftl: 'P' },
    cal: { st: 'P', scope: '注入微基准', hw: 'RTX PRO 6000 + PCIe5', mdl: 'PTX 微基准 · Qwen3-30B', err1: '注入准度 0.152%', err2: '非真实 HBF 器件误差' }
  },
  {
    g: 2, name: 'LLMSimulator / Duplex', org: ['Seoul National Univ.', '高校 · 首尔大学'],
    st: { batch: 'S', kv: 'P', pd: 'P', tp: 'S', pp: 'N', ep: 'S', link: 'P', hbm: 'S', nand: 'NA', ftl: 'NA' },
    cal: { st: 'N', scope: '无实机对照', hw: '—（仅模拟 GPU+PIM）', mdl: 'Mixtral / GPT 仿真内对比', err1: '无实机误差', err2: '仅相对基线加速比' }
  },
  {
    g: 3, name: 'LLMCompass', org: ['Princeton Univ.', '高校 · 普林斯顿大学'],
    st: { batch: 'N', kv: 'P', pd: 'N', tp: 'S', pp: 'N', ep: 'N', link: 'N', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'S', scope: '算子与单层', hw: 'A100×4 / MI210 / TPUv3', mdl: 'GPT-3 175B 单层 + 算子', err1: 'P 0.69% / D 7.5%', err2: '独立原语算子 10.9%' }
  },
  {
    g: 3, name: 'GenZ / ECHELON', org: ['Georgia Tech', '高校+企业 · Intel / Meta'],
    st: { batch: 'N', kv: 'P', pd: 'N', tp: 'S', pp: 'P', ep: 'S', link: 'N', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'S', scope: '标定后阶段', hw: 'H100 / A100 / MI300X 等', mdl: 'vLLM + Llama2-7B / 13B', err1: 'TTFT 5.88%', err2: 'TPOT 3.86% · 几何均值' }
  },
  {
    g: 4, name: 'Ramulator 2.1', org: ['ETH Zürich', '高校 · 苏黎世联邦理工'],
    st: { batch: 'NA', kv: 'NA', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'NA', hbm: 'S', nand: 'NA', ftl: 'NA' },
    cal: { st: 'P', scope: 'Verilog 契约', hw: 'Micron DDR4 Verilog', mdl: '合成访存 + JEDEC 回归', err1: '命令门控零违例', err2: '无统一实物百分比' }
  },
  {
    g: 4, name: 'DRAMsim3', org: ['Univ. of Maryland', '高校 · 马里兰大学'],
    st: { batch: 'NA', kv: 'NA', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'NA', hbm: 'S', nand: 'NA', ftl: 'NA' },
    cal: { st: 'P', scope: 'Verilog 对照', hw: 'DDR3/4/LPDDR Verilog', mdl: 'cmd.trace + CPU 访存', err1: '周期协议零违例', err2: '无统一实物百分比' }
  },
  {
    g: 4, name: 'ASTRA-sim 2.x', org: ['Georgia Tech 等', '高校+企业 · Meta / Intel'],
    st: { batch: 'NA', kv: 'NA', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'S', hbm: 'NA', nand: 'NA', ftl: 'NA' },
    cal: { st: 'P', scope: '集合通信基准', hw: 'HGX-H100 (2/4/8卡)', mdl: 'NCCL Ring All-Reduce', err1: '8卡 9.69% 误差', err2: '4卡 12.0% / 2卡 20.6%' }
  },
  {
    g: 4, name: 'MQSim', org: ['ETH Zürich & CMU', '高校 · 苏黎世理工 / CMU'],
    st: { batch: 'NA', kv: 'NA', pd: 'NA', tp: 'NA', pp: 'NA', ep: 'NA', link: 'NA', hbm: 'NA', nand: 'S', ftl: 'S' },
    cal: { st: 'S', scope: '商用 SSD 实测', hw: '4 款商用 NVMe/SATA SSD', mdl: '稳态多队列块级 I/O', err1: '响应偏差 6%–18%', err2: 'FAST 2018 实测对照' }
  }
];

// Verify all 14 rows' 10 capability ratings & cal.st match HTML source
for (const r of ROWS) {
  const escName = r.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const blockMatch = htmlSrc.match(new RegExp(`name:'${escName}'[\\s\\S]*?links:\\[`));
  if (!blockMatch) throw new Error('Tool not found in HTML: ' + r.name);
  const block = blockMatch[0];
  for (const k of COL_KEYS) {
    const m = block.match(new RegExp(`\\b${k}:'([A-Z]+)'`));
    const expected = m ? m[1] : 'NA';
    if (r.st[k] !== expected) {
      throw new Error(`Rating mismatch on ${r.name}.${k}: ppt=${r.st[k]} vs html=${expected}`);
    }
  }
  const cm = block.match(/"st":\s*"([SPN])"/);
  if (!cm || cm[1] !== r.cal.st) {
    throw new Error(`Cal status mismatch on ${r.name}: ppt=${r.cal.st} vs html=${cm && cm[1]}`);
  }
}

const s = p.addSlide();
s.background = { color: C.paper };
const S = p.ShapeType;

// ---- Header Area (y = 0.22" .. 0.84") ----
s.addShape(S.rect, {
  x: 0.34, y: 0.25, w: 0.075, h: 0.44,
  fill: { color: C.good }, line: { color: C.good, width: 0 }
});
s.addText('14 个 LLM 与内存仿真器能力矩阵对比', {
  x: 0.49, y: 0.22, w: 7.4, h: 0.34,
  fontFace: 'Microsoft YaHei', fontSize: 19.5, bold: true, color: C.ink, margin: 0, valign: 'mid'
});
s.addText('统一判准：仅当修改该机制参数会直接改变仿真结果时记 ●；实测校准按各工具官方声明拆分列出', {
  x: 0.49, y: 0.56, w: 7.9, h: 0.22,
  fontFace: 'Microsoft YaHei', fontSize: 9.5, color: C.muted, margin: 0, valign: 'mid'
});

// Top-right compact legend pills
const legendItems = [
  { sym: '●', lbl: '支持 ', desc: '已实现且参数生效', symSz: 10.5, fg: C.good, bg: C.goodBg, bd: 'BFDBFE', w: 1.52 },
  { sym: '◐', lbl: '部分 ', desc: '缺关键环节', symSz: 8.6, fg: C.part, bg: 'F8FAFC', bd: '93C5FD', w: 1.24 },
  { sym: '✕', lbl: '不提供 ', desc: '属范围但未做', symSz: 8.6, fg: C.bad, bg: C.badBg, bd: 'FECACA', w: 1.38 },
  { sym: '—', lbl: '不适用 ', desc: '非设计范围', symSz: 8.8, fg: C.muted, bg: 'F1F5F9', bd: C.line, w: 1.26 }
];
let lx = 12.9933 - legendItems.reduce((a, b) => a + b.w, 0) - (legendItems.length - 1) * 0.08;
for (const it of legendItems) {
  s.addShape(S.roundRect, {
    x: lx, y: 0.31, w: it.w, h: 0.34, rectRadius: 0.08,
    fill: { color: it.bg }, line: { color: it.bd, width: 0.8 }
  });
  s.addText([
    { text: it.sym + ' ', options: { bold: true, color: it.fg, fontSize: it.symSz } },
    { text: it.lbl, options: { bold: true, color: it.fg, fontSize: 8.6 } },
    { text: it.desc, options: { color: C.ink2, fontSize: 8.0 } }
  ], {
    x: lx + 0.04, y: 0.31, w: it.w - 0.08, h: 0.34,
    fontFace: 'Microsoft YaHei', align: 'center', valign: 'mid', margin: 0
  });
  lx += it.w + 0.08;
}

// ---- Capability Matrix Table (16 columns, 2 header rows + 14 tool rows) ----
// Total width = 12.6533" (x = 0.34" .. 12.9933")
const colW = [
  0.72,   // 0: 分类
  1.30,   // 1: 所属组织
  1.62,   // 2: 仿真器
  0.69,   // 3: 连续批处理
  0.55,   // 4: KV 容量
  0.55,   // 5: PD 分离
  0.38,   // 6: TP
  0.38,   // 7: PP
  0.38,   // 8: EP
  0.58,   // 9: 链路争用
  0.64,   // 10: HBM 命令
  0.70,   // 11: NAND 时序
  0.58,   // 12: FTL / GC
  0.88,   // 13: 对齐状态
  1.52,   // 14: 对齐硬件与模型
  1.1833  // 15: 代表性实测误差
];

const bThin = { pt: 0.6, color: C.line };
const bGroupLeft = { pt: 1.1, color: C.lineStrong };
const bHead = { pt: 0.7, color: '3B4D6B' };
const bGroupTop = { pt: 1.0, color: C.lineStrong };

function hdrCell(text, extra = {}) {
  return {
    text,
    options: {
      fontFace: 'Microsoft YaHei',
      fontSize: 8.8,
      bold: true,
      color: 'FFFFFF',
      fill: { color: C.navy },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.03, 0.02, 0.03],
      border: [bHead, bHead, bHead, bHead],
      ...extra
    }
  };
}

const headRow1 = [
  hdrCell('分类', { rowspan: 2, fontSize: 9.0 }),
  hdrCell('所属组织', { rowspan: 2, fontSize: 9.0 }),
  hdrCell('仿真器', { rowspan: 2, fontSize: 9.0 }),
  hdrCell('服务调度', { colspan: 3, fontSize: 9.2, fill: { color: C.navy } }),
  hdrCell('多卡并行与互联', { colspan: 4, fontSize: 9.2, fill: { color: C.navy } }),
  hdrCell('内存器件', { colspan: 3, fontSize: 9.2, fill: { color: C.navy } }),
  hdrCell('真机实测数据对齐（官方声明）', { colspan: 3, fontSize: 9.2, fill: { color: '16355D' } })
];

const subCols = [
  ['连续批处理', false], ['KV 容量', false], ['PD 分离', false],
  ['TP', true], ['PP', false], ['EP', false], ['链路争用', false],
  ['HBM 命令', true], ['NAND 时序', false], ['FTL / GC', false],
  ['对齐状态', true], ['对齐硬件与模型', false], ['代表性实测误差', false]
];
const headRow2 = subCols.map(([label, isCalOrGrp], idx) =>
  hdrCell(label, {
    fontSize: 8.0,
    fill: { color: idx >= 10 ? '1E4477' : C.navySub },
    border: [bHead, bHead, { pt: 1.2, color: C.navy }, isCalOrGrp ? { pt: 1.1, color: '64748B' } : bHead]
  })
);

const groupStartRow = new Map();
{
  let cursor = 0;
  for (const g of GROUPS) {
    groupStartRow.set(cursor, g);
    cursor += g.span;
  }
}

const bodyRows = ROWS.map((r, ri) => {
  const grpInfo = groupStartRow.get(ri);
  const isGroupTop = Boolean(grpInfo) && ri > 0;
  const topBorder = isGroupTop ? bGroupTop : bThin;
  const rowBg = r.mine ? C.mineBg : (ri % 2 === 1 ? C.zebra : C.paper);
  const cells = [];

  // Col 0: Category (only pushed on first row of each group due to rowspan)
  if (grpInfo) {
    cells.push({
      text: grpInfo.label,
      options: {
        rowspan: grpInfo.span,
        fontFace: 'Microsoft YaHei',
        fontSize: 8.2,
        bold: true,
        color: r.mine ? C.mineInk : C.ink2,
        fill: { color: r.mine ? C.mineCatBg : C.grpBg },
        align: 'center',
        valign: 'middle',
        margin: [0.02, 0.03, 0.02, 0.03],
        border: [topBorder, bThin, bGroupTop, bThin]
      }
    });
  }

  // Col 1: Organization (2 centered lines)
  cells.push({
    text: [
      { text: r.org[0] + '\n', options: { fontSize: 8.3, bold: true, color: r.mine ? C.mineInk : C.ink } },
      { text: r.org[1], options: { fontSize: 6.8, color: C.muted } }
    ],
    options: {
      fontFace: 'Microsoft YaHei',
      fill: { color: rowBg },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.03, 0.02, 0.03],
      border: [topBorder, bThin, bThin, bThin]
    }
  });

  // Col 2: Tool name (single centered line)
  cells.push({
    text: r.name,
    options: {
      fontFace: 'Microsoft YaHei',
      fontSize: 8.5,
      bold: true,
      color: r.mine ? C.mineInk : C.ink,
      fill: { color: rowBg },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.03, 0.02, 0.03],
      border: [topBorder, bGroupLeft, bThin, bThin]
    }
  });

  // Cols 3..12: 10 Mechanism ratings
  COL_KEYS.forEach((k, ki) => {
    const st = r.st[k];
    const leftB = (k === 'batch' || k === 'tp' || k === 'hbm') ? bGroupLeft : bThin;
    let fg = C.na, bg = rowBg, fSize = 10.0;
    if (st === 'S') { fg = C.good; bg = r.mine ? 'F4F3E6' : C.goodBg; fSize = 13.0; }
    else if (st === 'P') { fg = C.part; bg = rowBg; fSize = 10.0; }
    else if (st === 'N') { fg = C.bad; bg = r.mine ? 'FCECE9' : C.badBg; fSize = 9.6; }
    cells.push({
      text: SYM[st],
      options: {
        fontFace: 'Microsoft YaHei',
        fontSize: fSize,
        bold: true,
        color: fg,
        fill: { color: bg },
        align: 'center',
        valign: 'middle',
        margin: [0.01, 0.015, 0.01, 0.015],
        border: [topBorder, ki === 9 ? bGroupLeft : bThin, bThin, leftB]
      }
    });
  });

  // Col 13: Calibration status + scope
  const cs = r.cal.st;
  const calFg = cs === 'S' ? C.good : (cs === 'P' ? C.partCal : C.muted);
  const calBg = r.mine ? rowBg : (cs === 'S' ? C.goodBg : (cs === 'P' ? C.partCalBg : rowBg));
  const calSym = cs === 'S' ? '●' : (cs === 'P' ? '◐' : '✕');
  const calTxt = cs === 'S' ? ' 已对齐' : (cs === 'P' ? ' 局部对齐' : ' 未对齐');
  cells.push({
    text: [
      { text: calSym, options: { fontSize: cs === 'S' ? 9.8 : 8.0, bold: true, color: calFg } },
      { text: calTxt + '\n', options: { fontSize: 8.0, bold: true, color: calFg } },
      { text: r.cal.scope, options: { fontSize: 6.8, color: C.muted } }
    ],
    options: {
      fontFace: 'Microsoft YaHei',
      fill: { color: calBg },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.03, 0.02, 0.03],
      border: [topBorder, bThin, bThin, bGroupLeft]
    }
  });

  // Col 14: Aligned hardware + model/workload
  cells.push({
    text: [
      { text: r.cal.hw + '\n', options: { fontSize: 7.3, bold: cs !== 'N', color: cs === 'N' ? C.muted : C.ink2 } },
      { text: r.cal.mdl, options: { fontSize: 6.8, color: C.muted } }
    ],
    options: {
      fontFace: 'Microsoft YaHei',
      fill: { color: rowBg },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.035, 0.02, 0.035],
      border: [topBorder, bThin, bThin, bThin]
    }
  });

  // Col 15: Representative error
  cells.push({
    text: [
      { text: r.cal.err1 + '\n', options: { fontSize: 8.0, bold: cs !== 'N', color: cs === 'N' ? C.muted : C.ink } },
      { text: r.cal.err2, options: { fontSize: 6.8, color: C.muted } }
    ],
    options: {
      fontFace: 'Microsoft YaHei',
      fill: { color: rowBg },
      align: 'center',
      valign: 'middle',
      margin: [0.02, 0.035, 0.02, 0.035],
      border: [topBorder, bThin, bThin, bThin]
    }
  });

  return cells;
});

const rowH = [0.29, 0.31, ...Array(14).fill(0.378)]; // total = 0.60 + 5.292 = 5.892"
s.addTable([headRow1, headRow2, ...bodyRows], {
  x: 0.34,
  y: 0.86,
  w: 12.6533,
  colW,
  rowH,
  autoPage: false,
  valign: 'middle'
});

// ---- Footer Takeaways & Source ----
s.addText([
  { text: '阅读边界  ', options: { bold: true, color: C.good } },
  { text: '1) ✕ 表示属研究范围但未建模（如 HBF 缺 FTL/GC），— 表示非设计范围（服务仿真器仍通过真机 profile 隐含内存耗时）；  ', options: { color: C.ink2 } },
  { text: '2) 各工具验证对象（端到端请求延迟 / 单层算子 / 训练集群 / Verilog 协议）不同，不可直接横向排名。', options: { color: C.ink2 } }
], {
  x: 0.34, y: 6.89, w: 12.65, h: 0.22,
  fontFace: 'Microsoft YaHei', fontSize: 8.2, margin: 0, valign: 'mid'
});

s.addShape(S.line, {
  x: 0.34, y: 7.15, w: 12.6533, h: 0,
  line: { color: C.line, width: 0.6 }
});
s.addText('数据来源：14 个仿真器公开仓库主干源码与对应论文逐项核查（截至 2026-11-12） · 详见 docs/llm-simulator-comparison-20261110.html', {
  x: 0.34, y: 7.19, w: 11.8, h: 0.20,
  fontFace: 'Microsoft YaHei', fontSize: 8.0, color: C.muted, margin: 0, valign: 'mid'
});
s.addText('1', {
  x: 12.40, y: 7.19, w: 0.59, h: 0.20,
  fontFace: 'Microsoft YaHei', fontSize: 8.5, color: C.muted, align: 'right', margin: 0, valign: 'mid'
});

s.addNotes(
  '本页汇总 14 个 LLM 与内存仿真器的能力矩阵与官方真机实测对齐情况（截至 2026-11-12 公开仓库主干与对应论文）。\n' +
  '1. 统一判准：不按文档提及打分，仅当修改某机制参数会直接改变仿真输出时记为“支持（●）”；有该路径但缺关键环节记为“部分（◐）”；属于该工具研究范围但未实现记为“不提供（✕）”；不在设计范围记为“不适用（—）”。\n' +
  '2. 两个同名 HBFSim 完全独立：FlashLabAE 版（MBZUAI）为单封装事件驱动模型，含主机 FTL/GC；SlugLab 版（UCSC 等）为真实单卡 GPU 地址插桩延迟注入。\n' +
  '3. 公开基线与论文扩展区分：LLMSimulator / Duplex 与 ASTRA-sim 2.x 均按当前公开主干代码评定（Exploring HBF 扩展与 ASTRA-sim 3.0 PR #389 未合入主干）。\n' +
  '4. 实测误差不可横向排名：服务仿真器对齐端到端请求延迟/吞吐，硬件探索工具对齐单层或标定后阶段，DRAM 仿真器对齐 Verilog 协议合法性，MQSim 对齐传统 NVMe/SATA SSD；HeteroLLM-sim 4.0.0 报告中的 10% 为采样条件预测预算而非实机误差。'
);

const outFile = path.join(__dirname, 'LLM与内存仿真器能力矩阵对比.pptx');
p.writeFile({ fileName: outFile }).then(() => {
  console.log('WROTE PPTX:', outFile);
});
