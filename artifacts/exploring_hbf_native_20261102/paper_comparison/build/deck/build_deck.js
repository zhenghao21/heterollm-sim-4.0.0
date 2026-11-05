// Four-page lecture deck for the 20261102 scan's paper comparison. Numbers come from deck_data.json
// (prepare_data.py); the XY charts are added afterwards by finish_deck.py.
// House style: white page with content in clearly separate titled panels; pictures before words; table text centred;
// theme fonts Malgun Gothic + Microsoft YaHei; pure blue text for strengths, pure red for weaknesses. Paper is grey and native blue.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_SCRIPTS + "/apply_theme.js");

const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = process.argv[2];
const THEME = {
  name: "HBF Paper Comparison",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "14181F", lt1: "FFFFFF", dk2: "1E2A38", lt2: "F2F5F9",
    accent1: "2A78D6", accent2: "EB6834", accent3: "0E8F7E", accent4: "D03B3B", accent5: "6250D6", accent6: "52514E",
    hlink: "1C5CAB", folHlink: "6250D6",
  },
};
const NATIVE = "2A78D6", NATIVE_LIGHT = "9EC5F4", PAPER = "A9ADB3", PAPER_LIGHT = "D9DBDE", INK = "14181F", INK2 = "52514E", MUTED = "6F6D67", GRID = "E1E0D9", EDGE = "C9D1DB", PAPER_INK = "83878E";
const BLUE = "0000FF", RED = "FF0000";
const HW = ["HBM4", "HBF", "HBF+", "CONV", "CONV+"], WL = ["SHORT", "MID", "LONG"];
const MODELS = [["llama3_405b", "405B"], ["llama4_maverick", "Maverick"]];
const l3 = "llama3_405b", mav = "llama4_maverick";
const key = {};
for (const r of DATA.rows) key[`${r.model}|${r.workload}|${r.hardware}`] = r;
const get = (m, w, h) => key[`${m}|${w}|${h}`];
const f2 = v => v.toFixed(2), f1 = v => v.toFixed(1);
const f0 = v => Math.round(v).toLocaleString("en-US");
const values = (hardware, field) => MODELS.flatMap(([m]) => WL.flatMap(w => hardware.map(h => get(m, w, h)[field])));
const span = (hardware, field) => { const v = values(hardware, field); return [Math.min(...v), Math.max(...v)]; };
const range = (hardware, field) => span(hardware, field).map(f2).join("–");
const S = DATA.summary, G = DATA.groups, W = DATA.whatifs;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Exploring HBF 论文对照";
pres.author = "heterollm-sim";
pres.subject = "native 全量扫描 exploring_hbf_native_20261102 与 arXiv 2608.13868 的对照";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.2, w: 11.2, h: 0.3, fontSize: 13, bold: true, color: INK2, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.5, w: 12.33, h: 0.62, fontSize: 26, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "source", type: "body", x: 0.5, y: 6.84, w: 11.6, h: 0.5, fontSize: 10, color: MUTED, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.23, y: 6.92, w: 0.6, h: 0.34, fontSize: 10, color: MUTED, align: "right" },
});

// "[b:…]" marks a strength in pure blue, "[r:…]" a weakness in pure red; "\n" starts a new paragraph.
function rich(value, o = {}) {
  const runs = [];
  const lines = String(value).split("\n");
  lines.forEach((line, i) => {
    const parts = line.split(/(\[[br]:[^\]]*\])/).filter(Boolean);
    parts.forEach((part, j) => {
      const mark = part.match(/^\[([br]):([^\]]*)\]$/);
      const options = Object.assign({}, o, mark ? { color: mark[1] === "b" ? BLUE : RED, bold: true } : {});
      if (j === parts.length - 1 && i < lines.length - 1) options.breakLine = true;
      runs.push({ text: mark ? mark[2] : part, options });
    });
  });
  return runs;
}
function page(kicker, title, source, notes) {
  const slide = pres.addSlide({ masterName: "CONTENT" });
  slide.addText(kicker, { placeholder: "kicker" });
  slide.addText(rich(title), { placeholder: "title" });
  slide.addText(rich(source), { placeholder: "source" });
  slide.addNotes(notes);
  return slide;
}
function text(slide, value, x, y, w, h, o = {}) {
  slide.addText(Array.isArray(value) ? value : rich(value), Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, align: "left", valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 0 }, o));
}
function box(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: "FFFFFF", width: 0.75 }, objectName: name });
}
// A bordered block with a dark title bar: the unit that keeps one piece of content apart from the next.
function panel(slide, x, y, w, h, title, name) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "FFFFFF" }, line: { color: EDGE, width: 1 }, objectName: name });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.38, fill: { color: C.text2 }, line: { type: "none" }, objectName: name + " 标题栏" });
  text(slide, title, x + 0.14, y, w - 0.28, 0.38, { fontSize: 13, bold: true, color: "FFFFFF", valign: "middle", objectName: name + " 标题" });
}
// A tinted block inside a panel, or a free-standing number tile.
function tile(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: EDGE, width: 0.5 }, objectName: name });
}
function dot(slide, x, y, d, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x: x - d / 2, y: y - d / 2, w: d, h: d, fill: { color: fill }, line: { color: "FFFFFF", width: 0.75 }, objectName: name });
}
// A straight line between two points; PowerPoint draws a rising line as a flipped box.
function segment(slide, x1, y1, x2, y2, color, width, name) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: (x2 - x1) * (y2 - y1) < 0, line: { color, width }, objectName: name });
}
// One paper or native bar on a horizontal value scale, labelled inside when it fits and to its right when not.
function rangeBar(slide, scale, y, h, [lo, hi], fill, ink, label, name, labelFrom) {
  const x = scale(lo), w = Math.max(scale(hi) - x, 0.03);
  box(slide, x, y, w, h, fill, name);
  const inside = w > 0.075 * label.length + 0.1;
  if (inside) text(slide, label, x, y, w, h, { fontSize: 10.5, color: ink, align: "center", valign: "middle", objectName: name + " 数值" });
  else text(slide, label, Math.max(x + w + 0.05, labelFrom || 0), y, 1.6, h, { fontSize: 10.5, color: INK2, valign: "middle", objectName: name + " 数值" });
}

// ---------------------------------------------------------------- 1 trend
{
  const p = DATA.previous;
  const n = (m, w, h, f) => get(m, w, h)[f];
  const pairs = [
    { model: "Maverick SHORT", a: "HBF+", b: "HBF", m: mav, w: "SHORT", why: "写入挡读，见第 3 页" },
    { model: "Maverick MID", a: "HBF+", b: "HBM4", m: mav, w: "MID", why: "导入开销，见第 3 页" },
    { model: "405B SHORT", a: "HBF+", b: "HBM4", m: l3, w: "SHORT", why: "论文差 0.015，基本打平" },
    { model: "Maverick MID", a: "CONV", b: "CONV+", m: mav, w: "MID", why: "[r:未验证]" },
  ];
  const s = page("问题 1：五种配置谁强谁弱，和论文一致吗？", `[b:相对趋势与论文一致]：五种配置分档相同，${S.order_agree}/${S.order_total} 对配置排序相同`,
    `native = 本项目按硬件时序逐步仿真的结果。归一化吞吐 = 每 GPU 吞吐 ÷ 同负载 HBM4，所以 HBM4 恒为 1。范围：2 个模型 × 3 种负载 × 5 种配置 = 30 个点。\n数据：扫描 exploring_hbf_native_20261102，代码 cbee2ea3；论文值读自 Fig. 4，约 ±0.01。交叉小图各用自己的纵向刻度。`,
    `【这一页回答】
五种存储配置谁强谁弱，native 仿真和论文是否一致。答案是一致。

【先解释名词】
native 是本项目自己的物理仿真后端：HBM 按 JEDEC 时序，NAND 按页和 plane 建模，不用恒定带宽。论文是 Exploring HBF（arXiv 2608.13868）。
五种配置：HBM4 是基准。HBF 是高带宽闪存（每 die 25 个 plane，读 1 µs），CONV 是常规闪存（每 die 16 个 plane，读 3 µs），这两种配置仍带 HBM。HBF+、CONV+ 没有 HBM，只有闪存加 SRAM。
模型是 Llama3 405B 和 Llama4 Maverick，负载是 SHORT、MID、LONG。TPOT 是生成每个 token 的时间，约束是不超过 100 ms。每个点都在 8 GPU 下搜索 GPU 分工和 batch，取每 GPU 吞吐最大。
归一化吞吐是每 GPU 吞吐除以同模型、同负载下 HBM4 的每 GPU 吞吐。两边都这样算，比较的就是配置之间的相对关系。

【左边的散点图】
每个点是一组“模型 × 负载 × 配置”，共 24 个。HBM4 自身恒为 1，所以不画。横轴是论文，纵轴是 native。点落在粗对角线上表示两边相等，两条细线是 ±0.15。红字标的是 native 明显低于论文的点。
图上分成两团，这就是论文的核心结论：黑点 HBF 系与 HBM4 同档，论文 ${range(["HBF", "HBF+"], "paper_norm_tps")}，native ${range(["HBF", "HBF+"], "native_norm_tps")}；灰点 CONV 系明显落后，论文 ${range(["CONV", "CONV+"], "paper_norm_tps")}，native ${range(["CONV", "CONV+"], "native_norm_tps")}。

【右上三个数字】
${f2(S.norm_geomean)}×：24 个点的 native ÷ 论文取几何平均，说明整体没有偏高或偏低，上一轮是 ${f2(p.norm_geomean)}×。
${S.norm_within_015}/${S.norm_count}：差值在 ±0.15 以内的点数，上一轮是 ${p.norm_within_015}。
${S.order_agree}/${S.order_total}：6 组“模型 × 负载”，每组 5 种配置两两比较 10 对，共 60 对，其中 ${S.order_agree} 对谁强谁弱相同，上一轮是 ${p.order_agree}。

【右下四张交叉图】
每张小图是排序不同的一对配置，左端是论文，右端是 native。两条线交叉，就是谁强谁弱变了。每张小图用自己的刻度，交叉的幅度不能跨图比较。
前两对第 3 页解释。第三对论文只差 0.015，基本是打平。第四对里 CONV+ 的 PD 导入额外花 16.9 ms，CONV 只有 1.6 ms，这只是线索，没有做反事实实验，所以写“未验证”。
另外，超出 ±0.15 的 6 个点都是 HBF 系：4 个 native 偏低（Maverick SHORT、MID 的 HBF 和 HBF+），2 个偏高（405B LONG HBF、405B SHORT HBF+）。

【要提醒听众】
趋势一致不等于绝对值一致。绝对吞吐差多少，下一页讲。`);
  panel(s, 0.5, 1.22, 7.2, 5.54, "每个点是一组配置：越靠近对角线，和论文越接近（细线为 ±0.15）", "散点图板块");
  [[`${f2(S.norm_geomean)}×`, "native ÷ 论文\n几何平均（24 点）"], [`${S.norm_within_015}/${S.norm_count}`, "两边差值\n在 ±0.15 以内"], [`${S.order_agree}/${S.order_total}`, "两两比较\n谁强谁弱相同"]].forEach(([v, l], i) => {
    const x = 7.9 + i * 1.68;
    tile(s, x, 1.22, 1.57, 1.36, "统计卡 " + i);
    text(s, v, x + 0.05, 1.28, 1.47, 0.56, { fontSize: 24, bold: true, color: BLUE, align: "center", valign: "middle", objectName: "统计值 " + i });
    text(s, l, x + 0.05, 1.86, 1.47, 0.62, { fontSize: 11, color: INK2, align: "center", objectName: "统计说明 " + i });
  });
  panel(s, 7.9, 2.74, 4.93, 4.02, "4 对排序不同：两线交叉 = 谁强谁弱变了", "交叉图板块");
  pairs.forEach((pair, i) => {
    const X = 8.0 + (i % 2) * 2.4, Y = 3.2 + Math.floor(i / 2) * 1.76, cw = 2.33, ch = 1.7;
    tile(s, X, Y, cw, ch, "交叉图 " + i);
    const [pa, pb, na, nb] = [n(pair.m, pair.w, pair.a, "paper_norm_tps"), n(pair.m, pair.w, pair.b, "paper_norm_tps"), n(pair.m, pair.w, pair.a, "native_norm_tps"), n(pair.m, pair.w, pair.b, "native_norm_tps")];
    const lo = Math.min(pa, pb, na, nb), hi = Math.max(pa, pb, na, nb), top = Y + 0.54, H = 0.64, y = v => top + (hi - v) / (hi - lo) * H;
    const xl = X + 0.76, xr = X + 1.56;
    text(s, [{ text: pair.model + "：", options: { color: INK2 } }, { text: pair.a, options: { bold: true, color: INK } }, { text: " 比 ", options: { color: INK2 } }, { text: pair.b, options: { bold: true, color: PAPER_INK } }], X, Y + 0.05, cw, 0.24, { fontSize: 10.5, align: "center", valign: "middle", objectName: "交叉图名称 " + i });
    segment(s, xl, top - 0.04, xl, top + H + 0.04, EDGE, 0.75, "交叉图论文轴");
    segment(s, xr, top - 0.04, xr, top + H + 0.04, EDGE, 0.75, "交叉图 native 轴");
    segment(s, xl, y(pb), xr, y(nb), PAPER, 2.25, `交叉图 ${pair.b}`);
    segment(s, xl, y(pa), xr, y(na), INK, 2.25, `交叉图 ${pair.a}`);
    for (const [cx, v, color] of [[xl, pb, PAPER], [xr, nb, PAPER], [xl, pa, INK], [xr, na, INK]]) dot(s, cx, y(v), 0.11, color, "交叉图端点");
    // Keep the two value labels on a side apart when a pair is nearly tied.
    const spread = (u, v) => { const yu = y(u), yv = y(v), gap = 0.17 - Math.abs(yu - yv); if (gap <= 0) return [yu, yv]; return yu < yv ? [yu - gap / 2, yv + gap / 2] : [yu + gap / 2, yv - gap / 2]; };
    const [la, lb] = spread(pa, pb), [ra, rb] = spread(na, nb);
    text(s, f2(pa), X + 0.3, la - 0.09, 0.38, 0.18, { fontSize: 10.5, color: INK, align: "right", valign: "middle", objectName: "交叉图论文值" });
    text(s, f2(pb), X + 0.3, lb - 0.09, 0.38, 0.18, { fontSize: 10.5, color: PAPER_INK, align: "right", valign: "middle", objectName: "交叉图论文值" });
    text(s, `[r:${f2(na)}]`, xr + 0.08, ra - 0.09, 0.38, 0.18, { fontSize: 10.5, valign: "middle", objectName: "交叉图 native 值" });
    text(s, `[r:${f2(nb)}]`, xr + 0.08, rb - 0.09, 0.38, 0.18, { fontSize: 10.5, valign: "middle", objectName: "交叉图 native 值" });
    text(s, "论文", xl - 0.3, Y + 0.3, 0.6, 0.16, { fontSize: 9.5, color: MUTED, align: "center", valign: "middle", objectName: "交叉图轴名" });
    text(s, "native", xr - 0.3, Y + 0.3, 0.6, 0.16, { fontSize: 9.5, color: MUTED, align: "center", valign: "middle", objectName: "交叉图轴名" });
    text(s, pair.why, X, Y + 1.39, cw, 0.28, { fontSize: 10.5, align: "center", valign: "middle", objectName: "交叉图原因" });
  });
}
// ---------------------------------------------------------------- 2 absolute gap
{
  const cargo = r => {
    const kv = r.native_context_bytes_per_gpu / 1e9, paperKv = r.paper_batch_per_gpu * kv / r.native_batch_per_gpu;
    return { nFix: r.other_bytes_gb, nKv: kv, pFix: r.paper_total_gb - paperKv, pKv: paperKv, nTotal: r.other_bytes_gb + kv, pTotal: r.paper_total_gb };
  };
  const change = (now, was) => { const v = Math.round((now / was - 1) * 100); return v < 0 ? `少 ${-v}%` : `多 ${v}%`; };
  const hbm = DATA.rows.filter(x => x.hardware === "HBM4" && !(x.model === l3 && x.workload === "SHORT"));
  const rng = (rows, f) => `${f2(Math.min(...rows.map(f)))}–${f2(Math.max(...rows.map(f)))}`;
  const nativeUtil = rng(hbm, x => x.native_bandwidth_fraction), paperUtil = rng(hbm, x => x.paper_util);
  const pct = ([a, b]) => `${Math.round(a * 100)}–${Math.round(b * 100)}%`;
  const ms = ([a, b]) => `${Math.round(a)}–${Math.round(b)} ms`;
  const perHw = HW.map(h => DATA.per_hardware[h].abs);
  const longPlus = get(l3, "LONG", "HBF+"), convMid = get(l3, "MID", "CONV+"), dp4 = get(l3, "SHORT", "HBF"), midPlus = get(mav, "MID", "HBF+");
  const cA = cargo(longPlus), cB = cargo(convMid), cC = cargo(dp4), cD = cargo(midPlus);
  const trucks = [
    { title: "货几乎全是 KV", who: "LONG 的 HBF 系（4 点）", group: G.long_hbf, example: "405B LONG HBF+", detail: `总量${change(cA.nTotal, cA.pTotal)}`, c: cA },
    { title: "固定货占一大块", who: "CONV 系（12 点）", group: G.conv, example: "405B MID CONV+", detail: `总量${change(cB.nTotal, cB.pTotal)}`, c: cB },
    { title: "多拉了三份权重", who: "405B SHORT/MID HBF 系（4 点）", group: G.dp4, example: "405B SHORT HBF（四副本）", detail: `总量${change(cC.nTotal, cC.pTotal)}`, c: cC },
    { title: "车只跑了一半时间", who: "Maverick MID 的 HBF 系（2 点）", group: G.maverick_mid_hbf, example: "Maverick MID HBF+", detail: `TPOT ${Math.round(midPlus.native_tpot_ms)} 对 ${Math.round(midPlus.paper_tpot_ms)} ms`, c: cD },
  ];
  const s = page("问题 2：绝对吞吐和论文差多少，差在哪？", `绝对吞吐[r:只有论文的六成左右]，主要差在 batch，各组原因不同`,
    "比值都是 native ÷ 论文。货车单位 GB / 每步 / 每 GPU，每行各自缩放，只比同一行的两辆车。HBF 系 = HBF、HBF+，CONV 系 = CONV、CONV+。论文的固定货按单份权重估算（布局未公开，实际可能更多），所以“多拉权重”不单独算作和论文的差异。论文的简化（native 不采用）：恒定峰值带宽、理想 Prefill、SRAM 只放激活。",
    `【这一页回答】
绝对吞吐差多少、差在哪。答案是：全部 30 个点都低于论文，几何平均 ${f2(DATA.all_factors.abs)}，范围 0.37–0.80。主要差在 batch，而且各组原因不同。

【左上的数字：五种配置一起偏低】
按配置取几何平均，HBM4 ${f2(perHw[0])}、HBF ${f2(perHw[1])}、HBF+ ${f2(perHw[2])}、CONV ${f2(perHw[3])}、CONV+ ${f2(perHw[4])}，都在 ${f2(Math.min(...perHw))}–${f2(Math.max(...perHw))}，非常接近。除以 HBM4 时这部分互相抵消，这就是第 1 页趋势一致的原因。它只说明差距是系统性的，不代表物理过程相同。
单点最低是 Maverick SHORT HBF+ 的 0.37，最高是 Maverick MID CONV 的 0.80，完整表格在 HTML 报告里。

【左下：差距分两类】
吞吐 = batch ÷ TPOT，所以吞吐比 = batch 比 ÷ TPOT 比，30 个点上残差小于 0.1%。图上每个点是一组配置，横轴是 TPOT 比，越右表示单步比论文越慢；纵轴是 batch 比，越低表示同时服务的请求比论文越少。对角线上吞吐等于论文，30 个点全部在线下方。
右上这一团黑点是 HBM4：batch 比 0.77–0.94，货差不多；TPOT 比 1.25–1.79，路上慢。两边都受容量限制，论文的 TPOT 只有 24–34 ms，native 要支付行切换、tRRD/tFAW 和刷新。这 5 个点 native 利用率 ${nativeUtil}，论文估算 ${paperUtil}，两者之比 1.37–1.60，和 TPOT 比同一量级。这只是量级对照，没有逐项拆分。例外是 405B SHORT HBM4，选了 DP2/TP4，batch 比降到 0.43。
左下这一团蓝点是其余配置：TPOT 没有比论文长，差距全在 batch。论文里时延先到 100 ms 上限的 23 个点，几何平均 batch 比 ${f2(G.slo.batch)}，TPOT 比 ${f2(G.slo.tpot)}。Maverick SHORT HBF+ 单独在右下角，第 3 页解释。

【右边：为什么货装得少】
把每一步想成一趟货车。车厢大小是有效带宽乘 TPOT。权重等是每趟必拉的固定货，装完以后剩下的车厢才能装 KV，而 KV 字节 = batch × 每请求 KV 字节。这是帮助理解的比喻，不是精确等式：利用率按步时长算，TPOT 还含导入等开销。
灰色是固定货，蓝色是 KV。每块里上面一辆是论文，下面一辆是 native。每块各自缩放，只比同一块的两辆车。最右边红字是这个例子里 KV 比论文少多少，下面是这一组的几何平均 batch 比。
第一块，货几乎全是 KV：LONG 的 HBF、HBF+，KV 占 ${pct(G.long_hbf.kv_share_span)}。总运力打几折，batch 基本就打几折。405B LONG HBF+ 总量 ${Math.round(cA.nTotal)} 对 ${Math.round(cA.pTotal)} GB，KV ${Math.round(cA.nKv)} 对 ${Math.round(cA.pKv)} GB。Maverick 的两点还选了 DP2，每步固定货约 205 GB。
第二块，固定货占一大块：CONV、CONV+，KV 只占 ${pct(G.conv.kv_share_span)}。固定货减不了，所以总量只少一点，KV 就少很多。405B MID CONV+ 总量 ${Math.round(cB.nTotal)} 对 ${Math.round(cB.pTotal)} GB，少 22%，KV 少 50%。
第三块，多拉了三份权重：405B SHORT、MID 的 HBF、HBF+ 选 DP4/TP2 四副本，每步字节是单副本的 ${f2(G.dp4.factor_span[0])}–${f2(G.dp4.factor_span[1])} 倍，KV 只占 ${pct(G.dp4.kv_share_span)}。405B SHORT HBF 的总量比论文估算还多，KV 却少一半以上。论文布局未公开，Fig. 4 中 405B SHORT HBF+ 从 1 到 16 GPU 每 GPU 吞吐不变，推测论文也用多副本，所以这一块不单独算作和论文的差异。
第四块，车只跑了一半时间：Maverick MID 的 HBF、HBF+，TPOT 只用到 ${ms(G.maverick_mid_hbf.tpot_ms_span)}，第 3 页解释为什么停在这里。

【为什么不能只说“带宽用不满”】
时延受限的 23 个点上，利用率和 batch 比的对数相关系数只有 ${f2(DATA.util_batch_correlation)}。CONV+ 利用率最高（${range(["CONV+"], "native_bandwidth_fraction")}），batch 比仍只有 ${range(["CONV+"], "batch_ratio")}。第二、第三块就是原因：车开得再满，固定货也要先装。

【要提醒听众】
Maverick SHORT HBF 还没有单独定位，这是分析上的缺口。论文的固定货按单份权重估算，是下界。页脚列出的论文简化，native 都不采用，这也是绝对值偏低的背景。`);
  panel(s, 0.5, 1.22, 4.55, 5.54, "差距分两类：货装得少，或路上慢", "差距分类板块");
  tile(s, 0.62, 1.7, 4.31, 0.86, "五种配置数字");
  text(s, `[r:${f2(Math.min(...perHw))}–${f2(Math.max(...perHw))}]`, 0.68, 1.7, 1.62, 0.86, { fontSize: 20, align: "center", valign: "middle", objectName: "五种配置吞吐比" });
  text(s, "五种配置的吞吐比，一起偏低，除以 HBM4 后抵消，所以趋势还一致", 2.34, 1.7, 2.52, 0.86, { fontSize: 11, color: INK2, valign: "middle", objectName: "五种配置说明" });
  panel(s, 5.25, 1.22, 7.58, 5.54, "货为什么装得少？把每一步想成一趟货车", "货车板块");
  text(s, "车厢 = 有效带宽 × TPOT。固定货（权重等）先装，剩下的才装 KV。", 5.39, 1.66, 5.5, 0.34, { fontSize: 11.5, color: INK2, valign: "middle", objectName: "货车说明" });
  let lx = 11.0;
  for (const [label, color, w] of [["固定货", PAPER, 0.5], ["KV", NATIVE, 0.25]]) {
    box(s, lx, 1.76, 0.18, 0.14, color, "货车图例 " + label);
    text(s, label, lx + 0.22, 1.71, w, 0.24, { fontSize: 11, color: INK2, valign: "middle", objectName: "货车图例文字" });
    lx += 0.22 + w + 0.1;
  }
  trucks.forEach((t, i) => {
    const X = 5.37, y = 2.04 + i * 1.04, x0 = 8.2;
    tile(s, X, y, 7.34, 0.98, "货车块 " + i);
    text(s, t.title, X + 0.1, y + 0.08, 2.3, 0.28, { fontSize: 13, bold: true, valign: "middle", objectName: "货车原因" });
    text(s, t.who, X + 0.1, y + 0.37, 2.3, 0.24, { fontSize: 10, color: INK2, valign: "middle", objectName: "货车分组" });
    text(s, "例：" + t.example, X + 0.1, y + 0.63, 2.3, 0.24, { fontSize: 10, color: INK2, valign: "middle", objectName: "货车例子" });
    const unit = 2.8 / Math.max(t.c.nTotal, t.c.pTotal);
    [["论文", t.c.pFix, t.c.pKv, PAPER_LIGHT, NATIVE_LIGHT, INK], ["native", t.c.nFix, t.c.nKv, PAPER, NATIVE, "FFFFFF"]].forEach(([who, fix, kv, fixColor, kvColor, kvInk], row) => {
      const by = y + 0.14 + row * 0.39, h = 0.25, fw = fix * unit, kw = kv * unit;
      text(s, who, 7.72, by, 0.46, h, { fontSize: 10, color: INK2, valign: "middle", objectName: "货车名称" });
      box(s, x0, by, fw, h, fixColor, `${who} 固定货`);
      box(s, x0 + fw, by, kw, h, kvColor, `${who} KV`);
      text(s, String(Math.round(fix)), x0, by, fw, h, { fontSize: 10, color: INK, align: "center", valign: "middle", objectName: "固定货数值" });
      text(s, `KV ${Math.round(kv)}`, x0 + fw, by, kw, h, { fontSize: 10, color: kvInk, align: "center", valign: "middle", objectName: "KV 数值" });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + fw + kw + 0.02, y: by + 0.05, w: 0.17, h: h - 0.05, rectRadius: 0.03, fill: { color: C.text2 }, line: { type: "none" }, objectName: "车头" });
      for (const wheel of [x0 + 0.14, x0 + fw + kw - 0.1, x0 + fw + kw + 0.1]) dot(s, wheel, by + h + 0.015, 0.08, INK, "车轮");
    });
    text(s, `[r:KV ${change(t.c.nKv, t.c.pKv)}]`, 11.26, y + 0.08, 1.4, 0.3, { fontSize: 14, align: "center", valign: "middle", objectName: "货车 KV 变化" });
    text(s, t.detail, 11.26, y + 0.39, 1.4, 0.24, { fontSize: 10, color: INK2, align: "center", valign: "middle", objectName: "货车细节" });
    text(s, `该组 batch 比 ${f2(t.group.batch)}`, 11.26, y + 0.65, 1.4, 0.24, { fontSize: 10, color: INK2, align: "center", valign: "middle", objectName: "货车 batch 比" });
  });
  text(s, `[r:还没有定位]：Maverick SHORT HBF（1 点，batch 比 ${f2(G.maverick_short_hbf.batch)}）。Maverick SHORT HBF+ 见第 3 页。`, 5.39, 6.24, 7.3, 0.46, { fontSize: 11.5, valign: "middle", objectName: "未定位说明" });
}
// ---------------------------------------------------------------- 3 largest gaps
{
  const norm = (w, h) => f2(get(mav, w, h).native_norm_tps);
  const bothNorm = f2(W.both.rate / get(mav, "SHORT", "HBM4").native_tps);
  const s = page("问题 3：偏差最大的点，原因是什么？", "最大偏差在 Maverick HBF+：[r:闪存写入不能暂停，挡住了读]",
    `上面两块是 Maverick SHORT HBF+ 的诊断实验，不是扫描结果。B6048 约 756/GPU，接近论文的 768/GPU。时间线为 layer-045、rank 0，空白是其它任务。\nplane：闪存内可并行工作的单元。尾页：KV 没写满的最后一页。当前每 256 token 一块，每请求一块 = 2048 token 一块。`,
    `【这一页回答】
偏差最大的几个点为什么偏。上面两块讲 Maverick SHORT HBF+，下面两块讲 Maverick MID。

【先解释名词】
plane 是闪存内部能并行工作的单元，可以想成一条单车道。写入（编程）一次约 100 µs，本轮模型里写入不能暂停，写的时候同一 plane 上的读只能在后面等。
KV 是每个请求的上下文缓存，每生成一个 token，每一层都要读写。尾页是 KV 还没写满 4 KiB 的最后一页，HBF+ 先放在每堆 40 MB 的 SRAM 里。块是 KV 的分配单位，当前每 256 token 一块，“每请求一块”就是 2048 token 一块。
诊断实验只改一项配置重新仿真，不是扫描结果。

【左上的时间线】
Maverick SHORT HBF+ 论文 0.98，native ${norm("SHORT", "HBF+")}，是最大偏差。扫描里 batch 从 3088 加到 4096，吞吐反而从 6,958 降到 6,339，所以继续加 batch 没有用。于是固定 B6048 看机理。
这是同一层、同一 rank 的时间线，横轴单位 µs。红色是被写入占用，蓝色是注意力读，灰色是前馈，粉色是后台写入。
第一行，SRAM 放不下：98% 的尾页直接写闪存，写入进入关键路径，写 KV 这一段 607 µs，之后才开始注意力读。
第二行，SRAM 放得下：写入转到后台，但同一 plane 堆了 4 次写入，跨度 401 µs，下一层第一次读要等 402 µs。
第三行，再改成每请求一块：只堆 2 次，跨度 200 µs，等待降到 203 µs。和第一行比，这一层从 1,822 µs 降到 1,315 µs。
把主机命令额度放大 100 倍，TPOT 完全不变，排除了排队原因。

【右上的柱图】
还是固定 B6048 看 TPOT。当前配置 ${f1(W.current.tpot)} ms；两项都改，也就是 SRAM 放下全部尾页加每请求一块，是 ${f1(W.both.tpot)} ms，归一化吞吐回到 ${bothNorm}；再把写入时间压到接近 0（仍是 256 token 块、大 SRAM），是 ${f1(W.no_program.tpot)} ms，接近计算下限。所以约一半 TPOT 耗在闪存写路径上。
图上没画的两个中间结果：只让 SRAM 放下全部尾页是 ${f1(W.big_sram.tpot)} ms，只改每请求一块是 ${f1(W.one_block.tpot)} ms。
当前配置这根浅色柱预测区间过宽（${f1(W.current.lower)}–${f1(W.current.upper)} ms），没有通过接受，只作估计。

【下面两张小图】
都是横轴 batch 总数、纵轴 TPOT。蓝线是 native，虚线是预测区间过宽、只作估计的点，大圆点是选中的点，灰色菱形是论文，红线是 100 ms 上限。一眼能看到 native 停在哪、论文在哪。
左图 MID HBF+：论文 1.19，native ${norm("MID", "HBF+")}。B1504 时 TPOT 49.9 ms，还有余量；但加到 B1672，新请求导入额外花 20.5–49 ms，估计吞吐不再增长。论文的 Prefill 是理想的，没有这项开销。
右图 MID HBF：论文 1.00，native ${norm("MID", "HBF")}。B2776 是 106 ms，超限；插值得到 B2604，离已测点不足一步，搜索停在通过端 B1648（65.5 ms）。吞吐只低约 3.5%，但 batch 被低估，所以第 2 页这一组的 batch 比要打折。

【和论文的区别】
论文用恒定写带宽，看不到同一 plane 上的写入堆叠，这是 Maverick SHORT HBF+ 这个点 native 偏低的主要原因。这个结论只针对本轮模型。下一页会讲，本轮的设备内读改写不符合 OCP 规范，这条路径已经按 OCP 重做，正在重跑。`);
  // Timeline panel.
  panel(s, 0.5, 1.22, 7.95, 3.36, "SHORT HBF+：同一层的时间线（µs），写入挡住后面的读", "时间线板块");
  const lanes = [
    ["当前配置", [["block", 3, 18], ["block", 18, 625], ["attn", 626, 1120], ["ffn", 1122, 1154], ["ffn", 1383, 1608], ["flush", 1722, 1822]], "98% 尾页直接写闪存\n[r:写入进入关键路径]"],
    ["大 SRAM", [["block", 3, 402], ["attn", 404, 898], ["ffn", 900, 932], ["ffn", 1076, 1301], ["flush", 1415, 1816]], "同一 plane 堆 4 次写入\n[r:下一层要等 402 µs]"],
    ["大 SRAM + 每请求一块", [["block", 3, 203], ["attn", 205, 698], ["ffn", 700, 732], ["ffn", 775, 1001], ["flush", 1115, 1315]], "只堆 2 次\n[b:这一层 1,822→1,315 µs]"],
  ];
  const X0 = 3.1, XW = 5.15, T = 2000, sx = t => X0 + t / T * XW;
  const COL = { block: "D03B3B", attn: NATIVE, ffn: PAPER, flush: "F0A3A3" };
  let cursor = X0;
  for (const [label, color, width] of [["被写入占用", COL.block, 0.8], ["读", COL.attn, 0.2], ["前馈", COL.ffn, 0.36], ["后台写入", COL.flush, 0.66]]) {
    s.addShape(pres.shapes.RECTANGLE, { x: cursor, y: 1.73, w: 0.15, h: 0.15, fill: { color }, line: { type: "none" }, objectName: "图例 " + label });
    text(s, label, cursor + 0.19, 1.67, width, 0.26, { fontSize: 10.5, color: INK2, valign: "middle", objectName: "图例文字 " + label });
    cursor += 0.19 + width + 0.12;
  }
  lanes.forEach(([name, segs, caption], i) => {
    const y = 1.98 + i * 0.77, h = 0.71;
    tile(s, 0.62, y, 7.71, h, "时间线行 " + i);
    text(s, name, 0.72, y + 0.04, 2.3, 0.24, { fontSize: 11, bold: true, valign: "middle", objectName: "时间线行名" });
    text(s, caption, 0.72, y + 0.27, 2.3, 0.42, { fontSize: 10.5, objectName: "时间线说明" });
  });
  for (const t of [0, 500, 1000, 1500, 2000]) {
    s.addShape(pres.shapes.LINE, { x: sx(t), y: 1.98, w: 0, h: 2.25, line: { color: EDGE, width: 0.75 }, objectName: "时间刻度线" });
    text(s, String(t), sx(t) - 0.3, 4.3, 0.6, 0.2, { fontSize: 9.5, color: MUTED, align: "center", valign: "middle", objectName: "时间刻度" });
  }
  lanes.forEach(([name, segs], i) => {
    const y = 2.1 + i * 0.77, h = 0.47;
    for (const [kind, a, b] of segs) box(s, sx(a), y, Math.max((b - a) / T * XW, 0.02), h, COL[kind], `时间线 ${name} ${kind} ${a}-${b}`);
    const label = i === 0 ? ["写 KV 607", 18, 625] : [`等 ${segs[0][2]}`, 3, segs[0][2]];
    text(s, label[0], sx(label[1]), y, (label[2] - label[1]) / T * XW, h, { fontSize: 10.5, color: "FFFFFF", bold: true, align: "center", valign: "middle", objectName: "时间线标注" });
    const f = segs[segs.length - 1];
    text(s, String(f[2] - f[1]), sx(f[1]), y, Math.max((f[2] - f[1]) / T * XW, 0.3), h, { fontSize: 10.5, color: INK, bold: true, align: "center", valign: "middle", objectName: "时间线标注" });
  });
  // TPOT panel.
  panel(s, 8.65, 1.22, 4.18, 3.36, "约一半 TPOT 耗在闪存写路径", "TPOT 板块");
  const rows = [["当前配置", W.current], ["两项都改", W.both], ["写入耗时≈0", W.no_program]];
  s.addChart(pres.charts.BAR, [{ name: "TPOT (ms)", labels: rows.map(r => r[0]), values: rows.map(r => +r[1].tpot.toFixed(1)) }], {
    x: 8.7, y: 1.64, w: 4.08, h: 2.1, barDir: "bar", barGapWidthPct: 40, chartColors: [NATIVE_LIGHT, NATIVE, NATIVE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11.5, dataLabelFontSize: 12, catAxisLabelColor: INK, dataLabelColor: INK,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisHidden: true, valAxisLineShow: false, showLegend: false,
    valAxisMinVal: 0, valAxisMaxVal: 130, catAxisOrientation: "maxMin",
    altText: "B6048 的 TPOT：" + rows.map(r => `${r[0]} ${f1(r[1].tpot)} ms`).join("，"),
  });
  text(s, `B6048 的 TPOT（ms）。两项 = SRAM 放下全部尾页 + 每请求一块，归一化回到 [b:${bothNorm}]。\n浅色：预测区间过宽，只作估计。`, 8.79, 3.74, 3.9, 0.78, { fontSize: 10.5, color: INK2, valign: "middle", objectName: "柱图说明" });
  // Two curve panels (charts from finish_deck.py).
  [[0.5, "MID HBF+：导入开销让吞吐不再涨", "1.19", norm("MID", "HBF+"), "B1504 还有余量；到 B1672，新请求导入多花 20.5–49 ms，吞吐不再涨。\n虚线只作估计。"],
   [6.75, "MID HBF：搜索停在通过端，batch 被低估", "1.00", norm("MID", "HBF"), "B2776 已超限，插值点离它不足一步，搜索停在 B1648。\n吞吐只少约 3.5%。"]].forEach(([x, title, paper, native, detail]) => {
    panel(s, x, 4.72, 6.08, 2.04, title, "案例板块 " + title);
    text(s, `论文 ${paper} → native [r:${native}]`, x + 3.95, 5.14, 2.05, 0.3, { fontSize: 12, bold: true, valign: "middle", objectName: "案例数值" });
    text(s, detail, x + 3.95, 5.46, 2.05, 1.24, { fontSize: 10.5, color: INK2, objectName: "案例说明" });
  });
}
// ---------------------------------------------------------------- 4 verdict and next steps
{
  const mid = m => get(m, "MID", "HBF+").fig3_gain / get(m, "MID", "HBF").fig3_gain;
  const midGain = Math.round(((mid(l3) + mid(mav)) / 2 - 1) * 100);
  const trend = field => WL.map(w => (get(l3, w, "HBF+")[field] + get(mav, w, "HBF+")[field]) / 2);
  const workloads = MODELS.flatMap(([m]) => WL.map(w => [m, w]));
  const paperAbove = workloads.filter(([m, w]) => get(m, w, "HBF+").paper_norm_tps > 1), nativeAbove = workloads.filter(([m, w]) => get(m, w, "HBF+").native_norm_tps > 1);
  const swapped = paperAbove.filter(p => !nativeAbove.includes(p)).length;
  const programShare = Math.round((1 - W.no_program.tpot / W.current.tpot) * 100);
  const s = page("问题 4：论文的主张成立吗？接下来做什么？", "论文主张[b:方向都成立]，但[r:幅度偏小]；OCP 控制规范的重跑已在进行",
    "只检验 8 GPU。batch 增益 = 全部实测点中 TPOT 上界 ≤ 100 ms 的最大 batch/GPU ÷ 同负载 HBM4。论文 HBM4 的绝对值是取整标注。\n论文 KV Write 只统计新请求导入，和 native 的写入诊断不是同一个量，只比方向。",
    `【这一页回答】
论文的主张是否成立，以及接下来做什么。答案是：方向都成立，幅度普遍偏小；按 OCP 控制规范的全量重跑已经在进行。

【颜色约定】
灰色是论文，蓝色是 native。蓝字表示成立，红字表示幅度偏小或有差异。这里的 batch 增益用最大 batch：全部实测点中 TPOT 上界不超过 100 ms 的最大 batch/GPU，再除以 HBM4。

【左上：HBF 系比 HBM4 大，CONV 系比 HBM4 小】
竖线是 HBM4 = 1。native 的 HBF ${range(["HBF"], "fig3_gain")}×、HBF+ ${range(["HBF+"], "fig3_gain")}×，论文是 HBF ${range(["HBF"], "paper_norm_batch")}×、HBF+ ${range(["HBF+"], "paper_norm_batch")}×。方向相同，蓝条都比灰条短。405B SHORT 的 HBM4 最大 batch 是 TP8 B941，即 117.6/GPU，所以 HBF 在这个负载上是 0.98，等于打平，蓝条起点压在竖线上。CONV 系 native ${range(["CONV", "CONV+"], "fig3_gain")}×，论文 ${range(["CONV", "CONV+"], "paper_norm_batch")}×，都在竖线左边，成立。

【中上：上下文越长，HBF+ 增益越大】
数字是 HBF+ 在两个模型上的平均。两边都是短 < 中 < 长，native 幅度更小。分开看，native 有两处不单调：405B MID HBF+ 是 1.48，低于 SHORT 的 1.59；Maverick MID HBF 是 1.38，低于 SHORT 的 1.56。这两点搜索都停在通过端，比值被低估。

【左下：另外三条主张】
第一条，HBF+ 的 batch 胜过 HBF：两边都是 5/6 个负载，唯一例外都是 Maverick SHORT，完全一致。MID 平均增益论文 +24%，native +${midGain}%。这个 +24% 是用图上读数算的，和论文原文一致，也说明读数可靠。
第二条，HBF+ 的吞吐胜过 HBM4：两边都是 ${paperAbove.length}/6，但有 ${swapped} 个负载互换：论文里是 Maverick MID 胜出，native 里是 405B SHORT 胜出。第 1 页的交叉图也画了这两对。
第三条，改善 KV 写入能放大 HBF 优势：方向成立。论文的 KV Write 只统计新请求导入，占 Llama4 时间 5–13.9%；native 是诊断实验，写入耗时接近 0 时 TPOT 从约 100 降到 47 ms，约少 ${programShare}%。两个不是同一个量，只比方向。

【右上：下一步】
第一步已完成：NAND 的控制行为改为按 OCP HBF v0.7，几何和时序仍按论文 Table I。原因是本轮的设备内读改写其实不符合 OCP，也没有计入擦除，而这正是第 3 页最大偏差所在的路径。代码已提交。
第二步进行中：30 个点的全量重跑在 exploring_hbf_native_20261103 目录。目前只有单点：Maverick SHORT HBF+ B3088 的 TPOT 从 55.48 ms 降到 34.22 ms。这只是一个点，不下结论。
第三步：全量结果出来后重做这份对照。还需要确认规范没有规定的产品参数：块 1 MiB、擦除 2 ms、全 plane 同时擦除的功耗。如果只有 N 组能同时擦除，损失约为 N 倍。

【右下：这份结果的边界】
墙钟没有影响结果：可能改变结果的墙钟受限布局 0 个，因墙钟被拒的点 0 个，最长单点 ${f0(DATA.wall_max)} s，上限 21,600 s。
4 个选中点受预测区间限制，是下界：405B CONV MID、405B HBF SHORT、Maverick CONV SHORT、Maverick HBF+ MID。把被拒点的估计值代入，排序仍是 ${DATA.order_estimate}/60；全部取乐观端是 ${DATA.order_optimistic}/60，结论不变。
论文 HBM4 的绝对值是图上取整的标注，读数精度有限。
本轮 native 也有自己的缺口：NAND 没有擦除和挂起，HBM 是线性交织，没有 XOR 哈希，GPU 用 roofline 模型。
这一轮没有检验 GPU 数量扩展和最小 GPU 需求、时延分解、SLO 敏感性、NAND 寿命，也就是论文 Fig. 5 到 7。`);
  // Top row: two pictures of the batch-gain claims, then next steps.
  panel(s, 0.5, 1.22, 4.2, 3.08, "HBF 系比 HBM4 大，CONV 系比 HBM4 小", "主张一");
  text(s, "[b:成立]，[r:幅度偏小]（batch 增益，竖线 = HBM4）", 0.62, 1.65, 3.96, 0.26, { fontSize: 11, valign: "middle", objectName: "主张一判定" });
  const scale = v => 1.36 + v / 5.5 * 3.2;
  segment(s, scale(1), 1.94, scale(1), 3.96, INK, 1.25, "HBM4 竖线");
  [["HBF+", ["HBF+"]], ["HBF", ["HBF"]], ["CONV 系", ["CONV", "CONV+"]]].forEach(([label, hw], i) => {
    const y = 1.98 + i * 0.66;
    text(s, label, 0.62, y, 0.7, 0.5, { fontSize: 11, bold: true, valign: "middle", objectName: "主张一行名" });
    rangeBar(s, scale, y, 0.24, span(hw, "paper_norm_batch"), PAPER, INK, `论文 ${range(hw, "paper_norm_batch")}`, "论文增益 " + label, scale(1) + 0.06);
    rangeBar(s, scale, y + 0.26, 0.24, span(hw, "fig3_gain"), NATIVE, "FFFFFF", `native ${range(hw, "fig3_gain")}`, "native 增益 " + label, scale(1) + 0.06);
  });
  for (const t of [0, 1, 2, 3, 4, 5]) text(s, String(t), scale(t) - 0.2, 3.98, 0.4, 0.2, { fontSize: 10, color: t === 1 ? INK : MUTED, align: "center", valign: "middle", objectName: "主张一刻度" });
  panel(s, 4.86, 1.22, 3.84, 3.08, "上下文越长，HBF+ 增益越大", "主张二");
  text(s, "[b:成立]，[r:幅度偏小]（两个模型平均）", 4.98, 1.65, 3.6, 0.26, { fontSize: 11, valign: "middle", objectName: "主张二判定" });
  s.addChart(pres.charts.BAR, [{ name: "论文", labels: WL, values: trend("paper_norm_batch").map(v => +v.toFixed(2)) }, { name: "native", labels: WL, values: trend("fig3_gain").map(v => +v.toFixed(2)) }], {
    x: 4.9, y: 1.92, w: 3.76, h: 2.34, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [PAPER, NATIVE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", catAxisLabelFontSize: 10.5, valAxisLabelFontSize: 9, dataLabelFontSize: 10.5, catAxisLabelColor: INK2, valAxisLabelColor: MUTED, dataLabelColor: INK,
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisLineShow: false, showLegend: true, legendPos: "t", legendFontSize: 10.5, legendColor: INK2, valAxisMinVal: 0, valAxisMaxVal: 6.5,
    altText: `HBF+ 平均 batch 增益，短中长：论文 ${trend("paper_norm_batch").map(f2).join("、")}；native ${trend("fig3_gain").map(f2).join("、")}`,
  });
  panel(s, 8.86, 1.22, 3.97, 3.08, "下一步", "下一步");
  const steps = [
    ["✓", NATIVE, "[b:控制行为改按 OCP HBF v0.7]", "几何和时序仍按 Table I，代码已提交"],
    ["2", INK2, "全量重跑：进行中", "单点 B3088 的 TPOT [b:55.48→34.22 ms]，不下结论"],
    ["3", PAPER, "重做这份论文对照", "[r:待确认]：块 1 MiB、擦除 2 ms、同擦功耗"],
  ];
  segment(s, 9.15, 1.86, 9.15, 3.5, PAPER, 1.25, "步骤连线");
  steps.forEach(([mark, color, title, detail], i) => {
    const y = 1.7 + i * 0.84;
    s.addShape(pres.shapes.OVAL, { x: 9.0, y, w: 0.3, h: 0.3, fill: { color }, line: { color: "FFFFFF", width: 1.5 }, objectName: "步骤圆点" });
    text(s, mark, 9.0, y, 0.3, 0.3, { fontSize: 11, bold: true, color: "FFFFFF", align: "center", valign: "middle", objectName: "步骤序号" });
    text(s, title, 9.4, y, 3.34, 0.3, { fontSize: 12, bold: true, valign: "middle", objectName: "步骤名称" });
    text(s, detail, 9.4, y + 0.3, 3.34, 0.46, { fontSize: 10.5, color: INK2, objectName: "步骤说明" });
  });
  // Bottom row: the remaining claims as number tiles, then the limits of this result.
  panel(s, 0.5, 4.46, 8.2, 2.3, "另外三条主张", "其余主张");
  [["[b:5/6]", "HBF+ 的 batch 胜过 HBF", "[b:两边相同]\n例外都是 Maverick SHORT"],
   [`${paperAbove.length}/6`, "HBF+ 的吞吐胜过 HBM4", `[b:数量相同]\n[r:${swapped} 个负载互换]`],
   ["[b:100→47 ms]", "写入耗时≈0 时 native 的 TPOT", "[b:方向成立]\n论文 KV Write 占 5–13.9%"]].forEach(([value, claim, verdict], i) => {
    const x = 0.62 + i * 2.68;
    tile(s, x, 4.94, 2.6, 1.72, "主张块 " + i);
    text(s, value, x + 0.06, 5.0, 2.48, 0.56, { fontSize: i === 2 ? 18 : 22, bold: true, align: "center", valign: "middle", objectName: "主张数字" });
    text(s, claim, x + 0.08, 5.58, 2.44, 0.28, { fontSize: 11, bold: true, align: "center", valign: "middle", objectName: "主张内容" });
    text(s, verdict, x + 0.08, 5.88, 2.44, 0.7, { fontSize: 10.5, color: INK2, align: "center", objectName: "主张判定" });
  });
  panel(s, 8.86, 4.46, 3.97, 2.3, "这份结果的边界", "结果边界");
  [["[b:0]", "个点的结果受墙钟影响"], [`[b:${DATA.order_estimate}–${DATA.order_optimistic}/60]`, "代入被拒点估计后，排序相同"], ["4", "个选中点受预测区间限制，是下界"], ["[r:未检验]", "论文 Fig. 5–7；本轮 NAND 无擦除"]].forEach(([value, label], i) => {
    const y = 4.9 + i * 0.45;
    text(s, value, 8.92, y, 1.14, 0.42, { fontSize: i === 0 || i === 2 ? 18 : 13, bold: true, align: "center", valign: "middle", objectName: "结果边界数字" });
    text(s, label, 10.1, y, 2.66, 0.42, { fontSize: 11, valign: "middle", objectName: "结果边界说明" });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
