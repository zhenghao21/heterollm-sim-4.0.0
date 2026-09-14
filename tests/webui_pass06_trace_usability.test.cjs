"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const app = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));
const UiI18n = require(path.join(webui, "ui-i18n.js"));

function functionSource(name) {
  const start = app.indexOf(`function ${name}`);
  assert.ok(start >= 0, `${name} must exist`);
  const end = app.indexOf("\nfunction ", start + 1);
  return app.slice(start, end > start ? end : app.length);
}

function tagForId(source, tagName, id) {
  const pattern = new RegExp(`<${tagName}\\b(?=[^>]*\\bid=["']${id}["'])[^>]*>`, "u");
  const match = source.match(pattern);
  assert.ok(match, `missing <${tagName}> with id=${id}`);
  return match[0];
}

function attributeValue(tag, name) {
  const pattern = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "u");
  const match = tag.match(pattern);
  return match ? match[1] ?? match[2] : null;
}

function blockFromOpeningBrace(source, openIndex) {
  assert.equal(source[openIndex], "{", "block must start at an opening brace");
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(openIndex + 1, index);
    }
  }
  assert.fail("unterminated CSS block");
}

function atRuleBlock(pattern) {
  const match = pattern.exec(css);
  assert.ok(match, `missing CSS at-rule ${pattern}`);
  const openIndex = css.indexOf("{", match.index);
  assert.ok(openIndex > match.index, `missing CSS block for ${pattern}`);
  return blockFromOpeningBrace(css, openIndex);
}

function atRuleBlockContaining(pattern, marker) {
  for (const match of css.matchAll(new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`))) {
    const openIndex = css.indexOf("{", match.index);
    assert.ok(openIndex > match.index, `missing CSS block for ${pattern}`);
    const block = blockFromOpeningBrace(css, openIndex);
    if (block.includes(marker)) return block;
  }
  assert.fail(`missing CSS at-rule ${pattern} containing ${marker}`);
}

function cssBodiesContaining(source, selectorFragment) {
  const bodies = [];
  for (const match of source.matchAll(/([^{}]+)\{([^{}]*)\}/gu)) {
    if (match[1].replace(/\s+/gu, " ").includes(selectorFragment)) bodies.push(match[2]);
  }
  return bodies;
}

function assertCssContract(source, selectorFragment, patterns) {
  const bodies = cssBodiesContaining(source, selectorFragment);
  assert.ok(bodies.length, `missing CSS selector containing ${selectorFragment}`);
  for (const pattern of patterns) {
    assert.ok(
      bodies.some((body) => pattern.test(body)),
      `no ${selectorFragment} rule matched ${pattern}`,
    );
  }
}

function loadTraceHelpers() {
  UiI18n.setLanguage("zh-CN", null);
  const sandbox = {
    AbortController,
    CSS: { escape: String },
    Intl,
    Map,
    ModelGraphCore,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    UiI18n,
    URL,
    URLSearchParams,
    clearTimeout,
    console,
    document: {
      addEventListener() {},
      documentElement: { dataset: {}, style: {} },
      getElementById() { return null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    matchMedia: () => ({ matches: false }),
    requestAnimationFrame(callback) { callback(0); return 1; },
    setTimeout,
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(`${app}\n;globalThis.__pass06 = { traceRevealScrollTarget, traceRevealScrollBehavior };`, context, { filename: appPath });
  return context.__pass06;
}

function filterLabelFor(selectId) {
  const selectIndex = html.indexOf(`id="${selectId}"`);
  assert.ok(selectIndex >= 0, `missing ${selectId}`);
  const labelStart = html.lastIndexOf("<label", selectIndex);
  const labelEnd = html.indexOf("</label>", selectIndex);
  assert.ok(labelStart >= 0 && labelEnd > selectIndex, `${selectId} must be inside a label`);
  return html.slice(labelStart, labelEnd + "</label>".length);
}

test("locate-active control is localized, registered, wired, and refreshed after canvas scrolls", () => {
  const button = tagForId(html, "button", "traceLocateActiveButton");
  assert.match(button, /\bclass=["'][^"']*\btrace-locate-active\b/u);
  assert.equal(attributeValue(button, "aria-controls"), "traceTopologyCanvas");
  assert.equal(attributeValue(button, "data-i18n-zh"), "定位当前事件");
  assert.equal(attributeValue(button, "data-i18n-en"), "Locate active event");
  assert.match(button, /\bhidden\b/u);

  const cacheDom = functionSource("cacheDom");
  assert.match(cacheDom, /"traceLocateActiveButton"/u);

  const bindStaticEvents = functionSource("bindStaticEvents");
  assert.match(bindStaticEvents, /dom\.traceLocateActiveButton\.addEventListener\("click",\s*revealTraceTopology\)/u);
  assert.match(bindStaticEvents, /dom\.traceTopologyCanvas\.addEventListener\("scroll",\s*updateTraceLocateActiveButton,\s*\{\s*passive:\s*true\s*\}\)/u);

  const updateButton = functionSource("updateTraceLocateActiveButton");
  assert.match(updateButton, /const button = dom\.traceLocateActiveButton/u);
  assert.match(updateButton, /button\.hidden\s*=\s*!visible/u);
  assert.match(updateButton, /traceTopologyPanel\?\.classList\.toggle\("has-offscreen-active",\s*visible\)/u);

  const scrollCanvas = functionSource("traceScrollCanvasTo");
  assert.match(scrollCanvas, /requestAnimationFrame\(updateTraceLocateActiveButton\)/u);
});

test("reveal target calculation is extracted, reused, and autoplay does not force reveal", () => {
  const target = functionSource("traceTopologyRevealTarget");
  assert.match(target, /traceRevealTargetRect\(/u);
  assert.match(target, /traceRevealScrollTarget\(/u);

  const updateButton = functionSource("updateTraceLocateActiveButton");
  const reveal = functionSource("revealTraceTopology");
  assert.match(updateButton, /traceTopologyRevealTarget\(\)/u);
  assert.match(reveal, /traceTopologyRevealTarget\(\)/u);
  assert.doesNotMatch(reveal, /traceRevealTargetRect\(/u, "reveal should reuse the extracted target helper");
  assert.doesNotMatch(reveal, /traceRevealScrollTarget\(/u, "reveal should reuse the extracted target helper");

  const autoplay = functionSource("traceAnimationStep");
  assert.match(autoplay, /setTraceTime\(event\.start_ns,\s*\{\s*force:\s*true,\s*eventId:\s*event\.event_id\s*\}\)/u);
  assert.doesNotMatch(autoplay, /reveal\s*:\s*true/u);
});

test("trace reveal scroll target becomes inactive after the canvas scrolls to it", () => {
  const ui = loadTraceHelpers();
  const viewport = {
    clientWidth: 300,
    clientHeight: 200,
    scrollWidth: 900,
    scrollHeight: 700,
    scrollLeft: 0,
    scrollTop: 0,
  };
  const rect = { x: 470, y: 310, width: 80, height: 50 };
  const target = ui.traceRevealScrollTarget(viewport, rect, 20);
  assert.equal(target.left, 270);
  assert.equal(target.top, 180);

  viewport.scrollLeft = target.left;
  viewport.scrollTop = target.top;
  assert.equal(ui.traceRevealScrollTarget(viewport, rect, 20), null);

  const oversized = { x: 100, y: 150, width: 500, height: 300 };
  viewport.scrollLeft = 0;
  viewport.scrollTop = 0;
  const oversizedTarget = ui.traceRevealScrollTarget(viewport, oversized, 20);
  assert.equal(oversizedTarget.left, 200);
  assert.equal(oversizedTarget.top, 200);
  viewport.scrollLeft = oversizedTarget.left;
  viewport.scrollTop = oversizedTarget.top;
  assert.equal(ui.traceRevealScrollTarget(viewport, oversized, 20), null, "an oversized target is located once the best centered position is reached");
});

test("narrow playback stacks panels while preserving canvas height and vertical scroll chaining", () => {
  const narrow = atRuleBlockContaining(/@media\s*\(max-width:\s*1100px\)/u, ".playback-view .trace-workspace");
  assertCssContract(narrow, ".playback-view .trace-workspace", [
    /grid-template-columns:\s*minmax\(0,\s*1fr\)/u,
  ]);
  assertCssContract(narrow, ".playback-view .trace-side-panel", [
    /grid-template-columns:\s*minmax\(0,\s*1fr\)/u,
  ]);

  const canvasBodies = cssBodiesContaining(narrow, ".playback-view .trace-topology-canvas");
  assert.ok(canvasBodies.length, "narrow playback canvas rule must exist");
  const canvas = canvasBodies.join("\n");
  assert.match(canvas, /min-height:\s*clamp\([^;]*dvh[^;]*\)/u);
  assert.match(canvas, /max-height:\s*min\([^;]*dvh[^;]*\)/u);
  assert.ok(
    /overscroll-behavior-y:\s*auto/u.test(canvas) || /overscroll-behavior:\s*(?:auto|contain\s+auto)\b/u.test(canvas),
    "narrow canvas must allow vertical scroll chaining while staying internally scrollable",
  );

  assertCssContract(narrow, ".playback-view .trace-event-details", [
    /max-height:\s*none/u,
    /overflow:\s*visible/u,
  ]);
});

test("large and extreme font bands keep playback detail content unbounded and canvas viewport-sized", () => {
  assertCssContract(css, ':root[data-font-band="large"] .playback-view .trace-topology-canvas', [
    /min-height:\s*min\([^;]*dvh[^;]*\)/u,
    /max-height:\s*min\([^;]*dvh[^;]*\)/u,
  ]);
  assertCssContract(css, ':root[data-font-band="extreme"] .playback-view .trace-topology-canvas', [
    /min-height:\s*min\([^;]*dvh[^;]*\)/u,
    /max-height:\s*min\([^;]*dvh[^;]*\)/u,
  ]);
  assertCssContract(css, ':root[data-font-band="large"] .playback-view .trace-event-details', [
    /max-height:\s*none/u,
    /overflow:\s*visible/u,
  ]);
  assertCssContract(css, ':root[data-font-band="extreme"] .playback-view .trace-event-details', [
    /max-height:\s*none/u,
    /overflow:\s*visible/u,
  ]);
});

test("normal font desktop density preserves the full playback topology toolbar at medium widths", () => {
  const mediumDesktop = atRuleBlock(/@media\s*\(min-width:\s*1101px\)\s*and\s*\(max-width:\s*1450px\)/u);
  assertCssContract(mediumDesktop, ':root[data-font-band="normal"] .playback-view .trace-topology-heading > div:first-child', [
    /min-width:\s*min\(100%,\s*calc\(120px\s*\*\s*var\(--layout-scale\)\)\)/u,
  ]);
  assertCssContract(mediumDesktop, ':root[data-font-band="normal"] .playback-view .trace-layout-actions', [
    /gap:\s*3px/u,
  ]);
  assertCssContract(mediumDesktop, ':root[data-font-band="normal"] .playback-view .trace-layout-actions :is(.tool-button, .button)', [
    /min-width:\s*0/u,
    /padding-inline:\s*calc\(6px\s*\*\s*var\(--layout-scale\)\)/u,
  ]);
  assertCssContract(mediumDesktop, ':root[data-font-band="normal"] .playback-view .trace-layout-actions .zoom-value', [
    /min-width:\s*3\.8em/u,
    /padding-inline:\s*2px/u,
  ]);
});

test("light themes keep active protocol strokes mixed and inactive topology readable", () => {
  assertCssContract(css, ':root:is([data-theme="ivory"], [data-theme="mist"], [data-theme="softgray"]) :is(.trace-link.is-active .trace-link-path, .trace-link-path.is-active)', [
    /--protocol-active-color:\s*color-mix\(in srgb,\s*var\(--protocol-color,\s*var\(--cyan\)\)\s*66%,\s*var\(--text\)\s*34%\)/u,
    /stroke:\s*var\(--protocol-active-color\)/u,
  ]);
  assertCssContract(css, ".trace-topology-panel.has-active-trace .trace-link:not(.is-active) .trace-link-path", [
    /opacity:\s*\.(?:4[0-9]|[5-9][0-9])/u,
  ]);
  assertCssContract(css, ".trace-topology-panel.has-active-trace .trace-node:not(.is-active):not(.is-selected)", [
    /opacity:\s*\.(?:7[5-9]|[89][0-9])/u,
  ]);
});

test("playback request, batch, and rank filters bind explicit concept help", () => {
  for (const [selectId, helpKey] of [
    ["traceRequestFilter", "request"],
    ["traceBatchFilter", "batch"],
    ["traceRankFilter", "rank"],
  ]) {
    const label = filterLabelFor(selectId);
    assert.match(label, new RegExp(`data-concept-help=["']${helpKey}["']`, "u"));
    assert.match(label, new RegExp(`<select\\b[^>]*\\bid=["']${selectId}["']`, "u"));
  }
});
