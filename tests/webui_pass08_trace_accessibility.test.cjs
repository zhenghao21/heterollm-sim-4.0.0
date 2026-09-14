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
const styles = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));
const UiI18n = require(path.join(webui, "ui-i18n.js"));

function functionSource(name) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} must exist`);
  const end = app.indexOf("\nfunction ", start + 1);
  return app.slice(start, end > start ? end : app.length);
}

function tagForId(source, tagName, id) {
  const match = source.match(new RegExp(`<${tagName}\\b(?=[^>]*\\bid=["']${id}["'])[^>]*>`, "u"));
  assert.ok(match, `missing <${tagName}> with id=${id}`);
  return match[0];
}

function attributeValue(tag, name) {
  const match = tag.match(new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "u"));
  return match ? match[1] ?? match[2] : null;
}

function loadHelpers() {
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
      body: {},
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
  vm.runInContext(`${app}
;globalThis.__pass08 = {
  traceNodeNavigationTarget,
  traceNodeKeyboardDirection,
  handleTraceNodeKeydown,
  handleTraceTopologyKeydown,
};`, context, { filename: appPath });
  return context.__pass08;
}

test("trace topology exposes keyboard help, node focus targets, and the drawer relationship", () => {
  const canvas = tagForId(html, "div", "traceTopologyCanvas");
  assert.equal(attributeValue(canvas, "role"), "region");
  assert.equal(attributeValue(canvas, "aria-describedby"), "traceTopologyKeyboardHelp");
  assert.match(attributeValue(canvas, "aria-keyshortcuts"), /ArrowUp/);
  assert.match(html, /id="traceTopologyKeyboardHelp"[\s\S]*keyboard navigation does not change the simulation or layout/iu);

  const drawer = tagForId(html, "aside", "traceEventDrawer");
  assert.equal(attributeValue(drawer, "aria-labelledby"), "traceEventDetailTitle");
  const close = tagForId(html, "button", "traceDrawerCloseButton");
  assert.equal(attributeValue(close, "aria-controls"), "traceEventDrawer");
  const open = tagForId(html, "button", "traceDrawerOpenButton");
  assert.equal(attributeValue(open, "aria-controls"), "traceEventDrawer");
  assert.equal(attributeValue(open, "aria-expanded"), "true");
});

test("trace keyboard handlers are bound without changing playback or scenario state", () => {
  const binding = functionSource("bindStaticEvents");
  assert.match(binding, /dom\.traceTopologyCanvas\.addEventListener\("keydown",\s*handleTraceTopologyKeydown\)/u);
  assert.match(functionSource("renderTraceTopology"), /focusedNodeId/);
  assert.match(functionSource("renderTraceTopology"), /tabindex="0"/);
  assert.match(functionSource("renderTraceTopology"), /handleTraceNodeKeydown/);
  assert.match(functionSource("handleTraceNodeKeydown"), /traceNodeNavigationTarget/);
  assert.match(functionSource("handleTraceTopologyKeydown"), /traceTopologyScrollBy/);
  assert.match(functionSource("handleTraceNodeKeydown"), /event\.stopPropagation\(\)/);
  assert.match(functionSource("handleTraceTopologyKeydown"), /event\.preventDefault\(\)/);
  assert.match(styles, /\.playback-view \.trace-node:focus-visible\s*\{[\s\S]*outline:\s*2px solid var\(--focus\)/u);
});

test("trace spatial keyboard navigation chooses the nearest component in the requested direction", () => {
  const ui = loadHelpers();
  const layout = {
    rects: [
      { id: "center", x: 100, y: 100, width: 40, height: 40 },
      { id: "left", x: 20, y: 106, width: 40, height: 40 },
      { id: "right-far", x: 220, y: 106, width: 40, height: 40 },
      { id: "right-near", x: 175, y: 108, width: 40, height: 40 },
      { id: "above", x: 104, y: 20, width: 40, height: 40 },
      { id: "below", x: 105, y: 220, width: 40, height: 40 },
    ],
  };
  assert.equal(ui.traceNodeKeyboardDirection("ArrowLeft"), "left");
  assert.equal(ui.traceNodeKeyboardDirection("ArrowDown"), "down");
  assert.equal(ui.traceNodeNavigationTarget(layout, "center", "left"), "left");
  assert.equal(ui.traceNodeNavigationTarget(layout, "center", "right"), "right-near");
  assert.equal(ui.traceNodeNavigationTarget(layout, "center", "up"), "above");
  assert.equal(ui.traceNodeNavigationTarget(layout, "center", "down"), "below");
  assert.equal(ui.traceNodeNavigationTarget(layout, "center", "diagonal"), null);
});

test("fullscreen and drawer transitions preserve explicit focus contracts", () => {
  const fullscreen = functionSource("toggleTraceFullscreen");
  assert.match(fullscreen, /traceFullscreenPreviousFocus/);
  assert.match(fullscreen, /previousFocusId/);
  assert.match(fullscreen, /focusTraceElement\(dom\.traceTopologyCanvas\)/);
  assert.match(fullscreen, /focusTraceElement\(returnFocus\)/);
  const drawer = functionSource("setTraceDrawerOpen");
  assert.match(drawer, /aria-hidden|renderTracePlayback/);
  assert.match(drawer, /focusTraceElement\(dom\.traceDrawerCloseButton\)/);
  assert.match(drawer, /focusTraceElement\(dom\.traceDrawerOpenButton\)/);
  const render = functionSource("renderTracePlayback");
  assert.match(render, /traceEventDrawer\.setAttribute\("aria-hidden",\s*String\(!drawerOpen\)\)/);
  assert.match(render, /traceDrawerOpenButton\.setAttribute\("aria-expanded",\s*String\(drawerOpen\)\)/);
});
