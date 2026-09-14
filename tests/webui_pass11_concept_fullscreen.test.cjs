"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");
const UiI18n = require("../src/heterollm_sim/webui/ui-i18n.js");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const app = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
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
  assert.fail("unterminated block");
}

function functionSource(name) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} must exist`);
  const open = app.indexOf("{", start);
  assert.ok(open > start, `${name} must have a body`);
  return app.slice(start, open + 1) + blockFromOpeningBrace(app, open) + "}";
}

function sourceBetweenMarkers(startMarker, endMarker) {
  const start = app.indexOf(startMarker);
  assert.ok(start >= 0, `missing source marker: ${startMarker}`);
  const end = app.indexOf(endMarker, start + startMarker.length);
  assert.ok(end > start, `missing source end marker: ${endMarker}`);
  return app.slice(start, end);
}

function tagForId(source, tagName, id) {
  const pattern = `<${tagName}\\b(?=[^>]*\\bid=["']${escapeRegExp(id)}["'])[^>]*>`;
  const match = source.match(new RegExp(pattern, "u"));
  assert.ok(match, `missing <${tagName}> with id=${id}`);
  return match[0];
}

function attributeValue(tag, name) {
  const pattern = `\\b${escapeRegExp(name)}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`;
  const match = tag.match(new RegExp(pattern, "u"));
  return match ? match[1] ?? match[2] : null;
}

class FakeClassList {
  constructor(element) {
    this.element = element;
    this.tokens = new Set();
  }

  add(...names) {
    names.forEach((name) => this.tokens.add(String(name)));
  }

  remove(...names) {
    names.forEach((name) => this.tokens.delete(String(name)));
  }

  toggle(name, force) {
    const token = String(name);
    const enabled = force === undefined ? !this.tokens.has(token) : Boolean(force);
    if (enabled) this.tokens.add(token);
    else this.tokens.delete(token);
    return enabled;
  }

  contains(name) {
    return this.tokens.has(String(name));
  }

  toString() {
    return Array.from(this.tokens).join(" ");
  }
}

class FakeElement {
  constructor(tagName = "div", ownerDocument = null, textContent = "") {
    this.tagName = String(tagName).toUpperCase();
    this.nodeName = this.tagName;
    this.ownerDocument = ownerDocument;
    this.id = "";
    this.dataset = {};
    this.attributes = new Map();
    this.classList = new FakeClassList(this);
    this.children = [];
    this.parentNode = null;
    this.parentElement = null;
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.isConnected = true;
    this.focused = false;
    this.value = "";
    this.textContent = textContent;
    this.innerHTML = "";
    this.style = {
      setProperty(name, value) { this[name] = String(value); },
      removeProperty(name) { delete this[name]; },
    };
    this._querySelectorAll = null;
  }

  appendChild(child) {
    child.parentNode = this;
    child.parentElement = this;
    child.ownerDocument ||= this.ownerDocument;
    this.children.push(child);
    return child;
  }

  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }

  insertAdjacentHTML(_position, markup) {
    this.innerHTML += String(markup);
  }

  setAttribute(name, value = "") {
    const key = String(name);
    const stringValue = String(value);
    this.attributes.set(key, stringValue);
    if (key === "id") this.id = stringValue;
    if (key.startsWith("data-")) {
      const datasetKey = key.slice(5).replace(/-([a-z])/gu, (_match, letter) => letter.toUpperCase());
      this.dataset[datasetKey] = stringValue;
    }
  }

  getAttribute(name) {
    return this.attributes.has(String(name)) ? this.attributes.get(String(name)) : null;
  }

  hasAttribute(name) {
    return this.attributes.has(String(name));
  }

  removeAttribute(name) {
    this.attributes.delete(String(name));
  }

  addEventListener(type, listener) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(listener);
    this.listeners.set(type, callbacks);
  }

  focus() {
    this.focused = true;
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }

  contains(node) {
    if (node === this) return true;
    return this.children.some((child) => child === node || child.contains?.(node));
  }

  closest(selector) {
    if (this.matches(selector)) return this;
    return this.parentElement?.closest?.(selector) || null;
  }

  matches(selector) {
    const source = String(selector || "").trim();
    if (!source) return false;
    if (source.includes(",")) return source.split(",").some((part) => this.matches(part));
    if (source === "*") return true;
    const tag = source.match(/^[a-z][\w-]*/iu)?.[0];
    if (tag && this.tagName.toLowerCase() !== tag.toLowerCase()) return false;
    const id = source.match(/#([\w-]+)/u)?.[1];
    if (id && this.id !== id) return false;
    for (const className of source.matchAll(/\.([\w-]+)/gu)) {
      if (!this.classList.contains(className[1])) return false;
    }
    for (const attr of source.matchAll(/\[([:\w-]+)(?:\s*=\s*["']?([^\]"']+)["']?)?\]/gu)) {
      if (!this.hasAttribute(attr[1])) return false;
      if (attr[2] !== undefined && this.getAttribute(attr[1]) !== attr[2]) return false;
    }
    return Boolean(tag || id || source.startsWith(".") || source.startsWith("["));
  }

  querySelectorAll(selector) {
    if (this._querySelectorAll) return this._querySelectorAll(selector);
    const descendants = [];
    const visit = (node) => {
      node.children.forEach((child) => {
        descendants.push(child);
        visit(child);
      });
    };
    visit(this);
    return descendants.filter((node) => node.matches(selector));
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
    this.listeners = new Map();
    this.documentElement = new FakeElement("html", this);
    this.body = new FakeElement("body", this);
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }

  addEventListener(type, listener, options) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push({ listener, options });
    this.listeners.set(type, callbacks);
  }

  getElementById() {
    return null;
  }

  querySelectorAll(selector) {
    return this.body.querySelectorAll(selector);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

function loadRuntime() {
  UiI18n.setLanguage("zh-CN", null);
  const document = new FakeDocument();
  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    CustomEvent: class CustomEvent {},
    document,
    HTMLElement: FakeElement,
    HTMLInputElement: class HTMLInputElement extends FakeElement {},
    HTMLSelectElement: class HTMLSelectElement extends FakeElement {},
    HTMLTextAreaElement: class HTMLTextAreaElement extends FakeElement {},
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
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    matchMedia: () => ({ matches: false }),
    requestAnimationFrame(callback) { callback(0); return 1; },
    setTimeout,
    structuredClone,
  });
  context.window = context;
  vm.runInContext(`${app}
;globalThis.__pass11 = {
  state,
  dom,
  CONCEPT_HELP_COVERAGE_BY_VIEW,
  emptyTracePlaybackState,
  hydrateConceptHelp,
  syncTraceFullscreenSemantics,
  clearTraceFullscreenState,
  renderTracePlayback,
  handleTraceFullscreenKeydown,
  handleTraceFullscreenFocusIn,
  traceFullscreenFocusableElements,
};`, context, { filename: appPath });
  return { context, document, ui: context.__pass11 };
}

function installPlaybackRenderDom(ui) {
  Object.assign(ui.dom, {
    playbackCount: new FakeElement("span"),
    playbackStatus: new FakeElement("span"),
    playbackFidelity: new FakeElement("strong"),
    traceContent: new FakeElement("section"),
    traceEmpty: new FakeElement("div"),
    traceTopologyPanel: new FakeElement("section"),
    traceFullscreenButton: new FakeElement("button"),
    traceEventDrawer: new FakeElement("aside"),
    traceDrawerOpenButton: new FakeElement("button"),
    traceDrawerCloseButton: new FakeElement("button"),
  });
}

function keyboardEvent(key, extra = {}) {
  return {
    key,
    defaultPrevented: false,
    altKey: false,
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    prevented: false,
    stopped: false,
    preventDefault() { this.prevented = true; },
    stopPropagation() { this.stopped = true; },
    ...extra,
  };
}

test("concept coverage separates architecture, playback, and results concepts", () => {
  const { ui } = loadRuntime();
  const coverage = ui.CONCEPT_HELP_COVERAGE_BY_VIEW;
  assert.ok(coverage.architecture.includes("roofline"), "architecture coverage must include roofline");
  assert.equal(coverage.playback.includes("component_timeseries"), false, "playback coverage must not expose results-only component timeseries help");
  assert.ok(coverage.results.includes("component_timeseries"), "results coverage must include component timeseries");

  const scanHeading = tagForId(html, "h2", "architectureScanDialogTitle");
  assert.equal(attributeValue(scanHeading, "data-concept-help"), "roofline");
});

test("batch concept auto-matching only recognizes standalone batch labels", () => {
  const { document, ui } = loadRuntime();
  const root = new FakeElement("section", document);
  const batchZh = new FakeElement("span", document, "批次");
  const batchEn = new FakeElement("span", document, "Batch");
  const backendZh = new FakeElement("span", document, "批量计算后端（Batch Backend）");
  const backendEn = new FakeElement("span", document, "Batch Backend");
  [batchZh, batchEn, backendZh, backendEn].forEach((element) => root.appendChild(element));
  root._querySelectorAll = (selector) => {
    if (selector === "[data-concept-help]") return root.children.filter((element) => element.dataset.conceptHelp);
    return root.children;
  };

  ui.hydrateConceptHelp(root);

  assert.equal(batchZh.dataset.conceptHelp, "batch");
  assert.equal(batchEn.dataset.conceptHelp, "batch");
  assert.equal(backendZh.dataset.conceptHelp, undefined);
  assert.equal(backendEn.dataset.conceptHelp, undefined);
  assert.match(app, /\["batch",\s*\/\^\(\?:批次\|Batch\)\$\/iu\]/u, "batch fallback matcher must be anchored to the whole label");
});

test("trace topology static accessibility copy describes page and node navigation", () => {
  const panel = tagForId(html, "section", "traceTopologyPanel");
  assert.equal(attributeValue(panel, "role"), "region");

  const canvas = tagForId(html, "div", "traceTopologyCanvas");
  const shortcuts = attributeValue(canvas, "aria-keyshortcuts") || "";
  for (const key of ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"]) {
    assert.match(shortcuts, new RegExp(`\\b${key}\\b`, "u"), `canvas shortcuts include ${key}`);
  }

  const help = tagForId(html, "span", "traceTopologyKeyboardHelp");
  const zh = attributeValue(help, "data-i18n-zh") || "";
  const en = attributeValue(help, "data-i18n-en") || "";
  assert.match(zh, /PageUp\s*\/\s*PageDown/u);
  assert.match(en, /PageUp\s*\/\s*PageDown/u);
  assert.match(zh, /组件节点后[\s\S]*Home\s*\/\s*End/u);
  assert.match(en, /component node has focus[\s\S]*Home\s*\/\s*End/u);
});

test("trace fullscreen render state switches between region and modal dialog semantics", () => {
  const { ui } = loadRuntime();
  installPlaybackRenderDom(ui);
  ui.state.scenario = { hardware: { components: [], links: [] }, placement: {} };
  ui.state.report = null;
  ui.state.tracePlayback = ui.emptyTracePlaybackState();
  ui.state.tracePlayback.reportRef = null;
  ui.state.tracePlayback.jobIdRef = "";

  ui.state.tracePlayback.fullscreen = true;
  ui.renderTracePlayback();
  assert.equal(ui.dom.traceTopologyPanel.getAttribute("role"), "dialog");
  assert.equal(ui.dom.traceTopologyPanel.getAttribute("aria-modal"), "true");
  assert.equal(ui.dom.traceTopologyPanel.classList.contains("is-fullscreen"), true);

  ui.state.tracePlayback.fullscreen = false;
  ui.renderTracePlayback();
  assert.equal(ui.dom.traceTopologyPanel.getAttribute("role"), "region");
  assert.equal(ui.dom.traceTopologyPanel.hasAttribute("aria-modal"), false);
  assert.equal(ui.dom.traceTopologyPanel.classList.contains("is-fullscreen"), false);

  ui.syncTraceFullscreenSemantics(true);
  ui.dom.traceFullscreenButton.setAttribute("aria-pressed", "true");
  ui.clearTraceFullscreenState();
  assert.equal(ui.state.tracePlayback.fullscreen, false);
  assert.equal(ui.dom.traceTopologyPanel.getAttribute("role"), "region");
  assert.equal(ui.dom.traceTopologyPanel.hasAttribute("aria-modal"), false);
  assert.equal(ui.dom.traceFullscreenButton.getAttribute("aria-pressed"), "false");
});

test("fullscreen capture guards close on Escape, wrap Tab, and yield to native modal dialogs", () => {
  const { document, ui } = loadRuntime();
  const panel = new FakeElement("section", document);
  const canvas = new FakeElement("div", document);
  const middle = new FakeElement("button", document);
  const last = new FakeElement("button", document);
  const outside = new FakeElement("button", document);
  panel.append(canvas, middle, last);
  panel._querySelectorAll = () => [canvas, middle, last];
  Object.assign(ui.dom, {
    traceTopologyPanel: panel,
    traceTopologyCanvas: canvas,
    traceFullscreenButton: last,
    settingsDialog: { open: false },
  });
  ui.state.tracePlayback = ui.emptyTracePlaybackState();
  ui.state.tracePlayback.fullscreen = true;

  let event = keyboardEvent("Tab");
  document.activeElement = last;
  assert.equal(ui.handleTraceFullscreenKeydown(event), true);
  assert.equal(event.prevented, true);
  assert.equal(event.stopped, true);
  assert.equal(document.activeElement, canvas, "Tab on the last control wraps to the first control");

  event = keyboardEvent("Tab", { shiftKey: true });
  document.activeElement = canvas;
  assert.equal(ui.handleTraceFullscreenKeydown(event), true);
  assert.equal(event.prevented, true);
  assert.equal(document.activeElement, last, "Shift+Tab on the first control wraps to the last control");

  event = keyboardEvent("Tab");
  document.activeElement = outside;
  assert.equal(ui.handleTraceFullscreenKeydown(event), true);
  assert.equal(document.activeElement, canvas, "Tab from outside the fullscreen panel returns to the canvas fallback");

  document.activeElement = outside;
  assert.equal(ui.handleTraceFullscreenFocusIn({ target: outside }), true);
  assert.equal(document.activeElement, canvas, "focus entering outside the fullscreen panel is redirected back inside");

  ui.state.tracePlayback.fullscreen = true;
  ui.dom.settingsDialog.open = true;
  event = keyboardEvent("Escape");
  assert.equal(ui.handleTraceFullscreenKeydown(event), false);
  assert.equal(event.prevented, false);
  assert.equal(event.stopped, false);
  assert.equal(ui.state.tracePlayback.fullscreen, true, "native modals suspend fullscreen Escape trapping");

  document.activeElement = outside;
  assert.equal(ui.handleTraceFullscreenFocusIn({ target: outside }), false);
  assert.equal(document.activeElement, outside, "native modals suspend fullscreen focus trapping");

  ui.dom.settingsDialog.open = false;
  event = keyboardEvent("Escape");
  assert.equal(ui.handleTraceFullscreenKeydown(event), true);
  assert.equal(event.prevented, true);
  assert.equal(event.stopped, true);
  assert.equal(ui.state.tracePlayback.fullscreen, false);
});

test("fullscreen guards are registered in capture phase and group Escape remains available", () => {
  const binding = functionSource("bindStaticEvents");
  assert.match(
    binding,
    /document\.addEventListener\("keydown",\s*handleTraceFullscreenKeydown,\s*(?:true|\{\s*capture:\s*true\s*\})\)/u,
    "fullscreen keydown guard must run during capture",
  );
  assert.match(
    binding,
    /document\.addEventListener\("focusin",\s*handleTraceFullscreenFocusIn,\s*(?:true|\{\s*capture:\s*true\s*\})\)/u,
    "fullscreen focus guard must run during capture",
  );

  const keydown = functionSource("handleTraceFullscreenKeydown");
  assert.match(keydown, /hasOpenModalDialog\(\)/u, "fullscreen keydown guard must bypass native modal dialogs");
  assert.match(keydown, /event\.key === "Escape"[\s\S]*toggleTraceFullscreen\(false\)/u);
  assert.match(keydown, /event\.key !== "Tab"/u);
  assert.match(keydown, /event\.shiftKey[\s\S]*target = last/u);
  assert.match(keydown, /!event\.shiftKey[\s\S]*target = first/u);

  const focusin = functionSource("handleTraceFullscreenFocusIn");
  assert.match(focusin, /hasOpenModalDialog\(\)/u, "fullscreen focus guard must bypass native modal dialogs");
  assert.match(focusin, /panel\.contains\?\.\(event\.target\)/u);
  assert.match(focusin, /focusTraceElement\(traceFullscreenFallbackFocusTarget\(\)\)/u);

  const renderGroups = sourceBetweenMarkers("function renderTraceGroups", "function bindTraceLinkTooltip");
  assert.match(
    renderGroups,
    /button\.addEventListener\("keydown",\s*\(event\) => \{\s*if\s*\(!\["Enter", " "\]\.includes\(event\.key\)\)\s*return;\s*event\.preventDefault\(\);\s*event\.stopPropagation\(\);/u,
    "Escape and other non-activation keys must remain available to the fullscreen capture guard and global fallback",
  );
});
