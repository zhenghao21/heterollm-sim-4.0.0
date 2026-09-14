"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const appSource = fs.readFileSync(appPath, "utf8");
const I18n = require(path.join(webui, "ui-i18n.js"));
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

const CJK = /[\u3400-\u9fff]/u;

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  tokens() {
    return new Set(String(this.element.className || "").split(/\s+/u).filter(Boolean));
  }

  write(tokens) {
    this.element.className = Array.from(tokens).join(" ");
  }

  add(...names) {
    const tokens = this.tokens();
    names.forEach((name) => tokens.add(String(name)));
    this.write(tokens);
  }

  remove(...names) {
    const tokens = this.tokens();
    names.forEach((name) => tokens.delete(String(name)));
    this.write(tokens);
  }

  toggle(name, force) {
    const tokens = this.tokens();
    const value = String(name);
    const enabled = force === undefined ? !tokens.has(value) : Boolean(force);
    if (enabled) tokens.add(value);
    else tokens.delete(value);
    this.write(tokens);
    return enabled;
  }

  contains(name) {
    return this.tokens().has(String(name));
  }
}

function decodeAttribute(value) {
  return String(value)
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">");
}

class FakeElement {
  constructor(tagName = "div", ownerDocument = null) {
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.id = "";
    this.className = "";
    this.classList = new FakeClassList(this);
    this.dataset = {};
    this.style = {
      left: "",
      top: "",
      setProperty(name, value) { this[name] = String(value); },
      removeProperty(name) { delete this[name]; },
    };
    this.attributes = {};
    this.children = [];
    this.parentNode = null;
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.open = false;
    this.isConnected = true;
    this.removed = false;
    this.value = "";
    this.focused = false;
    this._textContent = "";
    this._innerHTML = "";
  }

  set textContent(value) {
    this._textContent = String(value ?? "");
    this.children.forEach((child) => { child.parentNode = null; });
    this.children = [];
  }

  get textContent() {
    return this._textContent;
  }

  set innerHTML(value) {
    this._innerHTML = String(value ?? "");
    this.children.forEach((child) => { child.parentNode = null; });
    this.children = [];

    // The toast test only needs the close control to be represented as a DOM
    // child. Keep the parser intentionally small so this harness does not
    // become a second HTML implementation.
    const closeTag = this._innerHTML.match(/<button\b[^>]*class\s*=\s*(["'])[^"']*toast-close[^"']*\1[^>]*>/iu);
    if (closeTag) {
      const button = new FakeElement("button", this.ownerDocument);
      button.className = "toast-close";
      for (const match of closeTag[0].matchAll(/([:\w-]+)\s*=\s*(["'])(.*?)\2/gu)) {
        button.setAttribute(match[1], decodeAttribute(match[3]));
      }
      this.append(button);
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }

  get firstElementChild() {
    return this.children[0] || null;
  }

  append(...items) {
    items.filter(Boolean).forEach((item) => {
      item.parentNode = this;
      item.ownerDocument ||= this.ownerDocument;
      this.children.push(item);
    });
  }

  appendChild(item) {
    this.append(item);
    return item;
  }

  remove() {
    this.removed = true;
    this.isConnected = false;
    if (this.parentNode) {
      this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
      this.parentNode = null;
    }
  }

  addEventListener(type, listener) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(listener);
    this.listeners.set(type, callbacks);
  }

  dispatch(type, event = {}) {
    const payload = {
      type,
      target: this,
      currentTarget: this,
      preventDefault() {},
      stopPropagation() {},
      ...event,
    };
    for (const listener of this.listeners.get(type) || []) listener(payload);
    return payload;
  }

  focus() {
    this.focused = true;
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }

  setAttribute(name, value) {
    const key = String(name);
    const normalized = String(value);
    this.attributes[key] = normalized;
    if (key === "id") this.id = normalized;
    if (key.startsWith("data-")) {
      const datasetKey = key.slice(5).replace(/-([a-z])/gu, (_match, letter) => letter.toUpperCase());
      this.dataset[datasetKey] = normalized;
    }
  }

  getAttribute(name) {
    const key = String(name);
    return Object.hasOwn(this.attributes, key) ? this.attributes[key] : null;
  }

  hasAttribute(name) {
    return Object.hasOwn(this.attributes, String(name));
  }

  removeAttribute(name) {
    delete this.attributes[String(name)];
  }

  descendants() {
    const result = [];
    const visit = (node) => {
      node.children.forEach((child) => {
        result.push(child);
        visit(child);
      });
    };
    visit(this);
    return result;
  }

  matches(selector) {
    const source = String(selector).trim();
    if (source.includes(",")) return source.split(",").some((part) => this.matches(part));
    const tag = source.match(/^[a-z][\w-]*/iu)?.[0];
    if (tag && this.tagName.toLowerCase() !== tag.toLowerCase()) return false;
    const id = source.match(/#([\w-]+)/u)?.[1];
    if (id && this.id !== id) return false;
    for (const className of source.matchAll(/\.([\w-]+)/gu)) {
      if (!this.classList.contains(className[1])) return false;
    }
    for (const attribute of source.matchAll(/\[([:\w-]+)(?:\s*=\s*["']?([^\]"']+)["']?)?\]/gu)) {
      if (!this.hasAttribute(attribute[1])) return false;
      if (attribute[2] !== undefined && this.getAttribute(attribute[1]) !== attribute[2]) return false;
    }
    return Boolean(tag || id || source.startsWith(".") || source.startsWith("[") || source === "*");
  }

  querySelector(selector) {
    return this.descendants().find((node) => node.matches(selector)) || null;
  }

  querySelectorAll(selector) {
    return this.descendants().filter((node) => node.matches(selector));
  }

  getBoundingClientRect() {
    return { left: 0, top: 0, right: 100, bottom: 40, width: 100, height: 40 };
  }
}

class FakeDialog extends FakeElement {
  showModal() {
    this.open = true;
  }

  close(returnValue = "") {
    this.open = false;
    this.returnValue = returnValue;
    this.dispatch("close", { target: this });
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
    this.listeners = new Map();
    this.elements = new Map();
    this.documentElement = new FakeElement("html", this);
    this.documentElement.dataset = {};
    this.documentElement.clientWidth = 1280;
    this.documentElement.clientHeight = 720;
    this.body = new FakeElement("body", this);
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }

  register(element) {
    if (element.id) this.elements.set(element.id, element);
    return element;
  }

  getElementById(id) {
    return this.elements.get(String(id)) || null;
  }

  querySelector(selector) {
    return this.body.querySelector(selector) || this.documentElement.querySelector(selector);
  }

  querySelectorAll(selector) {
    return [...this.body.querySelectorAll(selector), ...this.documentElement.querySelectorAll(selector)];
  }

  addEventListener(type, listener) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(listener);
    this.listeners.set(type, callbacks);
  }

  dispatchEvent(event) {
    for (const listener of this.listeners.get(event.type) || []) listener(event);
    return true;
  }

  execCommand() {
    return true;
  }
}

class FakeMutationObserver {
  constructor() {}
  observe() {}
  disconnect() {}
}

class FakeCustomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
}

function makeTimerHarness() {
  const timers = [];
  const setTimeoutImpl = (callback, delay) => {
    const id = timers.length + 1;
    timers.push({ callback, delay, id });
    return id;
  };
  const clearTimeoutImpl = (id) => {
    const timer = timers.find((item) => item.id === id);
    if (timer) timer.cleared = true;
  };
  return { timers, setTimeoutImpl, clearTimeoutImpl };
}

function loadApp() {
  const document = new FakeDocument();
  const timerHarness = makeTimerHarness();
  const window = {
    clearTimeout: timerHarness.clearTimeoutImpl,
    setTimeout: timerHarness.setTimeoutImpl,
    addEventListener() {},
    removeEventListener() {},
    innerHeight: 720,
    innerWidth: 1280,
  };
  window.window = window;
  window.document = document;

  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    CustomEvent: FakeCustomEvent,
    document,
    HTMLElement: FakeElement,
    Intl,
    Map,
    ModelGraphCore,
    MutationObserver: FakeMutationObserver,
    NodeFilter: { SHOW_TEXT: 4 },
    Option: class Option {},
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    URL,
    URLSearchParams,
    UiI18n: I18n,
    clearTimeout: timerHarness.clearTimeoutImpl,
    console,
    fetch: async () => ({ ok: true, json: async () => ({}) }),
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    navigator: { clipboard: { async writeText() {} } },
    requestAnimationFrame(callback) { callback(0); return 1; },
    setTimeout: timerHarness.setTimeoutImpl,
    structuredClone,
    window,
  });

  vm.runInContext(`${appSource}
;globalThis.__pass04 = {
  state,
  dom,
  bindModalDialogLifecycle,
  createTopologyNode,
  openSettingsDialog,
  restoreDialogFocus,
  showModalDialog,
  syncSettingsForm,
  toast,
  updateTopologyNode,
  updateUiSettings,
  validateScenario,
};`, context, { filename: appPath });
  return { context, document, timers: timerHarness.timers, ui: context.__pass04 };
}

function setVmBinding(context, name, value) {
  context.__pass04Binding = value;
  vm.runInContext(`${name} = globalThis.__pass04Binding;`, context, { filename: appPath });
}

function setLanguage(language, ui) {
  I18n.setLanguage(language, null);
  ui.state.settings.language = language;
}

function latestToast(toasts) {
  assert.ok(toasts.length, "expected at least one toast");
  return toasts[toasts.length - 1];
}

function sourceBetween(startMarker, endMarker) {
  const start = appSource.indexOf(startMarker);
  const end = appSource.indexOf(endMarker, start + startMarker.length);
  assert.ok(start >= 0, `missing source marker: ${startMarker}`);
  assert.ok(end > start, `missing source end marker: ${endMarker}`);
  return appSource.slice(start, end);
}

function assertUiTextPair(body, chineseText, label) {
  const index = body.indexOf(chineseText);
  assert.ok(index >= 0, `${label} must remain identifiable in its source path`);
  const nearby = body.slice(Math.max(0, index - 80), index + chineseText.length);
  assert.match(nearby, /uiText\s*\(/u, `${label} must pass its UI-authored Chinese copy through uiText`);
}

test("topology node keeps the component ID visible while localizing its accessible label", () => {
  const harness = loadApp();
  const { document, ui } = harness;
  const node = new FakeElement("button", document);
  const title = new FakeElement("span", document);
  title.className = "node-title";
  node.append(title);
  ui.state.nodePositions = { gpu0: { x: 12, y: 24 } };

  setLanguage("zh-CN", ui);
  ui.updateTopologyNode(node, { component_id: "gpu0", kind: "gpu" });
  const zhLabel = node.getAttribute("aria-label");
  assert.equal(title.textContent, "gpu0");
  assert.match(zhLabel, /gpu0/u);
  assert.match(zhLabel, /通用计算（GPU）/u);
  assert.match(zhLabel, /Ctrl.*Command.*多选/u);

  setLanguage("en", ui);
  ui.updateTopologyNode(node, { component_id: "gpu0", kind: "gpu" });
  const enLabel = node.getAttribute("aria-label");
  assert.equal(title.textContent, "gpu0", "language changes must not replace the visible component ID");
  assert.match(enLabel, /gpu0/u);
  assert.match(enLabel, /GPU Compute/u);
  assert.doesNotMatch(enLabel, CJK, "English topology accessibility copy must not retain Chinese chrome");
  assert.notEqual(enLabel, zhLabel);

  setLanguage("zh-CN", ui);
  ui.updateTopologyNode(node, { component_id: "gpu0", kind: "gpu" });
  assert.equal(node.getAttribute("aria-label"), zhLabel, "switching back must restore the Chinese label");
  assert.equal(title.textContent, "gpu0");
});

test("toast close control follows the active language and removes its toast", () => {
  const harness = loadApp();
  const { document, ui } = harness;
  const toastRegion = new FakeElement("section", document);
  ui.dom.toastRegion = toastRegion;

  setLanguage("zh-CN", ui);
  ui.toast("测试通知", "保留的通知正文", "info", 10_000);
  const zhToast = toastRegion.children.at(-1);
  const zhClose = zhToast?.querySelector(".toast-close");
  assert.ok(zhClose, "toast must expose a close button");
  assert.equal(zhToast.getAttribute("role"), "status");
  assert.equal(zhToast.getAttribute("aria-atomic"), "true");
  assert.equal(zhClose.getAttribute("aria-label"), "关闭通知");

  setLanguage("en", ui);
  ui.toast("Test notification", "Backend-owned prose is not translated by this test", "info", 10_000);
  const enToast = toastRegion.children.at(-1);
  const enClose = enToast?.querySelector(".toast-close");
  assert.ok(enClose, "English toast must expose a close button");
  assert.equal(enClose.getAttribute("aria-label"), "Close notification");

  enClose.dispatch("click");
  assert.equal(enToast.removed, true, "close button must remove the toast node");
  assert.equal(toastRegion.children.includes(enToast), false);
  assert.equal(zhToast.removed, false, "closing one toast must not remove a different toast");

  setLanguage("zh-CN", ui);
});

test("validation toast copy uses bilingual uiText while backend prose stays opaque", async () => {
  const validationBody = sourceBetween("async function validateScenario", "function runJobIsActive");
  for (const text of [
    "已忽略过期校验响应",
    "校验通过",
    "未发现错误",
    "校验未通过",
  ]) {
    assertUiTextPair(validationBody, text, `validation copy ${text}`);
  }
  assert.match(validationBody, /uiText\s*\(\s*["'`]([^"'`\r\n]*条警告)/u);
  assert.match(validationBody, /uiText\s*\(\s*["'`]([^"'`\r\n]*条错误)/u);

  const harness = loadApp();
  const { context, ui } = harness;
  const toasts = [];
  let response = null;
  setVmBinding(context, "toast", (...args) => { toasts.push(args); });
  setVmBinding(context, "apiRequest", async () => response);
  setVmBinding(context, "scenarioPayloadForTransport", () => ({}));
  setVmBinding(context, "scenarioRequestSnapshot", () => ({}));
  setVmBinding(context, "scenarioRequestIsCurrent", () => true);
  setVmBinding(context, "reconcileMappingFingerprint", () => {});
  for (const name of ["setBusy", "renderSteps", "renderDiagnostics", "openDiagnostics", "renderControlPlaneStatus"]) {
    setVmBinding(context, name, () => {});
  }

  ui.state.scenario = { name: "pass04" };
  ui.state.busy = false;

  response = {
    validation: {
      valid: true,
      errors: [],
      warnings: [{ message_zh: "后端保留警告", message_en: "Backend-owned warning" }],
    },
  };
  setLanguage("zh-CN", ui);
  await ui.validateScenario();
  const zhValid = latestToast(toasts);
  assert.equal(zhValid[0], "校验通过");
  assert.match(zhValid[1], /1.*警告/u);
  assert.doesNotMatch(zhValid[1], /后端保留警告|Backend-owned warning/u);

  setLanguage("en", ui);
  await ui.validateScenario();
  const enValid = latestToast(toasts);
  assert.notEqual(enValid[0], zhValid[0]);
  assert.doesNotMatch(enValid[0], CJK);
  assert.doesNotMatch(enValid[1], CJK);
  assert.match(enValid[1], /1.*warning/u);
  assert.doesNotMatch(enValid[1], /后端保留警告|Backend-owned warning/u);

  response = {
    validation: {
      valid: false,
      errors: [{ message_zh: "后端原始错误", message_en: "Backend raw error" }],
      warnings: [],
    },
  };
  setLanguage("zh-CN", ui);
  await ui.validateScenario();
  const zhInvalid = latestToast(toasts);
  assert.equal(zhInvalid[0], "校验未通过");
  assert.match(zhInvalid[1], /1.*错误/u);
  assert.doesNotMatch(zhInvalid[1], /后端原始错误|Backend raw error/u);

  setLanguage("en", ui);
  await ui.validateScenario();
  const enInvalid = latestToast(toasts);
  assert.notEqual(enInvalid[0], zhInvalid[0]);
  assert.doesNotMatch(enInvalid[0], CJK);
  assert.doesNotMatch(enInvalid[1], CJK);
  assert.match(enInvalid[1], /1.*error/u);
  assert.doesNotMatch(enInvalid[1], /后端原始错误|Backend raw error/u);

  setLanguage("zh-CN", ui);
});

test("Settings at 200% focuses the font-scale control and restores the opener on close", () => {
  const harness = loadApp();
  const { context, document, ui } = harness;
  const opener = new FakeElement("button", document);
  opener.id = "settingsButton";
  const dialog = new FakeDialog("settingsDialog", document);

  const controls = {
    fontScaleInput: new FakeElement("input", document),
    uiLanguageInput: new FakeElement("select", document),
    fontScaleNumberInput: new FakeElement("input", document),
    fontScaleValue: new FakeElement("output", document),
    customWorkspaceEnabled: new FakeElement("input", document),
    workspaceBackgroundInput: new FakeElement("input", document),
    compactLayoutInput: new FakeElement("input", document),
    topologyGridInput: new FakeElement("input", document),
    reduceMotionInput: new FakeElement("input", document),
  };
  Object.assign(ui.dom, controls, { settingsButton: opener, settingsDialog: dialog });
  ui.state.settings.fontScale = 200;
  setVmBinding(context, "renderRuntimeHealth", () => {});

  document.activeElement = opener;
  ui.bindModalDialogLifecycle();
  ui.openSettingsDialog();

  assert.equal(dialog.open, true);
  assert.equal(document.activeElement, controls.fontScaleInput, "initial focus must remain on the scale control");
  assert.equal(controls.fontScaleInput.value, "200");
  assert.equal(controls.fontScaleInput.getAttribute("aria-valuetext"), "200%");
  assert.equal(controls.fontScaleNumberInput.value, "200");
  assert.equal(controls.fontScaleValue.textContent, "200%");

  dialog.close("close");
  assert.equal(document.activeElement, opener, "closing Settings must restore focus to its trigger");
});
