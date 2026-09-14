"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");

class FakeHTMLElement {
  constructor() {
    this.isConnected = true;
    this.listeners = {};
    this.open = false;
    this.focused = false;
  }

  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }

  dispatch(type, event = {}) {
    for (const listener of this.listeners[type] || []) listener(event);
  }

  focus() {
    this.focused = true;
  }
}

class FakeDialog extends FakeHTMLElement {
  showModal() {
    this.open = true;
  }

  close(returnValue = "") {
    this.open = false;
    this.returnValue = returnValue;
    this.dispatch("close", { target: this });
  }
}

function loadDialogHelpers() {
  const context = vm.createContext({
    AbortController,
    HTMLElement: FakeHTMLElement,
    Intl,
    ModelGraphCore,
    Promise,
    TopologyCore,
    TraceViewCore,
    URL,
    clearTimeout,
    console,
    document: { addEventListener() {}, activeElement: null },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) { callback(); },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__architectureScan = {
    state,
    dom,
    showModalDialog,
    modalDialogs,
    bindModalDialogLifecycle,
    hasOpenModalDialog,
  };`, context);
  return context.__architectureScan;
}

test("architecture scan DOM and static wiring are complete and bilingual", () => {
  for (const id of [
    "architectureScanButton",
    "architectureScanDialog",
    "closeArchitectureScanButton",
    "architectureScanBackend",
    "architectureScanTopN",
    "runArchitectureScanButton",
    "architectureScanStatus",
    "architectureScanSummary",
    "architectureScanBody",
    "architectureScanDiagnostics",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
    assert.match(app, new RegExp(`"${id}"`));
  }
  assert.match(html, /批量计算后端（Batch Backend）/);
  assert.match(html, /返回候选数（Top N）/);
  assert.match(html, /扫描当前架构（Run Scan）/);
  assert.match(app, /dom\.architectureScanButton\.addEventListener\("click", openArchitectureScanDialog\)/);
  assert.match(app, /dom\.runArchitectureScanButton\.addEventListener\("click"/);
  assert.match(app, /dom\.closeArchitectureScanButton\.addEventListener\("click"/);
  assert.match(app, /apiRequest\("\/architecture-scan",\s*\{[\s\S]*method: "POST"/);
});

test("architecture scan participates in unified modal backdrop, focus, and keyboard guard state", () => {
  const ui = loadDialogHelpers();
  const dialog = new FakeDialog();
  const opener = new FakeHTMLElement();
  const initialFocus = new FakeHTMLElement();
  ui.dom.architectureScanDialog = dialog;

  assert.deepEqual(Array.from(ui.modalDialogs()), [dialog]);
  ui.bindModalDialogLifecycle();
  ui.showModalDialog(dialog, opener, initialFocus);
  assert.equal(dialog.open, true);
  assert.equal(initialFocus.focused, true);
  assert.equal(ui.hasOpenModalDialog(), true);

  dialog.dispatch("click", { target: dialog });
  assert.equal(dialog.open, false);
  assert.equal(dialog.returnValue, "backdrop");
  assert.equal(opener.focused, true);
  assert.equal(ui.hasOpenModalDialog(), false);
});
