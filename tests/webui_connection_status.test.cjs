"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");
const appPath = path.join(__dirname, "..", "src", "heterollm_sim", "webui", "app.js");
const htmlPath = path.join(__dirname, "..", "src", "heterollm_sim", "webui", "index.html");
const source = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(htmlPath, "utf8");

function response(status, payload) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => "application/json" },
    async json() { return payload; },
  };
}

function loadApp() {
  let language = "zh-CN";
  let fetchImpl = async () => response(200, {});
  const listeners = new Map();
  const connectionLabel = { textContent: "连接检查中" };
  const connectionState = {
    className: "connection-state is-checking",
    querySelector(selector) { return selector === "span" ? connectionLabel : null; },
  };
  const document = {
    addEventListener(type, listener) {
      const callbacks = listeners.get(type) || [];
      callbacks.push(listener);
      listeners.set(type, callbacks);
    },
    dispatchEvent(event) {
      (listeners.get(event.type) || []).forEach((listener) => listener(event));
    },
    querySelector(selector) {
      if (selector === "#connectionState span") return connectionLabel;
      if (selector === "#connectionState") return connectionState;
      return null;
    },
    querySelectorAll() { return []; },
  };
  const uiI18n = {
    language: () => language,
    pair: (zh, en) => (language === "en" ? en : zh),
    // Simulate the old static paired node being reset by a full-document pass.
    localize: () => { connectionLabel.textContent = "连接检查中"; },
    setLanguage(next) {
      language = next === "en" ? "en" : "zh-CN";
      document.dispatchEvent({ type: "ui-languagechange", detail: { language } });
    },
  };
  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    Intl,
    ModelGraphCore,
    Promise,
    clearTimeout,
    console,
    document,
    fetch: (...args) => fetchImpl(...args),
    localStorage: { setItem() {}, getItem() { return null; }, removeItem() {} },
    setTimeout,
    TopologyCore,
    TraceViewCore,
    UiI18n: uiI18n,
  });
  vm.runInContext(`${source}
;globalThis.__connectionStatus = {
  state,
  dom,
  apiRequest,
  bindConnectionEvents,
  probeConnection,
  renderAll,
  renderConnectionState,
  runtimeConnectionLabel,
  setConnection,
};`, context, { filename: appPath });
  const app = context.__connectionStatus;
  app.dom.connectionState = connectionState;
  app.dom.dirtyMark = { textContent: "" };

  return {
    app,
    context,
    connectionLabel,
    setFetch(next) { fetchImpl = next; },
    setLanguage(next) { uiI18n.setLanguage(next); },
  };
}

test("connection state keeps bilingual online, HTTP-error, and offline status through render/localize", async () => {
  const connectionMarkup = html.split(/\r?\n/u).find((line) => line.includes('id="connectionState"')) || "";
  assert.match(connectionMarkup, /id="connectionState"[^>]*>[\s\S]*<span data-i18n-skip>连接检查中<\/span>/);
  assert.doesNotMatch(connectionMarkup, /data-i18n-en=/);

  const harness = loadApp();
  const { app, context, connectionLabel } = harness;
  app.bindConnectionEvents();
  harness.setLanguage("en");
  assert.equal(connectionLabel.textContent, "Checking connection");
  harness.setLanguage("zh-CN");

  harness.setFetch(async () => response(200, {
    ok: true,
    version: "0.6.1",
    runtime: { python_version: "3.12.0" },
    ortools: { available: true, version: "9.15.6755", probe_ok: true },
    available_solvers: ["auto", "builtin", "ortools"],
  }));
  await app.probeConnection();
  assert.equal(app.state.connection.status, "online");
  assert.equal(connectionLabel.textContent, "本地 API · v0.6.1 · OR-Tools 9.15.6755");

  harness.setLanguage("en");
  assert.equal(connectionLabel.textContent, "Local API · v0.6.1 · OR-Tools 9.15.6755");

  // renderAll still performs a full localization pass; the connection state is
  // reapplied afterward and therefore cannot fall back to the checking label.
  vm.runInContext(`
    ensureTracePlaybackData = () => {};
    renderSteps = () => {};
    renderArchitecture = () => {};
    renderModel = () => {};
    renderMapping = () => {};
    renderWorkload = () => {};
    renderTracePlayback = () => {};
    renderResults = () => {};
    renderDiagnostics = () => {};
    hydrateConceptHelp = () => {};
  `, context);
  app.state.scenario = { name: "regression" };
  app.renderAll();
  assert.equal(connectionLabel.textContent, "Local API · v0.6.1 · OR-Tools 9.15.6755");

  harness.setFetch(async () => response(503, { error: { message_zh: "服务暂时不可用" } }));
  await app.probeConnection();
  assert.equal(app.state.connection.status, "offline");
  assert.equal(app.state.connection.labels.zh, "API 响应异常");
  assert.equal(app.state.connection.labels.en, "API response error");
  assert.equal(connectionLabel.textContent, "API response error");

  harness.setLanguage("zh-CN");
  harness.setFetch(async () => { throw new TypeError("network down"); });
  await app.probeConnection();
  assert.equal(app.state.connection.status, "offline");
  assert.equal(app.state.connection.labels.zh, "API 不可达");
  assert.equal(app.state.connection.labels.en, "API unreachable");
  assert.equal(connectionLabel.textContent, "API 不可达");
});

test("topbar keeps only a persistent two-state modification badge", () => {
  const topbar = html.slice(html.indexOf('<header class="topbar">'), html.indexOf("</header>"));
  assert.match(topbar, /id="dirtyMark">未修改<\/span>/);
  assert.doesNotMatch(topbar, /当前场景|scenarioName/);
  assert.match(source, /dirtyMark\.textContent = state\.dirty \? uiText\("已修改", "Modified"\) : uiText\("未修改", "Unmodified"\)/);
});
