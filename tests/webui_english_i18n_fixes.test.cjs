"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appSource = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const I18n = require(path.join(webui, "ui-i18n.js"));
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function fakeElement() {
  return {
    className: "",
    _innerHTML: "",
    set innerHTML(value) { this._innerHTML = String(value ?? ""); },
    get innerHTML() { return this._innerHTML; },
    querySelectorAll() { return []; },
  };
}

function loadApp() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    Intl,
    ModelGraphCore,
    Promise,
    TopologyCore,
    TraceViewCore,
    UiI18n: I18n,
    clearTimeout,
    console,
    document: { addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; } },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
  });
  vm.runInContext(`${appSource}
;globalThis.__englishI18nFixes = { state, dom, renderRequestTable, renderRuntimeHealth };`, context, {
    filename: path.join(webui, "app.js"),
  });
  return context.__englishI18nFixes;
}

test("standalone Mapping is title-cased without changing lowercase phrase translation", () => {
  assert.equal(I18n.localizeText("映射", "zh-CN"), "映射");
  assert.equal(I18n.localizeText("映射", "en"), "Mapping");
  assert.equal(I18n.localizeText("Rank 映射摘要", "en"), "Rank mapping summary");
});

test("language changes re-render dynamic runtime diagnostics after updating i18n state", () => {
  const languageBranch = appSource.match(/if \(state\.settings\.language !== previousLanguage\) \{[\s\S]*?\n  \}/u)?.[0] || "";
  assert.match(languageBranch, /setLanguage/u);
  assert.match(languageBranch, /renderRuntimeHealth\(\)/u);
  assert.ok(
    languageBranch.indexOf("setLanguage") < languageBranch.indexOf("renderRuntimeHealth()"),
    "runtime diagnostics must re-render after the active language changes",
  );
});

test("request-table aria labels follow the active language and preserve request IDs", () => {
  const ui = loadApp();
  ui.dom.requestTableBody = fakeElement();
  ui.state.scenario = {
    workload: {
      requests: [{ request_id: "req-7", arrival_ns: 100, prompt_tokens: 8, output_tokens: 4, priority: 2, deadline_ns: 900 }],
    },
  };

  I18n.setLanguage("zh-CN", null);
  ui.renderRequestTable();
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="请求 req-7 优先级"/u);
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="请求 req-7 截止时间（纳秒，可选）"/u);
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="删除请求 req-7"/u);

  I18n.setLanguage("en", null);
  ui.renderRequestTable();
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="Request req-7 priority"/u);
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="Request req-7 deadline \(nanoseconds, optional\)"/u);
  assert.match(ui.dom.requestTableBody.innerHTML, /aria-label="Delete request req-7"/u);
  I18n.setLanguage("zh-CN", null);
});

test("runtime-health diagnostics localize front-owned copy while preserving runtime values", () => {
  const ui = loadApp();
  ui.dom.runtimeHealthPanel = fakeElement();
  const health = {
    ok: true,
    serviceVersion: "0.6.1",
    pythonVersion: "3.12.0",
    pythonExecutable: "C:\\Python312\\python.exe",
    architectureBits: 64,
    ortoolsAvailable: true,
    ortoolsVersion: "9.15.6755",
    ortoolsProbeOk: true,
    availableSolvers: ["auto", "builtin", "ortools"],
    diagnosticLogPath: "C:\\logs\\runtime.log",
    ortoolsError: "后端保留的诊断说明",
  };
  ui.state.runtimeHealth = health;

  I18n.setLanguage("zh-CN", null);
  ui.renderRuntimeHealth();
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /仿真器版本/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /64 位/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /可用/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /0\.6\.1|3\.12\.0|C:\\Python312\\python\.exe|auto \/ builtin \/ ortools/u);

  I18n.setLanguage("en", null);
  ui.renderRuntimeHealth();
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /Simulator version/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /64-bit/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /Available/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /0\.6\.1|3\.12\.0|C:\\Python312\\python\.exe|auto \/ builtin \/ ortools/u);
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, /后端保留的诊断说明/u);
  assert.doesNotMatch(ui.dom.runtimeHealthPanel.innerHTML, /仿真器版本|解释器路径|可用求解器|诊断日志|64 位/u);

  ui.state.runtimeHealth = { ...health, ortoolsAvailable: false, ortoolsProbeOk: false };
  ui.renderRuntimeHealth();
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, />Unavailable</u);
  I18n.setLanguage("zh-CN", null);
  ui.renderRuntimeHealth();
  assert.match(ui.dom.runtimeHealthPanel.innerHTML, />不可用</u);
  I18n.setLanguage("zh-CN", null);
});
