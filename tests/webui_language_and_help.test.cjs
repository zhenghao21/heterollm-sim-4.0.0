"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.join(__dirname, "../src/heterollm_sim/webui");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");
const preload = fs.readFileSync(path.join(root, "ui-settings-preload.js"), "utf8");

test("global UI font scale caps at 200% and clamps persisted values before first paint", () => {
  assert.match(html, /id="fontScaleInput"[^>]+min="80"[^>]+max="200"/);
  assert.match(html, /id="fontScaleNumberInput"[^>]+min="80"[^>]+max="200"/);
  assert.match(app, /const MAX_FONT_SCALE = 200/);
  assert.match(app, /state\.settings = normalizeUiSettings\(readStoredJson\(STORAGE_UI_SETTINGS, DEFAULT_UI_SETTINGS\)\)/);
  assert.match(preload, /Math\.min\(200, Math\.max\(80/);

  const rootElement = {
    dataset: {},
    style: { setProperty() {}, removeProperty() {} },
  };
  const context = vm.createContext({
    document: { documentElement: rootElement },
    localStorage: { getItem() { return JSON.stringify({ fontScale: 500 }); } },
  });
  vm.runInContext(preload, context, { filename: path.join(root, "ui-settings-preload.js") });
  assert.equal(rootElement.dataset.fontScale, "200");
});

test("language is a persisted UI-only setting initialized before first paint", () => {
  assert.match(preload, /language:\s*"zh-CN"/);
  assert.match(preload, /root\.lang\s*=\s*language/);
  assert.match(html, /id="uiLanguageInput"/);
  assert.match(html, /src="\.\/ui-i18n\.js"[\s\S]*src="\.\/app\.js"/);
  assert.match(app, /UI_LANGUAGES\s*=\s*Object\.freeze\(\["zh-CN",\s*"en"\]\)/);
  assert.match(app, /updateUiSettings\(\{\s*language:/);
  assert.equal((html.match(/<option value="(?:zh-CN|en)">/g) || []).length, 2);
  const updateStart = app.indexOf("function updateUiSettings");
  const updateEnd = app.indexOf("function resetFontScale", updateStart);
  assert.doesNotMatch(app.slice(updateStart, updateEnd), /state\.dirty\s*=/);
});

test("automatic localization covers dynamic text and accessible attributes", () => {
  const i18n = fs.readFileSync(path.join(root, "ui-i18n.js"), "utf8");
  assert.match(i18n, /MutationObserver/);
  assert.match(i18n, /characterData:\s*true/);
  assert.match(i18n, /\["placeholder",\s*"aria-label",\s*"title"\]/);
  assert.doesNotMatch(html, /scenario-eyebrow-english/);
});

test("concept help and backend issues select stable localized fields", () => {
  assert.match(app, /const CONCEPT_HELP_ZH = Object\.freeze/);
  assert.match(app, /const CONCEPT_HELP_EN = Object\.freeze/);
  assert.match(app, /Object\.freeze\(\{\s*"zh-CN": zh, en:/);
  assert.match(app, /item\.message_en, item\.detail_en, item\.reason_en/);
  assert.match(app, /ISSUE_MESSAGE_BY_CODE/);
  assert.match(app, /localizedIssueMessage\(item\)/);
});

test("the bilingual glossary covers representative hardware, model, mapping, replay, and metric terms", () => {
  for (const key of [
    "hbm", "hbf", "cim", "chiplet", "pcie", "cxl", "ucie", "nvlink", "roce",
    "qkv", "gqa", "mqa", "moe", "rmsnorm", "rope", "tp", "pp", "ep", "cp_sat",
    "kv_cache", "des", "ttft", "tbt", "tpot", "percentile", "makespan",
  ]) {
    assert.match(app, new RegExp(`\\n\\s*${key}:`), key);
  }
  assert.match(app, /const scopedRoots = root === document\s*\? \[document\]/);
  assert.match(app, /h1, h2, h3, h4, th, dt, legend/);
  assert.match(app, /hydrateConceptHelp\(dom\.inspectorContent\)/);
  assert.match(app, /hydrateConceptHelp\(dom\.modelGraphInspectorContent\)/);
});

test("concept help uses the term itself instead of a visible question-mark control", () => {
  const start = app.indexOf("function fieldHelpMarkup");
  const end = app.indexOf("function componentInspectorEvidence", start);
  const implementation = app.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.doesNotMatch(implementation, /field-help-trigger[^>]*>\s*\?\s*</);
  assert.match(implementation, /class="[^"]*field-help-trigger/);
  assert.match(implementation, /pointerenter/);
  assert.match(implementation, /addEventListener\("focus"/);
  assert.match(implementation, /event\.key\s*===\s*"Escape"/);
  assert.match(implementation, /event\.key\s*===\s*"Enter"\s*\|\|\s*event\.key\s*===\s*" "/);
  assert.match(css, /\.field-help-trigger\s*\{[^}]*cursor:\s*help/s);
  assert.match(css, /\.field-help-trigger\s*\{[^}]*text-decoration:\s*none/s);
  assert.doesNotMatch(css, /text-decoration-style:\s*dotted/);
  assert.doesNotMatch(implementation, /fieldHelpMarkup\(helpKey,\s*label\)/);
  assert.doesNotMatch(implementation, /field-help-popover[^\n]*<strong>/);
  assert.match(implementation, /field-help-popover[^\n]*role="tooltip" hidden/);
});

test("Chinese chrome has no known English glosses and CSS pseudo-copy switches by language", () => {
  const I18n = require(path.join(root, "ui-i18n.js"));
  const textLeaves = [...html.replace(/<!--[\s\S]*?-->/g, "").matchAll(/>([^<>]+)</g)]
    .map((match) => match[1].replace(/&amp;/g, "&").replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .map((value) => I18n.localizeText(value, "zh-CN"));
  const knownGlosses = /\b(?:Overview|Operator-first|Host Memory|Fabric Switch|Prediction Layer|Auxiliary Head|Proposal Logits|Explicit Requests|Runtime & Solver Diagnostics)\b/u;
  textLeaves.forEach((value) => assert.doesNotMatch(value, knownGlosses, value));
  assert.match(css, /\.topology-canvas::after\s*\{[^}]*content:\s*"封装拓扑 \/ SCHEMA 4\.0\.0"/s);
  assert.match(css, /data-language="en"\] \.topology-canvas::after\s*\{[^}]*content:\s*"PACKAGE TOPOLOGY \/ SCHEMA 4\.0\.0"/s);
});

test("leaf model overview names and inspector copy use explicit bilingual localization", () => {
  assert.match(app, /uiText\("未选择组件", "No component selected"\)/);
  assert.match(app, /function modelGraphOverviewDisplayText\(value\)[\s\S]*globalThis\.UiI18n\?\.localizeText\?\.\(text, state\.settings\.language\)/);
  assert.match(app, /function modelGraphOverviewName\(node\)[\s\S]*return modelGraphOverviewDisplayText\(label\);/);
  const I18n = require(path.join(root, "ui-i18n.js"));
  for (const [zh, en] of [
    ["输入（Input）", "Input"],
    ["重复 Block 组（Repeated Block Group）", "Repeated Block Group"],
    ["多 Token 预测层（MTP Prediction Layer）", "MTP Prediction Layer"],
    ["多 Token 辅助头（MTP Auxiliary Head）", "MTP Auxiliary Head"],
  ]) {
    assert.ok(app.includes(zh), `missing leaf label: ${zh}`);
    assert.equal(I18n.localizeText(zh, "zh-CN"), zh.replace(/（[^（）]+）/u, ""));
    assert.equal(I18n.localizeText(zh, "en"), en);
  }
  for (const [zh, en] of [
    ["组件说明", "Component description"],
    ["结构角色", "Structural role"],
    ["代表算子 ID", "Representative operator ID"],
    ["实例数量", "Instance count"],
    ["具体端口", "Concrete ports"],
  ]) assert.match(app, new RegExp(`uiText\\("${zh}", "${en}"\\)`));
  assert.match(app, /function modelGraphOverviewIsMtp\(nodeValue\)[\s\S]*String\(node\.op_kind \|\| ""\)\.startsWith\("mtp_"\)/);
  assert.match(app, /mtp_prediction_layer: "多 Token 预测层（MTP Prediction Layer）"/);
  assert.match(app, /mtp_aux_head: "多 Token 辅助头（MTP Auxiliary Head）"/);
  assert.doesNotMatch(app, /uiText\("解码器堆栈", "Decoder Stack"\)/);
  assert.doesNotMatch(app, /uiText\("多 Token 预测器（MTP）", "Multi-Token Predictor \(MTP\)"\)/);
  assert.doesNotMatch(app, /实线子图来自权威 operator \/ typed port \/ tensor edge/);
});

test("all six page shells expose English-only static copy and accessible attributes", () => {
  const I18n = require(path.join(root, "ui-i18n.js"));
  for (const view of ["architecture", "model", "mapping", "workload", "playback", "results"]) {
    const start = html.indexOf(`id="view-${view}"`);
    const next = html.indexOf('<section class="view', start + 1);
    const section = html.slice(start, next < 0 ? html.length : next).replace(/<!--[\s\S]*?-->/g, "");
    const values = [
      ...section.matchAll(/>([^<>]+)</g),
      ...section.matchAll(/(?:placeholder|aria-label|title)="([^"]+)"/g),
    ].map((match) => match[1].replace(/&amp;/g, "&").replace(/\s+/g, " ").trim()).filter(Boolean);
    values.forEach((value) => {
      const localized = I18n.localizeText(value, "en");
      assert.doesNotMatch(localized, /interface text|[\u3400-\u9fff]/iu, `${view}: ${value} -> ${localized}`);
    });
  }
});

test("heading chrome keeps one prominent title while preserving independent kickers", () => {
  // A plain eyebrow immediately before a heading is presentation-only chrome
  // and must not repeat the heading. Independent state/scope kickers opt into
  // the explicit `kicker` class so they remain reviewable and intentional.
  assert.doesNotMatch(html, /<span class="eyebrow">[^<]*<\/span>\s*<h[123]\b/);
  assert.doesNotMatch(html, /<p>Explicit Requests<\/p>/);
  assert.match(html, /<span class="eyebrow kicker" id="traceNarrativeTitle">此刻发生了什么<\/span>/);
  assert.match(html, /id="traceNarrativeState"/);
  assert.match(html, /id="modelGraphCanvasTitle" class="sr-only">模型语义组件图画布<\/h2>/);

  const I18n = require(path.join(root, "ui-i18n.js"));
  const headingTexts = [...html.matchAll(/<h[123](?:\s[^>]*)?>([\s\S]*?)<\/h[123]>/g)]
    .map((match) => match[1].replace(/<[^>]*>/g, "").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim())
    .filter(Boolean);
  headingTexts.forEach((value) => {
    assert.doesNotMatch(I18n.localizeText(value, "en"), /[\u3400-\u9fff]/u, `heading remains Chinese in English: ${value}`);
  });
  assert.equal(I18n.localizeText("组件", "en"), "Components");
  assert.equal(I18n.localizeText("诊断", "en"), "Diagnostics");
});

test("workload headings and field labels render exactly one active language", () => {
  const renderStart = app.indexOf("function renderWorkload()");
  const renderEnd = app.indexOf("function workloadFieldLabel", renderStart);
  const body = app.slice(renderStart, renderEnd);
  assert.match(body, /uiText\("请求生成", "Request Generation"\)/);
  assert.match(body, /uiText\("连续批处理与调度", "Continuous Batching & Scheduling"\)/);
  assert.match(body, /uiText\("多 Token 预测（MTP）", "Multi-Token Prediction \(MTP\)"\)/);
  assert.doesNotMatch(body, /<p>Request Generation|<p>Continuous Batching|<p>Multi-Token Prediction/);
  const labelBody = app.slice(renderEnd, app.indexOf("function nestedNumberField", renderEnd));
  assert.match(labelBody, /uiText\(primary, secondary\)/);
  assert.doesNotMatch(labelBody, /<small>/);
});

test("real-browser dynamic model, mapping, and workload paths use explicit language pairs", () => {
  const contracts = [
    [/uiText\("已物化", "Materialized"\)/, "control-plane ready state"],
    [/uiText\("尚未物化", "Not materialized"\)/, "control-plane idle state"],
    [/"No internal runtime placement decision is materialized yet\./, "control-plane empty guidance"],
    [/uiText\("尚无 Rank-aware 控制平面决策", "No Rank-aware control-plane decision yet"\)/, "rank-aware empty state"],
    [/"\{groups\} operator groups · \{targets\} Rank targets"/, "operator mapping counts"],
    [/"\{groups\} tensor groups · \{shards\} physical shards"/, "tensor mapping counts"],
    [/uiText\("无匹配数据", "No matching data"\)/, "mapping pagination empty state"],
    [/"Actual TP\/PP\/EP Rank coverage appears after the internal control plane materializes placement\."/, "operator mapping guidance"],
    [/"The current mapping does not declare per-Rank weight shards\."/, "tensor shard guidance"],
    [/uiText\("并行策略", "Parallel strategy"\)/, "parallel title"],
    [/uiText\("允许同组件多逻辑 Rank", "Allow colocated logical Ranks"\)/, "rank colocation"],
    [/"Changing TP\/PP\/EP clears the old Rank mapping; changing PP also clears the layer-to-stage mapping\./, "parallel explanation"],
    [/uiText\("KV 驻留策略", "KV residency strategy"\)/, "KV title"],
    [/"HBF uses UCIe; SSD and high-I\/O SSD use PCIe\/CXL\./, "backing planner explanation"],
    [/uiText\("允许调度器抢占进行中的序列", "Allow the scheduler to preempt active sequences"\)/, "preemption help"],
    [/uiText\("启用候选 Token 提议与接受模型", "Enable candidate-Token proposals and the acceptance model"\)/, "MTP help"],
  ];
  contracts.forEach(([pattern, label]) => assert.match(app, pattern, label));

  const I18n = require(path.join(root, "ui-i18n.js"));
  I18n.setLanguage("en", null);
  for (const [zh, en] of [
    ["已物化", "Materialized"],
    ["尚未物化", "Not materialized"],
    ["无匹配数据", "No matching data"],
    ["并行策略", "Parallel strategy"],
    ["KV 驻留策略", "KV residency strategy"],
  ]) assert.equal(I18n.pair(zh, en), en);
  I18n.setLanguage("zh-CN", null);
});
