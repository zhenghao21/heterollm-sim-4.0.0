"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const I18n = require("../src/heterollm_sim/webui/ui-i18n.js");

test("language normalization is deterministic and Chinese remains the default", () => {
  assert.equal(I18n.normalizeLanguage("en-US"), "en");
  assert.equal(I18n.normalizeLanguage("EN"), "en");
  assert.equal(I18n.normalizeLanguage("zh-TW"), "zh-CN");
  assert.equal(I18n.normalizeLanguage("unexpected"), "zh-CN");
});

test("direct pairs expose only the active language", () => {
  I18n.setLanguage("zh-CN", null);
  assert.equal(I18n.pair("你好，{name}", "Hello, {name}", { name: "Ada" }), "你好，Ada");
  I18n.setLanguage("en", null);
  assert.equal(I18n.pair("你好，{name}", "Hello, {name}", { name: "Ada" }), "Hello, Ada");
});

test("interpolation leaves unknown placeholders visible for diagnostics", () => {
  assert.equal(I18n.interpolate("{known}/{missing}", { known: 7 }), "7/{missing}");
});

test("the browser bundle exposes direct language pairs for static copy", () => {
  const source = require("node:fs").readFileSync(require("node:path").join(__dirname, "../src/heterollm_sim/webui/ui-i18n.js"), "utf8");
  assert.match(source, /data-i18n-en/);
  assert.match(source, /data-i18n-zh/);
  assert.match(source, /\["placeholder",\s*"aria-label",\s*"title"\]/);
  assert.match(source, /data-i18n-\$\{attribute\}-en/);
});

test("mixed labels expose one language while canonical acronyms remain unchanged", () => {
  assert.equal(I18n.localizeText("分析型 · ANALYTICAL", "zh-CN"), "分析型");
  assert.equal(I18n.localizeText("分析型 · ANALYTICAL", "en"), "ANALYTICAL");
  assert.equal(I18n.localizeText("GPU（图形处理器）", "zh-CN"), "GPU（图形处理器）");
  assert.equal(I18n.localizeText("运行回放", "en"), "Run playback");
  assert.equal(I18n.localizeText("未知界面文案", "en"), "未知界面文案", "unknown provenance is preserved instead of hidden behind a placeholder");
});

test("Chinese mode removes interface glosses but preserves canonical acronyms and schema keys", () => {
  const cases = new Map([
    ["封装拓扑 · PACKAGE TOPOLOGY / SCHEMA 4.0.0", "封装拓扑"],
    ["结构总览 / Overview", "结构总览"],
    ["线性注意力 Block（Linear Attention Block）", "线性注意力 Block"],
    ["从上到下 · 算子优先（TB · Operator-first）", "从上到下 · 算子优先"],
    ["operator · typed port · tensor edge", "算子 · 带类型端口 · 张量边"],
    ["预测层（Prediction Layer）", "预测层"],
    ["主机内存（Host Memory）", "主机内存"],
    ["互连交换结构（Fabric Switch）", "互连交换结构"],
    ["硬件组件（hardware.components）", "硬件组件（hardware.components）"],
    ["首 Token 延迟（TTFT）", "首 Token 延迟（TTFT）"],
  ]);
  cases.forEach((expected, source) => assert.equal(I18n.localizeText(source, "zh-CN"), expected, source));
});

test("known front-owned dynamic labels expose complete English without Chinese", () => {
  for (const source of [
    "封装拓扑 · PACKAGE TOPOLOGY / SCHEMA 4.0.0",
    "结构总览 / Overview",
    "线性注意力 Block（Linear Attention Block）",
    "预测层（Prediction Layer）",
    "主机内存（Host Memory）",
    "互连交换结构（Fabric Switch）",
  ]) assert.doesNotMatch(I18n.localizeText(source, "en"), /[\u3400-\u9fff]/u, source);
});

test("known application dynamics never use a generic translation placeholder", () => {
  const source = require("node:fs").readFileSync(require("node:path").join(__dirname, "../src/heterollm_sim/webui/ui-i18n.js"), "utf8");
  assert.doesNotMatch(source, /interface text/i);
  for (const value of [
    "10 组件 · 9 链路 · 0 组",
    "3 算子 · 3 张量",
    "通用计算（GPU）",
    "首 Token 延迟（TTFT）",
    "可滚动的 Trace 数据流拓扑；仅高亮当前时间实际活动的组件和链路",
  ]) {
    const localized = I18n.localizeText(value, "en");
    assert.doesNotMatch(localized, /interface text|[\u3400-\u9fff]/iu, `${value} -> ${localized}`);
  }
});

test("the language contract stays pair-based", () => {
  assert.equal(I18n.register, undefined);
  assert.equal(I18n.registerTextPairs, undefined);
  assert.equal(I18n.t, undefined);
  assert.deepEqual(I18n.LANGUAGES, ["zh-CN", "en"]);
});
