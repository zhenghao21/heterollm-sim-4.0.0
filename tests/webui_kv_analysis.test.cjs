"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");

test("KV residency explorer has a complete UI contract", () => {
  for (const id of [
    "kvAnalysisButton", "kvAnalysisDialog", "closeKvAnalysisButton",
    "runKvAnalysisButton", "kvAnalysisScope", "kvAnalysisStatus",
    "kvAnalysisSummary", "kvAnalysisBody", "kvAnalysisNote",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
    assert.match(app, new RegExp(`"${id}"`));
  }
  assert.match(html, /HBM、HBF 及混合候选/);
  assert.match(html, /TTFT p50/);
  assert.match(html, /KV 峰值 \/ 容量/);
  assert.match(app, /function kvAnalysisComponentCandidates/);
  assert.match(app, /function kvAnalysisReportMetrics/);
  assert.match(app, /apiRequest\("\/run"/);
  assert.match(app, /dom\.kvAnalysisDialog/);
  assert.match(css, /\.kv-analysis-table/);
});

