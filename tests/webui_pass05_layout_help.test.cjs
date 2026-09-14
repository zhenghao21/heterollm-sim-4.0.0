"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.join(__dirname, "..");
const app = fs.readFileSync(path.join(root, "src", "heterollm_sim", "webui", "app.js"), "utf8");
const styles = fs.readFileSync(path.join(root, "src", "heterollm_sim", "webui", "styles.css"), "utf8");
const passMarker = "/* Pass 05: responsive architecture, model tooling, and concept-help clarity. */";
const passStart = styles.indexOf(passMarker);

assert.ok(passStart >= 0, "Pass 05 style contract marker must exist");
const passStyles = styles.slice(passStart);

test("narrow architecture layout grows with its stacked panels instead of overlapping them", () => {
  assert.match(passStyles, /@media \(max-width: 1060px\)/u);
  assert.match(passStyles, /\.architecture-view\s*\{[\s\S]*?display:\s*block;[\s\S]*?overflow-y:\s*auto;/u);
  assert.match(passStyles, /\.architecture-grid\s*\{[\s\S]*?height:\s*auto;[\s\S]*?grid-template-rows:\s*auto minmax\(calc\(520px \* var\(--layout-scale\)\), auto\) auto;/u);
});

test("model toolbar reveals its status and moves tools to a dedicated narrow row", () => {
  assert.match(passStyles, /\.model-graph-toolbar\s*\{\s*flex-wrap:\s*wrap;\s*\}/u);
  assert.match(passStyles, /\.model-graph-status\s*\{[\s\S]*?overflow:\s*visible;[\s\S]*?white-space:\s*normal;/u);
  assert.match(passStyles, /\.model-graph-toolbar-tools\s*\{[\s\S]*?flex:\s*1 1 100%;[\s\S]*?margin-inline-start:\s*0;/u);
});

test("concept help has a readable card hierarchy and low-noise trigger feedback", () => {
  assert.match(passStyles, /\.field-help-viewport-popover\s*\{[\s\S]*?width:\s*min\(calc\(380px \* var\(--layout-scale\)\), calc\(100vw - 24px\)\);[\s\S]*?max-width:\s*min\(calc\(380px \* var\(--layout-scale\)\), calc\(100vw - 24px\)\);[\s\S]*?font-size:\s*max\(\.625rem, 11px\);[\s\S]*?scrollbar-gutter:\s*stable;/u);
  assert.match(passStyles, /\.field-help-section-title\s*\{[\s\S]*?color:\s*var\(--copper-bright\);/u);
  assert.match(passStyles, /\.field-help-trigger:is\(:hover, :focus-visible, \[aria-expanded="true"\]\)\s*\{[\s\S]*?background:[\s\S]*?box-shadow:/u);
  assert.doesNotMatch(passStyles, /content:\s*"\?"|text-decoration-style:\s*dotted/u);
});

test("trace detail semantics link to existing concept-help topics", () => {
  const start = app.indexOf("function renderTraceEventDetails");
  const end = app.indexOf("\nfunction ", start + 1);
  assert.ok(start >= 0 && end > start, "trace detail renderer must remain discoverable");
  const details = app.slice(start, end);
  const bindings = [
    ["事件类型", "runtime_event"],
    ["模型位置", "operator"],
    ["数据对象", "tensor"],
    ["流向", "protocol_path"],
    ["同时进行的工作", "resource_contention"],
    ["关联请求", "request"],
    ["记录完整性", "run_manifest"],
    ["记录方式", "fidelity"],
  ];
  for (const [label, key] of bindings) {
    const line = details.split(/\r?\n/u).find((candidate) => candidate.includes(`uiText("${label}`));
    assert.ok(line, `missing trace detail label: ${label}`);
    assert.ok(line.includes(`, "${key}")`), `${label} must bind to ${key}`);
  }
});
