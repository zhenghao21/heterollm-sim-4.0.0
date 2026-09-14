"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function ruleBody(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "s"));
  assert.ok(match, `missing CSS rule: ${selector}`);
  return match[1];
}

function sourceBetween(startName, endName) {
  const start = app.indexOf(`function ${startName}`);
  const end = app.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

test("1.9 concept help is structured, bilingual, and covers explicit latency and residency fields", () => {
  for (const key of [
    "read_latency", "write_latency", "transfer_granularity", "dma_latency",
    "kv_residency_policy", "model_weights_backing",
  ]) {
    assert.ok((app.match(new RegExp(`\\n\\s*${key}:`, "g")) || []).length >= 2, `${key} must have equivalent zh/en definitions`);
    assert.match(app, new RegExp(`${key}: Object\\.freeze\\(\\{[\\s\\S]*?"zh-CN":[\\s\\S]*?en:`, "u"), `${key} must have structured bilingual detail`);
  }
  assert.match(app, /const CONCEPT_HELP_SECTION_LABELS = Object\.freeze/);
  for (const label of ["定义", "在模拟器中的作用", "可填 / 可选内容", "单位、范围与特殊值", "影响", "限制与示例"]) {
    assert.ok(app.includes(label), `missing structured help section: ${label}`);
  }
  assert.match(app, /function metadataField\([^)]*helpKey = ""/);
  assert.match(app, /metadataField\("读取延迟（Read Latency, ns）"[\s\S]*?helpKey: "read_latency"/);
  assert.match(app, /quantityField\("传输粒度（Transfer Granularity, B\/KiB…PiB）"[\s\S]*?helpKey: "transfer_granularity"/);
  assert.match(app, /data-concept-help="kv_residency_policy"/);
  assert.match(app, /data-concept-help="model_weights_backing"/);
});

test("automatic help hydration binds only explicit keys or recognized professional terms", () => {
  const start = app.indexOf("function hydrateConceptHelp");
  const end = app.indexOf("function closeFieldHelp", start);
  const body = app.slice(start, end);
  assert.match(body, /\.control-section-title/);
  assert.match(body, /\.metric-cell/);
  assert.match(body, /\.canvas-legend/);
  assert.match(body, /if \(match\) title\.dataset\.conceptHelp = match\[0\]/);
  assert.doesNotMatch(body, /interface_field|interface_section|resultLike|legendLike|fieldLike/);
  assert.doesNotMatch(app, /\n\s*interface_(?:field|section):|\n\s*result_metric:/);
  for (const key of ["capacity", "peak_ops", "bandwidth", "port_parameters", "embedding", "residual_add", "placement", "protocol_path"]) {
    assert.match(app, new RegExp(`\\["${key}",`), `missing professional term pattern: ${key}`);
  }
  assert.match(ruleBody(".field-help-card"), /display:\s*grid/);
  assert.match(ruleBody(".field-help-section"), /grid-template-columns:/);
});

test("non-overridden help uses meaningful domain profiles and the portal remains scrollable", () => {
  for (const profile of ["hardware", "model", "mapping", "workload", "replay", "metric"]) {
    assert.match(app, new RegExp(`\\n\\s*${profile}: Object\\.freeze\\(\\{`), `missing help profile: ${profile}`);
  }
  assert.match(app, /function conceptHelpProfileName/);
  assert.match(app, /CONCEPT_HELP_DETAIL_PROFILES\[conceptHelpProfileName\(key\)\]/);
  assert.match(css, /\.field-help-viewport-popover\s*\{[^}]*pointer-events:\s*auto/s);
  assert.match(css, /\.field-help-viewport-popover\s*\{[^}]*touch-action:\s*pan-y/s);
  assert.match(css, /\.field-help-viewport-popover\s*\{[^}]*overscroll-behavior:\s*contain/s);
  assert.match(app, /fieldHelpPortal\?\.contains\?\.\(target\)/);
  assert.match(app, /event\.key === "Enter" \|\| event\.key === " "/);
});

test("runtime cards use measured shared title, label, and value tracks without clipping", () => {
  assert.match(ruleBody(".runtime-grid"), /align-items:\s*stretch/);
  assert.match(ruleBody(".runtime-group"), /grid-template-rows:\s*var\(--runtime-title-track, auto\) minmax\(0, 1fr\)/);
  assert.match(ruleBody(".runtime-stat"), /grid-template-rows:\s*var\(--runtime-label-track, auto\) var\(--runtime-value-track, auto\)/);
  assert.match(ruleBody(".runtime-stats"), /align-content:\s*start/);
  assert.match(app, /function runtimeSharedTrackMeasurements/);
  assert.match(app, /function applyRuntimeSharedTracks/);
  assert.match(app, /data-runtime-stat-row="\$\{Math\.floor\(index \/ 2\)\}"/);
  assert.match(app, /new globalThis\.ResizeObserver/);
  assert.match(app, /Math\.abs\(width - runtimeTrackObservedWidth\) < 0\.5/);
  const runtimeCss = `${ruleBody(".runtime-group h3")} ${ruleBody(".runtime-stat")} ${ruleBody(".runtime-stat dt")} ${ruleBody(".runtime-stat dd")}`;
  assert.doesNotMatch(runtimeCss, /line-clamp|overflow:\s*hidden|76px/);
  assert.match(css, /@media \(max-width: 1060px\)[\s\S]*?\.runtime-grid\s*\{[^}]*repeat\(2,/s);
});

test("preemption removes only the decorative box while preserving its wrapper and hit height", () => {
  assert.match(app, /class="field workload-toggle-field is-preemption-toggle"/);
  const body = ruleBody(".workload-toggle-field.is-preemption-toggle .workload-checkbox-control");
  assert.match(body, /min-height:\s*calc\(44px \* var\(--layout-scale\)\)/);
  assert.match(body, /padding:\s*0/);
  assert.match(body, /border:\s*0/);
  assert.match(body, /background:\s*transparent/);
});

test("architecture topology renders an occupancy-aware whole-graph plan and seeds drag previews from unchanged routes", () => {
  const stored = sourceBetween("topologyStoredRouteSegments", "renderLinks");
  const render = sourceBetween("renderLinks", "handleTopologyNodeClick");

  assert.match(stored, /Topology\.routeSegments/);
  assert.match(stored, /topologyRouteSourcePortKey/);
  assert.match(stored, /topologyRouteTargetPortKey/);
  assert.match(stored, /function planTopologyRoutesAtPositions/);
  assert.match(stored, /Topology\.planOrthogonalRoutes\(projection\.links, visibleRects/);
  assert.match(render, /planTopologyRoutesAtPositions\(positions, metrics, \{ routeIds, occupiedSegments \}\)/);
  assert.match(render, /routeIds/);
  assert.match(render, /occupiedSegments/);
  assert.match(render, /topologyStoredRouteSegments\(element\)/);
  assert.match(render, /dataset\.topologyRoutePoints = JSON\.stringify\(points\)/);
  assert.match(render, /dataset\.topologyRouteSourcePortKey = planned\.sourcePortKey/);
  assert.match(render, /dataset\.topologyRouteTargetPortKey = planned\.targetPortKey/);
  assert.doesNotMatch(render, /Topology\.routeOrthogonal/);
  assert.doesNotMatch(render, /const parallel = new Map/);
});
