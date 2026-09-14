"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function sourceBetween(startName, endName) {
  const start = app.indexOf(`function ${startName}`);
  const end = app.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

test("architecture node pointermove is frame-coalesced and defers global work until pointerup", () => {
  const move = sourceBetween("moveNodeDrag", "endNodeDrag");
  const frame = sourceBetween("applyNodeDragFrame", "scheduleNodeDragFrame");
  const end = sourceBetween("endNodeDrag", "selectItem");

  assert.match(move, /scheduleNodeDragFrame\(event\)/);
  assert.doesNotMatch(move, /resolveTopologyPlacement|renderGroups|updateWorldBounds|savePositions|commitTopologyHistory/);
  assert.match(frame, /drag\.element|style\.transform|changedComponentIds/);
  assert.match(frame, /previewTopologyGroupBounds\(previewPositions, Object\.keys\(desired\)\)/);
  assert.doesNotMatch(frame, /resolveTopologyPlacement|renderGroups|updateWorldBounds|savePositions|commitTopologyHistory/);
  assert.match(end, /resolveTopologyPlacement/);
  assert.match(end, /renderGroups\(\)[\s\S]*updateWorldBounds\(\)[\s\S]*renderLinks\(\)[\s\S]*savePositions\(\)/);
  assert.match(end, /if \(placement\)[\s\S]*else \{\s*renderGroups\(\);\s*renderLinks\(\);\s*\}/, "blocked drops restore the previewed group frame");
  assert.match(end, /commitTopologyHistory\(drag\.historyBefore, "移动组件"\)/);
  assert.match(end, /pointercancel|lostpointercapture/);
});

test("architecture canvas pan and marquee coalesce raw pointer bursts", () => {
  const move = sourceBetween("moveCanvasPointer", "endCanvasPointer");
  const frame = sourceBetween("applyTopologyCanvasPointerFrame", "scheduleTopologyCanvasPointerFrame");

  assert.match(move, /scheduleTopologyCanvasPointerFrame\(event\)/);
  assert.doesNotMatch(move, /syncTopologyViewport|updateMarqueeElement|saveTopologyView/);
  assert.match(frame, /syncTopologyViewport\(\)/);
  assert.match(frame, /updateMarqueeElement\(\)/);
});

test("trace drag and pan avoid layout, DOM rebuild, and particle restart during pointermove", () => {
  const move = sourceBetween("moveTraceCanvasPointer", "endTraceCanvasPointer");
  const frame = sourceBetween("applyTraceInteractionFrame", "scheduleTraceInteractionFrame");
  const end = sourceBetween("endTraceCanvasPointer", "traceNodeMemoryCapacityLabel");

  assert.match(move, /scheduleTraceInteractionFrame\(event\)/);
  assert.doesNotMatch(move, /traceTopologyLayout|dragTraceNode|renderTraceTopology|renderTraceParticles/);
  assert.match(frame, /drag\.element\.style\.transform/);
  assert.match(frame, /previewTraceNodeRoutes\(drag, target\)/);
  assert.doesNotMatch(frame, /traceTopologyLayout|dragTraceNode|renderTraceTopology|renderTraceParticles/);
  assert.match(end, /TraceView\.dragTraceNode/);
  assert.match(end, /playback\.layoutView\.positions = committed\.positions/);
  assert.match(end, /playback\.topologyLayout = null/);
  assert.match(end, /renderTraceTopology\(\)/);
  assert.match(end, /pointercancel|lostpointercapture/);
});

test("trace playback ticks do not rebuild topology during a pointer gesture", () => {
  const step = sourceBetween("traceAnimationStep", "startTracePlayback");
  assert.match(step, /if \(playback\.layoutDrag \|\| playback\.layoutPan\) return/);
  assert.match(step, /setTraceTime/);
});

test("model pointerup commits the actual release coordinates and defers resize during port drag", () => {
  const end = sourceBetween("endModelGraphPointer", "renderModel");
  const resize = sourceBetween("scheduleModelGraphOverviewResize", "bindModelGraphResizeObserver");

  assert.match(end, /const finalPoint = !cancelled \? \{/);
  assert.doesNotMatch(end, /editor\.pendingPointer \|\|/);
  assert.match(end, /scheduleModelGraphOverviewResize\(\);[\s\S]*return;/);
  assert.match(resize, /editor\.drag \|\| editor\.pan \|\| editor\.connectPointer/);
});

test("model zoom is transform-only and viewport persistence is debounced", () => {
  const zoom = sourceBetween("zoomModelGraph", "fitModelGraph");
  assert.match(zoom, /syncModelGraphViewport\(ui\)/);
  assert.match(zoom, /scheduleModelGraphViewportSave\(\)/);
  assert.doesNotMatch(zoom, /renderModelGraph|saveModelGraphLayout/);
  assert.match(css, /\.model-graph-world[^}]*will-change:\s*transform/);
  assert.match(css, /\.model-overview-node\.is-dragging[^}]*will-change:\s*transform/);
  assert.doesNotMatch(css, /\.model-overview-node\s*\{[^}]*will-change/);
  assert.doesNotMatch(css, /\.model-inline-operator\s*\{[^}]*will-change/);
});
