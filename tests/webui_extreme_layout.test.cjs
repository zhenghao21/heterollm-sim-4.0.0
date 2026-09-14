"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const css = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "webui", "styles.css"),
  "utf8",
);
const app = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "webui", "app.js"),
  "utf8",
);

function ruleBody(selectorPattern) {
  const match = css.match(new RegExp(`${selectorPattern}\\s*\\{([^}]*)\\}`, "s"));
  assert.ok(match, `缺少布局规则：${selectorPattern}`);
  return match[1];
}

function sourceBetween(startName, endName) {
  const start = app.indexOf(`function ${startName}`);
  const end = app.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

test("常规字号继续使用三栏架构布局", () => {
  const body = ruleBody("\\.architecture-grid");
  assert.match(body, /grid-template-columns:\s*clamp\([^;]+\)\s+minmax\([^;]+\)\s+clamp\([^;]+\);/s);
  assert.doesNotMatch(body, /height:\s*auto/);
});

test("大字号与极大字号由架构视图承载纵向滚动", () => {
  const body = ruleBody(
    ':root\\[data-font-band="large"\\] \\.architecture-view,\\s*' +
    ':root\\[data-font-band="extreme"\\] \\.architecture-view',
  );
  assert.match(body, /display:\s*block/);
  assert.match(body, /overflow-x:\s*hidden/);
  assert.match(body, /overflow-y:\s*auto/);
});

test("大字号架构网格按内容堆叠且画布行保留可用高度", () => {
  const body = ruleBody(
    ':root\\[data-font-band="large"\\] \\.architecture-grid,\\s*' +
    ':root\\[data-font-band="extreme"\\] \\.architecture-grid',
  );
  assert.match(body, /width:\s*100%/);
  assert.match(body, /height:\s*auto/);
  assert.match(body, /grid-template-columns:\s*minmax\(0,\s*1fr\)/);
  assert.match(
    body,
    /grid-template-rows:\s*auto\s+minmax\(calc\(520px\s*\*\s*var\(--layout-scale\)\),\s*auto\)\s+auto/,
  );
  assert.doesNotMatch(body, /grid-template-rows:[^;]*1fr/);
  assert.match(body, /align-content:\s*start/);
});

test("高字号工作区不再强制横向撑破单列架构", () => {
  const body = ruleBody(
    ':root\\[data-font-band="large"\\] \\.topology-workspace,\\s*' +
    ':root\\[data-font-band="extreme"\\] \\.topology-workspace',
  );
  assert.match(body, /min-width:\s*0/);
});

test("拓扑链路不再为常驻文字标签预留布局空间", () => {
  assert.doesNotMatch(css, /\.topology-link-label|\.trace-link-label/);
  assert.doesNotMatch(app, /Topology\.placeRouteLabels\(/);
  assert.match(app, /function topologyLinkTooltipText\(/);
});

test("架构节点只显示组件 ID，框体按内容紧缩", () => {
  const createStart = app.indexOf("function createTopologyNode");
  const updateStart = app.indexOf("function updateTopologyNode", createStart);
  const measureStart = app.indexOf("function measureTopologyNodes", updateStart);
  const createBody = app.slice(createStart, updateStart);
  const updateBody = app.slice(updateStart, measureStart);
  assert.match(createBody, /className = "node-title"/);
  assert.doesNotMatch(createBody, /node-kicker|node-meta/);
  assert.match(updateBody, /\.node-title[\s\S]*textContent = id/);
  assert.doesNotMatch(updateBody, /textContent\s*=\s*kindLabel|formatBytes|formatOps/);
  const node = ruleBody("\\.topology-node");
  assert.match(node, /width:\s*max-content/);
  assert.match(node, /min-width:\s*calc\(48px \* var\(--layout-scale\)\)/);
});

test("两页自动适配优先 100%、最低 80%，放不下时保留滚动", () => {
  const canvas = ruleBody("\\.topology-canvas");
  assert.match(canvas, /overflow:\s*auto/);
  const architectureFit = app.slice(app.indexOf("function fitTopologyViewport"), app.indexOf("function createSelectedGroup"));
  assert.match(architectureFit, /minScale:\s*0\.8/);
  assert.match(architectureFit, /maxScale:\s*1/);
  const traceFit = app.slice(app.indexOf("function traceFitMetrics"), app.indexOf("function traceScrollCanvasTo"));
  assert.match(traceFit, /minScale:\s*0\.8/);
  assert.match(traceFit, /maxScale:\s*1/);
  assert.match(traceFit, /Math\.max\(0\.8, Math\.min\(1,/);
});

test("节点尺寸变化触发重测量、重排和完整正交重路由", () => {
  const render = app.slice(app.indexOf("function renderTopology"), app.indexOf("function topologyNodeRect"));
  assert.match(render, /const sizesChanged = measureTopologyNodes\(\)/);
  assert.match(render, /if \(needsMeasuredLayout \|\| sizesChanged\)[\s\S]*ensureNodePositions\(true\)/);
  assert.match(render, /renderLinks\(\)/);
  assert.match(app, /new globalThis\.ResizeObserver\(scheduleTopologyResponsiveLayout\)/);
  assert.match(app, /state\.tracePlayback\.topologyLayout = null;[\s\S]*renderTraceTopology\(\);[\s\S]*fitTraceLayout/);
});

test("架构画布响应式重测量保留用户坐标和持久化视图", () => {
  const schedule = sourceBetween("scheduleTopologyResponsiveLayout", "bindTopologyResizeObservers");
  assert.match(schedule, /measureTopologyNodes\(\)/);
  assert.match(schedule, /renderGroups\(\)/);
  assert.match(schedule, /renderLinks\(\{ normalizeWorld: false \}\)/);
  assert.match(schedule, /fitTopologyViewport\(\{ save: false, normalizeWorld: false \}\)/);
  assert.doesNotMatch(schedule, /renderTopology\(\{ relayout: true \}\)/);
  assert.doesNotMatch(schedule, /ensureNodePositions|savePositions|saveTopologyView/);
});

test("架构和回放画布仅在 Ctrl 或 Meta 滚轮时缩放", () => {
  for (const [canvas, zoom] of [
    ["traceTopologyCanvas", "zoomTraceLayout"],
    ["topologyCanvas", "zoomTopology"],
  ]) {
    const pattern = new RegExp(
      `dom\\.${canvas}\\.addEventListener\\("wheel", \\(event\\) => \\{\\s*`
      + `if \\(!event\\.ctrlKey && !event\\.metaKey\\) return;\\s*`
      + `event\\.preventDefault\\(\\);[\\s\\S]*?${zoom}\\(`,
    );
    assert.match(app, pattern, `${canvas} leaves ordinary wheel/trackpad input available for scrolling`);
  }
});

test("架构世界原点归一化后原地重算并重绘链路", () => {
  const render = sourceBetween("renderLinks", "handleTopologyNodeClick");
  const normalization = render.slice(render.indexOf("if (normalizeWorld && (shift.x || shift.y))"));
  assert.match(render, /const shift = updateWorldBounds\(\{ normalize: normalizeWorld \}\)/);
  assert.match(normalization, /planTopologyRoutesAtPositions\(state\.nodePositions, metrics\)/);
  assert.match(normalization, /repaintTopologyRoutes\(routePlan\)/);
  assert.match(normalization, /state\.topologyOverlayRects = routePlan\.routes/);
  assert.match(normalization, /updateTopologyRouteStatus\(routeErrors\)/);
  assert.match(normalization, /updateWorldBounds\(\)/);
  assert.doesNotMatch(normalization, /\brenderLinks\s*\(/, "normalization must not recurse through the full renderer");
});
