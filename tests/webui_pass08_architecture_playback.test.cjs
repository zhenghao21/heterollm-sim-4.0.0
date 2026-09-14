"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const styles = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function escapeRegExp(value) {
  return String(value)
    .replace(/[.*+?^()|[\]\\]/gu, "\\$&")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}");
}

function functionSource(name) {
  const start = app.indexOf("function " + name + "(");
  assert.ok(start >= 0, name + " must exist");
  const end = app.indexOf("\nfunction ", start + 1);
  return app.slice(start, end > start ? end : app.length);
}

function blockFromOpeningBrace(source, openIndex) {
  assert.equal(source[openIndex], "{", "callback must start at an opening brace");
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(openIndex + 1, index);
    }
  }
  assert.fail("unterminated callback block");
}

function arrowCallbackBody(source, marker) {
  const start = source.indexOf(marker);
  assert.ok(start >= 0, "missing callback marker: " + marker);
  const arrow = source.indexOf("=>", start);
  assert.ok(arrow > start, "missing callback arrow for " + marker);
  const open = source.indexOf("{", arrow);
  assert.ok(open > arrow, "missing callback body for " + marker);
  return blockFromOpeningBrace(source, open);
}

function tagForId(source, tagName, id) {
  const pattern = "<" + tagName + "\\b(?=[^>]*\\bid=[\"']"
    + escapeRegExp(id) + "[\"'])[^>]*>";
  const match = source.match(new RegExp(pattern, "u"));
  assert.ok(match, "missing <" + tagName + "> with id=" + id);
  return match[0];
}

function attributeValue(tag, name) {
  const pattern = "\\b" + escapeRegExp(name)
    + "\\s*=\\s*(?:\"([^\"]*)\"|'([^']*)')";
  const match = tag.match(new RegExp(pattern, "u"));
  return match ? match[1] ?? match[2] : null;
}

function assertBilingualUiText(source, zhFragment, context) {
  const pattern = "uiText\\(\\s*\"([^\"]*" + escapeRegExp(zhFragment)
    + "[^\"]*)\"\\s*,\\s*\"([^\"]+)\"";
  const match = new RegExp(pattern, "u").exec(source);
  assert.ok(match, context + ": expected uiText(zh, en) for " + zhFragment);
  assert.notEqual(match[1], match[2], context + ": Chinese and English copy must remain distinct");
  return { index: match.index, call: match[0] };
}

function callTail(source, call, limit = 800) {
  return source.slice(call.index, call.index + limit);
}

function cssBodiesForExactSelector(source, selector) {
  const normalizedSelector = selector.replace(/\s+/gu, " ").trim();
  return Array.from(source.matchAll(/([^{}]+)\{([^{}]*)\}/gu))
    .filter((match) => match[1].replace(/\s+/gu, " ").trim() === normalizedSelector)
    .map((match) => match[2]);
}

test("architecture Space pan is reserved for direct topology-canvas targets", () => {
  const keydown = arrowCallbackBody(app, 'window.addEventListener("keydown"');
  const spaceChecks = Array.from(keydown.matchAll(/if\s*\(([^)]*event\.code\s*===\s*"Space"[^)]*)\)\s*\{/gu));
  assert.equal(spaceChecks.length, 1, "window keydown must have one architecture Space branch");
  assert.match(spaceChecks[0][1], /state\.view\s*===\s*"architecture"/u);
  assert.match(spaceChecks[0][1], /!editing/u);
  assert.match(spaceChecks[0][1], /event\.target\s*===\s*dom\.topologyCanvas/u);

  const branchEnd = spaceChecks[0].index + spaceChecks[0][0].length;
  const stateMutation = keydown.indexOf("state.spacePressed = true");
  const classMutation = keydown.indexOf('dom.topologyCanvas.classList.add("is-space-pan")');
  assert.ok(stateMutation >= branchEnd, "Space state must be mutated inside the guarded branch");
  assert.ok(classMutation >= branchEnd, "canvas pan styling must be mutated inside the guarded branch");
});

test("architecture dynamic status copy is bilingual and preserves runtime values", () => {
  const routeStatus = functionSource("updateTopologyRouteStatus");
  const routeCopy = assertBilingualUiText(routeStatus, "{count} 条链路无法安全布线", "route failure status");
  assert.match(callTail(routeStatus, routeCopy), /\{\s*count\s*:\s*routeErrors\.length\s*\}/u);
  assert.match(routeStatus, /error\.id/u, "route failure details must retain link IDs");

  const toolbar = functionSource("updateTopologyToolbar");
  assertBilingualUiText(toolbar, "展开组", "group expand status");
  assertBilingualUiText(toolbar, "折叠组", "group collapse status");
  const groupSelection = assertBilingualUiText(toolbar, "组 {label}", "group selection status");
  assert.match(callTail(toolbar, groupSelection), /label\s*:\s*group\?\.label[\s\S]*root\s*:\s*group\?\.root/u);
  const componentSelection = assertBilingualUiText(toolbar, "已选 {count} 个组件", "component selection status");
  assert.match(callTail(toolbar, componentSelection), /count\s*:\s*state\.selectedComponents\.size/u);
  assertBilingualUiText(toolbar, "已选链路", "link selection status");
  assertBilingualUiText(toolbar, "未选择", "empty selection status");

  const topologyTool = functionSource("setTopologyTool");
  const toolHint = assertBilingualUiText(topologyTool, "连接模式 · {protocol}", "connection-mode hint");
  assert.match(callTail(topologyTool, toolHint), /protocol\s*:\s*dom\.protocolSelect\.value/u);
  assertBilingualUiText(topologyTool, "选择节点或链路查看属性", "select-mode hint");

  const connectNode = functionSource("handleConnectNode");
  const sourceHint = assertBilingualUiText(connectNode, "起点 {componentId}", "connection source hint");
  assert.match(callTail(connectNode, sourceHint), /componentId\s*\}/u);
  assertBilingualUiText(connectNode, "连接已创建", "connection-created hint");

  const bindStaticEvents = functionSource("bindStaticEvents");
  const protocolChange = arrowCallbackBody(bindStaticEvents, 'dom.protocolSelect.addEventListener("change"');
  const protocolHint = assertBilingualUiText(protocolChange, "连接模式 · {protocol}", "protocol selection hint");
  assert.match(callTail(protocolChange, protocolHint), /protocol\s*:\s*dom\.protocolSelect\.value/u);

  const protocolPreset = functionSource("useProtocolPreset");
  const presetHint = assertBilingualUiText(protocolPreset, "连接模式 · {protocol}", "protocol preset hint");
  assert.match(callTail(protocolPreset, presetHint), /protocol\s*:\s*detail\.protocol/u);
});

test("architecture topology canvas separates horizontal containment from vertical chaining", () => {
  const bodies = cssBodiesForExactSelector(styles, ".topology-canvas");
  assert.ok(bodies.length, "base .topology-canvas rule must exist");
  for (const body of bodies) {
    assert.match(body, /overscroll-behavior-x\s*:\s*contain\s*;/u);
    assert.match(body, /overscroll-behavior-y\s*:\s*auto\s*;/u);
    assert.doesNotMatch(body, /overscroll-behavior\s*:\s*contain\s*;/u);
  }
});

test("trace concept help belongs to the playback heading, not the live fidelity badge", () => {
  const title = tagForId(html, "h1", "playbackTitle");
  assert.equal(attributeValue(title, "data-concept-help"), "trace");
  const fidelity = tagForId(html, "strong", "playbackFidelity");
  assert.equal(attributeValue(fidelity, "data-concept-help"), null);
});

test("leaving playback and resetting trace clear fullscreen state and stale focus", () => {
  const clear = functionSource("clearTraceFullscreenState");
  assert.match(clear, /state\.tracePlayback\.fullscreen\s*=\s*false/u);
  assert.match(clear, /document\.documentElement\?\.classList\?\.remove\?\.\(\s*"trace-fullscreen-open"\s*\)/u);
  assert.match(clear, /traceFullscreenPreviousFocus\s*=\s*null/u);

  const switchView = functionSource("switchView");
  assert.match(switchView, /if\s*\(\s*viewName\s*!==\s*"playback"\s*\)\s*clearTraceFullscreenState\(\)/u);

  const reset = functionSource("resetTracePlaybackState");
  const clearIndex = reset.indexOf("clearTraceFullscreenState()");
  const replacementIndex = reset.indexOf("state.tracePlayback = emptyTracePlaybackState()");
  assert.ok(clearIndex >= 0, "trace reset must clear fullscreen state");
  assert.ok(replacementIndex > clearIndex, "trace reset must clear fullscreen state before replacing playback data");
});
