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
  return String(value).replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function functionSource(name) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} must exist`);
  const end = app.indexOf("\nfunction ", start + 1);
  return app.slice(start, end > start ? end : app.length);
}

function blockFromOpeningBrace(source, openIndex) {
  assert.equal(source[openIndex], "{", "block must start at an opening brace");
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(openIndex + 1, index);
    }
  }
  assert.fail("unterminated block");
}

function arrowCallbackBody(source, marker) {
  const start = source.indexOf(marker);
  assert.ok(start >= 0, `missing callback marker: ${marker}`);
  const arrow = source.indexOf("=>", start);
  assert.ok(arrow > start, `missing callback arrow for ${marker}`);
  const open = source.indexOf("{", arrow);
  assert.ok(open > arrow, `missing callback body for ${marker}`);
  return blockFromOpeningBrace(source, open);
}

function blockAfterMarker(source, marker) {
  const start = source.indexOf(marker);
  assert.ok(start >= 0, `missing marker: ${marker}`);
  const open = source.indexOf("{", start);
  assert.ok(open > start, `missing block for marker: ${marker}`);
  return blockFromOpeningBrace(source, open);
}

function assertOrder(source, markers, context) {
  let previous = -1;
  for (const marker of markers) {
    const index = source.indexOf(marker);
    assert.ok(index >= 0, `${context}: missing ${marker}`);
    assert.ok(index > previous, `${context}: expected ${marker} after prior marker`);
    previous = index;
  }
}

function assertBilingualUiText(source, zhFragment, context) {
  const pattern = "uiText\\(\\s*\"([^\"]*" + escapeRegExp(zhFragment)
    + "[^\"]*)\"\\s*,\\s*\"([^\"]+)\"";
  const match = new RegExp(pattern, "u").exec(source);
  assert.ok(match, `${context}: expected uiText(zh, en) for ${zhFragment}`);
  assert.notEqual(match[1], match[2], `${context}: Chinese and English copy must stay distinct`);
  return match[0];
}

function tagForId(source, tagName, id) {
  const match = source.match(new RegExp(`<${tagName}\\b(?=[^>]*\\bid=["']${escapeRegExp(id)}["'])[^>]*>`, "u"));
  assert.ok(match, `missing <${tagName}> with id=${id}`);
  return match[0];
}

function attributeValue(tag, name) {
  const match = tag.match(new RegExp(`\\b${escapeRegExp(name)}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "u"));
  return match ? match[1] ?? match[2] : null;
}

function normalizeSelector(selector) {
  return selector.replace(/\s+/gu, " ").trim();
}

function cssRules(source = styles) {
  return Array.from(source.matchAll(/([^{}]+)\{([^{}]*)\}/gu), (match) => ({
    selector: normalizeSelector(match[1]),
    body: match[2],
  })).filter((rule) => !rule.selector.startsWith("@"));
}

function selectorTargets(selector, target) {
  const normalized = normalizeSelector(selector);
  return normalized.split(",").some((item) => item.trim() === target);
}

function cssBodiesTargeting(target, source = styles) {
  return cssRules(source)
    .filter((rule) => selectorTargets(rule.selector, target))
    .map((rule) => rule.body);
}

function declarationValue(body, property) {
  const match = body.match(new RegExp(`${escapeRegExp(property)}\\s*:\\s*([^;]+);`, "u"));
  return match ? match[1].trim() : null;
}

function lastDeclarationForTarget(target, property, source = styles) {
  let value = null;
  for (const body of cssBodiesTargeting(target, source)) {
    const candidate = declarationValue(body, property);
    if (candidate) value = candidate;
  }
  assert.ok(value, `missing ${property} for ${target}`);
  return value;
}

function assertSemanticCssValue(target, property, tokens) {
  const value = lastDeclarationForTarget(target, property);
  assert.ok(
    tokens.some((token) => value.includes(`var(${token}`)),
    `${target} ${property} must use one of ${tokens.join(", ")}, got ${value}`,
  );
}

function assertSemanticCssAnyValue(target, properties, tokens) {
  const candidates = properties.map((property) => {
    try {
      return [property, lastDeclarationForTarget(target, property)];
    } catch (_error) {
      return null;
    }
  }).filter(Boolean);
  assert.ok(candidates.length, `missing any of ${properties.join(", ")} for ${target}`);
  assert.ok(
    candidates.some(([, value]) => tokens.some((token) => value.includes(`var(${token}`))),
    `${target} ${properties.join("/")} must use one of ${tokens.join(", ")}, got ${candidates.map(([property, value]) => `${property}: ${value}`).join("; ")}`,
  );
}

function atRuleBlockContaining(pattern, marker) {
  for (const match of styles.matchAll(new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`))) {
    const openIndex = styles.indexOf("{", match.index);
    assert.ok(openIndex > match.index, `missing CSS block for ${pattern}`);
    const block = blockFromOpeningBrace(styles, openIndex);
    if (block.includes(marker)) return block;
  }
  assert.fail(`missing CSS at-rule ${pattern} containing ${marker}`);
}

test("model status refreshes after selection and global localization while prioritizing inline operators", () => {
  const statusTag = tagForId(html, "span", "modelGraphStatus");
  assert.equal(attributeValue(statusTag, "role"), "status");
  assert.equal(attributeValue(statusTag, "aria-live"), "polite");

  const renderAll = functionSource("renderAll");
  assertOrder(
    renderAll,
    [
      "globalThis.UiI18n?.localize?.(document)",
      "renderModelGraphDiagnostics()",
    ],
    "global localization must be followed by dynamic model status refresh",
  );

  const updateUiSettings = functionSource("updateUiSettings");
  const languageBranch = blockAfterMarker(updateUiSettings, "if (state.settings.language !== previousLanguage)");
  assertOrder(
    languageBranch,
    [
      "globalThis.UiI18n?.setLanguage?.(state.settings.language, document)",
      "renderAll()",
    ],
    "language changes must rerun the full localized render path",
  );

  const diagnostics = functionSource("renderModelGraphDiagnostics");
  assert.match(diagnostics, /selectedOperatorId/u);
  assert.match(
    diagnostics,
    /const fallback\s*=\s*selectedOperator\s*\?\s*uiText\([\s\S]*:\s*uiText\(\s*selected/u,
    "status fallback must prefer selectedOperatorId before overview selection",
  );
  assertBilingualUiText(diagnostics, "已选择 {name}", "selected operator model status");

  const select = functionSource("selectModelGraphOperatorElement");
  assert.match(select, /refreshModelGraphInlineSelectionState\(operatorId\)/u);
  assert.match(select, /renderModelGraphDiagnostics\(\)/u, "inline/detail selection must refresh the live status text");
});

test("inline selection state mirrors visual and aria state and collapse clears hidden child selection", () => {
  const refresh = functionSource("refreshModelGraphInlineSelectionState");
  assert.match(refresh, /node\.classList\.toggle\("is-selected",\s*selected\)/u);
  assert.match(refresh, /node\.setAttribute\("aria-pressed",\s*String\(selected\)\)/u);
  assert.match(refresh, /modelGraphInlineEdgeDirection\(/u);
  assert.match(refresh, /data-model-route-source-operator/u);
  assert.match(refresh, /data-model-route-target-operator/u);

  const select = functionSource("selectModelGraphOperatorElement");
  assert.match(select, /classList\.remove\("is-selected"\)/u);
  assert.match(select, /setAttribute\("aria-pressed",\s*String\(node === inlineNode\)\)/u);
  assert.match(select, /selectedNode\.classList\.add\("is-selected"\)/u);

  const clickHandler = arrowCallbackBody(functionSource("bindModelGraphDynamicEvents"), 'dom.modelGraphCanvas.addEventListener("click"');
  const collapseBranch = blockAfterMarker(clickHandler, "if (overviewGroupButton)");
  assert.match(collapseBranch, /willCollapse/u);
  assert.match(collapseBranch, /selectedOperatorId/u);
  assert.match(
    collapseBranch,
    /(?:node\.operator_ids|node\.representative_operator_id|inline_graph|group_ids)/u,
    "collapse must check whether the selected inline/detail child becomes hidden",
  );
  assert.match(
    collapseBranch,
    /selectedOperatorId\s*=\s*(?:null|""|undefined)/u,
    "collapsing a repeat group must clear the now-hidden child operator selection",
  );
  assert.match(
    collapseBranch,
    /selectedOverviewId\s*=\s*node\.display_id|selectModelGraphOverviewComponent\(/u,
    "collapsing a repeat group must keep or restore overview group selection",
  );
  assert.match(collapseBranch, /renderModelGraphDiagnostics\(\)|renderModelGraph\(\)|autoLayoutModelGraph\(/u);
});

test("detail and inline model nodes expose button semantics and share Enter and Space activation", () => {
  const render = functionSource("renderModelGraph");
  assert.match(render, /class="model-graph-node[\s\S]*data-model-operator="\$\{escapeHtml\(operator\.operator_id\)\}"/u);
  assert.match(render, /role="button"/u);
  assert.match(render, /tabindex="0"/u);
  assert.match(render, /aria-pressed="\$\{String\(selected\)\}"/u);
  assertBilingualUiText(render, "{name}；算子 ID", "detail node accessible label");

  const inline = functionSource("renderModelInlineAuthoritativeGraph");
  assert.match(inline, /class="model-inline-operator[\s\S]*data-model-select-operator="\$\{escapeHtml\(operator\.operator_id\)\}"/u);
  assert.match(inline, /role="button"/u);
  assert.match(inline, /tabindex="0"/u);
  assert.match(inline, /aria-pressed="\$\{String\(state\.modelGraphEditor\.selectedOperatorId === operator\.operator_id\)\}"/u);
  assertBilingualUiText(inline, "{name}；算子 ID", "inline node accessible label");

  const keydown = arrowCallbackBody(functionSource("bindModelGraphDynamicEvents"), 'dom.modelGraphCanvas.addEventListener("keydown"');
  assert.match(keydown, /\[\s*['"]Enter['"]\s*,\s*['"] ['"]\s*\]\.includes\(event\.key\)/u);
  assert.match(keydown, /\.model-inline-operator\[data-model-select-operator\]/u);
  assert.match(keydown, /\.model-graph-node\[data-model-operator\]/u);
  assert.match(
    keydown,
    /event\.target\s*(?:===|!==)\s*(?:node|selectable|selectableNode|target|inlineNode|detailNode)/u,
  );
  assert.match(keydown, /selectModelGraphOperatorElement\(/u);
});

test("authoritative inline SVG has image semantics and bilingual label", () => {
  const inline = functionSource("renderModelInlineAuthoritativeGraph");
  assert.match(inline, /<svg\b[^`]*role="img"[^`]*aria-label="\$\{escapeHtml\(uiText\(/u);
  assertBilingualUiText(inline, "权威张量连接", "authoritative inline SVG label");
  assert.match(inline, /data-authoritative-subgraph="true"/u);
  assert.match(inline, /data-authoritative="true"/u);
  assert.match(inline, /data-model-inline-edge-id/u);
});

test("model inspector parameter labels and port directions are localized through shared helpers", () => {
  const parameterLabel = functionSource("modelGraphParameterLabel");
  assert.match(app, /const MODEL_GRAPH_PARAMETER_LABELS_EN = Object\.freeze/u);
  assert.match(parameterLabel, /uiText\(/u);
  assert.match(parameterLabel, /MODEL_GRAPH_PARAMETER_LABELS\[key\]/u);
  assert.match(parameterLabel, /MODEL_GRAPH_PARAMETER_LABELS_EN\[key\]/u);

  const directionLabel = functionSource("modelPortDirectionLabel");
  assert.match(directionLabel, /input:\s*\["输入",\s*"Input"\]/u);
  assert.match(directionLabel, /output:\s*\["输出",\s*"Output"\]/u);
  assert.match(directionLabel, /weight:\s*\["权重",\s*"Weight"\]/u);
  assert.match(directionLabel, /uiText\(pair\[0\],\s*pair\[1\]\)/u);

  const inspector = functionSource("renderModelGraphInspector");
  assertBilingualUiText(inspector, "组件参数", "detail inspector parameter section");
  assertBilingualUiText(inspector, "Block 组参数", "detail inspector group-template section");
  assert.match(inspector, /modelGraphParameterLabel\(key\)/u);
  assert.match(inspector, /modelPortDirectionLabel\(item\.direction\)/u);
  assertBilingualUiText(inspector, "数据类型", "port dtype editor label");
  assertBilingualUiText(inspector, "形状", "port shape editor label");

  const overviewInspector = functionSource("renderModelGraphOverviewInspector");
  assertBilingualUiText(overviewInspector, "组件参数", "overview inspector parameter section");
  assert.match(overviewInspector, /modelGraphParameterLabel\(key\)/u);
  assert.match(overviewInspector, /modelPortDirectionLabel\(port\.direction\)/u);
});

test("model graph responsive layout and route colors use semantic contracts", () => {
  const responsive = atRuleBlockContaining(/@media\s*\(max-width:\s*1280px\)/u, ".model-graph-shell");
  assert.match(responsive, /\.model-graph-shell\s*\{[\s\S]*grid-template-columns:\s*minmax\(0,\s*1fr\)/u);
  assert.match(responsive, /\.model-graph-inspector\s*\{[\s\S]*border-top:\s*1px solid var\(--line\)/u);
  assert.match(responsive, /\.model-graph-inspector\s*\{[\s\S]*border-left:\s*0/u);
  assert.match(responsive, /#view-model\s*>\s*\.view-head\s*\{[\s\S]*flex-direction:\s*column/u);
  assert.match(responsive, /#view-model\s+\.view-head-actions\s*\{[\s\S]*width:\s*100%/u);
  assert.match(responsive, /#view-model\s+\.view-head-actions\s+\.button\s*\{[\s\S]*white-space:\s*normal/u);

  assertSemanticCssValue(".model-graph-edge", "stroke", ["--cyan", "--cyan-bright", "--text-soft"]);
  assertSemanticCssValue(".model-graph-edge.is-overview", "stroke", ["--cyan", "--cyan-bright", "--text-soft"]);
  assertSemanticCssValue(".model-inline-edge", "stroke", ["--cyan", "--cyan-bright"]);
  assertSemanticCssValue(".model-inline-edge.is-residual", "stroke", ["--copper", "--copper-bright", "--warning"]);
  assertSemanticCssValue(".model-graph-edge.is-connection-preview", "stroke", ["--cyan-bright"]);
  assertSemanticCssValue(".model-graph-edge.is-connection-preview.is-compatible", "stroke", ["--success"]);
  assertSemanticCssValue(".model-graph-edge.is-connection-preview.is-incompatible", "stroke", ["--danger"]);
  assertSemanticCssAnyValue(".model-graph-port i", ["background", "background-color"], ["--cyan", "--cyan-bright"]);
  assertSemanticCssAnyValue(".model-graph-port.is-output i", ["background", "background-color"], ["--copper", "--copper-bright"]);
  assertSemanticCssAnyValue(".model-graph-port.is-weight i", ["background", "background-color"], ["--warning", "--copper"]);
  assertSemanticCssAnyValue(".model-inline-port", ["border", "border-color"], ["--line", "--cyan", "--text-soft"]);
});
