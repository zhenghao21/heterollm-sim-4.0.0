"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const app = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));
const UiI18n = require(path.join(webui, "ui-i18n.js"));

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
  assert.ok(start >= 0, `missing event listener: ${marker}`);
  const arrow = source.indexOf("=>", start);
  assert.ok(arrow > start, `missing arrow callback for ${marker}`);
  const open = source.indexOf("{", arrow);
  assert.ok(open > arrow, `missing callback block for ${marker}`);
  return blockFromOpeningBrace(source, open);
}

function tagForId(source, tagName, id) {
  const pattern = new RegExp(`<${tagName}\\b(?=[^>]*\\bid=["']${id}["'])[^>]*>`, "u");
  const match = source.match(pattern);
  assert.ok(match, `missing <${tagName}> with id=${id}`);
  return match[0];
}

function attributeValue(tag, name) {
  const pattern = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "u");
  const match = tag.match(pattern);
  return match ? match[1] ?? match[2] : null;
}

function normalizeSelector(selector) {
  return selector.replace(/\s+/gu, " ").trim();
}

function cssRules(source) {
  return Array.from(source.matchAll(/([^{}]+)\{([^{}]*)\}/gu), (match) => ({
    selector: normalizeSelector(match[1]),
    body: match[2],
  })).filter((rule) => !rule.selector.startsWith("@"));
}

function cssBodiesContaining(source, selectorFragment) {
  return cssRules(source)
    .filter((rule) => normalizeSelector(rule.selector).includes(selectorFragment))
    .map((rule) => rule.body);
}

function assertCssContract(source, selectorFragment, patterns) {
  const bodies = cssBodiesContaining(source, selectorFragment);
  assert.ok(bodies.length, `missing CSS selector containing ${selectorFragment}`);
  for (const pattern of patterns) {
    assert.ok(
      bodies.some((body) => pattern.test(body)),
      `no ${selectorFragment} rule matched ${pattern}`,
    );
  }
}

function atRuleBlockContaining(pattern, marker) {
  for (const match of css.matchAll(new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`))) {
    const openIndex = css.indexOf("{", match.index);
    assert.ok(openIndex > match.index, `missing CSS block for ${pattern}`);
    const block = blockFromOpeningBrace(css, openIndex);
    if (block.includes(marker)) return block;
  }
  assert.fail(`missing CSS at-rule ${pattern} containing ${marker}`);
}

function declarationValue(body, property) {
  const escaped = property.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const match = body.match(new RegExp(`${escaped}\\s*:\\s*([^;]+);`, "u"));
  return match ? match[1].trim() : null;
}

function declarations(body) {
  const result = {};
  for (const match of body.matchAll(/(--[-\w]+|[-\w]+)\s*:\s*([^;]+);/gu)) {
    result[match[1]] = match[2].trim();
  }
  return result;
}

function selectorAppliesToTheme(selector, theme) {
  return !selector.includes("[data-theme=") || selector.includes(`[data-theme="${theme}"]`) || selector.includes(`[data-theme='${theme}']`);
}

function selectorTargets(selector, target) {
  const normalized = normalizeSelector(selector);
  if (!normalized.includes(target)) return false;
  const after = normalized.slice(normalized.indexOf(target) + target.length, normalized.indexOf(target) + target.length + 1);
  return target.includes(" ") || !after || /[,>{+~]/u.test(after);
}

function selectorAppliesToFontBand(selector, band) {
  return selector.includes(`[data-font-band="${band}"]`) || selector.includes(`[data-font-band='${band}']`);
}

function cssBodiesTargeting(source, target, predicate = () => true) {
  return cssRules(source)
    .filter((rule) => selectorTargets(rule.selector, target) && predicate(rule))
    .map((rule) => rule.body);
}

function assertAnyCssContract(source, selectorFragments, patterns) {
  const bodies = selectorFragments.flatMap((fragment) => cssBodiesContaining(source, fragment));
  assert.ok(bodies.length, `missing CSS selector containing any of ${selectorFragments.join(", ")}`);
  for (const pattern of patterns) {
    assert.ok(
      bodies.some((body) => pattern.test(body)),
      `no ${selectorFragments.join(" / ")} rule matched ${pattern}`,
    );
  }
}

function assertFontBandCssContract(band, target, patterns) {
  const bodies = cssBodiesTargeting(css, target, (rule) => selectorAppliesToFontBand(rule.selector, band));
  assert.ok(bodies.length, `missing ${band} CSS selector targeting ${target}`);
  for (const pattern of patterns) {
    assert.ok(
      bodies.some((body) => pattern.test(body)),
      `no ${band} ${target} rule matched ${pattern}`,
    );
  }
}

function variablesForTheme(theme) {
  const variables = {};
  for (const rule of cssRules(css)) {
    if (!rule.selector.includes(":root") || !selectorAppliesToTheme(rule.selector, theme)) continue;
    Object.assign(variables, declarations(rule.body));
  }
  return variables;
}

function lastThemedDeclarationForTarget(target, property, theme, { requireThemeSelector = false } = {}) {
  let value = null;
  let selector = "";
  for (const rule of cssRules(css)) {
    if (!selectorAppliesToTheme(rule.selector, theme)) continue;
    if (requireThemeSelector && !rule.selector.includes("[data-theme=")) continue;
    if (!selectorTargets(rule.selector, target)) continue;
    const candidate = declarationValue(rule.body, property);
    if (candidate) {
      value = candidate;
      selector = rule.selector;
    }
  }
  assert.ok(value, `missing ${property} for ${target} in ${theme}`);
  return { selector, value };
}

function splitTopLevel(value) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "(") depth += 1;
    else if (value[index] === ")") depth -= 1;
    else if (value[index] === "," && depth === 0) {
      parts.push(value.slice(start, index).trim());
      start = index + 1;
    }
  }
  parts.push(value.slice(start).trim());
  return parts;
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return [
    Number.parseInt(clean.slice(0, 2), 16),
    Number.parseInt(clean.slice(2, 4), 16),
    Number.parseInt(clean.slice(4, 6), 16),
  ];
}

function rgbToHex([red, green, blue]) {
  return `#${[red, green, blue].map((channel) => Math.round(channel).toString(16).padStart(2, "0")).join("")}`;
}

function resolveColor(value, variables) {
  const trimmed = String(value || "").trim();
  const directHex = trimmed.match(/^#[0-9a-f]{6}\b/iu);
  if (directHex) return directHex[0].toLowerCase();

  const varMatch = trimmed.match(/^var\((--[-\w]+)(?:,\s*(.+))?\)$/u);
  if (varMatch) {
    const [, name, fallback] = varMatch;
    return resolveColor(variables[name] ?? fallback, variables);
  }

  if (trimmed.startsWith("color-mix(") && trimmed.endsWith(")")) {
    const inner = trimmed.slice("color-mix(".length, -1).trim();
    const prefix = "in srgb,";
    assert.ok(inner.startsWith(prefix), `unsupported color-mix syntax: ${trimmed}`);
    const [first, second] = splitTopLevel(inner.slice(prefix.length));
    const firstMatch = first.match(/^(.+?)\s+([\d.]+)%$/u);
    assert.ok(firstMatch, `unsupported first color-mix stop: ${first}`);
    const secondMatch = second.match(/^(.+?)(?:\s+([\d.]+)%)?$/u);
    const firstPercent = Number(firstMatch[2]) / 100;
    const secondPercent = secondMatch[2] == null ? 1 - firstPercent : Number(secondMatch[2]) / 100;
    const firstRgb = hexToRgb(resolveColor(firstMatch[1], variables));
    const secondRgb = hexToRgb(resolveColor(secondMatch[1], variables));
    return rgbToHex(firstRgb.map((channel, index) => channel * firstPercent + secondRgb[index] * secondPercent));
  }

  const nestedHex = trimmed.match(/#[0-9a-f]{6}\b/iu);
  if (nestedHex) return nestedHex[0].toLowerCase();
  assert.fail(`cannot resolve CSS color: ${trimmed}`);
}

function relativeLuminance(hex) {
  const [red, green, blue] = hexToRgb(hex).map((channel) => {
    const scaled = channel / 255;
    return scaled <= 0.03928 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground, background) {
  const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background));
  const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

function elementStub() {
  return {
    innerHTML: "",
    textContent: "",
    hidden: false,
    dataset: {},
    style: { removeProperty() {}, setProperty() {} },
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
}

function loadPass07Harness() {
  UiI18n.setLanguage("zh-CN", null);
  const sandbox = {
    AbortController,
    CSS: { escape: String },
    Intl,
    Map,
    ModelGraphCore,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    UiI18n,
    URL,
    URLSearchParams,
    clearTimeout,
    console,
    document: {
      addEventListener() {},
      body: { appendChild() {} },
      createElement() { return elementStub(); },
      documentElement: { dataset: {}, style: {} },
      getElementById() { return null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    matchMedia: () => ({ matches: false }),
    requestAnimationFrame(callback) { callback(0); return 1; },
    setTimeout,
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(`${app}
;globalThis.__pass07 = {
  CONCEPT_HELP_COVERAGE_BY_VIEW,
  CONCEPT_HELP_LABEL_BINDING_MAP,
  conceptHelpText,
  dom,
  normalizedConceptHelpLabel,
  renderRequestResults,
  runManifestMarkup,
};`, context, { filename: appPath });
  return context.__pass07;
}

function renderRequestResultRowLabels(language) {
  const ui = loadPass07Harness();
  UiI18n.setLanguage(language, null);
  ui.dom.requestResultMeta = elementStub();
  ui.dom.requestResultBody = elementStub();
  ui.renderRequestResults({
    req_01: {
      arrival_ns: 0,
      status: "completed",
      ttft_ns: 1200,
      tbt_ns: [2000, 3000, 4000],
      tpot_ns: 2500,
      e2e_ns: 9000,
      visible_output_tokens: 4,
    },
  });
  return Array.from(ui.dom.requestResultBody.innerHTML.matchAll(/<td\b[^>]*\bdata-label="([^"]*)"/gu), (match) => match[1]);
}

test("effective mapping filters reflow without collapsed controls at narrow and extreme font sizes", () => {
  assertCssContract(css, ".effective-mapping-filters", [
    /min-width:\s*0/u,
    /grid-template-columns:\s*minmax\([^;]*\)\s+repeat\(2,\s*minmax\([^;]*\)\)\s+auto/u,
  ]);
  assertAnyCssContract(css, [
    ".effective-mapping-filters .field",
    ".effective-mapping-filters > .field",
    ".effective-mapping-filters > *",
    ".effective-mapping-filters :is(.field, .button)",
  ], [
    /min-width:\s*0/u,
  ]);
  assertAnyCssContract(css, [
    ".effective-mapping-filters .button",
    ".effective-mapping-filters > .button",
    ".effective-mapping-filters button",
    ".effective-mapping-filters > button",
    ".effective-mapping-filters :is(input, select, .button)",
  ], [
    /min-width:\s*(?:min|max|clamp|calc)\(/u,
  ]);

  const narrow = atRuleBlockContaining(/@media\s*\(max-width:\s*900px\)/u, ".effective-mapping-filters");
  assertCssContract(narrow, ".effective-mapping-filters", [
    /grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/u,
  ]);
  const phone = atRuleBlockContaining(/@media\s*\(max-width:\s*560px\)/u, ".effective-mapping-filters");
  assertCssContract(phone, ".effective-mapping-filters", [
    /grid-template-columns:\s*minmax\(0,\s*1fr\)/u,
  ]);
  assertFontBandCssContract("extreme", ".effective-mapping-filters", [
    /grid-template-columns:\s*minmax\(0,\s*1fr\)/u,
  ]);
});

test("request result rows emit stable bilingual data labels for all nine columns", () => {
  const table = tagForId(html, "tbody", "requestResultBody");
  assert.equal(attributeValue(table, "id"), "requestResultBody");

  assert.deepEqual(renderRequestResultRowLabels("zh-CN"), [
    "请求（Request）",
    "状态（Status）",
    "拒绝原因（Rejection Reason）",
    "到达时间（Arrival）",
    "首 Token 延迟（TTFT）",
    "Token 间延迟（TBT）p50 / p95",
    "每输出 Token 时间（TPOT）",
    "端到端延迟（E2E）",
    "可见 Token 数（Visible Tokens）",
  ]);
  assert.deepEqual(renderRequestResultRowLabels("en"), [
    "Request",
    "Status",
    "Rejection Reason",
    "Arrival",
    "Time to First Token (TTFT)",
    "Time Between Tokens (TBT) p50 / p95",
    "Time per Output Token (TPOT)",
    "End-to-End Latency (E2E)",
    "Visible Tokens",
  ]);
});

test("request result table keeps desktop columns but becomes readable cards for large fonts", () => {
  assertCssContract(css, ".request-result-table-shell", [
    /overflow-x:\s*auto/u,
  ]);
  assertCssContract(css, ".request-result-table", [
    /min-width:\s*calc\(1260px\s*\*\s*var\(--layout-scale\)\)/u,
    /table-layout:\s*fixed/u,
  ]);

  for (const band of ["large", "extreme"]) {
    assertFontBandCssContract(band, ".request-result-table thead", [
      /position:\s*absolute\s*!important/u,
      /width:\s*1px\s*!important/u,
      /overflow:\s*hidden\s*!important/u,
      /visibility:\s*hidden\s*!important/u,
    ]);
    assertFontBandCssContract(band, ".request-result-table", [
      /display:\s*block/u,
      /width:\s*100%/u,
    ]);
    assertFontBandCssContract(band, ".request-result-table tr", [
      /display:\s*block/u,
      /border:\s*1px solid var\(--line\)/u,
    ]);
    assertFontBandCssContract(band, ".request-result-table td", [
      /display:\s*grid/u,
      /grid-template-columns:\s*minmax\(/u,
    ]);
    assertFontBandCssContract(band, ".request-result-table td::before", [
      /content:\s*attr\(data-label\)/u,
    ]);
  }

  const narrow = atRuleBlockContaining(/@media\s*\(max-width:\s*1060px\)/u, ".request-result-table-shell:has(td[data-label])");
  assertCssContract(narrow, ".request-result-table", [
    /display:\s*block/u,
    /min-width:\s*0/u,
  ]);
  assertCssContract(narrow, ".request-result-table thead", [
    /visibility:\s*hidden\s*!important/u,
  ]);
  assertCssContract(narrow, ".request-result-table td[data-label]", [
    /display:\s*grid/u,
    /grid-template-columns:\s*minmax\(/u,
  ]);
});

test("utilization percentages reserve readable inline space", () => {
  assertCssContract(css, ".util-row", [
    /grid-template-columns:\s*minmax\([^)]*\)\s+minmax\([^)]*\)\s+minmax\(/u,
  ]);
  assertCssContract(css, ".util-value", [
    /min-width:\s*(?:[3-9](?:\.\d+)?rem|calc\(|clamp\(|max\()/u,
    /white-space:\s*nowrap/u,
  ]);
});

test("light themes keep rail text and selected model edges above contrast thresholds", () => {
  const railTargets = [".step-button", ".step-number", ".step-copy small", ".step-count", ".rail-stat > span", ".rail-stat strong"];
  for (const theme of ["ivory", "mist", "softgray"]) {
    const variables = variablesForTheme(theme);
    const railBackground = resolveColor(variables["--rail-bg"], variables);
    for (const target of railTargets) {
      const { value } = lastThemedDeclarationForTarget(target, "color", theme);
      const foreground = resolveColor(value, variables);
      assert.ok(
        contrastRatio(foreground, railBackground) >= 4.5,
        `${theme} ${target} color ${foreground} must contrast with rail ${railBackground}`,
      );
    }

    for (const direction of ["incoming", "outgoing"]) {
      const { selector, value } = lastThemedDeclarationForTarget(`.is-one-hop.is-${direction}`, "stroke", theme, { requireThemeSelector: true });
      assert.match(selector, /\.is-one-hop\.is-(?:incoming|outgoing)/u);
      const stroke = resolveColor(value, variables);
      const canvas = resolveColor(variables["--canvas-bg"], variables);
      assert.ok(
        contrastRatio(stroke, canvas) >= 3,
        `${theme} selected ${direction} model edge ${stroke} must contrast with canvas ${canvas}`,
      );
    }
  }
});

test("analytical result badge uses analytical-report help while technical details keep run-manifest help", () => {
  const ui = loadPass07Harness();
  const markup = ui.runManifestMarkup({
    manifest: {
      evidence: "analytical",
      run_id: "run-pass07",
      schema_version: "1",
      simulator_version: "test",
      model_name: "long-model-name",
      hardware_name: "test-hardware",
      workload_name: "test-workload",
    },
    summary: { task_count: 9 },
  });
  const summaryTag = markup.match(/<div\b(?=[^>]*\brun-manifest-summary\b)[^>]*>/u)?.[0] || "";
  assert.equal(attributeValue(summaryTag, "data-concept-help"), "analytical_report");
  assert.match(
    markup,
    /<(?:details|summary)\b[^>]*\b(?:class="[^"]*\brun-technical-details\b[^"]*"[^>]*data-concept-help="run_manifest"|data-concept-help="run_manifest"[^>]*(?:class="[^"]*\brun-technical-details\b[^"]*")?)/u,
  );

  assert.ok(
    ui.CONCEPT_HELP_COVERAGE_BY_VIEW.results.includes("analytical_report"),
    "results concept coverage must include analytical_report",
  );
  assert.notEqual(
    ui.conceptHelpText("analytical_report"),
    "This concept is documented by the current analytical interface.",
  );
});

test("model graph wheel zoom only captures modified wheel gestures", () => {
  const listener = arrowCallbackBody(functionSource("bindStaticEvents"), 'dom.modelGraphCanvas.addEventListener("wheel"');
  const guard = listener.match(/if\s*\(\s*(?:!\s*\(\s*event\.ctrlKey\s*\|\|\s*event\.metaKey\s*\)|!\s*event\.ctrlKey\s*&&\s*!\s*event\.metaKey)\s*\)\s*(?:\{\s*)?return\s*;/u);
  assert.ok(guard, "model graph wheel listener must return before zoom unless ctrl/meta is pressed");
  const preventIndex = listener.indexOf("event.preventDefault()");
  assert.ok(preventIndex >= 0, "modified model graph wheel gestures still prevent default before zooming");
  assert.ok(guard.index < preventIndex, "ordinary wheel must return before preventDefault");
  assert.match(listener, /zoomModelGraph\(/u);
});

test("long model summary values remain discoverable through title and aria text", () => {
  const summaryFact = functionSource("modelSummaryFact");
  assert.match(summaryFact, /const displayValue = String\(value \?\? "—"\)/u);
  assert.match(summaryFact, /title="\$\{escapeHtml\(displayValue\)\}"/u);
  assert.match(summaryFact, /aria-label="\$\{escapeHtml\(displayValue\)\}"/u);
  assert.match(functionSource("renderModel"), /modelSummaryFact\("模型名称（Model Name）", "name", model\.name\)/u);
});

test("time-series charts shrink to measured cards and follow responsive width changes", () => {
  const render = functionSource("renderComponentTimeseries");
  assert.match(render, /timeseriesRenderedCardWidth\(\)/u);
  assert.match(render, /Math\.abs\(renderedCardWidth - chartWidth\) >= 1/u);
  const resizeRender = functionSource("scheduleComponentTimeseriesResizeRender");
  assert.match(resizeRender, /state\.view !== "results"/u);
  const observer = functionSource("bindComponentTimeseriesResizeObserver");
  assert.match(observer, /new globalThis\.ResizeObserver/u);
  assert.match(observer, /scheduleComponentTimeseriesResizeRender\(\)/u);
  const settings = functionSource("updateUiSettings");
  assert.match(settings, /resultsLayoutChanged/u);
  assert.match(settings, /componentTimeseriesObservedWidth = null/u);
  assert.match(settings, /scheduleComponentTimeseriesResizeRender\(\)/u);

  const narrow = atRuleBlockContaining(/@media\s*\(max-width:\s*1060px\)/u, ".timeseries-chart-svg");
  assertCssContract(narrow, ".timeseries-chart-svg", [/min-width:\s*0/u]);
  for (const band of ["large", "extreme"]) {
    assertFontBandCssContract(band, ".timeseries-chart-svg", [/min-width:\s*0/u]);
  }
});
