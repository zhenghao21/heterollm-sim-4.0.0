"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const preloadPath = path.join(webui, "ui-settings-preload.js");
const app = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const preload = fs.readFileSync(preloadPath, "utf8");
const I18n = require(path.join(webui, "ui-i18n.js"));
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function decodeEntities(value) {
  return String(value || "")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

function parseAttributes(source) {
  const attributes = {};
  for (const match of String(source || "").matchAll(/\s([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:="([^"]*)")?/gu)) {
    attributes[match[1]] = match[2] ?? "";
  }
  return attributes;
}

function sourceBetween(startName, endName) {
  const start = app.indexOf(`function ${startName}`);
  const end = app.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

function cssRuleBody(selector) {
  const match = css.match(new RegExp(`${escapeRegex(selector)}\\s*\\{([^}]*)\\}`, "s"));
  assert.ok(match, `missing CSS rule: ${selector}`);
  return match[1];
}

function cssBodiesContaining(selectorFragment) {
  const bodies = [];
  for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/gu)) {
    if (match[1].includes(selectorFragment)) bodies.push(match[2]);
  }
  return bodies;
}

function assertCssContract(selectorFragment, bodyPattern, message) {
  const bodies = cssBodiesContaining(selectorFragment);
  assert.ok(bodies.length, `missing CSS selector containing ${selectorFragment}`);
  assert.ok(
    bodies.some((body) => bodyPattern.test(body)),
    message || `no ${selectorFragment} rule matched ${bodyPattern}`,
  );
}

function normalizeConceptLabel(value) {
  return String(value || "").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

function createAppRuntime() {
  const frameCallbacks = [];
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
    Date,
    Intl,
    Map,
    ModelGraphCore,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    URL,
    URLSearchParams,
    UiI18n: I18n,
    cancelAnimationFrame() {},
    clearInterval,
    clearTimeout,
    console,
    document: {
      activeElement: null,
      addEventListener() {},
      body: { appendChild() {} },
      createElement() {
        return {
          classList: { add() {}, remove() {} },
          dataset: {},
          getContext() { return { measureText(text) { return { width: String(text).length * 8 }; } }; },
          getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
          hidden: false,
          innerHTML: "",
          isConnected: true,
          querySelector() { return null; },
          querySelectorAll() { return []; },
          setAttribute() {},
          style: {},
        };
      },
      createElementNS() { return this.createElement(); },
      documentElement: { clientWidth: 1024, clientHeight: 768, dataset: {}, style: { setProperty() {}, removeProperty() {} } },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) {
      frameCallbacks.push(callback);
      return frameCallbacks.length;
    },
    setInterval,
    setTimeout,
    structuredClone,
  });
  vm.runInContext(`${app}
    let __pass02RenderCalls = 0;
    let __pass02FitCalls = 0;
    globalThis.__pass02 = {
      CONCEPT_HELP_EN,
      CONCEPT_HELP_ZH,
      CONCEPT_HELP_COVERAGE_BY_VIEW,
      CONCEPT_HELP_DETAIL_OVERRIDES,
      CONCEPT_HELP_VISIBLE_LABEL_BINDINGS,
      CONCEPT_TERM_PATTERNS,
      conceptHelpSections,
      dom,
      fieldTitleMarkup,
      fontScaleBand,
      hydrateConceptHelp,
      normalizeUiSettings,
      renderControlPlaneStatus,
      controlPlaneStatusView,
      scheduleModelGraphOverviewResize,
      state,
      timeseriesControlsMarkup,
      installOverviewResizeSpy() {
        __pass02RenderCalls = 0;
        __pass02FitCalls = 0;
        renderModelGraph = () => { __pass02RenderCalls += 1; };
        fitModelGraphOverviewViewportForWidth = () => { __pass02FitCalls += 1; return false; };
      },
      fitCalls() { return __pass02FitCalls; },
      renderCalls() { return __pass02RenderCalls; },
    };`, context, { filename: appPath });
  context.__pass02.flushAnimationFrames = () => {
    const pending = frameCallbacks.splice(0);
    pending.forEach((callback) => callback(0));
  };
  return context.__pass02;
}

function preloadFontBand(fontScale) {
  const root = {
    dataset: {},
    lang: "",
    style: { setProperty() {}, removeProperty() {} },
  };
  const context = vm.createContext({
    document: { documentElement: root },
    localStorage: { getItem() { return JSON.stringify({ fontScale }); } },
  });
  vm.runInContext(preload, context, { filename: preloadPath });
  return root.dataset.fontBand;
}

function controlPlaneDom() {
  return {
    controlPlaneStatus: {},
    controlPlaneStatusBadge: { className: "", textContent: "" },
    controlPlaneStatusSummary: { textContent: "" },
    controlPlaneStatusMetrics: { innerHTML: "" },
  };
}

function fakeOverviewCanvas(width) {
  return {
    dataset: { mode: "overview" },
    clientWidth: width,
    offsetWidth: width,
    setWidth(nextWidth) {
      this.clientWidth = nextWidth;
      this.offsetWidth = nextWidth;
    },
    getBoundingClientRect() { return { width: this.clientWidth }; },
  };
}

function selectedOptionExposure(markup) {
  const selectMatch = String(markup).match(/<select\b(?=[^>]*data-timeseries-field="series")([^>]*)>([\s\S]*?)<\/select>/u);
  assert.ok(selectMatch, "timeseries series select must render");
  const selectAttrs = parseAttributes(selectMatch[0]);
  const optionMatch = selectMatch[2].match(/<option\b(?=[^>]*\bselected\b)([^>]*)>([\s\S]*?)<\/option>/u);
  assert.ok(optionMatch, "timeseries series select must preserve the selected option");
  const optionAttrs = parseAttributes(optionMatch[0]);
  return [selectAttrs.title, selectAttrs["aria-label"], optionAttrs.title, optionAttrs["aria-label"]]
    .filter(Boolean)
    .map(decodeEntities);
}

function conceptKeyForLabel(runtime, label) {
  const normalized = normalizeConceptLabel(label);
  const explicit = runtime.CONCEPT_HELP_VISIBLE_LABEL_BINDINGS
    .find(([candidate]) => normalizeConceptLabel(candidate) === normalized);
  if (explicit) return explicit[1];
  const match = runtime.CONCEPT_TERM_PATTERNS.find(([, pattern]) => pattern.test(label));
  return match?.[0] || "";
}

test("font scale band switches to extreme at the 190-200 percent cap in runtime and preload", () => {
  const runtime = createAppRuntime();
  const expected = new Map([[150, "normal"], [151, "large"], [189, "large"], [190, "extreme"], [200, "extreme"]]);
  for (const [scale, band] of expected) {
    assert.equal(runtime.fontScaleBand(scale), band, `runtime fontScaleBand(${scale})`);
    assert.equal(preloadFontBand(scale), band, `preload font band for ${scale}`);
  }
});

test("control-plane placement status remains read-only in both supported languages", () => {
  const runtime = createAppRuntime();
  Object.assign(runtime.dom, controlPlaneDom());
  runtime.state.mappingStale = false;
  runtime.state.scenario = {
    placement: {
      metadata: {
        control_plane: {
          policy: { options: { mode: "heuristic", objective: "balanced" } },
          decision: { operator_execution_targets: { dense0: [{ rank_id: 0, component_id: "gpu0" }] } },
          evidence: { fingerprint_schema: "runtime-control-plane-v4", input_fingerprint: "abc" },
        },
      },
    },
  };

  for (const language of ["zh-CN", "en"]) {
    I18n.setLanguage(language, null);
    runtime.state.settings = runtime.normalizeUiSettings({ language });
    runtime.renderControlPlaneStatus();
    assert.equal(runtime.dom.controlPlaneStatusBadge.className, "control-plane-status-badge is-ready");
    if (language === "zh-CN") assert.match(runtime.dom.controlPlaneStatusBadge.textContent, /已物化/u);
    else assert.equal(runtime.dom.controlPlaneStatusBadge.textContent, "Materialized");
    assert.match(runtime.dom.controlPlaneStatusMetrics.innerHTML, /runtime-control-plane-v4/u);
  }

  I18n.setLanguage("zh-CN", null);
});

test("model overview resize only triggers automatic viewport fit or clamp for material shrinks", () => {
  const fitDecisionBody = sourceBetween("modelGraphOverviewNeedsViewportFit", "fitModelGraphOverviewViewportForWidth");
  assert.match(
    fitDecisionBody,
    /currentWidth\s*<[\s\S]*(?:previousWidth|previousWidthValue)|(?:previousWidth|previousWidthValue)[\s\S]*>\s*currentWidth/u,
    "resize fit decision should compare the new canvas width against the previous width as a shrink, not any width change",
  );
  assert.match(fitDecisionBody, /materialReduction/u, "resize fit decision should require a material shrink threshold");

  const resizeBody = sourceBetween("scheduleModelGraphOverviewResize", "bindModelGraphResizeObserver");
  assert.match(resizeBody, /modelGraphOverviewNeedsViewportFit\(previousWidth,\s*currentWidth\)/u);
  assert.match(resizeBody, /viewport/iu, "resize logic should adjust the viewport, not rebuild semantic or position state");
  assert.match(resizeBody, /fit|clamp/iu, "resize logic should explicitly fit or clamp the existing overview viewport");
  assert.doesNotMatch(resizeBody, /\b(autoLayoutModelGraph|overviewResponsiveLayout|ensureNodePositions)\b/u);
  assert.doesNotMatch(resizeBody, /(?:positions|inline_positions)\s*=/u);

  const runtime = createAppRuntime();
  runtime.installOverviewResizeSpy();
  runtime.state.view = "model";
  runtime.state.modelGraphEditor.overviewCanvasWidth = 800;
  runtime.state.modelGraphEditor.drag = null;
  runtime.state.modelGraphEditor.pan = null;
  runtime.state.modelGraphEditor.connectPointer = null;
  runtime.dom.modelGraphCanvas = fakeOverviewCanvas(800);

  runtime.dom.modelGraphCanvas.setWidth(804);
  assert.equal(runtime.scheduleModelGraphOverviewResize(804), true);
  runtime.flushAnimationFrames();
  assert.equal(runtime.fitCalls(), 0, "canvas growth must not auto-fit the overview");

  runtime.state.modelGraphEditor.overviewCanvasWidth = 800;
  runtime.dom.modelGraphCanvas.setWidth(792);
  assert.equal(runtime.scheduleModelGraphOverviewResize(792), true);
  runtime.flushAnimationFrames();
  assert.equal(runtime.fitCalls(), 0, "minor shrink must not auto-fit the overview");

  runtime.state.modelGraphEditor.overviewCanvasWidth = 800;
  runtime.dom.modelGraphCanvas.setWidth(680);
  assert.equal(runtime.scheduleModelGraphOverviewResize(680), true);
  runtime.flushAnimationFrames();
  assert.equal(runtime.fitCalls(), 1, "material shrink should auto-fit or clamp the overview viewport");
});

test("MTP workload settings occupy the full workload grid width", () => {
  const renderWorkload = sourceBetween("renderWorkload", "workloadFieldLabel");
  assert.match(renderWorkload, /data-workload-section="mtp"/u);
  assertCssContract(
    'data-workload-section="mtp"',
    /grid-column:\s*1\s*\/\s*-1/u,
    "the MTP workload section should span the full workload settings grid",
  );
});

test("trace filter row has wrapping and readable status text contracts", () => {
  assertCssContract(".trace-filter-bar", /display:\s*flex/u);
  assertCssContract(".trace-filter-bar", /flex-wrap:\s*wrap/u);
  const control = cssRuleBody(".trace-filter-bar .inline-control");
  assert.match(control, /min-width:\s*min\(100%/u);
  assert.match(control, /flex:\s*1\s+1/u);
  assert.match(cssRuleBody(".trace-filter-bar .inline-control > select"), /flex:\s*1\s+1/u);

  const meta = cssRuleBody(".trace-filter-bar .trace-event-meta");
  assert.match(meta, /min-width:\s*min\(100%/u);
  assert.match(meta, /flex:\s*1\s+1/u);
  assert.match(meta, /overflow-wrap:\s*anywhere/u);
  assert.doesNotMatch(meta, /white-space:\s*nowrap|text-overflow:\s*ellipsis|overflow:\s*hidden/u);
});

test("inline model ports and concept-help triggers expose keyboard and screen-reader semantics", () => {
  const overviewPorts = sourceBetween("modelOverviewPortMarkup", "modelOverviewPortPoint");
  assert.match(overviewPorts, /model-overview-port[\s\S]*aria-label=/u);
  assert.match(overviewPorts, /port\.operator_id[\s\S]*port\.port_id[\s\S]*contract/u);

  const inlineGraph = sourceBetween("renderModelInlineAuthoritativeGraph", "modelGraphOverviewName");
  const inlinePortMarkup = inlineGraph.slice(inlineGraph.indexOf("const portMarkup"), inlineGraph.indexOf("const nodeMarkup"));
  assert.match(inlinePortMarkup, /model-inline-port[\s\S]*aria-label=/u);
  for (const token of ["operator.operator_id", "port.port_id", "port.direction", "port.tensor_id", "port.dtype", "modelShapeText(port.shape)", "port.layout"]) {
    assert.ok(inlinePortMarkup.includes(token), `inline port label should include ${token}`);
  }

  const fieldTitle = sourceBetween("fieldTitleMarkup", "quantityField");
  assert.match(fieldTitle, /role="button"/u, "explicit concept-help trigger markup should expose role=button");
  const hydrate = sourceBetween("hydrateConceptHelp", "closeFieldHelp");
  assert.match(hydrate, /setAttribute\("role",\s*"button"\)/u, "auto-hydrated concept-help triggers should expose role=button");
});

test("result time-series series select exposes the complete selected label", () => {
  const runtime = createAppRuntime();
  const longLabel = "Rank seven remote-memory read bandwidth with a deliberately long operator shard label";
  const seriesKey = "component:gpu0:memory-read:7:hbm-channel-0:0";
  const normalized = {
    components: [{ component_id: "gpu0", component_kind: "gpu" }],
    links: [],
    series: [{
      owner_id: "gpu0",
      owner_key: "component:gpu0",
      owner_type: "component",
      component_kind: "gpu",
      series_key: seriesKey,
      series_id: "memory-read",
      label_cn: longLabel,
      metric: "memory_read_bandwidth_bytes_per_s",
      rank: 7,
      channel_id: "hbm-channel-0",
    }],
  };
  const slot = { id: 1, ownerKey: "component:gpu0", seriesKey, yScale: "linear", color: "#5eb6c0" };
  runtime.state.componentTimeseriesView.slots = [slot];

  const markup = runtime.timeseriesControlsMarkup(slot, normalized);
  const fullSelectedLabel = `${longLabel} · memory_read_bandwidth_bytes_per_s · Rank 7 · hbm-channel-0`;
  assert.ok(
    selectedOptionExposure(markup).some((value) => value.includes(fullSelectedLabel)),
    "series select or its selected option should expose the full selected label through title or aria-label",
  );
  assertCssContract(
    ".timeseries-curve-control",
    /grid-column\s*:\s*1\s*\/\s*-1/u,
    "the long series selector should span the narrow results control grid",
  );
});

test("targeted concept help additions are bilingual, detailed, and bound to existing UI vocabulary", () => {
  const runtime = createAppRuntime();
  const targets = [
    ["vocabulary_size", ["词表大小（Vocabulary Size）", "Vocabulary Size"]],
    ["max_sequence_length", ["最大序列长度（Max Sequence Length）", "Max Sequence Length"]],
    ["softmax", ["Softmax"]],
    ["expert_router", ["Expert Router", "MoE 路由器（MoE Router）"]],
    ["kv_cache", ["KV 缓存（KV Cache）", "KV Cache"]],
    ["logical_memory", ["逻辑地址范围", "Logical address range"]],
    ["rank", ["执行位置", "Execution location"]],
    ["protocol", ["协议（Protocol）", "Protocol"]],
    ["bandwidth_semantics", ["原始线路", "Effective One-Way"]],
  ];

  for (const [key, labels] of targets) {
    assert.ok(runtime.CONCEPT_HELP_ZH[key], `${key} needs Chinese concept help`);
    assert.ok(runtime.CONCEPT_HELP_EN[key], `${key} needs English concept help`);
    assert.ok(runtime.CONCEPT_HELP_DETAIL_OVERRIDES[key], `${key} needs a term-specific six-section detail override`);
    for (const language of ["zh-CN", "en"]) {
      I18n.setLanguage(language, null);
      runtime.state.settings = runtime.normalizeUiSettings({ language });
      const sections = runtime.conceptHelpSections(key);
      assert.equal(sections.length, 6, `${key}/${language} should keep six structured help sections`);
      assert.ok(sections.every((section) => String(section.title).trim() && String(section.body).trim()), `${key}/${language} has empty help detail`);
    }
    for (const label of labels) {
      assert.equal(conceptKeyForLabel(runtime, label), key, `${label} should bind to ${key}`);
    }
  }

  const renderModel = sourceBetween("renderModel", "modelSummaryFact");
  assert.match(renderModel, /modelSummaryFact\("词表大小（Vocabulary Size）", "vocabulary_size"/u);
  assert.match(renderModel, /modelSummaryFact\("最大序列长度（Max Sequence Length）", "max_sequence_length"/u);

  const traceDetails = sourceBetween("renderTraceEventDetails", "copyTraceDiagnosticInfo");
  assert.match(traceDetails, /traceDetailFact\(uiText\("执行位置", "Execution location"\)/u);
  assert.match(traceDetails, /traceDetailFact\(uiText\("逻辑地址范围", "Logical address range"\)/u);
  assert.match(traceDetails, /hydrateConceptHelp\(dom\.traceEventDrawer\)/u);

  const protocolCard = sourceBetween("protocolPresetBandwidthMarkup", "filteredProtocolPresets");
  assert.match(protocolCard, /generic_protocol|data-concept-help="protocol"|fieldTitleMarkup\([^)]*"protocol/u);
  assert.match(protocolCard, /bandwidth_semantics|data-concept-help="bandwidth"|fieldTitleMarkup\([^)]*"bandwidth/u);

  I18n.setLanguage("zh-CN", null);
});
