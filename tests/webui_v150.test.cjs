"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const TopologyCore = require(path.join(webui, "topology-core.js"));
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

class TestOption {
  constructor(text, value) {
    this.text = text;
    this.value = value;
  }
}

function helpers() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
    Intl,
    Map,
    ModelGraphCore,
    Option: TestOption,
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    URL,
    URLSearchParams,
    clearTimeout,
    console,
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__v150 = {
    CONCEPT_HELP,
    CONCEPT_TERM_PATTERNS,
    MAX_TIMESERIES_CHARTS,
    dom,
    modelPortCompatibility,
    modelPresetPageQuery,
    normalizeComponentTimeseries,
    normalizeModelGraphPayload,
    modelGraphLayerSpecs,
    state,
    syncPresetFamilies,
    timeseriesChartModel,
    timeseriesChartMarkup,
    timeseriesMetricGroup,
    timeseriesTimeDomain,
  };`, context);
  return context.__v150;
}

function mockSelect(value = "") {
  return {
    value,
    options: [],
    replaceChildren(...options) {
      this.options = options;
      if (!options.some((item) => item.value === this.value)) this.value = options[0]?.value || "";
    },
  };
}

test("model preset Dense and MoE filters use the backend model_kind contract", () => {
  const ui = helpers();
  ui.state.presetCatalog.limit = 24;
  ui.dom.presetSearchInput = { value: "" };
  ui.dom.presetFamilyFilter = { value: "" };
  ui.dom.presetSupportFilter = { value: "" };
  ui.dom.presetArchitectureFilter = { value: "dense" };
  const dense = new URLSearchParams(ui.modelPresetPageQuery(0).slice(1));
  assert.equal(dense.get("model_kind"), "dense");
  assert.equal(dense.has("architecture"), false);
  ui.dom.presetArchitectureFilter.value = "moe";
  const moe = new URLSearchParams(ui.modelPresetPageQuery(24).slice(1));
  assert.equal(moe.get("model_kind"), "moe");
  assert.equal(moe.get("offset"), "24");
});

test("model-kind facets accept the V4 page contract", () => {
  const ui = helpers();
  ui.dom.presetFamilyFilter = mockSelect();
  ui.dom.presetArchitectureFilter = mockSelect("moe");
  ui.syncPresetFamilies({
    facets: { family: [{ value: "Qwen" }], model_kind: [{ value: "dense", count: 117 }, { value: "moe", count: 30 }] },
    items: [],
  });
  assert.deepEqual(Array.from(ui.dom.presetArchitectureFilter.options, (item) => item.value), ["", "dense", "moe"]);
  assert.equal(ui.dom.presetArchitectureFilter.value, "moe");
});

test("mapping, workload, playback, and results expose one bilingual concept-help dictionary", () => {
  const ui = helpers();
  const pageKeys = {
    mapping: ["rank_mapping", "rank_shard", "control_plane", "mapping_fingerprint", "mapping_stale", "tp", "pp", "ep", "collective", "kv_policy", "weights_resident"],
    workload: ["workload", "request", "synthetic_workload", "arrival_time", "prompt_tokens", "output_tokens", "scheduler", "batched_tokens", "preemption", "mtp", "acceptance_rate", "slo"],
    playback: ["trace", "simulation_time", "playback_speed", "selected_event", "active_event", "event_interval", "logical_memory", "fidelity", "batch", "protocol_path"],
    results: ["component_timeseries", "busy_fraction", "compute_utilization", "memory_residency", "bandwidth_utilization", "storage_occupancy", "ttft", "tbt", "tpot", "e2e", "throughput", "makespan", "energy", "critical_path", "utilization"],
  };
  Object.entries(pageKeys).forEach(([page, keys]) => keys.forEach((key) => {
    assert.equal(typeof ui.CONCEPT_HELP[key], "object", `${page}.${key}`);
    assert.ok(ui.CONCEPT_HELP[key]["zh-CN"].length >= 18, `${page}.${key} should have explanatory Chinese guidance`);
    assert.ok(ui.CONCEPT_HELP[key].en.length >= 18, `${page}.${key} should have explanatory English guidance`);
    assert.match(ui.CONCEPT_HELP[key]["zh-CN"], /[\u3400-\u9fff]/u, `${page}.${key} should contain Chinese guidance`);
    assert.doesNotMatch(ui.CONCEPT_HELP[key].en, /[\u3400-\u9fff]/u, `${page}.${key} English guidance should not contain Chinese`);
  }));
  for (const id of ["view-mapping", "view-workload", "view-playback", "view-results"]) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(app, /hydrateConceptHelp\(dom\.traceEventDrawer\)/);
  assert.match(app, /hydrateConceptHelp\(dom\.resultsContent\)/);
  assert.match(app, /hydrateConceptHelp\(dom\.placementControls\)/);
});

test("the model page is a typed component graph and presets use the same canonical adapter", () => {
  const ui = helpers();
  const model = {
    name: "dense-demo",
    vocabulary_size: 32000,
  };
  const layerSpecs = [
    { layer_id: "layer-a", kind: "dense", hidden_size: 1024, intermediate_size: 4096, attention_heads: 16, kv_heads: 4, dtype: "bf16" },
    { layer_id: "layer-b", kind: "dense", hidden_size: 1024, intermediate_size: 4096, attention_heads: 16, kv_heads: 4, dtype: "bf16" },
  ];
  const graph = ui.normalizeModelGraphPayload(
    ModelGraphCore.buildModelGraphFromLayerSpecs(layerSpecs, model),
    model,
  );
  assert.equal(graph.attributes.authoritative, true);
  assert.ok(graph.operators.some((item) => item.op_kind === "layer_group"));
  assert.ok(graph.operators.every((item) => Array.isArray(item.ports)));
  assert.deepEqual(Array.from(ui.modelGraphLayerSpecs(graph), (item) => item.layer_id), ["layer-a", "layer-b"]);
  assert.throws(() => ui.modelGraphLayerSpecs({ graph_id: "invalid", executable: true, operators: [], tensors: [] }), /不可投影|缺少|不可执行/u);
  const mismatch = ui.modelPortCompatibility(
    { direction: "output", dtype: "bf16", shape: ["B", "T", 1024], layout: "row_major" },
    { direction: "input", dtype: "fp16", shape: ["B", "T", 2048], layout: "column_major" },
  );
  assert.equal(mismatch.compatible, false);
  assert.match(mismatch.message, /期望.*实际/u);
  assert.match(html, /id="modelGraphCanvas"[\s\S]*id="modelGraphInspectorContent"/u);
  assert.doesNotMatch(html, /id="modelLayerBody"/u);
  assert.match(css, /\.model-graph-node\b/u);
});

test("model and trace cores load before app and the app delegates semantic/view operations", () => {
  const topologyIndex = html.indexOf('src="./topology-core.js"');
  const modelIndex = html.indexOf('src="./model-graph-core.js"');
  const traceIndex = html.indexOf('src="./trace-view-core.js"');
  const appIndex = html.indexOf('src="./app.js"');
  assert.ok(topologyIndex >= 0 && topologyIndex < traceIndex);
  assert.ok(modelIndex >= 0 && modelIndex < appIndex);
  assert.ok(traceIndex >= 0 && traceIndex < appIndex);
  assert.equal(typeof ModelGraphCore.buildModelGraphFromLayerSpecs, "function");
  for (const call of ["ModelGraph.normalizeModelGraph", "ModelGraph.graphToLayerSpecs", "ModelGraph.createConnectionPreview", "ModelGraph.updateConnectionPreview", "ModelGraph.commitConnectionPreview", "ModelGraph.layoutDag", "TraceView.fitTraceViewport", "TraceView.toggleFullscreen", "TraceView.dragTraceNode", "TraceView.pointAtProgress", "TraceView.animationCapability"]) {
    assert.ok(app.includes(call), call);
  }
  assert.match(app, /setAttribute\("viewBox", `0 0 \$\{layout\.bounds\.width\} \$\{layout\.bounds\.height\}`\)/u);
});

test("trace playback uses a one-event wall-clock interval while particles keep adaptive animation", () => {
  assert.match(app, /const prefersReducedMotion = typeof globalThis\.matchMedia === "function"[\s\S]*globalThis\.matchMedia\("\(prefers-reduced-motion: reduce\)"\)\.matches;/u);
  assert.match(app, /TraceView\.animationCapability\(globalThis, \{\s*frameMs: 16,\s*reduceMotion: state\.settings\.reduceMotion,\s*prefersReducedMotion,\s*\}\)/u);
  assert.match(app, /const TRACE_PLAYBACK_STEP_MS = 1000/u);
  assert.match(app, /globalThis\.setInterval\(traceAnimationStep, TRACE_PLAYBACK_STEP_MS\)/u);
  assert.match(app, /const nextIndex = playback\.selectedIndex \+ 1/u);
  assert.doesNotMatch(app, /playback\.speed/u);
  assert.match(app, /const titleAttribute = title \? ` title="\$\{escapeHtml\(title\)\}"` : "";/u);
  assert.match(app, /const memoryAttribute = memorySummary \? ` data-trace-memory-summary="\$\{escapeHtml\(memorySummary\)\}"` : "";/u);
  assert.match(app, /<span class="trace-node-copy"><strong>\$\{escapeHtml\(componentId\)\}<\/strong>\$\{traceNodeMemoryMarkup\(componentId\)\}<\/span>/u);
  assert.doesNotMatch(app, /class="trace-node-copy"[^\n]*kindLabel/u);
});

test("component time series preserves fidelity and never fabricates unknown points", () => {
  const ui = helpers();
  const normalized = ui.normalizeComponentTimeseries({
    component_timeseries: {
      schema_version: "1.0",
      time_unit: "ns",
      execution_mode: "scalable_serving",
      fidelity: "aggregate",
      fidelity_description_cn: "批次包络聚合，不是硬件采样。",
      point_limit: 500,
      components: [{
        component_id: "gpu0",
        component_kind: "gpu",
        series: [
          { series_id: "busy", metric: "busy_fraction", unit: "fraction", rank: 0, fidelity: "aggregate", quality: "batch_envelope", points: [{ start_ns: 0, end_ns: 10, value: 0.75 }] },
          { series_id: "unknown-residency", metric: "memory_residency_bytes", unit: "bytes", fidelity: "unknown", quality: "unavailable", points: [] },
        ],
      }],
      links: [],
    },
  });
  assert.equal(normalized.point_limit, 500);
  assert.equal(normalized.series[0].fidelity, "aggregate");
  assert.equal(normalized.series[0].quality, "batch_envelope");
  assert.equal(normalized.series[1].points.length, 0);
  assert.equal(ui.timeseriesMetricGroup("busy_fraction"), "GPU 忙碌与模型计算（GPU Busy & Compute）");
  assert.equal(ui.timeseriesMetricGroup("memory_residency_bytes"), "HBM 容量驻留（HBM Capacity Residency）");
  const emptyMarkup = ui.timeseriesChartMarkup(normalized.series[1]);
  assert.match(emptyMarkup, /不会.*伪造曲线/u);
  assert.doesNotMatch(emptyMarkup, /<path\b/u);
  const computeMarkup = ui.timeseriesChartMarkup({
    ...normalized.series[0],
    metric: "modeled_compute_utilization",
    unit: "ratio",
    capacity: { value: 989_500_000_000_000, unit: "ops_per_s" },
  });
  assert.match(computeMarkup, /class="timeseries-tick-label timeseries-y-tick"[^>]*>100%<\/text>/u);
  assert.doesNotMatch(computeMarkup, /class="timeseries-tick-label timeseries-x-tick"[^>]*>100%<\/text>/u);
  assert.match(computeMarkup, /<svg class="timeseries-chart-svg"[^>]*\spreserveAspectRatio="[^"]+"/u);
  assert.match(computeMarkup, /class="[^"]*timeseries-x-axis-title"[^>]*>仿真时间（线性）<\/text>/u);
  const yAxisTitle = /<text class="timeseries-axis-title timeseries-y-axis-title"[^>]*>利用率（%） · 线性<\/text>/u.exec(computeMarkup)?.[0] || "";
  assert.ok(yAxisTitle, "the Y-axis title keeps the utilization label");
  assert.doesNotMatch(yAxisTitle, /\btransform=|rotate\(/u, "the Y-axis title remains horizontal");
  const timeseriesSvgCss = /\.timeseries-chart-svg\s*\{([^}]*)\}/u.exec(css)?.[1] || "";
  assert.match(timeseriesSvgCss, /\bheight\s*:\s*(?!auto\b)[^;]+;/u, "timeseries charts use an explicit bounded height");
  assert.match(timeseriesSvgCss, /\bheight\s*:\s*clamp\(|\bmax-height\s*:/u, "timeseries chart height has an upper bound");
  assert.doesNotMatch(timeseriesSvgCss, /\bmin-height\s*:\s*(?!0(?:\s|;|$))[^;]+;/u, "timeseries charts do not rely on an unbounded minimum height");
  assert.doesNotMatch(computeMarkup, /e\d+%/iu);

  const logMarkup = ui.timeseriesChartMarkup({
    ...normalized.series[0],
    unit: "ratio",
    points: [
      { start_ns: 0, end_ns: 10, value: 0.5 },
      { start_ns: 10, end_ns: 20, value: 0 },
      { start_ns: 20, end_ns: 30, value: 0.25 },
    ],
  }, { yScale: "log", xDomain: [0, 30], color: "#d57a3b" });
  assert.match(logMarkup, /利用率（%） · Log10/u);
  assert.match(logMarkup, /1 个非正值区间显示为断点/u);
  assert.match(logMarkup, /--timeseries-line-color: #d57a3b/u);
  const pathData = /class="timeseries-series-path" d="([^"]+)"/u.exec(logMarkup)?.[1] || "";
  assert.equal((pathData.match(/\bM\b/gu) || []).length, 2);
  assert.equal(ui.MAX_TIMESERIES_CHARTS, 4);
  assert.match(html, /id="addTimeseriesChartButton"/u);
  assert.match(app, /view\.slots\.length >= MAX_TIMESERIES_CHARTS/u);
  assert.doesNotMatch(app, /timeseries(?:Component|Rank)Filter/u);
});
