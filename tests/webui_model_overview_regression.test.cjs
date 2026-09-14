"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function datasetKey(attribute) {
  return attribute.slice("data-".length).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
}

function parseAttributes(source) {
  const attributes = {};
  for (const match of source.matchAll(/\s([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:="([^"]*)")?/g)) {
    attributes[match[1]] = match[2] ?? "";
  }
  return attributes;
}

function parseStyle(source) {
  return Object.fromEntries(String(source || "").split(";").map((entry) => {
    const separator = entry.indexOf(":");
    if (separator < 0) return null;
    return [entry.slice(0, separator).trim(), entry.slice(separator + 1).trim()];
  }).filter(Boolean));
}

function pixelValue(value) {
  const number = Number.parseFloat(String(value || ""));
  return Number.isFinite(number) ? number : 0;
}

class FakeClassList {
  constructor(classes = "") {
    this.values = new Set(String(classes).split(/\s+/u).filter(Boolean));
  }

  add(...classes) {
    classes.filter(Boolean).forEach((item) => this.values.add(String(item)));
  }

  remove(...classes) {
    classes.forEach((item) => this.values.delete(String(item)));
  }

  contains(value) {
    return this.values.has(String(value));
  }

  toggle(value, force) {
    const enabled = force === undefined ? !this.contains(value) : Boolean(force);
    if (enabled) this.add(value);
    else this.remove(value);
    return enabled;
  }

  toString() {
    return Array.from(this.values).join(" ");
  }
}

class FakeElement {
  constructor(tagName = "div", attributes = {}) {
    this.tagName = tagName.toUpperCase();
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.children = [];
    this.parentElement = null;
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.textContent = "";
    this.classList = new FakeClassList();
    this._innerHTML = "";
    Object.entries(attributes).forEach(([name, value]) => this.setAttribute(name, value));
  }

  set innerHTML(value) {
    this._innerHTML = String(value ?? "");
    this.children = [];
  }

  get innerHTML() {
    return this._innerHTML;
  }

  setAttribute(name, value) {
    const text = String(value ?? "");
    this.attributes[name] = text;
    if (name === "class") this.classList = new FakeClassList(text);
    if (name === "style") this.style = { ...this.style, ...parseStyle(text) };
    if (name.startsWith("data-")) this.dataset[datasetKey(name)] = text;
  }

  getAttribute(name) {
    if (name === "class") return this.classList.toString();
    if (name === "style") return Object.entries(this.style).map(([key, value]) => `${key}:${value}`).join(";");
    return this.attributes[name] ?? null;
  }

  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    return collectMatches(this, selector);
  }

  closest(selector) {
    let node = this;
    while (node) {
      if (matchesSelector(node, selector)) return node;
      node = node.parentElement;
    }
    return null;
  }

  contains(node) {
    if (node === this) return true;
    return this.children.some((child) => child.contains(node));
  }

  remove() {
    if (!this.parentElement) return;
    this.parentElement.children = this.parentElement.children.filter((child) => child !== this);
    this.parentElement = null;
  }

  getBoundingClientRect() {
    if (this.dataset.modelPortId) return this.portRect();
    return {
      left: pixelValue(this.style.left),
      top: pixelValue(this.style.top),
      width: pixelValue(this.style.width) || Number(this.clientWidth) || 0,
      height: pixelValue(this.style.height) || Number(this.clientHeight) || 0,
      right: pixelValue(this.style.left) + (pixelValue(this.style.width) || Number(this.clientWidth) || 0),
      bottom: pixelValue(this.style.top) + (pixelValue(this.style.height) || Number(this.clientHeight) || 0),
    };
  }

  portRect() {
    const parentRect = this.parentElement.getBoundingClientRect();
    const side = this.dataset.modelPortSide || (this.dataset.modelPortDirection === "output" ? "right" : "left");
    const styleFraction = Number.parseFloat(String(this.style["--model-port-position"] || ""));
    const fraction = Number.isFinite(styleFraction) ? styleFraction / 100 : 0.5;
    let centerX = parentRect.left + parentRect.width * fraction;
    let centerY = parentRect.top + parentRect.height * fraction;
    if (side === "left") centerX = parentRect.left;
    if (side === "right") centerX = parentRect.left + parentRect.width;
    if (side === "top") centerY = parentRect.top;
    if (side === "bottom") centerY = parentRect.top + parentRect.height;
    return {
      left: centerX - 7,
      top: centerY - 7,
      width: 14,
      height: 14,
      right: centerX + 7,
      bottom: centerY + 7,
    };
  }
}

class FakeNodeLayer extends FakeElement {
  set innerHTML(value) {
    this._innerHTML = String(value ?? "");
    this.children = [
      ...parseOverviewNodes(this._innerHTML, this),
      ...parsePaths(this._innerHTML, this),
    ];
  }
}

class FakeEdgeLayer extends FakeElement {
  set innerHTML(value) {
    this._innerHTML = String(value ?? "");
    this.children = parsePaths(this._innerHTML, this);
  }
}

function parseOverviewNodes(markup, parent) {
  const roots = [];
  const stack = [];
  for (const match of String(markup || "").matchAll(/<\/article>|<article\b([^>]*)>|<button\b([^>]*)[\s\S]*?<\/button>/gu)) {
    if (match[1] != null) {
      const node = new FakeElement("article", parseAttributes(match[1]));
      const container = stack.at(-1) || parent;
      node.parentElement = container;
      container.children.push(node);
      if (container === parent) roots.push(node);
      stack.push(node);
      continue;
    }
    if (match[2] != null) {
      const button = new FakeElement("button", parseAttributes(match[2]));
      const container = stack.at(-1) || parent;
      button.parentElement = container;
      container.children.push(button);
      continue;
    }
    if (stack.length) stack.pop();
  }
  return roots;
}

function parsePaths(markup, parent) {
  return Array.from(markup.matchAll(/<path\b([^>]*)>/gu), (match) => {
    const pathElement = new FakeElement("path", parseAttributes(match[1]));
    pathElement.parentElement = parent;
    return pathElement;
  });
}

function collectMatches(root, selector) {
  const results = [];
  const visit = (node) => {
    node.children.forEach((child) => {
      if (matchesSelector(child, selector)) results.push(child);
      visit(child);
    });
  };
  visit(root);
  return results;
}

function matchesSelector(element, selector) {
  return String(selector || "").split(",").some((part) => matchesSimpleSelector(element, part.trim()));
}

function matchesSimpleSelector(element, selector) {
  if (!selector) return false;
  const selectorWithoutAttributes = selector.replace(/\[[^\]]*\]/gu, "");
  const tag = /^[a-zA-Z][a-zA-Z0-9_-]*/u.exec(selectorWithoutAttributes)?.[0] || "";
  if (tag && element.tagName.toLowerCase() !== tag.toLowerCase()) return false;
  for (const match of selectorWithoutAttributes.matchAll(/\.([a-zA-Z0-9_-]+)/gu)) {
    if (!element.classList.contains(match[1])) return false;
  }
  for (const match of selector.matchAll(/\[([^\]=]+)(?:="([^"]*)")?\]/gu)) {
    const actual = element.getAttribute(match[1]);
    if (actual == null) return false;
    if (match[2] != null && actual !== match[2]) return false;
  }
  return true;
}

function appHelpers() {
  const context = vm.createContext({
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
    URL,
    URLSearchParams,
    clearTimeout,
    console,
    document: {
      addEventListener() {},
      createElement() {
        return { getContext() { return { measureText(text) { return { width: String(text).length * 8 }; } }; } };
      },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) { return setTimeout(() => callback(0), 0); },
    cancelAnimationFrame(id) { clearTimeout(id); },
    setTimeout,
  });
  vm.runInContext(`${app}
    const __markScenarioChangedCalls = [];
    markScenarioChanged = (message = "", options = {}) => {
      __markScenarioChangedCalls.push({ message, options });
      state.dirty = true;
      state.scenarioGeneration += 1;
      if (options.mappingImpact) state.mappingGeneration += 1;
    };
    globalThis.__overviewRegression = {
      beginModelGraphPointer,
      commitModelGraphHistory,
      dom,
      endModelGraphPointer,
      fontScaleBand,
      layoutScaleForFont,
      modelGraphApplyInteractionFrame,
      modelGraphOverviewProjection,
      modelGraphOverviewNodeSize,
      modelGraphOverviewPresentationLayout,
      modelGraphOverviewRoutePlan,
      modelGraphOverviewTextWidth,
      modelGraphSnapshot,
      modelGraphUi,
      renderModelGraph,
      state,
      travelModelGraphHistory,
      markScenarioChangedCalls: __markScenarioChangedCalls,
    };`, context);
  return context.__overviewRegression;
}

function attachModelDom(ui) {
  ui.dom.modelGraphCanvas = new FakeElement("section", { "data-mode": "overview" });
  ui.dom.modelGraphCanvas.clientWidth = 720;
  ui.dom.modelGraphCanvas.offsetWidth = 720;
  ui.dom.modelGraphCanvas.closest = () => null;

  ui.dom.modelGraphWorld = new FakeElement("div");
  ui.dom.modelGraphWorld.clientWidth = 720;
  ui.dom.modelGraphWorld.clientHeight = 480;
  ui.dom.modelGraphWorld.style.width = "720px";
  ui.dom.modelGraphWorld.style.height = "480px";

  ui.dom.modelGraphGroupLayer = new FakeElement("div");
  ui.dom.modelGraphNodeLayer = new FakeNodeLayer("div");
  ui.dom.modelGraphEdgeLayer = new FakeEdgeLayer("svg");
  ui.dom.modelGraphInspectorTitle = new FakeElement("h2");
  ui.dom.modelGraphInspectorContent = new FakeElement("div");
  ui.dom.modelGraphDiagnostics = new FakeElement("div");
  ui.dom.modelGraphStatus = new FakeElement("div");

  for (const child of [ui.dom.modelGraphGroupLayer, ui.dom.modelGraphNodeLayer, ui.dom.modelGraphEdgeLayer]) {
    child.parentElement = ui.dom.modelGraphWorld;
  }
  ui.dom.modelGraphWorld.children = [ui.dom.modelGraphGroupLayer, ui.dom.modelGraphNodeLayer, ui.dom.modelGraphEdgeLayer];
}

function mtpBranchGraph() {
  return ModelGraphCore.normalizeModelGraph({
    graph_id: "mtp-short-route",
    operators: [
      {
        operator_id: "backbone",
        op_kind: "dense_mlp",
        sequence_index: 0,
        ports: [
          { port_id: "out0", direction: "output", tensor_id: "hidden", dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
        ],
      },
      {
        operator_id: "mtp.prediction_layer.000",
        op_kind: "mtp_prediction_layer",
        sequence_index: 1,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "hidden", dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
          { port_id: "out0", direction: "output", tensor_id: "proposal", dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
        ],
      },
    ],
    tensors: [
      {
        tensor_id: "hidden",
        role: "activation",
        producer_operator_id: "backbone",
        consumer_operator_ids: ["mtp.prediction_layer.000"],
        dtype: "bf16",
        shape: ["B", "T", 512],
        layout: "logical",
      },
      {
        tensor_id: "proposal",
        role: "activation",
        producer_operator_id: "mtp.prediction_layer.000",
        consumer_operator_ids: [],
        dtype: "bf16",
        shape: ["B", "T", 512],
        layout: "logical",
      },
    ],
  });
}

function mtpCrowdedForkGraph() {
  const shape = ["B", "T", 512];
  return ModelGraphCore.normalizeModelGraph({
    graph_id: "mtp-crowded-fork",
    operators: [
      {
        operator_id: "final_norm",
        op_kind: "rms_norm",
        sequence_index: 0,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "hidden", dtype: "bf16", shape, layout: "logical" },
          { port_id: "out0", direction: "output", tensor_id: "normalized", dtype: "bf16", shape, layout: "logical" },
        ],
      },
      {
        operator_id: "lm_head",
        op_kind: "lm_head",
        sequence_index: 1,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "normalized", dtype: "bf16", shape, layout: "logical" },
          { port_id: "out0", direction: "output", tensor_id: "logits", dtype: "bf16", shape: ["B", "T", "V"], layout: "logical" },
        ],
      },
      {
        operator_id: "mtp.prediction_layer.000",
        op_kind: "mtp_prediction_layer",
        sequence_index: 1,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "normalized", dtype: "bf16", shape, layout: "logical" },
          { port_id: "out0", direction: "output", tensor_id: "proposal", dtype: "bf16", shape, layout: "logical" },
        ],
      },
      {
        operator_id: "mtp.aux_head",
        op_kind: "mtp_aux_head",
        sequence_index: 2,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "proposal", dtype: "bf16", shape, layout: "logical" },
          { port_id: "out0", direction: "output", tensor_id: "proposal_logits", dtype: "bf16", shape: ["B", "T", "V"], layout: "logical" },
        ],
      },
    ],
    tensors: [
      { tensor_id: "hidden", role: "input", producer_operator_id: null, consumer_operator_ids: ["final_norm"], dtype: "bf16", shape, layout: "logical" },
      { tensor_id: "normalized", role: "activation", producer_operator_id: "final_norm", consumer_operator_ids: ["lm_head", "mtp.prediction_layer.000"], dtype: "bf16", shape, layout: "logical" },
      { tensor_id: "logits", role: "output", producer_operator_id: "lm_head", consumer_operator_ids: [], dtype: "bf16", shape: ["B", "T", "V"], layout: "logical" },
      { tensor_id: "proposal", role: "activation", producer_operator_id: "mtp.prediction_layer.000", consumer_operator_ids: ["mtp.aux_head"], dtype: "bf16", shape, layout: "logical" },
      { tensor_id: "proposal_logits", role: "output", producer_operator_id: "mtp.aux_head", consumer_operator_ids: [], dtype: "bf16", shape: ["B", "T", "V"], layout: "logical" },
    ],
  });
}

function boundaryOnlyGraph() {
  return ModelGraphCore.normalizeModelGraph({
    graph_id: "boundary-only",
    operators: [
      {
        operator_id: "input",
        op_kind: "model_input",
        sequence_index: 0,
        ports: [
          { port_id: "out0", direction: "output", tensor_id: "input.hidden", dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
        ],
      },
      {
        operator_id: "output",
        op_kind: "model_output",
        sequence_index: 1,
        ports: [
          { port_id: "in0", direction: "input", tensor_id: "input.hidden", dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
        ],
      },
    ],
    tensors: [
      {
        tensor_id: "input.hidden",
        role: "activation",
        producer_operator_id: "input",
        consumer_operator_ids: ["output"],
        dtype: "bf16",
        shape: ["B", "T", 512],
        layout: "logical",
      },
    ],
  });
}

function multiBoundaryRepeatGraph() {
  const shape = ["B", "T", 512];
  const operators = [];
  const tensors = [];
  for (let index = 0; index < 4; index += 1) {
    const inputTensor = `hidden.in${index}`;
    const outputTensor = `hidden.out${index}`;
    operators.push({
      operator_id: `source${index}`,
      op_kind: "model_input",
      sequence_index: index,
      ports: [
        { port_id: "out0", direction: "output", tensor_id: inputTensor, dtype: "bf16", shape, layout: "logical" },
      ],
    });
    operators.push({
      operator_id: `sink${index}`,
      op_kind: "model_output",
      sequence_index: 20 + index,
      ports: [
        { port_id: "in0", direction: "input", tensor_id: outputTensor, dtype: "bf16", shape, layout: "logical" },
      ],
    });
    tensors.push({
      tensor_id: inputTensor,
      role: "activation",
      producer_operator_id: `source${index}`,
      consumer_operator_ids: ["block0"],
      dtype: "bf16",
      shape,
      layout: "logical",
    });
    tensors.push({
      tensor_id: outputTensor,
      role: "activation",
      producer_operator_id: "block0",
      consumer_operator_ids: [`sink${index}`],
      dtype: "bf16",
      shape,
      layout: "logical",
    });
  }
  operators.push({
    operator_id: "group0",
    op_kind: "layer_group",
    sequence_index: 10,
    parameters: { repeat: 1 },
    attributes: {},
    ports: [],
  });
  operators.push({
    operator_id: "block0",
    op_kind: "dense_mlp",
    sequence_index: 11,
    parameters: {},
    attributes: { parent_group_id: "group0" },
    ports: [
      ...Array.from({ length: 4 }, (_item, index) => ({
        port_id: `in${index}`,
        direction: "input",
        tensor_id: `hidden.in${index}`,
        dtype: "bf16",
        shape,
        layout: "logical",
      })),
      ...Array.from({ length: 4 }, (_item, index) => ({
        port_id: `out${index}`,
        direction: "output",
        tensor_id: `hidden.out${index}`,
        dtype: "bf16",
        shape,
        layout: "logical",
      })),
    ],
  });
  return ModelGraphCore.normalizeModelGraph({
    graph_id: "multi-boundary-repeat",
    operators,
    tensors,
    attributes: {
      ui: {
        overview: {
          collapsed_groups: ["group0"],
          positions: { "overview:group:group0": { x: 240, y: 120 } },
          viewport: { x: 0, y: 0, scale: 1 },
        },
      },
    },
  });
}

function modelGraphUiState(graph) {
  return JSON.parse(JSON.stringify(graph.attributes.ui));
}

function modelLayer(id, overrides = {}) {
  return {
    schema_version: "1.0",
    layer_id: id,
    kind: "dense",
    hidden_size: 512,
    intermediate_size: 2048,
    attention_heads: 8,
    kv_heads: 2,
    attention_head_dim: 64,
    sequence_mixer: "full_attention",
    linear_attention: null,
    num_experts: 1,
    experts_per_token: 1,
    shared_expert_intermediate_size: 0,
    shared_expert_gate: false,
    dtype: "bf16",
    quantization: null,
    weight_bytes: 4096,
    metadata: { pattern_index: 0 },
    ...overrides,
  };
}

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

function visibleText(markup) {
  const visibleMarkup = String(markup || "")
    .replace(/<title\b[^>]*>[\s\S]*?<\/title>/gu, " ")
    .replace(/<span\b(?=[^>]*\bclass="[^"]*\bsr-only\b[^"]*")[^>]*>[\s\S]*?<\/span>/gu, " ");
  return decodeEntities(visibleMarkup.replace(/<[^>]*>/gu, " ")).replace(/\s+/gu, " ").trim();
}

function elementTags(markup, tagName, className) {
  const tag = escapeRegex(tagName);
  const klass = escapeRegex(className);
  const pattern = new RegExp(`<${tag}\\b(?=[^>]*\\bclass="[^"]*\\b${klass}\\b[^"]*")[^>]*>`, "gu");
  return Array.from(String(markup || "").matchAll(pattern), (match) => match[0]);
}

function classTokens(tag) {
  return new Set(String(parseAttributes(tag).class || "").split(/\s+/u).filter(Boolean));
}

function expandedBlockOverviewFixture({ selectedOperatorId = "", selectedOverviewId = "", layerOverrides = {} } = {}) {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = ModelGraphCore.buildModelGraphFromLayerSpecs(
    [modelLayer("layer-0", layerOverrides)],
    {
    name: "expanded-block",
    vocabulary_size: 32000,
    },
  );
  ui.state.view = "model";
  ui.state.settings.fontScale = 100;
  ui.state.scenario = { model: { graph } };
  const graphUi = ui.modelGraphUi(graph);
  graphUi.mode = "overview";
  graphUi.overview.collapsed_groups = [];
  const projection = ui.modelGraphOverviewProjection(graph);
  const groupNode = projection.nodes.find((node) => node.kind === "repeat_group");
  assert.ok(groupNode);
  graphUi.overview.positions = Object.fromEntries(projection.nodes.map((node, index) => [
    node.display_id,
    { x: 40 + (index % 2) * 300, y: 40 + Math.floor(index / 2) * 420 },
  ]));
  graphUi.overview.viewport = { x: 0, y: 0, scale: 1 };
  ui.state.modelGraphEditor.selectedOperatorId = selectedOperatorId;
  ui.state.modelGraphEditor.selectedOverviewId = selectedOverviewId || groupNode.display_id;
  ui.renderModelGraph();
  return {
    ui,
    graph,
    projection: ui.modelGraphOverviewProjection(graph),
    groupNode,
    html: ui.dom.modelGraphNodeLayer._innerHTML,
    edgeHtml: ui.dom.modelGraphEdgeLayer._innerHTML,
  };
}

function fireModelGraphCanvasEvent(ui, type, target, eventValues = {}) {
  const listener = ui.dom.modelGraphCanvas.listeners[type]?.[0];
  assert.ok(listener, `model graph ${type} handler should be bound`);
  const event = {
    target,
    detail: 1,
    defaultPrevented: false,
    propagationStopped: false,
    preventDefault() { this.defaultPrevented = true; },
    stopPropagation() { this.propagationStopped = true; },
    ...eventValues,
  };
  listener(event);
  return event;
}

function portByIdentity(root, className, operatorId, portId) {
  return root.querySelectorAll(`.${className}[data-model-port-operator="${operatorId}"][data-model-port-id="${portId}"]`)[0] || null;
}

test("overview DOM refresh preserves the route plan path for short MTP branches", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = mtpBranchGraph();
  ui.state.view = "model";
  ui.state.scenario = { model: { graph, layers: [] } };
  const graphUi = ui.modelGraphUi(graph);
  graphUi.overview.positions = {
    "overview:backbone": { x: 40, y: 40 },
    "overview:mtp.prediction_layer.000": { x: 168, y: 48 },
  };
  graphUi.overview.viewport = { x: 0, y: 0, scale: 1 };

  const projection = ui.modelGraphOverviewProjection(graph);
  assert.equal(projection.edges.length, 1);
  const presentation = ui.modelGraphOverviewPresentationLayout(projection, graphUi.overview.positions);
  const routePlan = ui.modelGraphOverviewRoutePlan(projection, presentation.positions);
  const edge = projection.edges[0];
  const plannedRoute = routePlan.routes.get(`${edge.source_id}->${edge.target_id}`);
  assert.equal(plannedRoute.kind, "smooth", "fixture should exercise the short local MTP route");

  ui.renderModelGraph();

  const renderedPaths = ui.dom.modelGraphEdgeLayer.querySelectorAll("[data-model-route-source-node][data-model-route-target-node]");
  assert.equal(renderedPaths.length, 1);
  const pathElement = renderedPaths[0];
  assert.equal(pathElement.dataset.modelRouteProfile, "overview");
  assert.equal(pathElement.dataset.modelRouteOuter, "false");
  assert.equal(pathElement.classList.contains("is-skip-route"), true);
  assert.equal(pathElement.getAttribute("d"), plannedRoute.path);
  assert.equal(pathElement.classList.contains("is-smooth"), true);
});

test("200% overview rendering separates a stale MTP fork without rewriting saved coordinates", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = mtpCrowdedForkGraph();
  ui.state.view = "model";
  ui.state.settings.fontScale = 120;
  ui.state.scenario = { model: { graph, layers: [] } };
  const overview = ui.modelGraphUi(graph).overview;
  const projection = ui.modelGraphOverviewProjection(graph);
  const nodeById = new Map(projection.nodes.map((node) => [node.display_id, node]));
  const sizes120 = Object.fromEntries(projection.nodes.map((node) => [node.display_id, ui.modelGraphOverviewNodeSize(node)]));
  const ids = {
    finalNorm: "overview:final_norm",
    lmHead: "overview:lm_head",
    prediction: "overview:mtp.prediction_layer.000",
    auxiliary: "overview:mtp.aux_head",
  };
  const layoutScale120 = ui.layoutScaleForFont(120);
  const horizontalGap120 = 18 * layoutScale120;
  const layerGap120 = 28 * layoutScale120;
  overview.positions = {
    [ids.finalNorm]: { x: 196, y: 235 },
    [ids.lmHead]: { x: 208, y: 235 + sizes120[ids.finalNorm].height + layerGap120 },
    [ids.prediction]: { x: 196 + sizes120[ids.finalNorm].width + horizontalGap120, y: 235 },
    [ids.auxiliary]: {
      x: 201 + sizes120[ids.finalNorm].width + horizontalGap120,
      y: 235 + sizes120[ids.prediction].height + layerGap120,
    },
  };
  overview.viewport = { x: 0, y: 0, scale: 1 };
  const savedPositions = JSON.stringify(overview.positions);
  ui.state.settings.fontScale = 200;
  const staleRects = Object.entries(overview.positions).map(([id, position]) => ({
    id,
    left: position.x,
    top: position.y,
    right: position.x + ui.modelGraphOverviewNodeSize(nodeById.get(id)).width,
    bottom: position.y + ui.modelGraphOverviewNodeSize(nodeById.get(id)).height,
  }));
  const staleFinalNorm = staleRects.find((rect) => rect.id === ids.finalNorm);
  const stalePrediction = staleRects.find((rect) => rect.id === ids.prediction);
  assert.ok(
    staleFinalNorm.left < stalePrediction.right && staleFinalNorm.right > stalePrediction.left
      && staleFinalNorm.top < stalePrediction.bottom && staleFinalNorm.bottom > stalePrediction.top,
    "fixture must reproduce the stale 120%-coordinate overlap at 200%",
  );

  ui.renderModelGraph();

  const rects = ui.dom.modelGraphNodeLayer.querySelectorAll(".model-overview-node").map((node) => ({
    id: node.dataset.modelOverviewNode,
    ...node.getBoundingClientRect(),
  }));
  assert.equal(rects.length, 4);
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    const a = rects[left];
    const b = rects[right];
    const intersects = a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    assert.equal(intersects, false, `${a.id} overlaps ${b.id}`);
  }
  assert.equal(JSON.stringify(overview.positions), savedPositions, "render-only collision spacing must not rewrite saved manual coordinates");
  assert.equal(ui.state.dirty, false, "presentation spacing must not dirty the semantic scenario");
});

test("overview drag starts from the collision-safe presentation coordinate", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = mtpCrowdedForkGraph();
  ui.state.view = "model";
  ui.state.settings.fontScale = 120;
  ui.state.scenario = { model: { graph, layers: [] } };
  const overview = ui.modelGraphUi(graph).overview;
  const projection = ui.modelGraphOverviewProjection(graph);
  const sizes120 = Object.fromEntries(projection.nodes.map((node) => [node.display_id, ui.modelGraphOverviewNodeSize(node)]));
  const finalNormId = "overview:final_norm";
  const predictionId = "overview:mtp.prediction_layer.000";
  const layoutScale120 = ui.layoutScaleForFont(120);
  overview.positions = {
    [finalNormId]: { x: 196, y: 235 },
    "overview:lm_head": { x: 208, y: 235 + sizes120[finalNormId].height + 28 * layoutScale120 },
    [predictionId]: { x: 196 + sizes120[finalNormId].width + 18 * layoutScale120, y: 235 },
    "overview:mtp.aux_head": {
      x: 201 + sizes120[finalNormId].width + 18 * layoutScale120,
      y: 235 + sizes120[predictionId].height + 28 * layoutScale120,
    },
  };
  overview.viewport = { x: 0, y: 0, scale: 1 };
  const savedPositions = JSON.stringify(overview.positions);

  ui.state.settings.fontScale = 200;
  ui.renderModelGraph();

  const renderedOverview = ui.modelGraphUi(ui.state.scenario.model.graph).overview;
  assert.equal(JSON.stringify(renderedOverview.positions), savedPositions);
  const dragNode = ui.dom.modelGraphNodeLayer.querySelectorAll(".model-overview-node").find((node) => {
    const saved = renderedOverview.positions[node.dataset.modelOverviewNode];
    return pixelValue(node.style.left) !== saved.x || pixelValue(node.style.top) !== saved.y;
  });
  assert.ok(dragNode, "fixture must exercise a presentation-offset node");
  dragNode.offsetWidth = pixelValue(dragNode.style.width);
  dragNode.offsetHeight = pixelValue(dragNode.style.height);
  const dragId = dragNode.dataset.modelOverviewNode;
  const renderedX = pixelValue(dragNode.style.left);
  const renderedY = pixelValue(dragNode.style.top);

  ui.beginModelGraphPointer({
    button: 0,
    pointerId: 41,
    clientX: 500,
    clientY: 300,
    target: dragNode,
    preventDefault() {},
  });

  const drag = ui.state.modelGraphEditor.drag;
  assert.ok(drag);
  assert.equal(drag.originX, renderedX);
  assert.equal(drag.originY, renderedY);
  const dragOverview = ui.modelGraphUi(ui.state.scenario.model.graph).overview;
  assert.equal(JSON.stringify(dragOverview.positions), savedPositions, "starting a drag must not rewrite saved coordinates");

  ui.modelGraphApplyInteractionFrame({ pointerId: 41, clientX: 510, clientY: 300, target: dragNode });
  assert.equal(drag.previewPosition.x, renderedX + 10);
  assert.equal(drag.previewPosition.y, renderedY);
  assert.equal(dragNode.style.transform, "translate(10px, 0px)");
  assert.equal(JSON.stringify(dragOverview.positions), savedPositions, "drag preview must remain render-only until commit");
  assert.equal(ui.state.dirty, false);

  drag.moved = true;
  ui.endModelGraphPointer({ type: "pointerup", pointerId: 41, clientX: 510, clientY: 300, target: dragNode });
  const committedOverview = ui.modelGraphUi(ui.state.scenario.model.graph).overview;
  assert.equal(JSON.stringify(committedOverview.positions[dragId]), JSON.stringify({ x: renderedX + 10, y: renderedY }));
  assert.equal(ui.state.dirty, false, "committing a presentation-coordinate drag must stay semantic-only");
});

test("overview selection marks only direct one-hop edges with real source-to-target direction", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = mtpBranchGraph();
  ui.state.view = "model";
  ui.state.scenario = { model: { graph, layers: [] } };
  const graphUi = ui.modelGraphUi(graph);
  graphUi.overview.positions = {
    "overview:backbone": { x: 40, y: 40 },
    "overview:mtp.prediction_layer.000": { x: 168, y: 48 },
  };
  graphUi.overview.viewport = { x: 0, y: 0, scale: 1 };
  const edge = ui.modelGraphOverviewProjection(graph).edges[0];

  ui.state.modelGraphEditor.selectedOverviewId = edge.source_id;
  ui.renderModelGraph();
  let pathElement = ui.dom.modelGraphEdgeLayer.querySelector("[data-model-hop-direction]");
  assert.ok(pathElement);
  assert.equal(pathElement.dataset.modelHopDirection, "outgoing");
  assert.equal(pathElement.classList.contains("is-one-hop"), true);
  assert.equal(pathElement.classList.contains("is-outgoing"), true);

  ui.state.modelGraphEditor.selectedOverviewId = edge.target_id;
  ui.renderModelGraph();
  pathElement = ui.dom.modelGraphEdgeLayer.querySelector("[data-model-hop-direction]");
  assert.ok(pathElement);
  assert.equal(pathElement.dataset.modelHopDirection, "incoming");
  assert.equal(pathElement.classList.contains("is-incoming"), true);
});

for (const [name, graphFactory] of [
  ["explicit empty graph", () => ModelGraphCore.normalizeModelGraph({ graph_id: "empty", operators: [], tensors: [] })],
  ["boundary-only graph", boundaryOnlyGraph],
]) {
  test(`overview render handles an empty projection from ${name}`, () => {
    const ui = appHelpers();
    attachModelDom(ui);
    const graph = graphFactory();
    ui.state.view = "model";
    ui.state.scenario = { model: { graph, layers: [] } };
    const projection = ui.modelGraphOverviewProjection(graph);
    assert.equal(projection.nodes.length, 0);
    assert.equal(projection.edges.length, 0);

    assert.doesNotThrow(() => ui.renderModelGraph());
    assert.equal(ui.dom.modelGraphNodeLayer.children.length, 0);
  assert.equal(ui.dom.modelGraphEdgeLayer.children.length, 0);
  });
}

test("expanded repeat groups render compact inline labels without extra headings or visible operator IDs", () => {
  const { html } = expandedBlockOverviewFixture();
  const text = visibleText(html);

  assert.doesNotMatch(html, /class="model-inline-heading"/);
  assert.doesNotMatch(text, /Block 权威子图|operator · typed port · tensor edge/u);
  assert.match(text, /RMS Norm|rms_norm|RMS 归一化/u);
  assert.match(text, /Dense MLP|dense_mlp|MLP/u);
  assert.match(text, /Residual Add|residual_add|残差/u);
  for (const operatorId of [
    "block-group-000.norm1",
    "block-group-000.attention",
    "block-group-000.mlp",
    "block-group-000.residual2",
  ]) {
    assert.doesNotMatch(text, new RegExp(escapeRegex(operatorId), "u"), `${operatorId} should stay out of visible labels`);
  }

  assert.match(html, /data-model-inline-operator="block-group-000\.mlp"/);
  assert.match(html, /data-model-select-operator="block-group-000\.mlp"/);
  assert.match(html, /data-model-route-(?:source|target)-operator="block-group-000\.mlp"/);
});

test("expanded repeat group overview edges keep scoped boundary ports when inline ports share an identity", () => {
  const { ui, graph, projection } = expandedBlockOverviewFixture();
  const groupNode = projection.nodes.find((node) => node.kind === "repeat_group");
  assert.ok(groupNode);

  const graphUi = ui.modelGraphUi(graph);
  const routePlan = ui.modelGraphOverviewRoutePlan(projection, graphUi.overview.positions);
  const externalEdges = projection.edges.filter((edge) => (
    !edge.visual_only && (edge.source_id === groupNode.display_id || edge.target_id === groupNode.display_id)
  ));
  assert.ok(externalEdges.length >= 2, "fixture should include edges entering and leaving the expanded group");

  for (const edge of externalEdges) {
    assert.ok(edge.representative_edge, `${edge.source_id}->${edge.target_id} should retain representative boundary endpoints`);
    const rendered = ui.dom.modelGraphEdgeLayer.querySelector(
      `[data-model-route-source-node="${edge.source_id}"][data-model-route-target-node="${edge.target_id}"]`,
    );
    assert.ok(rendered, `missing rendered overview edge ${edge.source_id}->${edge.target_id}`);
    assert.equal(rendered.dataset.modelRouteProfile, "overview");
    assert.equal(rendered.dataset.modelRouteSourceNode, edge.source_id);
    assert.equal(rendered.dataset.modelRouteTargetNode, edge.target_id);
    assert.equal(rendered.dataset.modelRouteSourceOperator, edge.representative_edge.source.operator_id);
    assert.equal(rendered.dataset.modelRouteSourcePort, edge.representative_edge.source.port_id);
    assert.equal(rendered.dataset.modelRouteTargetOperator, edge.representative_edge.target.operator_id);
    assert.equal(rendered.dataset.modelRouteTargetPort, edge.representative_edge.target.port_id);

    const sourceNode = projection.nodes.find((node) => node.display_id === edge.source_id);
    const targetNode = projection.nodes.find((node) => node.display_id === edge.target_id);
    assert.ok(sourceNode.boundary_ports.some((port) => (
      port.operator_id === rendered.dataset.modelRouteSourceOperator
        && port.port_id === rendered.dataset.modelRouteSourcePort
        && port.direction === "output"
    )));
    assert.ok(targetNode.boundary_ports.some((port) => (
      port.operator_id === rendered.dataset.modelRouteTargetOperator
        && port.port_id === rendered.dataset.modelRouteTargetPort
        && port.direction === "input"
    )));

    const plannedRoute = routePlan.routes.get(`${edge.source_id}->${edge.target_id}`);
    assert.ok(plannedRoute);
    assert.equal(rendered.getAttribute("d"), plannedRoute.path, "DOM rerouting should still use overview boundary dots");
  }

  const inbound = externalEdges.find((edge) => edge.target_id === groupNode.display_id);
  assert.ok(inbound?.representative_edge);
  const target = inbound.representative_edge.target;
  const overviewPort = portByIdentity(ui.dom.modelGraphNodeLayer, "model-overview-port", target.operator_id, target.port_id);
  const inlinePort = portByIdentity(ui.dom.modelGraphNodeLayer, "model-inline-port", target.operator_id, target.port_id);
  assert.ok(overviewPort, "expanded group should render an overview boundary port");
  assert.ok(inlinePort, "expanded group should also render the same authoritative port inside the inline subgraph");
  assert.equal(overviewPort.dataset.modelRouteScope, "overview");
  assert.equal(inlinePort.dataset.modelRouteScope, "inline");
  assert.notEqual(overviewPort, inlinePort);
});

test("expanded MoE repeat group toggle flips the rendered compound state instead of acting as a dead button", () => {
  const { ui, projection } = expandedBlockOverviewFixture({
    layerOverrides: {
      kind: "moe",
      num_experts: 8,
      experts_per_token: 2,
      shared_expert_intermediate_size: 1024,
      shared_expert_gate: true,
    },
  });
  const groupNode = projection.nodes.find((node) => node.kind === "repeat_group");
  const compoundId = `${groupNode.group_id}.moe`;

  let toggle = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-compound-toggle[data-model-overview-moe="${compoundId}"]`);
  assert.ok(toggle);
  assert.equal(toggle.getAttribute("type"), "button");
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.doesNotMatch(toggle.getAttribute("aria-label"), /^\s*$/u);

  const collapseEvent = fireModelGraphCanvasEvent(ui, "click", toggle);
  assert.equal(collapseEvent.propagationStopped, true);

  toggle = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-compound-toggle[data-model-overview-moe="${compoundId}"]`);
  assert.ok(toggle);
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.match(ui.dom.modelGraphNodeLayer._innerHTML, /model-inline-collapsed-note/);

  const expandEvent = fireModelGraphCanvasEvent(ui, "click", toggle);
  assert.equal(expandEvent.propagationStopped, true);
  toggle = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-compound-toggle[data-model-overview-moe="${compoundId}"]`);
  assert.ok(toggle);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.doesNotMatch(ui.dom.modelGraphNodeLayer._innerHTML, /model-inline-collapsed-note/);
});

test("repeat-group expansion and collapse use the global responsive layout and keep view state semantic-only", () => {
  const { ui, groupNode } = expandedBlockOverviewFixture();
  const before = ui.modelGraphSnapshot();
  const currentGraph = () => ui.state.scenario.model.graph;

  const assertGlobalLayout = (label) => {
    const activeGraph = currentGraph();
    const overview = ui.modelGraphUi(activeGraph).overview;
    const projection = ui.modelGraphOverviewProjection(activeGraph);
    const scale = ui.layoutScaleForFont(ui.state.settings.fontScale);
    const nodeSizes = Object.fromEntries(projection.nodes.map((node) => [
      node.display_id,
      ui.modelGraphOverviewNodeSize(node),
    ]));
    const expected = ModelGraphCore.overviewResponsiveLayout(
      projection,
      ui.dom.modelGraphCanvas.clientWidth,
      scale,
      { nodeSizes },
    );
    assert.deepEqual(overview.positions, expected.positions, `${label}: all overview nodes should use the global layout pass`);
    assert.deepEqual(overview.viewport, expected.viewport, `${label}: viewport should come from the global layout pass`);
    assert.equal(
      ui.dom.modelGraphWorld.style.transform,
      `translate(${expected.viewport.x}px, ${expected.viewport.y}px) scale(${expected.viewport.scale})`,
      `${label}: rendered viewport should be synchronized`,
    );
  };

  let toggle = ui.dom.modelGraphNodeLayer.querySelector(`[data-model-toggle-overview-group="${groupNode.display_id}"]`);
  assert.ok(toggle);
  fireModelGraphCanvasEvent(ui, "click", toggle);
  assert.equal(ui.modelGraphOverviewProjection(currentGraph()).nodes.find((node) => node.display_id === groupNode.display_id).expanded, false);
  assertGlobalLayout("collapse");

  toggle = ui.dom.modelGraphNodeLayer.querySelector(`[data-model-toggle-overview-group="${groupNode.display_id}"]`);
  assert.ok(toggle);
  fireModelGraphCanvasEvent(ui, "click", toggle);
  assert.equal(ui.modelGraphOverviewProjection(currentGraph()).nodes.find((node) => node.display_id === groupNode.display_id).expanded, true);
  assertGlobalLayout("expand");

  assert.deepEqual(ui.modelGraphSnapshot(), before, "collapse/expand must not change model semantics or mapping inputs");
});

test("internal MoE expansion and collapse use the same global layout and viewport path", () => {
  const { ui, groupNode } = expandedBlockOverviewFixture({
    layerOverrides: {
      kind: "moe",
      num_experts: 8,
      experts_per_token: 2,
      shared_expert_intermediate_size: 1024,
      shared_expert_gate: true,
    },
  });
  const before = ui.modelGraphSnapshot();
  const currentGraph = () => ui.state.scenario.model.graph;
  const assertGlobalLayout = (label) => {
    const activeGraph = currentGraph();
    const overview = ui.modelGraphUi(activeGraph).overview;
    const projection = ui.modelGraphOverviewProjection(activeGraph);
    const scale = ui.layoutScaleForFont(ui.state.settings.fontScale);
    const nodeSizes = Object.fromEntries(projection.nodes.map((node) => [
      node.display_id,
      ui.modelGraphOverviewNodeSize(node),
    ]));
    const expected = ModelGraphCore.overviewResponsiveLayout(
      projection,
      ui.dom.modelGraphCanvas.clientWidth,
      scale,
      { nodeSizes },
    );
    assert.deepEqual(overview.positions, expected.positions, `${label}: all overview nodes should use the global layout pass`);
    assert.deepEqual(overview.viewport, expected.viewport, `${label}: viewport should come from the global layout pass`);
    assert.equal(
      ui.dom.modelGraphWorld.style.transform,
      `translate(${expected.viewport.x}px, ${expected.viewport.y}px) scale(${expected.viewport.scale})`,
      `${label}: rendered viewport should be synchronized`,
    );
  };

  const compoundId = `${groupNode.group_id}.moe`;
  let toggle = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-compound-toggle[data-model-overview-moe="${compoundId}"]`);
  assert.ok(toggle);
  fireModelGraphCanvasEvent(ui, "click", toggle);
  assertGlobalLayout("MoE collapse");

  toggle = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-compound-toggle[data-model-overview-moe="${compoundId}"]`);
  assert.ok(toggle);
  fireModelGraphCanvasEvent(ui, "click", toggle);
  assertGlobalLayout("MoE expand");

  assert.deepEqual(ui.modelGraphSnapshot(), before, "MoE collapse/expand must not change model semantics or mapping inputs");
});

test("expanded attention renders a derived-only replacement without shape readouts", () => {
  const { html } = expandedBlockOverviewFixture();
  const derived = /<aside class="model-attention-derived"[\s\S]*?<\/aside>/u.exec(html)?.[0] || "";
  const attentionNode = elementTags(html, "article", "model-inline-operator")
    .find((tag) => parseAttributes(tag)["data-model-inline-operator"] === "block-group-000.attention");

  assert.ok(derived, "expanded attention should render its derived lowering graph");
  assert.ok(attentionNode);
  assert.ok(classTokens(attentionNode).has("is-attention-derived"));
  assert.equal(parseAttributes(attentionNode)["data-model-select-operator"], "block-group-000.attention");
  assert.match(derived, /data-derived-only="true"/);
  assert.doesNotMatch(html, /data-model-overview-attention="block-group-000\.attention"/);
  assert.doesNotMatch(derived, /data-authoritative="true"/);
  assert.doesNotMatch(derived, /class="is-shape"/);
  assert.doesNotMatch(visibleText(derived), /\bB\s*×|512\b/u);
  assert.match(visibleText(derived), /GQA|Q|KV|head_dim/u);
});

test("inline operators expose focusable Enter and Space selection semantics", () => {
  const { ui, html, groupNode } = expandedBlockOverviewFixture();
  const selectedId = "block-group-000.mlp";
  const nodeTag = elementTags(html, "article", "model-inline-operator")
    .find((tag) => parseAttributes(tag)["data-model-inline-operator"] === selectedId);
  assert.ok(nodeTag);
  const attrs = parseAttributes(nodeTag);
  assert.equal(attrs.role, "button");
  assert.equal(attrs.tabindex, "0");
  assert.equal(attrs["data-model-select-operator"], selectedId);
  assert.equal(attrs["aria-pressed"], "false");

  const inlineNode = ui.dom.modelGraphNodeLayer.querySelector(`.model-inline-operator[data-model-select-operator="${selectedId}"]`);
  assert.ok(inlineNode);
  for (const key of ["Enter", " "]) {
    ui.state.modelGraphEditor.selectedOperatorId = "";
    inlineNode.classList.remove("is-selected");
    inlineNode.setAttribute("aria-pressed", "false");
    const event = fireModelGraphCanvasEvent(ui, "keydown", inlineNode, { key });
    assert.equal(event.defaultPrevented, true, `${key} should prevent page scrolling/form submission`);
    assert.equal(event.propagationStopped, true, `${key} should be handled by the inline operator`);
    assert.equal(ui.state.modelGraphEditor.selectedOperatorId, selectedId);
    assert.equal(ui.state.modelGraphEditor.selectedOverviewId, groupNode.display_id);
    assert.ok(inlineNode.classList.contains("is-selected"));
    assert.equal(inlineNode.getAttribute("aria-pressed"), "true");
  }
});

for (const [name, layerOverrides, selectedId] of [
  ["linear attention", { sequence_mixer: "linear_attention", linear_attention: { key_heads: 4, value_heads: 8, state_dtype: "fp32" } }, "block-group-000.linear_attention"],
  ["another group operator", {}, "block-group-000.mlp"],
]) {
  test(`mouse selection of ${name} keeps the inspector through a full overview redraw`, () => {
    const { ui, groupNode } = expandedBlockOverviewFixture({ layerOverrides });
    const inlineNode = ui.dom.modelGraphNodeLayer.querySelector(`[data-model-inline-operator="${selectedId}"]`);
    assert.ok(inlineNode, `${selectedId} should render inside the expanded group`);

    fireModelGraphCanvasEvent(ui, "click", inlineNode);
    assert.equal(ui.state.modelGraphEditor.selectedOperatorId, selectedId);
    assert.equal(ui.state.modelGraphEditor.selectedOverviewId, groupNode.display_id);
    assert.equal(ui.dom.modelGraphInspectorTitle.textContent, selectedId);
    assert.ok(inlineNode.classList.contains("is-selected"));
    assert.equal(inlineNode.getAttribute("aria-pressed"), "true");

    ui.renderModelGraph();
    const refreshedInlineNode = ui.dom.modelGraphNodeLayer.querySelector(`[data-model-inline-operator="${selectedId}"]`);
    assert.ok(refreshedInlineNode);
    assert.equal(ui.state.modelGraphEditor.selectedOperatorId, selectedId);
    assert.equal(ui.state.modelGraphEditor.selectedOverviewId, groupNode.display_id);
    assert.equal(ui.dom.modelGraphInspectorTitle.textContent, selectedId);
    assert.ok(refreshedInlineNode.classList.contains("is-selected"));
    assert.equal(refreshedInlineNode.getAttribute("aria-pressed"), "true");

    const selectedEdges = ui.dom.modelGraphNodeLayer.querySelectorAll(".model-inline-edge.is-one-hop");
    assert.ok(selectedEdges.length >= 2, "the selected operator should retain its one-hop animation state");
  });
}

test("overview redraw keeps the inspector empty when nothing is selected", () => {
  const { ui } = expandedBlockOverviewFixture();
  ui.state.modelGraphEditor.selectedOperatorId = null;
  ui.state.modelGraphEditor.selectedOverviewId = null;

  ui.renderModelGraph();

  assert.equal(ui.dom.modelGraphInspectorTitle.textContent, "未选择组件");
  assert.match(ui.dom.modelGraphInspectorContent._innerHTML, /class="inspector-empty"/);
});

test("selecting an operator inside an expanded group marks the same one-hop edge states as overview selection", () => {
  const { ui, html } = expandedBlockOverviewFixture({ selectedOperatorId: "block-group-000.mlp" });
  const selectedNode = elementTags(html, "article", "model-inline-operator")
    .find((tag) => parseAttributes(tag)["data-model-inline-operator"] === "block-group-000.mlp");
  assert.ok(selectedNode);
  assert.ok(classTokens(selectedNode).has("is-selected"));

  const edgeElements = ui.dom.modelGraphNodeLayer.querySelectorAll(".model-inline-edge");
  const oneHopEdges = edgeElements.filter((edge) => edge.classList.contains("is-one-hop"));
  assert.equal(oneHopEdges.length, 2);

  const incoming = oneHopEdges.find((edge) => edge.dataset.modelRouteTargetNode === "block-group-000.mlp");
  const outgoing = oneHopEdges.find((edge) => edge.dataset.modelRouteSourceNode === "block-group-000.mlp");
  assert.ok(incoming);
  assert.ok(outgoing);
  assert.equal(incoming.dataset.modelHopDirection, "incoming");
  assert.ok(incoming.classList.contains("is-incoming"));
  assert.equal(outgoing.dataset.modelHopDirection, "outgoing");
  assert.ok(outgoing.classList.contains("is-outgoing"));

  const unrelated = edgeElements.filter((edge) => {
    return edge.dataset.modelRouteSourceNode !== "block-group-000.mlp"
      && edge.dataset.modelRouteTargetNode !== "block-group-000.mlp";
  });
  assert.ok(unrelated.every((edge) => !edge.classList.contains("is-one-hop")));
});

test("residual inline routes remain local and opt out of outer graph routing", () => {
  const { ui } = expandedBlockOverviewFixture();
  const residualEdges = ui.dom.modelGraphNodeLayer.querySelectorAll(".model-inline-edge.is-residual");
  assert.ok(residualEdges.length >= 2, "fixture should include residual skip edges");

  for (const edge of residualEdges) {
    assert.equal(edge.dataset.modelRouteOuter, "false");
    assert.equal(edge.classList.contains("is-skip-route"), true);
  }
});

test("expanded repeat group shells and inline operators keep stable measured layout constraints", () => {
  const { ui, projection, html } = expandedBlockOverviewFixture();
  const groupNode = projection.nodes.find((node) => node.kind === "repeat_group");
  const expectedSize = ui.modelGraphOverviewNodeSize(groupNode);
  const groupTag = elementTags(html, "article", "model-overview-node")
    .find((tag) => parseAttributes(tag)["data-model-overview-node"] === groupNode.display_id);
  assert.ok(groupTag);
  const groupStyle = parseStyle(parseAttributes(groupTag).style);
  assert.equal(pixelValue(groupStyle.width), expectedSize.width);
  assert.equal(pixelValue(groupStyle.height), expectedSize.height);
  const inlineGraphTag = elementTags(html, "div", "model-inline-graph")[0];
  assert.ok(inlineGraphTag);
  const inlineGraphStyle = parseStyle(parseAttributes(inlineGraphTag).style);
  assert.ok(expectedSize.width >= pixelValue(inlineGraphStyle.width));
  assert.ok(expectedSize.height >= pixelValue(inlineGraphStyle.height));
  assert.ok(expectedSize.width - pixelValue(inlineGraphStyle.width) <= 24);
  assert.ok(expectedSize.height - pixelValue(inlineGraphStyle.height) <= 72);

  const inlineRects = elementTags(html, "article", "model-inline-operator").map((tag) => {
    const attrs = parseAttributes(tag);
    const style = parseStyle(attrs.style);
    return {
      id: attrs["data-model-inline-operator"],
      x: pixelValue(style.left),
      y: pixelValue(style.top),
      width: pixelValue(style.width),
      height: pixelValue(style.height),
    };
  });
  assert.ok(inlineRects.length >= 4);
  assert.ok(inlineRects.every((rect) => rect.width > 0 && rect.height > 0));
  const attentionRect = inlineRects.find((rect) => rect.id === "block-group-000.attention");
  assert.ok(attentionRect);
  const compactRects = inlineRects.filter((rect) => rect.id !== "block-group-000.attention");
  assert.ok(compactRects.every((rect) => rect.width <= 140 && rect.height <= 52));
  assert.ok(attentionRect.width > Math.max(...compactRects.map((rect) => rect.width)));
  assert.ok(attentionRect.height > Math.max(...compactRects.map((rect) => rect.height)));

  for (let left = 0; left < inlineRects.length; left += 1) for (let right = left + 1; right < inlineRects.length; right += 1) {
    const a = inlineRects[left];
    const b = inlineRects[right];
    const intersects = a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
    assert.equal(intersects, false, `${a.id} overlaps ${b.id}`);
  }
});

test("collapsed repeat group sizing at 200% reserves width for the toggle beside long labels", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  ui.state.settings.fontScale = 200;
  assert.equal(ui.fontScaleBand(200), "extreme");

  const label = "Custom Ultra Long Decoder Block Group Label For Accessible Zoom";
  const repeatNode = {
    kind: "repeat_group",
    op_kind: "custom_decoder_stack",
    label,
    repeat_count: 12,
    expanded: false,
    boundary_ports: [],
  };
  const sameTextLeaf = {
    ...repeatNode,
    kind: "operator",
    label: `${label} ×12`,
    repeat_count: 1,
  };

  const repeatSize = ui.modelGraphOverviewNodeSize(repeatNode);
  const leafSize = ui.modelGraphOverviewNodeSize(sameTextLeaf);
  const toggleWidth = 24 * ui.layoutScaleForFont(200);
  assert.ok(
    repeatSize.width >= leafSize.width + toggleWidth,
    `repeat group width ${repeatSize.width} should reserve at least one scaled toggle beyond leaf width ${leafSize.width}`,
  );
  assert.ok(
    repeatSize.width >= ui.modelGraphOverviewTextWidth(label) + toggleWidth,
    "repeat group measurement should include the long label plus at least the scaled toggle allowance",
  );
});

test("collapsed repeat groups grow for multiple boundary port rows while one-row groups stay compact", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = multiBoundaryRepeatGraph();
  ui.state.view = "model";
  ui.state.settings.fontScale = 100;
  ui.state.scenario = { model: { graph, layers: [] } };
  const projection = ui.modelGraphOverviewProjection(graph);
  const groupNode = projection.nodes.find((node) => node.kind === "repeat_group");
  assert.ok(groupNode);
  assert.equal(groupNode.expanded, false);
  assert.deepEqual(ModelGraphCore.overviewBoundaryPortCounts(groupNode), { inputs: 4, outputs: 4, rows: 4 });

  const singleRowNode = {
    ...groupNode,
    boundary_ports: groupNode.boundary_ports.filter((_port, index) => index === 0 || index === 4),
  };
  const compact = ui.modelGraphOverviewNodeSize(singleRowNode);
  const multi = ui.modelGraphOverviewNodeSize(groupNode);
  assert.ok(compact.height >= 36 && compact.height <= 38);
  assert.equal(multi.height, compact.height + 48);

  ui.renderModelGraph();
  const renderedGroup = ui.dom.modelGraphNodeLayer.querySelector(`[data-model-overview-node="${groupNode.display_id}"]`);
  assert.ok(renderedGroup);
  assert.equal(pixelValue(renderedGroup.style.height), multi.height);
  for (const direction of ["input", "output"]) {
    const ports = renderedGroup.querySelectorAll(`[data-model-port-direction="${direction}"]`);
    assert.equal(ports.length, 4);
    const centers = ports.map((port) => {
      const rect = port.getBoundingClientRect();
      return rect.top + rect.height / 2;
    }).sort((left, right) => left - right);
    for (let index = 1; index < centers.length; index += 1) {
      assert.ok(centers[index] - centers[index - 1] >= 14, `${direction} ports should not overlap`);
    }
  }
});

test("semantic model undo and redo preserve newer graph UI collapse layout and viewport", () => {
  const ui = appHelpers();
  attachModelDom(ui);
  const graph = multiBoundaryRepeatGraph();
  ui.state.view = "model";
  ui.state.scenario = { model: { graph, layers: [{ layer_id: "layer0", kind: "dense_mlp" }] } };
  const beforeSemantic = ui.modelGraphSnapshot();
  const activeGraph = ui.state.scenario.model.graph;
  activeGraph.operators.find((operator) => operator.operator_id === "block0").parameters.activation = "gelu";
  assert.equal(ui.commitModelGraphHistory(beforeSemantic, "编辑模型参数", true), true);
  assert.equal(ui.state.modelGraphEditor.history.undo.length, 1);

  const beforeUiOnly = ui.modelGraphSnapshot();
  const uiGraph = ui.state.scenario.model.graph;
  const graphUi = ui.modelGraphUi(uiGraph);
  graphUi.overview.collapsed_groups = [];
  graphUi.overview.positions["overview:group:group0"] = { x: 333, y: 222 };
  graphUi.overview.viewport = { x: 44, y: 55, scale: 0.75 };
  graphUi.detail.collapsed_groups = ["group0"];
  graphUi.detail.positions.block0 = { x: 111, y: 222 };
  graphUi.detail.viewport = { x: 9, y: 10, scale: 1.4 };
  ui.modelGraphUi(uiGraph);
  const expectedUi = modelGraphUiState(uiGraph);
  assert.equal(ui.commitModelGraphHistory(beforeUiOnly, "移动模型 UI", false), false);
  assert.equal(ui.state.modelGraphEditor.history.undo.length, 1);
  assert.equal(ui.state.dirty, false);

  ui.travelModelGraphHistory("undo");
  assert.equal(ui.state.scenario.model.graph.operators.find((operator) => operator.operator_id === "block0").parameters.activation, undefined);
  assert.deepEqual(modelGraphUiState(ui.state.scenario.model.graph), expectedUi);
  assert.equal(ui.state.modelGraphEditor.history.undo.length, 0);
  assert.equal(ui.state.modelGraphEditor.history.redo.length, 1);

  ui.travelModelGraphHistory("redo");
  assert.equal(ui.state.scenario.model.graph.operators.find((operator) => operator.operator_id === "block0").parameters.activation, "gelu");
  assert.deepEqual(modelGraphUiState(ui.state.scenario.model.graph), expectedUi);
  assert.equal(ui.markScenarioChangedCalls.length, 2);
  assert.ok(ui.markScenarioChangedCalls.every((call) => call.options.mappingImpact === true));
});
