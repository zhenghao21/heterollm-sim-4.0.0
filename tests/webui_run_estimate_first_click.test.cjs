"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const source = fs.readFileSync(appPath, "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

class FakeClassList {
  constructor(element) {
    this.element = element;
    this.values = new Set();
  }

  add(...names) {
    names.forEach((name) => this.values.add(String(name)));
    this.sync();
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(String(name)));
    this.sync();
  }

  toggle(name, force) {
    const value = String(name);
    const enabled = force === undefined ? !this.values.has(value) : Boolean(force);
    if (enabled) this.values.add(value);
    else this.values.delete(value);
    this.sync();
    return enabled;
  }

  sync() {
    if (this.values.size) this.element.className = Array.from(this.values).join(" ");
  }
}

class FakeHTMLElement {
  constructor(id = "") {
    this.id = id;
    this.isConnected = true;
    this.open = false;
    this.hidden = false;
    this.disabled = false;
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.className = "";
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.listeners = {};
    this.ownerDocument = null;
    this.classList = new FakeClassList(this);
  }

  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }

  dispatch(type, event = {}) {
    const payload = {
      target: this,
      currentTarget: this,
      preventDefault() {},
      stopPropagation() {},
      ...event,
    };
    for (const listener of this.listeners[type] || []) listener(payload);
  }

  querySelector(selector) {
    if (selector === "span") return this.children.find((item) => item.tagName === "span") || null;
    if (selector === "button, input, select, textarea") return this.children[0] || null;
    return null;
  }

  querySelectorAll() {
    return [];
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  focus() {
    this.focused = true;
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }
}

class FakeDialog extends FakeHTMLElement {
  showModal() {
    this.open = true;
  }

  close(returnValue = "") {
    this.open = false;
    this.returnValue = returnValue;
    this.dispatch("close", { target: this });
  }
}

function response(status, payload) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => "application/json" },
    async json() { return payload; },
    async text() { return JSON.stringify(payload); },
  };
}

function minimalModelGraph() {
  return ModelGraphCore.buildModelGraphFromLayerSpecs([{
    schema_version: "4.0.0",
    layer_id: "layer0",
    kind: "dense",
    hidden_size: 512,
    intermediate_size: 2048,
    attention_heads: 8,
    kv_heads: 8,
    attention_head_dim: 64,
    sequence_mixer: "full_attention",
    dtype: "bf16",
    num_experts: 1,
    experts_per_token: 1,
  }], { name: "first-click-model", schema_version: "4.0.0", vocabulary_size: 32000 });
}

function scenario() {
  return {
    schema_version: "4.0.0",
    hardware: {
      metadata: {},
      components: [
        { component_id: "gpu0", kind: "gpu", cost_profile_id: "gpu-default", peak_ops_per_s: 120e12, ports: [] },
        { component_id: "hbm0", kind: "hbm", cost_profile_id: "hbm-default", read_bandwidth_gbps: 4096, ports: [] },
        { component_id: "cpu0", kind: "cpu", cost_profile_id: "cpu-default", peak_ops_per_s: 100e9, ports: [] },
        { component_id: "ram0", kind: "host_memory", cost_profile_id: "host-memory-default", read_bandwidth_gbps: 1600, ports: [] },
      ],
      links: [],
    },
    model: { graph: minimalModelGraph() },
    placement: {},
    workload: {
      requests: [],
      request_count: 1,
      prompt_tokens: 16,
      output_tokens: 4,
    },
    profiles: {
      components: {
        gpu: { "gpu-default": { tensor_core: { sm_count: 1 }, cache_hierarchy: { levels: [{}] } } },
        hbm: { "hbm-default": { bandwidth_gb_s: 1 } },
        cpu: { "cpu-default": { pipeline: { core_count: 1 }, cache_hierarchy: { levels: [{}] } } },
        host_memory: { "host-memory-default": { bandwidth_gb_s: 1 } },
      },
      host_orchestration: {
        cpu_component_id: "cpu0",
        gpu_component_id: "gpu0",
        scheduler_resource_id: "cpu0.scheduler",
        pack_resource_id: "cpu0.pack",
        dma_resource_id: "cpu0.dma",
        submission_resource_id: "gpu0.queue",
      },
    },
  };
}

function estimatePayload() {
  return {
    schema_version: "1.0",
    risk_level: "low",
    risk_level_zh: "低",
    request_count: 1,
    prompt_tokens: 16,
    output_tokens: 4,
    layer_count: 1,
    world_size: 1,
    estimated_cohort_count: 1,
    estimated_event_task_count: 12,
    recommended_retention_policy: "exact",
    warnings: [],
  };
}

function loadRunHarness(fetchImpl) {
  const documentListeners = {};
  const document = {
    activeElement: null,
    addEventListener(type, listener) {
      (documentListeners[type] ||= []).push(listener);
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
  };
  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    HTMLElement: FakeHTMLElement,
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
    document,
    fetch: (...args) => fetchImpl(...args),
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) { callback(0); return 1; },
    setTimeout,
    structuredClone,
    window: { clearTimeout, setTimeout },
  });
  vm.runInContext(`${source}
    const __toasts = [];
    const __operationErrors = [];
    renderSteps = () => {};
    renderDiagnostics = () => {};
    openDiagnostics = () => {};
    renderRuntimeHealth = () => {};
    renderAll = () => {};
    switchView = (view) => { state.activeView = view; };
    scheduleRunJobPoll = () => {};
    toast = (title, message = "", kind = "info") => { __toasts.push({ title, message, kind }); };
    showOperationError = (title, error) => { __operationErrors.push({ title, code: error?.code, message: error?.message }); };
    globalThis.__runHarness = {
      state,
      dom,
      runScenario,
      startRunJob,
      finishRunJob,
      compareScenario,
      scenarioPayloadForTransport,
      controlPlaneDecision,
      controlPlaneEvidence,
      validationIssueTarget,
      inferValidationIssueFields,
      normalizeIssue,
      beginValidationNavigation,
      advanceValidationNavigation,
      setScenario,
      acceptRuntimePlacement,
      effectiveParallelRanks,
      bindModalDialogLifecycle,
      toasts: __toasts,
      operationErrors: __operationErrors,
    };`, context, { filename: appPath });
  const ui = context.__runHarness;
  const ids = [
    "runButton", "rerunButton", "emptyRunButton", "compareButton", "validateButton",
    "loadReferenceButton", "canonicalExportButton", "canonicalExportDialogButton",
    "busyOverlay", "busyTitle", "busyDetail",
    "runEstimateRisk", "runEstimateSummary", "runEstimateWarnings",
    "runJobProgressPanel", "runJobStatus", "runProgressStage", "runProgressCount",
    "runProgressBar", "runProgressMessage", "dismissRunJobButton",
    "cancelRunJobButton", "startRunJobButton",
    "inspectorContent", "inspectorTitle", "deleteSelectionButton", "nodeLayer", "linkLayer", "topologyCanvas",
  ];
  ids.forEach((id) => {
    ui.dom[id] = new FakeHTMLElement(id);
    ui.dom[id].ownerDocument = document;
  });
  ui.dom.runJobDialog = new FakeDialog("runJobDialog");
  ui.dom.runJobDialog.ownerDocument = document;
  ui.dom.runJobDialog.children.push(ui.dom.startRunJobButton);
  const connectionLabel = new FakeHTMLElement("connectionStateLabel");
  connectionLabel.tagName = "span";
  ui.dom.connectionState = new FakeHTMLElement("connectionState");
  ui.dom.connectionState.children.push(connectionLabel);
  ui.dom.connectionState.ownerDocument = document;
  ui.bindModalDialogLifecycle();
  ui.state.scenario = scenario();
  return ui;
}

test("topbar run first click estimates first, opens the run dialog, and does not start a job", async () => {
  assert.match(source, /dom\.runButton\.addEventListener\("click", runScenario\)/);

  const calls = [];
  const ui = loadRunHarness(async (url, options = {}) => {
    calls.push({ url, method: options.method || "GET", body: options.body || "" });
    if (url === "/api/validate") {
      return response(200, {
        validation: { valid: true, errors: [], warnings: [], information: [] },
        input_fingerprint: "fp-first-click",
        current_input_fingerprint: "fp-first-click",
      });
    }
    if (url === "/api/run-estimate") return response(200, estimatePayload());
    throw new TypeError(`unexpected fetch: ${url}`);
  });

  await ui.runScenario({ currentTarget: ui.dom.runButton });

  assert.deepEqual(calls.map((item) => `${item.method} ${item.url}`), [
    "POST /api/validate",
    "POST /api/run-estimate",
  ]);
  assert.equal(calls.some((item) => item.url === "/api/run" || item.url === "/api/run-jobs"), false);
  assert.equal(ui.operationErrors.length, 0);
  assert.equal(ui.state.connection.status, "online");
  assert.equal(ui.dom.runJobDialog.open, true);
  assert.equal(ui.dom.startRunJobButton.disabled, false);
  assert.match(ui.dom.runEstimateRisk.textContent, /风险：低/);
  assert.match(ui.dom.runEstimateSummary.innerHTML, /请求数（Requests）/);
});

test("a first-click transport reset keeps a reachable API online and leaves run retryable", async () => {
  const calls = [];
  let failEstimateOnce = true;
  const ui = loadRunHarness(async (url, options = {}) => {
    calls.push({ url, method: options.method || "GET" });
    if (url === "/api/validate") {
      return response(200, { validation: { valid: true, errors: [], warnings: [], information: [] } });
    }
    if (url === "/api/run-estimate" && failEstimateOnce) {
      failEstimateOnce = false;
      throw new TypeError("offline package transiently refused the first estimate request");
    }
    if (url === "/api/health") return response(200, { ok: true });
    if (url === "/api/run-estimate") return response(200, estimatePayload());
    throw new TypeError(`unexpected fetch: ${url}`);
  });

  await ui.runScenario({ currentTarget: ui.dom.runButton });

  assert.equal(ui.dom.runJobDialog.open, false);
  assert.equal(ui.operationErrors.at(-1)?.code, "request_transport_error");
  assert.equal(ui.state.busy, false);
  assert.equal(ui.dom.runButton.disabled, false);
  assert.equal(ui.state.connection.status, "online");

  await ui.runScenario({ currentTarget: ui.dom.runButton });

  assert.deepEqual(calls.map((item) => `${item.method} ${item.url}`), [
    "POST /api/validate",
    "POST /api/run-estimate",
    "GET /api/health",
    "POST /api/validate",
    "POST /api/run-estimate",
  ]);
  assert.equal(ui.operationErrors.length, 1);
  assert.equal(ui.state.connection.status, "online");
  assert.equal(ui.dom.runJobDialog.open, true);
  assert.equal(ui.dom.startRunJobButton.disabled, false);
});

test("a failed health probe does not mislabel the API as online", async () => {
  const calls = [];
  const ui = loadRunHarness(async (url, options = {}) => {
    calls.push({ url, method: options.method || "GET" });
    if (url === "/api/validate") {
      throw new TypeError("connection reset during upload");
    }
    if (url === "/api/health") {
      return response(500, { ok: false, error: "unhealthy" });
    }
    throw new TypeError(`unexpected fetch: ${url}`);
  });

  await ui.runScenario({ currentTarget: ui.dom.runButton });

  assert.deepEqual(calls.map((item) => `${item.method} ${item.url}`), [
    "POST /api/validate",
    "GET /api/health",
  ]);
  assert.equal(ui.operationErrors.at(-1)?.code, "network_error");
  assert.equal(ui.state.connection.status, "offline");
  assert.equal(ui.dom.runJobDialog.open, false);
  assert.equal(ui.state.busy, false);
});


test("V4 stale or incomplete previous placement never blocks a new validated run", async () => {
  const calls = [];
  const ui = loadRunHarness(async (url, options) => {
    calls.push({ url, payload: JSON.parse(options.body) });
    if (url === "/api/validate") return response(200, { valid: true, errors: [], mapping_stale: false });
    if (url === "/api/run-estimate") return response(200, estimatePayload());
    if (url === "/api/run-jobs") return response(202, { job_id: "fresh-job", status: "running" });
    throw new Error(`unexpected ${url}`);
  });
  ui.state.mappingStale = true;
  ui.state.mappingInputFingerprint = "old";
  ui.state.currentInputFingerprint = "edited";
  ui.state.scenario.placement.metadata = {
    retained_note: "preserve custom metadata",
    control_plane: {
      policy: { options: { objective: "latency" } },
      decision: { fully_placed: false, generated_op_keys: ["old-op"] },
      evidence: { input_fingerprint: "old" },
    },
};
  await ui.runScenario();
  assert.equal(ui.dom.runJobDialog.open, true);
  await ui.startRunJob();
  assert.deepEqual(calls.map(c => c.url), ["/api/validate", "/api/run-estimate", "/api/run-jobs"]);
  for (const { payload } of calls) {
    const authoring = payload.scenario || payload;
    assert.deepEqual(Object.keys(authoring.placement.metadata.control_plane), ["policy"]);
    assert.equal(authoring.placement.metadata.retained_note, "preserve custom metadata");
    assert.deepEqual(authoring.placement.op_to_component, {});
  }
  assert.equal(calls[2].payload.retention_policy, estimatePayload().recommended_retention_policy);
  assert.equal(ui.state.scenario.placement.metadata.control_plane.evidence.input_fingerprint, "old");
  assert.equal(ui.operationErrors.length, 0);
});

test("local authoring errors are diagnosed and leave run retryable without unhandled rejection", async () => {
  let requests = 0;
  const ui = loadRunHarness(async () => { requests++; throw new Error("must not send"); });
  ui.state.scenario.profiles.host_orchestration.cpu_component_id = "missing-cpu";
  await assert.doesNotReject(ui.runScenario());
  assert.equal(requests, 0);
  assert.equal(ui.operationErrors.length, 1);
  assert.equal(ui.state.busy, false);
  assert.equal(ui.dom.runButton.disabled, false);
  assert.equal(ui.dom.runJobDialog.open, false);
});

test("V4 run still stops on backend validation errors before estimate or submission", async () => {
  const calls = [];
  const ui = loadRunHarness(async (url) => {
    calls.push(url);
    return response(200, { valid: false, errors: ["invalid capacity"] });
  });
  ui.state.mappingStale = true;
  await ui.runScenario();
  assert.deepEqual(calls, ["/api/validate"]);
  assert.equal(ui.dom.runJobDialog.open, false);
  assert.equal(ui.state.validation.errors.length, 1);
});

function runtimeReport() {
  return {
    manifest: { run_id: "contract-test" },
    requests: {},
    runtime_placement: {
      schema_version: "runtime-placement/v1",
      read_only: true,
      parallel: { rank_mapping: [{ rank: 0, component_id: "actual-gpu", memory_component_id: "actual-hbm" }] },
      control_plane: {
        decision: { fully_placed: true, operator_execution_targets: { "layer0.qkv": [{ rank: 0, component_id: "actual-gpu" }] }, rank_weight_shards: { weight: [{ rank: 0 }] } },
        evidence: { input_fingerprint: "runtime-fingerprint" },
      },
    },
  };
}

test("completed runtime placement populates read-only views without polluting V4 authoring", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  ui.state.scenario = ui.scenarioPayloadForTransport();
  ui.state.mappingStale = true;
  ui.state.runJobScenarioGeneration = ui.state.scenarioGeneration;
  ui.state.runJobMappingGeneration = ui.state.mappingGeneration;
  ui.finishRunJob({ job_id: "completed", status: "completed", report: runtimeReport() });
  assert.equal(ui.state.reportStale, false);
  assert.equal(ui.state.mappingStale, false);
  assert.equal(ui.controlPlaneDecision(ui.state.scenario.placement).fully_placed, true);
  assert.equal(ui.controlPlaneEvidence(ui.state.scenario.placement).input_fingerprint, "runtime-fingerprint");
  assert.equal(ui.effectiveParallelRanks()[0].component_id, "actual-gpu");
  const payload = ui.scenarioPayloadForTransport();
  assert.equal(payload.runtime_placement, undefined);
  assert.equal(payload.placement.metadata.control_plane, undefined);
  assert.deepEqual(Object.keys(payload.placement.op_to_component), []);
  assert.deepEqual(Object.keys(payload.placement.tensor_to_component), []);
});

test("background completion after edits is stale and never replaces the current mapping view", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  ui.state.scenario = ui.scenarioPayloadForTransport();
  ui.state.runJobScenarioGeneration = ui.state.scenarioGeneration;
  ui.state.runJobMappingGeneration = ui.state.mappingGeneration;
  ui.state.scenarioGeneration++;
  ui.state.mappingStale = true;
  ui.finishRunJob({ job_id: "old-job", status: "completed", report: runtimeReport() });
  assert.equal(ui.state.reportStale, true);
  assert.equal(ui.state.mappingStale, true);
  assert.deepEqual(Object.keys(ui.controlPlaneDecision(ui.state.scenario.placement)), []);
});

test("GPU comparison cannot overwrite an active background job", async () => {
  let requests = 0;
  const ui = loadRunHarness(async () => { requests++; throw new Error("must not send"); });
  ui.state.runJob = { job_id: "still-running", status: "running" };
  await ui.compareScenario();
  assert.equal(requests, 0);
  assert.equal(ui.state.runJob.job_id, "still-running");
});

test("comparison ignores obsolete responses after scenario edits", async () => {
  let ui;
  ui = loadRunHarness(async () => {
    ui.state.scenarioGeneration++;
    return response(200, { candidate: runtimeReport() });
  });
  await ui.compareScenario();
  assert.equal(ui.state.report, null);
  assert.equal(ui.state.comparison, null);
  assert.equal(ui.state.busy, false);
});

test("validation navigation resolves structured paths and plain backend messages", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  const structured = ui.validationIssueTarget({ field_path: "hardware.components[1].metadata.read_latency_ns" });
  assert.equal(structured.componentId, "hbm0");
  assert.equal(structured.field, "read_latency_ns");
  const inferred = ui.validationIssueTarget(ui.normalizeIssue?.("component gpu0 capacity is missing", "scenario", "error") || {
    message: "component gpu0 capacity is missing",
  });
  assert.equal(inferred.componentId, "gpu0");
  assert.equal(inferred.field, "capacity_bytes");
});

test("validation cursor advances after the current error disappears", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  const first = { code: "first", field_path: "hardware.components[0].capacity_bytes", message: "first" };
  const second = { code: "second", field_path: "hardware.components[1].read_bandwidth_gbps", message: "second" };
  ui.beginValidationNavigation([first, second]);
  assert.equal(ui.state.validationNavigation.index, 0);
  ui.advanceValidationNavigation([second]);
  assert.equal(ui.state.validationNavigation.index, 0);
  assert.equal(ui.state.validationNavigation.signature.includes("second"), true);
});

test("loading a new scenario retires the old validation cursor", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  ui.state.validationNavigation = { active: true, errors: [{ code: "old" }], index: 0, signature: "old" };
  ui.setScenario?.(scenario(), { dirty: false });
  assert.equal(ui.state.validationNavigation, null);
});


test("accepting fresh placement retires imported old decisions rather than reviving them on reload", () => {
  const ui = loadRunHarness(async () => { throw new Error("no request"); });
  ui.state.scenario = ui.scenarioPayloadForTransport();
  ui.state.scenario.placement.metadata.control_plane = {
    policy: { objective: "balanced" },
    decision: { fully_placed: false },
    evidence: { input_fingerprint: "obsolete" },
  };
  ui.acceptRuntimePlacement(runtimeReport());
  assert.deepEqual(Object.keys(ui.state.scenario.placement.metadata.control_plane), ["policy"]);
  assert.equal(ui.state.scenario.placement.metadata.control_plane.policy.objective, "balanced");
});
