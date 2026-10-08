const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const core = require(path.join(root, "src/heterollm_sim/webui/model-graph-core.js"));

function loadApp({ scheduleTimer = setTimeout, cancelTimer = clearTimeout } = {}) {
  const source = fs.readFileSync(path.join(root, "src/heterollm_sim/webui/app.js"), "utf8");
  const storedValues = new Map();
  const context = {
    console,
    document: { addEventListener() {} },
    TopologyCore: {},
    ModelGraphCore: core,
    TraceViewCore: {},
    localStorage: {
      getItem(key) { return storedValues.has(key) ? storedValues.get(key) : null; },
      setItem(key, value) { storedValues.set(key, String(value)); },
      removeItem(key) { storedValues.delete(key); },
    },
    window: { addEventListener() {}, setTimeout: scheduleTimer, clearTimeout: cancelTimer },
    setTimeout: scheduleTimer,
    clearTimeout: cancelTimer,
    URL,
    JSON,
    Math,
    Number,
    String,
    Object,
    Array,
    Map,
    Set,
    Date,
    RegExp,
    Promise,
    Intl,
    navigator: {},
  };
  context.globalThis = context;
  const suffix = `
    globalThis.__webuiTest = {
      state,
      dom,
      addRequest,
      updateRequestField,
      renderWorkload,
      applyWorkloadPreset,
      workloadPresets: WORKLOAD_PRESETS,
      setWorkloadPresetCatalog(items) { workloadPresetCatalog = items; },
      setLlamaRuntimeMode,
      updateLlamaRuntimeField,
      markScenarioChanged,
      renderRunJobDialog,
      renderRunStepSummary,
      setStubs(changed, notices) {
        markWorkloadChanged = changed;
        toast = (...args) => notices.push(args);
        renderRequestTable = () => {};
        hydrateConceptHelp = () => {};
      },
      setScenarioChangedStub(changed) {
        markScenarioChanged = changed;
      },
      stubScenarioRendering() {
        resetTracePlaybackState = () => {};
        captureRenderInteractionState = () => null;
        restoreRenderInteractionState = () => {};
        renderAll = () => {};
        markMappingStale = () => {};
      },
      importScenarioFile,
      runScenario,
      setScenario,
      setBusy,
      syncRunButtons,
      scheduleValidationNavigationRecheck,
      stubImportRunRendering({ request, events }) {
        apiRequest = request;
        clearValidationFocus = () => {};
        resetTracePlaybackState = () => {};
        resetTopologyHistory = () => {};
        renderAll = () => {};
        renderSteps = () => {};
        renderDiagnostics = () => {};
        openDiagnostics = () => {};
        Topology.normalizeTopologyView = () => ({ layout: { positions: {} }, viewport: {} });
        toast = (...args) => events.push({ type: "toast", args });
        showOperationError = (...args) => events.push({ type: "error", args });
        renderRunJobDialog = () => {};
        openRunJobDialog = () => events.push({ type: "dialog" });
      },
      resetPlacementForArchitecturePreset,
      resetArchitectureDependentProfiles,
      applyArchitecturePresetDetail,
      architecturePresetDetailItem,
      travelTopologyHistory,
      loadArchitectureFromPreset,
      hostOrchestrationProfileDraft,
      remapPresetProfileResources,
      stubArchitectureHistoryRendering({ failFirstSave = false } = {}) {
        architecturePresetIsLoadable = () => true;
        collisionSafeArchitectureTopologyView = () => ({
          view: { layout: { positions: {}, bounds: {} }, viewport: { x: 0, y: 0, scale: 1 } },
          adjusted: false,
        });
        Topology.normalizeTopologyView = (view) => view;
        commitPendingGroupLabel = () => {};
        globalThis.requestAnimationFrame = () => {};
        toast = () => {};
        let saves = 0;
        saveTopologyView = () => {
          saves += 1;
          if (failFirstSave && saves === 1) throw new Error("save failure after hardware replacement");
        };
      },
      setArchitectureLoadStubs({ detail, apply, confirm, notices }) {
        architecturePresetDetail = detail;
        architecturePresetIsLoadable = () => true;
        applyArchitecturePresetDetail = apply;
        architecturePresetNames = () => ({ zh: "test preset" });
        globalThis.confirm = confirm;
        renderArchitecturePresets = () => {};
        showOperationError = (...args) => notices.push(args);
        toast = (...args) => notices.push(args);
      },
      applyPresetDetailToScenario,
      modelPresetCardMarkup,
      invalidateModelStructureBindings,
      openJsonDialog,
      exportScenario,
      exportHardwareInput,
      setOperationErrorStub(notices) {
        showOperationError = (...args) => notices.push(args);
      },
      validatePhysicalMemoryConfig,
      physicalMemoryContractIssues,
      hardwareInputForScenario,
      scenarioPayloadForTransport,
      restoreStoredScenario,
      restoreScenarioFromStorage,
      initializeEmptyWorkspace,
      createEmptyScenario,
      persistCurrentScenario,
      missingScenarioInputIssues,
      currentProtocolConnectionDefaults,
      storageScenarioKey: STORAGE_SCENARIO,
      protocolInputs: dom,
      localStorage,
      setBootstrapStubs({ setScenario: setScenarioStub, loadReference: loadReferenceStub, notices }) {
        setScenario = setScenarioStub;
        loadReference = loadReferenceStub;
        toast = (...args) => notices.push(args);
      },
    };
  `;
  vm.createContext(context);
  vm.runInContext(source + suffix, context, { timeout: 30_000 });
  return context.__webuiTest;
}

function workload(prompt_tokens = 512, output_tokens = 128) {
  return {
    name: "test",
    requests: [],
    request_count: 1,
    prompt_tokens,
    output_tokens,
    arrival_rate_rps: 0,
    random_seed: 0,
    scheduler: {
      mode: "continuous",
      max_num_seqs: 1,
      max_num_batched_tokens: 512,
      prefill_chunk_tokens: 512,
      preemption_enabled: false,
    },
    mtp: null,
    metadata: {},
  };
}

function graphWithPorts(firstContract, secondContract) {
  return {
    graph_id: "contract-test",
    tensors: [{
      tensor_id: "x",
      role: "activation",
      producer_operator_id: "source",
      consumer_operator_ids: ["sink"],
    }],
    operators: [
      {
        operator_id: "source",
        op_kind: "producer",
        ports: [{ port_id: "out", direction: "output", tensor_id: "x", ...firstContract }],
      },
      {
        operator_id: "sink",
        op_kind: "consumer",
        ports: [{ port_id: "in", direction: "input", tensor_id: "x", ...secondContract }],
      },
    ],
  };
}

function importScenario(name) {
  return {
    schema_version: "4.0.0",
    name,
    hardware: {
      name: "import-test-hardware",
      components: [
        { component_id: "cpu0", kind: "cpu", cost_profile_id: "cpu-profile", metadata: {}, ports: [] },
        { component_id: "gpu0", kind: "gpu", cost_profile_id: "gpu-profile", metadata: {}, ports: [] },
      ],
      links: [],
      metadata: {},
    },
    model: { name: "import-test-model", graph: graphWithPorts({}, {}) },
    placement: { metadata: {}, parallel: {} },
    workload: workload(),
    profiles: {
      components: { cpu: { "cpu-profile": {} }, gpu: { "gpu-profile": {} } },
      host_orchestration: {
        cpu_component_id: "cpu0", gpu_component_id: "gpu0",
        scheduler_resource_id: "host.scheduler", pack_resource_id: "host.pack",
        dma_resource_id: "host.dma", submission_resource_id: "host.submit",
      },
      fusion: {},
      runtime: { gpu_controllers: { gpu0: {} } },
    },
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

function prepareImportRun(api, request) {
  const events = [];
  for (const id of [
    "busyOverlay", "busyTitle", "busyDetail", "runButton", "rerunButton", "emptyRunButton",
    "traceRunButton", "importButton", "importHardwareButton", "fileInput", "hardwareFileInput",
  ]) api.dom[id] = { disabled: false, hidden: true, value: "selected.json" };
  api.stubImportRunRendering({ request, events });
  api.setScenario(importScenario("old-scenario"));
  return events;
}

test("scenario import blocks runs through file reading and normalization, then one run uses the imported input", async () => {
  const api = loadApp();
  const fileRead = deferred();
  const normalization = deferred();
  const requests = [];
  const events = prepareImportRun(api, (route, options) => {
    requests.push({ route, name: JSON.parse(options.body).name });
    if (route === "/normalize") return normalization.promise;
    if (route === "/validate") return Promise.resolve({ valid: true, errors: [], warnings: [] });
    if (route === "/run-estimate") return Promise.resolve({ risk_level: "low" });
    throw new Error(`Unexpected route: ${route}`);
  });
  const importTask = api.importScenarioFile({ text: () => fileRead.promise });
  for (const phase of ["reading", "normalizing"]) {
    assert.equal(api.state.busy, true, phase);
    assert.equal(api.dom.busyOverlay.hidden, false, phase);
    api.syncRunButtons();
    for (const id of ["runButton", "rerunButton", "emptyRunButton", "traceRunButton", "importButton", "importHardwareButton"]) {
      assert.equal(api.dom[id].disabled, true, `${id} while ${phase}`);
    }
    await api.runScenario();
    assert.equal(requests.some(({ route }) => route === "/validate"), false);
    assert.equal(api.state.scenario.name, "old-scenario");
    if (phase === "reading") {
      fileRead.resolve(JSON.stringify(importScenario("new-scenario")));
      await new Promise(setImmediate);
    }
  }
  normalization.resolve({ scenario: importScenario("new-scenario") });
  await importTask;
  assert.equal(api.state.busy, false);
  assert.equal(api.dom.busyOverlay.hidden, true);
  assert.equal(api.dom.runButton.disabled, false);
  assert.equal(api.dom.importButton.disabled, false);
  assert.equal(api.state.scenario.name, "new-scenario");
  await api.runScenario();
  assert.deepEqual(requests, [
    { route: "/normalize", name: "new-scenario" },
    { route: "/validate", name: "new-scenario" },
    { route: "/run-estimate", name: "new-scenario" },
  ]);
  assert.equal(events.filter(({ type }) => type === "dialog").length, 1);
  assert.equal(events.filter(({ type }) => type === "error").length, 0);
});

test("overlapping and externally busy imports do not read files or release another operation's busy state", async () => {
  const api = loadApp();
  const normalization = deferred();
  const requests = [];
  prepareImportRun(api, (route) => { requests.push(route); return normalization.promise; });
  let rejectedFileReads = 0;
  const rejectedFile = { text() { rejectedFileReads += 1; return Promise.resolve("{}"); } };
  api.setBusy(true);
  await api.importScenarioFile(rejectedFile);
  assert.equal(api.state.busy, true);
  assert.equal(api.dom.busyOverlay.hidden, false);
  assert.equal(rejectedFileReads, 0);
  api.setBusy(false);
  const importTask = api.importScenarioFile({ text: async () => JSON.stringify(importScenario("new-scenario")) });
  await new Promise(setImmediate);
  await api.importScenarioFile(rejectedFile);
  assert.equal(api.state.busy, true);
  assert.equal(api.dom.runButton.disabled, true);
  assert.equal(rejectedFileReads, 0);
  assert.deepEqual(requests, ["/normalize"]);
  assert.equal(api.state.scenario.name, "old-scenario");
  normalization.resolve(importScenario("new-scenario"));
  await importTask;
  assert.equal(api.state.busy, false);
  assert.equal(api.state.scenario.name, "new-scenario");
});

test("failed file reads, JSON parsing and normalization release import busy state and allow retry", async () => {
  for (const failure of ["read", "json", "normalize"]) {
    const api = loadApp();
    let failNormalize = failure === "normalize";
    const events = prepareImportRun(api, async (_route, options) => {
      if (failNormalize) throw new Error("normalization failed");
      return JSON.parse(options.body);
    });
    const oldScenario = api.state.scenario;
    await api.importScenarioFile({ text: async () => {
      if (failure === "read") throw new Error("file read failed");
      return failure === "json" ? "invalid JSON" : JSON.stringify(importScenario("new-scenario"));
    } });
    assert.equal(api.state.busy, false, failure);
    assert.equal(api.dom.busyOverlay.hidden, true, failure);
    assert.equal(api.dom.runButton.disabled, false, failure);
    assert.equal(api.dom.fileInput.value, "", failure);
    assert.equal(api.dom.hardwareFileInput.value, "", failure);
    assert.equal(api.state.scenario, oldScenario, failure);
    assert.equal(events.some(({ type, args }) => type === "toast" && args[2] === "error"), true, failure);
    failNormalize = false;
    await api.importScenarioFile({ text: async () => JSON.stringify(importScenario("retry-scenario")) });
    assert.equal(api.state.scenario.name, "retry-scenario", failure);
    assert.equal(api.state.busy, false, failure);
  }
});

test("a queued validation recheck waits for a busy import and resumes after import failure", async () => {
  const timers = new Map();
  let nextTimer = 0;
  const api = loadApp({
    scheduleTimer(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    cancelTimer(id) { timers.delete(id); },
  });
  const fireNextTimer = () => {
    assert.equal(timers.size, 1);
    const [id, callback] = timers.entries().next().value;
    timers.delete(id);
    callback();
  };
  const normalization = deferred();
  const requests = [];
  prepareImportRun(api, (route) => {
    requests.push(route);
    return route === "/normalize" ? normalization.promise : Promise.resolve({ valid: true, errors: [] });
  });
  api.state.validationNavigation = { active: true, pendingGeneration: 0, recheckTimer: null };
  api.scheduleValidationNavigationRecheck();
  const importTask = api.importScenarioFile({ text: async () => JSON.stringify(importScenario("new-scenario")) });
  await new Promise(setImmediate);
  fireNextTimer();
  assert.deepEqual(requests, ["/normalize"]);
  assert.equal(api.state.busy, true);
  assert.equal(api.state.validationNavigation.pendingGeneration, api.state.scenarioGeneration);
  normalization.reject(new Error("normalization failed"));
  await importTask;
  assert.equal(api.state.busy, false);
  assert.equal(api.state.scenario.name, "old-scenario");
  fireNextTimer();
  await new Promise(setImmediate);
  assert.deepEqual(requests, ["/normalize", "/validate"]);
  assert.equal(api.state.busy, false);
  assert.equal(timers.size, 0);
});

test("model graph rejects conflicting port contracts when tensor contract is absent", () => {
  for (const [field, first, second] of [
    ["dtype", "fp16", "int64"],
    ["shape", ["B", "T", "H"], ["B", "T"]],
    ["layout", "logical", "row_major"],
  ]) {
    const graph = graphWithPorts({ [field]: first }, { [field]: second });
    const result = core.validateModelGraph(graph);
    assert.equal(result.valid, false, `${field} conflict should fail closed`);
    assert.match(result.errors.join("\n"), new RegExp(`契约 ${field}`));
  }
});

test("model graph still accepts matching port contracts and fills the omitted tensor contract", () => {
  const graph = graphWithPorts(
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
  );
  assert.equal(core.validateModelGraph(graph).valid, true);
  const normalized = core.normalizeModelGraph(graph);
  assert.deepEqual(normalized.tensors[0].dtype, "fp16");
  assert.deepEqual(normalized.tensors[0].shape, ["B", "T", "H"]);
});

test("new explicit request inherits current synthetic token defaults", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload(2048, 37) };
  let changed = 0;
  api.setStubs(() => { changed += 1; }, []);
  api.addRequest();
  assert.equal(api.state.scenario.workload.requests[0].prompt_tokens, 2048);
  assert.equal(api.state.scenario.workload.requests[0].output_tokens, 37);
  assert.equal(changed, 1);
});

test("blank workload number is rejected and arrival rate is exposed", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload() };
  const notices = [];
  let changed = 0;
  const form = {
    innerHTML: "",
    querySelector(selector) {
      if (selector === "#workloadPresetSelect") return { addEventListener() {} };
      if (selector === "#mtpEnabledInput") return { addEventListener() {} };
      return null;
    },
    querySelectorAll(selector) {
      return selector === "[data-workload-field]" ? [this.promptControl, this.arrivalControl] : [];
    },
    promptControl: {
      type: "number",
      value: "",
      dataset: { workloadField: "prompt_tokens" },
      addEventListener(type, handler) { this[type] = handler; },
    },
    arrivalControl: {
      type: "number",
      value: "2.5",
      dataset: { workloadField: "arrival_rate_rps" },
      addEventListener(type, handler) { this[type] = handler; },
    },
  };
  api.dom.workloadMetaForm = form;
  api.setStubs(() => { changed += 1; }, notices);
  api.renderWorkload();
  assert.match(form.innerHTML, /arrival_rate_rps/);
  form.arrivalControl.change();
  assert.equal(api.state.scenario.workload.arrival_rate_rps, 2.5);
  assert.equal(changed, 1);
  form.promptControl.change();
  assert.equal(api.state.scenario.workload.prompt_tokens, 512);
  assert.equal(changed, 1);
  assert.equal(notices.length, 1);
});

test("empty deadline remains nullable while invalid request numbers are rejected", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload() };
  api.state.scenario.workload.requests.push({ request_id: "r0", arrival_ns: 0, prompt_tokens: 8, output_tokens: 2, priority: 0, deadline_ns: 10 });
  const notices = [];
  let changed = 0;
  api.dom.requestTableBody = { innerHTML: "" };
  api.setStubs(() => { changed += 1; }, notices);
  api.updateRequestField(0, { type: "number", value: "", dataset: { requestField: "deadline_ns" } });
  assert.equal(api.state.scenario.workload.requests[0].deadline_ns, null);
  assert.equal(changed, 1);
  api.updateRequestField(0, { type: "number", value: "", dataset: { requestField: "prompt_tokens" } });
  assert.equal(api.state.scenario.workload.requests[0].prompt_tokens, 8);
  assert.equal(changed, 1);
  assert.equal(notices.length, 1);
});

test("llama runtime mode fills an unspecified KV owner from GPU-local memory", () => {
  const api = loadApp();
  api.state.scenario = {
    hardware: {
      components: [
        { component_id: "gpu0", kind: "gpu", metadata: {} },
        { component_id: "gddr0", kind: "gddr", metadata: {} },
        { component_id: "hostmem0", kind: "host_memory", metadata: {} },
      ],
      links: [
        { source_component: "gpu0", target_component: "gddr0" },
        { source_component: "gpu0", target_component: "hostmem0" },
      ],
    },
    placement: { kv_policy: { cache_component: null }, metadata: {} },
    profiles: {},
    workload: {
      requests: [{ prompt_tokens: 8, output_tokens: 2 }],
      prompt_tokens: 8,
      output_tokens: 2,
      scheduler: { max_num_seqs: 1, max_num_batched_tokens: 16, max_num_ubatch_tokens: 16 },
      metadata: {},
    },
  };
  assert.equal(api.setLlamaRuntimeMode("llama_cpp"), true);
  assert.equal(api.state.scenario.placement.kv_policy.cache_component, "gddr0");
  assert.equal(api.state.scenario.profiles.llama_cpp.policy, "llama_cpp");
});

test("llama.cpp workload preset refreshes stale runtime batch and context defaults", () => {
  const api = loadApp();
  api.setWorkloadPresetCatalog(api.workloadPresets);
  const previous = workload(64, 4);
  previous.requests = [{ request_id: "request-0000", prompt_tokens: 64, output_tokens: 4 }];
  previous.scheduler.max_num_seqs = 4;
  previous.scheduler.max_num_batched_tokens = 256;
  previous.scheduler.max_num_ubatch_tokens = 256;
  api.state.scenario = {
    workload: previous,
    profiles: {
      llama_cpp: {
        policy: "llama_cpp",
        batch: 256,
        ubatch: 256,
        context: 68,
        parallel: 4,
      },
    },
  };
  api.setStubs(() => {}, []);
  api.setScenarioChangedStub(() => {});
  api.applyWorkloadPreset("llama_cpp_default");
  assert.equal(api.state.scenario.workload.prompt_tokens, 512);
  assert.equal(api.state.scenario.workload.output_tokens, 128);
  assert.equal(api.state.scenario.workload.mtp, null);
  assert.equal(api.state.scenario.profiles.llama_cpp.batch, 512);
  assert.equal(api.state.scenario.profiles.llama_cpp.ubatch, 512);
  assert.equal(api.state.scenario.profiles.llama_cpp.context, 640);
  assert.equal(api.state.scenario.profiles.llama_cpp.parallel, 1);
});

test("llama.cpp workload preset preserves explicit runtime overrides", () => {
  const api = loadApp();
  api.setWorkloadPresetCatalog(api.workloadPresets);
  api.state.scenario = {
    workload: workload(64, 4),
    profiles: {
      llama_cpp: {
        policy: "llama_cpp",
        batch: 1024,
        ubatch: 128,
        context: 4096,
        parallel: 2,
      },
    },
  };
  api.setStubs(() => {}, []);
  api.setScenarioChangedStub(() => {});
  api.applyWorkloadPreset("llama_cpp_default");
  assert.equal(api.state.scenario.profiles.llama_cpp.batch, 1024);
  assert.equal(api.state.scenario.profiles.llama_cpp.ubatch, 128);
  assert.equal(api.state.scenario.profiles.llama_cpp.context, 4096);
  assert.equal(api.state.scenario.profiles.llama_cpp.parallel, 2);
});

test("run dialog scopes risk to logical scale and explains the first cohort count", () => {
  const api = loadApp();
  const element = () => ({ textContent: "", innerHTML: "", hidden: false, setAttribute() {}, removeAttribute() {} });
  for (const name of [
    "runJobDialog", "runEstimateRisk", "runEstimateSummary", "runEstimateWarnings",
    "runJobProgressPanel", "runJobStatus", "runProgressStage", "runProgressCount",
    "runProgressBar", "runProgressMessage", "startRunJobButton", "cancelRunJobButton",
    "dismissRunJobButton",
    "playbackStatus", "playbackCount", "resultsStatus", "resultsCount",
  ]) api.dom[name] = element();
  api.state.runEstimate = { schema_version: "4.0.0", risk_level: "low", risk_level_zh: "低", warnings: [] };
  api.state.runJob = {
    job_id: "test-job", status: "running",
    progress: { stage: "serving_cohorts", unit: "serving_batches", completed: 0, total: null },
  };
  api.renderRunJobDialog();
  assert.equal(api.dom.runEstimateRisk.textContent, "逻辑规模风险：低");
  assert.match(api.dom.runEstimateWarnings.innerHTML, /不保证运行耗时或峰值内存/);
  assert.match(api.dom.runProgressMessage.textContent, /完成前批次计数保持 0/);
  assert.match(api.dom.runProgressMessage.textContent, /详细事件可能耗时较长/);
  assert.equal(api.dom.playbackStatus.textContent, "运行中");
  assert.equal(api.dom.resultsStatus.textContent, "等待结果");
  api.state.runJob.progress.completed = 1;
  api.renderRunJobDialog();
  assert.doesNotMatch(api.dom.runProgressMessage.textContent, /完成前批次计数保持 0/);
});

test("sidebar run status follows submission and polling while preserving completed report semantics", () => {
  const api = loadApp();
  for (const name of ["playbackStatus", "playbackCount", "resultsStatus", "resultsCount"]) {
    api.dom[name] = { textContent: "", innerHTML: "" };
  }
  api.renderRunStepSummary();
  assert.equal(api.dom.playbackStatus.textContent, "未运行");
  api.state.report = { requests: { old: {} } };
  api.state.runJobSubmitting = true;
  api.renderRunStepSummary();
  assert.equal(api.dom.playbackStatus.textContent, "正在提交");
  assert.equal(api.dom.resultsStatus.textContent, "等待结果");
  assert.equal(api.dom.resultsCount.innerHTML, "—");
  api.state.runJobSubmitting = false;
  for (const [status, expected] of [["queued", "已排队"], ["running", "运行中"]]) {
    api.state.runJob = { status, job_id: "new" };
    api.renderRunStepSummary();
    assert.equal(api.dom.playbackStatus.textContent, expected);
    assert.equal(api.dom.playbackCount.textContent, "—");
    assert.equal(api.dom.resultsStatus.textContent, "等待结果");
  }
  api.state.runJob.cancellation_requested = true;
  api.renderRunStepSummary();
  assert.equal(api.dom.playbackStatus.textContent, "正在取消");
  api.state.runJob = { status: "completed", job_id: "new" };
  api.state.tracePlayback.batchTraceIndex = [{}, {}];
  api.renderRunStepSummary();
  assert.equal(api.dom.playbackStatus.textContent, "2 个批次摘要");
  assert.equal(api.dom.resultsStatus.textContent, "报告就绪");
  api.state.reportStale = true;
  api.renderRunStepSummary();
  assert.equal(api.dom.resultsStatus.textContent, "结果已过期");
});

test("architecture preset reset removes model-specific placement while preserving parallel policy", () => {
  const api = loadApp();
  const placement = {
    hardware_name: "old-hardware",
    op_to_component: { "layer-000": "gpu-old" },
    tensor_to_component: { hidden: "gddr-old" },
    tensor_bytes: { hidden: 128 },
    parallel: {
      tp_degree: 2,
      pp_degree: 3,
      ep_degree: 4,
      rank_mapping: [{ rank: 0, component_id: "gpu-old" }],
      layer_to_stage: { "layer-000": 0, "layer-001": 1 },
      collective_algorithm: "ring",
      routing_policy: "bandwidth_aware",
      allow_padding: false,
    },
    kv_policy: {
      cache_component: "gddr-old",
      offload_component: "host-old",
      dtype: "fp16",
      tokens_per_page: 32,
    },
    metadata: { control_plane: { decision: { old: true } }, ui: { keep: true }, keep: "yes" },
  };
  api.resetPlacementForArchitecturePreset(placement, "new-hardware");
  assert.equal(placement.hardware_name, "new-hardware");
  assert.equal(Object.keys(placement.op_to_component).length, 0);
  assert.equal(Object.keys(placement.tensor_to_component).length, 0);
  assert.equal(Object.keys(placement.tensor_bytes).length, 0);
  assert.equal(placement.parallel.rank_mapping.length, 0);
  assert.equal(Object.keys(placement.parallel.layer_to_stage).length, 0);
  assert.equal(placement.parallel.tp_degree, 2);
  assert.equal(placement.parallel.pp_degree, 3);
  assert.equal(placement.parallel.ep_degree, 4);
  assert.equal(placement.parallel.collective_algorithm, "ring");
  assert.equal(placement.parallel.routing_policy, "bandwidth_aware");
  assert.equal(placement.parallel.allow_padding, false);
  assert.equal(placement.kv_policy.cache_component, null);
  assert.equal(placement.kv_policy.offload_component, null);
  assert.equal(placement.kv_policy.dtype, "fp16");
  assert.equal(placement.kv_policy.tokens_per_page, 32);
  assert.equal(placement.metadata.control_plane, undefined);
  assert.equal(placement.metadata.ui.keep, true);
  assert.equal(placement.metadata.keep, "yes");
});

test("model preset replacement clears stale authoring maps and pipeline stages", () => {
  const api = loadApp();
  const graph = graphWithPorts(
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
  );
  graph.graph_id = "new-model-graph";
  graph.attributes = { ui: { detail: { collapsed_groups: ["source"] } } };
  api.state.scenario = {
    model: { name: "old-model", graph: { ...graph, graph_id: "old-model-graph" } },
    placement: {
      op_to_component: { old: "gpu0" },
      tensor_to_component: { old_tensor: "gddr0" },
      tensor_bytes: { old_tensor: 64 },
      parallel: {
        tp_degree: 2, pp_degree: 2, ep_degree: 1,
        rank_mapping: [{ rank: 0, component_id: "gpu0" }],
        layer_to_stage: { old: 0 },
        collective_algorithm: "ring", routing_policy: "bandwidth_aware", allow_padding: false,
      },
      metadata: { control_plane: { decision: { stale: true } } },
    },
  };
  let changed = 0;
  api.setScenarioChangedStub(() => { changed += 1; });
  const applied = api.applyPresetDetailToScenario({
    preset: { support_level: "exact" },
    model: { schema_version: "4.0.0", name: "new-model", graph },
  });
  assert.equal(applied.level, "exact");
  assert.equal(applied.removedByGroup.op_to_component, 1);
  assert.equal(applied.removedByGroup.tensor_to_component, 1);
  assert.equal(applied.removedByGroup.tensor_bytes, 1);
  assert.equal(applied.layerToStage.removed, 1);
  assert.equal(api.state.scenario.model.name, "new-model");
  assert.equal(Object.keys(api.state.scenario.placement.op_to_component).length, 0);
  assert.equal(Object.keys(api.state.scenario.placement.tensor_to_component).length, 0);
  assert.equal(Object.keys(api.state.scenario.placement.tensor_bytes).length, 0);
  assert.equal(Object.keys(api.state.scenario.placement.parallel.layer_to_stage).length, 0);
  assert.equal(api.state.scenario.placement.parallel.tp_degree, 2);
  assert.equal(api.state.scenario.placement.parallel.pp_degree, 2);
  assert.equal(api.state.scenario.placement.parallel.collective_algorithm, "ring");
  assert.equal(api.state.scenario.placement.metadata.control_plane, undefined);
  assert.equal(changed, 1);
  assert.equal(graph.attributes.ui.detail.collapsed_groups[0], "source");
});

test("DDR physical config validates and survives hardware input and scenario transport", () => {
  const api = loadApp();
  const physicalMemoryConfig = {
    kind: "DDR",
    channels: 2,
    subchannels_per_channel: 1,
    pseudo_channels_per_channel: 1,
    stacks: 1,
    dies_per_stack: 1,
    ranks_per_channel: 2,
    bank_groups_per_rank: 4,
    banks_per_group: 4,
    rows_per_bank: 32768,
    row_bytes: 8192,
    burst_bytes: 64,
    data_width_bits: 64,
    data_rate_mt_s: 5600,
    open_ns: 14,
    close_ns: 14,
    read_latency_ns: 40,
    write_latency_ns: 35,
    burst_interval_ns: 2.5,
    read_recovery_ns: 0,
    write_recovery_ns: 15,
    read_to_write_ns: 0,
    write_to_read_ns: 0,
    max_outstanding_requests: 64,
    capacity_bytes: 16 * 1024 ** 3,
    metadata: { source: "webui-ddr-regression" },
  };
  const hostMemory = {
    component_id: "hostmem0",
    kind: "host_memory",
    cost_profile_id: "hostmem-profile",
    capacity_bytes: physicalMemoryConfig.capacity_bytes,
    read_bandwidth_gbps: 716.8,
    write_bandwidth_gbps: 716.8,
    metadata: { physical_memory_config: physicalMemoryConfig },
    ports: [],
  };

  const validated = api.validatePhysicalMemoryConfig(physicalMemoryConfig, hostMemory);
  assert.equal(validated.kind, "DDR");
  assert.deepEqual(JSON.parse(JSON.stringify(validated)), physicalMemoryConfig);

  const graph = graphWithPorts(
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
  );
  api.state.scenario = {
    schema_version: "4.0.0",
    name: "ddr-transport-test",
    hardware: {
      name: "ddr-transport-hardware",
      components: [
        { component_id: "cpu0", kind: "cpu", cost_profile_id: "cpu-profile", metadata: {}, ports: [] },
        { component_id: "gpu0", kind: "gpu", cost_profile_id: "gpu-profile", metadata: {}, ports: [] },
        hostMemory,
      ],
      links: [],
      metadata: {},
    },
    model: { name: "ddr-transport-model", graph },
    placement: { metadata: {}, parallel: {} },
    workload: { requests: [] },
    profiles: {
      components: {
        cpu: { "cpu-profile": { pipeline: { core_count: 1 }, cache_hierarchy: { levels: [{}] } } },
        gpu: { "gpu-profile": { tensor_core: { supported_dtypes: ["fp16"] }, cache_hierarchy: { levels: [{}] } } },
        host_memory: { "hostmem-profile": {} },
      },
      host_orchestration: {
        cpu_component_id: "cpu0",
        gpu_component_id: "gpu0",
        scheduler_resource_id: "host.scheduler",
        pack_resource_id: "host.pack",
        dma_resource_id: "host.dma",
        submission_resource_id: "host.submit",
      },
      fusion: {},
      runtime: { gpu_controllers: { gpu0: {} } },
    },
  };

  const hardwareInput = api.hardwareInputForScenario();
  const hardwareInputHostMemory = hardwareInput.hardware.components.find((item) => item.component_id === "hostmem0");
  assert.deepEqual(
    JSON.parse(JSON.stringify(hardwareInputHostMemory.metadata.physical_memory_config)),
    physicalMemoryConfig,
  );

  const payload = api.scenarioPayloadForTransport();
  const transportedHostMemory = payload.hardware_input.hardware.components.find((item) => item.component_id === "hostmem0");
  assert.deepEqual(
    JSON.parse(JSON.stringify(transportedHostMemory.metadata.physical_memory_config)),
    physicalMemoryConfig,
  );
  assert.equal(api.physicalMemoryContractIssues().length, 0);
  assert.equal(Object.hasOwn(payload.workload, "request_count"), false);
  assert.equal(Object.hasOwn(payload.workload, "prompt_tokens"), false);
  assert.equal(Object.hasOwn(payload.workload.scheduler, "mode"), false);
  assert.equal(Object.hasOwn(payload.placement.parallel, "tp_degree"), false);

  delete hostMemory.metadata.physical_memory_config;
  const missingConfigIssues = api.physicalMemoryContractIssues();
  assert.equal(missingConfigIssues.length, 1);
  assert.match(missingConfigIssues[0].message_zh, /physical_memory_config/);
});

test("every DRAM and NAND component kind requires explicit physical memory configuration", () => {
  const api = loadApp();
  const componentKinds = [
    "hbm", "hbm_stack", "gddr", "gddr_memory", "dram", "ddr", "ddr_memory",
    "lpddr", "lpddr_memory", "host_memory", "cxl_memory", "memory",
    "hbf", "ssd", "high_io_ssd", "nvme",
  ];
  api.state.scenario = {
    hardware: {
      components: componentKinds.map((kind, index) => ({
        component_id: `${kind}-${index}`,
        kind,
        capacity_bytes: 1024,
        metadata: {},
      })),
    },
  };
  assert.equal(api.physicalMemoryContractIssues().length, componentKinds.length);
});

test("opening a workspace stays empty and preserves the previous draft until explicit restore", async () => {
  const api = loadApp();
  prepareImportRun(api, () => { throw new Error("empty startup must not request a reference scenario"); });
  const saved = api.localStorage.getItem(api.storageScenarioKey);
  api.dom.restoreScenarioButton = { hidden: true };

  api.initializeEmptyWorkspace();
  assert.equal(api.state.scenario.hardware.components.length, 0);
  assert.equal(api.state.scenario.model.graph.operators.length, 0);
  assert.equal(Object.keys(api.state.scenario.profiles).length, 0);
  assert.equal(api.state.scenario.workload.requests[0].prompt_tokens, 512);
  assert.equal(api.state.scenario.workload.requests[0].output_tokens, 128);
  assert.equal(api.dom.restoreScenarioButton.hidden, false);
  api.persistCurrentScenario(); // Empty canvas layout changes must not erase saved input.
  assert.equal(api.localStorage.getItem(api.storageScenarioKey), saved);
  api.syncRunButtons();
  assert.equal(api.dom.runButton.disabled, true);
  assert.deepEqual(Array.from(api.missingScenarioInputIssues(), (item) => item.code), ["hardware_missing", "model_missing"]);

  assert.equal(await api.restoreScenarioFromStorage(), true);
  assert.equal(api.state.scenario.name, "old-scenario");
  assert.equal(api.dom.restoreScenarioButton.hidden, true);
});

test("a fresh workspace does not load a reference or persist an untouched blank draft", async () => {
  const api = loadApp();
  prepareImportRun(api, () => { throw new Error("unexpected request"); });
  api.localStorage.removeItem(api.storageScenarioKey);
  api.dom.restoreScenarioButton = { hidden: false };
  api.initializeEmptyWorkspace();
  assert.equal(api.dom.restoreScenarioButton.hidden, true);
  assert.equal(await api.restoreScenarioFromStorage(), false);
  api.persistCurrentScenario();
  assert.equal(api.localStorage.getItem(api.storageScenarioKey), null);

  api.state.dirty = true;
  api.state.scenario.name = "user-edited-draft";
  api.persistCurrentScenario();
  assert.equal(JSON.parse(api.localStorage.getItem(api.storageScenarioKey)).name, "user-edited-draft");
});

test("blank workspace payload actions report construction errors and stop cleanly", () => {
  const api = loadApp();
  api.state.scenario = api.createEmptyScenario();
  api.dom.jsonEditor = { value: "", classList: { remove() {} } };
  const notices = [];
  api.setOperationErrorStub(notices);

  assert.doesNotThrow(() => api.openJsonDialog());
  assert.doesNotThrow(() => api.exportScenario());
  assert.doesNotThrow(() => api.exportHardwareInput());

  assert.deepEqual(notices.map(([title]) => title), [
    "JSON 场景生成失败",
    "场景导出失败",
    "硬件参数导出失败",
  ]);
  assert.ok(notices.every(([, error]) => error?.name === "Error" && error.message));
});

test("applying the real Qwen3.8 GGUF preset preserves operators, quantization, layer overrides, and unrelated inputs", () => {
  const api = loadApp();
  api.stubScenarioRendering();
  const presetModel = JSON.parse(fs.readFileSync(path.join(
    root,
    "src/heterollm_sim/model_preset_data/qwen3_8_27b_iq3_s_iq4_xs.json",
  ), "utf8"));
  const scenario = api.createEmptyScenario();
  scenario.hardware = {
    schema_version: "4.0.0",
    name: "preserved-hardware",
    components: [{ schema_version: "4.0.0", component_id: "gpu0", kind: "gpu", ports: [], metadata: { keep: true } }],
    links: [],
    metadata: { keep: true },
  };
  scenario.workload.requests = [{ request_id: "preserved-request", prompt_tokens: 23, output_tokens: 7, metadata: { keep: true } }];
  const originalHardware = JSON.parse(JSON.stringify(scenario.hardware));
  const originalRequests = JSON.parse(JSON.stringify(scenario.workload.requests));
  api.state.scenario = scenario;

  api.applyPresetDetailToScenario({
    preset: { id: "qwen3_8-27b-iq3-s-iq4-xs", support_level: "analytical_approximation" },
    model: presetModel,
  });

  const appliedGraph = api.state.scenario.model.graph;
  assert.equal(appliedGraph.operators.length, 229);
  assert.equal(appliedGraph.operators.some((operator) => operator.operator_id === "lm_head"), true);
  const layerGroups = appliedGraph.operators.filter((operator) => operator.op_kind === "layer_group");
  assert.equal(layerGroups.length, 32);
  const overriddenLayers = new Set(layerGroups.flatMap((operator) => Object.keys(operator.parameters.overrides || {})));
  assert.equal(overriddenLayers.size, 64);
  const quantizationTypes = new Set(layerGroups
    .flatMap((operator) => Object.values(operator.parameters.overrides || {}))
    .flatMap((override) => override.metadata?.gguf_tensor_bindings || [])
    .map((binding) => binding.type));
  assert.deepEqual(Array.from(quantizationTypes).sort(), ["F32", "IQ3_S", "IQ4_XS", "Q5_K"]);
  assert.equal(appliedGraph.attributes.metadata.gguf_physical_weight_bytes, 14854119424);
  assert.deepEqual(JSON.parse(JSON.stringify(api.state.scenario.hardware)), originalHardware);
  assert.deepEqual(JSON.parse(JSON.stringify(api.state.scenario.workload.requests)), originalRequests);
});

test("failed stored scenario load preserves the draft and does not replace it with reference", async () => {
  const api = loadApp();
  const notices = [];
  let referenceLoads = 0;
  api.localStorage.setItem(api.storageScenarioKey, JSON.stringify({ name: "unfinished-draft" }));
  api.setBootstrapStubs({
    setScenario() { throw new Error("physical_memory_config missing"); },
    loadReference() { referenceLoads += 1; },
    notices,
  });

  const loaded = await api.restoreStoredScenario({ name: "unfinished-draft" });

  assert.equal(loaded, false);
  assert.equal(api.localStorage.getItem(api.storageScenarioKey), JSON.stringify({ name: "unfinished-draft" }));
  assert.equal(referenceLoads, 0);
  assert.match(notices[0][1], /主动点击/);
});

test("malformed stored scenario is retained and does not load the reference scenario", async () => {
  const api = loadApp();
  const notices = [];
  let referenceLoads = 0;
  const malformed = '{"name":';
  api.localStorage.setItem(api.storageScenarioKey, malformed);
  api.setBootstrapStubs({
    setScenario() { throw new Error("should not be called"); },
    loadReference() { referenceLoads += 1; },
    notices,
  });

  await api.restoreScenarioFromStorage();

  assert.equal(api.localStorage.getItem(api.storageScenarioKey), malformed);
  assert.equal(referenceLoads, 0);
  assert.match(notices[0][1], /不是有效 JSON/);
});

test("invalid protocol connection fields are rejected without replacing the entered values", () => {
  const api = loadApp();
  Object.assign(api.protocolInputs, {
    protocolSelect: { value: "PCIe" },
    protocolVersionInput: { value: "5.0" },
    protocolUnitsInput: { value: "bad" },
    protocolBandwidthInput: { value: "32 GB/s" },
    protocolLatencyInput: { value: "150" },
    protocolPayloadInput: { value: "" },
  });

  assert.throws(() => api.currentProtocolConnectionDefaults(), /通道数必须/);
  assert.equal(api.protocolInputs.protocolUnitsInput.value, "bad");
  assert.equal(api.protocolInputs.protocolBandwidthInput.value, "32 GB/s");
});

test("host orchestration rebinding preserves explicit costs for validation", () => {
  const api = loadApp();
  const original = {
    capacity_fixed_instructions: 555,
    schedule_instructions_per_token: 19,
    command_build_instructions_per_invocation: 27,
    dma_queue_submission_ns: 13.5,
    admission_ns: 650,
    input_decode_ns_per_token: 2,
    output_encode_ns_per_token: 7,
    pinned_memory: false,
    request_parse_ns: -1,
    cpu_component_id: "old-cpu",
    gpu_component_id: "old-gpu",
    scheduler_resource_id: "old-cpu.scheduler",
    pack_resource_id: "old-cpu.pack",
    dma_resource_id: "old-cpu.h2d_dma",
    submission_resource_id: "old-gpu.command_queue",
  };
  const scenario = {
    hardware: { components: [
      { component_id: "cpu-next", kind: "cpu" },
      { component_id: "gpu-next", kind: "gpu" },
    ] },
    profiles: { host_orchestration: original },
  };
  const rebound = api.hostOrchestrationProfileDraft(scenario);
  for (const [field, value] of Object.entries(original)) {
    if (!field.endsWith("_id")) assert.equal(rebound[field], value, field);
  }
  assert.equal(rebound.cpu_component_id, "cpu-next");
  assert.equal(rebound.gpu_component_id, "gpu-next");
  assert.equal(rebound.scheduler_resource_id, "cpu-next.scheduler");
  assert.equal(rebound.submission_resource_id, "gpu-next.command_queue");
  assert.equal(original.cpu_component_id, "old-cpu");
});

test("hardware replacement rebuilds a complete host profile and isolates memory resources", () => {
  const api = loadApp();
  const component = (component_id, kind, profileKind, template) => ({
    component_id, kind,
    metadata: { cost_profile_key: profileKind, cost_profile_template: template },
  });
  const scenario = {
    hardware: { components: [
      component("cpu-next", "cpu", "cpu", { pipeline: { resource_id: "cpu-template.pipeline" } }),
      component("gpu-next", "gpu", "gpu", { tensor_core: { resource_id: "gpu-template.tensor_core" } }),
      component("hbm0", "hbm", "hbm", { resource_id: "hbm-template.hbm_fabric" }),
      component("hbm1", "hbm", "hbm", { resource_id: "hbm-template.hbm_fabric" }),
    ] },
    profiles: {
      components: {},
      host_orchestration: { capacity_fixed_instructions: 555, admission_ns: 999, cpu_component_id: "old-cpu" },
      fusion: { flash_attention: false },
      runtime: { gpu_controllers: { "old-gpu": { launch_ns: 5 } } },
    },
  };
  api.resetArchitectureDependentProfiles(scenario);
  const profile = scenario.profiles.host_orchestration;
  assert.equal(profile.capacity_fixed_instructions, 96);
  assert.equal(profile.capacity_instructions_per_request, 64);
  assert.equal(profile.schedule_fixed_instructions, 192);
  assert.equal(profile.schedule_instructions_per_request, 48);
  assert.equal(profile.schedule_instructions_per_token, 8);
  assert.equal(profile.command_build_fixed_instructions, 128);
  assert.equal(profile.command_build_instructions_per_invocation, 12);
  assert.equal(profile.dma_queue_submission_ns, 62.5);
  assert.equal(profile.admission_ns, 0);
  assert.equal(profile.input_decode_ns_per_token, 0);
  assert.equal(profile.output_encode_ns_per_token, 0);
  assert.equal(profile.cpu_component_id, "cpu-next");
  assert.equal(profile.gpu_component_id, "gpu-next");
  assert.equal(scenario.profiles.fusion.flash_attention, false);
  const resources = Object.values(scenario.profiles.components.hbm).map((entry) => entry.resource_id).sort();
  assert.deepEqual(resources, ["hbm0.hbm_fabric", "hbm1.hbm_fabric"]);
  assert.equal(scenario.profiles.runtime.gpu_controllers["gpu-next"].launch_ns, 5);
});

test("llama kernel selection survives the actual scenario-changed and runtime-edit paths", () => {
  const api = loadApp();
  api.stubScenarioRendering();
  api.state.scenario = {
    hardware: { components: [{
      component_id: "gpu0", kind: "gpu", metadata: { component_preset_id: "nvidia-rtx-5080" },
    }] },
    profiles: {},
    placement: { metadata: {} },
    workload: {
      requests: [{ prompt_tokens: 512, output_tokens: 128 }],
      scheduler: { max_num_seqs: 1, max_num_batched_tokens: 512 },
      metadata: {},
    },
  };
  api.setLlamaRuntimeMode("llama_cpp");
  api.state.scenario.workload.metadata.llama_cpp_runtime_fingerprint = "previous-run";
  api.markScenarioChanged();
  assert.equal(api.state.scenario.workload.metadata.llama_cpp_kernel_model_preset, "blackwell_analytical_v1");
  assert.equal(api.state.scenario.workload.metadata.llama_cpp_runtime_fingerprint, undefined);
  assert.equal(JSON.parse(api.localStorage.getItem(api.storageScenarioKey)).workload.metadata.llama_cpp_kernel_model_preset, "blackwell_analytical_v1");
  assert.equal(api.updateLlamaRuntimeField("context", 1024), true);
  api.markScenarioChanged();
  assert.equal(api.state.scenario.workload.metadata.llama_cpp_kernel_model_preset, "blackwell_analytical_v1");
  api.setLlamaRuntimeMode("auto");
  api.markScenarioChanged();
  assert.equal(api.state.scenario.workload.metadata.llama_cpp_kernel_model_preset, undefined);
});

test("explicit llama F32 hidden storage survives runtime edits and scenario persistence", () => {
  for (const enabled of [true, false]) {
    const api = loadApp();
    const storageContract = { schema: "llama.cpp.gguf.tensor-storage/v1", embedding_output_storage_bits: 32 };
    const ropeContract = { schema: "llama.cpp.cuda-rope/v1", strategy: "runtime_sin_cos" };
    api.stubScenarioRendering();
    api.state.scenario = {
      hardware: { components: [{ component_id: "gpu0", kind: "gpu", metadata: {} }] },
      profiles: {},
      placement: { metadata: {} },
      workload: {
        requests: [{ prompt_tokens: 512, output_tokens: 128 }],
        scheduler: { max_num_seqs: 1, max_num_batched_tokens: 512 },
        metadata: { llama_cpp_f32_hidden_storage: enabled, llama_cpp_tensor_storage_contract: storageContract, native_rope_source_contract: ropeContract },
      },
    };
    api.setLlamaRuntimeMode("llama_cpp");
    assert.equal(api.state.scenario.workload.metadata.llama_cpp_f32_hidden_storage, enabled);
    assert.equal(api.updateLlamaRuntimeField("context", 1024), true);
    api.state.scenario.workload.metadata.llama_cpp_runtime_fingerprint = "previous-run";
    api.markScenarioChanged();
    assert.equal(api.state.scenario.workload.metadata.llama_cpp_f32_hidden_storage, enabled);
    assert.deepEqual(api.state.scenario.workload.metadata.llama_cpp_tensor_storage_contract, storageContract);
    assert.deepEqual(api.state.scenario.workload.metadata.native_rope_source_contract, ropeContract);
    assert.equal(api.state.scenario.workload.metadata.llama_cpp_runtime_fingerprint, undefined);
    const saved = JSON.parse(api.localStorage.getItem(api.storageScenarioKey));
    assert.equal(saved.workload.metadata.llama_cpp_f32_hidden_storage, enabled);
    assert.deepEqual(saved.workload.metadata.llama_cpp_tensor_storage_contract, storageContract);
    assert.deepEqual(saved.workload.metadata.native_rope_source_contract, ropeContract);
    assert.equal(saved.workload.metadata.llama_cpp_runtime_fingerprint, undefined);
  }
});

test("hardware replacement selects a kernel for the new hardware while llama mode is active", () => {
  const api = loadApp();
  const gpu = {
    component_id: "gpu0", kind: "gpu",
    metadata: {
      component_preset_id: "nvidia-rtx-5080", cost_profile_key: "gpu",
      cost_profile_template: { launch_resource_id: "gpu0.frontend" },
    },
  };
  const scenario = {
    hardware: { components: [gpu] },
    profiles: { llama_cpp: { policy: "llama_cpp" } },
    workload: { metadata: {} },
  };
  api.resetArchitectureDependentProfiles(scenario);
  assert.equal(scenario.workload.metadata.llama_cpp_kernel_model_preset, "blackwell_analytical_v1");
  gpu.metadata.component_preset_id = "nvidia-b200";
  api.resetArchitectureDependentProfiles(scenario);
  assert.equal(scenario.workload.metadata.llama_cpp_kernel_model_preset, undefined);
});

test("architecture preset load restores its button after success, cancellation, and failure", async () => {
  for (const outcome of ["success", "cancel", "failure"]) {
    const api = loadApp();
    const notices = [];
    const button = { textContent: "Load hardware", disabled: false };
    let applied = 0;
    let closed = 0;
    api.state.scenario = { hardware: { components: [{}], links: [] } };
    api.dom.hardwarePresetsDialog = { close() { closed += 1; } };
    api.setArchitectureLoadStubs({
      async detail() {
        if (outcome === "failure") throw new Error("preset unavailable");
        return {};
      },
      apply() {
        applied += 1;
        return { hardware: { components: [], links: [] }, adjusted: false };
      },
      confirm() { return outcome !== "cancel"; },
      notices,
    });
    await api.loadArchitectureFromPreset("local-hardware", button);
    assert.equal(button.disabled, false, outcome);
    assert.equal(button.textContent, "Load hardware", outcome);
    assert.equal(applied, outcome === "success" ? 1 : 0, outcome);
    assert.equal(closed, outcome === "success" ? 1 : 0, outcome);
    assert.equal(notices.length, outcome === "cancel" ? 0 : 1, outcome);
  }
});

test("GGUF catalog keeps pending entries visible and exposes only binding until ready", () => {
  const api = loadApp();
  const base = { id: "pending-model", name: "Pending Model", support_level: "out_of_domain", generation_allowed: false, source_status: "pending_gguf" };
  const pending = api.modelPresetCardMarkup(base);
  assert.match(pending, /待补齐 GGUF/);
  assert.match(pending, /data-apply-preset="pending-model" disabled/);
  assert.match(pending, /data-bind-gguf="pending-model"/);
  assert.doesNotMatch(pending, /data-edit-gguf-preset/);
  const ready = api.modelPresetCardMarkup({ ...base, source_status: "gguf_ready", support_level: "analytical_approximation", generation_allowed: true, quantization: "MOSTLY_Q8_0" });
  assert.match(ready, /data-edit-gguf-preset="pending-model"/);
  assert.match(ready, /MOSTLY_Q8_0/);
  assert.doesNotMatch(ready, /data-apply-preset="pending-model" disabled/);
});

test("GGUF unsupported and non-generatable presets disable apply and edit", () => {
  const api = loadApp();
  const unsupported = api.modelPresetCardMarkup({
    id: "unsupported-model", name: "Unsupported Model", support_level: "analytical_approximation",
    generation_allowed: false, source_status: "gguf_unsupported", quantization: "MOSTLY_Q4_K_M",
  });
  assert.match(unsupported, /已有 GGUF，架构待适配/);
  assert.match(unsupported, /data-apply-preset="unsupported-model" disabled/);
  assert.match(unsupported, /data-edit-gguf-preset="unsupported-model" disabled/);
});

test("GGUF source facts show actual files and provenance as escaped text", () => {
  const api = loadApp();
  const markup = api.modelPresetCardMarkup({
    id: "source-model", name: "Source Model", support_level: "analytical_approximation",
    generation_allowed: true, source_status: "gguf_ready", quantization: "MOSTLY_Q8_0",
    gguf_source: {
      filename: "fallback.gguf", repo: "org/<model>", revision: "a".repeat(40), variant: "Instruct",
      files: [
        { filename: "weights-00001-of-00002.gguf", size_bytes: 12, sha256: "a".repeat(64), url: "https://example.test/1" },
        { filename: "weights-00002-of-00002.gguf", size_bytes: 34, sha256: "b".repeat(64), url: "https://example.test/2" },
      ],
    },
  });
  assert.match(markup, /weights-00001-of-00002\.gguf/);
  assert.match(markup, /weights-00002-of-00002\.gguf/);
  assert.match(markup, /2 个分片/);
  assert.match(markup, /org\/&lt;model&gt;/);
  assert.match(markup, new RegExp("a".repeat(40)));
  assert.match(markup, /Instruct/);
  assert.match(markup, /MOSTLY_Q8_0/);
  assert.doesNotMatch(markup, /fallback\.gguf/);
  assert.doesNotMatch(markup, /<model>/);
});

test("GGUF local source displays only filename provenance that exists", () => {
  const api = loadApp();
  const markup = api.modelPresetCardMarkup({
    id: "local-model", name: "Local Model", support_level: "analytical_approximation",
    generation_allowed: true, source_status: "gguf_derived", gguf_source: { filename: "local.gguf", sha256: "c".repeat(64) },
  });
  assert.match(markup, /local\.gguf/);
  assert.doesNotMatch(markup, /<dt>仓库<\/dt>/);
  assert.doesNotMatch(markup, /<dt>仓库修订<\/dt>/);
  assert.doesNotMatch(markup, /<dt>变体<\/dt>/);
});

test("model structure changes invalidate graph bindings without changing user runtime costs", () => {
  const api = loadApp();
  const scenario = {
    workload: { metadata: { cuda_graph_structural_program: { contract: "old" }, cuda_graph_comparison_request_id: "old", cuda_graph_experiment: {}, llama_cpp_gpu_native_invocations: {}, llama_cpp_f32_hidden_storage: false, user_note: "keep" } },
    profiles: { kernel: { graph_enabled: true, graph_launch_ns: 17 } },
  };
  api.invalidateModelStructureBindings(scenario);
  assert.deepEqual(scenario.workload.metadata, { llama_cpp_f32_hidden_storage: false, user_note: "keep" });
  assert.deepEqual(scenario.profiles.kernel, { graph_enabled: true, graph_launch_ns: 17 });
});

function historyHardware(preset) {
  return {
    schema_version: "4.0.0",
    name: preset,
    components: [{
      component_id: "gpu0", kind: "gpu", ports: [],
      metadata: {
        component_preset_id: preset,
        cost_profile_key: "gpu", cost_profile_template: { launch_resource_id: "gpu0.frontend" },
      },
    }],
    links: [], metadata: {},
  };
}

function prepareArchitectureHistory(api, preset, metadata = {}) {
  api.stubScenarioRendering();
  api.state.scenario = {
    hardware: historyHardware(preset),
    profiles: { llama_cpp: { policy: "llama_cpp" } },
    placement: { metadata: {} },
    workload: { metadata },
  };
  api.state.topologyView = { layout: { positions: {}, bounds: {} }, viewport: { x: 0, y: 0, scale: 1 } };
}

test("hardware presets supply initial runtime policies and preserve explicitly edited policies", () => {
  const api = loadApp();
  prepareArchitectureHistory(api, "nvidia-rtx-5080");
  api.stubArchitectureHistoryRendering();
  const profiles = {
    fusion: { flash_attention: true },
    runtime: { gpu_controllers: { gpu0: { command_processor: { command_submission_latency_ns: 5000 } } } },
  };
  const detail = api.architecturePresetDetailItem({ preset: { id: "nvidia-rtx-5080" }, hardware: historyHardware("nvidia-rtx-5080"), profiles });
  api.applyArchitecturePresetDetail(detail);
  assert.equal(api.state.scenario.profiles.fusion.flash_attention, true);
  assert.equal(api.state.scenario.profiles.runtime.gpu_controllers.gpu0.command_processor.command_submission_latency_ns, 5000);
  api.state.scenario.profiles.fusion.flash_attention = false;
  api.state.scenario.profiles.runtime.gpu_controllers.gpu0.command_processor.command_submission_latency_ns = 123;
  api.applyArchitecturePresetDetail(detail);
  assert.equal(api.state.scenario.profiles.fusion.flash_attention, false);
  assert.equal(api.state.scenario.profiles.runtime.gpu_controllers.gpu0.command_processor.command_submission_latency_ns, 123);
  assert.equal(profiles.fusion.flash_attention, true);
  assert.equal(profiles.runtime.gpu_controllers.gpu0.command_processor.command_submission_latency_ns, 5000);
});

test("hardware import undo and redo restore the matching kernel without reverting workload edits", () => {
  for (const originalPreset of ["nvidia-rtx-5080", "nvidia-b200"]) {
    for (const hiddenStorage of [undefined, false, true]) {
      const api = loadApp();
      prepareArchitectureHistory(api, originalPreset);
      api.stubArchitectureHistoryRendering();
      api.setLlamaRuntimeMode("llama_cpp");
      const replacement = originalPreset === "nvidia-rtx-5080" ? "nvidia-b200" : "nvidia-rtx-5080";
      api.applyArchitecturePresetDetail({ id: replacement, hardware: historyHardware(replacement) });
      // A workload edit after import must not be undone with the hardware.
      api.state.scenario.workload.metadata.user_note = "later workload edit";
      if (hiddenStorage !== undefined) api.state.scenario.workload.metadata.llama_cpp_f32_hidden_storage = hiddenStorage;
      api.markScenarioChanged();
      const assertState = (preset) => {
        const metadata = api.state.scenario.workload.metadata;
        assert.equal(api.state.scenario.hardware.components[0].metadata.component_preset_id, preset);
        assert.equal(metadata.llama_cpp_kernel_model_preset, preset === "nvidia-rtx-5080" ? "blackwell_analytical_v1" : undefined);
        assert.equal(Object.hasOwn(metadata, "llama_cpp_kernel_model_preset"), preset === "nvidia-rtx-5080");
        assert.equal(metadata.user_note, "later workload edit");
        assert.equal(metadata.llama_cpp_f32_hidden_storage, hiddenStorage);
        assert.equal(Object.hasOwn(metadata, "llama_cpp_f32_hidden_storage"), hiddenStorage !== undefined);
      };
      assert.equal(api.travelTopologyHistory("undo"), true);
      assertState(originalPreset);
      assert.equal(api.travelTopologyHistory("redo"), true);
      assertState(replacement);
    }
  }
});

test("failed hardware import rolls back the exact kernel field and preserves explicit author inputs", () => {
  for (const kernel of [undefined, false, true, "blackwell_analytical_v1"]) {
    const api = loadApp();
    const metadata = { llama_cpp_f32_hidden_storage: false, user_note: "original workload" };
    if (kernel !== undefined) metadata.llama_cpp_kernel_model_preset = kernel;
    prepareArchitectureHistory(api, "nvidia-b200", metadata);
    api.stubArchitectureHistoryRendering({ failFirstSave: true });
    assert.throws(() => api.applyArchitecturePresetDetail({
      id: "nvidia-rtx-5080", hardware: historyHardware("nvidia-rtx-5080"),
    }), /save failure after hardware replacement/);
    const restored = api.state.scenario.workload.metadata;
    assert.equal(api.state.scenario.hardware.components[0].metadata.component_preset_id, "nvidia-b200");
    assert.equal(restored.llama_cpp_kernel_model_preset, kernel);
    assert.equal(Object.hasOwn(restored, "llama_cpp_kernel_model_preset"), kernel !== undefined);
    assert.equal(restored.llama_cpp_f32_hidden_storage, false);
    assert.equal(restored.user_note, "original workload");
    assert.equal(api.state.topologyHistory.undo.length, 0);
  }
});
