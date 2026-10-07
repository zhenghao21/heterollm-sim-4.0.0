const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const core = require(path.join(root, "src/heterollm_sim/webui/model-graph-core.js"));

function loadApp() {
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
    window: { addEventListener() {}, setTimeout, clearTimeout },
    setTimeout,
    clearTimeout,
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
      renderRunJobDialog,
      setStubs(changed, notices) {
        markWorkloadChanged = changed;
        toast = (...args) => notices.push(args);
        renderRequestTable = () => {};
        hydrateConceptHelp = () => {};
      },
      setScenarioChangedStub(changed) {
        markScenarioChanged = changed;
      },
      resetPlacementForArchitecturePreset,
      applyPresetDetailToScenario,
      validatePhysicalMemoryConfig,
      physicalMemoryContractIssues,
      hardwareInputForScenario,
      scenarioPayloadForTransport,
      restoreStoredScenario,
      restoreScenarioFromStorage,
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
  api.state.runJob.progress.completed = 1;
  api.renderRunJobDialog();
  assert.doesNotMatch(api.dom.runProgressMessage.textContent, /完成前批次计数保持 0/);
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
