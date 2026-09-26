"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function helpers() {
  const context = vm.createContext({
    AbortController,
    Blob,
    CSS: { escape: String },
    Intl,
    ModelGraphCore,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    Promise,
    TopologyCore,
    TraceViewCore,
    URL,
    clearTimeout,
    console,
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) { callback(); },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__architecturePresets = {
    state,
    dom,
    ensureScenarioShape,
    scenarioPayloadForTransport,
    backendChineseMessage,
    architecturePresetDetailItem,
    architecturePresetNames,
    architecturePresetFidelity,
    architecturePresetIsLoadable,
    architecturePresetCardMarkup,
    architecturePresetDetailsMarkup,
    normalizeArchitecturePresetHardware,
    collisionSafeArchitectureTopologyView,
    resetPlacementForArchitecturePreset,
    materializeMissingCostProfiles,
    materializeComponentPreset,
    materializeTopologyBundle,
    costProfileDraft,
    resetArchitectureDependentProfiles,
    rebuildRuntimeGpuControllers,
    runtimeGpuControllerIssue,
    stableMappingEqual,
    renameComponent,
    hostOrchestrationReferenceIssue,
    componentKindClass,
    isActiveMemoryComponent,
    isWritableActiveRankMemory,
    kindLabel,
    componentInspectorProfile,
    applyArchitecturePresetDetail,
    travelTopologyHistory,
    configure() {
      renderAll = () => {};
      renderTopology = () => {};
      fitTopologyViewport = () => {};
      renderControlPlaneStatus = () => {};
      toast = () => {};
    },
  };`, context, { filename: path.join(webui, "app.js") });
  return context.__architecturePresets;
}

test("new active-memory component kinds are visible and usable by V4 placement UI", () => {
  const ui = helpers();
  const dram = {
    component_id: "dram0",
    kind: "dram",
    capacity_bytes: 32 * 1024 ** 3,
    read_bandwidth_gbps: 2048,
    write_bandwidth_gbps: 2048,
    metadata: { memory_service_owner: "dram0.controller" },
  };
  assert.equal(ui.componentKindClass(dram), "io");
  assert.equal(ui.isActiveMemoryComponent(dram), true);
  assert.equal(ui.isWritableActiveRankMemory(dram), true);
  assert.equal(ui.kindLabel("dram"), "堆叠 DRAM");
  assert.equal(ui.componentInspectorProfile("dram", dram).latencyDma, true);

  const hbfMemory = {
    component_id: "hbf0",
    kind: "hbf",
    metadata: { access_mode: "memory", write_buffer_bytes: 0 },
  };
  assert.equal(ui.isActiveMemoryComponent(hbfMemory), true);
  assert.equal(ui.isWritableActiveRankMemory(hbfMemory), true);
  assert.equal(ui.componentKindClass(hbfMemory), "io");
});

test("single component preset templates become an instance profile instead of reusing legacy calibration", () => {
  const ui = helpers();
  const scenario = {
    hardware: { components: [], links: [] },
    profiles: { components: { hbm: { "legacy-hbm": { bandwidth_gb_s: 1 } } } },
  };
  ui.state.scenario = scenario;
  const catalogComponent = {
    schema_version: "4.0.0",
    component_id: "jedec_hbm3",
    kind: "hbm",
    cost_profile_id: "legacy-hbm",
    read_bandwidth_gbps: 6553.6,
    metadata: {
      cost_profile_key: "hbm",
      cost_profile_template: {
        bandwidth_gb_s: 819.2,
        efficiency: 1,
        energy_pj_per_byte: 4,
        resource_id: "jedec_hbm3.hbm_fabric",
        read_latency_ns: 40,
        write_latency_ns: 40,
        transaction_bytes: 256,
        max_outstanding_requests: 32,
        read_bandwidth_gb_s: 819.2,
        write_bandwidth_gb_s: 819.2,
      },
    },
  };
  const component = ui.materializeComponentPreset({ id: "jedec-hbm3", component: catalogComponent });
  scenario.hardware.components.push(component);
  ui.materializeMissingCostProfiles([component], scenario);
  assert.notEqual(component.cost_profile_id, "legacy-hbm");
  const profile = ui.costProfileDraft("hbm", component);
  assert.equal(profile.read_latency_ns, 40);
  assert.equal(profile.bandwidth_gb_s, 819.2);
  assert.equal(profile.resource_id, "hbm0.hbm_fabric");
  const second = ui.materializeComponentPreset({ id: "jedec-hbm3", component: catalogComponent });
  scenario.hardware.components.push(second);
  ui.materializeMissingCostProfiles([second], scenario);
  assert.notEqual(second.cost_profile_id, component.cost_profile_id);
  assert.equal(ui.costProfileDraft("hbm", second).resource_id, "hbm1.hbm_fabric");
  scenario.profiles.components.hbm[component.cost_profile_id].read_latency_ns = 53;
  ui.materializeMissingCostProfiles([component], scenario);
  assert.equal(ui.costProfileDraft("hbm", component).read_latency_ns, 53, "repeat reconciliation preserves edits");
  assert.equal(ui.costProfileDraft("hbm", second).read_latency_ns, 40, "instances are independent");
  assert.equal(catalogComponent.cost_profile_id, "legacy-hbm");
  assert.equal(catalogComponent.metadata.cost_profile_template.read_latency_ns, 40);
  assert.equal(scenario.profiles.components.hbm["legacy-hbm"].bandwidth_gb_s, 1);
  component.metadata.cost_profile_template.read_latency_ns = 99;
  assert.equal(catalogComponent.metadata.cost_profile_template.read_latency_ns, 40, "catalog metadata is deeply cloned");
});

test("runtime GPU controller keys follow an architecture replacement without cloning across cardinalities", () => {
  const ui = helpers();
  const scenario = {
    hardware: { components: [{ component_id: "new-gpu", kind: "gpu" }] },
    profiles: { runtime: { gpu_controllers: { "old-gpu": { marker: "preserve" } } } },
  };
  assert.equal(ui.rebuildRuntimeGpuControllers(scenario), true);
  assert.deepEqual(JSON.parse(JSON.stringify(scenario.profiles.runtime.gpu_controllers)), { "new-gpu": { marker: "preserve" } });
  assert.equal(ui.runtimeGpuControllerIssue(scenario), "");

  scenario.hardware.components.push({ component_id: "second-gpu", kind: "gpu" });
  assert.equal(ui.rebuildRuntimeGpuControllers(scenario), false, "cardinality changes stay visible for explicit user repair");
  assert.match(ui.runtimeGpuControllerIssue(scenario), /second-gpu/u);
});

test("runtime controller rebuild reserves unchanged GPU IDs before mapping unmatched IDs", () => {
  const ui = helpers();
  const scenario = {
    hardware: { components: [{ component_id: "B", kind: "gpu" }, { component_id: "C", kind: "gpu" }] },
    profiles: { runtime: { gpu_controllers: { A: { marker: "A-controller" }, B: { marker: "B-controller" } } } },
  };
  assert.equal(ui.rebuildRuntimeGpuControllers(scenario), true);
  assert.deepEqual(JSON.parse(JSON.stringify(scenario.profiles.runtime.gpu_controllers)), {
    B: { marker: "B-controller" }, C: { marker: "A-controller" },
  });
});

test("runtime controller validation rejects missing runtime and uses canonical profile equality", () => {
  const ui = helpers();
  assert.equal(ui.stableMappingEqual({ a: 1, b: { c: 2 } }, { b: { c: 2 }, a: 1 }), true);
  const scenario = { hardware: { components: [{ component_id: "gpu0", kind: "gpu" }] }, profiles: {} };
  assert.match(ui.runtimeGpuControllerIssue(scenario), /profiles\.runtime/u);
  scenario.profiles.runtime = { gpu_controllers: { wrong: {} } };
  assert.match(ui.runtimeGpuControllerIssue(scenario), /gpu0/u);
});

test("renaming a GPU migrates its runtime controller key", () => {
  const ui = helpers();
  ui.state.scenario = {
    hardware: { components: [{ component_id: "gpu0", kind: "gpu" }], links: [] },
    profiles: { runtime: { gpu_controllers: { gpu0: { marker: "keep" } } }, host_orchestration: {} },
    placement: { kv_policy: {}, parallel: { rank_mapping: [] } },
  };
  ui.state.nodePositions = {};
  ui.state.nodeSizes = {};
  ui.state.topologyView = { groups: [], layout: { positions: {} } };
  ui.state.selectedComponents = new Set();
  ui.state.selected = { id: "gpu0" };
  ui.renameComponent(ui.state.scenario.hardware.components[0], "b200");
  assert.deepEqual(ui.state.scenario.profiles.runtime.gpu_controllers, { b200: { marker: "keep" } });
  assert.equal(ui.state.selected.id, "b200");
});

function scenario() {
  const model = {
    schema_version: "4.0.0",
    name: "kept-model",
  };
  model.graph = ModelGraphCore.buildModelGraphFromLayerSpecs([
    {
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
    },
  ], model);
  return {
    schema_version: "4.0.0",
    name: "preserve-me",
    assumptions: ["user assumption"],
    weights_resident: false,
    hardware: {
      schema_version: "4.0.0",
      name: "old-hardware",
      metadata: {},
      components: [
        { component_id: "old-gpu", kind: "gpu", ports: [{ port_id: "p0" }, { port_id: "host" }] },
        { component_id: "old-hbm", kind: "hbm", ports: [{ port_id: "p0" }] },
        { component_id: "old-cpu", kind: "cpu", ports: [{ port_id: "gpu" }, { port_id: "memory" }] },
        { component_id: "old-host-memory", kind: "host_memory", read_bandwidth_gbps: 1600, ports: [{ port_id: "cpu" }] },
      ],
      links: [
        { link_id: "old-link", source_component: "old-gpu", source_port: "p0", target_component: "old-hbm", target_port: "p0", protocol: "HBM" },
        { link_id: "old-host-link", source_component: "old-cpu", source_port: "gpu", target_component: "old-gpu", target_port: "host", protocol: "PCIe" },
        { link_id: "old-memory-link", source_component: "old-host-memory", source_port: "cpu", target_component: "old-cpu", target_port: "memory", protocol: "DDR" },
      ],
    },
    model,
    placement: {
      schema_version: "4.0.0",
      model_name: "kept-model",
      hardware_name: "old-hardware",
      op_to_component: {},
      tensor_to_component: {},
      tensor_bytes: {},
      parallel: {
        tp_degree: 2,
        pp_degree: 1,
        ep_degree: 1,
        rank_mapping: [{ rank: 0, component_id: "old-gpu", tp_rank: 0, pp_rank: 0, ep_rank: 0 }],
        layer_to_stage: { layer0: 0 },
        collective_algorithm: "ring",
        routing_policy: "lowest_latency",
      },
      kv_policy: { cache_component: "old-hbm", offload_component: null, tokens_per_page: 64, dtype: "fp8" },
      metadata: {
        ui: { allow_colocated_logical_ranks: true, custom_setting: "keep" },
        control_plane: {
          policy: {},
          decision: {},
          evidence: { input_fingerprint: "old" },
        },
      },
    },
    workload: { schema_version: "4.0.0", name: "kept-workload", requests: [{ request_id: "r0", prompt_tokens: 8, output_tokens: 4 }] },
    profiles: {},
  };
}

function detail() {
  return {
    preset_id: "nvl2-analytical",
    name_zh: "双芯粒异构节点",
    name_en: "Dual Superchip Heterogeneous Node",
    support_level: "analytical_approximation",
    loadable: true,
    hardware: {
      schema_version: "4.0.0",
      name: "new-hardware",
      metadata: {},
      components: [
        { component_id: "new-gpu", kind: "gpu", ports: [{ port_id: "c2c" }, { port_id: "host" }] },
        { component_id: "new-memory", kind: "hbm", ports: [{ port_id: "c2c" }] },
        { component_id: "new-cpu", kind: "cpu", ports: [{ port_id: "gpu" }, { port_id: "memory" }] },
        { component_id: "new-host-memory", kind: "host_memory", read_bandwidth_gbps: 2048, ports: [{ port_id: "cpu" }] },
      ],
      links: [
        { link_id: "new-link", source_component: "new-gpu", source_port: "c2c", target_component: "new-memory", target_port: "c2c", protocol: "NVLink-C2C" },
        { link_id: "new-host-link", source_component: "new-cpu", source_port: "gpu", target_component: "new-gpu", target_port: "host", protocol: "NVLink-C2C" },
        { link_id: "new-memory-link", source_component: "new-host-memory", source_port: "cpu", target_component: "new-cpu", target_port: "memory", protocol: "LPDDR5X" },
      ],
    },
    topology_view: {
      groups: [{ group_id: "node", label: "Node", members: ["new-gpu", "new-memory", "new-cpu", "new-host-memory"], root: "new-gpu", collapsed: true }],
      layout: { positions: { "new-gpu": { x: 20, y: 20 }, "new-memory": { x: 20, y: 20 }, "new-cpu": { x: 20, y: 20 }, "new-host-memory": { x: 20, y: 20 } } },
    },
    compatibility: { loadable: true },
  };
}

test("one fullscreen hardware library keeps architecture and component APIs behaviorally separate", () => {
  for (const id of [
    "hardwarePresetsButton", "hardwarePresetsDialog", "hardwarePresetComponentTab", "hardwarePresetArchitectureTab",
    "architecturePresetsDialog", "componentPresetsDialog", "architecturePresetSearchInput",
    "architecturePresetCategoryFilter", "architecturePresetLevelFilter", "architecturePresetVendorFilter",
    "architecturePresetSupportFilter", "architecturePresetStatus", "architecturePresetList",
  ]) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(html, /组件预设追加到当前拓扑/);
  assert.match(html, /架构预设经确认后替换硬件/);
  assert.equal((html.match(/id="hardwarePresetsButton"/g) || []).length, 1, "hardware catalog has one primary entry");
  assert.doesNotMatch(html, /id="architecturePresetsButton"|id="componentPresetsButton"/);
  assert.match(app, /apiRequest\("\/architecture-presets"/);
  assert.match(app, /`\/architecture-presets\/\$\{encodeURIComponent\(id\)\}`/);
  assert.match(app, /globalThis\.confirm\("载入架构预设会替换当前全部硬件组件与链路/);
  assert.match(css, /\.hardware-presets-dialog,[\s\S]*?width:\s*100dvw;[\s\S]*?height:\s*100dvh;/);
  assert.match(app, /state\.hardwarePresetScroll\[previous\]/);
  assert.match(app, /apiRequest\("\/component-presets"/);
  assert.match(css, /\.fidelity-badge\.is-exact/);
  assert.match(css, /\.fidelity-badge\.is-analytical/);
  assert.match(css, /\.fidelity-badge\.is-experimental/);
});

test("bilingual names, fidelity and explicit loadability render without conflating evidence", () => {
  const ui = helpers();
  const parsed = ui.architecturePresetDetailItem({ preset: { preset_id: "x", name_zh: "中文名", name_en: "English Name", evidence_level: "S2_VENDOR_DECLARED" }, compatibility: { loadable: false } });
  assert.deepEqual(JSON.parse(JSON.stringify(ui.architecturePresetNames(parsed))), { zh: "中文名", en: "English Name" });
  assert.equal(ui.architecturePresetFidelity({ support_level: "exact" }), "exact");
  assert.equal(ui.architecturePresetFidelity({ support_level: "exact_public_topology" }), "exact");
  assert.equal(ui.architecturePresetFidelity({ support_level: "analytical_approximation" }), "analytical");
  assert.equal(ui.architecturePresetFidelity({ support_level: "experimental" }), "experimental");
  assert.equal(ui.architecturePresetIsLoadable(parsed), false);
  const markup = ui.architecturePresetCardMarkup(parsed);
  assert.match(markup, /中文名/);
  assert.match(markup, /English Name/);
  assert.match(markup, /S2 · 厂商声明/);
  assert.match(markup, /仅供查看（不可载入）/);
  assert.match(markup, /disabled/);
});

test("architecture details disclose evidence, parameter basis, aggregation, and planner compatibility when present", () => {
  const ui = helpers();
  const parsed = ui.architecturePresetDetailItem({
    preset: { preset_id: "x", topology_evidence: "vendor topology diagram" },
    parameter_basis: { bandwidth: "one-way payload", peak_ops: "published dense BF16" },
    physical_composition: { gpu: 8, fabric: 1 },
    aggregate_node_explanation: "fabric0 folds a nonblocking switch plane",
    planner_executable: false,
    requires_gpu_attachment: true,
    requires_cpu_attachment: false,
    requires_profile_review: true,
    compatibility: { loadable: true },
  });
  const markup = ui.architecturePresetDetailsMarkup(parsed);
  assert.match(markup, /vendor topology diagram/);
  assert.match(markup, /one-way payload/);
  assert.match(markup, /published dense BF16/);
  assert.match(markup, /fabric0 folds a nonblocking switch plane/);
  assert.match(markup, /Planner 不可执行|否（No）/);
  assert.match(markup, /需连接 GPU/);
  assert.match(markup, /需连接 CPU/);
  assert.match(markup, /需复核硬件 Profile/);
});

test("architecture component details render physical counts, per-unit capacity, count status, and existing derivation metadata in Chinese", () => {
  const ui = helpers();
  const markup = ui.architecturePresetDetailsMarkup({
    preset_id: "physical-hbm",
    loadable: true,
    hardware: {
      components: [{
        component_id: "hbm0",
        kind: "hbm",
        capacity_bytes: 80 * (1024 ** 3),
        metadata: {
          physical_composition: {
            simulator_representation: "aggregate_node",
            simulator_node_count: 1,
            physical_unit_kind: "HBM_stack",
            physical_unit_count: 5,
            physical_unit_count_status: "vendor_documented_active_stacks",
          },
          measurement_basis: "厂商整卡可见容量",
          derived_formula: "整卡容量除以有效堆栈数",
          evidence_level: "S2_VENDOR_DECLARED",
        },
      }],
      links: [],
    },
    compatibility: { loadable: true },
  });
  assert.match(markup, /5 × HBM 堆栈/);
  assert.match(markup, /每物理单元容量：16 GiB（仿真组件总容量 ÷ 5）/);
  assert.match(markup, /数量依据：厂商文档明确有效堆栈数/);
  assert.match(markup, /来源与推导/);
  assert.match(markup, /测量依据（Measurement Basis）: 厂商整卡可见容量/);
  assert.match(markup, /推导公式（Derived Formula）: 整卡容量除以有效堆栈数/);
});

test("physical HBM nodes disclose stable unit metadata and cross-check flat-group membership without inventing values", () => {
  const ui = helpers();
  const physical = (index) => ({
    simulator_representation: "single_physical_unit_node",
    simulator_node_count: 1,
    physical_unit_kind: "HBM_stack",
    physical_unit_count: 1,
    physical_unit_count_status: "explicit_physical_node",
    unit_index: index,
    unit_count_in_product: 2,
    unit_count_status: "vendor_documented_stacks",
    unit_count_formula: "vendor-documented product stack count = 2",
    unit_capacity_bytes: 16 * (1024 ** 3),
    product_total_capacity_bytes: 32 * (1024 ** 3),
    unit_bandwidth_gbps: 3200,
    product_total_bandwidth_gbps: 6400,
    component_preset_id: "",
    component_preset_status: "no_exact_per_stack_component_preset",
    source_basis: "vendor product architecture and aggregate memory specification",
  });
  const component = (index) => ({
    component_id: `hbm${index}`,
    kind: "hbm",
    capacity_bytes: 16 * (1024 ** 3),
    metadata: {
      physical_composition: physical(index),
      parameter_basis: { capacity_bytes: "product total divmod unit count" },
      provenance: { value_status: "derived_per_physical_stack" },
    },
  });
  const markup = ui.architecturePresetDetailsMarkup({
    preset_id: "physical-hbm-node",
    hardware: { components: [component(0), component(1)], links: [], metadata: {} },
    topology_view: { groups: [{ group_id: "package0", members: ["hbm0", "hbm1"], root: "hbm0", collapsed: true }] },
    compatibility: { loadable: true },
  });
  assert.match(markup, /当前仿真节点对应一颗物理单元/);
  assert.match(markup, /产品内第 1 \/ 2 颗/);
  assert.match(markup, /每物理单元容量：16 GiB/);
  assert.match(markup, /当前分组含 2 个 HBM 物理组件，与产品单元总数一致/);
  assert.match(markup, /单元数量状态（Unit Count Status）: 厂商文档明确堆栈数/);
  assert.match(markup, /单元数量公式（Unit Count Formula）: vendor-documented product stack count = 2/);
  assert.match(markup, /产品总容量（Product Total Capacity, B\/KiB…PiB）: 32 GiB/);
  assert.match(markup, /无精确的单颗堆栈组件预设/);
  assert.match(markup, /数值状态（Value Status）: 按物理堆栈推导/);
});

test("flat groups count HBM nodes by memory subsystem before falling back to controller", () => {
  const ui = helpers();
  const physicalHbm = (id, index, memorySubsystemId, controllerComponentId, unitCount = 8) => ({
    component_id: id,
    kind: "hbm",
    capacity_bytes: 23.25 * (1024 ** 3),
    metadata: {
      physical_composition: {
        simulator_representation: "single_physical_unit_node",
        simulator_node_count: 1,
        physical_unit_kind: "HBM_stack",
        physical_unit_count: 1,
        physical_unit_count_status: "explicit_physical_node",
        unit_index: index,
        unit_count_in_product: unitCount,
        memory_subsystem_id: memorySubsystemId,
        controller_component_id: controllerComponentId,
      },
    },
  });
  const components = [
    { component_id: "gpu0", kind: "gpu" },
    { component_id: "gpu1", kind: "gpu" },
    ...Array.from({ length: 8 }, (_, index) => physicalHbm(`hbm0_${index}`, index, "gpu0-memory", "shared-controller")),
    ...Array.from({ length: 8 }, (_, index) => physicalHbm(`hbm1_${index}`, index, "gpu1-memory", "shared-controller")),
  ];
  const markup = ui.architecturePresetDetailsMarkup({
    preset_id: "gb200-flat-group",
    hardware: { components, links: [], metadata: {} },
    topology_view: { groups: [{ group_id: "superchip0", members: components.map((component) => component.component_id) }] },
    compatibility: { loadable: true },
  });
  assert.equal((markup.match(/当前分组含 8 个 HBM 物理组件，与产品单元总数一致/g) || []).length, 16);
  assert.doesNotMatch(markup, /当前分组含 16 个 HBM 物理组件/);
});

test("flat group HBM counts fall back to controller_component_id when memory subsystem is absent", () => {
  const ui = helpers();
  const physicalHbm = (id, controllerComponentId) => ({
    component_id: id,
    kind: "hbm",
    metadata: {
      physical_composition: {
        simulator_representation: "single_physical_unit_node",
        simulator_node_count: 1,
        physical_unit_kind: "HBM_stack",
        physical_unit_count: 1,
        physical_unit_count_status: "explicit_physical_node",
        unit_count_in_product: 2,
        controller_component_id: controllerComponentId,
      },
    },
  });
  const components = [
    physicalHbm("hbm0_0", "gpu0"),
    physicalHbm("hbm0_1", "gpu0"),
    physicalHbm("hbm1_0", "gpu1"),
    physicalHbm("hbm1_1", "gpu1"),
  ];
  const markup = ui.architecturePresetDetailsMarkup({
    preset_id: "controller-fallback-group",
    hardware: { components, links: [], metadata: {} },
    topology_view: { groups: [{ group_id: "flat", members: components.map((component) => component.component_id) }] },
    compatibility: { loadable: true },
  });
  assert.equal((markup.match(/当前分组含 2 个 HBM 物理组件，与产品单元总数一致/g) || []).length, 4);
  assert.doesNotMatch(markup, /当前分组含 4 个 HBM 物理组件/);
});

test("architecture replacement is one atomic undo/redo unit and preserves model, workload and non-hardware settings", () => {
  const ui = helpers();
  ui.configure();
  const initial = ui.ensureScenarioShape(scenario());
  initial.profiles.components.cim = { "old-cim": { name: "old-cim-profile" } };
  initial.profiles.cim_interconnect = { name: "old-cim-link-profile" };
  ui.state.scenario = initial;
  ui.state.mappingStale = false;
  ui.state.mappingStaleReason = "";
  ui.state.topologyView = TopologyCore.normalizeTopologyView({}, initial.hardware.components.map((item) => item.component_id));
  ui.state.nodePositions = ui.state.topologyView.layout.positions;
  initial.hardware.metadata.topology_view = JSON.parse(JSON.stringify(ui.state.topologyView));
  ui.state.topologyHistory = { undo: [], redo: [], restoring: false };
  const oldModel = JSON.stringify(initial.model);
  const oldWorkload = JSON.stringify(initial.workload);
  const oldHardware = JSON.stringify(initial.hardware);
  const oldPlacement = JSON.stringify(initial.placement);
  const oldProfiles = JSON.stringify(initial.profiles);
  const oldFusion = JSON.stringify(initial.profiles.fusion);

  const result = ui.applyArchitecturePresetDetail(detail());
  assert.equal(result.adjusted, true);
  assert.equal(JSON.stringify(ui.state.scenario.model), oldModel);
  assert.equal(JSON.stringify(ui.state.scenario.workload), oldWorkload);
  assert.equal(ui.state.scenario.weights_resident, false);
  assert.equal(ui.state.scenario.hardware.name, "new-hardware");
  const components = Object.fromEntries(ui.state.scenario.hardware.components.map((component) => [component.component_id, component]));
  const profiles = ui.state.scenario.profiles.components;
  assert.equal(profiles.gpu[components["new-gpu"].cost_profile_id].tensor_core.resource_id, "new-gpu.tensor_core");
  assert.equal(profiles.cpu[components["new-cpu"].cost_profile_id].pipeline.resource_id, "new-cpu.pipeline");
  assert.equal(profiles.hbm[components["new-memory"].cost_profile_id].resource_id, "new-memory.hbm_fabric");
  assert.equal(profiles.host_memory[components["new-host-memory"].cost_profile_id].resource_id, "new-host-memory.memory");
  assert.equal(ui.state.scenario.profiles.host_orchestration.cpu_component_id, "new-cpu");
  assert.equal(ui.state.scenario.profiles.host_orchestration.gpu_component_id, "new-gpu");
  assert.equal(ui.state.scenario.profiles.host_orchestration.scheduler_resource_id, "new-cpu.scheduler");
  assert.equal(ui.state.scenario.profiles.host_orchestration.submission_resource_id, "new-gpu.command_queue");
  assert.equal(JSON.stringify(ui.state.scenario.profiles.fusion), oldFusion);
  assert.equal(Object.hasOwn(ui.state.scenario.profiles.components, "cim"), false);
  assert.equal(Object.hasOwn(ui.state.scenario.profiles, "cim_interconnect"), false);
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.op_to_component)), {});
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.tensor_to_component)), {});
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.tensor_bytes)), {});
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.parallel.rank_mapping)), []);
  assert.equal(ui.state.scenario.placement.parallel.tp_degree, 2);
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.parallel.layer_to_stage)), { layer0: 0 });
  assert.equal(ui.state.scenario.placement.parallel.collective_algorithm, "ring");
  assert.equal(ui.state.scenario.placement.kv_policy.tokens_per_page, 64);
  assert.equal(ui.state.scenario.placement.kv_policy.dtype, "fp8");
  assert.equal(ui.state.scenario.placement.kv_policy.cache_component, null);
  assert.equal(ui.state.scenario.placement.metadata.ui.custom_setting, "keep");
  assert.equal(Object.hasOwn(ui.state.scenario.placement.metadata, "control_plane"), false);
  assert.equal(Object.hasOwn(ui.state.scenario.placement.metadata, "auto_mapping"), false);
  assert.equal(ui.state.mappingStale, true);
  assert.match(ui.state.mappingStaleReason, /引用旧组件的算子、张量、KV 驻留与 Rank 映射已清空/);
  assert.equal(ui.state.topologyHistory.undo.length, 1);
  assert.equal(ui.state.topologyView.groups[0].collapsed, true, "preset collapse state survives architecture replacement");
  assert.equal(ui.state.scenario.hardware.metadata.topology_view.groups[0].collapsed, true);
  assert.notDeepEqual(ui.state.nodePositions["new-gpu"], ui.state.nodePositions["new-memory"]);
  const projection = TopologyCore.collapseProjection(
    ui.state.scenario.hardware.components,
    ui.state.scenario.hardware.links,
    ui.state.topologyView.groups,
  );
  assert.deepEqual(Array.from(projection.visibleComponentIds), ["new-gpu"], "collapsed group displays only its root");
  assert.equal(projection.links.length, 0, "internal links are hidden only in the canvas projection");
  assert.equal(ui.state.scenario.hardware.components.length, 4, "simulation JSON retains grouped members");
  assert.equal(ui.state.scenario.hardware.links.length, 3, "simulation JSON retains grouped links");
  const positionsBeforeExpand = JSON.stringify(ui.state.topologyView.layout.positions);
  const expanded = TopologyCore.setGroupCollapsed(ui.state.topologyView, "node", false);
  assert.equal(JSON.stringify(expanded.layout.positions), positionsBeforeExpand, "expanding preserves preset component coordinates");
  const rects = ui.state.scenario.hardware.components.map((component) => TopologyCore.rectForNode(
    component.component_id,
    expanded.layout.positions,
    { "new-gpu": { width: 130, height: 66 }, "new-memory": { width: 130, height: 66 } },
  ));
  assert.equal(TopologyCore.rectsIntersect(rects[0], rects[1]), false, "expanded components do not overlap");

  assert.equal(ui.travelTopologyHistory("undo"), true);
  assert.equal(JSON.stringify(ui.state.scenario.hardware), oldHardware);
  assert.equal(JSON.stringify(ui.state.scenario.placement), oldPlacement);
  assert.equal(JSON.stringify(ui.state.scenario.profiles), oldProfiles);
  assert.equal(ui.state.mappingStale, false);

  assert.equal(ui.travelTopologyHistory("redo"), true);
  assert.equal(ui.state.scenario.hardware.name, "new-hardware");
  assert.equal(ui.state.scenario.profiles.host_orchestration.cpu_component_id, "new-cpu");
  assert.deepEqual(JSON.parse(JSON.stringify(ui.state.scenario.placement.op_to_component)), {});
  assert.equal(ui.state.mappingStale, true);
  assert.match(ui.state.mappingStaleReason, /架构预设 nvl2-analytical 已替换硬件拓扑/);
});

test("GPU-only architecture invalidates orchestration until a real CPU is attached", () => {
  const ui = helpers();
  ui.configure();
  const initial = ui.ensureScenarioShape(scenario());
  ui.state.scenario = initial;
  ui.state.topologyView = TopologyCore.normalizeTopologyView({}, initial.hardware.components.map((item) => item.component_id));
  ui.state.nodePositions = ui.state.topologyView.layout.positions;
  ui.state.topologyHistory = { undo: [], redo: [], restoring: false };

  const gpuOnly = JSON.parse(JSON.stringify(detail()));
  gpuOnly.hardware.components = gpuOnly.hardware.components.filter((component) => !["cpu", "host_memory"].includes(component.kind));
  gpuOnly.hardware.links = gpuOnly.hardware.links.filter((link) => !link.link_id.includes("host") && !link.link_id.includes("memory"));
  gpuOnly.topology_view.groups[0].members = ["new-gpu", "new-memory"];
  delete gpuOnly.topology_view.layout.positions["new-cpu"];
  delete gpuOnly.topology_view.layout.positions["new-host-memory"];
  gpuOnly.compatibility = {
    planner_executable: false,
    requires_cpu_attachment: true,
    requires_gpu_attachment: false,
  };

  ui.applyArchitecturePresetDetail(gpuOnly);

  assert.equal(Object.hasOwn(ui.state.scenario.profiles, "host_orchestration"), false);
  const missingCpuIssue = ui.hostOrchestrationReferenceIssue(ui.state.scenario);
  assert.match(missingCpuIssue, /CPU/u);
  assert.match(missingCpuIssue, /cpu_component_id/u);
  assert.match(missingCpuIssue, /V4 场景/u);
  assert.throws(
    () => ui.scenarioPayloadForTransport(ui.state.scenario),
    /profiles\.host_orchestration/,
  );

  const attached = [
    { component_id: "attached-cpu", kind: "cpu", ports: [] },
    { component_id: "attached-host-memory", kind: "host_memory", read_bandwidth_gbps: 1600, ports: [] },
  ];
  ui.state.scenario.hardware.components.push(...attached);
  ui.materializeMissingCostProfiles(attached, ui.state.scenario);

  assert.equal(ui.hostOrchestrationReferenceIssue(ui.state.scenario), "");
  assert.equal(ui.state.scenario.profiles.host_orchestration.cpu_component_id, "attached-cpu");
  assert.equal(ui.state.scenario.profiles.host_orchestration.gpu_component_id, "new-gpu");
});

test("V4 transport rejects removed flat KV fields and preserves nested policy", () => {
  const ui = helpers();
  const nested = scenario();
  nested.profiles.runtime = { gpu_controllers: { "old-gpu": {} } };
  nested.placement.kv_policy.cache_component = "old-hbm";
  const payload = ui.scenarioPayloadForTransport(nested);
  assert.equal(payload.placement.kv_policy.cache_component, "old-hbm");
  assert.equal(Object.hasOwn(payload.placement, "kv_cache_component"), false);
  const removed = scenario();
  removed.profiles.runtime = { gpu_controllers: { "old-gpu": {} } };
  removed.placement.kv_cache_component = "old-hbm";
  assert.throws(() => ui.scenarioPayloadForTransport(removed), /已从 V4 schema 删除/);
});

test("backend field-path diagnostics win over a generic parse hint", () => {
  const ui = helpers();
  const message = ui.backendChineseMessage({
    message: "场景 JSON 解析失败；请检查字段类型、必填项和 rank_mapping 结构",
    details: {
      diagnostics: [{ detail_zh: "placement.kv_cache_component 与 placement.kv_policy.cache_component 冲突" }],
    },
  });
  assert.equal(message, "placement.kv_cache_component 与 placement.kv_policy.cache_component 冲突");
  assert.doesNotMatch(message, /rank_mapping/);

  const nestedMessage = ui.backendChineseMessage({
    message: "场景 JSON 解析失败；请检查 rank_mapping 结构",
    details: { validation: { errors: { scenario: [{ message_zh: "placement.kv_offload_component 与 placement.kv_policy.offload_component 冲突" }] } } },
  });
  assert.equal(nestedMessage, "placement.kv_offload_component 与 placement.kv_policy.offload_component 冲突");
});
