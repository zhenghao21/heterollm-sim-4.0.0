"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const source = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function load() {
  const context = vm.createContext({
    AbortController,
    console,
    document: { addEventListener() {} },
    Intl,
    localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
    ModelGraphCore,
    Promise,
    setTimeout,
    clearTimeout,
    TopologyCore,
    TraceViewCore,
  });
  vm.runInContext(`${source}
;globalThis.__adaptationTests = {
  state,
  cimCostProfileMarkup,
  memoryCostProfileMarkup,
  hbfMediaMarkup,
  thermalOperatingPointMarkup,
  thermalLinkMarkup,
  controlPlanePolicyMarkup,
  controlPlaneMemoryTierMaps,
  controlPlaneMemoryTierDetails,
  costProfileDraft,
};`, context, { filename: path.join(webui, "app.js") });
  return context.__adaptationTests;
}

function scenario() {
  return {
    hardware: {
      metadata: { thermal_operating_point: { domain_id: "stack0.thermal", enabled: false } },
      components: [
        { component_id: "gpu0", kind: "gpu", ports: [] },
        { component_id: "hbm0", kind: "hbm", cost_profile_id: "hbm0-hbm", ports: [] },
        { component_id: "hbf0", kind: "hbf", metadata: { thermal_domain_id: "stack0.thermal" }, ports: [] },
      ],
      links: [],
    },
    model: {},
    placement: {
      metadata: {
        linear_state_offload_mode: "pressure",
        memory_tiers: {
          kv_layer_components: { "layer.0": "hbm0" },
          linear_state_layer_components: { "layer.1": "hbf0" },
        },
        control_plane: { policy: { options: {} } },
      },
      parallel: { tp_degree: 1, pp_degree: 1, ep_degree: 1 },
      kv_policy: {},
    },
    profiles: { components: { hbm: { "hbm0-hbm": { bandwidth_gb_s: 100 } } } },
  };
}

test("memory profile markup exposes optional directional fields without inventing zero overrides", () => {
  const ui = load();
  ui.state.scenario = scenario();
  const markup = ui.memoryCostProfileMarkup("hbm", ui.state.scenario.hardware.components[1]);
  assert.match(markup, /read_bandwidth_gb_s/);
  assert.match(markup, /write_bandwidth_gb_s/);
  assert.match(markup, /optional_positive/);
  assert.match(markup, /transaction_bytes/);
  assert.match(markup, /max_outstanding_requests/);
  assert.equal(ui.costProfileDraft("hbm", ui.state.scenario.hardware.components[1]).read_bandwidth_gb_s, undefined);
});

test("CIM markup gates tile controls on tiled conversion and preserves backend contract fields", () => {
  const ui = load();
  ui.state.scenario = scenario();
  ui.state.scenario.hardware.components.push({ component_id: "cim0", kind: "cim", cost_profile_id: "cim0-cim", ports: [] });
  ui.state.scenario.profiles.components.cim = {
    "cim0-cim": {
      array_count: 1, p_m: 1, p_k: 4, p_n: 4, frequency_ghz: 1,
      arithmetic_mode: "fp16_fp32_analytical",
      float_cycles_per_eval: 2,
      float_accumulator_outputs_per_cycle: 8,
      float_contract_basis: "test evidence",
      weight_conversion_mode: "packed_to_fp16_tiled_cold",
      weight_decode_elements_per_ns: 2,
      conversion_scratch_capacity_bytes: 4096,
      activation_fp32_to_fp16_elements_per_ns: 3,
      conversion_contract_basis: "test converter evidence",
      tile_m: 8, tile_k: 256, tile_n: 256,
    },
  };
  const markup = ui.cimCostProfileMarkup(ui.state.scenario.hardware.components.at(-1));
  for (const field of ["weight_conversion_mode", "arithmetic_mode", "weight_decode_elements_per_ns", "conversion_scratch_capacity_bytes", "activation_fp32_to_fp16_elements_per_ns", "conversion_contract_basis", "tile_m", "tile_k", "tile_n"]) assert.match(markup, new RegExp(field));
  assert.match(markup, /tile_k/);
});

test("placement policy markup includes new targets and linear-state offload mode is represented separately", () => {
  const ui = load();
  ui.state.scenario = scenario();
  const markup = ui.controlPlanePolicyMarkup(ui.state.scenario.placement);
  for (const field of ["operator_targets", "weight_tensor_targets", "kv_cache_target", "linear_state_target", "linear_state_offload_target", "kv_layer_targets", "linear_state_layer_targets"]) assert.match(markup, new RegExp(field));
  assert.match(markup, /allow_cold_cim_streaming/);
  assert.match(markup, /gpu_loadable_order/);
  assert.match(markup, /保留 tied weight runtime copies/);
  assert.match(markup, /目标约束/);
});

test("HBF media and thermal UI are explicit, constrained, and non-fictional", () => {
  const ui = load();
  ui.state.scenario = scenario();
  const hbf = ui.state.scenario.hardware.components.find((item) => item.kind === "hbf");
  const hbfMarkup = ui.hbfMediaMarkup(hbf);
  assert.match(hbfMarkup, /启用 cold_page_v1 媒体模型/);
  assert.match(hbfMarkup, /cold_page_v1/);
  assert.match(hbfMarkup, /4096/);
  assert.match(hbfMarkup, /2 的幂/);
  assert.match(ui.thermalOperatingPointMarkup(hbf), /只读状态/);
  ui.state.scenario.hardware.links.push({
    link_id: "gpu-hbf0",
    source_component: "gpu0",
    target_component: "hbf0",
    metadata: { thermal_domain_id: "stack0.thermal" },
  });
  assert.match(ui.thermalLinkMarkup(ui.state.scenario.hardware.links[0]), /链路静态热降额（只读状态）/);
});

test("memory tier metadata is read-only data and remains available from placement or runtime payload", () => {
  const ui = load();
  ui.state.scenario = scenario();
  assert.deepEqual(JSON.parse(JSON.stringify(ui.controlPlaneMemoryTierMaps())), ui.state.scenario.placement.metadata.memory_tiers);
  assert.deepEqual(JSON.parse(JSON.stringify(ui.controlPlaneMemoryTierDetails())), {});
  ui.state.report = { runtime_placement: { schema_version: "runtime-placement/v1", metadata: { memory_tiers: { kv_layer_components: { "layer.2": "hbm0" } } } } };
  assert.deepEqual(JSON.parse(JSON.stringify(ui.controlPlaneMemoryTierMaps())), { kv_layer_components: { "layer.2": "hbm0" } });
  delete ui.state.scenario.placement.metadata.memory_tiers;
  assert.deepEqual(JSON.parse(JSON.stringify(ui.controlPlaneMemoryTierMaps())), { kv_layer_components: { "layer.2": "hbm0" } });
  ui.state.report.runtime_placement.memory_tiers = { kv_layer_components: { "layer.3": "hbm0" } };
  delete ui.state.report.runtime_placement.metadata;
  assert.deepEqual(JSON.parse(JSON.stringify(ui.controlPlaneMemoryTierMaps())), { kv_layer_components: { "layer.3": "hbm0" } });
});

test("contract and responsive styles give new controls a stable hierarchy", () => {
  assert.match(css, /\.placement-target-row\s*\{/);
  assert.match(css, /\.profile-contract-callout/);
  assert.match(css, /\.control-plane-memory-tiers/);
  assert.match(css, /min-height:\s*calc\(36px \* var\(--layout-scale\)\)/);
  assert.match(css, /@media \(max-width: 560px\)/);
});
