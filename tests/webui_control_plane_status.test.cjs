"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const source = fs.readFileSync(appPath, "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function loadHelpers() {
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
;globalThis.__controlPlaneTests = {
  state,
  dom,
  controlPlaneDecision,
  controlPlaneEvidence,
  controlPlanePolicy,
  controlPlaneStatusView,
  ensureScenarioShape,
  mappingFingerprintFrom,
  placementControlPlaneMetadata,
  renderControlPlaneStatus,
};`, context, { filename: appPath });
  return context.__controlPlaneTests;
}

function scenarioWithControlPlane() {
  return {
    placement: {
      metadata: {
        control_plane: {
          policy: {
            options: { mode: "heuristic", objective: "balanced" },
          },
          decision: {
            status: "feasible_timeout",
            solver: "builtin",
            objective: "balanced",
            objective_value: 123.5,
            lower_bound: 100,
            gap: 0.19,
            optimality_proven: false,
            fully_placed: true,
            operator_execution_targets: {
              "dense0.mlp": [{ rank_id: 0, component_id: "gpu0" }],
            },
            rank_weight_shards: {
              "dense0.mlp_weights": [
                { rank_id: 0, storage_component_id: "hbm0" },
                { rank_id: 1, storage_component_id: "hbm1" },
              ],
            },
          },
          evidence: {
            fingerprint_schema: "runtime-control-plane-v4",
            input_fingerprint: "abc123",
          },
        },
      },
    },
  };
}

function completeV4ModelGraph() {
  return ModelGraphCore.buildModelGraphFromLayerSpecs([{
    layer_id: "dense0",
    kind: "dense",
    hidden_size: 16,
    intermediate_size: 32,
    attention_heads: 2,
    kv_heads: 2,
    attention_head_dim: 8,
    sequence_mixer: "full_attention",
    linear_attention: null,
    num_experts: 1,
    experts_per_token: 1,
    shared_expert_intermediate_size: 0,
    shared_expert_gate: false,
    dtype: "bf16",
    quantization: null,
    weight_bytes: 1024,
    metadata: {},
  }], {
    name: "m",
    architecture: "transformer",
    vocabulary_size: 32000,
    max_sequence_length: 4096,
    embedding_weight_bytes: 1024,
  });
}

test("manual auto-map request and settings surfaces are absent", () => {
  assert.doesNotMatch(source, /\/api\/auto-map|runAutoMapping|mappingModeInput|mappingSolverInput|setMappingLocked|renderOpMappings|renderTensorMappings|addMapping/u);
  assert.doesNotMatch(html, /rerunAutoMappingButton|autoMappingResult|mappingModeInput|mappingSolverInput|manualMappingDisclosure|addOpMappingButton|addTensorMappingButton|opMappingBody|tensorMappingBody|modelWeightsBackingInput/u);
  assert.match(html, /id="controlPlaneStatus"/u);
  assert.match(html, /id="controlPlaneStatusBadge"/u);
  assert.match(html, /id="controlPlaneStatusMetrics"/u);
});

test("layered control-plane metadata renders as read-only materialized status", () => {
  const helpers = loadHelpers();
  helpers.state.scenario = scenarioWithControlPlane();
  helpers.state.mappingStale = false;
  helpers.state.mappingStaleReason = "";
  const view = helpers.controlPlaneStatusView();

  assert.equal(view.status, "ready");
  assert.equal(view.materialized, true);
  assert.match(view.label, /已物化/u);
  assert.deepEqual(
    JSON.parse(JSON.stringify(view.metrics)),
    [
      ["指纹架构", "runtime-control-plane-v4", "mapping_fingerprint"],
      ["策略模式", "heuristic", "control_plane"],
      ["目标", "balanced", "objective"],
      ["求解器", "builtin", "solver"],
      ["目标值", 123.5, "objective"],
      ["下界", 100, "lower_bound"],
      ["Gap", 0.19, "optimality_gap", "percent"],
      ["算子 Rank 目标", 1, "operator_targets"],
      ["张量分片", 2, "weight_tensor_shards"],
    ],
  );

  helpers.dom.controlPlaneStatus = {};
  helpers.dom.controlPlaneStatusBadge = { className: "", textContent: "" };
  helpers.dom.controlPlaneStatusSummary = { textContent: "" };
  helpers.dom.controlPlaneStatusMetrics = { innerHTML: "" };
  helpers.renderControlPlaneStatus();
  assert.equal(helpers.dom.controlPlaneStatusBadge.className, "control-plane-status-badge is-ready");
  assert.match(helpers.dom.controlPlaneStatusMetrics.innerHTML, /runtime-control-plane-v4/u);
  assert.match(helpers.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="objective"/u);
  assert.match(helpers.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="lower_bound"/u);
  assert.match(helpers.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="optimality_gap"/u);
});

test("control-plane accessors do not read or import legacy auto-mapping metadata", () => {
  const helpers = loadHelpers();
  const placement = {
    metadata: {
      auto_mapping: {
        input_fingerprint: "legacy-fingerprint",
        locked_op_keys: ["legacy.op"],
        locked_tensor_ids: ["legacy.tensor"],
      },
    },
  };

  assert.equal(helpers.mappingFingerprintFrom({ placement }), "");
  assert.deepEqual(JSON.parse(JSON.stringify(helpers.controlPlanePolicy(placement))), {});
  assert.deepEqual(JSON.parse(JSON.stringify(helpers.controlPlaneDecision(placement))), {});
  assert.deepEqual(JSON.parse(JSON.stringify(helpers.controlPlaneEvidence(placement))), {});

  assert.deepEqual(JSON.parse(JSON.stringify(helpers.placementControlPlaneMetadata(placement))), {});
  assert.deepEqual(placement.metadata.auto_mapping.locked_op_keys, ["legacy.op"]);
});

test("status distinguishes stale and not-yet-materialized placement", () => {
  const helpers = loadHelpers();
  helpers.state.scenario = scenarioWithControlPlane();
  helpers.state.mappingStale = true;
  helpers.state.mappingStaleReason = "fingerprint mismatch";
  assert.deepEqual(
    {
      status: helpers.controlPlaneStatusView().status,
      summary: helpers.controlPlaneStatusView().summary,
    },
    { status: "stale", summary: "fingerprint mismatch" },
  );

  helpers.state.mappingStale = false;
  helpers.state.scenario = { placement: { metadata: { control_plane: { policy: {}, decision: {}, evidence: {} } } } };
  const empty = helpers.controlPlaneStatusView();
  assert.equal(empty.status, "idle");
  assert.equal(empty.materialized, false);
  assert.match(empty.summary, /不提供手动规划操作/u);
});

test("V4 UI rejects retired placement and control-plane authoring fields", () => {
  const helpers = loadHelpers();
  const manual = scenarioWithControlPlane();
  manual.schema_version = "4.0.0";
  manual.hardware = { schema_version: "4.0.0", name: "h", metadata: {}, components: [], links: [] };
  manual.model = { schema_version: "4.0.0", name: "m", graph: completeV4ModelGraph() };
  manual.placement.schema_version = "4.0.0";
  manual.placement.model_name = "m";
  manual.placement.hardware_name = "h";
  manual.placement.op_to_component = { op: "gpu0" };
  manual.placement.tensor_to_component = {};
  manual.placement.tensor_bytes = {};
  assert.throws(() => helpers.ensureScenarioShape(manual), /V4 authoring.*placement\.op_to_component/u);

  manual.placement.op_to_component = {};
  manual.placement.metadata.control_plane.policy.locked_tensor_ids = [];
  assert.throws(() => helpers.ensureScenarioShape(manual), /V4 authoring.*locked_tensor_ids/u);
});
