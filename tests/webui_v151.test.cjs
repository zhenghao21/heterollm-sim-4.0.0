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
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function graphFromFixture({ layer_specs: layerSpecs, ...options }) {
  return ModelGraphCore.buildModelGraphFromLayerSpecs(layerSpecs, options);
}

function qwen38Model() {
  const linearAttention = {
    key_heads: 16,
    value_heads: 48,
    key_head_dim: 128,
    value_head_dim: 128,
    conv_kernel_size: 4,
    state_dtype: "fp32",
    output_gate: true,
    gate_activation: "silu",
  };
  return {
    name: "Qwen3.8-27B",
    architecture: "qwen3_5_hybrid_transformer",
    vocabulary_size: 248320,
    max_sequence_length: 262144,
    layer_specs: Array.from({ length: 64 }, (_, index) => {
      const fullAttention = index % 4 === 3;
      return {
        schema_version: "4.0.0",
        layer_id: `layer-${String(index).padStart(3, "0")}`,
        kind: "dense",
        hidden_size: 5120,
        intermediate_size: 17408,
        attention_heads: 24,
        kv_heads: 4,
        attention_head_dim: 256,
        sequence_mixer: fullAttention ? "full_attention" : "linear_attention",
        linear_attention: fullAttention ? null : { ...linearAttention },
        num_experts: 1,
        experts_per_token: 1,
        shared_expert_intermediate_size: 0,
        shared_expert_gate: false,
        dtype: "bf16",
        quantization: null,
        weight_bytes: fullAttention ? 681574400 : 765460480,
        metadata: {
          preset_pattern: fullAttention ? "full_attention_block" : "linear_attention_block",
          pattern_index: fullAttention ? 1 : 0,
          pattern_offset: fullAttention ? 0 : index % 4,
          weight_bytes_method: "bf16_matrix_shape_estimate",
        },
      };
    }),
  };
}

function withTypedMtp(graphValue, predictionLayers = 2, includeAuxiliaryHead = true) {
  const graph = structuredClone(graphValue);
  const groups = graph.operators.filter((item) => item.op_kind === "layer_group");
  const lastGroup = groups[groups.length - 1];
  const branchTensor = `${lastGroup.operator_id}.output`;
  const branch = graph.tensors.find((item) => item.tensor_id === branchTensor);
  const dtype = branch.dtype;
  const shape = branch.shape;
  const layout = branch.layout;
  const port = (portId, direction, tensorId, portShape = shape) => ({ port_id: portId, direction, tensor_id: tensorId, dtype, shape: portShape, layout });
  const finalNorm = graph.operators.find((item) => item.operator_id === "final_norm");
  const firstSequence = finalNorm.sequence_index;
  graph.operators.filter((item) => item.sequence_index >= firstSequence).forEach((item) => { item.sequence_index += predictionLayers + (includeAuxiliaryHead ? 1 : 0); });
  let previous = branchTensor;
  for (let index = 0; index < predictionLayers; index += 1) {
    const prefix = `mtp.prediction_layer.${String(index).padStart(3, "0")}`;
    const output = `${prefix}.output`;
    const weights = `${prefix}.weights`;
    graph.operators.push({
      operator_id: prefix, op_kind: "mtp_prediction_layer", sequence_index: firstSequence + index,
      input_tensor_ids: [previous], output_tensor_ids: [output], weight_tensor_ids: [weights],
      ports: [port("in0", "input", previous), port("out0", "output", output), port("weight0", "weight", weights, [shape[2], shape[2]])],
      parameters: { hidden_size: shape[2] }, attributes: {},
    });
    graph.tensors.push(
      { tensor_id: output, role: "activation", producer_operator_id: prefix, consumer_operator_ids: [], dtype, shape, layout },
      { tensor_id: weights, role: "weight", producer_operator_id: null, consumer_operator_ids: [prefix], dtype, shape: [shape[2], shape[2]], layout },
    );
    previous = output;
  }
  if (includeAuxiliaryHead) {
    graph.operators.push({
      operator_id: "mtp.aux_head", op_kind: "mtp_aux_head", sequence_index: firstSequence + predictionLayers,
      input_tensor_ids: [previous], output_tensor_ids: ["mtp.proposal_logits"], weight_tensor_ids: ["mtp.aux_head.weights"],
      ports: [port("in0", "input", previous), port("out0", "output", "mtp.proposal_logits", ["B", "T", "V"]), port("weight0", "weight", "mtp.aux_head.weights", [shape[2], "V"])],
      parameters: {}, attributes: {},
    });
    graph.tensors.push(
      { tensor_id: "mtp.proposal_logits", role: "output", producer_operator_id: "mtp.aux_head", consumer_operator_ids: [], dtype, shape: ["B", "T", "V"], layout },
      { tensor_id: "mtp.aux_head.weights", role: "weight", producer_operator_id: null, consumer_operator_ids: ["mtp.aux_head"], dtype, shape: [shape[2], "V"], layout },
    );
  }
  return ModelGraphCore.normalizeModelGraph(graph);
}

function groupOperatorIds(graph, groupId) {
  return graph.operators
    .filter((operator) => String(operator.attributes?.parent_group_id || "") === groupId)
    .map((operator) => operator.operator_id);
}

function authoritativeGroupProjection(graph, groupId, expandedComponents = []) {
  return ModelGraphCore.authoritativeOperatorProjection(
    graph,
    groupOperatorIds(graph, groupId),
    { group_id: groupId, expanded_components: expandedComponents },
  );
}

function appHelpers() {
  const animationFrames = [];
  let resizeCallback = null;
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
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame(callback) { animationFrames.push(callback); return animationFrames.length; },
    cancelAnimationFrame() {},
    ResizeObserver: class ResizeObserver {
      constructor(callback) { resizeCallback = callback; }
      observe() {}
    },
    setTimeout,
    __animationFrames: animationFrames,
    __triggerModelResize() { resizeCallback?.([]); },
  });
  vm.runInContext(`${app}
    let __modelRenderCount = 0;
    renderModelGraph = () => {
      __modelRenderCount += 1;
      if (dom.modelGraphCanvas) state.modelGraphEditor.overviewCanvasWidth = modelGraphOverviewContainerWidth();
    };
    globalThis.__v151 = {
      bindModelGraphResizeObserver,
      scheduleModelGraphOverviewResize,
      modelGraphOverviewContainerWidth,
      modelGraphActiveUi,
      modelGraphDetailUi,
      modelGraphUi,
      modelGraphOverviewNodeSize,
      modelGraphOverviewTextWidth,
      reconcileModelGraphOverviewPositions,
      modelGraphOverviewRoutePlan,
      modelOverviewStackSummary,
      setModelGraphMode,
      selectModelGraphOverviewComponent,
      modelInlineLayout,
      modelInlineRoutePlan,
      renderAttentionLoweringProjection,
      dom,
      flushAnimationFrames() { while (globalThis.__animationFrames.length) globalThis.__animationFrames.shift()(0); },
      triggerModelResize: globalThis.__triggerModelResize,
      modelRenderCount() { return __modelRenderCount; },
      state,
    };`, context);
  return context.__v151;
}

test("the real Qwen3.8-27B overview uses repeat groups with complete authoritative templates", () => {
  const graph = graphFromFixture(qwen38Model());
  assert.equal(graph.operators.length, 229);
  assert.equal(graph.tensors.length, 262);
  assert.equal(graph.operators.filter((item) => item.op_kind === "layer_group").length, 32);
  const graphBefore = JSON.stringify(graph);
  const semanticBefore = ModelGraphCore.semanticProjection(graph);
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  const groupNodes = projection.nodes.filter((node) => node.kind === "repeat_group");

  assert.equal(projection.direction, "TB");
  assert.equal(projection.operator_first, true);
  assert.equal(projection.attributes.visual_projection_only, true);
  assert.equal(projection.attributes.repeat_group_projection, true);
  assert.equal(projection.attributes.leaf_operator_projection, false);
  assert.equal(projection.attributes.deduplicated, false);
  assert.ok(projection.nodes.length > 0);
  assert.equal(groupNodes.length, 1);
  const mainPattern = groupNodes[0];
  assert.equal(mainPattern.pattern_period, 2);
  assert.equal(mainPattern.pattern_repetitions, 16);
  assert.equal(mainPattern.repeat_count, 64);
  assert.equal(mainPattern.group_ids.length, 32);
  assert.deepEqual(mainPattern.pattern.map((item) => item.repeat), [3, 1]);
  assert.deepEqual(mainPattern.pattern.map((item) => item.mixer_kind), ["linear_attention", "full_attention"]);
  assert.equal(new Set(groupNodes.flatMap((node) => node.group_ids)).size, 32);
  groupNodes.forEach((node) => {
    assert.ok(node.repeat_count >= 1);
    assert.equal(node.expanded, false);
    assert.ok(node.inline_graph.authoritative);
    assert.ok(node.inline_graph.operators.length >= 6);
    assert.equal(new Set(node.inline_graph.operators.map((item) => item.operator_id)).size, node.inline_graph.operators.length);
    assert.ok(node.inline_graph.edges.every((edge) => edge.authoritative === true && edge.tensor_id));
    assert.equal(node.boundary.input_count, 1);
    assert.equal(node.boundary.output_count, 1);
  });
  assert.ok(projection.edges.some((edge) => edge.visual_only && /^×\d+$/.test(edge.label)));
  assert.ok(projection.edges.filter((edge) => !edge.visual_only).every((edge) => edge.representative_edge));
  assert.ok(graph.operators.some((item) => item.op_kind === "model_input"));
  assert.ok(graph.operators.some((item) => item.op_kind === "model_output"));
  assert.equal(JSON.stringify(graph), graphBefore);
  assert.deepEqual(ModelGraphCore.semanticProjection(graph), semanticBefore);
  assert.equal(ModelGraphCore.validateModelGraph(graph).valid, true);
  assert.doesNotMatch(JSON.stringify(projection), /encoder|cross[_ -]?attention/i);
});

test("repeat pattern summary names heterogeneous mixer blocks and their nested counts", () => {
  const helpers = appHelpers();
  assert.equal(helpers.modelOverviewStackSummary({
    pattern_repetitions: 16,
    pattern: [
      { mixer_kind: "linear_attention", ffn_kind: "dense", repeat: 3 },
      { mixer_kind: "full_attention", ffn_kind: "dense", repeat: 1 },
    ],
  }), "[线性注意力 Block ×3 + 全注意力 Block ×1] ×16");
  assert.equal(helpers.modelOverviewStackSummary({
    pattern_repetitions: 4,
    pattern: [{ mixer_kind: "full_attention", ffn_kind: "moe", repeat: 2 }],
  }), "全注意力 MoE Block ×2 ×4");
});

test("repeat-group boundary proxies map to real typed endpoints and preserve fan-out", () => {
  const graph = graphFromFixture(qwen38Model());
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  const operatorById = new Map(graph.operators.map((operator) => [operator.operator_id, operator]));
  const nodeById = new Map(projection.nodes.map((node) => [node.display_id, node]));

  projection.nodes.forEach((node) => {
    const memberIds = new Set(node.operator_ids);
    assert.ok(memberIds.size > 0);
    node.boundary_ports.forEach((endpoint) => {
      assert.ok(memberIds.has(endpoint.operator_id));
      const operator = operatorById.get(endpoint.operator_id);
      const realPort = operator?.ports.find((port) => port.port_id === endpoint.port_id);
      assert.ok(realPort, `${endpoint.operator_id}.${endpoint.port_id}`);
      assert.deepEqual(
        { direction: endpoint.direction, tensor_id: endpoint.tensor_id, dtype: endpoint.dtype, shape: endpoint.shape, layout: endpoint.layout },
        { direction: realPort.direction, tensor_id: realPort.tensor_id, dtype: realPort.dtype, shape: realPort.shape, layout: realPort.layout },
      );
      assert.ok(endpoint.proxy_port_id);
      assert.ok(endpoint.mapped_ports.length >= 1);
      endpoint.mapped_ports.forEach((mapped) => assert.ok(
        operatorById.get(mapped.operator_id)?.ports.some((port) => port.port_id === mapped.port_id && port.tensor_id === endpoint.tensor_id),
      ));
      assert.ok(["input", "output"].includes(endpoint.direction));
    });
  });
  projection.edges.filter((edge) => !edge.visual_only).forEach((edge) => {
    const sourceNode = nodeById.get(edge.source_id);
    const targetNode = nodeById.get(edge.target_id);
    assert.ok(sourceNode);
    assert.ok(targetNode);
    const representative = edge.representative_edge;
    assert.ok(representative, `${edge.source_id} -> ${edge.target_id}`);
    const sourceOperator = operatorById.get(representative.source.operator_id);
    const targetOperator = operatorById.get(representative.target.operator_id);
    const sourcePort = sourceOperator?.ports.find((port) => port.port_id === representative.source.port_id);
    const targetPort = targetOperator?.ports.find((port) => port.port_id === representative.target.port_id);
    assert.ok(sourceNode.operator_ids.includes(representative.source.operator_id));
    assert.ok(targetNode.operator_ids.includes(representative.target.operator_id));
    assert.equal(sourcePort?.direction, "output");
    assert.equal(targetPort?.direction, "input");
    assert.equal(sourcePort?.tensor_id, representative.source.tensor_id);
    assert.equal(targetPort?.tensor_id, representative.target.tensor_id);
    assert.ok(representative.source.tensor_id);
    assert.ok(representative.target.tensor_id);
    assert.ok(edge.tensor_ids.length > 0);
  });

  const dangling = ModelGraphCore.normalizeModelGraph({
    graph_id: "dangling-boundary",
    operators: [{
      operator_id: "source", op_kind: "custom_source", sequence_index: 0,
      ports: [{ port_id: "out0", direction: "output", tensor_id: "dangling", dtype: "fp16", shape: [1, 8], layout: "row_major" }],
    }],
    tensors: [{ tensor_id: "dangling", role: "activation", producer_operator_id: "source", consumer_operator_ids: [], dtype: "fp16", shape: [1, 8], layout: "row_major" }],
  });
  const danglingNode = ModelGraphCore.buildOverviewProjection(dangling).nodes[0];
  assert.equal(danglingNode.kind, "operator");
  assert.deepEqual(danglingNode.boundary_ports.map((item) => [item.operator_id, item.port_id]), [["source", "out0"]]);
});

test("repeat-group templates distinguish Dense, MoE, full, linear, and unknown top-level operators", () => {
  const dense = qwen38Model();
  dense.name = "Dense-4";
  dense.layer_specs = Array.from({ length: 4 }, (_, index) => ({ ...dense.layer_specs[3], layer_id: `dense-${index}` }));
  const denseProjection = ModelGraphCore.buildOverviewProjection(graphFromFixture(dense));
  const denseGroup = denseProjection.nodes.find((node) => node.kind === "repeat_group");
  assert.equal(denseGroup.repeat_count, 4);
  assert.equal(denseGroup.pattern[0].mixer_kind, "full_attention");
  assert.equal(denseGroup.pattern[0].ffn_kind, "dense");
  assert.ok(denseGroup.inline_graph.operators.some((item) => item.op_kind === "attention"));
  assert.ok(denseGroup.inline_graph.operators.some((item) => item.op_kind === "dense_mlp"));

  const moeLayers = Array.from({ length: 8 }, (_, index) => ({
    ...dense.layer_specs[0],
    layer_id: `moe-${index}`,
    kind: "moe",
    num_experts: 8,
    experts_per_token: 2,
    shared_expert_intermediate_size: 1024,
    shared_expert_gate: true,
  }));
  const moeProjection = ModelGraphCore.buildOverviewProjection(graphFromFixture({ name: "MoE-8", layer_specs: moeLayers }));
  const moeGroup = moeProjection.nodes.find((node) => node.kind === "repeat_group");
  assert.equal(moeGroup.repeat_count, 8);
  assert.equal(moeGroup.pattern[0].ffn_kind, "moe");
  assert.ok(moeGroup.inline_graph.operators.some((node) => node.op_kind === "moe_router"));
  assert.ok(moeGroup.inline_graph.operators.some((node) => node.op_kind === "shared_expert"));

  const unknown = ModelGraphCore.normalizeModelGraph({
    graph_id: "unknown-safe",
    operators: [
      { operator_id: "source", op_kind: "custom_source", sequence_index: 0, ports: [{ port_id: "out0", direction: "output", tensor_id: "custom", dtype: "fp16", shape: [1], layout: "logical" }] },
      { operator_id: "sink", op_kind: "custom_sink", sequence_index: 1, ports: [{ port_id: "in0", direction: "input", tensor_id: "custom", dtype: "fp16", shape: [1], layout: "logical" }] },
    ],
    tensors: [],
  });
  const fallback = ModelGraphCore.buildOverviewProjection(unknown);
  assert.deepEqual(fallback.nodes.map((node) => node.op_kind), ["custom_source", "custom_sink"]);
  assert.ok(fallback.nodes.every((node) => node.kind === "operator"));
  assert.equal(fallback.edges.length, 1);
});

test("typed MTP instances remain concrete top-level operators beside repeat groups", () => {
  const dense = qwen38Model();
  dense.name = "Dense-MTP";
  dense.layer_specs = dense.layer_specs.slice(3, 4).map((layer) => ({ ...layer, layer_id: "dense-0" }));
  const denseGraph = withTypedMtp(graphFromFixture(dense), 2, true);
  const semanticBefore = ModelGraphCore.semanticProjection(denseGraph);
  const denseProjection = ModelGraphCore.buildOverviewProjection(denseGraph);
  const predictions = denseProjection.nodes.filter((node) => node.op_kind === "mtp_prediction_layer");
  const prediction = predictions[0];
  const auxiliary = denseProjection.nodes.find((node) => node.op_kind === "mtp_aux_head");
  assert.equal(predictions.length, 2);
  assert.equal(prediction.kind, "operator");
  assert.equal(prediction.instance_count, 1);
  assert.equal(prediction.operator_ids.length, 1);
  assert.equal(auxiliary.kind, "operator");
  assert.equal(auxiliary.instance_count, 1);
  assert.equal(denseProjection.nodes.some((node) => node.kind === "mtp_proposer"), false);
  const branch = denseProjection.edges.find((edge) => edge.target_id === prediction.display_id);
  assert.equal(denseProjection.nodes.find((node) => node.display_id === branch.source_id).kind, "repeat_group");
  assert.deepEqual(branch.contracts[0].shape, ["B", "T", 5120]);
  assert.deepEqual(ModelGraphCore.semanticProjection(denseGraph), semanticBefore);

  const moe = qwen38Model();
  moe.name = "MoE-MTP";
  moe.layer_specs = moe.layer_specs.slice(3, 4).map((layer) => ({
    ...layer, layer_id: "moe-0", kind: "moe", num_experts: 8, experts_per_token: 2,
    shared_expert_intermediate_size: 1024, shared_expert_gate: true,
  }));
  const moeProjection = ModelGraphCore.buildOverviewProjection(withTypedMtp(graphFromFixture(moe), 1, true));
  const moeGroup = moeProjection.nodes.find((node) => node.kind === "repeat_group");
  assert.ok(moeGroup.inline_graph.operators.some((node) => node.op_kind === "moe_router"));
  assert.ok(moeGroup.inline_graph.operators.some((node) => node.op_kind === "shared_expert"));
  assert.equal(moeProjection.nodes.find((node) => node.op_kind === "mtp_prediction_layer").instance_count, 1);
  assert.equal(moeProjection.nodes.some((node) => node.kind === "mtp_proposer"), false);
});

test("nested MTP configuration remains separate concrete operators in the overview", () => {
  const graph = graphFromFixture({ ...qwen38Model(), mtp: { prediction_layers: 4, auxiliary_head: true } });
  assert.equal(graph.operators.filter((item) => item.op_kind === "mtp_prediction_layer").length, 4);
  assert.equal(graph.operators.filter((item) => item.op_kind === "mtp_aux_head").length, 1);
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  assert.equal(projection.nodes.some((node) => node.kind === "mtp_proposer"), false);
  assert.equal(projection.nodes.filter((node) => node.op_kind === "mtp_prediction_layer").length, 4);
  assert.ok(projection.nodes.filter((node) => node.op_kind === "mtp_prediction_layer").every((node) => node.instance_count === 1 && node.operator_ids.length === 1));
  assert.equal(projection.nodes.find((node) => node.op_kind === "mtp_aux_head").instance_count, 1);
});

test("Dense, MoE, full/linear pattern, and concrete MTP signatures remain distinguishable", () => {
  const base = qwen38Model();
  base.layer_specs = base.layer_specs.slice(3, 4).map((layer) => ({ ...layer, layer_id: "layer-0" }));
  const denseGraph = graphFromFixture({ ...base, name: "dense" });
  const moeGraph = graphFromFixture({
    ...base,
    name: "moe",
    layer_specs: base.layer_specs.map((layer) => ({ ...layer, kind: "moe", num_experts: 8, experts_per_token: 2 })),
  });
  const leafSignature = (graph) => {
    const projection = ModelGraphCore.buildOverviewProjection(graph);
    const patterns = projection.nodes.filter((node) => node.kind === "repeat_group").flatMap((node) => node.pattern);
    return {
      mixers: new Set(patterns.map((item) => item.mixer_kind).filter(Boolean)),
      ffnKinds: new Set(patterns.map((item) => item.ffn_kind).filter(Boolean)),
      mtpKinds: new Set(projection.nodes.filter((node) => node.op_kind.startsWith("mtp_")).map((node) => node.op_kind)),
    };
  };
  assert.deepEqual(Array.from(leafSignature(denseGraph).mixers), ["full_attention"]);
  assert.deepEqual(Array.from(leafSignature(denseGraph).ffnKinds), ["dense"]);
  assert.deepEqual(Array.from(leafSignature(moeGraph).mixers), ["full_attention"]);
  assert.deepEqual(Array.from(leafSignature(moeGraph).ffnKinds), ["moe"]);
  assert.deepEqual(Array.from(leafSignature(denseGraph).mtpKinds), []);
  assert.deepEqual(Array.from(leafSignature(withTypedMtp(denseGraph, 2, true)).mtpKinds), ["mtp_prediction_layer", "mtp_aux_head"]);
  const hybrid = leafSignature(graphFromFixture(qwen38Model()));
  assert.deepEqual(Array.from(hybrid.mixers).sort(), ["full_attention", "linear_attention"]);
});

test("overview group selection opens repeat details without mode switching or dirtying the scenario", () => {
  const model = qwen38Model();
  const graph = graphFromFixture(model);
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  const firstLeaf = projection.nodes.find((node) => node.group_ids?.includes("block-group-000"));
  const secondLeaf = projection.nodes.find((node) => node.representative_operator_id === "final_norm");
  const semanticBefore = ModelGraphCore.semanticProjection(graph);
  const ui = appHelpers();
  ui.state.scenario = { model: { ...model, graph } };
  const root = ui.modelGraphUi();
  assert.equal(root.mode, "overview");
  root.overview.positions[firstLeaf.display_id] = { x: 111, y: 222 };
  root.overview.viewport = { x: 7, y: 9, scale: 0.8 };
  ui.state.dirty = false;
  ui.setModelGraphMode("focus", { groupId: "block-group-000" });
  assert.equal(ui.modelGraphUi().mode, "overview");
  assert.equal(ui.state.modelGraphEditor.selectedOverviewId, firstLeaf.display_id);
  assert.equal(ui.state.modelGraphEditor.selectedOperatorId, null);
  assert.ok(Array.isArray(ui.modelGraphUi().overview.collapsed_groups));
  assert.equal(ui.selectModelGraphOverviewComponent(secondLeaf.representative_operator_id), true);
  assert.equal(ui.state.modelGraphEditor.selectedOverviewId, secondLeaf.display_id);
  assert.equal(ui.state.dirty, false);
  assert.deepEqual(ui.modelGraphActiveUi().positions[firstLeaf.display_id], { x: 111, y: 222 });
  assert.deepEqual(ModelGraphCore.semanticProjection(ui.state.scenario.model.graph), semanticBefore);
});

test("authoritative operator projections retain tensor edges beside visual-only repeat edges", () => {
  const model = qwen38Model();
  model.layer_specs = [{
    ...model.layer_specs[3],
    layer_id: "moe-0",
    kind: "moe",
    num_experts: 8,
    experts_per_token: 2,
    shared_expert_intermediate_size: 1024,
    shared_expert_gate: true,
  }];
  const graph = withTypedMtp(graphFromFixture(model), 2, true);
  const groupId = graph.operators.find((operator) => operator.op_kind === "layer_group").operator_id;
  const attentionId = graph.operators.find((operator) => operator.op_kind === "attention").operator_id;
  const inline = authoritativeGroupProjection(graph, groupId, [attentionId, `${groupId}.moe`]);
  const mtpIds = graph.operators.filter((operator) => operator.op_kind.startsWith("mtp_")).map((operator) => operator.operator_id);
  const mtpInline = ModelGraphCore.authoritativeOperatorProjection(graph, mtpIds, { expanded_components: mtpIds });

  assert.equal(inline.authoritative, true);
  assert.ok(inline.residual_edges.length >= 2);
  assert.ok(inline.residual_edges.every((edge) => edge.authoritative === true && edge.tensor_id));
  assert.ok(inline.edges.every((edge) => edge.authoritative === true && edge.tensor_id));
  assert.ok(Object.values(inline.attention_projections).every((item) => item.derived_only === true));
  assert.ok(mtpInline.edges.every((edge) => edge.authoritative === true && edge.tensor_id));
  assert.ok(mtpInline.boundary_ports.every((port) => mtpIds.includes(port.operator_id)));
  assert.match(app, /data-visual-only="true"/);
  assert.match(app, /Visual repeat indicator only; never written to the authoritative DAG/);
  assert.match(app, /class="model-inline-edge \$\{modelGraphRouteClass\(route\)\}\$\{residual \? " is-residual is-skip-route" : ""\}" data-authoritative="true"/);
  assert.match(app, /ModelGraph\.modelGraphRouteEdge\(sourceRect, targetRect, obstacles/);
  assert.match(app, /preferOuter: Boolean\(layout\.constrained && residual\)/);
  assert.match(app, /const canvasWidth = Number\(dom\.modelGraphCanvas\?\.clientWidth\)/);
  assert.match(app, /ModelGraph\.overviewResponsiveLayout\(projection, canvasWidth, scale, \{ nodeSizes \}\)/);
  assert.match(css, /\.model-graph-canvas\[data-mode="overview"\]\s*\{[^}]*overflow-x:\s*hidden/);
  assert.match(app, /if \(state\.view === "model"\) requestAnimationFrame\(renderModelGraph\)/);
  assert.doesNotMatch(app, /state\.modelGraphEditor\.overviewLayoutKey = "";\s*renderModelGraph\(\);/);
  assert.match(app, /Existing coordinates are user-owned until the explicit “整理” action/);
  assert.ok(app.includes('<title>${escapeHtml(modelOverviewContractTitle(edge))}</title>'));
});

test("authoritative inline layout remains bounded and derived-only when built directly", () => {
  const model = qwen38Model();
  model.layer_specs = [{
    ...model.layer_specs[3],
    layer_id: "moe-0",
    kind: "moe",
    num_experts: 8,
    experts_per_token: 2,
    shared_expert_intermediate_size: 1024,
    shared_expert_gate: true,
  }];
  const graph = graphFromFixture(model);
  const groupId = graph.operators.find((operator) => operator.op_kind === "layer_group").operator_id;
  const attentionId = graph.operators.find((operator) => operator.op_kind === "attention").operator_id;
  const inline = authoritativeGroupProjection(graph, groupId, [attentionId, `${groupId}.moe`]);
  const helpers = appHelpers();
  const layout = helpers.modelInlineLayout(inline, 460);
  const clamped = helpers.modelInlineLayout(inline, 460, { [attentionId]: { x: -900, y: 9999 } });
  assert.ok(layout.operators.some((item) => item.op_kind === "moe_router"));
  assert.ok(layout.operators.some((item) => item.op_kind === "shared_expert"));
  layout.operators.forEach((operator) => {
    const point = layout.positions[operator.operator_id];
    assert.ok(point.x >= 14);
    assert.ok(point.x + layout.nodeWidth <= 446.001);
    assert.ok(point.y >= 14);
    assert.ok(point.y + layout.nodeHeight <= layout.graphHeight - 13.999);
  });
  layout.edges.forEach((edge) => {
    assert.ok(layout.positions[edge.source_operator_id].y <= layout.positions[edge.target_operator_id].y);
  });
  assert.equal(clamped.positions[attentionId].x, 14);
  assert.equal(clamped.positions[attentionId].y, clamped.graphHeight - clamped.nodeHeight - 14);
  layout.inline = inline;
  const routes = helpers.modelInlineRoutePlan(layout);
  routes.entries.forEach((entry) => {
    const route = routes.routes.get(entry.edge.edge_id);
    const obstacles = Array.from(routes.rectByNode.values()).filter((rect) => ![entry.sourceNodeId, entry.targetNodeId].includes(rect.id));
    for (let index = 1; index < route.points.length; index += 1) {
      obstacles.forEach((obstacle) => assert.equal(
        ModelGraphCore.modelGraphSegmentHitsRect(route.points[index - 1], route.points[index], obstacle),
        false,
        `${entry.edge.edge_id} crosses ${obstacle.id}`,
      ));
    }
  });
  const routeRects = Array.from(routes.rectByNode.values());
  const minRouteX = Math.min(...routeRects.map((rect) => rect.x));
  const maxRouteX = Math.max(...routeRects.map((rect) => rect.x + rect.width));
  assert.ok(routes.entries.filter((entry) => routes.residualIds.has(entry.edge.edge_id)).every((entry) => {
    const route = routes.routes.get(entry.edge.edge_id);
    return route.kind === "orthogonal" && route.points.some((point) => point.x < minRouteX || point.x > maxRouteX);
  }));
  assert.ok(Object.values(inline.attention_projections).every((item) => item.derived_only === true));
  const attentionProjection = Object.values(inline.attention_projections)[0];
  const attentionMarkup = helpers.renderAttentionLoweringProjection(attentionProjection);
  assert.equal((attentionMarkup.match(/data-derived-edge=/g) || []).length, attentionProjection.edges.length);
  assert.equal((attentionMarkup.match(/data-derived-id=/g) || []).length, attentionProjection.nodes.length);
  assert.doesNotMatch(attentionMarkup, /data-model-drag-kind|data-model-select-operator/);
});

test("model drag and pan pointermove is rAF-coalesced and updates transforms plus affected routes without full render", () => {
  const moveSource = app.slice(app.indexOf("function moveModelGraphPointer"), app.indexOf("function endModelGraphPointer"));
  const frameSource = app.slice(app.indexOf("function modelGraphApplyInteractionFrame"), app.indexOf("function scheduleModelGraphInteractionFrame"));
  const routeSource = app.slice(app.indexOf("function refreshModelGraphDragRoutes"), app.indexOf("function modelGraphApplyInteractionFrame"));
  const delegateSource = app.slice(app.indexOf("function bindModelGraphDynamicEvents"), app.indexOf("function syncModelGraphControls"));
  const saveLayoutSource = app.slice(app.indexOf("function saveModelGraphLayout"), app.indexOf("function applyModelGraphSemanticChange"));
  assert.doesNotMatch(moveSource, /renderModelGraph\s*\(/);
  assert.doesNotMatch(moveSource, /modelGraphActiveUi\s*\(/);
  assert.doesNotMatch(frameSource, /modelGraphActiveUi\s*\(/);
  assert.match(moveSource, /scheduleModelGraphInteractionFrame\(event\)/);
  assert.match(frameSource, /drag\.element\.style\.transform/);
  assert.match(frameSource, /refreshModelGraphDragRoutes\(drag\)/);
  assert.match(routeSource, /refreshModelGraphDomRoutes\(/);
  assert.match(frameSource, /dom\.modelGraphWorld\.style\.transform/);
  assert.match(app, /globalThis\.requestAnimationFrame \|\| \(\(callback\) => setTimeout\(callback, 16\)\)/);
  assert.match(delegateSource, /dataset\.modelGraphDelegated/);
  assert.doesNotMatch(delegateSource, /\.forEach\(\(button\) => button\.addEventListener/);
  assert.match(app, /performance\.fullRenders \+= 1/);
  assert.doesNotMatch(saveLayoutSource, /state\.dirty\s*=|dirtyMark/);
  assert.match(saveLayoutSource, /localStorage\.setItem\(STORAGE_SCENARIO/);
  assert.match(css, /\.model-inline-port\[data-model-port-side="top"\]/);
  assert.match(css, /\.model-overview-port\[data-model-port-side="bottom"\]/);
});

test("overview ResizeObserver reapplies the final width without invalidating manual positions", () => {
  const helpers = appHelpers();
  helpers.state.view = "model";
  helpers.state.modelGraphEditor.overviewCanvasWidth = 980;
  helpers.state.modelGraphEditor.overviewLayoutKey = "initial-wide-layout";
  helpers.dom.modelGraphCanvas = { clientWidth: 964, offsetWidth: 980, dataset: { mode: "overview" } };
  helpers.bindModelGraphResizeObserver();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 0);

  helpers.dom.modelGraphCanvas.clientWidth = 666;
  helpers.dom.modelGraphCanvas.offsetWidth = 682;
  helpers.triggerModelResize();
  helpers.triggerModelResize();
  assert.equal(helpers.state.modelGraphEditor.overviewResizeTarget, 682);
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 1);
  assert.equal(helpers.state.modelGraphEditor.overviewCanvasWidth, 682);
  assert.equal(helpers.state.modelGraphEditor.overviewLayoutKey, "initial-wide-layout");

  helpers.triggerModelResize();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 1, "stable canvas width must not start a render loop");
  assert.match(app, /new globalThis\.ResizeObserver/);
  assert.match(app, /if \(editor\.drag \|\| editor\.pan \|\| editor\.connectPointer\) \{/);
});

test("material overview width reduction safely fits the viewport without moving model nodes", () => {
  const graph = graphFromFixture({ ...qwen38Model(), layer_specs: qwen38Model().layer_specs.slice(0, 4) });
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  const helpers = appHelpers();
  helpers.state.scenario = { model: { graph } };
  helpers.state.view = "model";
  helpers.dom.modelGraphCanvas = { clientWidth: 666, offsetWidth: 682, dataset: { mode: "overview" } };
  helpers.state.modelGraphEditor.overviewCanvasWidth = 980;
  const overview = helpers.modelGraphUi().overview;
  overview.positions = Object.fromEntries(projection.nodes.map((node, index) => [node.display_id, { x: 40 + index * 180, y: 60 + index * 80 }]));
  overview.positions[projection.nodes.at(-1).display_id].x = 900;
  overview.viewport = { x: 240, y: 22, scale: 1 };
  const positionsBefore = JSON.stringify(overview.positions);

  helpers.scheduleModelGraphOverviewResize();
  helpers.flushAnimationFrames();

  const fittedOverview = helpers.modelGraphUi().overview;
  assert.ok(fittedOverview.viewport.scale < 1, "a stale wide-screen scale should shrink to the visible canvas");
  assert.ok(fittedOverview.viewport.x >= 16, "the fitted graph should remain inside the left safe margin");
  assert.equal(fittedOverview.viewport.y, 22, "a responsive width reduction preserves the user's vertical view");
  assert.equal(JSON.stringify(fittedOverview.positions), positionsBefore, "responsive fitting must not rewrite user-owned node coordinates");
  assert.equal(helpers.state.modelGraphEditor.overviewCanvasWidth, 682);
});

test("overview resize during drag is deferred and never clears the manual-layout key", () => {
  const helpers = appHelpers();
  helpers.state.view = "model";
  helpers.state.modelGraphEditor.overviewCanvasWidth = 700;
  helpers.state.modelGraphEditor.overviewLayoutKey = "manual-layout";
  helpers.state.modelGraphEditor.drag = { pointerId: 7 };
  helpers.dom.modelGraphCanvas = { clientWidth: 620, offsetWidth: 636, dataset: { mode: "overview" } };

  helpers.bindModelGraphResizeObserver();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 0);
  assert.equal(helpers.state.modelGraphEditor.overviewResizeTarget, 636);
  assert.equal(helpers.state.modelGraphEditor.overviewLayoutKey, "manual-layout");

  helpers.state.modelGraphEditor.drag = null;
  helpers.scheduleModelGraphOverviewResize();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 1);
  assert.equal(helpers.state.modelGraphEditor.overviewCanvasWidth, 636);
  assert.equal(helpers.state.modelGraphEditor.overviewLayoutKey, "manual-layout");
});

test("overview reconciliation preserves existing coordinates and places only new leaves", () => {
  const graph = graphFromFixture({ ...qwen38Model(), layer_specs: qwen38Model().layer_specs.slice(0, 2) });
  const projection = ModelGraphCore.buildOverviewProjection(graph);
  const helpers = appHelpers();
  helpers.state.scenario = { model: { graph } };
  helpers.dom.modelGraphCanvas = { clientWidth: 760, offsetWidth: 776, dataset: { mode: "overview" } };
  const overview = helpers.modelGraphUi().overview;
  const firstId = projection.nodes[0].display_id;
  overview.positions[firstId] = { x: 321, y: 123 };

  const reconciled = helpers.reconcileModelGraphOverviewPositions(projection, overview);
  assert.equal(reconciled.initialized, true);
  assert.equal(reconciled.changed, true);
  assert.deepEqual(overview.positions[firstId], { x: 321, y: 123 });
  assert.equal(Object.keys(overview.positions).length, projection.nodes.length);

  const snapshot = JSON.stringify(overview.positions);
  helpers.state.settings.fontScale = 160;
  helpers.dom.modelGraphCanvas.clientWidth = 520;
  helpers.dom.modelGraphCanvas.offsetWidth = 536;
  const measuredOnly = helpers.reconcileModelGraphOverviewPositions(projection, overview);
  assert.equal(measuredOnly.changed, false);
  assert.equal(JSON.stringify(overview.positions), snapshot);
});

test("nearby MTP leaves use the local shortest route instead of an outer detour", () => {
  const helpers = appHelpers();
  const projection = {
    nodes: [
      {
        display_id: "mtp-prediction", kind: "operator", op_kind: "mtp_prediction_layer", label: "多 Token 预测层",
        boundary_ports: [{ operator_id: "prediction", port_id: "out0", direction: "output", tensor_id: "mtp-next" }],
      },
      {
        display_id: "mtp-aux", kind: "operator", op_kind: "mtp_aux_head", label: "多 Token 辅助头",
        boundary_ports: [{ operator_id: "aux", port_id: "in0", direction: "input", tensor_id: "mtp-next" }],
      },
    ],
    edges: [{ source_id: "mtp-prediction", target_id: "mtp-aux", tensor_ids: ["mtp-next"], contracts: [] }],
  };
  const positions = { "mtp-prediction": { x: 100, y: 80 }, "mtp-aux": { x: 100, y: 145 } };
  const plan = helpers.modelGraphOverviewRoutePlan(projection, positions);
  const route = plan.routes.get("mtp-prediction->mtp-aux");
  const sourceRect = plan.rectByNode.get("mtp-prediction");
  const targetRect = plan.rectByNode.get("mtp-aux");
  const minX = Math.min(sourceRect.x, targetRect.x);
  const maxX = Math.max(sourceRect.x + sourceRect.width, targetRect.x + targetRect.width);
  assert.equal(route.kind, "smooth");
  assert.ok(route.points.every((point) => point.x >= minX && point.x <= maxX));
  assert.doesNotMatch(route.path, /Q /, "a clear nearby pair must not be sent around an outer rail");
});

test("overview scrollbar changes do not masquerade as a responsive resize or reset zoom", () => {
  const helpers = appHelpers();
  helpers.state.view = "model";
  helpers.state.modelGraphEditor.overviewCanvasWidth = 700;
  helpers.state.modelGraphEditor.overviewLayoutKey = "stable-layout";
  helpers.state.scenario = { model: { graph: { attributes: { ui: { mode: "overview", overview: { viewport: { x: 12, y: 18, scale: 0.6 } } } } } } };
  helpers.dom.modelGraphCanvas = { clientWidth: 684, offsetWidth: 700, dataset: { mode: "overview" } };
  helpers.bindModelGraphResizeObserver();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 0);

  helpers.dom.modelGraphCanvas.clientWidth = 700;
  helpers.triggerModelResize();
  helpers.triggerModelResize();
  helpers.flushAnimationFrames();
  assert.equal(helpers.modelRenderCount(), 0, "scrollbar-only clientWidth changes must not relayout");
  assert.equal(helpers.state.modelGraphEditor.overviewLayoutKey, "stable-layout");
  assert.equal(helpers.state.scenario.model.graph.attributes.ui.overview.viewport.scale, 0.6);
  assert.match(css, /model-graph-canvas\[data-mode="overview"\][^}]*overflow-y:\s*auto[^}]*scrollbar-gutter:\s*stable/);
});

test("model overview renders collapsible repeat groups, authoritative templates, and editable concrete ports", () => {
  assert.match(html, /class="model-graph-mode-badge"[\s\S]*id="modelGraphBackButton"[^>]*hidden/);
  assert.match(html, /<dl class="form-strip model-summary-strip" id="modelMetaForm" aria-label="只读模型摘要"><\/dl>/);
  assert.match(html, /id="modelGraphConnectButton"/);
  assert.doesNotMatch(html, /id="modelGraph(?:OverviewButton|DetailButton|AddKind|AddButton|DuplicateButton|DeleteButton)"/);
  assert.doesNotMatch(app, /data-model-field/);
  assert.match(app, /data-model-summary-field/);
  const overviewRender = app.slice(app.indexOf("function renderModelGraphOverview(graph, ui)"), app.indexOf("function renderModelGraph()"));
  const delegatedEvents = app.slice(app.indexOf("function bindModelGraphDynamicEvents"), app.indexOf("function syncModelGraphControls"));
  assert.match(overviewRender, /data-model-select-overview/);
  assert.match(overviewRender, /data-model-toggle-overview-group/);
  assert.match(overviewRender, /renderModelInlineAuthoritativeGraph/);
  assert.match(overviewRender, /node\.kind === "repeat_group"/);
  assert.match(overviewRender, /repeatLabel/);
  assert.match(overviewRender, /modelOverviewPortMarkup\(graph, node, routePlan\.placements\)/);
  assert.match(overviewRender, /renderModelGraphOverviewInspector\(graph, projection\)/);
  const inspectorSource = app.slice(app.indexOf("function renderModelGraphOverviewInspector"), app.indexOf("function renderModelGraphOverview(graph, ui)"));
  assert.match(inspectorSource, /instanceCount/);
  assert.match(inspectorSource, /node\.pattern_period/);
  assert.match(inspectorSource, /modelOverviewStackSummary\(node\)/);
  assert.match(inspectorSource, /node\.has_semantic_overrides/);
  assert.match(inspectorSource, /具体端口|Concrete ports/);
  assert.doesNotMatch(inspectorSource, /node\.kind === "decoder_stack"|node\.kind === "mtp_proposer"/);
  assert.match(delegatedEvents, /selectModelGraphOverviewComponent\(overviewButton\.dataset\.modelSelectOverview\)/);
  assert.match(delegatedEvents, /overview\.collapsed_groups = Array\.from\(collapsed\)/);
  assert.match(app, /ui\.mode = "overview"/);
  assert.match(app, /ui\.overview\.collapsed_groups = overviewCollapsed/);
  assert.match(app, /ModelGraph\.updatePortContract/);
  assert.match(app, /ModelGraph\.validateModelGraph/);
  assert.match(app, /data-model-port-mapped/);
  assert.match(app, /asArray\(target\.mappedPorts\)/);
  assert.match(app, /ModelGraph\.connect\(nextGraph, source, mapped\)/);
  assert.match(app, /selectedOverviewId/);
  assert.match(html, /id="modelGraphInspectorTitle">未选择组件</);
  assert.match(html, /class="model-graph-mode-badge"><strong>结构总览<\/strong>/);
  assert.equal((html.match(/>结构总览</g) || []).length, 1);
  assert.doesNotMatch(html, /class="model-graph-mode-badge"><strong>结构总览<\/strong><small>/);
  assert.match(app, /派生视图 \/ 不新增执行 IR/);
  assert.match(app, /const MODEL_OVERVIEW_HORIZONTAL_EXTRA = 20/);
  assert.match(app, /labelWidth \+ MODEL_OVERVIEW_HORIZONTAL_EXTRA/);
  assert.match(app, /modelGraphOverviewNodeSize\(node\)/);
  assert.match(css, /\.model-overview-select-button \{[^}]*padding: calc\(4px \* var\(--layout-scale\)\) 9px/);
  assert.match(css, /\.model-overview-node\.is-repeat-group\.is-expanded\s*\{[^}]*border:\s*1px dashed var\(--model-kind-accent\);[^}]*background:\s*transparent;/s);
  assert.match(css, /\.model-overview-node\.is-repeat-group\.is-selected\s*\{[^}]*border:\s*2px solid #72d5df;[^}]*border-color:\s*var\(--cyan-bright\);[^}]*background:\s*#173038;/s);
  assert.match(css, /\.model-overview-inline-expansion\s*\{[^}]*background:\s*transparent;/s);
  assert.match(css, /\.model-graph-edge\.is-visual-repeat[^}]*stroke-dasharray/);
  assert.match(css, /\.model-overview-node\.is-selected[^}]*border:\s*2px solid #72d5df[^}]*background:\s*#173038/s);
  assert.match(css, /\.model-overview-select-button > strong \{[^}]*max-width: none;[^}]*overflow: visible;[^}]*white-space: nowrap;/s);
  assert.doesNotMatch(css, /\.model-overview-select-button > strong \{[^}]*text-overflow:\s*ellipsis/s);
  const helpers = appHelpers();
  const shortLabel = "x";
  const longLabel = "a much longer custom operator";
  const shortNode = { kind: "operator", op_kind: "custom", label: shortLabel, boundary_ports: [] };
  const longNode = { kind: "operator", op_kind: "custom", label: longLabel, boundary_ports: [] };
  assert.equal(helpers.modelGraphOverviewNodeSize(shortNode).width, helpers.modelGraphOverviewTextWidth(shortLabel) + 20);
  assert.equal(helpers.modelGraphOverviewNodeSize(longNode).width, helpers.modelGraphOverviewTextWidth(longLabel) + 20);
  assert.ok(helpers.modelGraphOverviewNodeSize(longNode).width > helpers.modelGraphOverviewNodeSize(shortNode).width);
  assert.match(app, /canvasWidth > 0 \? String\(Math\.round\(Math\.max\(1, canvasWidth - 32\)\)\) : "hidden"/);
  assert.match(app, /if \(!dragHandle && event\.target\.closest\?\.\('button, input, select, textarea, a, \[role="button"\]'\)\) return;/);
  assert.match(app, /if \(result\.connected\) \{\s*editor\.connectSource = null;\s*editor\.connectMode = false;/);
  assert.match(app, /timeout_kind: "frontend_watchdog"/);
  for (const stateClass of ["source", "compatible", "incompatible", "unavailable"]) {
    assert.match(css, new RegExp(`\\.model-graph-port\\.is-connect-${stateClass}`));
  }

  for (const section of ["request-generation", "continuous-batching", "mtp"]) {
    assert.ok(app.includes(`data-workload-section="${section}"`), section);
  }
  assert.match(html, /data-workload-section="explicit-requests"/);
  assert.match(app, /uiText\("请求生成", "Request Generation"\)/);
  assert.match(app, /uiText\("连续批处理与调度", "Continuous Batching & Scheduling"\)/);
  assert.match(app, /uiText\("多 Token 预测（MTP）", "Multi-Token Prediction \(MTP\)"\)/);
  assert.match(app, /class="workload-field-label"/);
  assert.match(app, /hydrateConceptHelp\(dom\.workloadMetaForm\)/);
  for (const key of ["scheduler", "batched_tokens", "preemption", "mtp", "acceptance_rate", "synthetic_prompt_tokens", "synthetic_output_tokens"]) {
    assert.ok(app.includes(`helpKey: "${key}"`) || app.includes(`"${key}")`), key);
  }

  assert.match(css, /\.workload-field-grid\s*>\s*\.field\s*\{[^}]*grid-row:\s*span 2;[^}]*grid-template-rows:\s*subgrid;/s);
  assert.match(css, /\.workload-request-table-shell,\s*\.request-result-table-shell\s*\{[^}]*overflow-x:\s*auto;/s);
  assert.match(css, /\.request-input-table\s*\{[^}]*min-width:\s*calc\(980px \* var\(--layout-scale\)\);/s);
  assert.match(css, /\.request-input-table\s*\{[^}]*table-layout:\s*fixed;/s);
  assert.match(css, /\.request-result-table\s*\{[^}]*table-layout:\s*fixed;/s);
  assert.match(css, /:is\(\.request-input-table, \.request-result-table\) th\.concept-help-title\s*\{[^}]*display:\s*table-cell;/s);
  const requestInputTable = /<table class="data-table request-input-table">([\s\S]*?)<\/table>/u.exec(html)?.[1] || "";
  const requestResultTable = /<table class="data-table request-result-table">([\s\S]*?)<\/table>/u.exec(html)?.[1] || "";
  assert.equal((requestInputTable.match(/<col class="request-input-col-/gu) || []).length, 8);
  assert.equal((requestResultTable.match(/<col class="request-result-col-/gu) || []).length, 9);
  assert.match(css, /\.request-input-col-deadline\s*\{[^}]*width:/s);
  assert.match(css, /\.request-result-col-reason\s*\{[^}]*width:/s);
  assert.match(css, /:root\[data-font-band="extreme"\] \.workload-config-grid,[\s\S]*:root\[data-font-band="extreme"\] \.workload-field-grid\s*\{\s*grid-template-columns:\s*minmax\(0,\s*1fr\);/);
  assert.match(css, /:root\[data-font-band="extreme"\] #view-workload\s*\{[^}]*overflow-x:\s*hidden;/s);
  assert.match(css, /minmax\(min\(100%,\s*calc\(178px \* var\(--layout-scale\)\)\),\s*1fr\)/);
});
