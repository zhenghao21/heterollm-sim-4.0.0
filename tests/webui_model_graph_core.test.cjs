"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const Core = require("../src/heterollm_sim/webui/model-graph-core.js");

function graphFromFixture({ layer_specs: layerSpecs, ...options }) {
  return Core.buildModelGraphFromLayerSpecs(layerSpecs, options);
}

function layer(id, overrides = {}) {
  return {
    schema_version: "1.0",
    layer_id: id,
    kind: "dense",
    hidden_size: 512,
    intermediate_size: 2048,
    attention_heads: 8,
    kv_heads: 8,
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
    metadata: { pattern_index: 0, note: id },
    ...overrides,
  };
}

function editablePair(sourceShape = ["B", "T", 512], targetShape = ["B", "T", "H"]) {
  return Core.normalizeModelGraph({
    graph_id: "pair",
    operators: [
      {
        operator_id: "source", op_kind: "input", sequence_index: 0,
        ports: [{ port_id: "out0", direction: "output", tensor_id: "source.output", dtype: "bf16", shape: sourceShape, layout: "logical" }],
      },
      {
        operator_id: "target", op_kind: "dense_mlp", sequence_index: 1,
        ports: [{ port_id: "in0", direction: "input", tensor_id: "target.input", dtype: "bfloat16", shape: targetShape, layout: "logical" }],
      },
    ],
    tensors: [], transforms: [], attributes: { ui: { positions: {} } },
  });
}

test("UMD module publishes the same frozen API to CommonJS and globalThis", () => {
  assert.equal(globalThis.ModelGraphCore, Core);
  assert.equal(Object.isFrozen(Core), true);
  assert.equal(Core.GRAPH_VERSION, 1);
  for (const name of ["normalizeGraph", "validateTransform", "portBezierPath", "routeEdge", "startConnectionPreview", "layoutGraph"]) {
    assert.equal(Core[name], undefined, `${name} compatibility alias must not survive`);
  }
});

test("layer specs become a deterministic backend-compatible semantic graph and round-trip", () => {
  const model = {
    name: "tiny-dense", architecture: "transformer", vocabulary_size: 32000,
    max_sequence_length: 4096, embedding_weight_bytes: 1024,
    layer_specs: [layer("layer-000"), layer("layer-001")],
  };
  const first = graphFromFixture(model);
  const second = graphFromFixture(model);
  assert.deepEqual(first, second);
  assert.equal(first.graph_id, "tiny-dense");
  assert.equal(first.attributes.authoritative, true);
  assert.ok(first.operators.some((item) => item.op_kind === "attention"));
  assert.ok(first.operators.some((item) => item.op_kind === "dense_mlp"));
  const logits = first.tensors.find((item) => item.tensor_id === "logits");
  assert.equal(logits.role, "activation");
  assert.equal(logits.producer_operator_id, "lm_head");
  assert.deepEqual(logits.consumer_operator_ids, ["output"]);
  const group = first.operators.find((item) => item.op_kind === "layer_group");
  assert.equal(group.parameters.repeat, 2);
  assert.deepEqual(group.parameters.layer_ids, ["layer-000", "layer-001"]);
  assert.equal(first.operators.every((item) => item.ports.every((port) => port.port_id && port.tensor_id && port.dtype && port.layout)), true);
  const projected = Core.graphToLayerSpecs(first);
  assert.equal(projected.ok, true);
  assert.deepEqual(projected.layer_specs, model.layer_specs);
});

test("model summaries are derived only from graph-owned contracts", () => {
  const graph = graphFromFixture({
    name: "graph-summary",
    architecture: "decoder_only_transformer",
    vocabulary_size: 32000,
    max_sequence_length: 4096,
    embedding_weight_bytes: 1024,
    layer_specs: [layer("layer-000")],
  });
  assert.deepEqual(Core.modelGraphAuthoringSummary(graph), {
    architecture: "decoder_only_transformer",
    vocabulary_size: 32000,
    max_sequence_length: 4096,
    embedding_weight_bytes: 1024,
  });

  const conflicting = structuredClone(graph);
  conflicting.attributes.symbols.V = 64000;
  assert.throws(
    () => Core.modelGraphAuthoringSummary(conflicting),
    /vocabulary_size.*冲突/,
  );
});

test("MoE layer specs materialize router, experts, shared expert, and combine components", () => {
  const graph = graphFromFixture({
    name: "tiny-moe",
    layer_specs: [layer("moe-0", { kind: "moe", num_experts: 8, experts_per_token: 2, shared_expert_intermediate_size: 1024, shared_expert_gate: true })],
  });
  const kinds = new Set(graph.operators.map((item) => item.op_kind));
  assert.ok(["moe_router", "moe_experts", "shared_expert", "moe_combine"].every((kind) => kinds.has(kind)));
  assert.equal(Core.graphToLayerSpecs(graph).layer_specs[0].num_experts, 8);
});

test("nested MTP configuration materializes typed prediction and auxiliary operators", () => {
  const graph = graphFromFixture({
    name: "tiny-mtp", vocabulary_size: 32000,
    mtp: {
      prediction_layers: 2,
      auxiliary_head: true,
      prediction_layer_weight_bytes: 1234,
      auxiliary_head_weight_bytes: 5678,
    },
    layer_specs: [layer("mtp-backbone")],
  });
  const mtp = graph.operators.filter((item) => item.op_kind.startsWith("mtp_"));
  assert.deepEqual(mtp.map((item) => item.operator_id), [
    "mtp.prediction_layer.000", "mtp.prediction_layer.001", "mtp.aux_head",
  ]);
  assert.equal(graph.tensors.find((item) => item.tensor_id === "mtp.prediction_layer.000.weights").logical_bytes, 1234);
  assert.equal(graph.tensors.find((item) => item.tensor_id === "mtp.aux_head.weights").logical_bytes, 5678);
  assert.deepEqual(graph.tensors.find((item) => item.tensor_id === "mtp.proposal_logits").shape, ["B", "T", "V"]);
  assert.ok(graph.tensors.find((item) => item.tensor_id === "block-group-000.output").consumer_operator_ids.includes("mtp.prediction_layer.000"));
});

test("authoritative group detail keeps residual edges and a clearly derived GQA lowering", () => {
  const graph = graphFromFixture({
    name: "inline", layer_specs: [layer("inline-0", { attention_heads: 8, kv_heads: 2, attention_head_dim: 64 })],
  });
  const group = graph.operators.find((item) => item.op_kind === "layer_group");
  const operatorIds = graph.operators
    .filter((item) => item.attributes.parent_group_id === group.operator_id)
    .map((item) => item.operator_id);
  const projection = Core.authoritativeOperatorProjection(graph, operatorIds, {
    group_id: group.operator_id,
    expanded_components: ["block-group-000", "block-group-000.attention"],
  });
  assert.ok(projection.residual_edges.length >= 2);
  assert.ok(projection.residual_edges.every((edge) => edge.authoritative === true && edge.tensor_id));
  const lowering = projection.attention_projections["block-group-000.attention"];
  assert.equal(lowering.derived_only, true);
  assert.equal(lowering.source_operator_id, "block-group-000.attention");
  assert.equal(lowering.attention_mode, "GQA");
  assert.equal(lowering.attention_heads, 8);
  assert.equal(lowering.kv_heads, 2);
  assert.equal(lowering.head_dim, 64);
  assert.equal(lowering.disclaimer, "派生视图 / 不新增执行 IR");
  assert.ok(lowering.nodes.every((item) => item.authoritative === false && !item.operator_id));
  assert.ok(lowering.nodes.every((item) => item.derived_id.startsWith("block-group-000.attention:derived:")));
  assert.ok(lowering.edges.every((item) => item.authoritative === false && !item.tensor_id));
  assert.ok(lowering.nodes.some((item) => item.label === "Q Heads ×8"));
  assert.equal(Core.attentionLoweringProjection({ operator_id: "mha", parameters: { attention_heads: 8, kv_heads: 8, attention_head_dim: 64 } }).attention_mode, "MHA");
  assert.equal(Core.attentionLoweringProjection({ operator_id: "mqa", parameters: { attention_heads: 8, kv_heads: 1, attention_head_dim: 64 } }).attention_mode, "MQA");
});

test("repeat-group overview keeps one authoritative template and a visual-only ×N loop", () => {
  const layers = Array.from({ length: 96 }, (_, index) => layer(`dense-${index}`));
  const graph = graphFromFixture({ name: "dense-96", vocabulary_size: 32000, layer_specs: layers });
  const snapshot = structuredClone(graph);
  const projection = Core.buildOverviewProjection(graph);

  assert.deepEqual(graph, snapshot);
  assert.equal(projection.attributes.repeat_group_projection, true);
  assert.equal(projection.attributes.leaf_operator_projection, false);
  const repeated = projection.nodes.find((item) => item.kind === "repeat_group");
  assert.equal(repeated.repeat_count, 96);
  assert.equal(repeated.expanded, false);
  assert.equal(repeated.inline_graph.operators.length, 6);
  assert.ok(repeated.inline_graph.operators.every((item) => item.authoritative === true));
  assert.ok(repeated.inline_graph.edges.every((item) => item.authoritative === true && item.tensor_id));
  assert.equal(repeated.boundary_ports.filter((item) => item.direction === "input").length, 1);
  assert.equal(repeated.boundary_ports.find((item) => item.direction === "input").fan_out, 2);
  const expanded = Core.buildOverviewProjection(graph, { collapsed_groups: [] }).nodes.find((item) => item.kind === "repeat_group");
  assert.equal(expanded.expanded, true);
  assert.deepEqual(expanded.inline_graph.operators.map((item) => item.operator_id), repeated.inline_graph.operators.map((item) => item.operator_id));
  assert.deepEqual(graph, snapshot, "expansion is view-only");
  const loop = projection.edges.find((edge) => edge.visual_only);
  assert.equal(loop.source_id, repeated.display_id);
  assert.equal(loop.target_id, repeated.display_id);
  assert.equal(loop.label, "×96");
  assert.equal(loop.semantic_edge_count, 0);
  assert.equal(Core.modelGraphEdges(graph).some((edge) => edge.source_operator_id === edge.target_operator_id), false);
  const embedding = projection.nodes.find((item) => item.op_kind === "embedding");
  const finalNorm = projection.nodes.find((item) => item.representative_operator_id === "final_norm");
  assert.ok(projection.edges.some((edge) => edge.source_id === embedding.display_id && edge.target_id === repeated.display_id));
  assert.ok(projection.edges.some((edge) => edge.source_id === repeated.display_id && edge.target_id === finalNorm.display_id));

  const operatorById = new Map(graph.operators.map((item) => [item.operator_id, item]));
  projection.nodes.flatMap((item) => item.boundary_ports).forEach((port) => {
    const authoritative = operatorById.get(port.operator_id)?.ports.find((item) => item.port_id === port.port_id);
    assert.ok(authoritative, `${port.operator_id}.${port.port_id} is authoritative`);
    assert.deepEqual(
      { direction: port.direction, dtype: port.dtype, shape: port.shape, layout: port.layout },
      { direction: authoritative.direction, dtype: authoritative.dtype, shape: authoritative.shape, layout: authoritative.layout },
    );
  });
});

test("A-B-A-B groups compress to their minimal complete period while the tail remains external", () => {
  const dense = (id) => layer(id, { metadata: { pattern_index: 0 } });
  const moe = (id) => layer(id, {
    kind: "moe", num_experts: 8, experts_per_token: 2,
    shared_expert_intermediate_size: 1024,
    metadata: { pattern_index: 1 },
  });
  const linear = (id) => layer(id, {
    sequence_mixer: "linear_attention",
    linear_attention: { key_heads: 4, value_heads: 8, state_dtype: "fp32" },
    metadata: { pattern_index: 2 },
  });
  const graph = graphFromFixture({
    name: "mixed-pattern",
    layer_specs: [dense("d0"), moe("m0"), dense("d1"), moe("m1"), linear("l0"), linear("l1")],
  });
  const projection = Core.buildOverviewProjection(graph);

  const patterns = projection.nodes.filter((item) => item.kind === "repeat_group");
  const alternating = patterns.find((item) => item.pattern_period === 2);
  assert.equal(alternating.pattern_repetitions, 2);
  assert.equal(alternating.repeat_count, 4);
  assert.deepEqual(alternating.pattern.map((item) => item.ffn_kind), ["dense", "moe"]);
  assert.ok(alternating.inline_graph.operators.some((item) => item.op_kind === "dense_mlp"));
  assert.ok(alternating.inline_graph.operators.some((item) => item.op_kind === "moe_router"));
  const linearTail = patterns.find((item) => item !== alternating);
  assert.equal(linearTail.repeat_count, 2);
  assert.equal(linearTail.pattern[0].mixer_kind, "linear_attention");
  assert.equal(projection.edges.filter((edge) => edge.visual_only).length, 2);
});

test("explicit repeated groups stay atomic while following A-B ranges avoid the partial tail", () => {
  const dense = (id) => layer(id, { metadata: { pattern_index: 0 } });
  const moe = (id) => layer(id, {
    kind: "moe", num_experts: 8, experts_per_token: 2,
    shared_expert_intermediate_size: 1024,
    metadata: { pattern_index: 1 },
  });
  const graph = graphFromFixture({
    name: "explicit-repeat-tail",
    layer_specs: [dense("a0"), dense("a1"), moe("b0"), dense("a2"), moe("b1"), dense("a3"), moe("b2")],
  });
  const layerGroups = graph.operators.filter((item) => item.op_kind === "layer_group");
  assert.deepEqual(layerGroups.map((item) => item.parameters.repeat), [2, 1, 1, 1, 1, 1]);

  const projection = Core.buildOverviewProjection(graph);
  const repeatGroups = projection.nodes.filter((item) => item.kind === "repeat_group");
  const explicit = repeatGroups.find((item) => item.group_ids.length === 1 && item.group_ids[0] === "block-group-000");
  assert.equal(explicit.repeat_count, 2);
  assert.equal(explicit.pattern_period, 1);
  assert.equal(explicit.pattern_repetitions, 1);
  assert.deepEqual(explicit.pattern.map((item) => item.repeat), [2]);

  const alternating = repeatGroups.find((item) => item.pattern_period === 2 && item.pattern_repetitions === 2);
  assert.deepEqual(alternating.group_ids, ["block-group-001", "block-group-002", "block-group-003", "block-group-004"]);
  assert.equal(alternating.repeat_count, 4);
  assert.deepEqual(alternating.pattern.map((item) => item.repeat), [1, 1]);
  assert.deepEqual(alternating.pattern.map((item) => item.ffn_kind), ["moe", "dense"]);

  const tail = repeatGroups.find((item) => item.group_ids.length === 1 && item.group_ids[0] === "block-group-005");
  assert.equal(tail.repeat_count, 1);
  assert.equal(projection.nodes.some((item) => item.pattern_period === 2 && item.pattern_repetitions === 3 && item.repeat_count === 7), false);
});

test("repeat-aware segment signatures keep adjacent ordinary pattern ranges compact", () => {
  const dense = (id) => layer(id, { metadata: { pattern_index: 0 } });
  const moe = (id) => layer(id, { kind: "moe", num_experts: 8, experts_per_token: 2, metadata: { pattern_index: 1 } });
  const linear = (id) => layer(id, {
    sequence_mixer: "linear_attention",
    linear_attention: { key_heads: 4, value_heads: 8, state_dtype: "fp32" },
    metadata: { pattern_index: 2 },
  });
  const wide = (id) => layer(id, { intermediate_size: 3072, metadata: { pattern_index: 3 } });
  const sparse = (id) => layer(id, { kind: "moe", num_experts: 4, experts_per_token: 1, metadata: { pattern_index: 4 } });
  const graph = graphFromFixture({
    name: "explicit-repeat-neighbors",
    layer_specs: [
      moe("left-b0"), dense("left-a0"), moe("left-b1"), dense("left-a1"),
      linear("repeat-e0"), linear("repeat-e1"),
      wide("right-c0"), sparse("right-d0"), wide("right-c1"), sparse("right-d1"),
    ],
  });

  const projection = Core.buildOverviewProjection(graph);
  const repeatGroups = projection.nodes.filter((item) => item.kind === "repeat_group");
  const byGroupIds = (ids) => repeatGroups.find((item) => JSON.stringify(item.group_ids) === JSON.stringify(ids));
  const left = byGroupIds(["block-group-000", "block-group-001", "block-group-002", "block-group-003"]);
  const explicit = byGroupIds(["block-group-004"]);
  const right = byGroupIds(["block-group-005", "block-group-006", "block-group-007", "block-group-008"]);

  assert.equal(left.pattern_period, 2);
  assert.equal(left.pattern_repetitions, 2);
  assert.equal(left.repeat_count, 4);
  assert.equal(explicit.repeat_count, 2);
  assert.equal(explicit.pattern_repetitions, 1);
  assert.deepEqual(explicit.pattern.map((item) => item.repeat), [2]);
  assert.equal(right.pattern_period, 2);
  assert.equal(right.pattern_repetitions, 2);
  assert.equal(right.repeat_count, 4);
  assert.deepEqual(left.pattern.map((item) => item.repeat), [1, 1]);
  assert.deepEqual(right.pattern.map((item) => item.repeat), [1, 1]);
  assert.equal(repeatGroups.some((item) => item.group_ids.includes("block-group-004") && item.group_ids.length > 1), false);
});

test("contract and semantic parameter changes split repeat groups", () => {
  const graph = graphFromFixture({
    name: "contract-variants",
    layer_specs: [
      layer("base", { metadata: { pattern_index: 0 } }),
      layer("wide", { intermediate_size: 4096, metadata: { pattern_index: 0 } }),
      layer("fp16", { intermediate_size: 4096, dtype: "fp16", metadata: { pattern_index: 0 } }),
    ],
  });
  const projection = Core.buildOverviewProjection(graph);
  const groups = projection.nodes.filter((item) => item.kind === "repeat_group");
  assert.equal(groups.length, 3);
  assert.ok(groups.every((item) => item.repeat_count === 1 && item.pattern_repetitions === 1));
  const mlps = groups.map((item) => item.inline_graph.operators.find((operator) => operator.op_kind === "dense_mlp"));
  assert.deepEqual(new Set(mlps.map((item) => item.parameters.intermediate_size)), new Set([2048, 4096]));
  assert.deepEqual(new Set(mlps.map((item) => item.ports.find((port) => port.direction === "output").dtype)), new Set(["bf16", "fp16"]));
});

test("a repeated A-B candidate with a cross-group bypass is not compressed", () => {
  const dense = (id) => layer(id, { metadata: { pattern_index: 0 } });
  const moe = (id) => layer(id, { kind: "moe", num_experts: 8, experts_per_token: 2, metadata: { pattern_index: 1 } });
  const graph = graphFromFixture({ name: "bypassed-ab", layer_specs: [dense("d0"), moe("m0"), dense("d1"), moe("m1")] });
  const firstGroup = graph.operators.find((item) => item.operator_id === "block-group-000");
  const firstOutput = graph.operators.find((item) => item.attributes.parent_group_id === firstGroup.operator_id && item.operator_id.endsWith("residual2")).ports.find((item) => item.direction === "output");
  const lmHead = graph.operators.find((item) => item.operator_id === "lm_head");
  lmHead.ports.push({ ...structuredClone(firstOutput), port_id: "bypass", direction: "input" });
  const normalized = Core.normalizeModelGraph(graph);
  const projection = Core.buildOverviewProjection(normalized);
  const repeatGroups = projection.nodes.filter((item) => item.kind === "repeat_group");
  assert.equal(repeatGroups.length, 4);
  assert.ok(repeatGroups.every((item) => item.group_ids.length === 1 && item.pattern_repetitions === 1));
  assert.equal(projection.nodes.some((item) => item.pattern_period === 2), false);
});

test("multiple layer groups preserve real layer IDs and layer-id-keyed overrides across Dense and MoE", () => {
  const layers = [
    layer("decoder.layers.0"),
    layer("decoder.layers.1"),
    layer("decoder.layers.2", { kind: "moe", num_experts: 8, experts_per_token: 2, metadata: { pattern_index: 1, note: "moe" } }),
  ];
  const graph = graphFromFixture({ name: "hybrid", layer_specs: layers });
  const groups = graph.operators.filter((item) => item.op_kind === "layer_group");
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.flatMap((item) => item.parameters.layer_ids), layers.map((item) => item.layer_id));
  groups.forEach((group) => assert.deepEqual(Object.keys(group.parameters.overrides), group.parameters.layer_ids));
  const projected = Core.graphToLayerSpecs(graph);
  assert.equal(projected.ok, true);
  assert.deepEqual(projected.layer_specs, layers);
});

test("linear-attention layer contracts survive the semantic graph round-trip", () => {
  const linearAttention = {
    key_heads: 4, value_heads: 8, key_head_dim: 64, value_head_dim: 64,
    conv_kernel_size: 4, state_dtype: "fp32", output_gate: true, gate_activation: "silu",
  };
  const original = layer("linear-0", { sequence_mixer: "linear_attention", linear_attention: linearAttention });
  const graph = graphFromFixture({ name: "linear", layer_specs: [original] });
  const mixer = graph.operators.find((item) => item.op_kind === "linear_attention");
  assert.deepEqual(mixer.parameters.linear_attention, linearAttention);
  assert.deepEqual(Core.graphToLayerSpecs(graph).layer_specs, [original]);
});

test("normalization reconnects weight-port tensors to their actual consumer", () => {
  const graph = Core.normalizeModelGraph({
    graph_id: "weights",
    operators: [{
      operator_id: "linear", op_kind: "linear", sequence_index: 0,
      ports: [{ port_id: "weight0", direction: "weight", tensor_id: "linear.weight", dtype: "bf16", shape: [512, 512], layout: "logical" }],
    }],
    tensors: [{
      tensor_id: "linear.weight", role: "weight", producer_operator_id: null,
      consumer_operator_ids: ["stale-consumer"], dtype: "bf16", shape: [512, 512], layout: "logical",
    }],
  });
  const tensor = graph.tensors.find((item) => item.tensor_id === "linear.weight");
  assert.deepEqual(tensor.consumer_operator_ids, ["linear"]);
  assert.deepEqual(graph.operators[0].weight_tensor_ids, ["linear.weight"]);
});

test("normalization assigns stable node, port, and tensor IDs without moving UI into semantics", () => {
  const raw = {
    graph_id: "aliases",
    nodes: [
      { kind: "linear", outputs: [{ id: "activation" }] },
      { kind: "linear", inputs: [{ id: "activation" }] },
    ],
    attributes: { owner: "test", ui: { positions: { linear: { x: 12, y: 34 } }, viewport: { x: 2, y: 3, scale: 20 } } },
  };
  const first = Core.normalizeModelGraph(raw);
  const second = Core.normalizeModelGraph(raw);
  assert.deepEqual(first, second);
  assert.deepEqual(first.operators.map((item) => item.operator_id), ["linear", "linear-2"]);
  assert.equal(first.operators[0].ports[0].port_id, "out0");
  assert.equal(first.operators[0].ports[0].tensor_id, "activation");
  assert.equal(first.attributes.ui.viewport.scale, 4);
  assert.equal(Core.semanticProjection(first).attributes.ui, undefined);
  assert.deepEqual(Core.uiProjection(first).positions.linear, { x: 12, y: 34 });
});

test("symbolic shape, dtype aliases, and layout unify with Chinese expected/actual diagnostics", () => {
  const compatible = Core.unifyTensorContracts(
    { dtype: "bfloat16", shape: ["B", "T", "H"], layout: "row-major" },
    { dtype: "bf16", shape: ["B", "T", 512], layout: "row_major" },
  );
  assert.equal(compatible.ok, true);
  assert.equal(compatible.bindings.H, 512);
  assert.deepEqual(compatible.contract.shape, ["B", "T", 512]);
  const badShape = Core.unifyTensorContracts(
    { dtype: "bf16", shape: ["B", "T", 512], layout: "logical" },
    { dtype: "bf16", shape: ["B", "T", 513], layout: "logical" },
  );
  assert.equal(badShape.ok, false);
  assert.match(badShape.message, /期望.*实际/);
  const badDtype = Core.modelPortCompatibility(
    { dtype: "fp16", shape: [2, 4], layout: "logical" },
    { dtype: "bf16", shape: [2, 4], layout: "logical" },
  );
  assert.equal(badDtype.compatible, false);
  assert.match(badDtype.message, /cast/);
});

test("all six explicit transform rules infer valid output contracts and fail closed", () => {
  assert.deepEqual(
    Core.inferTransformContracts("reshape", [{ dtype: "bf16", shape: [2, 3, 4], layout: "logical" }], { target_shape: [6, -1] }).outputs[0].shape,
    [6, 4],
  );
  assert.deepEqual(
    Core.inferTransformContracts("transpose", [{ dtype: "bf16", shape: ["B", "T", "H"], layout: "logical" }], { permutation: [1, 0, 2] }).outputs[0].shape,
    ["T", "B", "H"],
  );
  assert.deepEqual(
    Core.inferTransformContracts("concat", [
      { dtype: "bf16", shape: [2, 3], layout: "logical" },
      { dtype: "bf16", shape: [2, 5], layout: "logical" },
    ], { axis: 1 }).outputs[0].shape,
    [2, 8],
  );
  assert.deepEqual(
    Core.inferTransformContracts("split", [{ dtype: "bf16", shape: [2, 8], layout: "logical" }], { axis: 1, sections: 2 }).outputs.map((item) => item.shape),
    [[2, 4], [2, 4]],
  );
  assert.equal(Core.inferTransformContracts("cast", [{ dtype: "bf16", shape: [2], layout: "logical" }], { to_dtype: "float32" }).outputs[0].dtype, "fp32");
  assert.deepEqual(
    Core.inferTransformContracts("broadcast", [{ dtype: "bf16", shape: ["B", 1, "H"], layout: "logical" }], { target_shape: ["B", "T", "H"] }).outputs[0].shape,
    ["B", "T", "H"],
  );
  const invalid = Core.inferTransformContracts("reshape", [{ dtype: "bf16", shape: [2, 3], layout: "logical" }], { target_shape: [5] });
  assert.equal(invalid.ok, false);
  assert.match(invalid.message, /期望.*实际/);
});

test("connect and disconnect are immutable, unify symbols, and update tensor producer/consumer indexes", () => {
  const original = editablePair();
  const snapshot = structuredClone(original);
  const diagnostic = Core.validateConnection(original,
    { operator_id: "source", port_id: "out0" },
    { operator_id: "target", port_id: "in0" });
  assert.equal(diagnostic.compatible, true);
  const connected = Core.connectPorts(original, "source", "out0", "target", "in0");
  assert.deepEqual(original, snapshot);
  const sourcePort = connected.operators.find((item) => item.operator_id === "source").ports[0];
  const targetPort = connected.operators.find((item) => item.operator_id === "target").ports[0];
  assert.equal(targetPort.tensor_id, sourcePort.tensor_id);
  assert.deepEqual(targetPort.shape, ["B", "T", 512]);
  assert.deepEqual(connected.tensors.find((item) => item.tensor_id === sourcePort.tensor_id).consumer_operator_ids, ["target"]);
  const disconnected = Core.disconnectPort(connected, "target", "in0");
  assert.notEqual(disconnected.operators.find((item) => item.operator_id === "target").ports[0].tensor_id, sourcePort.tensor_id);
  assert.deepEqual(connected.tensors.find((item) => item.tensor_id === sourcePort.tensor_id).consumer_operator_ids, ["target"]);
});

test("incompatible connection is rejected before mutation with exact expected/actual context", () => {
  const graph = editablePair(["B", "T", 513], ["B", "T", 512]);
  const diagnostic = Core.validateConnection(graph,
    { operator_id: "source", port_id: "out0" },
    { operator_id: "target", port_id: "in0" });
  assert.equal(diagnostic.compatible, false);
  assert.match(diagnostic.message, /期望.*实际/);
  assert.throws(() => Core.connectPorts(graph, "source", "out0", "target", "in0"), /期望.*实际/);
});

test("add, update, duplicate, and delete helpers never mutate their input graph", () => {
  const base = editablePair();
  const added = Core.addNode(base, {
    operator_id: "cast", op_kind: "cast",
    ports: [{ port_id: "out0", direction: "output", dtype: "fp16", shape: ["B", "T", 512], layout: "logical" }],
    parameters: { dtype: "fp16" },
  }, { position: { x: 300, y: 80 } });
  assert.equal(base.operators.some((item) => item.operator_id === "cast"), false);
  assert.equal(added.attributes.ui.positions.cast.x, 300);
  const updated = Core.updateNode(added, "cast", { parameters: { dtype: "fp32" } });
  assert.equal(added.operators.find((item) => item.operator_id === "cast").parameters.dtype, "fp16");
  assert.equal(updated.operators.find((item) => item.operator_id === "cast").parameters.dtype, "fp32");
  const duplicated = Core.duplicateNode(updated, "cast");
  assert.ok(duplicated.operators.some((item) => item.operator_id === "cast-2"));
  assert.notEqual(
    duplicated.operators.find((item) => item.operator_id === "cast").ports[0].tensor_id,
    duplicated.operators.find((item) => item.operator_id === "cast-2").ports[0].tensor_id,
  );
  const deleted = Core.deleteNode(duplicated, "cast");
  assert.equal(deleted.operators.some((item) => item.operator_id === "cast"), false);
  assert.equal(duplicated.operators.some((item) => item.operator_id === "cast"), true);
});

test("repeat groups support per-instance overrides without expanding the visual component graph", () => {
  const graph = graphFromFixture({ name: "override", layer_specs: [layer("a"), layer("b")] });
  const group = graph.operators.find((item) => item.op_kind === "layer_group");
  const changed = Core.setLayerGroupOverride(graph, group.operator_id, "b", { intermediate_size: 3072, metadata: { tuned: true } });
  assert.equal(graph.operators.length, changed.operators.length);
  assert.equal(graph.operators.find((item) => item.operator_id === group.operator_id).parameters.overrides.b.intermediate_size, undefined);
  const projected = Core.graphToLayerSpecs(changed);
  assert.equal(projected.ok, true);
  assert.equal(projected.layer_specs[0].intermediate_size, 2048);
  assert.equal(projected.layer_specs[1].intermediate_size, 3072);
  assert.deepEqual(projected.layer_specs[1].metadata, { tuned: true });
});

test("non-projectable component graphs return an explicit reason instead of stale layer specs", () => {
  const result = Core.graphToLayerSpecs(editablePair());
  assert.equal(result.ok, false);
  assert.equal(result.code, "missing_layer_group");
  assert.match(result.reason, /缺少 layer_group/);
  assert.deepEqual(result.layer_specs, []);
});

test("DAG layering is deterministic and produces non-overlapping variable-size boxes", () => {
  const graph = graphFromFixture({ name: "layout", layer_specs: [layer("a"), layer("b")] });
  const sizes = Object.fromEntries(graph.operators.map((item, index) => [item.operator_id, { width: 120 + index * 3, height: 60 + (index % 4) * 15 }]));
  const first = Core.layoutDag(graph, sizes);
  const second = Core.layoutDag(graph, sizes);
  assert.deepEqual(first, second);
  assert.equal(first.hasCycle, false);
  const rects = graph.operators.map((item) => ({ id: item.operator_id, ...first.positions[item.operator_id], ...sizes[item.operator_id] }));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    const a = rects[left]; const b = rects[right];
    const intersects = a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
    assert.equal(intersects, false, `${a.id} overlaps ${b.id}`);
  }
});

test("cycle detection rejects cyclic semantic graphs and UI projection is independently replaceable", () => {
  const cyclic = Core.normalizeModelGraph({
    graph_id: "cycle",
    operators: [
      { operator_id: "a", op_kind: "linear", sequence_index: 0, ports: [
        { port_id: "in0", direction: "input", tensor_id: "tb", dtype: "bf16", shape: [2], layout: "logical" },
        { port_id: "out0", direction: "output", tensor_id: "ta", dtype: "bf16", shape: [2], layout: "logical" },
      ] },
      { operator_id: "b", op_kind: "linear", sequence_index: 1, ports: [
        { port_id: "in0", direction: "input", tensor_id: "ta", dtype: "bf16", shape: [2], layout: "logical" },
        { port_id: "out0", direction: "output", tensor_id: "tb", dtype: "bf16", shape: [2], layout: "logical" },
      ] },
    ], tensors: [], attributes: { ui: { positions: { a: { x: 1, y: 2 } } } },
  });
  assert.equal(Core.layoutDag(cyclic).hasCycle, true);
  assert.equal(Core.validateModelGraph(cyclic).valid, false);
  assert.match(Core.validateModelGraph(cyclic).errors.join("\n"), /必须是 DAG/);
  const semanticBefore = Core.semanticProjection(cyclic);
  const moved = Core.withUiProjection(cyclic, { positions: { a: { x: 900, y: 800 } }, viewport: { x: 1, y: 2, scale: 2 } });
  assert.deepEqual(Core.semanticProjection(moved), semanticBefore);
  assert.deepEqual(Core.uiProjection(moved).positions.a, { x: 900, y: 800 });
});

test("overview geometry uses native SVG cubic paths between exact port coordinates", () => {
  assert.equal(
    Core.overviewPortBezierPath({ x: 10, y: 20 }, { x: 210, y: 80 }),
    "M 10 20 C 110 20, 110 80, 210 80",
  );
  assert.equal(
    Core.overviewPortBezierPath({ x: 210, y: 80 }, { x: 10, y: 20 }, { minHandle: 40, maxHandle: 40 }),
    "M 210 80 C 250 80, -30 20, 10 20",
  );
  assert.equal(
    Core.overviewPortBezierPath({ x: 0.0004, y: -0 }, { x: 12.3456, y: 7.8912 }),
    "M 0 0 C 36 0, -23.654 7.891, 12.346 7.891",
  );
});

test("hybrid model routing uses short curves only when unobstructed and chooses all four anchor sides", () => {
  const source = { id: "source", x: 20, y: 40, width: 80, height: 60 };
  const right = { id: "right", x: 150, y: 40, width: 80, height: 60 };
  const below = { id: "below", x: 20, y: 180, width: 80, height: 60 };
  const distant = { id: "distant", x: 620, y: 240, width: 80, height: 60 };
  const short = Core.modelGraphRouteEdge(source, right, [source, right]);
  assert.equal(short.kind, "smooth");
  assert.equal(short.source.side, "right");
  assert.equal(short.target.side, "left");
  assert.match(short.path, /^M .+ C /);

  const unobstructed = Core.modelGraphRouteEdge(source, distant, [source, distant]);
  assert.equal(unobstructed.kind, "smooth", "distance alone does not create an orthogonal detour");
  assert.equal(unobstructed.source.side, "right");
  assert.equal(unobstructed.target.side, "left");

  const vertical = Core.modelGraphRouteEdge(source, below, [source, below], { shortCurveDistance: 40 });
  assert.equal(vertical.kind, "orthogonal");
  assert.equal(vertical.source.side, "bottom");
  assert.equal(vertical.target.side, "top");
  assert.match(vertical.path, /(?: L | Q )/);
  assert.deepEqual(Core.modelGraphAnchor(source, "left", 0.25), { x: 20, y: 55, side: "left", fraction: 0.25 });
});

test("obstacle-aware and residual routes stay outside unrelated model nodes", () => {
  const source = { id: "source", x: 100, y: 20, width: 100, height: 60 };
  const blocker = { id: "blocker", x: 90, y: 130, width: 120, height: 80 };
  const target = { id: "target", x: 100, y: 280, width: 100, height: 60 };
  const routed = Core.modelGraphRouteEdge(source, target, [source, blocker, target], { shortCurveDistance: 500 });
  assert.equal(routed.kind, "orthogonal");
  assert.equal(routed.obstacle_free, true);
  for (let index = 1; index < routed.points.length; index += 1) {
    assert.equal(Core.modelGraphSegmentHitsRect(routed.points[index - 1], routed.points[index], blocker), false);
  }

  const residual = Core.modelGraphRouteEdge(source, target, [source, target], { preferOuter: true });
  assert.equal(residual.kind, "orthogonal");
  assert.ok(residual.points.some((point) => point.x < source.x || point.x > source.x + source.width));
  assert.match(residual.path, / Q /);
});

test("exact port points remain the endpoints of obstacle-aware committed and preview routes", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 60 };
  const target = { id: "target", x: 220, y: 140, width: 80, height: 60 };
  const sourcePoint = { x: 80, y: 45, side: "right" };
  const targetPoint = { x: 245, y: 140, side: "top" };
  const route = Core.modelGraphRouteEdge(source, target, [source, target], {
    sourcePoint,
    targetPoint,
    shortCurveDistance: 0,
  });
  assert.deepEqual(route.source, sourcePoint);
  assert.deepEqual(route.target, targetPoint);
  assert.equal(route.points[0].x, sourcePoint.x);
  assert.equal(route.points.at(-1).y, targetPoint.y);
});

test("occupied segments deterministically divert overlaps while zero penalties preserve the direct route", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 60 };
  const target = { id: "target", x: 320, y: 0, width: 80, height: 60 };
  const occupiedSegments = [{ x1: 80, y1: 30, x2: 320, y2: 30 }];
  const snapshot = structuredClone(occupiedSegments);
  const options = {
    sourcePoint: { x: 80, y: 30, side: "right" },
    targetPoint: { x: 320, y: 30, side: "left" },
    occupiedSegments,
    overlapPenalty: 180,
    crossingPenalty: 72,
  };
  const first = Core.modelGraphRouteEdge(source, target, [source, target], options);
  const second = Core.modelGraphRouteEdge(source, target, [source, target], options);
  assert.deepEqual(first, second);
  assert.deepEqual(occupiedSegments, snapshot);
  assert.equal(first.kind, "orthogonal");
  assert.ok(first.points.some((point) => point.y !== 30), "route leaves the occupied horizontal channel");
  assert.ok(first.routing_penalty > 0, "unavoidable shared endpoint stubs remain represented in the cost");

  const unpenalized = Core.modelGraphRouteEdge(source, target, [source, target], {
    ...options,
    overlapPenalty: 0,
    crossingPenalty: 0,
  });
  assert.equal(unpenalized.kind, "smooth");
  assert.equal(unpenalized.routing_penalty, 0);
});

test("crossingPenalty lets independently routed edges trade a crossing for a clear channel", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 60 };
  const target = { id: "target", x: 320, y: 0, width: 80, height: 60 };
  const baseOptions = {
    sourcePoint: { x: 80, y: 30, side: "right" },
    targetPoint: { x: 320, y: 30, side: "left" },
    occupiedSegments: [{ x1: 200, y1: -80, x2: 200, y2: 80 }],
  };
  const crossingAllowed = Core.modelGraphRouteEdge(source, target, [source, target], {
    ...baseOptions,
    crossingPenalty: 0,
  });
  assert.equal(crossingAllowed.kind, "smooth");
  assert.equal(crossingAllowed.crossings, 1);
  assert.equal(crossingAllowed.routing_penalty, 0);

  const crossingAvoided = Core.modelGraphRouteEdge(source, target, [source, target], {
    ...baseOptions,
    crossingPenalty: 300,
  });
  assert.equal(crossingAvoided.kind, "orthogonal");
  assert.equal(crossingAvoided.crossings, 0);
  assert.equal(crossingAvoided.routing_penalty, 0);
  assert.ok(crossingAvoided.points.some((point) => point.y > 80 || point.y < -80));
});

test("overview layout key and pure sizes change whenever boundary-port rows change", () => {
  const node = { display_id: "overview:source", kind: "operator", summary: "source", boundary_ports: [] };
  const projection = { graph_id: "overview-layout", nodes: [node] };
  const emptyKey = Core.overviewLayoutKey(projection, 1);
  const emptySize = Core.overviewNodeSize(node, 1);
  assert.deepEqual(Core.overviewBoundaryPortCounts(node), { inputs: 0, outputs: 0, rows: 0 });

  const withOne = structuredClone(projection);
  withOne.nodes[0].boundary_ports.push({ operator_id: "source", port_id: "out0", direction: "output" });
  const oneKey = Core.overviewLayoutKey(withOne, 1);
  const oneSize = Core.overviewNodeSize(withOne.nodes[0], 1);
  assert.notEqual(oneKey, emptyKey);
  assert.ok(oneSize.height > emptySize.height);

  const withFour = structuredClone(withOne);
  for (let index = 1; index < 4; index += 1) {
    withFour.nodes[0].boundary_ports.push({ operator_id: "source", port_id: `out${index}`, direction: "output" });
  }
  const fourSize = Core.overviewNodeSize(withFour.nodes[0], 1);
  assert.notEqual(Core.overviewLayoutKey(withFour, 1), oneKey);
  assert.equal(fourSize.height - oneSize.height, 48);
  assert.deepEqual(Core.overviewNodeSizes(withFour, 1)["overview:source"], fourSize);
  assert.notEqual(Core.overviewLayoutKey(withFour, 1.2), Core.overviewLayoutKey(withFour, 1));
});

test("repeat-group overview layout is deterministic, honors measured sizes, wraps narrow DAG layers, and never overlaps", () => {
  const graph = graphFromFixture({
    name: "narrow-leaf-overview",
    vocabulary_size: 32000,
    mtp: { prediction_layers: 2, auxiliary_head: true },
    layer_specs: [layer("narrow-0")],
  });
  const projection = Core.buildOverviewProjection(graph);
  const measured = Object.fromEntries(projection.nodes.map((node) => [node.display_id, { width: 140, height: 60 }]));
  const layout = Core.overviewResponsiveLayout(projection, 300, 1, { nodeSizes: measured });
  assert.deepEqual(layout, Core.overviewResponsiveLayout(projection, 300, 1, { nodeSizes: measured }));
  assert.equal(layout.viewport.scale, 1);
  assert.equal(layout.scroll_required, false);
  projection.nodes.forEach((node) => assert.deepEqual(layout.nodeSizes[node.display_id], measured[node.display_id]));
  const rects = projection.nodes.map((node) => ({
    id: node.display_id,
    ...layout.positions[node.display_id],
    ...layout.nodeSizes[node.display_id],
  }));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    const a = rects[left]; const b = rects[right];
    const intersects = a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
    assert.equal(intersects, false, `${a.id} overlaps ${b.id}`);
  }

  const prediction = projection.nodes.find((item) => item.op_kind === "mtp_prediction_layer");
  const finalNorm = projection.nodes.find((item) => item.representative_operator_id === "final_norm");
  const branchLayer = layout.layers.find((item) => item.node_ids.includes(prediction.display_id));
  assert.ok(branchLayer.node_ids.includes(finalNorm.display_id));
  assert.notEqual(layout.positions[prediction.display_id].y, layout.positions[finalNorm.display_id].y, "narrow branch rank wraps to another row");
});

test("connection preview is transient and commits only through output-to-input graph validation", () => {
  const graph = editablePair();
  const before = structuredClone(graph);
  const preview = Core.createConnectionPreview(
    graph,
    { operatorId: "source", portId: "out0" },
    { x: 12, y: 34 },
  );
  assert.deepEqual(preview.source, { operator_id: "source", port_id: "out0" });
  assert.deepEqual(preview.pointer, { x: 12, y: 34 });
  assert.equal(preview.active, true);
  assert.deepEqual(graph, before);

  const hovered = Core.updateConnectionPreview(graph, preview, {
    pointer: { x: 56, y: 78 },
    target: { operatorId: "target", portId: "in0" },
  });
  assert.equal(hovered.status, "compatible");
  assert.equal(hovered.diagnostic.compatible, true);
  assert.deepEqual(hovered.pointer, { x: 56, y: 78 });
  assert.deepEqual(graph, before);

  const committed = Core.commitConnectionPreview(graph, hovered);
  assert.equal(committed.connected, true);
  assert.notEqual(committed.graph, graph);
  assert.equal(committed.preview.active, false);
  assert.equal(committed.preview.status, "committed");
  assert.deepEqual(graph, before);
  const sourceTensor = committed.graph.operators.find((item) => item.operator_id === "source").ports[0].tensor_id;
  assert.equal(committed.graph.operators.find((item) => item.operator_id === "target").ports[0].tensor_id, sourceTensor);

  const wrongDirection = Core.createConnectionPreview(graph, { operatorId: "target", portId: "in0" });
  assert.equal(wrongDirection.active, false);
  assert.equal(wrongDirection.diagnostic.code, "direction_mismatch");
});

test("editing a concrete port contract synchronizes TensorValue and every referencing port", () => {
  const graph = editablePair();
  const connected = Core.connectPorts(graph, "source", "out0", "target", "in0");
  const tensorId = connected.operators.find((item) => item.operator_id === "source").ports[0].tensor_id;
  const updated = Core.updatePortContract(connected, "target", "in0", { dtype: "fp16", shape: ["B", 32] });
  const tensor = updated.tensors.find((item) => item.tensor_id === tensorId);
  const references = updated.operators.flatMap((item) => item.ports).filter((item) => item.tensor_id === tensorId);
  assert.deepEqual({ dtype: tensor.dtype, shape: tensor.shape }, { dtype: "fp16", shape: ["B", 32] });
  assert.ok(references.length >= 2);
  assert.ok(references.every((item) => item.dtype === "fp16" && JSON.stringify(item.shape) === '["B",32]'));
  assert.equal(Core.validateModelGraph(updated).valid, true);
  assert.equal(connected.tensors.find((item) => item.tensor_id === tensorId).dtype, "bf16");
  assert.equal(Object.hasOwn(updated.operators[0].ports[0], "quantization"), false);
});

test("incompatible, cyclic, and Esc-cancelled previews preserve the exact graph object", () => {
  const incompatibleGraph = editablePair(["B", "T", 513], ["B", "T", 512]);
  const incompatiblePreview = Core.createConnectionPreview(incompatibleGraph, { operator_id: "source", port_id: "out0" });
  const incompatible = Core.commitConnectionPreview(
    incompatibleGraph,
    incompatiblePreview,
    { operator_id: "target", port_id: "in0" },
  );
  assert.equal(incompatible.connected, false);
  assert.equal(incompatible.graph, incompatibleGraph);
  assert.equal(incompatible.preview.active, true);
  assert.equal(incompatible.preview.status, "incompatible");
  assert.equal(incompatible.diagnostic.code, "shape_mismatch");

  const dag = Core.normalizeModelGraph({
    graph_id: "preview-cycle",
    operators: [
      { operator_id: "a", op_kind: "linear", sequence_index: 0, ports: [
        { port_id: "in0", direction: "input", tensor_id: "a.in", dtype: "bf16", shape: [2], layout: "logical" },
        { port_id: "out0", direction: "output", tensor_id: "ab", dtype: "bf16", shape: [2], layout: "logical" },
      ] },
      { operator_id: "b", op_kind: "linear", sequence_index: 1, ports: [
        { port_id: "in0", direction: "input", tensor_id: "ab", dtype: "bf16", shape: [2], layout: "logical" },
        { port_id: "out0", direction: "output", tensor_id: "b.out", dtype: "bf16", shape: [2], layout: "logical" },
      ] },
    ],
    tensors: [],
  });
  const cyclePreview = Core.createConnectionPreview(dag, { operator_id: "b", port_id: "out0" });
  const cycle = Core.commitConnectionPreview(dag, cyclePreview, { operator_id: "a", port_id: "in0" });
  assert.equal(cycle.connected, false);
  assert.equal(cycle.graph, dag);
  assert.equal(cycle.diagnostic.code, "cycle");

  const cancelledPreview = Core.cancelConnectionPreview(incompatiblePreview, "escape");
  assert.equal(cancelledPreview.active, false);
  assert.equal(cancelledPreview.status, "cancelled");
  assert.equal(cancelledPreview.diagnostic.reason, "escape");
  const cancelledCommit = Core.commitConnectionPreview(incompatibleGraph, cancelledPreview, { operator_id: "target", port_id: "in0" });
  assert.equal(cancelledCommit.connected, false);
  assert.equal(cancelledCommit.graph, incompatibleGraph);
  assert.equal(cancelledCommit.diagnostic.code, "preview_inactive");
});
