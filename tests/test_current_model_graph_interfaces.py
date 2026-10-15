"""Browser graph editing preserves current contracts and rejects retired shapes."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import model_graph_from_dict
from heterollm_sim.serde import to_primitive


def test_browser_model_graph_current_roundtrip_editing_and_old_shape_rejection():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser graph contract checks")
    graph = build_reference_scenario().model.graph
    expected = [to_primitive(item.layer) for item in model_graph_execution_view(graph).layer_instances]
    module = Path(__file__).parents[1] / "src/heterollm_sim/webui/model-graph-core.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const core = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const original = input.graph;
const normalized = core.normalizeModelGraph(original);
assert.equal(core.validateModelGraph(normalized).valid, true);
assert.deepEqual(normalized.operators.map(row => row.parameters), original.operators.map(row => row.parameters));
assert.deepEqual(core.graphToLayerSpecs(normalized).layer_specs, input.expected);
assert.deepEqual(core.normalizeModelGraph(JSON.parse(JSON.stringify(normalized))), normalized);
const group = normalized.operators.find(row => row.op_kind === 'layer_group');
const layerId = group.parameters.layer_ids[0];
const changed = core.setLayerGroupOverride(normalized, group.operator_id, layerId,
  {intermediate_size: 8192, metadata: {note: 'preserved'}});
const projected = core.graphToLayerSpecs(changed);
assert.equal(projected.ok, true);
assert.equal(projected.layer_specs[0].intermediate_size, 8192);
assert.deepEqual(projected.layer_specs[0].metadata, {note: 'preserved'});
let edited = core.addNode(changed, {op_kind: 'annotation', ports: [], parameters: {label: 'current'}});
const added = edited.operators.find(row => row.op_kind === 'annotation');
assert.equal(added.parameters.label, 'current');
edited = core.updateNode(edited, added.operator_id, {parameters: {label: 'edited'}});
assert.equal(core.validateModelGraph(edited).valid, true);
const duplicated = core.duplicateNode(edited, added.operator_id);
assert.equal(duplicated.operators.length, edited.operators.length + 1);
edited = core.deleteNode(duplicated, duplicated.operators.at(-1).operator_id);
assert.equal(core.validateModelGraph(edited).valid, true);
assert.deepEqual(edited.operators.find(row => row.operator_id === group.operator_id).parameters,
  changed.operators.find(row => row.operator_id === group.operator_id).parameters);
const currentBuilt = core.buildModelGraphFromLayerSpecs(input.expected, {
  name: 'current-built', architecture: 'transformer', vocabulary_size: 128, max_sequence_length: 256,
});
assert.equal(core.validateModelGraph(currentBuilt).valid, true);
assert.equal(core.graphToLayerSpecs(currentBuilt).ok, true);
assert(core.graphToLayerSpecs(currentBuilt).layer_specs.every(row => row.schema_version === '4.0.0'));
const retired = [];
const rename = (record, old, current) => {record[old] = record[current]; delete record[current];};
for (const [old, current] of [['node_id', 'operator_id'], ['id', 'operator_id'], ['kind', 'op_kind'],
    ['type', 'op_kind'], ['inputs', 'input_tensor_ids'], ['outputs', 'output_tensor_ids'], ['weights', 'weight_tensor_ids']]) {
  const graph = structuredClone(original); rename(graph.operators[0], old, current); retired.push(graph);
}
{ const graph = structuredClone(original); rename(graph, 'nodes', 'operators'); retired.push(graph); }
{ const graph = structuredClone(original); rename(graph.tensors[0], 'id', 'tensor_id'); retired.push(graph); }
{ const graph = structuredClone(original); rename(graph.operators.find(row => row.ports.length).ports[0], 'id', 'port_id'); retired.push(graph); }
{ const graph = structuredClone(original); const row = graph.operators.find(row => row.output_tensor_ids.length);
  row.output_tensor_ids = row.output_tensor_ids.map(tensor_id => ({tensor_id})); retired.push(graph); }
retired.push({graph: original}, {model: {graph: original}});
for (const graph of retired) {
  assert.equal(core.validateModelGraph(graph).valid, false);
  assert.throws(() => core.normalizeModelGraph(graph));
}
for (const change of [
  group => {delete group.parameters.layer_ids;},
  group => {group.parameters.overrides = {'0': {metadata: {}}};},
  group => {group.parameters.layer_template = input.expected[0]; delete group.parameters.layer_ids;},
]) {
  const graph = structuredClone(original); change(graph.operators.find(row => row.op_kind === 'layer_group'));
  assert.equal(core.graphToLayerSpecs(graph).ok, false);
}
const gguf = core.buildGraphFromGGUFPreset({schema_version: 'heterollm.gguf-model-preset/v1',
  preset: {id: 'current', name: 'Current GGUF', source: 'gguf_import'},
  model: {architecture: 'llama', n_layer: 4, n_embd: 64, n_head: 4, n_head_kv: 2, vocab_size: 128, context_length: 256},
  evidence: {geometry: 'exact'}, weights: {tensor_count: 16}});
assert.equal(gguf.attributes.symbols.V, 128);
assert.equal(gguf.executable, false);
assert.equal(core.canonicalDtype('float16'), 'fp16');
process.stdout.write(JSON.stringify({edited, changed, built: currentBuilt}));
"""
    result = subprocess.run([node, "-e", script, str(module)],
        input=json.dumps({"graph": to_primitive(graph), "expected": expected}),
        text=True, encoding="utf-8", capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout)
    for value in outputs.values():
        restored = model_graph_from_dict(value)
        assert restored.operators
    edited_view = model_graph_execution_view(model_graph_from_dict(outputs["changed"]))
    assert edited_view.layer_instances[0].layer.intermediate_size == 8192
