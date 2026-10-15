"""Schema consumers share current names without migrating obsolete inputs."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from heterollm_sim.cli import main
from heterollm_sim.config import hardware_from_dict, model_from_dict
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.model_artifacts import load_model_artifact, save_model_artifact
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import (
    ModelGraph, OperatorNode, TensorTransform, TensorValue, model_graph_from_dict,
)
from heterollm_sim.serde import to_primitive, write_json
from heterollm_sim.web import scenario_to_payload


def test_memory_generation_alias_does_not_override_the_physical_config():
    scenario = build_reference_scenario()
    original = scenario.hardware.get_component("hbm0")
    current = replace(original, metadata={**original.metadata, "memory_generation": "HBM3E"})
    configured = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        current if item.component_id == current.component_id else item
        for item in scenario.hardware.components)))
    assert configured.hardware.get_component("hbm0").metadata["physical_memory_config"]["generation"] == "HBM3"
    declared = replace(current, metadata={**current.metadata, "generation": "HBM3E"})
    with pytest.raises(ValueError, match="conflicts with physical_memory_config.generation"):
        replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            declared if item.component_id == declared.component_id else item
            for item in scenario.hardware.components)))


@pytest.mark.parametrize("kind", ["DDR3", "DDR4", "DDR5", "LPDDR5", "LPDDR5X"])
def test_dram_generation_is_not_a_component_kind(kind):
    with pytest.raises(ValueError, match="declare generation in physical_memory_config"):
        ComponentSpec("memory", kind)
    with pytest.raises(ValueError, match="declare generation in physical_memory_config"):
        hardware_from_dict({"name": "old-kind", "components": [{"component_id": "memory", "kind": kind}], "links": []})


TRANSFORMS = [
    ("reshape", {"shape": [3, 2]}, (2, 3), (3, 2), "fp16", "logical", "target_shape"),
    ("transpose", {"permutation": [1, 0]}, (2, 3), (3, 2), "fp16", "logical_transposed", "perm"),
    ("cast", {"dtype": "float32"}, (2, 3), (2, 3), "fp32", "logical", "to_dtype"),
    ("broadcast", {"shape": [2, 3]}, (1, 3), (2, 3), "fp16", "logical", "target_shape"),
]


def _transform_graph(kind, attributes, input_shape, output_shape, output_dtype, output_layout):
    return ModelGraph("transform", operators=(OperatorNode("boundary", "model_input", 0),),
        tensors=(TensorValue("input", "input", dtype="float16", shape=input_shape),
                 TensorValue("output", "output", dtype=output_dtype, shape=output_shape, layout=output_layout)),
        transforms=(TensorTransform("transform", kind, "input", "output", attributes),), executable=False)


@pytest.mark.parametrize("kind,attributes,input_shape,output_shape,output_dtype,output_layout,old_field", TRANSFORMS)
def test_transform_roundtrip_uses_current_attributes(kind, attributes, input_shape, output_shape, output_dtype, output_layout, old_field):
    graph = _transform_graph(kind, attributes, input_shape, output_shape, output_dtype, output_layout)
    assert model_graph_from_dict(to_primitive(graph)) == graph
    payload = to_primitive(graph)
    payload["transforms"][0]["attributes"][old_field] = next(iter(attributes.values()))
    with pytest.raises(ValueError, match="retired fields"):
        model_graph_from_dict(payload)


def test_transpose_still_supports_default_axis_reversal():
    graph = _transform_graph("transpose", {}, (2, 3), (3, 2), "fp16", "logical_transposed")
    assert model_graph_from_dict(to_primitive(graph)) == graph


def test_browser_transform_contracts_match_backend_current_names():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser transform checks")
    module = Path(__file__).parents[1] / "src/heterollm_sim/webui/model-graph-core.js"
    graphs = [to_primitive(_transform_graph(*case[:6])) for case in TRANSFORMS]
    script = r"""
const assert = require('node:assert/strict');
const core = require(process.argv[1]);
const graphs = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const aliases = {reshape: 'target_shape', transpose: 'perm', cast: 'to_dtype', broadcast: 'target_shape'};
for (const graph of graphs) {
  const transform = graph.transforms[0];
  const input = graph.tensors.find(row => row.tensor_id === 'input');
  const output = graph.tensors.find(row => row.tensor_id === 'output');
  assert.equal(core.validateModelGraph(graph).valid, true);
  const current = core.inferTransformContracts(transform.kind, [input], transform.attributes);
  assert.equal(current.ok, true);
  assert.deepEqual(current.outputs[0], core.normalizedContract(output));
  transform.attributes[aliases[transform.kind]] = Object.values(transform.attributes)[0];
  assert.equal(core.inferTransformContracts(transform.kind, [input], transform.attributes).ok, false);
  assert.equal(core.validateModelGraph(graph).valid, false);
  assert.throws(() => core.normalizeModelGraph(graph));
}
const input = {dtype: 'fp16', shape: [6], layout: 'logical'};
assert.equal(core.inferTransformContracts('split', [input], {sizes: [2, 4]}).ok, true);
assert.equal(core.inferTransformContracts('split', [input], {sections: 3}).ok, true);
assert.equal(core.inferTransformContracts('split', [input], {split_sizes: [2, 4]}).ok, false);
assert.equal(core.inferTransformContracts('transpose', [{dtype: 'fp16', shape: [2, 3]}]).ok, true);
"""
    result = subprocess.run([node, "-e", script, str(module)], input=json.dumps(graphs),
        text=True, encoding="utf-8", capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_cli_and_model_artifact_callers_keep_current_graph_and_reject_nodes(tmp_path, capsys):
    scenario = build_reference_scenario()
    path = tmp_path / "scenario.json"
    payload = scenario_to_payload(scenario)
    write_json(path, payload)
    assert main(["validate", str(path)]) == 0
    capsys.readouterr()
    document = save_model_artifact(scenario.model, artifact_dir=tmp_path, artifact_id="current")
    restored = load_model_artifact("current", artifact_dir=tmp_path)
    assert restored["model"]["graph"] == document["model"]["graph"]
    loaded = model_from_dict({"artifact_id": "current"}, artifact_dir=tmp_path)
    assert loaded.graph == scenario.model.graph
    graph = payload["model"]["graph"]
    graph["nodes"] = graph.pop("operators")
    write_json(path, payload)
    assert main(["validate", str(path)]) == 2
    with pytest.raises(ValueError, match="unknown fields: nodes"):
        save_model_artifact(payload["model"], artifact_dir=tmp_path)
