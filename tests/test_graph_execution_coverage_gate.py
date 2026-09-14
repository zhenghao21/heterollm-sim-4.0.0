import unittest
import json
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
from dataclasses import replace

from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.compiler_ir import (
    CanonicalizationError,
    compile_canonical_scenario,
)
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import (
    ModelSpec,
    model_graph_execution_digest,
    model_graph_execution_layers,
)
from heterollm_sim.model_presets import (
    OUT_OF_DOMAIN,
    UnsupportedPresetError,
    list_model_presets,
    materialize_model_payload,
    model_preset_detail,
)
from heterollm_sim.planner import validate_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import (
    OperatorNode,
    OperatorPort,
    TensorTransform,
    TensorValue,
)
from heterollm_sim.serde import to_primitive


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_GRAPH_CORE = (
    _REPO_ROOT / "src" / "heterollm_sim" / "webui" / "model-graph-core.js"
)


def _frontend_normalize_graphs(items):
    if shutil.which("node") is None:
        raise unittest.SkipTest("Node.js is required for frontend graph normalization")
    script = """
const fs = require("fs");
const ModelGraph = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const result = input.items.map((item) => ModelGraph.normalizeModelGraph(item.graph, item.model || {}));
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(_MODEL_GRAPH_CORE)],
        input=json.dumps({"items": list(items)}),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def _graph_only_model(source, graph):
    return ModelSpec(
        name=source.name,
        graph=graph,
        text_backbone_only=source.text_backbone_only,
        supported_modalities=source.supported_modalities,
        excluded_subgraphs=source.excluded_subgraphs,
        metadata=source.metadata,
        schema_version=source.schema_version,
    )


class GraphExecutionCoverageGateTests(unittest.TestCase):
    def setUp(self):
        self.scenario = build_reference_scenario()
        self.source = self.scenario.model
        self.graph = self.source.graph

    def assert_gate_rejects(self, graph, pattern="兼容.*执行器|静默降级"):
        with self.assertRaisesRegex(ValueError, pattern):
            _graph_only_model(self.source, graph)

    def _scenario_with_model(self, model):
        return replace(
            self.scenario,
            model=model,
            placement=replace(
                self.scenario.placement,
                model_name=model.name,
                op_to_component={},
                tensor_to_component={},
                tensor_bytes={},
                parallel=replace(
                    self.scenario.placement.parallel,
                    layer_to_stage={},
                ),
            ),
        )

    def test_all_supported_presets_pass_the_execution_gate(self):
        supported = 0
        for item in list_model_presets():
            if item["support_level"] == OUT_OF_DOMAIN:
                continue
            payload = materialize_model_payload(item["id"])
            with self.subTest(preset_id=item["id"]):
                model = model_from_dict(payload)
                self.assertTrue(model.graph.executable)
                supported += 1
        self.assertGreater(supported, 0)

    def test_derived_embedding_logical_bytes_is_a_lossless_graph_overlay(self):
        payload = deepcopy(to_primitive(self.source))
        embedding = next(
            item
            for item in payload["graph"]["tensors"]
            if item["tensor_id"] == "embedding_weights"
        )
        self.assertIsNone(embedding["logical_bytes"])
        self.assertEqual(self.source.embedding_weight_bytes, 16_384_000)

        embedding["logical_bytes"] = self.source.embedding_weight_bytes
        model = model_from_dict(payload)

        self.assertEqual(
            model_graph_execution_digest(model.graph),
            model_graph_execution_digest(
                self.source.graph
            ),
        )

    def test_explicit_embedding_logical_bytes_is_graph_authoritative(self):
        payload = deepcopy(to_primitive(self.source))
        embedding = next(
            item
            for item in payload["graph"]["tensors"]
            if item["tensor_id"] == "embedding_weights"
        )
        embedding["logical_bytes"] = self.source.embedding_weight_bytes + 1

        model = model_from_dict(payload)
        self.assertEqual(
            model.embedding_weight_bytes,
            self.source.embedding_weight_bytes + 1,
        )
        self.assertNotEqual(
            model_graph_execution_digest(model.graph),
            model_graph_execution_digest(self.source.graph),
        )

    def test_consumed_logits_cannot_be_relabelled_as_terminal_output(self):
        payload = deepcopy(model_preset_detail("qwen3_8-27b")["model"])
        logits = next(
            item
            for item in payload["graph"]["tensors"]
            if item["tensor_id"] == "logits"
        )
        self.assertEqual(logits["consumer_operator_ids"], ["output"])
        logits["role"] = "output"

        with self.assertRaisesRegex(ValueError, "role.*activation.*output"):
            model_from_dict(payload)

    def test_all_presets_survive_real_frontend_normalization_and_control_plane_boundary(self):
        presets = list_model_presets()
        generation_allowed = [item for item in presets if item["generation_allowed"]]
        display_only = [item for item in presets if not item["generation_allowed"]]
        self.assertEqual(len(presets), 147)
        self.assertEqual(len(generation_allowed), 139)
        self.assertEqual(len(display_only), 8)

        payloads = []
        for item in generation_allowed:
            detail = model_preset_detail(item["id"])
            self.assertEqual(detail["preset"]["id"], item["id"])
            self.assertIsNotNone(detail["model"])
            payloads.append((item, detail["model"]))
        normalized_graphs = _frontend_normalize_graphs(
            {"graph": payload["graph"], "model": payload}
            for _, payload in payloads
        )
        for (item, payload), graph in zip(payloads, normalized_graphs):
            with self.subTest(preset_id=item["id"]):
                roundtrip_payload = dict(payload)
                roundtrip_payload["graph"] = graph
                model = model_from_dict(roundtrip_payload)
                self.assertTrue(model.graph.executable)
                self.assertEqual(model.num_layers, item["layer_count"])
                self.assertEqual(
                    model_graph_execution_digest(model.graph),
                    model_graph_execution_digest(model_from_dict(payload).graph),
                )
                scenario = self._scenario_with_model(model)
                mapping = plan_runtime_placement(scenario)
                self.assertTrue(mapping.decisions)
                self.assertFalse(mapping.mapping_stale)
                self.assertEqual(mapping.input_fingerprint, mapping.current_input_fingerprint)
                self.assertEqual(mapping.apply(scenario).model, model)

        ood_graphs = []
        for item in display_only:
            with self.subTest(preset_id=item["id"], phase="display-only"):
                detail = model_preset_detail(item["id"])
                self.assertEqual(detail["preset"]["support_level"], OUT_OF_DOMAIN)
                self.assertFalse(detail["preset"]["generation_allowed"])
                self.assertIsNone(detail["model"])
                self.assertFalse(detail["graph"]["executable"])
                with self.assertRaises(UnsupportedPresetError):
                    materialize_model_payload(item["id"])
                ood_graphs.append({"graph": detail["graph"], "model": {}})
        for item, graph in zip(display_only, _frontend_normalize_graphs(ood_graphs)):
            with self.subTest(preset_id=item["id"], phase="fail-closed"):
                self.assertFalse(graph["executable"])
                with self.assertRaisesRegex(ValueError, "不可执行|不能投影"):
                    model_from_dict(
                        {
                            "schema_version": self.source.schema_version,
                            "name": item["name"],
                            "graph": graph,
                        }
                    )

    def test_qwen38_and_moe_frontend_round_trip_regressions(self):
        cases = ("qwen3_8-27b", "qwen3_8-2_4t-a95b")
        payloads = [(preset_id, materialize_model_payload(preset_id)) for preset_id in cases]
        normalized_graphs = _frontend_normalize_graphs(
            {"graph": payload["graph"], "model": payload}
            for _, payload in payloads
        )

        for (preset_id, payload), graph in zip(payloads, normalized_graphs):
            with self.subTest(preset_id=preset_id):
                original_tensor_ids = [
                    item["tensor_id"] for item in payload["graph"]["tensors"]
                ]
                normalized_tensor_ids = [item["tensor_id"] for item in graph["tensors"]]
                self.assertEqual(set(normalized_tensor_ids), set(original_tensor_ids))
                self.assertNotEqual(normalized_tensor_ids, original_tensor_ids)
                self.assertIn("embedding.output", original_tensor_ids)
                self.assertIn("embedding_weights", original_tensor_ids)
                roundtrip_payload = dict(payload)
                roundtrip_payload["graph"] = graph
                model = model_from_dict(roundtrip_payload)
                self.assertEqual(
                    model_graph_execution_digest(model.graph),
                    model_graph_execution_digest(model_from_dict(payload).graph),
                )

        moe_graph = normalized_graphs[1]
        multi_consumer = [
            tensor for tensor in moe_graph["tensors"]
            if len(tensor["consumer_operator_ids"]) > 1
        ]
        self.assertTrue(multi_consumer)

    def test_representative_frontend_normalized_presets_reach_control_plane_gate(self):
        payloads = [
            (preset_id, materialize_model_payload(preset_id))
            for preset_id in ("qwen3_8-27b", "mixtral-8x7b-v0_1")
        ]
        normalized_graphs = _frontend_normalize_graphs(
            {"graph": payload["graph"], "model": payload}
            for _, payload in payloads
        )

        for (preset_id, payload), graph in zip(payloads, normalized_graphs):
            with self.subTest(preset_id=preset_id):
                payload = dict(payload)
                payload["graph"] = graph
                model = model_from_dict(payload)
                result = plan_runtime_placement(self._scenario_with_model(model))
                self.assertTrue(result.decisions)

    def test_ui_only_metadata_is_ignored_by_gate_and_digest(self):
        attributes = dict(self.graph.attributes)
        attributes["ui"] = {
            "zoom": 2.25,
            "positions": {"embedding": [123, 456]},
        }
        changed = replace(self.graph, attributes=attributes)

        model = _graph_only_model(self.source, changed)

        self.assertEqual(
            model_graph_execution_layers(model.graph),
            model_graph_execution_layers(self.source.graph),
        )
        self.assertEqual(
            model_graph_execution_digest(changed),
            model_graph_execution_digest(self.graph),
        )

    def test_outer_graph_order_and_consumer_index_order_are_digest_neutral(self):
        changed = replace(
            self.graph,
            operators=tuple(reversed(self.graph.operators)),
            tensors=tuple(
                replace(
                    tensor,
                    consumer_operator_ids=tuple(
                        reversed(tensor.consumer_operator_ids)
                    ),
                )
                for tensor in reversed(self.graph.tensors)
            ),
        )

        model = _graph_only_model(self.source, changed)

        self.assertEqual(
            model_graph_execution_layers(model.graph),
            model_graph_execution_layers(self.source.graph),
        )
        self.assertEqual(
            model_graph_execution_digest(changed),
            model_graph_execution_digest(self.graph),
        )

    def test_input_port_order_remains_semantic_and_is_rejected(self):
        target = next(
            item for item in self.graph.operators if len(item.input_tensor_ids) > 1
        )
        input_ports = tuple(port for port in target.ports if port.direction == "input")
        other_ports = tuple(port for port in target.ports if port.direction != "input")
        changed_operator = replace(
            target,
            input_tensor_ids=tuple(reversed(target.input_tensor_ids)),
            ports=tuple(reversed(input_ports)) + other_ports,
        )

        self.assert_gate_rejects(
            replace(
                self.graph,
                operators=tuple(
                    changed_operator if item.operator_id == target.operator_id else item
                    for item in self.graph.operators
                ),
            )
        )

    def test_known_per_layer_override_remains_losslessly_projectable(self):
        payload = materialize_model_payload("qwen2_5-0_5b")
        group = next(
            item
            for item in payload["graph"]["operators"]
            if item["op_kind"] == "layer_group"
        )
        second_layer_id = group["parameters"]["layer_ids"][1]
        dense = next(
            item
            for item in payload["graph"]["operators"]
            if item["op_kind"] == "dense_mlp"
        )
        expected = dense["parameters"]["intermediate_size"] + 1
        group["parameters"]["overrides"][second_layer_id][
            "intermediate_size"
        ] = expected

        model = model_from_dict(payload)

        self.assertEqual(
            model_graph_execution_layers(model.graph)[1].intermediate_size,
            expected,
        )

    def test_authoritative_graph_edit_updates_the_derived_execution_geometry(self):
        group = next(
            item for item in self.graph.operators if item.op_kind == "layer_group"
        )
        parameters = dict(group.parameters)
        overrides = {
            key: dict(value) for key, value in parameters["overrides"].items()
        }
        layer_id = parameters["layer_ids"][0]
        overrides[layer_id]["weight_bytes"] = parameters["weight_bytes"] + 1
        parameters["overrides"] = overrides
        changed_group = replace(group, parameters=parameters)
        changed_graph = replace(
            self.graph,
            operators=tuple(
                changed_group if item.operator_id == group.operator_id else item
                for item in self.graph.operators
            ),
        )

        changed_model = replace(self.source, graph=changed_graph)
        self.assertEqual(
            model_graph_execution_layers(changed_model.graph)[0].weight_bytes,
            parameters["weight_bytes"] + 1,
        )

    def test_extra_and_changed_operator_are_rejected(self):
        extra = OperatorNode(
            operator_id="custom-noop",
            op_kind="custom_noop",
            sequence_index=len(self.graph.operators),
        )
        self.assert_gate_rejects(
            replace(self.graph, operators=self.graph.operators + (extra,))
        )

        target = self.graph.operators[0]
        changed = replace(target, op_kind="custom_model_input")
        self.assert_gate_rejects(
            replace(
                self.graph,
                operators=(changed,) + self.graph.operators[1:],
            )
        )

    def test_missing_operator_and_valid_rewire_are_rejected(self):
        final_norm = next(
            item for item in self.graph.operators if item.operator_id == "final_norm"
        )
        lm_head = next(
            item for item in self.graph.operators if item.operator_id == "lm_head"
        )
        previous_id = final_norm.input_tensor_ids[0]
        previous_tensor = next(
            item for item in self.graph.tensors if item.tensor_id == previous_id
        )
        final_tensor_id = final_norm.output_tensor_ids[0]
        final_tensor = next(
            item
            for item in self.graph.tensors
            if item.tensor_id == final_tensor_id
        )
        rewired_head = replace(
            lm_head,
            input_tensor_ids=(previous_id,),
            ports=tuple(
                replace(
                    port,
                    tensor_id=previous_id,
                    dtype=previous_tensor.dtype,
                    shape=previous_tensor.shape,
                    layout=previous_tensor.layout,
                )
                if port.direction == "input"
                else port
                for port in lm_head.ports
            ),
        )
        rewired_previous = replace(
            previous_tensor,
            consumer_operator_ids=tuple(
                item
                for item in previous_tensor.consumer_operator_ids
                if item != final_norm.operator_id
            )
            + (lm_head.operator_id,),
        )
        operators = tuple(
            rewired_head if item.operator_id == lm_head.operator_id else item
            for item in self.graph.operators
            if item.operator_id != final_norm.operator_id
        )
        tensors = tuple(
            rewired_previous if item.tensor_id == previous_id else item
            for item in self.graph.tensors
            if item.tensor_id != final_tensor.tensor_id
        )

        self.assert_gate_rejects(
            replace(self.graph, operators=operators, tensors=tensors)
        )

    def test_extra_edge_with_synchronized_port_and_consumer_is_rejected(self):
        source_tensor = next(
            item for item in self.graph.tensors if item.tensor_id == "input.tokens"
        )
        target = next(
            item
            for item in self.graph.operators
            if item.operator_id.endswith(".norm1")
        )
        changed_target = replace(
            target,
            input_tensor_ids=target.input_tensor_ids + (source_tensor.tensor_id,),
            ports=target.ports
            + (
                OperatorPort(
                    "in1",
                    "input",
                    source_tensor.tensor_id,
                    source_tensor.dtype,
                    source_tensor.shape,
                    source_tensor.layout,
                ),
            ),
        )
        changed_source = replace(
            source_tensor,
            consumer_operator_ids=source_tensor.consumer_operator_ids
            + (target.operator_id,),
        )
        operators = tuple(
            changed_target if item.operator_id == target.operator_id else item
            for item in self.graph.operators
        )
        tensors = tuple(
            changed_source if item.tensor_id == source_tensor.tensor_id else item
            for item in self.graph.tensors
        )

        self.assert_gate_rejects(
            replace(self.graph, operators=operators, tensors=tensors)
        )

    def test_port_and_tensor_contract_change_is_rejected(self):
        target = next(
            item
            for item in self.graph.tensors
            if item.tensor_id == "embedding.output"
        )
        changed_tensor = replace(target, layout="blocked")
        operators = tuple(
            replace(
                operator,
                ports=tuple(
                    replace(port, layout="blocked")
                    if port.tensor_id == target.tensor_id
                    else port
                    for port in operator.ports
                ),
            )
            if any(port.tensor_id == target.tensor_id for port in operator.ports)
            else operator
            for operator in self.graph.operators
        )
        tensors = tuple(
            changed_tensor if item.tensor_id == target.tensor_id else item
            for item in self.graph.tensors
        )

        self.assert_gate_rejects(
            replace(self.graph, operators=operators, tensors=tensors)
        )

    def test_transform_and_unknown_parameter_are_rejected(self):
        source = next(
            item
            for item in self.graph.tensors
            if item.tensor_id == "embedding.output"
        )
        cast_output = TensorValue(
            tensor_id="cast.output",
            role="activation",
            dtype="fp32",
            shape=source.shape,
            layout=source.layout,
        )
        transform = TensorTransform(
            transform_id="cast",
            kind="cast",
            input_tensor_id=source.tensor_id,
            output_tensor_id=cast_output.tensor_id,
            attributes={"to_dtype": "fp32"},
        )
        self.assert_gate_rejects(
            replace(
                self.graph,
                tensors=self.graph.tensors + (cast_output,),
                transforms=(transform,),
            ),
            "Transform.*拒绝静默降级",
        )

        attention = next(
            item for item in self.graph.operators if item.op_kind == "attention"
        )
        parameters = dict(attention.parameters)
        parameters["unknown_execution_knob"] = 1
        changed_attention = replace(attention, parameters=parameters)
        self.assert_gate_rejects(
            replace(
                self.graph,
                operators=tuple(
                    changed_attention
                    if item.operator_id == attention.operator_id
                    else item
                    for item in self.graph.operators
                ),
            )
        )

        # ``provenance`` is ignored only when it is the schema field on a
        # graph/operator/tensor record.  The same spelling inside executable
        # parameters remains semantic and must not bypass the coverage gate or
        # mapping fingerprint.
        reserved_parameters = dict(attention.parameters)
        reserved_parameters["provenance"] = {
            "unexpected_execution_knob": 1
        }
        reserved_attention = replace(
            attention, parameters=reserved_parameters
        )
        self.assert_gate_rejects(
            replace(
                self.graph,
                operators=tuple(
                    reserved_attention
                    if item.operator_id == attention.operator_id
                    else item
                    for item in self.graph.operators
                ),
            )
        )

    def test_post_parse_mutation_is_rechecked_by_validation_and_control_plane(self):
        group = next(
            item for item in self.graph.operators if item.op_kind == "layer_group"
        )
        group.parameters["weight_bytes"] += 1

        report = validate_scenario(self.scenario)

        self.assertFalse(report.is_valid)
        self.assertRegex(report.errors[0], "兼容.*执行器|静默降级")
        with self.assertRaisesRegex(ValueError, "兼容.*执行器|静默降级"):
            plan_runtime_placement(self.scenario)
        with self.assertRaisesRegex(
            CanonicalizationError, "read_source.*权威模型图.*覆盖门禁"
        ):
            compile_canonical_scenario(self.scenario)

    def test_canonical_keeps_expanded_graph_and_authoring_evidence(self):
        canonical = compile_canonical_scenario(self.scenario)
        attributes = canonical.model.attributes

        self.assertIn("dense0.attention", {
            item.operator_id for item in canonical.model.operators
        })
        self.assertEqual(attributes["authoring_graph"]["graph_id"], self.graph.graph_id)
        self.assertEqual(
            attributes["authoring_graph_digest"],
            model_graph_execution_digest(self.graph),
        )
        self.assertTrue(attributes["execution_projection"]["validated"])
        self.assertTrue(attributes["execution_projection"]["lossless"])
        self.assertEqual(
            canonical.model.provenance[0].source_kind,
            "ModelSpec.graph",
        )


if __name__ == "__main__":
    unittest.main()
