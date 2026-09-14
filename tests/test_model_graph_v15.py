import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import heterollm_sim.ir as ir_module
from heterollm_sim.compiler_ir import compile_canonical_scenario
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import (
    model_graph_execution_digest,
    model_graph_execution_view,
)
from heterollm_sim.control_plane_state import (
    MAPPING_FINGERPRINT_SCHEMA,
    mapping_fingerprint_status,
    mapping_input_fingerprint,
)
from heterollm_sim.model_catalog import ModelCatalog
from heterollm_sim.model_presets import (
    OUT_OF_DOMAIN,
    list_model_presets,
    materialize_model_payload,
    model_preset_detail,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.planner import validate_scenario
from heterollm_sim.schema_v1 import ModelGraph, OperatorNode, OperatorPort, TensorTransform, TensorValue
from heterollm_sim.serde import to_primitive


def _transform_graph(kind, input_tensor, output_tensor, attributes=None):
    return ModelGraph(
        graph_id="transform-contract",
        operators=(OperatorNode("anchor", "contract_anchor", 0),),
        tensors=(input_tensor, output_tensor),
        transforms=(
            TensorTransform(
                transform_id="tx",
                kind=kind,
                input_tensor_id=input_tensor.tensor_id,
                output_tensor_id=output_tensor.tensor_id,
                attributes=dict(attributes or {}),
            ),
        ),
    )


def _tensor(tensor_id, dtype="bf16", shape=(2,), layout="logical"):
    return TensorValue(tensor_id, "activation", dtype=dtype, shape=shape, layout=layout)


def _graph_model_payload(payload):
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "name",
            "text_backbone_only",
            "supported_modalities",
            "excluded_subgraphs",
            "metadata",
            "graph",
        )
        if key in payload
    }


def _tiny_graph():
    layer = ir_module.LayerSpec(
        layer_id="dense0",
        kind="dense",
        hidden_size=8,
        intermediate_size=16,
        attention_heads=2,
        weight_bytes=123,
    )
    return layer, ir_module.build_model_graph_from_layer_specs(
        "tiny",
        (layer,),
        vocabulary_size=32,
        max_sequence_length=128,
        embedding_weight_bytes=456,
    )


def _with_lm_head_bytes(graph, logical_bytes):
    return replace(
        graph,
        tensors=tuple(
            replace(tensor, logical_bytes=logical_bytes)
            if tensor.tensor_id == "lm_head_weights"
            else tensor
            for tensor in graph.tensors
        ),
    )


class ModelGraphAuthoringTests(unittest.TestCase):
    def test_graph_execution_view_cache_reuses_and_invalidates_by_graph(self):
        payload = _graph_model_payload(
            materialize_model_payload("qwen3_8-27b")
        )
        graph = model_from_dict(payload).graph
        object.__delattr__(graph, "_execution_view_cache")

        with patch.object(
            ir_module,
            "_assert_execution_projection_coverage",
            wraps=ir_module._assert_execution_projection_coverage,
        ) as coverage:
            first = model_graph_execution_view(graph)
            second = model_graph_execution_view(graph)
            replaced = model_graph_execution_view(
                replace(graph, graph_id=graph.graph_id + "-copy")
            )

        self.assertIs(second, first)
        self.assertIsNot(replaced, first)
        self.assertEqual(coverage.call_count, 2)
        self.assertNotIn("_execution_view_cache", to_primitive(graph))

    def test_model_properties_reuse_one_validated_execution_view(self):
        payload = _graph_model_payload(
            materialize_model_payload("qwen3_8-27b")
        )

        with patch.object(
            ir_module,
            "model_graph_execution_view",
            wraps=ir_module.model_graph_execution_view,
        ) as execution_view:
            model = model_from_dict(payload)
            calls_after_construction = execution_view.call_count

            self.assertEqual(calls_after_construction, 1)
            self.assertGreater(model.num_layers, 0)
            self.assertTrue(model.architecture)
            self.assertGreater(model.vocabulary_size, 0)
            self.assertGreater(model.max_sequence_length, 0)
            self.assertGreater(model.embedding_weight_bytes, 0)
            self.assertGreater(model.total_declared_weight_bytes, 0)
            self.assertEqual(
                execution_view.call_count,
                calls_after_construction,
            )

        self.assertNotIn("_execution_view", to_primitive(model))

    def test_graph_only_model_round_trip_preserves_execution_view(self):
        payload = _graph_model_payload(
            to_primitive(build_reference_scenario().model)
        )
        original = model_from_dict(payload)
        graph = original.graph
        view = model_graph_execution_view(
            graph,
            schema_version=original.schema_version,
        )

        self.assertTrue(graph.executable)
        self.assertTrue(any(item.op_kind == "layer_group" for item in graph.operators))
        self.assertTrue(all(port.dtype and port.layout for item in graph.operators for port in item.ports))
        self.assertTrue(any(tensor.shape == ("B", "T", 512) for tensor in graph.tensors))
        self.assertEqual(len(view.layer_instances), 2)

        restored = model_from_dict(
            _graph_model_payload(to_primitive(original))
        )
        self.assertEqual(
            model_graph_execution_digest(restored.graph),
            model_graph_execution_digest(graph),
        )

    def test_graph_only_projects_layer_instances(self):
        payload = _graph_model_payload(
            materialize_model_payload("mixtral-8x7b-v0_1")
        )

        model = model_from_dict(payload)
        view = model_graph_execution_view(
            model.graph,
            schema_version=model.schema_version,
        )
        expected_count = sum(
            int(operator.parameters["repeat"])
            for operator in model.graph.operators
            if operator.op_kind == "layer_group"
        )

        self.assertEqual(len(view.layer_instances), expected_count)
        self.assertTrue(
            all(descriptor.layer.is_moe for descriptor in view.layer_instances)
        )

    def test_graph_layer_groups_define_execution_projection_order(self):
        model = model_from_dict(
            _graph_model_payload(materialize_model_payload("qwen2_5-0_5b"))
        )
        view = model_graph_execution_view(
            model.graph,
            schema_version=model.schema_version,
        )
        expected_layer_ids = tuple(
            str(layer_id)
            for operator in model.graph.operators
            if operator.op_kind == "layer_group"
            for layer_id in operator.parameters["layer_ids"]
        )

        self.assertEqual(
            tuple(descriptor.layer_id for descriptor in view.layer_instances),
            expected_layer_ids,
        )

    def test_lm_head_positive_logical_bytes_count_as_untied_storage(self):
        layer, graph = _tiny_graph()
        output_weight_bytes = 789
        model = ir_module.ModelSpec(
            name="untied-output-head",
            graph=_with_lm_head_bytes(graph, output_weight_bytes),
        )
        view = model_graph_execution_view(model.graph)

        self.assertEqual(view.output_weight_bytes, output_weight_bytes)
        self.assertEqual(model.output_weight_bytes, output_weight_bytes)
        self.assertEqual(
            model.total_declared_weight_bytes,
            model.embedding_weight_bytes + output_weight_bytes + layer.weight_bytes,
        )

    def test_lm_head_zero_logical_bytes_preserves_tied_alias(self):
        layer, graph = _tiny_graph()
        tied = ir_module.ModelSpec(
            name="zero-byte-output-head-alias",
            graph=_with_lm_head_bytes(graph, 0),
        )

        self.assertEqual(tied.output_weight_bytes, 0)
        self.assertEqual(
            tied.total_declared_weight_bytes,
            tied.embedding_weight_bytes + layer.weight_bytes,
        )
        self.assertEqual(
            model_graph_execution_digest(tied.graph),
            model_graph_execution_digest(graph),
        )

    def test_typed_mtp_branch_is_authoritative_execution_contract(self):
        payload = _graph_model_payload(
            materialize_model_payload("qwen3_8-27b")
        )
        original = model_from_dict(payload)
        graph = original.graph
        view = model_graph_execution_view(
            graph,
            schema_version=original.schema_version,
        )
        operators = {item.operator_id: item for item in graph.operators}
        tensors = {item.tensor_id: item for item in graph.tensors}
        descriptors = {
            descriptor.operator.operator_id: descriptor
            for descriptor in view.mtp_descriptors
        }

        self.assertEqual(operators["output"].input_tensor_ids, ("logits",))
        self.assertIn("logits", tensors)
        self.assertIn("mtp.proposal_logits", tensors)
        branch_source = operators["final_norm"].input_tensor_ids[0]
        prediction = operators["mtp.prediction_layer.000"]
        aux_head = operators["mtp.aux_head"]
        self.assertEqual(prediction.op_kind, "mtp_prediction_layer")
        self.assertEqual(prediction.input_tensor_ids, (branch_source,))
        self.assertEqual(
            prediction.weight_tensor_ids,
            ("mtp.prediction_layer.000.weights",),
        )
        self.assertEqual(aux_head.op_kind, "mtp_aux_head")
        self.assertEqual(
            aux_head.input_tensor_ids,
            ("mtp.prediction_layer.000.output",),
        )
        self.assertEqual(aux_head.output_tensor_ids, ("mtp.proposal_logits",))
        self.assertEqual(set(descriptors), {prediction.operator_id, aux_head.operator_id})
        mtp_weight_bytes = sum(
            descriptor.weight_bytes for descriptor in view.mtp_descriptors
        )
        self.assertGreater(tensors["mtp.prediction_layer.000.weights"].logical_bytes, 0)
        self.assertGreater(tensors["mtp.aux_head.weights"].logical_bytes, 0)
        self.assertEqual(
            mtp_weight_bytes,
            sum(
                tensors[descriptor.weight_tensor.tensor_id].logical_bytes
                for descriptor in view.mtp_descriptors
            ),
        )

        restored = model_from_dict(
            _graph_model_payload(to_primitive(original))
        )
        self.assertEqual(
            model_graph_execution_digest(restored.graph),
            model_graph_execution_digest(original.graph),
        )

    def test_graph_native_execution_view_expands_layers_and_orders_mtp_dependencies(self):
        model = model_from_dict(
            _graph_model_payload(materialize_model_payload("qwen3_8-27b"))
        )
        view = model_graph_execution_view(model.graph)
        expected_layer_ids = tuple(
            str(layer_id)
            for operator in model.graph.operators
            if operator.op_kind == "layer_group"
            for layer_id in operator.parameters["layer_ids"]
        )

        self.assertEqual(
            tuple(item.layer_id for item in view.layer_instances),
            expected_layer_ids,
        )
        self.assertEqual(
            [item.operator.op_kind for item in view.mtp_descriptors],
            ["mtp_prediction_layer", "mtp_aux_head"],
        )
        order = {
            operator.operator_id: index
            for index, operator in enumerate(view.operators)
        }
        self.assertLess(order["block-group-031.residual2"], order["final_norm"])
        self.assertLess(
            order["block-group-031.residual2"],
            order["mtp.prediction_layer.000"],
        )
        self.assertLess(
            order["mtp.prediction_layer.000"], order["mtp.aux_head"]
        )

    def test_typed_mtp_weight_bytes_survive_graph_only_round_trip(self):
        model = model_from_dict(
            _graph_model_payload(materialize_model_payload("qwen3_8-27b"))
        )
        view = model_graph_execution_view(
            model.graph,
            schema_version=model.schema_version,
        )
        expected = tuple(
            (
                descriptor.operator.operator_id,
                descriptor.weight_tensor.tensor_id,
                descriptor.weight_bytes,
            )
            for descriptor in view.mtp_descriptors
        )
        self.assertTrue(expected)
        self.assertTrue(all(weight_bytes > 0 for _, _, weight_bytes in expected))

        reparsed = model_from_dict(
            _graph_model_payload(to_primitive(model))
        )
        reparsed_view = model_graph_execution_view(
            reparsed.graph,
            schema_version=reparsed.schema_version,
        )
        self.assertEqual(
            tuple(
                (
                    descriptor.operator.operator_id,
                    descriptor.weight_tensor.tensor_id,
                    descriptor.weight_bytes,
                )
                for descriptor in reparsed_view.mtp_descriptors
            ),
            expected,
        )

    def test_port_tensor_dimension_mismatch_is_rejected_with_expected_and_actual(self):
        graph = build_reference_scenario().model.graph
        target = next(item for item in graph.tensors if item.shape == ("B", "T", 512))
        bad = replace(target, shape=("B", "T", 513))
        tensors = tuple(bad if item.tensor_id == target.tensor_id else item for item in graph.tensors)

        with self.assertRaisesRegex(ValueError, "期望.*实际"):
            replace(graph, tensors=tensors)

    def test_unknown_transform_kind_is_rejected_in_chinese(self):
        with self.assertRaisesRegex(ValueError, "显式变换.*不支持的类型"):
            TensorTransform("bad", "flatten", "input", "output")

    def test_cyclic_model_graph_is_rejected_in_chinese(self):
        dtype = "bf16"
        shape = (2,)
        operators = (
            OperatorNode(
                "a", "linear", 0,
                input_tensor_ids=("tb",),
                output_tensor_ids=("ta",),
                ports=(
                    OperatorPort("in0", "input", "tb", dtype, shape),
                    OperatorPort("out0", "output", "ta", dtype, shape),
                ),
            ),
            OperatorNode(
                "b", "linear", 1,
                input_tensor_ids=("ta",),
                output_tensor_ids=("tb",),
                ports=(
                    OperatorPort("in0", "input", "ta", dtype, shape),
                    OperatorPort("out0", "output", "tb", dtype, shape),
                ),
            ),
        )
        tensors = (
            TensorValue("ta", "activation", producer_operator_id="a", consumer_operator_ids=("b",), dtype=dtype, shape=shape),
            TensorValue("tb", "activation", producer_operator_id="b", consumer_operator_ids=("a",), dtype=dtype, shape=shape),
        )

        with self.assertRaisesRegex(ValueError, "模型组件图必须是 DAG"):
            ModelGraph("cycle", operators, tensors)

    def test_stale_weight_consumer_is_rejected(self):
        graph = build_reference_scenario().model.graph
        operator = next(item for item in graph.operators if item.weight_tensor_ids)
        changed = replace(
            operator,
            weight_tensor_ids=(),
            ports=tuple(port for port in operator.ports if port.direction != "weight"),
        )
        operators = tuple(changed if item.operator_id == operator.operator_id else item for item in graph.operators)

        with self.assertRaisesRegex(ValueError, "消费组件索引已过期.*input/weight"):
            replace(graph, operators=operators)

    def test_representable_single_tensor_transform_contracts_are_validated(self):
        cases = (
            (
                "reshape",
                _tensor("reshape.input", shape=(2, 3, 4)),
                _tensor("reshape.output", shape=(6, 4)),
                {"target_shape": (6, -1)},
            ),
            (
                "transpose",
                _tensor("transpose.input", shape=("B", "T", "H")),
                _tensor("transpose.output", shape=("T", "B", "H"), layout="logical_transposed"),
                {"permutation": (1, 0, 2)},
            ),
            (
                "cast",
                _tensor("cast.input", dtype="bf16", shape=(2,)),
                _tensor("cast.output", dtype="fp32", shape=(2,)),
                {"to_dtype": "float32"},
            ),
            (
                "broadcast",
                _tensor("broadcast.input", shape=("B", 1, "H")),
                _tensor("broadcast.output", shape=("B", "T", "H")),
                {"target_shape": ("B", "T", "H")},
            ),
        )

        for kind, input_tensor, output_tensor, attributes in cases:
            with self.subTest(kind=kind):
                graph = _transform_graph(kind, input_tensor, output_tensor, attributes)
                self.assertEqual(graph.transforms[0].kind, kind)

    def test_concat_and_split_are_known_but_fail_closed_until_schema_has_tuple_refs(self):
        for kind in ("concat", "split"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "无法完整表达"):
                    _transform_graph(kind, _tensor(kind + ".input"), _tensor(kind + ".output"), {"axis": 0, "sections": 2})

    def test_canonical_compiler_preserves_placement_node_ids_and_adds_ports(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        ids = {item.operator_id for item in canonical.model.operators}

        self.assertTrue({"dense0.attention", "dense0.mlp", "moe1.router", "moe1.experts"}.issubset(ids))
        dense = next(item for item in canonical.model.operators if item.operator_id == "dense0.mlp")
        self.assertEqual(dense.ports[0].shape, ("B", "T", 512))

    def test_graph_component_precision_contract_is_rejected_before_lowering(self):
        scenario = build_reference_scenario()
        payload = _graph_model_payload(to_primitive(scenario.model))
        group = next(item for item in payload["graph"]["operators"] if item["op_kind"] == "layer_group")
        group["parameters"]["dtype"] = "mystery"
        parent_id = group["operator_id"]
        feed_forward = next(
            item
            for item in payload["graph"]["operators"]
            if item.get("attributes", {}).get("parent_group_id") == parent_id
            and item["op_kind"] in {"dense_mlp", "moe_router", "moe_experts"}
        )
        feed_forward["parameters"]["quantization"] = None
        with self.assertRaisesRegex(ValueError, "无损覆盖.*dtype|dtype.*静默降级"):
            model_from_dict(payload)


class ModelPresetGraphTests(unittest.TestCase):
    def test_all_147_presets_materialize_deterministic_graphs(self):
        presets = list_model_presets()
        self.assertEqual(len(presets), 147)

        for item in presets:
            detail = model_preset_detail(item["id"])
            graph = detail["graph"]
            self.assertEqual(graph, model_preset_detail(item["id"])["graph"])
            self.assertIn("operators", graph)
            if item["support_level"] == OUT_OF_DOMAIN:
                self.assertIsNone(detail["model"])
                self.assertFalse(graph["executable"])
                self.assertTrue(any(node["op_kind"] == "unsupported_component_group" for node in graph["operators"]))
            else:
                self.assertTrue(graph["executable"])
                self.assertEqual(detail["model"]["graph"], graph)

    def test_model_kind_filter_facets(self):
        catalog = ModelCatalog()
        dense = catalog.page(model_kind="dense", limit=200)
        moe = catalog.page(model_kind="moe", limit=200)

        self.assertGreater(dense["total"], 0)
        self.assertGreater(moe["total"], 0)
        self.assertTrue(all(item["model_kind"] == "dense" for item in dense["items"]))
        self.assertTrue(all(item["model_kind"] == "moe" for item in moe["items"]))
        facet_values = {item["value"] for item in dense["facets"]["model_kind"]}
        self.assertTrue({"dense", "moe", "state_space"}.issubset(facet_values))


class ModelGraphMappingFingerprintTests(unittest.TestCase):
    def setUp(self):
        base = build_reference_scenario()
        fingerprint = mapping_input_fingerprint(base)
        metadata = dict(base.placement.metadata)
        metadata["control_plane"] = {
            "policy": {},
            "decision": {},
            "evidence": {
                "input_fingerprint": fingerprint,
                "fingerprint_schema": MAPPING_FINGERPRINT_SCHEMA,
            },
        }
        self.mapped = replace(base, placement=replace(base.placement, metadata=metadata))

    def _with_graph(self, graph):
        payload = _graph_model_payload(to_primitive(self.mapped.model))
        payload["graph"] = to_primitive(graph)
        model = model_from_dict(payload)
        return replace(
            self.mapped,
            model=model,
        )

    def test_layout_only_change_does_not_stale_mapping(self):
        graph = self.mapped.model.graph
        attributes = dict(graph.attributes)
        attributes["ui"] = {"zoom": 2.5, "positions": {"embedding": [123, 456]}}
        changed = self._with_graph(replace(graph, attributes=attributes))

        self.assertFalse(mapping_fingerprint_status(changed)["mapping_stale"])

    def test_unprojectable_semantic_parameter_and_edge_changes_are_rejected(self):
        graph = self.mapped.model.graph
        group = next(item for item in graph.operators if item.op_kind == "layer_group")
        parameters = dict(group.parameters)
        parameters["weight_bytes"] += 1
        changed_group = replace(group, parameters=parameters)
        parameter_graph = replace(
            graph,
            operators=tuple(changed_group if item.operator_id == group.operator_id else item for item in graph.operators),
        )
        with self.assertRaisesRegex(ValueError, "无损覆盖|静默降级"):
            self._with_graph(parameter_graph)

        residual = next(item for item in graph.operators if item.op_kind == "residual_add")
        changed_residual = replace(
            residual,
            input_tensor_ids=tuple(reversed(residual.input_tensor_ids)),
            ports=tuple(reversed(residual.ports)),
        )
        edge_graph = replace(
            graph,
            operators=tuple(changed_residual if item.operator_id == residual.operator_id else item for item in graph.operators),
        )
        with self.assertRaisesRegex(ValueError, "无损覆盖|静默降级"):
            self._with_graph(edge_graph)


if __name__ == "__main__":
    unittest.main()
