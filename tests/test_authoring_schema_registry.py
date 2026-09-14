import unittest
from dataclasses import fields, replace

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.ir import ModelSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive


class GraphAuthoritativeModelTests(unittest.TestCase):
    def test_model_constructor_and_json_have_no_writable_graph_summaries(self):
        model = build_reference_scenario().model
        field_names = {item.name for item in fields(ModelSpec)}
        payload = to_primitive(model)

        self.assertTrue(
            {
                "vocabulary_size",
                "max_sequence_length",
                "embedding_weight_bytes",
                "architecture",
            }.isdisjoint(field_names)
        )
        self.assertTrue(
            {
                "vocabulary_size",
                "max_sequence_length",
                "embedding_weight_bytes",
                "architecture",
            }.isdisjoint(payload)
        )
        self.assertEqual(model.architecture, "decoder_only_transformer")
        self.assertEqual(model.vocabulary_size, 32_000)
        self.assertEqual(model.max_sequence_length, 32_768)
        self.assertEqual(model.embedding_weight_bytes, 16_384_000)

        with self.assertRaises(TypeError):
            ModelSpec(
                name=model.name,
                graph=model.graph,
                vocabulary_size=1,
            )


class ComponentProfileRegistryTests(unittest.TestCase):
    def test_component_binding_resolves_typed_heterogeneous_profiles(self):
        base = build_reference_scenario()
        alternate = replace(
            base.hbm_profile,
            resource_id="gpu0.hbm_fabric.alternate",
        )
        registries = {
            kind: dict(registry)
            for kind, registry in base.component_profiles.items()
        }
        registries["hbm"]["alternate-hbm"] = alternate
        components = tuple(
            replace(component, cost_profile_id="alternate-hbm")
            if component.component_id == "hbm1"
            else component
            for component in base.hardware.components
        )
        scenario = replace(
            base,
            hardware=replace(base.hardware, components=components),
            component_profiles=registries,
        )

        self.assertIsInstance(
            scenario.resolve_component_profile("hbm0"), HBMProfile
        )
        self.assertIs(
            scenario.resolve_component_profile("hbm1"), alternate
        )
        with self.assertRaisesRegex(ValueError, "exactly one hbm profile"):
            _ = scenario.hbm_profile

    def test_typed_component_without_binding_fails_closed(self):
        base = build_reference_scenario()
        components = tuple(
            replace(component, cost_profile_id=None)
            if component.component_id == "hbm0"
            else component
            for component in base.hardware.components
        )

        with self.assertRaisesRegex(ValueError, "requires cost_profile_id"):
            replace(
                base,
                hardware=replace(base.hardware, components=components),
            )


if __name__ == "__main__":
    unittest.main()
