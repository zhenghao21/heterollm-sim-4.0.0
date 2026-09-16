"""Pure fixed-source declaration and row-shape contracts; no planner or device work."""
from dataclasses import FrozenInstanceError
import json
import unittest

from heterollm_sim.final_layer_output_selection import (
    SOURCE_KEY,
    FinalLayerOutputPolicy,
    FinalLayerOutputSelection,
    resolve_declaration,
    source_declaration,
)


class FinalLayerOutputSelectionTests(unittest.TestCase):
    def policy(self, architecture="qwen2_decoder"):
        return resolve_declaration(source_declaration(), architecture, mtp_present=False)

    def test_absence_keeps_legacy_path_without_new_architecture_or_mtp_checks(self):
        self.assertIsNone(resolve_declaration(None, "unrelated_legacy_graph", mtp_present=True))
        self.assertIsNone(resolve_declaration(None, None, mtp_present=None))

    def test_declaration_factory_is_fresh_and_does_not_choose_architecture(self):
        first, second = source_declaration(), source_declaration()
        self.assertEqual(SOURCE_KEY, "llama_cpp_final_layer_output_selection")
        self.assertEqual(first, second)
        first["embeddings"] = True
        self.assertFalse(second["embeddings"])
        self.assertNotIn("position", second)
        self.assertNotIn("graph_architecture", second)

    def test_all_exact_graph_architectures_select_the_source_position(self):
        for architecture, position in (
            ("qwen2_decoder", "before_last_ffn"),
            ("qwen2", "before_last_ffn"),
            ("llama_decoder", "before_last_ffn"),
            ("llama", "before_last_ffn"),
            ("qwen3_5_hybrid_transformer", "after_final_norm"),
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(self.policy(architecture).position, position)

    def test_display_names_formats_and_nearby_architectures_are_not_aliases(self):
        for architecture in ("Qwen2.5", "SmolLM2", "Qwen3.8", "IQ4_XS",
                             "qwen35", "qwen2_decoder ", "QWEN2_DECODER", "unknown", None, 1):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                self.policy(architecture)

    def test_malformed_declarations_are_rejected(self):
        for raw in (True, False, 1, [], "enabled", {}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_declaration(raw, "qwen2_decoder", mtp_present=False)
        for field in source_declaration():
            raw = source_declaration()
            del raw[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                resolve_declaration(raw, "qwen2_decoder", mtp_present=False)
        raw = {**source_declaration(), "position": "before_last_ffn"}
        with self.assertRaises(ValueError):
            resolve_declaration(raw, "qwen2_decoder", mtp_present=False)

    def test_wrong_source_branch_or_revision_is_rejected(self):
        changes = {
            "schema_version": "other/v1", "backend_commit": "main", "request_kind": "embedding",
            "explicit_logits": False, "embeddings": True, "embeddings_nextn_masked": True,
            "speculative_type": "mtp",
        }
        for field, value in changes.items():
            raw = {**source_declaration(), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_declaration(raw, "qwen3_5_hybrid_transformer", mtp_present=False)

    def test_boolean_source_fields_do_not_accept_integer_or_string_lookalikes(self):
        for field in ("explicit_logits", "embeddings", "embeddings_nextn_masked"):
            for value in (0, 1, "false", "true", None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resolve_declaration({**source_declaration(), field: value}, "llama_decoder", mtp_present=False)

    def test_any_mtp_presence_is_excluded_and_presence_flag_is_strict(self):
        for value in (True, 0, 1, None, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_declaration(source_declaration(), "llama_decoder", mtp_present=value)

    def test_sparse_group_lane_indices_determine_rows_without_assuming_last_lane(self):
        selection = self.policy().select(token_rows=128, selected_indices=(3, 17, 64, 110))
        self.assertEqual(selection.selected_indices, (3, 17, 64, 110))
        self.assertEqual((selection.token_rows, selection.logit_rows), (128, 4))
        self.assertEqual((selection.ffn_rows, selection.final_norm_rows, selection.gather_count), (4, 4, 2))

    def test_zero_output_is_a_real_selection_with_architecture_specific_ancestors(self):
        for architecture, remaining in (("qwen2_decoder", 0), ("llama_decoder", 0), ("qwen3_5_hybrid_transformer", 128)):
            with self.subTest(architecture=architecture):
                selection = self.policy(architecture).select(token_rows=128, selected_indices=())
                self.assertIsNotNone(selection)
                self.assertEqual(selection.logit_rows, 0)
                self.assertEqual(selection.ffn_rows, remaining)
                self.assertEqual(selection.final_norm_rows, remaining)
                self.assertEqual(selection.gather_count, 0)
                self.assertEqual(selection.source_gather_nodes, 1 if remaining else 2)

    def test_late_selection_keeps_all_rows_before_the_head(self):
        for indices in ((127,), (0, 15, 100)):
            with self.subTest(indices=indices):
                selection = self.policy("qwen3_5_hybrid_transformer").select(token_rows=128, selected_indices=indices)
                self.assertEqual((selection.ffn_rows, selection.final_norm_rows), (128, 128))
                self.assertEqual(selection.logit_rows, len(indices))
                self.assertEqual(selection.gather_count, 1)

    def test_all_selected_rows_and_single_row_do_not_erase_source_gathers(self):
        for architecture, gathers in (("qwen2_decoder", 2), ("llama_decoder", 2), ("qwen3_5_hybrid_transformer", 1)):
            for count in (1, 4, 128):
                with self.subTest(architecture=architecture, count=count):
                    selection = self.policy(architecture).select(token_rows=count, selected_indices=tuple(range(count)))
                    self.assertEqual(selection.logit_rows, count)
                    self.assertEqual(selection.gather_count, gathers)

    def test_invalid_token_rows_are_not_rounded_or_promoted(self):
        for count in (0, -1, True, False, 1.0, "128", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.policy().select(token_rows=count, selected_indices=())

    def test_indices_must_be_immutable_strict_integers_in_source_order(self):
        invalid = ([], [0], {0}, "0", None, (True,), (0.0,), ("0",), (-1,), (4,), (0, 4), (1, 1), (2, 0))
        for indices in invalid:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.policy().select(token_rows=4, selected_indices=indices)

    def test_direct_construction_cannot_bypass_shape_or_architecture_checks(self):
        with self.assertRaises(ValueError):
            FinalLayerOutputPolicy("Qwen3.8")
        with self.assertRaises(ValueError):
            FinalLayerOutputSelection("qwen2_decoder", 2, (0, 2))
        with self.assertRaises(TypeError):
            FinalLayerOutputPolicy("qwen2_decoder", position="after_final_norm")

    def test_policy_and_selection_are_frozen_and_hashable(self):
        policy = self.policy()
        selection = policy.select(token_rows=4, selected_indices=(1, 3))
        with self.assertRaises(FrozenInstanceError):
            policy.graph_architecture = "llama_decoder"
        with self.assertRaises(FrozenInstanceError):
            selection.token_rows = 1
        self.assertEqual({selection}, {policy.select(token_rows=4, selected_indices=(1, 3))})

    def test_audit_is_fresh_json_data_and_retains_zero_row_semantics(self):
        selection = self.policy("qwen3_5_hybrid_transformer").select(token_rows=128, selected_indices=())
        audit = selection.audit_metadata()
        self.assertEqual(json.loads(json.dumps(audit)), audit)
        self.assertEqual((audit["token_rows"], audit["logit_rows"], audit["final_norm_rows"]), (128, 0, 128))
        self.assertEqual((audit["source_gather_nodes"], audit["gather_count"]), (1, 0))
        audit["selected_indices"].append(1)
        audit["final_norm_rows"] = 0
        self.assertEqual(selection.audit_metadata()["selected_indices"], [])
        self.assertEqual(selection.final_norm_rows, 128)


if __name__ == "__main__":
    unittest.main()
