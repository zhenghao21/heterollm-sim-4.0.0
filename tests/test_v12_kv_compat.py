import unittest

from heterollm_sim.config import parallel_from_dict, placement_from_dict
from heterollm_sim.ir import KVCachePolicy


class V4KVPolicyParsingTests(unittest.TestCase):
    def test_nested_policy_parses_as_typed_v3_policy(self):
        placement = placement_from_dict(
            {
                "schema_version": "4.0.0",
                "model_name": "m",
                "hardware_name": "h",
                "parallel": {},
                "kv_policy": {
                    "cache_component": "hbm0",
                    "offload_component": "ssd0",
                },
            }
        )

        self.assertIsInstance(placement.kv_policy, KVCachePolicy)
        self.assertEqual(placement.kv_policy.cache_component, "hbm0")
        self.assertEqual(placement.kv_policy.offload_component, "ssd0")

    def test_nested_null_values_remain_unspecified(self):
        placement = placement_from_dict(
            {
                "schema_version": "4.0.0",
                "model_name": "m",
                "hardware_name": "h",
                "parallel": {},
                "kv_policy": {
                    "cache_component": None,
                    "offload_component": None,
                },
            }
        )

        self.assertIsNone(placement.kv_policy.cache_component)
        self.assertIsNone(placement.kv_policy.offload_component)

    def test_flat_kv_fields_are_rejected_even_when_null(self):
        for field_name in ("kv_cache_component", "kv_offload_component"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^V4 placement contains unknown fields: {field_name}$",
                ):
                    placement_from_dict(
                        {
                            "schema_version": "4.0.0",
                            "model_name": "m",
                            "hardware_name": "h",
                            "parallel": {},
                            "kv_policy": {},
                            field_name: None,
                        }
                    )

    def test_flat_kv_fields_are_rejected_when_nested_values_match(self):
        for flat_name, nested_name in (
            ("kv_cache_component", "cache_component"),
            ("kv_offload_component", "offload_component"),
        ):
            with self.subTest(flat_name=flat_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^V4 placement contains unknown fields: {flat_name}$",
                ):
                    placement_from_dict(
                        {
                            "schema_version": "4.0.0",
                            "model_name": "m",
                            "hardware_name": "h",
                            "parallel": {},
                            "kv_policy": {nested_name: "component0"},
                            flat_name: "component0",
                        }
                    )

    def test_old_page_tokens_alias_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"^V4 placement\.kv_policy contains unknown fields: page_tokens$",
        ):
            placement_from_dict(
                {
                    "schema_version": "4.0.0",
                    "model_name": "m",
                    "hardware_name": "h",
                    "parallel": {},
                    "kv_policy": {"page_tokens": 16},
                }
            )

    def test_empty_rank_mapping_remains_legal(self):
        parallel = parallel_from_dict(
            {
                "tp_degree": 2,
                "pp_degree": 2,
                "ep_degree": 2,
                "rank_mapping": [],
            }
        )

        self.assertEqual(parallel.rank_mapping, ())
        self.assertEqual(parallel.world_size, 8)


if __name__ == "__main__":
    unittest.main()
