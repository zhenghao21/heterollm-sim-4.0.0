import unittest

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import (
    scenario_to_payload,
    validation_payload,
)


class WebParseErrorLocalizationTests(unittest.TestCase):
    def _payload(self):
        return scenario_to_payload(build_reference_scenario())

    def test_null_flat_kv_field_is_rejected_with_exact_field_diagnostic(self):
        payload = self._payload()
        payload["placement"]["kv_cache_component"] = None

        result = validation_payload(payload)

        issue = result["errors"]["scenario"][0]
        self.assertEqual(issue["code"], "parse_error")
        self.assertEqual(
            issue["message_en"],
            "V4 placement contains unknown fields: kv_cache_component",
        )

    def test_matching_flat_and_typed_kv_fields_are_still_rejected(self):
        payload = self._payload()
        payload["placement"]["kv_cache_component"] = "hbm0"
        payload["placement"]["kv_policy"]["cache_component"] = "hbm0"

        result = validation_payload(payload)

        issue = result["errors"]["scenario"][0]
        self.assertEqual(issue["code"], "parse_error")
        self.assertEqual(
            issue["message_en"],
            "V4 placement contains unknown fields: kv_cache_component",
        )

    def test_malformed_rank_mapping_reports_the_actual_field_constraint(self):
        payload = self._payload()
        parallel = payload["placement"]["parallel"]
        parallel["tp_degree"] = 2
        parallel["rank_mapping"] = parallel["rank_mapping"][:1]

        result = validation_payload(payload)

        issue = result["errors"]["scenario"][0]
        self.assertEqual(issue["code"], "parse_error")
        self.assertIn("placement.parallel.rank_mapping", issue["message"])
        self.assertIn("world_size", issue["message"])
        self.assertNotIn("请检查字段类型、必填项", issue["message"])

if __name__ == "__main__":
    unittest.main()
