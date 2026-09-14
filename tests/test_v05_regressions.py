import http.client
import json
import threading
import unittest

from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.model_presets import list_model_presets
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import build_server, scenario_to_payload


def _json_clone(value):
    return json.loads(json.dumps(value))


def _max_depth(value):
    if isinstance(value, dict):
        return 1 + max((_max_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_max_depth(item) for item in value), default=0)
    return 0


def _issue_codes(section):
    return {str(item.get("code")) for item in section}


def _issue_text(section):
    return "\n".join(str(item.get("message", "")) for item in section)


def _facet_values(values):
    result = set()
    for item in values:
        if isinstance(item, dict):
            result.add(str(item.get("value") or item.get("family") or item.get("id")))
        else:
            result.add(str(item))
    return result


class V05ApiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = build_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5.0)
        cls.server.server_close()

    def json_request(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def reference_payload(self):
        return scenario_to_payload(build_reference_scenario())

    def test_external_malformed_rank_mapping_remains_strict_parse_error(self):
        scenario = self.reference_payload()
        placement = scenario["placement"]
        parallel = placement["parallel"]
        parallel["tp_degree"] = 1
        parallel["pp_degree"] = 1
        parallel["ep_degree"] = 1
        parallel["rank_mapping"] = [
            {
                "rank": 0,
                "component_id": "gpu0",
                "tp_rank": 0,
                "pp_rank": 0,
                "ep_rank": 0,
            },
            {
                "rank": 1,
                "component_id": "gpu0",
                "tp_rank": 1,
                "pp_rank": 0,
                "ep_rank": 0,
            },
        ]

        status, payload = self.json_request("POST", "/api/validate", scenario)

        self.assertEqual(status, 200, payload)
        scenario_errors = payload["errors"]["scenario"]
        self.assertFalse(payload["valid"])
        self.assertIn("parse_error", _issue_codes(scenario_errors))
        self.assertIn("rank_mapping", _issue_text(scenario_errors))

    def test_single_gpu_parallel_resource_shortage_is_validation_not_parse_error(self):
        scenario = self.reference_payload()
        scenario["placement"]["parallel"]["tp_degree"] = 2
        scenario["placement"]["parallel"]["allow_padding"] = False
        scenario["placement"]["parallel"]["rank_mapping"] = []

        status, payload = self.json_request("POST", "/api/validate", scenario)

        self.assertEqual(status, 200, payload)
        scenario_errors = payload["errors"]["scenario"]
        self.assertNotIn("parse_error", _issue_codes(scenario_errors))
        self.assertIn("validation_error", _issue_codes(scenario_errors))
        self.assertIn("并行域需要 2 个计算组件", _issue_text(scenario_errors))
        self.assertIn(
            "parallel world requires 2 compute components",
            "\n".join(item.get("message_en", "") for item in scenario_errors),
        )

    def test_rejected_request_reason_is_serialized_and_excluded_from_percentiles(self):
        scenario = self.reference_payload()
        scenario["placement"]["kv_policy"]["cache_component"] = "hbm7"
        hbm7 = next(
            component
            for component in scenario["hardware"]["components"]
            if component["component_id"] == "hbm7"
        )
        hbm7["capacity_bytes"] = 16 * 1024
        request = scenario["workload"]["requests"][0]
        good = dict(request)
        good.update({"request_id": "good", "prompt_tokens": 1, "output_tokens": 2})
        rejected = dict(request)
        rejected.update({"request_id": "too-large", "prompt_tokens": 17, "output_tokens": 1})
        scenario["workload"]["requests"] = [good, rejected]
        scenario["workload"]["request_count"] = 2

        status, payload = self.json_request("POST", "/api/run", scenario)

        self.assertEqual(status, 200, payload)
        good_row = payload["requests"]["good"]
        rejected_row = payload["requests"]["too-large"]
        self.assertEqual(good_row["status"], "finished")
        self.assertEqual(rejected_row["status"], "rejected")
        self.assertTrue(rejected_row.get("reason"), rejected_row)
        self.assertGreater(good_row["e2e_ns"], 0.0)
        self.assertEqual(payload["summary"]["e2e_ns"]["p50"], good_row["e2e_ns"])
        self.assertEqual(payload["summary"]["e2e_ns"]["p95"], good_row["e2e_ns"])
        self.assertEqual(payload["summary"]["e2e_ns"]["p99"], good_row["e2e_ns"])

    def test_generated_control_plane_without_fingerprint_fails_closed(self):
        scenario = self.reference_payload()
        request = scenario["workload"]["requests"][0]
        request["request_id"] = "missing-control-plane-fingerprint"
        request["prompt_tokens"] = 28
        request["output_tokens"] = 4
        stale_kv_bytes = 16 * 1024
        scenario["placement"]["tensor_bytes"]["kv_cache"] = stale_kv_bytes
        # Generated-output metadata marks this as a completed placement.  V4 then
        # requires both its deployment fingerprint and schema marker.
        scenario["placement"]["metadata"]["control_plane"] = {
            "policy": {},
            "decision": {
                "generated_tensor_ids": ["kv_cache"],
                "derived_tensor_bytes": {"kv_cache": stale_kv_bytes},
            },
            "evidence": {},
        }

        status, diagnostics = self.json_request("POST", "/api/validate", scenario)

        self.assertEqual(status, 200, diagnostics)
        self.assertFalse(diagnostics["valid"], diagnostics)
        scenario_errors = diagnostics["errors"]["scenario"]
        self.assertIn("parse_error", _issue_codes(scenario_errors))
        diagnostic_text_en = "\n".join(
            str(item.get("message_en", "")) for item in scenario_errors
        )
        self.assertIn("placement.tensor_bytes", diagnostic_text_en)
        self.assertIn("CPU control plane", diagnostic_text_en)
        self.assertNotIn("mapping_stale", diagnostics)

        status, report = self.json_request("POST", "/api/run", scenario)

        self.assertEqual(status, 422, report)
        self.assertEqual(report["error"]["code"], "invalid_scenario")
        run_errors = report["error"]["details"]["errors"]["scenario"]
        self.assertIn("parse_error", _issue_codes(run_errors))

    def test_model_preset_page_families_facet_is_global_under_family_filter(self):
        status, page = self.json_request(
            "GET",
            "/api/model-presets?limit=2&offset=0&family=Qwen3.8",
        )

        self.assertEqual(status, 200, page)
        self.assertTrue(all(item["family"] == "Qwen3.8" for item in page["items"]))
        facets = page.get("facets", {})
        self.assertIn("family", facets, page)
        expected_families = {item["family"] for item in list_model_presets()}
        self.assertLess({"Qwen3.8"}, expected_families)
        self.assertLessEqual(expected_families, _facet_values(facets["family"]))


class V05PublicApiRegressionTests(unittest.TestCase):
    def test_explicit_colocated_ranks_parse_and_build_plan(self):
        scenario = scenario_to_payload(build_reference_scenario())
        parallel = scenario["placement"]["parallel"]
        parallel["tp_degree"] = 2
        parallel["pp_degree"] = 1
        parallel["ep_degree"] = 1
        parallel["rank_mapping"] = [
            {
                "rank": 0,
                "component_id": "gpu0",
                "tp_rank": 0,
                "pp_rank": 0,
                "ep_rank": 0,
                "memory_component_id": "hbm0",
                "cim_component_id": "cim0",
            },
            {
                "rank": 1,
                "component_id": "gpu0",
                "tp_rank": 1,
                "pp_rank": 0,
                "ep_rank": 0,
                "memory_component_id": "hbm0",
                "cim_component_id": "cim0",
            },
        ]

        parsed = scenario_from_dict(scenario)
        plan = build_parallel_plan(parsed)

        self.assertEqual(plan.world_size, 2)
        self.assertEqual(
            tuple(rank.component_id for rank in plan.ranks),
            ("gpu0", "gpu0"),
        )
        self.assertEqual(
            {rank.coordinates for rank in plan.ranks},
            {(0, 0, 0), (1, 0, 0)},
        )

    def test_control_plane_apply_preserves_hardware_and_topology_view_depth(self):
        payload = scenario_to_payload(build_reference_scenario())
        payload["hardware"]["metadata"]["topology_view"] = {
            "version": 1,
            "layout": {
                "positions": {
                    "gpu0": {
                        "x": 10,
                        "y": 20,
                        "anchors": [{"label": "rack", "point": {"x": 1, "y": 2}}],
                    }
                },
                "viewport": {"x": 0, "y": 0, "scale": 1},
            },
            "groups": [
                {
                    "group_id": "package0",
                    "members": ["gpu0", "hbm0", "cim0"],
                    "root": "gpu0",
                    "collapsed": False,
                    "style": {"tokens": [{"name": "accent", "value": "copper"}]},
                }
            ],
        }
        scenario = scenario_from_dict(payload)
        before = scenario_to_payload(scenario)

        mapped = plan_runtime_placement(scenario).apply(scenario)
        after = scenario_to_payload(mapped)

        self.assertEqual(_max_depth(after["hardware"]), _max_depth(before["hardware"]))
        self.assertEqual(
            _max_depth(after["hardware"]["metadata"]["topology_view"]),
            _max_depth(before["hardware"]["metadata"]["topology_view"]),
        )
        self.assertEqual(
            after["hardware"]["metadata"]["topology_view"],
            before["hardware"]["metadata"]["topology_view"],
        )


if __name__ == "__main__":
    unittest.main()
