from contextlib import redirect_stderr
import http.client
from io import StringIO
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from heterollm_sim import __version__
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.schema_v1 import CanonicalScenario
from heterollm_sim.web import JSON_BODY_LIMIT_BYTES, build_server, scenario_to_payload


class WebApiTests(unittest.TestCase):
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

    def request(self, method, path, payload=None, body=None, headers=None):
        request_headers = dict(headers or {})
        request_body = body
        if payload is not None:
            request_body = json.dumps(payload).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            connection.request(method, path, body=request_body, headers=request_headers)
            response = connection.getresponse()
            response_body = response.read()
            return response.status, dict(response.getheaders()), response_body
        finally:
            connection.close()

    def raw_socket_request(self, method, path, body, headers=None, chunk_size=64 * 1024):
        request_headers = dict(headers or {})
        request_headers.setdefault("Content-Length", str(len(body)))
        request_headers.setdefault("Host", "127.0.0.1:{}".format(self.port))
        request_headers.setdefault("Connection", "close")
        request_lines = ["{} {} HTTP/1.1".format(method, path)]
        request_lines.extend(
            "{}: {}".format(name, value)
            for name, value in request_headers.items()
        )
        request_head = ("\r\n".join(request_lines) + "\r\n\r\n").encode("ascii")

        with socket.create_connection(("127.0.0.1", self.port), timeout=10.0) as sock:
            sock.settimeout(10.0)
            try:
                sock.sendall(request_head)
                view = memoryview(body)
                for offset in range(0, len(view), chunk_size):
                    sock.sendall(view[offset : offset + chunk_size])
                response = http.client.HTTPResponse(sock)
                response.begin()
                response_body = response.read()
            except OSError as exc:
                self.fail(
                    "oversized upload should return HTTP 413 JSON instead of "
                    "resetting the connection: {!r}".format(exc)
                )
            return response.status, dict(response.getheaders()), response_body

    def json_request(self, method, path, payload=None, body=None, headers=None):
        status, response_headers, response_body = self.request(
            method, path, payload=payload, body=body, headers=headers
        )
        return status, response_headers, json.loads(response_body.decode("utf-8"))

    def reference_payload(self):
        return scenario_to_payload(build_reference_scenario())

    def json_body_size(self, payload):
        return len(json.dumps(payload).encode("utf-8"))

    def reference_payload_at_least(self, minimum_bytes):
        payload = self.reference_payload()
        payload["assumptions"] = ["x"]
        filler_bytes = minimum_bytes - self.json_body_size(payload)
        self.assertGreater(filler_bytes, 0)
        payload["assumptions"] = ["x" * filler_bytes]
        while self.json_body_size(payload) < minimum_bytes:
            payload["assumptions"][0] += "x"
        return payload

    def test_health_endpoint_reports_package_version(self):
        status, _, payload = self.json_request("GET", "/api/health")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["version"], __version__)
        self.assertEqual(payload["service"], "heterollm-sim")
        self.assertTrue(payload["runtime"]["python_version"])
        self.assertTrue(payload["runtime"]["executable"])
        self.assertIn(payload["runtime"]["architecture_bits"], {32, 64})
        self.assertTrue(payload["runtime"]["platform"])
        self.assertIn("version", payload["ortools"])
        self.assertIsInstance(payload["ortools"]["cp_sat_available"], bool)
        self.assertIsInstance(payload["ortools"]["probe_ok"], bool)
        self.assertIn("error", payload["ortools"])
        self.assertIn("error_type", payload["ortools"])
        expected_solvers = ["auto", "builtin"]
        if payload["ortools"]["available"]:
            expected_solvers.append("ortools")
        self.assertEqual(payload["available_solvers"], expected_solvers)

    def test_unexpected_exception_returns_safe_diagnostic_and_logs_traceback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory).joinpath("diagnostics.log")
            previous_log_path = self.server.diagnostic_log_path
            self.server.diagnostic_log_path = log_path
            terminal = StringIO()
            try:
                with patch(
                    "heterollm_sim.web.build_reference_scenario",
                    side_effect=RuntimeError("secret D:/private/runtime-path"),
                ), redirect_stderr(terminal):
                    status, _, payload = self.json_request("GET", "/api/reference")
            finally:
                self.server.diagnostic_log_path = previous_log_path

            self.assertEqual(status, 500)
            error = payload["error"]
            self.assertEqual(error["code"], "internal_error")
            self.assertIn("诊断 ID", error["message_zh"])
            self.assertEqual(error["exception_type"], "RuntimeError")
            self.assertRegex(error["diagnostic_id"], r"^[0-9a-f]{32}$")
            response_text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("secret", response_text)
            self.assertNotIn("private/runtime-path", response_text)
            self.assertNotIn("Traceback", response_text)

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn(error["diagnostic_id"], log_text)
            self.assertIn("Traceback", log_text)
            self.assertIn("RuntimeError: secret D:/private/runtime-path", log_text)
            self.assertIn("RuntimeError: secret D:/private/runtime-path", terminal.getvalue())

    def test_reference_endpoint_returns_bundled_scenario(self):
        status, _, payload = self.json_request("GET", "/api/reference")

        self.assertEqual(status, 200)
        self.assertEqual(payload["name"], build_reference_scenario().name)
        self.assertEqual(len(payload["hardware"]["components"]), 12)
        self.assertEqual(scenario_from_dict(payload), build_reference_scenario())

    def test_protocol_preset_endpoints_filter_and_return_simulation_defaults(self):
        status, _, page = self.json_request(
            "GET",
            "/api/protocol-presets?query=x16&protocol=PCIe&organization=PCI-SIG",
        )

        self.assertEqual(status, 200)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], "pcie-5_0-x16")
        self.assertIn("protocol", page["filters"])
        self.assertEqual(page["catalog"]["version"], "1.0.0")

        status, _, detail = self.json_request(
            "GET", "/api/protocol-presets/pcie-5_0-x16"
        )

        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["version"], "5.0")
        self.assertAlmostEqual(
            detail["simulation_defaults"]["link"]["bandwidth_gbps"],
            504.12307692307695,
        )
        self.assertEqual(
            detail["simulation_defaults"]["link"]["metadata"]["bandwidth_semantics"],
            "one_way_capacity",
        )

    def test_unknown_protocol_preset_returns_bilingual_404(self):
        status, _, payload = self.json_request(
            "GET", "/api/protocol-presets/not-a-real-protocol"
        )

        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")
        self.assertEqual(payload["error"]["message_zh"], "未找到指定的通信协议预设")
        self.assertEqual(
            payload["error"]["message_en"],
            "unknown communication protocol preset",
        )

    def test_validate_endpoint_reports_valid_scenario(self):
        status, _, payload = self.json_request(
            "POST", "/api/validate", payload=self.reference_payload()
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"], payload["errors"])
        self.assertEqual(payload["errors"]["topology"], [])
        self.assertEqual(payload["errors"]["scenario"], [])
        self.assertGreater(len(payload["warnings"]["scenario"]), 0)

    def test_first_click_endpoints_accept_valid_scenario_just_over_one_mib(self):
        previous_limit = 1024 * 1024
        scenario = self.reference_payload_at_least(previous_limit + 1024)
        self.assertLess(self.json_body_size(scenario), JSON_BODY_LIMIT_BYTES)

        status, _, payload = self.json_request(
            "POST", "/api/validate", payload=scenario
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"], payload["errors"])

        status, _, estimate = self.json_request(
            "POST", "/api/run-estimate", payload=scenario
        )
        self.assertEqual(status, 200)
        self.assertIn(estimate["risk_level"], {"low", "medium", "high", "critical"})

    def test_raw_socket_oversized_upload_returns_json_413_without_reset(self):
        prefix = b'{"oversized":"'
        suffix = b'"}'
        body = prefix + b"x" * (JSON_BODY_LIMIT_BYTES + 1) + suffix

        status, headers, response_body = self.raw_socket_request(
            "POST",
            "/api/validate",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        payload = json.loads(response_body.decode("utf-8"))

        self.assertEqual(status, 413)
        self.assertIn("application/json", headers["Content-Type"])
        error = payload["error"]
        self.assertEqual(error["code"], "body_too_large")
        self.assertEqual(error["details"]["actual_bytes"], len(body))
        self.assertEqual(error["details"]["limit_bytes"], JSON_BODY_LIMIT_BYTES)

        status, _, health = self.json_request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])

    def test_validate_endpoint_keeps_information_out_of_warnings(self):
        status, _, payload = self.json_request(
            "POST", "/api/validate", payload=self.reference_payload()
        )

        self.assertEqual(status, 200)
        information_en = {
            item["message_en"] for item in payload["information"]["scenario"]
        }
        warnings_en = {
            item["message_en"] for item in payload["warnings"]["scenario"]
        }
        expected_information = {
            "all operators, collectives, KV traffic, and host orchestration use "
            "the task/transaction-level analytical lowering",
            "request scheduling uses deterministic continuous batching",
        }
        self.assertLessEqual(expected_information, information_en)
        self.assertFalse(expected_information & warnings_en)
        self.assertTrue(
            all(
                item["code"] == "validation_information"
                for item in payload["information"]["scenario"]
            )
        )

    def test_validate_parse_failure_keeps_three_channel_shape(self):
        status, _, payload = self.json_request(
            "POST", "/api/validate", payload={"name": "incomplete"}
        )

        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["errors"]["topology"], [])
        self.assertTrue(payload["errors"]["scenario"])
        self.assertEqual(payload["warnings"], {"topology": [], "scenario": []})
        self.assertEqual(payload["information"], {"topology": [], "scenario": []})

    def test_validate_endpoint_reports_topology_errors_with_v4_empty_placement(self):
        scenario = self.reference_payload()
        scenario["hardware"]["links"][0]["target_component"] = "missing-hbm"

        status, _, payload = self.json_request("POST", "/api/validate", payload=scenario)

        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertTrue(
            any(item["code"] == "unknown_component" for item in payload["errors"]["topology"])
        )
        self.assertEqual(payload["errors"]["scenario"], [])

    def test_validate_endpoint_rejects_nonempty_v4_placement_fields(self):
        for field, value in (
            ("op_to_component", {"dense0.input_norm.reduce": "cim0"}),
            ("tensor_to_component", {"model_weights": "hbm0"}),
            ("tensor_bytes", {"model_weights": 1}),
        ):
            with self.subTest(field=field):
                scenario = self.reference_payload()
                scenario["placement"][field] = value

                status, _, payload = self.json_request(
                    "POST", "/api/validate", payload=scenario
                )

                self.assertEqual(status, 200, payload)
                self.assertFalse(payload["valid"])
                self.assertEqual(payload["errors"]["topology"], [])
                self.assertEqual(payload["errors"]["scenario"][0]["code"], "parse_error")
                self.assertIn(
                    "manual placement fields",
                    payload["errors"]["scenario"][0]["message_en"],
                )
                self.assertEqual(
                    payload["warnings"], {"topology": [], "scenario": []}
                )
                self.assertEqual(
                    payload["information"], {"topology": [], "scenario": []}
                )

    def test_run_endpoint_returns_report_dict(self):
        status, _, payload = self.json_request(
            "POST", "/api/run", payload=self.reference_payload()
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["scenario"], build_reference_scenario().name)
        self.assertGreater(payload["summary"]["task_count"], 0)
        self.assertEqual(payload["manifest"]["evidence"], "analytical")
        self.assertEqual(payload["execution_mode"], "continuous_batching")
        json.dumps(payload, sort_keys=True)
        self.assertIn("scheduler", payload)
        self.assertIn("kv_cache", payload)
        self.assertIn("batch_history", payload)
        self.assertIn("scheduler_events", payload)
        self.assertGreater(payload["summary"]["mtp"]["proposed_tokens"], 0)
        self.assertEqual(payload["summary"]["parallel"]["world_size"], 1)
        self.assertEqual(payload["kv_cache"]["tokens_per_page"], 16)
        self.assertGreater(payload["scheduler"]["total_batches"], 0)

    def test_run_endpoint_pages_visualization_without_wrapping_scenario(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/run?trace_offset=1&trace_limit=1&memory_segment_limit=8",
            payload=self.reference_payload(),
        )

        self.assertEqual(status, 200)
        visualization = payload["visualization"]
        self.assertEqual(visualization["pagination"]["offset"], 1)
        self.assertEqual(visualization["pagination"]["limit"], 1)
        self.assertEqual(visualization["pagination"]["returned"], 1)
        self.assertEqual(visualization["fidelity"], "aggregate")

    def test_run_endpoint_rejects_oversized_visualization_page(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/run?trace_limit=5001",
            payload=self.reference_payload(),
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_trace_limit")

    def test_run_estimate_and_background_job_endpoints(self):
        scenario = self.reference_payload()
        status, _, estimate = self.json_request(
            "POST", "/api/run-estimate", payload=scenario
        )

        self.assertEqual(status, 200)
        self.assertEqual(estimate["schema_version"], "4.0.0")
        self.assertGreater(estimate["estimated_cohort_count"], 0)
        self.assertEqual(
            estimate["recommended_retention_policy"], "aggregate"
        )
        self.assertTrue(estimate["risk_level_zh"])

        status, _, submitted = self.json_request(
            "POST",
            "/api/run-jobs",
            payload={
                "scenario": scenario,
                "retention_policy": "aggregate",
            },
        )
        self.assertEqual(status, 202)
        self.assertIn(submitted["status"], {"queued", "running", "completed"})
        job_id = submitted["job_id"]

        deadline = time.monotonic() + 5.0
        snapshot = submitted
        while snapshot["status"] not in {"completed", "failed", "cancelled"}:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
            status, _, snapshot = self.json_request(
                "GET", "/api/run-jobs/{}".format(job_id)
            )
            self.assertEqual(status, 200)

        self.assertEqual(snapshot["status"], "completed", snapshot.get("error"))
        self.assertIsNotNone(snapshot["report"])
        self.assertEqual(snapshot["progress"]["message"], "仿真任务已完成")

    def test_background_job_batch_trace_endpoint_pages_real_tasks(self):
        status, _, submitted = self.json_request(
            "POST",
            "/api/run-jobs",
            payload={
                "scenario": self.reference_payload(),
                "retention_policy": "aggregate",
            },
        )
        self.assertEqual(status, 202)
        job_id = submitted["job_id"]
        deadline = time.monotonic() + 5.0
        snapshot = submitted
        while snapshot["status"] not in {"completed", "failed", "cancelled"}:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
            _, _, snapshot = self.json_request(
                "GET", "/api/run-jobs/{}".format(job_id)
            )
        self.assertEqual(snapshot["status"], "completed", snapshot.get("error"))
        batch = snapshot["report"]["batch_trace_index"]["batches"][1]

        status, _, first = self.json_request(
            "GET",
            "/api/run-jobs/{}/trace?batch_id={}&offset=0&limit=2".format(
                job_id,
                batch["batch_id"],
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(first["granularity"], "task")
        self.assertEqual(first["scope"]["batch_id"], batch["batch_id"])
        self.assertEqual(first["pagination"]["returned"], 2)
        self.assertEqual(first["pagination"]["next_offset"], 2)
        self.assertAlmostEqual(
            batch["start_ns"], first["scope"]["device_start_ns"]
        )
        self.assertTrue(
            all(
                event["start_ns"]
                >= first["scope"]["start_ns"] - 1.0e-6
                for event in first["events"]
            )
        )

        status, _, second = self.json_request(
            "GET",
            "/api/run-jobs/{}/trace?batch_id={}&offset=2&limit=2".format(
                job_id,
                batch["batch_id"],
            ),
        )
        self.assertEqual(status, 200)
        self.assertNotEqual(
            first["events"][0]["event_id"],
            second["events"][0]["event_id"],
        )

        for query, code, expected_status in (
            ("", "invalid_batch_id", 400),
            ("?batch_id={}&offset=-1".format(batch["batch_id"]), "invalid_trace_offset", 400),
            ("?batch_id={}&limit=5001".format(batch["batch_id"]), "invalid_trace_limit", 400),
            ("?batch_id=missing", "run_job_batch_not_found", 404),
        ):
            with self.subTest(query=query):
                status, _, error = self.json_request(
                    "GET",
                    "/api/run-jobs/{}/trace{}".format(job_id, query),
                )
                self.assertEqual(status, expected_status)
                self.assertEqual(error["error"]["code"], code)
                self.assertTrue(error["error"]["message_zh"])

    def test_canonical_ir_endpoint_exports_schema_v1(self):
        status, _, payload = self.json_request(
            "POST", "/api/canonical-ir", payload=self.reference_payload()
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["schema_version"], "1.1")
        self.assertTrue(payload["hardware"]["nodes"])
        self.assertTrue(payload["model"]["operators"])
        self.assertTrue(payload["workload"]["requests"])
        restored = CanonicalScenario.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)

    def test_architecture_scan_endpoint_is_explicitly_analytical(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/architecture-scan",
            payload={
                "scenario": self.reference_payload(),
                "backend": "numpy",
                "top_n": 5,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["analysis_kind"], "analytical_architecture_scan")
        self.assertFalse(payload["is_event_simulation"])
        self.assertEqual(payload["backend_requested"], "numpy")
        self.assertEqual(payload["backend_used"], "numpy")
        self.assertEqual(payload["counts"]["top_results"], 5)
        self.assertEqual(len(payload["top_results"]), 5)
        self.assertTrue(payload["rank_placement"])
        self.assertTrue(all("total_ns" in row for row in payload["top_results"]))

    def test_architecture_scan_endpoint_rejects_invalid_options_in_chinese(self):
        for field, value, code in (
            ("backend", "tpu", "invalid_backend"),
            ("top_n", 0, "invalid_top_n"),
        ):
            with self.subTest(field=field):
                request = {
                    "scenario": self.reference_payload(),
                    "backend": "numpy",
                    "top_n": 5,
                }
                request[field] = value
                status, _, payload = self.json_request(
                    "POST", "/api/architecture-scan", payload=request
                )

                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], code)
                self.assertIn("必须", payload["error"]["message_zh"])

    def test_background_job_errors_are_chinese_and_safe(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/run-jobs",
            payload={
                "scenario": self.reference_payload(),
                "retention_policy": "invalid",
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            payload["error"]["code"], "invalid_retention_policy"
        )
        self.assertIn("必须", payload["error"]["message_zh"])

        for invalid_policy in ([], {}, ["exact"], {"name": "exact"}):
            with self.subTest(invalid_policy=invalid_policy):
                status, _, payload = self.json_request(
                    "POST",
                    "/api/run-jobs",
                    payload={
                        "scenario": self.reference_payload(),
                        "retention_policy": invalid_policy,
                    },
                )
                self.assertEqual(status, 400)
                self.assertEqual(
                    payload["error"]["code"],
                    "invalid_retention_policy",
                )
                self.assertIn("必须", payload["error"]["message_zh"])

        status, _, payload = self.json_request(
            "GET", "/api/run-jobs/not-a-real-job"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "run_job_not_found")
        self.assertEqual(payload["error"]["message_zh"], "未找到指定的仿真任务")

    def test_background_job_accepts_only_v4_retention_policies(self):
        policies = ("exact", "streaming", "aggregate")
        scenario = self.reference_payload()
        scenario["workload"]["scheduler"]["mode"] = "static"
        with patch("heterollm_sim.run_jobs.run_scenario", return_value={}), patch(
            "heterollm_sim.run_jobs.report_dict", return_value={"ok": True}
        ):
            for retention_policy in policies:
                with self.subTest(retention_policy=retention_policy):
                    status, _, submitted = self.json_request(
                        "POST",
                        "/api/run-jobs",
                        payload={
                            "scenario": scenario,
                            "retention_policy": retention_policy,
                        },
                    )
                    self.assertEqual(status, 202)
                    deadline = time.monotonic() + 1.0
                    snapshot = submitted
                    while snapshot["status"] not in {"completed", "failed", "cancelled"}:
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.005)
                        _, _, snapshot = self.json_request(
                            "GET", "/api/run-jobs/{}".format(submitted["job_id"])
                        )
                    self.assertEqual(snapshot["status"], "completed")

        for retention_policy in ("exact", "streaming"):
            with self.subTest(continuous_retention_policy=retention_policy):
                status, _, payload = self.json_request(
                    "POST",
                    "/api/run-jobs",
                    payload={
                        "scenario": self.reference_payload(),
                        "retention_policy": retention_policy,
                    },
                )
                self.assertEqual(status, 400)
                self.assertEqual(
                    payload["error"]["code"],
                    "invalid_retention_policy",
                )

    def test_background_job_can_be_cancelled_through_api(self):
        started = threading.Event()

        def cancellable_run(_scenario, *, retention_policy, control):
            self.assertEqual(retention_policy, "aggregate")
            started.set()
            index = 0
            while True:
                control.raise_if_cancelled()
                control.report("serving_cohorts", index, 100, message="正在推进在线批次")
                index += 1
                time.sleep(0.002)

        with patch("heterollm_sim.run_jobs.run_scenario", side_effect=cancellable_run):
            status, _, submitted = self.json_request(
                "POST",
                "/api/run-jobs",
                payload={
                    "scenario": self.reference_payload(),
                    "retention_policy": "aggregate",
                },
            )
            self.assertEqual(status, 202)
            self.assertTrue(started.wait(1.0))
            job_id = submitted["job_id"]

            status, _, cancelling = self.json_request(
                "POST", "/api/run-jobs/{}/cancel".format(job_id), payload={}
            )
            self.assertEqual(status, 202)
            self.assertTrue(cancelling["cancellation_requested"])

            deadline = time.monotonic() + 2.0
            snapshot = cancelling
            while snapshot["status"] != "cancelled":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
                status, _, snapshot = self.json_request(
                    "GET", "/api/run-jobs/{}".format(job_id)
                )
                self.assertEqual(status, 200)

        self.assertIsNone(snapshot["report"])
        self.assertEqual(snapshot["progress"]["message"], "仿真任务已取消")

    def test_compare_endpoint_returns_gpu_baseline_comparison(self):
        status, _, payload = self.json_request(
            "POST", "/api/compare", payload=self.reference_payload()
        )

        self.assertEqual(status, 200)
        self.assertIn("candidate", payload)
        self.assertIn("gpu_baseline", payload)
        self.assertGreater(payload["comparison"]["latency_speedup"], 1.0)

    def test_compare_endpoint_handles_zero_visible_tokens(self):
        scenario = self.reference_payload()
        scenario["workload"]["requests"][0]["output_tokens"] = 0

        status, _, payload = self.json_request(
            "POST", "/api/compare", payload=scenario
        )

        self.assertEqual(status, 200)
        self.assertIsNone(payload["comparison"]["throughput_speedup"])

    def test_auto_map_get_and_post_are_unknown_endpoints(self):
        for method, payload in (
            ("GET", None),
            ("POST", {"scenario": self.reference_payload()}),
        ):
            with self.subTest(method=method):
                status, _, response = self.json_request(
                    method, "/api/auto-map", payload=payload
                )
                self.assertEqual(status, 404)
                self.assertEqual(response["error"]["code"], "not_found")
                self.assertEqual(
                    response["error"]["message_en"], "unknown API endpoint"
                )

    def test_validate_rejects_precision_that_lowering_cannot_run(self):
        scenario = self.reference_payload()
        # Graph is authoritative in V4. Edit its executable component parameter.
        group = next(
            item
            for item in scenario["model"]["graph"]["operators"]
            if item["op_kind"] == "layer_group"
        )
        group["parameters"]["dtype"] = "mystery"
        parent_id = group["operator_id"]
        feed_forward = next(
            item
            for item in scenario["model"]["graph"]["operators"]
            if item.get("attributes", {}).get("parent_group_id") == parent_id
            and item["op_kind"] in {"dense_mlp", "moe_router", "moe_experts"}
        )
        feed_forward["parameters"]["quantization"] = None

        status, _, payload = self.json_request(
            "POST", "/api/validate", payload=scenario
        )

        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertTrue(
            any(
                "graph.operators[1].ports[1].dtype" in item["message"]
                and "已拒绝静默降级" in item["message"]
                for item in payload["errors"]["scenario"]
            )
        )

    def test_bad_json_non_object_and_unknown_api_return_json_errors(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/validate",
            body=b"{",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "bad_json")
        self.assertEqual(payload["error"]["message"], "请求体必须是有效的 JSON")
        self.assertEqual(payload["error"]["message_zh"], payload["error"]["message"])
        self.assertEqual(payload["error"]["message_en"], "request body must be valid JSON")

        status, _, payload = self.json_request(
            "POST", "/api/validate", body=b"[]", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "bad_json_object")
        self.assertEqual(payload["error"]["message"], "JSON 顶层值必须是对象")

        status, _, payload = self.json_request(
            "POST",
            "/api/canonical-ir",
            body=b'{"metadata": NaN}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "bad_json")
        self.assertEqual(payload["error"]["message_zh"], "请求体必须是有效的 JSON")

        status, _, payload = self.json_request("GET", "/api/missing")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_post_requires_json_and_rejects_cross_origin_browser_requests(self):
        status, _, payload = self.json_request(
            "POST",
            "/api/validate",
            body=b"{}",
            headers={"Content-Type": "text/plain"},
        )
        self.assertEqual(status, 415)
        self.assertEqual(payload["error"]["code"], "unsupported_media_type")

        status, _, payload = self.json_request(
            "POST",
            "/api/validate",
            payload=self.reference_payload(),
            headers={"Origin": "https://example.invalid"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "cross_origin_forbidden")

    def test_static_homepage_is_served_without_caching(self):
        status, headers, body = self.request("GET", "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        content_security_policy = headers["Content-Security-Policy"]
        self.assertIn("default-src 'self'", content_security_policy)
        script_directive = next(
            directive.strip()
            for directive in content_security_policy.split(";")
            if directive.strip().startswith("script-src")
        )
        self.assertEqual(script_directive, "script-src 'self'")
        directives = {
            directive.strip()
            for directive in content_security_policy.split(";")
            if directive.strip()
        }
        self.assertIn("style-src 'self' 'unsafe-inline'", directives)
        self.assertIn("object-src 'none'", directives)
        self.assertIn("base-uri 'none'", directives)
        self.assertIn("frame-ancestors 'none'", directives)
        self.assertGreater(len(body), 0)
        self.assertIn(
            b'id="fontScaleInput" min="80" max="200" step="1"',
            body,
        )
        self.assertIn(
            b'id="fontScaleNumberInput" min="80" max="200" step="1"',
            body,
        )
        self.assertIn(b'id="fontScaleValue"', body)
        self.assertIn(b'id="controlPlaneStatus"', body)
        self.assertIn(b'id="controlPlaneStatusBadge"', body)
        self.assertIn(b'id="controlPlaneStatusSummary"', body)
        self.assertIn(b'id="controlPlaneStatusMetrics"', body)
        self.assertNotIn(b'id="mappingModeInput"', body)
        self.assertNotIn(b'id="mappingSolverInput"', body)
        self.assertNotIn(b'/api/auto-map', body)
        self.assertIn(b'id="canonicalExportButton"', body)
        self.assertIn(b'id="canonicalExportDialogButton"', body)
        self.assertIn(b'id="runJobDialog"', body)
        self.assertIn(b'id="runEstimateSummary"', body)
        self.assertIn(b'id="runProgressBar"', body)
        self.assertIn(b'id="cancelRunJobButton"', body)
        self.assertNotIn("锁定（Lock）".encode("utf-8"), body)
        preload_tag = b'<script src="./ui-settings-preload.js"></script>'
        stylesheet_tag = b'<link rel="stylesheet" href="./styles.css">'
        self.assertIn(preload_tag, body)
        self.assertIn(stylesheet_tag, body)
        self.assertLess(body.index(preload_tag), body.index(stylesheet_tag))
        self.assertIn(b'<script src="./app.js" defer></script>', body)
        self.assertNotRegex(body.decode("utf-8"), r"<script(?![^>]*\bsrc=)")
        self.assertNotIn(b"Apply local-only appearance", body)

        status, headers, body = self.request("GET", "/ui-settings-preload.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn(b"fontScale: 120", body)
        self.assertIn(b"Math.min(200", body)
        self.assertIn(b"heterollm-lab:ui-settings:v3", body)
        self.assertNotIn(b"fontSize", body)

        status, headers, body = self.request("GET", "/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn(b"API_ROOT", body)
        self.assertIn(b"MAX_FONT_SCALE = 200", body)
        self.assertIn(b"heterollm-lab:scenario:v4", body)
        self.assertIn(b"heterollm-lab:ui-settings:v3", body)
        self.assertNotIn(b"LEGACY_FONT_SCALES", body)
        self.assertIn(b"layoutScaleForFont", body)

        status, headers, body = self.request("GET", "/styles.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers["Content-Type"])
        self.assertIn(b"--copper", body)
        self.assertIn(b"--font-scale", body)
        self.assertIn(b'data-font-band="extreme"', body)
        self.assertNotIn(b"data-font-size", body)


if __name__ == "__main__":
    unittest.main()
