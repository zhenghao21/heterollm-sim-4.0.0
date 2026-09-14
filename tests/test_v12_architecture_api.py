import http.client
import json
import threading
import unittest

from heterollm_sim.architecture_presets import list_architecture_presets
from heterollm_sim.web import build_server


class ArchitecturePresetApiTests(unittest.TestCase):
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

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            body = None
            headers = {}
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def test_list_filters_and_detail(self):
        expected_ids = {item["id"] for item in list_architecture_presets()}
        status, page = self.request("GET", "/api/architecture-presets")
        self.assertEqual(status, 200)
        self.assertEqual({item["id"] for item in page["items"]}, expected_ids)
        self.assertEqual(page["replacement_policy"]["mode"], "replace_hardware")

        status, filtered = self.request(
            "GET",
            "/api/architecture-presets?vendor=NVIDIA&protocol=NVLink&loadable=true",
        )
        self.assertEqual(status, 200)
        self.assertGreater(filtered["total"], 0)
        self.assertTrue(all(item["vendor"] == "NVIDIA" for item in filtered["items"]))
        self.assertTrue(all("NVLink" in item["protocols"] for item in filtered["items"]))
        self.assertTrue(all(item["loadable"] for item in filtered["items"]))

        status, detail = self.request(
            "GET", "/api/architecture-presets/gpu-hbm-cim"
        )
        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["id"], "gpu-hbm-cim")
        self.assertTrue(detail["hardware"]["components"])
        self.assertTrue(detail["hardware"]["links"])
        self.assertEqual(detail["replacement_policy"]["preserve"], ["model", "workload"])
        self.assertEqual(
            detail["preset"]["topology_evidence"],
            detail["hardware"]["metadata"]["topology_evidence"],
        )
        self.assertEqual(
            detail["preset"]["parameter_basis"],
            detail["hardware"]["metadata"]["parameter_basis"],
        )
        self.assertFalse(detail["compatibility"]["planner_executable"])
        self.assertFalse(detail["compatibility"]["requires_gpu_attachment"])
        self.assertTrue(detail["compatibility"]["requires_cpu_attachment"])

        status, gh200 = self.request(
            "GET", "/api/architecture-presets/nvidia-gh200-superchip"
        )
        self.assertEqual(status, 200)
        self.assertTrue(gh200["compatibility"]["planner_executable"])
        self.assertFalse(gh200["compatibility"]["requires_gpu_attachment"])
        self.assertFalse(gh200["compatibility"]["requires_cpu_attachment"])

        status, cxl = self.request(
            "GET", "/api/architecture-presets/cxl-type3-memory-expander"
        )
        self.assertEqual(status, 200)
        self.assertFalse(cxl["compatibility"]["planner_executable"])
        self.assertTrue(cxl["compatibility"]["requires_gpu_attachment"])
        self.assertFalse(cxl["compatibility"]["requires_cpu_attachment"])

    def test_invalid_filter_unknown_and_wrong_method_are_chinese(self):
        status, payload = self.request(
            "GET", "/api/architecture-presets?loadable=maybe"
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_loadable_filter")
        self.assertIn("必须", payload["error"]["message"])

        status, payload = self.request(
            "GET", "/api/architecture-presets/does-not-exist"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["message"], "未找到指定的架构拓扑预设")

        status, payload = self.request("POST", "/api/architecture-presets", {})
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["message"], "此端点要求使用 GET 方法")


if __name__ == "__main__":
    unittest.main()
