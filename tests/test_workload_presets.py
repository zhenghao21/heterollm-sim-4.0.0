import http.client
import json
import threading
import unittest

from heterollm_sim.web import build_server


class WorkloadPresetApiTests(unittest.TestCase):
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

    def request(self, method, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
        finally:
            connection.close()

    def test_list_query_detail_and_wrong_method(self):
        status, page = self.request("GET", "/api/workload-presets")
        self.assertEqual(status, 200)
        self.assertEqual(page["catalog"]["kind"], "request_shape")
        self.assertGreaterEqual(page["total"], 10)
        ids = {item["id"] for item in page["items"]}
        self.assertIn("llama_cpp_default", ids)
        self.assertIn("software_development", ids)

        status, filtered = self.request("GET", "/api/workload-presets?query=C06")
        self.assertEqual(status, 200)
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["id"], "software_development")

        status, detail = self.request(
            "GET", "/api/workload-presets/software_development"
        )
        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["id"], "software_development")
        self.assertEqual(detail["workload"]["prompt_tokens"], 24576)
        self.assertEqual(detail["workload"]["output_tokens"], 4096)
        self.assertEqual(len(detail["workload"]["requests"]), 16)
        self.assertFalse(detail["workload"]["scheduler"]["preemption_enabled"])

        status, missing = self.request("GET", "/api/workload-presets/missing")
        self.assertEqual(status, 404)
        self.assertEqual(missing["error"]["code"], "not_found")

        status, wrong_method = self.request("POST", "/api/workload-presets")
        self.assertEqual(status, 405)
        self.assertEqual(wrong_method["error"]["code"], "method_not_allowed")


if __name__ == "__main__":
    unittest.main()
