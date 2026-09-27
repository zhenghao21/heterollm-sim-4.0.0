import http.client
import json
import threading
import tempfile
import unittest

from heterollm_sim.web import build_server


class ComponentPresetCrudApiTests(unittest.TestCase):
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
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def test_create_update_delete_component_preset(self):
        preset_id = "crud-test-component"
        payload = {
            "id": preset_id,
            "name": "CRUD test component",
            "family": "Test vendor",
            "component": {
                "component_id": "crud0",
                "kind": "hbm",
                "capacity_bytes": 1024,
                "read_bandwidth_gbps": 100,
                "write_bandwidth_gbps": 80,
                "ports": [],
            },
            "sources": [{
                "title": "Test source",
                "url": "https://example.com/spec",
                "publisher": "Test",
                "evidence_level": "S2_VENDOR_DECLARED",
            }],
        }
        try:
            status, detail = self.request("POST", "/api/component-presets", payload)
            self.assertEqual(status, 201)
            self.assertEqual(detail["preset"]["id"], preset_id)

            payload["name"] = "Updated component"
            status, detail = self.request("PUT", "/api/component-presets/{}".format(preset_id), payload)
            self.assertEqual(status, 200)
            self.assertEqual(detail["preset"]["name"], "Updated component")

            status, page = self.request("GET", "/api/component-presets?query=Updated")
            self.assertEqual(status, 200)
            self.assertEqual([item["id"] for item in page["items"]], [preset_id])

            status, deleted = self.request("DELETE", "/api/component-presets/{}".format(preset_id))
            self.assertEqual(status, 200)
            self.assertEqual(deleted["deleted"], preset_id)

            status, payload = self.request("GET", "/api/component-presets/{}".format(preset_id))
            self.assertEqual(status, 404)
            self.assertEqual(payload["error"]["code"], "not_found")
        finally:
            # Tests are safe to rerun if a failure occurs before DELETE.
            self.request("DELETE", "/api/component-presets/{}".format(preset_id))

    def test_invalid_create_is_client_error(self):
        status, payload = self.request("POST", "/api/component-presets", {})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_component")

    def test_gpu_preset_drops_external_memory_fields(self):
        preset_id = "crud-gpu-component"
        payload = {
            "id": preset_id,
            "name": "GPU compute only",
            "component": {
                "component_id": "gpu0",
                "kind": "gpu",
                "capacity_bytes": 180_000_000_000,
                "peak_ops_per_s": 1.0e15,
                "read_bandwidth_gbps": 64_000,
                "write_bandwidth_gbps": 64_000,
                "ports": [],
            },
        }
        try:
            status, detail = self.request("POST", "/api/component-presets", payload)
            self.assertEqual(status, 201)
            component = detail["component"]
            self.assertEqual(component["capacity_bytes"], 0)
            self.assertEqual(component["read_bandwidth_gbps"], 0.0)
            self.assertEqual(component["write_bandwidth_gbps"], 0.0)
            self.assertEqual(component["peak_ops_per_s"], 1.0e15)
        finally:
            self.request("DELETE", "/api/component-presets/{}".format(preset_id))

    def test_catalog_persists_across_server_instances(self):
        payload = {
            "id": "crud-persisted-component",
            "name": "Persistent component",
            "component": {"component_id": "persist0", "kind": "ssd", "ports": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            server = build_server("127.0.0.1", 0, component_preset_cache_dir=directory)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                body = json.dumps(payload).encode("utf-8")
                connection.request("POST", "/api/component-presets", body, {"Content-Type": "application/json"})
                self.assertEqual(connection.getresponse().status, 201)
                connection.close()
            finally:
                server.shutdown(); thread.join(timeout=5); server.server_close()

            server = build_server("127.0.0.1", 0, component_preset_cache_dir=directory)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/api/component-presets/crud-persisted-component")
                response = connection.getresponse()
                detail = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200)
                self.assertEqual(detail["preset"]["name"], "Persistent component")
                connection.close()
            finally:
                server.shutdown(); thread.join(timeout=5); server.server_close()


if __name__ == "__main__":
    unittest.main()
