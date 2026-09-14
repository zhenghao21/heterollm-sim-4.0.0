import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from heterollm_sim.cli import main
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.topology import validate_topology


class ArchitecturePresetCliTests(unittest.TestCase):
    def test_list_and_detail_emit_stable_json(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "architecture-presets",
                    "list",
                    "--vendor",
                    "NVIDIA",
                    "--protocol",
                    "NVLink",
                    "--loadable",
                    "true",
                ]
            )

        self.assertEqual(status, 0)
        page = json.loads(output.getvalue())
        self.assertGreater(page["total"], 0)
        self.assertEqual(page["replacement_policy"]["mode"], "replace_hardware")
        self.assertTrue(all(item["vendor"] == "NVIDIA" for item in page["items"]))
        self.assertTrue(all("NVLink" in item["protocols"] for item in page["items"]))

        output = StringIO()
        with redirect_stdout(output):
            status = main(
                ["architecture-presets", "detail", "nvidia-gh200-superchip"]
            )

        self.assertEqual(status, 0)
        detail = json.loads(output.getvalue())
        self.assertEqual(detail["preset"]["id"], "nvidia-gh200-superchip")
        self.assertTrue(detail["preset"]["hardware_only"])
        self.assertNotIn("model", detail["hardware"])
        self.assertNotIn("workload", detail["hardware"])
        self.assertNotIn("placement", detail["hardware"])

    def test_export_hardware_writes_loadable_topology_only_json(self):
        output = StringIO()
        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "hardware.json"
            with redirect_stdout(output), redirect_stderr(errors):
                status = main(
                    [
                        "architecture-presets",
                        "export-hardware",
                        "gpu-hbm-cim",
                        "--output",
                        str(target),
                    ]
                )

            self.assertEqual(status, 0, errors.getvalue())
            self.assertEqual(output.getvalue(), "")
            self.assertIn("架构硬件 JSON 已写入", errors.getvalue())
            payload = json.loads(target.read_text(encoding="utf-8"))

        hardware = hardware_from_dict(payload)
        report = validate_topology(hardware)
        self.assertTrue(report.is_valid, report.format())
        self.assertEqual(payload["metadata"]["architecture_preset"]["id"], "gpu-hbm-cim")
        self.assertNotIn("model", payload)
        self.assertNotIn("workload", payload)
        self.assertNotIn("placement", payload)


if __name__ == "__main__":
    unittest.main()
