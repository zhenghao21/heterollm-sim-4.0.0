import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from heterollm_sim.cli import main
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import CanonicalScenario


class CliCanonicalExportTests(unittest.TestCase):
    def test_export_ir_writes_canonical_schema_v1(self):
        scenario = build_reference_scenario()
        output = StringIO()
        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "canonical.json"
            with patch("heterollm_sim.cli.load_scenario", return_value=scenario), redirect_stdout(output), redirect_stderr(errors):
                status = main(["export-ir", "scenario.json", "--output", str(target)])

            self.assertEqual(status, 0, errors.getvalue())
            payload = json.loads(target.read_text(encoding="utf-8"))
            restored = CanonicalScenario.from_dict(payload)
            self.assertEqual(restored.schema_version, "1.1")
            self.assertIn("Canonical IR 已写入", errors.getvalue())

    def test_export_ir_stdout_is_stable_json(self):
        scenario = build_reference_scenario()
        output = StringIO()
        with patch("heterollm_sim.cli.load_scenario", return_value=scenario), redirect_stdout(output):
            status = main(["export-ir", "scenario.json"])

        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema_version"], "1.1")


if __name__ == "__main__":
    unittest.main()
