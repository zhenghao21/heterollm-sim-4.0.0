from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from heterollm_sim.cli import main
from heterollm_sim.reference import build_reference_scenario


class CliReportReuseTests(unittest.TestCase):
    def test_text_report_reuses_payload_when_output_is_requested(self):
        scenario = build_reference_scenario()
        result = object()
        payload = {"summary": {"makespan_ns": 1}}
        output = StringIO()
        errors = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with patch("heterollm_sim.cli.load_scenario", return_value=scenario), \
                patch("heterollm_sim.cli.run_scenario", return_value=result), \
                patch("heterollm_sim.cli.report_dict", return_value=payload) as report, \
                patch("heterollm_sim.cli.format_report", return_value="formatted") as formatter, \
                redirect_stdout(output), redirect_stderr(errors):
                status = main(["run", "scenario.json", "--output", str(target)])

        self.assertEqual(status, 0, errors.getvalue())
        report.assert_called_once_with(result)
        formatter.assert_called_once_with(result, data=payload)
        self.assertEqual(output.getvalue().strip(), "formatted")


if __name__ == "__main__":
    unittest.main()
