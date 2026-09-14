"""Runtime capability and diagnostics log tests."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from heterollm_sim import __version__
from heterollm_sim.cli import main
from heterollm_sim.runtime_diagnostics import (
    format_doctor_report,
    health_payload,
    ortools_diagnostics,
    probe_cp_sat,
    record_unexpected_exception,
)


class _FakeVariable:
    def __eq__(self, other):
        return ("equals", other)


class _FakeModel:
    def NewBoolVar(self, name):
        return _FakeVariable()

    def Add(self, constraint):
        self.constraint = constraint


class _FakeParameters:
    num_search_workers = None
    random_seed = None


class _FakeSolver:
    def __init__(self):
        self.parameters = _FakeParameters()

    def Solve(self, model):
        return _FakeCpModel.OPTIMAL

    def Value(self, variable):
        return 1


class _FakeCpModel:
    OPTIMAL = 4
    CpModel = _FakeModel
    CpSolver = _FakeSolver


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_health_contract_is_complete_and_solver_list_matches_probe(self):
        payload = health_payload(__version__)

        self.assertEqual(payload["version"], __version__)
        self.assertEqual(
            set(payload["runtime"]),
            {"python_version", "executable", "architecture_bits", "platform"},
        )
        self.assertEqual(
            set(payload["ortools"]),
            {
                "available",
                "version",
                "cp_sat_available",
                "error",
                "error_type",
                "probe_ok",
            },
        )
        self.assertEqual(payload["ortools"]["available"], payload["ortools"]["probe_ok"])
        self.assertEqual(
            "ortools" in payload["available_solvers"],
            payload["ortools"]["available"],
        )

    def test_minimal_cp_sat_probe_checks_an_actual_solution_value(self):
        ok, error, error_type = probe_cp_sat(_FakeCpModel)

        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertIsNone(error_type)

    def test_cp_sat_import_failure_has_complete_chinese_context(self):
        import_error = ImportError("No module named 'ortools.sat.python.cp_model'")
        with patch(
            "heterollm_sim.runtime_diagnostics.distribution_version",
            return_value="9.test",
        ), patch(
            "heterollm_sim.runtime_diagnostics.import_module",
            side_effect=import_error,
        ):
            payload = ortools_diagnostics()

        self.assertFalse(payload["available"])
        self.assertFalse(payload["cp_sat_available"])
        self.assertFalse(payload["probe_ok"])
        self.assertEqual(payload["version"], "9.test")
        self.assertEqual(payload["error_type"], "ImportError")
        self.assertIn("导入 OR-Tools CP-SAT 模块失败", payload["error"])
        self.assertIn(str(import_error), payload["error"])

    def test_unexpected_exception_log_preserves_full_traceback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory).joinpath("diagnostics.log")
            terminal = StringIO()
            try:
                raise LookupError("完整异常消息")
            except LookupError as exc:
                with redirect_stderr(terminal):
                    diagnostic_id = record_unexpected_exception(
                        exc, context="unit-test", log_path=log_path
                    )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn(diagnostic_id, log_text)
            self.assertIn("Traceback", log_text)
            self.assertIn("LookupError: 完整异常消息", log_text)
            self.assertIn("LookupError: 完整异常消息", terminal.getvalue())

    def test_doctor_report_is_chinese_and_actionable(self):
        payload = {
            "version": "1.2.3",
            "runtime": {
                "python_version": "3.test",
                "executable": "python-test",
                "architecture_bits": 64,
                "platform": "test-platform",
            },
            "ortools": {
                "available": False,
                "version": None,
                "cp_sat_available": False,
                "probe_ok": False,
                "error": "导入失败（ImportError）：完整原因",
            },
            "available_solvers": ["auto", "builtin"],
        }

        report = format_doctor_report(payload)

        self.assertIn("运行环境诊断", report)
        self.assertIn("CP-SAT 最小求解：失败", report)
        self.assertIn(payload["ortools"]["error"], report)
        self.assertIn("诊断结果：未通过", report)

    def test_cli_doctor_json_uses_health_contract_and_failure_exit_status(self):
        payload = health_payload(__version__)
        payload["ortools"] = dict(payload["ortools"], available=False, probe_ok=False)
        payload["available_solvers"] = ["auto", "builtin"]
        output = StringIO()

        with patch("heterollm_sim.cli.health_payload", return_value=payload), redirect_stdout(output):
            status = main(["doctor", "--json"])

        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()
