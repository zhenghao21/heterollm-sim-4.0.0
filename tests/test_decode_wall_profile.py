"""The diagnostic wall categories must close without nested double counting."""
import importlib.util
from pathlib import Path
import unittest


_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_decode_wall.py"
_spec = importlib.util.spec_from_file_location("profile_decode_wall", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
WallCollector = _module.WallCollector


class DecodeWallProfileTests(unittest.TestCase):
    def test_nested_exclusive_intervals_close_exactly(self):
        ticks = iter((0, 10, 30, 50))
        collector = WallCollector(lambda: next(ticks))
        collector.enabled = True
        collector.interval_start_ns = 0
        inner = collector.wrap(lambda: None, "inner", "numeric")
        collector.wrap(inner, "outer", "adapter")()
        collector.complete_cohort(60, 1, "decode", 100.0)
        row = collector.result()["cohorts"][0]
        self.assertEqual({r["category"]: r["exclusive_ns"] for r in row["categories"]},
                         {"adapter": 30, "numeric": 20, "unattributed": 10})
        self.assertEqual(row["closure_error_ns"], 0)
        self.assertEqual(row["wall_ns"], 60)

    def test_exception_unwinds_timer_and_restores_staticmethod(self):
        ticks = iter((1, 7))
        collector = WallCollector(lambda: next(ticks))
        collector.enabled = True

        class Subject:
            @staticmethod
            def fail(message):
                raise ValueError(message)

        original = Subject.__dict__["fail"]
        collector.patch(Subject, "fail", "prepare")
        with self.assertRaisesRegex(ValueError, "expected"):
            Subject().fail("expected")
        self.assertEqual(collector.stack, [])
        collector.restore()
        self.assertIs(Subject.__dict__["fail"], original)

    def test_windows_reset_per_cohort(self):
        ticks = iter((2, 7, 15, 19))
        collector = WallCollector(lambda: next(ticks))
        collector.enabled = True
        collector.complete_cohort(0, 1, "prefill", 1.0)
        work = collector.wrap(lambda: 42, "work", "binding")
        self.assertEqual(work(), 42)
        collector.complete_cohort(10, 2, "decode", 2.0)
        work()
        collector.complete_cohort(20, 3, "decode", 3.0)
        result = collector.result()
        self.assertEqual(result["windows"]["first"]["categories"][0]["exclusive_ns"], 5)
        self.assertEqual(result["windows"]["steady"]["functions"][0]["exclusive_ns"], 4)
        self.assertEqual(result["windows"]["all"]["closure_error_ns"], 0)

    def test_preview_nested_kernel_cost_is_not_labelled_live(self):
        ticks = iter((0, 10, 30, 50))
        collector = WallCollector(lambda: next(ticks))
        collector.enabled = True
        collector.interval_start_ns = 0
        inner = collector.wrap(lambda: None, "kernel", "live_dispatch")
        collector.wrap(inner, "preview", "preview_schedule")()
        collector.complete_cohort(60, 1, "decode", 100.0)
        row = collector.result()["cohorts"][0]
        self.assertEqual({r["category"]: r["exclusive_ns"] for r in row["categories"]},
                         {"preview_schedule": 50, "unattributed": 10})


if __name__ == "__main__":
    unittest.main()
