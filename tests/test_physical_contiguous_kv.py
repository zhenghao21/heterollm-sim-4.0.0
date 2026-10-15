"""Admission must agree with physical contiguous KV buffer reservations."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import BatchCost, RequestStatus, _OnlineRuntime, compile_serving_plan


class PhysicalContiguousKVTests(unittest.TestCase):
    @staticmethod
    def _lazy_scenario():
        config = scenario(hbf=True, kv_budget=65536, count=4, slots=4)
        return replace(config, placement=replace(config.placement,
            kv_policy=replace(config.placement.kv_policy, allocation_policy="lazy")))

    def test_small_physical_tier_queues_requests_before_allocating_full_buffers(self):
        result = run_scenario(self._lazy_scenario()).serving
        self.assertTrue(all(item.status is RequestStatus.FINISHED
                            for item in result.request_metrics.values()))
        self.assertEqual(result.kv_metrics.requested_allocation_policy, "lazy")
        self.assertEqual(result.kv_metrics.effective_allocation_policy, "eager")
        self.assertIsNotNone(result.kv_metrics.allocation_policy_reason)
        self.assertLessEqual(result.kv_metrics.peak_used_bytes, 65536)

    def test_custom_lowerer_keeps_explicit_lazy_semantics(self):
        runtime = _OnlineRuntime(compile_serving_plan(self._lazy_scenario()),
                                 lambda *_: BatchCost(1.0))
        self.assertEqual(runtime.plan.kv_policy.allocation_policy, "lazy")
        self.assertIsNone(runtime._kv_allocation_policy_reason)


if __name__ == "__main__":
    unittest.main()
