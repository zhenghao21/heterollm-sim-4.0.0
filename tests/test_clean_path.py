from unittest.mock import patch

from heterollm_sim import serving
from heterollm_sim.reference import build_reference_scenario


def test_frontend_scenario_uses_topology_provider_and_unified_kernel():
    scenario = build_reference_scenario()
    original = serving.TopologyAwareBatchCostProvider
    created = []

    class SpyProvider(original):
        def __init__(self, bound_scenario):
            created.append(bound_scenario)
            super().__init__(bound_scenario)

    with patch.object(serving, "TopologyAwareBatchCostProvider", SpyProvider):
        result = serving.simulate_online(scenario)

    assert created == [scenario]
    assert result.makespan_ns > 0
    assert result.runtime_kernel_metrics["completed_count"] > 0
