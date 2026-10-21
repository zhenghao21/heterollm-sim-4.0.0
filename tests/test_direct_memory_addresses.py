"""Direct backing uses one phase address and one component default."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.cost_models import CostPhase
from heterollm_sim.data_motion import endpoint_service, deferred_endpoint_construction
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbf_config
from heterollm_sim.planner import _direct_memory_phase


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("component_offset,phase_metadata,overrides,expected", [
    (32768, {}, {}, 32768),
    (32768, {"page_offset_bytes": 16384}, {}, 16384),
    (32768, {"page_offset_bytes": 16384}, {"page_offset_bytes": 49152}, 49152),
    (None, {"layer_id": "layer1"}, {}, 16384),
    (None, {"layer_id": "layer1", "dram_address_bytes": 0}, {}, 16384),
])
def test_direct_backing_preserves_current_address_precedence(
        component_offset, phase_metadata, overrides, expected, defer):
    config = make_hbf_config(capacity_bytes=65536)
    component = ComponentSpec("flash", "hbf", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config,
                  "memory_access_offset_bytes": component_offset,
                  "page_offset_bytes": 0})
    rank = SimpleNamespace(rank=0, memory_component_id="hbm", component_id="gpu")
    local = SimpleNamespace(resource_id="hbm", effective_bandwidth_gb_s=1.,
                            memory_service=lambda *args, **kwargs: {"service_ns": 0.})
    phase = CostPhase("read", TaskCategory.MEMORY,
        (ResourceDemand("hbm", 64., bytes_moved=64),),
        {"kernel_model": {"read_bytes": 64, "write_bytes": 0}, **phase_metadata})
    with patch("heterollm_sim.planner._direct_device_memory", return_value=True), \
         patch("heterollm_sim.planner._gpu_profiles", return_value=(None, local)), \
         patch("heterollm_sim.planner._component", return_value=component), \
         patch("heterollm_sim.planner._resolve_component_profile", return_value=None), \
         patch("heterollm_sim.planner._execution_layers", return_value=("layer1",)), \
         patch("heterollm_sim.planner._topology_router", return_value=SimpleNamespace(route=lambda *args: ())), \
         deferred_endpoint_construction(defer):
        result = _direct_memory_phase(None, rank, phase, "flash", read_bytes=64, metadata=overrides)
    assert result.metadata["page_offset_bytes"] == expected
    assert result.metadata["memory_accesses"][0]["address"] == expected
    bill = result.metadata["direct_memory_access"]["read_service"]
    assert bill["physical_bytes"] == (0 if defer else config.page_bytes)
    if defer:
        assert bill["physical_cost_deferred"] is True
        assert bill["timing_model"] == "native_dispatch_pending"


def test_endpoint_requires_its_current_address_parameter():
    config = make_hbf_config(capacity_bytes=65536)
    component = ComponentSpec("flash", "hbf", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config, "memory_access_offset_bytes": 16384})
    assert endpoint_service(component, 64, operation="read", name="read").metadata["address"] == 16384
    with pytest.raises(TypeError, match="dram_address_bytes"):
        endpoint_service(component, 64, operation="read", name="read", dram_address_bytes=0)
