"""Memory endpoints expose one explicit operation to physical execution."""
import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import endpoint_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbf_config
from test_current_topology_inputs import PRESETS, _hardware


def nand_component():
    config = make_hbf_config(capacity_bytes=65536, write_buffer_bytes=0,
                             page_read_ns=100., page_program_ns=200., block_erase_ns=300.)
    return ComponentSpec("flash", "hbf", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config, "memory_access_offset_bytes": 0})


@pytest.mark.parametrize("operation,counter", [
    ("read", "pages_read"), ("write", "pages_programmed"), ("erase", "erase_operations"),
])
def test_endpoint_operation_reaches_the_matching_nand_action(operation, counter):
    endpoint = endpoint_service(nand_component(), 16384, operation=operation, name="probe")
    access, = endpoint.metadata["memory_accesses"]
    assert access["operation"] == operation
    physical, = endpoint.metadata["physical_execution_by_owner"].values()
    assert physical["operation"] == operation
    assert physical[counter] == 1
    assert physical["service_ns"] > 0
    if operation == "erase":
        assert endpoint.demands[0].bytes_moved == 0
        assert physical["physical_bytes"] == 0
    else:
        assert physical["physical_" + operation + "_bytes"] == 16384


@pytest.mark.parametrize("arguments", [{"read": True}, {"operation": "read", "read": True}])
def test_endpoint_rejects_the_retired_read_boolean(arguments):
    with pytest.raises(TypeError, match="read"):
        endpoint_service(nand_component(), 16384, name="old", **arguments)


def test_endpoint_requires_an_explicit_operation():
    with pytest.raises(TypeError, match="operation"):
        endpoint_service(nand_component(), 16384, name="missing")


@pytest.mark.parametrize("source,target,operation,address", [
    ("memory", "gpu", "read", 0), ("gpu", "memory", "write", 256),
])
def test_router_preserves_source_read_target_write_and_separate_addresses(source, target, operation, address):
    hardware = _hardware(*PRESETS[0][:2])
    phases = TopologyRouter(hardware).transfer_phases(source, target, 256,
        source_page_offset_bytes=0, target_page_offset_bytes=256,
        coherent_dma_mode="strict_serialized")
    accesses = [access for phase in phases for access in phase.metadata.get("memory_accesses", ())]
    assert len(accesses) == 1
    assert accesses[0]["operation"] == operation
    assert accesses[0]["address"] == address


def test_dram_endpoint_still_rejects_the_nand_erase_operation():
    memory = _hardware(*PRESETS[0][:2]).get_component("memory")
    with pytest.raises(ValueError, match="DRAM does not support ERASE"):
        endpoint_service(memory, 256, operation="erase", name="invalid", page_offset_bytes=0)
