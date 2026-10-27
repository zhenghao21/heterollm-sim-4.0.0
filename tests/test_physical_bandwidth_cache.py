"""Repeated bandwidth queries retain every mutable physical-contract guard."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim import physical_contract
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbm_config, make_hbf_config


def memory(factory, typed=False):
    config = factory(metadata={"custom": {"rows": [1, 2]}})
    return ComponentSpec("cache-memory", config.kind.value.lower(),
        capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config if typed else asdict(config)})


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
@pytest.mark.parametrize("typed", [False, True])
def test_caps_reuse_validation_only_while_complete_typed_content_is_unchanged(factory, typed, monkeypatch):
    component = memory(factory, typed)
    parse = physical_contract.parse_required_physical_config
    calls = []

    def counted(*args):
        calls.append(1)
        return parse(*args)

    monkeypatch.setattr(physical_contract, "parse_required_physical_config", counted)
    caps = physical_contract.physical_memory_bandwidth_caps_gbps(component)
    assert physical_contract.physical_memory_bandwidth_caps_gbps(component) == caps
    assert len(calls) == 1
    raw = component.metadata["physical_memory_config"]
    metadata = raw.metadata if typed else raw["metadata"]
    metadata["custom"]["rows"][0] = True  # True == 1 must still invalidate.
    assert physical_contract.physical_memory_bandwidth_caps_gbps(component) == caps
    assert len(calls) == 2
    metadata["custom"]["rows"].append({"new": 4})
    assert physical_contract.physical_memory_bandwidth_caps_gbps(component) == caps
    assert len(calls) == 3


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
def test_cached_caps_do_not_hide_missing_fields_invalid_types_or_obsolete_mode(factory):
    component = memory(factory)
    raw = component.metadata["physical_memory_config"]
    physical_contract.physical_memory_bandwidth_caps_gbps(component)
    channels = raw["channels"]
    raw["channels"] = True
    with pytest.raises(ValueError):
        physical_contract.physical_memory_bandwidth_caps_gbps(component)
    raw["channels"] = channels
    burst_field = "page_bytes" if factory is make_hbf_config else "burst_bytes"
    value = raw.pop(burst_field)
    with pytest.raises(ValueError, match="missing explicit fields"):
        physical_contract.physical_memory_bandwidth_caps_gbps(component)
    raw[burst_field] = value
    physical_contract.physical_memory_bandwidth_caps_gbps(component)
    component.metadata["memory_execution_mode"] = None
    with pytest.raises(ValueError, match="no longer configurable"):
        physical_contract.physical_memory_bandwidth_caps_gbps(component)


def test_typed_config_nested_hbm_validation_cannot_be_skipped():
    component = memory(make_hbm_config, typed=True)
    physical_contract.physical_memory_bandwidth_caps_gbps(component)
    component.metadata["physical_memory_config"].metadata["control_status"] = "jedec_invalid"
    with pytest.raises(ValueError, match="JEDEC control status"):
        physical_contract.physical_memory_bandwidth_caps_gbps(component)


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
def test_same_component_id_does_not_reuse_another_components_capacity_or_bandwidth(factory):
    component = memory(factory)
    physical_contract.physical_memory_bandwidth_caps_gbps(component)
    invalid = replace(component, capacity_bytes=component.capacity_bytes - 1)
    with pytest.raises(ValueError, match="capacity_bytes"):
        physical_contract.physical_memory_bandwidth_caps_gbps(invalid)
    invalid = replace(component, bandwidth_gbps=1.e20)
    with pytest.raises(ValueError, match="bandwidth exceeds"):
        physical_contract.physical_memory_bandwidth_caps_gbps(invalid)


def test_explicit_config_and_public_parses_remain_independent_of_cached_caps():
    component = memory(make_hbm_config)
    expected = physical_contract.physical_memory_bandwidth_caps_gbps(component)
    first = physical_contract.require_physical_memory_config(component)
    first.metadata["custom"]["rows"].append(99)
    second = physical_contract.require_physical_memory_config(component)
    assert second is not first
    assert second.metadata["custom"]["rows"] == [1, 2]
    assert physical_contract.physical_memory_bandwidth_caps_gbps(component) == expected
    override = replace(first, interface_bandwidth_gb_s=first.physical_interface_bandwidth_gb_s / 2,
        read_bandwidth_gb_s=None, write_bandwidth_gb_s=None)
    assert physical_contract.physical_memory_bandwidth_caps_gbps(component, physical_config=override) != expected


def test_custom_config_containers_keep_their_live_validation_protocol(monkeypatch):
    class ConfigDict(dict):
        pass

    component = memory(make_hbm_config)
    component.metadata["physical_memory_config"] = ConfigDict(component.metadata["physical_memory_config"])
    parse = physical_contract.parse_required_physical_config
    calls = []

    def counted(*args):
        calls.append(1)
        return parse(*args)

    monkeypatch.setattr(physical_contract, "parse_required_physical_config", counted)
    for _ in range(2):
        physical_contract.physical_memory_bandwidth_caps_gbps(component)
    assert len(calls) == 2


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
def test_borrowed_pricing_matches_owned_config_and_live_core_keeps_its_own_metadata(factory, monkeypatch):
    from heterollm_sim.data_motion import AccessKind, PhysicalRuntimeContext, resolve_service
    from test_decode_execution_acceleration import exact_values

    component = memory(factory)
    service = resolve_service(component)
    actual_runtime, expected_runtime = PhysicalRuntimeContext(), PhysicalRuntimeContext()
    borrow = physical_contract._borrow_physical_memory_config
    for index in range(3):
        actual = service._price(AccessKind.READ, 64, page_offset_bytes=index * 64,
            runtime=actual_runtime, arrival_ns=index * 11.3, summary_only=True)
        with monkeypatch.context() as owned:
            owned.setattr(physical_contract, "_borrow_physical_memory_config",
                          physical_contract.require_physical_memory_config)
            expected = service._price(AccessKind.READ, 64, page_offset_bytes=index * 64,
                runtime=expected_runtime, arrival_ns=index * 11.3, summary_only=True)
        exact_values(actual, expected)
    core = actual_runtime.runtimes[service.physical_owner].core
    core.config.metadata["custom"]["rows"].append(99)
    assert borrow(component).metadata["custom"]["rows"] == [1, 2]
    preview_runtime = PhysicalRuntimeContext()
    preview = preview_runtime.preview_runtime(borrow(component), service.physical_owner)
    preview.core.config.metadata["custom"]["rows"].append(100)
    assert borrow(component).metadata["custom"]["rows"] == [1, 2]


def test_custom_metadata_serialization_hooks_are_never_executed():
    class Value(int):
        def __reduce_ex__(self, protocol):
            raise AssertionError("the content key must not run extension hooks")

    component = memory(make_hbm_config)
    component.metadata["physical_memory_config"]["metadata"]["custom"]["rows"].append(Value(3))
    # Parsing arbitrary metadata retains its own deepcopy rules; the key
    # itself must decline before invoking this extension's reduce protocol.
    assert physical_contract._physical_content_key(component.metadata["physical_memory_config"]) is None
