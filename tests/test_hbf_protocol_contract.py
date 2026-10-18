"""New NAND execution choices survive the editor and serialized boundaries."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.web import HttpError, physical_memory_config_payload
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.memory_mapping import map_nand_address
from heterollm_sim.nand_core import NandCore
from heterollm_sim.planner import _summarize_nand_task_traffic
from heterollm_sim.reporting import _sum_batch_storage_traffic


FIELDS = ("protocol_mode", "host_channels", "host_full_duplex", "read_cache_pages_per_bank",
          "queue_scope", "max_outstanding_write_requests")


def test_ocp_preset_declares_256_independent_banks_and_512_gib():
    config = resolve_physical_config("hbf_ocp_0_7")
    assert config.channels == config.host_channels == 16
    assert config.targets_per_channel == config.dies_per_target == config.luns_per_die == 1
    assert config.planes_per_lun == 16
    assert config.planes_independent is True and config.parallel_units is None
    assert config.array_units == 256
    assert (config.blocks_per_plane, config.pages_per_block, config.page_bytes) == (2048, 256, 4096)
    assert config.computed_capacity_bytes == config.effective_capacity_bytes == 512 * 1024**3
    assert config.queue_scope == "host_channel"
    assert config.max_outstanding_requests == 256
    assert config.max_outstanding_write_requests == 64
    assert config.host_read_bandwidth_gb_s == config.host_write_bandwidth_gb_s == 1536.
    # 101.25 GB/s raw includes the 224-byte OOB alongside each 4096-byte page.
    assert config.internal_bandwidth_gb_s == 101.25 * 4096 / (4096 + 224) == 96.
    assert (config.page_read_ns, config.page_program_ns, config.block_erase_ns) == (4000., 75000., 2_000_000.)


def test_ocp_mapping_reaches_every_bank_and_uses_its_physical_host_channel():
    config = resolve_physical_config("hbf_ocp_0_7")
    core = NandCore(config)
    first_pages = [map_nand_address(config, index * config.page_bytes)
                   for index in range(config.array_units)]
    assert {mapping.unit for mapping in first_pages} == set(range(256))
    assert {(mapping.channel, mapping.plane) for mapping in first_pages} == {
        (channel, plane) for channel in range(16) for plane in range(16)}
    for mapping in first_pages:
        assert (mapping.page, mapping.block, mapping.page_offset) == (0, 0, 0)
        assert mapping.unit == mapping.channel + mapping.plane * config.channels
        assert core._host_id(mapping, "read") == f"nand:host:{mapping.channel}:read"
    second_page = map_nand_address(config, config.array_units * config.page_bytes)
    assert (second_page.unit, second_page.page, second_page.block) == (0, 1, 0)
    last = map_nand_address(config, config.effective_capacity_bytes - 1)
    assert (last.channel, last.plane, last.block, last.page, last.page_offset) == (15, 15, 2047, 255, 4095)


@pytest.mark.parametrize("field", FIELDS)
def test_serialized_nand_cannot_silently_acquire_protocol_choices(field):
    raw = asdict(resolve_physical_config("hbf_ocp_0_7"))
    raw.pop(field)
    with pytest.raises(ValueError, match=field):
        parse_execution_physical_config(raw)


def test_explicit_unlimited_write_subset_survives_serialization_and_inspector():
    raw = asdict(resolve_physical_config("hbf_ocp_0_7"))
    raw["max_outstanding_write_requests"] = None
    parsed = parse_execution_physical_config(raw)
    assert parsed.max_outstanding_write_requests is None
    assert parsed.queue_scope == "host_channel"
    updated = physical_memory_config_payload({"config": raw, "overrides": {}})
    assert "max_outstanding_write_requests" in updated
    assert updated["max_outstanding_write_requests"] is None


def test_inspector_round_trip_keeps_ocp_protocol_and_independent_channels():
    raw = asdict(resolve_physical_config("hbf_ocp_0_7"))
    updated = physical_memory_config_payload({"config": raw, "overrides": {"host_channels": 1}})
    assert updated["protocol_mode"] == "ocp_hbf_0_7"
    assert updated["host_channels"] == 1
    assert updated["host_full_duplex"] is True
    assert updated["read_cache_pages_per_bank"] == 2
    assert raw["host_channels"] == resolve_physical_config("hbf_ocp_0_7").host_channels


def test_inspector_rejects_hbf_geometry_that_conflicts_with_selected_protocol():
    raw = asdict(resolve_physical_config("hbf_ocp_0_7"))
    with pytest.raises(HttpError) as caught:
        physical_memory_config_payload({"config": raw, "overrides": {"page_bytes": 16384}})
    assert caught.value.status == 422


def test_browser_and_execution_agree_on_new_nand_contract():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    required = re.search(r"const NAND_PHYSICAL_REQUIRED_FIELDS = \[.*?\];", source, re.S).group()
    validate = re.search(r"^function validatePhysicalMemoryConfig\([^\n]*\) \{.*?^\}", source, re.M | re.S).group()
    raw = asdict(resolve_physical_config("hbf_ocp_0_7"))
    script = """
const assert = require('node:assert/strict');
const PHYSICAL_MEMORY_KINDS = new Set(['HBF', 'SSD']);
const normalizedComponentKind = kind => kind.toLowerCase();
""" + required + "\n" + validate + "\nconst raw = " + json.dumps(raw) + ";\n" + """
const component = {kind: 'hbf', component_id: 'hbf0', capacity_bytes: raw.capacity_bytes};
assert.deepEqual(validatePhysicalMemoryConfig(raw, component), raw);
for (const field of ['protocol_mode', 'host_channels', 'host_full_duplex', 'read_cache_pages_per_bank', 'queue_scope', 'max_outstanding_write_requests']) {
  const missing = {...raw}; delete missing[field];
  assert.throws(() => validatePhysicalMemoryConfig(missing, component));
}
for (const patch of [
  {protocol_mode: 'legacy'}, {page_bytes: 16384}, {host_granularity_bytes: 4096},
  {host_full_duplex: false}, {host_channels: 17}, {read_cache_pages_per_bank: 1},
  {write_buffer_bytes: 4096}, {queue_scope: 'device'}, {parallel_units: 1},
  {planes_independent: false}, {max_outstanding_write_requests: 257},
]) {
  assert.throws(() => validatePhysicalMemoryConfig({...raw, ...patch}, component));
}
console.log(JSON.stringify(validatePhysicalMemoryConfig({...raw, host_channels: 1}, component)));
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr
    actual = parse_execution_physical_config(json.loads(result.stdout))
    assert actual.protocol_mode == "ocp_hbf_0_7"
    assert actual.host_channels == 1


def test_rollback_restores_ocp_page_order_and_cache_on_repeated_retry():
    config = resolve_physical_config("hbf_ocp_0_7", overrides={
        "channels": 1, "host_channels": 1, "parallel_units": None,
        "planes_per_lun": 1, "planes_independent": True,
        "capacity_bytes": 65536,
    })
    runtime = PhysicalRuntimeContext()
    core = runtime.runtime(config, "hbf").core
    core.submit(AccessRequest("page0", "write", 0, 4096))
    core.submit(AccessRequest("cache0", "read", 0, 64))
    saved = runtime.snapshot()
    for attempt in range(2):
        core.submit(AccessRequest(f"page1-{attempt}", "write", 4096, 4096))
        # Evict the cached page before rolling back the failed task.
        core.submit(AccessRequest(f"read2-{attempt}", "read", 8192, 64))
        core.submit(AccessRequest(f"read3-{attempt}", "read", 12288, 64))
        runtime.restore(saved)
        with pytest.raises(ValueError, match="write order"):
            core.submit(AccessRequest("skip-page1", "write", 8192, 4096))
        reread = core.submit(AccessRequest(f"reread-{attempt}", "read", 0, 64))
        assert reread.counters["read_cache_hits"] == 1
        assert reread.counters["pages_read"] == 0
        runtime.restore(saved)


def test_service_batch_reports_actual_cache_traffic():
    config = resolve_physical_config("hbf_ocp_0_7")
    component = ComponentSpec("hbf", "hbf", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config})
    result = resolve_service(component).price_batch([
        {"operation": "read", "address": 0, "byte_count": 64},
        {"operation": "read", "address": 64, "byte_count": 64},
    ], runtime=PhysicalRuntimeContext())
    assert result["read_cache_misses"] == 1
    assert result["read_cache_hits"] == 1
    assert result["host_transfer_bytes"] == 128
    assert result["physical_read_bytes"] == 4096


def test_cache_observations_survive_task_and_online_aggregation():
    core = NandCore(resolve_physical_config("hbf_ocp_0_7"))
    ledgers = []
    for index in range(2):
        result = core.submit(AccessRequest(str(index), "read", index * 64, 64))
        task = SimpleNamespace(task_id=str(index), demands=(), metadata={
            "physical_execution_by_owner": {"hbf": {"kind": "HBF", **result.counters}}})
        ledgers.append(_summarize_nand_task_traffic([task]))
    result = SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={"storage_traffic": row})) for row in ledgers]))
    summary = _sum_batch_storage_traffic(result)
    assert summary["read_cache_hits"] == 1
    assert summary["read_cache_misses"] == 1
    assert summary["metric_availability"]["read_cache_hits"]["status"] == "recorded"
    # An old or incomplete observation is unavailable, not an invented zero.
    del ledgers[1]["read_cache_hits"]
    assert _sum_batch_storage_traffic(result)["read_cache_hits"] is None
