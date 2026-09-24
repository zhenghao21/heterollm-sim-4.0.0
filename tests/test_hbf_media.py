from types import SimpleNamespace
import pytest
from heterollm_sim.hbf_media import hbf_media_service, validate_hbf_media

def component(**updates):
    contract = {"version": "cold_page_v1", "host_transaction_bytes": 64, "host_max_request_bytes": 4096, "media_page_bytes": 4096, "command_queue_depth": 256, "media_parallelism": 4, "page_read_latency_ns": 100, "page_program_latency_ns": 200, "access_pattern": "contiguous_page_aligned", "physical_planes": 2}
    contract.update(updates.pop("hbf_media", {}))
    values = {"read_bandwidth_gbps": 100, "write_bandwidth_gbps": 100, "metadata": {"hbf_media": contract, "read_energy_pj_per_byte": 2, "write_energy_pj_per_byte": 3}}
    values.update(updates)
    return SimpleNamespace(**values)

def test_read_rounding_and_page_ops():
    r = hbf_media_service(component(), 1, True)
    assert (r["host_transfer_bytes"], r["physical_read_bytes"], r["command_count"]) == (64, 4096, 1)
    assert hbf_media_service(component(), 4096, True)["physical_read_bytes"] == 4096

def test_write_rmw_and_full_page():
    assert hbf_media_service(component(), 1, False)["rmw_read_operations"] == 1
    assert hbf_media_service(component(), 4096, False)["rmw_read_operations"] == 0

def test_zero_and_unknown_alignment_bound():
    z = hbf_media_service(component(), 0, False)
    assert z["service_ns"] == 0 and z["command_count"] == 0
    unknown = component(hbf_media={"access_pattern": "unknown_alignment_conservative"})
    one_tx = hbf_media_service(unknown, 64, False)
    page = hbf_media_service(unknown, 4096, False)
    assert one_tx["physical_write_bytes"] == 4096 and one_tx["rmw_read_operations"] == 1
    assert page["physical_write_bytes"] == 8192 and page["rmw_read_operations"] == 2
    assert page["command_count"] == 2 and page["media_command_count"] == 4

def test_qd_limits_media_waves_and_omitted_planes_do_not_force_serial():
    wide = component(hbf_media={"command_queue_depth": 1024, "media_parallelism": 1024, "physical_planes": 1024})
    narrow = component(hbf_media={"command_queue_depth": 256, "media_parallelism": 1024, "physical_planes": 1024})
    assert hbf_media_service(wide, 1024 * 4096, True)["service_ns"] < hbf_media_service(narrow, 1024 * 4096, True)["service_ns"]
    omitted = component(hbf_media={"physical_planes": None, "media_parallelism": 96})
    omitted.metadata["hbf_media"].pop("physical_planes")
    assert hbf_media_service(omitted, 96 * 4096, True)["effective_parallelism"] == 96
    one_plane_small_qd = component(hbf_media={"physical_planes": 1, "command_queue_depth": 256})
    one_plane_large_qd = component(hbf_media={"physical_planes": 1, "command_queue_depth": 1024})
    assert hbf_media_service(one_plane_small_qd, 1024 * 4096, True)["service_ns"] == hbf_media_service(one_plane_large_qd, 1024 * 4096, True)["service_ns"]


def test_latency_parallelism_and_directional_bandwidth_monotonicity():
    slow = hbf_media_service(component(hbf_media={"page_program_latency_ns": 1000}), 4097, False)
    fast = hbf_media_service(component(hbf_media={"page_program_latency_ns": 200}), 4097, False)
    assert slow["service_ns"] >= fast["service_ns"]
    more = hbf_media_service(component(hbf_media={"physical_planes": 4}), 4096 * 8, True)
    less = hbf_media_service(component(hbf_media={"physical_planes": 1}), 4096 * 8, True)
    assert more["service_ns"] <= less["service_ns"]
    write = hbf_media_service(component(write_bandwidth_gbps=1), 4096, False)
    assert write["service_ns"] > hbf_media_service(component(read_bandwidth_gbps=1), 4096, False)["service_ns"]

@pytest.mark.parametrize("bad", [{"command_queue_depth": 255}, {"media_parallelism": True}, {"page_read_latency_ns": float("nan")}, {"nope": 1}])
def test_invalid_contract(bad):
    with pytest.raises(ValueError):
        validate_hbf_media(component(hbf_media=bad))

def test_no_contract_is_noop():
    validate_hbf_media(SimpleNamespace(metadata={}))




def test_exact_physical_bytes_energy_and_serialized_service():
    c = component(read_energy_pj_per_byte=2, write_energy_pj_per_byte=3)
    r = hbf_media_service(c, 64, True)
    w = hbf_media_service(c, 64, False)
    assert r["physical_read_bytes"] == 4096 and r["read_energy_pj"] == 8192
    assert w["physical_read_bytes"] == 4096 and w["physical_write_bytes"] == 4096
    assert w["read_energy_pj"] == 8192 and w["write_energy_pj"] == 12288
    assert w["service_ns"] == pytest.approx(w["host_service_ns"] + w["media_read_service_ns"] + w["media_program_service_ns"])


@pytest.mark.parametrize("bad", [0, -1, 4097, 65, True])
def test_invalid_max_request(bad):
    with pytest.raises(ValueError):
        validate_hbf_media(component(hbf_media={"host_max_request_bytes": bad}))


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), -1, True])
def test_invalid_energy(energy):
    c = component()
    c.metadata["read_energy_pj_per_byte"] = energy
    with pytest.raises(ValueError):
        hbf_media_service(c, 64, True)
