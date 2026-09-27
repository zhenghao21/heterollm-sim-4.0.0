from heterollm_sim.component_presets import component_preset_detail, list_component_presets
import pytest


PUBLIC_IDS = {
    "amd-ryzen-9-9950x3d",
    "samsung-hbm3e-36gb-9_2",
    "gddr7-16gb-30_0-256bit",
    "sk-hynix-hbf-512gb",
    "ymtc-zhitai-ti-pro9100",
    "samsung-ddr5-32gb-udimm-5600",
    "acer-local-ddr5-128gb-5600-dual-channel",
    "sram-cim-analytical-tile",
    "nvidia-b200-sxm-gpu",
    "nvidia-rtx-5080",
}


def _detail(preset_id):
    return component_preset_detail(preset_id)["component"]


def test_public_hardware_catalog_is_the_curated_catalog():
    assert {item["id"] for item in list_component_presets()} == PUBLIC_IDS


def test_vendor_parameter_units_and_provenance_are_explicit():
    samsung_hbm = _detail("samsung-hbm3e-36gb-9_2")
    assert samsung_hbm["capacity_bytes"] == 36_000_000_000
    assert samsung_hbm["read_bandwidth_gbps"] == 9420.8
    assert samsung_hbm["metadata"]["vendor_parameter_provenance"]["pin_speed_gbps"]["unit"] == "Gb/s_per_pin"
    assert samsung_hbm["metadata"]["sources"][0]["url"].startswith("https://semiconductor.samsung.com/")

    hbf = _detail("sk-hynix-hbf-512gb")
    assert hbf["capacity_bytes"] == 512_000_000_000
    assert hbf["read_bandwidth_gbps"] == 3904.0
    assert hbf["write_bandwidth_gbps"] == 217.6
    assert hbf["metadata"]["read_latency_ns"] == 4000.0
    assert hbf["metadata"]["write_latency_ns"] == 75000.0
    assert hbf["metadata"]["access_mode"] == "memory"
    assert hbf["metadata"]["vendor_parameter_provenance"]["write_bandwidth_gbps"]["status"] == "analytical_user_configured"
    assert hbf["metadata"]["sources"][0]["url"] == "https://news.skhynix.com/en/hbf-at-fms-2026/"

    dram = _detail("samsung-ddr5-32gb-udimm-5600")
    assert dram["capacity_bytes"] == 32_000_000_000
    assert dram["read_bandwidth_gbps"] == 358.4
    assert dram["metadata"]["technology"]["data_rate_mt_s"] == 5600

    b200 = _detail("nvidia-b200-sxm-gpu")
    assert b200["metadata"]["technology"]["device_memory_gb"] == 180.0
    assert b200["metadata"]["technology"]["device_memory_bandwidth_gbps"] == 64_000.0

    gddr7 = _detail("gddr7-16gb-30_0-256bit")
    assert gddr7["metadata"]["technology"]["memory_type"] == "GDDR7"
    assert gddr7["ports"][0]["protocol"] == "GDDR7"
    assert gddr7["ports"][0]["bandwidth_gbps"] == 7680.0


def test_local_ddr5_preset_matches_the_installed_dual_channel_configuration():
    local = _detail("acer-local-ddr5-128gb-5600-dual-channel")
    assert local["capacity_bytes"] == 4 * 32 * 1024 ** 3
    assert local["read_bandwidth_gbps"] == 716.8
    assert local["write_bandwidth_gbps"] == 716.8
    assert local["ports"][0]["protocol"] == "DDR5"
    assert local["ports"][0]["bandwidth_gbps"] == 716.8
    snapshot = local["metadata"]["local_hardware_snapshot"]
    assert snapshot["part_numbers"] == ["BL.9BWWR.424", "BL.9BWWR.373"]
    assert snapshot["configured_speed_mt_s"] == 5600
    assert snapshot["channel_count"] == 2
    assert snapshot["theoretical_bandwidth_gb_s"] == 89.6


def test_desktop_prediction_hardware_has_official_sources_and_roles():
    cpu = _detail("amd-ryzen-9-9950x3d")
    assert cpu["metadata"]["sources"][0]["url"] == (
        "https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/"
        "amd-ryzen-9-9950x3d.html"
    )
    assert cpu["metadata"]["technology"]["core_count"] == 16
    assert cpu["metadata"]["technology"]["thread_count"] == 32
    assert cpu["capacity_bytes"] == 0
    assert cpu["peak_ops_per_s"] == 0.0
    assert cpu["ports"][0]["bandwidth_gbps"] == pytest.approx(716.8)

    gpu = _detail("nvidia-rtx-5080")
    assert gpu["metadata"]["sources"][0]["url"] == (
        "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/"
    )
    assert gpu["metadata"]["technology"]["cuda_cores"] == 10_752
    assert gpu["metadata"]["technology"]["device_memory_gb"] == 16.0
    assert gpu["metadata"]["technology"]["device_memory_bandwidth_gb_s"] == 960.0
    assert "memory_capacity_gb" not in gpu["ports"][0]["metadata"]
    # GPU compute endpoints do not duplicate attached memory fields.
    assert gpu["capacity_bytes"] == 0
    assert gpu["read_bandwidth_gbps"] == 0.0
    assert gpu["write_bandwidth_gbps"] == 0.0
    assert gpu["peak_ops_per_s"] == pytest.approx(112.6e12)
    assert gpu["metadata"]["memory_requires_explicit_component"] is True
    assert gpu["metadata"]["sources"][1]["url"].endswith("nvidia-rtx-blackwell-gpu-architecture.pdf")


def test_user_tipro9100_name_maps_to_official_tipplus9100_sku():
    ssd = _detail("ymtc-zhitai-ti-pro9100")
    provenance = ssd["metadata"]["vendor_parameter_provenance"]
    assert ssd["capacity_bytes"] == 1_024_000_000_000
    assert ssd["read_bandwidth_gbps"] == 96.0
    assert ssd["write_bandwidth_gbps"] == 85.6
    assert ssd["ports"][0]["bandwidth_gbps"] == 126.03076923076924
    assert provenance["mpn"]["value"] == "ZTSS3CB08D6CMC"
    assert provenance["random_iops"]["value"] == 1850
    assert ssd["metadata"]["sources"][0]["url"] == "https://www.ymtc.com/cn/products/77.html?cat=44"
