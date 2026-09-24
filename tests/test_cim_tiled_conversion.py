from dataclasses import replace

import pytest

from heterollm_sim.config import _cim_profile_from_dict
from heterollm_sim.cost_models import DigitalSramCimProfile, GemmWorkload, estimate_cim_gemm
from heterollm_sim.serde import to_primitive


def tiled_profile(**changes):
    values = dict(
        arithmetic_mode="fp16_fp32_analytical",
        float_cycles_per_eval=2,
        float_accumulator_outputs_per_cycle=8,
        float_contract_basis="unvalidated tiled FP16/FP32 analytical contract",
        weight_conversion_mode="packed_to_fp16_tiled_cold",
        weight_decode_elements_per_ns=2,
        activation_fp32_to_fp16_elements_per_ns=3,
        conversion_scratch_capacity_bytes=1 * 1024**2,
        weight_capacity_bytes=2 * 1024**2,
        conversion_contract_basis="unvalidated serial 8x256x256 tile stream",
        tile_m=8,
        tile_k=256,
        tile_n=256,
        max_m_replication=1,
        p_k=4,
        p_n=4,
    )
    values.update(changes)
    return DigitalSramCimProfile(**values)


def tiled_workload(**changes):
    values = dict(
        m=8, k=512, n=1024, activation_bits=16, weight_bits=4,
        accumulator_bits=32, cim_arithmetic="fp16",
        packed_weight_formats=("IQ4_XS",),
        weight_storage_bytes=262144,  # 1024 columns * 2 blocks * 128 B
        weight_metadata_bytes=16384,  # 1024 columns * 2 blocks * 8 B
        packed_weight_transform_operations=1,
        activation_storage_bytes=2 * 8 * 512,
        output_bits=16,
        output_storage_bytes=2 * 8 * 1024,
    )
    values.update(changes)
    return GemmWorkload(**values)


def test_tiled_conversion_is_bounded_and_charges_partial_stream():
    estimate = estimate_cim_gemm(tiled_profile(), tiled_workload())
    audit = estimate.metadata["weight_conversion"]
    assert audit["tile_count"] == 8
    assert audit["conversion_count"] == 8
    assert audit["packed_read_bytes"] == 278528
    assert audit["transfer_input_bytes"] == 32768
    assert audit["transfer_output_bytes"] == 16384
    assert audit["transfer_partial_read_bytes"] > 0
    assert audit["transfer_partial_write_bytes"] > 0
    assert audit["array_peak_bytes"] <= 2 * 1024**2
    assert audit["scratch_peak_bytes"] <= 1 * 1024**2
    assert audit["no_free_traffic"] is True
    assert audit["calibrated"] is False
    assert audit["numerical_equivalence_verified"] is False


def test_tiled_profile_roundtrip_and_capacity_rejection():
    profile = tiled_profile()
    assert _cim_profile_from_dict(to_primitive(profile)) == profile
    with pytest.raises(ValueError, match="scratch"):
        estimate_cim_gemm(replace(profile, conversion_scratch_capacity_bytes=178175), tiled_workload())
    with pytest.raises(ValueError, match="capacity"):
        estimate_cim_gemm(replace(profile, weight_capacity_bytes=131071), tiled_workload())


def test_tiled_conversion_rejects_warm_or_bad_block_layout():
    with pytest.raises(ValueError, match="warm"):
        estimate_cim_gemm(tiled_profile(), tiled_workload(), weights_resident=True)
    with pytest.raises(ValueError, match="authoritative packed payload"):
        estimate_cim_gemm(tiled_profile(), tiled_workload(weight_storage_bytes=1))
    with pytest.raises(ValueError, match="multiple of"):
        tiled_profile(tile_k=128)
