from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.architecture_scan import (
    _gpu_peak_tops_for_operator,
    _shard_size,
    build_batched_gemm_candidates,
    scan_architecture_candidates,
)
from heterollm_sim.reference import build_reference_scenario


def test_scan_rejects_invalid_rank_mapping_instead_of_scanning_without_ranks():
    scenario = build_reference_scenario()
    original = scenario.placement.parallel.rank_mapping[0]
    parallel = replace(
        scenario.placement.parallel,
        rank_mapping=(replace(original, component_id="missing_gpu"),),
    )
    scenario = replace(scenario, placement=replace(scenario.placement, parallel=parallel))
    with pytest.raises(ValueError, match="unknown component missing_gpu"):
        scan_architecture_candidates(scenario)
    with pytest.raises(ValueError, match="unknown component missing_gpu"):
        build_batched_gemm_candidates(scenario)


def test_no_padding_shard_rejects_indivisible_dimension():
    with pytest.raises(ValueError, match="not divisible"):
        _shard_size(10, 3, False)


def test_unsupported_gpu_dtype_does_not_use_authored_peak():
    component = SimpleNamespace(peak_ops_per_s=1_000_000_000_000.0)
    tensor_core = SimpleNamespace(peak_tops=lambda dtype: (_ for _ in ()).throw(ValueError("unsupported dtype " + dtype)))
    profile = SimpleNamespace(tensor_core=tensor_core, default_tensor_dtype="fp16")
    operator = SimpleNamespace(activation_bits=8, weight_bits=8)
    with pytest.raises(ValueError, match="unsupported dtype int8"):
        _gpu_peak_tops_for_operator(component, profile, operator)
