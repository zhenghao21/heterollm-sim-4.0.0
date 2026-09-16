from dataclasses import replace
from heterollm_sim import planner
from heterollm_sim.cost_models import GemmWorkload,TensorKernelWorkload
from heterollm_sim.conversion_work import ConversionSourceContract,SOURCE_SHA256,derive_conversion_work
from tests.test_mmq_planner import scenario
import pytest

def configured(enabled):
    case=scenario(tokens=4,weight_format="Q5_0")
    binding=dict(compute_capability=1200,highest_compiled_arch=1200,warp_size=32,
        source_hashes=dict(SOURCE_SHA256),runtime_binary_sha256="a"*64,ordinary_contiguous_2d=True)
    target=next(c for c in case.hardware.components if c.component_id=="gpu0")
    target=replace(target,metadata={**target.metadata,"llama_cpp_conversion_source_contract":binding})
    return replace(case,hardware=replace(case.hardware,components=tuple(target if c.component_id=="gpu0" else c for c in case.hardware.components)),
        workload=replace(case.workload,metadata={**case.workload.metadata,"llama_cpp_conversion_cta_costs":enabled})),target

def test_optin_lowering_grid_matches_source_no_io_change():
    case,target=configured(True);load=GemmWorkload(4,896,128,activation_bits=16,activation_storage_bytes=4*4*896,weight_bits=5,output_bits=32,packed_weight_formats=("Q5_0",))
    original=TensorKernelWorkload(11*4*1024,4*4*896,36*4*1024//32,streaming_fraction=1.)
    result,audit=planner._source_conversion_parallelism(case,target,load,original,"MMVQ_Q8_1")
    assert result.source_grid_ctas==16 and audit["applied"] is True
    assert replace(result,source_grid_ctas=None)==original

def test_disabled_returns_original_identity():
    case,target=configured(False);load=GemmWorkload(4,896,128)
    original=TensorKernelWorkload(1,4,4)
    value,audit=planner._source_conversion_parallelism(case,target,load,original,"MMVQ_Q8_1")
    assert value is original and audit is None

def test_uncovered_format_returns_analytical_and_reason():
    case,target=configured(True);load=GemmWorkload(4,896,128,packed_weight_formats=("Q4_K",))
    original=TensorKernelWorkload(1,4,4)
    value,audit=planner._source_conversion_parallelism(case,target,load,original,"MMVQ_Q8_1")
    assert value is original and audit["applied"] is False

def test_full_planner_optin_only_changes_declared_conversion_demands():
    a,_=configured(False);b,_=configured(True)
    old,new=planner.compile_scenario(a),planner.compile_scenario(b)
    assert len(old.tasks)==len(new.tasks)
    changed=0
    for x,y in zip(old.tasks,new.tasks):
        assert (x.task_id,x.dependencies,x.category)==(y.task_id,y.dependencies,y.category)
        if x.demands!=y.demands:
            assert y.metadata.get("conversion_source_work",{}).get("applied") is True
            changed+=1
    assert any(t.metadata.get("conversion_source_work",{}).get("applied") for t in new.tasks)
