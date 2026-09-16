"""Source-qualified rectangular non-Flash KV geometry; pure tiny task graphs."""
from dataclasses import replace
from unittest.mock import patch
import copy
import json
from pathlib import Path

import pytest
from heterollm_sim import planner
from heterollm_sim.ir import SchedulerSpec
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import BatchCohort, BatchItem
from tests.test_final_layer_output_selection_planner import _scenario, _cohort

KEY = "llama_cpp_nonflash_kv_view"
# Fixed-source declaration used by the bounded fixture. Integration below
# independently re-derives and checks the real build/object/module linkage.
SOURCE_SHA = {
    "src/llama-kv-cache.cpp": "e4d2aa977c8aa048ddd79c878683e94909f5f24ac109b0fcd1c0b59ef70f3899",
    "src/llama-graph.cpp": "a6a8241c2d149961801d0fdeaa68f1bb176297b4d5156b6138db0b3d169abb04",
    "src/llama-model.cpp": "94ede4e7ac8119c5a4d2fad97e3432008ec7d30b42ab37395db6e2a8047d1984",
    "annotation/src/llama-context.cpp": "e677c1e6e56fc08561fa56d9861501740405190e578d690e9efcad26fa48622e",
}


def scenario(parallel=2, slot=2048, enabled=True, architecture="llama_decoder"):
    case = _scenario(architecture)
    runtime = LlamaCppRuntimeConfig(batch=64, ubatch=64, context=slot, parallel=parallel,
        flash_attn=False, kv_unified=True, kv_type_k="f16", kv_type_v="f16")
    contract = {"schema": "heterollm.llama-nonflash-kv-view/v1", "runtime_binding_status": "verified",
        "source_sha256": SOURCE_SHA, "n_pad": 1, "n_kv_padding": 256,
        "configuration": {"batch": 64, "ubatch": 64, "parallel": parallel,
            "simulator_slot_context_tokens": slot, "native_context_tokens": slot * parallel,
            "flash_attn": False, "kv_unified": True, "kv_type_k": "f16", "kv_type_v": "f16"}}
    scheduler = SchedulerSpec(mode="continuous", max_num_seqs=parallel,
        max_num_batched_tokens=64, max_num_ubatch_tokens=64, prefill_chunk_tokens=64,
        mixed_phase_batching=True, phase_candidate_order="stable_admission")
    return replace(case, llama_cpp_config=runtime,
        fusion_policy=replace(case.fusion_policy, flash_attention=False),
        workload=replace(case.workload, scheduler=scheduler,
            metadata={**case.workload.metadata, **({KEY: contract} if enabled else {})}))


def mixed(context=128):
    return BatchCohort("mixed", "mixed", 0.0, (
        BatchItem("A", "decode", 1, context, logit_tokens=1),
        BatchItem("B", "prefill", 63, 0, logit_tokens=0)))


def qk_tasks(schedule):
    return [task for task in schedule.tasks if ".attention_qk." in task.name and task.metadata.get("phase") in {"gpu_gemm", "cpu_gemm"}]


def shape_tasks(schedule):
    return [task for task in schedule.tasks if task.metadata.get("phase") in {"gpu_gemm", "cpu_gemm"}
        and ("attention_qk" in task.name or "attention_pv" in task.name)]


@pytest.mark.parametrize("parallel", [1, 2, 4])
def test_first64_fixed_protocol_retains_logical_lanes_and_rectangular_width(parallel):
    case = scenario(parallel)
    cohort = _cohort(64, 0)
    group, = planner._serving_invocation_groups(case, cohort)
    assert group.context_tokens == 33 and group.kv_scan_tokens == 0
    assert group.nonflash_kv_view["physical_k_tokens"] == 256
    assert [x.context_tokens for x in group.lanes] == list(range(1, 65))
    schedule = planner.compile_serving_cohort_schedule(case, cohort)
    physical = shape_tasks(schedule)
    assert len(physical) == 4  # two layers, one QK and one PV invocation each
    for task in physical:
        assert task.metadata[KEY]["physical_k_tokens"] == 256
        assert task.metadata["physical_ubatch_rows"] == 64
    old = planner.compile_serving_cohort_schedule(scenario(parallel, enabled=False), cohort)
    assert len(shape_tasks(old)) == len(physical)
    assert len(schedule.tasks) == len(old.tasks)  # no per-row launches or extra KV tasks
    for task in schedule.tasks:
        if task.metadata.get("event_kind") == "kv_read":
            assert task.metadata["kv_token_accesses"] == 256
            assert task.metadata["resource_accounting"] == "included_in_attention_kernel"
            assert not task.demands


def test_mixed_decode_prefill_keeps_masks_logits_and_projection_batch():
    case = scenario()
    cohort = mixed()
    group, = planner._serving_invocation_groups(case, cohort)
    assert (group.token_batch, group.context_tokens, group.kv_read_tokens, group.kv_scan_tokens) == (64, 34, 128, 0)
    assert group.nonflash_kv_view["physical_k_tokens"] == 256
    assert group.logit_token_batch == 1
    assert [(l.request_id, l.phase, l.position, l.requires_logits) for l in group.lanes] == [
        ("A", "decode", 0, True), *[("B", "prefill", p, False) for p in range(63)]]
    new = planner.compile_serving_cohort_schedule(case, cohort)
    old = planner.compile_serving_cohort_schedule(scenario(enabled=False), cohort)
    assert len(new.tasks) == len(old.tasks)
    gemms = [t for t in new.tasks if t.metadata.get("phase") == "gpu_gemm"]
    assert shape_tasks(new)
    assert all(t.metadata[KEY]["logical_context_mean"] == 34 for t in gemms)
    assert [(t.name, t.metadata.get("physical_ubatch_rows")) for t in gemms] == [
        (t.name, t.metadata.get("physical_ubatch_rows")) for t in old.tasks if t.metadata.get("phase") == "gpu_gemm"]
    # Only attention QK/PV and score work changes; output/projection calls stay batched.
    for a, b in zip(new.tasks, old.tasks):
        if a.metadata.get("projection_id"):
            assert a.demands == b.demands


@pytest.mark.parametrize("context,expected", [(0,256),(255,256),(256,512),(511,512),(512,768),(2047,2048)])
def test_physical_extent_grows_with_provable_retained_context(context, expected):
    group, = planner._serving_invocation_groups(scenario(), _cohort(1,1,context=context,phase="decode"))
    assert group.nonflash_kv_view["physical_k_tokens"] == expected


def test_concurrent_sequence_contexts_not_summed_and_capacity_is_total_pool():
    case = scenario(parallel=4, slot=512)
    cohort = BatchCohort("decode", "decode", 0., tuple(BatchItem(str(i),"decode",1,511,logit_tokens=1) for i in range(4)))
    group, = planner._serving_invocation_groups(case, cohort)
    assert group.nonflash_kv_view["native_cache_cells"] == 2048
    assert group.nonflash_kv_view["physical_k_tokens"] == 512
    assert group.nonflash_kv_view["occupied_cells_lower_bound"] == 512
    assert "shared_prefix_union_unknown" in group.nonflash_kv_view["uncovered_reasons"][0]
    single, = planner._serving_invocation_groups(scenario(parallel=1,slot=256), _cohort(1,1,context=255,phase="decode"))
    assert single.nonflash_kv_view["physical_k_tokens"] == single.nonflash_kv_view["native_cache_cells"] == 256


@pytest.mark.parametrize("change,reason", [
    ("flash", "requires_nonflash_unified_kv"), ("nonunified", "requires_nonflash_unified_kv"),
    ("generic", "llama_cpp_runtime_config_required"), ("unverified", "source_runtime_build_link_not_verified"),
    ("configuration", "frozen_native_configuration_mismatch"), ("overflow", "context_shift_or_slot_overflow_not_modeled")])
def test_unsupported_paths_fail_closed_with_explicit_reason(change, reason):
    case = scenario()
    cohort = mixed(2048 if change == "overflow" else 128)
    if change == "flash":
        case = replace(case,llama_cpp_config=replace(case.llama_cpp_config,flash_attn=True),fusion_policy=replace(case.fusion_policy,flash_attention=True))
    if change == "nonunified": case = replace(case,llama_cpp_config=replace(case.llama_cpp_config,kv_unified=False))
    if change == "generic": case = replace(case,llama_cpp_config=None)
    if change in {"unverified","configuration"}:
        meta = copy.deepcopy(case.workload.metadata)
        if change == "unverified": meta[KEY]["runtime_binding_status"] = "conditional"
        else: meta[KEY]["configuration"]["parallel"] = 4
        case = replace(case,workload=replace(case.workload,metadata=meta))
    group, = planner._serving_invocation_groups(case,cohort)
    assert group.nonflash_kv_view["applied"] is False
    assert group.nonflash_kv_view["uncovered_reasons"] == (reason,)


def test_dynamic_replay_recomputes_physical_width_and_reads_at_padding_boundary():
    case = scenario(architecture="qwen3_5_hybrid_transformer")
    # Native Qwen full attention includes an explicit QK scale; use its
    # existing execution descriptor so the unfused replay contract is valid.
    from tests.test_physical_projection_invocations import _attention_execution_descriptors, _full_layer
    from tests.model_helpers import replace_model_layer_specs
    first = planner._execution_layers(case)[0]
    first = replace(first, hidden_size=128, intermediate_size=256)
    full = _full_layer(_attention_execution_descriptors())
    case = replace(case, model=replace_model_layer_specs(case.model, (first, full)))
    cold = _cohort(1,1,name="cold",context=255,phase="decode")
    warm = _cohort(1,1,name="warm",context=256,phase="decode")
    expected = planner.compile_serving_cohort_schedule(case,warm)
    context = planner.CompilationContext(case,eager_full_attention_segments=False,compiled_serving_invocation_segments=True)
    with patch.object(planner,"_compile_parallel_iteration",wraps=planner._compile_parallel_iteration) as body:
        with planner._compilation_scope(case,context):
            planner.compile_serving_cohort_schedule(case,cold)
            count = body.call_count
            actual = planner.compile_serving_cohort_schedule(case,warm)
    assert body.call_count == count  # true dynamic replay, not a fresh compile
    assert actual == expected
    assert all(t.metadata[KEY]["physical_k_tokens"] == 512 for t in shape_tasks(actual))


def test_exact_cohort_cache_distinguishes_context_boundary():
    case = scenario()
    ctx = planner.CompilationContext(case)
    a = _cohort(1,1,name="a",context=255,phase="decode")
    b = _cohort(1,1,name="b",context=256,phase="decode")
    with planner._compilation_scope(case,ctx):
        planner.compile_serving_cohort_schedule(case,a)
        replay = planner.compile_serving_cohort_schedule(case,b)
    assert replay == planner.compile_serving_cohort_schedule(case,b)


def test_qk_pv_and_softmax_receive_rectangular_workloads_not_only_audit_metadata():
    case = scenario()
    with patch.object(planner, "_add_rank_gemm", wraps=planner._add_rank_gemm) as gemm, \
            patch.object(planner, "_add_rank_primitive", wraps=planner._add_rank_primitive) as primitive:
        planner.compile_serving_cohort_schedule(case, mixed())
    qk = [call.args[5] for call in gemm.call_args_list if call.args[5].name == "attention_qk_tp"]
    pv = [call.args[5] for call in gemm.call_args_list if call.args[5].name == "attention_pv_tp"]
    assert len(qk) == len(pv) == 2
    assert all(work.m == 64 and work.n == 256 for work in qk)
    assert all(work.m == 64 and work.k == 256 for work in pv)
    reductions = [call.args[6] for call in primitive.call_args_list if call.args[6].name == "softmax_reduce_max_sum"]
    norms = [call.args[6] for call in primitive.call_args_list if call.args[6].name == "softmax_sub_exp_normalize"]
    assert len(reductions) == len(norms) == 2
    assert all(work.input_elements == 64 * 256 for work in reductions)
    assert all(work.elements == 64 * 256 for work in norms)


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "qwen3_5_hybrid_transformer"])
def test_actual_gguf_import_architectures_are_covered(architecture):
    case = scenario(architecture=architecture)
    groups = planner._serving_invocation_groups(case, mixed())
    assert all(group.nonflash_kv_view["applied"] for group in groups)
    assert all(group.nonflash_kv_view["physical_k_tokens"] == 256 for group in groups)


def test_sliding_window_metadata_is_explicitly_uncovered():
    case = scenario()
    raw = {**case.workload.metadata[KEY], "model_cache_scope": {"ordinary_retained_prefix": False}}
    case = replace(case, workload=replace(case.workload, metadata={**case.workload.metadata, KEY: raw}))
    group, = planner._serving_invocation_groups(case, mixed())
    assert group.nonflash_kv_view["uncovered_reasons"] == ("sliding_or_nonordinary_retained_prefix_not_proven",)


def test_mixed_long_prefix_never_falls_back_to_mean_or_constant_256():
    case = scenario()
    cohort = mixed(context=1024)
    group, = planner._serving_invocation_groups(case, cohort)
    assert group.context_tokens == 48
    assert group.nonflash_kv_view["physical_k_tokens"] == 1280
    assert group.nonflash_kv_view["occupied_cells_lower_bound"] == 1025
    assert group.kv_scan_tokens == 0
    with patch.object(planner, "_add_rank_gemm", wraps=planner._add_rank_gemm) as gemm:
        planner.compile_serving_cohort_schedule(case, cohort)
    assert all(call.args[5].n == 1280 for call in gemm.call_args_list
        if call.args[5].name == "attention_qk_tp")
