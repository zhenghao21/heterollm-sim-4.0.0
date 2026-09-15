"""Source-qualified GPU invocation and F32-to-KV boundaries; no timing fit."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from heterollm_sim import planner
from heterollm_sim.llama_gpu_invocations import (
    SOURCE_KEY, SCHEMA, _hash, apply_llama_gpu_invocation_contract, derive_llama_gpu_invocation_contract,
)
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import BatchCohort,BatchItem
from heterollm_sim.serde import to_primitive
from heterollm_sim.mmq_work import MMVQ_MAX_BATCH_SIZE
from tests.model_helpers import model_from_layer_specs
from tests.test_mmq_planner import scenario as base_scenario


def contract(*,fusion=True,status="conditional",device=None):
    c={"schema":SCHEMA,"status":status,"runtime_binding_sha256":"a"*64,
       "runtime_modules":{},"cuda_compute_capability":1200,"fusion_enabled":fusion,
       "architecture_rope_modes":{"llama":"normal","qwen2":"neox","qwen3_5_hybrid_transformer":"imrope"},
       "kernel_environment":{"GGML_CUDA_DISABLE_FUSION":{"captured":fusion is not None,"value":None}},
       "packed_qkv_cache_layout":{"schema":"llama.cpp.packed-qkv-reshape-v1",
           "matmul_output_dtype":"F32","view":"reshape_3d_with_packed_token_stride",
           "copy_required_when_flash_off":True,"strided_copy":"f32_scalar_kernel",
           "single_token_copy":"cuda_memcpy_d2d"},
       "mmq_device_evidence":device or {"available":False},
       "uncovered_reasons":[] if fusion is not None else ["historical_cuda_fusion_environment_unknown"],
       "source_refs":[],"unpriced_terms":["set_rows_conversion_instructions"],"native_dispatch_proven":False,
       "conditional_reasons":["model-specific unity include historical source-body hashes were not captured"]}
    c["content_sha256"]=_hash(c);return c


def scenario(*,m=8,fmt="IQ4_XS",flash=False,legacy_cpu_flag=False,architecture="qwen2"):
    base=base_scenario(enabled=None,tokens=m,weight_format=fmt)
    layer=planner._execution_layers(base)[0]
    metadata=deepcopy(layer.metadata)
    projections=metadata["weight_projection_descriptors"]["projections"]
    for key in ("attention.q","attention.k","attention.v"):projections.pop(key,None)
    bindings={}
    for projection in projections.values():
        for seg in projection["segments"]:
            bindings[seg["physical_tensor_name"]]={"name":seg["physical_tensor_name"],"shape":[seg["k"],seg["n"]],
                "type":seg["format"],"n_bytes":seg["physical_bytes"],"offset":0}
    metadata["gguf_tensor_bindings"]=list(bindings.values())
    layer=replace(layer,metadata=metadata)
    model=model_from_layer_specs(base.model.name,(layer,),architecture=architecture,metadata={"gguf_sha256":"c"*64})
    flags=dict(base.workload.metadata)
    flags.pop("llama_cpp_physical_projection_invocations",None)
    flags.pop("llama_cpp_f32_hidden_storage",None)
    if legacy_cpu_flag:flags["llama_cpp_physical_projection_invocations"]=True
    runtime=LlamaCppRuntimeConfig(gpu_layers=-1,kv_type_k="f16",kv_type_v="f16",flash_attn=flash)
    return replace(base,model=model,llama_cpp_config=runtime,workload=replace(base.workload,metadata=flags))


def changed_layer(base,fn):
    layer=planner._execution_layers(base)[0]
    metadata=deepcopy(layer.metadata);fn(metadata)
    layer=replace(layer,metadata=metadata)
    model=model_from_layer_specs(base.model.name,(layer,),architecture=base.model.architecture,metadata=base.model.metadata)
    return replace(base,model=model)


def lower(base):
    return planner.compile_scenario(base)


def projections(schedule):
    return [t for t in schedule.tasks if t.metadata.get("phase")=="gpu_gemm" and t.metadata.get("projection_id")]


def signature(schedule):
    return [(t.task_id,t.dependencies,to_primitive(t.demands)) for t in schedule.tasks]


class GPUInvocationTests(unittest.TestCase):
    def test_default_and_missing_contract_are_exact_identity(self):
        base=scenario()
        self.assertIs(apply_llama_gpu_invocation_contract(base),base)
        self.assertIs(apply_llama_gpu_invocation_contract(base,contract()),base)
        self.assertIs(apply_llama_gpu_invocation_contract(base,None,enabled=True),base)
        self.assertEqual([to_primitive(t) for t in lower(base).tasks],
            [to_primitive(t) for t in lower(apply_llama_gpu_invocation_contract(base,contract(),enabled=False)).tasks])

    def test_aliases_only_in_returned_model_and_cpu_flag_not_retroactively_enabled(self):
        base=scenario(legacy_cpu_flag=True)
        original=deepcopy(base.model)
        candidate=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        self.assertEqual(base.model,original)
        layer=planner._execution_layers(candidate)[0]
        self.assertTrue(planner._declared_physical_projections(candidate,layer,("attention.q","attention.k","attention.v"),
            combined_projection_id="attention.qkv",execution_component_id="gpu0"))
        self.assertFalse(planner._declared_physical_projections(candidate,layer,("attention.q","attention.k","attention.v"),
            combined_projection_id="attention.qkv",execution_component_id="cpu0"))
        self.assertNotIn("attention.q",planner._execution_layers(base)[0].metadata["weight_projection_descriptors"]["projections"])

    def test_m1_fuses_matching_quantized_ffn_m2_to_m64_split(self):
        for m in (1,2,4,5,6,7,8,64):
            with self.subTest(m=m):
                base=scenario(m=m)
                graph=lower(apply_llama_gpu_invocation_contract(base,contract(),enabled=True))
                tasks=projections(graph)
                self.assertEqual(len(tasks),6 if m==1 else 7)
                ids=[t.metadata["projection_id"] for t in tasks]
                self.assertEqual(set(ids)&{"attention.q","attention.k","attention.v"},{"attention.q","attention.k","attention.v"})
                self.assertEqual("mlp.up_gate" in ids,m==1)
                if m==1:
                    fused=next(t for t in tasks if t.metadata["projection_id"]=="mlp.up_gate")
                    self.assertTrue(fused.metadata["fusion_enabled"])
                    self.assertEqual(fused.metadata["projection_segment_count"],2)
                else:
                    for t in tasks:
                        if t.metadata["projection_id"] in ("mlp.gate","mlp.up"):
                            self.assertFalse(t.metadata["fusion_enabled"])
                self.assertFalse(any(t.metadata.get("mmq_source_work") for t in tasks))
                self.assertTrue(all(t.metadata["gpu_native_invocation"]["native_dispatch_proven"] is False for t in tasks))

    def test_all_regular_matrix_outputs_are_f32_before_kv_cast(self):
        for m in (1,8,64):
            graph=lower(apply_llama_gpu_invocation_contract(scenario(m=m),contract(),enabled=True))
            for t in projections(graph):
                if t.metadata["projection_id"]=="mlp.up_gate":continue
                n=t.metadata["gemm_n"]
                self.assertEqual(t.metadata["cost_model"]["output_bytes"],4*m*n,(t.name,m,n))
                self.assertEqual(t.metadata.get("modeled_kv_write_bytes",0),0)
            append=[t for t in graph.tasks if t.metadata.get("event_kind")=="kv_append"]
            self.assertEqual(len(append),1)
            self.assertEqual(append[0].metadata["physical_bytes"],2*2*m*128)
            self.assertEqual(append[0].metadata["resource_accounting"],"native_cache_write_kernels")

    def test_f16_set_rows_cast_and_transposed_v_index_geometry(self):
        for flash in (False,True):
            for fusion in (False,True):
                graph=lower(apply_llama_gpu_invocation_contract(scenario(flash=flash),contract(fusion=fusion),enabled=True))
                writes=[t for t in graph.tasks if t.metadata.get("event_kind")=="kv_native_set_rows"
                        and t.metadata.get("phase")!="kernel_launch"]
                self.assertEqual(len(writes),1 if fusion else 2)
                v=next(t for t in writes if t.metadata["native_kv_work"]["stage"]=="v_set_rows")
                audit=v.metadata["native_kv_work"]
                self.assertEqual(audit["source_input_dtype"],"F32")
                self.assertEqual(audit["cache_output_dtype"],"F16")
                self.assertEqual(audit["source_value_bytes"],4*8*128)
                self.assertEqual(audit["persistent_write_bytes"],2*8*128)
                self.assertEqual(audit["index_unique_bytes"],8*8*(1 if flash else 128))
                self.assertEqual(audit["conversion_execution"],"inside_set_rows_no_extra_conversion_launch")
                self.assertFalse(audit["conversion_instructions_priced"])

    def test_explicit_cuda_fusion_disabled_keeps_m1_separate(self):
        graph=lower(apply_llama_gpu_invocation_contract(scenario(m=1),contract(fusion=False),enabled=True))
        self.assertEqual(len(projections(graph)),7)
        for t in projections(graph):
            if t.metadata["projection_id"] in ("mlp.gate","mlp.up"):
                self.assertFalse(t.metadata["fusion_enabled"])
                self.assertEqual(t.metadata["fusion_decision"],"captured_cuda_fusion_disabled")

    def test_unknown_environment_preserves_demands_with_uncovered_reason(self):
        base=scenario()
        candidate=apply_llama_gpu_invocation_contract(base,contract(fusion=None,status="uncovered"),enabled=True)
        self.assertEqual(signature(lower(base)),signature(lower(candidate)))
        self.assertFalse(candidate.workload.metadata[SOURCE_KEY]["applied"])
        self.assertIn("historical_cuda_fusion_environment_unknown",candidate.workload.metadata[SOURCE_KEY]["reasons"])

    def test_true_packed_qkv_stays_one_physical_matrix(self):
        def pack(meta):
            ps=meta["weight_projection_descriptors"]["projections"]
            segments=ps["attention.qkv"]["segments"]
            packed={**segments[0],"physical_tensor_name":"blk.0.attn_qkv.weight","n":sum(s["n"] for s in segments),
                    "physical_bytes":sum(s["physical_bytes"] for s in segments)}
            ps["attention.qkv"]["segments"]=[packed]
            meta["gguf_tensor_bindings"].append({"name":packed["physical_tensor_name"],"shape":[packed["k"],packed["n"]],
                "type":packed["format"],"n_bytes":packed["physical_bytes"],"offset":0})
        base=changed_layer(scenario(),pack)
        candidate=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        layer=planner._execution_layers(candidate)[0]
        self.assertNotIn("attention.q",layer.metadata["weight_projection_descriptors"]["projections"])
        tasks=projections(lower(candidate))
        qkv=next(t for t in tasks if t.metadata["projection_id"]=="attention.qkv")
        self.assertEqual(qkv.metadata["projection_segment_count"],1)
        self.assertFalse(qkv.metadata["fusion_enabled"])
        self.assertFalse(qkv.metadata.get("phase_metadata",{}).get("epilogue_name"))
        self.assertEqual(qkv.metadata["cost_model"]["output_bytes"],4*8*512)

    def test_weight_bytes_conserved_and_malformed_group_uncovered(self):
        base=scenario();candidate=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        def bytes_by_group(graph):
            out={"qkv":0,"ffn":0}
            for t in projections(graph):
                pid=t.metadata["projection_id"]
                key="qkv" if pid.startswith("attention.") and pid!="attention.output" else "ffn" if pid.startswith("mlp.") and pid!="mlp.down" else None
                if key:out[key]+=sum(s["local_physical_bytes"] for s in t.metadata["projection_segments"])
            return out
        self.assertEqual(bytes_by_group(lower(base)),bytes_by_group(lower(candidate)))
        def corrupt(meta):meta["gguf_tensor_bindings"][0]["shape"][0]+=256
        bad=apply_llama_gpu_invocation_contract(changed_layer(base,corrupt),contract(),enabled=True)
        layer=planner._execution_layers(bad)[0]
        self.assertFalse(layer.metadata[SOURCE_KEY]["groups"]["attention.qkv"]["applied"])
        self.assertEqual(layer.metadata[SOURCE_KEY]["groups"]["attention.qkv"]["reason"],"source_tensor_shape_or_format_mismatch")


    def test_reapply_preserves_gpu_alias_ownership_and_cpu_graph(self):
        base=scenario(m=8,legacy_cpu_flag=True)
        base=replace(base,workload=replace(base.workload,metadata={
            **base.workload.metadata,"llama_cpp_f32_hidden_storage":True}),
            placement=replace(base.placement,
                op_to_component={**{k:"cpu0" for k in base.placement.op_to_component},"full0.norm":"cpu0"},
                tensor_to_component={"kv_cache":"hostmem0","linear_state":"hostmem0"},
                kv_policy=replace(base.placement.kv_policy,cache_component="hostmem0")),
            llama_cpp_config=replace(base.llama_cpp_config,gpu_layers=0,op_offload=False))
        once=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        twice=apply_llama_gpu_invocation_contract(once,contract(),enabled=True)
        self.assertEqual(signature(lower(once)),signature(lower(twice)))
        layer=planner._execution_layers(twice)[0]
        self.assertEqual(layer.metadata[SOURCE_KEY]["generated_gpu_aliases"],
            ["attention.q","attention.k","attention.v"])
        self.assertFalse(planner._declared_physical_projections(twice,layer,
            ("attention.q","attention.k","attention.v"),combined_projection_id="attention.qkv",
            execution_component_id="cpu0"))

    def test_failed_group_is_not_counted_as_qualified_gpu_invocation(self):
        def corrupt(meta):meta["gguf_tensor_bindings"][0]["n_bytes"]+=1
        candidate=apply_llama_gpu_invocation_contract(changed_layer(scenario(m=64),corrupt),contract(),enabled=True)
        graph=lower(candidate)
        qkv=next(t for t in projections(graph) if t.metadata["projection_id"]=="attention.qkv")
        self.assertFalse(qkv.metadata["gpu_native_invocation"]["applied"])
        self.assertEqual(qkv.metadata["gpu_native_invocation"]["reason"],"source_tensor_storage_mismatch")
        summary=planner.summarize_gpu_invocations(graph.tasks)
        self.assertGreaterEqual(summary["uncovered_tasks"],1)
        self.assertIn("source_tensor_storage_mismatch",summary["uncovered_reason_counts"])

    def test_packed_v_flash_off_copy_and_unknown_layout_boundary(self):
        def pack(meta):
            ps=meta["weight_projection_descriptors"]["projections"]
            segments=ps["attention.qkv"]["segments"]
            packed={**segments[0],"physical_tensor_name":"blk.0.attn_qkv.weight",
                "n":sum(v["n"] for v in segments),"physical_bytes":sum(v["physical_bytes"] for v in segments)}
            ps["attention.qkv"]["segments"]=[packed]
            meta["gguf_tensor_bindings"].append({"name":packed["physical_tensor_name"],
                "shape":[packed["k"],packed["n"]],"type":packed["format"],
                "n_bytes":packed["physical_bytes"],"offset":0})
        for m in (1,8):
            for flash in (False,True):
                base=changed_layer(scenario(m=m,flash=flash),pack)
                graph=lower(apply_llama_gpu_invocation_contract(base,contract(),enabled=True))
                copies=[t for t in graph.tasks if t.metadata.get("event_kind")=="kv_native_v_contiguous"]
                self.assertEqual(bool(copies),not flash)
                if not flash:
                    launches=[t for t in copies if t.metadata.get("phase")=="kernel_launch"]
                    self.assertEqual(len(launches),int(m>1))
                    main=next(t for t in copies if t.metadata.get("phase")!="kernel_launch")
                    work=main.metadata["native_kv_work"]
                    self.assertEqual(work["read_bytes"],4*m*128)
                    self.assertEqual(work["write_bytes"],4*m*128)
                    self.assertEqual(work["persistent_write_bytes"],0)
                    v=next(t for t in graph.tasks if t.metadata.get("native_kv_work",{}).get("stage")=="v_set_rows")
                    self.assertIn(main.task_id,v.dependencies)
        old=contract();old.pop("packed_qkv_cache_layout");old["content_sha256"]=_hash({k:v for k,v in old.items() if k!="content_sha256"})
        graph=lower(apply_llama_gpu_invocation_contract(changed_layer(scenario(m=8),pack),old,enabled=True))
        self.assertFalse(any(t.metadata.get("event_kind")=="kv_native_v_contiguous" for t in graph.tasks))
        self.assertTrue(any(t.metadata.get("native_kv_work",{}).get("reason")=="packed_qkv_v_view_layout_not_proven" for t in graph.tasks))

    def test_q5k_boundary_is_six_independent_of_graph_treatment(self):
        self.assertEqual(MMVQ_MAX_BATCH_SIZE["Q5_K"],6)
        self.assertEqual(MMVQ_MAX_BATCH_SIZE["Q4_K"],5)
        self.assertEqual(MMVQ_MAX_BATCH_SIZE["Q6_K"],7)

    def test_q5k_m6_uses_mmvq_and_m7_can_enter_separate_mmq_treatment(self):
        device={"available":True,"sm_count":84,"max_shared_memory_per_block_optin_bytes":101376}
        for m,status in ((6,"mmvq_precedes_mmq"),(7,"applied")):
            with self.subTest(m=m):
                candidate=apply_llama_gpu_invocation_contract(scenario(m=m,fmt="Q5_K"),
                    contract(device=device),enabled=True,enable_mmq_source_costs=True)
                tasks=projections(lower(candidate))
                self.assertEqual(len(tasks),7)
                if m==6:
                    self.assertTrue(all(t.metadata.get("mmq_source_work",{}).get("status")==status for t in tasks))
                else:
                    by_projection={t.metadata["projection_id"]:t.metadata["mmq_source_work"] for t in tasks}
                    self.assertEqual(by_projection["mlp.gate"]["status"],"applied")
                    self.assertEqual(by_projection["attention.q"]["status"],"applied")
                    # The independent MMQ range proof may still reject the down
                    # projection; the dispatch threshold must not suppress it.
                    self.assertEqual(by_projection["mlp.down"]["status"],"uncovered")
                    self.assertIn("allocation high-water",by_projection["mlp.down"]["reason"])

    def test_mm_q_costs_require_separate_switch_and_properties(self):
        base=scenario(m=64)
        no=apply_llama_gpu_invocation_contract(base,contract(),enabled=True,enable_mmq_source_costs=True)
        self.assertNotIn("llama_cpp_mmq_source_work",no.workload.metadata)
        self.assertFalse(no.workload.metadata[SOURCE_KEY]["mmq_source_costs"]["applied"])
        device={"available":True,"sm_count":84,"max_shared_memory_per_block_optin_bytes":101376}
        separate=apply_llama_gpu_invocation_contract(base,contract(device=device),enabled=True)
        self.assertNotIn("llama_cpp_mmq_source_work",separate.workload.metadata)
        yes=apply_llama_gpu_invocation_contract(base,contract(device=device),enabled=True,enable_mmq_source_costs=True)
        self.assertTrue(yes.workload.metadata[SOURCE_KEY]["mmq_source_costs"]["applied"])
        graph=lower(yes)
        mains=projections(graph)
        self.assertTrue(all(t.metadata.get("mmq_source_work",{}).get("status")=="applied" for t in mains))

    def test_nested_model_identity_supported_but_conflict_fails_closed(self):
        base=scenario()
        nested=replace(base,model=replace(base.model,metadata={"metadata":{"gguf_sha256":"c"*64}}))
        result=apply_llama_gpu_invocation_contract(nested,contract(),enabled=True)
        self.assertTrue(result.workload.metadata[SOURCE_KEY]["applied"])
        self.assertEqual(result.model.graph.attributes,nested.model.graph.attributes)
        self.assertEqual(result.model.metadata,nested.model.metadata)
        conflict=replace(base,model=replace(base.model,metadata={"gguf_sha256":"a"*64,"metadata":{"gguf_sha256":"c"*64}}))
        result=apply_llama_gpu_invocation_contract(conflict,contract(),enabled=True)
        self.assertFalse(result.workload.metadata[SOURCE_KEY]["applied"])
        self.assertEqual(result.workload.metadata[SOURCE_KEY]["reasons"],["conflicting_flat_and_nested_model_identity"])

    def test_output_head_weight_bytes_and_dtype_are_preserved(self):
        base=scenario()
        layer=planner._execution_layers(base)[0]
        from heterollm_sim.ir import build_model_graph_from_layer_specs
        graph=build_model_graph_from_layer_specs(base.model.name,(layer,),architecture="qwen2",vocabulary_size=256,
            output_weight_bytes=123456,embedding_weight_bytes=98765,output_head_dtype="fp32",metadata=base.model.metadata)
        base=replace(base,model=replace(base.model,graph=graph))
        result=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        self.assertEqual(result.model.output_weight_bytes,123456)
        self.assertEqual(result.model.embedding_weight_bytes,98765)
        self.assertEqual(next(t.dtype for t in result.model.graph.tensors if t.tensor_id=="logits"),"fp32")

    def test_contract_mutation_rejected_and_model_name_does_not_qualify(self):
        bad=contract();bad["cuda_compute_capability"]=900
        with self.assertRaises(ValueError):apply_llama_gpu_invocation_contract(scenario(),bad,enabled=True)
        base=scenario();base=replace(base,model=replace(base.model,metadata={}))
        candidate=apply_llama_gpu_invocation_contract(base,contract(),enabled=True)
        self.assertEqual(candidate.workload.metadata[SOURCE_KEY]["reasons"],["source_model_identity_missing"])


class GPUInvocationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=next((p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").is_file()),None)
        if root is None:raise unittest.SkipTest("project source evidence unavailable")
        directory=root/"artifacts/development/native_long_grid_135_20260915"
        path=directory/"optimization_loop/round_004/runtime_source_binding_structural_audit.json"
        if not path.is_file():raise unittest.SkipTest("recorded R4 build evidence not included in this checkout")
        cls.binding=json.loads(path.read_text(encoding="utf-8"))["build_binding"]
        cls.property_path=directory/"optimization_loop/operator_microbench_v2/driver_device_properties.json"
        cls.hardware_path=directory/"hardware.json"

    def derive(self,environment=None,cc=1200,probe=None):
        return derive_llama_gpu_invocation_contract(self.binding,
            captured_kernel_environment={"GGML_CUDA_DISABLE_FUSION":None} if environment is None else environment,
            cuda_compute_capability=cc,mmq_device_evidence=probe)

    def test_recorded_source_rule_remains_conditional(self):
        result=self.derive()
        self.assertEqual(result["status"],"conditional")
        self.assertFalse(result["native_dispatch_proven"])
        self.assertFalse(result["mmq_device_evidence"]["available"])
        self.assertGreaterEqual(len(result["source_refs"]),10)

    def test_fusion_absence_numeric_zero_presence_and_unknown(self):
        for value,expected in ((None,True),("0",True),("1",False),("-2",False)):
            with self.subTest(value=value):
                result=self.derive({"GGML_CUDA_DISABLE_FUSION":value})
                self.assertEqual(result["status"],"conditional")
                self.assertEqual(result["fusion_enabled"],expected)
        self.assertEqual(self.derive({})["status"],"uncovered")
        self.assertEqual(self.derive({"GGML_CUDA_DISABLE_FUSION":"maybe"})["status"],"uncovered")

    def test_mutated_runtime_binding_and_mismatched_architecture_rejected(self):
        binding=deepcopy(self.binding);binding["status"]="not_verified"
        with self.assertRaises(ValueError):derive_llama_gpu_invocation_contract(binding,
            captured_kernel_environment={"GGML_CUDA_DISABLE_FUSION":None},cuda_compute_capability=1200)
        with self.assertRaises(ValueError):self.derive(cc=900)

    def test_device_probe_sha_uuid_and_architecture_bound_before_optional_mmq(self):
        if not self.property_path.is_file() or not self.hardware_path.is_file():
            self.skipTest("requires both captured static device probe and local native hardware record")
        def ref(p):return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
        evidence={"source_ref":ref(self.property_path),"selected_hardware_ref":ref(self.hardware_path)}
        result=self.derive(probe=evidence)
        self.assertTrue(result["mmq_device_evidence"]["available"])
        self.assertEqual(result["mmq_device_evidence"]["max_shared_memory_per_block_optin_bytes"],101376)
        self.assertEqual(result["mmq_device_evidence"]["gpu_uuid"],json.loads(self.hardware_path.read_text(encoding="utf-8"))["gpu"]["uuid"])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"wrong.json";doc=json.loads(self.property_path.read_text(encoding="utf-8"));doc["gpu_uuid"]="GPU-wrong"
            p.write_text(json.dumps(doc),encoding="utf-8")
            with self.assertRaises(ValueError):self.derive(probe={**evidence,"source_ref":ref(p)})
        bad=deepcopy(evidence);bad["source_ref"]["sha256"]="0"*64
        with self.assertRaises(ValueError):self.derive(probe=bad)


if __name__=="__main__":unittest.main()
