"""Explicit source-qualified GPU invocation geometry; disabled unless applied.

Physical tensor aliases live only in the scenario returned by apply(). No
GGUF importer defaults, model-name rules, empirical rates or native execution
are involved. Historical model-specific unity source bodies are not all
captured, so every admitted contract remains conditional.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .ir import build_model_graph_from_layer_specs, model_graph_execution_layers, model_graph_execution_view
from .projection_descriptors import ARTIFACT_QUANTIZATION_REGISTRY
from .runtime_adapters import _source_function

SOURCE_KEY = "llama_cpp_gpu_native_invocations"
SCHEMA = "llama.cpp.gpu-native-invocations/v1"
_FUSION_ENV = "GGML_CUDA_DISABLE_FUSION"


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest()


def _path_key(value):
    return str(Path(value).resolve()).replace("\\", "/").casefold()


def _checked_text(path, expected, refs):
    p=Path(path).resolve()
    if p.suffix.lower() in {".dll",".exe",".gguf",".obj",".lib",".pch"}:
        raise ValueError("compiled artifacts must remain recorded identities")
    if p.stat().st_size > 32*1024*1024:
        raise ValueError("source evidence exceeds bounded text size")
    raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError("GPU invocation source identity mismatch: "+str(p))
    refs[_path_key(p)]={"path":str(p),"sha256":digest,"size_bytes":len(raw)}
    return raw.decode("utf-8-sig")


def _unique_digest(mapping, path):
    vals=[v for k,v in mapping.items() if _path_key(k)==_path_key(path)]
    if len(vals)!=1:
        raise ValueError("source history missing or ambiguous: "+str(path))
    return vals[0]


def _historical_sources(binding, refs):
    """Recheck extra source bodies against R4's digest-bound base receipt."""
    base_receipts=[]; annotation=[]; headers=[]
    for ref in binding.get("evidence_refs",()):
        name=Path(ref["path"]).name
        if name.endswith(".json") and ("receipt" in name or name=="header_snapshot.json"):
            document=json.loads(_checked_text(ref["path"],ref["sha256"],refs))
            if "source_sha256_before" in document:base_receipts.append(document)
            if "old_link_inputs_postverified" in document:annotation.append(document)
            if "files" in document and "header" in name:headers.append(document)
    if len(base_receipts)!=1 or len(annotation)!=1 or len(headers)!=1:
        raise ValueError("R4 binding requires unique original-source, annotation and header receipts")
    base,overlay,header=base_receipts[0],annotation[0],headers[0]
    if base.get("returncode")!=0 or base.get("source_unchanged") is not True or overlay.get("old_link_inputs_postverified") is not True:
        raise ValueError("original source or inherited link inputs were not verified")
    root=Path(binding["source_paths"]["scheduler"]).parents[2]
    wanted={"graph":"src/llama-graph.cpp","model":"src/llama-model.cpp", "kv_cache":"src/llama-kv-cache.cpp",
            "set_rows":"ggml/src/ggml-cuda/set-rows.cu", "mmvq":"ggml/src/ggml-cuda/mmvq.cu", "mmq":"ggml/src/ggml-cuda/mmq.cu",
            "copy":"ggml/src/ggml-cuda/cpy.cu", "quantize":"ggml/src/ggml-cuda/quantize.cu"}
    texts={}
    for role,relative in wanted.items():
        path=root/relative
        before=_unique_digest(base["source_sha256_before"],path)
        if before!=_unique_digest(base["source_sha256_after"],path):
            raise ValueError("historical source changed during build: "+relative)
        texts[role]=_checked_text(path,before,refs)
    for role,relative in (("mmvq_header","ggml/src/ggml-cuda/mmvq.cuh"),("cuda_common","ggml/src/ggml-cuda/common.cuh"),
                          ("mmq_header","ggml/src/ggml-cuda/mmq.cuh"),
                          ("mmq_load_tiles","ggml/src/ggml-cuda/mmq-load-tiles.cuh")):
        path=root/relative
        texts[role]=_checked_text(path,_unique_digest(header["files"],path),refs)
    return texts


def derive_llama_gpu_invocation_contract(
    runtime_binding: Mapping[str,Any], *,
    captured_kernel_environment: Mapping[str,str|None],
    cuda_compute_capability: int,
    mmq_device_evidence: Mapping[str,Any]|None=None,
) -> dict[str,Any]:
    """Derive from a verified R4 build binding and captured kernel controls.

    cuda_compute_capability uses llama.cpp units, e.g. 1200 for CUDA 12.0.
    mmq_device_evidence is optional and separate from invocation geometry.
    Missing or unknown controls remain uncovered; this function never reads
    today's environment or calls any native executable/device API.
    """
    if not isinstance(runtime_binding,Mapping):raise ValueError("R4 runtime binding must be a mapping")
    binding=dict(runtime_binding);claimed=binding.pop("content_sha256",None)
    if binding.get("schema")!="llama-recorded-runtime-source-binding/v1" or binding.get("status")!="verified_build_chain" or claimed!=_hash(binding):
        raise ValueError("R4 runtime source binding is unverified or mutated")
    if type(cuda_compute_capability) is not int or cuda_compute_capability//10 not in binding.get("compiled_cuda_architectures",()):
        raise ValueError("captured CUDA capability does not match compiled architecture")
    if not isinstance(captured_kernel_environment,Mapping):raise ValueError("kernel environment must be captured explicitly")
    refs={}
    source={}
    for role in ("cuda","operators"):
        path=binding["source_paths"][role]
        expected=_unique_digest(binding["source_contract"]["source_sha256"],path)
        source[role]=_checked_text(path,expected,refs)
    source.update(_historical_sources(binding,refs))
    cuda=source["cuda"]
    vector_fusion=_source_function(cuda,"static bool ggml_cuda_should_fuse_mul_mat_vec_q(")
    fusion_requirements=("src1->type == GGML_TYPE_F32","dst->type == GGML_TYPE_F32",
                         "dst->ne[1] != 1","cc <= GGML_CUDA_CC_PASCAL","bad_padding_clear")
    if any(v not in vector_fusion for v in fusion_requirements):raise ValueError("quantized M1 fusion rule is not recognized")
    graph=source["graph"]
    ffn=_source_function(graph,"ggml_tensor * llm_graph_context::build_ffn(")
    if "build_lora_mm(up, cur)" not in ffn or "build_lora_mm(gate, cur)" not in ffn:
        raise ValueError("separate physical gate/up projection source not recognized")
    if any(v not in graph for v in ("if (layer.wqkv)","build_lora_mm(layer.wqkv, cur", "build_lora_mm(layer.wq, cur", "build_lora_mm(layer.wk, cur", "build_lora_mm(layer.wv, cur")):
        raise ValueError("packed versus separate physical QKV source not recognized")
    mul_mat=_source_function(source["operators"],"struct ggml_tensor * ggml_mul_mat(")
    if "ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)" not in mul_mat:raise ValueError("native GEMM F32 output is not proven")
    cache=source["kv_cache"]
    if any(v not in cache for v in ("return ggml_set_rows(ctx, k, k_cur, k_idxs);","if (!v_trans)",
            "ggml_reshape_2d(ctx, v, 1, ggml_nelements(v))","ggml_cont_2d", "return ggml_set_rows(ctx, v_view, v_cur, v_idxs);")):
        raise ValueError("native K/V cache layout and writes are not recognized")
    if "!cparams.flash_attn" not in source["model"] or "dst->type == GGML_TYPE_F16" not in source["set_rows"] or "set_rows_cuda<float, int64_t>" not in source["set_rows"]:
        raise ValueError("F32 SET_ROWS to F16 or V-transpose source missing")
    if any(v not in graph for v in (
            "Vcur = ggml_view_3d(ctx0, qkv", "ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1]")):
        raise ValueError("packed QKV token-stride source is not recognized")
    copy = source["copy"]
    if any(v not in copy for v in (
            "void ggml_cuda_dup(", "ggml_cuda_cpy(ctx, src0, dst)",
            "const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1)",
            "src0->type == src1->type && contiguous_srcs", "cudaMemcpyDeviceToDevice",
            "!ggml_are_same_shape(src0, src1)", "ggml_cpy_scalar_cuda<float, float>")):
        raise ValueError("packed V contiguous-copy dispatch source is not recognized")
    rope=_source_function(cuda,"static bool ggml_cuda_should_fuse_rope_set_rows(")
    if any(v not in rope for v in ("set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16",
                                  "set_rows->src[1]->type != GGML_TYPE_I64","GGML_ROPE_TYPE_NORMAL","GGML_ROPE_TYPE_NEOX")):
        raise ValueError("cache-write RoPE fusion boundary not recognized")
    model_rope=_source_function(source["model"],"llama_rope_type llama_model_rope_type(")
    rope_modes={}
    for ir_arch,native_arch,expected in (("llama","LLAMA","NORM"),("qwen2","QWEN2","NEOX"),("qwen3_5_hybrid_transformer","QWEN35","IMROPE")):
        match=re.search(r"case LLM_ARCH_"+native_arch+r":.*?return LLAMA_ROPE_TYPE_([A-Z_]+);",model_rope,re.S)
        if match is None or match[1]!=expected:raise ValueError("fixed architecture RoPE rule changed")
        rope_modes[ir_arch]={"NORM":"normal","NEOX":"neox","IMROPE":"imrope"}[expected]
    env_present=_FUSION_ENV in captured_kernel_environment
    env_value=captured_kernel_environment.get(_FUSION_ENV)
    reasons=[];fusion_enabled=None
    if not env_present:
        reasons.append("historical_cuda_fusion_environment_unknown")
    elif env_value is None:
        fusion_enabled=True
    elif isinstance(env_value,str) and re.fullmatch(r"[+-]?\d+",env_value.strip()):
        fusion_enabled=int(env_value.strip())==0
    else:
        reasons.append("unsupported_cuda_fusion_environment_value")
    if 'getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi' not in cuda:
        raise ValueError("CUDA fusion environment semantics changed")
    mmq={"available":False,"reason":"captured_mmq_device_properties_not_supplied"}
    if mmq_device_evidence is not None:
        raw=dict(mmq_device_evidence)
        if not all(isinstance(raw.get(key), Mapping) for key in ("source_ref", "selected_hardware_ref")):
            raise ValueError("MMQ device evidence requires probe and selected hardware SHA references")
        property_ref, hardware_ref = raw["source_ref"], raw["selected_hardware_ref"]
        document=json.loads(_checked_text(property_ref["path"],property_ref["sha256"],refs))
        hardware=json.loads(_checked_text(hardware_ref["path"],hardware_ref["sha256"],refs))
        if document.get("schema") != "cuda-driver-static-device-properties/v1" or document.get("native_configuration_modified") is not False:
            raise ValueError("unsupported CUDA property probe")
        attrs=document.get("attributes",{})
        major=attrs.get("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR")
        minor=attrs.get("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR")
        sm=attrs.get("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT")
        shared=attrs.get("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
        warp=attrs.get("CU_DEVICE_ATTRIBUTE_WARP_SIZE")
        gpu=hardware.get("gpu",{})
        if (any(type(v) is not int for v in (major,minor,sm,shared,warp))
                or 100*major+10*minor != cuda_compute_capability
                or gpu.get("compute_capability") != str(major)+"."+str(minor)
                or not document.get("gpu_uuid") or document["gpu_uuid"] != gpu.get("uuid")
                or sm < 1 or shared < 49152 or warp != 32):
            raise ValueError("MMQ probe SHA/UUID/architecture does not match the selected native hardware")
        # Force flags belong to the actual MMQ compile unit, not to the driver
        # properties or a global peak-rate declaration.
        compile_refs=[r for r in binding.get("evidence_refs",()) if Path(r["path"]).name=="compile_commands.json"]
        if len(compile_refs)!=1:raise ValueError("bound MMQ compile commands are missing")
        commands=json.loads(_checked_text(compile_refs[0]["path"],compile_refs[0]["sha256"],refs))
        units=[r for r in commands if str(r.get("file", "")).replace("\\", "/").endswith("ggml-cuda/mmq.cu")]
        if len(units)!=1:raise ValueError("bound MMQ translation unit is missing or ambiguous")
        flags=str(units[0].get("command", " ".join(units[0].get("arguments",[]))))
        if re.search(r"(?:-D|/D)\s*GGML_CUDA_FORCE_CUBLAS\b",flags):
            raise ValueError("compiled MMQ source is forced to cuBLAS")
        if cuda_compute_capability != 1200:
            raise ValueError("fixed source MMQ work only covers captured Blackwell 1200")
        mmq={"available":True,"reason":None,"source_ref":dict(property_ref),
             "selected_hardware_ref":dict(hardware_ref),"gpu_uuid":document["gpu_uuid"],
             "sm_count":sm,"max_shared_memory_per_block_optin_bytes":shared,
             "cuda_compute_capability":cuda_compute_capability,"warp_size":warp,
             "cache_properties_recorded_not_applied":{"l2_bytes":attrs.get("CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"),
                 "per_sm_shared_bytes":attrs.get("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR")}}
    # The tail extension depends on logical-block scheduling and a final
    # tensor buffer pad, not a per-row stride change. Pin all source bodies.
    tail_requirements = {
        "mmq": ("GGML_PAD(ne10, MATRIX_ROW_PADDING)", "ne00, ne01"),
        "mmq_header": ("args.ncols_x / ggml_cuda_type_traits<type>::qk",
                       "fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter",
                       "kb0 < kb0_stop"),
        "quantize": ("i0 < ne00",),
        "cuda_common": ("#define MATRIX_ROW_PADDING 512",),
        "cuda": ("size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)",),
    }
    for role, literals in tail_requirements.items():
        if any(literal not in source[role] for literal in literals):
            raise ValueError("MMQ tail source rule not recognized: " + role)
    result={"schema":SCHEMA,"status":"conditional" if not reasons else "uncovered",
            "runtime_binding_sha256":claimed,"runtime_modules":binding["runtime_modules"],
            "cuda_compute_capability":cuda_compute_capability,
            "kernel_environment":{_FUSION_ENV:{"captured":env_present,"value":env_value}},
            "fusion_enabled":fusion_enabled,"minimum_quantized_fusion_m":1,"maximum_quantized_fusion_m":1,
            "architecture_rope_modes":rope_modes,"native_matmul_output_dtype":"F32","cache_write_dtype":"F16",
            "packed_qkv_cache_layout": {
                "schema":"llama.cpp.packed-qkv-reshape-v1", "matmul_output_dtype":"F32",
                "view":"reshape_3d_with_packed_token_stride", "copy_required_when_flash_off":True,
                "strided_copy":"f32_scalar_kernel", "single_token_copy":"cuda_memcpy_d2d",
            },
            "source_rule":"physical projection boundaries; quantized M1 FFN fusion only; F32 outputs then SET_ROWS casts",
            "mmq_reduction_tail_contract":"logical-k-streamk-tail/v1",
            "mmq_device_evidence":mmq,"source_refs":list(refs.values()),"uncovered_reasons":reasons,
            "native_dispatch_proven":False,"accuracy_validated":False,
            "conditional_reasons":["model-specific unity include historical source-body hashes were not captured; caller graph remains conditional"],
            "unpriced_terms":["native cache conversion instruction throughput","exact gather/scatter index transaction multiplicity","vector-kernel main throughput without an independent MMVQ cost treatment"]}
    result["content_sha256"]=_hash(result)
    return result


def _matrix_evidence(segment, bindings):
    if not isinstance(segment,Mapping):return None,"malformed_physical_segment"
    name=segment.get("physical_tensor_name")
    matches=bindings.get(name,())
    if len(matches)!=1:return None,"physical_tensor_binding_missing_or_ambiguous"
    binding=matches[0]; k,n=segment.get("k"),segment.get("n");fmt=str(segment.get("format","")).upper()
    spec=ARTIFACT_QUANTIZATION_REGISTRY.get(fmt)
    if spec is None:return None,"physical_quantization_not_supported"
    if type(k) is not int or type(n) is not int or k<1 or n<1 or k%spec.block_size:
        return None,"physical_matrix_dimensions_not_proven"
    if not isinstance(binding.get("shape"),(tuple,list)) or tuple(binding["shape"])!=(k,n) or str(binding.get("type","")).upper()!=fmt:
        return None,"source_tensor_shape_or_format_mismatch"
    size=n*(k//spec.block_size)*(spec.payload_bytes+spec.metadata_bytes)
    if segment.get("physical_bytes")!=size or binding.get("n_bytes")!=size or type(binding.get("offset")) is not int or binding["offset"]<0:
        return None,"source_tensor_storage_mismatch"
    return {"name":name,"k":k,"n":n,"format":fmt,"physical_bytes":size},None


def _layer_qualification(layer, contract):
    metadata=deepcopy(layer.metadata)
    projections=metadata.get("weight_projection_descriptors",{}).get("projections",{})
    bindings={}
    for binding in metadata.get("gguf_tensor_bindings",()):
        if isinstance(binding,Mapping):bindings.setdefault(binding.get("name"),[]).append(binding)
    previous = metadata.get(SOURCE_KEY, {})
    previous_aliases = previous.get("generated_gpu_aliases", ()) if isinstance(previous, Mapping) else ()
    if not isinstance(previous_aliases, (tuple, list)) or any(
            key not in ("attention.q", "attention.k", "attention.v") for key in previous_aliases):
        raise ValueError("generated GPU projection alias provenance is malformed")
    # Ownership survives requalification, including a subsequent failed group.
    # Existing GPU-created aliases must never become authored CPU capabilities.
    groups={};generated=list(dict.fromkeys(previous_aliases))
    def group(name, segments, expected):
        if not isinstance(segments,(tuple,list)) or len(segments)!=expected:
            return {"applied":False,"reason":"physical_projection_group_missing"}
        facts=[]
        for segment in segments:
            fact,reason=_matrix_evidence(segment,bindings)
            if reason:return {"applied":False,"reason":reason}
            facts.append(fact)
        if len({v["name"] for v in facts})!=len(facts):
            return {"applied":False,"reason":"group_repeats_a_physical_tensor"}
        return {"applied":True,"reason":None,"physical_matrices":facts,"group":name}
    qkv=projections.get("attention.qkv",{}).get("segments",())
    if len(qkv) in (1,3):
        groups["attention.qkv"]=group("attention.qkv",qkv,len(qkv))
        info=groups["attention.qkv"]
        if info["applied"]:
            info["packed_qkv"]=len(qkv)==1
            layout = contract.get("packed_qkv_cache_layout", {})
            info["packed_qkv_cache_layout"] = dict(layout) if isinstance(layout, Mapping) else {}
            if len(qkv)==3:
                for suffix,segment in zip(("q","k","v"),qkv):
                    key="attention."+suffix
                    if key in projections and projections[key].get("segments")!=[segment]:
                        info.update(applied=False,reason="existing_qkv_alias_disagrees_with_physical_tensor")
                        break
                if info["applied"]:
                    for suffix,segment in zip(("q","k","v"),qkv):
                        key="attention."+suffix
                        if key not in projections:
                            projections[key]={"segments":[dict(segment)],"alias_of":"attention.qkv"}
                            if key not in generated:generated.append(key)
    else:groups["attention.qkv"]={"applied":False,"reason":"physical_qkv_layout_not_covered"}
    segments=projections.get("mlp.up_gate",{}).get("segments",())
    groups["mlp.up_gate"]=group("mlp.up_gate",segments,2)
    if groups["mlp.up_gate"]["applied"]:
        facts=groups["mlp.up_gate"]["physical_matrices"]
        aliases=[projections.get("mlp."+suffix,{}).get("segments") for suffix in ("gate","up")]
        if aliases!=[[segments[0]],[segments[1]]]:
            groups["mlp.up_gate"].update(applied=False,reason="physical_gate_up_aliases_not_proven")
        groups["mlp.up_gate"]["fusion_extra_inputs_present"] = any(
            isinstance(name,str) and ("ffn_gate" in name or "ffn_up" in name)
            and not name.endswith(".weight") for name in bindings)
        groups["mlp.up_gate"]["fusion_shape_compatible"]=(
            facts[0]["k"],facts[0]["n"],facts[0]["format"]
        )==(facts[1]["k"],facts[1]["n"],facts[1]["format"])
    controls=[]
    for suffix in ("alpha","beta"):
        controls.extend(projections.get("linear_attention."+suffix,{}).get("segments",()))
    groups["linear_attention.controls"]=group("linear_attention.controls",controls,2)
    # A single stored linear QKV (and a separate Q+gate physical matrix) is
    # retained as imported; no alias of an individual tensor becomes a GEMM.
    parent_groups = {
        **{key:"attention.qkv" for key in ("attention.qkv", "attention.q", "attention.k", "attention.v")},
        **{key:"mlp.up_gate" for key in ("mlp.up_gate", "mlp.gate", "mlp.up")},
        **{key:"linear_attention.controls" for key in ("linear_attention.alpha", "linear_attention.beta")},
    }
    projected = {}
    supported = set(parent_groups) | {"attention.output", "mlp.down", "linear_attention.qkv",
                                      "linear_attention.output", "linear_attention.output_gate"}
    for key in supported & projections.keys():
        parent = groups.get(parent_groups.get(key, ""))
        if parent is not None and parent.get("applied") is not True:
            projected[key] = {**parent, "group":parent_groups[key]}
            continue
        expected = len(qkv) if key == "attention.qkv" else 2 if key == "mlp.up_gate" else 1
        projected[key] = group(key, projections[key].get("segments", ()), expected)
        projected[key]["group"] = parent_groups.get(key, key)
    metadata[SOURCE_KEY]={"groups":groups,"projections":projected,"generated_gpu_aliases":generated,
                          "native_dispatch_proven":False,"conditional":True}
    return replace(layer,metadata=metadata),groups


def apply_llama_gpu_invocation_contract(
    scenario, contract: Mapping[str,Any]|None=None, *,
    enabled: bool=False, enable_mmq_source_costs: bool=False,
):
    """Return an explicitly qualified scenario; default/no-contract is identity.

    Applying a source contract is conditional while historical model caller
    bytes remain unproven. MMQ pricing is an independent optional treatment.
    Existing scenarios are not undone by calling this function with disabled.
    The caller must run its standard final placement replan after composition.
    """
    if type(enabled) is not bool or type(enable_mmq_source_costs) is not bool:
        raise ValueError("GPU invocation treatment switches must be explicit booleans")
    if not enabled or contract is None:return scenario
    if not isinstance(contract,Mapping):raise ValueError("GPU invocation contract must be a mapping")
    payload=dict(contract);claimed=payload.pop("content_sha256",None)
    if payload.get("schema")!=SCHEMA or claimed!=_hash(payload):raise ValueError("GPU invocation contract is invalid or mutated")
    audit={"schema":SCHEMA,"applied":False,"status":"uncovered","contract_sha256":claimed,
           "native_dispatch_proven":False,"conditional":True,
           "reasons":list(payload.get("uncovered_reasons",())),
           "mmq_source_costs":{"requested":enable_mmq_source_costs,"applied":False}}
    def uncovered(reason):
        flags=dict(scenario.workload.metadata)
        flags[SOURCE_KEY]={**audit,"reasons":[*audit["reasons"],reason]}
        return replace(scenario,workload=replace(scenario.workload,metadata=flags))
    if payload.get("status")!="conditional" or payload.get("fusion_enabled") not in (True,False):
        return uncovered("captured_kernel_fusion_control_required")
    model=scenario.model
    model_sha=model.metadata.get("gguf_sha256")
    nested=model.metadata.get("metadata",{})
    nested_sha=nested.get("gguf_sha256") if isinstance(nested,Mapping) else None
    if model_sha is not None and nested_sha is not None and str(model_sha).lower()!=str(nested_sha).lower():
        return uncovered("conflicting_flat_and_nested_model_identity")
    if model_sha is None:model_sha=nested_sha
    if not isinstance(model_sha,str) or re.fullmatch(r"[0-9a-fA-F]{64}",model_sha) is None:
        return uncovered("source_model_identity_missing")
    rope_mode=payload.get("architecture_rope_modes",{}).get(model.architecture)
    if rope_mode is None:return uncovered("model_architecture_source_not_covered")
    layers=model_graph_execution_layers(model.graph,schema_version=model.schema_version)
    view=model_graph_execution_view(model.graph,schema_version=model.schema_version)
    if scenario.workload.mtp is not None or view.mtp_descriptors or any(l.kind!="dense" or l.shared_expert_intermediate_size for l in layers):
        return uncovered("experts_or_mtp_outside_dense_invocation_contract")
    parallel=scenario.placement.parallel
    if any(getattr(parallel,k)!=1 for k in ("tp_degree","pp_degree","ep_degree")):
        return uncovered("sharded_or_multirank_layout_not_proven")
    gpu=[c for c in scenario.hardware.components if str(c.kind).lower()=="gpu"]
    if len(gpu)!=1:return uncovered("one_captured_gpu_required")
    cc=payload["cuda_compute_capability"]
    if gpu[0].metadata.get("cuda_compute_capability") not in (None,cc):
        return uncovered("component_cuda_capability_conflicts_with_capture")
    qualified_layers=[];reason_counts={};applied_groups=0
    for layer in layers:
        qualified,groups=_layer_qualification(layer,payload);qualified_layers.append(qualified)
        for group in groups.values():
            if group.get("applied"):applied_groups+=1
            else:
                reason=group.get("reason","uncovered")
                reason_counts[reason]=reason_counts.get(reason,0)+1
    if not applied_groups:return uncovered("no_source_qualified_physical_projection_groups")
    graph=build_model_graph_from_layer_specs(model.name,qualified_layers,architecture=model.architecture,
        vocabulary_size=model.vocabulary_size,max_sequence_length=model.max_sequence_length,
        embedding_weight_bytes=model.embedding_weight_bytes,output_weight_bytes=model.output_weight_bytes,
        output_head_dtype=next((t.dtype for t in view.tensors if t.tensor_id == "logits"), None),
        metadata=model.graph.attributes.get("metadata", model.metadata))
    # Keep the original model identity and graph-level source declarations at
    # their existing nesting depth; only per-layer aliases/qualification change.
    graph=replace(graph, attributes=dict(model.graph.attributes))
    model=replace(model,graph=graph)
    runtime=scenario.llama_cpp_config
    cache_qualified=(runtime is not None and type(runtime.flash_attn) is bool and runtime.kv_type_k=="f16"
                     and runtime.kv_type_v=="f16" and runtime.kv_unified is True)
    audit.update(applied=True,status="conditional",reasons=[],gpu_component_ids=[gpu[0].component_id],
                 model_sha256=model_sha,fusion_enabled=payload["fusion_enabled"],rope_type=rope_mode,
                 source_refs=payload["source_refs"],qualified_projection_groups=applied_groups,
                 uncovered_group_reason_counts=reason_counts,
                 cache_write_qualified=cache_qualified,
                 cache_write_reason=None if cache_qualified else "captured_unified_f16_cache_runtime_required",
                 v_cache_transposed=(not runtime.flash_attn) if cache_qualified else None,
                 unpriced_terms=payload["unpriced_terms"])
    flags={**scenario.workload.metadata,SOURCE_KEY:audit,"llama_cpp_f32_hidden_storage":True}
    component_metadata={**gpu[0].metadata,"cuda_compute_capability":cc}
    if enable_mmq_source_costs:
        device=payload.get("mmq_device_evidence",{})
        profile=scenario.resolve_component_profile(gpu[0])
        if device.get("available") is True and device.get("sm_count")==profile.tensor_core.sm_count:
            component_metadata["llama_cpp_mmq_contract"]={"backend_commit":"0f3a71be15af836d277c9f918adfafb45732677e",
                "compiled_int8_mma":True,"force_cublas":False,"ordinary_contiguous_2d":True,
                "max_shared_memory_per_block_optin_bytes":device["max_shared_memory_per_block_optin_bytes"],
                "reduction_tail_contract":payload.get("mmq_reduction_tail_contract")}
            flags.update(llama_cpp_mmq_source_work=True,llama_cpp_f32_q8_1_mmvq=True)
            audit["mmq_source_costs"].update(applied=True,reason=None)
        else:audit["mmq_source_costs"].update(reason="source_mmq_device_properties_missing_or_profile_mismatch")
    hardware=replace(scenario.hardware,components=tuple(
        replace(c,metadata=component_metadata) if c.component_id==gpu[0].component_id else c
        for c in scenario.hardware.components))
    result=replace(scenario,model=model,hardware=hardware,workload=replace(scenario.workload,metadata=flags))
    # The wrapper performs its normal final placement replan after all source
    # treatments are composed. Do not recompute or fabricate a fingerprint here.
    return result


def gpu_invocation_group(scenario,layer,target_component_id,group):
    """Planner-only lookup; generated aliases never authorize a CPU call."""
    audit=scenario.workload.metadata.get(SOURCE_KEY,{})
    if not isinstance(audit,Mapping) or audit.get("applied") is not True or target_component_id not in audit.get("gpu_component_ids",()):return {}
    layer_audit=layer.metadata.get(SOURCE_KEY,{})
    value=layer_audit.get("groups",{}).get(group,{}) if isinstance(layer_audit,Mapping) else {}
    if not isinstance(value,Mapping) or value.get("applied") is not True:return {}
    return {**audit,**value}


def gpu_projection_invocation_audit(scenario, layer, target, projection_id, *, m, k, n):
    """A per-call diagnostic; a scenario label alone is not projection coverage."""
    active = scenario.workload.metadata.get(SOURCE_KEY, {})
    if (not isinstance(active, Mapping) or active.get("applied") is not True
            or target not in active.get("gpu_component_ids", ())):
        return {}
    declared = layer.metadata.get(SOURCE_KEY, {}) if layer is not None else {}
    proof = declared.get("projections", {}).get(projection_id, {}) if isinstance(declared, Mapping) else {}
    facts = proof.get("physical_matrices", ())
    applied = proof.get("applied") is True
    reason = proof.get("reason", "physical_projection_group_not_qualified")
    if applied and (any(f["k"] != k for f in facts) or sum(f["n"] for f in facts) != n):
        applied, reason = False, "physical_projection_call_shape_mismatch"
    return {"schema":active["schema"], "status":"conditional" if applied else "uncovered",
            "applied":applied, "reason":None if applied else reason,
            "group":proof.get("group", projection_id or "unqualified_weight_projection"),
            "projection_id":projection_id, "m":m, "k":k, "n":n,
            "physical_weight_matrices":len(facts) if applied else 0,
            "scope":"per_projection_source_tensor_and_call_shape", "native_dispatch_proven":False,
            "mmq_source_costs_requested":active["mmq_source_costs"]["requested"]}


__all__=["SOURCE_KEY","SCHEMA","derive_llama_gpu_invocation_contract","apply_llama_gpu_invocation_contract"]
