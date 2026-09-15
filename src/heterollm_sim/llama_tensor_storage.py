"""Opt-in source-qualified ordinary GGUF tensor storage and GET_ROWS traffic.

This module contains no latency values or conversion throughput. It separates
full embedding capacity/staging from selected-row gather traffic, and records
unpriced row conversion work instead of assigning an invented kernel speed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from .ir import model_graph_execution_view
from .runtime_adapters import _cpp_source_function

SCHEMA = 'llama.cpp.gguf.tensor-storage/v1'
SOURCE_KEY = 'llama_cpp_tensor_storage_contract'
AUDIT_KEY = 'llama_cpp_tensor_storage'
_ARCHITECTURES = {'qwen2', 'qwen2_decoder', 'llama', 'llama_decoder', 'qwen3_5_hybrid_transformer'}


def derive_llama_tensor_storage_contract(source_root: str | Path) -> dict[str, Any]:
    """Read the locked source rules; the caller binds these files to its build."""
    root = Path(source_root)
    paths = {'operators': root/'ggml/src/ggml.c', 'cpu_get_rows': root/'ggml/src/ggml-cpu/ops.cpp',
             'cuda_get_rows': root/'ggml/src/ggml-cuda/getrows.cu', 'input_graph': root/'src/llama-graph.cpp'}
    raw = {k: p.read_bytes() for k, p in paths.items()}
    source = {k: v.decode('utf-8') for k, v in raw.items()}
    operators = source['operators']
    get_rows = _cpp_source_function(operators, 'struct ggml_tensor * ggml_get_rows(')
    mul_mat = _cpp_source_function(operators, 'struct ggml_tensor * ggml_mul_mat(')
    cpu_rows = _cpp_source_function(source['cpu_get_rows'], 'static void ggml_compute_forward_get_rows_q(')
    cpu_dispatch = _cpp_source_function(source['cpu_get_rows'], 'void ggml_compute_forward_get_rows(')
    input_graph = _cpp_source_function(source['input_graph'], 'ggml_tensor * llm_graph_context::build_inp_embd(')
    conditions = [
        ('get_rows_output', all(v in get_rows for v in ('b->type == GGML_TYPE_I32', 'enum ggml_type type = GGML_TYPE_F32', 'a->type == GGML_TYPE_I32', 'GGML_OP_GET_ROWS'))),
        ('mul_mat_output', 'ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)' in mul_mat),
        ('cpu_indexed_quantized_rows', all(v in cpu_rows for v in ('const int64_t nr = ggml_nelements(src1)', 'ggml_get_type_traits(type)->to_float', 'i01 = *(int32_t *)', 'src0->data + i01*nb01', '(float *)', 'dequantize_row_q('))),
        ('cuda_indexed_rows', all(v in source['cuda_get_rows'] for v in ('const int i01 = src1[', 'src0 + i01*nb01', 'dequantize_kernel(src0_row', 'dst_row['))),
        ('token_input_graph', all(v in input_graph for v in ('GGML_TYPE_I32, ubatch.n_tokens', 'cur = ggml_get_rows(ctx0, tok_embd, inp->tokens)', 'for (const auto & lora : *loras)'))),
    ]
    hidden_sources = {}
    for name in ('ggml_add_impl', 'ggml_rms_norm_impl', 'ggml_unary_impl'):
        function = _cpp_source_function(operators, 'static struct ggml_tensor * '+name+'(')
        hidden_sources[name] = 'ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a)' in function
    for name in ('ggml_ssm_conv', 'ggml_gated_delta_net'):
        function = _cpp_source_function(operators, 'struct ggml_tensor * '+name+'(')
        hidden_sources[name] = 'GGML_TYPE_F32' in function
    conditions.append(('ordinary_hidden_type_preservation', all(hidden_sources.values())))
    failures = [name for name, good in conditions if not good]
    if failures:
        raise ValueError('unrecognized native tensor storage rule: '+', '.join(failures))
    cpu_types = set(re.findall(r'case GGML_TYPE_([A-Z0-9_]+):', cpu_dispatch)) - {'I32'}
    cuda_types = set(re.findall(r'case GGML_TYPE_([A-Z0-9_]+):', source['cuda_get_rows'])) - {'I32'}
    return {'schema': SCHEMA, 'status': 'source_derived',
        'source_sha256': {str(paths[k].resolve()): hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        'source_symbols': ['llm_graph_context::build_inp_embd', 'ggml_get_rows', 'ggml_mul_mat',
                           'ggml_compute_forward_get_rows_q', 'k_get_rows', *hidden_sources],
        'embedding_index_bits': 32, 'embedding_output_storage_bits': 32,
        'get_rows_access': 'selected_packed_rows_per_index',
        'supported_weight_types': {'cpu': sorted(cpu_types), 'gpu': sorted(cuda_types)},
        'hidden_storage_evidence': hidden_sources,
        'architectures': sorted(_ARCHITECTURES),
        'scope': 'unsharded contiguous ordinary causal-text GGUF backbone; no supplied embeddings/adapters/MTP/experts; KV cache and non-hidden state dtypes remain separate',
        'dequantization_timing': 'unmodeled_selected_rows_only; no conversion throughput inferred',
        'cross_device_staging': 'full_tensor_transfer_if_existing_native_placement_requires_it; never collapsed into selected_row_bytes',
        'accuracy_validated': False, 'native_latency_used': False}


def _qualification(scenario, contract) -> dict[str, Any]:
    reasons = []
    if not isinstance(contract, Mapping):
        return {'qualified': False, 'reasons': ['no_tensor_storage_contract']}
    hashes = contract.get('source_sha256')
    hidden_evidence = contract.get('hidden_storage_evidence')
    valid_hashes = (isinstance(hashes, Mapping) and bool(hashes)
                    and all(isinstance(value, str) and re.fullmatch('[0-9a-fA-F]{64}', value)
                            for value in hashes.values()))
    valid_hidden_evidence = (isinstance(hidden_evidence, Mapping)
        and all(hidden_evidence.get(name) is True for name in (
            'ggml_add_impl', 'ggml_rms_norm_impl', 'ggml_unary_impl',
            'ggml_ssm_conv', 'ggml_gated_delta_net')))
    if (contract.get('schema') != SCHEMA or contract.get('status') != 'source_derived'
            or contract.get('embedding_index_bits') != 32 or contract.get('embedding_output_storage_bits') != 32
            or contract.get('get_rows_access') != 'selected_packed_rows_per_index'
            or not valid_hashes or not valid_hidden_evidence or contract.get('native_latency_used') is not False
            or contract.get('accuracy_validated') is not False):
        reasons.append('tensor_storage_source_contract_unverified')
    view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
    # The GGUF graph builder nests its audit metadata under "metadata".
    # Accept legacy flat bindings too, but reject conflicting identities.
    metadata = {}
    for owner in (scenario.model.graph.attributes, scenario.model.metadata):
        if not isinstance(owner, Mapping):
            continue
        nested = owner.get("metadata")
        for mapping in ((nested if isinstance(nested, Mapping) else {}), owner):
            for key, value in mapping.items():
                if key == "metadata":
                    continue
                if key in {"gguf_sha256", "gguf_embedding_binding"} and key in metadata and metadata[key] != value:
                    reasons.append("conflicting_nested_gguf_evidence")
                metadata[key] = value
    architectures = contract.get('architectures', ())
    if (not isinstance(architectures, (list, tuple, set, frozenset))
            or view.architecture not in _ARCHITECTURES or view.architecture not in architectures):
        reasons.append('graph_architecture_not_source_qualified')
    if not isinstance(metadata.get('gguf_sha256'), str) or not re.fullmatch('[0-9a-fA-F]{64}', metadata['gguf_sha256']):
        reasons.append('typed_gguf_identity_required')
    binding = metadata.get('gguf_embedding_binding')
    if not isinstance(binding, Mapping):
        reasons.append('gguf_embedding_binding_missing')
        binding = {}
    shape = binding.get('shape')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)):
        reasons.append('contiguous_2d_embedding_shape_required')
    if (scenario.model.vocabulary_size <= 0 or type(binding.get('n_bytes')) is not int
            or binding.get('n_bytes', 0) <= 0):
        reasons.append('positive_gguf_embedding_capacity_required')
    embedding_ops = [operator for operator in view.operators if operator.op_kind == 'embedding']
    embedding_output = None
    if len(embedding_ops) == 1 and len(embedding_ops[0].output_tensor_ids) == 1:
        embedding_output = next((tensor for tensor in view.tensors
            if tensor.tensor_id == embedding_ops[0].output_tensor_ids[0]), None)
    if (embedding_output is None or not embedding_output.shape
            or not isinstance(shape, (list, tuple))
            or tuple(shape) != (embedding_output.shape[-1], scenario.model.vocabulary_size)
            or binding.get('n_bytes') != view.embedding_weight_bytes):
        reasons.append('typed_graph_embedding_binding_mismatch')
    supported_types = contract.get('supported_weight_types', {})
    if (not isinstance(supported_types, Mapping)
            or not any(str(binding.get('type', '')).upper() in supported_types.get(kind, ())
                       for kind in ('cpu', 'gpu'))):
        reasons.append('get_rows_target_or_weight_type_unproven')
    if binding.get('strides') is not None or binding.get('is_view') or binding.get('transposed'):
        reasons.append('embedding_view_or_strides_unproven')
    if scenario.placement.parallel.tp_degree != 1:
        reasons.append('sharded_embedding_row_layout_unproven')
    if scenario.workload.mtp is not None or view.mtp_descriptors:
        reasons.append('mtp_hidden_storage_unproven')
    if any(i.layer.kind != 'dense' or i.layer.shared_expert_intermediate_size for i in view.layer_instances):
        reasons.append('expert_hidden_storage_unproven')
    if not scenario.model.text_backbone_only or set(scenario.model.supported_modalities) != {'text'}:
        reasons.append('non_text_hidden_storage_unproven')
    for item in [metadata, scenario.workload.metadata, *(r.metadata for r in scenario.workload.requests)]:
        raw_modalities = item.get('modalities', item.get('modality', ('text',)))
        modalities = (raw_modalities,) if isinstance(raw_modalities, str) else raw_modalities
        if (not isinstance(modalities, (list, tuple, set, frozenset)) or set(modalities) != {'text'}
                or any(item.get(k) for k in ('lora', 'loras', 'adapters', 'input_embeddings', 'embeddings', 'need_embd'))):
            reasons.append('dynamic_embedding_or_adapter_path_unproven')
    return {'qualified': not reasons, 'reasons': list(dict.fromkeys(reasons)),
            'graph_architecture': view.architecture, 'gguf_sha256': metadata.get('gguf_sha256'),
            'embedding_binding': dict(binding)}


def apply_llama_tensor_storage_contract(scenario, contract, *, f32_hidden_storage: bool = True):
    """Opt in only a source-qualified graph; caller replans before compiling.

    `f32_hidden_storage=False` isolates the GET_ROWS storage/traffic change.
    True additionally uses the existing ordinary-hidden storage capability,
    without changing KV caches, quantized weights or recurrent state dtypes.
    """
    if type(f32_hidden_storage) is not bool:
        raise ValueError('f32_hidden_storage must be explicit boolean')
    if contract is None:
        return scenario
    if not isinstance(contract, Mapping):
        raise ValueError('tensor storage contract must be a mapping')
    audit = _qualification(scenario, contract)
    previous = scenario.workload.metadata.get(AUDIT_KEY, {})
    prior_f32 = previous.get('previous_f32_hidden_storage', scenario.workload.metadata.get('llama_cpp_f32_hidden_storage', False))
    metadata = {**scenario.workload.metadata, SOURCE_KEY: dict(contract), AUDIT_KEY: {
        **audit, 'status': 'enabled' if audit['qualified'] else 'unsupported',
        'previous_f32_hidden_storage': prior_f32, 'f32_hidden_storage_requested': f32_hidden_storage,
        'source_contract_schema': contract.get('schema'), 'scope': contract.get('scope'),
        'timing_completeness': 'partial; row conversion/dequantization compute is unpriced'}}
    if audit['qualified'] and f32_hidden_storage:
        metadata['llama_cpp_f32_hidden_storage'] = True
    elif previous.get('status') == 'enabled':
        metadata['llama_cpp_f32_hidden_storage'] = prior_f32
    return replace(scenario, workload=replace(scenario.workload, metadata=metadata))


def qualify_llama_tensor_storage_contract(scenario):
    return _qualification(scenario, scenario.workload.metadata.get(SOURCE_KEY))


def resolve_embedding_gather_access(scenario, *, token_rows: int, hidden_width: int,
                                  rank_weight_capacity_bytes: int, target_kind: str,
                                  tp_degree: int, qualification=None) -> dict[str, Any]:
    """Return logical native gather traffic, retaining capacity as a distinct field."""
    contract = scenario.workload.metadata.get(SOURCE_KEY)
    audit = _qualification(scenario, contract) if qualification is None else qualification
    result = {'requested': contract is not None, **audit, 'fallback': 'legacy_embedding_memory_model'}
    if not audit['qualified']:
        return result
    binding = audit['embedding_binding'];shape = binding['shape'];weight_type = str(binding.get('type', '')).upper()
    reasons = []
    if tp_degree != 1 or tuple(shape) != (hidden_width, scenario.model.vocabulary_size):
        reasons.append('physical_embedding_shape_or_shard_mismatch')
    if (type(token_rows) is not int or token_rows <= 0 or type(rank_weight_capacity_bytes) is not int
            or rank_weight_capacity_bytes <= 0 or binding.get('n_bytes') != rank_weight_capacity_bytes
            or rank_weight_capacity_bytes % scenario.model.vocabulary_size):
        reasons.append('physical_packed_embedding_row_bytes_unverified')
    if target_kind not in ('cpu', 'gpu') or weight_type not in contract.get('supported_weight_types', {}).get(target_kind, []):
        reasons.append('get_rows_target_or_weight_type_unproven')
    if reasons:
        return {**result, 'qualified': False, 'reasons': reasons}
    row_bytes = rank_weight_capacity_bytes // scenario.model.vocabulary_size
    conversion = weight_type != 'F32'
    return {**result, 'fallback': None, 'selected_rows': token_rows, 'selected_elements': token_rows*hidden_width,
        'rank_weight_capacity_bytes': rank_weight_capacity_bytes, 'packed_row_bytes': row_bytes,
        'weight_type': weight_type, 'selected_weight_read_bytes': token_rows*row_bytes,
        'index_read_bytes': token_rows*4, 'output_write_bytes': token_rows*hidden_width*4,
        'output_storage_bits': 32,
        'dequantization_compute': {'status': 'unmodeled_selected_rows_only' if conversion else 'not_required',
                                  'selected_rows': token_rows, 'selected_elements': token_rows*hidden_width,
                                  'source_conversion': 'to_float/gpu_dequantize(selected_row)' if conversion else 'F32 row copy',
                                  'throughput_assigned': False},
        'traffic_semantics': 'indexed logical row reads; repeated indices are not deduplicated; cache-line/page/write-allocation effects remain unproved',
        'capacity_semantics': 'entire packed table remains resident/allocated or separately staged',
        'timing_completeness': 'partial; conversion compute unpriced'}