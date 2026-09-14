"""Source facts for the fixed small Qwen3.5 ordinary-attention CUDA graph.

This declaration is separate from the older active attention descriptor so a
common input can carry new evidence without changing the accepted compiler.
It contains no measured duration, hardware price, or residual-derived value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .cost_models import TensorKernelWorkload
from .projection_descriptors import (
    AttentionExecutionDescriptor,
    resolve_attention_execution_descriptor,
    resolve_weight_projection,
)


SOURCE_KEY = "llama_cpp_qwen35_attention_source_work"
SOURCE_SCHEMA = "heterollm.qwen35-attention-source/v1"
BACKEND_COMMIT = "0f3a71be15af836d277c9f918adfafb45732677e"
GGUF_SHA256 = frozenset((
    "619917ae92f61eb6515a7070e944a0a7c2b198a2cf6536386f475485188a36ff",
    "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
))
ORDINARY_BLOCKS = (3, 7, 11, 15, 19, 23)


def source_declaration(*, gguf_sha256: str, block_index: int) -> dict:
    """Return only facts established for these two files and ordinary blocks."""
    if not isinstance(gguf_sha256, str) or gguf_sha256 not in GGUF_SHA256:
        raise ValueError("Qwen3.5 attention source GGUF identity is not covered")
    if type(block_index) is not int or block_index not in ORDINARY_BLOCKS:
        raise ValueError("Qwen3.5 attention source block is not an ordinary block")
    return {
        "schema_version": SOURCE_SCHEMA,
        "backend_commit": BACKEND_COMMIT,
        "architecture": "qwen35",
        "gguf_sha256": gguf_sha256,
        "block_index": block_index,
        "hidden_size": 1024,
        "query_heads": 8,
        "kv_heads": 2,
        "head_dim": 256,
        "query_width": 2048,
        "gate_width": 2048,
        "q_projection_width": 4096,
        "qk_scale": 0.0625,
        "qg_view": {
            "layout": "per_head_interleaved",
            "row_stride_elements": 512,
            "token_stride_elements": 4096,
            "query_offset_elements": 0,
            "gate_offset_elements": 256,
            "gate_contiguous_width": 2048,
        },
        "qk_rmsnorm": {
            "epsilon": 9.999999974752427e-7,
            "q_weight": {"name": "blk.{}.attn_q_norm.weight".format(block_index), "dtype": "f32", "elements": 256},
            "k_weight": {"name": "blk.{}.attn_k_norm.weight".format(block_index), "dtype": "f32", "elements": 256},
        },
        "rope": {
            "type": "imrope", "rotary_dim": 64, "sections": [11, 11, 10, 0],
            "freq_base": 10000000.0, "freq_scale": 1.0, "ext_factor": 0.0,
            "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
            "n_ctx_orig": 262144, "n_offs": 0, "freq_factors": None,
            "position_dtype": "i32", "position_components_per_token": 4,
        },
        "gate_activation": "sigmoid",
        "execution": {
            "standard_llama_graph": True,
            "default_cpu_cuda_scheduler": True,
            "gallocr_no_alloc_graph": True,
            "external_tensor_prebinding": False,
            "evaluation_callback": False,
            "extra_intermediate_outputs": False,
            "lora_adapters": False,
            "projection_scale_overrides": False,
            "cuda_graph_optimizer": False,
            "cuda_fusion_disabled": False,
            "attention_rotation_disabled": False,
        },
    }


def _same_source_value(actual: object, expected: object) -> bool:
    """Reject changed facts, including bool/int equality, without coercion."""
    if isinstance(expected, dict):
        return (isinstance(actual, Mapping) and set(actual) == set(expected)
                and all(_same_source_value(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_same_source_value(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


@dataclass(frozen=True)
class Qwen35AttentionSourceWork:
    gguf_sha256: str
    block_index: int
    attention: AttentionExecutionDescriptor

    @property
    def norm_weight_names(self):
        return tuple("blk.{}.attn_{}_norm.weight".format(self.block_index, part) for part in ("q", "k"))

    def tensor_kernel(self, stage: str, tokens: int, *, heads: int = 8):
        """Map the fixed source expressions to existing analytical resources.

        Repeated source requests and unique touched content are separate.
        Compiler load reuse, instruction lowering and cache transactions are
        not claimed as observed by this source-level budget.
        """
        if type(tokens) is not int or tokens <= 0 or type(heads) is not int or heads not in (2, 8):
            raise ValueError("Qwen3.5 kernel needs positive tokens and two or eight heads")
        elements, rows = 256 * heads * tokens, heads * tokens
        audit = {
            "backend_commit": BACKEND_COMMIT, "gguf_sha256": self.gguf_sha256,
            "block_index": self.block_index, "stage": stage,
            "tokens": tokens, "heads": heads, "logical_elements": elements,
            "source_kernel_launches": 1,
            "accounting_basis": "cuda_source_thread_expressions_not_machine_loads_or_dram_transactions",
            "resource_mapping": "existing_analytical_scalar_sfu_and_memory_profiles",
            "unpriced_terms": (
                "integer_index_control_and_fastmodulo", "pdl_and_block_synchronization",
                "shuffle_and_shared_memory_service", "compiler_dce_fma_and_special_function_lowering",
                "native_grid_occupancy_and_cache_transactions", "host_kernel_dispatch_instructions",
            ),
        }
        special = 0
        if stage == "rms_norm_mul":
            # norm.cu<256,true>, common.cuh block_reduce<SUM,256,float>.
            operations, special = 16 * elements, elements
            read, write = 12 * elements, 4 * elements
            unique = 8 * elements + 1024
            depth = 17  # square/add + two five-step reductions + mean/eps/rsqrt + two multiplies
            audit.update(source_float_multiply=3 * elements, source_float_add=12 * elements,
                         source_float_divide=elements, source_rsqrt_calls=elements,
                         source_shuffle_calls=10 * elements, shared_read_requests=256 * rows,
                         shared_write_requests=32 * rows, shared_unique_bytes_per_block=32,
                         shared_reserved_bytes_per_block=128, blocks=rows, threads_per_block=256,
                         input_unique_bytes=4 * elements, weight_unique_bytes=1024,
                         weight_read_requests=4 * elements, output_unique_bytes=4 * elements,
                         weight_tensor_name=self.norm_weight_names[0 if heads == 8 else 1],
                         weight_dtype="f32", weight_capacity_already_in_gguf_block_inventory=True,
                         global_statistics_bytes=0, norm_output_alias="distinct_from_input_source_rule_derived")
        elif stage == "imrope":
            rotated, pairs = 64 * heads * tokens, 32 * heads * tokens
            # Default ext_factor=0, no frequency factors: two divides, eight
            # multiplies and two +/- expressions, plus pow/cos/sin per pair.
            operations, special = 12 * pairs, 3 * pairs
            read, write = 4 * rotated + 4 * pairs, 4 * rotated
            unique, depth = 4 * rotated + 12 * tokens, 9
            audit.update(rotated_elements=rotated, rotated_pairs=pairs,
                         source_float_multiply=8 * pairs, source_float_add=2 * pairs,
                         source_float_divide=2 * pairs, source_pow_calls=pairs,
                         source_cos_calls=pairs, source_sin_calls=pairs,
                         position_read_requests=4 * pairs, position_unique_bytes=12 * tokens,
                         position_allocation_bytes=16 * tokens,
                         allocation_alias="same_address_source_rule_derived",
                         logical_output_bytes=4 * elements, untouched_elements=elements - rotated,
                         precomputed_sin_cos_table_bytes=0)
        elif stage in ("fwht256", "fwht64"):
            size = 256 if stage == "fwht256" else 64
            stages = size.bit_length() - 1
            operations = (1 + stages) * elements
            read = write = 4 * elements
            unique, depth = 8 * elements, 1 + stages
            audit.update(rotation_size=size, source_float_multiply=elements,
                         source_float_add_sub=stages * elements, source_shuffle_calls=5 * elements,
                         matrix_kernel_read_bytes=0, rows=elements // size,
                         blocks=(elements // size + 3) // 4, threads_per_block=128)
        elif stage == "gate_contiguous":
            if heads != 8:
                raise ValueError("gate materialization requires eight heads")
            operations, read, write = 0, 4 * elements, 4 * elements
            unique, depth = 8 * elements, 1
            audit.update(source_view_offset_elements=256, source_view_head_stride_elements=512,
                         source_view_token_stride_elements=4096, contiguous_output_width=2048,
                         input_unique_bytes=4 * elements, output_unique_bytes=4 * elements)
        elif stage == "sigmoid_mul":
            if heads != 8:
                raise ValueError("attention gate requires eight heads")
            operations, special = 4 * elements, elements
            read, write = 8 * elements, 4 * elements
            unique, depth = 12 * elements, 5
            audit.update(source_float_negate=elements, source_float_add=elements,
                         source_float_divide=elements, source_float_multiply=elements,
                         source_exp_calls=elements, sigmoid_global_intermediate_bytes=0)
        else:
            raise ValueError("unknown Qwen3.5 source kernel stage: " + stage)
        audit.update(read_requests_bytes=read, write_requests_bytes=write, unique_touched_bytes=unique,
                     source_dependency_steps=depth)
        workload = TensorKernelWorkload(
            operations=operations, transcendental_operations=special,
            read_bytes=read, write_bytes=write, working_set_bytes=unique,
            reuse_factor=max(1.0, (read + write) / unique), dependency_depth=depth,
            streaming_fraction=1.0 if stage == "gate_contiguous" else 0.0,
            name="qwen35_" + stage,
        )
        return workload, audit


def resolve_source_work(
    metadata: Mapping[str, object], *, hidden_size: int,
    attention_heads: int, kv_heads: int, head_dim: int,
) -> Optional[Qwen35AttentionSourceWork]:
    if SOURCE_KEY not in metadata:
        return None
    raw = metadata[SOURCE_KEY]
    if not isinstance(raw, Mapping):
        raise ValueError("{} must be a mapping".format(SOURCE_KEY))
    declared = source_declaration(gguf_sha256=raw.get("gguf_sha256"), block_index=raw.get("block_index"))
    if not _same_source_value(raw, declared):
        raise ValueError("{} differs from its supported source facts".format(SOURCE_KEY))
    if (hidden_size, attention_heads, kv_heads, head_dim) != (1024, 8, 2, 256):
        raise ValueError("Qwen3.5 ordinary-attention layer geometry differs from its source")
    if (type(metadata.get("gguf_block_index")) is not int
            or metadata["gguf_block_index"] != declared["block_index"]):
        raise ValueError("Qwen3.5 attention source block does not match the layer inventory")
    descriptor = {
        "schema_version": "heterollm.attention-execution/v1",
        **{key: declared[key] for key in ("query_heads", "kv_heads", "head_dim", "query_width", "gate_width", "q_projection_width", "qk_scale", "gate_activation")},
        "qk_norm": True,
        "rotary_dim": declared["rope"]["rotary_dim"],
    }
    if "attention_execution_descriptor" in metadata and not _same_source_value(metadata["attention_execution_descriptor"], descriptor):
        raise ValueError("old and source Qwen3.5 attention descriptors disagree")
    # The temporary mapping is internal; never write the old active key into
    # common input, model metadata, or the caller's descriptor mapping.
    attention = resolve_attention_execution_descriptor(
        {**metadata, "attention_execution_descriptor": descriptor},
        hidden_size=hidden_size, attention_heads=attention_heads,
        kv_heads=kv_heads, head_dim=head_dim,
    )
    if attention is None:  # pragma: no cover - explicit descriptor above
        raise AssertionError("source attention descriptor was not resolved")
    for projection, suffixes in (("attention.qkv", ("q", "k", "v")), ("attention.output", ("output",))):
        segments = resolve_weight_projection(metadata, projection)
        expected_names = {"blk.{}.attn_{}.weight".format(declared["block_index"], suffix) for suffix in suffixes}
        if segments is None or {segment.physical_tensor_name for segment in segments} != expected_names:
            raise ValueError("Qwen3.5 attention projection names do not match the declared block")
    return Qwen35AttentionSourceWork(declared["gguf_sha256"], declared["block_index"], attention)
