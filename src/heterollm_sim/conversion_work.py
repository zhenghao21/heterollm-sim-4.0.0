"""Exact locked-source activation-conversion work; no latency or bandwidth model.

The native caller chooses MMVQ/MMQ. This module only describes that declared
ordinary 2-D path. Source expressions are not compiled instructions and CTA
counts are not measured occupancy or an HBM bandwidth multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import prod
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

SOURCE_SHA256 = MappingProxyType({
    "ggml-cuda/quantize.cu": "12ab0782e5e9cf273dcc584edc7413729b18609dc20b4bc6c7a53efd5656c48f",
    "ggml-cuda/quantize.cuh": "1f87649a70eee707623b54ccd63fbe49ad3f135bed4eb2a29c65c32296bcb387",
    "ggml-cuda/common.cuh": "1cc3186a56426d5b929e9c29f9fae2d61d583f5f7abf0764d363c21cbbcbe932",
    "ggml-cuda/mmq.cuh": "d4fac91062e5cbcb083ec4c4f16cb4bf5ca741db903b5e2cd413e2e8dcfa2fae",
    "ggml-cuda/mmq.cu": "049996cb8387abab65a1181588c143179cabaac3f736d7d43a34b4b76d018c0e",
    "ggml-cuda/mmvq.cu": "14026871030393662628abdbd4937d5cab72031e20ddf582c9de1d7b424bb368",
    "ggml-common.h": "3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea",
})
_PATH_FORMATS = MappingProxyType({
    "MMVQ_Q8_1": frozenset({"Q5_0", "Q8_0"}),
    "MMQ_D4": frozenset({"Q5_0", "Q8_0"}),
    # DS4 source branch is distinct; these are the verified 32-value weight formats.
    # Q5_0/Q8_0 never enter DS4. No DS4 runtime/accuracy claim is made here.
    "MMQ_DS4": frozenset({"Q4_0", "Q4_1", "Q5_1"}),
})


class UnsupportedConversion(ValueError):
    """The declared invocation is outside the narrow source contract."""


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise UnsupportedConversion(name + " must be a positive integer")
    return value


@dataclass(frozen=True)
class ConversionSourceContract:
    compute_capability: int
    highest_compiled_arch: int
    warp_size: int
    source_hashes: Mapping[str, str]
    runtime_binary_sha256: str
    ordinary_contiguous_2d: bool
    input_dtype: str = "F32"
    channels: int = 1
    samples: int = 1
    has_ids: bool = False
    scatter: bool = False
    native_fp4: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_hashes, Mapping) or dict(self.source_hashes) != dict(SOURCE_SHA256):
            raise UnsupportedConversion("immutable conversion source identity differs")
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))
        if (type(self.compute_capability) is not int or self.compute_capability != 1200
                or type(self.highest_compiled_arch) is not int or self.highest_compiled_arch != 1200
                or type(self.warp_size) is not int or self.warp_size != 32):
            raise UnsupportedConversion("requires CC1200 compiled CC1200 and warp32")
        if (not isinstance(self.runtime_binary_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.runtime_binary_sha256) is None):
            raise UnsupportedConversion("explicit runtime binary SHA required")
        if (self.ordinary_contiguous_2d is not True or self.input_dtype != "F32"
                or type(self.channels) is not int or self.channels != 1
                or type(self.samples) is not int or self.samples != 1
                or self.has_ids is not False or self.scatter is not False or self.native_fp4 is not False):
            raise UnsupportedConversion("only ordinary F32 2-D, no ids/scatter/fp4 conversion supported")


def verify_conversion_source_tree(src_directory: str | Path) -> Mapping[str, str]:
    """Optional read-only verification. Import and work derivation perform no IO."""
    root = Path(src_directory)
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        with (root / relative).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise UnsupportedConversion("conversion source hash mismatch: " + relative)
        actual[relative] = digest
    return MappingProxyType(actual)


@dataclass(frozen=True)
class ConversionDAGNode:
    node_id: str
    operation: str
    count: int
    unit: str
    dependencies: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {"id": self.node_id, "operation": self.operation, "count": self.count,
                "unit": self.unit, "dependencies": self.dependencies, "cost_cycles": None}


@dataclass(frozen=True)
class ConversionWork:
    m: int
    k: int
    k_padded: int
    path: str
    weight_format: str
    runtime_binary_sha256: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    values_per_thread: int
    dag: tuple[ConversionDAGNode, ...]

    @property
    def cta_count(self) -> int:
        return prod(self.grid)

    @property
    def threads_per_cta(self) -> int:
        return prod(self.block)

    @property
    def warps_per_cta(self) -> int:
        return self.threads_per_cta // 32

    @property
    def launched_threads(self) -> int:
        return self.cta_count * self.threads_per_cta

    @property
    def launched_warps(self) -> int:
        return self.launched_threads // 32

    @property
    def padded_elements(self) -> int:
        return self.m * self.k_padded

    @property
    def logical_input_elements(self) -> int:
        return self.m * self.k

    @property
    def padding_only_threads(self) -> int:
        return (self.padded_elements - self.logical_input_elements) // self.values_per_thread

    @property
    def read_bytes(self) -> int:
        return 4 * self.logical_input_elements

    @property
    def write_bytes(self) -> int:
        return 36 * self.padded_elements // 32

    @property
    def partial_scalar_operations(self) -> int:
        """Preserve existing source proxy only, not a complete instruction count."""
        if self.path == "MMVQ_Q8_1":
            return 11 * self.padded_elements
        return (20 if self.path == "MMQ_DS4" else 14) * self.launched_threads

    @property
    def source_expression_counts(self) -> Mapping[str, int]:
        total = {}
        for node in self.dag:
            if node.unit in {"lane_source_expressions", "lane_source_expressions_upper_bound"}:
                total[node.operation] = total.get(node.operation, 0) + node.count
        return MappingProxyType(total)

    def active_sm_upper_bound(self, sm_count: int) -> int:
        """Source block-count upper bound, never a measured occupancy fraction."""
        return min(self.cta_count, _positive(sm_count, "SM count"))

    def to_metadata(self) -> dict[str, object]:
        return {"schema": "heterollm.source-conversion-work/v1", "path": self.path,
            "M": self.m, "Klogical": self.k, "Kpadded": self.k_padded,
            "weight_format": self.weight_format, "input_dtype": "F32", "output_layout": self.path,
            "compute_capability": 1200, "runtime_binary_sha256": self.runtime_binary_sha256,
            "source_hashes": dict(SOURCE_SHA256), "grid": self.grid, "block": self.block,
            "cta_count": self.cta_count, "threads_per_cta": self.threads_per_cta,
            "warps_per_cta": self.warps_per_cta, "launched_threads": self.launched_threads,
            "launched_warps": self.launched_warps, "values_per_thread": self.values_per_thread,
            "padding_only_threads": self.padding_only_threads,
            "read_bytes_logical": self.read_bytes, "write_bytes_physical": self.write_bytes,
            "partial_scalar_operations": self.partial_scalar_operations,
            "source_expression_counts": dict(self.source_expression_counts),
            "dag": tuple(node.to_metadata() for node in self.dag),
            "kernel_launch_count": 1, "host_synchronization_count_added": 0,
            "device_pdl_condition": "GGML_CUDA_USE_PDL && __CUDA_ARCH__ >= HOPPER; compiled/native behavior unproven",
            "pdl_cost_cycles": None, "measured_occupancy": None, "hbm_utilization_multiplier": None,
            "binary_source_equivalence_proven": False, "native_dispatch_proven": False,
            "cost_model_applied": False, "validated_llm_scope": False,
            "timing_completeness": "source_expressions_and_dependency_graph_without_cost_rates",
            "work_units": "source lane expressions/bytes; not SASS instructions or measured DRAM traffic",
            "unpriced_operations": ("shuffle_latency_and_issue", "division", "round_and_cast", "address_and_branch",
                "CTA_dispatch_and_tail", "device_PDL", "cache_transaction_behavior"),
            "zero_block_note": ("MMVQ zero-amax skips division/round branch; logical-block nonzero work is only an upper bound"
                if self.path == "MMVQ_Q8_1" else "MMQ zero-amax source inverse has infinity/NaN intermediates; this work description proves no bitwise numeric behavior")}


def _dag(m: int, k: int, kp: int, path: str) -> tuple[ConversionDAGNode, ...]:
    elements, logical = m * kp, m * k
    threads = elements if path == "MMVQ_Q8_1" else elements // 4
    nodes: list[ConversionDAGNode] = []
    def add(name: str, op: str, count: int, deps: tuple[str, ...] = (), unit: str = "lane_source_expressions") -> str:
        nodes.append(ConversionDAGNode(name, op, count, unit, deps))
        return name
    load = add("input", "read_F32", 4 * logical, unit="bytes")
    absolute = add("abs", "abs", elements, (load,))
    if path == "MMVQ_Q8_1":
        max_node, sum_node = absolute, load
        for i in range(5):
            sh = add(f"max_shuffle_{i}", "shuffle", threads, (max_node,))
            max_node = add(f"max_{i}", "float_max", threads, (max_node, sh))
            sh = add(f"sum_shuffle_{i}", "shuffle", threads, (sum_node,))
            sum_node = add(f"sum_{i}", "float_add", threads, (sum_node, sh))
        scale = add("scale", "division", threads, (max_node,))
        norm = add("normalize_nonzero_upper", "division", logical, (scale, load), "lane_source_expressions_upper_bound")
        rounded = add("round_nonzero_upper", "roundf", logical, (norm,), "lane_source_expressions_upper_bound")
        cast = add("int8_nonzero_upper", "int8_cast", logical, (rounded,), "lane_source_expressions_upper_bound")
        meta = add("half_scale_and_original_sum", "fp16_cast", 2 * elements // 32, (scale, sum_node))
    else:
        max_node = absolute
        for i in range(3):
            max_node = add(f"within_float4_max_{i}", "float_max", threads, (max_node, absolute))
        for i in range(3):
            sh = add(f"max_shuffle_{i}", "shuffle", threads, (max_node,))
            max_node = add(f"max_{i}", "float_max", threads, (max_node, sh))
        inverse = add("inverse_scale", "division", threads, (max_node,))
        product = add("normalize", "multiply", 4 * threads, (inverse, load))
        rounded = add("round", "roundf", 4 * threads, (product,))
        cast = add("int8", "int8_cast", 4 * threads, (rounded,))
        scale = add("inverse_reciprocal_scale", "division", threads, (inverse,))
        if path == "MMQ_DS4":
            sum_node = load
            for i in range(3):
                sum_node = add(f"within_float4_sum_{i}", "float_add", threads, (sum_node, load))
            for i in range(3):
                sh = add(f"sum_shuffle_{i}", "shuffle", threads, (sum_node,))
                sum_node = add(f"sum_{i}", "float_add", threads, (sum_node, sh))
            meta = add("half_scale_and_original_sum", "fp16_cast", 2 * elements // 32, (scale, sum_node))
        else:
            meta = scale
    add("q_values", "store_q8", elements, (cast,), "bytes")
    add("metadata", "store_scale_or_half2", 4 * elements // 32, (meta,), "bytes")
    return tuple(nodes)


def derive_conversion_work(*, m: int, k: int, weight_format: str, path: str,
                           contract: ConversionSourceContract) -> ConversionWork:
    """Derive ordinary single-channel/sample conversion, never select kernel dispatch.

    MMVQ initially covers Q5_0/Q8_0 M<=8. MMQ D4 covers those formats,
    DS4 covers only its verified32-value Q4_0/Q4_1/Q5_1 source branch. The
    declared MMQ path is not inferred merely from M or a model architecture.
    """
    if not isinstance(contract, ConversionSourceContract):
        raise UnsupportedConversion("explicit conversion source/runtime contract required")
    _positive(m, "M"); _positive(k, "K")
    if not isinstance(path, str) or path not in _PATH_FORMATS:
        raise UnsupportedConversion("unsupported conversion path")
    if not isinstance(weight_format, str) or weight_format not in _PATH_FORMATS[path]:
        raise UnsupportedConversion("weight format does not select declared conversion layout")
    if k % 32:
        raise UnsupportedConversion("logical K must be32-value aligned; no silent shape rewrite")
    if path == "MMVQ_Q8_1" and m > 8:
        raise UnsupportedConversion("outside source-qualified MMVQ batch domain")
    kp = ((k + 511) // 512) * 512
    grid, block, width = ((kp // 256, m, 1), (256, 1, 1), 1) if path == "MMVQ_Q8_1" else ((m, kp // 512, 1), (128, 1, 1), 4)
    if grid[0] > 2147483647 or grid[1] > 65535:
        raise UnsupportedConversion("launch exceeds supported CUDA grid axes")
    return ConversionWork(m, k, kp, path, weight_format, contract.runtime_binary_sha256,
                          grid, block, width, _dag(m, k, kp, path))
