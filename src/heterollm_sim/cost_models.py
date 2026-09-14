"""Analytical GPU/HBM, CPU/host-memory, and digital SRAM-CIM cost models.

The models in this module deliberately produce *resource service demand*.
They do not model queueing or resource contention: the event engine owns that
part of the simulation.  Ordered :class:`CostPhase` objects are suitable for a
planner to lower into sequential tasks, while demands inside one phase may run
concurrently and therefore form a roofline-style maximum.

Units are decimal GB/s, GHz, TOPS, nanoseconds, bytes, and picojoules.  These
units are convenient because 1 GB/s is 1 byte/ns and 1 GHz is 1 cycle/ns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Tuple

from .contracts import (
    EvidenceStatus,
    OperatorClass,
    ResourceDemand,
    TaskCategory,
)
from .mmq_work import MMQWork


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _storage_bytes(elements: int, bits_per_element: int) -> int:
    return _ceil_div(elements * bits_per_element, 8)


def _require_positive(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError("%s must be positive" % name)


def _require_non_negative(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("%s must be non-negative" % name)


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)


def _require_efficiency(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 < value <= 1.0
    ):
        raise ValueError("%s must be in (0, 1]" % name)


@dataclass(frozen=True)
class _DMASetupService:
    """One controller-side DMA submission envelope for a single engine.

    Queue depth and the maximum outstanding request count determine how many
    descriptor waves the controller must submit.  Independent DMA engines are
    intentionally absent: the event kernel represents them through the shared
    DMA resource capacity, so folding them into this service time would count
    engine parallelism twice.
    """

    transaction_count: int
    queue_parallelism: int
    wave_count: int
    service_ns: float


def _dma_setup_service(
    bytes_moved: int,
    *,
    batch_bytes: int,
    queue_depth: int,
    max_outstanding: int,
    fixed_latency_ns: float,
    submission_ns_per_wave: float,
) -> _DMASetupService:
    """Return the shared DMA-controller setup contract used by every path."""

    _require_non_negative_int("bytes_moved", bytes_moved)
    _require_positive_int("batch_bytes", batch_bytes)
    _require_positive_int("queue_depth", queue_depth)
    _require_positive_int("max_outstanding", max_outstanding)
    _require_non_negative("fixed_latency_ns", fixed_latency_ns)
    _require_non_negative("submission_ns_per_wave", submission_ns_per_wave)
    if bytes_moved == 0:
        return _DMASetupService(0, 0, 0, 0.0)
    transaction_count = _ceil_div(bytes_moved, batch_bytes)
    queue_parallelism = min(queue_depth, max_outstanding)
    wave_count = _ceil_div(transaction_count, queue_parallelism)
    return _DMASetupService(
        transaction_count=transaction_count,
        queue_parallelism=queue_parallelism,
        wave_count=wave_count,
        service_ns=(
            float(fixed_latency_ns)
            + wave_count * float(submission_ns_per_wave)
        ),
    )


@dataclass(frozen=True)
class GemmWorkload:
    """A quantized GEMM ``[M, K] x [K, N] -> [M, N]``.

    The optional storage-byte overrides keep logical tensor-core geometry
    separate from a physically compressed or shared RHS/output.  This is
    useful for GQA/MQA KV operands, whose compute view is query-head-wide even
    though only the rank-local KV heads are stored.
    """

    m: int
    k: int
    n: int
    activation_bits: int = 8
    weight_bits: int = 8
    output_bits: int = 16
    accumulator_bits: int = 32
    packed_weight_formats: Tuple[str, ...] = ()
    packed_weight_transform_operations: int = 0
    # (format, local N columns, format-local fused transform operations).
    # This is internal lowering evidence, not user-supplied model metadata.
    packed_weight_format_segments: Tuple[Tuple[str, int, int], ...] = ()
    weight_metadata_bytes: int = 0
    epilogue_operations: int = 0
    epilogue_transcendental_operations: int = 0
    epilogue_output_elements: int = 0
    epilogue_name: str = ""
    name: str = "gemm"
    weight_storage_bytes: Optional[int] = None
    activation_storage_bytes: Optional[int] = None
    output_storage_bytes: Optional[int] = None
    mmq_work: Optional[MMQWork] = None
    # Source-qualified partial primitive work which shares the scalar GPU
    # resource with fused unpack/epilogue work.  The field is intentionally
    # optional: unknown mixed-port MMVQ service remains unpriced.
    source_partial_service_ns: float = 0.0
    source_partial_work_units: int = 0
    source_partial_name: str = ""

    def __post_init__(self) -> None:
        for field_name in ("m", "k", "n"):
            _require_positive_int(field_name, getattr(self, field_name))
        for field_name in (
            "activation_bits",
            "weight_bits",
            "output_bits",
            "accumulator_bits",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_non_negative_int(
            "packed_weight_transform_operations",
            self.packed_weight_transform_operations,
        )
        if not isinstance(self.packed_weight_formats, tuple):
            raise ValueError("packed_weight_formats must be a tuple")
        normalized_formats = []
        for weight_format in self.packed_weight_formats:
            if not isinstance(weight_format, str) or not weight_format.strip():
                raise ValueError(
                    "packed_weight_formats must contain non-empty text"
                )
            normalized_formats.append(weight_format.strip().casefold())
        if len(normalized_formats) != len(set(normalized_formats)):
            raise ValueError("packed_weight_formats must be unique")
        if (
            self.packed_weight_transform_operations > 0
            and not self.packed_weight_formats
        ):
            raise ValueError(
                "packed weight transform work requires a packed format"
            )
        if not isinstance(self.packed_weight_format_segments, tuple):
            raise ValueError("packed_weight_format_segments must be a tuple")
        if self.packed_weight_format_segments:
            segment_n = 0
            segment_operations = 0
            for segment in self.packed_weight_format_segments:
                if not isinstance(segment, tuple) or len(segment) != 3:
                    raise ValueError("packed weight format segment must be (format, n, operations)")
                format_name, local_n, transform_operations = segment
                if (
                    not isinstance(format_name, str)
                    or format_name.strip().casefold() not in normalized_formats
                ):
                    raise ValueError("packed weight format segment must use a declared format")
                _require_positive_int("packed weight format segment n", local_n)
                _require_non_negative_int(
                    "packed weight format segment operations", transform_operations
                )
                segment_n += local_n
                segment_operations += transform_operations
            if segment_n != self.n:
                raise ValueError("packed weight format segment widths must equal n")
            if segment_operations != self.packed_weight_transform_operations:
                raise ValueError("packed weight format segment operations must equal total")
        _require_non_negative_int(
            "weight_metadata_bytes", self.weight_metadata_bytes
        )
        _require_non_negative("source_partial_service_ns", self.source_partial_service_ns)
        _require_non_negative_int("source_partial_work_units", self.source_partial_work_units)
        if not isinstance(self.source_partial_name, str):
            raise ValueError("source_partial_name must be text")
        if self.source_partial_service_ns > 0.0 and not self.source_partial_name:
            raise ValueError("source_partial_name is required for partial work")
        _require_non_negative_int(
            "epilogue_operations", self.epilogue_operations
        )
        _require_non_negative_int(
            "epilogue_transcendental_operations",
            self.epilogue_transcendental_operations,
        )
        _require_non_negative_int(
            "epilogue_output_elements", self.epilogue_output_elements
        )
        if self.weight_storage_bytes is not None:
            _require_non_negative_int(
                "weight_storage_bytes", self.weight_storage_bytes
            )
        if self.activation_storage_bytes is not None:
            _require_non_negative_int(
                "activation_storage_bytes", self.activation_storage_bytes
            )
        if self.output_storage_bytes is not None:
            _require_non_negative_int(
                "output_storage_bytes", self.output_storage_bytes
            )
        if self.mmq_work is not None:
            if not isinstance(self.mmq_work, MMQWork):
                raise ValueError("mmq_work must be an MMQWork or None")
            if (self.m, self.k, self.n) != (
                self.mmq_work.m,
                self.mmq_work.k,
                self.mmq_work.n,
            ):
                raise ValueError("mmq_work shape must match GEMM shape")
            if tuple(normalized_formats) != (
                self.mmq_work.weight_format.casefold(),
            ):
                raise ValueError("mmq_work requires one matching physical weight format")
            if self.activation_storage_bytes != self.mmq_work.consumer_unique_bytes:
                raise ValueError("mmq_work requires the derived consumer input bytes")
            if (
                self.output_bits != 32
                or self.output_bytes != self.mmq_work.native_output_bytes
            ):
                raise ValueError("mmq_work requires a native F32 output of 4MN bytes")
        if (
            self.epilogue_operations
            or self.epilogue_transcendental_operations
            or self.epilogue_output_elements
        ) and not self.epilogue_name:
            raise ValueError("fused GEMM epilogue must have a name")
        if not self.name:
            raise ValueError("name must not be empty")

    @property
    def operations(self) -> int:
        """Conventional operation count, with one MAC equal to two ops."""

        return 2 * self.m * self.k * self.n

    @property
    def activation_bytes(self) -> int:
        if self.activation_storage_bytes is not None:
            return self.activation_storage_bytes
        return _storage_bytes(self.m * self.k, self.activation_bits)

    @property
    def weight_bytes(self) -> int:
        return (
            (
                _storage_bytes(self.k * self.n, self.weight_bits)
                if self.weight_storage_bytes is None
                else self.weight_storage_bytes
            )
            + self.weight_metadata_bytes
        )

    @property
    def output_bytes(self) -> int:
        if self.output_storage_bytes is not None:
            return self.output_storage_bytes
        output_elements = self.epilogue_output_elements or self.m * self.n
        return _storage_bytes(output_elements, self.output_bits)

    @property
    def minimum_io_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes + self.output_bytes


@dataclass(frozen=True)
class TensorKernelWorkload:
    """A generic analytical tensor kernel with explicit work and traffic.

    This is used for elementwise gates/norms, local convolutions, and
    scan/recurrent updates whose arithmetic is not a GEMM.  It deliberately
    exposes read and write traffic separately so recurrent state is not hidden
    inside an attention-shaped matrix multiply.
    """

    operations: int
    read_bytes: int
    write_bytes: int
    transcendental_operations: int = 0
    dependency_depth: int = 1
    working_set_bytes: int = 0
    reuse_factor: float = 1.0
    streaming_fraction: float = 0.0
    name: str = "tensor_kernel"
    launch_only: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int("operations", self.operations)
        _require_non_negative_int("read_bytes", self.read_bytes)
        _require_non_negative_int("write_bytes", self.write_bytes)
        _require_non_negative_int(
            "transcendental_operations", self.transcendental_operations
        )
        _require_positive_int("dependency_depth", self.dependency_depth)
        _require_non_negative_int("working_set_bytes", self.working_set_bytes)
        _require_positive("reuse_factor", self.reuse_factor)
        if not 0.0 <= self.streaming_fraction <= 1.0:
            raise ValueError("streaming_fraction must be in [0, 1]")
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.launch_only, bool):
            raise ValueError("launch_only must be boolean")
        if self.launch_only and (self.operations or self.transcendental_operations or self.minimum_io_bytes):
            raise ValueError("launch_only requires zero declared arithmetic and traffic")
        if (
            self.operations == 0
            and self.transcendental_operations == 0
            and self.minimum_io_bytes == 0
            and not self.launch_only
        ):
            raise ValueError("tensor kernel must declare operations or bytes")

    @property
    def minimum_io_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def effective_working_set_bytes(self) -> int:
        return self.working_set_bytes or self.minimum_io_bytes


@dataclass(frozen=True)
class FusedAttentionKVPhysicalContract:
    """Exact persisted KV bytes consumed by one token of fused attention."""

    payload_bytes_per_token: int
    metadata_bytes_per_token: int = 0
    dequant_operations_per_token: int = 0
    artifact_format: str = ""

    def __post_init__(self) -> None:
        _require_non_negative_int(
            "payload_bytes_per_token", self.payload_bytes_per_token
        )
        _require_non_negative_int(
            "metadata_bytes_per_token", self.metadata_bytes_per_token
        )
        _require_non_negative_int(
            "dequant_operations_per_token",
            self.dequant_operations_per_token,
        )
        if not isinstance(self.artifact_format, str):
            raise ValueError("artifact_format must be text")

    @property
    def bytes_per_token(self) -> int:
        return self.payload_bytes_per_token + self.metadata_bytes_per_token

    @property
    def scale_bytes_per_token(self) -> int:
        return self.metadata_bytes_per_token


@dataclass(frozen=True)
class FusedAttentionWorkload:
    """FlashAttention-style tiled ``QK -> softmax -> PV`` workload.

    The score matrix is an on-chip intermediate and therefore is not charged
    to backing memory.  Tensor, scalar, and special-function work remain
    independently visible so contention on each GPU execution engine is
    preserved by the unified event kernel.  Optional KV fields separate the
    query-head compute geometry from the physical rank-local KV representation
    and exact persisted-cache token accesses.
    """

    batch_tokens: int
    context_tokens: int
    hidden_size: int
    input_bits: int = 16
    output_bits: int = 16
    softmax_scalar_ops_per_score: int = 5
    softmax_transcendental_ops_per_score: int = 1
    query_tile_tokens: int = 64
    key_tile_tokens: int = 64
    name: str = "flash_attention"
    kv_hidden_size: Optional[int] = None
    kv_input_bits: Optional[int] = None
    kv_read_tokens: Optional[int] = None
    kv_physical_contract: Optional[FusedAttentionKVPhysicalContract] = None
    score_heads: int = 1
    qk_scale: Optional[float] = None
    # Explicit backend-lowered Q4_0 MMA input view, not live token accesses.
    # The current serving allocator proves only an occupied-row lower bound.
    q4_mma_view_tokens_lower_bound: int = 0
    q4_mma_head_dim: int = 0
    # Backing Q storage can differ from the tiled matrix input precision.
    query_storage_bits: Optional[int] = None

    def __post_init__(self) -> None:
        for field_name in (
            "batch_tokens",
            "context_tokens",
            "hidden_size",
            "input_bits",
            "output_bits",
            "softmax_scalar_ops_per_score",
            "query_tile_tokens",
            "key_tile_tokens",
            "score_heads",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_non_negative_int(
            "softmax_transcendental_ops_per_score",
            self.softmax_transcendental_ops_per_score,
        )
        if self.kv_hidden_size is not None:
            _require_positive_int("kv_hidden_size", self.kv_hidden_size)
        if self.kv_input_bits is not None:
            _require_positive_int("kv_input_bits", self.kv_input_bits)
        if self.kv_read_tokens is not None:
            _require_non_negative_int("kv_read_tokens", self.kv_read_tokens)
        if self.kv_physical_contract is not None and not isinstance(
            self.kv_physical_contract, FusedAttentionKVPhysicalContract
        ):
            raise ValueError(
                "kv_physical_contract must be a "
                "FusedAttentionKVPhysicalContract"
            )
        if self.qk_scale is not None and (
            isinstance(self.qk_scale, bool)
            or not isinstance(self.qk_scale, (int, float))
            or not math.isfinite(float(self.qk_scale))
            or float(self.qk_scale) <= 0.0
        ):
            raise ValueError("qk_scale must be a positive finite number")
        if not self.name:
            raise ValueError("name must not be empty")
        _require_non_negative_int(
            "q4_mma_view_tokens_lower_bound", self.q4_mma_view_tokens_lower_bound
        )
        _require_non_negative_int("q4_mma_head_dim", self.q4_mma_head_dim)
        if self.query_storage_bits is not None:
            _require_positive_int("query_storage_bits", self.query_storage_bits)
        if self.q4_mma_view_tokens_lower_bound:
            if (
                self.batch_tokens < 3
                or self.q4_mma_head_dim not in {64, 128, 256}
                or self.hidden_size % self.q4_mma_head_dim
                or self.effective_kv_hidden_size % self.q4_mma_head_dim
                or self.q4_mma_view_tokens_lower_bound % 256
                or self.q4_mma_view_tokens_lower_bound < self.effective_kv_read_tokens
                or self.kv_artifact_format.casefold() != "q4_0"
            ):
                raise ValueError("Q4 MMA materialization requires a supported aligned physical view")
        elif self.q4_mma_head_dim:
            raise ValueError("Q4 MMA head dimension requires an explicit view lower bound")

    @property
    def effective_kv_hidden_size(self) -> int:
        return (
            self.hidden_size
            if self.kv_hidden_size is None
            else self.kv_hidden_size
        )

    @property
    def effective_query_storage_bits(self) -> int:
        return (
            self.input_bits
            if self.query_storage_bits is None
            else self.query_storage_bits
        )

    @property
    def effective_kv_input_bits(self) -> int:
        return (
            self.input_bits
            if self.kv_input_bits is None
            else self.kv_input_bits
        )

    @property
    def effective_kv_read_tokens(self) -> int:
        return (
            self.context_tokens
            if self.kv_read_tokens is None
            else self.kv_read_tokens
        )

    @property
    def score_elements(self) -> int:
        return self.score_heads * self.batch_tokens * self.context_tokens

    @property
    def tensor_operations(self) -> int:
        return 4 * self.batch_tokens * self.context_tokens * self.hidden_size

    @property
    def transcendental_operations(self) -> int:
        return (
            self.score_elements
            * self.softmax_transcendental_ops_per_score
        )

    @property
    def softmax_scalar_operations(self) -> int:
        return self.score_elements * self.softmax_scalar_ops_per_score

    @property
    def qk_scale_operations(self) -> int:
        return self.score_elements if self.qk_scale is not None else 0

    @property
    def kv_dequant_operations(self) -> int:
        if self.kv_physical_contract is None:
            return 0
        return (
            self.effective_kv_read_tokens
            * self.kv_physical_contract.dequant_operations_per_token
        )

    @property
    def scalar_operations(self) -> int:
        return (
            self.softmax_scalar_operations
            + self.qk_scale_operations
            + self.kv_dequant_operations
        )

    @property
    def query_read_bytes(self) -> int:
        return _storage_bytes(
            self.batch_tokens * self.hidden_size,
            self.effective_query_storage_bits,
        )

    @property
    def kv_payload_bytes_per_token(self) -> int:
        if self.kv_physical_contract is not None:
            return self.kv_physical_contract.payload_bytes_per_token
        return self._kv_bytes_for_tokens(1)

    @property
    def kv_metadata_bytes_per_token(self) -> int:
        if self.kv_physical_contract is None:
            return 0
        return self.kv_physical_contract.metadata_bytes_per_token

    @property
    def kv_dequant_operations_per_token(self) -> int:
        if self.kv_physical_contract is None:
            return 0
        return self.kv_physical_contract.dequant_operations_per_token

    def _kv_bytes_for_tokens(self, token_count: int) -> int:
        if self.kv_physical_contract is not None:
            return token_count * self.kv_physical_contract.bytes_per_token
        return _storage_bytes(
            2
            * token_count
            * self.effective_kv_hidden_size,
            self.effective_kv_input_bits,
        )

    @property
    def kv_payload_bytes(self) -> int:
        return self.effective_kv_read_tokens * self.kv_payload_bytes_per_token

    @property
    def kv_metadata_bytes(self) -> int:
        return (
            self.effective_kv_read_tokens
            * self.kv_metadata_bytes_per_token
        )

    @property
    def read_bytes(self) -> int:
        # Q plus persisted K and V.  Score tiles never spill to backing memory.
        return self.query_read_bytes + self.kv_read_bytes

    @property
    def kv_read_bytes(self) -> int:
        return self._kv_bytes_for_tokens(self.effective_kv_read_tokens)

    @property
    def kv_artifact_format(self) -> str:
        if self.kv_physical_contract is None:
            return ""
        return self.kv_physical_contract.artifact_format

    @property
    def write_bytes(self) -> int:
        return _storage_bytes(
            self.batch_tokens * self.hidden_size, self.output_bits
        )

    @property
    def score_matrix_bytes_elided(self) -> int:
        return _storage_bytes(self.score_elements, self.output_bits)

    @property
    def onchip_working_set_bytes(self) -> int:
        query_tokens = min(self.batch_tokens, self.query_tile_tokens)
        # The same rectangular consumer extent must reach fusion admission.
        # b10760 only trims masked tail blocks for M >= 1024 or multiple streams;
        # this explicit materialization contract declares a single unified stream.
        context_tokens = max(
            self.context_tokens,
            self.q4_mma_view_tokens_lower_bound if self.batch_tokens < 1024 else 0,
        )
        key_tokens = min(context_tokens, self.key_tile_tokens)
        kv_tile_bytes = (
            4 * key_tokens * self.effective_kv_hidden_size
            if self.q4_mma_view_tokens_lower_bound
            else self._kv_bytes_for_tokens(key_tokens)
        )
        return (
            _storage_bytes(
                query_tokens * self.hidden_size, self.input_bits
            )
            + kv_tile_bytes
            + _storage_bytes(
                self.score_heads * query_tokens * key_tokens,
                self.output_bits,
            )
            + _storage_bytes(
                query_tokens * self.hidden_size, self.output_bits
            )
        )

    @property
    def dependency_depth(self) -> int:
        return max(1, int(math.ceil(math.log2(self.context_tokens))) + 3)


@dataclass(frozen=True)
class ElementwiseWorkload:
    """Shape-derived element-wise arithmetic and traffic.

    ``input_count`` equally sized tensors are read and one tensor is written.
    Traffic is derived from the logical shape so increasing ``elements`` can
    never decrease either work or bytes.
    """

    elements: int
    operations_per_element: int = 1
    fixed_operations: int = 0
    input_count: int = 1
    input_bits: int = 16
    output_bits: int = 16
    transcendental_ops_per_element: int = 0
    fixed_transcendental_operations: int = 0
    dependency_depth: int = 1
    working_set_bytes: int = 0
    reuse_factor: float = 1.0
    streaming_fraction: float = 0.0
    read_storage_bytes: Optional[int] = None
    write_storage_bytes: Optional[int] = None
    name: str = "elementwise"

    def __post_init__(self) -> None:
        for field_name in (
            "elements",
            "operations_per_element",
            "input_count",
            "input_bits",
            "output_bits",
            "dependency_depth",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_non_negative_int(
            "fixed_operations", self.fixed_operations
        )
        _require_non_negative_int(
            "transcendental_ops_per_element",
            self.transcendental_ops_per_element,
        )
        _require_non_negative_int(
            "fixed_transcendental_operations",
            self.fixed_transcendental_operations,
        )
        _require_non_negative_int("working_set_bytes", self.working_set_bytes)
        if self.read_storage_bytes is not None:
            _require_non_negative_int(
                "read_storage_bytes", self.read_storage_bytes
            )
        if self.write_storage_bytes is not None:
            _require_non_negative_int(
                "write_storage_bytes", self.write_storage_bytes
            )
        _require_positive("reuse_factor", self.reuse_factor)
        if not 0.0 <= self.streaming_fraction <= 1.0:
            raise ValueError("streaming_fraction must be in [0, 1]")
        if not self.name:
            raise ValueError("name must not be empty")

    @property
    def operations(self) -> int:
        return (
            self.elements * self.operations_per_element
            + self.fixed_operations
        )

    @property
    def transcendental_operations(self) -> int:
        return (
            self.elements * self.transcendental_ops_per_element
            + self.fixed_transcendental_operations
        )

    @property
    def read_bytes(self) -> int:
        if self.read_storage_bytes is not None:
            return self.read_storage_bytes
        return _storage_bytes(
            self.elements * self.input_count, self.input_bits
        )

    @property
    def write_bytes(self) -> int:
        if self.write_storage_bytes is not None:
            return self.write_storage_bytes
        return _storage_bytes(self.elements, self.output_bits)

    @property
    def minimum_io_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def effective_working_set_bytes(self) -> int:
        return self.working_set_bytes or self.minimum_io_bytes


@dataclass(frozen=True)
class ReductionWorkload:
    """A segmented reduction from input elements to output elements."""

    input_elements: int
    output_elements: int = 1
    operations_per_combine: int = 1
    fixed_operations: int = 0
    input_bits: int = 16
    output_bits: int = 16
    dependency_depth: int = 1
    working_set_bytes: int = 0
    reuse_factor: float = 1.0
    streaming_fraction: float = 0.0
    name: str = "reduction"

    def __post_init__(self) -> None:
        for field_name in (
            "input_elements",
            "output_elements",
            "operations_per_combine",
            "input_bits",
            "output_bits",
            "dependency_depth",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_non_negative_int("working_set_bytes", self.working_set_bytes)
        _require_non_negative_int(
            "fixed_operations", self.fixed_operations
        )
        _require_positive("reuse_factor", self.reuse_factor)
        if not 0.0 <= self.streaming_fraction <= 1.0:
            raise ValueError("streaming_fraction must be in [0, 1]")
        if self.output_elements > self.input_elements:
            raise ValueError(
                "output_elements must not exceed input_elements"
            )
        if not self.name:
            raise ValueError("name must not be empty")

    @property
    def operations(self) -> int:
        return (
            (self.input_elements - self.output_elements)
            * self.operations_per_combine
            + self.fixed_operations
        )

    @property
    def read_bytes(self) -> int:
        return _storage_bytes(self.input_elements, self.input_bits)

    @property
    def write_bytes(self) -> int:
        return _storage_bytes(self.output_elements, self.output_bits)

    @property
    def minimum_io_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def effective_working_set_bytes(self) -> int:
        return self.working_set_bytes or self.minimum_io_bytes


@dataclass(frozen=True)
class MemoryWorkload:
    """A pure data-movement primitive with explicit byte traffic."""

    read_bytes: int = 0
    write_bytes: int = 0
    working_set_bytes: int = 0
    reuse_factor: float = 1.0
    streaming_fraction: float = 1.0
    name: str = "memory"

    def __post_init__(self) -> None:
        _require_non_negative_int("read_bytes", self.read_bytes)
        _require_non_negative_int("write_bytes", self.write_bytes)
        _require_non_negative_int("working_set_bytes", self.working_set_bytes)
        _require_positive("reuse_factor", self.reuse_factor)
        if not 0.0 <= self.streaming_fraction <= 1.0:
            raise ValueError("streaming_fraction must be in [0, 1]")
        if self.minimum_io_bytes == 0:
            raise ValueError("memory workload must move at least one byte")
        if not self.name:
            raise ValueError("name must not be empty")

    @property
    def operations(self) -> int:
        return 0

    @property
    def minimum_io_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def effective_working_set_bytes(self) -> int:
        return self.working_set_bytes or self.minimum_io_bytes


@dataclass(frozen=True)
class CostPhase:
    """One ordered cost-model phase containing concurrent resource demands."""

    name: str
    category: TaskCategory
    demands: Tuple[ResourceDemand, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name must not be empty")
        if not self.demands:
            raise ValueError("a cost phase must contain at least one demand")
        resource_ids = tuple(d.resource_id for d in self.demands)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("a cost phase may demand each resource at most once")

    @property
    def service_ns(self) -> float:
        """Elapsed service time without contention: concurrent-demand maximum."""

        return max(demand.service_ns for demand in self.demands)

    @property
    def energy_pj(self) -> float:
        return sum(demand.energy_pj for demand in self.demands)


@dataclass(frozen=True)
class CostEstimate:
    """Ordered analytical phases plus common metrics used by planners."""

    phases: Tuple[CostPhase, ...]
    useful_ops: int
    utilization: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("an estimate must contain at least one phase")
        _require_non_negative("useful_ops", self.useful_ops)
        if not 0.0 <= self.utilization <= 1.0:
            raise ValueError("utilization must be in [0, 1]")

    @property
    def service_ns(self) -> float:
        """Sequential phase service time, before queueing and contention."""

        return sum(phase.service_ns for phase in self.phases)

    @property
    def latency_ns(self) -> float:
        return self.service_ns

    @property
    def energy_pj(self) -> float:
        return sum(phase.energy_pj for phase in self.phases)

    @property
    def bytes_moved(self) -> int:
        return sum(
            demand.bytes_moved
            for phase in self.phases
            for demand in phase.demands
        )

    @property
    def phase_names(self) -> Tuple[str, ...]:
        return tuple(phase.name for phase in self.phases)

    def phase(self, name: str) -> CostPhase:
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(name)


@dataclass(frozen=True)
class CacheLevelProfile:
    """One SRAM cache/scratchpad level in a v3 hardware profile.

    The simulator consumes aggregate working-set and reuse information rather
    than pretending to know unavailable physical addresses.  ``max_outstanding``
    and ``banks`` bound latency overlap; bandwidth is still an independent
    shared resource demand in the event kernel.
    """

    name: str
    capacity_bytes: int
    line_bytes: int
    hit_latency_ns: float
    bandwidth_gb_s: float
    associativity: int = 1
    banks: int = 1
    read_ports: int = 1
    write_ports: int = 1
    max_outstanding: int = 1
    energy_pj_per_byte: float = 0.0
    resource_id: str = "cache"

    def __post_init__(self) -> None:
        if not self.name or not self.resource_id:
            raise ValueError("cache name and resource_id must not be empty")
        for field_name in (
            "capacity_bytes",
            "line_bytes",
            "associativity",
            "banks",
            "read_ports",
            "write_ports",
            "max_outstanding",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_non_negative("hit_latency_ns", self.hit_latency_ns)
        _require_positive("bandwidth_gb_s", self.bandwidth_gb_s)
        _require_non_negative(
            "energy_pj_per_byte", self.energy_pj_per_byte
        )

    @property
    def transaction_parallelism(self) -> int:
        return self.banks * max(self.read_ports, self.write_ports) * self.max_outstanding


@dataclass(frozen=True)
class CacheHierarchyProfile:
    """Ordered near-to-far SRAM cache hierarchy."""

    levels: Tuple[CacheLevelProfile, ...]
    write_back: bool = True
    write_allocate: bool = True

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("cache hierarchy must contain at least one level")
        if not all(isinstance(level, CacheLevelProfile) for level in self.levels):
            raise ValueError("cache levels must contain CacheLevelProfile values")
        names = tuple(level.name for level in self.levels)
        resources = tuple(level.resource_id for level in self.levels)
        if len(names) != len(set(names)):
            raise ValueError("cache level names must be unique")
        if len(resources) != len(set(resources)):
            raise ValueError("cache resource ids must be unique")
        if any(
            left.capacity_bytes > right.capacity_bytes
            for left, right in zip(self.levels, self.levels[1:])
        ):
            raise ValueError("cache capacity must be non-decreasing by level")
        if not isinstance(self.write_back, bool) or not isinstance(
            self.write_allocate, bool
        ):
            raise ValueError("cache write policies must be boolean")


@dataclass(frozen=True)
class TensorCoreProfile:
    """Structural tensor-core issue model for one GPU generation."""

    sm_count: int
    tensor_cores_per_sm: int
    frequency_ghz: float
    mma_m: int = 16
    mma_n: int = 16
    mma_k: int = 16
    cycles_per_mma: float = 1.0
    supported_dtypes: Tuple[str, ...] = ("fp16", "bf16", "int8")
    dtype_throughput_scale: Mapping[str, float] = field(default_factory=dict)
    resource_id: str = "gpu.tensor_core"

    def __post_init__(self) -> None:
        for field_name in (
            "sm_count",
            "tensor_cores_per_sm",
            "mma_m",
            "mma_n",
            "mma_k",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_positive("frequency_ghz", self.frequency_ghz)
        _require_positive("cycles_per_mma", self.cycles_per_mma)
        if not self.supported_dtypes:
            raise ValueError("tensor core must support at least one dtype")
        if not self.resource_id:
            raise ValueError("tensor-core resource_id must not be empty")
        for dtype_name in self.supported_dtypes:
            if not str(dtype_name).strip():
                raise ValueError("tensor-core dtype names must not be empty")
        for dtype_name, scale in self.dtype_throughput_scale.items():
            if dtype_name not in self.supported_dtypes:
                raise ValueError(
                    "dtype throughput scale references unsupported dtype {}".format(
                        dtype_name
                    )
                )
            _require_positive(
                "dtype_throughput_scale[{}]".format(dtype_name), scale
            )

    @property
    def operations_per_mma(self) -> int:
        return 2 * self.mma_m * self.mma_n * self.mma_k

    def peak_tops(self, dtype_name: str) -> float:
        normalized = str(dtype_name).lower()
        if normalized not in self.supported_dtypes:
            raise ValueError(
                "tensor core does not support dtype {}".format(dtype_name)
            )
        scale = float(self.dtype_throughput_scale.get(normalized, 1.0))
        operations_per_ns = (
            self.sm_count
            * self.tensor_cores_per_sm
            * self.frequency_ghz
            * self.operations_per_mma
            * scale
            / self.cycles_per_mma
        )
        return operations_per_ns / 1000.0


@dataclass(frozen=True)
class CPUQuantizedDotCapability:
    """One explicitly declared packed-weight CPU kernel/ISA path.

    GEMM useful work keeps the conventional two-operations-per-MAC unit.
    ``effective_ops_per_instruction`` is therefore an effective retired-dot
    instruction rate: a 256-bit AVX2 maddubs/madd pair that realizes 32 MACs
    is 32 ops/instruction, while one AVX-VNNI dpbusd is 64 ops/instruction.
    Packed storage precision remains a workload fact and is deliberately
    separate from the internal dot operand widths declared here.
    """

    name: str
    supported_weight_formats: Tuple[str, ...]
    source_activation_bits: Tuple[int, ...]
    dot_activation_bits: int
    dot_weight_bits: int
    accumulator_bits: int
    effective_ops_per_instruction: float
    dot_issue_instructions_per_cycle_per_core: float
    auxiliary_ops_per_instruction: float
    activation_quantization_block_elements: int
    activation_quantization_instructions_per_block: Optional[float] = None
    evidence: str = ""
    # Per-format source primitive budgets for one K block of one row dot.
    # Intrinsics and indexed loads are instruction proxies, not disassembly.
    source_dot_work: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    source_dot_work_max_m: Optional[int] = None
    source_dot_work_all_m: bool = False
    source_dot_work_max_k: Optional[int] = None
    # Upper bound for selecting this kernel capability.  This is distinct
    # from source_dot_work_max_m, which only bounds the optional auxiliary
    # work ledger after a capability has already matched.
    maximum_m: Optional[int] = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("quantized-dot capability name must not be empty")
        if not isinstance(self.evidence, str):
            raise ValueError("quantized-dot capability evidence must be text")
        if not isinstance(self.supported_weight_formats, tuple) or not (
            self.supported_weight_formats
        ):
            raise ValueError(
                "supported_weight_formats must be a non-empty tuple"
            )
        normalized_formats = []
        for weight_format in self.supported_weight_formats:
            if not isinstance(weight_format, str) or not weight_format.strip():
                raise ValueError(
                    "supported_weight_formats must contain non-empty text"
                )
            normalized_formats.append(weight_format.strip().casefold())
        if len(normalized_formats) != len(set(normalized_formats)):
            raise ValueError("supported_weight_formats must be unique")
        if not isinstance(self.source_activation_bits, tuple) or not (
            self.source_activation_bits
        ):
            raise ValueError("source_activation_bits must be a non-empty tuple")
        for activation_bits in self.source_activation_bits:
            _require_positive_int("source_activation_bits", activation_bits)
        if len(self.source_activation_bits) != len(
            set(self.source_activation_bits)
        ):
            raise ValueError("source_activation_bits must be unique")
        for field_name in (
            "dot_activation_bits",
            "dot_weight_bits",
            "accumulator_bits",
            "activation_quantization_block_elements",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        for field_name in (
            "effective_ops_per_instruction",
            "dot_issue_instructions_per_cycle_per_core",
            "auxiliary_ops_per_instruction",
        ):
            _require_positive(field_name, getattr(self, field_name))
        if self.activation_quantization_instructions_per_block is not None:
            _require_non_negative(
                "activation_quantization_instructions_per_block",
                self.activation_quantization_instructions_per_block,
            )
        if not isinstance(self.source_dot_work, Mapping):
            raise ValueError("source_dot_work must be a mapping")
        if self.source_dot_work_max_m is not None:
            _require_positive_int("source_dot_work_max_m", self.source_dot_work_max_m)
        if not isinstance(self.source_dot_work_all_m, bool):
            raise ValueError("source_dot_work_all_m must be a boolean")
        if self.source_dot_work_all_m and self.source_dot_work_max_m is not None:
            raise ValueError("source_dot_work_all_m and an M bound are mutually exclusive")
        if self.source_dot_work_max_k is not None:
            _require_positive_int("source_dot_work_max_k", self.source_dot_work_max_k)
        if self.maximum_m is not None:
            _require_positive_int("maximum_m", self.maximum_m)
        if (self.source_dot_work_all_m or self.source_dot_work_max_k is not None) and not self.source_dot_work:
            raise ValueError("source dot scope requires source_dot_work and evidence")
        if self.source_dot_work and (
            (self.source_dot_work_max_m is None and not self.source_dot_work_all_m)
            or not self.evidence.strip()
        ):
            raise ValueError("source_dot_work requires an M bound or explicit all-M scope, and evidence")
        source_formats = []
        for weight_format, budget in self.source_dot_work.items():
            if not isinstance(weight_format, str) or weight_format.casefold() not in normalized_formats:
                raise ValueError("source_dot_work format must be a supported weight format")
            source_formats.append(weight_format.casefold())
            expected = {"block_elements", "auxiliary_vector_ops", "vector_loads", "scalar_lut_loads"}
            row_fields = {"row_auxiliary_vector_ops", "row_vector_loads", "row_scalar_lut_loads", "row_store_issue_ops"}
            if not isinstance(budget, Mapping) or set(budget) not in (expected, expected | row_fields):
                raise ValueError("source_dot_work budget has missing or unknown fields")
            _require_positive_int("source_dot_work block_elements", budget["block_elements"])
            for key in set(budget) - {"block_elements"}:
                _require_non_negative_int("source_dot_work " + key, budget[key])
        if len(set(source_formats)) != len(source_formats):
            raise ValueError("source_dot_work formats must be unique")

    def supports(self, workload: GemmWorkload) -> bool:
        required_formats = {
            weight_format.strip().casefold()
            for weight_format in workload.packed_weight_formats
        }
        supported_formats = {
            weight_format.strip().casefold()
            for weight_format in self.supported_weight_formats
        }
        return bool(required_formats) and (
            required_formats.issubset(supported_formats)
            and workload.activation_bits in self.source_activation_bits
            and workload.accumulator_bits == self.accumulator_bits
            and (self.maximum_m is None or workload.m <= self.maximum_m)
        )


@dataclass(frozen=True)
class CPUPipelineProfile:
    """Aggregate out-of-order CPU issue/retire model."""

    core_count: int
    frequency_ghz: float
    simd_width_bits: int
    decode_width: int
    issue_width: int
    retire_width: int
    vector_fma_units_per_core: int = 1
    vector_alu_units_per_core: int = 1
    load_units_per_core: int = 1
    store_units_per_core: int = 1
    branch_units_per_core: int = 1
    special_function_units_per_core: int = 1
    special_function_cycles_per_vector: float = 8.0
    reorder_buffer_entries: int = 128
    load_store_queue_entries: int = 64
    memory_level_parallelism: int = 8
    branch_mispredict_ns: float = 0.0
    resource_id: str = "cpu.pipeline"

    def __post_init__(self) -> None:
        for field_name in (
            "core_count",
            "simd_width_bits",
            "decode_width",
            "issue_width",
            "retire_width",
            "vector_fma_units_per_core",
            "vector_alu_units_per_core",
            "load_units_per_core",
            "store_units_per_core",
            "branch_units_per_core",
            "special_function_units_per_core",
            "reorder_buffer_entries",
            "load_store_queue_entries",
            "memory_level_parallelism",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_positive("frequency_ghz", self.frequency_ghz)
        _require_positive(
            "special_function_cycles_per_vector",
            self.special_function_cycles_per_vector,
        )
        _require_non_negative(
            "branch_mispredict_ns", self.branch_mispredict_ns
        )
        if not self.resource_id:
            raise ValueError("CPU pipeline resource_id must not be empty")


@dataclass(frozen=True)
class HostOrchestrationProfile:
    """Aggregate CPU preparation and physical invocation submission costs.

    Instruction counts describe one serial control thread.  ``core_count`` is
    deliberately not folded into these counts or their service rate; multiple
    host threads are represented by execution-resource capacity instead.
    """

    request_parse_ns: float
    batch_fixed_ns: float
    token_pack_ns: float
    submission_ns: float
    capacity_fixed_instructions: int = 96
    capacity_instructions_per_request: int = 64
    schedule_fixed_instructions: int = 192
    schedule_instructions_per_request: int = 48
    schedule_instructions_per_token: int = 8
    command_build_fixed_instructions: int = 128
    command_build_instructions_per_invocation: int = 12
    dma_queue_submission_ns: float = 62.5
    descriptor_bytes_per_request: int = 64
    token_bytes: int = 4
    dma_bandwidth_gb_s: float = 32.0
    dma_latency_ns: float = 1000.0
    max_inflight_batches: int = 2
    pinned_memory: bool = True
    kv_page_lookup_ns: float = 8.0
    kv_descriptor_ns: float = 4.0
    kv_descriptor_bytes: int = 32
    # Host API/front-end costs measured per request/token.  These are kept
    # separate from GEMM and memory terms so client TTFT can explain request
    # admission and stream serialization overhead explicitly.
    admission_ns: float = 0.0
    input_decode_ns_per_token: float = 0.0
    output_encode_ns_per_token: float = 0.0
    cpu_component_id: str = "cpu0"
    gpu_component_id: str = "gpu0"
    scheduler_resource_id: str = "host.scheduler"
    pack_resource_id: str = "host.pack"
    dma_resource_id: str = "host.h2d_dma"
    submission_resource_id: str = "gpu.command_queue"

    def __post_init__(self) -> None:
        for field_name in (
            "request_parse_ns",
            "batch_fixed_ns",
            "token_pack_ns",
            "submission_ns",
            "dma_queue_submission_ns",
            "dma_latency_ns",
            "kv_page_lookup_ns",
            "kv_descriptor_ns",
            "admission_ns",
            "input_decode_ns_per_token",
            "output_encode_ns_per_token",
        ):
            _require_non_negative(field_name, getattr(self, field_name))
        for field_name in (
            "capacity_fixed_instructions",
            "capacity_instructions_per_request",
            "schedule_fixed_instructions",
            "schedule_instructions_per_request",
            "schedule_instructions_per_token",
            "command_build_fixed_instructions",
            "command_build_instructions_per_invocation",
            "descriptor_bytes_per_request",
            "token_bytes",
            "max_inflight_batches",
            "kv_descriptor_bytes",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        _require_positive("dma_bandwidth_gb_s", self.dma_bandwidth_gb_s)
        if not isinstance(self.pinned_memory, bool):
            raise ValueError("pinned_memory must be boolean")
        resources = (
            self.scheduler_resource_id,
            self.pack_resource_id,
            self.dma_resource_id,
            self.submission_resource_id,
        )
        if any(not resource for resource in resources):
            raise ValueError("orchestration resource ids must not be empty")
        if len(resources) != len(set(resources)):
            raise ValueError("orchestration resource ids must be distinct")
        if not self.cpu_component_id or not self.gpu_component_id:
            raise ValueError("orchestration component ids must not be empty")

    def capacity_instruction_count(self, request_count: int) -> int:
        _require_positive_int("request_count", request_count)
        return (
            self.capacity_fixed_instructions
            + request_count * self.capacity_instructions_per_request
        )

    def schedule_instruction_count(
        self, request_count: int, token_count: int
    ) -> int:
        _require_positive_int("request_count", request_count)
        _require_non_negative_int("token_count", token_count)
        return (
            self.schedule_fixed_instructions
            + request_count * self.schedule_instructions_per_request
            + token_count * self.schedule_instructions_per_token
        )

    def command_build_instruction_count(self, invocation_count: int) -> int:
        _require_positive_int("invocation_count", invocation_count)
        return (
            self.command_build_fixed_instructions
            + invocation_count
            * self.command_build_instructions_per_invocation
        )

    def payload_bytes(self, request_count: int, token_count: int) -> int:
        _require_positive_int("request_count", request_count)
        _require_non_negative_int("token_count", token_count)
        return (
            request_count * self.descriptor_bytes_per_request
            + token_count * self.token_bytes
        )


@dataclass(frozen=True)
class HBMProfile:
    """Effective HBM service and energy characteristics."""

    bandwidth_gb_s: float
    efficiency: float = 1.0
    energy_pj_per_byte: float = 0.0
    resource_id: str = "hbm.channel"

    def __post_init__(self) -> None:
        _require_positive("bandwidth_gb_s", self.bandwidth_gb_s)
        _require_efficiency("efficiency", self.efficiency)
        _require_non_negative("energy_pj_per_byte", self.energy_pj_per_byte)
        if not self.resource_id:
            raise ValueError("resource_id must not be empty")

    @property
    def effective_bandwidth_gb_s(self) -> float:
        return self.bandwidth_gb_s * self.efficiency


@dataclass(frozen=True)
class HostMemoryProfile:
    """Effective CPU-visible memory service and energy characteristics."""

    bandwidth_gb_s: float
    efficiency: float = 1.0
    energy_pj_per_byte: float = 0.0
    resource_id: str = "host.memory"
    name: str = "host-memory"

    def __post_init__(self) -> None:
        _require_positive("bandwidth_gb_s", self.bandwidth_gb_s)
        _require_efficiency("efficiency", self.efficiency)
        _require_non_negative("energy_pj_per_byte", self.energy_pj_per_byte)
        if not self.resource_id or not self.name:
            raise ValueError("resource and profile names must not be empty")

    @property
    def effective_bandwidth_gb_s(self) -> float:
        # Decimal GB/s is numerically equal to bytes/ns.
        return self.bandwidth_gb_s * self.efficiency


@dataclass(frozen=True)
class HostGemmOffloadCapability:
    """Effective runtime support for executing host-operation GEMMs on a GPU.

    This capability applies only to GEMM.  A GEMM whose physical ``M`` is at
    least ``minimum_m`` may execute on the GPU even when its weights remain
    statically owned by host memory or the CPU.  Declaring the capability does
    not move or otherwise change ownership of those weights.
    """

    minimum_m: int
    evidence: str

    def __post_init__(self) -> None:
        _require_positive_int("minimum_m", self.minimum_m)
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("host GEMM offload capability evidence must be non-empty text")


@dataclass(frozen=True)
class HostRecurrentOffloadCapability:
    """Source-declared CUDA path for a host-owned recurrent mixer.

    The contract is intentionally exact.  It names the graph architecture,
    recurrent geometry, and complete adjacent-op set validated from one
    runtime source revision.  Planner lowering must fail closed when any field
    differs; declaring this capability never changes persistent state
    ownership.
    """

    op_offload: bool
    minimum_m: int
    architecture: str
    supported_ops: Tuple[str, ...]
    query_width: int
    key_width: int
    value_width: int
    conv_kernel_size: int
    state_dtype: str
    evidence: str
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.op_offload, bool):
            raise ValueError(
                "host recurrent offload op_offload must be boolean"
            )
        _require_positive_int("minimum_m", self.minimum_m)
        for field_name in (
            "architecture",
            "state_dtype",
            "evidence",
            "provenance",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "host recurrent offload {} must be non-empty text".format(
                        field_name
                    )
                )
        if not isinstance(self.supported_ops, tuple) or not self.supported_ops:
            raise ValueError(
                "host recurrent offload supported_ops must be a non-empty tuple"
            )
        normalized_ops = []
        for operation in self.supported_ops:
            if not isinstance(operation, str) or not operation.strip():
                raise ValueError(
                    "host recurrent offload supported_ops must contain "
                    "non-empty text"
                )
            normalized_ops.append(operation.strip().casefold())
        if len(normalized_ops) != len(set(normalized_ops)):
            raise ValueError(
                "host recurrent offload supported_ops must be unique"
            )
        for field_name in (
            "query_width",
            "key_width",
            "value_width",
            "conv_kernel_size",
        ):
            _require_positive_int(field_name, getattr(self, field_name))


@dataclass(frozen=True)
class GPUQuantizedMatmulCapability:
    """One explicitly evidenced packed-weight GPU matmul kernel path.

    Packed storage and source activation widths remain workload facts.  The
    capability only declares the internal tensor-core dtype selected by a
    particular kernel family after any kernel-local activation conversion.
    ``min_m`` bounds dispatch by physical input rows, not logical concurrency.
    """

    name: str
    supported_weight_formats: Tuple[str, ...]
    source_activation_bits: Tuple[int, ...]
    internal_activation_bits: int
    accumulator_bits: int
    kernel_family: str
    tensor_core_dtype: str
    evidence: str
    provenance: str
    min_m: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("min_m", self.min_m)
        for field_name in (
            "name",
            "kernel_family",
            "tensor_core_dtype",
            "evidence",
            "provenance",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "GPU quantized-matmul capability {} must be non-empty text"
                    .format(field_name)
                )
        if not isinstance(self.supported_weight_formats, tuple) or not (
            self.supported_weight_formats
        ):
            raise ValueError(
                "supported_weight_formats must be a non-empty tuple"
            )
        normalized_formats = []
        for weight_format in self.supported_weight_formats:
            if not isinstance(weight_format, str) or not weight_format.strip():
                raise ValueError(
                    "supported_weight_formats must contain non-empty text"
                )
            normalized_formats.append(weight_format.strip().casefold())
        if len(normalized_formats) != len(set(normalized_formats)):
            raise ValueError("supported_weight_formats must be unique")
        if not isinstance(self.source_activation_bits, tuple) or not (
            self.source_activation_bits
        ):
            raise ValueError("source_activation_bits must be a non-empty tuple")
        for activation_bits in self.source_activation_bits:
            _require_positive_int("source_activation_bits", activation_bits)
        if len(self.source_activation_bits) != len(
            set(self.source_activation_bits)
        ):
            raise ValueError("source_activation_bits must be unique")
        _require_positive_int(
            "internal_activation_bits", self.internal_activation_bits
        )
        _require_positive_int("accumulator_bits", self.accumulator_bits)
        expected_internal_bits = {
            "int8": 8,
            "fp16": 16,
            "bf16": 16,
            "fp32": 32,
        }.get(self.tensor_core_dtype.strip().casefold())
        if (
            expected_internal_bits is not None
            and self.internal_activation_bits != expected_internal_bits
        ):
            raise ValueError(
                "internal_activation_bits does not match tensor_core_dtype"
            )

    def supports(self, workload: GemmWorkload) -> bool:
        required_formats = {
            weight_format.strip().casefold()
            for weight_format in workload.packed_weight_formats
        }
        supported_formats = {
            weight_format.strip().casefold()
            for weight_format in self.supported_weight_formats
        }
        return bool(required_formats) and (
            required_formats.issubset(supported_formats)
            and workload.m >= self.min_m
            and workload.activation_bits in self.source_activation_bits
            and workload.accumulator_bits == self.accumulator_bits
        )


@dataclass(frozen=True)
class GPUProfile:
    """Version-3 structural GPU profile.

    Tensor, scalar/SFU, launch, cache, and HBM resources are distinct.  Peak
    tensor throughput is derived from the declared SM/MMA geometry instead of
    being accepted as an opaque TOPS scalar.
    """

    tensor_core: TensorCoreProfile
    cache_hierarchy: CacheHierarchyProfile
    scalar_lanes_per_sm: int
    scalar_ops_per_cycle: float
    reduction_ops_per_cycle_per_sm: float
    special_function_units_per_sm: int
    special_function_ops_per_cycle: float
    host_gemm_offload: Optional[HostGemmOffloadCapability] = None
    host_recurrent_offload: Optional[
        HostRecurrentOffloadCapability
    ] = None
    quantized_matmul_capabilities: Tuple[
        GPUQuantizedMatmulCapability, ...
    ] = ()
    occupancy: float = 1.0
    attainable_efficiency: float = 1.0
    kernel_launch_ns: float = 0.0
    tensor_energy_pj_per_op: float = 0.0
    scalar_energy_pj_per_op: float = 0.0
    special_function_energy_pj_per_op: float = 0.0
    launch_energy_pj: float = 0.0
    scalar_resource_id: str = "gpu.scalar"
    special_function_resource_id: str = "gpu.sfu"
    launch_resource_id: str = "gpu.frontend"
    default_tensor_dtype: str = "int8"
    name: str = "gpu"

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_core, TensorCoreProfile):
            raise ValueError("tensor_core must be a TensorCoreProfile")
        if not isinstance(self.cache_hierarchy, CacheHierarchyProfile):
            raise ValueError("cache_hierarchy must be a CacheHierarchyProfile")
        if self.host_gemm_offload is not None and not isinstance(
            self.host_gemm_offload, HostGemmOffloadCapability
        ):
            raise ValueError(
                "host_gemm_offload must be a HostGemmOffloadCapability or None"
            )
        if self.host_recurrent_offload is not None and not isinstance(
            self.host_recurrent_offload, HostRecurrentOffloadCapability
        ):
            raise ValueError(
                "host_recurrent_offload must be a "
                "HostRecurrentOffloadCapability or None"
            )
        if not isinstance(self.quantized_matmul_capabilities, tuple):
            raise ValueError("quantized_matmul_capabilities must be a tuple")
        if not all(
            isinstance(capability, GPUQuantizedMatmulCapability)
            for capability in self.quantized_matmul_capabilities
        ):
            raise ValueError(
                "quantized_matmul_capabilities must contain "
                "GPUQuantizedMatmulCapability values"
            )
        capability_names = tuple(
            capability.name.casefold()
            for capability in self.quantized_matmul_capabilities
        )
        if len(capability_names) != len(set(capability_names)):
            raise ValueError(
                "GPU quantized-matmul capability names must be unique"
            )
        unsupported_capability_dtypes = tuple(
            capability.tensor_core_dtype
            for capability in self.quantized_matmul_capabilities
            if capability.tensor_core_dtype.casefold()
            not in {
                dtype_name.casefold()
                for dtype_name in self.tensor_core.supported_dtypes
            }
        )
        if unsupported_capability_dtypes:
            raise ValueError(
                "GPU quantized-matmul capability tensor dtype is not supported: {}"
                .format(", ".join(unsupported_capability_dtypes))
            )
        _require_positive_int("scalar_lanes_per_sm", self.scalar_lanes_per_sm)
        _require_positive("scalar_ops_per_cycle", self.scalar_ops_per_cycle)
        _require_positive(
            "reduction_ops_per_cycle_per_sm",
            self.reduction_ops_per_cycle_per_sm,
        )
        _require_positive_int(
            "special_function_units_per_sm",
            self.special_function_units_per_sm,
        )
        _require_positive(
            "special_function_ops_per_cycle",
            self.special_function_ops_per_cycle,
        )
        _require_efficiency("occupancy", self.occupancy)
        _require_efficiency(
            "attainable_efficiency", self.attainable_efficiency
        )
        for field_name in (
            "kernel_launch_ns",
            "tensor_energy_pj_per_op",
            "scalar_energy_pj_per_op",
            "special_function_energy_pj_per_op",
            "launch_energy_pj",
        ):
            _require_non_negative(field_name, getattr(self, field_name))
        if (
            not self.scalar_resource_id
            or not self.special_function_resource_id
            or not self.launch_resource_id
            or not self.name
            or not self.default_tensor_dtype
        ):
            raise ValueError("resource and profile names must not be empty")
        if self.default_tensor_dtype not in self.tensor_core.supported_dtypes:
            raise ValueError("default tensor dtype is not supported")
        resource_ids = (
            self.tensor_core.resource_id,
            self.scalar_resource_id,
            self.special_function_resource_id,
            self.launch_resource_id,
        ) + tuple(
            level.resource_id for level in self.cache_hierarchy.levels
        )
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("GPU compute/cache resource ids must be distinct")

    def resolve_quantized_matmul_capability(
        self, workload: GemmWorkload
    ) -> Optional[GPUQuantizedMatmulCapability]:
        matches = tuple(
            capability
            for capability in self.quantized_matmul_capabilities
            if capability.supports(workload)
        )
        if len(matches) > 1:
            raise ValueError(
                "multiple GPU quantized-matmul capabilities match workload {}: {}"
                .format(
                    workload.name,
                    ", ".join(capability.name for capability in matches),
                )
            )
        return matches[0] if matches else None

    @property
    def sm_count(self) -> int:
        return self.tensor_core.sm_count

    @property
    def peak_tops(self) -> float:
        return self.tensor_core.peak_tops(self.default_tensor_dtype)

    @property
    def attainable_tops(self) -> float:
        return self.peak_tops * self.attainable_efficiency * self.occupancy

    @property
    def elementwise_gops(self) -> float:
        return (
            self.sm_count
            * self.scalar_lanes_per_sm
            * self.scalar_ops_per_cycle
            * self.tensor_core.frequency_ghz
            * self.attainable_efficiency
            * self.occupancy
        )

    @property
    def reduction_gops(self) -> float:
        return (
            self.sm_count
            * self.reduction_ops_per_cycle_per_sm
            * self.tensor_core.frequency_ghz
            * self.attainable_efficiency
            * self.occupancy
        )

    @property
    def special_function_gops(self) -> float:
        return (
            self.sm_count
            * self.special_function_units_per_sm
            * self.special_function_ops_per_cycle
            * self.tensor_core.frequency_ghz
            * self.attainable_efficiency
            * self.occupancy
        )

    @property
    def compute_resource_id(self) -> str:
        return self.tensor_core.resource_id

    @property
    def energy_pj_per_op(self) -> float:
        return self.tensor_energy_pj_per_op

    @property
    def elementwise_energy_pj_per_op(self) -> float:
        return self.scalar_energy_pj_per_op

    @property
    def reduction_energy_pj_per_op(self) -> float:
        return self.scalar_energy_pj_per_op

    def estimate_gemm(
        self, hbm: HBMProfile, workload: GemmWorkload
    ) -> CostEstimate:
        return estimate_gpu_gemm(self, hbm, workload)

    def estimate_tensor_kernel(
        self, hbm: HBMProfile, workload: TensorKernelWorkload
    ) -> CostEstimate:
        return estimate_gpu_tensor_kernel(self, hbm, workload)

    def estimate_elementwise(
        self, hbm: HBMProfile, workload: ElementwiseWorkload
    ) -> CostEstimate:
        return estimate_gpu_elementwise(self, hbm, workload)

    def estimate_reduction(
        self, hbm: HBMProfile, workload: ReductionWorkload
    ) -> CostEstimate:
        return estimate_gpu_reduction(self, hbm, workload)

    def estimate_memory(
        self, hbm: HBMProfile, workload: MemoryWorkload
    ) -> CostEstimate:
        return estimate_gpu_memory(self, hbm, workload)


@dataclass(frozen=True)
class CPUProfile:
    """Version-3 CPU pipeline plus SRAM-cache hierarchy."""

    pipeline: CPUPipelineProfile
    cache_hierarchy: CacheHierarchyProfile
    quantized_dot_capabilities: Tuple[CPUQuantizedDotCapability, ...] = ()
    attainable_efficiency: float = 1.0
    dispatch_ns: float = 0.0
    gemm_energy_pj_per_op: float = 0.0
    elementwise_energy_pj_per_op: float = 0.0
    reduction_energy_pj_per_op: float = 0.0
    special_function_energy_pj_per_op: float = 0.0
    dispatch_energy_pj: float = 0.0
    name: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, CPUPipelineProfile):
            raise ValueError("pipeline must be a CPUPipelineProfile")
        if not isinstance(self.cache_hierarchy, CacheHierarchyProfile):
            raise ValueError("cache_hierarchy must be a CacheHierarchyProfile")
        if not isinstance(self.quantized_dot_capabilities, tuple):
            raise ValueError("quantized_dot_capabilities must be a tuple")
        if not all(
            isinstance(capability, CPUQuantizedDotCapability)
            for capability in self.quantized_dot_capabilities
        ):
            raise ValueError(
                "quantized_dot_capabilities must contain "
                "CPUQuantizedDotCapability values"
            )
        capability_names = tuple(
            capability.name.casefold()
            for capability in self.quantized_dot_capabilities
        )
        if len(capability_names) != len(set(capability_names)):
            raise ValueError("quantized-dot capability names must be unique")
        _require_efficiency(
            "attainable_efficiency", self.attainable_efficiency
        )
        for field_name in (
            "dispatch_ns",
            "gemm_energy_pj_per_op",
            "elementwise_energy_pj_per_op",
            "reduction_energy_pj_per_op",
            "special_function_energy_pj_per_op",
            "dispatch_energy_pj",
        ):
            _require_non_negative(field_name, getattr(self, field_name))
        if not self.name:
            raise ValueError("resource and profile names must not be empty")
        cache_resources = tuple(
            level.resource_id for level in self.cache_hierarchy.levels
        )
        if self.pipeline.resource_id in cache_resources:
            raise ValueError("CPU pipeline/cache resource ids must be distinct")

    @property
    def compute_resource_id(self) -> str:
        return self.pipeline.resource_id

    @property
    def gemm_gops(self) -> float:
        lanes = self.pipeline.simd_width_bits / 16.0
        return (
            self.pipeline.core_count
            * self.pipeline.vector_fma_units_per_core
            * lanes
            * 2.0
            * self.pipeline.frequency_ghz
        )

    @property
    def elementwise_gops(self) -> float:
        lanes = self.pipeline.simd_width_bits / 16.0
        return (
            self.pipeline.core_count
            * self.pipeline.vector_alu_units_per_core
            * lanes
            * self.pipeline.frequency_ghz
        )

    @property
    def reduction_gops(self) -> float:
        return self.elementwise_gops / max(
            1.0, math.log2(self.pipeline.simd_width_bits / 16.0)
        )

    @property
    def attainable_gemm_gops(self) -> float:
        return self.gemm_gops * self.attainable_efficiency

    def resolve_quantized_dot_capability(
        self, workload: GemmWorkload
    ) -> Optional[CPUQuantizedDotCapability]:
        matches = tuple(
            capability
            for capability in self.quantized_dot_capabilities
            if capability.supports(workload)
        )
        if len(matches) > 1:
            raise ValueError(
                "multiple CPU quantized-dot capabilities match workload {}: {}"
                .format(
                    workload.name,
                    ", ".join(capability.name for capability in matches),
                )
            )
        return matches[0] if matches else None

    def attainable_quantized_dot_gops(
        self, capability: CPUQuantizedDotCapability
    ) -> float:
        if capability not in self.quantized_dot_capabilities:
            raise ValueError(
                "quantized-dot capability is not declared by this CPU profile"
            )
        return (
            self.pipeline.core_count
            * self.pipeline.frequency_ghz
            * capability.dot_issue_instructions_per_cycle_per_core
            * capability.effective_ops_per_instruction
            * self.attainable_efficiency
        )

    @property
    def attainable_elementwise_gops(self) -> float:
        return self.elementwise_gops * self.attainable_efficiency

    @property
    def attainable_reduction_gops(self) -> float:
        return self.reduction_gops * self.attainable_efficiency

    def estimate_gemm(
        self, memory: HostMemoryProfile, workload: GemmWorkload
    ) -> CostEstimate:
        return estimate_cpu_gemm(self, memory, workload)

    def estimate_elementwise(
        self, memory: HostMemoryProfile, workload: ElementwiseWorkload
    ) -> CostEstimate:
        return estimate_cpu_elementwise(self, memory, workload)

    def estimate_reduction(
        self, memory: HostMemoryProfile, workload: ReductionWorkload
    ) -> CostEstimate:
        return estimate_cpu_reduction(self, memory, workload)

    def estimate_memory(
        self, memory: HostMemoryProfile, workload: MemoryWorkload
    ) -> CostEstimate:
        return estimate_cpu_memory(self, memory, workload)


@dataclass(frozen=True)
class DigitalSramCimProfile:
    """ADC-free digital SRAM-CIM profile.

    ``p_m``, ``p_k`` and ``p_n`` are effective logical dimensions per array
    evaluation at the declared mode; physical spare/ECC columns should already
    be removed.  ``input_parallel_bits`` and ``weight_parallel_bits`` select
    bit-parallel width.  Larger operand widths are evaluated as bit slices.
    """

    array_count: int = 64
    p_m: int = 1
    p_k: int = 128
    p_n: int = 128
    frequency_ghz: float = 1.0
    input_parallel_bits: int = 1
    weight_parallel_bits: int = 1
    cycles_per_eval: int = 1
    weight_capacity_bytes: int = 64 * 1024 * 1024
    max_m_replication: int = 1

    load_bandwidth_gb_s: float = 512.0
    activation_bandwidth_gb_s: float = 1024.0
    output_bandwidth_gb_s: float = 1024.0
    noc_bandwidth_gb_s: float = 2048.0
    accumulator_outputs_per_cycle: float = 1024.0
    peripheral_elements_per_cycle: float = 1024.0
    load_latency_ns: float = 0.0
    noc_hop_latency_ns: float = 0.0
    noc_reduce_fan_in: int = 4
    peripheral_latency_ns: float = 0.0

    accumulator_bits: int = 32
    accumulator_guard_bits: int = 0
    supported_activation_bits: Tuple[int, ...] = (1, 2, 4, 8, 16)
    supported_weight_bits: Tuple[int, ...] = (1, 2, 4, 8, 16)

    eval_energy_pj: float = 0.0
    load_energy_pj_per_byte: float = 0.0
    activation_energy_pj_per_byte: float = 0.0
    output_energy_pj_per_byte: float = 0.0
    noc_energy_pj_per_byte: float = 0.0
    accumulator_energy_pj_per_op: float = 0.0
    peripheral_energy_pj_per_element: float = 0.0

    array_resource_id: str = "cim.array"
    load_resource_id: str = "cim.load"
    activation_resource_id: str = "cim.activation"
    noc_resource_id: str = "cim.noc"
    accumulator_resource_id: str = "cim.accumulator"
    peripheral_resource_id: str = "cim.peripheral"
    name: str = "digital-sram-cim"

    def __post_init__(self) -> None:
        for field_name in (
            "array_count",
            "p_m",
            "p_k",
            "p_n",
            "input_parallel_bits",
            "weight_parallel_bits",
            "cycles_per_eval",
            "weight_capacity_bytes",
            "max_m_replication",
            "noc_reduce_fan_in",
            "accumulator_bits",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        for field_name in (
            "frequency_ghz",
            "load_bandwidth_gb_s",
            "activation_bandwidth_gb_s",
            "output_bandwidth_gb_s",
            "noc_bandwidth_gb_s",
            "accumulator_outputs_per_cycle",
            "peripheral_elements_per_cycle",
        ):
            _require_positive(field_name, getattr(self, field_name))
        _require_non_negative_int(
            "accumulator_guard_bits", self.accumulator_guard_bits
        )
        for field_name in (
            "load_latency_ns",
            "noc_hop_latency_ns",
            "peripheral_latency_ns",
            "eval_energy_pj",
            "load_energy_pj_per_byte",
            "activation_energy_pj_per_byte",
            "output_energy_pj_per_byte",
            "noc_energy_pj_per_byte",
            "accumulator_energy_pj_per_op",
            "peripheral_energy_pj_per_element",
        ):
            _require_non_negative(field_name, getattr(self, field_name))
        if not self.supported_activation_bits or not self.supported_weight_bits:
            raise ValueError("supported bit-width tuples must not be empty")
        for bits in self.supported_activation_bits:
            _require_positive_int("supported activation bits", bits)
        for bits in self.supported_weight_bits:
            _require_positive_int("supported weight bits", bits)

        resource_ids = (
            self.array_resource_id,
            self.load_resource_id,
            self.activation_resource_id,
            self.noc_resource_id,
            self.accumulator_resource_id,
            self.peripheral_resource_id,
        )
        if not self.name or any(not resource_id for resource_id in resource_ids):
            raise ValueError("profile and resource names must not be empty")
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("CIM resource ids must be distinct")

    def estimate_gemm(
        self, workload: GemmWorkload, weights_resident: bool = False
    ) -> CostEstimate:
        return estimate_cim_gemm(self, workload, weights_resident=weights_resident)


def _cache_memory_demands(
    *,
    hierarchy: CacheHierarchyProfile,
    read_bytes: int,
    write_bytes: int,
    working_set_bytes: int,
    reuse_factor: float,
    streaming_fraction: float,
    backing_bandwidth_gb_s: float,
    backing_energy_pj_per_byte: float,
    backing_resource_id: str,
) -> Tuple[Tuple[ResourceDemand, ...], Mapping[str, object]]:
    """Derive deterministic cache-level traffic from an aggregate working set.

    No physical address trace exists in the system simulator.  The v3 cache
    contract therefore exposes the exact analytical assumption: reusable
    traffic is captured in proportion to capacity/working-set, while declared
    streaming traffic bypasses temporal reuse.  Each level reports accessed,
    hit, and miss bytes instead of fabricating cache lines that were never
    observed.
    """

    _require_non_negative_int("read_bytes", read_bytes)
    _require_non_negative_int("write_bytes", write_bytes)
    _require_positive_int("working_set_bytes", max(1, working_set_bytes))
    _require_positive("reuse_factor", reuse_factor)
    if not 0.0 <= streaming_fraction <= 1.0:
        raise ValueError("streaming_fraction must be in [0, 1]")
    _require_positive("backing_bandwidth_gb_s", backing_bandwidth_gb_s)

    incoming = read_bytes + write_bytes
    reusable_fraction = (
        (1.0 - streaming_fraction)
        * max(0.0, 1.0 - 1.0 / float(reuse_factor))
    )
    demands = []
    rows = []
    for level in hierarchy.levels:
        accessed = incoming
        capacity_fraction = min(
            1.0,
            level.capacity_bytes / float(max(1, working_set_bytes)),
        )
        local_hit_ratio = min(1.0, reusable_fraction * capacity_fraction)
        hit_bytes = min(
            accessed, int(math.floor(accessed * local_hit_ratio + 0.5))
        )
        miss_bytes = accessed - hit_bytes
        line_count = _ceil_div(max(0, accessed), level.line_bytes)
        latency_waves = _ceil_div(
            line_count, max(1, level.transaction_parallelism)
        )
        service_ns = max(
            accessed / level.bandwidth_gb_s,
            latency_waves * level.hit_latency_ns,
        )
        demands.append(
            ResourceDemand(
                resource_id=level.resource_id,
                service_ns=service_ns,
                bytes_moved=accessed,
                energy_pj=accessed * level.energy_pj_per_byte,
            )
        )
        rows.append(
            {
                "level": level.name,
                "resource_id": level.resource_id,
                "accessed_bytes": accessed,
                "hit_bytes": hit_bytes,
                "miss_bytes": miss_bytes,
                "hit_ratio": (
                    hit_bytes / float(accessed) if accessed else 0.0
                ),
                "service_ns": service_ns,
            }
        )
        incoming = miss_bytes

    backing_bytes = incoming
    backing_service_ns = backing_bytes / backing_bandwidth_gb_s
    demands.append(
        ResourceDemand(
            resource_id=backing_resource_id,
            service_ns=backing_service_ns,
            bytes_moved=backing_bytes,
            energy_pj=backing_bytes * backing_energy_pj_per_byte,
        )
    )
    return tuple(demands), {
        "cache_model": "v3_working_set_reuse",
        "working_set_bytes": working_set_bytes,
        "reuse_factor": reuse_factor,
        "streaming_fraction": streaming_fraction,
        "levels": tuple(rows),
        "backing_bytes": backing_bytes,
        "backing_service_ns": backing_service_ns,
        "write_back": hierarchy.write_back,
        "write_allocate": hierarchy.write_allocate,
    }


def _gemm_tensor_dtype(workload: GemmWorkload) -> str:
    if workload.activation_bits <= 8 and workload.weight_bits <= 8:
        return "int8"
    if workload.activation_bits <= 16 and workload.weight_bits <= 16:
        return "fp16"
    return "fp32"


def estimate_gpu_gemm(
    gpu: GPUProfile, hbm: HBMProfile, workload: GemmWorkload
) -> CostEstimate:
    """Estimate a tiled tensor-core GEMM through GPU SRAM and HBM."""

    if (
        workload.mmq_work is not None
        and workload.mmq_work.sm_count != gpu.tensor_core.sm_count
    ):
        raise ValueError("mmq_work SM count must match the GPU profile")
    mmq_metadata = (
        {
            "mmq_source_work": workload.mmq_work.to_metadata(),
            "mmq_source_qualified_int8": True,
            "mmq_main_partial_write_bytes": (
                workload.mmq_work.main_partial_write_bytes
            ),
            "mmq_main_partial_write_accounting": (
                "same_matrix_memory_phase_no_extra_launch"
            ),
        }
        if workload.mmq_work is not None
        else {}
    )
    source_dtype_name = _gemm_tensor_dtype(workload)
    quantized_capability = (
        gpu.resolve_quantized_matmul_capability(workload)
        if workload.mmq_work is None else None
    )
    # Keep physical format coverage explicit in every GEMM estimate.  A
    # quantized artifact without a declared backend capability still follows
    # the historical generic roofline, but callers must be able to distinguish
    # that analytical fallback from a source-qualified path.
    if workload.packed_weight_formats:
        if workload.mmq_work is not None:
            format_coverage = "source_qualified_mmq"
        elif quantized_capability is not None:
            format_coverage = "declared_quantized_capability"
        else:
            format_coverage = "generic_quantized_fallback"
    else:
        format_coverage = "unpacked"
    mmq_metadata["quantized_format_coverage"] = format_coverage
    dtype_name = (
        "int8" if workload.mmq_work is not None else
        quantized_capability.tensor_core_dtype.casefold()
        if quantized_capability is not None
        else source_dtype_name
    )
    if workload.mmq_work is not None:
        mmq_metadata.update(
            source_tensor_dtype=source_dtype_name,
            internal_tensor_dtype="int8",
            kernel_family="cuda_mmq",
        )
    tensor_core = gpu.tensor_core
    if dtype_name not in tensor_core.supported_dtypes:
        raise ValueError(
            "GPU tensor core does not support GEMM dtype {}".format(dtype_name)
        )
    m_tile_count = _ceil_div(workload.m, tensor_core.mma_m)
    n_tile_count = _ceil_div(workload.n, tensor_core.mma_n)
    serial_k_tile_count = _ceil_div(workload.k, tensor_core.mma_k)
    tile_count = m_tile_count * n_tile_count * serial_k_tile_count
    issued_operations = tile_count * tensor_core.operations_per_mma
    structural_peak_tops = tensor_core.peak_tops(dtype_name)
    attainable_tops = (
        structural_peak_tops
        * gpu.attainable_efficiency
        * gpu.occupancy
    )
    capability_metadata = (
        {
            "name": quantized_capability.name,
            "kernel_family": quantized_capability.kernel_family,
            "tensor_core_dtype": dtype_name,
            "internal_activation_bits": (
                quantized_capability.internal_activation_bits
            ),
            **({"min_m": quantized_capability.min_m}
               if quantized_capability.min_m != 1 else {}),
            "accumulator_bits": quantized_capability.accumulator_bits,
            "supported_weight_formats": (
                quantized_capability.supported_weight_formats
            ),
            "evidence": quantized_capability.evidence,
            "provenance": quantized_capability.provenance,
        }
        if quantized_capability is not None
        else None
    )
    quantized_path_metadata = (
        {
            "source_tensor_dtype": source_dtype_name,
            "source_activation_bits": workload.activation_bits,
            "source_weight_bits": workload.weight_bits,
            "accumulator_bits": workload.accumulator_bits,
            "internal_tensor_dtype": dtype_name,
            "structural_peak_tops": structural_peak_tops,
            "quantized_matmul_capability": capability_metadata,
            "kernel_family": quantized_capability.kernel_family,
        }
        if quantized_capability is not None
        else {}
    )
    compute_ns = issued_operations / (attainable_tops * 1000.0)
    epilogue_scalar_ns = (
        workload.epilogue_operations / gpu.elementwise_gops
        if workload.epilogue_operations > 0
        else 0.0
    )
    epilogue_sfu_ns = (
        workload.epilogue_transcendental_operations
        / gpu.special_function_gops
        if workload.epilogue_transcendental_operations > 0
        else 0.0
    )
    # Packed conversion and the scalar epilogue share an execution resource.
    # Account here so placement estimates and serving use the same cost.
    transform_ns = workload.packed_weight_transform_operations / gpu.elementwise_gops
    scalar_ns = (
        epilogue_scalar_ns
        + transform_ns
        + workload.source_partial_service_ns
    )
    transform_metadata = (
        {
            "dequant_execution_model": "fused_quantized_dot",
            "fused_dequant_operations": workload.packed_weight_transform_operations,
            "fused_dequant_service_ns": transform_ns,
            "fused_dequant_accounting": "shared_gemm_scalar_demand",
            "fused_dequant_service_merge": "scalar_sum",
        }
        if workload.packed_weight_transform_operations else {}
    )
    partial_metadata = (
        {
            "source_partial_name": workload.source_partial_name,
            "source_partial_service_ns": workload.source_partial_service_ns,
            "source_partial_work_units": workload.source_partial_work_units,
            "source_partial_accounting": "shared_gpu_scalar_resource",
            "source_partial_timing_completeness": "partial",
        }
        if workload.source_partial_service_ns > 0.0
        else {}
    )
    # GemmWorkload.minimum_io_bytes already counts each input, packed RHS,
    # metadata byte, and output exactly once.  Those are compulsory backing
    # accesses for a stateless kernel estimate: arithmetic intensity describes
    # work performed per byte, not evidence that the byte was resident in an
    # on-chip cache before the kernel started.  Applying the generic cache
    # reuse curve here used to erase a large fraction of quantized GEMV weight
    # traffic, most noticeably at M=1.  Intra-kernel tile reuse is already
    # reflected by counting the RHS once rather than M times.
    reuse_factor = 1.0
    streaming_fraction = 1.0
    independent_output_tile_count = m_tile_count * n_tile_count
    warp_equivalent_count = (
        tensor_core.sm_count * tensor_core.tensor_cores_per_sm
    )
    resident_warp_equivalent_capacity = (
        warp_equivalent_count * gpu.occupancy
    )
    parallel_tile_slots = max(
        1,
        int(math.floor(resident_warp_equivalent_capacity)),
    )
    output_tile_wave_count = _ceil_div(
        independent_output_tile_count, parallel_tile_slots
    )
    output_tile_wave_utilization = independent_output_tile_count / float(
        output_tile_wave_count * parallel_tile_slots
    )
    peak_effective_hbm_bandwidth_gb_s = hbm.effective_bandwidth_gb_s
    shape_effective_hbm_bandwidth_gb_s = (
        peak_effective_hbm_bandwidth_gb_s
        * output_tile_wave_utilization
    )
    hbm_bandwidth_metadata = {
        "model": "mma_output_tile_wave_proxy_v1",
        "fallback_to_fixed_bandwidth": False,
        "m_tile_count": m_tile_count,
        "n_tile_count": n_tile_count,
        "serial_k_tile_count": serial_k_tile_count,
        "independent_output_tile_count": independent_output_tile_count,
        "warp_equivalent_count": warp_equivalent_count,
        "parallel_tile_slots": parallel_tile_slots,
        "resident_warp_equivalent_capacity": (
            resident_warp_equivalent_capacity
        ),
        "output_tile_wave_count": output_tile_wave_count,
        "output_tile_wave_utilization": output_tile_wave_utilization,
        "profile_occupancy": gpu.occupancy,
        "peak_effective_hbm_bandwidth_gb_s": (
            peak_effective_hbm_bandwidth_gb_s
        ),
        "shape_effective_hbm_bandwidth_gb_s": (
            shape_effective_hbm_bandwidth_gb_s
        ),
        "tile_utilization_applied_to_hbm": False,
        "parallelism_basis": (
            "independent MxN output tiles; K tiles are serial reduction"
        ),
    }
    memory_demands, cache_metadata = _cache_memory_demands(
        hierarchy=gpu.cache_hierarchy,
        read_bytes=workload.activation_bytes + workload.weight_bytes,
        write_bytes=(
            workload.output_bytes
            + (
                workload.mmq_work.main_partial_write_bytes
                if workload.mmq_work is not None else 0
            )
        ),
        working_set_bytes=(
            workload.minimum_io_bytes
            + (
                workload.mmq_work.main_partial_write_bytes
                if workload.mmq_work is not None else 0
            )
        ),
        reuse_factor=reuse_factor,
        streaming_fraction=streaming_fraction,
        backing_bandwidth_gb_s=shape_effective_hbm_bandwidth_gb_s,
        backing_energy_pj_per_byte=hbm.energy_pj_per_byte,
        backing_resource_id=hbm.resource_id,
    )
    memory_ns = max(
        (demand.service_ns for demand in memory_demands), default=0.0
    )

    phases = []
    if gpu.kernel_launch_ns > 0.0 or gpu.launch_energy_pj > 0.0:
        phases.append(
            CostPhase(
                name="kernel_launch",
                category=TaskCategory.COMPUTE,
                demands=(
                    ResourceDemand(
                        resource_id=gpu.launch_resource_id,
                        service_ns=gpu.kernel_launch_ns,
                        energy_pj=gpu.launch_energy_pj,
                    ),
                ),
                metadata={
                    "operator_class": OperatorClass.GEMM.value,
                    "evidence": EvidenceStatus.ANALYTICAL.value,
                },
            )
        )

    roofline_ns = max(
        compute_ns, scalar_ns, epilogue_sfu_ns, memory_ns
    )
    compute_demands = [
        ResourceDemand(
            resource_id=tensor_core.resource_id,
            service_ns=compute_ns,
            energy_pj=(issued_operations * gpu.tensor_energy_pj_per_op),
            work_units=float(workload.operations),
        )
    ]
    if (
        workload.epilogue_operations > 0
        or workload.packed_weight_transform_operations > 0
        or workload.source_partial_service_ns > 0.0
    ):
        compute_demands.append(
            ResourceDemand(
                resource_id=gpu.scalar_resource_id,
                service_ns=scalar_ns,
                energy_pj=(
                    (workload.epilogue_operations + workload.packed_weight_transform_operations)
                    * gpu.scalar_energy_pj_per_op
                ),
                work_units=float(
                    workload.epilogue_operations
                    + workload.packed_weight_transform_operations
                    + workload.source_partial_work_units
                ),
            )
        )
    if workload.epilogue_transcendental_operations > 0:
        compute_demands.append(
            ResourceDemand(
                resource_id=gpu.special_function_resource_id,
                service_ns=epilogue_sfu_ns,
                energy_pj=(
                    workload.epilogue_transcendental_operations
                    * gpu.special_function_energy_pj_per_op
                ),
                work_units=float(
                    workload.epilogue_transcendental_operations
                ),
            )
        )
    phases.append(
        CostPhase(
            name="gpu_gemm",
            category=TaskCategory.COMPUTE,
            demands=tuple(compute_demands) + memory_demands,
            metadata={
                "operator_class": OperatorClass.GEMM.value,
                "evidence": EvidenceStatus.ANALYTICAL.value,
                "compute_service_ns": compute_ns,
                "memory_service_ns": memory_ns,
                **transform_metadata,
                **partial_metadata,
                "tensor_dtype": dtype_name,
                **quantized_path_metadata,
                "tile_count": tile_count,
                "issued_operations": issued_operations,
                "epilogue_name": workload.epilogue_name,
                "epilogue_operations": workload.epilogue_operations,
                "epilogue_transcendental_operations": (
                    workload.epilogue_transcendental_operations
                ),
                "epilogue_output_elements": workload.epilogue_output_elements,
                "epilogue_scalar_service_ns": epilogue_scalar_ns,
                "epilogue_special_function_service_ns": epilogue_sfu_ns,
                "fusion_group": (
                    "gemm_epilogue"
                    if workload.epilogue_operations
                    or workload.epilogue_transcendental_operations
                    else ""
                ),
                "cache": cache_metadata,
                "hbm_bandwidth": hbm_bandwidth_metadata,
                "memory_traffic_semantics": (
                    "source_unique_and_partial_write_inherited_stateless_mapping"
                    if workload.mmq_work is not None else "compulsory_minimum_io"
                ),
                **mmq_metadata,
            },
        )
    )

    bound = max(
        (
            (compute_ns, "tensor_core"),
            (scalar_ns, "scalar"),
            (epilogue_sfu_ns, "special_function"),
            (memory_ns, "memory"),
        ),
        key=lambda item: (item[0], item[1]),
    )[1]
    tile_utilization = workload.operations / float(issued_operations)
    compute_utilization = (
        tile_utilization * compute_ns / roofline_ns if roofline_ns else 0.0
    )
    return CostEstimate(
        phases=tuple(phases),
        useful_ops=(
            workload.operations
            + workload.epilogue_operations
            + workload.epilogue_transcendental_operations
        ),
        utilization=compute_utilization,
        metadata={
            "model": "gpu_hbm_roofline",
            "profile": gpu.name,
            "operator_class": OperatorClass.GEMM.value,
            "evidence": EvidenceStatus.ANALYTICAL.value,
            "bound": bound,
            "compute_service_ns": compute_ns,
            "epilogue_scalar_service_ns": epilogue_scalar_ns,
            "epilogue_special_function_service_ns": epilogue_sfu_ns,
            "memory_service_ns": memory_ns,
            "roofline_service_ns": roofline_ns,
            **transform_metadata,
            "attainable_tops": attainable_tops,
            "tensor_dtype": dtype_name,
            **quantized_path_metadata,
            "mma_shape": (
                tensor_core.mma_m,
                tensor_core.mma_n,
                tensor_core.mma_k,
            ),
            "tile_count": tile_count,
            "issued_operations": issued_operations,
            "tile_utilization": tile_utilization,
            "activation_bytes": workload.activation_bytes,
            "weight_bytes": workload.weight_bytes,
            "output_bytes": workload.output_bytes,
            "minimum_io_bytes": workload.minimum_io_bytes,
            **mmq_metadata,
            "epilogue_name": workload.epilogue_name,
            "epilogue_operations": workload.epilogue_operations,
            "epilogue_transcendental_operations": (
                workload.epilogue_transcendental_operations
            ),
            "epilogue_output_elements": workload.epilogue_output_elements,
            "fusion_group": (
                "gemm_epilogue"
                if workload.epilogue_operations
                or workload.epilogue_transcendental_operations
                else ""
            ),
            "cache": cache_metadata,
            "hbm_bandwidth": hbm_bandwidth_metadata,
        },
    )


def estimate_gpu_tensor_kernel(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: TensorKernelWorkload,
) -> CostEstimate:
    """Estimate a generic non-GEMM GPU kernel on scalar/SFU engines."""

    if workload.launch_only:
        # A source-required launch may have no data work (MMQ fixup with P=0).
        # Preserve the launch even when its declared profile service is zero;
        # do not invent one arithmetic op just to satisfy the normal roofline.
        audit = {
            "device": "gpu", "profile": gpu.name, "workload": workload.name,
            "operator_class": OperatorClass.ELEMENTWISE.value,
            "launch_only": True, "operations": 0, "read_bytes": 0, "write_bytes": 0,
            "timing_completeness": "launch_only_control_work_unpriced",
        }
        return CostEstimate(
            phases=(CostPhase(
                name="kernel_launch", category=TaskCategory.COMPUTE,
                demands=(ResourceDemand(
                    resource_id=gpu.launch_resource_id, service_ns=gpu.kernel_launch_ns,
                    energy_pj=gpu.launch_energy_pj,
                ),), metadata=audit,
            ),), useful_ops=0, utilization=0.0, metadata=audit,
        )
    return _estimate_typed_roofline(
        device_kind="gpu",
        profile_name=gpu.name,
        operator_class=OperatorClass.ELEMENTWISE,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=(
            float(gpu.elementwise_gops)
            if workload.operations > 0
            else None
        ),
        memory_bandwidth_gb_s=hbm.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=gpu.scalar_energy_pj_per_op,
        memory_energy_pj_per_byte=hbm.energy_pj_per_byte,
        compute_resource_id=gpu.scalar_resource_id,
        memory_resource_id=hbm.resource_id,
        dispatch_resource_id=gpu.launch_resource_id,
        dispatch_name="kernel_launch",
        dispatch_ns=gpu.kernel_launch_ns,
        dispatch_energy_pj=gpu.launch_energy_pj,
        cache_hierarchy=gpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        special_function_operations=workload.transcendental_operations,
        special_function_throughput_gops=gpu.special_function_gops,
        special_function_energy_pj_per_op=(
            gpu.special_function_energy_pj_per_op
        ),
        special_function_resource_id=gpu.special_function_resource_id,
        dependency_depth=workload.dependency_depth,
        frequency_ghz=gpu.tensor_core.frequency_ghz,
    )


def _estimate_gpu_q4_mma_materialized_attention(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: FusedAttentionWorkload,
) -> CostEstimate:
    """Lower the two source-required Q4_0 -> F16 producers before MMA.

    b10760 convert.cu:85-109,286-289: 32 threads produce 256 values;
    each thread shares dm=-8*d over eight d*q+dm expressions.  Count
    17E/8 source FLOPs, not SASS instructions or unmeasured cast work.
    """

    view_tokens = workload.q4_mma_view_tokens_lower_bound
    elements = workload.effective_kv_hidden_size * view_tokens
    conversion_workload = TensorKernelWorkload(
        operations=17 * elements // 8,
        read_bytes=18 * elements // 32,
        write_bytes=2 * elements,
        streaming_fraction=1.0,
        name="q4_0_to_f16_cache_view",
    )
    conversion = estimate_gpu_tensor_kernel(gpu, hbm, conversion_workload)
    # b10760 fattn-common.cuh:1094 and fattn-mma-f16.cuh:1815: below
    # 1024 query columns in one stream, masking happens after rectangular QK.
    # The allocator supplies only a view lower bound, not an exact high-water mark.
    rectangular_scan = workload.batch_tokens < 1024
    consumer_context = (
        max(workload.context_tokens, view_tokens)
        if rectangular_scan else workload.context_tokens
    )
    consumer_workload = replace(
        workload,
        q4_mma_view_tokens_lower_bound=0,
        q4_mma_head_dim=0,
        kv_input_bits=16,
        kv_physical_contract=None,
        kv_read_tokens=view_tokens,
        context_tokens=consumer_context,
    )
    consumer = estimate_gpu_fused_attention(gpu, hbm, consumer_workload)
    audit = {
        "q4_mma_materialization_applied": True,
        "materialization_view_tokens_lower_bound": view_tokens,
        "materialization_head_dim": workload.q4_mma_head_dim,
        "materialization_extent_completeness": "lower_bound",
        "materialization_timing_completeness": "partial",
        "materialization_source": "llama.cpp_b10760_fattn_common_1022_1083_convert_85_109",
        "materialization_view_layout": "unified_contiguously_allocated_independent_k_v",
        "materialization_memory_price": "existing_streaming_backing_memory_approximation",
        "materialization_unpriced_work": "bit_unpack_casts_register_residency_and_dependencies",
        "materialization_elements_per_operand": elements,
        "materialization_read_bytes_per_operand": conversion_workload.read_bytes,
        "materialization_write_bytes_per_operand": conversion_workload.write_bytes,
        "materialization_float_operations_per_operand": conversion_workload.operations,
        "materialization_grid_blocks_per_operand": _ceil_div(elements, 256),
        "materialization_threads_per_block": 32,
        "materialization_kernel_count": 2,
        "materialization_removed_legacy_dequant_operations": workload.kv_dequant_operations,
        "materialization_input_artifact_format": "Q4_0",
        "materialization_consumer_logical_context_tokens": workload.context_tokens,
        "materialization_consumer_context_tokens": consumer_context,
        "materialization_consumer_rectangular_scan_applied": rectangular_scan,
        "materialization_consumer_compute_extent": (
            "occupied_rows_rectangular_lower_bound" if rectangular_scan
            else "legacy_logical_context_mask_trim_unknown"
        ),
        "physical_kernel_count": 3,
    }
    phases = []
    for operand in ("k", "v"):
        phases.extend(
            replace(
                phase,
                name="q4_{}_materialize.{}".format(operand, phase.name),
                metadata={
                    **dict(phase.metadata),
                    **audit,
                    "materialization_operand": operand,
                    "kernel_phase": phase.name,
                },
            )
            for phase in conversion.phases
        )
    phases.extend(
        replace(phase, metadata={**dict(phase.metadata), **audit})
        for phase in consumer.phases
    )
    return CostEstimate(
        phases=tuple(phases),
        useful_ops=consumer.useful_ops + 2 * conversion.useful_ops,
        utilization=consumer.utilization,
        metadata={
            **dict(consumer.metadata),
            **audit,
            "model": "gpu_fused_attention_q4_mma_materialization_v1",
            "bound": "sequential_materialization_and_attention",
            "roofline_service_ns": (
                float(consumer.metadata["roofline_service_ns"])
                + 2 * sum(phase.service_ns for phase in conversion.phases
                          if phase.name != "kernel_launch")
            ),
            "read_bytes": consumer_workload.read_bytes + 2 * conversion_workload.read_bytes,
            "write_bytes": consumer_workload.write_bytes + 2 * conversion_workload.write_bytes,
            "minimum_io_bytes": (
                consumer_workload.read_bytes + consumer_workload.write_bytes
                + 2 * conversion_workload.minimum_io_bytes
            ),
        },
    )


def estimate_gpu_fused_attention(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: FusedAttentionWorkload,
) -> CostEstimate:
    """Estimate one audited FlashAttention-style fused GPU kernel."""

    if workload.q4_mma_view_tokens_lower_bound:
        return _estimate_gpu_q4_mma_materialized_attention(gpu, hbm, workload)

    tensor_core = gpu.tensor_core
    if workload.input_bits <= 8:
        dtype_name = "int8"
    elif workload.input_bits <= 16:
        dtype_name = "fp16"
    else:
        dtype_name = "fp32"
    if dtype_name not in tensor_core.supported_dtypes:
        raise ValueError(
            "GPU tensor core does not support attention dtype {}".format(
                dtype_name
            )
        )

    qk_tiles = (
        _ceil_div(workload.batch_tokens, tensor_core.mma_m)
        * _ceil_div(workload.context_tokens, tensor_core.mma_n)
        * _ceil_div(workload.hidden_size, tensor_core.mma_k)
    )
    pv_tiles = (
        _ceil_div(workload.batch_tokens, tensor_core.mma_m)
        * _ceil_div(workload.hidden_size, tensor_core.mma_n)
        * _ceil_div(workload.context_tokens, tensor_core.mma_k)
    )
    issued_tensor_operations = (
        qk_tiles + pv_tiles
    ) * tensor_core.operations_per_mma
    attainable_tops = (
        tensor_core.peak_tops(dtype_name)
        * gpu.attainable_efficiency
        * gpu.occupancy
    )
    tensor_ns = issued_tensor_operations / (attainable_tops * 1000.0)
    scalar_ns = max(
        workload.scalar_operations / gpu.elementwise_gops,
        workload.dependency_depth / tensor_core.frequency_ghz,
    )
    special_function_ns = (
        workload.transcendental_operations / gpu.special_function_gops
        if workload.transcendental_operations > 0
        else 0.0
    )
    minimum_io_bytes = workload.read_bytes + workload.write_bytes
    memory_demands, cache_metadata = _cache_memory_demands(
        hierarchy=gpu.cache_hierarchy,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        working_set_bytes=workload.onchip_working_set_bytes,
        # Fused attention streams each invocation's Q, persisted physical KV,
        # KV metadata, and O exactly once from backing memory.  Tile reuse only
        # keeps score/softmax/partial accumulators on chip; it must not erase
        # the compulsory persisted-cache bytes.
        reuse_factor=1.0,
        streaming_fraction=1.0,
        backing_bandwidth_gb_s=hbm.effective_bandwidth_gb_s,
        backing_energy_pj_per_byte=hbm.energy_pj_per_byte,
        backing_resource_id=hbm.resource_id,
    )
    memory_ns = max(
        (demand.service_ns for demand in memory_demands), default=0.0
    )
    phases = []
    if gpu.kernel_launch_ns > 0.0 or gpu.launch_energy_pj > 0.0:
        phases.append(
            CostPhase(
                name="kernel_launch",
                category=TaskCategory.COMPUTE,
                demands=(
                    ResourceDemand(
                        resource_id=gpu.launch_resource_id,
                        service_ns=gpu.kernel_launch_ns,
                        energy_pj=gpu.launch_energy_pj,
                    ),
                ),
                metadata={
                    "operator_class": OperatorClass.GEMM.value,
                    "evidence": EvidenceStatus.ANALYTICAL.value,
                    "fusion_group": "qk_softmax_pv",
                },
            )
        )

    compute_demands = [
        ResourceDemand(
            resource_id=tensor_core.resource_id,
            service_ns=tensor_ns,
            energy_pj=(
                issued_tensor_operations * gpu.tensor_energy_pj_per_op
            ),
            work_units=float(workload.tensor_operations),
        ),
        ResourceDemand(
            resource_id=gpu.scalar_resource_id,
            service_ns=scalar_ns,
            energy_pj=(
                workload.scalar_operations * gpu.scalar_energy_pj_per_op
            ),
            work_units=float(workload.scalar_operations),
        ),
    ]
    if workload.transcendental_operations > 0:
        compute_demands.append(
            ResourceDemand(
                resource_id=gpu.special_function_resource_id,
                service_ns=special_function_ns,
                energy_pj=(
                    workload.transcendental_operations
                    * gpu.special_function_energy_pj_per_op
                ),
                work_units=float(workload.transcendental_operations),
            )
        )
    roofline_ns = max(
        tensor_ns, scalar_ns, special_function_ns, memory_ns
    )
    phase_metadata = {
        "operator_class": OperatorClass.GEMM.value,
        "evidence": EvidenceStatus.ANALYTICAL.value,
        "fusion_group": "qk_softmax_pv",
        "fusion_model": "flash_attention_tiled_online_softmax",
        "tensor_service_ns": tensor_ns,
        "scalar_service_ns": scalar_ns,
        "special_function_service_ns": special_function_ns,
        "memory_service_ns": memory_ns,
        "score_matrix_bytes_elided": workload.score_matrix_bytes_elided,
        "onchip_working_set_bytes": workload.onchip_working_set_bytes,
        "query_hidden_size": workload.hidden_size,
        **(
            {
                "effective_query_storage_bits": (
                    workload.effective_query_storage_bits
                )
            }
            if workload.query_storage_bits is not None
            else {}
        ),
        "score_heads": workload.score_heads,
        "score_elements": workload.score_elements,
        "qk_scale": workload.qk_scale,
        "query_read_bytes": workload.query_read_bytes,
        "kv_hidden_size": workload.effective_kv_hidden_size,
        "kv_input_bits": workload.effective_kv_input_bits,
        "kv_read_tokens": workload.effective_kv_read_tokens,
        "kv_read_bytes": workload.kv_read_bytes,
        "kv_payload_bytes_per_token": workload.kv_payload_bytes_per_token,
        "kv_metadata_bytes_per_token": workload.kv_metadata_bytes_per_token,
        "kv_scale_bytes_per_token": workload.kv_metadata_bytes_per_token,
        "kv_dequant_operations_per_token": (
            workload.kv_dequant_operations_per_token
        ),
        "kv_payload_bytes": workload.kv_payload_bytes,
        "kv_metadata_bytes": workload.kv_metadata_bytes,
        "kv_scale_bytes": workload.kv_metadata_bytes,
        "kv_dequant_operations": workload.kv_dequant_operations,
        "kv_artifact_format": workload.kv_artifact_format,
        "kv_physical_contract_applied": (
            workload.kv_physical_contract is not None
        ),
        "softmax_scalar_operations": workload.softmax_scalar_operations,
        "qk_scale_operations": workload.qk_scale_operations,
        "memory_traffic_semantics": "compulsory_minimum_io",
        "cache": cache_metadata,
    }
    phases.append(
        CostPhase(
            name="gpu_fused_attention",
            category=TaskCategory.COMPUTE,
            demands=tuple(compute_demands) + memory_demands,
            metadata=phase_metadata,
        )
    )
    bound = max(
        (
            (tensor_ns, "tensor_core"),
            (scalar_ns, "scalar"),
            (special_function_ns, "special_function"),
            (memory_ns, "memory"),
        ),
        key=lambda item: (item[0], item[1]),
    )[1]
    total_useful_operations = (
        workload.tensor_operations
        + workload.scalar_operations
        + workload.transcendental_operations
    )
    return CostEstimate(
        phases=tuple(phases),
        useful_ops=total_useful_operations,
        utilization=(
            workload.tensor_operations
            / float(max(1, issued_tensor_operations))
            * tensor_ns
            / roofline_ns
            if roofline_ns
            else 0.0
        ),
        metadata={
            "model": "gpu_fused_attention_v3",
            "profile": gpu.name,
            "kernel": workload.name,
            "operator_class": OperatorClass.GEMM.value,
            "evidence": EvidenceStatus.ANALYTICAL.value,
            "bound": bound,
            "roofline_service_ns": roofline_ns,
            "tensor_dtype": dtype_name,
            "tensor_operations": workload.tensor_operations,
            "issued_tensor_operations": issued_tensor_operations,
            "qk_tile_count": qk_tiles,
            "pv_tile_count": pv_tiles,
            "scalar_operations": workload.scalar_operations,
            "special_function_operations": (
                workload.transcendental_operations
            ),
            "read_bytes": workload.read_bytes,
            "write_bytes": workload.write_bytes,
            "minimum_io_bytes": minimum_io_bytes,
            **phase_metadata,
        },
    )


def _estimate_typed_roofline(
    *,
    device_kind: str,
    profile_name: str,
    operator_class: OperatorClass,
    workload_name: str,
    operations: int,
    read_bytes: int,
    write_bytes: int,
    compute_throughput_gops: Optional[float],
    memory_bandwidth_gb_s: float,
    compute_energy_pj_per_op: float,
    additional_compute_energy_pj: float = 0.0,
    memory_energy_pj_per_byte: float,
    compute_resource_id: str,
    memory_resource_id: str,
    dispatch_resource_id: Optional[str] = None,
    dispatch_name: str,
    dispatch_ns: float,
    dispatch_energy_pj: float,
    cache_hierarchy: Optional[CacheHierarchyProfile] = None,
    working_set_bytes: int = 0,
    reuse_factor: float = 1.0,
    streaming_fraction: float = 0.0,
    special_function_operations: int = 0,
    special_function_throughput_gops: Optional[float] = None,
    special_function_energy_pj_per_op: float = 0.0,
    special_function_resource_id: Optional[str] = None,
    dependency_depth: int = 1,
    frequency_ghz: Optional[float] = None,
    compute_service_override_ns: Optional[float] = None,
    compute_work_units: Optional[float] = None,
    instruction_metadata: Optional[Mapping[str, object]] = None,
) -> CostEstimate:
    """Build one typed roofline stage without double-counting elapsed time.

    GOP/s is operations/ns and decimal GB/s is bytes/ns.  Compute and memory
    demands are placed in the same :class:`CostPhase`, so the phase service is
    their maximum rather than their sum.
    """

    _require_non_negative_int("operations", operations)
    _require_non_negative_int("read_bytes", read_bytes)
    _require_non_negative_int("write_bytes", write_bytes)
    _require_positive("memory_bandwidth_gb_s", memory_bandwidth_gb_s)
    _require_non_negative(
        "compute_energy_pj_per_op", compute_energy_pj_per_op
    )
    _require_non_negative(
        "additional_compute_energy_pj", additional_compute_energy_pj
    )
    _require_non_negative(
        "memory_energy_pj_per_byte", memory_energy_pj_per_byte
    )
    _require_non_negative("dispatch_ns", dispatch_ns)
    _require_non_negative("dispatch_energy_pj", dispatch_energy_pj)
    _require_non_negative_int(
        "special_function_operations", special_function_operations
    )
    _require_positive_int("dependency_depth", dependency_depth)
    if operations > 0:
        if compute_throughput_gops is None:
            raise ValueError("compute throughput is required for arithmetic")
        _require_positive(
            "compute_throughput_gops", compute_throughput_gops
        )
    if not workload_name or not profile_name or not device_kind:
        raise ValueError("model and workload names must not be empty")
    if not compute_resource_id or not memory_resource_id:
        raise ValueError("resource ids must not be empty")
    effective_dispatch_resource_id = (
        dispatch_resource_id or compute_resource_id
    )
    if not effective_dispatch_resource_id:
        raise ValueError("dispatch resource id must not be empty")
    if special_function_operations > 0:
        if special_function_throughput_gops is None:
            raise ValueError("special-function throughput is required")
        _require_positive(
            "special_function_throughput_gops",
            special_function_throughput_gops,
        )
        if not special_function_resource_id:
            raise ValueError("special-function resource id is required")
        if special_function_resource_id == compute_resource_id:
            raise ValueError("scalar and special-function resources must differ")
    _require_non_negative(
        "special_function_energy_pj_per_op",
        special_function_energy_pj_per_op,
    )
    if frequency_ghz is not None:
        _require_positive("frequency_ghz", frequency_ghz)
    if compute_service_override_ns is not None:
        _require_non_negative(
            "compute_service_override_ns", compute_service_override_ns
        )
    if compute_work_units is not None:
        _require_non_negative("compute_work_units", compute_work_units)

    bytes_moved = read_bytes + write_bytes
    has_pipeline_service = float(compute_service_override_ns or 0.0) > 0.0
    if operations == 0 and bytes_moved == 0 and not has_pipeline_service:
        raise ValueError("typed workload must declare operations or bytes")
    if (
        operations > 0
        and bytes_moved > 0
        and compute_resource_id == memory_resource_id
    ):
        raise ValueError("compute and memory resource ids must be distinct")

    throughput_compute_ns = (
        operations / float(compute_throughput_gops)
        if operations > 0
        else 0.0
    )
    dependency_ns = (
        dependency_depth / float(frequency_ghz)
        if operations > 0 and frequency_ghz is not None
        else 0.0
    )
    compute_ns = max(
        throughput_compute_ns,
        dependency_ns,
        float(compute_service_override_ns or 0.0),
    )
    special_function_ns = (
        special_function_operations
        / float(special_function_throughput_gops)
        if special_function_operations > 0
        and special_function_throughput_gops is not None
        else 0.0
    )
    if cache_hierarchy is not None and bytes_moved > 0:
        memory_demands, cache_metadata = _cache_memory_demands(
            hierarchy=cache_hierarchy,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            working_set_bytes=max(1, working_set_bytes or bytes_moved),
            reuse_factor=reuse_factor,
            streaming_fraction=streaming_fraction,
            backing_bandwidth_gb_s=memory_bandwidth_gb_s,
            backing_energy_pj_per_byte=memory_energy_pj_per_byte,
            backing_resource_id=memory_resource_id,
        )
    else:
        backing_service_ns = bytes_moved / memory_bandwidth_gb_s
        memory_demands = (
            ResourceDemand(
                resource_id=memory_resource_id,
                service_ns=backing_service_ns,
                bytes_moved=bytes_moved,
                energy_pj=bytes_moved * memory_energy_pj_per_byte,
            ),
        ) if bytes_moved > 0 else ()
        cache_metadata = {
            "cache_model": "none",
            "backing_bytes": bytes_moved,
            "backing_service_ns": backing_service_ns,
        }
    memory_ns = max(
        (demand.service_ns for demand in memory_demands), default=0.0
    )
    roofline_ns = max(compute_ns, special_function_ns, memory_ns)
    metadata = {
        "operator_class": operator_class.value,
        "evidence": EvidenceStatus.ANALYTICAL.value,
        "compute_service_ns": compute_ns,
        "throughput_compute_service_ns": throughput_compute_ns,
        "dependency_service_ns": dependency_ns,
        "special_function_service_ns": special_function_ns,
        "special_function_operations": special_function_operations,
        "memory_service_ns": memory_ns,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "cache": cache_metadata,
        "instruction_schedule": dict(instruction_metadata or {}),
    }

    phases = []
    if dispatch_ns > 0.0 or dispatch_energy_pj > 0.0:
        phases.append(
            CostPhase(
                name=dispatch_name,
                category=TaskCategory.COMPUTE,
                demands=(
                    ResourceDemand(
                        resource_id=effective_dispatch_resource_id,
                        service_ns=dispatch_ns,
                        energy_pj=dispatch_energy_pj,
                    ),
                ),
                metadata={
                    "operator_class": operator_class.value,
                    "evidence": EvidenceStatus.ANALYTICAL.value,
                },
            )
        )

    demands = []
    if operations > 0 or has_pipeline_service:
        demands.append(
            ResourceDemand(
                resource_id=compute_resource_id,
                service_ns=compute_ns,
                energy_pj=(
                    operations * compute_energy_pj_per_op
                    + additional_compute_energy_pj
                ),
                work_units=float(
                    operations
                    if compute_work_units is None
                    else compute_work_units
                ),
            )
        )
    if special_function_operations > 0:
        demands.append(
            ResourceDemand(
                resource_id=str(special_function_resource_id),
                service_ns=special_function_ns,
                energy_pj=(
                    special_function_operations
                    * special_function_energy_pj_per_op
                ),
                work_units=float(special_function_operations),
            )
        )
    demands.extend(memory_demands)
    phases.append(
        CostPhase(
            name="%s_%s" % (device_kind, operator_class.value),
            category=(
                TaskCategory.MEMORY
                if operator_class == OperatorClass.MEMORY
                else TaskCategory.COMPUTE
            ),
            demands=tuple(demands),
            metadata=metadata,
        )
    )

    bound = max(
        (
            (compute_ns, "compute"),
            (special_function_ns, "special_function"),
            (memory_ns, "memory"),
        ),
        key=lambda item: (item[0], item[1]),
    )[1]
    estimate_metadata = dict(metadata)
    estimate_metadata.update(
        {
            "model": "%s_%s_roofline"
            % (device_kind, operator_class.value),
            "profile": profile_name,
            "kernel": workload_name,
            "bound": bound,
            "operations": operations,
            "roofline_service_ns": roofline_ns,
            "throughput_gops": compute_throughput_gops,
            "memory_bandwidth_gb_s": memory_bandwidth_gb_s,
            "special_function_throughput_gops": (
                special_function_throughput_gops
            ),
        }
    )
    return CostEstimate(
        phases=tuple(phases),
        useful_ops=operations,
        utilization=(compute_ns / roofline_ns if roofline_ns else 0.0),
        metadata=estimate_metadata,
    )


def estimate_gpu_elementwise(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: ElementwiseWorkload,
) -> CostEstimate:
    """Estimate a GPU element-wise kernel using scalar GOP/s throughput."""

    return _estimate_typed_roofline(
        device_kind="gpu",
        profile_name=gpu.name,
        operator_class=OperatorClass.ELEMENTWISE,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=float(gpu.elementwise_gops),
        memory_bandwidth_gb_s=hbm.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=float(
            gpu.elementwise_energy_pj_per_op
        ),
        memory_energy_pj_per_byte=hbm.energy_pj_per_byte,
        compute_resource_id=gpu.scalar_resource_id,
        memory_resource_id=hbm.resource_id,
        dispatch_resource_id=gpu.launch_resource_id,
        dispatch_name="kernel_launch",
        dispatch_ns=gpu.kernel_launch_ns,
        dispatch_energy_pj=gpu.launch_energy_pj,
        cache_hierarchy=gpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        special_function_operations=workload.transcendental_operations,
        special_function_throughput_gops=gpu.special_function_gops,
        special_function_energy_pj_per_op=(
            gpu.special_function_energy_pj_per_op
        ),
        special_function_resource_id=gpu.special_function_resource_id,
        dependency_depth=workload.dependency_depth,
        frequency_ghz=gpu.tensor_core.frequency_ghz,
    )


def estimate_gpu_reduction(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: ReductionWorkload,
) -> CostEstimate:
    """Estimate a GPU reduction using its dedicated reduction throughput."""

    return _estimate_typed_roofline(
        device_kind="gpu",
        profile_name=gpu.name,
        operator_class=OperatorClass.REDUCTION,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=float(gpu.reduction_gops),
        memory_bandwidth_gb_s=hbm.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=float(gpu.reduction_energy_pj_per_op),
        memory_energy_pj_per_byte=hbm.energy_pj_per_byte,
        compute_resource_id=gpu.scalar_resource_id,
        memory_resource_id=hbm.resource_id,
        dispatch_resource_id=gpu.launch_resource_id,
        dispatch_name="kernel_launch",
        dispatch_ns=gpu.kernel_launch_ns,
        dispatch_energy_pj=gpu.launch_energy_pj,
        cache_hierarchy=gpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        dependency_depth=workload.dependency_depth,
        frequency_ghz=gpu.tensor_core.frequency_ghz,
    )


def estimate_gpu_memory(
    gpu: GPUProfile,
    hbm: HBMProfile,
    workload: MemoryWorkload,
) -> CostEstimate:
    """Estimate a pure GPU HBM data-movement kernel."""

    return _estimate_typed_roofline(
        device_kind="gpu",
        profile_name=gpu.name,
        operator_class=OperatorClass.MEMORY,
        workload_name=workload.name,
        operations=0,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=None,
        memory_bandwidth_gb_s=hbm.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=0.0,
        memory_energy_pj_per_byte=hbm.energy_pj_per_byte,
        compute_resource_id=gpu.scalar_resource_id,
        memory_resource_id=hbm.resource_id,
        dispatch_resource_id=gpu.launch_resource_id,
        dispatch_name="kernel_launch",
        dispatch_ns=gpu.kernel_launch_ns,
        dispatch_energy_pj=gpu.launch_energy_pj,
        cache_hierarchy=gpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
    )


def _cpu_instruction_schedule(
    cpu: CPUProfile,
    *,
    operator_class: OperatorClass,
    operations: int,
    read_bytes: int,
    write_bytes: int,
    element_bits: int,
    dependency_depth: int,
    special_function_operations: int = 0,
    quantized_dot_capability: Optional[CPUQuantizedDotCapability] = None,
    packed_weight_transform_operations: int = 0,
    activation_elements: int = 0,
    source_dot_work: Optional[Mapping[str, int]] = None,
    source_dot_blocks: int = 0,
    source_dot_totals: Optional[Mapping[str, int]] = None,
    source_dot_row_totals: Optional[Mapping[str, int]] = None,
    effective_core_count: Optional[int] = None,
    scalar_execution: bool = False,
) -> Tuple[float, Mapping[str, object]]:
    """Convert aggregate work into an auditable OoO instruction schedule."""

    pipeline = cpu.pipeline
    _require_non_negative_int("operations", operations)
    _require_non_negative_int("read_bytes", read_bytes)
    _require_non_negative_int("write_bytes", write_bytes)
    _require_positive_int("element_bits", element_bits)
    _require_positive_int("dependency_depth", dependency_depth)
    _require_non_negative_int(
        "special_function_operations", special_function_operations
    )
    _require_non_negative_int(
        "packed_weight_transform_operations",
        packed_weight_transform_operations,
    )
    _require_non_negative_int("activation_elements", activation_elements)
    if not isinstance(scalar_execution, bool):
        raise ValueError("scalar_execution must be a boolean")
    if scalar_execution and (
        operator_class != OperatorClass.ELEMENTWISE
        or element_bits != 32
        or special_function_operations != 0
        or quantized_dot_capability is not None
        or packed_weight_transform_operations != 0
        or activation_elements != 0
        or source_dot_work is not None
        or source_dot_blocks != 0
        or source_dot_totals is not None
        or source_dot_row_totals is not None
    ):
        raise ValueError(
            "scalar_execution requires plain 32-bit elementwise work"
        )
    if effective_core_count is not None:
        _require_positive_int("effective_core_count", effective_core_count)
        if effective_core_count > pipeline.core_count:
            raise ValueError(
                "effective_core_count must not exceed physical core_count"
            )
    if (
        quantized_dot_capability is not None
        and operator_class != OperatorClass.GEMM
    ):
        raise ValueError("quantized-dot capability is valid only for GEMM")
    if source_dot_row_totals is not None:
        row_fields = {"auxiliary_vector_ops", "vector_loads", "scalar_lut_loads", "store_issue_ops"}
        if quantized_dot_capability is None or not isinstance(source_dot_row_totals, Mapping) or set(source_dot_row_totals) != row_fields:
            raise ValueError("source_dot_row_totals requires a quantized dot and all four row totals")
        for key, value in source_dot_row_totals.items():
            _require_non_negative_int("source_dot_row_totals " + key, value)

    is_quantized_dot = quantized_dot_capability is not None
    effective_element_bits = (
        quantized_dot_capability.dot_activation_bits
        if quantized_dot_capability is not None
        else element_bits
    )
    # The sampler's conditional heap loop reads one f32 field from each
    # 12-byte record. Do not assume contiguous SIMD packing for that path.
    vector_lanes = (
        1 if scalar_execution
        else max(1, pipeline.simd_width_bits // effective_element_bits)
    )
    vector_bytes = 4 if scalar_execution else max(1, pipeline.simd_width_bits // 8)
    if quantized_dot_capability is not None:
        operations_per_instruction = (
            quantized_dot_capability.effective_ops_per_instruction
        )
        execution_units = (
            quantized_dot_capability
            .dot_issue_instructions_per_cycle_per_core
        )
    elif operator_class == OperatorClass.GEMM:
        operations_per_instruction = 2 * vector_lanes
        execution_units = pipeline.vector_fma_units_per_core
    else:
        operations_per_instruction = vector_lanes
        execution_units = pipeline.vector_alu_units_per_core
    if operations > 0:
        compute_instructions = (
            int(math.ceil(operations / float(operations_per_instruction)))
            if is_quantized_dot
            else _ceil_div(operations, operations_per_instruction)
        )
    else:
        compute_instructions = 0
    packed_weight_transform_instructions = 0
    packed_weight_transform_issue_instructions = 0
    activation_quantization_blocks = 0
    activation_quantization_instructions = 0
    activation_quantization_accounted = True
    if quantized_dot_capability is not None:
        if source_dot_work is not None:
            packed_weight_transform_instructions = (
                source_dot_blocks * source_dot_work["auxiliary_vector_ops"]
            )
        elif source_dot_totals is not None:
            packed_weight_transform_instructions = int(
                source_dot_totals["auxiliary_vector_ops"]
            )
        if source_dot_row_totals is not None:
            packed_weight_transform_instructions += source_dot_row_totals["auxiliary_vector_ops"]
        if packed_weight_transform_operations > 0:
            packed_weight_transform_instructions = int(
                packed_weight_transform_instructions + math.ceil(
                    packed_weight_transform_operations /
                    quantized_dot_capability.auxiliary_ops_per_instruction
                )
            )
        activation_quantization_blocks = (
            _ceil_div(
                activation_elements,
                quantized_dot_capability
                .activation_quantization_block_elements,
            )
            if activation_elements > 0
            else 0
        )
        activation_instructions_per_block = (
            quantized_dot_capability
            .activation_quantization_instructions_per_block
        )
        activation_quantization_accounted = (
            activation_instructions_per_block is not None
        )
        if (
            activation_quantization_blocks > 0
            and activation_instructions_per_block is not None
        ):
            activation_quantization_instructions = int(
                math.ceil(
                    activation_quantization_blocks
                    * activation_instructions_per_block
                )
            )
    elif packed_weight_transform_operations > 0:
        # A quantized GEMM without a declared ISA capability still executes
        # the packed-weight dequant/unpack work.  The old generic path dropped
        # this field entirely, making Q4/Q5 CPU estimates optimistic.  Keep
        # the operation count in the analytical schedule.  Generic GGML
        # dequantization includes bit extraction, scale lookup and indexed
        # loads whose exact ISA packing is not proven by the profile.  Keep the
        # primitive count for audit, and use a conservative two-primitive
        # issue grouping for the shared vector-ALU envelope.  ISA-specific
        # profiles may replace this with measured
        # ``auxiliary_ops_per_instruction``; no request timing is fitted here.
        packed_weight_transform_instructions = int(
            packed_weight_transform_operations
        )
        packed_weight_transform_issue_instructions = _ceil_div(
            packed_weight_transform_operations, 2
        )
        activation_quantization_accounted = False
    auxiliary_instructions = (
        (packed_weight_transform_issue_instructions
         if quantized_dot_capability is None
         else packed_weight_transform_instructions)
        + activation_quantization_instructions
    )
    special_function_instructions = (
        _ceil_div(special_function_operations, vector_lanes)
        if special_function_operations > 0
        else 0
    )
    load_instructions = _ceil_div(read_bytes, vector_bytes) if read_bytes else 0
    minimum_io_load_proxy = load_instructions
    source_load_proxy = 0
    if source_dot_work is not None or source_dot_totals is not None:
        # These are register-side reads, including repeated activations and
        # table lookups. They must not multiply compulsory backing-memory IO.
        source_load_proxy = (
            source_dot_blocks * (
                source_dot_work["vector_loads"] + source_dot_work["scalar_lut_loads"]
            )
            if source_dot_work is not None
            else int(source_dot_totals["vector_loads"])
            + int(source_dot_totals["scalar_lut_loads"])
        )
    if source_dot_row_totals is not None:
        source_load_proxy += source_dot_row_totals["vector_loads"] + source_dot_row_totals["scalar_lut_loads"]
    load_instructions = max(minimum_io_load_proxy, source_load_proxy)
    store_instructions = (
        _ceil_div(write_bytes, vector_bytes) if write_bytes else 0
    )
    minimum_io_store_proxy = store_instructions
    if source_dot_row_totals is not None:
        # The fixed row budget already includes the result store. Stack saves
        # consume issue resources but do not add compulsory backing-memory IO.
        store_instructions = max(store_instructions, source_dot_row_totals["store_issue_ops"])
    total_instructions = (
        compute_instructions
        + auxiliary_instructions
        + special_function_instructions
        + load_instructions
        + store_instructions
    )
    cores = (
        pipeline.core_count
        if effective_core_count is None
        else effective_core_count
    )

    frontend_cycles = max(
        _ceil_div(total_instructions, cores * pipeline.decode_width),
        _ceil_div(total_instructions, cores * pipeline.issue_width),
        _ceil_div(total_instructions, cores * pipeline.retire_width),
    )
    execution_cycles = 0
    if compute_instructions:
        execution_cycles = (
            int(
                math.ceil(
                    compute_instructions / float(cores * execution_units)
                )
            )
            if is_quantized_dot
            else _ceil_div(compute_instructions, cores * execution_units)
        )
    # Generic quantized fallback shares the vector ALU issue ports with the
    # GEMM body.  Include its unpack/dequant cycles in the same execution
    # envelope; previously they were metadata-only and could not lengthen a
    # compute-bound operator.
    if quantized_dot_capability is None and packed_weight_transform_issue_instructions:
        execution_cycles += _ceil_div(
            packed_weight_transform_issue_instructions,
            cores * pipeline.vector_alu_units_per_core,
        )
    auxiliary_cycles = (
        _ceil_div(
            auxiliary_instructions,
            cores * pipeline.vector_alu_units_per_core,
        )
        if auxiliary_instructions
        else 0
    )
    load_cycles = (
        _ceil_div(
            load_instructions, cores * pipeline.load_units_per_core
        )
        if load_instructions
        else 0
    )
    store_cycles = (
        _ceil_div(
            store_instructions, cores * pipeline.store_units_per_core
        )
        if store_instructions
        else 0
    )
    special_function_cycles = (
        _ceil_div(
            special_function_instructions,
            cores * pipeline.special_function_units_per_core,
        )
        * pipeline.special_function_cycles_per_vector
        if special_function_instructions
        else 0.0
    )
    rob_waves = (
        _ceil_div(
            total_instructions,
            cores * pipeline.reorder_buffer_entries,
        )
        if total_instructions
        else 0
    )
    memory_instructions = load_instructions + store_instructions
    lsq_waves = (
        _ceil_div(
            memory_instructions,
            cores * pipeline.load_store_queue_entries,
        )
        if memory_instructions
        else 0
    )
    mlp_waves = (
        _ceil_div(
            memory_instructions,
            cores * pipeline.memory_level_parallelism,
        )
        if memory_instructions
        else 0
    )
    dependency_cycles = dependency_depth * max(1, rob_waves)
    cycle_rows = {
        "frontend": float(frontend_cycles),
        "execution": float(execution_cycles),
        "load_issue": float(load_cycles),
        "store_issue": float(store_cycles),
        "special_function": float(special_function_cycles),
        "dependency": float(dependency_cycles),
        "lsq_window": float(lsq_waves),
        "memory_level_parallelism": float(mlp_waves),
    }
    if is_quantized_dot:
        cycle_rows["auxiliary_execution"] = float(auxiliary_cycles)
    limiting_stage, total_cycles = max(
        cycle_rows.items(), key=lambda item: (item[1], item[0])
    )
    service_ns = (
        total_cycles
        / pipeline.frequency_ghz
        / cpu.attainable_efficiency
    )
    metadata = {
        "model": "cpu_ooo_instruction_schedule_v3",
        "operator_class": operator_class.value,
        "physical_core_count": pipeline.core_count,
        "effective_core_count": cores,
        "vector_lanes": vector_lanes,
        "vector_bytes": vector_bytes,
        "compute_instructions": compute_instructions,
        "packed_weight_transform_instructions": packed_weight_transform_instructions,
        "packed_weight_transform_issue_instructions": packed_weight_transform_issue_instructions,
        "auxiliary_instructions": auxiliary_instructions,
        "special_function_instructions": special_function_instructions,
        "load_instructions": load_instructions,
        "store_instructions": store_instructions,
        "total_instructions": total_instructions,
        "frontend_width": min(
            pipeline.decode_width,
            pipeline.issue_width,
            pipeline.retire_width,
        ),
        "reorder_buffer_entries_per_core": (
            pipeline.reorder_buffer_entries
        ),
        "load_store_queue_entries_per_core": (
            pipeline.load_store_queue_entries
        ),
        "memory_level_parallelism_per_core": (
            pipeline.memory_level_parallelism
        ),
        "rob_waves": rob_waves,
        "lsq_waves": lsq_waves,
        "mlp_waves": mlp_waves,
        "dependency_depth": dependency_depth,
        "cycle_demands": cycle_rows,
        "limiting_stage": limiting_stage,
        "scheduled_cycles": total_cycles,
        "service_ns": service_ns,
    }
    if scalar_execution:
        metadata["scalar_execution"] = True
        metadata["comparison_issue_model"] = "generic_alu_not_measured_comparator"
    if quantized_dot_capability is not None:
        metadata.update(
            {
                "model": "cpu_quantized_dot_instruction_schedule_v1",
                "quantized_dot_capability": quantized_dot_capability.name,
                "source_activation_bits": element_bits,
                "dot_activation_bits": (
                    quantized_dot_capability.dot_activation_bits
                ),
                "dot_weight_bits": quantized_dot_capability.dot_weight_bits,
                "accumulator_bits": quantized_dot_capability.accumulator_bits,
                "effective_ops_per_instruction": (
                    quantized_dot_capability.effective_ops_per_instruction
                ),
                "dot_issue_instructions_per_cycle_per_core": (
                    quantized_dot_capability
                    .dot_issue_instructions_per_cycle_per_core
                ),
                "auxiliary_ops_per_instruction": (
                    quantized_dot_capability.auxiliary_ops_per_instruction
                ),
                "packed_weight_transform_operations": (
                    packed_weight_transform_operations
                ),
                "packed_weight_transform_instructions": (
                    packed_weight_transform_instructions
                ),
                "activation_elements": activation_elements,
                "activation_quantization_block_elements": (
                    quantized_dot_capability
                    .activation_quantization_block_elements
                ),
                "activation_quantization_blocks": (
                    activation_quantization_blocks
                ),
                "activation_quantization_instructions_per_block": (
                    quantized_dot_capability
                    .activation_quantization_instructions_per_block
                ),
                "activation_quantization_instructions": (
                    activation_quantization_instructions
                ),
                "activation_quantization_accounted": (
                    activation_quantization_accounted
                ),
                "auxiliary_instructions": auxiliary_instructions,
                "timing_completeness": (
                    "complete"
                    if activation_quantization_accounted
                    else "partial"
                ),
                "evidence": quantized_dot_capability.evidence,
            }
        )
    if source_dot_work is not None or source_dot_totals is not None:
        source_vector_loads = (
            source_dot_blocks * source_dot_work["vector_loads"]
            if source_dot_work is not None
            else int(source_dot_totals["vector_loads"])
        )
        source_scalar_lut_loads = (
            source_dot_blocks * source_dot_work["scalar_lut_loads"]
            if source_dot_work is not None
            else int(source_dot_totals["scalar_lut_loads"])
        )
        metadata.update({
            "instruction_work_basis": "source_primitive_proxy",
            "source_dot_blocks": source_dot_blocks,
            "source_dot_work": (
                dict(source_dot_work) if source_dot_work is not None else None
            ),
            "source_dot_segment_totals": (
                dict(source_dot_totals) if source_dot_totals is not None else None
            ),
            "source_vector_loads": source_vector_loads,
            "source_scalar_lut_loads": source_scalar_lut_loads,
            "minimum_io_load_proxy": minimum_io_load_proxy,
            "load_proxy_merge": "max_non_additive_proxy",
            "timing_completeness": "partial",
            "source_work_limitations": (
                "Source intrinsics and indexed reads are not retired instructions; "
                "compiler folding, scalar indexing, activation conversion and "
                "function-level setup/reduction are not fully modeled."
            ),
        })
    if source_dot_row_totals is not None:
        metadata.update({
            "source_dot_row_totals": dict(source_dot_row_totals),
            "minimum_io_store_proxy": minimum_io_store_proxy,
            "store_proxy_merge": "max_non_additive_proxy",
            "timing_completeness": "partial",
        })
    if quantized_dot_capability is not None and (
        quantized_dot_capability.source_dot_work_all_m
        or quantized_dot_capability.source_dot_work_max_k is not None
        or any("row_auxiliary_vector_ops" in budget for budget in quantized_dot_capability.source_dot_work.values())
    ):
        # Explicit source contracts remain partial even when their shape or
        # format scope is ineligible and no source budget was applied.
        metadata.update({
            "timing_completeness": "partial",
            "source_work_limitations": (
                "Source resource roles are not deduplicated frontend or retired "
                "instructions. Scalar work, implicit stack work, activation "
                "conversion, cache behavior and micro-op decomposition remain "
                "partial; ineligible source scope is not zero work."
            ),
        })
    return service_ns, metadata


def estimate_cpu_gemm(
    cpu: CPUProfile,
    memory: HostMemoryProfile,
    workload: GemmWorkload,
) -> CostEstimate:
    """Estimate CPU GEMM; CPU GOP/s is not converted through GPU TOPS."""

    if workload.mmq_work is not None:
        raise ValueError("MMQ source work is GPU-only")
    quantized_dot_capability = cpu.resolve_quantized_dot_capability(workload)
    source_dot_work = None
    source_dot_blocks = 0
    source_dot_totals = None
    source_dot_rows = 0
    source_dot_row_totals = None
    residual_packed_weight_transform_operations = workload.packed_weight_transform_operations
    formats = {value.strip().casefold() for value in workload.packed_weight_formats}
    if (
        quantized_dot_capability is not None
        and quantized_dot_capability.source_dot_work
        and (
            quantized_dot_capability.source_dot_work_all_m
            or workload.m <= quantized_dot_capability.source_dot_work_max_m
        )
        and (
            quantized_dot_capability.source_dot_work_max_k is None
            or workload.k <= quantized_dot_capability.source_dot_work_max_k
        )
    ):
        if workload.packed_weight_format_segments:
            source_totals = {"auxiliary_vector_ops": 0, "vector_loads": 0, "scalar_lut_loads": 0}
            residual_packed_weight_transform_operations = 0
            source_budget_by_format = {
                format_name.casefold(): budget
                for format_name, budget in quantized_dot_capability.source_dot_work.items()
            }
            for format_name, local_n, transform_operations in workload.packed_weight_format_segments:
                budget = source_budget_by_format.get(format_name.casefold())
                if budget is None or workload.k % budget["block_elements"]:
                    residual_packed_weight_transform_operations += transform_operations
                    continue
                block_count = workload.m * local_n * (workload.k // budget["block_elements"])
                source_dot_blocks += block_count
                for key in source_totals:
                    source_totals[key] += block_count * budget[key]
                if "row_auxiliary_vector_ops" in budget:
                    if source_dot_row_totals is None:
                        source_dot_row_totals = {"auxiliary_vector_ops": 0, "vector_loads": 0, "scalar_lut_loads": 0, "store_issue_ops": 0}
                    row_count = workload.m * local_n
                    source_dot_rows += row_count
                    for key in source_dot_row_totals:
                        source_dot_row_totals[key] += row_count * budget["row_" + key]
            if source_dot_blocks:
                source_dot_totals = source_totals
        elif len(formats) == 1:
            budget = next((
                value for key, value in quantized_dot_capability.source_dot_work.items()
                if key.casefold() in formats
            ), None)
            if budget is not None and workload.k % budget["block_elements"] == 0:
                source_dot_work = budget
                source_dot_blocks = workload.m * workload.n * (workload.k // budget["block_elements"])
                residual_packed_weight_transform_operations = 0
                if "row_auxiliary_vector_ops" in budget:
                    source_dot_rows = workload.m * workload.n
                    source_dot_row_totals = {
                        key: source_dot_rows * budget["row_" + key]
                        for key in ("auxiliary_vector_ops", "vector_loads", "scalar_lut_loads", "store_issue_ops")
                    }
    compute_throughput_gops = (
        cpu.attainable_quantized_dot_gops(quantized_dot_capability)
        if quantized_dot_capability is not None
        else cpu.attainable_gemm_gops
    )
    dependency_depth = max(1, int(math.ceil(math.log2(workload.k))))
    instruction_ns, instruction_metadata = _cpu_instruction_schedule(
        cpu,
        operator_class=OperatorClass.GEMM,
        operations=workload.operations,
        read_bytes=workload.activation_bytes + workload.weight_bytes,
        write_bytes=workload.output_bytes,
        element_bits=(
            workload.activation_bits
            if quantized_dot_capability is not None
            else max(workload.activation_bits, workload.weight_bits)
        ),
        dependency_depth=dependency_depth,
        quantized_dot_capability=quantized_dot_capability,
        # Preserve physical dequant/unpack work even when no ISA-specific
        # quantized-dot capability is declared.  Generic CPU fallback prices
        # this through the SIMD instruction schedule instead of silently
        # discarding it.
        # Projection metadata stores dequant primitives per output row.  A
        # GEMM with M rows performs that unpack for each row on CPU; retain
        # this shape dependence in the generic fallback.
        packed_weight_transform_operations=(
            residual_packed_weight_transform_operations
            if quantized_dot_capability is not None
            else (
                residual_packed_weight_transform_operations * workload.m
                if workload.packed_weight_format_segments
                else residual_packed_weight_transform_operations
            )
        ),
        activation_elements=(
            workload.m * workload.k
            if quantized_dot_capability is not None
            else 0
        ),
        source_dot_work=source_dot_work,
        source_dot_blocks=source_dot_blocks,
        source_dot_totals=source_dot_totals,
        source_dot_row_totals=source_dot_row_totals,
    )
    instruction_metadata = dict(instruction_metadata)
    known_quantized_formats = any(
        str(value).strip().upper().startswith(("Q", "IQ"))
        for value in workload.packed_weight_formats
    )
    if known_quantized_formats:
        instruction_metadata["quantized_format_coverage"] = (
            "declared_quantized_capability"
            if quantized_dot_capability is not None
            else "generic_quantized_fallback"
        )
    if source_dot_row_totals is not None:
        instruction_metadata["source_dot_rows"] = source_dot_rows
    auxiliary_compute_energy_pj = 0.0
    if quantized_dot_capability is not None:
        auxiliary_equivalent_operations = (
            workload.packed_weight_transform_operations
            + int(
                instruction_metadata[
                    "activation_quantization_instructions"
                ]
            )
            * quantized_dot_capability.auxiliary_ops_per_instruction
        )
        if source_dot_work is not None or source_dot_totals is not None:
            auxiliary_equivalent_operations = (
                instruction_metadata["auxiliary_instructions"]
                * quantized_dot_capability.auxiliary_ops_per_instruction
            )
        auxiliary_compute_energy_pj = (
            auxiliary_equivalent_operations
            * cpu.elementwise_energy_pj_per_op
        )
    return _estimate_typed_roofline(
        device_kind="cpu",
        profile_name=cpu.name,
        operator_class=OperatorClass.GEMM,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.activation_bytes + workload.weight_bytes,
        write_bytes=workload.output_bytes,
        compute_throughput_gops=compute_throughput_gops,
        memory_bandwidth_gb_s=memory.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=cpu.gemm_energy_pj_per_op,
        additional_compute_energy_pj=auxiliary_compute_energy_pj,
        memory_energy_pj_per_byte=memory.energy_pj_per_byte,
        compute_resource_id=cpu.compute_resource_id,
        memory_resource_id=memory.resource_id,
        dispatch_name="cpu_dispatch",
        dispatch_ns=cpu.dispatch_ns,
        dispatch_energy_pj=cpu.dispatch_energy_pj,
        cache_hierarchy=cpu.cache_hierarchy,
        working_set_bytes=workload.minimum_io_bytes,
        # The GEMM workload is already a minimum-I/O projection.  Cache
        # capacity cannot remove its compulsory activation/RHS/output bytes;
        # it only avoids counting the RHS once per M row.
        reuse_factor=1.0,
        streaming_fraction=1.0,
        dependency_depth=dependency_depth,
        frequency_ghz=cpu.pipeline.frequency_ghz,
        compute_service_override_ns=instruction_ns,
        compute_work_units=float(instruction_metadata["total_instructions"]),
        instruction_metadata=instruction_metadata,
    )


def estimate_cpu_elementwise(
    cpu: CPUProfile,
    memory: HostMemoryProfile,
    workload: ElementwiseWorkload,
) -> CostEstimate:
    """Estimate CPU element-wise work with its scalar GOP/s contract."""

    instruction_ns, instruction_metadata = _cpu_instruction_schedule(
        cpu,
        operator_class=OperatorClass.ELEMENTWISE,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        element_bits=max(workload.input_bits, workload.output_bits),
        dependency_depth=workload.dependency_depth,
        special_function_operations=workload.transcendental_operations,
    )

    return _estimate_typed_roofline(
        device_kind="cpu",
        profile_name=cpu.name,
        operator_class=OperatorClass.ELEMENTWISE,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=cpu.attainable_elementwise_gops,
        memory_bandwidth_gb_s=memory.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=cpu.elementwise_energy_pj_per_op,
        additional_compute_energy_pj=(
            workload.transcendental_operations
            * cpu.special_function_energy_pj_per_op
        ),
        memory_energy_pj_per_byte=memory.energy_pj_per_byte,
        compute_resource_id=cpu.compute_resource_id,
        memory_resource_id=memory.resource_id,
        dispatch_name="cpu_dispatch",
        dispatch_ns=cpu.dispatch_ns,
        dispatch_energy_pj=cpu.dispatch_energy_pj,
        cache_hierarchy=cpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        dependency_depth=workload.dependency_depth,
        frequency_ghz=cpu.pipeline.frequency_ghz,
        compute_service_override_ns=instruction_ns,
        compute_work_units=float(instruction_metadata["total_instructions"]),
        instruction_metadata=instruction_metadata,
    )


def estimate_cpu_reduction(
    cpu: CPUProfile,
    memory: HostMemoryProfile,
    workload: ReductionWorkload,
) -> CostEstimate:
    """Estimate CPU reduction with its dedicated reduction GOP/s contract."""

    tree_depth = max(
        workload.dependency_depth,
        int(
            math.ceil(
                math.log2(
                    max(
                        1,
                        workload.input_elements
                        // workload.output_elements,
                    )
                )
            )
        ),
    )
    instruction_ns, instruction_metadata = _cpu_instruction_schedule(
        cpu,
        operator_class=OperatorClass.REDUCTION,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        element_bits=max(workload.input_bits, workload.output_bits),
        dependency_depth=tree_depth,
    )

    return _estimate_typed_roofline(
        device_kind="cpu",
        profile_name=cpu.name,
        operator_class=OperatorClass.REDUCTION,
        workload_name=workload.name,
        operations=workload.operations,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=cpu.attainable_reduction_gops,
        memory_bandwidth_gb_s=memory.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=cpu.reduction_energy_pj_per_op,
        memory_energy_pj_per_byte=memory.energy_pj_per_byte,
        compute_resource_id=cpu.compute_resource_id,
        memory_resource_id=memory.resource_id,
        dispatch_name="cpu_dispatch",
        dispatch_ns=cpu.dispatch_ns,
        dispatch_energy_pj=cpu.dispatch_energy_pj,
        cache_hierarchy=cpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        dependency_depth=tree_depth,
        frequency_ghz=cpu.pipeline.frequency_ghz,
        compute_service_override_ns=instruction_ns,
        compute_work_units=float(instruction_metadata["total_instructions"]),
        instruction_metadata=instruction_metadata,
    )


def estimate_cpu_memory(
    cpu: CPUProfile,
    memory: HostMemoryProfile,
    workload: MemoryWorkload,
) -> CostEstimate:
    """Estimate pure CPU-visible host-memory traffic."""

    instruction_ns, instruction_metadata = _cpu_instruction_schedule(
        cpu,
        operator_class=OperatorClass.MEMORY,
        operations=0,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        element_bits=8,
        dependency_depth=1,
    )

    return _estimate_typed_roofline(
        device_kind="cpu",
        profile_name=cpu.name,
        operator_class=OperatorClass.MEMORY,
        workload_name=workload.name,
        operations=0,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        compute_throughput_gops=None,
        memory_bandwidth_gb_s=memory.effective_bandwidth_gb_s,
        compute_energy_pj_per_op=0.0,
        memory_energy_pj_per_byte=memory.energy_pj_per_byte,
        compute_resource_id=cpu.compute_resource_id,
        memory_resource_id=memory.resource_id,
        dispatch_name="cpu_dispatch",
        dispatch_ns=cpu.dispatch_ns,
        dispatch_energy_pj=cpu.dispatch_energy_pj,
        cache_hierarchy=cpu.cache_hierarchy,
        working_set_bytes=workload.effective_working_set_bytes,
        reuse_factor=workload.reuse_factor,
        streaming_fraction=workload.streaming_fraction,
        dependency_depth=1,
        frequency_ghz=cpu.pipeline.frequency_ghz,
        compute_service_override_ns=instruction_ns,
        compute_work_units=float(instruction_metadata["total_instructions"]),
        instruction_metadata=instruction_metadata,
    )


def estimate_cpu_logical_stream(
    cpu: CPUProfile,
    workload: MemoryWorkload,
    *,
    serial_repetitions: int = 1,
    comparison_count: int = 0,
) -> CostEstimate:
    """Estimate one-core logical load/store issue without inventing DRAM IO.

    ``workload`` describes one serial iteration.  Repetitions are evaluated as
    a strict serial sum so per-iteration instruction rounding is preserved.
    Optional f32 comparisons share the same scalar load/compute envelope.
    The result is an optimistic pipeline estimate under the declared CPU
    issue model, not a guaranteed mathematical lower bound for compiled code.
    """

    _require_positive_int("serial_repetitions", serial_repetitions)
    _require_non_negative_int("comparison_count", comparison_count)
    per_iteration_ns, instruction_metadata = _cpu_instruction_schedule(
        cpu,
        operator_class=(
            OperatorClass.ELEMENTWISE if comparison_count else OperatorClass.MEMORY
        ),
        operations=comparison_count,
        read_bytes=workload.read_bytes,
        write_bytes=workload.write_bytes,
        element_bits=32 if comparison_count else 8,
        dependency_depth=1,
        effective_core_count=1,
        scalar_execution=comparison_count > 0,
    )
    service_ns = per_iteration_ns * serial_repetitions
    logical_read_bytes = workload.read_bytes * serial_repetitions
    logical_write_bytes = workload.write_bytes * serial_repetitions
    total_instructions = (
        int(instruction_metadata["total_instructions"])
        * serial_repetitions
    )
    common_metadata = {
        "model": "cpu_logical_stream_issue_v1",
        "profile": cpu.name,
        "evidence": EvidenceStatus.ANALYTICAL.value,
        "timing_completeness": "partial",
        "timing_interpretation": (
            "optimistic_single_core_pipeline_estimate"
        ),
        "strict_mathematical_lower_bound": False,
        "partial_reason": (
            "compiler_lowering_cache_residency_write_allocation_and_"
            "single_thread_memory_throughput_not_declared"
        ),
        "serial_repetitions": serial_repetitions,
        "row_service_aggregation": "serial_sum",
        "per_iteration_service_ns": per_iteration_ns,
        "per_iteration_logical_read_bytes": workload.read_bytes,
        "per_iteration_logical_write_bytes": workload.write_bytes,
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_write_bytes,
        "logical_stream_bytes": logical_read_bytes + logical_write_bytes,
        "physical_memory_traffic_status": "unknown_not_charged",
        "instruction_schedule": dict(instruction_metadata),
    }
    if comparison_count:
        common_metadata.update({
            "model": "cpu_logical_scalar_comparison_issue_v1",
            "per_iteration_comparison_count": comparison_count,
            "comparison_count": comparison_count * serial_repetitions,
            "comparison_bits": 32,
            "partial_reason": (
                "generic_alu_issue_compiler_fusion_strided_candidate_reads_"
                "and_conditional_heap_dependencies_not_resolved"
            ),
        })
    return CostEstimate(
        phases=(
            CostPhase(
                name="cpu_logical_stream_issue",
                category=(
                    TaskCategory.COMPUTE if comparison_count else TaskCategory.MEMORY
                ),
                demands=(
                    ResourceDemand(
                        resource_id=cpu.compute_resource_id,
                        service_ns=service_ns,
                        work_units=float(total_instructions),
                    ),
                ),
                metadata=common_metadata,
            ),
        ),
        useful_ops=comparison_count * serial_repetitions,
        utilization=1.0,
        metadata=common_metadata,
    )


def _tree_levels(term_count: int, fan_in: int) -> int:
    levels = 0
    remaining = term_count
    while remaining > 1:
        remaining = _ceil_div(remaining, fan_in)
        levels += 1
    return levels


def _required_accumulator_bits(workload: GemmWorkload, guard_bits: int) -> int:
    return (
        workload.activation_bits
        + workload.weight_bits
        + int(math.ceil(math.log2(workload.k)))
        + guard_bits
    )


def estimate_cim_gemm(
    profile: DigitalSramCimProfile,
    workload: GemmWorkload,
    weights_resident: bool = False,
) -> CostEstimate:
    """Estimate an ADC-free weight-stationary digital SRAM-CIM GEMM."""

    if workload.mmq_work is not None:
        raise ValueError("MMQ source work is GPU-only")
    if workload.activation_bits not in profile.supported_activation_bits:
        raise ValueError(
            "unsupported activation bit width: %d" % workload.activation_bits
        )
    if workload.weight_bits not in profile.supported_weight_bits:
        raise ValueError("unsupported weight bit width: %d" % workload.weight_bits)

    required_accumulator_bits = _required_accumulator_bits(
        workload, profile.accumulator_guard_bits
    )
    available_accumulator_bits = min(
        workload.accumulator_bits, profile.accumulator_bits
    )
    if available_accumulator_bits < required_accumulator_bits:
        raise ValueError(
            "accumulator width %d is smaller than required width %d"
            % (available_accumulator_bits, required_accumulator_bits)
        )

    n_m = _ceil_div(workload.m, profile.p_m)
    n_k = _ceil_div(workload.k, profile.p_k)
    n_n = _ceil_div(workload.n, profile.p_n)
    q_a = _ceil_div(workload.activation_bits, profile.input_parallel_bits)
    q_w = _ceil_div(workload.weight_bits, profile.weight_parallel_bits)
    bit_slice_count = q_a * q_w

    padded_weight_elements = n_k * profile.p_k * n_n * profile.p_n
    base_weight_storage_bytes = (
        _storage_bytes(padded_weight_elements, workload.weight_bits)
        + workload.weight_metadata_bytes
    )
    if base_weight_storage_bytes > profile.weight_capacity_bytes:
        raise ValueError(
            "padded weight storage (%d bytes) exceeds CIM capacity (%d bytes)"
            % (base_weight_storage_bytes, profile.weight_capacity_bytes)
        )

    placements = n_k * n_n
    replication_by_arrays = max(1, profile.array_count // placements)
    replication_by_capacity = max(
        1, profile.weight_capacity_bytes // base_weight_storage_bytes
    )
    replication = min(
        profile.max_m_replication,
        n_m,
        replication_by_arrays,
        replication_by_capacity,
    )
    a_eff = min(profile.array_count, placements * replication)

    block_waves = _ceil_div(n_m * placements, a_eff)
    array_cycles = (
        block_waves * bit_slice_count * profile.cycles_per_eval
    )
    array_service_ns = array_cycles / profile.frequency_ghz
    logical_evaluations = n_m * placements * bit_slice_count

    shape_utilization = (
        (workload.m * workload.k * workload.n)
        / float(
            n_m
            * profile.p_m
            * n_k
            * profile.p_k
            * n_n
            * profile.p_n
        )
    )
    wave_utilization = (n_m * placements) / float(block_waves * a_eff)
    bit_utilization = (
        workload.activation_bits
        / float(q_a * profile.input_parallel_bits)
        * workload.weight_bits
        / float(q_w * profile.weight_parallel_bits)
    )
    utilization = shape_utilization * wave_utilization * bit_utilization

    phases = []
    transfer_weight_bytes = workload.weight_bytes * replication
    resident_weight_bytes = base_weight_storage_bytes * replication
    if not weights_resident:
        load_service_ns = (
            profile.load_latency_ns
            + transfer_weight_bytes / profile.load_bandwidth_gb_s
        )
        phases.append(
            CostPhase(
                name="weight_load",
                category=TaskCategory.MEMORY,
                demands=(
                    ResourceDemand(
                        resource_id=profile.load_resource_id,
                        service_ns=load_service_ns,
                        bytes_moved=transfer_weight_bytes,
                        energy_pj=(
                            transfer_weight_bytes
                            * profile.load_energy_pj_per_byte
                        ),
                    ),
                ),
                metadata={"resident_bytes": resident_weight_bytes},
            )
        )

    activation_service_ns = (
        workload.activation_bytes / profile.activation_bandwidth_gb_s
    )
    compute_demands = [
        ResourceDemand(
            resource_id=profile.array_resource_id,
            service_ns=array_service_ns,
            energy_pj=logical_evaluations * profile.eval_energy_pj,
            work_units=float(logical_evaluations),
        ),
        ResourceDemand(
            resource_id=profile.activation_resource_id,
            service_ns=activation_service_ns,
            bytes_moved=workload.activation_bytes,
            energy_pj=(
                workload.activation_bytes
                * profile.activation_energy_pj_per_byte
            ),
        ),
    ]

    activation_broadcast_bytes = workload.activation_bytes * max(0, n_n - 1)
    if activation_broadcast_bytes:
        broadcast_service_ns = (
            profile.noc_hop_latency_ns
            + activation_broadcast_bytes / profile.noc_bandwidth_gb_s
        )
        compute_demands.append(
            ResourceDemand(
                resource_id=profile.noc_resource_id,
                service_ns=broadcast_service_ns,
                bytes_moved=activation_broadcast_bytes,
                energy_pj=(
                    activation_broadcast_bytes * profile.noc_energy_pj_per_byte
                ),
            )
        )
    else:
        broadcast_service_ns = 0.0

    phases.append(
        CostPhase(
            name="cim_array_eval",
            category=TaskCategory.CIM,
            demands=tuple(compute_demands),
            metadata={
                "array_service_ns": array_service_ns,
                "activation_service_ns": activation_service_ns,
                "broadcast_service_ns": broadcast_service_ns,
            },
        )
    )

    partial_terms = n_k * bit_slice_count
    accumulation_ops = workload.m * workload.n * max(0, partial_terms - 1)
    psum_elements = workload.m * workload.n * max(0, n_k - 1)
    psum_bytes = _storage_bytes(
        psum_elements, available_accumulator_bits
    )
    reduce_service_ns = 0.0
    if accumulation_ops:
        accumulation_cycles = int(
            math.ceil(
                accumulation_ops / profile.accumulator_outputs_per_cycle
            )
        )
        accumulator_service_ns = accumulation_cycles / profile.frequency_ghz
        reduce_demands = [
            ResourceDemand(
                resource_id=profile.accumulator_resource_id,
                service_ns=accumulator_service_ns,
                energy_pj=(
                    accumulation_ops * profile.accumulator_energy_pj_per_op
                ),
                work_units=float(accumulation_ops),
            )
        ]

        noc_reduce_service_ns = 0.0
        if psum_bytes:
            levels = _tree_levels(n_k, profile.noc_reduce_fan_in)
            noc_reduce_service_ns = (
                levels * profile.noc_hop_latency_ns
                + psum_bytes / profile.noc_bandwidth_gb_s
            )
            reduce_demands.append(
                ResourceDemand(
                    resource_id=profile.noc_resource_id,
                    service_ns=noc_reduce_service_ns,
                    bytes_moved=psum_bytes,
                    energy_pj=psum_bytes * profile.noc_energy_pj_per_byte,
                )
            )
        reduce_service_ns = max(
            demand.service_ns for demand in reduce_demands
        )
        phases.append(
            CostPhase(
                name="cim_accumulate_reduce",
                category=TaskCategory.CIM,
                demands=tuple(reduce_demands),
                metadata={
                    "accumulation_ops": accumulation_ops,
                    "psum_bytes": psum_bytes,
                    "accumulator_service_ns": accumulator_service_ns,
                    "noc_service_ns": noc_reduce_service_ns,
                },
            )
        )

    output_elements = workload.m * workload.n
    peripheral_cycles = int(
        math.ceil(output_elements / profile.peripheral_elements_per_cycle)
    )
    peripheral_compute_ns = peripheral_cycles / profile.frequency_ghz
    output_service_ns = workload.output_bytes / profile.output_bandwidth_gb_s
    peripheral_service_ns = profile.peripheral_latency_ns + max(
        peripheral_compute_ns, output_service_ns
    )
    phases.append(
        CostPhase(
            name="cim_peripheral_output",
            category=TaskCategory.OUTPUT,
            demands=(
                ResourceDemand(
                    resource_id=profile.peripheral_resource_id,
                    service_ns=peripheral_service_ns,
                    bytes_moved=workload.output_bytes,
                    energy_pj=(
                        output_elements * profile.peripheral_energy_pj_per_element
                        + workload.output_bytes * profile.output_energy_pj_per_byte
                    ),
                    work_units=float(output_elements),
                ),
            ),
            metadata={
                "compute_service_ns": peripheral_compute_ns,
                "output_service_ns": output_service_ns,
            },
        )
    )

    return CostEstimate(
        phases=tuple(phases),
        useful_ops=workload.operations,
        utilization=utilization,
        metadata={
            "model": "digital_sram_cim",
            "profile": profile.name,
            "weights_resident": weights_resident,
            "n_m": n_m,
            "n_k": n_k,
            "n_n": n_n,
            "q_a": q_a,
            "q_w": q_w,
            "bit_slice_count": bit_slice_count,
            "a_eff": a_eff,
            "weight_replication": replication,
            "resident_weight_bytes": resident_weight_bytes,
            "transfer_weight_bytes": transfer_weight_bytes,
            "required_accumulator_bits": required_accumulator_bits,
            "array_cycles": array_cycles,
            "array_service_ns": array_service_ns,
            "reduce_service_ns": reduce_service_ns,
            "shape_utilization": shape_utilization,
            "wave_utilization": wave_utilization,
            "bit_utilization": bit_utilization,
        },
    )


def break_even_reuse(
    gpu_service_ns: float,
    cim_resident_service_ns: float,
    cim_load_ns: float,
) -> Optional[float]:
    """Return the strict reuse threshold for CIM latency benefit.

    The amortized CIM cost is ``resident + load / reuse``.  The returned value
    is therefore a real-valued threshold: an integer reuse count must be
    strictly greater than it.  ``None`` means resident CIM is not faster than
    the GPU and no finite reuse count can break even.
    """

    for name, value in (
        ("gpu_service_ns", gpu_service_ns),
        ("cim_resident_service_ns", cim_resident_service_ns),
        ("cim_load_ns", cim_load_ns),
    ):
        _require_non_negative(name, value)
    advantage_ns = gpu_service_ns - cim_resident_service_ns
    if advantage_ns <= 0.0:
        return None
    return cim_load_ns / advantage_ns


__all__ = [
    "CPUPipelineProfile",
    "CPUProfile",
    "CacheHierarchyProfile",
    "CacheLevelProfile",
    "CostEstimate",
    "CostPhase",
    "DigitalSramCimProfile",
    "ElementwiseWorkload",
    "FusedAttentionKVPhysicalContract",
    "FusedAttentionWorkload",
    "GPUProfile",
    "GemmWorkload",
    "HBMProfile",
    "HostGemmOffloadCapability",
    "HostRecurrentOffloadCapability",
    "HostMemoryProfile",
    "HostOrchestrationProfile",
    "MemoryWorkload",
    "ReductionWorkload",
    "TensorKernelWorkload",
    "TensorCoreProfile",
    "break_even_reuse",
    "estimate_cim_gemm",
    "estimate_cpu_elementwise",
    "estimate_cpu_gemm",
    "estimate_cpu_logical_stream",
    "estimate_cpu_memory",
    "estimate_cpu_reduction",
    "estimate_gpu_elementwise",
    "estimate_gpu_fused_attention",
    "estimate_gpu_gemm",
    "estimate_gpu_memory",
    "estimate_gpu_reduction",
    "estimate_gpu_tensor_kernel",
]
