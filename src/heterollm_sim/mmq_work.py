"""Fixed-source MMQ work derivation; this module deliberately has no cost model.

All values are source-accounting quantities, not measured instructions or
memory transactions.  Callers either consume the explicit fields or leave the
work unsupported; this module never falls back to a different CUDA path.
"""
from __future__ import annotations

from dataclasses import dataclass


MMVQ_MAX_BATCH_SIZE = {
    "IQ4_XS": 8,
    "Q4_K": 5,
    "Q5_0": 8,
    "IQ3_S": 8,
    "Q5_K": 6,
    "Q8_0": 8,
    "Q6_K": 7,
}

_DS4_FORMATS = frozenset(("Q4_K", "Q5_K"))
_QK_32_FORMATS = frozenset(("Q5_0", "Q8_0"))
# Packed block sizes from ggml-common.h, not nominal bits per weight.
_WEIGHT_BLOCK_BYTES = {
    "IQ4_XS": 136, "Q4_K": 144, "Q5_0": 22, "IQ3_S": 110,
    "Q5_K": 176, "Q8_0": 34, "Q6_K": 210,
}
_J_NO_TAIL = (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)
_J_TAIL = (8, 16, 32, 64, 128)
_I = 128
_WEIGHT_SHARED_BYTES = 38_912
_MIN_SHARED_MEMORY_PER_BLOCK = 49_152
_SOURCE_ALLOC_LIMIT = 2**30


class UnsupportedMMQ(ValueError):
    """The fixed-source MMQ formula has no valid description for this call."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _shared_bytes(j: int) -> int:
    return 4 * j + _WEIGHT_SHARED_BYTES + 1024 * _ceil_div(144 * j, 1024)


@dataclass(frozen=True)
class MMQWork:
    """One source-qualified fixed-backend MMQ conversion/main/fixup description."""

    m: int
    k: int
    n: int
    weight_format: str
    sm_count: int
    shared_memory_per_block: int
    qk: int
    conversion_layout: str
    i: int
    j: int
    jmax: int
    k_padded: int
    k_execution: int
    x_tiles: int
    y_tiles: int
    u_tiles: int
    k_iterations: int
    consumer_window_bytes: int
    consumer_extra_bytes: int
    source_repeated_bytes: int
    source_allocation_bytes: int
    block_count: int
    stream_k_boundaries_qblocks: tuple[int, ...]
    partial_writer_count: int
    fixup_tile_indices: tuple[int, ...]
    fixup_valid_elements: int

    @property
    def conversion_read_bytes(self) -> int:
        return 4 * self.m * self.k

    @property
    def conversion_write_bytes(self) -> int:
        return 144 * self.m * self.k_padded // 128

    @property
    def conversion_operations(self) -> int:
        elements = self.m * self.k_padded // 4
        return (20 if self.conversion_layout == "DS4" else 14) * elements

    @property
    def consumer_unique_bytes(self) -> int:
        return self.conversion_effective_bytes + self.consumer_extra_bytes

    @property
    def main_partial_write_bytes(self) -> int:
        return 4 * self.partial_writer_count * self.i * self.j

    @property
    def fixup_launch(self) -> bool:
        return self.u_tiles % self.block_count != 0

    @property
    def fixup_read_bytes(self) -> int:
        return self.main_partial_write_bytes + 4 * self.fixup_valid_elements

    @property
    def fixup_write_bytes(self) -> int:
        return 4 * self.fixup_valid_elements

    @property
    def fixup_operations(self) -> int:
        return self.partial_writer_count * self.i * self.j + self.fixup_valid_elements

    @property
    def native_output_bytes(self) -> int:
        return 4 * self.m * self.n

    @property
    def conversion_effective_bytes(self) -> int:
        # A q8_1_mmq block holds 128 logical values. A partly populated block
        # still occupies its complete physical block in the converted layout.
        return 144 * self.m * _ceil_div(self.k, 128)

    @property
    def source_nonempty_block_count(self) -> int:
        return sum(
            start != end for start, end in zip(
                self.stream_k_boundaries_qblocks,
                self.stream_k_boundaries_qblocks[1:],
            )
        )

    @property
    def logical_arithmetic_operations(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def execution_arithmetic_operations(self) -> int:
        """Logical output shape, including mandatory 256-value K iterations."""
        return 2 * self.m * self.n * self.k_execution

    @property
    def source_nominal_arithmetic_operations(self) -> int:
        """Source tile MAC work; this is not a measured instruction count."""
        return 2 * self.u_tiles * self.i * self.j * self.k_execution

    @property
    def weight_block_bytes(self) -> int:
        return _WEIGHT_BLOCK_BYTES[self.weight_format]

    @property
    def logical_weight_bytes(self) -> int:
        return self.n * (self.k // self.qk) * self.weight_block_bytes

    @property
    def weight_tail_read_bytes_per_row(self) -> int:
        return ((self.k_execution - self.k) // self.qk) * self.weight_block_bytes

    @property
    def weight_read_high_water_bytes(self) -> int:
        # Source rows retain their logical K stride. Intermediate row tails
        # overlap later row data; only the final row extends beyond the tensor.
        # Multiplying the tail by N would invent a row-padded weight allocation.
        return self.logical_weight_bytes + self.weight_tail_read_bytes_per_row

    @property
    def weight_tail_range_bytes(self) -> tuple[int, int]:
        """Half-open final tensor tail range; allocation validity is external."""
        return self.logical_weight_bytes, self.weight_read_high_water_bytes

    def to_metadata(self) -> dict[str, object]:
        source_threads = self.m * self.k_padded // 4
        summed_layout = self.conversion_layout == "DS4"
        return {
            "model": "fixed_source_mmq_work/v1",
            "backend_commit": "0f3a71be15af836d277c9f918adfafb45732677e",
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "weight_format": self.weight_format,
            "sm_count": self.sm_count,
            "shared_memory_per_block": self.shared_memory_per_block,
            "i": self.i,
            "j": self.j,
            "jmax": self.jmax,
            "template_shared_bytes": _shared_bytes(self.j),
            "qk": self.qk,
            "conversion_layout": self.conversion_layout,
            "k_padded": self.k_padded,
            "k_execution": self.k_execution,
            "k_quantization_blocks": self.k // self.qk,
            "k_blocks_per_iteration": 256 // self.qk,
            "k_conversion_nonzero_block_extent": 128 * _ceil_div(self.k, 128),
            "logical_arithmetic_operations": self.logical_arithmetic_operations,
            "execution_arithmetic_operations": self.execution_arithmetic_operations,
            "source_nominal_arithmetic_operations": self.source_nominal_arithmetic_operations,
            "source_nominal_arithmetic_policy": "full_I_J_tiles_and_256_value_K_iterations",
            "weight_block_bytes": self.weight_block_bytes,
            "logical_weight_bytes": self.logical_weight_bytes,
            "weight_tail_read_bytes_per_row": self.weight_tail_read_bytes_per_row,
            "weight_read_high_water_bytes": self.weight_read_high_water_bytes,
            "weight_tail_range_bytes": self.weight_tail_range_bytes,
            "weight_tail_allocation_proven": self.k_execution == self.k,
            "weight_tail_policy": "logical_row_stride_final_tensor_tail_only_no_per_row_allocation",
            "x_tiles": self.x_tiles,
            "y_tiles": self.y_tiles,
            "u_tiles": self.u_tiles,
            "k_iterations": self.k_iterations,
            "conversion_effective_bytes": self.conversion_effective_bytes,
            "conversion_read_bytes": self.conversion_read_bytes,
            "conversion_write_bytes": self.conversion_write_bytes,
            "conversion_operations": self.conversion_operations,
            "consumer_unique_bytes": self.consumer_unique_bytes,
            "consumer_load_window_bytes": self.consumer_window_bytes,
            "consumer_extra_bytes": self.consumer_extra_bytes,
            "source_repeated_bytes": self.source_repeated_bytes,
            "source_allocation_bytes": self.source_allocation_bytes,
            "source_allocation_covers_consumer": True,
            "block_count": self.block_count,
            "stream_k_boundaries_qblocks": self.stream_k_boundaries_qblocks,
            "source_nonempty_block_count": self.source_nonempty_block_count,
            "stream_k_boundary_policy": "raw=floor(b*U*B/G); boundary=raw-(raw%B)%R",
            "main_partial_write_bytes": self.main_partial_write_bytes,
            "partial_writer_count": self.partial_writer_count,
            "fixup_launch": self.fixup_launch,
            "fixup_requested_allocation_bytes": (
                4 * self.block_count * self.i * self.j if self.fixup_launch else 0
            ),
            "fixup_tile_indices": self.fixup_tile_indices,
            "fixup_valid_elements": self.fixup_valid_elements,
            "fixup_read_bytes": self.fixup_read_bytes,
            "fixup_write_bytes": self.fixup_write_bytes,
            "fixup_operations": self.fixup_operations,
            "native_output_bytes": self.native_output_bytes,
            "conversion_source_expressions": {
                "abs": 4 * source_threads,
                "max": 6 * source_threads,
                "multiply": 4 * source_threads,
                "add": 6 * source_threads if summed_layout else 0,
                "shuffle": (6 if summed_layout else 3) * source_threads,
                "division": 2 * source_threads,
                "roundf": 4 * source_threads,
                "int8_conversion": 4 * source_threads,
                "metadata_writes": source_threads // 8,
            },
            "timing_completeness": "partial_inherited_resource_mapping",
            "matrix_geometry_policy": "inherited_mma_geometry_and_output_tile_wave_proxy",
            "source_traffic_is_not_measured_cache_or_dram_traffic": True,
            "unmodeled_terms": (
                "source_repeated_loads_are_not_promoted_to_backing_memory",
                "conversion_division_control_rounding_exchange_and_address_work",
                "compiled_instruction_and_cache_transaction_behavior",
                "temporary_allocation_lifetime_and_pool_granularity",
            ),
        }


def derive_mmq_work(
    *,
    m: int,
    k: int,
    n: int,
    weight_format: str,
    sm_count: int,
    shared_memory_per_block: int,
) -> MMQWork:
    """Derive the fixed-source MMQ work or reject an uncovered call exactly."""
    m = _positive_int("m", m)
    k = _positive_int("k", k)
    n = _positive_int("n", n)
    sm_count = _positive_int("sm_count", sm_count)
    shared_memory_per_block = _positive_int(
        "shared_memory_per_block", shared_memory_per_block
    )
    if not isinstance(weight_format, str) or not weight_format.strip():
        raise ValueError("weight_format must be non-empty text")
    weight_format = weight_format.strip().upper()
    if weight_format not in MMVQ_MAX_BATCH_SIZE:
        raise UnsupportedMMQ("unsupported fixed-source MMQ format {}".format(weight_format))
    qk = 32 if weight_format in _QK_32_FORMATS else 256
    if k % qk:
        raise UnsupportedMMQ(
            "fixed-source MMQ requires k to be a multiple of {} for {}".format(qk, weight_format)
        )
    if m <= MMVQ_MAX_BATCH_SIZE[weight_format]:
        raise UnsupportedMMQ("m={} remains in MMVQ range for {}".format(m, weight_format))
    if shared_memory_per_block < _MIN_SHARED_MEMORY_PER_BLOCK:
        raise UnsupportedMMQ("fixed-source MMQ requires at least 49152 shared bytes per block")

    conversion_layout = "DS4" if weight_format in _DS4_FORMATS else "D4"
    allowed_j = _J_NO_TAIL if n % _I == 0 else _J_TAIL
    chosen_j = None
    chosen_x = None
    for candidate in allowed_j:
        if _shared_bytes(candidate) > shared_memory_per_block:
            continue
        tiles = _ceil_div(m, candidate)
        if chosen_x is None or tiles < chosen_x:
            chosen_j, chosen_x = candidate, tiles
    if chosen_j is None:
        raise UnsupportedMMQ("no fixed-source MMQ J template fits shared memory")

    jmax = max((item for item in allowed_j if item <= min(m, 512)), default=0)
    k_padded = 512 * _ceil_div(k, 512)
    k_iterations = _ceil_div(k, 256)
    k_execution = 256 * k_iterations
    x_tiles = _ceil_div(m, chosen_j)
    y_tiles = _ceil_div(n, _I)
    u_tiles = x_tiles * y_tiles
    consumer_window = 1024 * _ceil_div(36 * chosen_j, 256)
    # Every process_tile K iteration loads two complete 128-value q8 windows,
    # even for K=896. The final window's exclusive end gives the source union
    # high-water because successive J windows and K segments overlap or touch.
    consumer_unique = (
        144 * m * (2 * k_iterations - 1)
        + 144 * (x_tiles - 1) * chosen_j + consumer_window
    )
    consumer_extra = consumer_unique - 144 * m * _ceil_div(k, 128)
    source_repeated = u_tiles * 2 * k_iterations * consumer_window
    source_allocation = 144 * m * k_padded // 128 + 144 * jmax
    if consumer_unique > source_allocation:
        raise UnsupportedMMQ("source allocation high-water does not cover MMQ consumer range")
    if u_tiles * (k // qk) >= _SOURCE_ALLOC_LIMIT:
        raise UnsupportedMMQ("MMQ source allocation integer limit would overflow")

    efficiency = (100 * u_tiles) // (sm_count * _ceil_div(u_tiles, sm_count))
    block_count = u_tiles if efficiency >= 90 else sm_count
    # Partition in *logical quantization blocks*, then align each boundary
    # relative to its own output tile. Replacing B with ceil(K/256) changes
    # source stream-K scheduling for a tail such as K=896 (B=28, R=8).
    blocks_per_row = k // qk
    blocks_per_iteration = 256 // qk
    boundaries = []
    for block in range(block_count + 1):
        raw = block * u_tiles * blocks_per_row // block_count
        boundaries.append(raw - (raw % blocks_per_row) % blocks_per_iteration)
    boundaries = tuple(boundaries)
    partial_tiles = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start != end and end % blocks_per_row:
            partial_tiles.append(end // blocks_per_row)
    fixup_tiles = tuple(sorted(set(partial_tiles)))
    valid_elements = 0
    for tile in fixup_tiles:
        y, x = divmod(tile, x_tiles)
        valid_m = min(chosen_j, m - x * chosen_j)
        valid_n = min(_I, n - y * _I)
        if valid_m <= 0 or valid_n <= 0:
            raise UnsupportedMMQ("MMQ fixup tile lies outside the native output")
        valid_elements += valid_m * valid_n

    return MMQWork(
        m=m, k=k, n=n, weight_format=weight_format, sm_count=sm_count,
        shared_memory_per_block=shared_memory_per_block, qk=qk,
        conversion_layout=conversion_layout, i=_I, j=chosen_j, jmax=jmax,
        k_padded=k_padded, k_execution=k_execution,
        x_tiles=x_tiles, y_tiles=y_tiles, u_tiles=u_tiles,
        k_iterations=k_iterations,
        consumer_window_bytes=consumer_window,
        consumer_extra_bytes=consumer_extra, source_repeated_bytes=source_repeated,
        source_allocation_bytes=source_allocation, block_count=block_count,
        stream_k_boundaries_qblocks=boundaries,
        partial_writer_count=len(partial_tiles), fixup_tile_indices=fixup_tiles,
        fixup_valid_elements=valid_elements,
    )


__all__ = ("MMVQ_MAX_BATCH_SIZE", "MMQWork", "UnsupportedMMQ", "derive_mmq_work")
