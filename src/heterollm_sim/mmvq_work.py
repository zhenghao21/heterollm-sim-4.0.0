"""Source-derived CC1200 MMVQ geometry, with no latency or bandwidth model.

Only ordinary contiguous Q5_0/Q8_0 MUL_MAT is described.  An explicit runtime
contract is required; this module never chooses MMVQ from a model name and
never assumes that a CTA count is an HBM utilization percentage.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SOURCE_SHA256 = MappingProxyType({
    'ggml-cuda/mmvq.cu': '14026871030393662628abdbd4937d5cab72031e20ddf582c9de1d7b424bb368',
    'ggml-cuda/mmvq.cuh': '93ef0ea631585f93c35b00087d265e4ca45f82aaca8e7c363d382dc561d85fbd',
    'ggml-cuda/vecdotq.cuh': '9d165e8e36db9bdfb69bfc2301e550a58c4a3057907bbf4d075b5f3eb72347a2',
    'ggml-common.h': '3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea',
    'ggml-cuda/common.cuh': '1cc3186a56426d5b929e9c29f9fae2d61d583f5f7abf0764d363c21cbbcbe932',
})

class UnsupportedMMVQ(ValueError):
    """The narrow source/runtime contract cannot describe this invocation."""


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise UnsupportedMMVQ(name + ' must be a positive integer')
    return value


@dataclass(frozen=True)
class MMVQSourceContract:
    compute_capability: int
    highest_compiled_arch: int
    warp_size: int
    source_hashes: Mapping[str, str]
    runtime_binary_sha256: str
    ordinary_contiguous_2d: bool
    force_cublas: bool
    mmvq_dispatch_enabled: bool
    has_ids: bool = False
    has_fusion: bool = False
    channels: int = 1
    samples: int = 1

    def __post_init__(self) -> None:
        if dict(self.source_hashes) != dict(SOURCE_SHA256):
            raise UnsupportedMMVQ('immutable MMVQ source identity differs')
        object.__setattr__(self, 'source_hashes', MappingProxyType(dict(self.source_hashes)))
        if (type(self.compute_capability) is not int or self.compute_capability != 1200
                or type(self.highest_compiled_arch) is not int or self.highest_compiled_arch != 1200
                or type(self.warp_size) is not int or self.warp_size != 32):
            raise UnsupportedMMVQ('requires CC1200 compiled CC1200 and warp32')
        if (not isinstance(self.runtime_binary_sha256, str) or len(self.runtime_binary_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.runtime_binary_sha256)):
            raise UnsupportedMMVQ('runtime binary SHA required')
        if (self.ordinary_contiguous_2d is not True or self.force_cublas is not False
                or self.mmvq_dispatch_enabled is not True or self.has_ids is not False
                or self.has_fusion is not False or type(self.channels) is not int or self.channels != 1
                or type(self.samples) is not int or self.samples != 1):
            raise UnsupportedMMVQ('only declared ordinary unfused single-channel/sample MMVQ supported')


def verify_mmvq_source_tree(src_directory: str | Path) -> Mapping[str, str]:
    """Optional explicit read-only identity check; importing performs no IO."""
    root = Path(src_directory)
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        with (root / relative).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected:
            raise UnsupportedMMVQ('source hash mismatch: ' + relative)
        actual[relative] = digest
    return MappingProxyType(actual)


@dataclass(frozen=True)
class MMVQWork:
    m: int
    k: int
    n: int
    weight_format: str
    runtime_binary_sha256: str
    qk: int
    qi: int
    vdr: int
    warps_per_cta: int
    rows_per_cta: int
    small_k: bool
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    blocks_per_row: int
    blocks_per_loop_step: int
    loop_iterations_by_thread: tuple[int, ...]
    weight_block_bytes: int

    @property
    def cta_count(self) -> int:
        return self.grid[0]

    @property
    def launched_warps(self) -> int:
        return self.cta_count * self.warps_per_cta

    @property
    def active_k_threads_per_cta(self) -> int:
        return sum(count > 0 for count in self.loop_iterations_by_thread)

    @property
    def maximum_thread_k_iterations(self) -> int:
        return max(self.loop_iterations_by_thread)

    @property
    def source_vector_dot_calls(self) -> int:
        return sum(self.loop_iterations_by_thread) * self.m * self.rows_per_cta * self.cta_count

    @property
    def logical_weight_bytes(self) -> int:
        return self.n * self.blocks_per_row * self.weight_block_bytes

    @property
    def logical_input_f32_bytes(self) -> int:
        return 4 * self.m * self.k

    @property
    def consumer_q8_1_unique_bytes(self) -> int:
        return 36 * self.m * self.k // 32

    @property
    def output_f32_bytes(self) -> int:
        return 4 * self.m * self.n

    @property
    def reduction_shared_array_bytes(self) -> int:
        # Main tmp_shared only. The unused gate array and compiler elimination
        # are not a proof of final cubin static allocation.
        return 4 * max(1, self.warps_per_cta - 1) * self.m * self.rows_per_cta * 32

    def to_metadata(self) -> dict[str, object]:
        return {**asdict(self), 'schema': 'heterollm.source-mmvq-work/v1',
            'source_hashes': dict(SOURCE_SHA256), 'parameter_table': 'MMVQ_PARAMETERS_GENERIC',
            'template_batch': self.m, 'has_fusion': False, 'halve_iters': False,
            'cta_count': self.cta_count, 'launched_warps': self.launched_warps,
            'active_k_threads_per_cta': self.active_k_threads_per_cta,
            'maximum_thread_k_iterations': self.maximum_thread_k_iterations,
            'source_vector_dot_calls': self.source_vector_dot_calls,
            'logical_weight_bytes': self.logical_weight_bytes,
            'logical_input_f32_bytes': self.logical_input_f32_bytes,
            'consumer_q8_1_unique_bytes': self.consumer_q8_1_unique_bytes,
            'output_f32_bytes': self.output_f32_bytes,
            'reduction_shared_array_bytes': self.reduction_shared_array_bytes,
            'dynamic_shared_launch_bytes': 0, 'cta_barriers': 1,
            'final_warp_reduce_rounds': 5, 'native_dispatch_proven': False,
            'binary_source_equivalence_proven': False, 'cost_model_applied': False,
            'hbm_efficiency': None, 'measured_occupancy': None,
            'work_units': 'source vector-dot calls and logical bytes; not SASS instructions or DRAM transactions',
            'timing_completeness': 'source_geometry_only_no_cost_rates',
            'utilization_candidate': 'Use vector CTA/warp/K-loop geometry instead of MMA tiles only after independent bandwidth/occupancy validation.'}


def derive_mmvq_work(*, m: int, k: int, n: int, weight_format: str,
                     contract: MMVQSourceContract) -> MMVQWork:
    """Derive exact source launch geometry, not GPU performance.

    Q5_0/Q8_0 dispatch on the generic CC1200 path accepts M<=8. M1 may
    increase rows/CTA to four when logical K-blocks fit the strict small-K
    predicate. M2..4 use four warps/two rows; M5..8 two warps/two rows.
    Odd/partial output tiles require independent weight-tail allocation proof
    and are deliberately not handled by this initial contract.
    """
    if not isinstance(contract, MMVQSourceContract):
        raise UnsupportedMMVQ('explicit source/runtime contract required')
    for value, name in ((m, 'M'), (k, 'K'), (n, 'N')):
        _positive(value, name)
    if not isinstance(weight_format, str) or weight_format not in {'Q5_0', 'Q8_0'}:
        raise UnsupportedMMVQ('format outside Q5_0/Q8_0 verified geometry scope')
    if m > 8 or k % 32:
        raise UnsupportedMMVQ('outside MMVQ batch or logical K block domain')
    qi, block_bytes = (4, 22) if weight_format == 'Q5_0' else (8, 34)
    vdr, qk, warp = 2, 32, 32
    warps = 4 if m <= 4 else 2
    qblocks = k // qk
    small = m == 1 and qblocks < warps * vdr * warp // qi
    rows = warps if small else 1 if m == 1 else 2
    if n % rows:
        raise UnsupportedMMVQ('partial output CTA weight-tail allocation not proven')
    step = vdr * warps * warp // qi
    iterations = tuple(max(0, (qblocks - 1 - tid // (qi // vdr)) // step + 1)
                       for tid in range(warp * warps))
    return MMVQWork(m, k, n, weight_format, contract.runtime_binary_sha256, qk, qi, vdr,
                   warps, rows, small, (n // rows, 1, 1), (warp, warps, 1),
                   qblocks, step, iterations, block_bytes)
