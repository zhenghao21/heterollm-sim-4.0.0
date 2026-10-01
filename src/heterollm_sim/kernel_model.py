"""Kernel-specific service models; rates are declarations, never inferred measurements.

A profile is scoped to one device/runtime build. Surfaces interpolate only complete
joint grid cells. Unknown kernels and out-of-domain shapes retain analytical costs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
import math
import re
from typing import Mapping

from .contracts import ResourceDemand, TaskCategory


def _number(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def _integer(value, name, *, zero=False):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    _number(value, name, zero=zero)


@dataclass(frozen=True)
class KernelSample:
    m: int
    n: int
    k: int
    device_ns: float
    analytical_ns: float
    stddev_ns: float
    sample_count: int
    evidence: str
    achieved_bandwidth_gb_s: float | None = None
    dispatch_signature: str | None = None

    def __post_init__(self):
        for key in ('m', 'n', 'k', 'sample_count'):
            _integer(getattr(self, key), key)
        if self.sample_count < 2:
            raise ValueError('kernel calibration needs repeated measurements')
        for key in ('device_ns', 'analytical_ns'):
            _number(getattr(self, key), key)
        _number(self.stddev_ns, 'stddev_ns', zero=True)
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError('kernel samples require independent microbenchmark evidence')
        if self.dispatch_signature is not None and (not isinstance(self.dispatch_signature, str) or not self.dispatch_signature):
            raise ValueError('dispatch_signature must be nonempty text')
        if self.achieved_bandwidth_gb_s is not None:
            _number(self.achieved_bandwidth_gb_s, 'achieved_bandwidth_gb_s')

    @property
    def shape(self):
        return self.m, self.n, self.k


@dataclass(frozen=True)
class KernelCapability:
    kernel_family: str
    weight_formats: tuple[str, ...]
    activation_dtype: str
    phase: str
    compute_primitive: str
    evidence: str
    internal_dtype: str = 'int8'
    operator: str = 'gemm'
    layout: str = 'contiguous'
    min_shape: tuple[int, int, int] = (1, 1, 1)
    max_shape: tuple[int, int, int] = (2147483647, 2147483647, 2147483647)
    cta_geometry: tuple[int, int, int] = (1, 128, 32)
    warps_per_cta: int = 4
    registers_per_thread: int = 32
    shared_memory_per_cta: int = 0
    native_resource_variants: tuple[tuple[int, int, int], ...] = ()
    attainable_efficiency: float = 1.0
    # DP4A/SIMT issue rate is distinct from tensor TOPS; operations are scalar MAC ops.
    dot_ops_per_sm_cycle: float = 128.0
    unpack_ops_per_weight: float = 0.0
    scale_ops_per_weight: float = 0.0
    reduction_ops_per_output: float = 0.0
    dependency_cycles: float = 0.0
    overlap_factor: float = 1.0
    transaction_efficiency: float = 1.0
    saturation_ctas_per_sm: int = 2
    memory_latency_ns: float = 0.0
    access_pattern: str = 'coalesced'
    transaction_bytes: int = 128
    samples: tuple[KernelSample, ...] = ()
    surface_model: str = 'calibrated_analytical'
    interpolation_space: str = 'log'
    # A measured wall does not identify per-resource occupancy. Only explicitly
    # declared resource ratios may be used for calibrated contention predictions.
    resource_occupancy_verified: bool = False
    cache_protocol: str = 'cold_streaming'
    epilogue: str = ''
    output_bits: int = 16
    accumulator_bits: int = 32
    attention_stream_k: bool = False
    attention_fixup_registers: int = 40
    attention_query_tile: int = 32
    attention_heads_per_tile: int = 2
    attention_kv_tile: int = 64
    attention_mask: str | None = None
    attention_kv_hidden_size: int | None = None
    attention_score_heads: int | None = None
    calibration_source_bound: bool = False
    # Optional source specialization selector.  A descriptor without one is
    # the analytical fallback; a bound descriptor is considered first when
    # the typed source dispatch has the same signature.
    dispatch_signature: str | None = None
    # Registered complete joint cells accepted by an independent holdout.
    calibration_cells: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = ()
    validation_relative_error: float | None = None

    def __post_init__(self):
        if type(self.attention_stream_k) is not bool:
            raise ValueError('attention_stream_k must be boolean')
        if type(self.calibration_source_bound) is not bool:
            raise ValueError('calibration_source_bound must be boolean')
        if self.dispatch_signature is not None and (
                not isinstance(self.dispatch_signature, str) or not self.dispatch_signature.strip()):
            raise ValueError('dispatch_signature must be nonempty text')
        if not isinstance(self.calibration_cells, tuple):
            raise ValueError('calibration_cells must be immutable')
        for cell in self.calibration_cells:
            if not isinstance(cell, tuple) or len(cell) != 2:
                raise ValueError('calibration cell requires lower/upper shapes')
            for shape in cell:
                if not isinstance(shape, tuple) or len(shape) != 3:
                    raise ValueError('calibration cell shape requires three integers')
                for value in shape:
                    _integer(value, 'calibration cell coordinate')
            if any(a > b for a, b in zip(*cell)):
                raise ValueError('inverted calibration cell')
        if self.validation_relative_error is not None:
            _number(self.validation_relative_error, 'validation_relative_error', zero=True)
        for key in ('attention_query_tile', 'attention_heads_per_tile', 'attention_kv_tile', 'attention_fixup_registers'):
            _integer(getattr(self, key), key)
        if self.attention_stream_k and (self.operator != 'attention' or self.samples):
            raise ValueError('stream-K requires analytical attention; whole-kernel samples are incompatible')
        if self.attention_mask not in (None, 'none', 'causal'):
            raise ValueError('attention_mask must be none or causal')
        if self.operator != 'attention' and self.attention_mask is not None:
            raise ValueError('attention_mask requires attention operator')
        for key in ('output_bits', 'accumulator_bits'):
            _integer(getattr(self, key), key)
        for key in ('attention_kv_hidden_size', 'attention_score_heads'):
            if getattr(self, key) is not None:
                _integer(getattr(self, key), key)
        if self.phase not in ('prefill', 'decode'):
            raise ValueError('kernel phase must be explicit prefill or decode')
        if self.operator not in ('gemm', 'attention'):
            raise ValueError('unsupported kernel operator')
        if self.compute_primitive not in ('tensor', 'dp4a', 'simt'):
            raise ValueError('compute primitive must be tensor, dp4a or simt')
        if self.internal_dtype not in ('fp16', 'bf16', 'fp32', 'int8'):
            raise ValueError('unsupported internal dtype')
        if self.activation_dtype not in ('fp16', 'bf16', 'fp32', 'int8'):
            raise ValueError('unsupported activation dtype')
        for key in ('kernel_family', 'evidence', 'layout', 'access_pattern', 'cache_protocol'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f'{key} must be nonempty')
        if not isinstance(self.weight_formats, tuple):
            raise ValueError('weight_formats must be an immutable tuple')
        if not isinstance(self.samples, tuple):
            raise ValueError('samples must be an immutable tuple')
        if not self.weight_formats or len(set(self.weight_formats)) != len(self.weight_formats) or any(not isinstance(x, str) or x != x.casefold() or not x for x in self.weight_formats):
            raise ValueError('weight_formats must be unique canonical lower-case names')
        for key in ('min_shape', 'max_shape', 'cta_geometry'):
            values = getattr(self, key)
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError(f'{key} must have three dimensions')
            for value in values:
                _integer(value, key)
        if any(a > b for a, b in zip(self.min_shape, self.max_shape)):
            raise ValueError('invalid dispatch range')
        for key in ('warps_per_cta', 'registers_per_thread', 'saturation_ctas_per_sm', 'transaction_bytes'):
            _integer(getattr(self, key), key)
        _integer(self.shared_memory_per_cta, 'shared_memory_per_cta', zero=True)
        if self.native_resource_variants:
            raise ValueError('native_resource_variants requires full binary/specialization binding; column-only resource tables are unsupported')
        for key in ('attainable_efficiency', 'transaction_efficiency'):
            _number(getattr(self, key), key)
            if getattr(self, key) > 1:
                raise ValueError(f'{key} must not exceed one')
        for key in ('dot_ops_per_sm_cycle', 'overlap_factor'):
            _number(getattr(self, key), key)
        if self.overlap_factor < 1:
            raise ValueError('overlap_factor cannot make service faster than its resource demands')
        for key in ('unpack_ops_per_weight', 'scale_ops_per_weight', 'reduction_ops_per_output', 'dependency_cycles', 'memory_latency_ns'):
            _number(getattr(self, key), key, zero=True)
        if self.interpolation_space not in ('linear', 'log'):
            raise ValueError('interpolation_space must be linear or log')
        if self.surface_model not in ('calibrated_analytical', 'measured_surrogate'):
            raise ValueError('unknown surface model')
        if self.resource_occupancy_verified is True:
            raise ValueError('resource occupancy cannot be verified by a device-wall sample alone')
        if type(self.resource_occupancy_verified) is not bool:
            raise ValueError('resource_occupancy_verified must be boolean')
        if any(not isinstance(s, KernelSample) for s in self.samples) or len({s.shape for s in self.samples}) != len(self.samples):
            raise ValueError('invalid or duplicate kernel samples')
        if self.samples and len(self.weight_formats) != 1:
            raise ValueError('measured surfaces must bind exactly one physical weight format')
        if self.samples and self.operator == 'attention' and (self.attention_kv_hidden_size is None or self.attention_score_heads is None):
            raise ValueError('attention calibration must bind KV width and score heads')
        if self.calibration_source_bound and not self.samples:
            raise ValueError('calibration_source_bound requires samples')
        if any(not self.contains(s.shape) for s in self.samples):
            raise ValueError('sample outside kernel dispatch range')
        sample_shapes = {sample.shape for sample in self.samples}
        for low, high in self.calibration_cells:
            corners = product(*(tuple(sorted({a, b})) for a, b in zip(low, high)))
            if any(point not in sample_shapes for point in corners):
                raise ValueError('calibration cell has unmeasured corner')

    def contains(self, shape):
        return all(lo <= x <= hi for x, lo, hi in zip(shape, self.min_shape, self.max_shape))


@dataclass(frozen=True)
class KernelModelProfile:
    hardware_id: str
    runtime_id: str
    architecture: str
    kernels: tuple[KernelCapability, ...] = ()
    registers_per_sm: int = 65536
    shared_memory_per_sm: int = 65536
    max_threads_per_sm: int = 2048
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 16
    warp_size: int = 32
    launch_ns: float | None = None
    graph_launch_ns: float | None = None
    launch_evidence: str | None = None
    # Optional independent runtime residual split.  ``launch_ns`` remains the
    # device enqueue component for backwards compatibility; this component is
    # the host/runtime submission envelope and is charged once per ordinary
    # kernel.  Graph replay is still charged once per explicit replay.
    runtime_submission_ns: float | None = None
    runtime_submission_evidence: str | None = None
    calibration_provenance: str = 'caller_declared'
    graph_enabled: bool = False
    stateful_l2: bool = False
    calibration_hardware_id: str | None = None
    calibration_runtime_sha256: str | None = None
    hardware_limits_evidence: str | None = None
    unverified_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ('hardware_id', 'runtime_id', 'architecture'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f'{key} is required')
        for key in ('registers_per_sm', 'shared_memory_per_sm', 'max_threads_per_sm', 'max_warps_per_sm', 'max_ctas_per_sm', 'warp_size'):
            _integer(getattr(self, key), key)
        for key in ('launch_ns', 'graph_launch_ns', 'runtime_submission_ns'):
            if getattr(self, key) is not None:
                _number(getattr(self, key), key, zero=True)
                if key == 'runtime_submission_ns':
                    if not self.runtime_submission_evidence:
                        raise ValueError('runtime submission calibration requires evidence')
                elif not self.launch_evidence:
                    raise ValueError('launch calibration requires evidence')
        if not isinstance(self.unverified_parameters, tuple) or any(not isinstance(x, str) or not x for x in self.unverified_parameters):
            raise ValueError('unverified_parameters must be a tuple of nonempty names')
        if self.hardware_limits_evidence is not None and (not isinstance(self.hardware_limits_evidence, str) or not self.hardware_limits_evidence.strip()):
            raise ValueError('hardware_limits_evidence must be nonempty text')
        if not isinstance(self.calibration_provenance, str) or not self.calibration_provenance.strip():
            raise ValueError('calibration_provenance must be nonempty text')
        if self.calibration_hardware_id is not None and not isinstance(self.calibration_hardware_id, str):
            raise ValueError('calibration_hardware_id must be text')
        if self.calibration_runtime_sha256 is not None and (
                not isinstance(self.calibration_runtime_sha256, str)
                or not re.fullmatch(r'[0-9a-f]{64}', self.calibration_runtime_sha256)):
            raise ValueError('calibration_runtime_sha256 must be lowercase SHA-256')
        if type(self.graph_enabled) is not bool or type(self.stateful_l2) is not bool:
            raise ValueError('graph_enabled and stateful_l2 must be boolean')
        if not isinstance(self.kernels, tuple) or any(not isinstance(k, KernelCapability) for k in self.kernels):
            raise ValueError('kernels must contain KernelCapability')

    def dispatch(self, *, shape, formats, dtype, phase, operator='gemm', layout='contiguous', output_bits=None, accumulator_bits=None, epilogue=None, attention_kv_hidden_size=None, attention_score_heads=None, attention_mask=None, dispatch_signature=None):
        matches = [kernel for kernel in self.kernels if kernel.operator == operator and kernel.phase == phase
                   and kernel.activation_dtype == dtype and kernel.layout == layout
                   and (output_bits is None or kernel.output_bits == output_bits)
                   and (accumulator_bits is None or kernel.accumulator_bits == accumulator_bits)
                   and (epilogue is None or kernel.epilogue == epilogue)
                   and (kernel.attention_kv_hidden_size is None or kernel.attention_kv_hidden_size == attention_kv_hidden_size)
                   and (kernel.attention_mask is None or kernel.attention_mask == attention_mask)
                   and (kernel.attention_score_heads is None or kernel.attention_score_heads == attention_score_heads)
                   and (kernel.dispatch_signature is None or kernel.dispatch_signature == dispatch_signature)
                   and set(formats).issubset(kernel.weight_formats) and formats and kernel.contains(shape)]
        specialized = [kernel for kernel in matches if kernel.dispatch_signature is not None]
        if specialized:
            matches = specialized
        if len(matches) > 1:
            raise ValueError('ambiguous kernel dispatch: ' + ', '.join(k.kernel_family for k in matches))
        return matches[0] if matches else None

    def occupancy(self, kernel):
        threads = kernel.warps_per_cta * self.warp_size
        limits = {
            'register': self.registers_per_sm // (kernel.registers_per_thread * threads),
            'shared_memory': self.shared_memory_per_sm // kernel.shared_memory_per_cta if kernel.shared_memory_per_cta else self.max_ctas_per_sm,
            'thread': self.max_threads_per_sm // threads,
            'warp': self.max_warps_per_sm // kernel.warps_per_cta,
            'architecture': self.max_ctas_per_sm,
        }
        resident = min(limits.values())
        if resident < 1:
            raise ValueError('kernel cannot reside on this GPU: register/shared/thread limits')
        return {'resident_ctas': resident, 'limits': limits,
                'occupancy': min(1.0, resident * kernel.warps_per_cta / self.max_warps_per_sm)}


def attention_stream_k_schedule(context, output_tiles, kv_tile_tokens, max_active_blocks):
    """Dense Ada+ stream-K rounding from locked fattn-common.cuh."""
    for value in (context, output_tiles, kv_tile_tokens, max_active_blocks):
        _integer(value, 'stream-K geometry')
    raw = min(max_active_blocks, math.ceil(context / kv_tile_tokens) * output_tiles)
    rounded = raw // output_tiles * output_tiles
    loss = 100 * (raw - rounded) // raw if rounded else 100
    blocks = rounded if loss <= 5 else raw
    return {'main_blocks': blocks, 'output_tiles': output_tiles,
            'uniform': blocks % output_tiles == 0,
            'fixup_required': output_tiles % blocks != 0,
            'blocks_per_tile': blocks // output_tiles if blocks % output_tiles == 0 else None}


def attention_uniform_fixup_traffic(*, blocks, output_tiles, columns_per_tile,
                                    live_rows, head_dim):
    """Unique logical bytes from the uniform CUDA write/merge branches.

    Metadata is loaded by each lane, but repeated addresses are counted once;
    cache transactions and instruction issue are separate from this byte count.
    Partial blocks write padded columns; final output and fixup skip padding.
    """
    for value in (blocks, output_tiles, columns_per_tile, live_rows, head_dim):
        _integer(value, 'uniform fixup geometry')
    if blocks % output_tiles or live_rows > output_tiles * columns_per_tile:
        raise ValueError('invalid uniform fixup geometry')
    parts = blocks // output_tiles
    allocation = blocks * columns_per_tile * (16 + 4 * head_dim) if parts > 1 else 0
    scratch_write = ((blocks - output_tiles) * columns_per_tile * 4 * head_dim
                     + blocks * columns_per_tile * 8) if parts > 1 else 0
    scratch_read = (live_rows * ((parts - 1) * 4 * head_dim + parts * 8)) if parts > 1 else 0
    return {'allocation_bytes': allocation, 'scratch_write_bytes': scratch_write,
            'scratch_read_bytes': scratch_read, 'output_bytes': live_rows * head_dim * 4,
            'byte_semantics': 'unique_logical_addresses_not_dram_transactions'}


def kernel_model_from_dict(raw: Mapping):
    def construct(cls, value, tuples=()):
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError(f'invalid {cls.__name__} fields')
        values = dict(value)
        for key in tuples:
            if key in values:
                if not isinstance(values[key], (list, tuple)):
                    raise ValueError(f'{key} must be an array')
                values[key] = tuple(values[key])
        return values
    values = construct(KernelModelProfile, raw, ('unverified_parameters',))
    if 'kernels' in values and not isinstance(values['kernels'], (list, tuple)):
        raise ValueError('kernels must be an array')
    kernels = []
    for entry in values.get('kernels', ()):
        item = construct(KernelCapability, entry, ('weight_formats', 'min_shape', 'max_shape', 'cta_geometry', 'native_resource_variants'))
        if 'calibration_cells' in item:
            item['calibration_cells'] = tuple(tuple(tuple(shape) for shape in cell) for cell in item['calibration_cells'])
        if 'samples' in item and not isinstance(item['samples'], (list, tuple)):
            raise ValueError('samples must be an array')
        item['samples'] = tuple(KernelSample(**construct(KernelSample, sample)) for sample in item.get('samples', ()))
        kernels.append(KernelCapability(**item))
    values['kernels'] = tuple(kernels)
    return KernelModelProfile(**values)


def performance_surface(kernel, shape, analytical_ns):
    """Exact samples, or log-shape multilinear interpolation over a FULL cell."""
    samples = {s.shape: s for s in kernel.samples}
    base = {'model': 'analytical', 'confidence': 'low', 'distance_to_calibration_domain': None,
            'uncertainty_ns': None, 'uncertainty_kind': 'unvalidated', 'extrapolated': False}
    if not samples:
        return analytical_ns, None, {**base, 'reason': 'no_measurements'}
    axes = [sorted({p[i] for p in samples}) for i in range(3)]
    if kernel.calibration_cells:
        supported = [cell for cell in kernel.calibration_cells
                     if all(a <= x <= b for x, a, b in zip(shape, *cell))]
        if not supported:
            distance = min(math.sqrt(sum(math.log2(x / min(max(x, a), b)) ** 2
                                        for x, a, b in zip(shape, *cell)))
                           for cell in kernel.calibration_cells)
            return analytical_ns, None, {**base, 'reason': 'outside_validated_calibration_cells',
                                         'distance_to_calibration_domain': distance}
        # Use a registered joint cell, not the bounding box of unrelated grids.
        low, high = min(supported, key=lambda cell: sum(math.log2(b / a) for a, b in zip(*cell)))
        axes = [sorted({a, b}) for a, b in zip(low, high)]
    distance = min(math.sqrt(sum(math.log2(x / y) ** 2 for x, y in zip(shape, point))) for point in samples)
    # Domain distance measures distance to the supported bounding box, not
    # nearest training point. Keep sampling density as a separate diagnostic.
    domain_distance = math.sqrt(sum(
        math.log2(x / min(max(x, axis[0]), axis[-1])) ** 2
        for x, axis in zip(shape, axes)))
    base['distance_to_calibration_domain'] = domain_distance
    base['nearest_sample_distance'] = distance
    brackets = []
    for x, axis in zip(shape, axes):
        low = [v for v in axis if v <= x]
        high = [v for v in axis if v >= x]
        if not low or not high:
            return analytical_ns, None, {**base, 'reason': 'outside_calibration_domain'}
        brackets.append(tuple(sorted({max(low), min(high)})))
    corners = list(product(*brackets))
    if any(point not in samples for point in corners):
        return analytical_ns, None, {**base, 'distance_to_calibration_domain': None,
                                     'reason': 'joint_cell_not_measured'}
    weighted = []
    for point in corners:
        weight = 1.0
        for x, coordinate, bounds in zip(shape, point, brackets):
            if len(bounds) == 2:
                fraction = ((x - bounds[0]) / (bounds[1] - bounds[0]) if kernel.interpolation_space == 'linear'
                            else math.log(x / bounds[0]) / math.log(bounds[1] / bounds[0]))
                weight *= fraction if coordinate == bounds[1] else 1 - fraction
        weighted.append((weight, samples[point]))
    exact = shape in samples
    signatures = {s.dispatch_signature for _, s in weighted}
    if not exact and len(signatures) > 1:
        return analytical_ns, None, {**base, 'distance_to_calibration_domain': None,
                                     'reason': 'kernel_specialization_boundary'}
    # Level 2 keeps the analytical resource model and applies a measured
    # correction ratio. Level 3 is the only mode allowed to use device_ns
    # directly for an exact sample.
    measured = kernel.surface_model == 'measured_surrogate'
    predicted = sum(w * (s.device_ns if measured else analytical_ns * s.device_ns / s.analytical_ns) for w, s in weighted)
    deviation = sum(w * s.stddev_ns * (1 if measured else analytical_ns / s.analytical_ns) for w, s in weighted)
    bandwidth = (sum(w * s.achieved_bandwidth_gb_s for w, s in weighted)
                 if all(s.achieved_bandwidth_gb_s is not None for _, s in weighted) else None)
    validation_error = kernel.validation_relative_error
    uncertainty = max(deviation, predicted * validation_error) if validation_error is not None else deviation
    return predicted, bandwidth, {
        **base, 'model': 'measured_surrogate' if measured else 'calibrated_analytical',
        'confidence': 'medium' if exact else 'low', 'reason': 'exact' if exact else 'joint_grid_interpolation',
        'distance_to_calibration_domain': 0.0, 'nearest_sample_distance': distance,
        'uncertainty_ns': uncertainty,
        'measurement_stddev_ns': deviation, 'validation_relative_error': validation_error,
        'uncertainty_kind': ('heldout_error_envelope_not_statistical_prediction_interval'
                             if validation_error is not None else 'measurement_stddev_not_prediction_interval'),
        'evidence': tuple(s.evidence for _, s in weighted),
        'resource_occupancy_calibrated': kernel.resource_occupancy_verified,
        'contention_assumption': 'analytical_resource_ratios_scaled_to_device_wall',
        'validated_llm_scope': False, 'evidence_integrity': 'caller_declared_not_verified',
    }


def execution_phase(workload):
    explicit = getattr(workload, 'execution_phase', 'unspecified')
    return explicit if explicit in ('prefill', 'decode') else 'unspecified'


def kernel_calibration_dispatch_signature(workload):
    """Canonical source specialization key used by Level-2 surfaces.

    MMVQ and MMQ use different source dispatch contracts.  The key deliberately
    contains only compile/source specialization fields; shape is still carried
    by ``KernelSample`` so interpolation cannot cross a different template.
    """
    type_ids = {
        'q4_k': 12, 'q5_k': 13, 'q6_k': 14,
        'iq3_s': 21, 'iq4_xs': 23,
        'q4_0': 2, 'q5_0': 6, 'q8_0': 8,
    }
    source = getattr(workload, 'mmvq_work', None)
    if source is not None:
        fmt = str(source.weight_format).casefold()
        type_id = type_ids.get(fmt)
        if type_id is None:
            return None
        return 'mmvq:type={}:m={}:fusion=0:small_k={}:halve_iters=0:warps={}:rows={}'.format(
            type_id, source.m, int(source.small_k), source.warps_per_cta, source.rows_per_cta)
    source = getattr(workload, 'mmq_work', None)
    if source is None:
        return None
    fmt = str(source.weight_format).casefold()
    type_id = type_ids.get(fmt)
    if type_id is None:
        return None
    # MMQ's compiled template is selected by physical format, J tile, the
    # fallback boundary (N % 128), and stream-K.  Runtime M/N/K remain sample
    # coordinates and must not be folded into this specialization key.
    return 'mmq:type={}:j={}:fallback={}:stream_k={}'.format(
        type_id, source.j, int(source.n % 128 != 0), int(source.fixup_launch))


def mmvq_calibration_dispatch_signature(workload):
    """Backward-compatible MMVQ-only spelling used by old surface tools."""
    source = getattr(workload, 'mmvq_work', None)
    if source is None:
        return None
    return kernel_calibration_dispatch_signature(workload)


def estimate_kernel(gpu, hbm, workload, *, attention=False):
    """Return None for an uncovered dispatch; never invent a CUDA implementation."""
    from .cost_models import CostEstimate, CostPhase
    profile = gpu.kernel_model
    if profile is None:
        return None
    if attention and workload.q4_mma_view_tokens_lower_bound:
        return None
    phase = execution_phase(workload)
    dtype = getattr(workload, 'activation_dtype', None) or {8: 'int8', 16: 'fp16', 32: 'fp32'}.get(workload.input_bits if attention else workload.activation_bits, 'unknown')
    shape = ((workload.batch_tokens, workload.context_tokens, workload.hidden_size) if attention else (workload.m, workload.n, workload.k))
    formats = ((workload.kv_artifact_format.casefold(),) if attention and workload.kv_artifact_format else
               (({8: 'int8', 16: 'fp16', 32: 'fp32'}.get(workload.effective_kv_input_bits, 'unknown'),) if attention else
                (tuple(f.casefold() for f in workload.packed_weight_formats) or ({8: 'int8', 16: 'fp16', 32: 'fp32'}.get(workload.weight_bits, 'unknown'),))))
    kernel = profile.dispatch(shape=shape, formats=formats, dtype=dtype, phase=phase,
                              operator='attention' if attention else 'gemm', layout=getattr(workload, 'layout', 'contiguous'),
                              output_bits=workload.output_bits, accumulator_bits=None if attention else workload.accumulator_bits,
                              epilogue=None if attention else workload.epilogue_name,
                              attention_kv_hidden_size=workload.effective_kv_hidden_size if attention else None,
                              attention_score_heads=workload.score_heads if attention else None,
                               attention_mask=('causal' if workload.causal_query_positions else 'none') if attention else None,
                              dispatch_signature=(kernel_calibration_dispatch_signature(workload)
                                                   if not attention else None))
    if kernel is None:
        return None
    cache_surface_excluded = bool(kernel.samples and kernel.cache_protocol != workload.cache_protocol)
    if cache_surface_excluded:
        kernel = replace(kernel, samples=(), calibration_source_bound=False, calibration_cells=(), validation_relative_error=None)
    source_surface_excluded = False
    source_surface_exclusion_reason = None
    if kernel.samples and not attention:
        source = (getattr(workload, 'mmvq_work', None)
                  or getattr(workload, 'mmq_work', None))
        if source is None and workload.mmq_work is not None:
            source_surface_excluded = True
            source_surface_exclusion_reason = 'source_launch_geometry_not_bound_to_surface'
        elif source is None:
            # A source-bound descriptor is only valid for the typed MMVQ path;
            # do not silently reuse it for an ordinary GEMM dispatch.
            if kernel.calibration_source_bound:
                source_surface_excluded = True
                source_surface_exclusion_reason = 'source_launch_geometry_not_bound_to_surface'
        elif not kernel.calibration_source_bound:
            # Level-2 samples need an explicit source contract.  Without it,
            # the analytical descriptor remains the safe fallback.
            source_surface_excluded = True
            source_surface_exclusion_reason = 'source_calibration_binding_missing'
        else:
            expected_signature = kernel_calibration_dispatch_signature(workload)
            sample_signatures = {
                sample.dispatch_signature for sample in kernel.samples
                if sample.m == workload.m
            }
            runtime_matches = (
                profile.calibration_runtime_sha256 is not None
                and getattr(source, 'runtime_binary_sha256', '') == profile.calibration_runtime_sha256
            )
            signature_matches = (
                expected_signature is not None
                and sample_signatures == {expected_signature}
            )
            if not runtime_matches or not signature_matches:
                source_surface_excluded = True
                source_surface_exclusion_reason = 'source_calibration_binding_mismatch'
    if source_surface_excluded:
        kernel = replace(kernel, samples=(), calibration_source_bound=False, calibration_cells=(), validation_relative_error=None)
    if kernel.output_bits != workload.output_bits:
        return None
    if not attention and kernel.accumulator_bits != workload.accumulator_bits:
        return None
    if attention and ((kernel.attention_kv_hidden_size is not None and kernel.attention_kv_hidden_size != workload.effective_kv_hidden_size)
                      or (kernel.attention_score_heads is not None and kernel.attention_score_heads != workload.score_heads)):
        return None
    if not attention and kernel.epilogue != workload.epilogue_name:
        return None
    if not attention and workload.mmvq_work is not None and 'mmvq' not in kernel.kernel_family.casefold():
        return None
    if not attention and workload.mmq_work is not None and 'mmq' not in kernel.kernel_family.casefold():
        return None
    source_residency = None
    if not attention and workload.mmvq_work is not None:
        source = workload.mmvq_work
        kernel = replace(kernel, warps_per_cta=source.warps_per_cta,
                         shared_memory_per_cta=max(kernel.shared_memory_per_cta,
                                                   source.reduction_shared_array_bytes))
        source_residency = 'mmvq_source_warps_shared_lower_bound_registers_descriptor'
    observed_resources = None
    if not attention and workload.mmvq_work is not None:
        from .native_kernel_resources import observed_mmvq_resources
        # API occupancy belongs to this exact device, not arbitrary profile limits.
        if (profile.registers_per_sm, profile.shared_memory_per_sm, profile.max_threads_per_sm,
                profile.max_warps_per_sm, profile.max_ctas_per_sm, profile.warp_size) == (65536,102400,1536,48,24,32):
            observed_resources = observed_mmvq_resources(workload.mmvq_work, hardware_id=profile.hardware_id)
        if observed_resources is not None:
            kernel = replace(kernel, registers_per_thread=observed_resources['registers'],
                             shared_memory_per_cta=observed_resources['static_shared'] +
                             observed_resources['dynamic_shared'] + observed_resources['reserved_shared'])
            source_residency = 'native_cubin_cuda_occupancy_api_exact_specialization'
    residency = profile.occupancy(kernel)
    if observed_resources is not None:
        resident = observed_resources['resident_ctas']
        residency = {**residency, 'analytical_resident_ctas': residency['resident_ctas'],
                     'resident_ctas': resident, 'occupancy': resident * kernel.warps_per_cta / profile.max_warps_per_sm,
                     'source': 'cuda_occupancy_api', 'native_resources': observed_resources}

    if not attention and workload.mmq_work is not None and workload.mmq_work.sm_count != gpu.sm_count:
        raise ValueError('MMQ SM count differs from kernel GPU')
    m, n, k = shape
    stream_k = None
    if attention and kernel.attention_stream_k:
        # The registered decode contract is paged and has no Stream-K
        # scratch/fixup or page-table binding.  Keep custom descriptors from
        # silently pricing an unsupported decode layout until that contract is
        # defined independently.
        if phase != 'prefill':
            return None
        # Single-sequence dense view only; no inference for paged/quantized KV.
        if (workload.kv_physical_contract is not None or workload.effective_kv_input_bits != 16
                or workload.output_bits != 32 or workload.hidden_size % workload.score_heads
                or workload.hidden_size // workload.score_heads != 64):
            return None
        gqa = workload.hidden_size // workload.effective_kv_hidden_size
        if (gqa < 1
                or workload.hidden_size % workload.effective_kv_hidden_size
                or gqa % kernel.attention_heads_per_tile
                or workload.score_heads % kernel.attention_heads_per_tile):
            return None
        tiles = math.ceil(m / kernel.attention_query_tile) * workload.score_heads // kernel.attention_heads_per_tile
        stream_k = attention_stream_k_schedule(n, tiles, kernel.attention_kv_tile,
                                              residency['resident_ctas'] * gpu.sm_count)
        stream_k.update(query_tokens=m, query_tile=kernel.attention_query_tile,
                        heads_per_tile=kernel.attention_heads_per_tile)
        if not stream_k['uniform']:
            return None  # General fixup requires a different work contract.
    ctas = (workload.mmq_work.block_count if not attention and workload.mmq_work is not None else
            workload.mmvq_work.cta_count if not attention and workload.mmvq_work is not None else
            math.ceil(m / kernel.cta_geometry[0]) * math.ceil(n / kernel.cta_geometry[1]))
    if stream_k is not None:
        ctas = stream_k['main_blocks']
    active = min(1.0, ctas / gpu.sm_count)
    efficiency = kernel.attainable_efficiency * residency['occupancy'] * active
    if attention:
        tiled_workload = (replace(workload, query_tile_tokens=kernel.attention_query_tile,
                                  key_tile_tokens=kernel.attention_kv_tile)
                          if stream_k is not None else workload)
        attention_work = attention_tile_work(tiled_workload)
        if stream_k is not None and not workload.causal_query_positions:
            attention_work = {**attention_work, 'executed_pairs':
                math.ceil(m / kernel.attention_query_tile) * kernel.attention_query_tile
                * math.ceil(n / kernel.attention_kv_tile) * kernel.attention_kv_tile}
        if stream_k is not None:
            attention_work.update(query_tile_tokens=kernel.attention_query_tile,
                                  key_tile_tokens=kernel.attention_kv_tile,
                                  tile_source='kernel_descriptor')
        if kernel.samples and workload.causal_query_positions:
            # A rectangular sample does not qualify a causal visibility pattern.
            kernel = replace(kernel, samples=(), calibration_source_bound=False, calibration_cells=(), validation_relative_error=None)
        ratio = attention_work['executed_pairs'] / (workload.batch_tokens * workload.context_tokens)
        read, write = workload.read_bytes, workload.write_bytes
        operations = workload.tensor_operations * ratio
        scalar_ops = workload.softmax_scalar_operations * ratio + workload.qk_scale_operations * ratio + workload.kv_dequant_operations
        reduce_ops = workload.score_elements * ratio * kernel.reduction_ops_per_output
        sfu_ops = workload.transcendental_operations * ratio
        unpack = 0  # Already included in workload.scalar_operations.
    else:
        read, write = workload.activation_bytes + workload.weight_bytes, workload.output_bytes
        operations = workload.operations
        if workload.mmq_work is not None:
            # Main only: conversion and fixup remain separate planner tasks.
            read += max(0, workload.mmq_work.weight_read_high_water_bytes - workload.mmq_work.logical_weight_bytes)
            write += workload.mmq_work.main_partial_write_bytes
            operations = workload.mmq_work.source_nominal_arithmetic_operations
        unpack = k * n * kernel.unpack_ops_per_weight
        scalar_ops = k * n * kernel.scale_ops_per_weight + workload.epilogue_operations + workload.packed_weight_transform_operations
        # Explicit descriptor unpack/scale replaces, rather than doubles, legacy dequant work.
        if kernel.unpack_ops_per_weight or kernel.scale_ops_per_weight:
            scalar_ops -= workload.packed_weight_transform_operations
        reduce_ops = m * n * kernel.reduction_ops_per_output
        sfu_ops = workload.epilogue_transcendental_operations
    if kernel.compute_primitive == 'tensor':
        internal_dtype = kernel.internal_dtype
        if internal_dtype not in gpu.tensor_core.supported_dtypes:
            raise ValueError('kernel tensor primitive unsupported by GPU')
        issued = operations
        if not attention and workload.mmq_work is None:
            tile = gpu.tensor_core
            issued = math.ceil(m / tile.mma_m) * math.ceil(n / tile.mma_n) * math.ceil(k / tile.mma_k) * tile.operations_per_mma
        dot_ns = issued / (gpu.tensor_core.peak_tops(internal_dtype) * 1000 * efficiency)
        dot_resource = gpu.tensor_core.resource_id
    else:
        dot_ns = operations / (gpu.sm_count * gpu.tensor_core.frequency_ghz * kernel.dot_ops_per_sm_cycle * efficiency)
        dot_resource = gpu.scalar_resource_id
    scalar_rate = gpu.sm_count * gpu.scalar_lanes_per_sm * gpu.scalar_ops_per_cycle * gpu.tensor_core.frequency_ghz * efficiency
    unpack_ns, scale_ns = unpack / scalar_rate, scalar_ops / scalar_rate
    reduce_ns = reduce_ops / (gpu.sm_count * gpu.reduction_ops_per_cycle_per_sm * gpu.tensor_core.frequency_ghz * efficiency)
    sfu_ns = sfu_ops / (gpu.sm_count * gpu.special_function_units_per_sm * gpu.special_function_ops_per_cycle * gpu.tensor_core.frequency_ghz * efficiency)
    scratch_bytes = 0
    scratch_traffic = None
    if stream_k is not None and stream_k['fixup_required']:
        columns = kernel.attention_query_tile * kernel.attention_heads_per_tile
        head_dim = workload.hidden_size // workload.score_heads
        scratch_traffic = attention_uniform_fixup_traffic(
            blocks=stream_k['main_blocks'], output_tiles=stream_k['output_tiles'],
            columns_per_tile=columns, live_rows=m * workload.score_heads, head_dim=head_dim)
        scratch_bytes = scratch_traffic['scratch_write_bytes']
        write += scratch_bytes
    transactions = math.ceil(read / kernel.transaction_bytes) * kernel.transaction_bytes + math.ceil(write / kernel.transaction_bytes) * kernel.transaction_bytes
    transaction_utilization = (read + write) / transactions if transactions else 1.0
    bw = hbm.effective_bandwidth_gb_s * kernel.transaction_efficiency * transaction_utilization * min(1.0, ctas / (gpu.sm_count * kernel.saturation_ctas_per_sm))
    memory_ns = hbm.memory_service(read, write, bandwidth_gb_s=bw)['service_ns'] + kernel.memory_latency_ns
    resources = {}
    def add(resource, ns, byte_count=0, work=0):
        old = resources.get(resource, ResourceDemand(resource, 0))
        resources[resource] = ResourceDemand(resource, old.service_ns + ns, bytes_moved=old.bytes_moved + byte_count, work_units=old.work_units + work)
    add(dot_resource, dot_ns, work=operations)
    add(gpu.scalar_resource_id, unpack_ns + scale_ns + reduce_ns + (0 if attention else workload.source_partial_service_ns), work=unpack + scalar_ops + reduce_ops)
    add(gpu.special_function_resource_id, sfu_ns, work=sfu_ops)
    add(hbm.resource_id, memory_ns, read + write)
    dependency_ns = kernel.dependency_cycles / gpu.tensor_core.frequency_ghz
    bottleneck = max(resources.items(), key=lambda item: item[1].service_ns)[0]
    analytical = max(d.service_ns for d in resources.values()) * kernel.overlap_factor + dependency_ns
    if profile.stateful_l2 and kernel.samples:
        raise ValueError('measured kernel surfaces require a fixed cache protocol; disable stateful_l2')
    predicted, measured_bw, prediction = performance_surface(kernel, shape, analytical)
    if attention:
        # Descriptor integration must not turn diagnostic hot-buffer evidence
        # into a production timing surface.  Report the explicit acceptance
        # decision alongside the retained causal/KV/resource ledger.
        from .attention_level2 import attention_level2_status
        prediction['level2_attention_status'] = attention_level2_status(
            phase, mask='causal' if workload.causal_query_positions else 'none',
            kv_format=formats[0] if len(formats) == 1 else 'mixed',
            stream_k=stream_k is not None,
        )
        if prediction['model'] == 'analytical':
            prediction['reason'] = 'attention_level2_not_production_qualified'
            prediction['fallback_kind'] = 'explicit_attention_analytical'
    if cache_surface_excluded:
        prediction['reason'] = 'measurement_cache_protocol_mismatch'
    if source_surface_excluded:
        prediction['reason'] = source_surface_exclusion_reason or 'source_launch_geometry_not_bound_to_surface'
    if measured_bw is not None:
        if measured_bw > hbm.bandwidth_gb_s:
            raise ValueError('measured achieved HBM bandwidth exceeds declared physical peak')
        # Keep the achieved rate as an auditable observation.  Do not rebuild
        # the analytical HBM demand from it: the standalone kernel trace and
        # the simulator's logical byte ledger can have different transaction
        # accounting (padding, source staging, and partial buffers).  The
        # calibrated wall correction below is the qualified service envelope;
        # replacing the HBM demand here would double-count that mismatch and
        # could make a validated sample slower than its measured wall.
        bw = measured_bw
    # Dependency/overlap stalls occupy the stream, not every hardware resource.
    factor = predicted / analytical if prediction['model'] != 'analytical' else 1.0
    demands = tuple(replace(d, energy_pj=(d.bytes_moved * hbm.energy_pj_per_byte if d.resource_id == hbm.resource_id else
                            d.work_units * (gpu.tensor_energy_pj_per_op if d.resource_id == gpu.tensor_core.resource_id else
                                            gpu.special_function_energy_pj_per_op if d.resource_id == gpu.special_function_resource_id else gpu.scalar_energy_pj_per_op)),
                            service_ns=d.service_ns * kernel.overlap_factor * factor)
                    for d in resources.values())
    envelope = max(predicted, max(d.service_ns for d in demands))
    device_launch = gpu.kernel_launch_ns if profile.launch_ns is None else profile.launch_ns
    runtime_submission = profile.runtime_submission_ns or 0.0
    launch = device_launch + runtime_submission
    stream_id = gpu.launch_resource_id + '.device_stream'
    demands += (ResourceDemand(stream_id, envelope),)
    prediction.update(prediction_ns=envelope, analytical_ns=analytical,
                      evidence_integrity=profile.calibration_provenance,
                      measured_wall_conflicts_with_resource_floor=envelope > predicted,
                      hardware_id=profile.hardware_id, runtime_id=profile.runtime_id,
                      runtime_residual={
                          'ordinary_device_launch_ns': (gpu.kernel_launch_ns if profile.launch_ns is None else profile.launch_ns),
                          'runtime_submission_ns': runtime_submission,
                          'ordinary_kernel_submission_ns': launch,
                          'graph_replay_bound': bool(profile.graph_enabled and profile.graph_launch_ns is not None),
                          'runtime_submission_evidence': profile.runtime_submission_evidence,
                      })
    audit = {'schema': 'kernel-aware-cost/v1', 'kernel_family': kernel.kernel_family,
             'dispatch_evidence': kernel.evidence, 'phase': phase, 'shape': dict(zip(('m', 'n', 'k'), shape)),
             'formats': formats, 'compute_primitive': kernel.compute_primitive, 'occupancy': residency,
             'cta_count': ctas, 'prediction': prediction,
             'hardware_limits_evidence': profile.hardware_limits_evidence,
             'unverified_parameters': profile.unverified_parameters,
             'residency_source': source_residency or 'kernel_descriptor',
             'warps_per_cta': kernel.warps_per_cta,
             'shared_memory_per_cta': kernel.shared_memory_per_cta,
             'pipeline_ns': {'load': memory_ns, 'unpack': unpack_ns, 'scale': scale_ns, 'dot': dot_ns, 'reduce': reduce_ns, 'sfu': sfu_ns, 'dependency': dependency_ns},
             'resource_service_ns': {key: demand.service_ns for key, demand in resources.items()},
             'resource_bottleneck': bottleneck,
             'analytical_service_ns': analytical,
             'achieved_bandwidth_gb_s': bw, 'bandwidth_source': 'measurement' if measured_bw is not None else 'uncalibrated_cta_transaction_model',
             'bandwidth_resource_floor_policy': (
                 'surface_wall_correction_scales_logical_resource_demands'
                 if measured_bw is not None else 'analytical_logical_resource_demand'
             ),
             'kernel_count': 1, 'read_bytes': read, 'write_bytes': write,
             **({'scratch_traffic': scratch_traffic} if scratch_traffic else {}),
             'cache_protocol': kernel.cache_protocol,
             'epilogue': kernel.epilogue,
             'output_bits': kernel.output_bits, 'accumulator_bits': kernel.accumulator_bits,
             **({'attention_tile_work': attention_work} if attention else {}),
             'attention_model': ('flash_prefill' if phase == 'prefill' else 'paged_decode') if attention else None}
    phases = []
    # Graph replay launch is owned by the graph lifecycle, not charged per node.
    # Only a real graph-lifecycle lowering may elide per-node submission.
    # A profile flag alone is not proof that this invocation is captured.
    if profile.graph_enabled:
        prediction['graph_launch_status'] = 'requires_runtime_capture_binding_per_node_launch_preserved'
    if launch or gpu.launch_energy_pj:
        phases.append(CostPhase('kernel_launch', TaskCategory.COMPUTE, (ResourceDemand(gpu.launch_resource_id, launch, energy_pj=gpu.launch_energy_pj),), {'launch_evidence': profile.launch_evidence, 'kernel_count': 1}))
    phases.append(CostPhase('gpu_fused_attention' if attention else 'gpu_gemm', TaskCategory.COMPUTE, demands, {'kernel_model': audit}))
    if scratch_bytes:
        elements = workload.batch_tokens * workload.hidden_size
        merges = stream_k['blocks_per_tile'] - 1
        # CUDA uniform fixup performs two exp operations and a weighted merge
        # per lane, followed by normalization. No measured-rate claim.
        fix_descriptor = replace(kernel, warps_per_cta=2,
                                 registers_per_thread=kernel.attention_fixup_registers,
                                 shared_memory_per_cta=0)
        fix_residency = profile.occupancy(fix_descriptor)
        fix_ctas = stream_k['output_tiles'] * kernel.attention_query_tile * kernel.attention_heads_per_tile
        fix_efficiency = kernel.attainable_efficiency * fix_residency['occupancy'] * min(1.0, fix_ctas / gpu.sm_count)
        fix_scalar_rate = gpu.sm_count * gpu.scalar_lanes_per_sm * gpu.scalar_ops_per_cycle * gpu.tensor_core.frequency_ghz * fix_efficiency
        fix_scalar = elements * (12 * merges + 1)
        fix_sfu = elements * 2 * merges
        fix_read = scratch_traffic['scratch_read_bytes'] + workload.write_bytes
        fix_write = workload.write_bytes
        fix_transactions = (math.ceil(fix_read / kernel.transaction_bytes)
                            + math.ceil(fix_write / kernel.transaction_bytes)) * kernel.transaction_bytes
        fix_bw = (hbm.effective_bandwidth_gb_s * kernel.transaction_efficiency
                  * (fix_read + fix_write) / fix_transactions
                  * min(1.0, fix_ctas / (gpu.sm_count * kernel.saturation_ctas_per_sm)))
        fix_demands = (
            ResourceDemand(gpu.scalar_resource_id, fix_scalar / fix_scalar_rate,
                           work_units=fix_scalar, energy_pj=fix_scalar * gpu.scalar_energy_pj_per_op),
            ResourceDemand(gpu.special_function_resource_id, fix_sfu / (gpu.sm_count * gpu.special_function_units_per_sm * gpu.special_function_ops_per_cycle * gpu.tensor_core.frequency_ghz * fix_efficiency),
                           work_units=fix_sfu, energy_pj=fix_sfu * gpu.special_function_energy_pj_per_op),
            ResourceDemand(hbm.resource_id, hbm.memory_service(fix_read, fix_write, bandwidth_gb_s=fix_bw)['service_ns'] + kernel.memory_latency_ns,
                           bytes_moved=fix_read + fix_write, energy_pj=(fix_read + fix_write) * hbm.energy_pj_per_byte))
        fix_ns = max(d.service_ns for d in fix_demands)
        fix_audit = {**audit, 'kernel_family': 'flash_attn_stream_k_fixup_uniform',
                     'read_bytes': fix_read, 'write_bytes': fix_write,
                     'occupancy': fix_residency, 'cta_count': fix_ctas,
                     'prediction': {**prediction, 'prediction_ns': fix_ns, 'analytical_ns': fix_ns,
                                    'reason': 'source_uniform_fixup_analytical_bound'},
                     'stream_k': stream_k, 'scratch_traffic': scratch_traffic,
                     'achieved_bandwidth_gb_s': fix_bw,
                     'traffic_model': 'source_uniform_unique_logical_bytes',
                     'pipeline_ns': {'dependency': 0}}
        if launch or gpu.launch_energy_pj:
            phases.append(CostPhase('kernel_launch', TaskCategory.COMPUTE,
                          (ResourceDemand(gpu.launch_resource_id, launch, energy_pj=gpu.launch_energy_pj),),
                          {'kernel_count': 1, 'launch_evidence': profile.launch_evidence}))
        phases.append(CostPhase('gpu_attention_stream_k_fixup', TaskCategory.COMPUTE,
                      fix_demands + (ResourceDemand(stream_id, fix_ns),),
                      {'kernel_model': fix_audit, 'persistent_kv_read_bytes': 0}))
    if stream_k is not None:
        audit['stream_k'] = stream_k
    return CostEstimate(tuple(phases), int(workload.tensor_operations if attention else workload.operations), residency['occupancy'],
                        {'model': 'kernel_aware', 'kernel_model': audit, 'prediction': prediction,
                         'kernel_count': 2 if scratch_bytes else 1, 'graph_launch_requires_lifecycle_owner': profile.graph_enabled,
                         'stateful_l2_requested': profile.stateful_l2,
                         **({'activation_bytes': workload.activation_bytes, 'weight_bytes': workload.weight_bytes,
                             'output_bytes': workload.output_bytes} if not attention else
                            {'query_read_bytes': workload.query_read_bytes, 'kv_read_bytes': workload.kv_read_bytes,
                             'read_bytes': workload.read_bytes, 'write_bytes': workload.write_bytes})})


def summarize_kernel_predictions(tasks):
    """Bounded coverage summary, without pretending correlated errors are IID."""
    rows = {}
    kernel_count = launch_count = fallback_count = 0
    for task in tasks:
        metadata = task.metadata
        if metadata.get('phase') == 'kernel_launch':
            launch_count += 1
            continue
        audit = metadata.get('phase_metadata', {}).get('kernel_model', metadata.get('cost_model', {}).get('kernel_model'))
        if not audit:
            fallback = metadata.get('cost_model', {}).get('prediction')
            if fallback:
                fallback_count += 1
            continue
        kernel_count += 1
        prediction = audit['prediction']
        key = (audit['kernel_family'], audit['phase'], prediction['model'], prediction['reason'])
        row = rows.setdefault(key, {'kernel_family': key[0], 'phase': key[1], 'model': key[2],
                                   'reason': key[3], 'count': 0, 'predicted_service_ns': 0.0,
                                   'confidence': prediction['confidence'], 'max_domain_distance': 0.0,
                                   'uncertainty_kind': prediction['uncertainty_kind']})
        row['count'] += 1
        row['predicted_service_ns'] += prediction['prediction_ns']
        distance = prediction.get('distance_to_calibration_domain')
        if distance is not None:
            row['max_domain_distance'] = max(row['max_domain_distance'], distance)
    return {'kernel_count': kernel_count, 'launch_count': launch_count, 'fallback_count': fallback_count, 'groups': tuple(rows.values()),
            'validated_llm_accuracy': False, 'end_to_end_uncertainty_ns': None,
            'uncertainty_note': 'per-kernel measurement spread is not an end-to-end prediction interval'}


def graph_launch_phases(profile, frontend_resource, kernel_count, *, captured):
    """Explicit graph-level API: caller owns capture/replay identity and DAG edges."""
    from .cost_models import CostPhase
    _integer(kernel_count, 'kernel_count')
    if type(captured) is not bool:
        raise ValueError('captured must be boolean')
    # ``launch_ns`` is the calibrated device enqueue component.  A standalone
    # runtime residual record can still provide the ordinary host submission
    # rate when no device enqueue value was supplied; graph replay always uses
    # its own measured replay rate.
    rate = profile.graph_launch_ns if captured else (
        profile.launch_ns if profile.launch_ns is not None else profile.runtime_submission_ns
    )
    if rate is None:
        raise ValueError('selected runtime submission mode has no calibrated launch cost')
    ordinary_rate = (
        (profile.launch_ns if profile.launch_ns is not None else 0.0)
        + (profile.runtime_submission_ns or 0.0)
    )
    return CostPhase('cuda_graph_replay' if captured else 'kernel_launch', TaskCategory.COMPUTE,
                     (ResourceDemand(frontend_resource, rate if captured else kernel_count * ordinary_rate),),
                     {'kernel_count': kernel_count, 'submission_count': 1 if captured else kernel_count,
                      'runtime_id': profile.runtime_id, 'evidence': profile.launch_evidence,
                      'captured': captured,
                      'runtime_residual': {
                          'ordinary_device_launch_ns': profile.launch_ns,
                          'runtime_submission_ns': profile.runtime_submission_ns or 0.0,
                          'graph_replay_ns': profile.graph_launch_ns if captured else None,
                          'fusion_launches_elided': max(0, kernel_count - 1) if captured else 0,
                          'runtime_submission_evidence': profile.runtime_submission_evidence,
                      }})


def apply_captured_graph_launch(tasks, profile):
    """Replace submissions only for explicitly bound capture groups.

    Each member needs cuda_graph_id AND cuda_graph_captured=True, assigned by
    the runtime adapter. Group IDs must identify a single replay, not a cached
    executable reused over multiple tokens. No graph is inferred from shapes.
    """
    if not profile.graph_enabled:
        return tuple(tasks)
    groups = {}
    for index, task in enumerate(tasks):
        if task.metadata.get('cuda_graph_captured') is True:
            group = task.metadata.get('cuda_graph_id')
            if not isinstance(group, str) or not group:
                raise ValueError('captured launch requires a graph replay ID')
            if task.metadata.get('phase') == 'kernel_launch':
                groups.setdefault(group, []).append(index)
    result = list(tasks)
    for group, indices in groups.items():
        members = [tasks[i] for i in indices]
        first = members[0]
        if len({tuple(d.resource_id for d in t.demands) for t in members}) != 1 or len(first.demands) != 1:
            raise ValueError('a graph replay must use one device frontend resource')
        launch = graph_launch_phases(profile, first.demands[0].resource_id, len(members), captured=True)
        for ordinal, index in enumerate(indices):
            task = tasks[index]
            metadata = {**task.metadata, 'graph_launch': {**launch.metadata, 'graph_id': group,
                         'cost_owner': first.task_id, 'charged': ordinal == 0}}
            demands = launch.demands if ordinal == 0 else tuple(replace(d,service_ns=0.0,energy_pj=0.0) for d in task.demands)
            dependencies = task.dependencies if ordinal == 0 else tuple(dict.fromkeys((*task.dependencies, first.task_id)))
            result[index] = replace(task, demands=demands, dependencies=dependencies, metadata=metadata)
    return tuple(result)

def load_microbenchmark_surface(path, *, expected_sha256, profile, kernel, analytical_cost):
    """Build samples from hash-bound raw repeated kernel timings, not LLM totals.

    analytical_cost(m,n,k) must use the same descriptor without its samples.
    This validates the supplied data/identity, not the truth of instrumentation.
    No measurements are launched, and no target LLM observations are consumed.
    """
    import hashlib
    import json
    from pathlib import Path
    import statistics

    if len(kernel.weight_formats) != 1 or kernel.operator != 'gemm':
        raise ValueError('this import schema supports one GEMM physical format only')
    data_bytes = Path(path).read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != expected_sha256:
        raise ValueError('microbenchmark evidence SHA-256 mismatch')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate measurement field: ' + key)
            result[key] = value
        return result
    data = json.loads(data_bytes.decode('utf-8-sig'), object_pairs_hook=unique)
    expected = {'schema': 'heterollm.kernel-microbenchmark/v1',
                'source_kind': 'independent_synthetic_operator', 'target_llm_latency_used': False,
                'measurement_boundary': 'cuda_device_kernel_interval',
                'hardware_id': profile.hardware_id, 'runtime_id': profile.runtime_id,
                'architecture': profile.architecture, 'kernel_family': kernel.kernel_family,
                'phase': kernel.phase, 'weight_format': kernel.weight_formats[0],
                'activation_dtype': kernel.activation_dtype, 'layout': kernel.layout,
                'output_bits': kernel.output_bits, 'accumulator_bits': kernel.accumulator_bits,
                'epilogue': kernel.epilogue, 'cache_protocol': kernel.cache_protocol}
    if not isinstance(data, dict) or set(data) != set(expected) | {'samples'}:
        raise ValueError('invalid independent microbenchmark schema')
    for key, value in expected.items():
        if type(data[key]) is not type(value) or data[key] != value:
            raise ValueError('microbenchmark identity mismatch: ' + key)
    if len(kernel.weight_formats) != 1 or kernel.operator != 'gemm':
        raise ValueError('this import schema supports one GEMM physical format only')
    if not isinstance(data['samples'], list) or not data['samples']:
        raise ValueError('microbenchmark samples must be a nonempty array')
    samples = []
    for row in data['samples']:
        required = {'m','n','k','device_durations_ns','hbm_bytes','hbm_durations_ns'}
        if (not isinstance(row, dict) or not required <= set(row)
                or set(row) - required - {'dispatch_signature'}):
            raise ValueError('invalid microbenchmark sample')
        shape = tuple(row[key] for key in ('m','n','k'))
        for value in shape:
            _integer(value, 'shape')
        durations = row['device_durations_ns']
        if not isinstance(durations,list) or len(durations) < 2:
            raise ValueError('at least two independent device timing samples required')
        for value in durations:
            _number(value, 'device duration')
        bw = None
        memory_durations = row['hbm_durations_ns']
        byte_count = row['hbm_bytes']
        if byte_count is not None or memory_durations is not None:
            _integer(byte_count, 'measured HBM bytes')
            if not isinstance(memory_durations,list) or len(memory_durations) < 2:
                raise ValueError('measured bandwidth requires repeated HBM timing samples')
            for value in memory_durations:
                _number(value, 'HBM duration')
            bw = byte_count / statistics.median(memory_durations)
        samples.append(KernelSample(*shape, statistics.median(durations), analytical_cost(*shape),
                                    statistics.stdev(durations), len(durations),
                                    'sha256:' + expected_sha256, bw, row.get('dispatch_signature')))
    return replace(kernel, samples=tuple(samples))

def llama_blackwell_analytical_profile(hardware_id, runtime_id, *, calibrated=False):
    """Opt-in source-rule approximation, NOT a native dispatch observation.

    Batch limits: locked ggml-cuda/mmvq.cu ggml_cuda_should_use_mmvq.
    Resources/rates are provisional analytical assumptions pending ptxas and
    independent microbenchmarks. No target LLM latency is used.
    """
    kernels = []
    limits = {'q4_0': 8, 'q4_1': 8, 'q5_0': 8, 'q5_1': 8, 'q8_0': 8,
              'q2_k': 5, 'q3_k': 5, 'q4_k': 5, 'q5_k': 6, 'q6_k': 7,
              'iq3_s': 8, 'iq4_xs': 8, 'iq4_nl': 8}
    evidence = ('source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu:ggml_cuda_should_use_mmvq; '
                'mmq.cu:ggml_cuda_should_use_mmq; conditional ordinary contiguous CUDA MUL_MAT; '
                'registers/smem/issue/bandwidth are unmeasured analytical assumptions')
    for fmt, limit in limits.items():
        for phase in ('prefill', 'decode'):
            for dtype in ('fp16', 'fp32'):
                for output in (16, 32):
                    for low, high, family, primitive, tile, warps in (
                        (1, min(4, limit), 'mmvq', 'dp4a', (1, 1, 32), 4),
                        (5, limit, 'mmvq', 'dp4a', (1, 2, 32), 2),
                        (limit + 1, 2147483647, 'mmq', 'tensor', (64, 64, 32), 4),
                    ):
                        if low > high:
                            continue
                        kernels.append(KernelCapability(
                            'cuda_' + family + '_' + fmt, (fmt,), dtype, phase, primitive, evidence,
                            min_shape=(low, 1, 32), max_shape=(high, 2147483647, 2147483647),
                            cta_geometry=tile, warps_per_cta=warps, output_bits=output,
                            registers_per_thread=32, shared_memory_per_cta=0,
                            dot_ops_per_sm_cycle=128, attainable_efficiency=0.65,
                            transaction_efficiency=0.8, reduction_ops_per_output=5))
    # Attention is a separate source family.  The independent synthetic
    # probes are diagnostic only (the probe DLL SHA differs from the calibrated
    # MMQ runtime and the causal M=64 fixup holdout is over the production
    # gate), so these descriptors install the phase/traffic contract while
    # deliberately retaining the analytical fallback.  No attention sample is
    # copied into a GEMM surface and no paged/quantized KV dispatch is inferred.
    attention_evidence = (
        'source:ggml-cuda/fattn.cuh+fattn-common.cuh; '
        'independent synthetic attention_causal_20260929 and '
        'attention_streamk_20260929; runtime/module SHA not matched to '
        'calibrated MMQ binary; holdouts diagnostic_only'
    )
    for phase, family, primitive, tile, warps in (
        ('prefill', 'flash_attention_prefill_l2', 'tensor', (64, 64, 32), 4),
        ('decode', 'paged_attention_decode_l2', 'tensor', (1, 64, 32), 4),
    ):
        for output in (16, 32):
            kernels.append(KernelCapability(
                family, ('fp16',), 'fp16', phase, primitive, attention_evidence,
                internal_dtype='fp16', operator='attention', output_bits=output,
                min_shape=(1, 1, 64), max_shape=(2147483647, 2147483647, 2147483647),
                cta_geometry=tile, warps_per_cta=warps,
                # Resource fields stay conservative until a same-runtime
                # resource trace is available; source symbols are not enough.
                registers_per_thread=32, shared_memory_per_cta=0,
                attainable_efficiency=0.65, transaction_efficiency=0.8,
                reduction_ops_per_output=5, attention_mask=None,
                attention_query_tile=64 if phase == 'prefill' else 1,
                attention_heads_per_tile=2 if phase == 'prefill' else 1,
                attention_kv_tile=64))
    if calibrated:
        from .kernel_level2_surfaces import (
            level2_samples,
            level2_resource,
            LEVEL2_VALIDATION_ERROR_BY_FORMAT,
            LEVEL2_VALIDATION_ERROR_BY_KEY,
        )

        def complete_cells(group):
            """Return only measured joint M/N/K cells for one specialization."""
            points = {sample.shape for sample in group}
            axes = [sorted({point[index] for point in points}) for index in range(3)]
            cells = []
            for m_low, m_high in zip(axes[0], axes[0][1:] or axes[0]):
                for n_low, n_high in zip(axes[1], axes[1][1:] or axes[1]):
                    for k_low, k_high in zip(axes[2], axes[2][1:] or axes[2]):
                        corners = tuple(product(
                            (m_low, m_high), (n_low, n_high), (k_low, k_high)
                        ))
                        if all(corner in points for corner in corners):
                            cells.append(((m_low, n_low, k_low),
                                          (m_high, n_high, k_high)))
            return tuple(cells)

        calibrated_kernels = []
        for kernel in kernels:
            samples = level2_samples(kernel.weight_formats[0], output_bits=kernel.output_bits,
                                      activation_dtype=kernel.activation_dtype,
                                      phase=kernel.phase,
                                      kernel_family=kernel.kernel_family.split('_', 2)[1])
            if samples and kernel.compute_primitive == 'dp4a' and kernel.output_bits == 32:
                # Do not let a three-dimensional surface silently interpolate
                # across M.  The current evidence only qualifies M=1/2/4;
                # retain an analytical descriptor for an unmeasured M=3 and
                # keep the ordinary M=5..limit range unchanged.  Small-K
                # source dispatch is split by its signature as well; IQ4_XS
                # has a distinct source specialization at the short-K point.
                low, high = kernel.min_shape[0], kernel.max_shape[0]
                if high < 2147483647:
                    for m in range(low, high + 1):
                        scoped = tuple(sample for sample in samples if sample.m == m)
                        # Generic analytical fallback for this exact M.  The
                        # dispatch routine gives a matching source-bound
                        # descriptor precedence when one covers the shape.
                        calibrated_kernels.append(replace(
                            kernel,
                            min_shape=(m, kernel.min_shape[1], kernel.min_shape[2]),
                            max_shape=(m, kernel.max_shape[1], kernel.max_shape[2]),
                            samples=(),
                            calibration_source_bound=False,
                        ))
                        signatures = sorted({sample.dispatch_signature for sample in scoped})
                        for signature in signatures:
                            group = tuple(sample for sample in scoped
                                          if sample.dispatch_signature == signature)
                            n_values = [sample.n for sample in group]
                            k_values = [sample.k for sample in group]
                            calibrated_kernels.append(replace(
                                kernel,
                                min_shape=(m, min(n_values), min(k_values)),
                                max_shape=(m, max(n_values), max(k_values)),
                                samples=group,
                                surface_model='calibrated_analytical',
                                cache_protocol='cold_streaming',
                                calibration_source_bound=True,
                                dispatch_signature=signature,
                                calibration_cells=complete_cells(group),
                                validation_relative_error=LEVEL2_VALIDATION_ERROR_BY_FORMAT.get(
                                    kernel.weight_formats[0]
                                ),
                            ))
                    continue
            if samples and kernel.compute_primitive == 'tensor' and kernel.output_bits == 32:
                # Prefill MMQ is a separate source family.  Keep the broad
                # analytical MMQ descriptor as fallback and give each exact
                # M/J/source specialization its own bounded surface so it
                # cannot interpolate from decode MMVQ or another MMQ tile.
                calibrated_kernels.append(kernel)
                signatures = sorted({sample.dispatch_signature for sample in samples})
                for signature in signatures:
                    group = tuple(sample for sample in samples
                                  if sample.dispatch_signature == signature)
                    if not group:
                        continue
                    for m in sorted({sample.m for sample in group}):
                        scoped = tuple(sample for sample in group if sample.m == m)
                        n_values = [sample.n for sample in scoped]
                        k_values = [sample.k for sample in scoped]
                        key = '{}:{}:{}'.format(kernel.phase, 'mmq', kernel.weight_formats[0])
                        validation_error = LEVEL2_VALIDATION_ERROR_BY_KEY.get(
                            key, LEVEL2_VALIDATION_ERROR_BY_FORMAT.get(kernel.weight_formats[0])
                        )
                        resource = level2_resource(
                            kernel.weight_formats[0], signature,
                            phase=kernel.phase, kernel_family='mmq'
                        ) or {}
                        calibrated_kernels.append(replace(
                            kernel,
                            min_shape=(m, min(n_values), min(k_values)),
                            max_shape=(m, max(n_values), max(k_values)),
                            samples=scoped,
                            surface_model='calibrated_analytical',
                            cache_protocol='cold_streaming',
                            calibration_source_bound=True,
                            dispatch_signature=signature,
                            calibration_cells=complete_cells(scoped),
                            validation_relative_error=validation_error,
                            registers_per_thread=int(resource.get('registers_per_thread', kernel.registers_per_thread)),
                            shared_memory_per_cta=int(resource.get('shared_memory_per_cta', kernel.shared_memory_per_cta)),
                            warps_per_cta=int(resource.get('warps_per_cta', kernel.warps_per_cta)),
                        ))
                continue
                kernel = replace(kernel, samples=samples, surface_model='calibrated_analytical',
                                 cache_protocol='cold_streaming', calibration_source_bound=True)
            calibrated_kernels.append(kernel)
        kernels = calibrated_kernels
    return KernelModelProfile(hardware_id, runtime_id,
                              'sm120-source-rule-calibrated' if calibrated else 'sm120-source-rule-analytical', tuple(kernels),
                              registers_per_sm=65536, shared_memory_per_sm=102400,
                              max_threads_per_sm=1536, max_warps_per_sm=48, max_ctas_per_sm=24,
                              calibration_provenance=(
                                  'independent_synthetic_cold_shape_holdouts_v1' if calibrated
                                  else 'caller_declared'),
                              calibration_hardware_id=('nvidia-rtx-5080' if calibrated else None),
                              calibration_runtime_sha256=(
                                  '8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297'
                                  if calibrated else None),
                              hardware_limits_evidence=(
                                  'RTX 5080 CC12.0 cuDeviceGetAttribute device0, 2026-09-29: '
                                  'attributes 39=1536,81=102400,82=65536,106=24; '
                                  'docs/KERNEL_AWARE_MODEL.md#rtx-5080-parameter-evidence'),
                              unverified_parameters=(
                                  'registers_per_thread', 'shared_memory_per_cta_without_source_binding',
                                  'attainable_efficiency', 'dot_ops_per_sm_cycle',
                                  'transaction_efficiency', 'reduction_ops_per_output',
                                  'saturation_ctas_per_sm', 'dependency_cycles', 'overlap_factor',
                                  'memory_latency_ns', 'cta_geometry_without_source_binding',
                                  'attention_resources_without_same_runtime_binding'))


def attention_tile_work(workload):
    """Executed tiled scores vs useful causal scores, for explicit Q positions."""
    positions = workload.causal_query_positions
    if not positions:
        return {'useful_pairs': workload.batch_tokens * workload.context_tokens,
                'executed_pairs': workload.batch_tokens * workload.context_tokens,
                'causal_tiles_skipped': 0, 'causal_positions_bound': False}
    qtile, ktile = workload.query_tile_tokens, workload.key_tile_tokens
    total_tiles = (workload.context_tokens + ktile - 1) // ktile
    executed = skipped = 0
    for offset in range(0, len(positions), qtile):
        group = positions[offset:offset + qtile]
        visible_tiles = (group[-1] + 1 + ktile - 1) // ktile
        executed += qtile * visible_tiles * ktile
        skipped += total_tiles - visible_tiles
    return {'useful_pairs': sum(p + 1 for p in positions), 'executed_pairs': executed,
            'causal_tiles_skipped': skipped, 'causal_positions_bound': True}
