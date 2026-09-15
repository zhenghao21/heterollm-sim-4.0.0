"""Reference lowering pipeline from scenario IR to deterministic ScheduleIR.

This is intentionally a vertical-slice compiler rather than a model-name
adapter.  It handles generic Dense/MoE transformer layer parameters and keeps
every unimplemented optimization explicit instead of silently applying an
optimistic scaling factor.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import OrderedDict, deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from typing import Callable, Dict, FrozenSet, Hashable, Iterator, List, Optional, Sequence, Set, Tuple

from . import __version__
from .config import ScenarioConfig
from .calibration import (
    calibrate_cost_phase,
    decode_first_invocation_extra_ns,
    launch_calibration_ns,
    phase_boundary_calibration_ns,
    request_marker_calibration_ns,
    profile_from_mapping,
)
from .communication import TopologyRouter, plan_collective
from .contracts import (
    ANALYTICAL_MODEL_VERSION,
    EvidenceStatus,
    OperatorClass,
    ResourceDemand,
    RunManifest,
    TaskCategory,
    TaskSpec,
    TraceMarker,
    _PreparedExecutionStage,
    _PreparedExecutionTask,
)
from .cost_models import (
    CPUIQPanelDispatch,
    CPUProfile,
    CostEstimate,
    DigitalSramCimProfile,
    ElementwiseWorkload,
    FusedAttentionKVPhysicalContract,
    FusedAttentionWorkload,
    GemmWorkload,
    GPUProfile,
    HBMProfile,
    HostMemoryProfile,
    MemoryWorkload,
    ReductionWorkload,
    TensorKernelWorkload,
    _dma_setup_service,
    estimate_cim_gemm,
    estimate_cpu_elementwise,
    estimate_cpu_gemm,
    estimate_cpu_logical_stream,
    estimate_cpu_memory,
    estimate_cpu_reduction,
    estimate_gpu_elementwise,
    estimate_gpu_fused_attention,
    estimate_gpu_gemm,
    estimate_gpu_memory,
    estimate_gpu_reduction,
    estimate_gpu_tensor_kernel,
)
from .engine import ScheduleIR
from .execution_control import ExecutionControl
from .ir import (
    ACTIVE_MEMORY_COMPONENT_KINDS,
    OFFLOAD_STORAGE_COMPONENT_KINDS,
    STORAGE_COMPONENT_KINDS,
    ComponentSpec,
    LayerSpec,
    MTPExecutionDescriptor,
    ModelGraphExecutionView,
    RequestSpec,
    WorkloadSpec,
    model_graph_execution_view,
    normalize_component_kind,
)
from .control_plane_state import (
    control_plane_decision,
    is_control_plane_generated_tensor,
    mapping_fingerprint_status,
)
from .mtp import MTPRequestCursor, expected_prefix_tokens, round_accepted_prefix
from .mmq_work import MMVQ_MAX_BATCH_SIZE, MMQWork, UnsupportedMMQ, derive_mmq_work
from .parallel import LogicalRank, ParallelPlan, build_parallel_plan, shard_extent
from .precision import canonical_dtype, dtype_bits, layer_precision_bits
from .projection_descriptors import (
    ARTIFACT_QUANTIZATION_REGISTRY as _ARTIFACT_QUANTIZATION_REGISTRY,
    ArtifactQuantizationSpec as _ArtifactQuantizationSpec,
    AttentionExecutionDescriptor as _AttentionExecutionDescriptor,
    MaterializedWeightProjection as _MaterializedWeightProjection,
    canonical_artifact_quantization as _canonical_primitive_artifact_quantization,
    materialize_weight_projection,
    resolve_attention_execution_descriptor,
    resolve_weight_projection,
)
from .qwen35_attention_work import (
    SOURCE_KEY as _QWEN35_ATTENTION_SOURCE_KEY,
    Qwen35AttentionSourceWork as _Qwen35AttentionSourceWork,
    resolve_source_work as _resolve_qwen35_source_work,
)
from .final_layer_output_selection import (
    SOURCE_KEY as _FINAL_OUTPUT_SELECTION_KEY,
    FinalLayerOutputSelection as _FinalOutputSelection,
    resolve_declaration as _resolve_final_output_declaration,
)
from .serde import stable_hash


def _model_gguf_sha256(model: object) -> Optional[str]:
    """Read the GGUF digest from both legacy and graph-nested metadata."""
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("gguf_sha256")
    if value is None and isinstance(metadata.get("metadata"), Mapping):
        value = metadata["metadata"].get("gguf_sha256")
    return str(value) if value is not None else None
from .scalable_serving import (
    CompiledGraphExecutor,
    ExactTemplateCache,
    TaskExecutionRecord,
    execute_cost_schedule,
)
from .topology import assert_valid_topology


_NON_INHERITED_METADATA_SUBTREES = frozenset(
    {"weight_projection_descriptors", "attention_execution_descriptor", _QWEN35_ATTENTION_SOURCE_KEY}
)
_UNSAFE_TASK_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
_HOST_OUTPUT_COMPLETION_INTERRUPT_RESOURCE_ID = "interrupt"
_LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION = "llama_cpp_cpu_chain"
_LLAMA_CPP_CPU_SAMPLER_RECORD_BYTES = 12
_LLAMA_CPP_CPU_SAMPLER_ROWS_EXECUTION = "serial_slot_loop"
_LLAMA_CPP_CPU_SAMPLER_TOP_K_ALGORITHM = "std_partial_sort"
_LLAMA_CPP_CPU_SAMPLER_PROVENANCE = (
    "llama.cpp b10760 commit "
    "0f3a71be15af836d277c9f918adfafb45732677e; "
    "common/sampling.cpp:130-162,594-640; "
    "src/llama-sampler.cpp:192-204; "
    "tools/server/server-context.cpp:2703-2724,3778-3834"
)
_LLAMA_CPP_CPU_SAMPLER_SCAN_PROVENANCE = (
    "llama.cpp 0f3a71be15af836d277c9f918adfafb45732677e "
    "llama_sampler_top_k_impl/llama_token_data_array_partial_sort_inplace; "
    "Microsoft STL vs-2022-17.13 stl/inc/algorithm partial_sort tail loop; "
    "LLVM llvmorg-19.1.1 libcxx/include/__algorithm/partial_sort.h"
)


class MappingResolutionError(ValueError):
    """A placement requested a target that lowering must not rewrite."""

    def __init__(
        self,
        message: str,
        details: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.details = dict(details)


class ScenarioValidationError(ValueError):
    """Scenario validation failed with machine-readable diagnostics."""

    def __init__(
        self,
        message: str,
        diagnostics: Sequence[Mapping[str, object]] = (),
    ) -> None:
        super().__init__(message)
        serialized = [dict(diagnostic) for diagnostic in diagnostics]
        self.details: Dict[str, object] = {"diagnostics": serialized}
        if len(serialized) == 1:
            self.details.update(serialized[0])


@dataclass(frozen=True)
class _MetadataLookup:
    found: bool
    value: object = None


@dataclass
class _MetadataSourceCache:
    """Strongly own one source and its exact BFS lookup results."""

    source: Mapping[object, object]
    lookups: Dict[FrozenSet[str], _MetadataLookup]
    flat_values: Optional[Mapping[str, Tuple[int, object]]]
    identity_token: int


_SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY = object()


@dataclass
class CompilationContext:
    """Immutable-scenario indexes shared by one exact compilation.

    The scenario IR is frozen.  Keeping these derived projections request-local
    avoids rebuilding the typed execution graph and re-scanning weight metadata
    for every GEMM without introducing a process-global cache or stale-input
    semantics.  Mutable dictionaries below are memoization indexes only; they
    never contain runtime KV state or task dependencies.  Leaf cost results
    use exact immutable keys and a bounded LRU local to this context.
    """

    scenario: ScenarioConfig
    _execution_view_value: Optional[ModelGraphExecutionView] = None
    _execution_layers_value: Optional[Tuple[LayerSpec, ...]] = None
    _mtp_descriptors_by_weight: Optional[
        Dict[str, MTPExecutionDescriptor]
    ] = None
    _control_plane_decision_value: Optional[Mapping[object, object]] = None
    _canonical_weight_ids: Optional[Dict[str, str]] = None
    _rank_shards_by_tensor: Optional[
        Dict[str, Tuple[Mapping[object, object], ...]]
    ] = None
    _rank_shards_by_rank_target: Optional[
        Dict[Tuple[str, str, str], Mapping[object, object]]
    ] = None
    _rank_shards_by_rank: Optional[
        Dict[Tuple[str, str], Mapping[object, object]]
    ] = None
    _rank_shards_by_target: Optional[
        Dict[Tuple[str, str], Mapping[object, object]]
    ] = None
    _parallel_plan_value: Optional[ParallelPlan] = None
    _router_value: Optional[TopologyRouter] = None
    _component_map_value: Optional[Dict[str, ComponentSpec]] = None
    _invariant_values: Optional[Dict[Tuple[object, ...], object]] = None
    _metadata_sources_by_id: Optional[
        OrderedDict[int, _MetadataSourceCache]
    ] = None
    _metadata_lookup_hits: int = 0
    _metadata_lookup_misses: int = 0
    _metadata_flat_resolutions: int = 0
    _metadata_source_next_token: int = 0
    _artifact_workload_values: Optional[
        OrderedDict[
            Hashable,
            Tuple[object, Tuple[Tuple[str, object], ...]],
        ]
    ] = None
    _artifact_workload_hits: int = 0
    _artifact_workload_misses: int = 0
    _artifact_workload_bypasses: int = 0
    metadata_source_cache_entries: int = 65536
    artifact_workload_cache_entries: int = 65536
    leaf_cache_entries: int = 4096
    eager_full_attention_segments: bool = True
    compiled_serving_invocation_segments: bool = False
    _serving_invocation_lean_replay_capability: Optional[object] = None
    _leaf_values: Optional[
        "OrderedDict[Hashable, object]"
    ] = None
    _leaf_hits: int = 0
    _leaf_misses: int = 0
    _leaf_bypasses: int = 0
    _leaf_evictions: int = 0
    _leaf_hits_by_kind: Optional[Dict[str, int]] = None
    _leaf_misses_by_kind: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.leaf_cache_entries <= 0:
            raise ValueError("leaf_cache_entries must be positive")
        if self.metadata_source_cache_entries <= 0:
            raise ValueError("metadata_source_cache_entries must be positive")
        if self.artifact_workload_cache_entries <= 0:
            raise ValueError(
                "artifact_workload_cache_entries must be positive"
            )

    def component_map(self) -> Mapping[str, ComponentSpec]:
        if self._component_map_value is None:
            self._component_map_value = self.scenario.hardware.component_map()
        return self._component_map_value

    def component(self, component_id: str) -> ComponentSpec:
        try:
            return self.component_map()[component_id]
        except KeyError:
            raise KeyError("unknown component_id: {}".format(component_id))

    def invariant(
        self,
        key: Tuple[object, ...],
        factory: Callable[[], object],
    ) -> object:
        if self._invariant_values is None:
            self._invariant_values = {}
        try:
            return self._invariant_values[key]
        except KeyError:
            value = factory()
            self._invariant_values[key] = value
            return value

    def _metadata_source_cache(
        self,
        source: Mapping[object, object],
    ) -> _MetadataSourceCache:
        """Return an identity-validated cache that strongly owns ``source``."""

        if self._metadata_sources_by_id is None:
            self._metadata_sources_by_id = OrderedDict()
        marker = id(source)
        cached = self._metadata_sources_by_id.get(marker)
        if cached is not None and cached.source is source:
            self._metadata_sources_by_id.move_to_end(marker)
            return cached
        if cached is not None:
            self._metadata_sources_by_id.pop(marker, None)

        flat_values: Optional[Dict[str, Tuple[int, object]]] = None
        if (
            type(source) is dict
            and all(
                type(key) is str and not isinstance(value, Mapping)
                for key, value in source.items()
            )
        ):
            flat_values = {}
            for encounter, (key, value) in enumerate(source.items()):
                flat_values.setdefault(
                    key.casefold(),
                    (encounter, value),
                )
        identity_token = self._metadata_source_next_token
        self._metadata_source_next_token += 1
        cached = _MetadataSourceCache(
            source,
            {},
            flat_values,
            identity_token,
        )
        # ``source`` is retained strongly and identity is checked on every
        # access, so an object id can never alias stale results after reuse.
        self._metadata_sources_by_id[marker] = cached
        self._metadata_sources_by_id.move_to_end(marker)
        while (
            len(self._metadata_sources_by_id)
            > self.metadata_source_cache_entries
        ):
            self._metadata_sources_by_id.popitem(last=False)
        return cached

    def metadata_source_identity(
        self,
        source: Mapping[object, object],
    ) -> int:
        """Return a context-owned token while strongly retaining ``source``."""

        return self._metadata_source_cache(source).identity_token

    def metadata_value(
        self,
        sources: Sequence[Mapping[str, object]],
        keys: Sequence[str],
    ) -> object:
        """Resolve metadata from context-local immutable-source indexes."""

        wanted = frozenset(str(key).casefold() for key in keys)
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            source_cache = self._metadata_source_cache(source)
            cached = source_cache.lookups.get(wanted)
            if cached is not None:
                self._metadata_lookup_hits += 1
                if cached.found:
                    return cached.value
                continue

            self._metadata_lookup_misses += 1
            if source_cache.flat_values is not None:
                self._metadata_flat_resolutions += 1
                first_match: Optional[Tuple[int, object]] = None
                for key in wanted:
                    match = source_cache.flat_values.get(key)
                    if match is not None and (
                        first_match is None or match[0] < first_match[0]
                    ):
                        first_match = match
                if first_match is not None:
                    result = _MetadataLookup(True, first_match[1])
                    source_cache.lookups[wanted] = result
                    return result.value
                source_cache.lookups[wanted] = _MetadataLookup(False)
                continue

            pending = deque((source,))
            visited: Set[int] = set()
            while pending:
                current = pending.popleft()
                marker = id(current)
                if marker in visited:
                    continue
                visited.add(marker)
                for key, value in current.items():
                    if str(key).casefold() in wanted:
                        result = _MetadataLookup(True, value)
                        source_cache.lookups[wanted] = result
                        # This deliberately returns even when ``value`` is
                        # None, before touching later mappings or sources.
                        return result.value
                for key, value in current.items():
                    if (
                        str(key).casefold()
                        not in _NON_INHERITED_METADATA_SUBTREES
                        and isinstance(value, Mapping)
                    ):
                        pending.append(value)
            source_cache.lookups[wanted] = _MetadataLookup(False)
        return None

    @property
    def metadata_cache_stats(self) -> Mapping[str, int]:
        sources = self._metadata_sources_by_id or {}
        return {
            "hits": self._metadata_lookup_hits,
            "misses": self._metadata_lookup_misses,
            "sources": len(sources),
            "lookups": sum(len(source.lookups) for source in sources.values()),
            "flat_sources": sum(
                source.flat_values is not None for source in sources.values()
            ),
            "flat_resolutions": self._metadata_flat_resolutions,
        }

    def artifact_workload_metadata(
        self,
        key: Optional[Hashable],
        factory: Callable[[], Tuple[object, Dict[str, object]]],
    ) -> Tuple[object, Dict[str, object]]:
        """Memoize one exact physical artifact derivation in this compilation."""

        if key is None:
            self._artifact_workload_bypasses += 1
            return factory()
        try:
            hash(key)
        except TypeError:
            self._artifact_workload_bypasses += 1
            return factory()
        if self._artifact_workload_values is None:
            self._artifact_workload_values = OrderedDict()
        try:
            spec, derived_items = self._artifact_workload_values[key]
        except KeyError:
            self._artifact_workload_misses += 1
            # Exceptions deliberately escape before an entry is installed.
            spec, derived = factory()
            derived_items = tuple(derived.items())
            self._artifact_workload_values[key] = (spec, derived_items)
            self._artifact_workload_values.move_to_end(key)
            while (
                len(self._artifact_workload_values)
                > self.artifact_workload_cache_entries
            ):
                self._artifact_workload_values.popitem(last=False)
        else:
            self._artifact_workload_hits += 1
            self._artifact_workload_values.move_to_end(key)
        # The frozen spec is shareable; caller-owned metadata never is.
        return spec, dict(derived_items)

    @property
    def artifact_workload_cache_stats(self) -> Mapping[str, int]:
        return {
            "hits": self._artifact_workload_hits,
            "misses": self._artifact_workload_misses,
            "bypasses": self._artifact_workload_bypasses,
            "entries": len(self._artifact_workload_values or {}),
        }

    def leaf(
        self,
        key: Hashable,
        factory: Callable[[], object],
    ) -> object:
        kind = (
            str(key[0])
            if isinstance(key, tuple) and key
            else "other"
        )
        try:
            hash(key)
        except TypeError:
            self._leaf_bypasses += 1
            return factory()
        if self._leaf_values is None:
            self._leaf_values = OrderedDict()
        try:
            cached = self._leaf_values[key]
        except KeyError:
            self._leaf_misses += 1
            if self._leaf_misses_by_kind is None:
                self._leaf_misses_by_kind = {}
            self._leaf_misses_by_kind[kind] = (
                self._leaf_misses_by_kind.get(kind, 0) + 1
            )
            value = factory()
            self._leaf_values[key] = value
            self._leaf_values.move_to_end(key)
            if len(self._leaf_values) > self.leaf_cache_entries:
                self._leaf_values.popitem(last=False)
                self._leaf_evictions += 1
            return value
        self._leaf_hits += 1
        if self._leaf_hits_by_kind is None:
            self._leaf_hits_by_kind = {}
        self._leaf_hits_by_kind[kind] = (
            self._leaf_hits_by_kind.get(kind, 0) + 1
        )
        self._leaf_values.move_to_end(key)
        return cached

    @property
    def leaf_cache_stats(self) -> Mapping[str, int]:
        stats = {
            "hits": self._leaf_hits,
            "misses": self._leaf_misses,
            "bypasses": self._leaf_bypasses,
            "evictions": self._leaf_evictions,
            "size": len(self._leaf_values or ()),
            "max_entries": self.leaf_cache_entries,
        }
        for kind, count in (self._leaf_hits_by_kind or {}).items():
            stats["{}_hits".format(kind)] = count
        for kind, count in (self._leaf_misses_by_kind or {}).items():
            stats["{}_misses".format(kind)] = count
        return stats

    def execution_view(self) -> ModelGraphExecutionView:
        if self._execution_view_value is None:
            execution_view = model_graph_execution_view(
                self.scenario.model.graph,
                schema_version=self.scenario.model.schema_version,
            )
            execution_layers = tuple(
                descriptor.layer for descriptor in execution_view.layer_instances
            )
            self._execution_view_value = execution_view
            self._execution_layers_value = execution_layers
        return self._execution_view_value

    def execution_layers(self) -> Tuple[LayerSpec, ...]:
        if self._execution_layers_value is None:
            self.execution_view()
        if self._execution_layers_value is None:  # pragma: no cover - defensive
            raise AssertionError("execution layers were not initialized")
        return self._execution_layers_value

    def mtp_descriptor_for_weight(
        self, tensor_id: str
    ) -> Optional[MTPExecutionDescriptor]:
        if self._mtp_descriptors_by_weight is None:
            descriptors: Dict[str, MTPExecutionDescriptor] = {}
            for descriptor in self.execution_view().mtp_descriptors:
                # Preserve the historical ``next(...)`` first-match behavior.
                descriptors.setdefault(
                    str(descriptor.weight_tensor.tensor_id), descriptor
                )
            self._mtp_descriptors_by_weight = descriptors
        return self._mtp_descriptors_by_weight.get(str(tensor_id))

    def control_plane_decision(self) -> Mapping[object, object]:
        if self._control_plane_decision_value is None:
            self._control_plane_decision_value = control_plane_decision(
                self.scenario
            )
        return self._control_plane_decision_value

    def canonical_weight_tensor_id(self, tensor_id: str) -> str:
        requested = str(tensor_id)
        if self._canonical_weight_ids is None:
            self._canonical_weight_ids = {}
        cached = self._canonical_weight_ids.get(requested)
        if cached is not None:
            return cached
        aliases = self.control_plane_decision().get(
            "logical_weight_aliases", {}
        )
        if not isinstance(aliases, Mapping):
            self._canonical_weight_ids[requested] = requested
            return requested
        current = requested
        visited = set()
        while current in aliases and current not in visited:
            visited.add(current)
            target = aliases[current]
            if target is None:
                break
            current = str(target)
        self._canonical_weight_ids[requested] = current
        return current

    def _ensure_rank_shard_indexes(self) -> None:
        if self._rank_shards_by_tensor is not None:
            return
        decision = self.control_plane_decision()
        top_level_raw = decision.get("rank_weight_shards", {})
        top_level = top_level_raw if isinstance(top_level_raw, Mapping) else {}
        by_tensor: Dict[str, Tuple[Mapping[object, object], ...]] = {}
        by_rank_target: Dict[
            Tuple[str, str, str], Mapping[object, object]
        ] = {}
        by_rank: Dict[Tuple[str, str], Mapping[object, object]] = {}
        by_target: Dict[Tuple[str, str], Mapping[object, object]] = {}
        for tensor_id in {str(name) for name in top_level}:
            shards_raw = top_level.get(tensor_id, ())
            shards = (
                tuple(
                    shard
                    for shard in shards_raw
                    if isinstance(shard, Mapping)
                )
                if isinstance(shards_raw, (list, tuple))
                else ()
            )
            by_tensor[tensor_id] = shards
            for shard in shards:
                rank_id = str(shard.get("rank_id", shard.get("rank", "")))
                target = str(shard.get("compute_component_id", ""))
                by_rank.setdefault((tensor_id, rank_id), shard)
                by_target.setdefault((tensor_id, target), shard)
                by_rank_target.setdefault((tensor_id, rank_id, target), shard)
        self._rank_shards_by_tensor = by_tensor
        self._rank_shards_by_rank_target = by_rank_target
        self._rank_shards_by_rank = by_rank
        self._rank_shards_by_target = by_target

    def rank_shards(
        self, tensor_id: str
    ) -> Tuple[Mapping[object, object], ...]:
        self._ensure_rank_shard_indexes()
        if self._rank_shards_by_tensor is None:  # pragma: no cover - defensive
            return ()
        return self._rank_shards_by_tensor.get(str(tensor_id), ())

    def select_rank_shard(
        self,
        tensor_id: str,
        *,
        rank_id: Optional[int],
        target_component_id: Optional[str],
    ) -> Optional[Mapping[object, object]]:
        self._ensure_rank_shard_indexes()
        tensor_key = str(tensor_id)
        target = str(target_component_id) if target_component_id else ""
        selected: Optional[Mapping[object, object]] = None
        if rank_id is not None:
            rank_key = str(rank_id)
            if target:
                if self._rank_shards_by_rank_target is not None:
                    selected = self._rank_shards_by_rank_target.get(
                        (tensor_key, rank_key, target)
                    )
            elif self._rank_shards_by_rank is not None:
                selected = self._rank_shards_by_rank.get((tensor_key, rank_key))
        if (
            selected is None
            and target
            and self._rank_shards_by_target is not None
        ):
            selected = self._rank_shards_by_target.get((tensor_key, target))
        return selected

    def parallel_plan(self) -> ParallelPlan:
        if self._parallel_plan_value is None:
            self._parallel_plan_value = build_parallel_plan(
                self.scenario,
                execution_view=self.execution_view(),
            )
        return self._parallel_plan_value

    def router(self) -> TopologyRouter:
        if self._router_value is None:
            self._router_value = TopologyRouter(
                self.scenario.hardware,
                coherent_dma_mode=_coherent_dma_mode(self.scenario),
            )
        return self._router_value


_COMPILATION_CONTEXT: ContextVar[Optional[CompilationContext]] = ContextVar(
    "planner_compilation_context", default=None
)


@contextmanager
def _compilation_scope(
    scenario: ScenarioConfig,
    context: Optional[CompilationContext] = None,
) -> Iterator[CompilationContext]:
    """Activate a scenario context, honoring an explicitly selected owner."""

    current = _COMPILATION_CONTEXT.get()
    if context is not None and context.scenario is not scenario:
        raise ValueError("compilation context is bound to another scenario")
    if (
        current is not None
        and current.scenario is scenario
        and (context is None or context is current)
    ):
        yield current
        return
    selected = context or CompilationContext(scenario)
    token = _COMPILATION_CONTEXT.set(selected)
    try:
        yield selected
    finally:
        _COMPILATION_CONTEXT.reset(token)


def _active_compilation_context(
    scenario: ScenarioConfig,
) -> Optional[CompilationContext]:
    context = _COMPILATION_CONTEXT.get()
    return context if context is not None and context.scenario is scenario else None


def _component_map(scenario: ScenarioConfig) -> Mapping[str, ComponentSpec]:
    context = _active_compilation_context(scenario)
    if context is not None:
        return context.component_map()
    return scenario.hardware.component_map()


def _component(scenario: ScenarioConfig, component_id: str) -> ComponentSpec:
    context = _active_compilation_context(scenario)
    if context is not None:
        return context.component(component_id)
    return scenario.hardware.get_component(component_id)


def _resolve_component_profile(
    scenario: ScenarioConfig,
    component: object,
    expected_type: Optional[type] = None,
) -> object:
    context = _active_compilation_context(scenario)
    if context is None:
        return scenario.resolve_component_profile(component, expected_type)
    if isinstance(component, str):
        component_id = component
    elif isinstance(component, ComponentSpec):
        canonical = context.component_map().get(component.component_id)
        if canonical is not component:
            return scenario.resolve_component_profile(component, expected_type)
        component_id = component.component_id
    else:
        return scenario.resolve_component_profile(component, expected_type)
    return context.invariant(
        ("component_profile", component_id, expected_type),
        lambda: scenario.resolve_component_profile(component, expected_type),
    )


def _memoized_cost_estimate(
    scenario: ScenarioConfig,
    key: Tuple[object, ...],
    factory: Callable[[], CostEstimate],
) -> CostEstimate:
    context = _active_compilation_context(scenario)
    if context is None:
        return factory()
    value = context.leaf(("cost",) + key, factory)
    if not isinstance(value, CostEstimate):  # pragma: no cover - defensive
        raise AssertionError("cost memo returned a non-CostEstimate value")
    return value


_TRANSFER_TEMPLATE_NAME = "__planner_exact_transfer__"


def _transfer_phases(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    source_component: str,
    target_component: str,
    byte_count: int,
    *,
    policy: str,
    name: str,
) -> Tuple[object, ...]:
    context = _active_compilation_context(scenario)
    if context is None or router is not context.router():
        return router.transfer_phases(
            source_component,
            target_component,
            byte_count,
            policy=policy,
            name=name,
        )
    template = context.leaf(
        (
            "transfer",
            source_component,
            target_component,
            byte_count,
            policy,
        ),
        lambda: router.transfer_phases(
            source_component,
            target_component,
            byte_count,
            policy=policy,
            name=_TRANSFER_TEMPLATE_NAME,
        ),
    )
    phases = tuple(template)
    if any(
        not str(phase.name).startswith(_TRANSFER_TEMPLATE_NAME)
        for phase in phases
    ):  # pragma: no cover - defensive fail-closed fallback
        return router.transfer_phases(
            source_component,
            target_component,
            byte_count,
            policy=policy,
            name=name,
        )
    return tuple(
        replace(
            phase,
            name=name + phase.name[len(_TRANSFER_TEMPLATE_NAME) :],
        )
        for phase in phases
    )


def _collective_plan(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    kind: str,
    participants: Sequence[str],
    tensor_bytes: int,
    *,
    algorithm: str,
    routing_policy: str,
) -> object:
    ordered = tuple(participants)
    context = _active_compilation_context(scenario)
    factory = lambda: plan_collective(
        router,
        kind,
        ordered,
        tensor_bytes,
        algorithm=algorithm,
        routing_policy=routing_policy,
    )
    if context is None or router is not context.router():
        return factory()
    return context.leaf(
        (
            "collective",
            kind,
            ordered,
            tensor_bytes,
            algorithm,
            routing_policy,
        ),
        factory,
    )


def _parallel_plan(scenario: ScenarioConfig) -> ParallelPlan:
    context = _active_compilation_context(scenario)
    return context.parallel_plan() if context is not None else build_parallel_plan(scenario)


def _topology_router(scenario: ScenarioConfig) -> TopologyRouter:
    context = _active_compilation_context(scenario)
    return context.router() if context is not None else TopologyRouter(
        scenario.hardware, coherent_dma_mode=_coherent_dma_mode(scenario)
    )


def _coherent_dma_mode(scenario: ScenarioConfig) -> str:
    """Resolve the explicit DMA span mode from scenario metadata."""

    metadata = scenario.placement.metadata
    value = metadata.get("coherent_dma_mode")
    if value is None:
        value = scenario.hardware.metadata.get("coherent_dma_mode")
    return str(value or "pipelined")


@dataclass(frozen=True)
class ScenarioValidationReport:
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    information: Tuple[str, ...] = ()
    input_fingerprint: Optional[str] = None
    current_input_fingerprint: Optional[str] = None
    mapping_stale: bool = False
    errors_en: Tuple[str, ...] = ()
    warnings_en: Tuple[str, ...] = ()
    information_en: Tuple[str, ...] = ()
    diagnostics: Tuple[Mapping[str, object], ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ScenarioValidationError(
                "场景校验失败：\n- " + "\n- ".join(self.errors),
                self.diagnostics,
            )


_PLANNER_EXACT_MESSAGES_ZH = {
    "placement.model_name does not match model.name": "placement.model_name 与 model.name 不一致",
    "placement.hardware_name does not match hardware.name": "placement.hardware_name 与 hardware.name 不一致",
    "reference planner requires at least one GPU component": "参考规划器至少需要一个 GPU 组件",
    "V4 full-attention workloads require a KV cache component": "V4 完整注意力工作负载需要配置 KV 缓存组件",
    "kv_policy.cache_component conflicts with tensor_to_component[kv_cache]": "kv_policy.cache_component 与 tensor_to_component[kv_cache] 冲突",
    "linear state must target writable active memory": "线性注意力状态必须放在可写的活动内存中",
    "linear state cache component is read-only": "线性注意力状态缓存组件为只读",
    "linear state offload must target active memory or offload storage": "线性注意力状态卸载必须使用活动内存或卸载存储",
    "linear state offload component is read-only": "线性注意力状态卸载组件为只读",
    "linear state cache and offload components must be distinct": "线性注意力状态缓存与卸载组件不能相同",
    "CIM placement requires both cim_profile and cim_interconnect": "CIM 放置同时需要 cim_profile 和 cim_interconnect",
    "KV cache and offload components must be distinct": "KV 缓存组件与卸载组件不能相同",
    "KV swap preemption requires an offload component": "KV 交换式抢占需要配置卸载组件",
    "KV swap preemption requires a positive offload_ratio": "KV 交换式抢占要求 offload_ratio 大于零",
    "swap preemption for a linear mixer requires tensor_to_component[linear_state_offload]": "线性混合器使用交换式抢占时必须配置 tensor_to_component[linear_state_offload]",
    "linear_state_offload_ratio must be in [0, 1]": "linear_state_offload_ratio 必须在 [0, 1] 范围内",
    "linear_state_offload_ratio must be numeric": "linear_state_offload_ratio 必须是数值",
    "workload must contain at least one request": "工作负载至少需要包含一个请求",
    "CIM results use warm weight-resident semantics": "CIM 结果采用权重常驻的热启动语义",
    "CIM results include per-operation cold weight loads": "CIM 结果包含每次算子的冷权重加载开销",
    "all operators, collectives, KV traffic, and host orchestration use the task/transaction-level analytical lowering": "所有算子、集合通信、KV 流量和主机编排均使用任务/事务级解析降低模型",
    "request scheduling uses deterministic continuous batching": "请求调度采用确定性的连续批处理",
    "request scheduling uses deterministic static task ordering": "请求调度采用确定性的静态任务顺序",
}


def _format_iec_bytes(value: str) -> str:
    """Format an integer byte count for user-visible Chinese diagnostics."""

    byte_count = int(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(byte_count)
    unit_index = 0
    while amount >= 1024.0 and unit_index < len(units) - 1:
        amount /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return "{} B".format(byte_count)
    return "{} {}".format(format(amount, ".4g"), units[unit_index])


def _planner_message_zh(
    message: str,
    *,
    warning: bool = False,
    information: bool = False,
) -> str:
    """Translate planner diagnostics while retaining their English originals."""

    mixed_layer = re.fullmatch(r"layer (.+): (.+)", message, flags=re.DOTALL)
    if mixed_layer is not None and re.search(r"[\u3400-\u9fff]", mixed_layer.group(2)):
        return "层 {}：{}".format(*mixed_layer.groups())
    if re.search(r"[\u3400-\u9fff]", message):
        return message
    exact = _PLANNER_EXACT_MESSAGES_ZH.get(message)
    if exact is not None:
        return exact

    non_gemm_cim = re.fullmatch(
        r"non-GEMM operator (.+) \((.+)\) requests CIM target (.+); "
        r"compatible target suggestion: (.+); resolution_applied=false",
        message,
    )
    if non_gemm_cim is not None:
        operator_id, operator_class, requested, resolved = non_gemm_cim.groups()
        return (
            "非 GEMM 算子 {}（{}）请求了 CIM 目标 {}；兼容目标建议为 {}；"
            "resolution_applied=false，未自动改写"
        ).format(operator_id, operator_class, requested, resolved)

    patterns = (
        (
            r"control-plane decision is stale: stored input fingerprint (.+) does not match current input fingerprint (.+); rerun control-plane placement before simulation",
            lambda match: "控制平面决策已过期：保存的输入指纹 {} 与当前输入指纹 {} 不一致；请重新执行控制平面放置后再仿真".format(match.group(1), match.group(2)),
        ),
        (
            r"op placement (.+) references unknown component (.+)",
            lambda match: "算子放置 {} 引用了未知组件 {}".format(*match.groups()),
        ),
        (
            r"op placement (.+) targets unsupported component kind (.+)",
            lambda match: "算子放置 {} 指向了不支持的组件类型 {}".format(*match.groups()),
        ),
        (
            r"op placement (.+) targets CPU without CPU/host-memory profiles",
            lambda match: "算子放置 {} 指向 CPU，但缺少执行成本配置；请同时配置并绑定 profiles.components.cpu 与 profiles.components.host_memory".format(
                match.group(1)
            ),
        ),
        (
            r"tensor placement (.+) references unknown component (.+)",
            lambda match: "张量放置 {} 引用了未知组件 {}".format(*match.groups()),
        ),
        (
            r"KV cache references unknown component (.+)",
            lambda match: "KV 缓存引用了未知组件 {}".format(match.group(1)),
        ),
        (
            r"KV cache must target writable active memory; (.+) is (.+) storage",
            lambda match: "KV 缓存必须放在可写的活动内存中；{} 属于 {} 存储".format(*match.groups()),
        ),
        (
            r"KV cache component (.+) is declared read-only",
            lambda match: "KV 缓存组件 {} 被声明为只读".format(match.group(1)),
        ),
        (
            r"full-attention rank (\d+) has no writable KV cache placement; configure kv_policy\.cache_component, tensor_to_component\[kv_cache\], or rank memory_component_id",
            lambda match: "完整注意力 Rank {} 没有可写的 KV 缓存放置；请配置 kv_policy.cache_component、tensor_to_component[kv_cache] 或 Rank memory_component_id".format(match.group(1)),
        ),
        (
            r"KV offload references unknown component (.+)",
            lambda match: "KV 卸载引用了未知组件 {}".format(match.group(1)),
        ),
        (
            r"KV offload must target active memory or offload storage, not (.+)",
            lambda match: "KV 卸载必须使用活动内存或卸载存储，不能使用 {}".format(match.group(1)),
        ),
        (
            r"KV offload component (.+) is declared read-only",
            lambda match: "KV 卸载组件 {} 被声明为只读".format(match.group(1)),
        ),
        (
            r"linear state references unknown component (.+)",
            lambda match: "线性注意力状态引用了未知组件 {}".format(match.group(1)),
        ),
        (
            r"linear state offload references unknown component (.+)",
            lambda match: "线性注意力状态卸载引用了未知组件 {}".format(match.group(1)),
        ),
        (
            r"tensor_bytes\[linear_state\] declares (\d+) bytes but one request requires (\d+) bytes",
            lambda match: "tensor_bytes[linear_state] 声明了 {}，但单个请求需要 {}".format(
                _format_iec_bytes(match.group(1)),
                _format_iec_bytes(match.group(2)),
            ),
        ),
        (
            r"tensor_bytes\[model_weights\] declares (\d+) bytes but the model declares (\d+) weight bytes",
            lambda match: "tensor_bytes[model_weights] 声明了 {}，但模型声明了 {} 的权重".format(
                _format_iec_bytes(match.group(1)),
                _format_iec_bytes(match.group(2)),
            ),
        ),
        (
            r"control_plane\.decision total-physical-byte metadata disagrees for (.+)",
            lambda match: "control_plane.decision 中张量 {} 的总物理字节元数据不一致".format(match.group(1)),
        ),
        (
            r"CIM tensor (.+) total physical bytes (\d+) do not match (\d+) padded bytes across (\d+) replicas",
            lambda match: "CIM 张量 {} 的总物理字节数（{}）与 {} 个副本、每个 {} 的填充结果不一致".format(
                match.group(1),
                _format_iec_bytes(match.group(2)),
                match.group(4),
                _format_iec_bytes(match.group(3)),
            ),
        ),
        (
            r"cold/streamed weights require tensor_to_component\[model_weights\] and tensor_bytes\[model_weights\] backing storage, or mapped detailed weight tensors",
            lambda match: "冷权重或流式权重需要配置 tensor_to_component[model_weights] 与 tensor_bytes[model_weights] 后备存储，或提供已映射的详细权重张量",
        ),
        (
            r"remaining resident model weights require (\d+) bytes across physical TP/PP/EP placements, but (.+) provide (\d+) bytes",
            lambda match: "其余常驻模型权重在物理 TP/PP/EP 放置上需要 {}，但 {} 仅能提供 {}".format(
                _format_iec_bytes(match.group(1)),
                match.group(2),
                _format_iec_bytes(match.group(3)),
            ),
        ),
        (
            r"resident aggregate model_weights component (.+) must be writable active memory, not read-only (.+)",
            lambda match: "常驻 aggregate model_weights 组件 {} 必须是可写活动内存，不能是只读的 {}".format(
                *match.groups()
            ),
        ),
        (
            r"weight tensor (.+) route to (.+): (.+)",
            lambda match: "权重张量 {} 到 {} 的路由失败；请检查拓扑连通性".format(match.group(1), match.group(2)),
        ),
        (
            r"layer (.+): unsupported layer dtype (.+); add a model adapter or explicit quantization",
            lambda match: "层 {} 使用了不支持的层数据类型 {}；请添加模型适配器或明确配置量化".format(*match.groups()),
        ),
        (
            r"parallel world requires (\d+) compute components, found (\d+); provide an explicit rank_mapping to model colocated logical ranks",
            lambda match: "并行域需要 {} 个计算组件，但只找到 {} 个；如需逻辑 rank 共置，请显式配置 rank_mapping".format(*match.groups()),
        ),
        (
            r"rank_mapping must contain exactly (\d+) ranks",
            lambda match: "rank_mapping 必须恰好包含 {} 个 rank".format(match.group(1)),
        ),
        (
            r"rank_mapping rank values must be unique",
            lambda match: "rank_mapping 中的 rank 值必须唯一",
        ),
        (
            r"rank_mapping must cover every TP/PP/EP coordinate exactly once",
            lambda match: "rank_mapping 必须且只能覆盖每个 TP/PP/EP 坐标一次",
        ),
        (
            r"rank (\d+) references unknown component (.+)",
            lambda match: "rank {} 引用了未知组件 {}".format(*match.groups()),
        ),
        (
            r"rank (\d+) must map to a GPU component for the reference cost provider, not (.+)",
            lambda match: "参考成本模型要求 rank {} 映射到 GPU 组件，不能映射到 {}".format(*match.groups()),
        ),
        (
            r"rank (\d+) references unknown (memory|CIM) component (.+)",
            lambda match: "rank {} 引用了未知的 {} 组件 {}".format(*match.groups()),
        ),
        (
            r"rank (\d+) memory component (.+) must be writable active memory, not (.+)",
            lambda match: "rank {} 的内存组件 {} 必须是可写活动内存，不能是 {}".format(*match.groups()),
        ),
        (
            r"rank (\d+) CIM component (.+) has non-CIM kind (.+)",
            lambda match: "rank {} 的 CIM 组件 {} 实际类型为非 CIM 的 {}".format(*match.groups()),
        ),
        (
            r"request (.+) must have at least one prompt token",
            lambda match: "请求 {} 至少需要一个提示词元".format(match.group(1)),
        ),
        (
            r"request (.+) uses unsupported modalities: (.+)",
            lambda match: "请求 {} 使用了不支持的模态：{}".format(*match.groups()),
        ),
        (
            r"request (.+) has no executable IR for modalities: (.+); the analytical core currently lowers only the ordered text backbone",
            lambda match: "请求 {} 的以下模态没有可执行 IR：{}；当前分析核心仅支持有序文本主干".format(*match.groups()),
        ),
        (
            r"request (.+) KV working set requires (\d+) bytes \((\d+) pages for (\d+) tokens\), exceeding cache capacity (\d+) bytes \((\d+) pages\)",
            lambda match: "请求 {} 的 KV 工作集需要 {}（{} 页、{} 个词元），超过缓存容量 {}（{} 页）".format(
                match.group(1),
                _format_iec_bytes(match.group(2)),
                match.group(3),
                match.group(4),
                _format_iec_bytes(match.group(5)),
                match.group(6),
            ),
        ),
        (
            r"request (.+) sequence has (\d+) tokens, exceeding model max_sequence_length (\d+)",
            lambda match: "请求 {} 的完整序列包含 {} 个词元，超过模型 max_sequence_length {}".format(
                *match.groups()
            ),
        ),
        (
            r"V4 placement does not support non-zero prefetch_distance; proactive KV lookahead is not modeled",
            lambda match: "V4 放置不支持非零 prefetch_distance；当前未建模主动 KV 前瞻",
        ),
        (
            r"V4 placement requires positive capacity_bytes for storage component (.+) before placing (\d+) bytes of tensor (.+)",
            lambda match: "V4 放置要求存储组件 {} 在放置 {} 字节张量 {} 前声明正 capacity_bytes".format(
                match.group(1), _format_iec_bytes(match.group(2)), match.group(3)
            ),
        ),
        (
            r"V4 placement requires positive capacity_bytes for active memory component (.+) before placing (\d+) tensor bytes",
            lambda match: "V4 放置要求活动内存组件 {} 在放置 {} 字节张量前声明正 capacity_bytes".format(
                match.group(1), _format_iec_bytes(match.group(2))
            ),
        ),
        (
            r"V4 placement requires positive capacity_bytes for resident model-weight components; capacity is unknown for (.+)",
            lambda match: "V4 放置要求常驻模型权重组件声明正 capacity_bytes；以下组件容量未知：{}".format(
                match.group(1)
            ),
        ),
        (
            r"detailed weight tensors explicitly cover (\d+) of (\d+) declared bytes; remaining resident capacity will be checked across physical TP/PP/EP placements",
            lambda match: "详细权重张量已显式覆盖声明的 {} 中的 {}；其余常驻容量将按物理 TP/PP/EP 放置校验".format(
                _format_iec_bytes(match.group(2)),
                _format_iec_bytes(match.group(1)),
            ),
        ),
        (
            r"manual CIM weight tensor (.+) lacks complete padded/replica metadata; physical resident capacity may be underestimated",
            lambda match: "手动配置的 CIM 权重张量 {} 缺少完整的填充/副本元数据；物理常驻容量可能被低估".format(match.group(1)),
        ),
        (
            r"resident model weights are capacity-checked as physical TP/PP/EP shards across (\d+) component\(s\); (\d+) bytes are explicit detailed placements",
            lambda match: "常驻模型权重按 {} 个组件上的物理 TP/PP/EP 分片校验容量；其中 {} 为显式详细放置".format(
                match.group(1),
                _format_iec_bytes(match.group(2)),
            ),
        ),
        (
            r"explicit workload\.requests take precedence over conflicting top-level synthetic fields: (.+)",
            lambda match: "显式 workload.requests 优先于发生冲突的顶层合成字段：{}".format(match.group(1)),
        ),
        (
            r"warm CIM resident capacity is checked from (.+) across (\d+) physical component\(s\)",
            lambda match: "CIM 热常驻容量根据 {} 在 {} 个物理组件上进行校验".format(
                "显式详细权重张量" if match.group(1) == "explicit detailed weight tensors" else "映射层权重估算",
                match.group(2),
            ),
        ),
        (
            r"component (.+) has capacity_bytes=0; unspecified capacity is treated as unbounded by the control plane",
            lambda match: "组件 {} 的 capacity_bytes=0；控制平面会将未指定容量视为无上限".format(match.group(1)),
        ),
        (
            r"offload storage component (.+) has capacity_bytes=0; declare a positive capacity before placing (\d+) tensor bytes",
            lambda match: "卸载存储组件 {} 的 capacity_bytes=0（容量未知）；请先声明可容纳 {} 字节张量的正容量".format(
                match.group(1), _format_iec_bytes(match.group(2))
            ),
        ),
        (
            r"op placement key (.+) is not consumed by the reference lowering",
            lambda match: "算子放置键 {} 不会被参考降低流程使用".format(match.group(1)),
        ),
        (
            r"OR-Tools adapter failed.*",
            lambda match: "OR-Tools 适配器运行失败，已改用内置求解器",
        ),
    )
    for pattern, render in patterns:
        matched = re.fullmatch(pattern, message, flags=re.DOTALL)
        if matched is not None:
            return render(matched)

    if message.startswith("serving admission preflight failed: "):
        detail = message[len("serving admission preflight failed: ") :]
        return "服务准入预检失败：{}".format(_planner_message_zh(detail))
    if message.startswith("topology validation failed:"):
        return "拓扑校验失败；具体问题请查看拓扑错误列表"
    if message.startswith("parallel plan:"):
        nested = message[len("parallel plan:") :].strip()
        translated = _planner_message_zh(
            nested,
            warning=warning,
            information=information,
        )
        if translated not in {"场景校验未通过；请查看 message_en 获取技术细节", "场景校验警告；请查看 message_en 获取技术细节"}:
            return "并行计划无效：{}".format(translated)
        return "并行计划无效；请检查并行度、rank_mapping 和层级划分"
    if message.startswith("serving admission preflight failed:"):
        return "服务准入预检查失败；请检查请求规模和缓存容量"
    if message.startswith("serving admission will reject this request:"):
        return "服务准入预计会拒绝该请求；请检查请求规模和缓存容量"
    if information:
        return "场景校验信息；请查看 message_en 获取技术细节"
    if warning:
        return "场景校验警告；请查看 message_en 获取技术细节"
    return "场景校验未通过；请查看 message_en 获取技术细节"


_MANIFEST_ASSUMPTION_EN_TO_ZH = {
    "Reference results are ANALYTICAL and are not calibrated to a named commercial chip.": "参考结果为分析型结果，未针对任何命名商业芯片进行校准。",
    "MoE routing is represented by a uniform aggregate active-expert model.": "MoE 路由采用均匀聚合活跃专家模型表示。",
    "CIM is ADC-free digital SRAM-CIM with warm resident weights.": "CIM 为无 ADC 的数字 SRAM-CIM，权重采用热常驻。",
    "The workload is inference-only; no training graph or optimizer state is represented.": "工作负载仅包含推理；未表示训练图或优化器状态。",
    "Continuous batching admits, chunks, and preempts only at scheduler boundaries.": "连续批处理仅在调度器边界进行准入、分块与抢占。",
    "online cohort lowered with ragged-context weighted mean": "在线 cohort 使用不规则上下文加权均值降低。",
    "swap I/O follows the declared physical topology": "交换 I/O 遵循声明的物理拓扑。",
}
_MANIFEST_ASSUMPTION_ZH_TO_EN = {
    zh: en for en, zh in _MANIFEST_ASSUMPTION_EN_TO_ZH.items()
}
_GENERIC_PLANNER_MESSAGES_ZH = {
    "场景校验未通过；请查看 message_en 获取技术细节",
    "场景校验警告；请查看 message_en 获取技术细节",
}


def localized_manifest_assumptions(
    assumptions: Sequence[str],
    *,
    assumptions_zh: Optional[Sequence[str]] = None,
    assumptions_en: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return stable Chinese/English projections for manifest assumptions.

    ``RunManifest.assumptions`` is the canonical source
    array.  Callers that already have the English planner diagnostics can pass
    them through ``assumptions_en`` so warning translations remain exact.  For
    ordinary assumptions only the small, stable built-in vocabulary is
    translated.  Unrecognised/user-authored text is copied verbatim to both
    projections to preserve provenance.
    """

    source = tuple(str(item) for item in assumptions)
    provided_zh = (
        tuple(str(item) for item in assumptions_zh)
        if assumptions_zh is not None
        else ()
    )
    provided_en = (
        tuple(str(item) for item in assumptions_en)
        if assumptions_en is not None
        else ()
    )
    zh_values: List[str] = []
    en_values: List[str] = []
    for index, raw in enumerate(source):
        explicit_zh = provided_zh[index] if index < len(provided_zh) else None
        explicit_en = provided_en[index] if index < len(provided_en) else None
        if explicit_zh is not None:
            zh = explicit_zh
        elif re.search(r"[\u3400-\u9fff]", raw):
            zh = raw
        elif raw in _MANIFEST_ASSUMPTION_EN_TO_ZH:
            zh = _MANIFEST_ASSUMPTION_EN_TO_ZH[raw]
        elif explicit_en is not None:
            translated = _planner_message_zh(explicit_en, warning=True)
            zh = translated if translated not in _GENERIC_PLANNER_MESSAGES_ZH else raw
        else:
            zh = raw

        if explicit_en is not None:
            en = explicit_en
        elif raw in _MANIFEST_ASSUMPTION_ZH_TO_EN:
            en = _MANIFEST_ASSUMPTION_ZH_TO_EN[raw]
        elif raw in _MANIFEST_ASSUMPTION_EN_TO_ZH:
            en = raw
        else:
            en = raw
        zh_values.append(zh)
        en_values.append(en)
    return tuple(zh_values), tuple(en_values)


def _execution_view(scenario: ScenarioConfig):
    """Return the fail-closed graph-native execution view."""

    context = _active_compilation_context(scenario)
    if context is None:
        context = CompilationContext(scenario)
    return context.execution_view()


def _execution_layers(scenario: ScenarioConfig) -> Tuple[LayerSpec, ...]:
    """Cost geometry derived from the authoritative typed graph."""

    context = _active_compilation_context(scenario)
    if context is None:
        context = CompilationContext(scenario)
    return context.execution_layers()


def _mtp_execution_descriptors(
    scenario: ScenarioConfig,
) -> Tuple[MTPExecutionDescriptor, ...]:
    return _execution_view(scenario).mtp_descriptors


def _execution_view_declared_weight_bytes(
    execution_view: ModelGraphExecutionView,
) -> int:
    return (
        int(execution_view.embedding_weight_bytes)
        + int(execution_view.output_weight_bytes)
        + sum(
            int(descriptor.layer.weight_bytes)
            for descriptor in execution_view.layer_instances
        )
        + sum(
            int(descriptor.weight_bytes)
            for descriptor in execution_view.mtp_descriptors
        )
    )


def _reference_lowering_op_mapping_keys(
    scenario: ScenarioConfig,
    *,
    execution_view: Optional[ModelGraphExecutionView] = None,
) -> Set[str]:
    """Return every placement key consumed by the reference lowering.

    Keep validation aligned with the exact typed primitive keys queried by the
    lowering. Control-plane placement emits those keys directly.
    """

    if execution_view is None:
        execution_view = _execution_view(scenario)
    layers = tuple(
        descriptor.layer for descriptor in execution_view.layer_instances
    )
    supported: Set[str] = set()
    if execution_view.vocabulary_size > 0:
        supported.update({"embedding", "lm_head"})
    aggregate_groups = {
        "attention": any(not layer.is_linear_attention for layer in layers),
        "linear_attention": any(layer.is_linear_attention for layer in layers),
        "mlp": any(not layer.is_moe for layer in layers),
        "experts": any(layer.is_moe for layer in layers),
        "shared_expert": any(layer.has_shared_expert for layer in layers),
    }
    for group, present in aggregate_groups.items():
        if present:
            supported.update({group, "*.{}".format(group)})
    mtp_descriptors = execution_view.mtp_descriptors
    for layer in layers:
        layer_id = layer.layer_id
        supported.update(
            {
                layer_id,
                "{}.norm".format(layer_id),
                "{}.input_norm.reduce".format(layer_id),
                "{}.input_norm.apply".format(layer_id),
                "{}.post_attention_norm.reduce".format(layer_id),
                "{}.post_attention_norm.apply".format(layer_id),
            }
        )
        if layer.is_linear_attention:
            supported.update(
                {
                    "{}.linear_attention".format(layer_id),
                    "{}.linear_state_update".format(layer_id),
                    "{}.linear_attention.local_conv".format(layer_id),
                    "{}.linear_attention.state_update".format(layer_id),
                    "{}.linear_attention.gate_norm.reduce".format(layer_id),
                    "{}.linear_attention.gate_norm.apply".format(layer_id),
                    "{}.linear_attention.residual".format(layer_id),
                }
            )
        else:
            supported.update(
                {
                    "{}.attention".format(layer_id),
                    "{}.softmax".format(layer_id),
                    "{}.attention.rope".format(layer_id),
                    "{}.attention.qk".format(layer_id),
                    "{}.attention.pv".format(layer_id),
                    "{}.attention.softmax.reduce".format(layer_id),
                    "{}.attention.softmax.normalize".format(layer_id),
                    "{}.attention.residual".format(layer_id),
                }
            )
        if layer.is_moe:
            supported.update(
                {
                    "{}.router".format(layer_id),
                    "{}.router.softmax.reduce".format(layer_id),
                    "{}.router.softmax.normalize".format(layer_id),
                    "{}.router.topk".format(layer_id),
                    "{}.experts".format(layer_id),
                    "{}.experts.activation".format(layer_id),
                    "{}.moe.residual".format(layer_id),
                }
            )
            if layer.has_shared_expert:
                supported.update(
                    {
                        "{}.shared_expert".format(layer_id),
                        "{}.shared_expert.activation".format(layer_id),
                    }
                )
                if layer.shared_expert_gate:
                    supported.update(
                        {
                            "{}.shared_expert_gate".format(layer_id),
                            "{}.shared_expert.gate_apply".format(layer_id),
                        }
                    )
        else:
            supported.update(
                {
                    "{}.mlp".format(layer_id),
                    "{}.mlp.activation".format(layer_id),
                    "{}.mlp.residual".format(layer_id),
                }
            )
    supported.update(
        descriptor.operator.operator_id
        for descriptor in mtp_descriptors
    )
    return supported


def _typed_primitive_mapping_uses(
    scenario: ScenarioConfig,
) -> Tuple[Tuple[str, str, OperatorClass, Tuple[str, ...]], ...]:
    """Return non-GEMM primitive keys and their legacy fallback keys."""

    uses: List[Tuple[str, str, OperatorClass, Tuple[str, ...]]] = []

    def add(
        operator_id: str,
        operator_class: OperatorClass,
        *fallback_keys: str,
    ) -> None:
        uses.append(
            (layer_id, operator_id, operator_class, tuple(fallback_keys))
        )

    for layer in _execution_layers(scenario):
        layer_id = layer.layer_id
        norm_key = "{}.norm".format(layer_id)
        add(
            "{}.input_norm.reduce".format(layer_id),
            OperatorClass.REDUCTION,
            norm_key,
        )
        add(
            "{}.input_norm.apply".format(layer_id),
            OperatorClass.ELEMENTWISE,
            norm_key,
        )
        add(
            "{}.post_attention_norm.reduce".format(layer_id),
            OperatorClass.REDUCTION,
            norm_key,
        )
        add(
            "{}.post_attention_norm.apply".format(layer_id),
            OperatorClass.ELEMENTWISE,
            norm_key,
        )
        if layer.is_linear_attention:
            add(
                "{}.linear_attention.local_conv".format(layer_id),
                OperatorClass.REDUCTION,
                "{}.linear_attention".format(layer_id),
            )
            add(
                "{}.linear_attention.state_update".format(layer_id),
                OperatorClass.ELEMENTWISE,
                "{}.linear_state_update".format(layer_id),
            )
            add(
                "{}.linear_attention.gate_norm.reduce".format(layer_id),
                OperatorClass.REDUCTION,
                norm_key,
            )
            add(
                "{}.linear_attention.gate_norm.apply".format(layer_id),
                OperatorClass.ELEMENTWISE,
                norm_key,
            )
            add(
                "{}.linear_attention.residual".format(layer_id),
                OperatorClass.ELEMENTWISE,
                norm_key,
            )
        else:
            add(
                "{}.attention.rope".format(layer_id),
                OperatorClass.ELEMENTWISE,
                "{}.attention".format(layer_id),
            )
            add(
                "{}.attention.softmax.reduce".format(layer_id),
                OperatorClass.REDUCTION,
                "{}.softmax".format(layer_id),
            )
            add(
                "{}.attention.softmax.normalize".format(layer_id),
                OperatorClass.ELEMENTWISE,
                "{}.softmax".format(layer_id),
            )
            add(
                "{}.attention.residual".format(layer_id),
                OperatorClass.ELEMENTWISE,
                norm_key,
            )
        if layer.is_moe:
            router_key = "{}.router".format(layer_id)
            add(
                "{}.router.softmax.reduce".format(layer_id),
                OperatorClass.REDUCTION,
                router_key,
            )
            add(
                "{}.router.softmax.normalize".format(layer_id),
                OperatorClass.ELEMENTWISE,
                router_key,
            )
            add(
                "{}.router.topk".format(layer_id),
                OperatorClass.REDUCTION,
                router_key,
            )
            add(
                "{}.experts.activation".format(layer_id),
                OperatorClass.ELEMENTWISE,
                "{}.experts".format(layer_id),
            )
            add(
                "{}.moe.residual".format(layer_id),
                OperatorClass.ELEMENTWISE,
                norm_key,
            )
            if layer.has_shared_expert:
                shared_key = "{}.shared_expert".format(layer_id)
                add(
                    "{}.shared_expert.activation".format(layer_id),
                    OperatorClass.ELEMENTWISE,
                    shared_key,
                )
                if layer.shared_expert_gate:
                    add(
                        "{}.shared_expert.gate_apply".format(layer_id),
                        OperatorClass.ELEMENTWISE,
                        shared_key,
                    )
        else:
            add(
                "{}.mlp.activation".format(layer_id),
                OperatorClass.ELEMENTWISE,
                "{}.mlp".format(layer_id),
            )
            add(
                "{}.mlp.residual".format(layer_id),
                OperatorClass.ELEMENTWISE,
                norm_key,
            )
    return tuple(uses)


def validate_scenario(scenario: ScenarioConfig) -> ScenarioValidationReport:
    """Validate one scenario with a request-local compilation context."""

    with _compilation_scope(scenario):
        return _validate_scenario_uncached(scenario)


def _validate_scenario_uncached(
    scenario: ScenarioConfig,
) -> ScenarioValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    information: List[str] = []
    diagnostics: List[Mapping[str, object]] = []
    try:
        execution_view = _execution_view(scenario)
    except (AttributeError, TypeError, ValueError) as exc:
        message = str(exc)
        return ScenarioValidationReport(
            errors=(message,),
            errors_en=(
                "model graph execution coverage rejected: {}".format(
                    message
                ),
            ),
        )
    execution_layers = tuple(
        descriptor.layer for descriptor in execution_view.layer_instances
    )
    declared_weight_bytes = _execution_view_declared_weight_bytes(
        execution_view
    )
    fingerprint_status = mapping_fingerprint_status(scenario)
    if fingerprint_status["mapping_stale"]:
        errors.append(
            "control-plane decision is stale: stored input fingerprint {} does not "
            "match current input fingerprint {}; rerun control-plane placement "
            "before simulation".format(
                fingerprint_status["input_fingerprint"],
                fingerprint_status["current_input_fingerprint"],
            )
        )

    try:
        assert_valid_topology(scenario.hardware)
    except ValueError as exc:
        errors.append(str(exc))

    if scenario.placement.model_name != scenario.model.name:
        errors.append("placement.model_name does not match model.name")
    if scenario.placement.hardware_name != scenario.hardware.name:
        errors.append("placement.hardware_name does not match hardware.name")

    components = _component_map(scenario)
    gpu_components = [component for component in components.values() if _kind(component) == "gpu"]
    if not gpu_components:
        errors.append("reference planner requires at least one GPU component")

    for placement_name, component_id in scenario.placement.op_to_component.items():
        if component_id not in components:
            errors.append(
                "op placement {} references unknown component {}".format(
                    placement_name, component_id
                )
            )
        elif _kind(components[component_id]) not in {"gpu", "cpu"} and not _is_cim(
            components[component_id]
        ):
            errors.append(
                "op placement {} targets unsupported component kind {}".format(
                    placement_name, components[component_id].kind
                )
            )
        elif _kind(components[component_id]) == "cpu":
            try:
                _cpu_profiles(scenario, component_id)
            except (KeyError, TypeError, ValueError):
                errors.append(
                    "op placement {} targets CPU without CPU/host-memory profiles".format(
                        placement_name
                    )
                )
    mapping_diagnostics = non_gemm_cim_mapping_diagnostics(scenario)
    diagnostics.extend(mapping_diagnostics)
    errors.extend(
        _mapping_resolution_message(diagnostic)
        for diagnostic in mapping_diagnostics
    )
    for tensor_name, component_id in scenario.placement.tensor_to_component.items():
        if component_id not in components:
            errors.append(
                "tensor placement {} references unknown component {}".format(
                    tensor_name, component_id
                )
            )
            continue
        component = components[component_id]
        declared_bytes = int(scenario.placement.tensor_bytes.get(tensor_name, 0))
        if (
            declared_bytes > 0
            and component.capacity_bytes <= 0
            and _kind(component) in STORAGE_COMPONENT_KINDS
        ):
            errors.append(
                "V4 placement requires positive capacity_bytes for storage "
                "component {} before placing {} bytes of tensor {}".format(
                    component_id, declared_bytes, tensor_name
                )
            )

    kv_policy = scenario.placement.kv_policy
    kv_from_tensor = scenario.placement.tensor_to_component.get("kv_cache")
    kv_component_id = kv_policy.cache_component or kv_from_tensor
    if (
        kv_policy.cache_component
        and kv_from_tensor
        and kv_policy.cache_component != kv_from_tensor
    ):
        errors.append(
            "kv_policy.cache_component conflicts with "
            "tensor_to_component[kv_cache]"
        )
    if kv_component_id:
        kv_component = components.get(kv_component_id)
        if kv_component is None:
            errors.append("KV cache references unknown component {}".format(kv_component_id))
        elif _kind(kv_component) not in ACTIVE_MEMORY_COMPONENT_KINDS:
            errors.append(
                "KV cache must target writable active memory; {} is {} storage".format(
                    kv_component_id,
                    kv_component.kind
                )
            )
        elif not _is_writable_storage(kv_component):
            errors.append(
                "KV cache component {} is declared read-only".format(kv_component_id)
            )
    prefetch_distance = int(kv_policy.prefetch_distance)
    if prefetch_distance > 0:
        errors.append(
            "V4 placement does not support non-zero prefetch_distance; "
            "proactive KV lookahead is not modeled"
        )
    offload_component_id = kv_policy.offload_component
    if offload_component_id:
        offload_component = components.get(offload_component_id)
        if offload_component is None:
            errors.append(
                "KV offload references unknown component {}".format(
                    offload_component_id
                )
            )
        elif _kind(offload_component) not in STORAGE_COMPONENT_KINDS:
            errors.append(
                "KV offload must target active memory or offload storage, not {}".format(
                    offload_component.kind
                )
            )
        elif not _is_writable_storage(offload_component):
            errors.append(
                "KV offload component {} is declared read-only".format(
                    offload_component_id
                )
            )

    linear_state_component_id = scenario.placement.tensor_to_component.get(
        "linear_state"
    )
    linear_state_offload_id = scenario.placement.tensor_to_component.get(
        "linear_state_offload"
    )
    if linear_state_component_id:
        state_component = components.get(linear_state_component_id)
        if state_component is None:
            errors.append(
                "linear state references unknown component {}".format(
                    linear_state_component_id
                )
            )
        elif _kind(state_component) not in ACTIVE_MEMORY_COMPONENT_KINDS:
            errors.append("linear state must target writable active memory")
        elif not _is_writable_storage(state_component):
            errors.append("linear state cache component is read-only")
    if linear_state_offload_id:
        state_offload = components.get(linear_state_offload_id)
        if state_offload is None:
            errors.append(
                "linear state offload references unknown component {}".format(
                    linear_state_offload_id
                )
            )
        elif _kind(state_offload) not in STORAGE_COMPONENT_KINDS:
            errors.append(
                "linear state offload must target active memory or offload storage"
            )
        elif not _is_writable_storage(state_offload):
            errors.append("linear state offload component is read-only")
    if (
        linear_state_component_id
        and linear_state_offload_id
        and linear_state_component_id == linear_state_offload_id
    ):
        errors.append("linear state cache and offload components must be distinct")
    linear_state_footprint = sum(
        _linear_state_bytes(layer, 1)
        for layer in execution_layers
        if layer.is_linear_attention
    )
    if (
        linear_state_footprint > 0
        and "linear_state" in scenario.placement.tensor_bytes
        and not is_control_plane_generated_tensor(scenario, "linear_state")
        and int(scenario.placement.tensor_bytes["linear_state"])
        < linear_state_footprint
    ):
        errors.append(
            "tensor_bytes[linear_state] declares {} bytes but one request "
            "requires {} bytes".format(
                int(scenario.placement.tensor_bytes["linear_state"]),
                linear_state_footprint,
            )
        )

    weight_component_id = _weight_storage_component(scenario)
    weight_component = components.get(weight_component_id) if weight_component_id else None
    mapped_weight_bytes = int(
        scenario.placement.tensor_bytes.get("model_weights", 0)
    )
    decision_metadata = control_plane_decision(scenario)
    weight_details_raw = decision_metadata.get(
        "weight_tensor_details", {}
    )
    weight_tensor_details = (
        weight_details_raw if isinstance(weight_details_raw, Mapping) else {}
    )
    rank_weight_shards_raw = decision_metadata.get(
        "rank_weight_shards", {}
    )
    rank_weight_shards = (
        rank_weight_shards_raw
        if isinstance(rank_weight_shards_raw, Mapping)
        else {}
    )
    if rank_weight_shards_raw and not isinstance(
        rank_weight_shards_raw, Mapping
    ):
        errors.append(
            "control_plane.decision.rank_weight_shards must be a mapping"
        )
    derived_bytes_raw = decision_metadata.get("derived_tensor_bytes", {})
    derived_tensor_bytes = (
        derived_bytes_raw if isinstance(derived_bytes_raw, Mapping) else {}
    )
    physical_bytes_raw = decision_metadata.get(
        "physical_tensor_bytes", {}
    )
    physical_tensor_bytes = (
        physical_bytes_raw if isinstance(physical_bytes_raw, Mapping) else {}
    )
    if physical_bytes_raw and not isinstance(physical_bytes_raw, Mapping):
        errors.append(
            "control_plane.decision.physical_tensor_bytes must be a mapping"
        )
    detailed_weight_tensors = tuple(
        sorted(
            {
                str(tensor_name)
                for tensor_name in (
                    set(scenario.placement.tensor_to_component)
                    | set(scenario.placement.tensor_bytes)
                    | set(weight_tensor_details)
                    | set(rank_weight_shards)
                    | set(derived_tensor_bytes)
                )
                if str(tensor_name) != "model_weights"
                and "weight" in str(tensor_name).lower()
            }
        )
    )
    logical_weight_bytes: Dict[str, int] = {}
    for tensor_name in detailed_weight_tensors:
        canonical = _canonical_weight_tensor_id(scenario, tensor_name)
        logical_weight_bytes[canonical] = max(
            logical_weight_bytes.get(canonical, 0),
            _logical_weight_tensor_bytes(scenario, tensor_name),
        )
    detailed_weight_bytes = sum(logical_weight_bytes.values())
    active_detailed_weight_tensors = {
        _canonical_weight_tensor_id(scenario, tensor_name)
        for tensor_name in detailed_weight_tensors
        if _weight_tensor_has_active_resident_placement(
            scenario,
            tensor_name,
            weight_tensor_details,
            rank_weight_shards,
        )
    }
    active_detailed_weight_bytes = sum(
        byte_count
        for tensor_name, byte_count in logical_weight_bytes.items()
        if tensor_name in active_detailed_weight_tensors
    )
    resident_replica_raw = decision_metadata.get(
        "resident_cim_replicas", {}
    )
    resident_cim_replicas: Dict[object, object] = (
        dict(resident_replica_raw)
        if isinstance(resident_replica_raw, Mapping)
        else {}
    )
    for tensor_name, detail_raw in weight_tensor_details.items():
        if not isinstance(detail_raw, Mapping):
            continue
        if str(detail_raw.get("residency", "")) != "warm_cim_resident":
            continue
        detail_replicas = detail_raw.get("replica_component_ids")
        if not isinstance(detail_replicas, (list, tuple)) or not detail_replicas:
            continue
        if tensor_name in resident_cim_replicas:
            top_level_replicas = resident_cim_replicas[tensor_name]
            if isinstance(top_level_replicas, (list, tuple)) and {
                str(item) for item in top_level_replicas
            } != {str(item) for item in detail_replicas}:
                errors.append(
                    "control_plane.decision replica metadata disagrees for {}".format(
                        tensor_name
                    )
                )
        else:
            resident_cim_replicas[tensor_name] = detail_replicas
    if declared_weight_bytes > 0:
        if weight_component_id:
            if mapped_weight_bytes <= 0:
                errors.append(
                    "tensor_bytes[model_weights] must declare at least {} bytes".format(
                        declared_weight_bytes
                    )
                )
            elif mapped_weight_bytes < declared_weight_bytes:
                errors.append(
                    "tensor_bytes[model_weights] declares {} bytes but the model "
                    "declares {} weight bytes".format(
                        mapped_weight_bytes, declared_weight_bytes
                    )
                )
            elif mapped_weight_bytes > declared_weight_bytes:
                warnings.append(
                    "tensor_bytes[model_weights] exceeds total_declared_weight_bytes "
                    "by {} bytes".format(mapped_weight_bytes - declared_weight_bytes)
                )
        elif mapped_weight_bytes > 0:
            errors.append(
                "tensor_bytes[model_weights] requires "
                "tensor_to_component[model_weights]"
            )
        aggregate_backing_complete = bool(
            weight_component_id and mapped_weight_bytes >= declared_weight_bytes
        )
        if not scenario.weights_resident:
            if not aggregate_backing_complete and not detailed_weight_tensors:
                errors.append(
                    "cold/streamed weights require tensor_to_component[model_weights] "
                    "and tensor_bytes[model_weights] backing storage, or mapped "
                    "detailed weight tensors"
                )
            elif not aggregate_backing_complete:
                warnings.append(
                    "cold weights use mapped detailed backing tensors "
                    "without an aggregate model_weights backing declaration"
                )
        elif not weight_component_id and not detailed_weight_tensors:
            warnings.append(
                "resident weights have no aggregate or detailed tensor mapping; "
                "physical TP/PP/EP capacity will be checked from model declarations"
            )
        elif not weight_component_id and detailed_weight_bytes < declared_weight_bytes:
            warnings.append(
                "detailed weight tensors explicitly cover {} of {} declared "
                "bytes; remaining resident capacity will be checked across "
                "physical TP/PP/EP placements".format(
                    detailed_weight_bytes, declared_weight_bytes
                )
            )
    elif weight_component_id or mapped_weight_bytes:
        warnings.append(
            "model weight placement cannot be reconciled because "
            "total_declared_weight_bytes is zero"
        )
    if weight_component_id and weight_component is not None:
        if _kind(weight_component) not in STORAGE_COMPONENT_KINDS:
            errors.append(
                "tensor_to_component[model_weights] must target memory or storage, not {}".format(
                    weight_component.kind
                )
            )
        else:
            if (
                scenario.weights_resident
                and _kind(weight_component) in ACTIVE_MEMORY_COMPONENT_KINDS
                and not weight_component.is_writable
            ):
                errors.append(
                    "resident aggregate model_weights component {} must be "
                    "writable active memory, not read-only {}".format(
                        weight_component_id,
                        weight_component.kind,
                    )
                )
            router = _topology_router(scenario)
            for gpu_component in gpu_components:
                if gpu_component.component_id == weight_component_id:
                    continue
                try:
                    router.route(weight_component_id, gpu_component.component_id, 1)
                except ValueError as exc:
                    errors.append("model weight route: {}".format(exc))
            if _kind(weight_component) in OFFLOAD_STORAGE_COMPONENT_KINDS:
                if scenario.weights_resident:
                    warnings.append(
                        "preloaded resident weights use {} {} as capacity-checked "
                        "backing; runtime offload backing reads are suppressed".format(
                            weight_component_id, weight_component.kind
                        )
                    )
                else:
                    warnings.append(
                        "cold streamed weights read from {} {} once per physical "
                        "rank and weight-bearing GEMM invocation".format(
                            weight_component_id, weight_component.kind
                        )
                    )
            else:
                warnings.append(
                    "model weights are read from {} storage through the declared topology".format(
                        weight_component.kind
                    )
                )

    logical_views_raw = decision_metadata.get("logical_weight_views", {})
    logical_weight_views = (
        {str(tensor_name) for tensor_name in logical_views_raw}
        if isinstance(logical_views_raw, Mapping)
        else set()
    )
    logical_weight_views.update(
        str(tensor_name)
        for tensor_name, detail_raw in weight_tensor_details.items()
        if isinstance(detail_raw, Mapping)
        and str(detail_raw.get("residency", ""))
        == "aggregate_backing_logical_view"
    )
    padded_raw = decision_metadata.get("padded_tensor_bytes", {})
    padded_tensor_bytes = padded_raw if isinstance(padded_raw, Mapping) else {}
    if padded_raw and not isinstance(padded_raw, Mapping):
        errors.append(
            "control_plane.decision.padded_tensor_bytes must be a mapping"
        )
    total_physical_raw = decision_metadata.get(
        "cim_total_physical_bytes", {}
    )
    cim_total_physical_bytes = (
        total_physical_raw if isinstance(total_physical_raw, Mapping) else {}
    )
    if total_physical_raw and not isinstance(total_physical_raw, Mapping):
        errors.append(
            "control_plane.decision.cim_total_physical_bytes must be a mapping"
        )

    for tensor_name, metadata_bytes in physical_tensor_bytes.items():
        placement_bytes = scenario.placement.tensor_bytes.get(str(tensor_name))
        try:
            expected_bytes = int(metadata_bytes)
        except (TypeError, ValueError):
            errors.append(
                "control_plane.decision physical_tensor_bytes[{}] must be an integer".format(
                    tensor_name
                )
            )
            continue
        if placement_bytes is None or int(placement_bytes) != expected_bytes:
            errors.append(
                "control_plane.decision physical bytes for {} do not match "
                "PlacementSpec.tensor_bytes".format(tensor_name)
            )
    for tensor_name, detail_raw in weight_tensor_details.items():
        if not isinstance(detail_raw, Mapping):
            errors.append(
                "control_plane.decision weight_tensor_details[{}] must be a mapping".format(
                    tensor_name
                )
            )
            continue
        tensor_key = str(tensor_name)
        if "rank_shards" in detail_raw:
            errors.append(
                "control_plane.decision weight_tensor_details[{}].rank_shards "
                "is not part of V4; use rank_weight_shards".format(
                    tensor_key
                )
            )
        for label, detail_value, top_value in (
            (
                "placement-byte",
                detail_raw.get("placement_bytes"),
                physical_tensor_bytes.get(tensor_key),
            ),
            (
                "padded-byte",
                detail_raw.get("padded_bytes_per_replica"),
                padded_tensor_bytes.get(tensor_key),
            ),
            (
                "total-physical-byte",
                detail_raw.get("total_physical_bytes"),
                cim_total_physical_bytes.get(tensor_key),
            ),
        ):
            if detail_value is None or top_value is None:
                continue
            try:
                disagrees = int(detail_value) != int(top_value)
            except (TypeError, ValueError):
                errors.append(
                    "control_plane.decision {} metadata for {} must be integers".format(
                        label, tensor_key
                    )
                )
                continue
            if disagrees:
                errors.append(
                    "control_plane.decision {} metadata disagrees for {}".format(
                        label, tensor_key
                    )
                )

    bytes_by_component: Dict[str, int] = {}
    for tensor_name, byte_count in scenario.placement.tensor_bytes.items():
        component_id = scenario.placement.tensor_to_component.get(tensor_name)
        if component_id is None:
            warnings.append("tensor {} declares bytes but has no placement".format(tensor_name))
            continue
        if str(tensor_name) in logical_weight_views:
            continue
        bytes_by_component[component_id] = bytes_by_component.get(component_id, 0) + int(byte_count)

    rank_shards_by_tensor: Dict[str, Sequence[object]] = {}
    for tensor_name in sorted(str(name) for name in rank_weight_shards):
        shards_raw = rank_weight_shards[tensor_name]
        if not isinstance(shards_raw, (list, tuple)):
            errors.append(
                "control_plane.decision rank_weight_shards[{}] must be a list".format(
                    tensor_name
                )
            )
            continue
        if shards_raw:
            rank_shards_by_tensor[tensor_name] = shards_raw

    rank_sharded_tensors = set(rank_shards_by_tensor)
    rank_shard_storage_targets: Dict[str, set] = {}
    rank_shard_logical_bytes: Dict[str, int] = {}
    for tensor_key, shards_raw in rank_shards_by_tensor.items():
        detail_raw = weight_tensor_details.get(tensor_key, {})
        detail = detail_raw if isinstance(detail_raw, Mapping) else {}
        primary = scenario.placement.tensor_to_component.get(tensor_key)
        representative_present = tensor_key in scenario.placement.tensor_bytes
        representative_bytes = int(
            scenario.placement.tensor_bytes.get(tensor_key, 0)
        )
        if (
            primary
            and representative_present
            and tensor_key not in logical_weight_views
        ):
            bytes_by_component[primary] = max(
                0,
                bytes_by_component.get(primary, 0) - representative_bytes,
            )
        seen_ranks = set()
        seen_replicas = set()
        unique_logical_shards: Dict[str, int] = {}
        shard_usage: Dict[str, int] = {}
        primary_physical: List[int] = []
        total_physical = 0
        storage_targets = set()
        cold_cim_transient = str(detail.get("residency", "")) == "cold_cim_transient"
        for shard_position, shard_raw in enumerate(shards_raw):
            if not isinstance(shard_raw, Mapping):
                errors.append(
                    "权重张量 {} 的 rank shard[{}] 必须是对象".format(
                        tensor_key, shard_position
                    )
                )
                continue
            rank_value = shard_raw.get(
                "rank_id", shard_raw.get("rank")
            )
            component_value = shard_raw.get("component_id")
            storage_value = shard_raw.get("storage_component_id")
            if (
                component_value is not None
                and storage_value is not None
                and str(component_value) != str(storage_value)
            ):
                errors.append(
                    "权重张量 {} 的 Rank {} component_id 与 storage_component_id 不一致".format(
                        tensor_key, rank_value
                    )
                )
            storage_id = str(
                storage_value
                if storage_value is not None
                else component_value or ""
            )
            compute_id = str(shard_raw.get("compute_component_id", ""))
            shard_id_raw = shard_raw.get("shard_id")
            replica_id_raw = shard_raw.get("replica_id")
            shard_id = str(shard_id_raw or "")
            replica_id = str(replica_id_raw or "")
            if not shard_id_raw or not replica_id_raw:
                errors.append(
                    "权重张量 {} 的 Rank {} 缺少 shard_id 或 replica_id".format(
                        tensor_key, rank_value
                    )
                )
            try:
                logical_bytes = int(shard_raw.get("logical_bytes", 0))
                physical_bytes = int(shard_raw.get("physical_bytes", 0))
            except (TypeError, ValueError):
                logical_bytes = -1
                physical_bytes = -1
            if rank_value in seen_ranks:
                errors.append(
                    "权重张量 {} 的 Rank {} 分片重复".format(
                        tensor_key, rank_value
                    )
                )
            seen_ranks.add(rank_value)
            if replica_id in seen_replicas:
                errors.append(
                    "权重张量 {} 的物理副本 {} 重复".format(
                        tensor_key, replica_id
                    )
                )
                continue
            seen_replicas.add(replica_id)
            if not storage_id or storage_id not in components:
                errors.append(
                    "权重张量 {} 的 Rank {} 引用了未知存储组件 {}".format(
                        tensor_key, rank_value, storage_id or "<空>"
                    )
                )
                continue
            storage_component = components[storage_id]
            if (
                _kind(storage_component) not in STORAGE_COMPONENT_KINDS
                and not _is_cim(storage_component)
            ):
                errors.append(
                    "权重张量 {} 的 Rank {} 目标 {} 不是存储或 CIM 组件".format(
                        tensor_key, rank_value, storage_id
                    )
                )
            elif (
                scenario.weights_resident
                and not cold_cim_transient
                and not _is_active_resident_weight_storage(storage_component)
            ):
                errors.append(
                    "resident rank weight shard {} Rank {} must target writable active memory or CIM, not {}".format(
                        tensor_key,
                        rank_value,
                        storage_component.kind,
                    )
                )
            if not compute_id or compute_id not in components:
                errors.append(
                    "权重张量 {} 的 Rank {} 引用了未知计算组件 {}".format(
                        tensor_key, rank_value, compute_id or "<空>"
                    )
                )
            if logical_bytes < 0 or physical_bytes < 0:
                errors.append(
                    "权重张量 {} 的 Rank {} logical_bytes/physical_bytes 必须是非负整数".format(
                        tensor_key, rank_value
                    )
                )
                continue
            existing_logical = unique_logical_shards.get(shard_id)
            if (
                existing_logical is not None
                and existing_logical != logical_bytes
            ):
                errors.append(
                    "权重张量 {} 的逻辑分片 {} 在副本间字节数不一致".format(
                        tensor_key, shard_id
                    )
                )
            else:
                unique_logical_shards[shard_id] = logical_bytes
            storage_targets.add(storage_id)
            shard_usage[storage_id] = (
                shard_usage.get(storage_id, 0) + physical_bytes
            )
            if primary and storage_id == str(primary):
                primary_physical.append(physical_bytes)
            total_physical += physical_bytes
        unique_logical_total = sum(unique_logical_shards.values())
        rank_shard_logical_bytes[tensor_key] = unique_logical_total
        rank_shard_storage_targets[tensor_key] = storage_targets
        declared_logical = detail.get("logical_bytes")
        if declared_logical is not None:
            try:
                logical_matches = int(declared_logical) == unique_logical_total
            except (TypeError, ValueError):
                logical_matches = False
            if not logical_matches:
                errors.append(
                    "权重张量 {} 的唯一逻辑分片字节数与 logical_bytes 不一致".format(
                        tensor_key
                    )
                )
        for storage_id, byte_count in shard_usage.items():
            bytes_by_component[storage_id] = (
                bytes_by_component.get(storage_id, 0) + byte_count
            )
        if primary and representative_present:
            primary_component = components.get(str(primary))
            expected_representative = (
                max(primary_physical, default=0)
                if primary_component is not None
                and _is_cim(primary_component)
                else sum(primary_physical)
            )
            if representative_bytes != expected_representative:
                errors.append(
                    "权重张量 {} 的兼容代表字节数 {} 与主组件 {} 的代表分片字节数 {} 不一致".format(
                        tensor_key,
                        representative_bytes,
                        primary,
                        expected_representative,
                    )
                )
        declared_total = detail.get("total_physical_bytes")
        if declared_total is not None:
            try:
                total_matches = int(declared_total) == total_physical
            except (TypeError, ValueError):
                total_matches = False
            if not total_matches:
                errors.append(
                    "权重张量 {} 的总物理字节数与 rank shards 不一致".format(
                        tensor_key
                    )
                )

    # A complete warm rank-local layout is the physical deployment.  The
    # aggregate model_weights tensor is a canonical/cold backing declaration
    # and must not be charged again on top of active local rank shards.
    rank_local_logical_total = sum(rank_shard_logical_bytes.values())
    if (
        scenario.weights_resident
        and weight_component_id
        and weight_component is not None
        and _kind(weight_component) in ACTIVE_MEMORY_COMPONENT_KINDS
        and mapped_weight_bytes > 0
        and declared_weight_bytes > 0
        and rank_local_logical_total >= declared_weight_bytes
    ):
        bytes_by_component[weight_component_id] = max(
            0,
            bytes_by_component.get(weight_component_id, 0)
            - mapped_weight_bytes,
        )
    for tensor_name, replica_targets in resident_cim_replicas.items():
        if not isinstance(replica_targets, (list, tuple)):
            errors.append(
                "control_plane.decision resident_cim_replicas[{}] must be a list".format(
                    tensor_name
                )
            )
            continue
        tensor_key = str(tensor_name)
        primary = scenario.placement.tensor_to_component.get(tensor_key)
        targets = tuple(dict.fromkeys(str(item) for item in replica_targets))
        for component_id in targets:
            if component_id not in components:
                errors.append(
                    "control_plane.decision resident replica {} references unknown component {}".format(
                        tensor_name, component_id
                    )
                    )
            elif not _is_cim(components[component_id]):
                errors.append(
                    "control_plane.decision resident replica {} target {} is not CIM".format(
                        tensor_name, component_id
                    )
                )
        if tensor_key in rank_sharded_tensors:
            expected_targets = {
                component_id
                for component_id in rank_shard_storage_targets.get(
                    tensor_key, set()
                )
                if component_id in components
                and _is_cim(components[component_id])
            }
            if set(targets) != expected_targets:
                errors.append(
                    "control_plane.decision resident replicas for {} disagree with "
                    "rank_weight_shards".format(tensor_key)
                )
            # Rank-local metadata already accounts every physical replica and
            # validates its total.  The schema-0 padded/scalar fields are only
            # UI representatives and must not multiply capacity.
            continue
        detail_raw = weight_tensor_details.get(tensor_key, {})
        detail = detail_raw if isinstance(detail_raw, Mapping) else {}
        residency = str(detail.get("residency", ""))
        warm_resident = scenario.weights_resident and residency != "cold_cim_transient"
        if not warm_resident:
            declared_total = cim_total_physical_bytes.get(tensor_key)
            if declared_total is not None:
                try:
                    if int(declared_total) != 0:
                        errors.append(
                            "cold CIM tensor {} must declare zero total physical bytes".format(
                                tensor_key
                            )
                        )
                except (TypeError, ValueError):
                    errors.append(
                        "control_plane.decision cim_total_physical_bytes[{}] must be an integer".format(
                            tensor_key
                        )
                    )
            continue
        padded_value = padded_tensor_bytes.get(
            tensor_key, detail.get("padded_bytes_per_replica")
        )
        try:
            padded_bytes = int(padded_value) if padded_value is not None else 0
        except (TypeError, ValueError):
            padded_bytes = 0
            errors.append(
                "control_plane.decision padded bytes for {} must be an integer".format(
                    tensor_key
                )
            )
        if padded_bytes <= 0:
            warnings.append(
                "warm CIM weight tensor {} lacks padded physical-byte metadata; "
                "capacity is checked only from its declared placement bytes".format(
                    tensor_key
                )
            )
            continue
        if primary and str(primary) not in targets:
            errors.append(
                "control_plane.decision resident replicas for {} omit primary component {}".format(
                    tensor_key, primary
                )
            )
        placement_bytes = scenario.placement.tensor_bytes.get(tensor_key)
        if placement_bytes is None:
            errors.append(
                "warm CIM tensor {} lacks PlacementSpec.tensor_bytes".format(
                    tensor_key
                )
            )
        elif int(placement_bytes) != padded_bytes:
            errors.append(
                "warm CIM tensor {} placement bytes {} do not match padded "
                "bytes per replica {}".format(
                    tensor_key, int(placement_bytes), padded_bytes
                )
            )
        for component_id in targets:
            if component_id == primary or component_id not in components:
                continue
            bytes_by_component[component_id] = (
                bytes_by_component.get(component_id, 0) + padded_bytes
            )
        expected_total = padded_bytes * len(targets)
        declared_total = cim_total_physical_bytes.get(
            tensor_key, detail.get("total_physical_bytes")
        )
        if declared_total is not None:
            try:
                if int(declared_total) != expected_total:
                    errors.append(
                        "CIM tensor {} total physical bytes {} do not match {} "
                        "padded bytes across {} replicas".format(
                            tensor_key,
                            int(declared_total),
                            padded_bytes,
                            len(targets),
                        )
                    )
            except (TypeError, ValueError):
                errors.append(
                    "control_plane.decision total physical bytes for {} must be an integer".format(
                        tensor_key
                    )
                )

    replica_metadata_tensors = {str(name) for name in resident_cim_replicas}
    for tensor_name in detailed_weight_tensors:
        component_id = scenario.placement.tensor_to_component.get(tensor_name)
        component = components.get(component_id) if component_id else None
        detail_raw = weight_tensor_details.get(tensor_name, {})
        detail = detail_raw if isinstance(detail_raw, Mapping) else {}
        has_padded_metadata = (
            tensor_name in padded_tensor_bytes
            or detail.get("padded_bytes_per_replica") is not None
        )
        has_replica_metadata = (
            tensor_name in replica_metadata_tensors
            or bool(detail.get("replica_component_ids"))
        )
        if (
            scenario.weights_resident
            and component is not None
            and _is_cim(component)
            and (not has_padded_metadata or not has_replica_metadata)
        ):
            warnings.append(
                "manual CIM weight tensor {} lacks complete padded/replica metadata; "
                "physical resident capacity may be underestimated".format(
                    tensor_name
                )
            )
    for component_id, byte_count in bytes_by_component.items():
        component = components.get(component_id)
        capacity = (
            _component_weight_capacity(scenario, component)
            if component is not None
            else 0
        )
        if (
            component is not None
            and byte_count > 0
            and _kind(component) in OFFLOAD_STORAGE_COMPONENT_KINDS
            and component.capacity_bytes <= 0
        ):
            errors.append(
                "offload storage component {} has capacity_bytes=0; declare a positive "
                "capacity before placing {} tensor bytes".format(
                    component_id, byte_count
                )
            )
        elif (
            component is not None
            and byte_count > 0
            and _kind(component) in ACTIVE_MEMORY_COMPONENT_KINDS
            and component.capacity_bytes <= 0
        ):
            errors.append(
                "V4 placement requires positive capacity_bytes for active "
                "memory component {} before placing {} tensor bytes".format(
                    component_id, byte_count
                )
            )
        elif component is not None and capacity and byte_count > capacity:
            errors.append(
                "declared tensors require {} bytes on {}, capacity is {}".format(
                    byte_count, component_id, capacity
                )
            )

    mapped_cim_component_ids = tuple(
        sorted(
            {
                component_id
                for component_id in scenario.placement.op_to_component.values()
                if component_id in components
                and _is_cim(components[component_id])
            }
        )
    )
    mapped_to_cim = bool(mapped_cim_component_ids)
    for component_id in mapped_cim_component_ids:
        try:
            _resolve_component_profile(
                scenario,
                component_id, DigitalSramCimProfile
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                "CIM placement target {} has no valid CIM profile: {}".format(
                    component_id, exc
                )
            )
    if mapped_cim_component_ids and scenario.cim_interconnect is None:
        errors.append("CIM placement requires both cim_profile and cim_interconnect")

    for layer in execution_layers:
        try:
            _layer_precision_bits(layer)
            if layer.is_linear_attention and layer.linear_attention is not None:
                _dtype_bits(layer.linear_attention.state_dtype)
        except ValueError as exc:
            errors.append("layer {}: {}".format(layer.layer_id, exc))

    parallel_plan: Optional[ParallelPlan] = None
    try:
        parallel_plan = _parallel_plan(scenario)
        router = _topology_router(scenario)
        if any(
            not layer.is_linear_attention
            for layer in execution_layers
        ):
            for rank in parallel_plan.ranks:
                cache_component_id, _offload, _ratio = _kv_components(
                    scenario, rank
                )
                if not cache_component_id:
                    errors.append(
                        "full-attention rank {} has no writable KV cache "
                        "placement; configure kv_policy.cache_component, "
                        "tensor_to_component[kv_cache], "
                        "or rank memory_component_id".format(rank.rank)
                    )
                    continue
                cache_component = components.get(cache_component_id)
                if (
                    cache_component is None
                    or _kind(cache_component)
                    not in ACTIVE_MEMORY_COMPONENT_KINDS
                    or not _is_writable_storage(cache_component)
                ):
                    # The global placement and parallel-plan checks emit the
                    # actionable component-specific diagnostic.
                    continue
                local_kv_component = (
                    rank.memory_component_id or rank.component_id
                )
                if cache_component_id != local_kv_component:
                    try:
                        router.route(
                            cache_component_id,
                            local_kv_component,
                            1,
                            policy=parallel_plan.routing_policy,
                        )
                    except ValueError as exc:
                        errors.append(
                            "rank {} KV cache route: {}".format(
                                rank.rank, exc
                            )
                        )
        for stage in range(parallel_plan.pp_degree):
            for group in (
                parallel_plan.tp_group(stage, 0),
                parallel_plan.ep_group(stage, 0),
            ):
                component_ids = tuple(dict.fromkeys(rank.component_id for rank in group))
                if len(component_ids) > 1:
                    _collective_plan(
                        scenario,
                        router,
                        "all_reduce",
                        component_ids,
                        1,
                        algorithm=parallel_plan.collective_algorithm,
                        routing_policy=parallel_plan.routing_policy,
                    )
        for stage in range(parallel_plan.pp_degree - 1):
            for tp_rank in range(parallel_plan.tp_degree):
                source = parallel_plan.rank_at(tp_rank, stage, 0).component_id
                target = parallel_plan.rank_at(tp_rank, stage + 1, 0).component_id
                router.route(
                    source, target, 1, policy=parallel_plan.routing_policy
                )
        for rank in parallel_plan.ranks:
            for endpoint, label in (
                (rank.memory_component_id, "memory"),
                (rank.cim_component_id, "CIM"),
            ):
                if endpoint and endpoint != rank.component_id:
                    try:
                        router.route(
                            rank.component_id,
                            endpoint,
                            1,
                            policy=parallel_plan.routing_policy,
                        )
                    except ValueError:
                        raise ValueError(
                            "rank {} 的 {} 路由无效；请检查拓扑连通性".format(
                                rank.rank, "内存" if label == "memory" else label
                            )
                        )
        plan_ranks = {rank.rank: rank for rank in parallel_plan.ranks}
        for tensor_name, tensor_rank_shards in rank_shards_by_tensor.items():
            for shard_raw in tensor_rank_shards:
                if not isinstance(shard_raw, Mapping):
                    continue
                try:
                    rank_number = int(
                        shard_raw.get("rank_id", shard_raw.get("rank", -1))
                    )
                except (TypeError, ValueError):
                    continue
                plan_rank = plan_ranks.get(rank_number)
                compute_id = str(
                    shard_raw.get("compute_component_id", "")
                )
                storage_id = str(
                    shard_raw.get(
                        "storage_component_id",
                        shard_raw.get("component_id", ""),
                    )
                )
                if plan_rank is None:
                    errors.append(
                        "权重张量 {} 的分片引用了当前并行计划中不存在的 Rank {}".format(
                            tensor_name, rank_number
                        )
                    )
                    continue
                if compute_id != plan_rank.component_id:
                    errors.append(
                        "权重张量 {} 的 Rank {} 计算组件 {} 与当前并行计划 {} 不一致".format(
                            tensor_name,
                            rank_number,
                            compute_id,
                            plan_rank.component_id,
                        )
                    )
                    continue
                if storage_id and storage_id != compute_id:
                    try:
                        router.route(
                            storage_id,
                            compute_id,
                            1,
                            policy=parallel_plan.routing_policy,
                        )
                    except ValueError as exc:
                        errors.append(
                            "权重张量 {} 的 Rank {} 本地分片路由无效：{}".format(
                                tensor_name, rank_number, exc
                            )
                        )
        for tensor_name in detailed_weight_tensors:
            for target_component_id in _weight_execution_targets(
                scenario, parallel_plan, tensor_name
            ):
                source_component_id, _, logical_tensor = (
                    _weight_source_for_tensor(
                        scenario, tensor_name, target_component_id
                    )
                )
                if source_component_id is None:
                    errors.append(
                        "weight tensor {} has no physical source for target {}".format(
                            logical_tensor, target_component_id
                        )
                    )
                    continue
                if source_component_id == target_component_id:
                    continue
                try:
                    router.route(
                        source_component_id,
                        target_component_id,
                        1,
                        policy=parallel_plan.routing_policy,
                    )
                except ValueError as exc:
                    errors.append(
                        "weight tensor {} route to {}: {}".format(
                            logical_tensor, target_component_id, exc
                        )
                    )
        if weight_component_id and not scenario.weights_resident:
            for cim_target in _configured_cim_targets(scenario, parallel_plan):
                if cim_target == weight_component_id:
                    continue
                try:
                    router.route(
                        weight_component_id,
                        cim_target,
                        1,
                        policy=parallel_plan.routing_policy,
                    )
                except ValueError as exc:
                    errors.append("model weight CIM route: {}".format(exc))
        if any(layer.is_linear_attention for layer in execution_layers):
            for rank in parallel_plan.ranks:
                active = (
                    linear_state_component_id
                    or rank.memory_component_id
                    or kv_component_id
                    or rank.component_id
                )
                if active != rank.component_id:
                    router.route(
                        str(active),
                        rank.component_id,
                        1,
                        policy=parallel_plan.routing_policy,
                    )
                if linear_state_offload_id and active != linear_state_offload_id:
                    router.route(
                        str(linear_state_offload_id),
                        str(active),
                        1,
                        policy=parallel_plan.routing_policy,
                    )
        if (
            declared_weight_bytes > 0
            and scenario.weights_resident
        ):
            resident_components = _resident_weight_components(
                scenario, parallel_plan
            )
            aggregate_active_bytes = (
                min(mapped_weight_bytes, declared_weight_bytes)
                if weight_component is not None
                and _kind(weight_component) in ACTIVE_MEMORY_COMPONENT_KINDS
                and weight_component.is_writable
                else 0
            )
            remaining_weight_bytes = max(
                0,
                declared_weight_bytes
                - active_detailed_weight_bytes
                - aggregate_active_bytes,
            )
            known_capacity = sum(
                _unallocated_component_capacity(scenario, component_id)
                for component_id in resident_components
            )
            unknown_capacity = tuple(
                component_id
                for component_id in resident_components
                if components[component_id].capacity_bytes <= 0
            )
            if known_capacity < remaining_weight_bytes and not unknown_capacity:
                errors.append(
                    "remaining resident model weights require {} bytes across "
                    "physical TP/PP/EP placements, but {} provide {} bytes".format(
                        remaining_weight_bytes,
                        ", ".join(resident_components) or "no components",
                        known_capacity,
                    )
                )
            elif remaining_weight_bytes > 0 and unknown_capacity:
                errors.append(
                    "V4 placement requires positive capacity_bytes for resident "
                    "model-weight components; capacity is unknown for {}".format(
                        ", ".join(unknown_capacity)
                    )
                )
            else:
                information.append(
                    "resident model weights are capacity-checked as physical "
                    "TP/PP/EP shards across {} component(s); {} bytes are "
                    "explicit detailed placements".format(
                        len(resident_components), detailed_weight_bytes
                    )
                )
        if declared_weight_bytes > 0 and scenario.weights_resident and mapped_to_cim:
            cim_targets = _configured_cim_targets(scenario, parallel_plan)
            explicit_cim_bytes = 0
            for tensor_name in detailed_weight_tensors:
                declared_total = cim_total_physical_bytes.get(tensor_name)
                if declared_total is not None:
                    try:
                        explicit_cim_bytes += max(0, int(declared_total))
                    except (TypeError, ValueError):
                        pass
                    continue
                replica_count = _physical_cim_replica_count(
                    scenario,
                    tensor_name,
                    cim_targets,
                    resident_cim_replicas,
                )
                placement_bytes = scenario.placement.tensor_bytes.get(
                    tensor_name
                )
                if replica_count > 0 and placement_bytes is not None:
                    explicit_cim_bytes += int(placement_bytes) * replica_count
            cim_required_bytes = explicit_cim_bytes or _estimated_cim_weight_bytes(
                scenario, execution_view=execution_view
            )
            cim_capacity = sum(
                _resident_weight_capacity(scenario, component_id)
                for component_id in cim_targets
            )
            unknown_cim_capacity = tuple(
                component_id
                for component_id in cim_targets
                if _component_weight_capacity(
                    scenario, components[component_id]
                )
                <= 0
            )
            if cim_capacity < cim_required_bytes and not unknown_cim_capacity:
                errors.append(
                    "warm CIM resident weights require {} bytes across {}, "
                    "capacity is {} bytes".format(
                        cim_required_bytes,
                        ", ".join(cim_targets) or "no CIM components",
                        cim_capacity,
                    )
                )
            elif unknown_cim_capacity:
                warnings.append(
                    "warm CIM resident weight capacity is not fully verifiable; "
                    "capacity_bytes is unspecified for {}".format(
                        ", ".join(unknown_cim_capacity)
                    )
                )
            else:
                source = (
                    "explicit detailed weight tensors"
                    if explicit_cim_bytes
                    else "mapped-layer weight estimates"
                )
                warnings.append(
                    "warm CIM resident capacity is checked from {} across {} "
                    "physical component(s)".format(source, len(cim_targets))
                )
    except (KeyError, ValueError) as exc:
        errors.append("parallel plan: {}".format(exc))

    if offload_component_id and kv_component_id:
        if str(offload_component_id) == str(kv_component_id):
            errors.append("KV cache and offload components must be distinct")
        try:
            _topology_router(scenario).route(
                str(kv_component_id), str(offload_component_id), 1
            )
        except ValueError as exc:
            errors.append("KV offload route: {}".format(exc))

    scheduler = getattr(scenario.workload, "scheduler", None)
    scheduler_preemption = str(
        getattr(scheduler, "preemption_policy", "auto")
    ).lower()
    scheduler_preemption_enabled = bool(
        getattr(scheduler, "preemption_enabled", True)
    )
    scheduler_mode = str(getattr(scheduler, "mode", "static"))
    kv_preemption = str(
        getattr(kv_policy, "preemption_mode", "auto")
        if kv_policy is not None
        else "auto"
    ).lower()
    swap_required = scheduler_mode == "continuous" and scheduler_preemption_enabled and (
        scheduler_preemption == "swap"
        or (scheduler_preemption == "auto" and kv_preemption == "swap")
    )
    if swap_required and not offload_component_id:
        errors.append("KV swap preemption requires an offload component")
    if (
        swap_required
        and kv_policy is not None
        and float(getattr(kv_policy, "offload_ratio", 1.0)) <= 0
    ):
        errors.append("KV swap preemption requires a positive offload_ratio")
    if (
        swap_required
        and any(layer.is_linear_attention for layer in execution_layers)
        and not linear_state_offload_id
    ):
        errors.append(
            "swap preemption for a linear mixer requires "
            "tensor_to_component[linear_state_offload]"
        )
    try:
        linear_state_offload_ratio = float(
            scenario.placement.metadata.get("linear_state_offload_ratio", 1.0)
        )
        if not 0.0 <= linear_state_offload_ratio <= 1.0:
            errors.append("linear_state_offload_ratio must be in [0, 1]")
    except (TypeError, ValueError):
        errors.append("linear_state_offload_ratio must be numeric")

    supported_mapping_keys = _reference_lowering_op_mapping_keys(
        scenario, execution_view=execution_view
    )
    for key in scenario.placement.op_to_component:
        if key not in supported_mapping_keys:
            warnings.append("op placement key {} is not consumed by the reference lowering".format(key))

    requests = materialize_requests(scenario, execution_view=execution_view)
    if not requests:
        errors.append("workload must contain at least one request")
    if scenario.workload.requests:
        synthetic_conflicts: List[str] = []
        if (
            scenario.workload.request_count != 1
            and scenario.workload.request_count
            != len(scenario.workload.requests)
        ):
            synthetic_conflicts.append("request_count")
        if scenario.workload.prompt_tokens > 0 and any(
            request.prompt_tokens != scenario.workload.prompt_tokens
            for request in scenario.workload.requests
        ):
            synthetic_conflicts.append("prompt_tokens")
        if scenario.workload.output_tokens > 0 and any(
            request.output_tokens != scenario.workload.output_tokens
            for request in scenario.workload.requests
        ):
            synthetic_conflicts.append("output_tokens")
        if scenario.workload.arrival_rate_rps > 0:
            interval_ns = 1_000_000_000.0 / scenario.workload.arrival_rate_rps
            ordered_requests = sorted(
                scenario.workload.requests,
                key=lambda item: (item.arrival_ns, item.request_id),
            )
            first_arrival = ordered_requests[0].arrival_ns
            if any(
                not math.isclose(
                    request.arrival_ns,
                    first_arrival + index * interval_ns,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-6,
                )
                for index, request in enumerate(ordered_requests)
            ):
                synthetic_conflicts.append("arrival_rate_rps")
        if synthetic_conflicts:
            warnings.append(
                "explicit workload.requests take precedence over conflicting "
                "top-level synthetic fields: {}".format(
                    ", ".join(synthetic_conflicts)
                )
            )
    validation_requests: Sequence[RequestSpec]
    if scenario.workload.requests:
        validation_requests = requests
    elif requests:
        # Synthetic workloads are homogeneous.  Validate the deterministic
        # template request once instead of materializing every synthetic row.
        validation_requests = (requests[0],)
    else:
        validation_requests = ()

    model_max_sequence_length = int(
        execution_view.max_sequence_length or 0
    )
    for request in validation_requests:
        if request.prompt_tokens <= 0:
            errors.append("request {} must have at least one prompt token".format(request.request_id))
        full_sequence_tokens = max(0, int(request.prompt_tokens)) + max(
            0, int(request.output_tokens)
        )
        if (
            model_max_sequence_length > 0
            and full_sequence_tokens > model_max_sequence_length
        ):
            errors.append(
                "request {} sequence has {} tokens, exceeding model "
                "max_sequence_length {}".format(
                    request.request_id,
                    full_sequence_tokens,
                    model_max_sequence_length,
                )
            )
        request_modalities = _request_modalities(request)
        unsupported = sorted(
            set(request_modalities) - set(scenario.model.supported_modalities)
        )
        if unsupported:
            errors.append(
                "request {} uses unsupported modalities: {}".format(
                    request.request_id, ", ".join(unsupported)
                )
            )
        non_text = sorted(set(request_modalities) - {"text"})
        if non_text:
            errors.append(
                "request {} has no executable IR for modalities: {}; "
                "the analytical core currently lowers only the ordered text backbone".format(
                    request.request_id, ", ".join(non_text)
                )
            )

    # Admission always uses the current workload.  Auto-generated runtime-state
    # tensor sizes are estimates from the mapping run, not capacity declarations;
    # the serving planner derives their budgets from physical placement instead.
    if requests and not any(
        "does not support non-zero prefetch_distance" in error
        for error in errors
    ):
        try:
            from .serving import serving_admission_diagnostics

            admission_failures = serving_admission_diagnostics(scenario)
            # The planner validates the complete prompt+output sequence above,
            # while serving admission performs the same guard before runtime.
            # Preserve one canonical diagnostic instead of reporting the same
            # request twice when both layers reject it.
            admission_failures = tuple(
                failure
                for failure in admission_failures
                if failure not in errors and failure not in warnings
            )
            if (
                admission_failures
                and not scenario.workload.requests
            ):
                errors.extend(admission_failures)
            elif admission_failures and len(admission_failures) >= len(requests):
                errors.extend(admission_failures)
            elif admission_failures:
                warnings.extend(
                    "serving admission will reject this request: {}".format(
                        failure
                    )
                    for failure in admission_failures
                )
        except (TypeError, ValueError) as exc:
            errors.append("serving admission preflight failed: {}".format(exc))

    if mapped_to_cim:
        if scenario.weights_resident:
            warnings.append("CIM results use warm weight-resident semantics")
        else:
            warnings.append("CIM results include per-operation cold weight loads")
    information.append(
        "all operators, collectives, KV traffic, and host orchestration use "
        "the task/transaction-level analytical lowering"
    )
    scheduler = getattr(scenario.workload, "scheduler", None)
    scheduler_mode = str(getattr(scheduler, "mode", "static")) if scheduler else "static"
    if scheduler_mode == "continuous":
        information.append("request scheduling uses deterministic continuous batching")
    else:
        information.append("request scheduling uses deterministic static task ordering")
    errors_en = tuple(errors)
    warnings_en = tuple(warnings)
    information_en = tuple(information)
    return ScenarioValidationReport(
        errors=tuple(_planner_message_zh(message) for message in errors_en),
        warnings=tuple(
            _planner_message_zh(message, warning=True) for message in warnings_en
        ),
        information=tuple(
            _planner_message_zh(message, information=True)
            for message in information_en
        ),
        input_fingerprint=fingerprint_status["input_fingerprint"],
        current_input_fingerprint=fingerprint_status[
            "current_input_fingerprint"
        ],
        mapping_stale=bool(fingerprint_status["mapping_stale"]),
        errors_en=errors_en,
        warnings_en=warnings_en,
        information_en=information_en,
        diagnostics=tuple(diagnostics),
    )


class _SyntheticRequestSequence(Sequence[RequestSpec]):
    """Lazy view over deterministic synthetic request rows.

    Explicit trace requests stay as their original tuple.  Synthetic workloads
    may intentionally declare very large ``request_count`` values, so this view
    preserves the old index/iteration semantics without allocating a
    ``RequestSpec`` tuple up front.
    """

    def __init__(
        self,
        workload: WorkloadSpec,
        *,
        max_sequence_length: int = 0,
    ) -> None:
        self._count = max(0, int(workload.request_count))
        self._prompt_tokens = int(workload.prompt_tokens)
        self._output_tokens = int(workload.output_tokens)
        self._max_sequence_length = max(0, int(max_sequence_length or 0))
        self._interval_ns = (
            1_000_000_000.0 / float(workload.arrival_rate_rps)
            if workload.arrival_rate_rps > 0
            else 0.0
        )

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[RequestSpec]:
        for index in range(self._count):
            yield self._at(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(
                self._at(position)
                for position in range(*index.indices(self._count))
            )
        if not isinstance(index, int):
            raise TypeError("request index must be an integer or slice")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("request index out of range")
        return self._at(index)

    def __repr__(self) -> str:
        return (
            "{}(count={!r}, prompt_tokens={!r}, output_tokens={!r}, max_sequence_length={!r})"
            .format(
                type(self).__name__,
                self._count,
                self._prompt_tokens,
                self._output_tokens,
                self._max_sequence_length,
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _SyntheticRequestSequence):
            return (
                self._count,
                self._prompt_tokens,
                self._output_tokens,
                self._max_sequence_length,
                self._interval_ns,
            ) == (
                other._count,
                other._prompt_tokens,
                other._output_tokens,
                other._max_sequence_length,
                other._interval_ns,
            )
        if isinstance(other, Sequence):
            return len(other) == self._count and all(
                left == right for left, right in zip(self, other)
            )
        return False

    def _at(self, index: int) -> RequestSpec:
        return RequestSpec(
            request_id="request-{:04d}".format(index),
            arrival_ns=index * self._interval_ns,
            prompt_tokens=self._prompt_tokens,
            output_tokens=self._output_tokens,
        )


def materialize_requests(
    scenario: ScenarioConfig,
    *,
    execution_view: Optional[ModelGraphExecutionView] = None,
) -> Sequence[RequestSpec]:
    workload = scenario.workload
    if workload.requests:
        return workload.requests
    if workload.request_count <= 0:
        return ()
    if execution_view is None:
        execution_view = _execution_view(scenario)
    return _SyntheticRequestSequence(
        workload,
        max_sequence_length=execution_view.max_sequence_length,
    )


def _request_modalities(request: RequestSpec) -> Tuple[str, ...]:
    raw = request.metadata.get("modalities", request.metadata.get("modality", ()))
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = tuple(str(item) for item in raw)
    elif raw in (None, ()):
        values = ()
    else:
        values = (str(raw),)
    inferred = list(values)
    for modality in ("image", "video", "audio"):
        if request.metadata.get("has_{}".format(modality)) is True:
            inferred.append(modality)
    normalized = tuple(
        dict.fromkeys(
            str(item).strip().lower().replace("-", "_")
            for item in inferred
            if str(item).strip()
        )
    )
    return normalized or ("text",)


class _TaskBuilder:
    def __init__(self, request: RequestSpec) -> None:
        self.request = request
        self.tasks: List[TaskSpec] = []
        self.previous: Optional[str] = None
        self.counter = 0
        self._rank_value_components: Dict[str, Dict[int, str]] = {}
        self._last_coherent_dma_task: Optional[str] = None
        # Planner-private payloads used only by the capability-guarded serving
        # invocation emitter.  They are keyed by the already validated task id
        # and never become part of ScheduleIR or its public metadata contract.
        self._task_segment_dynamic_payloads: Dict[str, object] = {}

    def coherent_dma_dependencies(
        self, dependencies: Sequence[str]
    ) -> Tuple[str, ...]:
        # Independent movements contend through the shared DMA resource
        # capacity.  A synthetic predecessor would force capacity>1 back to
        # one lane and double-model the hardware limit.
        return tuple(dict.fromkeys(dependencies))

    def record_coherent_dma_task(self, task_id: str) -> None:
        self._last_coherent_dma_task = task_id

    def rank_value_component(
        self,
        dependencies: Sequence[str],
        rank_id: int,
    ) -> Optional[str]:
        """Return the unambiguous component holding a rank activation."""

        components = {
            component
            for dependency in dependencies
            for component in (
                self._rank_value_components.get(dependency, {}).get(rank_id),
            )
            if component is not None
        }
        return next(iter(components)) if len(components) == 1 else None

    def record_rank_value(
        self,
        task_id: str,
        rank_id: int,
        component_id: str,
    ) -> None:
        self._rank_value_components.setdefault(task_id, {})[rank_id] = (
            component_id
        )

    def discard_rank_value(self, task_id: str, rank_id: int) -> None:
        values = self._rank_value_components.get(task_id)
        if values is None:
            return
        values.pop(rank_id, None)
        if not values:
            self._rank_value_components.pop(task_id, None)

    def add(
        self,
        name: str,
        category: TaskCategory,
        demands: Sequence[ResourceDemand] = (),
        *,
        metadata: Optional[Mapping[str, object]] = None,
        marker: Optional[TraceMarker] = None,
        token_index: Optional[int] = None,
        dependency: Optional[str] = None,
        dependencies: Optional[Sequence[str]] = None,
        advance: bool = True,
        earliest_start_ns: float = 0.0,
    ) -> str:
        self.counter += 1
        safe_name = _UNSAFE_TASK_NAME_PATTERN.sub("-", name)
        task_id = "{}.{:05d}.{}".format(self.request.request_id, self.counter, safe_name)
        if dependencies is not None:
            dependency_ids = tuple(item for item in dependencies if item)
        else:
            predecessor = self.previous if dependency is None else dependency
            dependency_ids = (predecessor,) if predecessor else ()
        self.tasks.append(
            TaskSpec(
                task_id=task_id,
                request_id=self.request.request_id,
                name=name,
                category=category,
                dependencies=dependency_ids,
                demands=tuple(demands),
                earliest_start_ns=earliest_start_ns,
                marker=marker,
                token_index=token_index,
                metadata=dict(metadata or {}),
            )
        )
        inherited: Dict[int, str] = {}
        ambiguous = set()
        for dependency_id in dependency_ids:
            for rank_id, component_id in self._rank_value_components.get(
                dependency_id, {}
            ).items():
                current = inherited.get(rank_id)
                if current is not None and current != component_id:
                    ambiguous.add(rank_id)
                else:
                    inherited[rank_id] = component_id
        for rank_id in ambiguous:
            inherited.pop(rank_id, None)
        if inherited:
            self._rank_value_components[task_id] = inherited
        if advance:
            self.previous = task_id
        return task_id

    def drain(self) -> Tuple[TaskSpec, ...]:
        """Return newly lowered tasks while retaining request-local identity.

        ``counter`` and ``previous`` deliberately survive the drain. A
        streaming executor can therefore lower one request a token/cohort at
        a time without changing task ids or dependency edges, while the
        full compiler can concatenate the same chunks into ``ScheduleIR``.
        """

        tasks = tuple(self.tasks)
        referenced = {
            dependency
            for task in tasks
            for dependency in task.dependencies
        }
        live_task_ids = {
            task.task_id for task in tasks if task.task_id not in referenced
        }
        if self.previous is not None:
            live_task_ids.add(self.previous)
        self._rank_value_components = {
            task_id: values
            for task_id, values in self._rank_value_components.items()
            if task_id in live_task_ids
        }
        self.tasks.clear()
        return tasks


def _task_segment_namespace_value(
    value: str,
    *,
    source_request_id: str,
    source_prefix: str,
    request_id: str,
    prefix: str,
) -> str:
    """Rebind one compiler-generated namespace without touching user ids."""

    for separator in (".", ":"):
        qualified_prefix = source_request_id + separator + source_prefix
        if qualified_prefix in value:
            # Temporary staged-allocation ids prepend a runtime namespace;
            # weight invocation ids use the colon form.  Replace this exact
            # compiler-generated pair without rewriting coincidental model or
            # user identifiers elsewhere in the value.
            return value.replace(
                qualified_prefix,
                request_id + separator + prefix,
            )
    if source_prefix in value:
        return value.replace(source_prefix, prefix)
    for separator in (".", ":"):
        source_start = source_request_id + separator
        if value.startswith(source_start):
            rebound = request_id + separator + value[len(source_start) :]
            # Invocation/allocation ids commonly contain both the outer
            # request namespace and the operation prefix after a colon.
            return rebound.replace(source_prefix, prefix)
    return value


_TASK_SEGMENT_ATOMIC_METADATA_TYPES = {
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
}


def _task_segment_metadata_key_sort_key(key: object) -> Tuple[object, ...]:
    """Return a deterministic order for value-like custom Mapping keys."""

    key_type = type(key)
    if key_type is type(None):
        return (0,)
    if key_type is bool:
        return (1, bool(key))
    if key_type is int:
        return (2, int(key))
    if key_type is float:
        return (3, float(key).hex())
    if key_type is complex:
        value = complex(key)
        return (4, value.real.hex(), value.imag.hex())
    if key_type is str:
        return (5, key)
    if key_type is bytes:
        return (6, key)
    raise ValueError("task segment metadata key is not value-like")


_TASK_SEGMENT_TASK_SPEC_FIELDS = (
    "task_id",
    "request_id",
    "name",
    "category",
    "dependencies",
    "demands",
    "earliest_start_ns",
    "marker",
    "token_index",
    "metadata",
)
_TASK_SEGMENT_TRUSTED_TASK_SPEC_LAYOUT = (
    tuple(TaskSpec.__dataclass_fields__) == _TASK_SEGMENT_TASK_SPEC_FIELDS
)


def _task_segment_trusted_task_spec(
    template: TaskSpec,
    *,
    task_id: Optional[str] = None,
    request_id: Optional[str] = None,
    name: Optional[str] = None,
    dependencies: Optional[Tuple[str, ...]] = None,
    demands: Optional[Tuple[ResourceDemand, ...]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> TaskSpec:
    """Instantiate a capture-validated planner task without revalidating it.

    Segment templates originate only from already constructed ``TaskSpec``
    values.  Replay changes planner-owned non-empty ids/names, compiled
    dependencies, and an isolated metadata mapping; demands and timing fields
    remain the previously validated objects.  Keep a dataclass-layout guard so
    a future ``TaskSpec`` schema change automatically falls back to its public
    constructor and validation path.
    """

    rebound_task_id = template.task_id if task_id is None else task_id
    rebound_request_id = template.request_id if request_id is None else request_id
    rebound_name = template.name if name is None else name
    rebound_dependencies = (
        template.dependencies if dependencies is None else dependencies
    )
    rebound_demands = template.demands if demands is None else demands
    rebound_metadata = template.metadata if metadata is None else metadata
    if not _TASK_SEGMENT_TRUSTED_TASK_SPEC_LAYOUT:
        return TaskSpec(
            task_id=rebound_task_id,
            request_id=rebound_request_id,
            name=rebound_name,
            category=template.category,
            dependencies=rebound_dependencies,
            demands=rebound_demands,
            earliest_start_ns=template.earliest_start_ns,
            marker=template.marker,
            token_index=template.token_index,
            metadata=rebound_metadata,
        )
    rebound = object.__new__(TaskSpec)
    object.__setattr__(rebound, "task_id", rebound_task_id)
    object.__setattr__(rebound, "request_id", rebound_request_id)
    object.__setattr__(rebound, "name", rebound_name)
    object.__setattr__(rebound, "category", template.category)
    object.__setattr__(rebound, "dependencies", rebound_dependencies)
    object.__setattr__(rebound, "demands", rebound_demands)
    object.__setattr__(rebound, "earliest_start_ns", template.earliest_start_ns)
    object.__setattr__(rebound, "marker", template.marker)
    object.__setattr__(rebound, "token_index", template.token_index)
    object.__setattr__(rebound, "metadata", rebound_metadata)
    return rebound


@dataclass(frozen=True)
class _FusedAttentionTaskReplayPayload:
    """Exact cost-model input retained for one fused-attention phase."""

    workload: FusedAttentionWorkload
    rank: LogicalRank
    layer_id: str
    phase_index: int
    phase_name: str
    fusion_targets: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class _KVReadTaskReplayPayload:
    """Exact rank/layer geometry retained for one local KV bookkeeping task."""

    layer: LayerSpec
    rank: LogicalRank
    tp_degree: int


@dataclass(frozen=True)
class _DynamicAttentionCostTaskReplayPayload:
    """One phase of the exact unfused attention fallback cost graph."""

    role: str
    workload: object
    fused_workload: FusedAttentionWorkload
    fusion_targets: Tuple[Tuple[str, str], ...]
    layer: LayerSpec
    rank: LogicalRank
    tp_degree: int
    token_batch: int
    score_heads: int
    target_component_id: str = ""
    operator_class: OperatorClass = OperatorClass.GEMM
    phase_index: int = -1
    phase_name: str = ""
    cost_model_suffix: Tuple[Tuple[str, object], ...] = ()


@dataclass(frozen=True)
class _TaskSegmentDynamicReplayContext:
    """The two values allowed to vary in a compiled serving invocation."""

    scenario: ScenarioConfig
    context_tokens: int
    kv_read_tokens: int


def _task_segment_clone_metadata(
    value: object,
    memo: Optional[Dict[int, object]] = None,
) -> object:
    """Clone JSON-like task metadata without generic ``deepcopy`` overhead.

    Segment metadata is dominated by immutable scalars and short tuples, but
    contains mutable diagnostic mappings that must not leak between capture
    and replay.  The generic copier performs dispatch and memo bookkeeping for
    every scalar.  This copier memoizes containers only, preserves aliases
    between mutable subtrees, and shares deeply atomic tuples safely.  Unknown
    object types fail back to their ordinary copy protocol via ``copy`` only
    when such a value actually appears.
    """

    value_type = type(value)
    if value_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
        return value
    if memo is None:
        memo = {}
    object_id = id(value)
    cached = memo.get(object_id)
    if cached is not None:
        return cached
    if value_type is dict:
        cloned_dict: Dict[object, object] = {}
        memo[object_id] = cloned_dict
        for key, item in value.items():  # type: ignore[union-attr]
            cloned_dict[
                _task_segment_clone_metadata(key, memo)
            ] = _task_segment_clone_metadata(item, memo)
        return cloned_dict
    if value_type is list:
        cloned_list: List[object] = []
        memo[object_id] = cloned_list
        cloned_list.extend(
            _task_segment_clone_metadata(item, memo)
            for item in value  # type: ignore[union-attr]
        )
        return cloned_list
    if value_type is tuple:
        if all(
            type(item) in _TASK_SEGMENT_ATOMIC_METADATA_TYPES
            for item in value  # type: ignore[union-attr]
        ):
            memo[object_id] = value
            return value
        cloned_tuple = tuple(
            _task_segment_clone_metadata(item, memo)
            for item in value  # type: ignore[union-attr]
        )
        memo[object_id] = cloned_tuple
        return cloned_tuple
    if value_type is set:
        cloned_set: set[object] = set()
        memo[object_id] = cloned_set
        cloned_set.update(
            _task_segment_clone_metadata(item, memo)
            for item in value  # type: ignore[union-attr]
        )
        return cloned_set
    if value_type is frozenset:
        cloned_frozenset = frozenset(
            _task_segment_clone_metadata(item, memo)
            for item in value  # type: ignore[union-attr]
        )
        memo[object_id] = cloned_frozenset
        return cloned_frozenset

    # TaskSpec metadata is intentionally value-like.  Retaining an explicit
    # fallback keeps third-party/custom metadata safe without charging the
    # common scalar path for generic copy dispatch.
    from copy import deepcopy

    cloned = deepcopy(value, memo)
    # Clone plans are compiled before replay and carry template identities.
    # Preserve aliases even when deepcopy internally memoizes by the live
    # replay object's identity instead.
    memo[object_id] = cloned
    return cloned


def _task_segment_string_needs_namespace(
    value: str,
    source_request_id: str,
    source_prefix: str,
    memo: Dict[str, bool],
) -> bool:
    """Return a memoized exact namespace-rebind decision for one string."""

    cached = memo.get(value)
    if cached is not None:
        return cached
    source_request_id_length = len(source_request_id)
    candidate = source_prefix in value or (
        len(value) > source_request_id_length
        and value.startswith(source_request_id)
        and value[source_request_id_length] in ".:"
    )
    required = bool(
        candidate
        and _task_segment_namespace_value(
            value,
            source_request_id=source_request_id,
            source_prefix=source_prefix,
            request_id="__rebound_request__",
            prefix="__rebound_prefix__",
        )
        != value
    )
    memo[value] = required
    return required


def _task_segment_capture_metadata(
    value: object,
    *,
    source_request_id: str,
    source_prefix: str,
    memo: Optional[Dict[int, object]] = None,
    plan_memo: Optional[Dict[int, Optional[object]]] = None,
    namespace_memo: Optional[Dict[str, bool]] = None,
    active: Optional[Set[int]] = None,
) -> Tuple[object, Optional[object]]:
    """Clone capture metadata and compile its replay plan in one traversal."""

    value_type = type(value)
    if namespace_memo is None:
        namespace_memo = {}
    if value_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
        plan = None
        if (
            value_type is str
            and _task_segment_string_needs_namespace(
                value,  # type: ignore[arg-type]
                source_request_id,
                source_prefix,
                namespace_memo,
            )
        ):
            plan = ("namespace", id(value), ())
        return value, plan

    if memo is None:
        memo = {}
    if plan_memo is None:
        plan_memo = {}
    if active is None:
        active = set()
    object_id = id(value)
    if object_id in active:
        raise ValueError("cyclic task segment metadata is not cacheable")
    if object_id in memo:
        return memo[object_id], plan_memo[object_id]
    if object_id in plan_memo:
        plan = plan_memo[object_id]
        cloned = _task_segment_clone_metadata_with_plan(
            value,
            plan,
            memo,
            source_request_id=source_request_id,
            source_prefix=source_prefix,
            request_id=source_request_id,
            prefix=source_prefix,
        )
        return cloned, plan

    if value_type is dict:
        active.add(object_id)
        cloned_dict: Dict[object, object] = {}
        memo[object_id] = cloned_dict
        children = []
        try:
            for key, item in value.items():  # type: ignore[union-attr]
                if type(key) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    raise ValueError(
                        "task segment metadata key is not value-like"
                    )
                if (
                    isinstance(key, str)
                    and _task_segment_string_needs_namespace(
                        key,
                        source_request_id,
                        source_prefix,
                        namespace_memo,
                    )
                ):
                    raise ValueError(
                        "task segment metadata key contains a namespace"
                    )
                item_type = type(item)
                if item_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    cloned_item = item
                    child_plan = None
                    if (
                        item_type is str
                        and _task_segment_string_needs_namespace(
                            item,
                            source_request_id,
                            source_prefix,
                            namespace_memo,
                        )
                    ):
                        child_plan = ("namespace", id(item), ())
                else:
                    cloned_item, child_plan = _task_segment_capture_metadata(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        memo=memo,
                        plan_memo=plan_memo,
                        namespace_memo=namespace_memo,
                        active=active,
                    )
                cloned_dict[key] = cloned_item
                if child_plan is not None:
                    children.append((key, child_plan))
        finally:
            active.remove(object_id)
        plan = ("dict", object_id, tuple(children))
        plan_memo[object_id] = plan
        return cloned_dict, plan

    if isinstance(value, Mapping):
        active.add(object_id)
        cloned_dict: Dict[object, object] = {}
        memo[object_id] = cloned_dict
        children = []
        try:
            items = tuple(value.items())
            items = tuple(
                sorted(
                    items,
                    key=lambda pair: _task_segment_metadata_key_sort_key(
                        pair[0]
                    ),
                )
            )
            for key, item in items:
                if type(key) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    raise ValueError(
                        "task segment metadata key is not value-like"
                    )
                if (
                    isinstance(key, str)
                    and _task_segment_string_needs_namespace(
                        key,
                        source_request_id,
                        source_prefix,
                        namespace_memo,
                    )
                ):
                    raise ValueError(
                        "task segment metadata key contains a namespace"
                    )
                item_type = type(item)
                if item_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    cloned_item = item
                    child_plan = None
                    if (
                        item_type is str
                        and _task_segment_string_needs_namespace(
                            item,
                            source_request_id,
                            source_prefix,
                            namespace_memo,
                        )
                    ):
                        child_plan = ("namespace", id(item), ())
                else:
                    cloned_item, child_plan = _task_segment_capture_metadata(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        memo=memo,
                        plan_memo=plan_memo,
                        namespace_memo=namespace_memo,
                        active=active,
                    )
                cloned_dict[key] = cloned_item
                if child_plan is not None:
                    children.append((key, child_plan))
        finally:
            active.remove(object_id)
        plan = ("dict", object_id, tuple(children))
        plan_memo[object_id] = plan
        return cloned_dict, plan

    if value_type is list:
        active.add(object_id)
        cloned_list: List[object] = []
        memo[object_id] = cloned_list
        children = []
        try:
            for index, item in enumerate(value):  # type: ignore[arg-type]
                item_type = type(item)
                if item_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    cloned_item = item
                    child_plan = None
                    if (
                        item_type is str
                        and _task_segment_string_needs_namespace(
                            item,
                            source_request_id,
                            source_prefix,
                            namespace_memo,
                        )
                    ):
                        child_plan = ("namespace", id(item), ())
                else:
                    cloned_item, child_plan = _task_segment_capture_metadata(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        memo=memo,
                        plan_memo=plan_memo,
                        namespace_memo=namespace_memo,
                        active=active,
                    )
                cloned_list.append(cloned_item)
                if child_plan is not None:
                    children.append((index, child_plan))
        finally:
            active.remove(object_id)
        plan = ("list", object_id, tuple(children))
        plan_memo[object_id] = plan
        return cloned_list, plan

    if value_type is tuple:
        active.add(object_id)
        cloned_items = []
        children = []
        try:
            for index, item in enumerate(value):  # type: ignore[arg-type]
                item_type = type(item)
                if item_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    cloned_item = item
                    child_plan = None
                    if (
                        item_type is str
                        and _task_segment_string_needs_namespace(
                            item,
                            source_request_id,
                            source_prefix,
                            namespace_memo,
                        )
                    ):
                        child_plan = ("namespace", id(item), ())
                else:
                    cloned_item, child_plan = _task_segment_capture_metadata(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        memo=memo,
                        plan_memo=plan_memo,
                        namespace_memo=namespace_memo,
                        active=active,
                    )
                cloned_items.append(cloned_item)
                if child_plan is not None:
                    children.append((index, child_plan))
        finally:
            active.remove(object_id)
        if not children and all(
            cloned is original
            for cloned, original in zip(cloned_items, value)  # type: ignore[arg-type]
        ):
            memo[object_id] = value
            plan_memo[object_id] = None
            return value, None
        cloned_tuple = tuple(cloned_items)
        memo[object_id] = cloned_tuple
        plan = ("tuple", object_id, tuple(children))
        plan_memo[object_id] = plan
        return cloned_tuple, plan

    if value_type in {set, frozenset}:
        items = tuple(value)  # type: ignore[arg-type]
        if any(
            isinstance(item, str)
            and _task_segment_string_needs_namespace(
                item,
                source_request_id,
                source_prefix,
                namespace_memo,
            )
            for item in items
        ):
            raise ValueError("task segment set metadata contains a namespace")
        if any(
            type(item) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES
            for item in items
        ):
            raise ValueError(
                "non-atomic task segment set metadata is not cacheable"
            )
        if value_type is set:
            cloned_set = set(value)  # type: ignore[arg-type]
            memo[object_id] = cloned_set
            plan = ("set", object_id, ())
            plan_memo[object_id] = plan
            return cloned_set, plan
        memo[object_id] = value
        plan_memo[object_id] = None
        return value, None

    from copy import deepcopy

    cloned = deepcopy(value, memo)
    memo[object_id] = cloned
    plan = ("deepcopy", object_id, ())
    plan_memo[object_id] = plan
    return cloned, plan


def _task_segment_metadata_clone_plan(
    value: object,
    namespace_paths: Sequence[Sequence[object]] = (),
    *,
    source_request_id: Optional[str] = None,
    source_prefix: Optional[str] = None,
    _active: Optional[Set[int]] = None,
) -> Optional[object]:
    """Compile mutable branches and namespace replacements for one replay.

    Rebinding is part of the memoized clone traversal.  A later path-copy pass
    would split aliases when two metadata keys refer to the same subtree.
    """

    if namespace_paths and any(not path for path in namespace_paths):
        if not isinstance(value, str):  # pragma: no cover - capture invariant
            raise ValueError("task segment namespace path is not text")
        return ("namespace", id(value), ())

    value_type = type(value)
    if (
        value_type is str
        and source_request_id is not None
        and source_prefix is not None
        and (
            source_prefix in value  # type: ignore[operator]
            or value.startswith(source_request_id + ".")  # type: ignore[union-attr]
            or value.startswith(source_request_id + ":")  # type: ignore[union-attr]
        )
        and _task_segment_namespace_value(
            value,  # type: ignore[arg-type]
            source_request_id=source_request_id,
            source_prefix=source_prefix,
            request_id="__rebound_request__",
            prefix="__rebound_prefix__",
        )
        != value
    ):
        return ("namespace", id(value), ())
    if value_type in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
        return None
    if value_type in {dict, list, tuple}:
        if _active is None:
            _active = set()
        object_id = id(value)
        if object_id in _active:
            raise ValueError("cyclic task segment metadata is not cacheable")
        _active.add(object_id)
    else:
        object_id = id(value)
    if value_type is dict:
        try:
            children = []
            for key, item in value.items():  # type: ignore[union-attr]
                if type(key) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    raise ValueError(
                        "task segment metadata key is not value-like"
                    )
                if (
                    isinstance(key, str)
                    and source_request_id is not None
                    and source_prefix is not None
                    and _task_segment_namespace_value(
                        key,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        request_id="__rebound_request__",
                        prefix="__rebound_prefix__",
                    )
                    != key
                ):
                    raise ValueError(
                        "task segment metadata key contains a namespace"
                    )
                child_paths = (
                    tuple(
                        path[1:]
                        for path in namespace_paths
                        if path and path[0] == key
                    )
                    if namespace_paths
                    else ()
                )
                child_plan = _task_segment_metadata_clone_plan(
                    item,
                    child_paths,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    _active=_active,
                )
                if child_plan is not None:
                    children.append((key, child_plan))
            return ("dict", object_id, tuple(children))
        finally:
            _active.remove(object_id)
    if value_type is list:
        try:
            children = []
            for index, item in enumerate(value):  # type: ignore[arg-type]
                child_paths = (
                    tuple(
                        path[1:]
                        for path in namespace_paths
                        if path and path[0] == index
                    )
                    if namespace_paths
                    else ()
                )
                child_plan = _task_segment_metadata_clone_plan(
                    item,
                    child_paths,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    _active=_active,
                )
                if child_plan is not None:
                    children.append((index, child_plan))
            return ("list", object_id, tuple(children))
        finally:
            _active.remove(object_id)
    if value_type is tuple:
        try:
            children = []
            for index, item in enumerate(value):  # type: ignore[arg-type]
                child_paths = (
                    tuple(
                        path[1:]
                        for path in namespace_paths
                        if path and path[0] == index
                    )
                    if namespace_paths
                    else ()
                )
                child_plan = _task_segment_metadata_clone_plan(
                    item,
                    child_paths,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    _active=_active,
                )
                if child_plan is not None:
                    children.append((index, child_plan))
            return (
                ("tuple", object_id, tuple(children))
                if children
                else None
            )
        finally:
            _active.remove(object_id)
    if value_type in {set, frozenset}:
        if namespace_paths:
            raise ValueError("task segment set metadata contains a namespace")
        items = tuple(value)  # type: ignore[arg-type]
        if source_request_id is not None and source_prefix is not None:
            if any(
                isinstance(item, str)
                and (
                    source_prefix in item
                    or item.startswith(source_request_id + ".")
                    or item.startswith(source_request_id + ":")
                )
                and _task_segment_namespace_value(
                    item,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    request_id="__rebound_request__",
                    prefix="__rebound_prefix__",
                )
                != item
                for item in items
            ):
                raise ValueError(
                    "task segment set metadata contains a namespace"
                )
            if any(
                type(item) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES
                for item in items
            ):
                raise ValueError(
                    "non-atomic task segment set metadata is not cacheable"
                )
        if value_type is set:
            if all(
                type(item) in _TASK_SEGMENT_ATOMIC_METADATA_TYPES
                for item in items
            ):
                return ("set", object_id, ())
            return ("deepcopy", object_id, ())
        if all(
            type(item) in _TASK_SEGMENT_ATOMIC_METADATA_TYPES
            for item in items
        ):
            return None
    return ("deepcopy", object_id, ())


def _task_segment_clone_metadata_with_plan(
    value: object,
    plan: Optional[object],
    memo: Optional[Dict[int, object]] = None,
    *,
    source_request_id: Optional[str] = None,
    source_prefix: Optional[str] = None,
    request_id: Optional[str] = None,
    prefix: Optional[str] = None,
    counter_delta: int = 0,
) -> object:
    """Execute a precompiled metadata clone plan with container memoization."""

    if plan is None:
        return value
    if memo is None:
        memo = {}
    return _task_segment_clone_metadata_plan_node(
        value,
        plan,
        memo,
        (
            source_request_id,
            source_prefix,
            request_id,
            prefix,
            counter_delta,
        ),
    )


def _task_segment_clone_metadata_plan_node(
    value: object,
    plan: object,
    memo: Dict[int, object],
    namespace_binding: Tuple[
        Optional[str], Optional[str], Optional[str], Optional[str], int
    ],
) -> object:
    """Run one clone-plan node with one shared namespace binding.

    Segment replay visits millions of short metadata branches in long-output
    schedules.  Passing the immutable rebinding payload as one positional tuple
    keeps that recursive hot path exact while avoiding repeated keyword binding
    and tuple construction at every child.
    """

    kind, object_id, children = plan  # type: ignore[misc]
    if object_id in memo:
        return memo[object_id]
    if kind == "namespace":
        (
            source_request_id,
            source_prefix,
            request_id,
            prefix,
            counter_delta,
        ) = namespace_binding
        if (
            not isinstance(value, str)
            or source_request_id is None
            or source_prefix is None
            or request_id is None
            or prefix is None
        ):  # pragma: no cover - capture/replay invariant
            raise AssertionError("cached namespace clone plan is invalid")
        cloned_namespace = _task_segment_counter_value(
            _task_segment_namespace_value(
                value,
                source_request_id=source_request_id,
                source_prefix=source_prefix,
                request_id=request_id,
                prefix=prefix,
            ),
            counter_delta,
        )
        memo[object_id] = cloned_namespace
        return cloned_namespace
    if kind == "dict":
        cloned_dict = dict(value)  # type: ignore[arg-type]
        memo[object_id] = cloned_dict
        for key, child_plan in children:
            cloned_dict[key] = _task_segment_clone_metadata_plan_node(
                value[key],  # type: ignore[index]
                child_plan,
                memo,
                namespace_binding,
            )
        return cloned_dict
    if kind == "list":
        cloned_list = list(value)  # type: ignore[arg-type]
        memo[object_id] = cloned_list
        for index, child_plan in children:
            cloned_list[index] = _task_segment_clone_metadata_plan_node(
                value[index],  # type: ignore[index]
                child_plan,
                memo,
                namespace_binding,
            )
        return cloned_list
    if kind == "tuple":
        cloned_items = list(value)  # type: ignore[arg-type]
        for index, child_plan in children:
            cloned_items[index] = _task_segment_clone_metadata_plan_node(
                value[index],  # type: ignore[index]
                child_plan,
                memo,
                namespace_binding,
            )
        cloned_tuple = tuple(cloned_items)
        memo[object_id] = cloned_tuple
        return cloned_tuple
    if kind == "set":
        cloned_set = set(value)  # type: ignore[arg-type]
        memo[object_id] = cloned_set
        return cloned_set
    from copy import deepcopy

    cloned = deepcopy(value, memo)
    memo[object_id] = cloned
    return cloned


def _task_segment_namespace_paths(
    value: object,
    *,
    source_request_id: str,
    source_prefix: str,
    path: Tuple[object, ...] = (),
    _active: Optional[Set[int]] = None,
) -> Tuple[Tuple[object, ...], ...]:
    """Locate only compiler namespace strings that a segment replay patches."""

    if isinstance(value, str):
        rebound = _task_segment_namespace_value(
            value,
            source_request_id=source_request_id,
            source_prefix=source_prefix,
            request_id="__rebound_request__",
            prefix="__rebound_prefix__",
        )
        return (path,) if rebound != value else ()
    if isinstance(value, (Mapping, tuple, list, set, frozenset)):
        if _active is None:
            _active = set()
        object_id = id(value)
        if object_id in _active:
            raise ValueError("cyclic task segment metadata is not cacheable")
        _active.add(object_id)
    else:
        object_id = None
    if isinstance(value, Mapping):
        paths: List[Tuple[object, ...]] = []
        try:
            for key, item in value.items():
                # Compiler metadata keys are schema names.  Custom mutable or
                # identity-bearing keys are outside the segment-cache contract.
                if type(key) not in _TASK_SEGMENT_ATOMIC_METADATA_TYPES:
                    raise ValueError(
                        "task segment metadata key is not value-like"
                    )
                if isinstance(key, str) and _task_segment_namespace_value(
                    key,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    request_id="__rebound_request__",
                    prefix="__rebound_prefix__",
                ) != key:
                    raise ValueError(
                        "task segment metadata key contains a namespace"
                    )
                paths.extend(
                    _task_segment_namespace_paths(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        path=path + (key,),
                        _active=_active,
                    )
                )
            return tuple(paths)
        finally:
            assert object_id is not None
            _active.remove(object_id)
    if isinstance(value, (tuple, list)):
        paths = []
        try:
            for index, item in enumerate(value):
                paths.extend(
                    _task_segment_namespace_paths(
                        item,
                        source_request_id=source_request_id,
                        source_prefix=source_prefix,
                        path=path + (index,),
                        _active=_active,
                    )
                )
            return tuple(paths)
        finally:
            assert object_id is not None
            _active.remove(object_id)
    if isinstance(value, (set, frozenset)):
        try:
            if any(
                _task_segment_namespace_paths(
                    item,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    _active=_active,
                )
                for item in value
            ):
                raise ValueError(
                    "task segment set metadata contains a namespace"
                )
            return ()
        finally:
            assert object_id is not None
            _active.remove(object_id)
    return ()


def _task_segment_counter_value(value: str, counter_delta: int) -> str:
    """Shift the TaskBuilder counter embedded in a staged allocation id."""

    prefix = "runtime.staged_weight."
    if not counter_delta or not value.startswith(prefix):
        return value
    counter_text, separator, remainder = value[len(prefix) :].partition(".")
    if not separator or not counter_text.isdigit():
        return value
    return "{}{:05d}.{}".format(
        prefix,
        int(counter_text) + counter_delta,
        remainder,
    )


def _task_segment_lean_root_namespace_keys(
    plan: Optional[object],
) -> Optional[Tuple[str, ...]]:
    """Compile the provider-private root-only namespace fast path.

    ``None`` deliberately means that replay must use the ordinary clone plan.
    The lean path never path-copies nested containers, so a namespace below the
    metadata root is an exact fail-closed boundary rather than an optimization
    opportunity.
    """

    if not isinstance(plan, tuple) or len(plan) != 3 or plan[0] != "dict":
        return None

    def contains_namespace(node: object) -> bool:
        if not isinstance(node, tuple) or len(node) != 3:
            return False
        kind, _object_id, children = node
        if kind == "namespace":
            return True
        return any(
            contains_namespace(child_plan)
            for _selector, child_plan in children
        )

    root_keys: List[str] = []
    for key, child_plan in plan[2]:
        if (
            isinstance(child_plan, tuple)
            and len(child_plan) == 3
            and child_plan[0] == "namespace"
        ):
            if not isinstance(key, str):
                return None
            root_keys.append(key)
            continue
        if contains_namespace(child_plan):
            return None
    return tuple(root_keys)


@dataclass(frozen=True)
class _TaskSegmentTemplate:
    """Exact context-independent TaskBuilder segment with rebound namespaces."""

    tasks: Tuple[TaskSpec, ...]
    source_request_id: str
    source_prefix: str
    source_counter_before: int
    source_dependencies: Tuple[str, ...]
    source_initial_previous: Optional[str]
    source_final_previous: Optional[str]
    source_initial_dma: Optional[str]
    source_final_dma: Optional[str]
    terminal_task_id: str
    dependency_refs: Tuple[Tuple[object, ...], ...]
    terminal_task_index: int
    source_final_previous_ref: Optional[object]
    source_final_dma_ref: Optional[object]
    rank_values: Tuple[Tuple[Tuple[int, str], ...], ...]
    metadata_clone_plans: Tuple[Optional[object], ...]
    lean_metadata_namespace_keys: Optional[Tuple[Tuple[str, ...], ...]]
    dynamic_payloads: Tuple[Optional[object], ...]
    source_phase: Optional[str]

    @classmethod
    def capture(
        cls,
        builder: _TaskBuilder,
        *,
        first_task_index: int,
        source_prefix: str,
        source_counter_before: int,
        source_dependencies: Sequence[str],
        source_initial_previous: Optional[str],
        source_initial_dma: Optional[str],
        terminal_task_id: str,
        source_phase: Optional[str] = None,
        required_dynamic_attention_invocations: int = 0,
    ) -> Optional["_TaskSegmentTemplate"]:
        source_tasks = tuple(builder.tasks[first_task_index:])
        if not source_tasks:
            return None
        dynamic_payloads = tuple(
            builder._task_segment_dynamic_payloads.get(task.task_id)
            for task in source_tasks
        )
        if required_dynamic_attention_invocations:
            dynamic_attention_payload_types = (
                _FusedAttentionTaskReplayPayload,
                _DynamicAttentionCostTaskReplayPayload,
            )
            if any(
                task.metadata.get("fusion_group") == "flash_attention"
                and not isinstance(payload, dynamic_attention_payload_types)
                for task, payload in zip(source_tasks, dynamic_payloads)
            ):
                # A split flash-attention placement can insert transfers whose
                # service demand and fusion working-set audit both depend on the
                # live context shape.  Such a transfer has no dynamic replay
                # payload, so admitting the segment would preserve stale bytes.
                return None
            kv_keys = {
                (str(payload.layer.layer_id), payload.rank.rank)
                for task, payload in zip(source_tasks, dynamic_payloads)
                if isinstance(payload, _KVReadTaskReplayPayload)
                and task.metadata.get("event_kind") == "kv_read"
                and not task.demands
                and task.metadata.get("resource_accounting")
                == "included_in_attention_kernel"
            }
            fused_keys = {
                (payload.layer_id, payload.rank.rank)
                for task, payload in zip(source_tasks, dynamic_payloads)
                if isinstance(payload, _FusedAttentionTaskReplayPayload)
                and task.metadata.get("event_kind") == "fused_attention"
                and payload.phase_name == "gpu_fused_attention"
            }
            unfused_roles: Dict[Tuple[str, int], Set[str]] = {}
            for payload in dynamic_payloads:
                if isinstance(payload, _DynamicAttentionCostTaskReplayPayload):
                    unfused_roles.setdefault(
                        (str(payload.layer.layer_id), payload.rank.rank),
                        set(),
                    ).add(payload.role)
            required_unfused_roles = {
                "qk",
                "qk_scale",
                "softmax_reduce",
                "softmax_normalize",
                "pv",
            }
            if len(kv_keys) != required_dynamic_attention_invocations:
                return None
            if any(
                key not in fused_keys
                and unfused_roles.get(key) != required_unfused_roles
                for key in kv_keys
            ):
                return None
            if (fused_keys | set(unfused_roles)) != kv_keys:
                return None
            if any(
                task.metadata.get("event_kind") == "kv_read_skipped"
                for task in source_tasks
            ):
                return None
        allowed_external_ids = {
            dependency for dependency in source_dependencies if dependency
        }
        if source_initial_previous:
            allowed_external_ids.add(source_initial_previous)
        if source_initial_dma:
            allowed_external_ids.add(source_initial_dma)

        task_index_by_id = {
            task.task_id: position for position, task in enumerate(source_tasks)
        }
        if len(task_index_by_id) != len(source_tasks):
            return None
        dependency_refs = []
        has_forward_dependency = False
        for position, task in enumerate(source_tasks):
            task_dependency_refs = []
            for dependency in task.dependencies:
                if dependency in task_index_by_id:
                    dependency_index = task_index_by_id[dependency]
                    task_dependency_refs.append(dependency_index)
                    has_forward_dependency |= dependency_index >= position
                elif dependency in allowed_external_ids:
                    task_dependency_refs.append(dependency)
                else:
                    return None
            dependency_refs.append(tuple(task_dependency_refs))
        if has_forward_dependency:
            # Late-bound input transfers can depend on later-created tasks.
            # Preserve their authored order, but retain the acyclic contract
            # that the ordered fast path previously guaranteed implicitly.
            try:
                TopologicalSorter({
                    position: tuple(ref for ref in refs if isinstance(ref, int))
                    for position, refs in enumerate(dependency_refs)
                }).prepare()
            except CycleError:
                return None
        terminal_task_index = task_index_by_id.get(terminal_task_id)
        if terminal_task_index is None:
            return None
        final_previous = builder.previous
        final_dma = builder._last_coherent_dma_task
        if (
            final_previous is not None
            and final_previous not in task_index_by_id
            and final_previous not in allowed_external_ids
        ):
            return None
        if (
            final_dma is not None
            and final_dma not in task_index_by_id
            and final_dma not in allowed_external_ids
        ):
            return None

        def compiled_state_ref(source: Optional[str]) -> Optional[object]:
            if source is None:
                return None
            if source in task_index_by_id:
                return task_index_by_id[source]
            return source

        source_request_id = builder.request.request_id
        try:
            captured_tasks = []
            metadata_clone_plans = []
            lean_metadata_namespace_keys = []
            lean_metadata_replay_safe = True
            metadata_plan_memo: Dict[int, Optional[object]] = {}
            metadata_namespace_memo: Dict[str, bool] = {}
            for task in source_tasks:
                cloned_metadata, clone_plan = _task_segment_capture_metadata(
                    task.metadata,
                    source_request_id=source_request_id,
                    source_prefix=source_prefix,
                    memo={},
                    plan_memo=metadata_plan_memo,
                    namespace_memo=metadata_namespace_memo,
                )
                captured_tasks.append(
                    _task_segment_trusted_task_spec(
                        task,
                        metadata=cloned_metadata,
                    )
                )
                metadata_clone_plans.append(clone_plan)
                root_namespace_keys = _task_segment_lean_root_namespace_keys(
                    clone_plan
                )
                if root_namespace_keys is None:
                    lean_metadata_replay_safe = False
                    lean_metadata_namespace_keys.append(())
                else:
                    lean_metadata_namespace_keys.append(root_namespace_keys)
            tasks = tuple(captured_tasks)
            metadata_clone_plans_tuple = tuple(metadata_clone_plans)
            lean_metadata_namespace_keys_tuple = (
                tuple(lean_metadata_namespace_keys)
                if lean_metadata_replay_safe
                else None
            )
        except Exception:
            # A valid uncached compilation must remain valid when metadata is
            # outside the value-like segment-cache contract.
            return None
        return cls(
            tasks=tasks,
            source_request_id=source_request_id,
            source_prefix=source_prefix,
            source_counter_before=source_counter_before,
            source_dependencies=tuple(source_dependencies),
            source_initial_previous=source_initial_previous,
            source_final_previous=final_previous,
            source_initial_dma=source_initial_dma,
            source_final_dma=final_dma,
            terminal_task_id=terminal_task_id,
            dependency_refs=tuple(dependency_refs),
            terminal_task_index=terminal_task_index,
            source_final_previous_ref=compiled_state_ref(final_previous),
            source_final_dma_ref=compiled_state_ref(final_dma),
            rank_values=tuple(
                tuple(
                    sorted(
                        builder._rank_value_components.get(
                            task.task_id, {}
                        ).items()
                    )
                )
                for task in tasks
            ),
            metadata_clone_plans=metadata_clone_plans_tuple,
            lean_metadata_namespace_keys=lean_metadata_namespace_keys_tuple,
            dynamic_payloads=dynamic_payloads,
            source_phase=source_phase,
        )

    def replay(
        self,
        builder: _TaskBuilder,
        *,
        prefix: str,
        dependencies: Sequence[str],
        metadata_overrides: Optional[Mapping[str, object]] = None,
        metadata_override_rules: Sequence[
            Tuple[str, object, Mapping[str, object]]
        ] = (),
        dynamic_context: Optional[_TaskSegmentDynamicReplayContext] = None,
    ) -> Optional[str]:
        if len(dependencies) != len(self.source_dependencies):
            return None
        if not (
            len(self.tasks)
            == len(self.dependency_refs)
            == len(self.rank_values)
            == len(self.metadata_clone_plans)
            == len(self.dynamic_payloads)
        ):
            return None
        active_context = (
            _active_compilation_context(dynamic_context.scenario)
            if dynamic_context is not None
            else None
        )
        lean_namespace_keys = self.lean_metadata_namespace_keys
        use_lean_metadata_replay = bool(
            dynamic_context is not None
            and active_context is not None
            and active_context._serving_invocation_lean_replay_capability
            is _SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY
            and lean_namespace_keys is not None
            and len(lean_namespace_keys) == len(self.tasks)
        )
        if use_lean_metadata_replay:
            assert lean_namespace_keys is not None
            use_lean_metadata_replay = all(
                isinstance(task.metadata.get(key), str)
                for task, root_keys in zip(self.tasks, lean_namespace_keys)
                for key in root_keys
            )
        if (
            self.terminal_task_index < 0
            or self.terminal_task_index >= len(self.tasks)
        ):
            return None
        dynamic_task_overrides: Mapping[
            int, Tuple[Tuple[ResourceDemand, ...], Mapping[str, object]]
        ] = {}
        if dynamic_context is not None:
            prepared_dynamic_overrides = _task_segment_dynamic_task_overrides(
                self.dynamic_payloads,
                dynamic_context,
            )
            if prepared_dynamic_overrides is None:
                return None
            dynamic_task_overrides = prepared_dynamic_overrides
        external_ids: Dict[str, str] = {}
        for source, current in zip(self.source_dependencies, dependencies):
            existing = external_ids.get(source)
            if existing is not None and existing != current:
                return None
            external_ids[source] = current

        current_previous = builder.previous
        if self.source_initial_previous is not None:
            existing = external_ids.get(self.source_initial_previous)
            if existing is not None and existing != current_previous:
                return None
            if current_previous is None:
                return None
            external_ids[self.source_initial_previous] = current_previous
        current_dma = builder._last_coherent_dma_task
        if self.source_initial_dma is not None:
            existing = external_ids.get(self.source_initial_dma)
            if existing is not None and existing != current_dma:
                return None
            if current_dma is None:
                return None
            external_ids[self.source_initial_dma] = current_dma
        elif current_dma is not None:
            return None

        request_id = builder.request.request_id
        counter_delta = builder.counter - self.source_counter_before
        rebound_source_phase = (
            _task_segment_namespace_value(
                self.source_phase,
                source_request_id=self.source_request_id,
                source_prefix=self.source_prefix,
                request_id=request_id,
                prefix=prefix,
            )
            if self.source_phase is not None
            else None
        )
        rebound_names = tuple(
            _task_segment_namespace_value(
                task.name,
                source_request_id=self.source_request_id,
                source_prefix=self.source_prefix,
                request_id=request_id,
                prefix=prefix,
            )
            for task in self.tasks
        )
        rebound_task_ids = tuple(
            "{}.{:05d}.{}".format(
                request_id,
                builder.counter + task_index + 1,
                _UNSAFE_TASK_NAME_PATTERN.sub("-", name),
            )
            for task_index, name in enumerate(rebound_names)
        )
        for task_index, (task, dependency_refs, rank_values, clone_plan) in enumerate(zip(
            self.tasks,
            self.dependency_refs,
            self.rank_values,
            self.metadata_clone_plans,
        )):
            name = rebound_names[task_index]
            if use_lean_metadata_replay:
                assert lean_namespace_keys is not None
                replay_metadata = dict(task.metadata)
                for key in lean_namespace_keys[task_index]:
                    source_value = replay_metadata[key]
                    assert isinstance(source_value, str)
                    replay_metadata[key] = _task_segment_counter_value(
                        _task_segment_namespace_value(
                            source_value,
                            source_request_id=self.source_request_id,
                            source_prefix=self.source_prefix,
                            request_id=request_id,
                            prefix=prefix,
                        ),
                        counter_delta,
                    )
            else:
                metadata: object = _task_segment_clone_metadata_with_plan(
                    task.metadata,
                    clone_plan,
                    source_request_id=self.source_request_id,
                    source_prefix=self.source_prefix,
                    request_id=request_id,
                    prefix=prefix,
                    counter_delta=counter_delta,
                )
                metadata_source = (
                    metadata if isinstance(metadata, Mapping) else task.metadata
                )
                # A normal TaskSpec metadata root is a dict and its capture plan
                # always materializes a fresh root.  Keep that isolated root
                # instead of copying it once more in TaskBuilder.add, while
                # retaining the historical dict-normalization fallback for
                # custom mappings.
                replay_metadata = (
                    metadata_source
                    if type(metadata_source) is dict
                    else dict(metadata_source or {})
                )
            if metadata_overrides:
                for key, value in metadata_overrides.items():
                    if key not in replay_metadata:
                        continue
                    if (
                        key == "phase"
                        and self.source_phase is not None
                        and replay_metadata[key] != self.source_phase
                        and replay_metadata[key] != rebound_source_phase
                    ):
                        continue
                    replay_metadata[key] = value
            if metadata_override_rules:
                for selector_key, selector_value, rule_overrides in (
                    metadata_override_rules
                ):
                    if replay_metadata.get(selector_key) != selector_value:
                        continue
                    for key, value in rule_overrides.items():
                        if key in replay_metadata:
                            replay_metadata[key] = value
            dynamic_override = dynamic_task_overrides.get(task_index)
            if dynamic_override is not None:
                _dynamic_demands, dynamic_metadata = dynamic_override
                replay_metadata.update(dynamic_metadata)
            try:
                rebound_dependencies = tuple(
                    (
                        rebound_task_ids[dependency_ref]
                        if isinstance(dependency_ref, int)
                        else external_ids[dependency_ref]
                    )
                    for dependency_ref in dependency_refs
                )
            except (IndexError, KeyError, TypeError):
                return None
            builder.counter += 1
            task_id = rebound_task_ids[task_index]
            builder.tasks.append(
                _task_segment_trusted_task_spec(
                    task,
                    task_id=task_id,
                    request_id=request_id,
                    name=name,
                    dependencies=tuple(
                        dependency
                        for dependency in rebound_dependencies
                        if dependency
                    ),
                    demands=(
                        task.demands
                        if dynamic_override is None
                        else dynamic_override[0]
                    ),
                    metadata=replay_metadata,
                )
            )
            if rank_values:
                builder._rank_value_components[task_id] = dict(rank_values)
            else:
                builder._rank_value_components.pop(task_id, None)

        def rebound_state_ref(source_ref: Optional[object]) -> Optional[str]:
            if source_ref is None:
                return None
            if isinstance(source_ref, int):
                return rebound_task_ids[source_ref]
            return external_ids[source_ref]

        try:
            builder.previous = rebound_state_ref(self.source_final_previous_ref)
            builder._last_coherent_dma_task = rebound_state_ref(
                self.source_final_dma_ref
            )
            return rebound_task_ids[self.terminal_task_index]
        except (IndexError, KeyError, TypeError):
            return None


def _task_segment_cache(
    context: CompilationContext,
    key: Tuple[object, ...],
) -> Optional["OrderedDict[Hashable, _TaskSegmentTemplate]"]:
    if context.leaf_cache_entries <= 0:
        return None
    cache_value = context.invariant(key, OrderedDict)
    if not isinstance(cache_value, OrderedDict):  # pragma: no cover - defensive
        return None
    return cache_value


def _task_segment_cache_get(
    cache: "OrderedDict[Hashable, _TaskSegmentTemplate]",
    key: Hashable,
) -> Optional[_TaskSegmentTemplate]:
    try:
        template = cache[key]
    except KeyError:
        return None
    if not isinstance(template, _TaskSegmentTemplate):  # pragma: no cover
        return None
    cache.move_to_end(key)
    return template


def _task_segment_cache_put(
    context: CompilationContext,
    cache: "OrderedDict[Hashable, _TaskSegmentTemplate]",
    key: Hashable,
    template: _TaskSegmentTemplate,
) -> None:
    if context.leaf_cache_entries <= 0:
        return
    cache[key] = template
    cache.move_to_end(key)
    while len(cache) > context.leaf_cache_entries:
        cache.popitem(last=False)


@dataclass(frozen=True)
class RequestTaskChunk:
    """One bounded lowering segment for a static request task graph."""

    tasks: Tuple[TaskSpec, ...]
    terminal_task_id: str
    final: bool = False


@dataclass(frozen=True)
class StreamingScheduleIR:
    """A static schedule whose request graphs are lowered on demand."""

    manifest: RunManifest
    scenario: ScenarioConfig
    requests: Sequence[RequestSpec]
    # The same physical execution lanes used by the online control-plane
    # runtime.  Keeping this on the IR prevents streaming execution from
    # silently falling back to single-lane resources.
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    resource_owners: Mapping[str, str] = field(default_factory=dict)


def _scenario_resource_capacities(scenario: ScenarioConfig) -> Mapping[str, int]:
    """Resolve shared execution lanes without importing control-plane at load time."""

    # control_plane_planner imports this module, so the import must remain
    # lazy to avoid a planner <-> control-plane cycle during package startup.
    from .control_plane import _execution_resource_capacities

    return _execution_resource_capacities(scenario)


def _scenario_resource_owners(scenario: ScenarioConfig) -> Mapping[str, str]:
    from .communication import declared_resource_owners

    return declared_resource_owners(scenario.hardware)


def compile_scenario(scenario: ScenarioConfig) -> ScheduleIR:
    """Compile one scenario inside a single exact, request-local context."""

    with _compilation_scope(scenario):
        return _compile_scenario_in_context(scenario)


def compile_streaming_scenario(scenario: ScenarioConfig) -> StreamingScheduleIR:
    """Validate a static scenario without materializing its complete graph."""

    with _compilation_scope(scenario):
        validation = validate_scenario(scenario)
        validation.raise_for_errors()
        manifest_assumptions = (
            scenario.assumptions + validation.warnings + validation.information
        )
        manifest_assumptions_zh, manifest_assumptions_en = localized_manifest_assumptions(
            manifest_assumptions,
            assumptions_en=(
                scenario.assumptions
                + validation.warnings_en
                + validation.information_en
            ),
        )
        manifest = RunManifest(
            schema_version=scenario.schema_version,
            run_id=stable_hash(scenario)[:16],
            random_seed=scenario.workload.random_seed,
            simulator_version=__version__,
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
            calibration_version=ANALYTICAL_MODEL_VERSION,
            evidence=EvidenceStatus.ANALYTICAL,
            assumptions=manifest_assumptions,
            assumptions_zh=manifest_assumptions_zh,
            assumptions_en=manifest_assumptions_en,
            metadata={
                "scenario_name": scenario.name,
                "execution_kernel": "unified_event_kernel",
                "schedule_materialization": "incremental",
            },
        )
        return StreamingScheduleIR(
            manifest=manifest,
            scenario=scenario,
            requests=materialize_requests(scenario),
            resource_capacities=_scenario_resource_capacities(scenario),
            resource_owners=_scenario_resource_owners(scenario),
        )


def _compile_scenario_in_context(scenario: ScenarioConfig) -> ScheduleIR:
    validation = validate_scenario(scenario)
    validation.raise_for_errors()
    manifest_assumptions = (
        scenario.assumptions + validation.warnings + validation.information
    )
    manifest_assumptions_zh, manifest_assumptions_en = localized_manifest_assumptions(
        manifest_assumptions,
        assumptions_en=(
            scenario.assumptions
            + validation.warnings_en
            + validation.information_en
        ),
    )
    manifest = RunManifest(
        schema_version=scenario.schema_version,
        run_id=stable_hash(scenario)[:16],
        random_seed=scenario.workload.random_seed,
        simulator_version=__version__,
        model_name=scenario.model.name,
        hardware_name=scenario.hardware.name,
        workload_name=scenario.workload.name,
        calibration_version=ANALYTICAL_MODEL_VERSION,
        evidence=EvidenceStatus.ANALYTICAL,
        assumptions=manifest_assumptions,
        assumptions_zh=manifest_assumptions_zh,
        assumptions_en=manifest_assumptions_en,
        metadata={"scenario_name": scenario.name},
    )
    tasks: List[TaskSpec] = []
    for request in materialize_requests(scenario):
        tasks.extend(_compile_parallel_request(scenario, request))
    iq_panel_contract = scenario.workload.metadata.get("llama_cpp_cpu_iq_panel_reuse")
    if isinstance(iq_panel_contract, Mapping) and iq_panel_contract.get("enabled") is True:
        manifest = replace(manifest, metadata={
            **manifest.metadata, "cpu_iq_panel_reuse": summarize_cpu_iq_panel_reuse(tasks),
        })
    return ScheduleIR(
        manifest=manifest,
        tasks=tuple(tasks),
        resource_capacities=_scenario_resource_capacities(scenario),
        resource_owners=_scenario_resource_owners(scenario),
    )


def iter_request_task_chunks(
    scenario: ScenarioConfig, request: RequestSpec
) -> Iterator[RequestTaskChunk]:
    """Yield exact request-local lowering chunks without retaining the graph.

    The yielded task sequence is identical to the corresponding slice of
    :func:`compile_scenario`. Chunk boundaries occur only after a
    prefill/decode/MTP commit unit, so future work depends on the advertised
    terminal task exactly as it does in the complete graph.
    """

    with _compilation_scope(scenario):
        yield from _iter_parallel_request_task_chunks(scenario, request)


def _mtp_configuration(scenario: ScenarioConfig) -> Optional[object]:
    policy = scenario.workload.mtp
    return policy if policy is not None and policy.enabled else None


def _mtp_expected_acceptance(
    mtp: object, proposed_tokens: int, mtp_round: int
) -> float:
    """Evaluate the explicitly selected MTP acceptance model.

    ``acceptance_trace`` is optional evidence for expected-prefix models and
    must not silently override ``acceptance_rate``.  Trace mode, conversely,
    consumes only its cyclic trace.  This defensive boundary mirrors the IR
    contract and keeps planner execution fail-closed if a mutated nested value
    bypasses frozen-dataclass construction.
    """

    candidate_count = int(proposed_tokens)
    if candidate_count <= 0:
        raise ValueError("MTP proposed_tokens must be positive")
    round_index = int(mtp_round)
    if round_index < 0:
        raise ValueError("MTP round must be non-negative")
    model = str(getattr(mtp, "acceptance_model", "")).strip().lower()
    if model in {"expected", "expected_prefix"}:
        raw_rate = getattr(mtp, "acceptance_rate", None)
        rate = 0.0 if raw_rate is None else float(raw_rate)
    elif model == "trace":
        trace = tuple(getattr(mtp, "acceptance_trace", ()) or ())
        if not trace:
            raise ValueError("MTP acceptance_model='trace' requires acceptance_trace")
        rate = float(trace[round_index % len(trace)])
    else:
        raise ValueError(
            "unsupported MTP acceptance_model {}; expected expected, "
            "expected_prefix, or trace".format(model or "<empty>")
        )
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("MTP acceptance value must be in [0, 1]")
    return expected_prefix_tokens(candidate_count, rate)


def _kv_configuration(scenario: ScenarioConfig) -> object:
    return scenario.placement.kv_policy


def _weight_storage_component(scenario: ScenarioConfig) -> Optional[str]:
    component_id = scenario.placement.tensor_to_component.get("model_weights")
    return str(component_id) if component_id else None


def _control_plane_decision_metadata(
    scenario: ScenarioConfig,
) -> Mapping[object, object]:
    context = _active_compilation_context(scenario)
    if context is not None:
        return context.control_plane_decision()
    return control_plane_decision(scenario)


def _canonical_weight_tensor_id(
    scenario: ScenarioConfig, tensor_id: str
) -> str:
    context = _active_compilation_context(scenario)
    if context is not None:
        return context.canonical_weight_tensor_id(tensor_id)
    aliases = _control_plane_decision_metadata(scenario).get(
        "logical_weight_aliases", {}
    )
    if not isinstance(aliases, Mapping):
        return str(tensor_id)
    current = str(tensor_id)
    visited = set()
    while current in aliases and current not in visited:
        visited.add(current)
        target = aliases[current]
        if target is None:
            break
        current = str(target)
    return current


def _mtp_descriptor_for_weight(
    scenario: ScenarioConfig, tensor_id: str
) -> Optional[MTPExecutionDescriptor]:
    requested = str(tensor_id)
    context = _active_compilation_context(scenario)
    if context is not None:
        return context.mtp_descriptor_for_weight(requested)
    return next(
        (
            descriptor
            for descriptor in _mtp_execution_descriptors(scenario)
            if descriptor.weight_tensor.tensor_id == requested
        ),
        None,
    )


def _weight_source_for_tensor(
    scenario: ScenarioConfig,
    tensor_id: Optional[str],
    target_component_id: Optional[str] = None,
    *,
    rank: Optional[LogicalRank] = None,
) -> Tuple[Optional[str], str, str]:
    """Resolve physical source, physical tensor, and logical tensor IDs.

    Detailed placements take precedence over the aggregate ``model_weights``
    backing.  Control-plane logical views still report the aggregate tensor as
    the physical read, while warm replicated CIM tensors resolve to the local
    replica.  An explicitly marked runtime copy uses its own placement while
    preserving the canonical file owner in access metadata.  Cold CIM
    placements are transient compute targets rather than physical sources and
    therefore resolve through their declared backing.
    """

    requested = str(tensor_id or "model_weights")
    canonical_owner = _canonical_weight_tensor_id(scenario, requested)
    decision = _control_plane_decision_metadata(scenario)
    details_raw = decision.get("weight_tensor_details", {})
    details_by_tensor = details_raw if isinstance(details_raw, Mapping) else {}
    requested_detail_raw = details_by_tensor.get(requested, {})
    requested_detail = (
        requested_detail_raw
        if isinstance(requested_detail_raw, Mapping)
        else {}
    )
    is_runtime_copy = (
        requested != canonical_owner
        and str(requested_detail.get("runtime_copy_of") or "")
        == canonical_owner
    )
    placement_tensor = requested if is_runtime_copy else canonical_owner
    detail_raw = details_by_tensor.get(placement_tensor, {})
    detail = detail_raw if isinstance(detail_raw, Mapping) else {}
    residency = str(detail.get("residency", ""))

    target = str(target_component_id) if target_component_id else None
    context = _active_compilation_context(scenario)
    if context is not None:
        selected_shard = context.select_rank_shard(
            placement_tensor,
            rank_id=rank.rank if rank is not None else None,
            target_component_id=target,
        )
    else:
        rank_shards_by_tensor_raw = decision.get("rank_weight_shards", {})
        if isinstance(rank_shards_by_tensor_raw, Mapping):
            rank_shards_raw = rank_shards_by_tensor_raw.get(
                placement_tensor,
                rank_shards_by_tensor_raw.get(requested, ()),
            )
        else:
            rank_shards_raw = ()
        rank_shards = (
            tuple(
                shard
                for shard in rank_shards_raw
                if isinstance(shard, Mapping)
            )
            if isinstance(rank_shards_raw, (list, tuple))
            else ()
        )
        selected_shard: Optional[Mapping[object, object]] = None
        if rank is not None:
            selected_shard = next(
                (
                    shard
                    for shard in rank_shards
                    if str(shard.get("rank_id", shard.get("rank", "")))
                    == str(rank.rank)
                    and (
                        not target
                        or str(shard.get("compute_component_id", "")) == target
                    )
                ),
                None,
            )
        if selected_shard is None and target:
            selected_shard = next(
                (
                    shard
                    for shard in rank_shards
                    if str(shard.get("compute_component_id", "")) == target
                ),
                None,
            )
    if selected_shard is not None and residency != "cold_cim_transient":
        storage = selected_shard.get(
            "storage_component_id", selected_shard.get("component_id")
        )
        if storage:
            return str(storage), placement_tensor, canonical_owner

    replicas_raw = detail.get("replica_component_ids", ())
    if not isinstance(replicas_raw, (list, tuple)):
        replica_map = decision.get("resident_cim_replicas", {})
        replicas_raw = (
            replica_map.get(placement_tensor, ())
            if isinstance(replica_map, Mapping)
            else ()
        )
    replicas = {
        str(component_id)
        for component_id in replicas_raw
        if component_id is not None
    }
    if (
        scenario.weights_resident
        and residency != "cold_cim_transient"
        and target
        and target in replicas
    ):
        return target, placement_tensor, canonical_owner

    aggregate_component = _weight_storage_component(scenario)
    backing_component = detail.get("backing_component_id")
    backing_tensor = str(detail.get("backing_tensor_id") or "model_weights")
    if residency == "cold_cim_transient":
        source = str(backing_component) if backing_component else aggregate_component
        return source, backing_tensor, canonical_owner

    mapped_component = scenario.placement.tensor_to_component.get(placement_tensor)
    if mapped_component is None and placement_tensor != requested:
        mapped_component = scenario.placement.tensor_to_component.get(requested)
    if mapped_component is not None:
        source = str(mapped_component)
        physical_tensor = (
            backing_tensor
            if backing_component is not None
            and str(backing_component) == source
            else placement_tensor
        )
        # A manual cold CIM mapping describes the transient destination.  If
        # aggregate backing is available, it remains the physical source.
        if not scenario.weights_resident and aggregate_component:
            component = _component_map(scenario).get(source)
            if component is not None and _is_cim(component):
                return aggregate_component, "model_weights", canonical_owner
        return source, physical_tensor, canonical_owner

    if backing_component is not None:
        return str(backing_component), backing_tensor, canonical_owner
    if aggregate_component:
        return aggregate_component, "model_weights", canonical_owner
    return None, canonical_owner, canonical_owner


@dataclass(frozen=True)
class _WeightBackingReadDecision:
    """Auditable per-GEMM lifecycle gate for a resolved weight source.

    Source resolution and lifecycle are deliberately separate: detailed tensor
    placement still wins over aggregate ``model_weights``, while
    ``weights_resident`` alone decides whether an HBF/SSD source is a preload
    backing or a per-use cold stream.  This object is immutable and scoped to
    one physical-rank GEMM invocation; no batch-level loaded set is involved.
    """

    lifecycle_mode: str
    source_kind: str
    source_is_offload_backing: bool
    source_is_remote: bool
    source_is_compute_local_backing: bool
    emit_source_transfer: bool
    gate_reason: str

    def audit_metadata(self, invocation_id: str) -> Dict[str, object]:
        return {
            "weight_lifecycle_mode": self.lifecycle_mode,
            "weight_read_invocation_id": invocation_id,
            "weight_source_kind": self.source_kind,
            "weight_source_is_offload_backing": self.source_is_offload_backing,
            "weight_source_is_compute_local_backing": (
                self.source_is_compute_local_backing
            ),
            "weight_source_transfer_emitted": self.emit_source_transfer,
            "weight_backing_read_emitted": (
                self.emit_source_transfer and self.source_is_offload_backing
            ),
            "weight_backing_read_gate": self.gate_reason,
        }


@dataclass(frozen=True)
class _HostGemmOffloadDecision:
    """Invocation-local execution override for one statically host GEMM."""

    placement_component_id: str
    execution_component_id: str
    applied: bool
    reason: str
    evidence: str
    physical_m: int
    minimum_m: int
    weight_owner_component_id: Optional[str]
    execution_backend: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    @property
    def audit_metadata(self) -> Dict[str, object]:
        override = self.execution_component_id != self.placement_component_id
        return {
            "placement_component": self.placement_component_id,
            "placement_component_id": self.placement_component_id,
            "execution_component": self.execution_component_id,
            "execution_component_id": self.execution_component_id,
            "execution_backend": self.execution_backend,
            "execution_override": override,
            "override": override,
            "host_gemm_offload_override": override,
            "host_gemm_offload_applied": self.applied,
            "host_gemm_offload_reason": self.reason,
            "host_gemm_offload_evidence": self.evidence,
            "host_gemm_offload_provenance": dict(self.provenance or {}),
            "host_gemm_offload_physical_m": self.physical_m,
            "host_gemm_offload_minimum_m": self.minimum_m,
            "physical_m": self.physical_m,
            "minimum_m": self.minimum_m,
            "host_gemm_offload_weight_owner": (
                self.weight_owner_component_id
            ),
            "weight_owner": self.weight_owner_component_id,
            "weight_owner_component_id": self.weight_owner_component_id,
            "temporary_weight_bytes": 0,
            "temporary_weight_read_only": True,
            "temporary_weight_residency": "fully_resident_after_h2d",
            "temporary_weight_release_semantics": "clean_discard",
            "temporary_weight_dirty_writeback": False,
            "clean_discard_semantics": "free_clean_discard_no_writeback",
        }


_HOST_RECURRENT_OFFLOAD_REQUIRED_OPS = frozenset(
    {
        "rms_norm",
        "l2_norm",
        "silu",
        "sigmoid",
        "softplus",
        "ssm_conv",
        "gated_delta_net",
        "gate",
    }
)


@dataclass(frozen=True)
class _HostRecurrentOffloadDecision:
    """Invocation-local placement for one host-owned recurrent subgraph."""

    placement_component_id: str
    execution_component_id: str
    state_owner_component_id: Optional[str]
    applied: bool
    reason: str
    evidence: str
    provenance: str
    physical_m: int
    minimum_m: int
    state_bytes: int
    supported_ops: Tuple[str, ...]

    @property
    def audit_metadata(self) -> Dict[str, object]:
        override = self.execution_component_id != self.placement_component_id
        return {
            "host_recurrent_offload_applied": self.applied,
            "host_recurrent_offload_reason": self.reason,
            "host_recurrent_offload_evidence": self.evidence,
            "host_recurrent_offload_provenance": self.provenance,
            "host_recurrent_offload_physical_m": self.physical_m,
            "host_recurrent_offload_minimum_m": self.minimum_m,
            "host_recurrent_offload_supported_ops": self.supported_ops,
            "host_recurrent_offload_state_bytes": self.state_bytes,
            "host_recurrent_offload_state_owner": (
                self.state_owner_component_id
            ),
            "host_recurrent_offload_placement_component": (
                self.placement_component_id
            ),
            "host_recurrent_offload_execution_component": (
                self.execution_component_id
            ),
            "host_recurrent_offload_execution_override": override,
        }


def _route_is_available(
    router: TopologyRouter,
    source_component_id: str,
    target_component_id: str,
    byte_count: int,
    *,
    routing_policy: str,
) -> bool:
    """Probe a required invocation route without mutating the schedule."""

    if byte_count <= 0 or source_component_id == target_component_id:
        return True
    try:
        router.route(
            source_component_id,
            target_component_id,
            byte_count,
            policy=routing_policy,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _host_gemm_offload_decision(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    workload: GemmWorkload,
    placement_component_id: str,
    activation_source_component_id: str,
    weight_owner_component_id: Optional[str],
    *,
    model_weight_read: bool,
    dynamic_rhs: bool,
) -> Optional[_HostGemmOffloadDecision]:
    """Return an auditable GPU override, or ``None`` for legacy profiles.

    The capability is declared on the rank GPU rather than on static
    placement.  A missing capability intentionally returns ``None`` so old
    scenarios retain their exact task metadata and schedule.
    """

    try:
        gpu_profile = _resolve_component_profile(
            scenario, rank.component_id, GPUProfile
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    capability = gpu_profile.host_gemm_offload
    if capability is None:
        return None

    minimum_m = int(capability.minimum_m)
    evidence = str(capability.evidence)
    execution_component_id = placement_component_id
    applied = False
    placement = _component(scenario, placement_component_id)
    weight_owner = (
        str(weight_owner_component_id)
        if weight_owner_component_id is not None
        else None
    )

    if _kind(placement) != "cpu":
        reason = "placement_not_cpu"
    elif scenario.llama_cpp_config is not None and not scenario.llama_cpp_config.op_offload:
        reason = "runtime_op_offload_disabled"
    elif not model_weight_read or dynamic_rhs:
        reason = "dynamic_rhs_not_model_weight"
    elif workload.m < minimum_m:
        reason = "physical_m_below_minimum"
    elif not capability.supports_workload_format(workload):
        reason = "weight_format_not_source_supported"
    elif not weight_owner:
        reason = "weight_owner_missing"
    else:
        owner_component = _component_map(scenario).get(weight_owner)
        owner_kind = (
            _kind(owner_component) if owner_component is not None else ""
        )
        host_cpu_local = owner_kind in {"cpu", "host_memory"} and (
            _weight_source_is_compute_local_backing(
                scenario,
                weight_owner,
                placement_component_id,
            )
        )
        if not host_cpu_local:
            reason = "weight_owner_not_host_cpu_local"
        else:
            try:
                _gpu_profiles(
                    scenario,
                    rank.component_id,
                    rank.memory_component_id,
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                reason = "gpu_hbm_profile_unavailable"
            else:
                weight_route = _route_is_available(
                    router,
                    weight_owner,
                    rank.component_id,
                    workload.weight_bytes,
                    routing_policy=plan.routing_policy,
                )
                activation_route = _route_is_available(
                    router,
                    activation_source_component_id,
                    rank.component_id,
                    workload.activation_bytes,
                    routing_policy=plan.routing_policy,
                )
                if not weight_route:
                    reason = "weight_route_unavailable"
                elif not activation_route:
                    reason = "activation_route_unavailable"
                else:
                    execution_component_id = rank.component_id
                    applied = True
                    reason = "eligible_host_model_weight_gemm"

    return _HostGemmOffloadDecision(
        placement_component_id=placement_component_id,
        execution_component_id=execution_component_id,
        applied=applied,
        reason=reason,
        evidence=evidence,
        physical_m=int(workload.m),
        minimum_m=minimum_m,
        weight_owner_component_id=weight_owner,
        execution_backend=("gpu" if applied else _kind(placement)),
        provenance=capability.provenance,
    )


def _weight_source_is_compute_local_backing(
    scenario: ScenarioConfig,
    source_component_id: Optional[str],
    target_component_id: str,
) -> bool:
    """Return whether the compute roofline already owns the source traffic."""

    source = str(source_component_id) if source_component_id else ""
    target = str(target_component_id)
    if not source:
        return False
    if source == target:
        return True
    target_component = _component_map(scenario).get(target)
    if target_component is None or _kind(target_component) != "cpu":
        return False
    attached_host_memory = _nearest_profile_component_id(
        scenario, target, "host_memory"
    )
    return bool(attached_host_memory and source == attached_host_memory)


def _weight_backing_read_gate(
    scenario: ScenarioConfig,
    source_component_id: Optional[str],
    target_component_id: str,
    *,
    model_weight_read: bool,
) -> _WeightBackingReadDecision:
    """Gate one resolved source read through the scenario weight lifecycle."""

    lifecycle_mode = (
        "preloaded_resident"
        if scenario.weights_resident
        else "cold_stream_per_use"
    )
    source = str(source_component_id) if source_component_id else ""
    source_component = _component_map(scenario).get(source)
    source_kind = (
        _kind(source_component) if source_component is not None else "missing"
    )
    source_is_offload = source_kind in OFFLOAD_STORAGE_COMPONENT_KINDS
    source_is_remote = bool(source and source != str(target_component_id))
    source_is_compute_local = _weight_source_is_compute_local_backing(
        scenario,
        source_component_id,
        target_component_id,
    )

    if not model_weight_read:
        emit_source_transfer = False
        reason = "not_weight_bearing_gemm"
    elif not source:
        emit_source_transfer = False
        reason = "cold_missing_backing" if not scenario.weights_resident else "resident_implicit_local"
    elif source_is_compute_local:
        emit_source_transfer = False
        reason = (
            "source_is_compute_local"
            if not source_is_remote
            else "source_is_cpu_attached_host_memory"
        )
    elif scenario.weights_resident and source_is_offload:
        emit_source_transfer = False
        reason = "preloaded_offload_backing_suppressed"
    else:
        emit_source_transfer = True
        reason = (
            "cold_stream_per_use_emitted"
            if not scenario.weights_resident
            else "resident_active_memory_route_emitted"
        )
    return _WeightBackingReadDecision(
        lifecycle_mode=lifecycle_mode,
        source_kind=source_kind,
        source_is_offload_backing=source_is_offload,
        source_is_remote=source_is_remote,
        source_is_compute_local_backing=source_is_compute_local,
        emit_source_transfer=emit_source_transfer,
        gate_reason=reason,
    )


def _weight_transfer_metadata(
    source_tensor: str,
    logical_tensor: str,
    source_component: str,
    target_component: str,
) -> Dict[str, object]:
    return {
        "event_kind": "model_weight_read",
        "tensor": source_tensor,
        "logical_weight_tensor": logical_tensor,
        "weight_source_component": source_component,
        "weight_target_component": target_component,
    }


def _logical_weight_tensor_bytes(
    scenario: ScenarioConfig, tensor_id: str
) -> int:
    canonical = _canonical_weight_tensor_id(scenario, tensor_id)
    mtp_descriptor = _mtp_descriptor_for_weight(scenario, canonical)
    if mtp_descriptor is not None:
        return max(0, int(mtp_descriptor.weight_bytes))
    decision = _control_plane_decision_metadata(scenario)
    details_raw = decision.get("weight_tensor_details", {})
    if isinstance(details_raw, Mapping):
        detail_raw = details_raw.get(canonical, {})
        if isinstance(detail_raw, Mapping) and detail_raw.get("logical_bytes") is not None:
            try:
                return max(0, int(detail_raw["logical_bytes"]))
            except (TypeError, ValueError):
                return 0
    for metadata_key in ("derived_tensor_bytes", "logical_weight_views"):
        values = decision.get(metadata_key, {})
        if isinstance(values, Mapping) and canonical in values:
            try:
                return max(0, int(values[canonical]))
            except (TypeError, ValueError):
                return 0
    try:
        return max(
            0,
            int(
                scenario.placement.tensor_bytes.get(
                    canonical,
                    scenario.placement.tensor_bytes.get(tensor_id, 0),
                )
            ),
        )
    except (TypeError, ValueError):
        return 0


def _weight_execution_targets(
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    tensor_id: str,
) -> Tuple[str, ...]:
    """Return every physical compute target that consumes a weight tensor."""

    canonical = _canonical_weight_tensor_id(scenario, tensor_id)
    if canonical in {"embedding_weights", "lm_head_weights"}:
        ranks = plan.tp_group(plan.pp_degree - 1, 0)
        targets = (
            _parallel_named_target(scenario, "lm_head", rank)
            for rank in ranks
        )
    else:
        mtp_descriptor = _mtp_descriptor_for_weight(scenario, canonical)
        if mtp_descriptor is not None:
            ranks = plan.tp_group(plan.pp_degree - 1, 0)
            consumer_ids = []
            if _canonical_weight_tensor_id(
                scenario, "lm_head_weights"
            ) == canonical:
                consumer_ids.append("lm_head")
            consumer_ids.append(mtp_descriptor.operator.operator_id)
            targets = (
                _parallel_named_target(scenario, consumer_id, rank)
                for rank in ranks
                for consumer_id in consumer_ids
            )
            return tuple(
                dict.fromkeys(str(target) for target in targets)
            )
        target_spec = None
        layer = None
        for candidate in _execution_layers(scenario):
            specifications = {
                "{}.attention_weights".format(candidate.layer_id): (
                    "attention",
                    "tp",
                ),
                "{}.linear_attention_weights".format(candidate.layer_id): (
                    "linear_attention",
                    "tp",
                ),
                "{}.mlp_weights".format(candidate.layer_id): ("mlp", "tp"),
                "{}.expert_weights".format(candidate.layer_id): (
                    "experts",
                    "all",
                ),
                "{}.shared_expert_weights".format(candidate.layer_id): (
                    "shared_expert",
                    "tp",
                ),
                "{}.router_weights".format(candidate.layer_id): (None, "tp"),
                "{}.shared_expert_gate_weights".format(candidate.layer_id): (
                    None,
                    "tp",
                ),
            }
            if canonical in specifications:
                layer = candidate
                target_spec = specifications[canonical]
                break
        if layer is None or target_spec is None:
            targets = (rank.component_id for rank in plan.ranks)
        else:
            group, rank_mode = target_spec
            stage = plan.stage_for_layer(layer)
            ranks = (
                plan.ranks_for_stage(stage)
                if rank_mode == "all"
                else plan.tp_group(stage, 0)
            )
            targets = (
                rank.component_id
                if group is None
                else _parallel_target(scenario, layer, group, rank)
                for rank in ranks
            )
    return tuple(dict.fromkeys(str(target) for target in targets))


def _configured_cim_targets(
    scenario: ScenarioConfig, plan: Optional[ParallelPlan] = None
) -> Tuple[str, ...]:
    components = _component_map(scenario)
    configured = {
        str(component_id)
        for component_id in scenario.placement.op_to_component.values()
        if component_id in components and _is_cim(components[component_id])
    }
    if configured and plan is not None:
        configured.update(
            str(rank.cim_component_id)
            for rank in plan.ranks
            if rank.cim_component_id
        )
    return tuple(sorted(configured))


def _weight_tensor_has_active_resident_placement(
    scenario: ScenarioConfig,
    tensor_id: str,
    weight_tensor_details: Mapping[object, object],
    rank_weight_shards: Mapping[object, object],
) -> bool:
    """Whether a detailed tensor accounts real active resident capacity.

    Aggregate logical views and cold CIM destinations are backing/transient
    descriptions, not rank-local resident allocations.  In particular, an
    HBF/SSD placement must never reduce the active-memory requirement.
    """

    canonical = _canonical_weight_tensor_id(scenario, str(tensor_id))
    detail_raw = weight_tensor_details.get(canonical, {})
    if not isinstance(detail_raw, Mapping):
        detail_raw = weight_tensor_details.get(str(tensor_id), {})
    detail = detail_raw if isinstance(detail_raw, Mapping) else {}
    if str(detail.get("residency", "")) in {
        "aggregate_backing_logical_view",
        "cold_cim_transient",
    }:
        return False

    component_ids = set()
    shards_raw = rank_weight_shards.get(canonical)
    if shards_raw is None:
        shards_raw = rank_weight_shards.get(str(tensor_id), ())
    if isinstance(shards_raw, (list, tuple)):
        component_ids.update(
            str(
                shard.get(
                    "storage_component_id",
                    shard.get("component_id", ""),
                )
            )
            for shard in shards_raw
            if isinstance(shard, Mapping)
        )
    mapped = scenario.placement.tensor_to_component.get(canonical)
    if mapped is None:
        mapped = scenario.placement.tensor_to_component.get(str(tensor_id))
    if mapped is not None:
        component_ids.add(str(mapped))

    components = _component_map(scenario)
    return any(
        component_id in components
        and _is_active_resident_weight_storage(components[component_id])
        for component_id in component_ids
        if component_id
    )


def _resident_weight_components(
    scenario: ScenarioConfig, plan: ParallelPlan
) -> Tuple[str, ...]:
    """Return unique physical stores participating in a warm resident plan."""

    components = _component_map(scenario)
    resident = {
        str(rank.memory_component_id or rank.component_id)
        for rank in plan.ranks
    }
    resident.update(_configured_cim_targets(scenario, plan))
    backing = _weight_storage_component(scenario)
    if (
        backing
        and backing in components
        and _is_active_resident_weight_storage(components[backing])
    ):
        resident.add(backing)
    for tensor_name, component_id in scenario.placement.tensor_to_component.items():
        component = components.get(component_id)
        if (
            "weight" in tensor_name.lower()
            and component is not None
            and _is_active_resident_weight_storage(component)
        ):
            resident.add(str(component_id))
    decision = _control_plane_decision_metadata(scenario)
    details_raw = decision.get("weight_tensor_details", {})
    if isinstance(details_raw, Mapping):
        for detail_raw in details_raw.values():
            if not isinstance(detail_raw, Mapping):
                continue
            resident.update(
                str(component_id)
                for component_id in detail_raw.get(
                    "replica_component_ids",
                    (),
                )
                if component_id
            )
    rank_shards_raw = decision.get("rank_weight_shards", {})
    if isinstance(rank_shards_raw, Mapping):
        for shards_raw in rank_shards_raw.values():
            if not isinstance(shards_raw, (list, tuple)):
                continue
            resident.update(
                str(shard.get("storage_component_id"))
                for shard in shards_raw
                if isinstance(shard, Mapping)
                and shard.get("storage_component_id")
            )
    return tuple(sorted(component_id for component_id in resident if component_id in components))


def _resident_weight_capacity(
    scenario: ScenarioConfig, component_id: str
) -> int:
    """Capacity usable by weights without double-counting declared weight tensors."""

    component = _component(scenario, component_id)
    physical_capacity = _component_weight_capacity(scenario, component)
    if physical_capacity <= 0:
        return 0
    non_weight_bytes = sum(
        int(byte_count)
        for tensor_name, byte_count in scenario.placement.tensor_bytes.items()
        if "weight" not in tensor_name.lower()
        and scenario.placement.tensor_to_component.get(tensor_name) == component_id
    )
    return max(0, physical_capacity - non_weight_bytes)


def _component_weight_capacity(
    scenario: ScenarioConfig, component: ComponentSpec
) -> int:
    capacity = int(component.capacity_bytes)
    if _is_cim(component):
        cim_profile = _resolve_component_profile(
            scenario,
            component, DigitalSramCimProfile
        )
        profile_capacity = int(cim_profile.weight_capacity_bytes)
        if capacity <= 0:
            return profile_capacity
        return min(capacity, profile_capacity)
    return max(0, capacity)


def _unallocated_component_capacity(
    scenario: ScenarioConfig, component_id: str
) -> int:
    component = _component(scenario, component_id)
    physical_capacity = _component_weight_capacity(scenario, component)
    if physical_capacity <= 0:
        return 0
    occupied = sum(
        int(byte_count)
        for tensor_name, byte_count in scenario.placement.tensor_bytes.items()
        if scenario.placement.tensor_to_component.get(tensor_name) == component_id
    )
    return max(0, physical_capacity - occupied)


def _estimated_cim_weight_bytes(
    scenario: ScenarioConfig,
    *,
    execution_view: Optional[ModelGraphExecutionView] = None,
) -> int:
    """Conservative fallback when warm CIM weights lack detailed tensors."""

    if execution_view is None:
        execution_view = _execution_view(scenario)
    layers = tuple(
        descriptor.layer for descriptor in execution_view.layer_instances
    )
    components = _component_map(scenario)
    total = 0
    for layer in layers:
        groups = [
            "linear_attention" if layer.is_linear_attention else "attention",
            "experts" if layer.is_moe else "mlp",
        ]
        if layer.has_shared_expert:
            groups.append("shared_expert")
        if any(
            target in components and _is_cim(components[target])
            for target in (_resolve_target(scenario, layer, group) for group in groups)
        ):
            total += int(layer.weight_bytes)
    lm_target = scenario.placement.op_to_component.get("lm_head")
    if lm_target in components and _is_cim(components[lm_target]):
        total += int(
            execution_view.output_weight_bytes
            or execution_view.embedding_weight_bytes
        )
    for descriptor in execution_view.mtp_descriptors:
        target = scenario.placement.op_to_component.get(
            descriptor.operator.operator_id
        )
        if target in components and _is_cim(components[target]):
            total += int(descriptor.weight_bytes)
    return min(_execution_view_declared_weight_bytes(execution_view), total)


def _physical_cim_replica_count(
    scenario: ScenarioConfig,
    tensor_name: str,
    cim_targets: Sequence[str],
    resident_cim_replicas: Mapping[object, object],
) -> int:
    targets = {
        str(scenario.placement.tensor_to_component.get(tensor_name, ""))
    }
    replicas = resident_cim_replicas.get(tensor_name, ())
    if isinstance(replicas, (list, tuple)):
        targets.update(str(component_id) for component_id in replicas)
    return len(targets.intersection(cim_targets))


def _add_join(
    builder: _TaskBuilder,
    name: str,
    dependencies: Sequence[str],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    return builder.add(
        name,
        TaskCategory.SYNCHRONIZATION,
        dependencies=tuple(dict.fromkeys(item for item in dependencies if item)),
        advance=False,
        metadata=metadata,
    )


def _communication_task_metadata(
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    payload = dict(metadata or {})
    operator_class = payload.get("operator_class")
    if operator_class and operator_class != OperatorClass.COMMUNICATION.value:
        payload.setdefault("source_operator_class", operator_class)
    payload["operator_class"] = OperatorClass.COMMUNICATION.value
    return payload


def _coherent_dma_engine_demands(
    scenario: ScenarioConfig,
    demands: Sequence[ResourceDemand],
) -> Tuple[Tuple[ResourceDemand, ...], str, float, bool]:
    """Occupy the configured DMA engine once without re-owning bulk bytes."""

    retained = tuple(demands)
    resource_id = scenario.host_orchestration_profile.dma_resource_id
    existing = tuple(
        demand for demand in retained if demand.resource_id == resource_id
    )
    if len(existing) > 1:  # pragma: no cover - coherent phases are unique
        raise ValueError(
            "coherent DMA phase declares its shared engine more than once"
        )
    service_ns = max(
        (demand.service_ns for demand in retained),
        default=0.0,
    )
    if service_ns <= 0.0:  # pragma: no cover - non-empty coherent invariant
        raise ValueError("coherent DMA phase has no positive physical service")
    if existing:
        existing_demand = existing[0]
        if (
            existing_demand.bytes_moved != 0
            or existing_demand.energy_pj != 0.0
            or existing_demand.work_units != 0.0
        ):
            raise ValueError(
                "configured DMA engine resource {} collides with a physical "
                "transfer demand; use a dedicated zero-accounting DMA "
                "resource id".format(resource_id)
            )
        normalized = tuple(
            ResourceDemand(resource_id=resource_id, service_ns=service_ns)
            if demand.resource_id == resource_id
            else demand
            for demand in retained
        )
        return normalized, resource_id, service_ns, False
    return (
        retained
        + (
            ResourceDemand(
                resource_id=resource_id,
                service_ns=service_ns,
            ),
        ),
        resource_id,
        service_ns,
        True,
    )


def _add_transfer_tasks(
    builder: _TaskBuilder,
    router: TopologyRouter,
    source_component: str,
    target_component: str,
    byte_count: int,
    dependencies: Sequence[str],
    *,
    name: str,
    routing_policy: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    transfer_metadata = dict(metadata or {})
    context = _COMPILATION_CONTEXT.get()
    # A serving lowerer can outlive one request scenario.  Never read an
    # optional component model from a stale context whose router is not the
    # router that owns this transfer.
    if context is not None and router is not context.router():
        context = None
    # Host-resident model weights use pageable H2D staging in the locked
    # llama.cpp CUDA path.  A component benchmark may opt in a measured
    # staging curve; the default remains the topology-only transfer model.
    pageable_spec = None
    pageable_d2h_spec = None
    scenario = context.scenario if context is not None else None
    if (
        scenario is not None
        and transfer_metadata.get("event_kind") == "model_weight_read"
        and transfer_metadata.get("host_gemm_offload_applied") is True
    ):
        raw_spec = scenario.hardware.metadata.get(
            "pageable_h2d_component_model"
        )
        if isinstance(raw_spec, Mapping) and raw_spec.get("enabled") is True:
            try:
                source_kind = _kind(_component(scenario, source_component))
                target_kind = _kind(_component(scenario, target_component))
                bandwidth_gbps = float(raw_spec.get("bandwidth_gbps", 0.0))
                fixed_latency_ns = float(raw_spec.get("fixed_latency_ns", 0.0))
            except (KeyError, TypeError, ValueError):
                bandwidth_gbps = 0.0
                fixed_latency_ns = 0.0
                source_kind = target_kind = ""
            if (
                source_kind in {"host_memory", "cpu"}
                and target_kind == "gpu"
                and bandwidth_gbps > 0.0
                and fixed_latency_ns >= 0.0
            ):
                pageable_spec = {
                    "bandwidth_gbps": bandwidth_gbps,
                    "fixed_latency_ns": fixed_latency_ns,
                    "graph_split_count": max(
                        0, int(raw_spec.get("graph_split_count", 0))
                    ),
                    "control_calls_per_split": max(
                        0, int(raw_spec.get("control_calls_per_split", 0))
                    ),
                    "resource_id": str(
                        raw_spec.get(
                            "resource_id", "pageable_h2d_component"
                        )
                    ),
                    "evidence": raw_spec.get("evidence"),
                }
    if (
        scenario is not None
        and transfer_metadata.get("event_kind") == "logits_d2h"
    ):
        raw_spec = scenario.hardware.metadata.get(
            "pageable_d2h_component_model"
        )
        if isinstance(raw_spec, Mapping) and raw_spec.get("enabled") is True:
            try:
                source_kind = _kind(_component(scenario, source_component))
                target_kind = _kind(_component(scenario, target_component))
                bandwidth_gbps = float(raw_spec.get("bandwidth_gbps", 0.0))
                fixed_latency_ns = float(raw_spec.get("fixed_latency_ns", 0.0))
            except (KeyError, TypeError, ValueError):
                bandwidth_gbps = 0.0
                fixed_latency_ns = 0.0
                source_kind = target_kind = ""
            if (
                source_kind in {"gpu", "hbm"}
                and target_kind in {"host_memory", "cpu"}
                and bandwidth_gbps > 0.0
                and fixed_latency_ns >= 0.0
            ):
                pageable_d2h_spec = {
                    "bandwidth_gbps": bandwidth_gbps,
                    "fixed_latency_ns": fixed_latency_ns,
                    "resource_id": str(
                        raw_spec.get(
                            "resource_id", "pageable_d2h_component"
                        )
                    ),
                    "evidence": raw_spec.get("evidence"),
                }
    phases = (
        _transfer_phases(
            context.scenario,
            router,
            source_component,
            target_component,
            byte_count,
            policy=routing_policy,
            name=name,
        )
        if context is not None and router is context.router()
        else router.transfer_phases(
            source_component,
            target_component,
            byte_count,
            policy=routing_policy,
            name=name,
        )
    )
    previous = tuple(dependencies)
    if pageable_spec is not None:
        split_count = pageable_spec["graph_split_count"]
        control_calls = pageable_spec["control_calls_per_split"]
        if (
            split_count > 0
            and control_calls > 0
            and not getattr(
                builder, "_pageable_h2d_graph_split_scheduled", False
            )
        ):
            control = builder.add(
                name + ".pageable_h2d_graph_split_control",
                TaskCategory.COMMUNICATION,
                (
                    ResourceDemand(
                        pageable_spec["resource_id"],
                        split_count
                        * control_calls
                        * pageable_spec["fixed_latency_ns"],
                        work_units=float(split_count * control_calls),
                    ),
                ),
                dependencies=previous,
                advance=False,
                metadata={
                    **_communication_task_metadata(
                        {
                            "event_kind": "pageable_h2d_graph_split_control",
                            "transfer_kind": "instruction",
                            "source_component": source_component,
                            "target_component": target_component,
                            "graph_split_count": split_count,
                            "control_calls_per_split": control_calls,
                            "control_fixed_latency_ns": pageable_spec[
                                "fixed_latency_ns"
                            ],
                            "component_model_evidence": pageable_spec[
                                "evidence"
                            ],
                        }
                    )
                },
            )
            setattr(builder, "_pageable_h2d_graph_split_scheduled", True)
            previous = (control,)
        stage_service_ns = pageable_spec["fixed_latency_ns"] + (
            8.0 * float(byte_count) / pageable_spec["bandwidth_gbps"]
        )
        staged = builder.add(
            name + ".pageable_h2d_stage",
            TaskCategory.COMMUNICATION,
            (
                ResourceDemand(
                    pageable_spec["resource_id"],
                    stage_service_ns,
                    bytes_moved=byte_count,
                ),
            ),
            dependencies=previous,
            advance=False,
            metadata={
                **_communication_task_metadata(
                    {
                        "event_kind": "pageable_h2d_component_stage",
                        "transfer_kind": "data",
                        "source_component": source_component,
                        "target_component": target_component,
                        "bytes": int(byte_count),
                        "component_model_bandwidth_gbps": pageable_spec[
                            "bandwidth_gbps"
                        ],
                        "component_model_fixed_latency_ns": pageable_spec[
                            "fixed_latency_ns"
                        ],
                        "component_model_evidence": pageable_spec[
                            "evidence"
                        ],
                        **transfer_metadata,
                    }
                )
            },
        )
        previous = (staged,)
    if pageable_d2h_spec is not None:
        stage_service_ns = pageable_d2h_spec["fixed_latency_ns"] + (
            8.0 * float(byte_count) / pageable_d2h_spec["bandwidth_gbps"]
        )
        staged = builder.add(
            name + ".pageable_d2h_stage",
            TaskCategory.COMMUNICATION,
            (
                ResourceDemand(
                    pageable_d2h_spec["resource_id"],
                    stage_service_ns,
                    bytes_moved=byte_count,
                ),
            ),
            dependencies=previous,
            advance=False,
            metadata={
                **_communication_task_metadata(
                    {
                        "event_kind": "pageable_d2h_component_stage",
                        "transfer_kind": "data",
                        "source_component": source_component,
                        "target_component": target_component,
                        "bytes": int(byte_count),
                        "component_model_bandwidth_gbps": pageable_d2h_spec[
                            "bandwidth_gbps"
                        ],
                        "component_model_fixed_latency_ns": pageable_d2h_spec[
                            "fixed_latency_ns"
                        ],
                        "component_model_evidence": pageable_d2h_spec[
                            "evidence"
                        ],
                        **transfer_metadata,
                    }
                )
            },
        )
        previous = (staged,)
    if not phases:
        return _add_join(
            builder,
            name + ".local",
            previous,
            metadata=_communication_task_metadata({
                "event_kind": "local_transfer",
                "transfer_kind": "data",
                "source_component": source_component,
                "target_component": target_component,
                "bytes": int(byte_count),
                **transfer_metadata,
            }),
        )
    last = ""
    for phase in phases:
        phase_metadata = dict(phase.metadata)
        phase_demands = phase.demands
        coherent_dma = (
            phase_metadata.get("transfer_execution") == "coherent_dma"
        )
        phase_event_kind = str(phase_metadata.get("event_kind", ""))
        phase_direction = None
        if phase_event_kind == "memory_read":
            phase_direction = "read"
        elif phase_event_kind == "memory_write":
            phase_direction = "write"
        elif phase_event_kind == "dma":
            dma_direction = str(phase_metadata.get("direction", ""))
            if dma_direction == "out":
                phase_direction = "read"
            elif dma_direction == "in":
                phase_direction = "write"
        phase_metadata.update(transfer_metadata)
        if coherent_dma and context is not None and router is context.router():
            (
                phase_demands,
                dma_engine_resource_id,
                dma_engine_service_ns,
                dma_engine_demand_added,
            ) = _coherent_dma_engine_demands(
                context.scenario,
                phase_demands,
            )
            phase_metadata.update(
                {
                    "dma_engine_resource_id": dma_engine_resource_id,
                    "dma_engine_service_ns": dma_engine_service_ns,
                    "dma_engine_demand_added": dma_engine_demand_added,
                    "dma_engine_accounting": (
                        "once_per_coherent_movement_zero_byte_occupancy"
                        if dma_engine_demand_added
                        else "normalized_existing_unique_zero_byte_occupancy"
                    ),
                }
            )
        phase_metadata.setdefault("transfer_kind", "data")
        phase_metadata["transfer_phase_event_kind"] = phase_event_kind
        if phase_direction is not None:
            # Keep the logical transfer event kind (for KV/event aggregation),
            # but report each endpoint/DMA phase using its physical direction.
            phase_metadata["memory_direction"] = phase_direction
        phase_metadata = _communication_task_metadata(phase_metadata)
        phase_dependencies = (
            builder.coherent_dma_dependencies(previous)
            if coherent_dma
            else previous
        )
        last = builder.add(
            phase.name,
            TaskCategory.COMMUNICATION,
            phase_demands,
            dependencies=phase_dependencies,
            advance=False,
            metadata=phase_metadata,
        )
        if coherent_dma:
            builder.record_coherent_dma_task(last)
        previous = (last,)
    return last


def _coherent_staged_weight_target(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    rank: LogicalRank,
    source_component: str,
    compute_component: str,
    byte_count: int,
    *,
    routing_policy: str,
) -> str:
    """Return an explicit active-memory target only for a coherent DMA span."""

    memory_component = rank.memory_component_id
    if not memory_component or memory_component == compute_component:
        return compute_component
    try:
        target = _component(scenario, memory_component)
    except KeyError:  # pragma: no cover - validated scenario invariant
        return compute_component
    if not target.is_active_memory:
        return compute_component
    try:
        phases = _transfer_phases(
            scenario,
            router,
            source_component,
            memory_component,
            byte_count,
            policy=routing_policy,
            name="coherent_staged_weight_probe",
        )
    except ValueError:
        return compute_component
    if (
        len(phases) == 1
        and phases[0].metadata.get("transfer_execution") == "coherent_dma"
    ):
        return memory_component
    return compute_component


def _add_host_orchestration(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    dependencies: Sequence[str],
    *,
    name: str,
    request_count: int,
    token_count: int,
) -> str:
    """Lower one aggregate V4 CPU/controller prep sequence per realized cohort.

    The high-level scheduler decision is deliberately decomposed into a small
    number of instruction-class and controller-transaction batches.  The
    batches are aggregate facts for the whole cohort: this accounts for CPU,
    cache, IOMMU, DMA, and PCIe payload staging time without creating a task
    per instruction, cache line, page, or packet.
    """

    profile = scenario.host_orchestration_profile
    if request_count <= 0 or token_count < 0:
        raise ValueError("orchestration counts are invalid")
    rank = plan.rank_at(0, 0, 0)
    cpu_id = profile.cpu_component_id
    gpu_id = profile.gpu_component_id
    cpu_profile, host_memory_profile = _cpu_profiles(scenario, cpu_id)
    runtime_profile = scenario.runtime_profile
    cpu_control = runtime_profile.cpu
    transport_control = runtime_profile.pcie_dma_iommu
    gpu_control = runtime_profile.gpu_controllers.get(gpu_id)
    if gpu_control is None:
        # ScenarioConfig validates the runtime registry.  A non-default GPU
        # reference can still be supplied by a custom orchestration profile;
        # fail closed instead of silently borrowing another GPU's controller.
        raise ValueError(
            "runtime controller profile is missing GPU {}".format(gpu_id)
        )
    payload_bytes = profile.payload_bytes(request_count, token_count)
    admission_ns = request_count * profile.admission_ns
    input_decode_ns = token_count * profile.input_decode_ns_per_token
    prepare_ns = (
        profile.batch_fixed_ns
        + request_count * profile.request_parse_ns
        + admission_ns
        + input_decode_ns
    )
    cpu_issue_width = max(
        1,
        min(
            cpu_profile.pipeline.decode_width,
            cpu_profile.pipeline.issue_width,
            cpu_profile.pipeline.retire_width,
        ),
    )
    # Each dependency chain is issued by one control thread.  Dividing its
    # service by every physical core would manufacture intra-chain parallelism;
    # core_count belongs to the execution resource's capacity instead.
    cpu_instructions_per_ns = max(
        1.0e-12,
        cpu_issue_width * cpu_profile.pipeline.frequency_ghz,
    )
    cpu_pipeline_resource = _component_resource_id(
        cpu_profile.pipeline.resource_id,
        reference_component_id=profile.cpu_component_id,
        target_component_id=cpu_id,
    )
    capacity_instructions = profile.capacity_instruction_count(request_count)
    capacity_service_ns = capacity_instructions / cpu_instructions_per_ns
    capacity = builder.add(
        name + ".capacity_accounting",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.scheduler_resource_id,
                capacity_service_ns,
                work_units=float(capacity_instructions),
            ),
            ResourceDemand(
                "{}.control_cache".format(cpu_id),
                cpu_control.cache_hit_latency_ns,
                bytes_moved=min(
                    payload_bytes,
                    cpu_control.request_batch_size
                    * cpu_control.cache_line_bytes,
                ),
            ),
        ),
        dependencies=tuple(dependencies),
        advance=False,
        metadata={
            "event_kind": "cpu_capacity_accounting",
            "runtime_phase": "capacity_check",
            "orchestration_stage": "host_prefix",
            "instruction_class": "integer_control",
            "instruction_count": capacity_instructions,
            "aggregation": "cohort_instruction_batch",
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "cpu_core_capacity": cpu_profile.pipeline.core_count,
            "cpu_control_resource_id": profile.scheduler_resource_id,
            "cpu_service_semantics": "single_serial_control_thread",
            "target_component": cpu_id,
        },
    )
    schedule_instructions = profile.schedule_instruction_count(
        request_count, token_count
    )
    schedule_service_ns = schedule_instructions / cpu_instructions_per_ns
    scheduled = builder.add(
        name + ".batch_operator_schedule",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.scheduler_resource_id,
                schedule_service_ns,
                work_units=float(schedule_instructions),
            ),
        ),
        dependencies=(capacity,),
        advance=False,
        metadata={
            "event_kind": "cpu_batch_operator_schedule",
            "runtime_phase": "operator_schedule",
            "orchestration_stage": "host_prefix",
            "instruction_class": "frontend_branch_integer",
            "instruction_count": schedule_instructions,
            "aggregation": "cohort_instruction_batch",
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "cpu_core_capacity": cpu_profile.pipeline.core_count,
            "cpu_control_resource_id": profile.scheduler_resource_id,
            "cpu_service_semantics": "single_serial_control_thread",
            "target_component": cpu_id,
        },
    )
    prepare = builder.add(
        name + ".prepare",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.scheduler_resource_id,
                prepare_ns,
                work_units=float(request_count),
            ),
        ),
        dependencies=(scheduled,),
        advance=False,
        metadata={
            "event_kind": "host_cohort_prepare",
            "orchestration_stage": "host_prefix",
            "batch_fixed_setup_ns": profile.batch_fixed_ns,
            "request_parse_total_ns": request_count * profile.request_parse_ns,
            "admission_total_ns": admission_ns,
            "input_decode_total_ns": input_decode_ns,
            "host_frontend_cost_model": "admission_plus_token_decode_v1",
            "host_frontend_cost_units": {
                "admission_ns_per_request": profile.admission_ns,
                "input_decode_ns_per_token": profile.input_decode_ns_per_token,
                "output_encode_ns_per_token": profile.output_encode_ns_per_token,
            },
            "cohort_setup_scope": "request_local" if request_count == 1 else "scheduler_cohort",
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "payload_bytes": payload_bytes,
            "cpu_control_resource_id": profile.scheduler_resource_id,
            "target_component": cpu_id,
        },
    )

    pack_workload = MemoryWorkload(
        read_bytes=payload_bytes,
        write_bytes=payload_bytes,
        working_set_bytes=2 * payload_bytes,
        reuse_factor=1.0,
        streaming_fraction=1.0,
        name="cohort_payload_pack",
    )
    pack_estimate = _memoized_cost_estimate(
        scenario,
        ("cpu_memory", cpu_id, pack_workload),
        lambda: estimate_cpu_memory(
            cpu_profile,
            host_memory_profile,
            pack_workload,
        ),
    )
    prior: Tuple[str, ...] = (prepare,)
    pack_ns = token_count * profile.token_pack_ns
    for phase in pack_estimate.phases:
        phase_demands = []
        for demand in phase.demands:
            namespaced = _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=cpu_id,
            )
            if (
                phase.name == "cpu_memory"
                and namespaced.resource_id
                == _component_resource_id(
                    cpu_profile.pipeline.resource_id,
                    reference_component_id=profile.cpu_component_id,
                    target_component_id=cpu_id,
                )
            ):
                namespaced = ResourceDemand(
                    namespaced.resource_id,
                    max(namespaced.service_ns, pack_ns),
                    bytes_moved=namespaced.bytes_moved,
                    energy_pj=namespaced.energy_pj,
                    work_units=namespaced.work_units,
                )
            phase_demands.append(namespaced)
        if phase.name == "cpu_memory":
            phase_demands.append(
                ResourceDemand(
                    profile.pack_resource_id,
                    pack_ns,
                    bytes_moved=payload_bytes,
                    work_units=float(token_count),
                )
            )
        packed = builder.add(
            "{}.pack.{}".format(name, phase.name),
            phase.category,
            tuple(phase_demands),
            dependencies=prior,
            advance=False,
            metadata={
                "event_kind": "host_cohort_pack",
                "orchestration_stage": "host_prefix",
                "phase": phase.name,
                "payload_bytes": payload_bytes,
                "cohort_request_count": request_count,
                "cohort_token_count": token_count,
                "cost_model": dict(pack_estimate.metadata),
                "target_component": cpu_id,
            },
        )
        prior = (packed,)

    iommu_pages = max(
        1,
        (payload_bytes + transport_control.iommu_page_size_bytes - 1)
        // transport_control.iommu_page_size_bytes,
    )
    iommu_misses = max(
        1,
        (iommu_pages + transport_control.iommu_tlb_entries - 1)
        // transport_control.iommu_tlb_entries,
    )
    iommu_walk_batches = max(
        1,
        (
            iommu_misses
            + transport_control.iommu_max_outstanding_walks
            - 1
        )
        // transport_control.iommu_max_outstanding_walks,
    )
    translated = builder.add(
        name + ".iommu_translate",
        TaskCategory.MEMORY,
        (
            ResourceDemand(
                "{}.iommu".format(cpu_id),
                iommu_walk_batches * transport_control.iommu_miss_latency_ns,
                bytes_moved=payload_bytes,
                work_units=float(iommu_pages),
            ),
        ),
        dependencies=prior,
        advance=False,
        metadata={
            "event_kind": "iommu_translation_batch",
            "runtime_phase": "iommu_translate",
            "orchestration_stage": "host_prefix",
            "transaction_kind": "address_translation",
            "transaction_count": iommu_pages,
            "transaction_batches": iommu_walk_batches,
            "payload_bytes": payload_bytes,
            "aggregation": "controller_transaction_batch",
            "source_component": cpu_id,
            "target_component": gpu_id,
        },
    )
    dma_setup = _dma_setup_service(
        payload_bytes,
        batch_bytes=transport_control.dma_batch_bytes,
        queue_depth=transport_control.dma_queue_depth,
        max_outstanding=transport_control.dma_max_outstanding,
        fixed_latency_ns=profile.dma_latency_ns,
        submission_ns_per_wave=profile.dma_queue_submission_ns,
    )
    dma_queued = builder.add(
        name + ".dma_queue",
        TaskCategory.COMMUNICATION,
        (
            ResourceDemand(
                profile.dma_resource_id,
                dma_setup.service_ns,
                work_units=float(dma_setup.transaction_count),
            ),
        ),
        dependencies=(translated,),
        advance=False,
        metadata=_communication_task_metadata(
            {
                "event_kind": "dma_controller_batch",
                "runtime_phase": "dma_queue",
                "orchestration_stage": "host_prefix",
                "transaction_kind": "host_to_device_dma",
                "transaction_count": dma_setup.transaction_count,
                "transaction_batches": dma_setup.wave_count,
                "queue_parallelism": dma_setup.queue_parallelism,
                "queue_depth": transport_control.dma_queue_depth,
                "max_outstanding": transport_control.dma_max_outstanding,
                "payload_bytes": payload_bytes,
                "aggregation": "controller_transaction_batch",
                "cost_owner": "dma_descriptor_queue",
                "dma_engine_accounting": "per_engine_setup_zero_byte_occupancy",
                "excludes_costs": (
                    "driver_submit",
                    "gpu_command_processor",
                    "operator_kernel_launch",
                ),
                "source_component": cpu_id,
                "target_component": gpu_id,
            }
        ),
    )
    h2d_source_component = cpu_id
    h2d_target_component = gpu_id
    try:
        candidate_source = _compute_local_runtime_memory_component_id(
            scenario,
            cpu_id,
        )
        candidate_target = _compute_local_runtime_memory_component_id(
            scenario,
            gpu_id,
        )
        coherent_h2d_probe = _transfer_phases(
            scenario,
            router,
            candidate_source,
            candidate_target,
            payload_bytes,
            policy=plan.routing_policy,
            name=name + ".h2d_probe",
        )
    except (KeyError, TypeError, ValueError):
        coherent_h2d_probe = ()
    if (
        len(coherent_h2d_probe) == 1
        and coherent_h2d_probe[0].metadata.get("transfer_execution")
        == "coherent_dma"
    ):
        h2d_source_component = candidate_source
        h2d_target_component = candidate_target
    # A topology that has not explicitly opted its whole memory span into
    # coherent DMA retains the legacy one-hop logical CPU-to-GPU transfer.
    transferred = _add_transfer_tasks(
        builder,
        router,
        h2d_source_component,
        h2d_target_component,
        payload_bytes,
        (dma_queued,),
        name=name + ".h2d",
        routing_policy=plan.routing_policy,
        metadata={
            "event_kind": "host_cohort_h2d",
            "transfer_kind": "data",
            "orchestration_stage": "host_prefix",
            "payload_bytes": payload_bytes,
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "single_dma_transaction": True,
            "controller_source_component": cpu_id,
            "controller_target_component": gpu_id,
        },
    )

    return transferred


def _add_physical_invocation_frontend(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    dependencies: Sequence[str],
    *,
    name: str,
    request_count: int,
    token_count: int,
    invocation_count: int,
    invocation_family: str,
    orchestration_stage: str,
    invocation_group_ids: Sequence[str] = (),
    execution_phase: Optional[str] = None,
    first_decode_invocation: bool = False,
) -> str:
    """Lower aggregate command build, driver submit, and GPU CP work."""

    profile = scenario.host_orchestration_profile
    if request_count <= 0 or token_count < 0 or invocation_count <= 0:
        raise ValueError("physical invocation frontend counts are invalid")
    if not invocation_family or not orchestration_stage:
        raise ValueError("physical invocation frontend identity is invalid")

    cpu_id = profile.cpu_component_id
    gpu_id = profile.gpu_component_id
    cpu_profile, _host_memory_profile = _cpu_profiles(scenario, cpu_id)
    gpu_control = scenario.runtime_profile.gpu_controllers.get(gpu_id)
    if gpu_control is None:
        raise ValueError(
            "runtime controller profile is missing GPU {}".format(gpu_id)
        )
    payload_bytes = profile.payload_bytes(request_count, token_count)
    cpu_issue_width = max(
        1,
        min(
            cpu_profile.pipeline.decode_width,
            cpu_profile.pipeline.issue_width,
            cpu_profile.pipeline.retire_width,
        ),
    )
    cpu_instructions_per_ns = max(
        1.0e-12,
        cpu_issue_width * cpu_profile.pipeline.frequency_ghz,
    )
    normalized_group_ids = tuple(
        dict.fromkeys(
            str(group_id)
            for group_id in invocation_group_ids
            if str(group_id)
        )
    )

    command_build_instructions = profile.command_build_instruction_count(
        invocation_count
    )
    command_build_service_ns = (
        command_build_instructions / cpu_instructions_per_ns
    )
    command_built = builder.add(
        name + ".invocation_command_build",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.scheduler_resource_id,
                command_build_service_ns,
                work_units=float(command_build_instructions),
            ),
        ),
        dependencies=tuple(dependencies),
        advance=False,
        metadata={
            "event_kind": "cpu_invocation_command_build",
            "runtime_phase": "command_build",
            "orchestration_stage": orchestration_stage,
            "instruction_class": "graph_and_command_build",
            "instruction_count": command_build_instructions,
            "physical_invocation_family": invocation_family,
            "physical_invocation_group_ids": normalized_group_ids,
            "physical_invocation_group_count": invocation_count,
            "aggregation": "cohort_invocation_batch",
            "cost_owner": "host_cpu_command_build",
            "excludes_costs": (
                "driver_submit",
                "gpu_command_processor",
                "operator_kernel_launch",
            ),
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "cpu_core_capacity": cpu_profile.pipeline.core_count,
            "cpu_control_resource_id": profile.scheduler_resource_id,
            "cpu_service_semantics": "single_serial_control_thread",
            "target_component": cpu_id,
        },
    )
    submitted = builder.add(
        name + ".submit",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.submission_resource_id,
                invocation_count * profile.submission_ns,
                work_units=float(invocation_count),
            ),
        ),
        dependencies=(command_built,),
        advance=False,
        metadata={
            "event_kind": "host_cohort_submit",
            "runtime_phase": "driver_submit",
            "transfer_kind": "instruction",
            "orchestration_stage": orchestration_stage,
            "payload_bytes": payload_bytes,
            "cohort_request_count": request_count,
            "cohort_token_count": token_count,
            "physical_invocation_family": invocation_family,
            "physical_invocation_group_ids": normalized_group_ids,
            "submission_count": invocation_count,
            "physical_invocation_group_count": invocation_count,
            "batching_contract": "one_driver_submit_per_physical_invocation_group",
            "cost_owner": "driver_submission_queue",
            "excludes_costs": (
                "gpu_command_processor",
                "operator_kernel_launch",
            ),
            "source_component": cpu_id,
            "target_component": gpu_id,
        },
    )
    command_profile = gpu_control.command_processor
    command_batches = max(
        1,
        (invocation_count + command_profile.launch_batch_size - 1)
        // command_profile.launch_batch_size,
    )
    command_processor = builder.add(
        name + ".gpu_command_processor",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                "{}.command_processor".format(gpu_id),
                command_batches
                * command_profile.command_submission_latency_ns,
                bytes_moved=payload_bytes,
                work_units=float(command_batches),
            ),
        ),
        dependencies=(submitted,),
        advance=False,
        metadata={
            "event_kind": "gpu_command_processor_batch",
            "runtime_phase": "gpu_command_processor",
            "orchestration_stage": orchestration_stage,
            "instruction_class": "gpu_command",
            "command_batches": command_batches,
            "physical_invocation_family": invocation_family,
            "physical_invocation_group_ids": normalized_group_ids,
            "physical_invocation_group_count": invocation_count,
            "payload_bytes": payload_bytes,
            "aggregation": "controller_transaction_batch",
            "cost_owner": "gpu_command_processor",
            "excludes_costs": (
                "driver_submit",
                "operator_kernel_launch",
            ),
            "source_component": cpu_id,
            "target_component": gpu_id,
        },
    )
    # CUDA API launch/synchronize rows are not one-to-one with lowered
    # operators.  If an explicit profile proves a phase aggregate, charge it
    # exactly once after command submission for this physical invocation.
    # Missing policy, identity, or coverage keeps the model unchanged.
    calibration_metadata = scenario.placement.metadata
    if calibration_metadata.get("native_calibration_apply_phase_boundary") is True:
        calibration = profile_from_mapping(
            calibration_metadata.get("native_calibration")
        )
        execution_phase = execution_phase or _execution_phase_from_name(name)
        boundary_ns = phase_boundary_calibration_ns(
            calibration,
            execution_phase,
            model_sha256=_model_gguf_sha256(scenario.model),
            hardware_fingerprint=calibration_metadata.get("hardware_fingerprint"),
            runtime_fingerprint=(calibration_metadata.get("llama_cpp_runtime_fingerprint")
                                 or (scenario.llama_cpp_config.fingerprint
                                     if scenario.llama_cpp_config is not None else None)),
        )
        # CUDA semantic traces show a one-time gap on the first M=1 decode
        # invocation (the initial kq graph/kernel path); later decode
        # invocations have no comparable gap.  Keep this residual on the
        # explicit phase-boundary task so it is charged once, rather than
        # multiplying a per-token or per-kernel coefficient.  The profile
        # helper enforces identity and coverage; CPU-only paths never use a
        # CUDA startup residual.
        first_decode_invocation = (
            (
                bool(first_decode_invocation)
                or (
                    str(execution_phase or "").strip().lower() == "decode"
                    and re.search(r"(?:^|\.)decode0*1(?:\.|$)", str(name)) is not None
                )
            )
            and scenario.llama_cpp_config is not None
            and int(scenario.llama_cpp_config.gpu_layers) != 0
        )
        startup_extra_ns = decode_first_invocation_extra_ns(
            calibration,
            first_invocation=first_decode_invocation,
            model_sha256=_model_gguf_sha256(scenario.model),
            hardware_fingerprint=calibration_metadata.get("hardware_fingerprint"),
            runtime_fingerprint=(calibration_metadata.get("llama_cpp_runtime_fingerprint")
                                 or (scenario.llama_cpp_config.fingerprint
                                     if scenario.llama_cpp_config is not None else None)),
        )
        if boundary_ns is not None and startup_extra_ns is not None:
            boundary_ns += startup_extra_ns
        if boundary_ns is not None:
            return builder.add(
                name + ".phase_boundary",
                TaskCategory.SYNCHRONIZATION,
                (
                    ResourceDemand(
                        "{}.frontend".format(gpu_id),
                        boundary_ns,
                    ),
                ),
                dependencies=(command_processor,),
                advance=False,
                metadata={
                    "event_kind": "native_phase_boundary",
                    "runtime_phase": execution_phase,
                    "phase_boundary_policy": "one_task_per_phase_invocation",
                    "phase_boundary_ns": boundary_ns,
                    "decode_first_invocation": bool(first_decode_invocation),
                    "decode_first_invocation_extra_ns": startup_extra_ns,
                    "physical_invocation_family": invocation_family,
                    "physical_invocation_group_ids": normalized_group_ids,
                    "physical_invocation_group_count": invocation_count,
                    "orchestration_stage": orchestration_stage,
                    "cost_owner": "native_cuda_api_phase_boundary",
                    "source_component": cpu_id,
                    "target_component": gpu_id,
                    "evidence_scope": "aggregate_launch_plus_sync_once_per_phase_invocation",
                },
            )
    return command_processor


def _add_request_marker_boundary(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    request: RequestSpec,
    dependency: str,
    *,
    marker: str,
) -> str:
    """Add one request-lifecycle marker cost when exact evidence exists.

    Request marker timing is a request-level additive action.  It is never
    inferred from phase totals or spread over operators.  The calibration
    helper requires a covered marker row, model/hardware/runtime identity,
    and exact prompt/output shape, so an absent or mismatched profile leaves
    the dependency unchanged (fail-closed).
    """
    metadata = scenario.placement.metadata
    if metadata.get("native_calibration_apply_request_boundary") is not True:
        return dependency
    profile = profile_from_mapping(metadata.get("native_calibration"))
    if profile is None:
        return dependency
    request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    prompt_fingerprint = (
        request_metadata.get("prompt_fingerprint")
        or metadata.get("prompt_fingerprint")
        or metadata.get("request_prompt_fingerprint")
    )
    value = request_marker_calibration_ns(
        profile,
        marker,
        model_sha256=_model_gguf_sha256(scenario.model),
        hardware_fingerprint=metadata.get("hardware_fingerprint"),
        runtime_fingerprint=(metadata.get("llama_cpp_runtime_fingerprint")
                             or (scenario.llama_cpp_config.fingerprint
                                 if scenario.llama_cpp_config is not None else None)),
        prompt_tokens=request.prompt_tokens,
        output_tokens=request.output_tokens,
        prompt_fingerprint=str(prompt_fingerprint) if prompt_fingerprint is not None else None,
    )
    # A zero-valued measured marker is valid evidence but needs no task.  Do
    # not create a synthetic zero-duration event that could perturb ordering.
    if value is None or value <= 0:
        return dependency
    return builder.add(
        "request_boundary." + str(marker),
        TaskCategory.SYNCHRONIZATION,
        (ResourceDemand(
            scenario.host_orchestration_profile.scheduler_resource_id,
            value,
        ),),
        dependencies=(dependency,),
        advance=False,
        metadata={
            "event_kind": "native_request_marker_boundary",
            "request_marker": str(marker),
            "request_marker_policy": "additive_once_per_request",
            "request_marker_ns": value,
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.output_tokens,
            "cost_owner": "native_request_marker_boundary",
            "evidence_scope": "exact_request_marker_once_per_request",
        },
    )


def _add_swap_control(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    dependencies: Sequence[str],
    *,
    name: str,
    page_count: int,
    byte_count: int,
) -> str:
    """Model CPU page-table lookup and descriptor submission for KV swap."""

    profile = scenario.host_orchestration_profile
    units = max(1, int(page_count))
    descriptor_bytes = units * profile.kv_descriptor_bytes
    control_ns = profile.batch_fixed_ns + units * (
        profile.kv_page_lookup_ns + profile.kv_descriptor_ns
    )
    rank = plan.rank_at(0, 0, 0)
    cpu_profile, host_memory_profile = _cpu_profiles(
        scenario, profile.cpu_component_id
    )
    control = builder.add(
        name + ".page_table",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.scheduler_resource_id,
                control_ns,
                bytes_moved=descriptor_bytes,
                work_units=float(units),
            ),
        ),
        dependencies=tuple(dependencies),
        advance=False,
        metadata={
            "event_kind": "kv_swap_cpu_control",
            "orchestration_stage": "swap_control",
            "page_count": units,
            "descriptor_bytes": descriptor_bytes,
            "transfer_bytes": byte_count,
        },
    )
    descriptor_workload = MemoryWorkload(
        read_bytes=descriptor_bytes,
        write_bytes=descriptor_bytes,
        working_set_bytes=2 * descriptor_bytes,
        reuse_factor=2.0,
        streaming_fraction=0.0,
        name="kv_swap_descriptor_pack",
    )
    descriptor_estimate = _memoized_cost_estimate(
        scenario,
        ("cpu_memory", profile.cpu_component_id, descriptor_workload),
        lambda: estimate_cpu_memory(
            cpu_profile,
            host_memory_profile,
            descriptor_workload,
        ),
    )
    prior: Tuple[str, ...] = (control,)
    for phase in descriptor_estimate.phases:
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=profile.cpu_component_id,
            )
            for demand in phase.demands
        )
        descriptor_task = builder.add(
            "{}.descriptor.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                "event_kind": "kv_swap_descriptor_pack",
                "orchestration_stage": "swap_control",
                "phase": phase.name,
                "page_count": units,
                "descriptor_bytes": descriptor_bytes,
                "transfer_bytes": byte_count,
                "cost_model": dict(descriptor_estimate.metadata),
            },
        )
        prior = (descriptor_task,)
    return builder.add(
        name + ".submit",
        TaskCategory.POLICY,
        (
            ResourceDemand(
                profile.dma_resource_id,
                profile.submission_ns,
                bytes_moved=descriptor_bytes,
                work_units=1.0,
            ),
        ),
        dependencies=prior,
        advance=False,
        metadata={
            "event_kind": "kv_swap_submit",
            "transfer_kind": "instruction",
            "orchestration_stage": "swap_control",
            "page_count": units,
            "descriptor_bytes": descriptor_bytes,
            "transfer_bytes": byte_count,
            "submission_count": 1,
            "source_component": profile.cpu_component_id,
            "target_component": profile.gpu_component_id,
        },
    )


def _add_collective_tasks(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    name: str,
    kind: str,
    ranks: Sequence[LogicalRank],
    tensor_bytes: int,
    dependencies: Sequence[str],
    *,
    tensor_elements: Optional[int] = None,
    element_bits: int = 8,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    rank_value_components: Dict[int, str] = {}
    collective_dependencies = tuple(dependencies)
    if len(ranks) == 1:
        rank = ranks[0]
        rank_value_components[rank.rank] = (
            builder.rank_value_component(dependencies, rank.rank)
            or rank.component_id
        )
    else:
        normalized_inputs: List[str] = []
        for rank in ranks:
            source_component = (
                builder.rank_value_component(dependencies, rank.rank)
                or rank.component_id
            )
            if source_component != rank.component_id:
                normalized_inputs.append(
                    _add_transfer_tasks(
                        builder,
                        router,
                        source_component,
                        rank.component_id,
                        tensor_bytes,
                        dependencies,
                        name="{}.rank{:03d}.input_to_collective".format(
                            name, rank.rank
                        ),
                        routing_policy=plan.routing_policy,
                        metadata={
                            "event_kind": "operator_output_transfer",
                            "operator_id": name,
                            "rank": rank.rank,
                            "transport": "activation",
                            "placement_boundary": "collective",
                        },
                    )
                )
            rank_value_components[rank.rank] = rank.component_id
        if normalized_inputs:
            collective_dependencies = tuple(normalized_inputs)
    components = tuple(
        dict.fromkeys(rank_value_components[rank.rank] for rank in ranks)
    )
    collective = _collective_plan(
        scenario,
        router,
        kind,
        components,
        tensor_bytes,
        algorithm=plan.collective_algorithm,
        routing_policy=plan.routing_policy,
    )
    normalized_kind = collective.kind
    reduction_elements: Optional[int] = None
    if (
        normalized_kind in {"all_reduce", "reduce_scatter"}
        and tensor_bytes > 0
        and len(ranks) > 1
    ):
        if (
            isinstance(element_bits, bool)
            or not isinstance(element_bits, int)
            or element_bits <= 0
        ):
            raise ValueError("element_bits must be a positive integer")
        if tensor_elements is None:
            reduction_elements = (
                tensor_bytes * 8 + element_bits - 1
            ) // element_bits
        else:
            if (
                isinstance(tensor_elements, bool)
                or not isinstance(tensor_elements, int)
                or tensor_elements < 0
            ):
                raise ValueError(
                    "tensor_elements must be a non-negative integer"
                )
            reduction_elements = tensor_elements
        declared_bytes = (
            reduction_elements * element_bits + 7
        ) // 8
        if declared_bytes != tensor_bytes:
            raise ValueError(
                "tensor_elements and element_bits describe {} bytes, not {}"
                .format(declared_bytes, tensor_bytes)
            )

    base_metadata = dict(metadata or {})
    prior = collective_dependencies
    if not collective.rounds:
        final = _add_join(
            builder,
            name + ".local",
            prior,
            metadata=_communication_task_metadata({
                "event_kind": "collective",
                "collective_kind": normalized_kind,
                "algorithm": "local",
                "planned_algorithm": collective.algorithm,
                "bytes": tensor_bytes,
                **base_metadata,
            }),
        )
    else:
        for round_spec in collective.rounds:
            round_ends: List[str] = []
            for transfer_index, transfer in enumerate(round_spec.transfers):
                round_ends.append(
                    _add_transfer_tasks(
                        builder,
                        router,
                        transfer.source_component,
                        transfer.target_component,
                        transfer.byte_count,
                        prior,
                        name="{}.round{:02d}.tx{:02d}".format(
                            name, round_spec.index, transfer_index
                        ),
                        routing_policy=plan.routing_policy,
                        metadata={
                            "event_kind": "collective_transfer",
                            "collective_kind": normalized_kind,
                            "algorithm": collective.algorithm,
                            "collective_name": name,
                            **base_metadata,
                        },
                    )
                )
            barrier = _add_join(
                builder,
                "{}.round{:02d}.barrier".format(name, round_spec.index),
                round_ends,
                metadata=_communication_task_metadata({
                    "event_kind": "collective_barrier",
                    "collective_kind": normalized_kind,
                    "algorithm": collective.algorithm,
                    "round": round_spec.index,
                    **base_metadata,
                }),
            )
            prior = (barrier,)
        final = prior[0]

    if reduction_elements is not None:
        participant_count = len(ranks)
        dependency_depth = max(
            1, int(math.ceil(math.log2(participant_count)))
        )
        reduction_ends: List[str] = []
        for rank_index, rank in enumerate(ranks):
            shard = shard_extent(
                reduction_elements,
                participant_count,
                rank_index,
                allow_padding=plan.allow_padding,
            )
            workload = ReductionWorkload(
                input_elements=participant_count * shard.local_size,
                output_elements=shard.local_size,
                input_bits=element_bits,
                output_bits=element_bits,
                dependency_depth=dependency_depth,
                streaming_fraction=1.0,
                name="{}_local_collective_reduction".format(
                    normalized_kind
                ),
            )
            gpu_profile, hbm_profile = _gpu_profiles(
                scenario,
                rank.component_id,
                rank.memory_component_id,
            )
            estimate = _memoized_cost_estimate(
                scenario,
                (
                    "gpu_reduction",
                    rank.component_id,
                    rank.memory_component_id,
                    workload,
                ),
                lambda: estimate_gpu_reduction(
                    gpu_profile,
                    hbm_profile,
                    workload,
                ),
            )
            rank_prior = (final,)
            last = ""
            for phase in estimate.phases:
                demands = tuple(
                    _namespace_demand(
                        scenario,
                        demand,
                        rank=rank,
                        target_component_id=rank.component_id,
                    )
                    for demand in phase.demands
                )
                last = builder.add(
                    "{}.rank{:03d}.local_reduce.{}".format(
                        name, rank.rank, phase.name
                    ),
                    TaskCategory.COLLECTIVE,
                    demands,
                    dependencies=rank_prior,
                    advance=False,
                    metadata={
                        **base_metadata,
                        "event_kind": base_metadata.get(
                            "event_kind", "collective_reduce"
                        ),
                        "collective_phase": "local_reduction",
                        "collective_kind": normalized_kind,
                        "algorithm": collective.algorithm,
                        "collective_name": name,
                        "rank": rank.rank,
                        "tp_rank": rank.tp_rank,
                        "pp_rank": rank.pp_rank,
                        "ep_rank": rank.ep_rank,
                        "target_component": rank.component_id,
                        "operator_class": OperatorClass.REDUCTION.value,
                        "reduction_schedule": "post_collective_proxy",
                        "logical_tensor_elements": reduction_elements,
                        "physical_tensor_elements": shard.padded_size,
                        "padding_elements": shard.padding,
                        "element_bits": element_bits,
                        "rank_reduction_input_elements": (
                            workload.input_elements
                        ),
                        "rank_reduction_output_elements": (
                            workload.output_elements
                        ),
                        "rank_result_elements": (
                            reduction_elements
                            if normalized_kind == "all_reduce"
                            else shard.local_size
                        ),
                        "op_name": name,
                        "phase": phase.name,
                        "cost_model": dict(estimate.metadata),
                        "phase_metadata": dict(phase.metadata),
                        "analytical_ops": sum(
                            demand.work_units for demand in demands
                        ),
                        "analytical_bytes": sum(
                            demand.bytes_moved for demand in demands
                        ),
                        "analytical_energy_pj": sum(
                            demand.energy_pj for demand in demands
                        ),
                        "analytical_service_ns": max(
                            (
                                demand.service_ns
                                for demand in demands
                            ),
                            default=0.0,
                        ),
                    },
                )
                rank_prior = (last,)
            if not last:  # pragma: no cover - estimator contract
                raise AssertionError(
                    "collective reduction estimate produced no phases"
                )
            reduction_ends.append(
                last
            )
        final = _add_join(
            builder,
            name + ".complete",
            reduction_ends,
            metadata=_communication_task_metadata({
                "event_kind": "collective_complete",
                "collective_kind": normalized_kind,
                "algorithm": collective.algorithm,
                **base_metadata,
            }),
        )
    for rank in ranks:
        builder.record_rank_value(
            final,
            rank.rank,
            rank_value_components[rank.rank],
        )
    return final


def _component_resource_id(
    configured: str,
    *,
    reference_component_id: str,
    target_component_id: str,
) -> str:
    """Rebind one component-local V4 resource to a logical rank target."""

    value = str(configured)
    if target_component_id == reference_component_id:
        return value
    prefix = reference_component_id + "."
    if value.startswith(prefix):
        return target_component_id + value[len(reference_component_id) :]
    suffix = value.split(".", 1)[-1]
    return "{}.{}".format(target_component_id, suffix)


def _rank_gpu_resource(
    scenario: ScenarioConfig,
    rank: LogicalRank,
    configured: str,
) -> str:
    return _component_resource_id(
        configured,
        reference_component_id=(
            scenario.host_orchestration_profile.gpu_component_id
        ),
        target_component_id=rank.component_id,
    )


def _rank_compute_resource(scenario: ScenarioConfig, rank: LogicalRank) -> str:
    """Return the rank-local scalar/ALU resource for policy/reduction work."""

    gpu_profile = _resolve_component_profile(
        scenario,
        rank.component_id, GPUProfile
    )
    return _rank_gpu_resource(
        scenario, rank, gpu_profile.scalar_resource_id
    )


def _rank_memory_resource(scenario: ScenarioConfig, rank: LogicalRank) -> str:
    reference = scenario.host_orchestration_profile.gpu_component_id
    _gpu_profile, hbm_profile = _gpu_profiles(
        scenario,
        rank.component_id,
        rank.memory_component_id,
    )
    owner = rank.memory_component_id or rank.component_id
    if rank.component_id == reference and rank.memory_component_id is None:
        return str(hbm_profile.resource_id)
    return _component_resource_id(
        hbm_profile.resource_id,
        reference_component_id=reference,
        target_component_id=owner,
    )


def _nearest_profile_component_id(
    scenario: ScenarioConfig,
    source_component_id: str,
    component_kind: str,
) -> Optional[str]:
    """Return the nearest bidirectionally reachable component of one kind."""

    def resolve() -> Optional[str]:
        router = _topology_router(scenario)
        ranked: List[Tuple[float, str]] = []
        for component in scenario.hardware.components:
            if _kind(component) != component_kind:
                continue
            try:
                outward = router.route(
                    source_component_id, component.component_id, 1
                )
                inward = router.route(
                    component.component_id, source_component_id, 1
                )
                _resolve_component_profile(scenario, component)
            except (KeyError, TypeError, ValueError):
                continue
            ranked.append(
                (
                    sum(hop.latency_ns for hop in outward)
                    + sum(hop.latency_ns for hop in inward),
                    component.component_id,
                )
            )
        return min(ranked)[1] if ranked else None

    context = _active_compilation_context(scenario)
    if context is None:
        return resolve()
    value = context.invariant(
        ("nearest_profile_component", source_component_id, component_kind),
        resolve,
    )
    return None if value is None else str(value)


def _gpu_profiles(
    scenario: ScenarioConfig,
    gpu_component_id: str,
    memory_component_id: Optional[str] = None,
) -> Tuple[GPUProfile, HBMProfile]:
    """Resolve GPU/HBM cost profiles from the exact target components."""

    def resolve() -> Tuple[GPUProfile, HBMProfile]:
        gpu_profile = _resolve_component_profile(
            scenario, gpu_component_id, GPUProfile
        )
        selected_memory = memory_component_id
        if selected_memory is not None:
            memory_component = _component(scenario, selected_memory)
            if _kind(memory_component) != "hbm":
                selected_memory = None
        if selected_memory is None:
            selected_memory = _nearest_profile_component_id(
                scenario, gpu_component_id, "hbm"
            )
        if selected_memory is None:
            raise ValueError(
                "GPU component {} has no reachable HBM profile target".format(
                    gpu_component_id
                )
            )
        hbm_profile = _resolve_component_profile(
            scenario, selected_memory, HBMProfile
        )
        return gpu_profile, hbm_profile

    context = _active_compilation_context(scenario)
    if context is None:
        return resolve()
    value = context.invariant(
        (
            "gpu_profiles",
            gpu_component_id,
            memory_component_id,
        ),
        resolve,
    )
    return value


def _cpu_profiles(
    scenario: ScenarioConfig,
    cpu_component_id: Optional[str] = None,
    memory_component_id: Optional[str] = None,
) -> Tuple[CPUProfile, HostMemoryProfile]:
    """Resolve CPU/host-memory profiles from the exact CPU target."""

    selected_cpu = cpu_component_id or (
        scenario.host_orchestration_profile.cpu_component_id
    )
    selected_memory = memory_component_id

    def resolve() -> Tuple[CPUProfile, HostMemoryProfile]:
        cpu_profile = _resolve_component_profile(
            scenario, selected_cpu, CPUProfile
        )
        # A native llama.cpp CPU path is bounded by n_threads (and batch work
        # by n_threads_batch).  Project the declared thread count onto the
        # existing typed CPU pipeline so ngl=0 runs carry real timing changes.
        llama_cfg = getattr(scenario, "llama_cpp_config", None)
        if llama_cfg is not None:
            thread_count = llama_cfg.threads if llama_cfg.threads > 0 else cpu_profile.pipeline.core_count
            thread_count = min(cpu_profile.pipeline.core_count, thread_count)
            if thread_count != cpu_profile.pipeline.core_count:
                cpu_profile = replace(
                    cpu_profile,
                    pipeline=replace(cpu_profile.pipeline, core_count=thread_count),
                )
        if selected_memory is not None:
            memory_component = _component(scenario, selected_memory)
            if _kind(memory_component) != "host_memory":
                raise ValueError(
                    "CPU component {} host-memory target {} is not host_memory"
                    .format(selected_cpu, selected_memory)
                )
            router = _topology_router(scenario)
            try:
                router.route(selected_cpu, selected_memory, 1)
                router.route(selected_memory, selected_cpu, 1)
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "CPU component {} cannot bidirectionally reach host-memory target {}"
                    .format(selected_cpu, selected_memory)
                )
            host_memory_component = selected_memory
        else:
            host_memory_component = _nearest_profile_component_id(
                scenario, selected_cpu, "host_memory"
            )
        if host_memory_component is None:
            raise ValueError(
                "CPU component {} has no reachable host-memory profile target"
                .format(selected_cpu)
            )
        host_memory_profile = _resolve_component_profile(
            scenario, host_memory_component, HostMemoryProfile
        )
        return cpu_profile, host_memory_profile

    context = _active_compilation_context(scenario)
    if context is None:
        return resolve()
    value = context.invariant(
        ("cpu_profiles", selected_cpu, selected_memory), resolve
    )
    return value


def _compute_local_runtime_memory_component_id(
    scenario: ScenarioConfig,
    target_component_id: str,
) -> str:
    """Resolve the memory backend physically local to one CPU/GPU target."""

    target = _component(scenario, target_component_id)
    target_kind = _kind(target)
    memory_kind = {
        "cpu": "host_memory",
        "gpu": "hbm",
    }.get(target_kind)
    if memory_kind is None:
        raise ValueError(
            "runtime memory target {} must be a CPU or GPU, not {}".format(
                target_component_id, target.kind
            )
        )
    memory_component_id = _nearest_profile_component_id(
        scenario,
        target_component_id,
        memory_kind,
    )
    if memory_component_id is None:
        raise ValueError(
            "{} component {} has no reachable {} runtime-memory target".format(
                target_kind.upper(),
                target_component_id,
                memory_kind.replace("_", "-"),
            )
        )
    return memory_component_id


def _layer_local_runtime_memory_component_id(
    scenario: ScenarioConfig,
    rank: LogicalRank,
    target_component_id: Optional[str],
    configured_component_id: Optional[str],
) -> Optional[str]:
    """Rebind runtime state only when a layer executes away from its rank."""

    target = str(target_component_id or rank.component_id)
    target_kind = _kind(_component(scenario, target))
    if target == rank.component_id or target_kind not in {"cpu", "gpu"}:
        return (
            str(configured_component_id)
            if configured_component_id is not None
            else None
        )
    return _compute_local_runtime_memory_component_id(scenario, target)


def _runtime_memory_source_is_compute_local(
    scenario: ScenarioConfig,
    source_component_id: str,
    target_component_id: str,
) -> bool:
    """Return whether a runtime operand is covered by the target roofline."""

    source = str(source_component_id)
    target = str(target_component_id)
    if source == target:
        return True
    target_kind = _kind(_component(scenario, target))
    if target_kind not in {"cpu", "gpu"}:
        return False
    return source == _compute_local_runtime_memory_component_id(
        scenario,
        target,
    )


def _nearest_non_cim_compute(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    rank: LogicalRank,
    source_component_id: str,
) -> Optional[str]:
    """Return the legacy deterministic CPU/GPU suggestion without applying it."""

    candidates: List[str] = []
    try:
        _gpu_profiles(
            scenario,
            rank.component_id,
            rank.memory_component_id,
        )
    except (KeyError, TypeError, ValueError):
        pass
    else:
        candidates.append(rank.component_id)
    for component in scenario.hardware.components:
        if _kind(component) != "cpu":
            continue
        try:
            _cpu_profiles(scenario, component.component_id)
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(component.component_id)
    ranked: List[Tuple[float, str]] = []
    for component_id in dict.fromkeys(candidates):
        try:
            outward = router.route(source_component_id, component_id, 1)
            inward = router.route(component_id, source_component_id, 1)
        except ValueError:
            continue
        ranked.append(
            (
                sum(hop.latency_ns for hop in outward)
                + sum(hop.latency_ns for hop in inward),
                component_id,
            )
        )
    return min(ranked)[1] if ranked else None


def _mapping_resolution_message(details: Mapping[str, object]) -> str:
    resolved = details.get("resolved_target")
    return (
        "non-GEMM operator {} ({}) requests CIM target {}; compatible "
        "target suggestion: {}; resolution_applied=false"
    ).format(
        details.get("operator_id", details.get("sub_operator_id", "unknown")),
        details.get("operator_class", "unknown"),
        details.get("requested_target", "unknown"),
        resolved if resolved is not None else "none",
    )


def _non_gemm_cim_mapping_diagnostic(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    rank: Optional[LogicalRank],
    operator_class: OperatorClass,
    operator_id: str,
    requested_target: str,
    requested_mapping_key: str,
) -> Dict[str, object]:
    resolved_target = (
        _nearest_non_cim_compute(
            scenario,
            router,
            rank,
            requested_target,
        )
        if rank is not None
        else None
    )
    details: Dict[str, object] = {
        "code": "non_gemm_cim_target",
        "operator_id": operator_id,
        "sub_operator_id": operator_id,
        "operator_class": operator_class.value,
        "requested_target": requested_target,
        "resolved_target": resolved_target,
        "resolution_applied": False,
        "requested_mapping_key": requested_mapping_key,
    }
    if rank is not None:
        details["rank_id"] = rank.rank
    details["message_en"] = _mapping_resolution_message(details)
    return details


def non_gemm_cim_mapping_diagnostics(
    scenario: ScenarioConfig,
) -> Tuple[Mapping[str, object], ...]:
    """Diagnose every typed primitive whose effective target is CIM."""

    components = _component_map(scenario)
    try:
        router = _topology_router(scenario)
        plan = _parallel_plan(scenario)
    except (KeyError, TypeError, ValueError):
        router = TopologyRouter(
            scenario.hardware,
            coherent_dma_mode=_coherent_dma_mode(scenario),
        )
        plan = None
    layers_by_id = {
        layer.layer_id: layer for layer in _execution_layers(scenario)
    }
    diagnostics: List[Mapping[str, object]] = []
    for (
        layer_id,
        operator_id,
        operator_class,
        fallback_keys,
    ) in _typed_primitive_mapping_uses(scenario):
        requested_mapping_key: Optional[str] = None
        if operator_id in scenario.placement.op_to_component:
            requested_mapping_key = operator_id
        else:
            requested_mapping_key = next(
                (
                    key
                    for key in fallback_keys
                    if key in scenario.placement.op_to_component
                ),
                None,
            )
        if requested_mapping_key is None:
            continue
        requested_target = str(
            scenario.placement.op_to_component[requested_mapping_key]
        )
        component = components.get(requested_target)
        if component is None or not _is_cim(component):
            continue
        layer = layers_by_id.get(layer_id)
        layer_ranks = (
            tuple(
                sorted(
                    plan.ranks_for_layer(layer),
                    key=lambda item: item.rank,
                )
            )
            if plan is not None and layer is not None
            else ()
        )
        rank = layer_ranks[0] if layer_ranks else None
        diagnostics.append(
            _non_gemm_cim_mapping_diagnostic(
                scenario,
                router,
                rank,
                operator_class,
                operator_id,
                requested_target,
                requested_mapping_key,
            )
        )
    return tuple(diagnostics)


def _primitive_target(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    rank: LogicalRank,
    operator_class: OperatorClass,
    mapping_key: str,
    *,
    fallback_keys: Sequence[str] = (),
) -> str:
    """Resolve an exact primitive mapping with explicit-key precedence."""

    configured = scenario.placement.op_to_component.get(mapping_key)
    if configured is None:
        configured = next(
            (
                scenario.placement.op_to_component[key]
                for key in fallback_keys
                if key in scenario.placement.op_to_component
            ),
            rank.component_id,
        )
    component_id = str(configured)
    component = _component(scenario, component_id)
    if _is_cim(component) and operator_class != OperatorClass.GEMM:
        requested_mapping_key = (
            mapping_key
            if mapping_key in scenario.placement.op_to_component
            else next(
                (
                    key
                    for key in fallback_keys
                    if key in scenario.placement.op_to_component
                ),
                mapping_key,
            )
        )
        details = _non_gemm_cim_mapping_diagnostic(
            scenario,
            router,
            rank,
            operator_class,
            mapping_key,
            component_id,
            requested_mapping_key,
        )
        raise MappingResolutionError(
            _mapping_resolution_message(details),
            details,
        )
    if _kind(component) == "cpu":
        _cpu_profiles(scenario, component_id)
        return component_id
    if _kind(component) == "gpu":
        return rank.component_id
    if _kind(component) not in {"gpu", "cpu"} and not _is_cim(component):
        return rank.component_id
    return component_id


def _fusion_allowed(
    scenario: ScenarioConfig,
    group: str,
    target_component_id: str,
    working_set_bytes: int,
) -> Tuple[bool, Mapping[str, object]]:
    """Return an explicit, auditable V4 GPU fusion decision."""

    enabled = bool(getattr(scenario.fusion_policy, group))
    target = _component(scenario, target_component_id)
    is_gpu = _kind(target) == "gpu"
    hardware_limit = (
        max(
            level.capacity_bytes
            for level in _resolve_component_profile(
                scenario, target_component_id, GPUProfile
            ).cache_hierarchy.levels
        )
        if is_gpu
        else 0
    )
    configured_limit = int(
        scenario.fusion_policy.max_fused_working_set_bytes
    )
    limit = (
        min(hardware_limit, configured_limit)
        if configured_limit > 0
        else hardware_limit
    )
    fits = max(0, int(working_set_bytes)) <= limit
    allowed = enabled and is_gpu and fits
    if not enabled:
        reason = "disabled_by_policy"
    elif not is_gpu:
        reason = "target_is_not_gpu"
    elif not fits:
        reason = "working_set_exceeds_sram_limit"
    else:
        reason = "enabled_same_gpu_sram_resident"
    return allowed, {
        "fusion_group": group,
        "fusion_enabled": allowed,
        "fusion_decision": reason,
        "fusion_working_set_bytes": max(0, int(working_set_bytes)),
        "fusion_sram_limit_bytes": limit,
        "fusion_target_component": target_component_id,
    }


def _same_rank_gpu_fusion_decision(
    scenario: ScenarioConfig,
    group: str,
    rank: LogicalRank,
    working_set_bytes: int,
    member_targets: Sequence[Tuple[str, str]],
) -> Tuple[bool, Mapping[str, object]]:
    anchor_target = (
        member_targets[0][1]
        if member_targets
        else rank.component_id
    )
    allowed, raw_audit = _fusion_allowed(
        scenario,
        group,
        anchor_target,
        working_set_bytes,
    )
    member_target_map = {
        op_key: target_component_id
        for op_key, target_component_id in member_targets
    }
    rank_is_gpu = _kind(_component(scenario, rank.component_id)) == "gpu"
    same_rank_gpu = rank_is_gpu and all(
        target_component_id == rank.component_id
        for _op_key, target_component_id in member_targets
    )
    audit = dict(raw_audit)
    audit.update(
        {
            "fusion_rank_component": rank.component_id,
            "fusion_member_targets": member_target_map,
            "co_located_op_keys": tuple(member_target_map),
            "colocation_required": allowed,
            "fusion_members_on_same_rank_gpu": same_rank_gpu,
        }
    )
    if allowed and not same_rank_gpu:
        allowed = False
        audit.update(
            {
                "fusion_enabled": False,
                "fusion_decision": "fusion_members_not_on_same_rank_gpu",
            }
        )
    return allowed, audit


def _residual_norm_fusion_decision(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    rank: LogicalRank,
    layer: LayerSpec,
    hidden_elements: int,
) -> Tuple[bool, Mapping[str, object]]:
    residual_key = "{}.{}.residual".format(
        layer.layer_id,
        "linear_attention" if layer.is_linear_attention else "attention",
    )
    residual_target = _primitive_target(
        scenario,
        router,
        rank,
        OperatorClass.ELEMENTWISE,
        residual_key,
        fallback_keys=("{}.norm".format(layer.layer_id),),
    )
    reduce_target = _primitive_target(
        scenario,
        router,
        rank,
        OperatorClass.REDUCTION,
        "{}.post_attention_norm.reduce".format(layer.layer_id),
        fallback_keys=("{}.norm".format(layer.layer_id),),
    )
    apply_target = _primitive_target(
        scenario,
        router,
        rank,
        OperatorClass.ELEMENTWISE,
        "{}.post_attention_norm.apply".format(layer.layer_id),
        fallback_keys=("{}.norm".format(layer.layer_id),),
    )
    allowed, audit = _same_rank_gpu_fusion_decision(
        scenario,
        "residual_norm",
        rank,
        _activation_bytes(layer, hidden_elements * 3, scenario=scenario),
        (
            (residual_key, residual_target),
            (
                "{}.post_attention_norm.reduce".format(layer.layer_id),
                reduce_target,
            ),
            (
                "{}.post_attention_norm.apply".format(layer.layer_id),
                apply_target,
            ),
        ),
    )
    audit = dict(audit)
    audit.update(
        {
            "fusion_residual_target": residual_target,
            "fusion_reduce_target": reduce_target,
            "fusion_apply_target": apply_target,
        }
    )
    return allowed, audit


def _namespace_demand(
    scenario: ScenarioConfig,
    demand: ResourceDemand,
    *,
    rank: LogicalRank,
    target_component_id: str,
    memory_component_id: Optional[str] = None,
) -> ResourceDemand:
    def namespace() -> ResourceDemand:
        resource_id = demand.resource_id
        target = _component(scenario, target_component_id)
        if _kind(target) == "gpu":
            gpu_profile, hbm_profile = _gpu_profiles(
                scenario,
                target_component_id,
                rank.memory_component_id,
            )
            gpu_resources = {
                gpu_profile.tensor_core.resource_id,
                gpu_profile.scalar_resource_id,
                gpu_profile.special_function_resource_id,
                gpu_profile.launch_resource_id,
                *(
                    level.resource_id
                    for level in gpu_profile.cache_hierarchy.levels
                ),
            }
            if resource_id in gpu_resources:
                resource_id = _rank_gpu_resource(
                    scenario, rank, resource_id
                )
            elif resource_id == hbm_profile.resource_id:
                resource_id = _rank_memory_resource(scenario, rank)
        elif _is_cim(target):
            base_cim = next(
                (
                    component.component_id
                    for component in scenario.hardware.components
                    if _is_cim(component)
                ),
                target_component_id,
            )
            if target_component_id != base_cim and resource_id.startswith(
                base_cim + "."
            ):
                resource_id = (
                    target_component_id + resource_id[len(base_cim) :]
                )
        elif _kind(target) == "cpu":
            cpu_profile, host_memory_profile = _cpu_profiles(
                scenario,
                target_component_id,
                memory_component_id=memory_component_id,
            )
            cpu_resources = {
                cpu_profile.pipeline.resource_id,
                *(
                    level.resource_id
                    for level in cpu_profile.cache_hierarchy.levels
                ),
            }
            reference_cpu = (
                scenario.host_orchestration_profile.cpu_component_id
            )
            if resource_id in cpu_resources:
                resource_id = _component_resource_id(
                    resource_id,
                    reference_component_id=reference_cpu,
                    target_component_id=target_component_id,
                )
            elif resource_id == host_memory_profile.resource_id:
                resource_id = _component_resource_id(
                    resource_id,
                    reference_component_id=reference_cpu,
                    target_component_id=target_component_id,
                )
        return ResourceDemand(
            resource_id=resource_id,
            service_ns=demand.service_ns,
            bytes_moved=demand.bytes_moved,
            energy_pj=demand.energy_pj,
            work_units=demand.work_units,
        )

    context = _active_compilation_context(scenario)
    if context is None:
        return namespace()
    value = context.leaf(
        (
            "namespace_demand",
            demand,
            rank,
            target_component_id,
            memory_component_id,
        ),
        namespace,
    )
    if not isinstance(value, ResourceDemand):  # pragma: no cover - defensive
        raise AssertionError("demand memo returned a non-ResourceDemand value")
    return value


def _parallel_target(
    scenario: ScenarioConfig, layer: LayerSpec, group: str, rank: LogicalRank
) -> str:
    configured = _resolve_target(scenario, layer, group)
    component = _component(scenario, configured)
    if _is_cim(component):
        return rank.cim_component_id or configured
    if _kind(component) == "cpu":
        _cpu_profiles(scenario, str(configured))
    if _kind(component) == "cpu":
        return str(configured)
    return rank.component_id


def _parallel_named_target(
    scenario: ScenarioConfig,
    mapping_key: str,
    rank: LogicalRank,
    *,
    fallback_keys: Sequence[str] = (),
) -> str:
    configured = scenario.placement.op_to_component.get(mapping_key)
    if configured is None:
        configured = next(
            (
                scenario.placement.op_to_component[key]
                for key in fallback_keys
                if key in scenario.placement.op_to_component
            ),
            None,
        )
    if configured is None:
        return rank.component_id
    component_id = str(configured)
    component = _component(scenario, component_id)
    if _is_cim(component):
        return rank.cim_component_id or component_id
    if _kind(component) == "cpu":
        _cpu_profiles(scenario, component_id)
    if _kind(component) == "cpu":
        return component_id
    return rank.component_id


def _declared_physical_projections(
    scenario: ScenarioConfig, layer: LayerSpec, projection_ids: Sequence[str],
    *, combined_projection_id: Optional[str] = None,
) -> bool:
    """Use physical call boundaries only for the declared backend contract."""

    if not _serving_bool_capability(
        scenario, ("llama_cpp_physical_projection_invocations",)
    ):
        return False
    root = layer.metadata.get("weight_projection_descriptors")
    if root is None:
        return False
    if not isinstance(root, Mapping) or not isinstance(root.get("projections"), Mapping):
        raise ValueError("weight_projection_descriptors.projections must be a mapping")
    present = [key in root["projections"] for key in projection_ids]
    if any(present) and not all(present):
        raise ValueError("physical projection group is incomplete: {}".format(", ".join(projection_ids)))
    if all(present):
        segments = []
        for key in projection_ids:
            resolved = resolve_weight_projection(layer.metadata, key)
            if resolved is None or len(resolved) != 1:
                raise ValueError("physical projection must have exactly one segment: " + key)
            segments.extend(resolved)
        if len({segment.physical_tensor_name for segment in segments}) != len(segments):
            raise ValueError("physical projection group repeats a weight tensor")
        if combined_projection_id is not None and tuple(segments) != resolve_weight_projection(
            layer.metadata, combined_projection_id
        ):
            raise ValueError("split projections differ from " + combined_projection_id)
    return all(present)


def _mmvq_activation_conversion_workload(
    scenario: ScenarioConfig, workload: GemmWorkload, *, max_m: int = 4
) -> Optional[TensorKernelWorkload]:
    """Lower the declared F32-to-Q8_1 MMVQ input conversion minimum."""

    if not _serving_bool_capability(scenario, ("llama_cpp_f32_q8_1_mmvq",)):
        return None
    formats = {value.upper() for value in workload.packed_weight_formats}
    if (
        not 1 <= workload.m <= max_m or workload.k % 32
        or not formats
        or not formats.issubset({
            "Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ3_S", "IQ4_XS",
        })
    ):
        return None
    padded_k = ((workload.k + 511) // 512) * 512
    padded_elements = workload.m * padded_k
    # One absolute value and two five-round 32-lane floating reductions.
    # Shuffle, normalization, rounding and type-conversion cycles are unknown.
    return TensorKernelWorkload(
        operations=11 * padded_elements,
        read_bytes=4 * workload.m * workload.k,
        write_bytes=36 * padded_elements // 32,
        streaming_fraction=1.0,
        name="llama_cpp_mmvq_f32_to_q8_1",
    )



def _cpu_gemm_cost_key(
    target_component_id: str,
    workload: GemmWorkload,
    dispatch: Optional[CPUIQPanelDispatch] = None,
) -> Tuple[object, ...]:
    # Keep the historical key unchanged unless native dispatch facts are used.
    key = ("cpu_gemm", target_component_id, workload)
    return key if dispatch is None else key + (dispatch,)


def _cpu_iq_panel_weight_bindings(
    scenario: ScenarioConfig,
) -> Mapping[str, Optional[Mapping[str, object]]]:
    """Index imported physical tensor evidence once, without reading live files."""
    def resolve():
        bindings: Dict[str, Optional[Mapping[str, object]]] = {}
        sources = []
        for layer in _execution_layers(scenario):
            values = layer.metadata.get("gguf_tensor_bindings", ())
            if isinstance(values, (tuple, list)):
                sources.extend(values)
        model_metadata = scenario.model.metadata
        metadata_sources = [model_metadata]
        if isinstance(model_metadata.get("metadata"), Mapping):
            metadata_sources.append(model_metadata["metadata"])
        for metadata in metadata_sources:
            sources.extend(metadata.get(key) for key in (
                "gguf_embedding_binding", "gguf_output_binding",
            ))
        for binding in sources:
            if not isinstance(binding, Mapping):
                continue
            name = binding.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in bindings and bindings[name] != binding:
                bindings[name] = None  # Conflicting source records cannot prove a layout.
            elif name not in bindings:
                bindings[name] = binding
        return bindings

    context = _active_compilation_context(scenario)
    return resolve() if context is None else context.invariant(
        ("cpu_iq_panel_weight_bindings",), resolve
    )


def _llama_source_cpu_iq_panel_dispatch(
    scenario: ScenarioConfig,
    workload: GemmWorkload,
    target: ComponentSpec,
    operation_metadata: Mapping[str, object],
    *,
    model_weight_read: bool,
    rhs_is_activation: bool,
) -> Tuple[Optional[CPUIQPanelDispatch], Optional[Mapping[str, object]]]:
    """Prove ordinary F32 x physical GGUF weight geometry before the cost gate.

    The adapter declares source provenance and the experimental switch only.
    Tensor dtype/layout facts must come from this actual planner invocation.
    """
    raw = scenario.workload.metadata.get("llama_cpp_cpu_iq_panel_reuse")
    if raw is None or (isinstance(raw, Mapping) and raw.get("enabled", False) is False):
        return None, None
    audit = {
        "schema": "heterollm.cpu-iq-panel-reuse/v1",
        "status": "uncovered", "applied": False, "native_dispatch_proven": False,
        "default_unset_assumption_used": False,
        "m": workload.m, "k": workload.k, "n": workload.n,
        "execution_component": target.component_id,
    }

    def uncovered(reason: str):
        return None, {**audit, "reason": reason, "rejection_reasons": (reason,)}

    def valid_sha(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None

    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return uncovered("invalid_iq_panel_experiment_contract")
    audit.update(
        environment_state=raw.get("no_iq_panel_environment_state", "unknown"),
        assume_default_unset=raw.get("assume_default_unset", False),
        source_sha256=raw.get("source_sha256", ""),
        cpu_backend_sha256=raw.get("cpu_backend_sha256", ""),
    )
    if (not model_weight_read or rhs_is_activation
            or operation_metadata.get("dynamic_rhs") is True
            or operation_metadata.get("rhs_operand_kind") == "activation"):
        return uncovered("runtime_rhs_is_not_a_physical_model_weight")
    if (
        "expert_index" in operation_metadata
        or operation_metadata.get("coverage_component") in {"routed_expert", "shared_expert"}
        or operation_metadata.get("ffn_path") in {"routed", "shared"}
    ):
        return uncovered("expert_or_scatter_layout")
    if (
        workload.epilogue_operations or workload.epilogue_transcendental_operations
        or workload.epilogue_output_elements or workload.epilogue_name
        or operation_metadata.get("fused") is True
        or (operation_metadata.get("fusion_group") not in (None, "")
            and operation_metadata.get("fusion_enabled") is not False)
    ):
        return uncovered("fused_epilogue_has_no_native_iq_panel_contract")
    formats = tuple(value.upper() for value in workload.packed_weight_formats)
    if len(formats) != 1 or formats[0] not in {"IQ3_S", "IQ4_XS"}:
        return uncovered("unsupported_or_mixed_physical_weight_format")
    segments = operation_metadata.get("projection_segments", ())
    if (
        operation_metadata.get("weight_projection_descriptor_applied") is not True
        or operation_metadata.get("projection_segment_count") != 1
        or not isinstance(segments, (tuple, list)) or len(segments) != 1
        or not isinstance(segments[0], Mapping)
    ):
        return uncovered("one_physical_projection_not_proven")
    segment = segments[0]
    if (
        segment.get("local_k") != workload.k or segment.get("global_k") != workload.k
        or segment.get("local_n") != workload.n or segment.get("global_n") != workload.n
        or str(segment.get("format", "")).upper() != formats[0]
        or not isinstance(segment.get("physical_tensor_name"), str)
        or not segment.get("physical_tensor_name")
    ):
        return uncovered("physical_projection_shape_or_format_mismatch")
    tensor_name = segment["physical_tensor_name"]
    audit.update(weight_format=formats[0], physical_tensor_name=tensor_name)
    model_sha = _model_gguf_sha256(scenario.model)
    if not valid_sha(model_sha):
        return uncovered("source_model_identity_missing")
    audit["model_sha256"] = model_sha
    binding = _cpu_iq_panel_weight_bindings(scenario).get(tensor_name)
    if binding is None:
        return uncovered("source_tensor_binding_missing_or_ambiguous")
    shape = binding.get("shape")
    if (
        not isinstance(shape, (tuple, list)) or len(shape) != 2
        or any(type(value) is not int for value in shape)
        or tuple(shape) != (workload.k, workload.n)
        or str(binding.get("type", "")).upper() != formats[0]
    ):
        return uncovered("source_tensor_shape_or_format_mismatch")
    spec = _ARTIFACT_QUANTIZATION_REGISTRY[formats[0]]
    physical_bytes = workload.n * ((workload.k + spec.block_size - 1) // spec.block_size) * (
        spec.payload_bytes + spec.metadata_bytes
    )
    if (
        type(binding.get("offset")) is not int or binding["offset"] < 0
        or binding.get("block_size", spec.block_size) != spec.block_size
        or binding.get("n_bytes") != physical_bytes
        or segment.get("physical_bytes") != physical_bytes
        or segment.get("local_physical_bytes") != physical_bytes
        or workload.weight_bytes != physical_bytes
    ):
        return uncovered("source_tensor_physical_storage_mismatch")
    if (
        not _f32_hidden_storage_enabled(scenario)
        or workload.activation_bytes != 4 * workload.m * workload.k
        or workload.output_bytes != 4 * workload.m * workload.n
        or workload.accumulator_bits != 32
    ):
        return uncovered("ordinary_f32_input_output_not_proven")
    refs = raw.get("source_refs", ())
    if (
        not valid_sha(raw.get("source_sha256"))
        or not valid_sha(raw.get("cpu_backend_sha256"))
        or not isinstance(refs, (tuple, list)) or not refs
        or any(not isinstance(value, str) or not value for value in refs)
    ):
        return uncovered("source_or_backend_identity_missing")
    try:
        dispatch = CPUIQPanelDispatch(
            compiled_avx2=raw.get("compiled_avx2", False),
            source_activation_dtype="F32", source_output_dtype="F32",
            source_weight_layout="ordinary_contiguous_2d", source_activation_ne3=1,
            no_iq_panel_environment_state=raw.get("no_iq_panel_environment_state", "unknown"),
            assume_default_unset=raw.get("assume_default_unset", False),
            source_sha256=raw["source_sha256"], cpu_backend_sha256=raw["cpu_backend_sha256"],
            source_refs=tuple(refs),
        )
    except (TypeError, ValueError):
        return uncovered("invalid_iq_panel_dispatch_facts")
    return dispatch, {**audit, "tensor_evidence": "imported_gguf_ordinary_2d_matrix",
                      "source_tensor_offset": binding["offset"], "physical_bytes": physical_bytes}


def summarize_cpu_iq_panel_reuse(tasks: Sequence[TaskSpec]) -> Mapping[str, object]:
    """Count physical CPU GEMMs, excluding dispatch phases and cache lookups."""
    total = audited = applied = conditional = proven = 0
    reasons: Dict[str, int] = {}
    for task in tasks:
        if task.metadata.get("phase") != "cpu_gemm":
            continue
        total += 1
        audit = task.metadata.get("cpu_iq_panel_reuse")
        if not isinstance(audit, Mapping):
            continue
        audited += 1
        if audit.get("applied") is True:
            applied += 1
            conditional += int(audit.get("default_unset_assumption_used") is True)
            proven += int(audit.get("native_dispatch_proven") is True)
        else:
            for reason in set(audit.get("rejection_reasons", ())):
                reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "schema": "heterollm.cpu-iq-panel-reuse-coverage/v1",
        "counting_unit": "physical_cpu_gemm_task",
        "cpu_gemm_tasks": total, "audited_tasks": audited, "applied_tasks": applied,
        "conditional_tasks": conditional, "native_dispatch_proven_tasks": proven,
        "uncovered_tasks": audited - applied, "uncovered_reason_counts": dict(sorted(reasons.items())),
    }


def _declared_mmq_work(
    scenario: ScenarioConfig,
    workload: GemmWorkload,
    target: ComponentSpec,
    gpu_profile: GPUProfile,
    operation_metadata: Mapping[str, object],
    *,
    model_weight_read: bool,
    rhs_is_activation: bool,
) -> Tuple[Optional[MMQWork], Optional[Mapping[str, object]], int]:
    """Bind a physical projection to the fixed, explicitly declared backend.

    The final integer is the source MMVQ limit when that earlier dispatch wins.
    Missing coverage remains visible; it does not change the old partial cost.
    """
    enabled = scenario.workload.metadata.get("llama_cpp_mmq_source_work", False)
    if not isinstance(enabled, bool):
        raise ValueError("llama_cpp_mmq_source_work must be an explicit boolean")
    if not enabled:
        return None, None, 0
    audit = {
        "schema": "heterollm.cuda-mmq-source-work/v1",
        "status": "uncovered",
        "m": workload.m, "k": workload.k, "n": workload.n,
        "execution_component": target.component_id,
    }

    def uncovered(reason: str):
        return None, {**audit, "reason": reason}, 0

    if not model_weight_read or rhs_is_activation:
        return uncovered("runtime_rhs_is_not_a_physical_model_weight")
    formats = tuple(f.upper() for f in workload.packed_weight_formats)
    if len(formats) != 1 or formats[0] not in MMVQ_MAX_BATCH_SIZE:
        return uncovered("unsupported_or_mixed_physical_weight_format")
    if (
        "expert_index" in operation_metadata
        or operation_metadata.get("coverage_component") in {"routed_expert", "shared_expert"}
        or operation_metadata.get("ffn_path") in {"routed", "shared"}
    ):
        return uncovered("expert_or_scatter_layout")
    segments = operation_metadata.get("projection_segments", ())
    if (
        operation_metadata.get("projection_segment_count") != 1
        or not isinstance(segments, (tuple, list)) or len(segments) != 1
        or not isinstance(segments[0], Mapping)
    ):
        return uncovered("one_physical_projection_not_proven")
    segment = segments[0]
    if (
        segment.get("local_k") != workload.k
        or segment.get("local_n") != workload.n
        or str(segment.get("format", "")).upper() != formats[0]
        or not segment.get("physical_tensor_name")
    ):
        return uncovered("physical_projection_shape_or_format_mismatch")
    if (
        not _f32_hidden_storage_enabled(scenario)
        or workload.activation_bytes != 4 * workload.m * workload.k
        or workload.output_bytes != 4 * workload.m * workload.n
        or workload.accumulator_bits != 32
    ):
        return uncovered("ordinary_f32_input_output_not_proven")
    if (workload.epilogue_operations or workload.epilogue_transcendental_operations
            or workload.epilogue_output_elements or workload.epilogue_name):
        return uncovered("fused_epilogue_has_no_native_mmq_contract")
    contract = target.metadata.get("llama_cpp_mmq_contract")
    if not isinstance(contract, Mapping):
        return uncovered("missing_fixed_backend_dispatch_contract")
    if (
        contract.get("backend_commit") != "0f3a71be15af836d277c9f918adfafb45732677e"
        or target.metadata.get("cuda_compute_capability") != 1200
        or contract.get("compiled_int8_mma") is not True
        or contract.get("force_cublas") is not False
        or contract.get("ordinary_contiguous_2d") is not True
    ):
        return uncovered("unsupported_backend_architecture_or_layout_declaration")
    shared = contract.get("max_shared_memory_per_block_optin_bytes")
    if type(shared) is not int or shared < 48 * 1024:
        return uncovered("missing_or_insufficient_per_block_shared_memory")
    audit.update(
        weight_format=formats[0],
        physical_tensor_name=segment["physical_tensor_name"],
        backend_commit=contract["backend_commit"],
        shared_memory_per_block=shared,
        sm_count=gpu_profile.sm_count,
    )
    mmvq_limit = MMVQ_MAX_BATCH_SIZE[formats[0]]
    if workload.m <= mmvq_limit:
        return None, {**audit, "status": "mmvq_precedes_mmq", "mmvq_max_m": mmvq_limit}, mmvq_limit
    try:
        work = derive_mmq_work(
            m=workload.m, k=workload.k, n=workload.n,
            weight_format=formats[0], sm_count=gpu_profile.sm_count,
            shared_memory_per_block=shared,
        )
    except UnsupportedMMQ as error:
        return uncovered(str(error))
    return work, {**audit, **work.to_metadata(), "status": "applied"}, 0


def _declared_mmvq_prmt_partial_work(
    scenario: ScenarioConfig,
    workload: GemmWorkload,
    target: ComponentSpec,
    operation_metadata: Mapping[str, object],
    gpu_profile: GPUProfile,
) -> Tuple[GemmWorkload, Optional[Mapping[str, object]]]:
    """Replace the old unpack proxy with source-qualified PRMT work only.

    DP4A is deliberately not priced here: the current-driver migration probe
    did not pass its stability gate.  The retained tensor-core/memory phases
    therefore remain the existing analytical path; this helper only replaces
    the opaque packed-transform proxy when the fixed source geometry and the
    current-driver PRMT contract are all explicit.
    """

    raw = scenario.hardware.metadata.get(
        "llama_cpp_mmvq_prmt_partial_contract"
    )
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return workload, None
    if _kind(target) != "gpu" or not operation_metadata.get("model_weight_read", True):
        return workload, {"status": "uncovered", "reason": "not_gpu_model_weight"}
    formats = tuple(value.upper() for value in workload.packed_weight_formats)
    if len(formats) != 1 or formats[0] not in {"Q4_K", "IQ4_XS"}:
        return workload, {"status": "uncovered", "reason": "format_not_in_prmt_scope"}
    if workload.m not in {2, 4} or workload.k % 1024 != 0 or workload.n % 2 != 0:
        return workload, {"status": "uncovered", "reason": "shape_outside_m2_m4_full_step_scope"}
    segments = workload.packed_weight_format_segments
    if len(segments) != 1 or segments[0][1] != workload.n:
        return workload, {"status": "uncovered", "reason": "physical_projection_not_proven"}
    if str(segments[0][0]).upper() != formats[0]:
        return workload, {"status": "uncovered", "reason": "physical_shape_or_format_mismatch"}
    if not operation_metadata.get("projection_id"):
        return workload, {"status": "uncovered", "reason": "projection_identity_not_proven"}
    if (
        operation_metadata.get("fused") is True
        or operation_metadata.get("fusion_group") not in (None, "")
        or operation_metadata.get("expert_index") is not None
        or operation_metadata.get("coverage_component") in {"routed_expert", "shared_expert"}
        or operation_metadata.get("ffn_path") in {"routed", "shared"}
    ):
        return workload, {"status": "uncovered", "reason": "fusion_or_expert_layout"}
    if raw.get("source_revision") != "0f3a71be15af836d277c9f918adfafb45732677e":
        return workload, {"status": "uncovered", "reason": "source_revision_mismatch"}
    if raw.get("driver_sha256") != "ba82badab512a9a0723e0bd380f391392405db38b7fab8ae6c75eeb358f1e3ad":
        return workload, {"status": "uncovered", "reason": "driver_identity_mismatch"}
    if raw.get("device_uuid") != "83b80720-113c-3f3d-c624-f1dc642b3f8b":
        return workload, {"status": "uncovered", "reason": "device_identity_mismatch"}
    if raw.get("sm_count") != gpu_profile.sm_count:
        return workload, {"status": "uncovered", "reason": "sm_count_mismatch"}
    rate = raw.get("prmt_thread_inst_per_sm_cycle")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or float(rate) <= 0.0:
        return workload, {"status": "uncovered", "reason": "missing_prmt_capacity"}
    frequency_ghz = float(gpu_profile.tensor_core.frequency_ghz)
    if frequency_ghz <= 0.0:
        return workload, {"status": "uncovered", "reason": "missing_gpu_frequency"}
    rows_per_warp_step = 2
    blocks_per_warp_step = 2 if formats[0] == "Q4_K" else 4
    warp_steps = (workload.n // rows_per_warp_step) * (
        workload.k // 256 // blocks_per_warp_step
    )
    prmt_slots = 32 * warp_steps * (
        4 * rows_per_warp_step * workload.m
        if formats[0] == "Q4_K"
        else 32 * rows_per_warp_step
    )
    service_ns = prmt_slots / (float(gpu_profile.sm_count) * float(rate) * frequency_ghz)
    if service_ns <= 0.0:
        return workload, {"status": "uncovered", "reason": "zero_prmt_work"}
    segments = tuple(
        (format_name, local_n, 0)
        for format_name, local_n, _operations in workload.packed_weight_format_segments
    )
    replaced = replace(
        workload,
        packed_weight_transform_operations=0,
        packed_weight_format_segments=segments,
        source_partial_service_ns=service_ns,
        source_partial_work_units=prmt_slots,
        source_partial_name="llama_cpp_mmvq_prmt_partial",
    )
    return replaced, {
        "status": "applied",
        "format": formats[0],
        "m": workload.m,
        "k": workload.k,
        "n": workload.n,
        "prmt_thread_slots": prmt_slots,
        "prmt_thread_inst_per_sm_cycle": float(rate),
        "sm_count": gpu_profile.sm_count,
        "frequency_ghz": frequency_ghz,
        "service_ns": service_ns,
        "replaced_old_packed_weight_transform_operations": workload.packed_weight_transform_operations,
        "source": raw.get("evidence"),
    }


def _execution_phase_from_name(name: object) -> Optional[str]:
    """Infer a serving phase from a fully-qualified task name.

    Serving task names carry a cohort prefix (for example,
    ``cohort-000001.decode.layer-000...``), so a ``startswith('decode')``
    check misses the phase and leaves calibration without an execution
    bucket.  Match explicit dotted path components while ignoring unrelated
    operator names that merely contain the word ``decode``.
    """

    text = str(name or "").strip().lower()
    qualified = ".{}.".format(text)
    if re.match(r"^prefill(?:\d+)?(?:\.|$)", text) or ".prefill." in qualified:
        return "prefill"
    if re.match(r"^decode(?:\d+)?(?:\.|$)", text) or ".decode." in qualified:
        return "decode"
    return None


def _add_rank_gemm(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    workload: GemmWorkload,
    target_component_id: str,
    name: str,
    dependencies: Sequence[str],
    *,
    model_weight_read: bool = True,
    weight_tensor_id: Optional[str] = None,
    activation_source_component_id: Optional[str] = None,
    dynamic_rhs_source_component_id: Optional[str] = None,
    dynamic_rhs: bool = False,
    keep_output_on_target: bool = False,
    metadata: Optional[Mapping[str, object]] = None,
    dynamic_attention_replay: Optional[
        _DynamicAttentionCostTaskReplayPayload
    ] = None,
) -> str:
    placement_component_id = str(target_component_id)
    prior = tuple(dependencies)
    activation_source = (
        builder.rank_value_component(prior, rank.rank)
        or activation_source_component_id
        or rank.component_id
    )
    iq_panel_audit = None
    operation_metadata = dict(metadata or {})
    operation_metadata.setdefault("operator_class", OperatorClass.GEMM.value)
    if _f32_hidden_storage_enabled(scenario):
        operation_metadata.update(
            runtime_input_storage_bits=32,
            runtime_output_storage_bits=32,
            runtime_output_storage_bytes=workload.output_bytes,
            matrix_compute_precision_policy="inherited_analytical_contract",
        )
    if weight_tensor_id is not None:
        operation_metadata.setdefault("weight_tensor_id", weight_tensor_id)
        operation_metadata.setdefault("tensor_id", weight_tensor_id)
    rhs_is_activation = (
        not model_weight_read
        and (
            dynamic_rhs
            or operation_metadata.get("rhs_operand_kind") == "activation"
        )
    )
    if rhs_is_activation:
        operation_metadata.setdefault("rhs_operand_kind", "activation")
        operation_metadata.setdefault("dynamic_rhs", True)
    quantization_metadata = _workload_quantization_metadata(
        scenario,
        workload,
        name,
        operation_metadata,
        dynamic_rhs=rhs_is_activation,
    )
    # Operation metadata is deliberately last so an explicit per-operation
    # annotation remains authoritative over derived workload facts.
    operation_metadata = {
        **quantization_metadata,
        **operation_metadata,
    }
    # Preserve the physical GEMM invocation geometry for native calibration.
    # Regular planner call sites do not carry ``gemm_m/gemm_n`` in their
    # hand-authored metadata, even though the typed workload already has the
    # exact [M, K] x [K, N] shape.  Exposing these dimensions lets the phase
    # boundary select an exact operator-wall bucket (for example
    # ``17408x1x1x1`` for a decode FFN) instead of silently using a phase-wide
    # average.  Explicit metadata remains authoritative for specialized
    # dynamic attention paths.
    operation_metadata.setdefault("gemm_m", workload.m)
    operation_metadata.setdefault("gemm_n", workload.n)
    operation_metadata.setdefault("token_batch", workload.m)
    operation_metadata.setdefault("prompt_tokens", builder.request.prompt_tokens)
    operation_metadata.setdefault("output_tokens", builder.request.output_tokens)
    weight_source, source_tensor, logical_tensor = _weight_source_for_tensor(
        scenario, weight_tensor_id, placement_component_id, rank=rank
    )
    offload_decision = _host_gemm_offload_decision(
        scenario,
        router,
        plan,
        rank,
        workload,
        placement_component_id,
        activation_source,
        weight_source,
        model_weight_read=model_weight_read,
        dynamic_rhs=rhs_is_activation,
    )
    if offload_decision is not None:
        target_component_id = offload_decision.execution_component_id
        offload_audit = offload_decision.audit_metadata
        if offload_decision.applied:
            offload_audit["temporary_weight_bytes"] = workload.weight_bytes
        operation_metadata.update(offload_audit)
    target = _component(scenario, target_component_id)
    weight_read_decision = _weight_backing_read_gate(
        scenario,
        weight_source,
        target_component_id,
        model_weight_read=model_weight_read,
    )
    if model_weight_read:
        invocation_id = "{}:{}:rank-{}:{}".format(
            builder.request.request_id,
            name,
            rank.rank,
            logical_tensor,
        )
        operation_metadata.update(
            weight_read_decision.audit_metadata(invocation_id)
        )
        operation_metadata["weight_read_bytes"] = workload.weight_bytes
    staged_weight_allocation_id: Optional[str] = None
    mmq_work: Optional[MMQWork] = None
    mmq_audit: Optional[Mapping[str, object]] = None
    staged_weight_metadata: Dict[str, object] = {}
    if offload_decision is not None and offload_decision.applied:
        allocation_name = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "-",
            "{}.{}.rank-{}.{}".format(
                builder.request.request_id,
                name,
                rank.rank,
                logical_tensor,
            ),
        )
        staged_weight_allocation_id = (
            "runtime.staged_weight.{:05d}.{}".format(
                builder.counter + 1,
                allocation_name,
            )
        )
        staged_weight_metadata = {
            **operation_metadata,
            "allocation_id": staged_weight_allocation_id,
            "staged_weight_allocation_id": staged_weight_allocation_id,
            "staged_weight_bytes": workload.weight_bytes,
            "temporary_weight_bytes": workload.weight_bytes,
            "weight_owner_component_id": weight_source,
            "weight_source_component": weight_source,
            "weight_target_component": target_component_id,
            "residency_component_id": rank.memory_component_id,
            "lifecycle": "temporary",
            "read_only": True,
            "fully_resident": True,
            "h2d_already_charged": True,
            "release_semantics": "clean_discard",
            "clean_eviction_service": "free_discard",
            "dirty_writeback": False,
            "rank": rank.rank,
        }
    extra_phase_demands: Dict[str, Tuple[ResourceDemand, ...]] = {}
    rank_local_weight_source = (
        _kind(target) == "gpu"
        and target_component_id == rank.component_id
        and weight_source in {
            rank.component_id,
            rank.memory_component_id,
        }
    )
    cpu_local_weight_source = (
        _kind(target) == "cpu"
        and weight_source is not None
        and _weight_source_is_compute_local_backing(
            scenario,
            weight_source,
            target_component_id,
        )
    )
    compute_local_weight_source = (
        rank_local_weight_source or cpu_local_weight_source
    )
    # Residency observes operator reads, not only topology transfers.  Emit one
    # marker even when the physical backing is already local or resident; any
    # route phases below retain the same invocation id and are deduplicated.
    if model_weight_read and weight_source:
        weight_access_metadata = {
            **operation_metadata,
            **_weight_transfer_metadata(
                source_tensor,
                logical_tensor,
                weight_source,
                target_component_id,
            ),
            # This zero-service marker feeds the residency ledger.  It is not
            # a physical backing read; only routed transfer phases use the
            # ``model_weight_read`` event kind and consume source bandwidth.
            "event_kind": "model_weight_access",
            "rank": rank.rank,
            "resource_accounting": (
                "included_in_gemm_roofline"
                if compute_local_weight_source
                else "weight_access_marker"
            ),
            "bytes": workload.weight_bytes,
        }
        if compute_local_weight_source:
            weight_access_metadata["compute_local_backing"] = (
                "cpu_attached_host_memory"
                if cpu_local_weight_source
                else "rank_local_accelerator_memory"
            )
        weight_access_metadata.setdefault("event_kind", "model_weight_read")
        weight_access_metadata = _communication_task_metadata(
            weight_access_metadata
        )
        prior = (
            builder.add(
                name + ".model_weight_read.access",
                TaskCategory.COMMUNICATION,
                dependencies=prior,
                advance=False,
                metadata=weight_access_metadata,
            ),
        )
    if (
        model_weight_read
        and _kind(target) in {"gpu", "cpu"}
        and weight_read_decision.emit_source_transfer
        and not compute_local_weight_source
    ):
        weight_transfer_target = target_component_id
        if staged_weight_allocation_id is not None:
            weight_transfer_target = _coherent_staged_weight_target(
                scenario,
                router,
                rank,
                weight_source,
                target_component_id,
                workload.weight_bytes,
                routing_policy=plan.routing_policy,
            )
        prior = (
            _add_transfer_tasks(
                builder,
                router,
                weight_source,
                weight_transfer_target,
                workload.weight_bytes,
                prior,
                name=name + ".model_weight_read",
                routing_policy=plan.routing_policy,
                metadata={
                    **operation_metadata,
                    **_weight_transfer_metadata(
                        source_tensor,
                        logical_tensor,
                        weight_source,
                        target_component_id,
                    ),
                    "rank": rank.rank,
                },
            ),
        )
    if (
        not _is_cim(target)
        and activation_source != target_component_id
        and workload.activation_bytes > 0
    ):
        prior = (
            _add_transfer_tasks(
                builder,
                router,
                activation_source,
                target_component_id,
                workload.activation_bytes,
                prior,
                name=name + ".activation_transfer",
                routing_policy=plan.routing_policy,
                metadata={
                    **operation_metadata,
                    "rank": rank.rank,
                    "event_kind": "operator_input_transfer",
                    "transport": "activation",
                },
            ),
        )
    dispatch_resource_id: Optional[str] = None
    dispatch_energy_pj = 0.0
    if _is_cim(target):
        cim_profile = _resolve_component_profile(
            scenario,
            target_component_id, DigitalSramCimProfile
        )
        dispatch_resource_id = cim_profile.load_resource_id
        prior = (
            _add_transfer_tasks(
                builder,
                router,
                activation_source,
                target_component_id,
                workload.activation_bytes,
                prior,
                name=name + ".activation_to_cim",
                routing_policy=plan.routing_policy,
                metadata={
                    **operation_metadata,
                    "rank": rank.rank,
                    "transport": "gpu-cim",
                },
            ),
        )
    dynamic_rhs_source = str(
        dynamic_rhs_source_component_id or rank.component_id
    )
    if (
        not model_weight_read
        and workload.weight_bytes > 0
        and not _runtime_memory_source_is_compute_local(
            scenario,
            dynamic_rhs_source,
            target_component_id,
        )
    ):
        # QK/PV use a runtime tensor as the RHS, not a model-weight tensor.
        # The GEMM estimator still names the RHS traffic ``weight_bytes``
        # because it describes the matrix role.  When execution is remote,
        # explicitly move that dynamic operand from the rank that owns the
        # materialized K/V activation; model-weight residency must never gate
        # this transfer.
        prior = (
            _add_transfer_tasks(
                builder,
                router,
                dynamic_rhs_source,
                target_component_id,
                workload.weight_bytes,
                prior,
                name=name + ".dynamic_rhs_transfer",
                routing_policy=plan.routing_policy,
                metadata={
                    **operation_metadata,
                    "rank": rank.rank,
                    "event_kind": "operator_input_transfer",
                    "transport": "dynamic_rhs",
                    "rhs_operand_kind": "activation",
                },
            ),
        )
    if _is_cim(target):
        weights_resident = scenario.weights_resident and model_weight_read
        estimate = _memoized_cost_estimate(
            scenario,
            (
                "cim_gemm",
                target_component_id,
                workload,
                weights_resident,
            ),
            lambda: estimate_cim_gemm(
                cim_profile,
                workload,
                weights_resident=weights_resident,
            ),
        )
        if model_weight_read and "weight_load" in estimate.phase_names:
            # One cold backing read materializes this GEMM's rank-local RHS
            # exactly once.  CIM tile/input reuse remains inside the active
            # cost model and must not multiply HBF/SSD traffic by batch M.
            byte_count = workload.weight_bytes
            if weight_read_decision.emit_source_transfer and weight_source:
                prior = (
                    _add_transfer_tasks(
                        builder,
                        router,
                        weight_source,
                        target_component_id,
                        byte_count,
                        prior,
                        name=name + ".model_weight_read",
                        routing_policy=plan.routing_policy,
                        metadata={
                            **operation_metadata,
                            **_weight_transfer_metadata(
                                source_tensor,
                                logical_tensor,
                                weight_source,
                                target_component_id,
                            ),
                            "rank": rank.rank,
                        },
                    ),
                )
            elif not weight_source:
                extra_phase_demands["weight_load"] = _shared_transfer_demands(
                    scenario, rank, byte_count
                )
    elif _kind(target) == "gpu":
        gpu_profile, hbm_profile = _gpu_profiles(
            scenario,
            target_component_id,
            rank.memory_component_id,
        )
        dispatch_resource_id = gpu_profile.launch_resource_id
        dispatch_energy_pj = gpu_profile.launch_energy_pj
        mmq_work, mmq_audit, mmvq_limit = _declared_mmq_work(
            scenario, workload, target, gpu_profile, operation_metadata,
            model_weight_read=model_weight_read, rhs_is_activation=rhs_is_activation,
        )
        if mmq_audit is not None:
            operation_metadata["mmq_source_work"] = {**mmq_audit, "stage": "matrix"}
        conversion = (
            _mmvq_activation_conversion_workload(scenario, workload, max_m=mmvq_limit or 4)
            if model_weight_read and not rhs_is_activation
            and target_component_id == rank.component_id
            and mmq_work is None
            and "expert_index" not in operation_metadata
            and operation_metadata.get("coverage_component") not in {"routed_expert", "shared_expert"}
            and operation_metadata.get("ffn_path") not in {"routed", "shared"}
            else None
        )
        if conversion is not None:
            consumer_bytes = 36 * workload.m * workload.k // 32
            conversion_audit = {
                "activation_conversion_applied": True,
                "activation_conversion_implementation": "llama_cpp_f32_q8_1_mmvq",
                "activation_conversion_bound": "one_per_lowered_projection",
                "activation_conversion_timing_completeness": "partial",
                "activation_conversion_input_rows": workload.m,
                "activation_conversion_input_columns": workload.k,
                "activation_conversion_padded_columns": ((workload.k + 511) // 512) * 512,
                "activation_conversion_read_bytes": conversion.read_bytes,
                "activation_conversion_write_bytes": conversion.write_bytes,
                "activation_conversion_scalar_operations": conversion.operations,
                "activation_conversion_consumer_bytes": consumer_bytes,
            }
            prior = (_add_rank_tensor_kernel(
                builder, scenario, router, plan, rank, conversion,
                name + ".activation_q8_1", prior,
                metadata={
                    "event_kind": "activation_quantization",
                    "operator_class": OperatorClass.ELEMENTWISE.value,
                    "layer_id": operation_metadata.get("layer_id"),
                    "projection_id": operation_metadata.get("projection_id"),
                    **conversion_audit,
                },
            ),)
            # Replace the consumer's input bytes; retaining the old activation
            # read here would count both source and temporary input in GEMM.
            workload = replace(workload, activation_storage_bytes=consumer_bytes)
            operation_metadata.update(conversion_audit)
        if mmq_work is not None:
            conversion = TensorKernelWorkload(
                operations=mmq_work.conversion_operations,
                read_bytes=mmq_work.conversion_read_bytes,
                write_bytes=mmq_work.conversion_write_bytes,
                streaming_fraction=1.0,
                name="llama_cpp_mmq_f32_input_repacking",
            )
            prior = (_add_rank_tensor_kernel(
                builder, scenario, router, plan, rank, conversion,
                name + ".activation_mmq", prior,
                execution_component_id=target_component_id, input_is_local=True,
                metadata={
                    "event_kind": "activation_quantization",
                    "operator_class": OperatorClass.ELEMENTWISE.value,
                    "layer_id": operation_metadata.get("layer_id"),
                    "projection_id": operation_metadata.get("projection_id"),
                    "mmq_source_work": {**dict(mmq_audit or {}), "stage": "conversion"},
                },
            ),)
            workload = replace(
                workload, activation_storage_bytes=mmq_work.consumer_unique_bytes,
                mmq_work=mmq_work,
            )
        if model_weight_read and not rhs_is_activation and mmq_work is None:
            workload, mmvq_prmt_audit = _declared_mmvq_prmt_partial_work(
                scenario,
                workload,
                target,
                operation_metadata,
                gpu_profile,
            )
            if mmvq_prmt_audit is not None:
                operation_metadata["mmvq_prmt_partial_work"] = mmvq_prmt_audit
        estimate = _memoized_cost_estimate(
            scenario,
            (
                "gpu_gemm",
                target_component_id,
                rank.memory_component_id,
                workload,
            ),
            lambda: estimate_gpu_gemm(gpu_profile, hbm_profile, workload),
        )
    elif _kind(target) == "cpu":
        cpu_profile, host_memory_profile = _cpu_profiles(
            scenario, target_component_id
        )
        # CPU packed-weight transform work is part of the capability-selected
        # instruction schedule in estimate_cpu_gemm().  Keeping the old
        # planner-side merge here would charge the same transform twice.
        dispatch_resource_id = cpu_profile.pipeline.resource_id
        dispatch_energy_pj = cpu_profile.dispatch_energy_pj
        iq_panel_dispatch, iq_panel_audit = _llama_source_cpu_iq_panel_dispatch(
            scenario, workload, target, operation_metadata,
            model_weight_read=model_weight_read, rhs_is_activation=rhs_is_activation,
        )
        estimate = _memoized_cost_estimate(
            scenario,
            _cpu_gemm_cost_key(target_component_id, workload, iq_panel_dispatch),
            lambda: estimate_cpu_gemm(
                cpu_profile, host_memory_profile, workload,
                iq_panel_dispatch=iq_panel_dispatch,
            ),
        )
        if iq_panel_audit is not None and iq_panel_dispatch is not None:
            compute_phase = next(phase for phase in estimate.phases if phase.name == "cpu_gemm")
            reuse = compute_phase.metadata["instruction_schedule"]["iq_panel_weight_reuse"]
            iq_panel_audit = {
                **iq_panel_audit, **reuse,
                "schema": "heterollm.cpu-iq-panel-reuse/v1",
                "status": "applied" if reuse["applied"] else "uncovered",
                "reason": None if reuse["applied"] else reuse["rejection_reasons"][0],
            }
    else:
        raise ValueError("并行 GEMM 目标 {} 不具备计算能力".format(target.kind))

    if staged_weight_allocation_id is not None:
        register = builder.add(
            name + ".staged_weight.register",
            TaskCategory.SYNCHRONIZATION,
            dependencies=prior,
            advance=False,
            metadata={
                **staged_weight_metadata,
                "event_kind": "staged_weight_register",
                "staged_weight_operation": "register",
                "resource_accounting": "zero_resource_lifecycle_marker",
            },
        )
        staged_read = builder.add(
            name + ".staged_weight.read",
            TaskCategory.SYNCHRONIZATION,
            dependencies=(register,),
            advance=False,
            metadata={
                **staged_weight_metadata,
                "event_kind": "staged_weight_read",
                "staged_weight_operation": "read",
                "resource_accounting": "gpu_gemm_hbm_roofline",
            },
        )
        prior = (staged_read,)

    explicit_dispatch_ns = float(
        quantization_metadata.get("dequant_dispatch_ns", 0.0)
    )
    if explicit_dispatch_ns > 0.0 and dispatch_resource_id is not None:
        dispatch = builder.add(
            name + ".dequant_dispatch",
            TaskCategory.COMPUTE,
            (
                ResourceDemand(
                    dispatch_resource_id,
                    explicit_dispatch_ns,
                    energy_pj=dispatch_energy_pj,
                ),
            ),
            dependencies=prior,
            advance=False,
            metadata={
                **operation_metadata,
                "op_name": name,
                "phase": "dequant_dispatch",
                "dispatch_ns": explicit_dispatch_ns,
                "dispatch_override": True,
            },
        )
        prior = (dispatch,)

    last = ""
    phase_metadata = {
        "rank": rank.rank,
        "tp_rank": rank.tp_rank,
        "pp_rank": rank.pp_rank,
        "ep_rank": rank.ep_rank,
        "target_component": target_component_id,
        **operation_metadata,
    }
    for phase_index, phase in enumerate(estimate.phases):
        phase_demands = phase.demands + extra_phase_demands.get(phase.name, ())
        # ``estimate_cpu_gemm`` already passes activation + physical weight
        # bytes through the cache hierarchy to the local backing demand.
        # A resident CPU weight needs no topology transfer, but adding its
        # bytes again here would double-charge the same DRAM read.
        effective_phase_metadata = dict(phase.metadata)
        # Semantic calibration buckets use the native operator's logical
        # output shape (N x M x 1 x 1).  Preserve it at the phase boundary so
        # an exact shape coefficient can be selected; missing shapes remain
        # fail-closed in calibrate_cost_phase.
        if ("token_shape" not in effective_phase_metadata
                and scenario.placement.metadata.get("native_calibration_shape_policy") != "phase"):
            # ``gemm_n``/``gemm_m`` are emitted by
            # ``_workload_quantization_metadata`` into ``phase_metadata``
            # (the operation-level envelope), while a CostPhase's own
            # metadata only contains cost-model details.  Looking solely at
            # ``phase.metadata`` therefore dropped the exact M=1 decode
            # shape and silently selected a phase aggregate.  Use the
            # operation envelope as a conservative fallback; it is constant
            # for every phase of this one physical GEMM and is unavailable
            # for dynamic attention/elementwise phases, which remain
            # fail-closed as before.
            gemm_n = effective_phase_metadata.get("gemm_n")
            if gemm_n is None:
                gemm_n = phase_metadata.get("gemm_n")
            gemm_m = effective_phase_metadata.get("gemm_m")
            if gemm_m is None:
                gemm_m = phase_metadata.get("gemm_m")
            if isinstance(gemm_n, int) and isinstance(gemm_m, int) and gemm_n > 0 and gemm_m > 0:
                effective_phase_metadata["token_shape"] = f"{gemm_n}x{gemm_m}x1x1"
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=target_component_id,
            )
            for demand in phase_demands
        )
        # Apply only evidence-backed stage/shape coefficients.  The helper is
        # deliberately fail-closed: without an explicit invocation identity,
        # phase, and token shape it returns the analytical demands unchanged.
        calibration = (
            profile_from_mapping(scenario.placement.metadata.get("native_calibration"))
            if (scenario.placement.metadata.get("native_calibration_apply_stage") is True
                    or scenario.placement.metadata.get("native_calibration_apply_memory") is True)
            else None
        )
        if calibration is not None:
            calibration_metadata = {**phase_metadata, **effective_phase_metadata}
            # Carry the scenario-level shape policy into the phase helper.  It
            # is deliberately metadata-only: the helper still requires an
            # explicit kernel-basis profile before selecting a shape bucket.
            calibration_metadata.setdefault(
                "native_calibration_shape_policy",
                scenario.placement.metadata.get("native_calibration_shape_policy"),
            )
            # CostPhase metadata carries an implementation phase such as
            # ``gpu_gemm`` or ``kernel_launch``.  That low-level label must
            # not hide the serving phase encoded by the qualified task name;
            # otherwise stage calibration silently misses every GPU GEMM.
            current_phase = str(
                calibration_metadata.get("execution_phase")
                or calibration_metadata.get("phase")
                or ""
            ).strip().lower()
            if current_phase not in {"prefill", "decode"}:
                inferred_phase = _execution_phase_from_name(name)
                if inferred_phase is not None:
                    calibration_metadata["execution_phase"] = inferred_phase
            calibrated_phase = calibrate_cost_phase(
                replace(phase, demands=demands, metadata=effective_phase_metadata),
                calibration_metadata,
                calibration,
                model_sha256=_model_gguf_sha256(scenario.model),
                hardware_fingerprint=scenario.placement.metadata.get("hardware_fingerprint"),
                runtime_fingerprint=(scenario.placement.metadata.get("llama_cpp_runtime_fingerprint")
                                     or (scenario.llama_cpp_config.fingerprint
                                         if scenario.llama_cpp_config is not None else None)),
                apply_memory=scenario.placement.metadata.get("native_calibration_apply_memory") is True,
            )
            demands = calibrated_phase.demands
            effective_phase_metadata = dict(calibrated_phase.metadata)
            # Apply a measured launch rate only to this invocation's explicit
            # kernel-launch phase.  The operator-wall coefficient above stays
            # scoped to the compute demand; composing them avoids charging a
            # global launch latency to every semantic sub-operator.
            launch_ns = launch_calibration_ns(
                calibration,
                calibration_metadata.get("execution_phase")
                or calibration_metadata.get("phase")
                or _execution_phase_from_name(name),
            )
            if launch_ns is not None and phase.name == "kernel_launch":
                demands = tuple(
                    ResourceDemand(
                        d.resource_id,
                        launch_ns,
                        d.bytes_moved,
                        d.energy_pj,
                        d.work_units,
                    )
                    if "frontend" in d.resource_id.lower()
                    or "launch" in d.resource_id.lower() else d
                    for d in demands
                )
        last = builder.add(
            "{}.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                **phase_metadata,
                **({"cpu_iq_panel_reuse": iq_panel_audit}
                   if phase.name == "cpu_gemm" and iq_panel_audit is not None else {}),
                "op_name": name,
                "phase": phase.name,
                "cost_model": {
                    **dict(estimate.metadata),
                    **quantization_metadata,
                    "operator_class": OperatorClass.GEMM.value,
                },
                "phase_metadata": effective_phase_metadata,
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands), default=0.0
                ),
            },
        )
        if dynamic_attention_replay is not None:
            builder._task_segment_dynamic_payloads[last] = replace(
                dynamic_attention_replay,
                target_component_id=target_component_id,
                operator_class=OperatorClass.GEMM,
                phase_index=phase_index,
                phase_name=phase.name,
                cost_model_suffix=tuple(
                    {
                        **quantization_metadata,
                        "operator_class": OperatorClass.GEMM.value,
                    }.items()
                ),
            )
        prior = (last,)
    if mmq_work is not None and mmq_work.fixup_launch:
        fixup = TensorKernelWorkload(
            operations=mmq_work.fixup_operations,
            read_bytes=mmq_work.fixup_read_bytes,
            write_bytes=mmq_work.fixup_write_bytes,
            streaming_fraction=1.0,
            name="llama_cpp_mmq_partial_result_fixup",
            launch_only=mmq_work.fixup_operations == 0,
        )
        last = _add_rank_tensor_kernel(
            builder, scenario, router, plan, rank, fixup,
            name + ".mmq_fixup", prior,
            execution_component_id=target_component_id, input_is_local=True,
            metadata={
                "event_kind": "mmq_partial_result_merge",
                "operator_class": OperatorClass.ELEMENTWISE.value,
                "layer_id": operation_metadata.get("layer_id"),
                "projection_id": operation_metadata.get("projection_id"),
                "mmq_source_work": {**dict(mmq_audit or {}), "stage": "fixup"},
            },
        )
        prior = (last,)
    if _kind(target) == "cpu" and (
        workload.epilogue_operations
        or workload.epilogue_transcendental_operations
    ):
        # estimate_cpu_gemm() models the GEMM pipeline but intentionally does
        # not fold genuine GemmWorkload activation epilogues into its typed
        # roofline.  Artifact dequantization never reaches this path: it is
        # accounted inside the quantized GEMM phase above.
        epilogue_operations = (
            workload.epilogue_operations
            + workload.epilogue_transcendental_operations
        )
        epilogue_throughput = max(
            1.0, cpu_profile.attainable_elementwise_gops
        )
        epilogue_service_ns = epilogue_operations / epilogue_throughput
        epilogue = builder.add(
            name + ".gemm_epilogue",
            TaskCategory.COMPUTE,
            (
                ResourceDemand(
                    cpu_profile.compute_resource_id,
                    epilogue_service_ns,
                    energy_pj=(
                        workload.epilogue_operations
                        * cpu_profile.elementwise_energy_pj_per_op
                        + workload.epilogue_transcendental_operations
                        * cpu_profile.special_function_energy_pj_per_op
                    ),
                    work_units=float(epilogue_operations),
                ),
            ),
            dependencies=prior,
            advance=False,
            metadata={
                **operation_metadata,
                "op_name": name,
                "phase": "gemm_epilogue",
                "epilogue_operations": workload.epilogue_operations,
                "epilogue_transcendental_operations": (
                    workload.epilogue_transcendental_operations
                ),
                "epilogue_name": workload.epilogue_name,
                "cost_model": {
                    **quantization_metadata,
                    "device": "cpu",
                    "epilogue_service_ns": epilogue_service_ns,
                },
            },
        )
        prior = (epilogue,)
        last = epilogue
    if staged_weight_allocation_id is not None:
        release = builder.add(
            name + ".staged_weight.release",
            TaskCategory.SYNCHRONIZATION,
            dependencies=prior,
            advance=False,
            metadata={
                **staged_weight_metadata,
                "event_kind": "staged_weight_release",
                "staged_weight_operation": "release",
                "resource_accounting": "zero_resource_lifecycle_marker",
            },
        )
        prior = (release,)
        last = release
    if _is_cim(target):
        last = _add_transfer_tasks(
            builder,
            router,
            target_component_id,
            rank.component_id,
            workload.output_bytes,
            prior,
            name=name + ".output_to_gpu",
            routing_policy=plan.routing_policy,
            metadata={
                **operation_metadata,
                "rank": rank.rank,
                "transport": "cim-gpu",
            },
        )
        output_component_id = rank.component_id
    else:
        output_component_id = target_component_id
    builder.record_rank_value(last, rank.rank, output_component_id)
    return last


def _rank_gpu_activation_input_dependencies(
    builder: _TaskBuilder,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    dependencies: Sequence[str],
    byte_count: int,
    *,
    name: str,
    operator_class: OperatorClass,
    metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[Tuple[str, ...], str]:
    prior = tuple(dependencies)
    source_component = (
        builder.rank_value_component(prior, rank.rank) or rank.component_id
    )
    if source_component == rank.component_id or byte_count <= 0:
        return prior, source_component
    base_metadata = dict(metadata or {})
    prior = (
        _add_transfer_tasks(
            builder,
            router,
            source_component,
            rank.component_id,
            byte_count,
            prior,
            name=name + ".input_transfer",
            routing_policy=plan.routing_policy,
            metadata={
                **base_metadata,
                "rank": rank.rank,
                "event_kind": "operator_input_transfer",
                "operator_class": operator_class.value,
                "operator_id": str(base_metadata.get("operator_id") or name),
                "transport": "activation",
                "source_component": source_component,
                "input_component": source_component,
                "target_component": rank.component_id,
            },
        ),
    )
    return prior, source_component


def _add_rank_tensor_kernel(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    workload: TensorKernelWorkload,
    name: str,
    dependencies: Sequence[str],
    *,
    metadata: Optional[Mapping[str, object]] = None,
    execution_component_id: Optional[str] = None,
    input_is_local: bool = False,
) -> str:
    """Lower a local non-GEMM tensor kernel on one logical GPU rank."""

    execution_id = execution_component_id or rank.component_id
    if _kind(_component(scenario, execution_id)) != "gpu":
        raise ValueError("tensor kernel execution component must be a GPU")
    if input_is_local:
        # The enclosing physical GEMM already moved the original input, or
        # this is its local fixup. Temporary buffers never trigger another route.
        prior, activation_source = tuple(dependencies), execution_id
    else:
        if execution_id != rank.component_id:
            raise ValueError("remote tensor kernel needs an explicitly local input")
        prior, activation_source = _rank_gpu_activation_input_dependencies(
            builder, router, plan, rank, dependencies,
            max(0, int(workload.read_bytes)), name=name,
            operator_class=OperatorClass.ELEMENTWISE, metadata=metadata,
        )
    memory_id = rank.memory_component_id if execution_id == rank.component_id else None
    gpu_profile, hbm_profile = _gpu_profiles(
        scenario,
        execution_id,
        memory_id,
    )
    estimate = _memoized_cost_estimate(
        scenario,
        (
            "gpu_tensor_kernel",
            execution_id,
            memory_id,
            workload,
        ),
        lambda: estimate_gpu_tensor_kernel(
            gpu_profile, hbm_profile, workload
        ),
    )
    last = ""
    for phase in estimate.phases:
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=execution_id,
            )
            for demand in phase.demands
        )
        # Tensor kernels (norm/quantize/attention-output) carry explicit
        # event_kind/coverage_component metadata.  Feed those markers through
        # the same fail-closed stage calibration gate used by GEMM lowering.
        calibration = (
            profile_from_mapping(scenario.placement.metadata.get("native_calibration"))
            if (scenario.placement.metadata.get("native_calibration_apply_stage") is True
                    or scenario.placement.metadata.get("native_calibration_apply_memory") is True)
            else None
        )
        if calibration is not None:
            execution_phase = ((metadata or {}).get("execution_phase")
                               or (metadata or {}).get("phase")) if metadata else None
            if str(execution_phase or "").strip().lower() not in {"prefill", "decode"}:
                inferred_phase = _execution_phase_from_name(name)
                if inferred_phase is not None:
                    execution_phase = inferred_phase
            cal_meta = {
                "phase": execution_phase or phase.name,
                "calibration_stage": (metadata or {}).get("calibration_stage") if metadata else None,
                "coverage_component": (metadata or {}).get("coverage_component") if metadata else None,
                "event_kind": (metadata or {}).get("event_kind") if metadata else None,
                "linear_op": (metadata or {}).get("linear_op") if metadata else None,
                "projection_id": (metadata or {}).get("projection_id") if metadata else None,
                "token_shape": (metadata or {}).get("token_shape") if metadata else None,
                "native_calibration_shape_policy": scenario.placement.metadata.get(
                    "native_calibration_shape_policy"
                ),
                "token_batch": (metadata or {}).get("token_batch") if metadata else None,
                # Bind provenance to the actual request being lowered.  A
                # scenario may contain several request shapes in one batch;
                # using scenario.workload here silently attributed a profile
                # to the wrong prompt/output pair.
                "prompt_tokens": builder.request.prompt_tokens,
                "output_tokens": builder.request.output_tokens,
            }
            calibrated_phase = calibrate_cost_phase(
                replace(phase, demands=demands), cal_meta, calibration,
                model_sha256=_model_gguf_sha256(scenario.model),
                hardware_fingerprint=scenario.placement.metadata.get("hardware_fingerprint"),
                runtime_fingerprint=(scenario.placement.metadata.get("llama_cpp_runtime_fingerprint")
                                     or (scenario.llama_cpp_config.fingerprint
                                         if scenario.llama_cpp_config is not None else None)),
                apply_memory=scenario.placement.metadata.get("native_calibration_apply_memory") is True,
            )
            demands = calibrated_phase.demands
            # A phase-scoped launch measurement belongs only to the explicit
            # kernel-launch phase.  Keep it separate from operator-wall
            # compute calibration so one graph launch is not charged once per
            # semantic operator.
            launch_ns = launch_calibration_ns(calibration, execution_phase)
            if launch_ns is not None and phase.name == "kernel_launch":
                demands = tuple(
                    ResourceDemand(
                        d.resource_id,
                        launch_ns,
                        d.bytes_moved,
                        d.energy_pj,
                        d.work_units,
                    )
                    if "frontend" in d.resource_id.lower()
                    or "launch" in d.resource_id.lower() else d
                    for d in demands
                )
        last = builder.add(
            "{}.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                "rank": rank.rank,
                "tp_rank": rank.tp_rank,
                "pp_rank": rank.pp_rank,
                "ep_rank": rank.ep_rank,
                "op_name": name,
                "phase": phase.name,
                "target_component": execution_id,
                "input_component": activation_source,
                "cost_model": dict(estimate.metadata),
                "phase_metadata": dict(phase.metadata),
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands), default=0.0
                ),
                **dict(metadata or {}),
            },
        )
        prior = (last,)
    builder.record_rank_value(last, rank.rank, execution_id)
    return last


def _add_rank_fused_attention(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    workload: FusedAttentionWorkload,
    name: str,
    dependencies: Sequence[str],
    *,
    metadata: Optional[Mapping[str, object]] = None,
    fusion_targets: Sequence[Tuple[str, str]] = (),
) -> str:
    """Lower one FlashAttention-style mixed-engine GPU kernel."""

    activation_input_bytes = max(
        0,
        int(workload.read_bytes) - int(workload.kv_read_bytes),
    )
    prior, activation_source = _rank_gpu_activation_input_dependencies(
        builder,
        router,
        plan,
        rank,
        dependencies,
        activation_input_bytes,
        name=name,
        operator_class=OperatorClass.GEMM,
        metadata=metadata,
    )
    gpu_profile, hbm_profile = _gpu_profiles(
        scenario,
        rank.component_id,
        rank.memory_component_id,
    )
    estimate = _memoized_cost_estimate(
        scenario,
        (
            "gpu_fused_attention",
            rank.component_id,
            rank.memory_component_id,
            workload,
        ),
        lambda: estimate_gpu_fused_attention(
            gpu_profile, hbm_profile, workload
        ),
    )
    last = ""
    for phase_index, phase in enumerate(estimate.phases):
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=rank.component_id,
            )
            for demand in phase.demands
        )
        calibration = (
            profile_from_mapping(scenario.placement.metadata.get("native_calibration"))
            if (scenario.placement.metadata.get("native_calibration_apply_stage") is True
                    or scenario.placement.metadata.get("native_calibration_apply_memory") is True)
            else None
        )
        effective_phase_metadata = dict(phase.metadata)
        if calibration is not None:
            execution_phase = ((metadata or {}).get("execution_phase")
                               or (metadata or {}).get("phase")) if metadata else None
            if str(execution_phase or "").strip().lower() not in {"prefill", "decode"}:
                inferred_phase = _execution_phase_from_name(name)
                if inferred_phase is not None:
                    execution_phase = inferred_phase
            cal_meta = {
                "phase": execution_phase or phase.name,
                "calibration_stage": (metadata or {}).get("calibration_stage") if metadata else None,
                "coverage_component": (metadata or {}).get("coverage_component") if metadata else None,
                "event_kind": (metadata or {}).get("event_kind") if metadata else None,
                "linear_op": (metadata or {}).get("linear_op") if metadata else None,
                "projection_id": (metadata or {}).get("projection_id") if metadata else None,
                "token_shape": (metadata or {}).get("token_shape") if metadata else None,
            }
            calibrated_phase = calibrate_cost_phase(
                replace(phase, demands=demands), cal_meta, calibration,
                model_sha256=_model_gguf_sha256(scenario.model),
                hardware_fingerprint=scenario.placement.metadata.get("hardware_fingerprint"),
                runtime_fingerprint=(scenario.placement.metadata.get("llama_cpp_runtime_fingerprint")
                                     or (scenario.llama_cpp_config.fingerprint
                                         if scenario.llama_cpp_config is not None else None)),
                apply_memory=scenario.placement.metadata.get("native_calibration_apply_memory") is True,
            )
            demands = calibrated_phase.demands
            effective_phase_metadata = dict(calibrated_phase.metadata)
        last = builder.add(
            "{}.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                "rank": rank.rank,
                "tp_rank": rank.tp_rank,
                "pp_rank": rank.pp_rank,
                "ep_rank": rank.ep_rank,
                "op_name": name,
                "phase": phase.name,
                "event_kind": (
                    "kv_materialization"
                    if phase.metadata.get("materialization_operand") else "fused_attention"
                ),
                "operator_class": (
                    OperatorClass.ELEMENTWISE.value
                    if phase.metadata.get("materialization_operand") else OperatorClass.GEMM.value
                ),
                "target_component": rank.component_id,
                "input_component": activation_source,
                "cost_model": dict(estimate.metadata),
                "phase_metadata": effective_phase_metadata,
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands), default=0.0
                ),
                **dict(metadata or {}),
            },
        )
        builder._task_segment_dynamic_payloads[last] = (
            _FusedAttentionTaskReplayPayload(
                workload=workload,
                rank=rank,
                layer_id=str((metadata or {}).get("layer_id", "")),
                phase_index=phase_index,
                phase_name=phase.name,
                fusion_targets=tuple(fusion_targets),
            )
        )
        prior = (last,)
    builder.record_rank_value(last, rank.rank, rank.component_id)
    return last


def _primitive_io_bytes(workload: object) -> Tuple[int, int]:
    return (
        int(getattr(workload, "read_bytes", 0)),
        int(getattr(workload, "write_bytes", 0)),
    )


def _estimate_typed_primitive(
    scenario: ScenarioConfig,
    target_component_id: str,
    operator_class: OperatorClass,
    workload: object,
    *,
    memory_component_id: Optional[str] = None,
) -> CostEstimate:
    target = _component(scenario, target_component_id)
    if _kind(target) == "gpu":
        gpu_profile, hbm_profile = _gpu_profiles(
            scenario,
            target_component_id,
            memory_component_id,
        )
        if operator_class == OperatorClass.ELEMENTWISE:
            return _memoized_cost_estimate(
                scenario,
                (
                    "gpu_elementwise",
                    target_component_id,
                    memory_component_id,
                    workload,
                ),
                lambda: estimate_gpu_elementwise(
                    gpu_profile, hbm_profile, workload
                ),
            )
        if operator_class == OperatorClass.REDUCTION:
            return _memoized_cost_estimate(
                scenario,
                (
                    "gpu_reduction",
                    target_component_id,
                    memory_component_id,
                    workload,
                ),
                lambda: estimate_gpu_reduction(
                    gpu_profile, hbm_profile, workload
                ),
            )
        if operator_class == OperatorClass.MEMORY:
            return _memoized_cost_estimate(
                scenario,
                (
                    "gpu_memory",
                    target_component_id,
                    memory_component_id,
                    workload,
                ),
                lambda: estimate_gpu_memory(
                    gpu_profile, hbm_profile, workload
                ),
            )
    elif _kind(target) == "cpu":
        cpu_memory_component_id = (
            memory_component_id
            if memory_component_id is not None
            and _kind(_component(scenario, memory_component_id))
            == "host_memory"
            else None
        )
        cpu_profile, host_memory_profile = _cpu_profiles(
            scenario,
            target_component_id,
            memory_component_id=cpu_memory_component_id,
        )
        if operator_class == OperatorClass.ELEMENTWISE:
            return _memoized_cost_estimate(
                scenario,
                (
                    "cpu_elementwise",
                    target_component_id,
                    cpu_memory_component_id,
                    workload,
                ),
                lambda: estimate_cpu_elementwise(
                    cpu_profile, host_memory_profile, workload
                ),
            )
        if operator_class == OperatorClass.REDUCTION:
            return _memoized_cost_estimate(
                scenario,
                (
                    "cpu_reduction",
                    target_component_id,
                    cpu_memory_component_id,
                    workload,
                ),
                lambda: estimate_cpu_reduction(
                    cpu_profile, host_memory_profile, workload
                ),
            )
        if operator_class == OperatorClass.MEMORY:
            return _memoized_cost_estimate(
                scenario,
                (
                    "cpu_memory",
                    target_component_id,
                    cpu_memory_component_id,
                    workload,
                ),
                lambda: estimate_cpu_memory(
                    cpu_profile, host_memory_profile, workload
                ),
            )
    raise ValueError(
        "{} 算子目标 {} 不支持 {}".format(
            operator_class.value, target_component_id, target.kind
        )
    )


def _add_rank_primitive(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    operator_class: OperatorClass,
    workload: object,
    mapping_key: str,
    name: str,
    dependencies: Sequence[str],
    *,
    source_component_id: str,
    target_component_id: Optional[str] = None,
    input_component_bytes: Optional[Sequence[Tuple[str, int]]] = None,
    fallback_keys: Sequence[str] = (),
    metadata: Optional[Mapping[str, object]] = None,
    dynamic_attention_replay: Optional[
        _DynamicAttentionCostTaskReplayPayload
    ] = None,
) -> Tuple[str, str]:
    """Lower one typed primitive and retain its output on the chosen device.

    Compute and local-memory demands inside the estimator remain concurrent.
    Only a real inter-component boundary emits a topology transfer, so adjacent
    primitives selected for the same CPU/GPU never pay duplicate movement.
    """

    target_component_id = (
        str(target_component_id)
        if target_component_id is not None
        else _primitive_target(
            scenario,
            router,
            rank,
            operator_class,
            mapping_key,
            fallback_keys=fallback_keys,
        )
    )
    prior = tuple(dependencies)
    source_component_id = (
        builder.rank_value_component(prior, rank.rank)
        or source_component_id
    )
    read_bytes, write_bytes = _primitive_io_bytes(workload)
    if input_component_bytes is None:
        input_sources = ((source_component_id, read_bytes),)
    else:
        input_sources = tuple(
            (str(component_id), max(0, int(byte_count)))
            for component_id, byte_count in input_component_bytes
        )
        accounted_read_bytes = sum(
            byte_count for _component_id, byte_count in input_sources
        )
        remaining_read_bytes = max(0, read_bytes - accounted_read_bytes)
        if remaining_read_bytes:
            input_sources = input_sources + (
                (source_component_id, remaining_read_bytes),
            )
    input_components = tuple(
        dict.fromkeys(
            component_id
            for component_id, byte_count in input_sources
            if byte_count > 0
        )
    )
    input_transfer_bytes: Dict[str, int] = {}
    for input_component_id, byte_count in input_sources:
        if byte_count <= 0:
            continue
        input_transfer_bytes[input_component_id] = (
            input_transfer_bytes.get(input_component_id, 0) + byte_count
        )
    transfer_dependencies: List[str] = []
    for input_component_id, byte_count in input_transfer_bytes.items():
        if input_component_id == target_component_id or byte_count <= 0:
            continue
        transfer_dependencies.append(
            _add_transfer_tasks(
                builder,
                router,
                input_component_id,
                target_component_id,
                byte_count,
                prior,
                name=name + ".input_transfer",
                routing_policy=plan.routing_policy,
                metadata={
                    "event_kind": "operator_input_transfer",
                    "operator_class": operator_class.value,
                    "operator_id": mapping_key,
                    "rank": rank.rank,
                    "input_component": input_component_id,
                    "input_components": input_components,
                    "input_bytes": byte_count,
                },
            ),
        )
    if transfer_dependencies:
        prior = tuple(transfer_dependencies)
    estimate = _estimate_typed_primitive(
        scenario,
        target_component_id,
        operator_class,
        workload,
        memory_component_id=rank.memory_component_id,
    )
    last = ""
    operation_metadata = {
        "rank": rank.rank,
        "tp_rank": rank.tp_rank,
        "pp_rank": rank.pp_rank,
        "ep_rank": rank.ep_rank,
        "operator_id": mapping_key,
        "operator_class": operator_class.value,
        "target_component": target_component_id,
        "input_component": source_component_id,
        "input_components": input_components,
        "output_bytes": write_bytes,
        **dict(metadata or {}),
    }
    for phase_index, phase in enumerate(estimate.phases):
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=target_component_id,
            )
            for demand in phase.demands
        )
        calibration = (
            profile_from_mapping(scenario.placement.metadata.get("native_calibration"))
            if (scenario.placement.metadata.get("native_calibration_apply_stage") is True
                    or scenario.placement.metadata.get("native_calibration_apply_memory") is True)
            else None
        )
        effective_phase_metadata = dict(phase.metadata)
        if calibration is not None:
            execution_phase = ((metadata or {}).get("execution_phase")
                               or (metadata or {}).get("phase")) if metadata else None
            if str(execution_phase or "").strip().lower() not in {"prefill", "decode"}:
                inferred_phase = _execution_phase_from_name(name)
                if inferred_phase is not None:
                    execution_phase = inferred_phase
            cal_meta = {
                "phase": execution_phase or phase.name,
                "calibration_stage": (metadata or {}).get("calibration_stage") if metadata else None,
                "coverage_component": (metadata or {}).get("coverage_component") if metadata else None,
                "event_kind": (metadata or {}).get("event_kind") if metadata else None,
                "linear_op": (metadata or {}).get("linear_op") if metadata else None,
                "projection_id": (metadata or {}).get("projection_id") if metadata else None,
                "token_shape": (metadata or {}).get("token_shape") if metadata else None,
            }
            calibrated_phase = calibrate_cost_phase(
                replace(phase, demands=demands), cal_meta, calibration,
                model_sha256=_model_gguf_sha256(scenario.model),
                hardware_fingerprint=scenario.placement.metadata.get("hardware_fingerprint"),
                runtime_fingerprint=(scenario.placement.metadata.get("llama_cpp_runtime_fingerprint")
                                     or (scenario.llama_cpp_config.fingerprint
                                         if scenario.llama_cpp_config is not None else None)),
                apply_memory=scenario.placement.metadata.get("native_calibration_apply_memory") is True,
            )
            demands = calibrated_phase.demands
            effective_phase_metadata = dict(calibrated_phase.metadata)
        last = builder.add(
            "{}.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                **operation_metadata,
                "op_name": name,
                "phase": phase.name,
                "cost_model": {
                    **dict(estimate.metadata),
                    "operator_class": operator_class.value,
                },
                "phase_metadata": effective_phase_metadata,
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands), default=0.0
                ),
            },
        )
        if dynamic_attention_replay is not None:
            builder._task_segment_dynamic_payloads[last] = replace(
                dynamic_attention_replay,
                target_component_id=target_component_id,
                operator_class=operator_class,
                phase_index=phase_index,
                phase_name=phase.name,
                cost_model_suffix=(
                    ("operator_class", operator_class.value),
                ),
            )
        prior = (last,)
    builder.record_rank_value(last, rank.rank, target_component_id)
    return last, target_component_id


def _return_rank_value(
    builder: _TaskBuilder,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    task_id: str,
    component_id: str,
    byte_count: int,
    *,
    name: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    """Retain a produced activation until a real consumer boundary."""

    builder.record_rank_value(task_id, rank.rank, component_id)
    # CPU/GPU outputs stay live on their producing component.  Consumers and
    # collectives lower the transfer when (and only when) their placement
    # creates a real boundary.
    return task_id


def _discard_side_branch_rank_value(
    builder: _TaskBuilder,
    task_id: str,
    rank: LogicalRank,
) -> str:
    """Prevent side-input/side-effect work from publishing activation liveness."""

    builder.discard_rank_value(task_id, rank.rank)
    return task_id


def _linear_state_components(
    scenario: ScenarioConfig,
    rank: LogicalRank,
    target_component_id: Optional[str] = None,
) -> Tuple[str, Optional[str], float]:
    configured_active = (
        scenario.placement.tensor_to_component.get("linear_state")
        or rank.memory_component_id
        or scenario.placement.kv_policy.cache_component
        or rank.component_id
    )
    active = _layer_local_runtime_memory_component_id(
        scenario,
        rank,
        target_component_id,
        str(configured_active),
    )
    if active is None:
        raise ValueError("linear state has no active runtime-memory placement")
    offload = scenario.placement.tensor_to_component.get("linear_state_offload")
    raw_ratio = scenario.placement.metadata.get("linear_state_offload_ratio", 1.0)
    ratio = min(1.0, max(0.0, float(raw_ratio)))
    return str(active), (str(offload) if offload else None), ratio


@dataclass(frozen=True)
class _LinearStateRuntimeSemantics:
    """Describe one invocation's committed and speculative state lifecycle.

    Ordinary decode/prefill invocations update the persistent state directly.
    An MTP verifier instead advances a request-local rolling scratch state for
    every materialized position, selects the accepted-prefix snapshot for the
    persistent commit, and releases scratch only after the final verifier
    position.  This keeps rejected suffix state physically distinct from the
    state visible to the next decoding round.
    """

    mode: str = "persistent_update"
    read_source: str = "committed"
    materialized_positions: Tuple[int, ...] = ()
    committed_prefix_positions: Tuple[int, ...] = ()
    rejected_positions: Tuple[int, ...] = ()
    commit_snapshot_positions: Tuple[int, ...] = ()
    materialized_lane_ids: Tuple[str, ...] = ()
    committed_lane_ids: Tuple[str, ...] = ()
    rejected_lane_ids: Tuple[str, ...] = ()
    commit_snapshot_lane_ids: Tuple[str, ...] = ()
    persistent_update_lane_ids: Tuple[str, ...] = ()
    release_temporary: bool = False
    independent_state_owner_count: int = 1

    def __post_init__(self) -> None:
        count = int(self.independent_state_owner_count)
        if count < 1:
            raise ValueError("linear state owner count must be positive")
        object.__setattr__(self, "independent_state_owner_count", count)

    @property
    def uses_temporary_state(self) -> bool:
        return self.mode in {
            "speculative_verification",
            "mixed_persistent_and_speculative_update",
        }

    @property
    def has_persistent_updates(self) -> bool:
        return bool(self.persistent_update_lane_ids)

    @property
    def commits_snapshot(self) -> bool:
        return bool(
            self.commit_snapshot_positions or self.commit_snapshot_lane_ids
        )

    def audit_metadata(self) -> Mapping[str, object]:
        metadata = {
            "linear_state_runtime_semantics": self.mode,
            "linear_state_read_source": self.read_source,
            "linear_state_materialized_positions": self.materialized_positions,
            "linear_state_committed_prefix_positions": (
                self.committed_prefix_positions
            ),
            "linear_state_rejected_positions": self.rejected_positions,
            "linear_state_commit_snapshot_positions": (
                self.commit_snapshot_positions
            ),
            "linear_state_materialized_lane_ids": self.materialized_lane_ids,
            "linear_state_committed_lane_ids": self.committed_lane_ids,
            "linear_state_rejected_lane_ids": self.rejected_lane_ids,
            "linear_state_commit_snapshot_lane_ids": (
                self.commit_snapshot_lane_ids
            ),
            "linear_state_persistent_update_lane_ids": (
                self.persistent_update_lane_ids
            ),
            "linear_state_temporary_release": self.release_temporary,
        }
        if self.independent_state_owner_count != 1:
            metadata["linear_state_independent_owner_count"] = (
                self.independent_state_owner_count
            )
        return metadata


def _linear_state_runtime_structure_identity(
    runtime: _LinearStateRuntimeSemantics,
) -> Optional[Tuple[object, ...]]:
    """Return the topology-bearing part of one linear-state runtime.

    Verifier positions and lane ids are audit identities: their concrete
    values do not change workloads, routes, demands, or task count.  The
    booleans below do change the state read/materialize/commit/release graph.
    Mixed persistent+speculative groups are eligible only because replay uses
    discriminator-scoped overrides for the outer and inner runtime blocks.
    """

    if runtime.mode not in {
        "persistent_update",
        "speculative_verification",
        "mixed_persistent_and_speculative_update",
    }:
        return None
    if runtime.read_source not in {"committed", "speculative"}:
        return None
    identity = (
        "linear_state_structure_v1",
        runtime.mode,
        runtime.read_source,
        runtime.uses_temporary_state,
        runtime.commits_snapshot,
        runtime.release_temporary,
        runtime.has_persistent_updates,
    )
    if runtime.independent_state_owner_count != 1:
        identity = identity + (
            (
                "independent_state_owner_count",
                runtime.independent_state_owner_count,
            ),
        )
    return identity


def _mtp_linear_state_runtime(
    verifier_positions: Sequence[int],
    committed_prefix_positions: Sequence[int],
    commit_snapshot_positions: Sequence[int],
    *,
    read_source: str,
    release_temporary: bool,
    materialized_lane_ids: Sequence[str] = (),
    committed_lane_ids: Sequence[str] = (),
    commit_snapshot_lane_ids: Sequence[str] = (),
    persistent_update_lane_ids: Sequence[str] = (),
) -> _LinearStateRuntimeSemantics:
    materialized = tuple(int(position) for position in verifier_positions)
    committed = tuple(int(position) for position in committed_prefix_positions)
    committed_set = set(committed)
    materialized_lanes = tuple(str(lane_id) for lane_id in materialized_lane_ids)
    committed_lanes = tuple(str(lane_id) for lane_id in committed_lane_ids)
    persistent_lanes = tuple(
        str(lane_id) for lane_id in persistent_update_lane_ids
    )
    committed_lane_set = set(committed_lanes)
    return _LinearStateRuntimeSemantics(
        mode=(
            "mixed_persistent_and_speculative_update"
            if persistent_lanes
            else "speculative_verification"
        ),
        read_source=str(read_source),
        materialized_positions=materialized,
        committed_prefix_positions=committed,
        rejected_positions=tuple(
            position for position in materialized if position not in committed_set
        ),
        commit_snapshot_positions=tuple(
            int(position) for position in commit_snapshot_positions
        ),
        materialized_lane_ids=materialized_lanes,
        committed_lane_ids=committed_lanes,
        rejected_lane_ids=tuple(
            lane_id
            for lane_id in materialized_lanes
            if lane_id not in committed_lane_set
        ),
        commit_snapshot_lane_ids=tuple(
            str(lane_id) for lane_id in commit_snapshot_lane_ids
        ),
        persistent_update_lane_ids=persistent_lanes,
        release_temporary=bool(release_temporary),
    )


def _linear_state_persistent_owner_count(
    runtime: _LinearStateRuntimeSemantics,
) -> int:
    return max(1, int(runtime.independent_state_owner_count))


def _linear_state_bytes(layer: LayerSpec, tp_degree: int) -> int:
    geometry = layer.linear_attention
    if geometry is None:
        return 0
    total_elements = (
        geometry.recurrent_state_elements
        + geometry.convolution_state_elements
    )
    total_bytes = int(math.ceil(total_elements * _dtype_bits(geometry.state_dtype) / 8.0))
    return int(math.ceil(total_bytes / float(tp_degree)))


def _host_recurrent_offload_decision(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    *,
    physical_m: int,
    placement_component_id: str,
    host_projection_decision: Optional[_HostGemmOffloadDecision],
) -> Optional[_HostRecurrentOffloadDecision]:
    """Select the evidenced CUDA recurrent path without moving state ownership."""

    try:
        gpu_profile = _resolve_component_profile(
            scenario, rank.component_id, GPUProfile
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    capability = gpu_profile.host_recurrent_offload
    if capability is None:
        return None

    geometry = layer.linear_attention
    minimum_m = int(capability.minimum_m)
    evidence = str(capability.evidence)
    provenance = str(capability.provenance)
    supported_ops = tuple(str(item) for item in capability.supported_ops)
    state_bytes = _linear_state_bytes(layer, plan.tp_degree)
    execution_component_id = str(placement_component_id)
    state_owner_component_id: Optional[str] = None
    applied = False
    placement_is_cpu = (
        _kind(_component(scenario, placement_component_id)) == "cpu"
    )
    if placement_is_cpu:
        try:
            state_owner_component_id, _offload, _ratio = (
                _linear_state_components(
                    scenario,
                    rank,
                    placement_component_id,
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            state_owner_component_id = None

    if not placement_is_cpu:
        reason = "placement_not_cpu"
    elif state_owner_component_id is None:
        reason = "state_owner_unavailable"
    elif not capability.op_offload:
        reason = "op_offload_disabled"
    elif physical_m < minimum_m:
        reason = "physical_m_below_minimum"
    elif capability.architecture != scenario.model.architecture:
        reason = "architecture_unsupported"
    elif geometry is None:
        reason = "linear_geometry_missing"
    elif {
        item.strip().casefold() for item in supported_ops
    } != _HOST_RECURRENT_OFFLOAD_REQUIRED_OPS:
        reason = "operator_set_unsupported"
    elif (
        int(capability.query_width) != geometry.query_width
        or int(capability.key_width) != geometry.key_width
        or int(capability.value_width) != geometry.value_width
        or int(capability.conv_kernel_size) != geometry.conv_kernel_size
        or capability.state_dtype.strip().casefold()
        != geometry.state_dtype.strip().casefold()
    ):
        reason = "shape_contract_unsupported"
    elif host_projection_decision is None:
        reason = "host_gemm_capability_missing"
    elif not host_projection_decision.applied:
        reason = "host_projection_not_offloaded"
    else:
        state_h2d = _route_is_available(
            router,
            state_owner_component_id,
            rank.component_id,
            state_bytes,
            routing_policy=plan.routing_policy,
        )
        state_d2h = _route_is_available(
            router,
            rank.component_id,
            state_owner_component_id,
            state_bytes,
            routing_policy=plan.routing_policy,
        )
        if not state_h2d:
            reason = "state_h2d_route_unavailable"
        elif not state_d2h:
            reason = "state_d2h_route_unavailable"
        else:
            execution_component_id = rank.component_id
            applied = True
            reason = "eligible_host_recurrent_subgraph"

    return _HostRecurrentOffloadDecision(
        placement_component_id=str(placement_component_id),
        execution_component_id=execution_component_id,
        state_owner_component_id=state_owner_component_id,
        applied=applied,
        reason=reason,
        evidence=evidence,
        provenance=provenance,
        physical_m=int(physical_m),
        minimum_m=minimum_m,
        state_bytes=state_bytes,
        supported_ops=supported_ops,
    )


def _add_linear_state_read(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    dependencies: Sequence[str],
    *,
    name: str,
    target_component_id: Optional[str] = None,
    storage_owner_component_id: Optional[str] = None,
    runtime: Optional[_LinearStateRuntimeSemantics] = None,
) -> str:
    runtime = runtime or _LinearStateRuntimeSemantics()
    state_target = str(target_component_id or rank.component_id)
    active, offload, ratio = _linear_state_components(
        scenario,
        rank,
        storage_owner_component_id or state_target,
    )
    byte_count = _linear_state_bytes(
        layer,
        plan.tp_degree,
    ) * _linear_state_persistent_owner_count(runtime)
    prefetched_bytes = int(math.ceil(byte_count * ratio))
    prior = tuple(dependencies)
    if offload and ratio > 0.0 and runtime.read_source == "committed":
        prior = (
            _add_transfer_tasks(
                builder,
                router,
                offload,
                active,
                prefetched_bytes,
                prior,
                name=name + ".prefetch",
                routing_policy=plan.routing_policy,
                metadata={
                    "event_kind": "linear_state_prefetch",
                    "operator_class": OperatorClass.MEMORY.value,
                    "state_kind": "recurrent_conv",
                    "layer_id": layer.layer_id,
                    "rank": rank.rank,
                    "bytes": prefetched_bytes,
                    "state_persistence": "committed",
                    "state_lifecycle": "prefetch",
                    **runtime.audit_metadata(),
                },
            ),
        )
    state_read = _add_transfer_tasks(
        builder,
        router,
        active,
        state_target,
        byte_count,
        prior,
        name=name + ".read",
        routing_policy=plan.routing_policy,
        metadata={
            "event_kind": "linear_state_read",
            "operator_class": OperatorClass.MEMORY.value,
            "state_kind": "recurrent_conv",
            "layer_id": layer.layer_id,
            "rank": rank.rank,
            "bytes": byte_count,
            "state_persistence": (
                "temporary_speculative"
                if runtime.read_source == "speculative"
                else "committed"
            ),
            "state_lifecycle": "read",
            "state_storage": (
                "rolling_speculative_buffer"
                if runtime.read_source == "speculative"
                else "persistent_state"
            ),
            **runtime.audit_metadata(),
        },
    )
    # Recurrent state is a side input to the mixer, not the layer's primary
    # activation.  Do not let its branch overwrite activation liveness at the
    # subsequent join.
    return _discard_side_branch_rank_value(builder, state_read, rank)


def _add_linear_state_write(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    dependencies: Sequence[str],
    *,
    name: str,
    source_component_id: Optional[str] = None,
    target_component_id: Optional[str] = None,
    storage_owner_component_id: Optional[str] = None,
    runtime: Optional[_LinearStateRuntimeSemantics] = None,
) -> str:
    runtime = runtime or _LinearStateRuntimeSemantics()
    active, offload, ratio = _linear_state_components(
        scenario,
        rank,
        storage_owner_component_id or target_component_id,
    )
    byte_count = _linear_state_bytes(
        layer,
        plan.tp_degree,
    ) * _linear_state_persistent_owner_count(runtime)
    offloaded_bytes = int(math.ceil(byte_count * ratio))
    source_component = source_component_id or (
        builder.rank_value_component(dependencies, rank.rank)
        or rank.component_id
    )
    written = _add_transfer_tasks(
        builder,
        router,
        source_component,
        active,
        byte_count,
        dependencies,
        name=name + ".write",
        routing_policy=plan.routing_policy,
        metadata={
            "event_kind": "linear_state_write",
            "operator_class": OperatorClass.MEMORY.value,
            "state_kind": "recurrent_conv",
            "layer_id": layer.layer_id,
            "rank": rank.rank,
            "bytes": byte_count,
            "state_persistence": "committed",
            "state_lifecycle": "commit",
            "state_storage": "persistent_state",
            "state_commit_source": (
                "retained_committed_prefix_snapshot"
                if runtime.uses_temporary_state
                else "current_state_update"
            ),
            **runtime.audit_metadata(),
        },
    )
    if not offload or ratio <= 0.0:
        return _discard_side_branch_rank_value(builder, written, rank)
    offloaded = _add_transfer_tasks(
        builder,
        router,
        active,
        offload,
        offloaded_bytes,
        (written,),
        name=name + ".offload",
        routing_policy=plan.routing_policy,
        metadata={
            "event_kind": "linear_state_offload",
            "operator_class": OperatorClass.MEMORY.value,
            "state_kind": "recurrent_conv",
            "layer_id": layer.layer_id,
            "rank": rank.rank,
            "bytes": offloaded_bytes,
            "state_persistence": "committed",
            "state_lifecycle": "offload",
            "state_storage": "persistent_state",
            **runtime.audit_metadata(),
        },
    )
    return _discard_side_branch_rank_value(builder, offloaded, rank)


def _add_linear_state_materialize(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    dependencies: Sequence[str],
    *,
    name: str,
    source_component_id: str,
    target_component_id: Optional[str] = None,
    storage_owner_component_id: Optional[str] = None,
    runtime: _LinearStateRuntimeSemantics,
) -> str:
    """Write the rolling verifier state without changing persistent state."""

    active, _offload, _ratio = _linear_state_components(
        scenario,
        rank,
        storage_owner_component_id or target_component_id,
    )
    byte_count = _linear_state_bytes(layer, plan.tp_degree)
    materialized = _add_transfer_tasks(
        builder,
        router,
        source_component_id,
        active,
        byte_count,
        dependencies,
        name=name + ".materialize",
        routing_policy=plan.routing_policy,
        metadata={
            "event_kind": "linear_state_materialize",
            "operator_class": OperatorClass.MEMORY.value,
            "state_kind": "recurrent_conv",
            "layer_id": layer.layer_id,
            "rank": rank.rank,
            "bytes": byte_count,
            "memory_direction": "write",
            "state_persistence": "temporary_speculative",
            "state_lifecycle": "materialize",
            "state_storage": "rolling_speculative_buffer",
            **runtime.audit_metadata(),
        },
    )
    return _discard_side_branch_rank_value(builder, materialized, rank)


def _add_linear_state_release(
    builder: _TaskBuilder,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    dependencies: Sequence[str],
    *,
    name: str,
    runtime: _LinearStateRuntimeSemantics,
) -> str:
    """End one verifier scratch lifetime after its last materialized state."""

    released = _add_join(
        builder,
        name + ".release",
        dependencies,
        metadata={
            "event_kind": "linear_state_release",
            "operator_class": OperatorClass.MEMORY.value,
            "state_kind": "recurrent_conv",
            "layer_id": layer.layer_id,
            "rank": rank.rank,
            "bytes": 0,
            "released_bytes": _linear_state_bytes(layer, plan.tp_degree),
            "resource_accounting": "lifecycle_only",
            "state_persistence": "temporary_speculative",
            "state_lifecycle": "release",
            "state_storage": "rolling_speculative_buffer",
            **runtime.audit_metadata(),
        },
    )
    return _discard_side_branch_rank_value(builder, released, rank)


def _kv_components(
    scenario: ScenarioConfig,
    rank: LogicalRank,
    target_component_id: Optional[str] = None,
    layer: Optional[LayerSpec] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    # llama.cpp allocates KV pages with the layer's execution arena when
    # --kvo is enabled.  Keep the authored policy as the fallback, but honor
    # an explicit per-layer map emitted by the native parity adapter.
    policy = _kv_configuration(scenario)
    configured_cache = (
        policy.cache_component
    ) or scenario.placement.tensor_to_component.get(
        "kv_cache"
    ) or rank.memory_component_id
    layer_map = scenario.placement.metadata.get("llama_cpp_kv_layer_components", {})
    mapped_cache = None
    if layer is not None and isinstance(layer_map, Mapping):
        value = layer_map.get(str(layer.layer_id))
        if value:
            mapped_cache = str(value)
    cache = mapped_cache or _layer_local_runtime_memory_component_id(
        scenario,
        rank,
        target_component_id,
        str(configured_cache) if configured_cache else None,
    )
    offload = policy.offload_component
    ratio = float(policy.offload_ratio)
    return (str(cache) if cache else None, str(offload) if offload else None, ratio)


def _effective_kv_dtype(scenario: ScenarioConfig, layer: LayerSpec) -> str:
    policy_dtype = scenario.placement.kv_policy.dtype
    return str(policy_dtype) if policy_dtype else layer.dtype


def _kv_artifact_spec(
    scenario: ScenarioConfig,
) -> Optional[_ArtifactQuantizationSpec]:
    """Resolve an explicitly declared runtime KV artifact format.

    KV policy metadata is kept separate from model-weight metadata: Q4_0 is a
    runtime activation format here, so it must not accidentally make QK/PV
    operands look like model weights or apply IQ3/IQ4 FFN selection.
    """

    sources = (
        scenario.placement.metadata,
        scenario.workload.metadata,
        scenario.model.metadata,
    )
    value = _metadata_value(
        sources,
        (
            "kv_artifact_quantization",
            "artifact_kv_quantization",
            "kv_cache_quantization",
        ),
    )
    if value is None:
        return None
    label = _canonical_artifact_quantization(value)
    if label is None or label == _MIXED_ARTIFACT_QUANTIZATION:
        raise ValueError("unsupported KV artifact quantization {}".format(value))
    return _ARTIFACT_QUANTIZATION_REGISTRY[label]


def _kv_dtype_bits(
    scenario: ScenarioConfig,
    layer: LayerSpec,
) -> Tuple[int, Optional[_ArtifactQuantizationSpec]]:
    artifact = _kv_artifact_spec(scenario)
    if artifact is None:
        policy_label = _canonical_artifact_quantization(
            _effective_kv_dtype(scenario, layer)
        )
        if policy_label == "Q4_0":
            artifact = _ARTIFACT_QUANTIZATION_REGISTRY[policy_label]
    if artifact is not None:
        return artifact.compute_weight_bits, artifact
    return _dtype_bits(_effective_kv_dtype(scenario, layer)), None


def _physical_kv_width_for_rank(layer: LayerSpec, tp_degree: int) -> int:
    """Materialized K (or V) elements stored by one physical TP rank."""

    local_heads = int(math.ceil(layer.effective_kv_heads / float(tp_degree)))
    return local_heads * layer.effective_attention_head_dim


def _kv_bytes_per_token(
    scenario: ScenarioConfig, layer: LayerSpec, tp_degree: int
) -> int:
    dtype_bits, artifact = _kv_dtype_bits(scenario, layer)
    local_width = _physical_kv_width_for_rank(layer, tp_degree)
    if artifact is not None:
        blocks = int(math.ceil(local_width / float(artifact.block_size)))
        return 2 * blocks * (
            artifact.payload_bytes + artifact.metadata_bytes
        )
    return int(math.ceil(2 * local_width * dtype_bits / 8.0))


def _kv_tensor_bytes(
    scenario: ScenarioConfig,
    layer: LayerSpec,
    tp_degree: int,
    token_accesses: int,
) -> int:
    """Physical bytes for only K or only V over exact token accesses."""

    dtype_bits, artifact = _kv_dtype_bits(scenario, layer)
    local_width = _physical_kv_width_for_rank(layer, tp_degree)
    if artifact is not None:
        blocks = int(math.ceil(local_width / float(artifact.block_size)))
        return max(0, token_accesses) * blocks * (
            artifact.payload_bytes + artifact.metadata_bytes
        )
    return int(
        math.ceil(max(0, token_accesses) * local_width * dtype_bits / 8.0)
    )


def _kv_tensor_storage_metadata_bytes(
    scenario: ScenarioConfig,
    layer: LayerSpec,
    tp_degree: int,
    token_accesses: int,
) -> Tuple[int, int]:
    """Return packed payload and metadata bytes for one K or V operand."""

    _dtype_bits_value, artifact = _kv_dtype_bits(scenario, layer)
    if artifact is None:
        return _kv_tensor_bytes(scenario, layer, tp_degree, token_accesses), 0
    local_width = _physical_kv_width_for_rank(layer, tp_degree)
    blocks = int(math.ceil(local_width / float(artifact.block_size)))
    accesses = max(0, token_accesses)
    return (
        accesses * blocks * artifact.payload_bytes,
        accesses * blocks * artifact.metadata_bytes,
    )


def _kv_fused_attention_physical_contract(
    scenario: ScenarioConfig,
    layer: LayerSpec,
    tp_degree: int,
) -> Tuple[int, Optional[FusedAttentionKVPhysicalContract]]:
    """Derive exact rank-local KV bytes for a fused attention invocation."""

    dtype_bits, artifact = _kv_dtype_bits(scenario, layer)
    if artifact is None:
        return dtype_bits, None

    payload_per_operand_token, metadata_per_operand_token = (
        _kv_tensor_storage_metadata_bytes(scenario, layer, tp_degree, 1)
    )
    local_width = _physical_kv_width_for_rank(layer, tp_degree)
    block_count = int(math.ceil(local_width / float(artifact.block_size)))
    return dtype_bits, FusedAttentionKVPhysicalContract(
        payload_bytes_per_token=2 * payload_per_operand_token,
        metadata_bytes_per_token=2 * metadata_per_operand_token,
        dequant_operations_per_token=(
            2
            * block_count
            * artifact.block_size
            * artifact.dequant_operations_per_weight
        ),
        artifact_format=artifact.name,
    )


def _logical_kv_bytes_per_token_for_rank(
    scenario: ScenarioConfig,
    layer: LayerSpec,
    tp_degree: int,
    tp_rank: int,
) -> int:
    """Logical (non-replicated) KV share attributed to one TP rank."""

    dtype_bits, _artifact = _kv_dtype_bits(scenario, layer)
    head_dim = layer.effective_attention_head_dim
    base, remainder = divmod(layer.effective_kv_heads, max(1, tp_degree))
    logical_heads = base + (1 if tp_rank < remainder else 0)
    return int(math.ceil(2 * logical_heads * head_dim * dtype_bits / 8.0))


def _add_kv_access(
    builder: _TaskBuilder,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    source_component: str,
    target_component: str,
    byte_count: int,
    dependencies: Sequence[str],
    *,
    name: str,
    metadata: Mapping[str, object],
) -> str:
    """Lower a KV access without double charging rank-local HBM traffic.

    The local QKV producer and attention consumer already include append/read
    traffic in their GPU/HBM rooflines.  A rank-local KV event is therefore
    retained as logical bookkeeping, while only a remote placement creates an
    additional topology transfer.
    """

    payload = {
        **dict(metadata),
        "operator_class": OperatorClass.MEMORY.value,
        "kv_access_id": str(name),
        "bytes": max(0, int(byte_count)),
        "physical_bytes": max(0, int(byte_count)),
        "source_component": source_component,
        "target_component": target_component,
    }
    if source_component == target_component or {
        source_component,
        target_component,
    } == {rank.component_id, rank.memory_component_id}:
        local = _add_join(
            builder,
            name + ".local",
            dependencies,
            metadata={
                **payload,
                "resource_accounting": "included_in_attention_kernel",
                "resource_transfer_bytes": 0,
                "source_component": source_component,
                "target_component": target_component,
            },
        )
        return _discard_side_branch_rank_value(builder, local, rank)
    remote = _add_transfer_tasks(
        builder,
        router,
        source_component,
        target_component,
        byte_count,
        dependencies,
        name=name,
        routing_policy=plan.routing_policy,
        metadata={
            **payload,
            "resource_accounting": "explicit_remote_transfer",
            "resource_transfer_bytes": max(0, int(byte_count)),
        },
    )
    return _discard_side_branch_rank_value(builder, remote, rank)


def _add_kv_read(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    read_token_count: int,
    dependencies: Sequence[str],
    *,
    name: str,
    target_component_id: Optional[str] = None,
) -> str:
    cache, _offload, _ratio = _kv_components(
        scenario,
        rank,
        target_component_id,
        layer,
    )
    if read_token_count <= 0:
        skipped = _add_join(
            builder,
            name + ".none",
            dependencies,
            metadata={
                "event_kind": "kv_read_skipped",
                "reason": "no_prior_kv",
                "rank": rank.rank,
                "layer_id": layer.layer_id,
                "bytes": 0,
                "logical_bytes": 0,
                "physical_bytes": 0,
            },
        )
        return _discard_side_branch_rank_value(builder, skipped, rank)
    if not cache:
        raise ValueError(
            "full-attention rank {} layer {} has no writable KV cache "
            "placement".format(rank.rank, layer.layer_id)
        )
    byte_count = (
        _kv_bytes_per_token(scenario, layer, plan.tp_degree) * read_token_count
    )
    logical_bytes = _logical_kv_bytes_per_token_for_rank(
        scenario, layer, plan.tp_degree, rank.tp_rank
    ) * read_token_count
    access_component = _layer_local_runtime_memory_component_id(
        scenario,
        rank,
        target_component_id,
        rank.memory_component_id or rank.component_id,
    )
    if access_component is None:
        raise ValueError(
            "full-attention rank {} layer {} has no local KV access component"
            .format(rank.rank, layer.layer_id)
        )
    task_id = _add_kv_access(
        builder,
        router,
        plan,
        rank,
        cache,
        access_component,
        byte_count,
        dependencies,
        name=name + ".read",
        metadata={
            "event_kind": "kv_read",
            "rank": rank.rank,
            "layer_id": layer.layer_id,
            "coverage_component": "full_attention",
            "kv_cache_component": cache,
            "logical_bytes": logical_bytes,
            "kv_token_accesses": read_token_count,
            "memory_direction": "read",
        },
    )
    if (
        builder.tasks
        and builder.tasks[-1].task_id == task_id
        and not builder.tasks[-1].demands
        and builder.tasks[-1].metadata.get("event_kind") == "kv_read"
        and builder.tasks[-1].metadata.get("resource_accounting")
        == "included_in_attention_kernel"
    ):
        builder._task_segment_dynamic_payloads[task_id] = (
            _KVReadTaskReplayPayload(
                layer=layer,
                rank=rank,
                tp_degree=plan.tp_degree,
            )
        )
    return task_id


def _task_segment_dynamic_task_overrides(
    payloads: Sequence[Optional[object]],
    replay: _TaskSegmentDynamicReplayContext,
) -> Optional[
    Mapping[int, Tuple[Tuple[ResourceDemand, ...], Mapping[str, object]]]
]:
    """Materialize the exact context-shaped slots in one compiled invocation.

    The serving emitter is admitted only after capture proved that every full
    attention invocation used the fused local-KV path.  Re-evaluate the fusion
    guard before mutating a builder, then run the original analytical estimator
    and namespace every demand in the same order as uncached lowering.
    """

    context_tokens = int(replay.context_tokens)
    kv_read_tokens = int(replay.kv_read_tokens)
    if context_tokens <= 0 or kv_read_tokens <= 0:
        return None
    scenario = replay.scenario
    overrides: Dict[
        int, Tuple[Tuple[ResourceDemand, ...], Mapping[str, object]]
    ] = {}
    estimates: Dict[Hashable, CostEstimate] = {}
    for task_index, payload in enumerate(payloads):
        if isinstance(payload, _KVReadTaskReplayPayload):
            byte_count = (
                _kv_bytes_per_token(
                    scenario,
                    payload.layer,
                    payload.tp_degree,
                )
                * kv_read_tokens
            )
            logical_bytes = (
                _logical_kv_bytes_per_token_for_rank(
                    scenario,
                    payload.layer,
                    payload.tp_degree,
                    payload.rank.tp_rank,
                )
                * kv_read_tokens
            )
            overrides[task_index] = (
                (),
                {
                    "logical_bytes": logical_bytes,
                    "kv_token_accesses": kv_read_tokens,
                    "bytes": byte_count,
                    "physical_bytes": byte_count,
                },
            )
            continue
        if isinstance(payload, _DynamicAttentionCostTaskReplayPayload):
            score_elements = max(
                1,
                payload.score_heads
                * payload.token_batch
                * context_tokens,
            )
            if payload.role == "qk":
                workload = replace(
                    payload.workload,
                    n=context_tokens,
                    weight_storage_bytes=(
                        _kv_tensor_storage_metadata_bytes(
                            scenario,
                            payload.layer,
                            payload.tp_degree,
                            kv_read_tokens,
                        )[0]
                    ),
                    weight_metadata_bytes=(
                        _kv_tensor_storage_metadata_bytes(
                            scenario,
                            payload.layer,
                            payload.tp_degree,
                            kv_read_tokens,
                        )[1]
                    ),
                    output_storage_bytes=_activation_bytes(
                        payload.layer,
                        score_elements,
                        scenario=scenario,
                    ),
                )
            elif payload.role == "pv":
                workload = replace(
                    payload.workload,
                    k=context_tokens,
                    weight_storage_bytes=(
                        _kv_tensor_storage_metadata_bytes(
                            scenario,
                            payload.layer,
                            payload.tp_degree,
                            kv_read_tokens,
                        )[0]
                    ),
                    weight_metadata_bytes=(
                        _kv_tensor_storage_metadata_bytes(
                            scenario,
                            payload.layer,
                            payload.tp_degree,
                            kv_read_tokens,
                        )[1]
                    ),
                    activation_storage_bytes=_activation_bytes(
                        payload.layer,
                        score_elements,
                        scenario=scenario,
                    ),
                )
            elif payload.role == "qk_scale":
                workload = replace(
                    payload.workload,
                    elements=score_elements,
                )
            elif payload.role == "softmax_reduce":
                workload = replace(
                    payload.workload,
                    input_elements=score_elements,
                    dependency_depth=max(
                        1,
                        int(
                            math.ceil(
                                math.log2(max(1, context_tokens))
                            )
                        ),
                    ),
                    working_set_bytes=_activation_bytes(
                        payload.layer,
                        score_elements,
                        scenario=scenario,
                    ),
                )
            elif payload.role == "softmax_normalize":
                workload = replace(
                    payload.workload,
                    elements=score_elements,
                    working_set_bytes=_activation_bytes(
                        payload.layer,
                        score_elements * 2,
                        scenario=scenario,
                    ),
                )
            else:  # pragma: no cover - capture validates the closed role set
                return None
            fused_workload = replace(
                payload.fused_workload,
                context_tokens=context_tokens,
                kv_read_tokens=kv_read_tokens,
            )
            flash_allowed, flash_audit = _same_rank_gpu_fusion_decision(
                scenario,
                "flash_attention",
                payload.rank,
                fused_workload.onchip_working_set_bytes,
                payload.fusion_targets,
            )
            if flash_allowed:
                return None
            estimate_key: Hashable = (
                "dynamic_attention_cost",
                payload.operator_class,
                payload.target_component_id,
                payload.rank.memory_component_id,
                workload,
            )
            estimate = estimates.get(estimate_key)
            if estimate is None:
                if payload.operator_class == OperatorClass.GEMM:
                    target_kind = _kind(
                        _component(scenario, payload.target_component_id)
                    )
                    if target_kind == "cpu":
                        cpu_profile, host_memory_profile = _cpu_profiles(
                            scenario,
                            payload.target_component_id,
                        )
                        estimate = _memoized_cost_estimate(
                            scenario,
                            _cpu_gemm_cost_key(payload.target_component_id, workload),
                            lambda: estimate_cpu_gemm(
                                cpu_profile,
                                host_memory_profile,
                                workload,
                            ),
                        )
                    elif target_kind == "gpu":
                        gpu_profile, hbm_profile = _gpu_profiles(
                            scenario,
                            payload.target_component_id,
                            payload.rank.memory_component_id,
                        )
                        estimate = _memoized_cost_estimate(
                            scenario,
                            (
                                "gpu_gemm",
                                payload.target_component_id,
                                payload.rank.memory_component_id,
                                workload,
                            ),
                            lambda: estimate_gpu_gemm(
                                gpu_profile,
                                hbm_profile,
                                workload,
                            ),
                        )
                    else:
                        return None
                else:
                    estimate = _estimate_typed_primitive(
                        scenario,
                        payload.target_component_id,
                        payload.operator_class,
                        workload,
                        memory_component_id=(
                            payload.rank.memory_component_id
                        ),
                    )
                estimates[estimate_key] = estimate
            if (
                payload.phase_index < 0
                or payload.phase_index >= len(estimate.phases)
            ):
                return None
            phase = estimate.phases[payload.phase_index]
            if phase.name != payload.phase_name:
                return None
            demands = tuple(
                _namespace_demand(
                    scenario,
                    demand,
                    rank=payload.rank,
                    target_component_id=payload.target_component_id,
                )
                for demand in phase.demands
            )
            cost_model_suffix = dict(payload.cost_model_suffix)
            workload_metadata_updates: Dict[str, object] = {}
            for metadata_key in (
                "weight_storage_bytes",
                "weight_metadata_bytes",
                "weight_bytes",
            ):
                if metadata_key not in cost_model_suffix:
                    continue
                metadata_value = getattr(workload, metadata_key)
                cost_model_suffix[metadata_key] = metadata_value
                workload_metadata_updates[metadata_key] = metadata_value
            dynamic_metadata: Dict[str, object] = {
                **dict(flash_audit),
                **workload_metadata_updates,
                "cost_model": {
                    **dict(estimate.metadata),
                    **cost_model_suffix,
                },
                "phase_metadata": dict(phase.metadata),
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands),
                    default=0.0,
                ),
            }
            if payload.operator_class == OperatorClass.GEMM and phase.name == "cpu_gemm":
                # Attention replay always has an activation RHS; refresh shape audit
                # while retaining the historical generic cost key and estimator.
                _, replay_iq_audit = _llama_source_cpu_iq_panel_dispatch(
                    scenario, workload, _component(scenario, payload.target_component_id), {},
                    model_weight_read=False, rhs_is_activation=True,
                )
                if replay_iq_audit is not None:
                    dynamic_metadata["cpu_iq_panel_reuse"] = replay_iq_audit
            if payload.role == "qk_scale":
                dynamic_metadata["score_elements"] = score_elements
            if (
                payload.operator_class == OperatorClass.GEMM
                and _f32_hidden_storage_enabled(scenario)
            ):
                dynamic_metadata["runtime_output_storage_bytes"] = workload.output_bytes
            if payload.operator_class != OperatorClass.GEMM:
                _read_bytes, output_bytes = _primitive_io_bytes(workload)
                dynamic_metadata["output_bytes"] = output_bytes
            overrides[task_index] = (demands, dynamic_metadata)
            continue
        if not isinstance(payload, _FusedAttentionTaskReplayPayload):
            continue
        workload = replace(
            payload.workload,
            context_tokens=context_tokens,
            kv_read_tokens=kv_read_tokens,
        )
        if not payload.fusion_targets:
            return None
        flash_allowed, flash_audit = _same_rank_gpu_fusion_decision(
            scenario,
            "flash_attention",
            payload.rank,
            workload.onchip_working_set_bytes,
            payload.fusion_targets,
        )
        if not flash_allowed:
            return None
        estimate_key = (payload.rank, workload)
        estimate = estimates.get(estimate_key)
        if estimate is None:
            gpu_profile, hbm_profile = _gpu_profiles(
                scenario,
                payload.rank.component_id,
                payload.rank.memory_component_id,
            )
            estimate = _memoized_cost_estimate(
                scenario,
                (
                    "gpu_fused_attention",
                    payload.rank.component_id,
                    payload.rank.memory_component_id,
                    workload,
                ),
                lambda: estimate_gpu_fused_attention(
                    gpu_profile,
                    hbm_profile,
                    workload,
                ),
            )
            estimates[estimate_key] = estimate
        if (
            payload.phase_index < 0
            or payload.phase_index >= len(estimate.phases)
        ):
            return None
        phase = estimate.phases[payload.phase_index]
        if phase.name != payload.phase_name:
            return None
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=payload.rank,
                target_component_id=payload.rank.component_id,
            )
            for demand in phase.demands
        )
        overrides[task_index] = (
            demands,
            {
                **dict(flash_audit),
                "cost_model": dict(estimate.metadata),
                "phase_metadata": dict(phase.metadata),
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
                "analytical_service_ns": max(
                    (demand.service_ns for demand in demands),
                    default=0.0,
                ),
            },
        )
    return overrides


def _native_local_kv_contract(
    scenario: ScenarioConfig, router: TopologyRouter, plan: ParallelPlan,
    rank: LogicalRank, layer: LayerSpec, token_batch: int,
    kv_materialized_tokens: int, kv_append_tokens: int,
) -> Mapping[str, object]:
    """Qualify the fixed ordinary CUDA cache graph before changing QKV output."""
    if scenario.workload.metadata.get("llama_cpp_mmq_source_work") is not True:
        return {}
    audit: Dict[str, object] = {"status": "uncovered"}
    declaration = scenario.model.metadata.get("llama_cpp_native_kv_writeback")
    if not isinstance(declaration, Mapping) or (
        declaration.get("backend_commit") != "0f3a71be15af836d277c9f918adfafb45732677e"
        or declaration.get("rope_type") not in {"normal", "neox", "imrope"}
        or any(declaration.get(key) is not True for key in (
            "unified_non_transposed_cache", "ordinary_internal_k",
            "fusion_environment_enabled", "explicit_flash_attention",
        ))
    ):
        return {**audit, "reason": "ordinary_cache_source_contract_not_declared"}
    if not _f32_hidden_storage_enabled(scenario) or not _declared_physical_projections(
        scenario, layer, ("attention.q", "attention.k", "attention.v"),
        combined_projection_id="attention.qkv",
    ):
        return {**audit, "reason": "f32_physical_qkv_required"}
    mtp = _mtp_configuration(scenario)
    if (plan.world_size != 1 or (mtp is not None and mtp.enabled)
            or token_batch <= 0 or token_batch != kv_materialized_tokens
            or token_batch != kv_append_tokens):
        return {**audit, "reason": "ordinary_single_rank_equal_cache_rows_required"}
    target = _parallel_target(scenario, layer, "attention", rank)
    cache, _offload, ratio = _kv_components(scenario, rank, target)
    if (target != rank.component_id or _kind(_component(scenario, target)) != "gpu"
            or cache not in {rank.component_id, rank.memory_component_id} or ratio != 0):
        return {**audit, "reason": "local_gpu_qkv_and_cache_without_offload_required"}
    primitive_names = [(OperatorClass.ELEMENTWISE, "rope", "attention")]
    execution = _attention_execution_descriptor(layer)
    if execution is None and layer.hidden_size != layer.attention_heads * layer.effective_attention_head_dim:
        return {**audit, "reason": "nonstandard_query_width_needs_attention_execution_descriptor"}
    if execution is not None and execution.qk_norm:
        primitive_names.extend((kind, suffix, "norm") for kind, suffix in (
            (OperatorClass.REDUCTION, "q_norm.reduce"),
            (OperatorClass.ELEMENTWISE, "q_norm.apply"),
            (OperatorClass.REDUCTION, "k_norm.reduce"),
            (OperatorClass.ELEMENTWISE, "k_norm.apply"),
        ))
    if any(_primitive_target(scenario, router, rank, kind,
                             "{}.attention.{}".format(layer.layer_id, suffix),
                             fallback_keys=("{}.{}".format(layer.layer_id, fallback),)) != target
           for kind, suffix, fallback in primitive_names):
        return {**audit, "reason": "qk_norm_and_rope_must_remain_on_producer_gpu"}
    bits, artifact = _kv_dtype_bits(scenario, layer)
    cache_format = artifact.name if artifact is not None else "f{}".format(bits)
    if cache_format not in {"f16", "f32", "Q4_0", "Q8_0"}:
        return {**audit, "reason": "cache_format_source_not_covered"}
    cache_width = _physical_kv_width_for_rank(layer, plan.tp_degree)
    expected_widths = (
        execution.q_projection_width if execution is not None else layer.hidden_size,
        cache_width, cache_width,
    )
    if any(resolve_weight_projection(layer.metadata, projection)[0].n != width
           for projection, width in zip(("attention.q", "attention.k", "attention.v"), expected_widths)):
        return {**audit, "reason": "physical_qkv_and_cache_row_widths_disagree"}
    if artifact is not None and cache_width % artifact.block_size:
        return {**audit, "reason": "quantized_set_rows_requires_aligned_row_width"}
    # The positive three-node fusion proof excludes Q/K normalization graphs.
    fused_k = (artifact is None and declaration["rope_type"] in {"normal", "neox"}
               and not (execution is not None and execution.qk_norm))
    if (artifact is None and declaration["rope_type"] in {"normal", "neox"}
            and execution is not None and execution.qk_norm):
        return {**audit, "reason": "normalized_rope_fusion_source_not_covered"}
    return {
        **dict(declaration), "status": "applied", "cache_format": cache_format,
        "cache_component": cache, "producer_component": target,
        "k_rope_set_rows_fused": fused_k, "rows": token_batch,
        "hadamard_unpriced": artifact is not None and layer.effective_attention_head_dim % 64 == 0,
        "unpriced_terms": ("set_rows_conversion_instructions_and_control",
                           "repeated_index_requests", "applicable_qkv_hadamard_work"),
        "rope_arithmetic": "inherited_three_ops_and_precomputed_sin_cos_approximation",
    }


def _add_native_local_kv_writeback(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, rank: LogicalRank, layer: LayerSpec,
    token_batch: int, query_width: int, kv_width: int,
    query_rotated_width: int, key_rotated_width: int,
    dependencies: Sequence[str], *, name: str, contract: Mapping[str, object],
    metadata: Mapping[str, object],
    already_rotated: Optional[Tuple[str, str]] = None,
) -> Tuple[str, str]:
    """Keep Q rotation, replace K's destination when fused, then write cache once."""
    common = {key: value for key, value in metadata.items() if key != "phase"}
    common["execution_phase"] = metadata.get("phase")
    target_bytes = _kv_tensor_bytes(scenario, layer, plan.tp_degree, token_batch)
    index_bytes = 8 * token_batch
    prior = tuple(dependencies)
    query_end = ""
    if already_rotated is not None:
        if contract["k_rope_set_rows_fused"] or not all(already_rotated):
            raise ValueError("pre-rotated source values require separate cache writes")
        query_end, key_end = already_rotated
        prior = (key_end,)
    rotation_parts = () if already_rotated is not None else (
        ("q", query_width, query_rotated_width), ("k", kv_width, key_rotated_width))
    for part, width, rotated_width in rotation_parts:
        fused = part == "k" and contract["k_rope_set_rows_fused"]
        elements, rotated = token_batch * width, token_batch * rotated_width
        read_bytes = 4 * elements + 8 * rotated + (index_bytes if fused else 0)
        write_bytes = target_bytes if fused else 4 * elements
        end = _add_rank_tensor_kernel(
            builder, scenario, router, plan, rank,
            TensorKernelWorkload(operations=3 * rotated, read_bytes=read_bytes,
                write_bytes=write_bytes, dependency_depth=3,
                working_set_bytes=read_bytes + write_bytes, reuse_factor=2.0,
                name="native_{}_rope".format(part)),
            name + "." + part + "_rope", prior, input_is_local=True,
            metadata={**common, "event_kind": "kv_native_k_rope" if fused else "rope",
                "modeled_memory_write_bytes": write_bytes,
                "native_kv_work": {**contract, "stage": part + "_rope",
                    "read_bytes": read_bytes, "write_bytes": write_bytes,
                    "persistent_write_bytes": target_bytes if fused else 0,
                    "index_unique_bytes": index_bytes if fused else 0},
                "persistent_output_bytes": target_bytes if fused else 0},
        )
        if part == "q":
            query_end = end
        else:
            _discard_side_branch_rank_value(builder, end, rank)
        prior = (end,)
    for part in (("v",) if contract["k_rope_set_rows_fused"] else ("k", "v")):
        read_bytes = 4 * token_batch * kv_width + index_bytes
        end = _add_rank_tensor_kernel(
            builder, scenario, router, plan, rank,
            TensorKernelWorkload(operations=0, read_bytes=read_bytes,
                write_bytes=target_bytes, streaming_fraction=1.0,
                name="native_{}_set_rows".format(part)),
            name + "." + part + "_set_rows", prior, input_is_local=True,
            metadata={**common, "event_kind": "kv_native_set_rows",
                "modeled_memory_write_bytes": target_bytes,
                "native_kv_work": {**contract, "stage": part + "_set_rows",
                    "read_bytes": read_bytes, "write_bytes": target_bytes,
                    "persistent_write_bytes": target_bytes, "index_unique_bytes": index_bytes},
                **({"input_tensor_id": contract[part + "_input_tensor_id"]}
                   if part + "_input_tensor_id" in contract else {}),
                "persistent_output_bytes": target_bytes},
        )
        prior = (_discard_side_branch_rank_value(builder, end, rank),)
    complete = _add_join(builder, name + ".append_complete", prior, metadata={
        **common, "event_kind": "kv_append", "rank": rank.rank,
        "kv_token_appends": token_batch, "bytes": 2 * target_bytes,
        "logical_bytes": _logical_kv_bytes_per_token_for_rank(
            scenario, layer, plan.tp_degree, rank.tp_rank) * token_batch,
        "physical_bytes": 2 * target_bytes, "memory_direction": "write",
        "resource_accounting": "native_cache_write_kernels", "resource_transfer_bytes": 0,
        "native_kv_work": {**contract, "stage": "append_complete"},
    })
    return query_end, _discard_side_branch_rank_value(builder, complete, rank)


def _qwen35_cuda_source_layer(
    scenario: ScenarioConfig, plan: ParallelPlan, rank: LogicalRank,
    layer: LayerSpec, source: _Qwen35AttentionSourceWork,
) -> bool:
    """Use verified dimensions on CPU; CUDA work requires the physical GPU path."""
    inventory = scenario.model.metadata.get("gguf_inventory", {})
    if not isinstance(inventory, Mapping) or inventory.get("sha256") != source.gguf_sha256:
        raise ValueError("Qwen3.5 source work does not match the model's actual GGUF inventory")
    if plan.world_size != 1 or scenario.workload.mtp is not None or not _f32_hidden_storage_enabled(scenario):
        raise ValueError("Qwen3.5 source geometry requires single-rank ordinary F32 execution")
    target = _parallel_target(scenario, layer, "attention", rank)
    kind = _kind(_component(scenario, target))
    if kind not in {"cpu", "gpu"}:
        raise ValueError("Qwen3.5 source geometry has no supported CPU/GPU execution path")
    return kind == "gpu"


def _qwen35_source_execution_contract(
    scenario: ScenarioConfig, router: TopologyRouter, plan: ParallelPlan,
    rank: LogicalRank, layer: LayerSpec, source: _Qwen35AttentionSourceWork,
    qkv_target: str, native_kv: Mapping[str, object],
) -> Mapping[str, object]:
    """Qualify the source-derived CUDA graph after explicit CPU separation."""
    inventory = scenario.model.metadata.get("gguf_inventory", {})
    if not isinstance(inventory, Mapping) or inventory.get("sha256") != source.gguf_sha256:
        raise ValueError("Qwen3.5 source work does not match the model's actual GGUF inventory")
    cache_format = str(native_kv.get("cache_format", "")).casefold()
    if (plan.world_size != 1 or scenario.workload.mtp is not None
            or not _f32_hidden_storage_enabled(scenario) or qkv_target != rank.component_id
            or _kind(_component(scenario, rank.component_id)) != "gpu"
            or native_kv.get("status") != "applied" or native_kv.get("rope_type") != "imrope"
            or native_kv.get("k_rope_set_rows_fused") is not False
            or cache_format not in {"f16", "q4_0"}):
        raise ValueError("Qwen3.5 source attention requires its single-GPU F32/native-KV graph contract")
    for operator_class, suffix, fallback in (
        (OperatorClass.REDUCTION, "attention.q_norm.reduce", "norm"),
        (OperatorClass.ELEMENTWISE, "attention.q_norm.apply", "norm"),
        (OperatorClass.REDUCTION, "attention.k_norm.reduce", "norm"),
        (OperatorClass.ELEMENTWISE, "attention.k_norm.apply", "norm"),
        (OperatorClass.ELEMENTWISE, "attention.rope", "attention"),
        (OperatorClass.ELEMENTWISE, "attention.gate", "attention"),
    ):
        target = _primitive_target(scenario, router, rank, operator_class,
            layer.layer_id + "." + suffix, fallback_keys=(layer.layer_id + "." + fallback,))
        if target != rank.component_id:
            raise ValueError("Qwen3.5 source fused tensors must stay in the same CUDA partition")
    gpu = _component(scenario, rank.component_id)
    mmq_contract = gpu.metadata.get("llama_cpp_mmq_contract", {})
    if (not isinstance(mmq_contract, Mapping)
            or mmq_contract.get("backend_commit") != native_kv.get("backend_commit")
            or gpu.metadata.get("cuda_compute_capability") != 1200
            or not scenario.fusion_policy.flash_attention):
        raise ValueError("Qwen3.5 source backend/default graph conditions are not declared")
    return {**native_kv, "qwen35_attention_source": True,
            "graph_order": "default_q_v_k_then_cache_fa_gate",
            "graph_fusions": ("q_rms_norm_mul", "k_rms_norm_mul", "sigmoid_mul"),
            "rope_allocation_alias": "same_address_source_rule_derived",
            "norm_allocation_alias": "distinct_from_input_source_rule_derived",
            "hadamard_enabled": cache_format == "q4_0",
            "hadamard_unpriced": False,
            "rope_arithmetic": "fixed_default_imrope_source_thread_expressions",
            "unpriced_terms": ("set_rows_conversion_instructions_and_control",
                               "repeated_set_rows_index_requests", "compiler_and_native_execution_details")}


def _add_qwen35_source_tensor(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, rank: LogicalRank, source: _Qwen35AttentionSourceWork,
    stage: str, token_batch: int, heads: int, name: str, dependencies: Sequence[str],
    *, inputs: Sequence[Tuple[str, int]], output: Tuple[str, int],
    metadata: Mapping[str, object], same_address: bool = False,
) -> str:
    workload, work = source.tensor_kernel(stage, token_batch, heads=heads)
    common = {key: value for key, value in metadata.items() if key != "phase"}
    geometry = {"input_bytes": sum(size for _, size in inputs), "output_bytes": output[1],
                "input_objects": tuple({"tensor_id": tensor, "allocation_bytes": size} for tensor, size in inputs),
                "output_tensor_id": output[0], "allocation_alias": "same_address" if same_address else "distinct_or_unproven"}
    return _add_rank_tensor_kernel(builder, scenario, router, plan, rank, workload, name, dependencies,
        input_is_local=True, metadata={**common, "execution_phase": metadata.get("phase"),
            "event_kind": "qwen35_attention_source", "qwen35_attention_work": work,
            "qwen35_tensor_geometry": geometry, "output_tensor_id": output[0],
            "input_tensor_id": inputs[0][0] if len(inputs) == 1 else name + ".inputs",
            "modeled_memory_write_bytes": workload.write_bytes})


def _add_qwen35_source_qkv(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, rank: LogicalRank, layer: LayerSpec,
    source: _Qwen35AttentionSourceWork, token_batch: int, prefix: str,
    norm_apply: str, norm_component: str, qkv_target: str,
    metadata: Mapping[str, object], contract: Mapping[str, object],
) -> Tuple[str, str, str]:
    """Default cgraph Q/V/K sequence, retaining QG until the later gate copy."""
    owner = lambda role: prefix + ".rank{:03d}.qwen35.{}".format(rank.rank, role)
    name = prefix + ".rank{:03d}".format(rank.rank)
    prior, qg, query_end, key_end = norm_apply, "", "", ""
    for part, width in (("q", 4096), ("v", 512), ("k", 512)):
        workload = _layer_gemm(layer, token_batch, layer.hidden_size, width,
            name={"q": "query_tp", "k": "key_tp", "v": "value_tp"}[part],
            projection_id="attention." + part, projection_tp_degree=1, projection_tp_rank=0,
            projection_allow_padding=plan.allow_padding, f32_storage=True)
        workload = replace(workload, output_storage_bytes=4 * token_batch * workload.n)
        role = "qg" if part == "q" else part
        end = _add_rank_gemm(builder, scenario, router, plan, rank, workload, qkv_target,
            name + ".attention_" + part, tuple(dict.fromkeys((prior, norm_apply))),
            weight_tensor_id=layer.layer_id + ".attention_weights",
            activation_source_component_id=norm_component, keep_output_on_target=True,
            metadata={**metadata, "projection_id": "attention." + part,
                "physical_projection": {"q": "query", "k": "key", "v": "value"}[part],
                "output_tensor_id": owner(role), "modeled_memory_write_bytes": workload.output_bytes,
                "modeled_memory_write_kind": "split_projection_activation",
                "qkv_transient_output_bytes": workload.output_bytes,
                "modeled_kv_write_bytes": 0, "qwen35_source_order": part})
        if part == "q":
            qg = end
        heads = 8 if part == "q" else 2
        if part in {"q", "k"}:
            normalized = part + "_norm"
            end = _add_qwen35_source_tensor(builder, scenario, router, plan, rank, source,
                "rms_norm_mul", token_batch, heads, name + "." + normalized, (end,),
                inputs=((owner(role), 4 * token_batch * width),),
                output=(owner(normalized), 4 * token_batch * 256 * heads), metadata=metadata)
            role = normalized
            end = _add_qwen35_source_tensor(builder, scenario, router, plan, rank, source,
                "imrope", token_batch, heads, name + "." + part + "_imrope", (end,),
                inputs=((owner(role), 4 * token_batch * 256 * heads),),
                output=(owner(role), 4 * token_batch * 256 * heads), metadata=metadata, same_address=True)
        if contract["hadamard_enabled"]:
            rotated = part + "_hadamard"
            end = _add_qwen35_source_tensor(builder, scenario, router, plan, rank, source,
                "fwht64" if part == "v" else "fwht256", token_batch, heads,
                name + "." + rotated, (end,), inputs=((owner(role), 4 * token_batch * 256 * heads),),
                output=(owner(rotated), 4 * token_batch * 256 * heads), metadata=metadata)
            role = rotated
        if part == "q":
            query_end = end
        elif part == "k":
            key_end = end
        prior = end
    cache_contract = {**contract,
        "k_input_tensor_id": owner("k_hadamard" if contract["hadamard_enabled"] else "k_norm"),
        "v_input_tensor_id": owner("v_hadamard" if contract["hadamard_enabled"] else "v")}
    query_end, appended = _add_native_local_kv_writeback(
        builder, scenario, router, plan, rank, layer, token_batch, 2048, 512, 512, 128,
        (key_end,), name=name + ".native_kv", contract=cache_contract, metadata=metadata,
        already_rotated=(query_end, key_end))
    return qg, query_end, appended


def _add_qwen35_shared_graph_inputs(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, token_batch: int, name: str, dependencies: Sequence[str],
) -> str:
    """One text-position input and, for Q4 KV, two shared rotations per graph."""
    sources = tuple((layer, _qwen35_attention_source_work(layer)) for layer in _execution_layers(scenario)
                    if _QWEN35_ATTENTION_SOURCE_KEY in layer.metadata)
    if not sources:
        return _add_join(builder, name + ".unused", dependencies)
    if len(sources) != 6 or {source.block_index for _, source in sources} != {3, 7, 11, 15, 19, 23}:
        raise ValueError("Qwen3.5 shared graph inputs require all six ordinary source layers")
    rank = plan.rank_at(0, 0, 0)
    gpu_sources = tuple((layer, source) for layer, source in sources
                        if _qwen35_cuda_source_layer(scenario, plan, rank, layer, source))
    for layer, source in gpu_sources:
        native_kv = _native_local_kv_contract(scenario, router, plan, rank, layer,
                                            token_batch, token_batch, token_batch)
        _qwen35_source_execution_contract(scenario, router, plan, rank, layer, source,
            _parallel_target(scenario, layer, "attention", rank), native_kv)
    cpu_id = scenario.host_orchestration_profile.cpu_component_id
    cpu_profile, _host_profile = _cpu_profiles(scenario, cpu_id)
    host_memory = _compute_local_runtime_memory_component_id(scenario, cpu_id)
    prior = tuple(dependencies)
    cache_formats = set()
    for layer, _source in sources:
        bits, artifact = _kv_dtype_bits(scenario, layer)
        cache_formats.add(artifact.name.casefold() if artifact is not None else "f{}".format(bits))
    if len(cache_formats) != 1 or not cache_formats <= {"f16", "q4_0"}:
        raise ValueError("Qwen3.5 shared inputs require a uniform declared F16/Q4_0 cache")
    quantized = "q4_0" in cache_formats
    common = {"event_kind": "qwen35_shared_graph_input", "physical_ubatch_rows": token_batch,
              "qwen35_shared_input_work": {"scope": "once_per_graph_input_update_and_compute",
                  "shared_ordinary_layers": 6, "tokens": token_batch,
                  "gpu_ordinary_layers": len(gpu_sources),
                  "cpu_ordinary_layers_with_existing_kernel_approximation": 6 - len(gpu_sources),
                  "hadamard_enabled": quantized, "input_copy_deduplication": "source_target_copy_slot",
                  "timing_completeness": "partial", "unpriced_terms": (
                      "vector_allocation_initialization_and_compiler_lowering",
                      "physical_cpu_cache_writeback_and_faults", "input_copy_driver_submission_and_completion")}}
    # Hybrid input set_input runs before positions. The host matrices were
    # generated at cache construction; each graph input update copies them.
    host_steps = (("k_rotation_set", 262144, 262144), ("v_rotation_set", 16384, 16384)) if quantized else ()
    host_steps += (("position_expand", 12 * token_batch, 16 * token_batch),
                   ("position_tensor_set", 16 * token_batch, 16 * token_batch))
    for step, read_bytes, write_bytes in host_steps:
        estimate = estimate_cpu_logical_stream(cpu_profile, MemoryWorkload(read_bytes=read_bytes, write_bytes=write_bytes))
        for phase in estimate.phases:
            end = builder.add(name + "." + step + "." + phase.name, phase.category,
                tuple(_namespace_demand(scenario, demand, rank=rank, target_component_id=cpu_id) for demand in phase.demands),
                dependencies=prior, advance=False,
                metadata={**common, "op_name": name + "." + step, "phase": phase.name,
                    "target_component": cpu_id, "cost_model": dict(estimate.metadata),
                    "qwen35_shared_input_work": {**common["qwen35_shared_input_work"], "stage": step,
                        "logical_read_bytes": read_bytes, "logical_write_bytes": write_bytes,
                        "physical_memory_traffic_status": "unknown_not_charged"}})
            prior = (end,)
    device_inputs = (("positions", 16 * token_batch),) if gpu_sources else ()
    if quantized and gpu_sources:
        device_inputs += (("k_rotation", 262144), ("v_rotation", 16384))
    for role, byte_count in device_inputs:
        end = _add_transfer_tasks(builder, router, host_memory, rank.component_id, byte_count,
            prior, name=name + "." + role + ".h2d", routing_policy=plan.routing_policy,
            metadata={**common, "qwen35_shared_input_work": {**common["qwen35_shared_input_work"],
                "stage": role + "_h2d", "h2d_payload_bytes": byte_count,
                "source_input_object": name + "." + role, "copy_count": 1,
                "blocking_input_copy": True, "not_a_per_layer_or_fwht_weight_read": True}})
        prior = (end,)
    # These inputs are positions/matrices, not a hidden activation rank value.
    return _add_join(builder, name + ".complete", prior, metadata={**common,
        "qwen35_shared_input_work": {**common["qwen35_shared_input_work"], "stage": "complete",
            "h2d_copy_count": len(device_inputs), "h2d_payload_bytes": sum(size for _, size in device_inputs),
            "host_read_requests_bytes": sum(read for _, read, _ in host_steps),
            "host_write_requests_bytes": sum(write for _, _, write in host_steps)}})


def _add_kv_append(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    router: TopologyRouter,
    plan: ParallelPlan,
    rank: LogicalRank,
    layer: LayerSpec,
    token_batch: int,
    dependencies: Sequence[str],
    *,
    name: str,
    target_component_id: Optional[str] = None,
) -> str:
    cache, _offload, _ratio = _kv_components(
        scenario,
        rank,
        target_component_id,
        layer,
    )
    if token_batch <= 0:
        skipped = _add_join(
            builder,
            name + ".none",
            dependencies,
            metadata={
                "event_kind": "kv_append_skipped",
                "reason": "no_new_kv",
                "rank": rank.rank,
                "layer_id": layer.layer_id,
                "bytes": 0,
                "logical_bytes": 0,
                "physical_bytes": 0,
            },
        )
        return _discard_side_branch_rank_value(builder, skipped, rank)
    if not cache:
        raise ValueError(
            "full-attention rank {} layer {} has no writable KV cache "
            "placement".format(rank.rank, layer.layer_id)
        )
    byte_count = _kv_bytes_per_token(scenario, layer, plan.tp_degree) * token_batch
    logical_bytes = _logical_kv_bytes_per_token_for_rank(
        scenario, layer, plan.tp_degree, rank.tp_rank
    ) * token_batch
    access_component = _layer_local_runtime_memory_component_id(
        scenario,
        rank,
        target_component_id,
        rank.memory_component_id or rank.component_id,
    )
    if access_component is None:
        raise ValueError(
            "full-attention rank {} layer {} has no local KV access component"
            .format(rank.rank, layer.layer_id)
        )
    return _add_kv_access(
        builder,
        router,
        plan,
        rank,
        access_component,
        cache,
        byte_count,
        dependencies,
        name=name + ".append",
        metadata={
            "event_kind": "kv_append",
            "rank": rank.rank,
            "layer_id": layer.layer_id,
            "coverage_component": "full_attention",
            "kv_cache_component": cache,
            "logical_bytes": logical_bytes,
            "kv_token_appends": token_batch,
            "memory_direction": "write",
        },
    )


def _compile_parallel_request(
    scenario: ScenarioConfig, request: RequestSpec
) -> Tuple[TaskSpec, ...]:
    return tuple(
        task
        for chunk in _iter_parallel_request_task_chunks(scenario, request)
        for task in chunk.tasks
    )


def _iter_parallel_request_task_chunks(
    scenario: ScenarioConfig, request: RequestSpec
) -> Iterator[RequestTaskChunk]:
    plan = _parallel_plan(scenario)
    router = _topology_router(scenario)
    builder = _TaskBuilder(request)
    arrival = builder.add(
        "request_arrival",
        TaskCategory.POLICY,
        marker=TraceMarker.REQUEST_ARRIVAL,
        earliest_start_ns=request.arrival_ns,
        dependency="",
    )
    request_begin = _add_request_marker_boundary(
        builder, scenario, request, arrival, marker="request_begin"
    )
    prepared = _add_host_orchestration(
        builder,
        scenario,
        router,
        plan,
        (request_begin,),
        name="prefill.host_orchestration",
        request_count=1,
        token_count=request.prompt_tokens,
    )
    prepared = _add_physical_invocation_frontend(
        builder,
        scenario,
        plan,
        (prepared,),
        name="prefill.invocation_frontend",
        request_count=1,
        token_count=request.prompt_tokens,
        invocation_count=1,
        invocation_family="target_operator",
        orchestration_stage="host_prefix",
    )
    first_invocation_task = len(builder.tasks)
    selection = _final_output_selection(scenario, plan, request.prompt_tokens, (request.prompt_tokens - 1,))
    invocation_prior, output_indices, upload_ids = _prepare_output_selection_inputs(
        builder, scenario, router, plan, selection, "prefill", (prepared,))
    end = _compile_parallel_iteration(
        builder,
        scenario,
        plan,
        router,
        token_batch=request.prompt_tokens,
        context_tokens=request.prompt_tokens,
        kv_read_tokens=0,
        kv_append_tokens=request.prompt_tokens,
        kv_materialized_tokens=request.prompt_tokens,
        phase="prefill",
        output_selection=selection,
        output_indices_dependency=output_indices,
        dependencies=invocation_prior,
    )
    end = _compile_parallel_lm_head(
        builder,
        scenario,
        plan,
        router,
        "prefill",
        (end,),
        # Autoregressive prefill consumes only the final prompt position's
        # logits.  Backbone/KV work still covers the whole prompt above.
        token_batch=1,
        output_selection=selection,
        output_indices_dependency=output_indices,
    )
    _bind_output_index_upload(builder, first_invocation_task, upload_ids)
    next_token = 0
    if request.output_tokens > 0:
        end = _add_host_visible_logits_sampling_commit(
            builder,
            scenario,
            plan,
            router,
            (end,),
            phase="prefill",
            logit_rows=1,
            committed_rows=1,
        )
        end = _add_request_marker_boundary(
            builder, scenario, request, end, marker="first_token"
        )
        end = _emit_parallel_token(builder, scenario, plan, 0, (end,))
        next_token = 1

    yield RequestTaskChunk(builder.drain(), end)

    mtp = _mtp_configuration(scenario)
    mtp_cursor = MTPRequestCursor(mtp) if mtp is not None else None
    while next_token < request.output_tokens:
        if mtp is None:
            phase = "decode{:04d}".format(next_token)
            prepared = _add_host_orchestration(
                builder,
                scenario,
                router,
                plan,
                (end,),
                name=phase + ".host_orchestration",
                request_count=1,
                token_count=1,
            )
            prepared = _add_physical_invocation_frontend(
                builder,
                scenario,
                plan,
                (prepared,),
                name=phase + ".invocation_frontend",
                request_count=1,
                token_count=1,
                invocation_count=1,
                invocation_family="target_operator",
                orchestration_stage="host_prefix",
                first_decode_invocation=True,
            )
            first_invocation_task = len(builder.tasks)
            selection = _final_output_selection(scenario, plan, 1, (0,))
            invocation_prior, output_indices, upload_ids = _prepare_output_selection_inputs(
                builder, scenario, router, plan, selection, phase, (prepared,))
            end = _compile_parallel_iteration(
                builder,
                scenario,
                plan,
                router,
                token_batch=1,
                context_tokens=request.prompt_tokens + next_token,
                kv_read_tokens=max(
                    0, request.prompt_tokens + next_token - 1
                ),
                kv_append_tokens=1,
                kv_materialized_tokens=1,
                phase=phase,
                output_selection=selection,
                output_indices_dependency=output_indices,
                dependencies=invocation_prior,
            )
            end = _compile_parallel_lm_head(
                builder, scenario, plan, router, phase, (end,),
                output_selection=selection, output_indices_dependency=output_indices,
            )
            _bind_output_index_upload(builder, first_invocation_task, upload_ids)
            end = _add_host_visible_logits_sampling_commit(
                builder,
                scenario,
                plan,
                router,
                (end,),
                phase=phase,
                logit_rows=1,
                committed_rows=1,
            )
            if next_token == 0:
                end = _add_request_marker_boundary(
                    builder, scenario, request, end, marker="first_token"
                )
            end = _emit_parallel_token(
                builder, scenario, plan, next_token, (end,)
            )
            next_token += 1
            yield RequestTaskChunk(builder.drain(), end)
            continue

        if mtp_cursor is None:
            raise AssertionError("enabled MTP requires a request cursor")
        mtp_step = mtp_cursor.next_round(
            request.output_tokens - next_token
        )
        verifier_tokens = mtp_step.verifier_tokens
        accepted = mtp_step.committed_tokens
        prepared = _add_host_orchestration(
            builder,
            scenario,
            router,
            plan,
            (end,),
            name="mtp_verify{:04d}.host_orchestration".format(next_token),
            request_count=1,
            token_count=verifier_tokens,
        )
        proposer_dependencies: Tuple[str, ...] = (prepared,)
        if mtp_step.draft_tokens > 0:
            proposer_dependencies = (
                _add_physical_invocation_frontend(
                    builder,
                    scenario,
                    plan,
                    proposer_dependencies,
                    name="mtp_verify{:04d}.mtp_proposer_frontend".format(
                        next_token
                    ),
                    request_count=1,
                    token_count=verifier_tokens,
                    invocation_count=mtp_step.draft_tokens,
                    invocation_family="mtp_proposer",
                    orchestration_stage="host_prefix",
                    # MTP verification/proposal is part of the decode
                    # request phase; this helper is outside the cohort
                    # lowering function and has no local ``kind`` variable.
                    execution_phase="decode",
                ),
            )
        proposer = _compile_parallel_mtp_proposer(
            builder,
            scenario,
            plan,
            router,
            next_token=next_token,
            draft_step_lanes=(1,) * mtp_step.draft_tokens,
            verifier_tokens=verifier_tokens,
            accepted_tokens=accepted,
            policy=mtp,
            dependencies=proposer_dependencies,
        )
        phase = "mtp_verify{:04d}".format(next_token)
        target_frontend = _add_physical_invocation_frontend(
            builder,
            scenario,
            plan,
            (proposer,),
            name=phase + ".target_invocation_frontend",
            request_count=1,
            token_count=verifier_tokens,
            invocation_count=1,
            invocation_family="target_operator",
            orchestration_stage="host_target_frontend",
        )
        end = _compile_parallel_iteration(
            builder,
            scenario,
            plan,
            router,
            token_batch=verifier_tokens,
            context_tokens=request.prompt_tokens + next_token,
            kv_read_tokens=verifier_tokens
            * max(0, request.prompt_tokens + next_token - 1),
            kv_append_tokens=accepted,
            kv_materialized_tokens=verifier_tokens,
            linear_state_runtime=_mtp_linear_state_runtime(
                range(verifier_tokens),
                range(accepted),
                ((accepted - 1,) if accepted > 0 else ()),
                read_source="committed",
                release_temporary=True,
            ),
            phase=phase,
            dependencies=(target_frontend,),
        )
        end = _compile_parallel_lm_head(
            builder,
            scenario,
            plan,
            router,
            phase,
            (end,),
            token_batch=verifier_tokens,
        )
        end = _add_host_visible_logits_sampling_commit(
            builder,
            scenario,
            plan,
            router,
            (end,),
            phase=phase,
            logit_rows=verifier_tokens,
            committed_rows=accepted,
        )
        end = builder.add(
            "mtp{:04d}.commit".format(next_token),
            TaskCategory.POLICY,
            dependencies=(end,),
            advance=False,
            metadata={
                "event_kind": "mtp_commit",
                "proposed_tokens": verifier_tokens,
                "accepted_tokens": accepted,
                "rejected_tokens": mtp_step.rejected_draft_tokens,
                "main_tokens": mtp_step.main_tokens,
                "draft_tokens": mtp_step.draft_tokens,
                "verifier_tokens": verifier_tokens,
                "committed_tokens": accepted,
                "accepted_draft_tokens": mtp_step.accepted_draft_tokens,
                "rejected_draft_tokens": mtp_step.rejected_draft_tokens,
                "expected_accepted_draft_tokens": (
                    mtp_step.expected_accepted_draft_tokens
                ),
                "mtp_round": mtp_step.round_index,
            },
        )
        if next_token == 0 and accepted > 0:
            end = _add_request_marker_boundary(
                builder, scenario, request, end, marker="first_token"
            )
        for offset in range(accepted):
            token_index = next_token + offset
            end = _emit_parallel_token(
                builder, scenario, plan, token_index, (end,)
            )
        next_token += accepted
        yield RequestTaskChunk(builder.drain(), end)

    end = _add_request_marker_boundary(
        builder, scenario, request, end, marker="request_end"
    )
    done = builder.add(
        "request_done",
        TaskCategory.OUTPUT,
        marker=TraceMarker.REQUEST_DONE,
        dependencies=(end,),
    )
    yield RequestTaskChunk(builder.drain(), done, final=True)


def _compile_parallel_mtp_proposer(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    *,
    next_token: int,
    draft_step_lanes: Sequence[int],
    verifier_tokens: int,
    accepted_tokens: int,
    policy: object,
    dependencies: Sequence[str],
    draft_step_request_ids: Sequence[Sequence[str]] = (),
    draft_step_group_ids: Sequence[str] = (),
    draft_step_indices: Sequence[int] = (),
    draft_step_group_indices: Sequence[int] = (),
    draft_step_chunk_indices: Sequence[int] = (),
    descriptor_kinds: Optional[Sequence[str]] = None,
    invocation_family: str = "proposer",
) -> str:
    """Lower serial draft steps, each containing the typed descriptor chain.

    ``draft_step_lanes[step]`` is the number of request lanes active at that
    draft position.  Lanes share the step's GEMM invocation; later draft steps
    are strictly dependent and read the typed weights again in full.
    """

    if invocation_family not in {"proposer", "draft_context_catchup"}:
        raise ValueError("unsupported MTP invocation family")
    prefix = (
        "mtp{:04d}".format(next_token)
        if invocation_family == "proposer"
        else "mtp_context_catchup{:04d}".format(next_token)
    )
    step_lanes = tuple(max(0, int(value)) for value in draft_step_lanes)
    if any(value <= 0 for value in step_lanes):
        raise ValueError("MTP draft_step_lanes must contain positive values")
    if draft_step_request_ids and len(draft_step_request_ids) != len(step_lanes):
        raise ValueError(
            "MTP draft_step_request_ids must align with draft_step_lanes"
        )
    for values, label in (
        (draft_step_group_ids, "draft_step_group_ids"),
        (draft_step_indices, "draft_step_indices"),
        (draft_step_group_indices, "draft_step_group_indices"),
        (draft_step_chunk_indices, "draft_step_chunk_indices"),
    ):
        if values and len(values) != len(step_lanes):
            raise ValueError(
                "MTP {} must align with draft_step_lanes".format(label)
            )
    single_group_metadata: Dict[str, object] = {}
    if len(step_lanes) == 1 and draft_step_group_ids:
        single_group_metadata = {
            "mtp_proposer_invocation_group_id": str(draft_step_group_ids[0]),
            "mtp_proposer_invocation_group_index": (
                max(0, int(draft_step_group_indices[0]))
                if draft_step_group_indices
                else 0
            ),
            "physical_chunk_index": (
                max(0, int(draft_step_chunk_indices[0]))
                if draft_step_chunk_indices
                else 0
            ),
            "physical_ubatch_rows": step_lanes[0],
        }
    draft_tokens = sum(step_lanes)
    descriptors = _mtp_execution_descriptors(scenario)
    if descriptor_kinds is not None:
        allowed_kinds = {str(kind) for kind in descriptor_kinds}
        descriptors = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.operator.op_kind in allowed_kinds
        )
    if not descriptors:
        proposer_scale = float(getattr(policy, "proposal_cost_scale", 0.15))
        demands = (
            (
                ResourceDemand(
                    _rank_compute_resource(
                        scenario, plan.rank_at(0, plan.pp_degree - 1, 0)
                    ),
                    max(50.0, proposer_scale * 1000.0 * draft_tokens),
                    work_units=float(draft_tokens),
                ),
            )
            if draft_tokens
            else ()
        )
        return builder.add(
            prefix
            + (
                ".propose"
                if invocation_family == "proposer"
                else ".catchup"
            ),
            TaskCategory.POLICY,
            demands,
            dependencies=dependencies,
            advance=False,
            metadata={
                "event_kind": (
                    "mtp_propose"
                    if invocation_family == "proposer"
                    else "mtp_draft_context_catchup"
                ),
                "mtp_invocation_family": invocation_family,
                "mtp_coverage": "policy_cost_fallback",
                "model_operator_ids": (),
                "weight_tensor_ids": (),
                "proposed_tokens": verifier_tokens,
                "draft_tokens": draft_tokens,
                "draft_steps": len(step_lanes),
                "proposal_cost_scale_applied": proposer_scale,
                "accepted_tokens": accepted_tokens,
                **single_group_metadata,
            },
        )

    layer = _execution_layers(scenario)[-1]
    stage = plan.pp_degree - 1
    step_prior = tuple(dependencies)
    for step_offset, lane_count in enumerate(step_lanes):
        draft_step = (
            max(0, int(draft_step_indices[step_offset]))
            if draft_step_indices
            else step_offset
        )
        group_index = (
            max(0, int(draft_step_group_indices[step_offset]))
            if draft_step_group_indices
            else draft_step
        )
        chunk_index = (
            max(0, int(draft_step_chunk_indices[step_offset]))
            if draft_step_chunk_indices
            else 0
        )
        request_ids = tuple(
            dict.fromkeys(
                str(request_id)
                for request_id in (
                    draft_step_request_ids[step_offset]
                    if draft_step_request_ids
                    else (builder.request.request_id,)
                )
            )
        )
        invocation_group_id = (
            str(draft_step_group_ids[step_offset])
            if draft_step_group_ids
            else (
                (
                    "mtp-proposer-invocation-group-token{:04d}-draft{:04d}"
                    if invocation_family == "proposer"
                    else (
                        "mtp-draft-catchup-invocation-group-token{:04d}-"
                        "batch{:04d}"
                    )
                )
            ).format(next_token, draft_step)
        )
        replica_priors = {
            ep_rank: step_prior for ep_rank in range(plan.ep_degree)
        }
        for descriptor in descriptors:
            operator = descriptor.operator
            is_prediction = operator.op_kind == "mtp_prediction_layer"
            output_width = (
                descriptor.hidden_size
                if is_prediction
                else descriptor.vocabulary_size
            )
            collective_kind = "all_reduce" if is_prediction else "all_gather"
            for ep_rank in range(plan.ep_degree):
                ranks = plan.tp_group(stage, ep_rank)
                ends: List[str] = []
                prior = replica_priors[ep_rank]
                replica_suffix = (
                    ".ep{:03d}".format(ep_rank) if plan.ep_degree > 1 else ""
                )
                for rank in ranks:
                    output_shard = shard_extent(
                        output_width,
                        plan.tp_degree,
                        rank.tp_rank,
                        allow_padding=plan.allow_padding,
                    )
                    workload = _layer_gemm(
                        layer,
                        lane_count,
                        descriptor.hidden_size,
                        output_shard.local_size,
                        name=operator.op_kind + "_tp",
                        f32_storage=_f32_hidden_storage_enabled(scenario),
                    )
                    # Typed logical bytes remain authoritative when they
                    # include quantization metadata or padding not recoverable
                    # from KxN.  Every draft step gets a new workload and thus
                    # another complete read of the same typed weight shard.
                    weight_shard = shard_extent(
                        descriptor.weight_bytes,
                        plan.tp_degree,
                        rank.tp_rank,
                        allow_padding=plan.allow_padding,
                    )
                    # Typed logical bytes are the complete physical shard
                    # contract, including any quantization metadata or
                    # padding that cannot be recovered from KxN.  Replace the
                    # fallback format's storage and metadata separately so
                    # its metadata is not retained (or subtracted) from the
                    # authoritative typed byte count.
                    workload = replace(
                        workload,
                        weight_storage_bytes=weight_shard.local_size,
                        weight_metadata_bytes=0,
                    )
                    operator_metadata = {
                        "event_kind": operator.op_kind,
                        "mtp_component": (
                            "prediction_layer" if is_prediction else "aux_head"
                        ),
                        "coverage_component": (
                            "mtp_prediction" if is_prediction else "mtp_aux_head"
                        ),
                        "model_operator_id": operator.operator_id,
                        "operator_id": operator.operator_id,
                        "weight_tensor_id": descriptor.weight_tensor.tensor_id,
                        "tensor_id": descriptor.weight_tensor.tensor_id,
                        "input_tensor_id": descriptor.input_tensor.tensor_id,
                        "output_tensor_id": descriptor.output_tensor.tensor_id,
                        "graph_sequence_index": operator.sequence_index,
                        "prediction_index": descriptor.prediction_index,
                        "declared_weight_bytes": descriptor.weight_bytes,
                        "rank_weight_bytes": weight_shard.local_size,
                        "mtp_ep_replica": ep_rank,
                        "proposed_tokens": verifier_tokens,
                        "draft_tokens": draft_tokens,
                        "draft_step": draft_step,
                        "draft_step_lanes": lane_count,
                        "physical_ubatch_rows": lane_count,
                        "physical_chunk_index": chunk_index,
                        "mtp_proposer_invocation_group_id": invocation_group_id,
                        "mtp_proposer_invocation_group_index": group_index,
                        "mtp_proposer_invocation_group_request_ids": request_ids,
                        "mtp_proposer_invocation_group_batching_semantics": (
                            (
                                "mtp_draft_step_physical_ubatch"
                                if draft_step_group_ids
                                else "mtp_draft_step_batch"
                            )
                            if invocation_family == "proposer"
                            else (
                                "mtp_draft_context_catchup_physical_ubatch"
                                if draft_step_group_ids
                                else "mtp_draft_context_catchup_batch"
                            )
                        ),
                        "mtp_invocation_family": invocation_family,
                        "proposal_cost_scale_ignored": True,
                    }
                    if len(request_ids) == 1:
                        operator_metadata["serving_request_id"] = request_ids[0]
                    ends.append(
                        _add_rank_gemm(
                            builder,
                            scenario,
                            router,
                            plan,
                            rank,
                            workload,
                            _parallel_named_target(
                                scenario,
                                operator.operator_id,
                                rank,
                            ),
                            "{}.draft{:03d}.{}{}.rank{:03d}".format(
                                prefix,
                                draft_step,
                                operator.operator_id,
                                replica_suffix,
                                rank.rank,
                            ),
                            prior,
                            weight_tensor_id=descriptor.weight_tensor.tensor_id,
                            metadata=operator_metadata,
                        )
                    )
                replica_priors[ep_rank] = (
                    _add_collective_tasks(
                        builder,
                        scenario,
                        router,
                        plan,
                        "{}.draft{:03d}.{}{}.{}".format(
                            prefix,
                            draft_step,
                            operator.operator_id,
                            replica_suffix,
                            collective_kind,
                        ),
                        collective_kind,
                        ranks,
                        _activation_bytes(layer, lane_count * output_width, scenario=scenario),
                        ends,
                        tensor_elements=lane_count * output_width,
                        element_bits=_activation_storage_bits(layer, scenario),
                        metadata={
                            **operator_metadata,
                            "event_kind": operator.op_kind + "_collective",
                        },
                    ),
                )
        step_prior = tuple(
            replica_priors[ep_rank][0]
            for ep_rank in range(plan.ep_degree)
        )

    prior = step_prior

    return builder.add(
        prefix
        + (
            ".propose"
            if invocation_family == "proposer"
            else ".catchup"
        ),
        TaskCategory.POLICY,
        dependencies=prior,
        advance=False,
        metadata={
            "event_kind": (
                "mtp_propose"
                if invocation_family == "proposer"
                else "mtp_draft_context_catchup"
            ),
            "mtp_invocation_family": invocation_family,
            "mtp_coverage": "structural",
            "model_operator_ids": tuple(
                descriptor.operator.operator_id
                for descriptor in descriptors
            ),
            "weight_tensor_ids": tuple(
                descriptor.weight_tensor.tensor_id
                for descriptor in descriptors
            ),
            "proposed_tokens": verifier_tokens,
            "draft_tokens": draft_tokens,
            "draft_steps": len(step_lanes),
            "draft_step_lanes": step_lanes,
            "proposal_cost_scale_ignored": True,
            "accepted_tokens": accepted_tokens,
            **single_group_metadata,
        },
    )


def _compile_parallel_mtp_draft_catchup(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    *,
    token_batch: int,
    request_ids: Sequence[str],
    policy: object,
    dependencies: Sequence[str],
    invocation_group_id: Optional[str] = None,
    invocation_group_index: int = 0,
    physical_chunk_index: int = 0,
) -> str:
    """Advance the independent MTP context after one target physical batch.

    llama.cpp feeds every target prompt/verification batch through ``ctx_dft``
    to keep its appended MTP-layer KV current.  Those rows execute the typed
    prediction block, but their output flags are false, so the auxiliary LM
    head is deliberately excluded here.
    """

    rows = max(0, int(token_batch))
    if rows <= 0:
        raise ValueError("MTP draft-context catch-up requires positive rows")
    unique_request_ids = tuple(dict.fromkeys(str(item) for item in request_ids))
    return _compile_parallel_mtp_proposer(
        builder,
        scenario,
        plan,
        router,
        next_token=max(0, int(invocation_group_index)),
        draft_step_lanes=(rows,),
        verifier_tokens=rows,
        accepted_tokens=0,
        policy=policy,
        dependencies=dependencies,
        draft_step_request_ids=(unique_request_ids,),
        draft_step_group_ids=(
            (str(invocation_group_id),) if invocation_group_id is not None else ()
        ),
        draft_step_indices=(0,),
        draft_step_group_indices=(max(0, int(invocation_group_index)),),
        draft_step_chunk_indices=(max(0, int(physical_chunk_index)),),
        descriptor_kinds=("mtp_prediction_layer",),
        invocation_family="draft_context_catchup",
    )


def _f32_hidden_storage_enabled(scenario: Optional[ScenarioConfig]) -> bool:
    if scenario is None:
        return False

    def resolve() -> bool:
        value = scenario.workload.metadata.get("llama_cpp_f32_hidden_storage", False)
        if not isinstance(value, bool):
            raise ValueError("llama_cpp_f32_hidden_storage must be an explicit boolean")
        if not value:
            return False
        if (
            scenario.workload.mtp is not None
            or any(
                layer.kind != "dense" or layer.shared_expert_intermediate_size
                for layer in _execution_layers(scenario)
            )
            or any(set(_request_modalities(request)) - {"text"}
                   for request in scenario.workload.requests)
        ):
            raise ValueError("F32 hidden storage requires an ordinary non-expert, non-MTP backbone")
        return True

    context = _active_compilation_context(scenario)
    return resolve() if context is None else context.invariant(
        ("llama_cpp_f32_hidden_storage",), resolve
    )


def _activation_storage_bits(
    layer: LayerSpec, scenario: Optional[ScenarioConfig] = None
) -> int:
    return 32 if _f32_hidden_storage_enabled(scenario) else _layer_precision_bits(layer)[0]


def _activation_bytes(
    layer: LayerSpec, elements: int, *, scenario: Optional[ScenarioConfig] = None
) -> int:
    activation_bits = _activation_storage_bits(layer, scenario)
    return int(math.ceil(max(0, elements) * activation_bits / 8.0))


def _compile_parallel_embedding(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    phase: str,
    dependencies: Sequence[str],
    *,
    token_batch: int,
) -> str:
    """Lower one typed embedding lookup for a backbone invocation."""

    execution_view = _execution_view(scenario)
    embedding_operators = tuple(
        operator
        for operator in execution_view.operators
        if operator.op_kind == "embedding"
    )
    if len(embedding_operators) != 1:
        raise ValueError("typed execution requires exactly one embedding operator")
    embedding = embedding_operators[0]
    if (
        len(embedding.weight_tensor_ids) != 1
        or len(embedding.output_tensor_ids) != 1
    ):
        raise ValueError(
            "embedding must declare one weight tensor and one output tensor"
        )
    weight_tensor_id = embedding.weight_tensor_ids[0]
    output_tensor_id = embedding.output_tensor_ids[0]
    output_tensors = tuple(
        tensor
        for tensor in execution_view.tensors
        if tensor.tensor_id == output_tensor_id
    )
    if len(output_tensors) != 1:
        raise ValueError("embedding output tensor is missing from typed execution")
    output_tensor = output_tensors[0]
    if not output_tensor.shape or not isinstance(output_tensor.shape[-1], int):
        raise ValueError("embedding output must declare a concrete hidden width")
    hidden_size = int(output_tensor.shape[-1])
    if hidden_size <= 0:
        raise ValueError("embedding hidden width must be positive")

    stage = 0
    declared_weight_bytes = _logical_weight_tensor_bytes(
        scenario, weight_tensor_id
    )
    if declared_weight_bytes <= 0:
        declared_weight_bytes = max(
            0, int(execution_view.embedding_weight_bytes)
        )
    if declared_weight_bytes <= 0:
        return _add_join(
            builder,
            phase + ".embedding.no_physical_weight",
            dependencies,
            metadata={
                "event_kind": "embedding_skipped",
                "phase": phase,
                "model_operator_id": embedding.operator_id,
                "reason": "no_physical_weight_tensor",
            },
        )
    declared_f32_storage = _f32_hidden_storage_enabled(scenario)
    if declared_f32_storage and plan.tp_degree != 1:
        raise ValueError("F32 embedding storage requires an unsharded physical weight row")
    output_bits = 32 if declared_f32_storage else _dtype_bits(output_tensor.dtype)
    rank_ends: List[str] = []
    for rank in plan.tp_group(stage, 0):
        target_component_id = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.MEMORY,
            embedding.operator_id,
        )
        hidden_shard = shard_extent(
            hidden_size,
            plan.tp_degree,
            rank.tp_rank,
            allow_padding=plan.allow_padding,
        )
        weight_shard = shard_extent(
            declared_weight_bytes,
            plan.tp_degree,
            rank.tp_rank,
            allow_padding=plan.allow_padding,
        )
        rank_weight_bytes = weight_shard.local_size
        weight_source, source_tensor, logical_tensor = _weight_source_for_tensor(
            scenario,
            weight_tensor_id,
            target_component_id,
            rank=rank,
        )
        weight_read_decision = _weight_backing_read_gate(
            scenario,
            weight_source,
            target_component_id,
            model_weight_read=True,
        )
        name = "{}.rank{:03d}.embedding".format(phase, rank.rank)
        invocation_id = "{}:{}:rank-{}:{}".format(
            builder.request.request_id,
            name,
            rank.rank,
            logical_tensor,
        )
        lookup_bytes = max(
            1,
            int(
                math.ceil(
                    max(1, token_batch)
                    * hidden_shard.local_size
                    * output_bits
                    / 8.0
                )
            ),
        )
        lookup_read_bytes = lookup_bytes
        if declared_f32_storage:
            vocabulary_size = scenario.model.vocabulary_size
            if vocabulary_size <= 0 or rank_weight_bytes % vocabulary_size:
                raise ValueError("F32 embedding storage requires exact physical weight row bytes")
            lookup_read_bytes = max(1, token_batch) * (rank_weight_bytes // vocabulary_size)
        detail_raw = _control_plane_decision_metadata(scenario).get(
            "weight_tensor_details", {}
        )
        embedding_detail_raw = (
            detail_raw.get(weight_tensor_id, {})
            if isinstance(detail_raw, Mapping)
            else {}
        )
        resident_sparse_lookup = (
            isinstance(embedding_detail_raw, Mapping)
            and embedding_detail_raw.get("runtime_copy_role")
            == "input_embedding"
        )
        operation_metadata = {
            "event_kind": "embedding",
            "stage": stage,
            "phase": phase,
            "rank": rank.rank,
            "model_operator_id": embedding.operator_id,
            "model_operator_kind": embedding.op_kind,
            "weight_tensor_id": weight_tensor_id,
            "tensor_id": weight_tensor_id,
            "output_tensor_id": output_tensor_id,
            "declared_weight_bytes": declared_weight_bytes,
            "rank_weight_bytes": rank_weight_bytes,
            "rank_weight_capacity_bytes": rank_weight_bytes,
            "weight_read_bytes": (
                lookup_read_bytes if declared_f32_storage or resident_sparse_lookup else rank_weight_bytes
            ),
            "lookup_workload_bytes": lookup_bytes,
            "physical_weight_row_bytes": int(
                math.ceil(
                    rank_weight_bytes
                    / float(max(1, scenario.model.vocabulary_size))
                )
            ),
            "physical_weight_row_bytes_semantics": (
                "capacity_derived_not_a_dram_transaction_claim"
            ),
            **weight_read_decision.audit_metadata(invocation_id),
        }
        if declared_f32_storage:
            operation_metadata.update(
                runtime_output_storage_bits=32,
                lookup_read_bytes=lookup_read_bytes,
                lookup_write_bytes=lookup_bytes,
                embedding_dequant_compute="unmodeled_not_added_by_storage_contract",
            )
        prior = tuple(dependencies)
        rank_local_weight_source = (
            _kind(_component(scenario, target_component_id)) == "gpu"
            and target_component_id == rank.component_id
            and weight_source in {
                rank.component_id,
                rank.memory_component_id,
            }
        )
        cpu_local_weight_source = (
            _kind(_component(scenario, target_component_id)) == "cpu"
            and weight_source is not None
            and _weight_source_is_compute_local_backing(
                scenario,
                weight_source,
                target_component_id,
            )
        )
        compute_local_weight_source = (
            rank_local_weight_source or cpu_local_weight_source
        )
        if weight_source:
            access_metadata = {
                **operation_metadata,
                **_weight_transfer_metadata(
                    source_tensor,
                    logical_tensor,
                    weight_source,
                    target_component_id,
                ),
                "event_kind": "model_weight_access",
                "resource_accounting": (
                    "included_in_embedding_memory_roofline"
                    if compute_local_weight_source
                    else "weight_access_marker"
                ),
                "bytes": operation_metadata["weight_read_bytes"],
            }
            if declared_f32_storage and not compute_local_weight_source:
                access_metadata.update(
                    bytes=rank_weight_bytes,
                    weight_read_bytes=rank_weight_bytes,
                    weight_access_semantics="full_weight_staging_before_row_lookup",
                )
            prior = (
                builder.add(
                    name + ".model_weight_read.access",
                    TaskCategory.COMMUNICATION,
                    dependencies=prior,
                    advance=False,
                    metadata=_communication_task_metadata(access_metadata),
                ),
            )
        if (
            weight_source
            and _kind(_component(scenario, target_component_id))
            in {"gpu", "cpu"}
            and weight_read_decision.emit_source_transfer
            and not compute_local_weight_source
        ):
            prior = (
                _add_transfer_tasks(
                    builder,
                    router,
                    weight_source,
                    target_component_id,
                    rank_weight_bytes,
                    prior,
                    name=name + ".model_weight_read",
                    routing_policy=plan.routing_policy,
                    metadata={
                        **operation_metadata,
                        **({
                            "weight_read_bytes": rank_weight_bytes,
                            "weight_access_semantics": "full_weight_staging_before_row_lookup",
                        } if declared_f32_storage else {}),
                        **_weight_transfer_metadata(
                            source_tensor,
                            logical_tensor,
                            weight_source,
                            target_component_id,
                        ),
                    },
                ),
            )
        end, _target_component_id = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.MEMORY,
            MemoryWorkload(
                # GET_ROWS on CPU streams the resident quantized embedding
                # backing through DRAM.  Preserve the row lookup write while
                # charging the physical tensor read once; GPU-local weights
                # keep the existing row-lookup behavior.
                read_bytes=(
                    max(lookup_read_bytes, rank_weight_bytes)
                    if cpu_local_weight_source
                    else lookup_read_bytes
                ),
                write_bytes=lookup_bytes,
                working_set_bytes=(
                    max(lookup_read_bytes, rank_weight_bytes) + lookup_bytes
                    if cpu_local_weight_source
                    else lookup_read_bytes + lookup_bytes
                ),
                reuse_factor=1.0,
                streaming_fraction=1.0,
                name="embedding_lookup",
            ),
            embedding.operator_id,
            name,
            prior,
            source_component_id=target_component_id,
            metadata=operation_metadata,
        )
        rank_ends.append(end)
    return _add_join(
        builder,
        phase + ".embedding.complete",
        rank_ends,
        metadata={
            "event_kind": "embedding_complete",
            "phase": phase,
            "model_operator_id": embedding.operator_id,
        },
    )


def _final_output_selection(
    scenario: ScenarioConfig, plan: ParallelPlan, token_rows: int,
    selected_indices: Tuple[int, ...],
) -> Optional[_FinalOutputSelection]:
    declaration = scenario.model.metadata.get(_FINAL_OUTPUT_SELECTION_KEY)
    if declaration is None:
        return None
    policy = _resolve_final_output_declaration(
        declaration,
        scenario.model.architecture, mtp_present=scenario.workload.mtp is not None,
    )
    if policy is None:
        return None
    if plan.world_size != 1 or not _f32_hidden_storage_enabled(scenario):
        raise ValueError("final output selection requires a single rank and F32 hidden storage")
    if _execution_layers(scenario)[-1].is_moe:
        raise ValueError("final output selection has no MoE tail source contract")
    if policy.position == "before_last_ffn" and _execution_layers(scenario)[-1].is_linear_attention:
        raise ValueError("early final output selection requires a full-attention last layer")
    return policy.select(token_rows=token_rows, selected_indices=selected_indices)


def _selection_audit(selection: _FinalOutputSelection, stage: str) -> Mapping[str, object]:
    return {**selection.audit_metadata(), "stage": stage,
            "timing_completeness": "partial",
            "placement_semantics": "inherited_single_rank_operator_mapping_not_native_assignment"}


def _tag_output_selection_tasks(
    builder: _TaskBuilder, first: int, selection: _FinalOutputSelection, stage: str,
) -> None:
    for index in range(first, len(builder.tasks)):
        task = builder.tasks[index]
        if "final_layer_output_selection" not in task.metadata:
            builder.tasks[index] = replace(task, metadata={**task.metadata,
                "final_layer_output_selection": _selection_audit(selection, stage)})


def _prepare_output_selection_inputs(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, selection: Optional[_FinalOutputSelection],
    phase: str, dependencies: Sequence[str],
) -> Tuple[Tuple[str, ...], Optional[str], Tuple[str, ...]]:
    """One host input update and one destination copy, outside layer caches."""
    if selection is None:
        return tuple(dependencies), None, ()
    rank = plan.rank_at(0, 0, 0)
    layer = _execution_layers(scenario)[-1]
    prefix = phase + "." + layer.layer_id + ".output_ids"
    cpu_id = scenario.host_orchestration_profile.cpu_component_id
    cpu, _memory = _cpu_profiles(scenario, cpu_id)
    reads = selection.token_rows if selection.logit_rows < selection.token_rows else 0
    writes = 4 * selection.logit_rows
    estimate = estimate_cpu_logical_stream(cpu, MemoryWorkload(
        read_bytes=reads, write_bytes=writes, name="output_row_indices"))
    prior = tuple(dependencies)
    for cost_phase in estimate.phases:
        host_end = builder.add(prefix + "." + cost_phase.name, cost_phase.category,
            tuple(_namespace_demand(scenario, demand, rank=rank, target_component_id=cpu_id)
                  for demand in cost_phase.demands), dependencies=prior, advance=False,
            metadata={"event_kind": "output_row_indices", "phase": cost_phase.name,
                "target_component": cpu_id, "cost_model": dict(estimate.metadata),
                "final_layer_output_selection": {**_selection_audit(selection, "output_indices"),
                    "index_tensor_id": prefix, "logical_read_bytes": reads,
                    "logical_write_bytes": writes, "physical_memory_traffic_status": "unknown_not_charged",
                    "unpriced_terms": ["compiled_I8_condition_address_and_loop_work", "input_sync_and_driver_control"]}})
        prior = (host_end,)
    target = (
        _parallel_target(scenario, layer, "attention", rank)
        if selection.position == "before_last_ffn"
        else _primitive_target(scenario, router, rank, OperatorClass.ELEMENTWISE,
                               "final_norm.apply", fallback_keys=("final_norm",))
    )
    if _kind(_component(scenario, target)) not in {"cpu", "gpu"}:
        raise ValueError("final output selection requires a declared CPU or GPU target")
    if not writes or _kind(_component(scenario, target)) == "cpu":
        return prior, host_end, ()
    first_transfer = len(builder.tasks)
    host_memory = _compute_local_runtime_memory_component_id(scenario, cpu_id)
    upload = _add_transfer_tasks(builder, router, host_memory, target, writes, prior,
        name=prefix + ".upload", routing_policy=plan.routing_policy,
        metadata={"event_kind": "output_row_index_transfer_internal", "rank": rank.rank,
            "final_layer_output_selection": {**_selection_audit(selection, "index_upload"),
                "index_tensor_id": prefix, "index_bytes": writes, "destination_component": target,
                "copy_scope": "one_source_tensor_destination_and_invocation",
                "unpriced_terms": ["input_copy_driver_submission_and_stream_sync"]}})
    ids = tuple(task.task_id for task in builder.tasks[first_transfer:])
    for index in range(first_transfer, len(builder.tasks)):
        task = builder.tasks[index]
        builder.discard_rank_value(task.task_id, rank.rank)
        if task.task_id == upload:
            builder.tasks[index] = replace(task, metadata={**task.metadata,
                "event_kind": "output_row_index_transfer"})
    return prior, upload, ids


def _bind_output_index_upload(
    builder: _TaskBuilder, first: int, upload_ids: Sequence[str],
) -> None:
    """Place the shared input at its first consuming declared device segment.

    This follows the existing planner's operator placement; it does not assert
    that the native backend scheduler made those same assignments.
    """
    if not upload_ids:
        return
    by_id = {task.task_id: index for index, task in enumerate(builder.tasks)}
    upload_first, upload_last = by_id[upload_ids[0]], upload_ids[-1]
    target = builder.tasks[upload_first].metadata["final_layer_output_selection"]["destination_component"]
    numerical = []
    for index in range(first, len(builder.tasks)):
        task = builder.tasks[index]
        if task.task_id in upload_ids or task.metadata.get("event_kind") in {
            "output_row_indices", "qwen35_shared_graph_input",
        }:
            continue
        component = task.metadata.get("target_component")
        cost = task.metadata.get("cost_model")
        if component and isinstance(cost, Mapping) and task.category != TaskCategory.COMMUNICATION:
            numerical.append((index, component))
    gather = next(index for index, component in numerical
                  if component == target and builder.tasks[index].metadata.get("event_kind") == "output_row_selection")
    segment = []
    for index, component in numerical:
        if index > gather:
            break
        if component == target:
            segment.append(index)
        else:
            segment.clear()
    if not segment:
        raise ValueError("output-index destination segment could not be located")
    entry_index = segment[0]
    entry, upload = builder.tasks[entry_index], builder.tasks[upload_first]
    if any(dependency in upload_ids for dependency in entry.dependencies):
        # No earlier node in this segment: the gather already waits for upload.
        return
    builder.tasks[upload_first] = replace(upload, dependencies=tuple(dict.fromkeys(
        (*upload.dependencies, *entry.dependencies))))
    builder.tasks[entry_index] = replace(entry, dependencies=tuple(dict.fromkeys(
        (*entry.dependencies, upload_last))))


def _add_output_row_selection(
    builder: _TaskBuilder, scenario: ScenarioConfig, router: TopologyRouter,
    plan: ParallelPlan, rank: LogicalRank, layer: LayerSpec,
    selection: _FinalOutputSelection, name: str, dependencies: Sequence[str],
    *, stage: str, source_component: str, target_component: str,
    indices_dependency: str, input_tensor_id: str,
) -> Tuple[str, str]:
    """Copy nonempty F32 rows; known logical work is separate from unknown fees."""
    if selection.logit_rows == 0:
        raise ValueError("empty source GET_ROWS must be skipped before workload construction")
    width, rows = layer.hidden_size, selection.logit_rows
    input_capacity, output_capacity = 4 * width * selection.token_rows, 4 * width * rows
    prior = tuple(dependencies)
    if source_component != target_component:
        moved = _add_transfer_tasks(builder, router, source_component, target_component,
            input_capacity, prior, name=name + ".full_input_transfer", routing_policy=plan.routing_policy,
            metadata={"event_kind": "output_selection_input_transfer", "rank": rank.rank,
                "input_tensor_id": input_tensor_id,
                "final_layer_output_selection": {**_selection_audit(selection, stage),
                    "full_input_layout_bytes": input_capacity}})
        prior = (moved,)
    prior = tuple(dict.fromkeys((*prior, indices_dependency)))
    audit = {**_selection_audit(selection, stage), "logical_data_read_bytes": output_capacity,
        "logical_data_write_bytes": output_capacity, "index_unique_bytes": 4 * rows,
        "unpriced_terms": ["compiled_copy_and_address_instructions", "physical_cache_transactions",
                           "repeated_gpu_index_loads_and_broadcast", "native_synchronization"]}
    metadata = {"event_kind": "output_row_selection", "layer_id": layer.layer_id,
        "input_tensor_id": input_tensor_id, "output_tensor_id": name + ".selected",
        "final_layer_output_selection": audit,
        "output_selection_tensor_geometry": {"input_bytes": input_capacity,
            "output_bytes": output_capacity, "input_rows": selection.token_rows,
            "output_rows": rows, "width": width, "element_bytes": 4,
            "allocation_alias": "distinct_get_rows_output"}}
    if _kind(_component(scenario, target_component)) == "gpu":
        return _add_rank_primitive(builder, scenario, router, plan, rank,
            OperatorClass.MEMORY, MemoryWorkload(read_bytes=output_capacity + 4 * rows,
                write_bytes=output_capacity, working_set_bytes=input_capacity + output_capacity + 4 * rows,
                name="output_row_selection"), name, name, prior,
            source_component_id=target_component, target_component_id=target_component,
            input_component_bytes=((target_component, output_capacity + 4 * rows),), metadata=metadata)
    if _kind(_component(scenario, target_component)) != "cpu":
        raise ValueError("GET_ROWS source work only covers CPU and GPU")
    if rows == 1:
        cpu, _memory = _cpu_profiles(scenario, target_component)
        estimate = estimate_cpu_logical_stream(cpu, MemoryWorkload(
            read_bytes=output_capacity + 4, write_bytes=output_capacity, name="output_row_selection"))
        cost_phase, = estimate.phases
        demands = tuple(_namespace_demand(scenario, d, rank=rank, target_component_id=target_component)
                        for d in cost_phase.demands)
        cost = dict(estimate.metadata)
    else:
        demands = ()
        cost = {"model": "cpu_get_rows_graph_threads_unpriced", "timing_completeness": "partial",
                "logical_read_bytes": output_capacity + 4 * rows, "logical_write_bytes": output_capacity,
                "unpriced_reason": "actual_CPU_split_thread_count_not_declared"}
        metadata["final_layer_output_selection"] = {**audit,
            "cpu_copy_fee_status": "unpriced_nonzero_work", "cpu_graph_threads": None}
    last = builder.add(name, TaskCategory.MEMORY, demands, dependencies=prior, advance=False,
        metadata={**metadata, "rank": rank.rank, "phase": "cpu_output_row_selection",
            "op_name": name, "target_component": target_component, "cost_model": cost})
    builder.record_rank_value(last, rank.rank, target_component)
    return last, target_component


def _compile_parallel_iteration(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    *,
    token_batch: int,
    context_tokens: int,
    kv_read_tokens: Optional[int] = None,
    kv_append_tokens: Optional[int] = None,
    kv_materialized_tokens: Optional[int] = None,
    q4_mma_view_tokens_lower_bound: int = 0,
    linear_state_runtime: Optional[_LinearStateRuntimeSemantics] = None,
    output_selection: Optional[_FinalOutputSelection] = None,
    output_indices_dependency: Optional[str] = None,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    previous = _add_join(
        builder,
        phase + ".start",
        dependencies,
        metadata={"event_kind": "iteration_start", "phase": phase},
    )
    if any(_QWEN35_ATTENTION_SOURCE_KEY in layer.metadata for layer in _execution_layers(scenario)):
        previous = _add_qwen35_shared_graph_inputs(
            builder, scenario, router, plan, token_batch, phase + ".qwen35_shared_inputs", (previous,))
    previous = _compile_parallel_embedding(
        builder,
        scenario,
        plan,
        router,
        phase,
        (previous,),
        token_batch=token_batch,
    )
    for stage in range(plan.pp_degree):
        layers = tuple(
            layer
            for layer in _execution_layers(scenario)
            if plan.stage_for_layer(layer) == stage
        )
        stage_end = previous
        for layer in layers:
            stage_end = _compile_parallel_layer(
                builder,
                scenario,
                plan,
                router,
                layer,
                token_batch=token_batch,
                context_tokens=context_tokens,
                kv_read_tokens=(
                    context_tokens if kv_read_tokens is None else kv_read_tokens
                ),
                kv_append_tokens=(
                    token_batch if kv_append_tokens is None else kv_append_tokens
                ),
                kv_materialized_tokens=(
                    token_batch
                    if kv_materialized_tokens is None
                    else kv_materialized_tokens
                ),
                q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
                linear_state_runtime=linear_state_runtime,
                output_selection=(output_selection if layer.layer_id == _execution_layers(scenario)[-1].layer_id else None),
                output_indices_dependency=output_indices_dependency,
                phase=phase,
                dependencies=(stage_end,),
            )
        if stage == plan.pp_degree - 1:
            previous = stage_end
            continue
        boundary_ends: List[str] = []
        last_layer = layers[-1]
        bytes_per_rank = _activation_bytes(
            last_layer,
            token_batch
            * shard_extent(
                last_layer.hidden_size,
                plan.tp_degree,
                0,
                allow_padding=plan.allow_padding,
            ).local_size,
            scenario=scenario,
        )
        for tp_rank in range(plan.tp_degree):
            source = plan.rank_at(tp_rank, stage, 0)
            target = plan.rank_at(tp_rank, stage + 1, 0)
            source_component = (
                builder.rank_value_component((stage_end,), source.rank)
                or source.component_id
            )
            boundary = _add_transfer_tasks(
                builder,
                router,
                source_component,
                target.component_id,
                bytes_per_rank,
                (stage_end,),
                name="{}.pp{:02d}_to_{:02d}.tp{:02d}".format(
                    phase, stage, stage + 1, tp_rank
                ),
                routing_policy=plan.routing_policy,
                metadata={
                    "event_kind": "pipeline_activation",
                    "phase": phase,
                    "source_stage": stage,
                    "target_stage": stage + 1,
                    "tp_rank": tp_rank,
                },
            )
            builder.record_rank_value(
                boundary, target.rank, target.component_id
            )
            boundary_ends.append(boundary)
        previous = _add_join(
            builder,
            "{}.pp{:02d}_to_{:02d}.complete".format(phase, stage, stage + 1),
            boundary_ends,
            metadata={"event_kind": "pipeline_barrier", "phase": phase},
        )
    return previous


def _compile_parallel_layer(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    *,
    token_batch: int,
    context_tokens: int,
    kv_read_tokens: int,
    kv_append_tokens: int,
    kv_materialized_tokens: int,
    q4_mma_view_tokens_lower_bound: int = 0,
    linear_state_runtime: Optional[_LinearStateRuntimeSemantics],
    output_selection: Optional[_FinalOutputSelection] = None,
    output_indices_dependency: Optional[str] = None,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    """Compile one layer, replaying exact full-attention payload templates.

    Full attention has context-shaped demands, KV traffic, and an SRAM-sensitive
    flash-fusion decision.  Those values therefore stay in the exact cache key;
    only compiler namespaces and the outer phase label are rebound.  Linear
    state has its own stricter runtime-structure cache below and deliberately
    bypasses this layer-level cache.
    """

    if layer.is_linear_attention:
        return _compile_parallel_layer_body(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            kv_read_tokens=kv_read_tokens,
            kv_append_tokens=kv_append_tokens,
            kv_materialized_tokens=kv_materialized_tokens,
            q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
            linear_state_runtime=linear_state_runtime,
            output_selection=output_selection,
            output_indices_dependency=output_indices_dependency,
            phase=phase,
            dependencies=dependencies,
        )
    context = _active_compilation_context(scenario)
    if context is None:
        return _compile_parallel_layer_body(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            kv_read_tokens=kv_read_tokens,
            kv_append_tokens=kv_append_tokens,
            kv_materialized_tokens=kv_materialized_tokens,
            q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
            linear_state_runtime=linear_state_runtime,
            output_selection=output_selection,
            output_indices_dependency=output_indices_dependency,
            phase=phase,
            dependencies=dependencies,
        )
    if not context.eager_full_attention_segments:
        # The provider's outer exact cohort cache owns reuse across cohorts.
        # Full-attention payloads are context-exact, so a newly seen outer key
        # cannot hit this same leaf key before the completed BatchCost is
        # retained.  Avoid compiling a large clone plan that would be dead on
        # arrival; direct CompilationContext callers keep eager leaf reuse.
        return _compile_parallel_layer_body(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            kv_read_tokens=kv_read_tokens,
            kv_append_tokens=kv_append_tokens,
            kv_materialized_tokens=kv_materialized_tokens,
            q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
            linear_state_runtime=linear_state_runtime,
            output_selection=output_selection,
            output_indices_dependency=output_indices_dependency,
            phase=phase,
            dependencies=dependencies,
        )
    cache = _task_segment_cache(
        context,
        ("full_attention_layer_task_segment_cache_v1",),
    )
    if cache is None:
        return _compile_parallel_layer_body(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            kv_read_tokens=kv_read_tokens,
            kv_append_tokens=kv_append_tokens,
            kv_materialized_tokens=kv_materialized_tokens,
            q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
            linear_state_runtime=linear_state_runtime,
            output_selection=output_selection,
            output_indices_dependency=output_indices_dependency,
            phase=phase,
            dependencies=dependencies,
        )

    ranks = plan.tp_group(plan.stage_for_layer(layer), 0)
    input_components = tuple(
        builder.rank_value_component(dependencies, rank.rank)
        for rank in ranks
    )
    cache_key = (
        str(layer.layer_id),
        int(token_batch),
        int(context_tokens),
        int(kv_read_tokens),
        int(kv_append_tokens),
        int(kv_materialized_tokens),
        int(q4_mma_view_tokens_lower_bound),
        input_components,
        len(tuple(dependencies)),
        builder._last_coherent_dma_task is not None,
    )
    if output_selection is not None:
        cache_key += (output_selection,)
    capture_dependencies = tuple(dependencies) + ((output_indices_dependency,) if output_selection is not None and output_indices_dependency is not None else ())
    prefix = "{}.{}".format(phase, layer.layer_id)
    template = _task_segment_cache_get(cache, cache_key)
    if template is not None:
        replayed = template.replay(
            builder,
            prefix=prefix,
            dependencies=capture_dependencies,
            metadata_overrides={"phase": phase},
        )
        if replayed is not None:
            return replayed

    first_task_index = len(builder.tasks)
    initial_counter = builder.counter
    initial_previous = builder.previous
    initial_dma = builder._last_coherent_dma_task
    terminal = _compile_parallel_layer_body(
        builder,
        scenario,
        plan,
        router,
        layer,
        token_batch=token_batch,
        context_tokens=context_tokens,
        kv_read_tokens=kv_read_tokens,
        kv_append_tokens=kv_append_tokens,
        kv_materialized_tokens=kv_materialized_tokens,
        q4_mma_view_tokens_lower_bound=q4_mma_view_tokens_lower_bound,
        linear_state_runtime=linear_state_runtime,
        output_selection=output_selection,
        output_indices_dependency=output_indices_dependency,
        phase=phase,
        dependencies=dependencies,
    )
    captured = _TaskSegmentTemplate.capture(
        builder,
        first_task_index=first_task_index,
        source_prefix=prefix,
        source_counter_before=initial_counter,
        source_dependencies=capture_dependencies,
        source_initial_previous=initial_previous,
        source_initial_dma=initial_dma,
        terminal_task_id=terminal,
        source_phase=phase,
    )
    if captured is not None:
        _task_segment_cache_put(context, cache, cache_key, captured)
    return terminal


def _compile_parallel_layer_body(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    *,
    token_batch: int,
    context_tokens: int,
    kv_read_tokens: int,
    kv_append_tokens: int,
    kv_materialized_tokens: int,
    q4_mma_view_tokens_lower_bound: int = 0,
    linear_state_runtime: Optional[_LinearStateRuntimeSemantics],
    output_selection: Optional[_FinalOutputSelection] = None,
    output_indices_dependency: Optional[str] = None,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    if layer.is_linear_attention:
        mixer_end = _compile_parallel_linear_mixer(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            linear_state_runtime=linear_state_runtime,
            phase=phase,
            dependencies=dependencies,
        )
        prefix = "{}.{}".format(phase, layer.layer_id)
        if layer.is_moe:
            return _compile_parallel_moe(
                builder,
                scenario,
                plan,
                router,
                layer,
                token_batch,
                prefix,
                (mixer_end,),
            )
        return _compile_parallel_dense_mlp(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch,
            prefix,
            (mixer_end,),
        )

    tail_rows = output_selection.ffn_rows if output_selection is not None else token_batch
    stage = plan.stage_for_layer(layer)
    tp_ranks = plan.tp_group(stage, 0)
    hidden_shard = shard_extent(
        layer.hidden_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    attention_execution = _attention_execution_descriptor(layer)
    source_geometry = _qwen35_attention_source_work(layer)
    head_dim = (
        attention_execution.head_dim
        if attention_execution is not None
        else layer.effective_attention_head_dim
    )
    query_heads = (
        attention_execution.query_heads
        if attention_execution is not None
        else layer.attention_heads
    )
    kv_heads = (
        attention_execution.kv_heads
        if attention_execution is not None
        else layer.effective_kv_heads
    )
    query_width = (
        attention_execution.query_width
        if attention_execution is not None
        else layer.hidden_size
    )
    gate_width = (
        attention_execution.gate_width
        if attention_execution is not None
        else 0
    )
    q_projection_width = (
        attention_execution.q_projection_width
        if attention_execution is not None
        else query_width
    )
    kv_projection_width = kv_heads * head_dim
    kv_input_bits, kv_physical_contract = (
        _kv_fused_attention_physical_contract(
            scenario,
            layer,
            plan.tp_degree,
        )
    )
    kv_hidden_size = _physical_kv_width_for_rank(layer, plan.tp_degree)
    kv_materialized_bytes = (
        _kv_bytes_per_token(scenario, layer, plan.tp_degree)
        * max(0, kv_materialized_tokens)
    )
    query_shard = shard_extent(
        query_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    gate_shard = (
        shard_extent(
            gate_width,
            plan.tp_degree,
            0,
            allow_padding=plan.allow_padding,
        )
        if gate_width
        else None
    )
    q_projection_shard = shard_extent(
        q_projection_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    kv_projection_shard = shard_extent(
        kv_projection_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    query_head_shard = shard_extent(
        query_heads,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    kv_head_shard = shard_extent(
        kv_heads,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    qkv_width = q_projection_width + 2 * kv_projection_width
    rope_width = query_width + kv_projection_width
    qkv_shard = shard_extent(
        qkv_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    rope_shard = shard_extent(
        rope_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    query_output_bytes = _activation_bytes(
        layer, token_batch * q_projection_shard.local_size,
        scenario=scenario,
    )
    qkv_transient_output_bytes = _activation_bytes(
        layer, token_batch * qkv_shard.local_size,
        scenario=scenario,
    )
    rank_ends: List[str] = []
    prefix = "{}.{}".format(phase, layer.layer_id)
    for rank in tp_ranks:
        source_attention = source_geometry
        if source_geometry is not None and not _qwen35_cuda_source_layer(scenario, plan, rank, layer, source_geometry):
            source_attention = None
        native_kv = _native_local_kv_contract(
            scenario, router, plan, rank, layer, token_batch,
            kv_materialized_tokens, kv_append_tokens,
        )
        native_qkv_output = native_kv.get("status") == "applied"
        qkv_is_transient = attention_execution is not None or native_qkv_output
        qkv_output_bytes = (
            qkv_transient_output_bytes if qkv_is_transient
            else query_output_bytes + kv_materialized_bytes
        )
        activation_bits = _activation_storage_bits(layer, scenario)
        hidden_elements = max(
            1, token_batch * hidden_shard.local_size
        )
        rank_meta = {
            "phase": phase,
            "layer_id": layer.layer_id,
            "stage": stage,
            "shard_padding": hidden_shard.padding,
            "sequence_mixer": "full_attention",
            "coverage_component": "full_attention",
        }
        if native_kv:
            rank_meta["native_kv_work"] = native_kv
        if source_geometry is not None:
            rank_meta["qwen35_attention_geometry"] = {
                "gguf_sha256": source_geometry.gguf_sha256,
                "block_index": source_geometry.block_index,
                "query_heads": 8, "kv_heads": 2, "head_dim": 256,
                "query_width": 2048, "gate_width": 2048,
                "execution_path": "cuda_source_work" if source_attention is not None else "cpu_existing_descriptor_approximation",
                "cpu_source_kernel_counts_verified": False,
            }
        if attention_execution is not None:
            rank_meta.update(
                {
                    "attention_execution_descriptor_applied": True,
                    "attention_execution_descriptor_schema_version": (
                        "heterollm.attention-execution/v1"
                    ),
                    "attention_query_heads": query_heads,
                    "attention_kv_heads": kv_heads,
                    "attention_head_dim": head_dim,
                    "attention_query_width": query_width,
                    "attention_gate_width": gate_width,
                    "attention_q_projection_width": q_projection_width,
                    "attention_rotary_dim": attention_execution.rotary_dim,
                    "attention_qk_scale": attention_execution.qk_scale,
                    "attention_qk_norm": attention_execution.qk_norm,
                    "attention_gate_activation": (
                        attention_execution.gate_activation
                    ),
                }
            )
        residual_input_component = builder.rank_value_component(dependencies, rank.rank) or rank.component_id
        norm_reduce, norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=hidden_elements,
                output_elements=max(1, token_batch),
                operations_per_combine=2,
                fixed_operations=max(1, token_batch),
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                name="attention_input_norm_reduce",
            ),
            "{}.input_norm.reduce".format(layer.layer_id),
            "{}.rank{:03d}.input_norm_reduce".format(prefix, rank.rank),
            dependencies,
            source_component_id=rank.component_id,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**rank_meta, "event_kind": "input_norm_reduce",
                **({"input_tensor_id": prefix + ".rank{:03d}.residual_input".format(rank.rank),
                    "output_selection_input_shape": [token_batch, layer.hidden_size]}
                   if output_selection is not None and output_selection.position == "before_last_ffn" else {})},
        )
        input_reduction_component = norm_component
        norm_apply, norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=hidden_elements,
                operations_per_element=2,
                fixed_operations=max(1, token_batch),
                fixed_transcendental_operations=max(1, token_batch),
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=4,
                working_set_bytes=_activation_bytes(
                    layer, hidden_elements * 3,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="attention_input_norm_apply",
            ),
            "{}.input_norm.apply".format(layer.layer_id),
            "{}.rank{:03d}.input_norm_apply".format(prefix, rank.rank),
            (norm_reduce,),
            source_component_id=norm_component,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**rank_meta, "event_kind": "input_norm_apply"},
        )
        qkv_metadata = {
            **rank_meta,
            "projection_id": "attention.qkv",
            "projection_tp_degree": plan.tp_degree,
            "projection_tp_rank": rank.tp_rank,
            "projection_allow_padding": plan.allow_padding,
            "modeled_memory_write_bytes": qkv_output_bytes,
            "modeled_memory_write_kind": "qkv_output_including_kv_append",
            "modeled_kv_write_bytes": kv_materialized_bytes,
            "kv_materialized_tokens": max(0, kv_materialized_tokens),
            "kv_materialized_bytes": kv_materialized_bytes,
            "kv_persistent_append_tokens": max(0, kv_append_tokens),
            "kv_persistent_append_bytes": (
                _kv_bytes_per_token(scenario, layer, plan.tp_degree)
                * max(0, kv_append_tokens)
            ),
        }
        if qkv_is_transient:
            qkv_metadata.update(
                {
                    "modeled_memory_write_bytes": qkv_transient_output_bytes,
                    "modeled_memory_write_kind": "transient_qkv_activation",
                    "modeled_kv_write_bytes": 0,
                    "qkv_transient_output_bytes": (
                        qkv_transient_output_bytes
                    ),
                    "qkv_transient_q_gate_bytes": query_output_bytes,
                    "qkv_transient_kv_bytes": _activation_bytes(
                        layer,
                        token_batch
                        * 2
                        * kv_projection_shard.local_size,
                        scenario=scenario,
                    ),
                    "persistent_kv_append_accounting": (
                        "separate_kv_append_task"
                    ),
                }
            )
        if native_qkv_output:
            qkv_metadata.update(
                native_qkv_f32_output=True,
                kv_materialized_bytes_geometry_only=kv_materialized_bytes,
                kv_materialized_bytes=0,
                persistent_output_bytes=0,
                persistent_kv_append_accounting="native_cache_write_kernels",
            )
        qkv_target = _parallel_target(scenario, layer, "attention", rank)
        source_contract = None
        if source_attention is not None:
            source_contract = _qwen35_source_execution_contract(
                scenario, router, plan, rank, layer, source_attention, qkv_target, native_kv)
            rank_meta["native_kv_work"] = source_contract
            rank_meta["qwen35_source_execution_contract"] = source_contract
            qkv_metadata.update(native_kv_work=source_contract, qwen35_source_execution_contract=source_contract)
        kv_runtime_operand_component = _layer_local_runtime_memory_component_id(
            scenario,
            rank,
            qkv_target,
            rank.memory_component_id or rank.component_id,
        )
        rope_traffic_elements = max(
            1,
            token_batch
            * (
                query_shard.local_size + kv_projection_shard.local_size
                if attention_execution is not None
                else rope_shard.local_size
            ),
        )
        rope_compute_elements = max(
            1,
            token_batch
            * (
                (
                    query_head_shard.local_size
                    + kv_head_shard.local_size
                )
                * attention_execution.rotary_dim
                if attention_execution is not None
                else rope_shard.local_size
            ),
        )
        split_qkv = _declared_physical_projections(
            scenario, layer, ("attention.q", "attention.k", "attention.v"),
            combined_projection_id="attention.qkv",
        )
        qkv_workload = _layer_gemm(
            layer,
            token_batch,
            layer.hidden_size,
            qkv_shard.local_size,
            name="qkv_tp",
            projection_id="attention.qkv",
            projection_tp_degree=plan.tp_degree,
            projection_tp_rank=rank.tp_rank,
            projection_allow_padding=plan.allow_padding,
            f32_storage=_f32_hidden_storage_enabled(scenario),
        )
        qkv_workload = replace(qkv_workload, output_storage_bytes=qkv_output_bytes)
        rope_target = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.ELEMENTWISE,
            "{}.attention.rope".format(layer.layer_id),
            fallback_keys=("{}.attention".format(layer.layer_id),),
        )
        qkv_rope_fused, qkv_rope_audit = _same_rank_gpu_fusion_decision(
            scenario,
            "qkv_rope",
            rank,
            qkv_workload.output_bytes
            + _activation_bytes(layer, rope_traffic_elements * 2, scenario=scenario),
            (
                ("{}.attention".format(layer.layer_id), qkv_target),
                (
                    "{}.attention.rope".format(layer.layer_id),
                    rope_target,
                ),
            ),
        )
        qkv_rope_audit = dict(qkv_rope_audit)
        qkv_rope_audit.update(
            {
                "fusion_attention_target": qkv_target,
                "fusion_rope_target": rope_target,
            }
        )
        if attention_execution is not None and attention_execution.qk_norm:
            qkv_rope_fused = False
            qkv_rope_audit.update(
                {
                    "fusion_applied": False,
                    "fusion_reason": "explicit_qk_rmsnorm_boundary",
                }
            )
        if split_qkv:
            qkv_rope_fused = False
            qkv_rope_audit = dict(qkv_rope_audit)
            qkv_rope_audit.update(
                {
                    "fusion_enabled": False,
                    "fusion_decision": "physical_projection_boundary",
                    "physical_projection_invocations": ("q", "k", "v"),
                }
            )
        if qkv_rope_fused:
            qkv_workload = _append_gemm_epilogue(
                qkv_workload,
                operations=3 * rope_compute_elements,
                name="rope",
            )
        source_rope = source_append = None
        if source_attention is not None:
            if not split_qkv or source_contract is None:
                raise ValueError("Qwen3.5 source attention requires three physical projections")
            qkv, source_rope, source_append = _add_qwen35_source_qkv(
                builder, scenario, router, plan, rank, layer, source_attention, token_batch,
                prefix, norm_apply, norm_component, qkv_target,
                {**qkv_metadata, **qkv_rope_audit}, source_contract)
        elif not split_qkv:
            qkv = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                qkv_workload,
                qkv_target,
                "{}.rank{:03d}.qkv".format(prefix, rank.rank),
                (norm_apply,),
                weight_tensor_id="{}.attention_weights".format(layer.layer_id),
                activation_source_component_id=norm_component,
                keep_output_on_target=True,
                metadata={**qkv_metadata, **qkv_rope_audit},
            )
        else:
            split_tasks = []
            split_specs = (
                ("q", query_output_bytes, "query"),
                ("k", (qkv_output_bytes - query_output_bytes) // 2, "key"),
                ("v", (qkv_output_bytes - query_output_bytes) // 2, "value"),
            )
            for projection_suffix, output_bytes, projection_name in split_specs:
                workload = _layer_gemm(
                    layer,
                    token_batch,
                    layer.hidden_size,
                    q_projection_shard.local_size
                    if projection_suffix == "q"
                    else kv_projection_shard.local_size,
                    name="{}_tp".format(projection_name),
                    projection_id="attention.{}".format(projection_suffix),
                    projection_tp_degree=plan.tp_degree,
                    projection_tp_rank=rank.tp_rank,
                    projection_allow_padding=plan.allow_padding,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                )
                if qkv_is_transient:
                    # Each physical projection pads its own output shard.
                    output_bytes = _activation_bytes(layer, token_batch * workload.n, scenario=scenario)
                workload = replace(workload, output_storage_bytes=output_bytes)
                split_tasks.append(
                    _add_rank_gemm(
                        builder,
                        scenario,
                        router,
                        plan,
                        rank,
                        workload,
                        qkv_target,
                        "{}.rank{:03d}.attention_{}".format(prefix, rank.rank, projection_suffix),
                        (norm_apply,),
                        weight_tensor_id="{}.attention_weights".format(layer.layer_id),
                        activation_source_component_id=norm_component,
                        keep_output_on_target=True,
                        metadata={
                            **qkv_metadata,
                            **qkv_rope_audit,
                            "projection_id": "attention.{}".format(projection_suffix),
                            "physical_projection": projection_name,
                            "modeled_memory_write_bytes": output_bytes,
                            "modeled_memory_write_kind": "split_projection_activation",
                            "qkv_transient_output_bytes": output_bytes,
                            "modeled_kv_write_bytes": (
                                0 if projection_suffix == "q"
                                else qkv_metadata["modeled_kv_write_bytes"] // 2
                            ),
                        },
                    )
                )
            qkv = _add_join(
                builder,
                "{}.rank{:03d}.attention_projections_ready".format(prefix, rank.rank),
                tuple(split_tasks),
                metadata={**rank_meta, **qkv_rope_audit, "event_kind": "attention_projection_join"},
            )
        qkv_component = (
            rank.component_id
            if _is_cim(_component(scenario, qkv_target))
            else qkv_target
        )
        rope_dependencies: Tuple[str, ...] = (qkv,)
        rope_source_component = qkv_component
        rope_input_component_bytes: Optional[Tuple[Tuple[str, int], ...]] = None
        if attention_execution is not None and attention_execution.qk_norm and source_attention is None:
            q_norm_elements = max(
                1, token_batch * query_shard.local_size
            )
            q_norm_groups = max(
                1, token_batch * query_head_shard.local_size
            )
            k_norm_elements = max(
                1, token_batch * kv_projection_shard.local_size
            )
            k_norm_groups = max(
                1, token_batch * kv_head_shard.local_size
            )
            q_norm_reduce, q_norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.REDUCTION,
                ReductionWorkload(
                    input_elements=q_norm_elements,
                    output_elements=q_norm_groups,
                    operations_per_combine=2,
                    fixed_operations=q_norm_groups,
                    input_bits=activation_bits,
                    output_bits=max(16, activation_bits),
                    name="attention_q_rmsnorm_reduce",
                ),
                "{}.attention.q_norm.reduce".format(layer.layer_id),
                "{}.rank{:03d}.q_norm_reduce".format(prefix, rank.rank),
                (qkv,),
                source_component_id=qkv_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    "event_kind": "attention_q_norm_reduce",
                    "norm_kind": "rmsnorm",
                    "norm_groups": q_norm_groups,
                },
            )
            q_norm_apply, q_norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=q_norm_elements,
                    operations_per_element=2,
                    fixed_transcendental_operations=q_norm_groups,
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=4,
                    name="attention_q_rmsnorm_apply",
                ),
                "{}.attention.q_norm.apply".format(layer.layer_id),
                "{}.rank{:03d}.q_norm_apply".format(prefix, rank.rank),
                (q_norm_reduce,),
                source_component_id=q_norm_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    "event_kind": "attention_q_norm_apply",
                    "norm_kind": "rmsnorm",
                    "norm_groups": q_norm_groups,
                },
            )
            k_norm_reduce, k_norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.REDUCTION,
                ReductionWorkload(
                    input_elements=k_norm_elements,
                    output_elements=k_norm_groups,
                    operations_per_combine=2,
                    fixed_operations=k_norm_groups,
                    input_bits=activation_bits,
                    output_bits=max(16, activation_bits),
                    name="attention_k_rmsnorm_reduce",
                ),
                "{}.attention.k_norm.reduce".format(layer.layer_id),
                "{}.rank{:03d}.k_norm_reduce".format(prefix, rank.rank),
                (qkv,),
                source_component_id=qkv_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    "event_kind": "attention_k_norm_reduce",
                    "norm_kind": "rmsnorm",
                    "norm_groups": k_norm_groups,
                },
            )
            k_norm_apply, k_norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=k_norm_elements,
                    operations_per_element=2,
                    fixed_transcendental_operations=k_norm_groups,
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=4,
                    name="attention_k_rmsnorm_apply",
                ),
                "{}.attention.k_norm.apply".format(layer.layer_id),
                "{}.rank{:03d}.k_norm_apply".format(prefix, rank.rank),
                (k_norm_reduce,),
                source_component_id=k_norm_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    "event_kind": "attention_k_norm_apply",
                    "norm_kind": "rmsnorm",
                    "norm_groups": k_norm_groups,
                },
            )
            qk_norm_ready = _add_join(
                builder,
                "{}.rank{:03d}.qk_norm_ready".format(prefix, rank.rank),
                (q_norm_apply, k_norm_apply),
                metadata={
                    **rank_meta,
                    "event_kind": "attention_qk_norm_ready",
                },
            )
            rope_dependencies = (qk_norm_ready,)
            rope_source_component = q_norm_component
            rope_input_component_bytes = (
                (
                    q_norm_component,
                    _activation_bytes(layer, q_norm_elements, scenario=scenario),
                ),
                (
                    k_norm_component,
                    _activation_bytes(layer, k_norm_elements, scenario=scenario),
                ),
            )
        native_kv_append = source_append
        if source_attention is not None:
            rope, rope_component = source_rope, rank.component_id
        elif native_qkv_output:
            rope, native_kv_append = _add_native_local_kv_writeback(
                builder, scenario, router, plan, rank, layer,
                token_batch, query_shard.local_size, kv_projection_shard.local_size,
                (query_head_shard.local_size * attention_execution.rotary_dim
                 if attention_execution is not None else query_shard.local_size),
                (kv_head_shard.local_size * attention_execution.rotary_dim
                 if attention_execution is not None else kv_projection_shard.local_size),
                rope_dependencies, name="{}.rank{:03d}.native_kv".format(prefix, rank.rank),
                contract=native_kv, metadata=rank_meta,
            )
            rope_component = rank.component_id
        elif qkv_rope_fused:
            rope, rope_component = qkv, qkv_component
        else:
            rope_read_bytes = None
            rope_write_bytes = None
            if attention_execution is not None:
                rope_read_bytes = _activation_bytes(
                    layer,
                    rope_traffic_elements + 2 * rope_compute_elements,
                    scenario=scenario,
                )
                rope_write_bytes = _activation_bytes(
                    layer, rope_traffic_elements,
                    scenario=scenario,
                )
            rope, rope_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=rope_compute_elements,
                    operations_per_element=3,
                    input_count=3,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=3,
                    working_set_bytes=_activation_bytes(
                        layer,
                        (
                            rope_traffic_elements * 2
                            + 2 * rope_compute_elements
                        ),
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    read_storage_bytes=rope_read_bytes,
                    write_storage_bytes=rope_write_bytes,
                    name="rope",
                ),
                "{}.attention.rope".format(layer.layer_id),
                "{}.rank{:03d}.rope".format(prefix, rank.rank),
                rope_dependencies,
                source_component_id=rope_source_component,
                input_component_bytes=rope_input_component_bytes,
                fallback_keys=("{}.attention".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    **qkv_rope_audit,
                    "event_kind": "rope",
                    "rope_table_strategy": "precomputed_sin_cos",
                    "rope_qk_traffic_elements": rope_traffic_elements,
                    "rope_rotated_elements": rope_compute_elements,
                },
            )
        qkv_ready = _return_rank_value(
            builder,
            router,
            plan,
            rank,
            rope,
            rope_component,
            _activation_bytes(layer, token_batch * qkv_workload.n, scenario=scenario),
            name="{}.rank{:03d}.rope.output_to_gpu".format(prefix, rank.rank),
            metadata={**rank_meta, "operator_id": "{}.attention.rope".format(layer.layer_id)},
        )
        kv_read = _add_kv_read(
            builder,
            scenario,
            router,
            plan,
            rank,
            layer,
            kv_read_tokens,
            dependencies,
            name="{}.rank{:03d}.kv".format(prefix, rank.rank),
            target_component_id=qkv_target,
        )
        attention_ready = _add_join(
            builder,
            "{}.rank{:03d}.attention_ready".format(prefix, rank.rank),
            (qkv_ready, kv_read) + ((native_kv_append,) if native_kv_append else ()),
            metadata=rank_meta,
        )
        qk_target = _parallel_named_target(
            scenario,
            "{}.attention.qk".format(layer.layer_id),
            rank,
            fallback_keys=("{}.attention".format(layer.layer_id),),
        )
        score_heads = (
            query_head_shard.local_size
            if attention_execution is not None
            else 1
        )
        score_elements = max(
            1, score_heads * token_batch * context_tokens
        )
        pv_target = _parallel_named_target(
            scenario,
            "{}.attention.pv".format(layer.layer_id),
            rank,
            fallback_keys=("{}.attention".format(layer.layer_id),),
        )
        softmax_reduce_target = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.REDUCTION,
            "{}.attention.softmax.reduce".format(layer.layer_id),
            fallback_keys=("{}.softmax".format(layer.layer_id),),
        )
        softmax_target = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.ELEMENTWISE,
            "{}.attention.softmax.normalize".format(layer.layer_id),
            fallback_keys=("{}.softmax".format(layer.layer_id),),
        )
        materialization_view = 0
        runtime_metadata = scenario.workload.metadata.get("serving_runtime", {})
        prompt_cache_metadata = (
            runtime_metadata.get("prompt_cache", {})
            if isinstance(runtime_metadata, Mapping) else {}
        )
        if (
            q4_mma_view_tokens_lower_bound
            and scenario.workload.metadata.get("llama_cpp_q4_kv_mma_materialization") is True
            and isinstance(prompt_cache_metadata, Mapping)
            and prompt_cache_metadata.get("unified_kv") is True
            and plan.world_size == 1
            and scenario.workload.mtp is None
            and scenario.placement.kv_policy.offload_ratio == 0.0
            and _kind(_component(scenario, rank.component_id)) == "gpu"
            and token_batch >= 3
            and head_dim in {64, 128, 256}
            and query_shard.local_size % head_dim == 0
            and kv_physical_contract is not None
            and kv_physical_contract.artifact_format.casefold() == "q4_0"
        ):
            # The explicit flag declares the fixed b10760 Ada+ MMA dispatch.
            # Preserve the logical context separately; the consumer cost model
            # uses this span for untrimmed rectangular execution where supported.
            materialization_view = q4_mma_view_tokens_lower_bound
        fused_attention_workload = FusedAttentionWorkload(
            batch_tokens=token_batch,
            context_tokens=max(1, context_tokens),
            hidden_size=query_shard.local_size,
            input_bits=_layer_precision_bits(layer)[0],
            output_bits=activation_bits,
            query_storage_bits=32 if _f32_hidden_storage_enabled(scenario) else None,
            name="attention_qk_softmax_pv",
            kv_hidden_size=kv_hidden_size,
            kv_input_bits=kv_input_bits,
            kv_read_tokens=max(0, kv_read_tokens),
            kv_physical_contract=kv_physical_contract,
            score_heads=score_heads,
            qk_scale=(
                attention_execution.qk_scale
                if attention_execution is not None
                else None
            ),
            q4_mma_view_tokens_lower_bound=materialization_view,
            q4_mma_head_dim=head_dim if materialization_view else 0,
        )
        flash_fusion_targets = (
            ("{}.attention.qk".format(layer.layer_id), qk_target),
            (
                "{}.attention.softmax.reduce".format(layer.layer_id),
                softmax_reduce_target,
            ),
            (
                "{}.attention.softmax.normalize".format(layer.layer_id),
                softmax_target,
            ),
            ("{}.attention.pv".format(layer.layer_id), pv_target),
        )
        flash_allowed, flash_audit = _same_rank_gpu_fusion_decision(
            scenario,
            "flash_attention",
            rank,
            fused_attention_workload.onchip_working_set_bytes,
            flash_fusion_targets,
        )
        flash_audit = dict(flash_audit)
        flash_audit.update(
            {
                "fusion_qk_target": qk_target,
                "fusion_softmax_reduce_target": softmax_reduce_target,
                "fusion_softmax_normalize_target": softmax_target,
                "fusion_pv_target": pv_target,
            }
        )
        if source_attention is not None and not flash_allowed:
            raise ValueError("Qwen3.5 source attention requires its declared same-GPU FlashAttention path")
        if flash_allowed:
            pv = _add_rank_fused_attention(
                builder,
                scenario,
                router,
                plan,
                rank,
                fused_attention_workload,
                "{}.rank{:03d}.attention_flash".format(
                    prefix, rank.rank
                ),
                (attention_ready,),
                metadata={**rank_meta, **flash_audit, **({
                    "output_tensor_id": prefix + ".rank{:03d}.qwen35.context".format(rank.rank),
                } if source_attention is not None else {})},
                fusion_targets=flash_fusion_targets,
            )
            pv_component = rank.component_id
        else:
            qk_workload = replace(
                _layer_gemm(
                    layer,
                    token_batch,
                    query_shard.local_size,
                    context_tokens,
                    name="attention_qk_tp",
                    dynamic_rhs=True,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                ),
                weight_storage_bytes=_kv_tensor_storage_metadata_bytes(
                    scenario,
                    layer,
                    plan.tp_degree,
                    kv_read_tokens,
                )[0],
                weight_metadata_bytes=_kv_tensor_storage_metadata_bytes(
                    scenario,
                    layer,
                    plan.tp_degree,
                    kv_read_tokens,
                )[1],
                output_storage_bytes=_activation_bytes(
                    layer, score_elements,
                    scenario=scenario,
                ),
            )
            qk = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                qk_workload,
                qk_target,
                "{}.rank{:03d}.attention_qk".format(prefix, rank.rank),
                (attention_ready,),
                model_weight_read=False,
                dynamic_rhs_source_component_id=kv_runtime_operand_component,
                dynamic_rhs=True,
                keep_output_on_target=True,
                metadata={**rank_meta, **flash_audit},
                dynamic_attention_replay=(
                    _DynamicAttentionCostTaskReplayPayload(
                        role="qk",
                        workload=qk_workload,
                        fused_workload=fused_attention_workload,
                        fusion_targets=flash_fusion_targets,
                        layer=layer,
                        rank=rank,
                        tp_degree=plan.tp_degree,
                        token_batch=token_batch,
                        score_heads=score_heads,
                    )
                ),
            )
            qk_component = (
                rank.component_id
                if _is_cim(_component(scenario, qk_target))
                else qk_target
            )
            qk_scaled = qk
            qk_scaled_component = qk_component
            if attention_execution is not None:
                qk_scale_workload = ElementwiseWorkload(
                    elements=score_elements,
                    operations_per_element=1,
                    input_count=1,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    name="attention_qk_scale",
                )
                qk_scaled, qk_scaled_component = _add_rank_primitive(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    OperatorClass.ELEMENTWISE,
                    qk_scale_workload,
                    "{}.attention.qk_scale".format(layer.layer_id),
                    "{}.rank{:03d}.qk_scale".format(prefix, rank.rank),
                    (qk,),
                    source_component_id=qk_component,
                    fallback_keys=(
                        "{}.attention".format(layer.layer_id),
                    ),
                    metadata={
                        **rank_meta,
                        **flash_audit,
                        "event_kind": "attention_qk_scale",
                        "qk_scale": attention_execution.qk_scale,
                        "score_heads": score_heads,
                        "score_elements": score_elements,
                    },
                    dynamic_attention_replay=(
                        _DynamicAttentionCostTaskReplayPayload(
                            role="qk_scale",
                            workload=qk_scale_workload,
                            fused_workload=fused_attention_workload,
                            fusion_targets=flash_fusion_targets,
                            layer=layer,
                            rank=rank,
                            tp_degree=plan.tp_degree,
                            token_batch=token_batch,
                            score_heads=score_heads,
                        )
                    ),
                )
            softmax_reduce_workload = ReductionWorkload(
                input_elements=score_elements,
                output_elements=max(1, score_heads * token_batch),
                operations_per_combine=2,
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                dependency_depth=max(
                    1, int(math.ceil(math.log2(max(1, context_tokens))))
                ),
                working_set_bytes=_activation_bytes(
                    layer, score_elements,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="softmax_reduce_max_sum",
            )
            softmax_reduce, softmax_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.REDUCTION,
                softmax_reduce_workload,
                "{}.attention.softmax.reduce".format(layer.layer_id),
                "{}.rank{:03d}.softmax_reduce".format(prefix, rank.rank),
                (qk_scaled,),
                source_component_id=qk_scaled_component,
                fallback_keys=("{}.softmax".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    **flash_audit,
                    "event_kind": "softmax_reduce",
                    "softmax_reductions": ("max", "sum"),
                    "softmax_reduction_groups": (
                        score_heads * token_batch
                    ),
                },
                dynamic_attention_replay=(
                    _DynamicAttentionCostTaskReplayPayload(
                        role="softmax_reduce",
                        workload=softmax_reduce_workload,
                        fused_workload=fused_attention_workload,
                        fusion_targets=flash_fusion_targets,
                        layer=layer,
                        rank=rank,
                        tp_degree=plan.tp_degree,
                        token_batch=token_batch,
                        score_heads=score_heads,
                    )
                ),
            )
            softmax_workload = ElementwiseWorkload(
                elements=score_elements,
                operations_per_element=2,
                transcendental_ops_per_element=1,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=3,
                working_set_bytes=_activation_bytes(
                    layer, score_elements * 2,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="softmax_sub_exp_normalize",
            )
            softmax, softmax_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                softmax_workload,
                "{}.attention.softmax.normalize".format(layer.layer_id),
                "{}.rank{:03d}.softmax_normalize".format(prefix, rank.rank),
                (softmax_reduce,),
                source_component_id=softmax_component,
                fallback_keys=("{}.softmax".format(layer.layer_id),),
                metadata={
                    **rank_meta,
                    **flash_audit,
                    "event_kind": "softmax_normalize",
                    "softmax_transcendental": "exp",
                },
                dynamic_attention_replay=(
                    _DynamicAttentionCostTaskReplayPayload(
                        role="softmax_normalize",
                        workload=softmax_workload,
                        fused_workload=fused_attention_workload,
                        fusion_targets=flash_fusion_targets,
                        layer=layer,
                        rank=rank,
                        tp_degree=plan.tp_degree,
                        token_batch=token_batch,
                        score_heads=score_heads,
                    )
                ),
            )
            pv_workload = replace(
                _layer_gemm(
                    layer,
                    token_batch,
                    context_tokens,
                    query_shard.local_size,
                    name="attention_pv_tp",
                    dynamic_rhs=True,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                ),
                weight_storage_bytes=_kv_tensor_storage_metadata_bytes(
                    scenario,
                    layer,
                    plan.tp_degree,
                    kv_read_tokens,
                )[0],
                weight_metadata_bytes=_kv_tensor_storage_metadata_bytes(
                    scenario,
                    layer,
                    plan.tp_degree,
                    kv_read_tokens,
                )[1],
                activation_storage_bytes=_activation_bytes(
                    layer, score_elements,
                    scenario=scenario,
                ),
            )
            pv = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                pv_workload,
                pv_target,
                "{}.rank{:03d}.attention_pv".format(prefix, rank.rank),
                (softmax,),
                model_weight_read=False,
                activation_source_component_id=softmax_component,
                dynamic_rhs_source_component_id=kv_runtime_operand_component,
                dynamic_rhs=True,
                keep_output_on_target=True,
                metadata={**rank_meta, **flash_audit},
                dynamic_attention_replay=(
                    _DynamicAttentionCostTaskReplayPayload(
                        role="pv",
                        workload=pv_workload,
                        fused_workload=fused_attention_workload,
                        fusion_targets=flash_fusion_targets,
                        layer=layer,
                        rank=rank,
                        tp_degree=plan.tp_degree,
                        token_batch=token_batch,
                        score_heads=score_heads,
                    )
                ),
            )
            pv_component = (
                rank.component_id
                if _is_cim(_component(scenario, pv_target))
                else pv_target
            )
        attention_context = pv
        attention_context_component = pv_component
        if source_attention is not None:
            owner = lambda role: prefix + ".rank{:03d}.qwen35.{}".format(rank.rank, role)
            gate_name = prefix + ".rank{:03d}".format(rank.rank)
            context_owner = owner("context")
            if source_contract["hadamard_enabled"]:
                attention_context = _add_qwen35_source_tensor(
                    builder, scenario, router, plan, rank, source_attention, "fwht64", token_batch, 8,
                    gate_name + ".context_hadamard", (pv,), inputs=((context_owner, 8192 * token_batch),),
                    output=(owner("context_hadamard"), 8192 * token_batch), metadata=rank_meta)
                context_owner = owner("context_hadamard")
            contiguous_gate = _add_qwen35_source_tensor(
                builder, scenario, router, plan, rank, source_attention, "gate_contiguous", token_batch, 8,
                gate_name + ".gate_contiguous", (attention_context, qkv),
                inputs=((owner("qg"), 16384 * token_batch),),
                output=(owner("gate_contiguous"), 8192 * token_batch), metadata=rank_meta)
            attention_context = _add_qwen35_source_tensor(
                builder, scenario, router, plan, rank, source_attention, "sigmoid_mul", token_batch, 8,
                gate_name + ".attention_gate", (attention_context, contiguous_gate),
                inputs=((context_owner, 8192 * token_batch), (owner("gate_contiguous"), 8192 * token_batch)),
                output=(owner("gated"), 8192 * token_batch), metadata=rank_meta)
            attention_context_component = rank.component_id
        elif attention_execution is not None:
            if gate_shard is None:  # pragma: no cover - descriptor invariant
                raise AssertionError("gated attention is missing a gate shard")
            gate_elements = max(
                1, token_batch * gate_shard.local_size
            )
            attention_context, attention_context_component = (
                _add_rank_primitive(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    OperatorClass.ELEMENTWISE,
                    ElementwiseWorkload(
                        elements=gate_elements,
                        operations_per_element=3,
                        transcendental_ops_per_element=1,
                        input_count=2,
                        input_bits=activation_bits,
                        output_bits=activation_bits,
                        dependency_depth=4,
                        working_set_bytes=_activation_bytes(
                            layer, gate_elements * 3,
                            scenario=scenario,
                        ),
                        reuse_factor=2.0,
                        name="attention_sigmoid_gate_context",
                    ),
                    "{}.attention.gate".format(layer.layer_id),
                    "{}.rank{:03d}.attention_gate".format(
                        prefix, rank.rank
                    ),
                    (pv, qkv),
                    source_component_id=pv_component,
                    input_component_bytes=(
                        (
                            pv_component,
                            _activation_bytes(layer, gate_elements, scenario=scenario),
                        ),
                        (
                            qkv_component,
                            _activation_bytes(layer, gate_elements, scenario=scenario),
                        ),
                    ),
                    fallback_keys=(
                        "{}.attention".format(layer.layer_id),
                    ),
                    metadata={
                        **rank_meta,
                        "event_kind": "attention_gate",
                        "gate_activation": (
                            attention_execution.gate_activation
                        ),
                        "gate_semantics": "sigmoid_gate_times_context",
                        "gate_elements": gate_elements,
                    },
                )
            )
        output_target = _parallel_target(scenario, layer, "attention", rank)
        output = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            _layer_gemm(
                layer,
                token_batch,
                hidden_shard.local_size,
                layer.hidden_size,
                name="attention_output_tp",
                projection_id="attention.output",
                projection_tp_degree=plan.tp_degree,
                projection_tp_rank=rank.tp_rank,
                projection_allow_padding=plan.allow_padding,
                f32_storage=_f32_hidden_storage_enabled(scenario),
            ),
            output_target,
            "{}.rank{:03d}.attention_output".format(prefix, rank.rank),
            (attention_context,),
            weight_tensor_id="{}.attention_weights".format(layer.layer_id),
            activation_source_component_id=attention_context_component,
            keep_output_on_target=True,
            metadata={
                **rank_meta,
                "projection_id": "attention.output",
                **({"output_tensor_id": prefix + ".rank{:03d}.attention_output_value".format(rank.rank)}
                   if output_selection is not None and output_selection.position == "before_last_ffn" else {}),
                **({"input_tensor_id": prefix + ".rank{:03d}.qwen35.gated".format(rank.rank)}
                   if source_attention is not None else {}),
                "projection_tp_degree": plan.tp_degree,
                "projection_tp_rank": rank.tp_rank,
                "projection_allow_padding": plan.allow_padding,
            },
        )
        output_component = (
            rank.component_id
            if _is_cim(_component(scenario, output_target))
            else output_target
        )
        tail_elements = tail_rows * hidden_shard.local_size
        residual_inputs = (output,)
        if output_selection is not None and output_selection.position == "before_last_ffn" and tail_rows:
            if output_indices_dependency is None:
                raise ValueError("selected last layer has no shared output-index dependency")
            original_output = output
            output, output_component = _add_output_row_selection(
                builder, scenario, router, plan, rank, layer, output_selection,
                prefix + ".rank{:03d}.attention_output_rows".format(rank.rank), (output,),
                stage="attention_output_rows", source_component=output_component,
                target_component=output_component, indices_dependency=output_indices_dependency,
                input_tensor_id=prefix + ".rank{:03d}.attention_output_value".format(rank.rank))
            # The input reduction consumed precisely this layer's original full
            # input. Its same-destination copy can be shared by that same source
            # tensor, not by an unrelated tensor merely living on the GPU.
            residual_source = (output_component if input_reduction_component == output_component
                               else residual_input_component)
            selected_residual, _ = _add_output_row_selection(
                builder, scenario, router, plan, rank, layer, output_selection,
                prefix + ".rank{:03d}.residual_input_rows".format(rank.rank),
                tuple(dict.fromkeys((*dependencies, original_output))), stage="residual_input_rows",
                source_component=residual_source, target_component=output_component,
                indices_dependency=output_indices_dependency,
                input_tensor_id=prefix + ".rank{:03d}.residual_input".format(rank.rank))
            residual_inputs = (output, selected_residual)
            rank_meta = {**rank_meta, "final_layer_output_selection": _selection_audit(output_selection, "last_ffn")}
        if tail_rows == 0:
            # Empty residual/FFN nodes do not execute; attention and KV remain.
            residual_norm_audit = {}
        else:
            residual_norm_fused, residual_norm_audit = (
                _residual_norm_fusion_decision(
                    scenario,
                    router,
                    rank,
                    layer,
                    max(1, tail_rows * layer.hidden_size),
                )
            )
            if residual_norm_fused:
                residual = (_add_join(builder, prefix + ".selected_residual_inputs", residual_inputs, metadata=rank_meta)
                            if output_selection is not None else output)
                residual_component = output_component
            else:
                residual, residual_component = _add_rank_primitive(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    OperatorClass.ELEMENTWISE,
                    ElementwiseWorkload(
                        elements=tail_elements,
                        operations_per_element=1,
                        input_count=2,
                        input_bits=activation_bits,
                        output_bits=activation_bits,
                        dependency_depth=1,
                        working_set_bytes=_activation_bytes(
                            layer, tail_elements * 3,
                            scenario=scenario,
                        ),
                        reuse_factor=2.0,
                        name="attention_residual",
                    ),
                    "{}.attention.residual".format(layer.layer_id),
                    "{}.rank{:03d}.attention_residual".format(prefix, rank.rank),
                    residual_inputs,
                    source_component_id=output_component,
                    fallback_keys=("{}.norm".format(layer.layer_id),),
                    metadata={
                        **rank_meta,
                        **residual_norm_audit,
                        "event_kind": "attention_residual",
                    },
                )
            output = _return_rank_value(
                builder,
                router,
                plan,
                rank,
                residual,
                residual_component,
                _activation_bytes(layer, tail_elements, scenario=scenario),
                name="{}.rank{:03d}.attention_residual.output_to_gpu".format(prefix, rank.rank),
                metadata={**rank_meta, **residual_norm_audit},
            )
        kv_append = native_kv_append or _add_kv_append(
            builder,
            scenario,
            router,
            plan,
            rank,
            layer,
            kv_append_tokens,
            (qkv_ready,),
            name="{}.rank{:03d}.kv".format(prefix, rank.rank),
            target_component_id=qkv_target,
        )
        rank_ends.append(
            _add_join(
                builder,
                "{}.rank{:03d}.attention_complete".format(prefix, rank.rank),
                (output, kv_append),
                metadata=rank_meta,
            )
        )
    if tail_rows == 0:
        return _add_join(builder, prefix + ".empty_tail", rank_ends,
            metadata={"event_kind": "empty_output_tail", "layer_id": layer.layer_id,
                "final_layer_output_selection": _selection_audit(output_selection, "empty_tail")})
    attention_bytes = _activation_bytes(layer, tail_rows * layer.hidden_size, scenario=scenario)
    attention_end = _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".attention_all_reduce",
        "all_reduce",
        tp_ranks,
        attention_bytes,
        rank_ends,
        tensor_elements=tail_rows * layer.hidden_size,
        element_bits=_activation_storage_bits(layer, scenario),
        metadata={
            "phase": phase,
            "layer_id": layer.layer_id,
            "stage": stage,
            "sequence_mixer": "full_attention",
            "coverage_component": "full_attention",
        },
    )
    if layer.is_moe:
        return _compile_parallel_moe(
            builder,
            scenario,
            plan,
            router,
            layer,
            tail_rows,
            prefix,
            (attention_end,),
        )
    first_ffn = len(builder.tasks)
    terminal = _compile_parallel_dense_mlp(
        builder,
        scenario,
        plan,
        router,
        layer,
        tail_rows,
        prefix,
        (attention_end,),
    )

    if output_selection is not None:
        _tag_output_selection_tasks(builder, first_ffn, output_selection, "last_ffn")
    return terminal

def _compile_parallel_linear_mixer(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    *,
    token_batch: int,
    context_tokens: int,
    linear_state_runtime: Optional[_LinearStateRuntimeSemantics] = None,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    """Compile or replay a linear mixer while refreshing context audit data.

    Linear attention carries fixed recurrent state and O(token_batch) work;
    historical context is recorded for audit but does not enter any workload
    or demand.  Runtime state semantics remain in the exact cache key, so MTP
    commit/release boundaries cannot share a segment accidentally.
    """

    context = _active_compilation_context(scenario)
    if context is None:
        return _compile_parallel_linear_mixer_uncached(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            linear_state_runtime=linear_state_runtime,
            phase=phase,
            dependencies=dependencies,
        )
    cache = _task_segment_cache(
        context, ("linear_mixer_task_segment_cache_v1",)
    )
    if cache is None:
        return _compile_parallel_linear_mixer_uncached(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch=token_batch,
            context_tokens=context_tokens,
            linear_state_runtime=linear_state_runtime,
            phase=phase,
            dependencies=dependencies,
        )
    state_runtime = linear_state_runtime or _LinearStateRuntimeSemantics()
    ranks = plan.tp_group(plan.stage_for_layer(layer), 0)
    input_components = tuple(
        builder.rank_value_component(dependencies, rank.rank)
        for rank in ranks
    )
    scaling = "O(T)" if token_batch > 1 or phase.startswith("prefill") else "O(1)"
    runtime_structure = _linear_state_runtime_structure_identity(
        state_runtime
    )
    cache_key = (
        str(layer.layer_id),
        int(token_batch),
        (
            runtime_structure
            if runtime_structure is not None
            else ("linear_state_exact_v1", state_runtime)
        ),
        scaling,
        input_components,
        len(tuple(dependencies)),
        builder._last_coherent_dma_task is not None,
    )
    prefix = "{}.{}".format(phase, layer.layer_id)
    template = _task_segment_cache_get(cache, cache_key)
    if template is not None:
        runtime_override_rules: Tuple[
            Tuple[str, object, Mapping[str, object]], ...
        ] = ()
        if runtime_structure is not None:
            rules = [
                (
                    "linear_state_runtime_semantics",
                    state_runtime.mode,
                    state_runtime.audit_metadata(),
                )
            ]
            if state_runtime.has_persistent_updates:
                persistent_runtime = _LinearStateRuntimeSemantics(
                    mode="persistent_update",
                    read_source="committed",
                    persistent_update_lane_ids=(
                        state_runtime.persistent_update_lane_ids
                    ),
                )
                rules.append(
                    (
                        "linear_state_runtime_semantics",
                        persistent_runtime.mode,
                        persistent_runtime.audit_metadata(),
                    )
                )
            runtime_override_rules = tuple(rules)
        replayed = template.replay(
            builder,
            prefix=prefix,
            dependencies=dependencies,
            metadata_overrides={
                "phase": phase,
                "context_tokens_observed": context_tokens,
            },
            metadata_override_rules=runtime_override_rules,
        )
        if replayed is not None:
            return replayed

    first_task_index = len(builder.tasks)
    initial_counter = builder.counter
    initial_previous = builder.previous
    initial_dma = builder._last_coherent_dma_task
    terminal = _compile_parallel_linear_mixer_uncached(
        builder,
        scenario,
        plan,
        router,
        layer,
        token_batch=token_batch,
        context_tokens=context_tokens,
        linear_state_runtime=linear_state_runtime,
        phase=phase,
        dependencies=dependencies,
    )
    captured = _TaskSegmentTemplate.capture(
        builder,
        first_task_index=first_task_index,
        source_prefix=prefix,
        source_counter_before=initial_counter,
        source_dependencies=dependencies,
        source_initial_previous=initial_previous,
        source_initial_dma=initial_dma,
        terminal_task_id=terminal,
        source_phase=phase,
    )
    if captured is not None:
        _task_segment_cache_put(context, cache, cache_key, captured)
    return terminal


def _compile_parallel_linear_mixer_uncached(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    *,
    token_batch: int,
    context_tokens: int,
    linear_state_runtime: Optional[_LinearStateRuntimeSemantics] = None,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    """Lower a linear mixer without attention/KV-shaped surrogate GEMMs."""

    geometry = layer.linear_attention
    if geometry is None:
        raise ValueError("线性注意力层缺少明确的几何参数")
    state_runtime = linear_state_runtime or _LinearStateRuntimeSemantics()
    stage = plan.stage_for_layer(layer)
    ranks = plan.tp_group(stage, 0)
    prefix = "{}.{}".format(phase, layer.layer_id)
    projected_width = (
        geometry.query_width + geometry.key_width + geometry.value_width
    )
    projected_shard = shard_extent(
        projected_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    value_shard = shard_extent(
        geometry.value_width,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    activation_bits = _activation_storage_bits(layer, scenario)
    activation_bytes = max(1, int(math.ceil(activation_bits / 8.0)))
    state_bytes = _linear_state_bytes(layer, plan.tp_degree)
    recurrent_elements = int(
        math.ceil(geometry.recurrent_state_elements / float(plan.tp_degree))
    )
    physical_projection_invocations = _declared_physical_projections(
        scenario,
        layer,
        ("linear_attention.alpha", "linear_attention.beta"),
    )
    rank_ends: List[str] = []
    scaling = "O(T)" if token_batch > 1 or phase.startswith("prefill") else "O(1)"
    for rank in ranks:
        target = _parallel_target(scenario, layer, "linear_attention", rank)
        projection_workload = _layer_gemm(
            layer,
            token_batch,
            layer.hidden_size,
            projected_shard.local_size,
            name="linear_qkv_projection_tp",
            projection_id="linear_attention.qkv",
            projection_tp_degree=plan.tp_degree,
            projection_tp_rank=rank.tp_rank,
            projection_allow_padding=plan.allow_padding,
            f32_storage=_f32_hidden_storage_enabled(scenario),
        )
        linear_weight_tensor_id = "{}.linear_attention_weights".format(
            layer.layer_id
        )
        weight_source, _source_tensor, _logical_tensor = (
            _weight_source_for_tensor(
                scenario,
                linear_weight_tensor_id,
                target,
                rank=rank,
            )
        )
        activation_source = (
            builder.rank_value_component(dependencies, rank.rank)
            or rank.component_id
        )
        host_projection_decision = _host_gemm_offload_decision(
            scenario,
            router,
            plan,
            rank,
            projection_workload,
            target,
            activation_source,
            weight_source,
            model_weight_read=True,
            dynamic_rhs=False,
        )
        recurrent_offload = _host_recurrent_offload_decision(
            scenario,
            router,
            plan,
            rank,
            layer,
            physical_m=token_batch,
            placement_component_id=target,
            host_projection_decision=host_projection_decision,
        )
        recurrent_target = (
            recurrent_offload.execution_component_id
            if recurrent_offload is not None
            else target
        )
        recurrent_audit = (
            recurrent_offload.audit_metadata
            if recurrent_offload is not None
            else {}
        )
        metadata = {
            "phase": phase,
            "layer_id": layer.layer_id,
            "stage": stage,
            "sequence_mixer": "linear_attention",
            "coverage_component": "linear_attention",
            "context_scaling": scaling,
            "context_tokens_observed": context_tokens,
            "state_bytes": state_bytes,
            **recurrent_audit,
        }
        hidden_elements = max(1, token_batch * layer.hidden_size)
        norm_reduce, norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=hidden_elements,
                output_elements=max(1, token_batch),
                operations_per_combine=2,
                fixed_operations=max(1, token_batch),
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                name="linear_input_norm_reduce",
            ),
            "{}.input_norm.reduce".format(layer.layer_id),
            "{}.rank{:03d}.linear_input_norm_reduce".format(prefix, rank.rank),
            dependencies,
            source_component_id=rank.component_id,
            target_component_id=recurrent_target,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**metadata, "linear_op": "input_norm_reduce"},
        )
        norm_apply, norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=hidden_elements,
                operations_per_element=2,
                fixed_operations=max(1, token_batch),
                fixed_transcendental_operations=max(1, token_batch),
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=4,
                working_set_bytes=_activation_bytes(
                    layer, hidden_elements * 3,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="linear_input_norm_apply",
            ),
            "{}.input_norm.apply".format(layer.layer_id),
            "{}.rank{:03d}.linear_input_norm_apply".format(prefix, rank.rank),
            (norm_reduce,),
            source_component_id=norm_component,
            target_component_id=recurrent_target,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**metadata, "linear_op": "input_norm_apply"},
        )
        projection = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            projection_workload,
            target,
            "{}.rank{:03d}.linear_qkv_projection".format(prefix, rank.rank),
            (norm_apply,),
            weight_tensor_id=linear_weight_tensor_id,
            activation_source_component_id=norm_component,
            metadata={
                **metadata,
                "linear_op": "projection",
                "projection_id": "linear_attention.qkv",
                "projection_tp_degree": plan.tp_degree,
                "projection_tp_rank": rank.tp_rank,
                "projection_allow_padding": plan.allow_padding,
            },
        )
        gate_projection = ""
        if geometry.output_gate:
            gate_projection = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                _layer_gemm(
                    layer,
                    token_batch,
                    layer.hidden_size,
                    value_shard.local_size,
                    name="linear_output_gate_projection_tp",
                    projection_id="linear_attention.output_gate",
                    projection_tp_degree=plan.tp_degree,
                    projection_tp_rank=rank.tp_rank,
                    projection_allow_padding=plan.allow_padding,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                ),
                target,
                "{}.rank{:03d}.linear_output_gate_projection".format(
                    prefix, rank.rank
                ),
                (norm_apply,),
                weight_tensor_id="{}.linear_attention_weights".format(
                    layer.layer_id
                ),
                activation_source_component_id=norm_component,
                metadata={
                    **metadata,
                    "linear_op": "output_gate_projection",
                    "projection_id": "linear_attention.output_gate",
                    "projection_tp_degree": plan.tp_degree,
                    "projection_tp_rank": rank.tp_rank,
                    "projection_allow_padding": plan.allow_padding,
                },
            )
        control_projections = []
        control_outputs: List[Tuple[str, int]] = []
        if physical_projection_invocations:
            for suffix in ("alpha", "beta"):
                projection_id = "linear_attention." + suffix
                segments = resolve_weight_projection(layer.metadata, projection_id)
                assert segments is not None
                segment, = segments
                if (segment.k, segment.n, segment.tp_shard_axis) != (
                    layer.hidden_size, geometry.value_heads, "n"
                ):
                    raise ValueError(projection_id + " must map hidden width to value heads, sharded on n")
                control_workload = _layer_gemm(
                    layer, token_batch, layer.hidden_size, geometry.value_heads,
                    name="linear_{}_projection_tp".format(suffix),
                    projection_id=projection_id,
                    projection_tp_degree=plan.tp_degree,
                    projection_tp_rank=rank.tp_rank,
                    projection_allow_padding=plan.allow_padding,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                )
                # Physical F32 storage is separate from the inherited compute-rate contract.
                control_workload = replace(
                    control_workload,
                    activation_storage_bytes=4 * control_workload.m * control_workload.k,
                    output_bits=32,
                )
                control_projections.append(_add_rank_gemm(
                    builder, scenario, router, plan, rank, control_workload,
                    target, "{}.rank{:03d}.linear_{}_projection".format(prefix, rank.rank, suffix),
                    (norm_apply,), weight_tensor_id=linear_weight_tensor_id,
                    activation_source_component_id=norm_component,
                    metadata={
                        **metadata, "linear_op": suffix + "_projection",
                        "projection_id": projection_id,
                        "projection_tp_degree": plan.tp_degree,
                        "projection_tp_rank": rank.tp_rank,
                        "projection_allow_padding": plan.allow_padding,
                        "control_transform_timing_completeness": "not_added_in_projection_boundary_candidate",
                        "runtime_input_storage_bits": 32,
                        "runtime_output_storage_bits": 32,
                        "runtime_output_storage_bytes": control_workload.output_bytes,
                        "matrix_compute_precision_policy": "inherited_analytical_contract",
                    },
                ))
                control_component = builder.rank_value_component((control_projections[-1],), rank.rank)
                if control_component is None:
                    raise AssertionError("physical control projection output location is missing")
                control_outputs.append((control_component, control_workload.output_bytes))
        state_read = _add_linear_state_read(
            builder,
            scenario,
            router,
            plan,
            rank,
            layer,
            dependencies,
            name="{}.rank{:03d}.linear_state".format(prefix, rank.rank),
            target_component_id=recurrent_target,
            storage_owner_component_id=target,
            runtime=state_runtime,
        )
        ready = _add_join(
            builder,
            "{}.rank{:03d}.linear_inputs_ready".format(prefix, rank.rank),
            (projection, gate_projection, state_read),
            metadata=metadata,
        )
        conv_channels = projected_shard.local_size
        conv_output_elements = max(1, token_batch * conv_channels)
        conv, conv_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=max(
                    conv_output_elements,
                    conv_output_elements * geometry.conv_kernel_size,
                ),
                output_elements=conv_output_elements,
                operations_per_combine=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                name="linear_local_conv",
            ),
            "{}.linear_attention.local_conv".format(layer.layer_id),
            "{}.rank{:03d}.linear_local_conv".format(prefix, rank.rank),
            (ready,),
            source_component_id=rank.component_id,
            target_component_id=recurrent_target,
            fallback_keys=("{}.linear_attention".format(layer.layer_id),),
            metadata={**metadata, "linear_op": "local_conv"},
        )
        scan, scan_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=max(1, token_batch * recurrent_elements),
                operations_per_element=6,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                read_storage_bytes=(
                    _activation_bytes(layer, 2 * max(1, token_batch * recurrent_elements), scenario=scenario)
                    + sum(byte_count for _, byte_count in control_outputs)
                    if control_outputs else None
                ),
                name="linear_recurrent_scan",
            ),
            "{}.linear_attention.state_update".format(layer.layer_id),
            "{}.rank{:03d}.linear_recurrent_scan".format(prefix, rank.rank),
            (conv, *control_projections),
            source_component_id=conv_component,
            target_component_id=recurrent_target,
            input_component_bytes=control_outputs or None,
            fallback_keys=("{}.linear_state_update".format(layer.layer_id),),
            metadata={**metadata, "linear_op": "scan_recurrent_update"},
        )
        gate_elements = max(1, token_batch * value_shard.local_size)
        gate_reduce, gate_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=gate_elements,
                output_elements=max(1, token_batch),
                operations_per_combine=2,
                fixed_operations=max(1, token_batch),
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                name="linear_gate_norm_reduce",
            ),
            "{}.linear_attention.gate_norm.reduce".format(layer.layer_id),
            "{}.rank{:03d}.linear_gate_norm_reduce".format(prefix, rank.rank),
            (scan,),
            source_component_id=scan_component,
            target_component_id=recurrent_target,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={
                **metadata,
                "linear_op": "gate_norm_reduce",
                "output_gate": geometry.output_gate,
                "gate_activation": geometry.gate_activation,
            },
        )
        gate_norm, gate_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=gate_elements,
                operations_per_element=8,
                fixed_transcendental_operations=max(1, token_batch),
                input_count=2 if geometry.output_gate else 1,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=5,
                working_set_bytes=_activation_bytes(
                    layer, gate_elements * 3,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="linear_gate_norm_apply",
            ),
            "{}.linear_attention.gate_norm.apply".format(layer.layer_id),
            "{}.rank{:03d}.linear_gate_norm_apply".format(prefix, rank.rank),
            (gate_reduce,),
            source_component_id=gate_component,
            target_component_id=recurrent_target,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={
                **metadata,
                "linear_op": "gate_norm_apply",
                "output_gate": geometry.output_gate,
                "gate_activation": geometry.gate_activation,
            },
        )
        output = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            _layer_gemm(
                layer,
                token_batch,
                value_shard.local_size,
                layer.hidden_size,
                name="linear_output_projection_tp",
                projection_id="linear_attention.output",
                projection_tp_degree=plan.tp_degree,
                projection_tp_rank=rank.tp_rank,
                projection_allow_padding=plan.allow_padding,
                f32_storage=_f32_hidden_storage_enabled(scenario),
            ),
            target,
            "{}.rank{:03d}.linear_output_projection".format(prefix, rank.rank),
            (gate_norm,),
            weight_tensor_id="{}.linear_attention_weights".format(
                layer.layer_id
            ),
            activation_source_component_id=gate_component,
            keep_output_on_target=True,
            metadata={
                **metadata,
                "linear_op": "output_projection",
                "projection_id": "linear_attention.output",
                "projection_tp_degree": plan.tp_degree,
                "projection_tp_rank": rank.tp_rank,
                "projection_allow_padding": plan.allow_padding,
            },
        )
        output_component = (
            builder.rank_value_component((output,), rank.rank)
            or (
                rank.component_id
                if _is_cim(_component(scenario, target))
                else target
            )
        )
        residual, residual_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=hidden_elements,
                operations_per_element=1,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                name="linear_attention_residual",
            ),
            "{}.linear_attention.residual".format(layer.layer_id),
            "{}.rank{:03d}.linear_attention_residual".format(prefix, rank.rank),
            (output,),
            source_component_id=output_component,
            target_component_id=recurrent_target,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**metadata, "linear_op": "residual"},
        )
        output = _return_rank_value(
            builder,
            router,
            plan,
            rank,
            residual,
            residual_component,
            _activation_bytes(layer, hidden_elements, scenario=scenario),
            name="{}.rank{:03d}.linear_attention_residual.output_to_gpu".format(
                prefix, rank.rank
            ),
            metadata=metadata,
        )
        scan_for_state = _return_rank_value(
            builder,
            router,
            plan,
            rank,
            scan,
            scan_component,
            max(1, state_bytes),
            name="{}.rank{:03d}.linear_state_update.output_to_gpu".format(
                prefix, rank.rank
            ),
            metadata=metadata,
        )
        if state_runtime.uses_temporary_state:
            speculative_state_write = _add_linear_state_materialize(
                builder,
                scenario,
                router,
                plan,
                rank,
                layer,
                (scan_for_state,),
                name="{}.rank{:03d}.linear_state".format(prefix, rank.rank),
                source_component_id=scan_component,
                target_component_id=target,
                storage_owner_component_id=target,
                runtime=state_runtime,
            )
            if state_runtime.commits_snapshot:
                speculative_state_write = _add_linear_state_write(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    layer,
                    (speculative_state_write,),
                    name="{}.rank{:03d}.linear_state".format(
                        prefix, rank.rank
                    ),
                    source_component_id=scan_component,
                    target_component_id=target,
                    storage_owner_component_id=target,
                    runtime=state_runtime,
                )
            if state_runtime.release_temporary:
                speculative_state_write = _add_linear_state_release(
                    builder,
                    plan,
                    rank,
                    layer,
                    (speculative_state_write,),
                    name="{}.rank{:03d}.linear_state".format(
                        prefix, rank.rank
                    ),
                    runtime=state_runtime,
                )
            state_writes = [speculative_state_write]
            if state_runtime.has_persistent_updates:
                persistent_runtime = _LinearStateRuntimeSemantics(
                    mode="persistent_update",
                    read_source="committed",
                    persistent_update_lane_ids=(
                        state_runtime.persistent_update_lane_ids
                    ),
                )
                state_writes.append(
                    _add_linear_state_write(
                        builder,
                        scenario,
                        router,
                        plan,
                        rank,
                        layer,
                        (scan_for_state,),
                        name=(
                            "{}.rank{:03d}.linear_state.persistent"
                        ).format(prefix, rank.rank),
                        source_component_id=scan_component,
                        target_component_id=target,
                        storage_owner_component_id=target,
                        runtime=persistent_runtime,
                    )
                )
            state_write = (
                state_writes[0]
                if len(state_writes) == 1
                else _add_join(
                    builder,
                    "{}.rank{:03d}.linear_state.mixed_complete".format(
                        prefix, rank.rank
                    ),
                    tuple(state_writes),
                    metadata={
                        **metadata,
                        **state_runtime.audit_metadata(),
                    },
                )
            )
        else:
            state_write = _add_linear_state_write(
                builder,
                scenario,
                router,
                plan,
                rank,
                layer,
                (scan_for_state,),
                name="{}.rank{:03d}.linear_state".format(prefix, rank.rank),
                target_component_id=target,
                storage_owner_component_id=target,
                runtime=state_runtime,
            )
        rank_ends.append(
            _add_join(
                builder,
                "{}.rank{:03d}.linear_complete".format(prefix, rank.rank),
                (output, state_write),
                metadata=metadata,
            )
        )
    return _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".linear_all_reduce",
        "all_reduce",
        ranks,
        _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
        rank_ends,
        tensor_elements=token_batch * layer.hidden_size,
        element_bits=_activation_storage_bits(layer, scenario),
        metadata={
            "phase": phase,
            "layer_id": layer.layer_id,
            "stage": stage,
            "sequence_mixer": "linear_attention",
            "coverage_component": "linear_attention",
            "context_scaling": scaling,
            "state_bytes": state_bytes * plan.tp_degree,
        },
    )


def _compile_parallel_dense_mlp(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    token_batch: int,
    prefix: str,
    dependencies: Sequence[str],
) -> str:
    """Compile or replay the context-independent dense-MLP task segment.

    Attention and linear-state lowering remain context-sensitive and always
    execute normally.  The dense MLP contract has no context argument: for an
    immutable scenario its task demands depend only on the layer, physical
    token batch, and incoming rank-value placement.  Replaying that exact
    segment avoids rebuilding thousands of identical tasks across MTP
    verifier positions and rounds while rebinding request-local identities.
    """

    context = _active_compilation_context(scenario)
    if context is None:
        return _compile_parallel_dense_mlp_uncached(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch,
            prefix,
            dependencies,
        )
    cache = _task_segment_cache(context, ("dense_mlp_task_segment_cache_v1",))
    if cache is None:
        return _compile_parallel_dense_mlp_uncached(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch,
            prefix,
            dependencies,
        )
    ranks = plan.tp_group(plan.stage_for_layer(layer), 0)
    input_components = tuple(
        builder.rank_value_component(dependencies, rank.rank)
        for rank in ranks
    )
    cache_key = (
        str(layer.layer_id),
        int(token_batch),
        input_components,
        len(tuple(dependencies)),
        builder._last_coherent_dma_task is not None,
    )
    template = _task_segment_cache_get(cache, cache_key)
    if template is not None:
        replayed = template.replay(
            builder,
            prefix=prefix,
            dependencies=dependencies,
        )
        if replayed is not None:
            return replayed

    first_task_index = len(builder.tasks)
    initial_counter = builder.counter
    initial_previous = builder.previous
    initial_dma = builder._last_coherent_dma_task
    terminal = _compile_parallel_dense_mlp_uncached(
        builder,
        scenario,
        plan,
        router,
        layer,
        token_batch,
        prefix,
        dependencies,
    )
    captured = _TaskSegmentTemplate.capture(
        builder,
        first_task_index=first_task_index,
        source_prefix=prefix,
        source_counter_before=initial_counter,
        source_dependencies=dependencies,
        source_initial_previous=initial_previous,
        source_initial_dma=initial_dma,
        terminal_task_id=terminal,
    )
    if captured is not None:
        _task_segment_cache_put(context, cache, cache_key, captured)
    return terminal


def _compile_parallel_dense_mlp_uncached(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    token_batch: int,
    prefix: str,
    dependencies: Sequence[str],
) -> str:
    stage = plan.stage_for_layer(layer)
    ranks = plan.tp_group(stage, 0)
    gated_mlp = bool(getattr(layer, "gated_mlp", True))
    up_shard = shard_extent(
        (2 if gated_mlp else 1) * layer.intermediate_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    down_shard = shard_extent(
        layer.intermediate_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    ends = []
    for rank in ranks:
        activation_bits = _activation_storage_bits(layer, scenario)
        hidden_elements = max(1, token_batch * layer.hidden_size)
        ffn_meta = {
            "layer_id": layer.layer_id,
            "stage": stage,
            "ffn_path": "dense",
            "coverage_component": "dense_ffn",
        }
        residual_norm_fused, residual_norm_audit = (
            _residual_norm_fusion_decision(
                scenario, router, rank, layer, hidden_elements
            )
        )
        if residual_norm_fused:
            element_bytes = max(1, int(math.ceil(activation_bits / 8.0)))
            norm_apply = _add_rank_tensor_kernel(
                builder,
                scenario,
                router,
                plan,
                rank,
                TensorKernelWorkload(
                    operations=5 * hidden_elements,
                    transcendental_operations=max(1, token_batch),
                    read_bytes=hidden_elements * element_bytes * 2,
                    write_bytes=hidden_elements * element_bytes,
                    dependency_depth=max(
                        4,
                        int(
                            math.ceil(
                                math.log2(
                                    max(1, hidden_elements // token_batch)
                                )
                            )
                        )
                        + 3,
                    ),
                    working_set_bytes=hidden_elements * element_bytes * 3,
                    reuse_factor=2.0,
                    name="fused_residual_rmsnorm",
                ),
                "{}.rank{:03d}.fused_residual_norm".format(
                    prefix, rank.rank
                ),
                dependencies,
                metadata={
                    **ffn_meta,
                    **residual_norm_audit,
                    "event_kind": "fused_residual_norm",
                    "norm_kind": "rmsnorm",
                },
            )
            norm_component = rank.component_id
        else:
            norm_reduce, norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.REDUCTION,
                ReductionWorkload(
                    input_elements=hidden_elements,
                    output_elements=max(1, token_batch),
                    operations_per_combine=2,
                    fixed_operations=max(1, token_batch),
                    input_bits=activation_bits,
                    output_bits=max(16, activation_bits),
                    dependency_depth=max(
                        1,
                        int(
                            math.ceil(
                                math.log2(
                                    max(1, hidden_elements // token_batch)
                                )
                            )
                        ),
                    ),
                    working_set_bytes=_activation_bytes(
                        layer, hidden_elements,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="post_attention_norm_reduce",
                ),
                "{}.post_attention_norm.reduce".format(layer.layer_id),
                "{}.rank{:03d}.post_attention_norm_reduce".format(
                    prefix, rank.rank
                ),
                dependencies,
                source_component_id=rank.component_id,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **ffn_meta,
                    **residual_norm_audit,
                    "event_kind": "post_attention_norm_reduce",
                },
            )
            norm_apply, norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=hidden_elements,
                    operations_per_element=2,
                    fixed_operations=max(1, token_batch),
                    fixed_transcendental_operations=max(1, token_batch),
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=4,
                    working_set_bytes=_activation_bytes(
                        layer, hidden_elements * 3,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="post_attention_norm_apply",
                ),
                "{}.post_attention_norm.apply".format(layer.layer_id),
                "{}.rank{:03d}.post_attention_norm_apply".format(
                    prefix, rank.rank
                ),
                (norm_reduce,),
                source_component_id=norm_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **ffn_meta,
                    **residual_norm_audit,
                    "event_kind": "post_attention_norm_apply",
                },
            )
        target = _parallel_target(scenario, layer, "mlp", rank)
        activation_elements = max(
            1, token_batch * down_shard.local_size
        )
        split_mlp = gated_mlp and _declared_physical_projections(
            scenario, layer, ("mlp.gate", "mlp.up"),
            combined_projection_id="mlp.up_gate",
        )
        up_workload = _layer_gemm(
            layer,
            token_batch,
            layer.hidden_size,
            up_shard.local_size,
            name="mlp_up_tp",
            projection_id=("mlp.up_gate" if gated_mlp else "mlp.up"),
            projection_tp_degree=plan.tp_degree,
            projection_tp_rank=rank.tp_rank,
            projection_allow_padding=plan.allow_padding,
            f32_storage=_f32_hidden_storage_enabled(scenario),
        )
        activation_target = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.ELEMENTWISE,
            "{}.mlp.activation".format(layer.layer_id),
            fallback_keys=("{}.mlp".format(layer.layer_id),),
        )
        activation_fused, activation_audit = _same_rank_gpu_fusion_decision(
            scenario,
            "gemm_epilogue_activation",
            rank,
            up_workload.output_bytes,
            (
                ("{}.mlp".format(layer.layer_id), target),
                (
                    "{}.mlp.activation".format(layer.layer_id),
                    activation_target,
                ),
            ),
        )
        if not gated_mlp:
            activation_fused = False
            activation_audit = {"fusion_enabled": False, "fusion_decision": "ungated_ffn"}
        activation_audit = dict(activation_audit)
        if split_mlp and token_batch != 1:
            activation_fused = False
            activation_audit.update(
                {
                    "fusion_enabled": False,
                    "fusion_decision": "backend_requires_single_row",
                    "fusion_physical_m": token_batch,
                }
            )
        if split_mlp and activation_fused:
            cc = _component(scenario, target).metadata.get("cuda_compute_capability")
            if not isinstance(cc, int) or isinstance(cc, bool) or cc <= 600:
                activation_fused = False
                activation_audit.update(
                    {"fusion_enabled": False, "fusion_decision": "backend_architecture_unknown_or_ineligible"}
                )
        if split_mlp and activation_fused:
            gate_descriptor = _materialize_weight_projection(layer, "mlp.gate")
            up_descriptor = _materialize_weight_projection(layer, "mlp.up")
            assert gate_descriptor is not None and up_descriptor is not None
            gate_segment, = gate_descriptor.segments
            up_segment, = up_descriptor.segments
            if (gate_descriptor.k, gate_descriptor.n, gate_segment.segment.artifact_spec,
                    gate_segment.segment.tp_shard_axis) != (
                    up_descriptor.k, up_descriptor.n, up_segment.segment.artifact_spec,
                    up_segment.segment.tp_shard_axis):
                activation_fused = False
                activation_audit.update(
                    {"fusion_enabled": False, "fusion_decision": "backend_gate_up_layout_mismatch"}
                )
        activation_audit.update(
            {
                "fusion_parent_target": target,
                "fusion_activation_target": activation_target,
            }
        )
        if activation_fused:
            up_workload = _append_gemm_epilogue(
                up_workload,
                operations=5 * activation_elements,
                transcendental_operations=activation_elements,
                output_elements=activation_elements,
                name="swiglu",
            )
        if not split_mlp or activation_fused:
            up = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                up_workload,
                target,
                "{}.rank{:03d}.mlp_up_gate".format(prefix, rank.rank),
                (norm_apply,),
                weight_tensor_id="{}.mlp_weights".format(layer.layer_id),
                activation_source_component_id=norm_component,
                keep_output_on_target=True,
                metadata={
                    **ffn_meta,
                    **activation_audit,
                    "projection_id": ("mlp.up_gate" if gated_mlp else "mlp.up"),
                    "projection_tp_degree": plan.tp_degree,
                    "projection_tp_rank": rank.tp_rank,
                    "projection_allow_padding": plan.allow_padding,
                },
            )
        else:
            split_mlp_tasks = []
            intermediate_shard = shard_extent(
                layer.intermediate_size,
                plan.tp_degree,
                0,
                allow_padding=plan.allow_padding,
            )
            for projection_suffix in ("gate", "up"):
                workload = _layer_gemm(
                    layer,
                    token_batch,
                    layer.hidden_size,
                    intermediate_shard.local_size,
                    name="mlp_{}_tp".format(projection_suffix),
                    projection_id="mlp.{}".format(projection_suffix),
                    projection_tp_degree=plan.tp_degree,
                    projection_tp_rank=rank.tp_rank,
                    projection_allow_padding=plan.allow_padding,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                )
                workload = replace(
                    workload,
                    output_storage_bytes=_activation_bytes(
                        layer, token_batch * intermediate_shard.local_size,
                        scenario=scenario,
                    ),
                )
                split_mlp_tasks.append(
                    _add_rank_gemm(
                        builder,
                        scenario,
                        router,
                        plan,
                        rank,
                        workload,
                        target,
                        "{}.rank{:03d}.mlp_{}".format(prefix, rank.rank, projection_suffix),
                        # llama.cpp's CPU graph emits ffn_gate followed by
                        # ffn_up on the same pipeline.  Preserve that
                        # physical order when the explicit split capability
                        # is active; GPU callers retain the historical
                        # parallel split semantics.
                        (
                            (split_mlp_tasks[-1],)
                            if split_mlp_tasks
                            and _kind(_component(scenario, target)) == "cpu"
                            else (norm_apply,)
                        ),
                        weight_tensor_id="{}.mlp_weights".format(layer.layer_id),
                        activation_source_component_id=norm_component,
                        keep_output_on_target=True,
                        metadata={
                            **ffn_meta,
                            **activation_audit,
                            "projection_id": "mlp.{}".format(projection_suffix),
                            "physical_projection": projection_suffix,
                            "projection_tp_degree": plan.tp_degree,
                            "projection_tp_rank": rank.tp_rank,
                            "projection_allow_padding": plan.allow_padding,
                        },
                    )
                )
            up = _add_join(
                builder,
                "{}.rank{:03d}.mlp_projections_ready".format(prefix, rank.rank),
                tuple(split_mlp_tasks),
                metadata={
                    **ffn_meta,
                    **activation_audit,
                    "event_kind": "mlp_projection_join",
                    "physical_projections": ("gate", "up"),
                },
            )
        up_component = (
            rank.component_id
            if _is_cim(_component(scenario, target))
            else target
        )
        if not gated_mlp:
            activation, activation_component = up, up_component
        elif activation_fused:
            activation, activation_component = up, up_component
        else:
            activation, activation_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=activation_elements,
                    operations_per_element=5,
                    transcendental_ops_per_element=1,
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=5,
                    working_set_bytes=_activation_bytes(
                        layer, activation_elements * 3,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="mlp_swiglu_activation",
                ),
                "{}.mlp.activation".format(layer.layer_id),
                "{}.rank{:03d}.mlp_activation".format(prefix, rank.rank),
                (up,),
                source_component_id=up_component,
                fallback_keys=("{}.mlp".format(layer.layer_id),),
                metadata={
                    **ffn_meta,
                    **activation_audit,
                    "event_kind": "mlp_activation",
                    "activation_function": "swiglu",
                },
            )
        down = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                _layer_gemm(
                    layer,
                    token_batch,
                    down_shard.local_size,
                    layer.hidden_size,
                    name="mlp_down_tp",
                    projection_id="mlp.down",
                    projection_tp_degree=plan.tp_degree,
                    projection_tp_rank=rank.tp_rank,
                    projection_allow_padding=plan.allow_padding,
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                ),
                target,
                "{}.rank{:03d}.mlp_down".format(prefix, rank.rank),
                (activation,),
                weight_tensor_id="{}.mlp_weights".format(layer.layer_id),
                activation_source_component_id=activation_component,
                keep_output_on_target=True,
                metadata={
                    **ffn_meta,
                    "projection_id": "mlp.down",
                    "projection_tp_degree": plan.tp_degree,
                    "projection_tp_rank": rank.tp_rank,
                    "projection_allow_padding": plan.allow_padding,
                },
        )
        down_component = (
            rank.component_id
            if _is_cim(_component(scenario, target))
            else target
        )
        residual, residual_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=hidden_elements,
                operations_per_element=1,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                name="mlp_residual",
            ),
            "{}.mlp.residual".format(layer.layer_id),
            "{}.rank{:03d}.mlp_residual".format(prefix, rank.rank),
            (down,),
            source_component_id=down_component,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={**ffn_meta, "event_kind": "mlp_residual"},
        )
        ends.append(
            _return_rank_value(
                builder,
                router,
                plan,
                rank,
                residual,
                residual_component,
                _activation_bytes(layer, hidden_elements, scenario=scenario),
                name="{}.rank{:03d}.mlp_residual.output_to_gpu".format(
                    prefix, rank.rank
                ),
                metadata=ffn_meta,
            )
        )
    return _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".mlp_all_reduce",
        "all_reduce",
        ranks,
        _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
        ends,
        tensor_elements=token_batch * layer.hidden_size,
        element_bits=_activation_storage_bits(layer, scenario),
        metadata={
            "layer_id": layer.layer_id,
            "stage": stage,
            "ffn_path": "dense",
            "coverage_component": "dense_ffn",
        },
    )


def _compile_parallel_moe(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    token_batch: int,
    prefix: str,
    dependencies: Sequence[str],
) -> str:
    stage = plan.stage_for_layer(layer)
    tp_ranks = plan.tp_group(stage, 0)
    norm_ends: List[str] = []
    for rank in tp_ranks:
        activation_bits = _activation_storage_bits(layer, scenario)
        hidden_elements = max(1, token_batch * layer.hidden_size)
        norm_meta = {
            "layer_id": layer.layer_id,
            "stage": stage,
            "ffn_path": "routed",
            "coverage_component": "routed_expert",
        }
        residual_norm_fused, residual_norm_audit = (
            _residual_norm_fusion_decision(
                scenario, router, rank, layer, hidden_elements
            )
        )
        if residual_norm_fused:
            element_bytes = max(1, int(math.ceil(activation_bits / 8.0)))
            norm_end = _add_rank_tensor_kernel(
                builder,
                scenario,
                router,
                plan,
                rank,
                TensorKernelWorkload(
                    operations=5 * hidden_elements,
                    transcendental_operations=max(1, token_batch),
                    read_bytes=hidden_elements * element_bytes * 2,
                    write_bytes=hidden_elements * element_bytes,
                    dependency_depth=max(
                        4,
                        int(
                            math.ceil(
                                math.log2(
                                    max(1, hidden_elements // token_batch)
                                )
                            )
                        )
                        + 3,
                    ),
                    working_set_bytes=hidden_elements * element_bytes * 3,
                    reuse_factor=2.0,
                    name="fused_residual_rmsnorm",
                ),
                "{}.rank{:03d}.fused_residual_norm".format(
                    prefix, rank.rank
                ),
                dependencies,
                metadata={
                    **norm_meta,
                    **residual_norm_audit,
                    "event_kind": "fused_residual_norm",
                    "norm_kind": "rmsnorm",
                },
            )
        else:
            norm_reduce, norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.REDUCTION,
                ReductionWorkload(
                    input_elements=hidden_elements,
                    output_elements=max(1, token_batch),
                    operations_per_combine=2,
                    fixed_operations=max(1, token_batch),
                    input_bits=activation_bits,
                    output_bits=max(16, activation_bits),
                    dependency_depth=max(
                        1,
                        int(
                            math.ceil(
                                math.log2(
                                    max(1, hidden_elements // token_batch)
                                )
                            )
                        ),
                    ),
                    working_set_bytes=_activation_bytes(
                        layer, hidden_elements,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="moe_post_attention_norm_reduce",
                ),
                "{}.post_attention_norm.reduce".format(layer.layer_id),
                "{}.rank{:03d}.post_attention_norm_reduce".format(
                    prefix, rank.rank
                ),
                dependencies,
                source_component_id=rank.component_id,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **norm_meta,
                    **residual_norm_audit,
                    "event_kind": "post_attention_norm_reduce",
                },
            )
            norm_end, _norm_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=hidden_elements,
                    operations_per_element=2,
                    fixed_operations=max(1, token_batch),
                    fixed_transcendental_operations=max(1, token_batch),
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=4,
                    working_set_bytes=_activation_bytes(
                        layer, hidden_elements * 3,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="moe_post_attention_norm_apply",
                ),
                "{}.post_attention_norm.apply".format(layer.layer_id),
                "{}.rank{:03d}.post_attention_norm_apply".format(
                    prefix, rank.rank
                ),
                (norm_reduce,),
                source_component_id=norm_component,
                fallback_keys=("{}.norm".format(layer.layer_id),),
                metadata={
                    **norm_meta,
                    **residual_norm_audit,
                    "event_kind": "post_attention_norm_apply",
                },
            )
        norm_ends.append(norm_end)
    dependencies = (
        _add_join(
            builder,
            prefix + ".post_attention_norm.complete",
            norm_ends,
            metadata={"event_kind": "post_attention_norm_barrier"},
        ),
    )
    shared_end = ""
    if layer.has_shared_expert:
        shared_end = _compile_parallel_shared_expert(
            builder,
            scenario,
            plan,
            router,
            layer,
            token_batch,
            prefix,
            dependencies,
        )
    router_ends: List[str] = []
    expert_score_shard = shard_extent(
        layer.num_experts,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    for rank in tp_ranks:
        activation_bits = _activation_storage_bits(layer, scenario)
        router_meta = {
            "layer_id": layer.layer_id,
            "stage": stage,
            "ffn_path": "routed",
            "coverage_component": "routed_expert",
        }
        router_target = _parallel_named_target(
            scenario, "{}.router".format(layer.layer_id), rank
        )
        router_gemm = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                _layer_gemm(
                    layer,
                    token_batch,
                    layer.hidden_size,
                    expert_score_shard.local_size,
                    name="moe_router_tp",
                    f32_storage=_f32_hidden_storage_enabled(scenario),
                ),
                router_target,
                "{}.rank{:03d}.moe_router".format(prefix, rank.rank),
                dependencies,
                weight_tensor_id="{}.router_weights".format(layer.layer_id),
                keep_output_on_target=True,
                metadata=router_meta,
        )
        router_component = (
            rank.component_id
            if _is_cim(_component(scenario, router_target))
            else router_target
        )
        score_elements = max(
            1, token_batch * expert_score_shard.local_size
        )
        router_reduce, router_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=score_elements,
                output_elements=max(1, token_batch),
                operations_per_combine=2,
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                dependency_depth=max(
                    1,
                    int(
                        math.ceil(
                            math.log2(
                                max(1, expert_score_shard.local_size)
                            )
                        )
                    ),
                ),
                working_set_bytes=_activation_bytes(layer, score_elements, scenario=scenario),
                reuse_factor=2.0,
                name="moe_router_softmax_reduce_max_sum",
            ),
            "{}.router.softmax.reduce".format(layer.layer_id),
            "{}.rank{:03d}.router_softmax_reduce".format(prefix, rank.rank),
            (router_gemm,),
            source_component_id=router_component,
            fallback_keys=("{}.router".format(layer.layer_id),),
            metadata={**router_meta, "event_kind": "router_softmax_reduce"},
        )
        router_softmax, router_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=score_elements,
                operations_per_element=2,
                transcendental_ops_per_element=1,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=3,
                working_set_bytes=_activation_bytes(
                    layer, score_elements * 2,
                    scenario=scenario,
                ),
                reuse_factor=2.0,
                name="moe_router_softmax_sub_exp_normalize",
            ),
            "{}.router.softmax.normalize".format(layer.layer_id),
            "{}.rank{:03d}.router_softmax_normalize".format(prefix, rank.rank),
            (router_reduce,),
            source_component_id=router_component,
            fallback_keys=("{}.router".format(layer.layer_id),),
            metadata={**router_meta, "event_kind": "router_softmax_normalize"},
        )
        router_topk, router_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=score_elements,
                output_elements=max(
                    1, token_batch * min(
                        layer.experts_per_token,
                        expert_score_shard.local_size,
                    )
                ),
                operations_per_combine=1,
                input_bits=activation_bits,
                output_bits=activation_bits,
                name="moe_router_topk",
            ),
            "{}.router.topk".format(layer.layer_id),
            "{}.rank{:03d}.router_topk".format(prefix, rank.rank),
            (router_softmax,),
            source_component_id=router_component,
            fallback_keys=("{}.router".format(layer.layer_id),),
            metadata={**router_meta, "event_kind": "router_topk"},
        )
        router_ends.append(
            _return_rank_value(
                builder,
                router,
                plan,
                rank,
                router_topk,
                router_component,
                _activation_bytes(layer, score_elements, scenario=scenario),
                name="{}.rank{:03d}.router.output_to_gpu".format(
                    prefix, rank.rank
                ),
                metadata=router_meta,
            )
        )
    router_ready = _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".router_all_gather",
        "all_gather",
        tp_ranks,
        _activation_bytes(layer, token_batch * layer.num_experts, scenario=scenario),
        router_ends,
        metadata={"layer_id": layer.layer_id, "stage": stage},
    )
    routed_tokens = token_batch * layer.experts_per_token
    dispatch_ends: List[str] = []
    for tp_rank in range(plan.tp_degree):
        ep_group = plan.ep_group(stage, tp_rank)
        dispatch_ends.append(
            _add_collective_tasks(
                builder,
                scenario,
                router,
                plan,
                "{}.tp{:02d}.expert_dispatch".format(prefix, tp_rank),
                "all_to_all",
                ep_group,
                _activation_bytes(
                    layer,
                    routed_tokens
                    * shard_extent(
                        layer.hidden_size,
                        plan.tp_degree,
                        tp_rank,
                        allow_padding=plan.allow_padding,
                    ).local_size,
                    scenario=scenario,
                ),
                (router_ready,),
                metadata={"layer_id": layer.layer_id, "stage": stage},
            )
        )
    dispatch_ready = _add_join(
        builder,
        prefix + ".expert_dispatch.complete",
        dispatch_ends,
        metadata={"event_kind": "moe_dispatch_barrier"},
    )
    base_experts, extra_experts = divmod(layer.num_experts, plan.ep_degree)
    base_tokens, extra_tokens = divmod(routed_tokens, layer.num_experts)
    expert_ends: List[str] = []
    for rank in plan.ranks_for_stage(stage):
        local_experts = base_experts + (1 if rank.ep_rank < extra_experts else 0)
        if local_experts <= 0:
            expert_ends.append(
                _add_join(
                    builder,
                    "{}.rank{:03d}.experts_idle".format(prefix, rank.rank),
                    (dispatch_ready,),
                )
            )
            continue
        target = _parallel_target(scenario, layer, "experts", rank)
        local_intermediate = shard_extent(
            layer.intermediate_size,
            plan.tp_degree,
            rank.tp_rank,
            allow_padding=plan.allow_padding,
        ).local_size
        expert_start = rank.ep_rank * base_experts + min(
            rank.ep_rank, extra_experts
        )
        rank_active = False
        for expert_index in range(expert_start, expert_start + local_experts):
            expert_tokens = base_tokens + (
                1 if expert_index < extra_tokens else 0
            )
            if expert_tokens <= 0:
                continue
            rank_active = True
            expert_metadata = {
                "layer_id": layer.layer_id,
                "stage": stage,
                "expert_index": expert_index,
                "local_experts": local_experts,
                "expert_tokens": expert_tokens,
                "routed_tokens": routed_tokens,
                "routing_model": "deterministic_balanced_histogram",
                "ffn_path": "routed",
                "coverage_component": "routed_expert",
            }
            activation_elements = max(
                1, expert_tokens * local_intermediate
            )
            expert_up_workload = _layer_gemm(
                layer,
                expert_tokens,
                layer.hidden_size,
                2 * local_intermediate,
                name="moe_expert_up_tp_ep",
                f32_storage=_f32_hidden_storage_enabled(scenario),
            )
            expert_activation_target = _primitive_target(
                scenario,
                router,
                rank,
                OperatorClass.ELEMENTWISE,
                "{}.experts.activation".format(layer.layer_id),
                fallback_keys=("{}.experts".format(layer.layer_id),),
            )
            expert_activation_fused, expert_activation_audit = (
                _same_rank_gpu_fusion_decision(
                    scenario,
                    "gemm_epilogue_activation",
                    rank,
                    expert_up_workload.output_bytes,
                    (
                        ("{}.experts".format(layer.layer_id), target),
                        (
                            "{}.experts.activation".format(layer.layer_id),
                            expert_activation_target,
                        ),
                    ),
                )
            )
            expert_activation_audit = dict(expert_activation_audit)
            expert_activation_audit.update(
                {
                    "fusion_parent_target": target,
                    "fusion_activation_target": expert_activation_target,
                }
            )
            if expert_activation_fused:
                expert_up_workload = _append_gemm_epilogue(
                    expert_up_workload,
                    operations=5 * activation_elements,
                    transcendental_operations=activation_elements,
                    output_elements=activation_elements,
                    name="swiglu",
                )
            up = _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                expert_up_workload,
                target,
                "{}.rank{:03d}.expert{:04d}.up_gate".format(
                    prefix, rank.rank, expert_index
                ),
                (dispatch_ready,),
                weight_tensor_id="{}.expert_weights".format(layer.layer_id),
                keep_output_on_target=True,
                metadata={
                    **expert_metadata,
                    **expert_activation_audit,
                },
            )
            up_component = (
                rank.component_id
                if _is_cim(_component(scenario, target))
                else target
            )
            activation_bits = _activation_storage_bits(layer, scenario)
            if expert_activation_fused:
                activation, activation_component = up, up_component
            else:
                activation, activation_component = _add_rank_primitive(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    OperatorClass.ELEMENTWISE,
                    ElementwiseWorkload(
                        elements=activation_elements,
                        operations_per_element=5,
                        transcendental_ops_per_element=1,
                        input_count=2,
                        input_bits=activation_bits,
                        output_bits=activation_bits,
                        dependency_depth=5,
                        working_set_bytes=_activation_bytes(
                            layer, activation_elements * 3,
                            scenario=scenario,
                        ),
                        reuse_factor=2.0,
                        name="moe_expert_swiglu_activation",
                    ),
                    "{}.experts.activation".format(layer.layer_id),
                    "{}.rank{:03d}.expert{:04d}.activation".format(
                        prefix, rank.rank, expert_index
                    ),
                    (up,),
                    source_component_id=up_component,
                    fallback_keys=("{}.experts".format(layer.layer_id),),
                    metadata={
                        **expert_metadata,
                        **expert_activation_audit,
                        "event_kind": "expert_activation",
                        "activation_function": "swiglu",
                    },
                )
            expert_ends.append(
                _add_rank_gemm(
                    builder,
                    scenario,
                    router,
                    plan,
                    rank,
                    _layer_gemm(
                        layer,
                        expert_tokens,
                        local_intermediate,
                        layer.hidden_size,
                        name="moe_expert_down_tp_ep",
                        f32_storage=_f32_hidden_storage_enabled(scenario),
                    ),
                    target,
                    "{}.rank{:03d}.expert{:04d}.down".format(
                        prefix, rank.rank, expert_index
                    ),
                    (activation,),
                    weight_tensor_id="{}.expert_weights".format(
                        layer.layer_id
                    ),
                    activation_source_component_id=activation_component,
                    metadata=expert_metadata,
                )
            )
        if not rank_active:
            expert_ends.append(
                _add_join(
                    builder,
                    "{}.rank{:03d}.experts_no_tokens".format(prefix, rank.rank),
                    (dispatch_ready,),
                    metadata={
                        "layer_id": layer.layer_id,
                        "stage": stage,
                        "local_experts": local_experts,
                        "routed_tokens": routed_tokens,
                    },
                )
            )
    experts_ready = _add_join(
        builder,
        prefix + ".experts.complete",
        expert_ends,
        metadata={"event_kind": "moe_expert_barrier"},
    )
    combine_ends = []
    for tp_rank in range(plan.tp_degree):
        combine_ends.append(
            _add_collective_tasks(
                builder,
                scenario,
                router,
                plan,
                "{}.tp{:02d}.expert_combine".format(prefix, tp_rank),
                "all_to_all",
                plan.ep_group(stage, tp_rank),
                _activation_bytes(
                    layer,
                    routed_tokens
                    * shard_extent(
                        layer.hidden_size,
                        plan.tp_degree,
                        tp_rank,
                        allow_padding=plan.allow_padding,
                    ).local_size,
                    scenario=scenario,
                ),
                (experts_ready,),
                metadata={"layer_id": layer.layer_id, "stage": stage},
            )
        )
    combined = _add_join(
        builder,
        prefix + ".expert_combine.complete",
        combine_ends,
        metadata={"event_kind": "moe_combine_barrier"},
    )
    routed_end = _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".moe_tp_all_reduce",
        "all_reduce",
        tp_ranks,
        _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
        (combined,),
        tensor_elements=token_batch * layer.hidden_size,
        element_bits=_activation_storage_bits(layer, scenario),
        metadata={"layer_id": layer.layer_id, "stage": stage},
    )
    combined_output = routed_end
    if shared_end:
        combined_output = _add_join(
            builder,
            prefix + ".moe_routed_shared_ready",
            (routed_end, shared_end),
            metadata={
                "event_kind": "moe_routed_shared_ready",
                "layer_id": layer.layer_id,
                "stage": stage,
                "routed_experts": layer.num_experts,
                "shared_expert_intermediate_size": (
                    layer.shared_expert_intermediate_size
                ),
                "shared_expert_gate": layer.shared_expert_gate,
            },
        )
    activation_bits = _activation_storage_bits(layer, scenario)
    residual_ends: List[str] = []
    for rank in tp_ranks:
        residual, residual_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=max(1, token_batch * layer.hidden_size),
                operations_per_element=1,
                input_count=3 if shared_end else 2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                name="moe_residual",
            ),
            "{}.moe.residual".format(layer.layer_id),
            "{}.rank{:03d}.moe_residual".format(prefix, rank.rank),
            (combined_output,),
            source_component_id=rank.component_id,
            fallback_keys=("{}.norm".format(layer.layer_id),),
            metadata={
                "event_kind": "moe_residual",
                "layer_id": layer.layer_id,
                "stage": stage,
            },
        )
        residual_ends.append(
            _return_rank_value(
                builder,
                router,
                plan,
                rank,
                residual,
                residual_component,
                _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
                name="{}.rank{:03d}.moe_residual.output_to_gpu".format(
                    prefix, rank.rank
                ),
            )
        )
    return _add_join(
        builder,
        prefix + ".moe_residual.complete",
        residual_ends,
        metadata={"event_kind": "moe_residual_barrier"},
    )


def _compile_parallel_shared_expert(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    layer: LayerSpec,
    token_batch: int,
    prefix: str,
    dependencies: Sequence[str],
) -> str:
    stage = plan.stage_for_layer(layer)
    ranks = plan.tp_group(stage, 0)
    up_shard = shard_extent(
        2 * layer.shared_expert_intermediate_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    down_shard = shard_extent(
        layer.shared_expert_intermediate_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    activation_bits = _activation_storage_bits(layer, scenario)
    element_bytes = max(1, int(math.ceil(activation_bits / 8.0)))
    ends = []
    for rank in ranks:
        target = _parallel_target(scenario, layer, "shared_expert", rank)
        common = {
            "event_kind": "moe_shared_expert",
            "ffn_path": "shared",
            "layer_id": layer.layer_id,
            "stage": stage,
            "shared_expert_intermediate_size": (
                layer.shared_expert_intermediate_size
            ),
            "shared_expert_gate": layer.shared_expert_gate,
            "ep_replication": plan.ep_degree,
            "coverage_component": "shared_expert",
        }
        activation_elements = max(
            1, token_batch * down_shard.local_size
        )
        shared_up_workload = _layer_gemm(
            layer,
            token_batch,
            layer.hidden_size,
            up_shard.local_size,
            name="moe_shared_expert_up_tp",
            f32_storage=_f32_hidden_storage_enabled(scenario),
        )
        shared_activation_target = _primitive_target(
            scenario,
            router,
            rank,
            OperatorClass.ELEMENTWISE,
            "{}.shared_expert.activation".format(layer.layer_id),
            fallback_keys=("{}.shared_expert".format(layer.layer_id),),
        )
        shared_activation_fused, shared_activation_audit = (
            _same_rank_gpu_fusion_decision(
                scenario,
                "gemm_epilogue_activation",
                rank,
                shared_up_workload.output_bytes,
                (
                    ("{}.shared_expert".format(layer.layer_id), target),
                    (
                        "{}.shared_expert.activation".format(layer.layer_id),
                        shared_activation_target,
                    ),
                ),
            )
        )
        shared_activation_audit = dict(shared_activation_audit)
        shared_activation_audit.update(
            {
                "fusion_parent_target": target,
                "fusion_activation_target": shared_activation_target,
            }
        )
        if shared_activation_fused:
            shared_up_workload = _append_gemm_epilogue(
                shared_up_workload,
                operations=5 * activation_elements,
                transcendental_operations=activation_elements,
                output_elements=activation_elements,
                name="swiglu",
            )
        up = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            shared_up_workload,
            target,
            "{}.rank{:03d}.shared_expert.up_gate".format(prefix, rank.rank),
            dependencies,
            weight_tensor_id="{}.shared_expert_weights".format(
                layer.layer_id
            ),
            keep_output_on_target=True,
            metadata={**common, **shared_activation_audit},
        )
        up_component = (
            rank.component_id
            if _is_cim(_component(scenario, target))
            else target
        )
        if shared_activation_fused:
            activation, activation_component = up, up_component
        else:
            activation, activation_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=activation_elements,
                    operations_per_element=5,
                    transcendental_ops_per_element=1,
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    dependency_depth=5,
                    working_set_bytes=_activation_bytes(
                        layer, activation_elements * 3,
                        scenario=scenario,
                    ),
                    reuse_factor=2.0,
                    name="moe_shared_expert_swiglu_activation",
                ),
                "{}.shared_expert.activation".format(layer.layer_id),
                "{}.rank{:03d}.shared_expert.activation".format(
                    prefix, rank.rank
                ),
                (up,),
                source_component_id=up_component,
                fallback_keys=("{}.shared_expert".format(layer.layer_id),),
                metadata={
                    **common,
                    **shared_activation_audit,
                    "ffn_op": "shared_activation",
                    "activation_function": "swiglu",
                },
            )
        down = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            _layer_gemm(
                layer,
                token_batch,
                down_shard.local_size,
                layer.hidden_size,
                name="moe_shared_expert_down_tp",
                f32_storage=_f32_hidden_storage_enabled(scenario),
            ),
            target,
            "{}.rank{:03d}.shared_expert.down".format(prefix, rank.rank),
            (activation,),
            weight_tensor_id="{}.shared_expert_weights".format(
                layer.layer_id
            ),
            activation_source_component_id=activation_component,
            metadata=common,
        )
        if not layer.shared_expert_gate:
            ends.append(down)
            continue
        gate = _add_rank_gemm(
            builder,
            scenario,
            router,
            plan,
            rank,
            _layer_gemm(
                layer,
                token_batch,
                layer.hidden_size,
                1,
                name="moe_shared_expert_gate_tp",
                f32_storage=_f32_hidden_storage_enabled(scenario),
            ),
            _parallel_named_target(
                scenario,
                "{}.shared_expert_gate".format(layer.layer_id),
                rank,
            ),
            "{}.rank{:03d}.shared_expert.gate".format(prefix, rank.rank),
            dependencies,
            weight_tensor_id="{}.shared_expert_gate_weights".format(
                layer.layer_id
            ),
            metadata={**common, "ffn_op": "shared_gate_projection"},
        )
        gate_apply, gate_component = _add_rank_primitive(
                builder,
                scenario,
                router,
                plan,
                rank,
                OperatorClass.ELEMENTWISE,
                ElementwiseWorkload(
                    elements=max(1, token_batch * layer.hidden_size),
                    operations_per_element=2,
                    input_count=2,
                    input_bits=activation_bits,
                    output_bits=activation_bits,
                    name="moe_shared_expert_gate",
                ),
                "{}.shared_expert.gate_apply".format(layer.layer_id),
                "{}.rank{:03d}.shared_expert.gate_apply".format(
                    prefix, rank.rank
                ),
                (down, gate),
                source_component_id=rank.component_id,
                fallback_keys=("{}.shared_expert".format(layer.layer_id),),
                metadata={**common, "ffn_op": "shared_gate_apply"},
        )
        ends.append(
            _return_rank_value(
                builder,
                router,
                plan,
                rank,
                gate_apply,
                gate_component,
                _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
                name="{}.rank{:03d}.shared_expert.gate_apply.output_to_gpu".format(
                    prefix, rank.rank
                ),
                metadata=common,
            )
        )
    return _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        prefix + ".shared_expert_tp_all_reduce",
        "all_reduce",
        ranks,
        _activation_bytes(layer, token_batch * layer.hidden_size, scenario=scenario),
        ends,
        tensor_elements=token_batch * layer.hidden_size,
        element_bits=_activation_storage_bits(layer, scenario),
        metadata={
            "event_kind": "moe_shared_expert_collective",
            "ffn_path": "shared",
            "layer_id": layer.layer_id,
            "stage": stage,
            "ep_replication": plan.ep_degree,
            "coverage_component": "shared_expert",
        },
    )


def _compile_parallel_final_norm(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    phase: str,
    dependencies: Sequence[str],
    token_batch: int = 1,
    output_selection: Optional[_FinalOutputSelection] = None,
) -> Mapping[int, str]:
    """Lower the typed model-level RMSNorm before the output projection."""

    execution_view = _execution_view(scenario)
    final_norms = tuple(
        operator
        for operator in execution_view.operators
        if operator.operator_id == "final_norm"
    )
    if len(final_norms) != 1:
        raise ValueError(
            "typed execution requires exactly one final_norm operator"
        )
    final_norm = final_norms[0]
    if final_norm.op_kind != "rms_norm" or len(final_norm.output_tensor_ids) != 1:
        raise ValueError(
            "final_norm must be one rms_norm operator with one output tensor"
        )
    output_tensor_id = final_norm.output_tensor_ids[0]
    output_tensors = tuple(
        tensor
        for tensor in execution_view.tensors
        if tensor.tensor_id == output_tensor_id
    )
    if len(output_tensors) != 1:
        raise ValueError("final_norm output tensor is missing from typed execution")
    output_tensor = output_tensors[0]
    if not output_tensor.shape or not isinstance(output_tensor.shape[-1], int):
        raise ValueError("final_norm output must declare a concrete hidden width")
    hidden_size = int(output_tensor.shape[-1])
    if hidden_size <= 0:
        raise ValueError("final_norm hidden width must be positive")

    activation_bits = 32 if _f32_hidden_storage_enabled(scenario) else _dtype_bits(output_tensor.dtype)
    batch_tokens = max(1, token_batch)
    stage = plan.pp_degree - 1
    ends: Dict[int, str] = {}
    for rank in plan.tp_group(stage, 0):
        hidden_shard = shard_extent(
            hidden_size,
            plan.tp_degree,
            rank.tp_rank,
            allow_padding=plan.allow_padding,
        )
        hidden_elements = max(1, batch_tokens * hidden_shard.local_size)
        metadata = {
            "stage": stage,
            "phase": phase,
            "model_operator_id": final_norm.operator_id,
            "model_operator_kind": final_norm.op_kind,
            "output_tensor_id": output_tensor_id,
            "hidden_size": hidden_size,
            "hidden_shard_size": hidden_shard.local_size,
            "shard_padding": hidden_shard.padding,
        }
        if output_selection is not None:
            metadata["final_layer_output_selection"] = _selection_audit(output_selection, "final_norm")
        norm_reduce, norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.REDUCTION,
            ReductionWorkload(
                input_elements=hidden_elements,
                output_elements=batch_tokens,
                operations_per_combine=2,
                fixed_operations=batch_tokens,
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                dependency_depth=max(
                    1,
                    int(math.ceil(math.log2(max(1, hidden_shard.local_size)))),
                ),
                working_set_bytes=int(
                    math.ceil(hidden_elements * activation_bits / 8.0)
                ),
                reuse_factor=2.0,
                name="final_norm_reduce",
            ),
            "final_norm.reduce",
            "{}.rank{:03d}.final_norm_reduce".format(phase, rank.rank),
            dependencies,
            source_component_id=rank.component_id,
            fallback_keys=("final_norm",),
            metadata={**metadata, "event_kind": "final_norm_reduce"},
        )
        norm_apply, _norm_component = _add_rank_primitive(
            builder,
            scenario,
            router,
            plan,
            rank,
            OperatorClass.ELEMENTWISE,
            ElementwiseWorkload(
                elements=hidden_elements,
                operations_per_element=2,
                fixed_operations=batch_tokens,
                fixed_transcendental_operations=batch_tokens,
                input_count=2,
                input_bits=activation_bits,
                output_bits=activation_bits,
                dependency_depth=4,
                working_set_bytes=int(
                    math.ceil(hidden_elements * 3 * activation_bits / 8.0)
                ),
                reuse_factor=2.0,
                name="final_norm_apply",
            ),
            "final_norm.apply",
            "{}.rank{:03d}.final_norm_apply".format(phase, rank.rank),
            (norm_reduce,),
            source_component_id=norm_component,
            fallback_keys=("final_norm",),
            metadata={**metadata, "event_kind": "final_norm_apply"},
        )
        ends[rank.rank] = norm_apply
    return ends


def _compile_parallel_lm_head(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    phase: str,
    dependencies: Sequence[str],
    token_batch: int = 1,
    output_selection: Optional[_FinalOutputSelection] = None,
    output_indices_dependency: Optional[str] = None,
) -> str:
    if output_selection is not None:
        if output_selection.logit_rows != token_batch:
            raise ValueError("output-head rows disagree with selected output indices")
        if output_selection.final_norm_rows == 0:
            return _add_join(builder, phase + ".empty_output_head", dependencies,
                metadata={"event_kind": "empty_output_head",
                    "final_layer_output_selection": _selection_audit(output_selection, "empty_tail")})
    execution_view = _execution_view(scenario)
    vocabulary_size = execution_view.vocabulary_size
    if vocabulary_size <= 0:
        return _add_join(builder, phase + ".lm_head.none", dependencies)
    layer = execution_view.layer_instances[-1].layer
    stage = plan.pp_degree - 1
    ranks = plan.tp_group(stage, 0)
    final_norm_ends = _compile_parallel_final_norm(
        builder,
        scenario,
        plan,
        router,
        phase,
        dependencies,
        token_batch=(output_selection.final_norm_rows if output_selection is not None else token_batch),
        output_selection=output_selection,
    )
    if output_selection is not None:
        if output_selection.logit_rows == 0:
            return _add_join(builder, phase + ".empty_output_head", tuple(final_norm_ends.values()),
                metadata={"event_kind": "empty_output_head",
                    "final_layer_output_selection": _selection_audit(output_selection, "empty_tail")})
        if output_selection.position == "after_final_norm":
            if output_indices_dependency is None:
                raise ValueError("final norm row selection has no shared index dependency")
            selected_ends = {}
            for rank in ranks:
                norm_end = final_norm_ends[rank.rank]
                source = builder.rank_value_component((norm_end,), rank.rank) or rank.component_id
                tensor_id = next(task.metadata["output_tensor_id"] for task in reversed(builder.tasks)
                                 if task.task_id == norm_end)
                selected_ends[rank.rank], _ = _add_output_row_selection(
                    builder, scenario, router, plan, rank, layer, output_selection,
                    phase + "." + layer.layer_id + ".rank{:03d}.final_norm_rows".format(rank.rank),
                    (norm_end,), stage="final_norm_rows", source_component=source,
                    target_component=source, indices_dependency=output_indices_dependency,
                    input_tensor_id=tensor_id)
            final_norm_ends = selected_ends
    vocab_shard = shard_extent(
        vocabulary_size,
        plan.tp_degree,
        0,
        allow_padding=plan.allow_padding,
    )
    configured = scenario.placement.op_to_component.get("lm_head")
    weight_tensor_id = "lm_head_weights"
    declared_weight_bytes = _logical_weight_tensor_bytes(
        scenario, weight_tensor_id
    )
    logit_bits, logit_precision_source = _lm_head_output_bits(scenario)
    vocabulary_source = "model_execution_view"
    vocabulary_status = _logits_vocabulary_status(
        scenario, execution_view.vocabulary_size
    )
    ends = []
    for rank in ranks:
        # Respect the explicit lm_head placement for every execution device.
        # Previously only CIM placement was honored; CPU/GPU mappings silently
        # fell back to the rank component.  That made a ``gpu_layers=0``
        # llama.cpp run execute the final projection on GPU (and inserted a
        # full host→GPU weight transfer), while native llama.cpp keeps lm_head
        # on CPU when offload is disabled.  The resulting TTFT/E2E error was
        # dominated by a placement mismatch rather than model compute.
        target = rank.component_id
        if configured:
            configured_id = str(configured)
            configured_component = _component(scenario, configured_id)
            if _is_cim(configured_component):
                target = rank.cim_component_id or configured_id
            elif _kind(configured_component) in {"cpu", "gpu"}:
                target = configured_id
            else:
                raise ValueError(
                    "lm_head placement {} is not a compute component".format(
                        configured_id
                    )
                )
        workload = _layer_gemm(
            layer,
            max(1, token_batch),
            layer.hidden_size,
            vocab_shard.local_size,
            name="lm_head_tp",
            f32_storage=_f32_hidden_storage_enabled(scenario),
        )
        if workload.output_bits != logit_bits:
            workload = replace(workload, output_bits=logit_bits)
        rank_weight_bytes = workload.weight_bytes
        if declared_weight_bytes > 0:
            weight_shard = shard_extent(
                declared_weight_bytes,
                plan.tp_degree,
                rank.tp_rank,
                allow_padding=plan.allow_padding,
            )
            rank_weight_bytes = weight_shard.local_size
            if rank_weight_bytes != workload.weight_bytes:
                workload = replace(
                    workload,
                    weight_storage_bytes=rank_weight_bytes,
                    weight_metadata_bytes=0,
                )
        ends.append(
            _add_rank_gemm(
                builder,
                scenario,
                router,
                plan,
                rank,
                workload,
                target,
                "{}.rank{:03d}.lm_head".format(phase, rank.rank),
                (final_norm_ends[rank.rank],),
                weight_tensor_id=weight_tensor_id,
                metadata={
                    "stage": stage,
                    "phase": phase,
                    "event_kind": "lm_head_projection",
                    "logits_output_bits": logit_bits,
                    "logits_precision_source": logit_precision_source,
                    "logits_vocabulary_size": vocabulary_size,
                    "logits_vocabulary_source": vocabulary_source,
                    "model_graph_vocabulary_size": (
                        execution_view.vocabulary_size
                    ),
                    "logits_vocabulary_status": vocabulary_status,
                    "declared_weight_bytes": declared_weight_bytes,
                    "rank_weight_bytes": rank_weight_bytes,
                    **({"final_layer_output_selection": _selection_audit(output_selection, "lm_head")}
                       if output_selection is not None else {}),
                },
            )
        )
    return _add_collective_tasks(
        builder,
        scenario,
        router,
        plan,
        phase + ".lm_head_all_gather",
        "all_gather",
        ranks,
        int(
            math.ceil(
                max(1, token_batch) * vocabulary_size * logit_bits / 8.0
            )
        ),
        ends,
        tensor_elements=max(1, token_batch) * vocabulary_size,
        element_bits=logit_bits,
        metadata={
            "stage": stage,
            "phase": phase,
            "event_kind": "lm_head_all_gather",
            "logits_output_bits": logit_bits,
            "logits_precision_source": logit_precision_source,
            "logits_vocabulary_size": vocabulary_size,
            "logits_vocabulary_source": vocabulary_source,
            "model_graph_vocabulary_size": execution_view.vocabulary_size,
            "logits_vocabulary_status": vocabulary_status,
        },
    )


def _lm_head_output_dtype(scenario: ScenarioConfig) -> Tuple[str, str]:
    """Resolve the model-graph representation produced by the LM head."""

    def resolve() -> Tuple[str, str]:
        if _f32_hidden_storage_enabled(scenario):
            return "fp32", "declared_llama_cpp_f32_hidden_storage"
        execution_view = _execution_view(scenario)
        tensor_by_id = {
            tensor.tensor_id: tensor for tensor in execution_view.tensors
        }
        output_ids = tuple(
            output_id
            for operator in execution_view.operators
            if operator.operator_id == "lm_head"
            for output_id in operator.output_tensor_ids
        )
        if not output_ids and "logits" in tensor_by_id:
            output_ids = ("logits",)
        output_dtypes = tuple(
            dict.fromkeys(
                tensor_by_id[output_id].dtype
                for output_id in output_ids
                if output_id in tensor_by_id
            )
        )
        if len(output_dtypes) == 1:
            return output_dtypes[0], "model_logits_tensor_dtype"
        layer = execution_view.layer_instances[-1].layer
        return layer.dtype, "partial_last_layer_activation_fallback"

    context = _active_compilation_context(scenario)
    if context is None:
        return resolve()
    return context.invariant(("lm_head_output_dtype",), resolve)


def _lm_head_output_bits(scenario: ScenarioConfig) -> Tuple[int, str]:
    """Resolve model-graph logits precision for the LM head projection."""

    dtype, source = _lm_head_output_dtype(scenario)
    return _dtype_bits(dtype), source


def _host_visible_logits_vocabulary_size(scenario: ScenarioConfig) -> int:
    contract = scenario.host_output_contract
    if contract is not None:
        return contract.vocabulary_size
    return _execution_view(scenario).vocabulary_size


def _host_visible_logits_output_bits(
    scenario: ScenarioConfig,
) -> Tuple[int, str]:
    contract = scenario.host_output_contract
    if contract is not None:
        return contract.logits_element_bytes * 8, "host_output_contract"
    return _lm_head_output_bits(scenario)


def _logits_vocabulary_status(
    scenario: ScenarioConfig,
    model_graph_vocabulary_size: int,
) -> str:
    contract = scenario.host_output_contract
    if contract is None:
        return "consistent"
    return (
        "consistent"
        if contract.vocabulary_size == model_graph_vocabulary_size
        else "partial_model_graph_disagreement"
    )


def _host_output_contract_projection(
    scenario: ScenarioConfig,
) -> Dict[str, object]:
    """Project model/host output agreement without inventing a conversion."""

    def resolve() -> Dict[str, object]:
        model_vocabulary_size = _execution_view(scenario).vocabulary_size
        model_logits_dtype, model_dtype_source = _lm_head_output_dtype(scenario)
        contract = scenario.host_output_contract
        vocabulary_status = _logits_vocabulary_status(
            scenario, model_vocabulary_size
        )
        if contract is None:
            precision_status = "partial"
            contract_status = "partial"
            partial_reason = "host_output_contract_not_declared"
            host_vocabulary_size = None
            host_logits_dtype = None
            host_logits_element_bytes = None
        else:
            host_vocabulary_size = contract.vocabulary_size
            host_logits_dtype = contract.logits_dtype
            host_logits_element_bytes = contract.logits_element_bytes
            if model_dtype_source not in {
                "model_logits_tensor_dtype",
                "declared_llama_cpp_f32_hidden_storage",
            }:
                precision_status = "partial_model_graph_dtype_unresolved"
            elif canonical_dtype(model_logits_dtype) == canonical_dtype(
                contract.logits_dtype
            ):
                precision_status = "consistent"
            else:
                precision_status = "partial_model_graph_disagreement"
            complete = (
                vocabulary_status == "consistent"
                and precision_status == "consistent"
            )
            contract_status = "complete" if complete else "partial"
            disagreements = []
            if vocabulary_status != "consistent":
                disagreements.append("vocabulary")
            if precision_status != "consistent":
                disagreements.append("logits_dtype")
            partial_reason = (
                None
                if complete
                else "host_output_contract_model_graph_{}_disagreement".format(
                    "_and_".join(disagreements)
                )
            )
        return {
            "model_graph_vocabulary_size": model_vocabulary_size,
            "host_contract_vocabulary_size": host_vocabulary_size,
            "logits_vocabulary_status": vocabulary_status,
            "model_graph_logits_dtype": model_logits_dtype,
            "model_graph_logits_dtype_source": model_dtype_source,
            "host_contract_logits_dtype": host_logits_dtype,
            "host_contract_logits_element_bytes": host_logits_element_bytes,
            "logits_precision_status": precision_status,
            "host_output_contract_status": contract_status,
            "host_output_contract_partial_reason": partial_reason,
        }

    context = _active_compilation_context(scenario)
    if context is None:
        return resolve()
    return context.invariant(("host_output_contract_projection",), resolve)


@dataclass(frozen=True)
class _HostOutputTailTarget:
    cpu_component_id: str
    output_target_component_id: Optional[str]
    output_target_kind: str
    memory_component_id: Optional[str]
    memory_semantics: str


def _host_output_tail_target(
    scenario: ScenarioConfig,
    router: TopologyRouter,
) -> _HostOutputTailTarget:
    cpu_id = scenario.host_orchestration_profile.cpu_component_id
    contract = scenario.host_output_contract
    if contract is None:
        return _HostOutputTailTarget(
            cpu_component_id=cpu_id,
            output_target_component_id=None,
            output_target_kind="undeclared",
            memory_component_id=None,
            memory_semantics="cpu_local_nearest_host_memory",
        )

    output_target = str(contract.target_component_id)
    target = _component(scenario, output_target)
    target_kind = _kind(target)
    if target_kind == "cpu":
        if output_target != cpu_id:
            raise ValueError(
                "host output CPU target {} must match orchestration CPU {}"
                .format(output_target, cpu_id)
            )
        memory_component_id = _nearest_profile_component_id(
            scenario, cpu_id, "host_memory"
        )
        if memory_component_id is None:
            raise ValueError(
                "host output CPU target {} has no reachable host-memory target"
                .format(cpu_id)
            )
        _cpu_profiles(
            scenario,
            cpu_id,
            memory_component_id=memory_component_id,
        )
        return _HostOutputTailTarget(
            cpu_component_id=cpu_id,
            output_target_component_id=output_target,
            output_target_kind=target_kind,
            memory_component_id=memory_component_id,
            memory_semantics=(
                "cpu_contract_target_normalized_to_nearest_host_memory"
            ),
        )
    if target_kind != "host_memory":
        raise ValueError(
            "host output target {} must be CPU-visible memory or a CPU".format(
                output_target
            )
        )

    _cpu_profiles(scenario, cpu_id, memory_component_id=output_target)
    return _HostOutputTailTarget(
        cpu_component_id=cpu_id,
        output_target_component_id=output_target,
        output_target_kind=target_kind,
        memory_component_id=output_target,
        memory_semantics=(
            "host_output_contract_exact_host_memory_cost_profile"
        ),
    )


def _add_host_output_cpu_workload(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    workload: object,
    operator_class: OperatorClass,
    dependencies: Sequence[str],
    *,
    name: str,
    metadata: Mapping[str, object],
    cpu_component_id: Optional[str] = None,
    memory_component_id: Optional[str] = None,
) -> str:
    """Lower one aggregate CPU output operation without element expansion."""

    cpu_id = cpu_component_id or scenario.host_orchestration_profile.cpu_component_id
    rank = plan.rank_at(0, plan.pp_degree - 1, 0)
    estimate = _estimate_typed_primitive(
        scenario,
        cpu_id,
        operator_class,
        workload,
        memory_component_id=memory_component_id,
    )
    prior = tuple(dependencies)
    last = ""
    for phase in estimate.phases:
        demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=cpu_id,
                memory_component_id=memory_component_id,
            )
            for demand in phase.demands
        )
        last = builder.add(
            "{}.{}".format(name, phase.name),
            phase.category,
            demands,
            dependencies=prior,
            advance=False,
            metadata={
                **dict(metadata),
                "phase": phase.name,
                "operator_class": operator_class.value,
                "target_component": cpu_id,
                "cost_model": dict(estimate.metadata),
                "phase_metadata": dict(phase.metadata),
                "analytical_ops": sum(
                    demand.work_units for demand in demands
                ),
                "analytical_bytes": sum(
                    demand.bytes_moved for demand in demands
                ),
                "analytical_energy_pj": sum(
                    demand.energy_pj for demand in demands
                ),
            },
        )
        prior = (last,)
    if not last:  # pragma: no cover - cost-estimator contract
        raise AssertionError("CPU output estimate produced no phases")
    return last


def _add_host_visible_logits_sampling_commit(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    dependencies: Sequence[str],
    *,
    phase: str,
    logit_rows: int,
    committed_rows: int,
) -> str:
    """Make one physical logits batch host-visible, sampled, and committed."""

    rows = max(0, int(logit_rows))
    commits = max(0, min(rows, int(committed_rows)))
    if rows <= 0:
        return _add_join(builder, phase + ".output.none", dependencies)
    vocabulary_size = _host_visible_logits_vocabulary_size(scenario)
    if vocabulary_size <= 0:
        return _add_join(builder, phase + ".output.no_vocabulary", dependencies)
    logit_bits, precision_source = _host_visible_logits_output_bits(scenario)
    logits_bytes = int(math.ceil(rows * vocabulary_size * logit_bits / 8.0))
    profile = scenario.host_orchestration_profile
    transport = scenario.runtime_profile.pcie_dma_iommu
    rank = plan.rank_at(0, plan.pp_degree - 1, 0)
    source_component = rank.component_id
    tail_target = _host_output_tail_target(scenario, router)
    transfer_source_component = source_component
    if tail_target.memory_component_id is not None:
        try:
            candidate_source = (
                rank.memory_component_id
                or _compute_local_runtime_memory_component_id(
                    scenario,
                    source_component,
                )
            )
            coherent_d2h_probe = _transfer_phases(
                scenario,
                router,
                candidate_source,
                tail_target.memory_component_id,
                logits_bytes,
                policy=plan.routing_policy,
                name=phase + ".output.d2h_probe",
            )
        except (KeyError, TypeError, ValueError):
            coherent_d2h_probe = ()
        if (
            len(coherent_d2h_probe) == 1
            and coherent_d2h_probe[0].metadata.get("transfer_execution")
            == "coherent_dma"
        ):
            transfer_source_component = candidate_source
    contract_projection = _host_output_contract_projection(scenario)
    common = {
        "logit_rows": rows,
        "committed_rows": commits,
        "vocabulary_size": vocabulary_size,
        "vocabulary_size_source": (
            "host_output_contract"
            if scenario.host_output_contract is not None
            else "model_execution_view"
        ),
        **contract_projection,
        "logits_output_bits": logit_bits,
        "logits_bytes": logits_bytes,
        "logits_precision_source": precision_source,
        "output_source_component_id": source_component,
        "output_transfer_source_component_id": transfer_source_component,
        "aggregation": "invocation_group_batch",
        "host_output_target_component": (
            tail_target.output_target_component_id
        ),
        "host_output_target_kind": tail_target.output_target_kind,
        "host_output_cpu_component": tail_target.cpu_component_id,
        "host_output_memory_component_id": tail_target.memory_component_id,
        "host_output_memory_semantics": tail_target.memory_semantics,
    }
    prior = tuple(dependencies)
    contract = scenario.host_output_contract
    if contract is None:
        transfer_end = builder.add(
            phase + ".output.d2h_unmodeled",
            TaskCategory.COMMUNICATION,
            dependencies=prior,
            advance=False,
            metadata=_communication_task_metadata(
                {
                    **common,
                    "event_kind": "logits_d2h_unmodeled",
                    "serving_output_stage": "device_output_completion",
                    "modeling_status": "partial",
                    "partial_reason": "host_output_contract_not_declared",
                    "direction": "device_to_host",
                }
            ),
        )
    else:
        target_component = tail_target.memory_component_id
        if target_component is None:  # pragma: no cover - validated contract
            raise AssertionError(
                "declared host output contract has no physical memory target"
            )
        pages = max(
            1,
            (logits_bytes + transport.iommu_page_size_bytes - 1)
            // transport.iommu_page_size_bytes,
        )
        misses = max(
            1,
            (pages + transport.iommu_tlb_entries - 1)
            // transport.iommu_tlb_entries,
        )
        walk_batches = max(
            1,
            (misses + transport.iommu_max_outstanding_walks - 1)
            // transport.iommu_max_outstanding_walks,
        )
        translated = builder.add(
            phase + ".output.iommu_translate",
            TaskCategory.MEMORY,
            (
                ResourceDemand(
                    "{}.iommu".format(profile.cpu_component_id),
                    walk_batches * transport.iommu_miss_latency_ns,
                    work_units=float(pages),
                ),
            ),
            dependencies=prior,
            advance=False,
            metadata={
                **common,
                "event_kind": "logits_d2h_iommu_translation",
                "serving_output_stage": "device_output_completion",
                "runtime_phase": "iommu_translate",
                "transaction_kind": "address_translation",
                "transaction_count": pages,
                "transaction_batches": walk_batches,
                "target_component": target_component,
                "direction": "device_to_host",
            },
        )
        dma_setup = _dma_setup_service(
            logits_bytes,
            batch_bytes=transport.dma_batch_bytes,
            queue_depth=transport.dma_queue_depth,
            max_outstanding=transport.dma_max_outstanding,
            fixed_latency_ns=profile.dma_latency_ns,
            submission_ns_per_wave=profile.dma_queue_submission_ns,
        )
        queued = builder.add(
            phase + ".output.dma_queue",
            TaskCategory.COMMUNICATION,
            (
                ResourceDemand(
                    profile.dma_resource_id,
                    dma_setup.service_ns,
                    work_units=float(dma_setup.transaction_count),
                ),
            ),
            dependencies=(translated,),
            advance=False,
            metadata=_communication_task_metadata(
                {
                    **common,
                    "event_kind": "logits_d2h_dma_controller_batch",
                    "serving_output_stage": "device_output_completion",
                    "runtime_phase": "dma_queue",
                    "transaction_kind": "device_to_host_dma",
                    "transaction_count": dma_setup.transaction_count,
                    "transaction_batches": dma_setup.wave_count,
                    "queue_parallelism": dma_setup.queue_parallelism,
                    "queue_depth": transport.dma_queue_depth,
                    "max_outstanding": transport.dma_max_outstanding,
                    "target_component": target_component,
                    "direction": "device_to_host",
                    "cost_owner": "dma_descriptor_queue",
                    "bulk_service_owner": False,
                }
            ),
        )
        transfer_end = _add_transfer_tasks(
            builder,
            router,
            transfer_source_component,
            target_component,
            logits_bytes,
            (queued,),
            name=phase + ".output.d2h",
            routing_policy=plan.routing_policy,
            metadata={
                **common,
                "event_kind": "logits_d2h",
                "serving_output_stage": "device_output_completion",
                "target_component": target_component,
                "direction": "device_to_host",
                "allocation_semantics": contract.allocation_semantics,
                "modeling_status": "complete",
            },
        )
    completed = builder.add(
        phase + ".output.completion_interrupt",
        TaskCategory.SYNCHRONIZATION,
        (
            ResourceDemand(
                _HOST_OUTPUT_COMPLETION_INTERRUPT_RESOURCE_ID,
                0.0,
            ),
        ),
        dependencies=(transfer_end,),
        advance=False,
        metadata={
            **common,
            "event_kind": "logits_d2h_completion_interrupt",
            "serving_output_stage": "device_output_completion",
            "opaque_device_fence": True,
            "completion_interrupt_service_ns": 0.0,
            "completion_interrupt_status": "partial",
            "completion_interrupt_resource_id": (
                _HOST_OUTPUT_COMPLETION_INTERRUPT_RESOURCE_ID
            ),
            "completion_interrupt_resource_status": (
                "declared_shared_runtime_resource"
            ),
            "partial_reason": "completion_interrupt_latency_not_declared",
            "wait_accounting": "dependency_elapsed_time",
            "host_wait_service_ns": 0.0,
            "timing_completeness": "partial",
            "direction": "device_to_host",
        },
    )

    # CPU lowering already charges the exact target HostMemoryProfile below.
    # Adding a second host-memory-to-CPU transfer would charge the same logits
    # read through both the topology endpoint and the CPU memory cost model.
    cpu_dependencies: Tuple[str, ...] = (completed,)

    sampling = scenario.sampling_policy
    sampling_implementation = (
        None
        if sampling is None or sampling.implementation is None
        else sampling.implementation.strip().lower()
    )
    if sampling_implementation == _LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION:
        if logit_bits != 32:
            raise ValueError(
                "llama_cpp_cpu_chain requires host-visible f32 logits"
            )
        cpu_profile, _host_memory_profile = _cpu_profiles(
            scenario,
            tail_target.cpu_component_id,
            memory_component_id=tail_target.memory_component_id,
        )
        per_row_logit_bytes = vocabulary_size * 4
        per_row_candidate_bytes = (
            vocabulary_size * _LLAMA_CPP_CPU_SAMPLER_RECORD_BYTES
        )
        candidate_workload = MemoryWorkload(
            read_bytes=per_row_logit_bytes,
            write_bytes=per_row_candidate_bytes,
            working_set_bytes=(
                per_row_logit_bytes + per_row_candidate_bytes
            ),
            name="llama_cpp_candidate_materialization",
        )
        candidate_estimate = estimate_cpu_logical_stream(
            cpu_profile,
            candidate_workload,
            serial_repetitions=rows,
        )
        candidate_phase = candidate_estimate.phases[0]
        candidate_demands = tuple(
            _namespace_demand(
                scenario,
                demand,
                rank=rank,
                target_component_id=tail_target.cpu_component_id,
                memory_component_id=tail_target.memory_component_id,
            )
            for demand in candidate_phase.demands
        )
        candidate_metadata = {
            **common,
            "event_kind": "cpu_logits_candidate_materialization",
            "serving_output_stage": "cpu_output_commit",
            "sampling_implementation": (
                _LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION
            ),
            "sampling_model_status": "partial",
            "candidate_record_bytes": (
                _LLAMA_CPP_CPU_SAMPLER_RECORD_BYTES
            ),
            "candidate_record_layout": (
                "x86_64 llama_token_data: int32 id, float logit, float p"
            ),
            "rows_execution": _LLAMA_CPP_CPU_SAMPLER_ROWS_EXECUTION,
            "row_service_aggregation": "serial_sum",
            "per_row_logical_read_bytes": per_row_logit_bytes,
            "per_row_logical_write_bytes": per_row_candidate_bytes,
            "logical_read_bytes": per_row_logit_bytes * rows,
            "logical_write_bytes": per_row_candidate_bytes * rows,
            "logical_stream_bytes": (
                (per_row_logit_bytes + per_row_candidate_bytes) * rows
            ),
            "physical_memory_traffic_status": "unknown_not_charged",
            "timing_completeness": "partial",
            "timing_interpretation": (
                "optimistic_single_core_pipeline_estimate"
            ),
            "strict_mathematical_lower_bound": False,
            "source_provenance": _LLAMA_CPP_CPU_SAMPLER_PROVENANCE,
            "cost_model": dict(candidate_estimate.metadata),
            "phase_metadata": dict(candidate_phase.metadata),
            "analytical_ops": sum(
                demand.work_units for demand in candidate_demands
            ),
            "analytical_bytes": 0,
            "analytical_energy_pj": sum(
                demand.energy_pj for demand in candidate_demands
            ),
        }
        candidate_materialized = builder.add(
            phase + ".output.cpu_candidate_materialization",
            candidate_phase.category,
            candidate_demands,
            dependencies=cpu_dependencies,
            advance=False,
            metadata=candidate_metadata,
        )
        filter_demands: Tuple[ResourceDemand, ...] = ()
        filter_scan_metadata: Dict[str, object] = {}
        if (
            sampling.top_k is not None
            and 2 <= sampling.top_k <= min(128, vocabulary_size)
            and sampling.top_k < vocabulary_size
        ):
            # Only the mandatory tail scan: every remaining candidate is
            # compared with the heap root. Heap repair and sorting stay unknown.
            per_row_comparisons = vocabulary_size - sampling.top_k
            filter_estimate = estimate_cpu_logical_stream(
                cpu_profile,
                MemoryWorkload(
                    read_bytes=4 * per_row_comparisons,
                    name="llama_cpp_top_k_mandatory_scan",
                ),
                serial_repetitions=rows,
                comparison_count=per_row_comparisons,
            )
            filter_demands = tuple(
                _namespace_demand(
                    scenario,
                    demand,
                    rank=rank,
                    target_component_id=tail_target.cpu_component_id,
                    memory_component_id=tail_target.memory_component_id,
                )
                for demand in filter_estimate.phases[0].demands
            )
            filter_scan_metadata = {
                "filter_work_model": "top_k_mandatory_tail_scan",
                "per_row_comparison_count": per_row_comparisons,
                "comparison_count": rows * per_row_comparisons,
                "per_row_logical_read_bytes": 4 * per_row_comparisons,
                "logical_read_bytes": 4 * rows * per_row_comparisons,
                "logical_write_bytes": 0,
                "logical_read_source": "materialized_candidate_logit_field",
                "candidate_record_stride_bytes": _LLAMA_CPP_CPU_SAMPLER_RECORD_BYTES,
                "row_service_aggregation": "serial_sum",
                "physical_memory_traffic_status": "unknown_not_charged",
                "timing_interpretation": "optimistic_single_core_pipeline_estimate",
                "strict_mathematical_lower_bound": False,
                "partial_reason": (
                    "heap_root_build_repair_moves_sort_branch_probability_"
                    "special_function_cache_and_compiler_cost_unknown"
                ),
                "scan_source_provenance": _LLAMA_CPP_CPU_SAMPLER_SCAN_PROVENANCE,
                "cost_model": dict(filter_estimate.metadata),
                "analytical_ops": sum(demand.work_units for demand in filter_demands),
                "analytical_bytes": 0,
            }
        sampled = builder.add(
            phase + ".output.cpu_sampler_filter_chain_unmodeled",
            TaskCategory.COMPUTE,
            filter_demands,
            dependencies=(candidate_materialized,),
            advance=False,
            metadata={
                **common,
                "event_kind": "cpu_logits_sampling",
                "serving_output_stage": "cpu_output_commit",
                "sampling_model": _LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION,
                "sampling_model_status": "partial",
                "sampling_implementation": (
                    _LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION
                ),
                "rows_execution": _LLAMA_CPP_CPU_SAMPLER_ROWS_EXECUTION,
                "top_k_algorithm": (
                    _LLAMA_CPP_CPU_SAMPLER_TOP_K_ALGORITHM
                ),
                "top_k": sampling.top_k,
                "top_p": sampling.top_p,
                "min_p": sampling.min_p,
                "min_keep": sampling.min_keep,
                "temperature": sampling.temperature,
                "filter_chain": (
                    "top_k",
                    "top_p",
                    "min_p",
                    "temperature",
                    "distribution",
                ),
                "comparison_complexity_proxy": "O(V log K)",
                "comparison_complexity_timing_use": "metadata_only",
                "filter_chain_service_ns": sum(
                    (demand.service_ns for demand in filter_demands), 0.0
                ),
                "timing_completeness": "partial",
                "partial_reason": (
                    "comparison_swap_branch_cache_and_compiler_cost_not_"
                    "declared"
                ),
                "source_provenance": _LLAMA_CPP_CPU_SAMPLER_PROVENANCE,
                **filter_scan_metadata,
            },
        )
    elif sampling is not None and sampling.mode.strip().lower() == "greedy":
        sampled = _add_host_output_cpu_workload(
            builder,
            scenario,
            plan,
            ReductionWorkload(
                input_elements=rows * vocabulary_size,
                output_elements=rows,
                operations_per_combine=1,
                input_bits=logit_bits,
                output_bits=max(8, profile.token_bytes * 8),
                dependency_depth=max(
                    1, int(math.ceil(math.log2(vocabulary_size)))
                ),
                working_set_bytes=logits_bytes,
                streaming_fraction=1.0,
                name="host_logits_greedy_argmax",
            ),
            OperatorClass.REDUCTION,
            cpu_dependencies,
            name=phase + ".output.cpu_sampling",
            metadata={
                **common,
                "event_kind": "cpu_logits_sampling",
                "serving_output_stage": "cpu_output_commit",
                "sampling_model": "greedy_argmax",
                "sampling_model_status": "complete",
                "temperature": sampling.temperature,
            },
            cpu_component_id=tail_target.cpu_component_id,
            memory_component_id=tail_target.memory_component_id,
        )
    else:
        sampled = builder.add(
            phase + ".output.cpu_sampling_unmodeled",
            TaskCategory.COMPUTE,
            dependencies=cpu_dependencies,
            advance=False,
            metadata={
                **common,
                "event_kind": "cpu_logits_sampling",
                "serving_output_stage": "cpu_output_commit",
                "sampling_model": (
                    "undeclared" if sampling is None else sampling.mode
                ),
                "sampling_model_status": "partial",
                "partial_reason": "sampling_algorithm_not_declared_or_modeled",
            },
        )

    commit_bytes = commits * profile.token_bytes
    if commit_bytes > 0:
        return _add_host_output_cpu_workload(
            builder,
            scenario,
            plan,
            MemoryWorkload(
                write_bytes=commit_bytes,
                working_set_bytes=commit_bytes,
                name="host_token_commit",
            ),
            OperatorClass.MEMORY,
            (sampled,),
            name=phase + ".output.cpu_token_commit",
            metadata={
                **common,
                "event_kind": "cpu_token_commit",
                "serving_output_stage": "cpu_output_commit",
                "commit_bytes": commit_bytes,
                "commit_model": "typed_host_memory_write",
            },
            cpu_component_id=tail_target.cpu_component_id,
            memory_component_id=tail_target.memory_component_id,
        )
    return builder.add(
        phase + ".output.cpu_token_commit_empty",
        TaskCategory.OUTPUT,
        dependencies=(sampled,),
        advance=False,
        metadata={
            **common,
            "event_kind": "cpu_token_commit",
            "serving_output_stage": "cpu_output_commit",
            "commit_bytes": 0,
            "commit_model": "no_committed_rows",
        },
    )


def _emit_parallel_token(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    token_index: int,
    dependencies: Sequence[str],
) -> str:
    marker = TraceMarker.FIRST_TOKEN if token_index == 0 else TraceMarker.TOKEN_EMIT
    return builder.add(
        "emit_token_{:04d}".format(token_index),
        TaskCategory.OUTPUT,
        dependencies=dependencies,
        advance=False,
        marker=marker,
        token_index=token_index,
        metadata={
            "event_kind": "token_visible",
            "committed_tokens": 1,
        },
    )


def _serving_item_verifier_tokens(item: object) -> int:
    explicit = getattr(item, "verifier_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    return max(
        0,
        int(
            getattr(item, "proposed_tokens", 0)
            or getattr(item, "token_count", 0)
        ),
    )


def _serving_item_main_tokens(item: object) -> int:
    explicit = getattr(item, "main_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    return 1 if str(getattr(item, "phase", "")) == "mtp" else 0


def _serving_item_draft_tokens(item: object) -> int:
    explicit = getattr(item, "draft_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    return max(
        0,
        _serving_item_verifier_tokens(item) - _serving_item_main_tokens(item),
    )


def _serving_item_committed_tokens(item: object) -> int:
    explicit = getattr(item, "committed_tokens", None)
    if explicit is not None:
        return max(0, int(explicit))
    proposed = _serving_item_verifier_tokens(item)
    return round_accepted_prefix(
        proposed,
        float(getattr(item, "expected_accepted_tokens", 1.0)),
    )


def _serving_item_logit_tokens(item: object) -> int:
    """Return output-head positions without conflating them with backbone work."""

    token_count = max(0, int(getattr(item, "token_count", 0)))
    explicit = getattr(item, "logit_tokens", None)
    if explicit is None:
        # Compatibility for custom cohorts created before logit positions
        # became explicit: their historical contract requested every output.
        return token_count
    return min(token_count, max(0, int(explicit)))


def _serving_item_is_mtp(item: object) -> bool:
    return str(getattr(item, "phase", "")).strip().lower() == "mtp"


def _serving_mtp_items(cohort: object) -> Tuple[object, ...]:
    return tuple(
        item
        for item in tuple(getattr(cohort, "items", ()) or ())
        if _serving_item_is_mtp(item)
    )


@dataclass(frozen=True)
class _ServingInvocationLane:
    """One scheduler item position retained through backend lowering."""

    item_index: int
    request_id: str
    phase: str
    position: int
    context_tokens: int
    kv_read_tokens: int
    kv_append_tokens: int
    kv_materialized_tokens: int
    requires_logits: bool
    state_in_committed_prefix: bool
    state_commit_boundary: bool
    state_release_boundary: bool

    def identity_metadata(self, kind: str) -> Mapping[str, object]:
        identity: Dict[str, object] = {
            "item_index": self.item_index,
            "request_id": self.request_id,
            "lane_id": "item{:04d}.position{:04d}".format(
                self.item_index, self.position
            ),
            "position": self.position,
            "phase": self.phase,
            "requires_logits": self.requires_logits,
        }
        if self.phase == "mtp":
            identity["verifier_position"] = self.position
            identity["linear_state_position_role"] = (
                "committed_prefix"
                if self.state_in_committed_prefix
                else "rejected_speculative"
            )
            identity["linear_state_commit_boundary"] = (
                self.state_commit_boundary
            )
            identity["linear_state_release_boundary"] = (
                self.state_release_boundary
            )
        return identity


@dataclass(frozen=True)
class _ServingInvocationGroup:
    """A backend operator invocation without erased request/state axes."""

    group_index: int
    kind: str
    lanes: Tuple[_ServingInvocationLane, ...]
    predecessor_group_id: Optional[str]
    batching_semantics: str
    kv_scan_tokens: int = 0
    q4_mma_view_tokens_lower_bound: int = 0

    @property
    def group_id(self) -> str:
        return "operator-invocation-group-{:04d}".format(self.group_index)

    @property
    def token_batch(self) -> int:
        return len(self.lanes)

    @property
    def context_tokens(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    sum(lane.context_tokens for lane in self.lanes)
                    / float(max(1, self.token_batch))
                )
            ),
        )

    @property
    def kv_read_tokens(self) -> int:
        return sum(lane.kv_read_tokens for lane in self.lanes)

    @property
    def kv_append_tokens(self) -> int:
        return sum(lane.kv_append_tokens for lane in self.lanes)

    @property
    def kv_materialized_tokens(self) -> int:
        return sum(lane.kv_materialized_tokens for lane in self.lanes)

    @property
    def logit_token_batch(self) -> int:
        return sum(bool(lane.requires_logits) for lane in self.lanes)

    @property
    def committed_logit_token_batch(self) -> int:
        return sum(
            bool(lane.requires_logits and lane.state_in_committed_prefix)
            for lane in self.lanes
        )

    @property
    def request_ids(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(lane.request_id for lane in self.lanes))

    @property
    def phase_token_counts(self) -> Mapping[str, int]:
        counts: Dict[str, int] = {}
        for lane in self.lanes:
            counts[lane.phase] = counts.get(lane.phase, 0) + 1
        return counts

    def linear_state_runtime(
        self,
    ) -> Optional[_LinearStateRuntimeSemantics]:
        mtp_lanes = tuple(lane for lane in self.lanes if lane.phase == "mtp")
        if not mtp_lanes:
            if (
                self.batching_semantics in {
                    "explicit_stateful_decode_batch_capability",
                    "explicit_equal_length_stateful_ubatch",
                }
                and len(self.request_ids) > 1
            ):
                return _LinearStateRuntimeSemantics(
                    mode="persistent_update",
                    read_source="committed",
                    persistent_update_lane_ids=tuple(
                        str(lane.identity_metadata(self.kind)["lane_id"])
                        for lane in self.lanes
                    ),
                    independent_state_owner_count=len(self.request_ids),
                )
            return None
        lane_ids = tuple(
            str(lane.identity_metadata(self.kind)["lane_id"])
            for lane in mtp_lanes
        )
        committed_lanes = tuple(
            lane_id
            for lane, lane_id in zip(mtp_lanes, lane_ids)
            if lane.state_in_committed_prefix
        )
        commit_lanes = tuple(
            lane_id
            for lane, lane_id in zip(mtp_lanes, lane_ids)
            if lane.state_commit_boundary
        )
        persistent_lanes = tuple(
            str(lane.identity_metadata(self.kind)["lane_id"])
            for lane in self.lanes
            if lane.phase != "mtp"
        )
        return _mtp_linear_state_runtime(
            (lane.position for lane in mtp_lanes),
            (
                lane.position
                for lane in mtp_lanes
                if lane.state_in_committed_prefix
            ),
            (
                lane.position
                for lane in mtp_lanes
                if lane.state_commit_boundary
            ),
            read_source=(
                "committed"
                if any(lane.position == 0 for lane in mtp_lanes)
                else "speculative"
            ),
            release_temporary=any(
                lane.state_release_boundary for lane in mtp_lanes
            ),
            materialized_lane_ids=lane_ids,
            committed_lane_ids=committed_lanes,
            commit_snapshot_lane_ids=commit_lanes,
            persistent_update_lane_ids=persistent_lanes,
        )

    def audit_metadata(self) -> Mapping[str, object]:
        verifier_positions = tuple(
            lane.position for lane in self.lanes if lane.phase == "mtp"
        )
        runtime = self.linear_state_runtime()
        return {
            "group_id": self.group_id,
            "group_index": self.group_index,
            "kind": self.kind,
            "request_ids": self.request_ids,
            "phase_token_counts": dict(self.phase_token_counts),
            "item_indexes": tuple(
                dict.fromkeys(lane.item_index for lane in self.lanes)
            ),
            "lane_count": len(self.lanes),
            "lanes": tuple(
                lane.identity_metadata(self.kind) for lane in self.lanes
            ),
            "verifier_positions": verifier_positions,
            "token_batch": self.token_batch,
            "physical_ubatch_index": self.group_index,
            "physical_ubatch_rows": self.token_batch,
            "context_tokens": self.context_tokens,
            "kv_read_tokens": self.kv_read_tokens,
            "kv_append_tokens": self.kv_append_tokens,
            "kv_materialized_tokens": self.kv_materialized_tokens,
            "logit_token_batch": self.logit_token_batch,
            "committed_logit_token_batch": self.committed_logit_token_batch,
            "predecessor_group_id": self.predecessor_group_id,
            "batching_semantics": self.batching_semantics,
            **({
                "llama_cpp_kv_scan_tokens": self.kv_scan_tokens,
                "scan_bound": "occupied_rows_lower_bound",
                "kv_scan_partial_timing": True,
            } if self.kv_scan_tokens else {}),
            **({
                "llama_cpp_q4_kv_materialization_view_tokens_lower_bound": self.q4_mma_view_tokens_lower_bound,
                "materialization_extent_completeness": "lower_bound",
            } if self.q4_mma_view_tokens_lower_bound else {}),
            **(runtime.audit_metadata() if runtime is not None else {}),
        }


def _serving_bool_capability(
    scenario: ScenarioConfig,
    keys: Sequence[str],
) -> bool:
    """Accept only an explicit boolean backend capability declaration."""

    value = _metadata_value(
        (
            scenario.workload.metadata,
            scenario.model.metadata,
            scenario.placement.metadata,
        ),
        keys,
    )
    return value is True


def _serving_invocation_lanes(
    cohort: object,
    *,
    inter_invocation_positions: bool,
) -> Tuple[Tuple[_ServingInvocationLane, ...], ...]:
    """Project each scheduler item into request-local ordered positions."""

    kind = str(getattr(cohort, "kind", "decode"))
    item_lanes: List[Tuple[_ServingInvocationLane, ...]] = []
    for item_index, item in enumerate(
        tuple(getattr(cohort, "items", ()) or ())
    ):
        token_count = max(0, int(getattr(item, "token_count", 0)))
        context = max(0, int(getattr(item, "context_tokens", 0)))
        phase = str(getattr(item, "phase", kind))
        append_raw = getattr(item, "kv_append_tokens", None)
        append_tokens = max(
            0, int(token_count if append_raw is None else append_raw)
        )
        materialized_raw = getattr(item, "kv_materialized_tokens", None)
        materialized_tokens = max(
            0,
            int(
                append_tokens
                if materialized_raw is None
                else materialized_raw
            ),
        )
        is_mtp = phase == "mtp"
        committed_tokens = (
            min(token_count, _serving_item_committed_tokens(item))
            if is_mtp
            else token_count
        )
        logit_tokens = _serving_item_logit_tokens(item)
        first_logit_position = token_count - logit_tokens
        fused_prefill_chunk = kind == "prefill" or phase in {
            "prefill",
            "recompute",
        }
        lanes = []
        for position in range(token_count):
            # ``context_tokens`` includes the current causal position for
            # operator compute.  A fused prefill/recompute chunk loads its
            # historical persisted K/V through the first lane and reuses that
            # tile across the remaining query rows.  Decode/MTP behavior is
            # unchanged, including separate verifier-position invocations.
            if fused_prefill_chunk:
                kv_read_tokens = context if position == 0 else 0
            else:
                kv_read_tokens = context + (
                    position if inter_invocation_positions else 0
                )
            lanes.append(
                _ServingInvocationLane(
                    item_index=item_index,
                    request_id=str(getattr(item, "request_id", "")),
                    phase=phase,
                    position=position,
                    context_tokens=context + position + 1,
                    kv_read_tokens=kv_read_tokens,
                    kv_append_tokens=int(position < append_tokens),
                    kv_materialized_tokens=int(position < materialized_tokens),
                    requires_logits=(position >= first_logit_position),
                    state_in_committed_prefix=(
                        not is_mtp or position < committed_tokens
                    ),
                    state_commit_boundary=(
                        is_mtp
                        and committed_tokens > 0
                        and position == committed_tokens - 1
                    ),
                    state_release_boundary=(
                        is_mtp and position == token_count - 1
                    ),
                )
            )
        item_lanes.append(tuple(lanes))
    return tuple(item_lanes)


def _serving_cohort_kv_scan_tokens(
    cohort: object, *, metadata_key: str = "llama_cpp_kv_scan_tokens",
) -> int:
    """Read a runtime-proven physical scan bound without changing KV owners."""

    metadata = getattr(cohort, "metadata", {})
    value = metadata.get(metadata_key) if isinstance(metadata, Mapping) else None
    if value is None:
        return 0
    items = tuple(getattr(cohort, "items", ()) or ())
    if metadata.get("llama_cpp_kv_scan_phase") == "prefill_rectangular":
        occupied = metadata.get("llama_cpp_kv_occupied_rows")
        append_rows = _serving_cohort_kv_shape(cohort)[2]
        if (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            or isinstance(occupied, bool) or not isinstance(occupied, int)
            or occupied < append_rows or occupied > value
            or str(getattr(cohort, "kind", "")) not in {"prefill", "mixed"}
            or not items
            or any(
                str(getattr(item, "phase", "")) not in {"prefill", "decode"}
                or int(getattr(item, "token_count", 0)) <= 0
                or (str(getattr(item, "phase", "")) == "decode"
                    and int(getattr(item, "token_count", 0)) != 1)
                or value < int(getattr(item, "context_tokens", 0))
                    + int(getattr(item, "token_count", 0))
                for item in items
            )
        ):
            raise ValueError("prefill KV scan requires final occupied rows and ordinary prefill/mixed items")
        return value
    if (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        or str(getattr(cohort, "kind", "")) != "decode" or not items
        or any(
            str(getattr(item, "phase", "")) != "decode"
            or int(getattr(item, "token_count", 0)) != 1
            or value < int(getattr(item, "context_tokens", 0)) + 1
            for item in items
        )
    ):
        raise ValueError("llama_cpp_kv_scan_tokens requires a positive bound for one-row decode items")
    return value


def _serving_invocation_groups(
    scenario: ScenarioConfig,
    cohort: object,
) -> Tuple[_ServingInvocationGroup, ...]:
    """Lower scheduler cohorts into capability-backed operator invocations.

    Stateless dense/full-attention models retain ordinary scheduler batching.
    Stateful linear-attention models fail closed: request slots remain distinct
    unless a stateful batch capability is declared, and dependent decode/MTP
    positions are serial unless a fused chunk capability is declared.
    Request-level prefill chunks remain intact, but are never fused across
    state owners without an explicit stateful-batch capability.
    """

    kind = str(getattr(cohort, "kind", "decode"))
    kv_scan_tokens = _serving_cohort_kv_scan_tokens(cohort)
    q4_view_tokens = _serving_cohort_kv_scan_tokens(
        cohort, metadata_key="llama_cpp_q4_kv_materialization_view_tokens_lower_bound",
    )
    if q4_view_tokens and (kv_scan_tokens or q4_view_tokens % 256):
        raise ValueError("Q4 materialization requires a separate 256-aligned view lower bound")
    has_stateful_linear_attention = any(
        layer.is_linear_attention for layer in _execution_layers(scenario)
    )
    fused_chunk = _serving_bool_capability(
        scenario,
        (
            "supports_fused_chunked_stateful_execution",
            "supports_fused_chunked_linear_attention",
            "supports_fused_chunked_gdn",
            "fused_chunked_stateful_execution",
        ),
    )
    stateful_batch = _serving_bool_capability(
        scenario,
        (
            "supports_batched_stateful_execution",
            "supports_batched_linear_attention",
            "supports_batched_gdn",
            "batched_stateful_execution",
        ),
    )
    split_positions = (
        has_stateful_linear_attention
        and kind in {"decode", "mtp"}
        and not fused_chunk
    )
    lanes_by_item = _serving_invocation_lanes(
        cohort,
        inter_invocation_positions=split_positions,
    )
    raw_metadata = getattr(cohort, "metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    prefill_scan = bool(kv_scan_tokens or q4_view_tokens) and (
        metadata.get("llama_cpp_kv_scan_phase") == "prefill_rectangular"
    )
    if prefill_scan and has_stateful_linear_attention and not (
        stateful_batch and scenario.workload.mtp is None
        and _serving_bool_capability(scenario, ("supports_equal_length_stateful_ubatches",))
    ):
        raise ValueError("stateful prefill KV scan requires explicit equal-length physical batches")
    scan_occupied_rows = (
        int(metadata["llama_cpp_kv_occupied_rows"])
        - sum(lane.kv_append_tokens for lanes in lanes_by_item for lane in lanes)
        if prefill_scan else 0
    )
    groups: List[_ServingInvocationGroup] = []
    scheduler = scenario.workload.scheduler
    logical_rows = (
        max(1, int(scheduler.max_num_batched_tokens))
        if scheduler is not None
        else max(
            1,
            sum(len(item_lanes) for item_lanes in lanes_by_item),
        )
    )
    physical_ubatch_rows = (
        logical_rows
        if scheduler is None or scheduler.max_num_ubatch_tokens is None
        else max(1, int(scheduler.max_num_ubatch_tokens))
    )

    def add_group(
        lanes: Sequence[_ServingInvocationLane],
        *,
        predecessor_group_id: Optional[str],
        batching_semantics: str,
    ) -> _ServingInvocationGroup:
        nonlocal scan_occupied_rows
        lane_rows = tuple(lanes)
        if not lane_rows:
            raise ValueError("backend invocation group must contain rows")
        last_group: Optional[_ServingInvocationGroup] = None
        predecessor = predecessor_group_id
        for row_start in range(0, len(lane_rows), physical_ubatch_rows):
            physical_lanes = lane_rows[row_start : row_start + physical_ubatch_rows]
            group_scan_tokens = kv_scan_tokens
            group_q4_view_tokens = q4_view_tokens
            if prefill_scan:
                if len(physical_lanes) >= 1024:
                    raise ValueError("prefill KV scan excludes mask-trim query groups >= 1024")
                if groups and predecessor != groups[-1].group_id:
                    raise ValueError("prefill KV scan requires serial physical invocation groups")
                scan_occupied_rows += sum(lane.kv_append_tokens for lane in physical_lanes)
                physical_span = min(
                    kv_scan_tokens or q4_view_tokens,
                    ((max(1, scan_occupied_rows) + 255) // 256) * 256,
                )
                group_scan_tokens = physical_span if kv_scan_tokens else 0
                group_q4_view_tokens = physical_span if q4_view_tokens else 0
            group = _ServingInvocationGroup(
                group_index=len(groups),
                kind=kind,
                lanes=physical_lanes,
                predecessor_group_id=predecessor,
                batching_semantics=batching_semantics,
                kv_scan_tokens=group_scan_tokens,
                q4_mma_view_tokens_lower_bound=group_q4_view_tokens,
            )
            groups.append(group)
            last_group = group
            predecessor = group.group_id
        assert last_group is not None
        return last_group

    if (
        has_stateful_linear_attention and stateful_batch
        and scenario.workload.mtp is None and kind in {"prefill", "mixed"}
        and _serving_bool_capability(scenario, ("supports_equal_length_stateful_ubatches",))
        and all(lane.phase in {"prefill", "decode"} for lanes in lanes_by_item for lane in lanes)
    ):
        owners = tuple(lanes[0].request_id for lanes in lanes_by_item if lanes)
        if not all(owners) or len(set(owners)) != len(owners):
            raise ValueError("equal-length stateful ubatches require distinct nonempty request owners")
        # Unified recurrent backends form rectangular sequence batches. A
        # decode row +127 prompt rows becomes (1+1), then126, not one M128.
        pending = [list(lanes) for lanes in lanes_by_item if lanes]
        predecessor = None
        while pending:
            active = pending[:physical_ubatch_rows]
            rows_per_owner = min(
                min(len(lanes) for lanes in active),
                physical_ubatch_rows // len(active),
            )
            rows = tuple(lane for lanes in active for lane in lanes[:rows_per_owner])
            group = add_group(rows, predecessor_group_id=predecessor,
                batching_semantics="explicit_equal_length_stateful_ubatch")
            predecessor = group.group_id
            for lanes in active:
                del lanes[:rows_per_owner]
            pending = [lanes for lanes in pending if lanes]
        return tuple(groups)

    if kind == "mixed":
        all_lanes = tuple(
            lane for item_lanes in lanes_by_item for lane in item_lanes
        )
        add_group(
            all_lanes,
            predecessor_group_id=None,
            batching_semantics="explicit_mixed_phase_physical_batch",
        )
        return tuple(groups)

    if not has_stateful_linear_attention:
        all_lanes = tuple(
            lane for item_lanes in lanes_by_item for lane in item_lanes
        )
        add_group(
            all_lanes,
            predecessor_group_id=None,
            batching_semantics="stateless_scheduler_batch",
        )
        return tuple(groups)

    if stateful_batch and (fused_chunk or kind not in {"decode", "mtp"}):
        all_lanes = tuple(
            lane for item_lanes in lanes_by_item for lane in item_lanes
        )
        add_group(
            all_lanes,
            predecessor_group_id=None,
            batching_semantics="explicit_stateful_batch_capability",
        )
        return tuple(groups)

    if stateful_batch and kind == "decode" and split_positions:
        nonempty_item_lanes = tuple(
            item_lanes for item_lanes in lanes_by_item if item_lanes
        )
        owner_request_ids = tuple(
            item_lanes[0].request_id for item_lanes in nonempty_item_lanes
        )
        can_batch_decode_waves = (
            bool(nonempty_item_lanes)
            and all(owner_request_ids)
            and len(owner_request_ids) == len(set(owner_request_ids))
            and all(
                lane.phase == "decode"
                for item_lanes in nonempty_item_lanes
                for lane in item_lanes
            )
        )
    else:
        nonempty_item_lanes = ()
        can_batch_decode_waves = False

    if can_batch_decode_waves:
        wave_lanes: List[Tuple[_ServingInvocationLane, ...]] = []
        max_positions = max(
            (len(item_lanes) for item_lanes in nonempty_item_lanes),
            default=0,
        )
        for position in range(max_positions):
            ready = tuple(
                item_lanes[position]
                for item_lanes in nonempty_item_lanes
                if position < len(item_lanes)
            )
            if ready:
                wave_lanes.append(ready)
        if physical_ubatch_rows > 1 and any(
            len(ready) > 1 for ready in wave_lanes
        ):
            predecessor_group_id: Optional[str] = None
            for ready in wave_lanes:
                group = add_group(
                    ready,
                    predecessor_group_id=predecessor_group_id,
                    batching_semantics=(
                        "explicit_stateful_decode_batch_capability"
                    ),
                )
                predecessor_group_id = group.group_id
            return tuple(groups)

    for item_lanes in lanes_by_item:
        if not item_lanes:
            continue
        if not split_positions:
            add_group(
                item_lanes,
                predecessor_group_id=None,
                batching_semantics=(
                    "request_local_fused_chunk"
                    if fused_chunk
                    else "request_local_prefill_chunk"
                ),
            )
            continue
        predecessor_group_id: Optional[str] = None
        for lane in item_lanes:
            group = add_group(
                (lane,),
                predecessor_group_id=predecessor_group_id,
                batching_semantics="serial_stateful_position",
            )
            predecessor_group_id = group.group_id
    return tuple(groups)


def _tag_serving_invocation_group_tasks(
    builder: _TaskBuilder,
    first_task_index: int,
    group: _ServingInvocationGroup,
) -> None:
    """Attach one auditable group identity to every task it lowered."""

    metadata = _serving_invocation_group_task_metadata(group)
    for index in range(first_task_index, len(builder.tasks)):
        task = builder.tasks[index]
        # ``_TaskBuilder.add`` gives every task its own plain dict.  Updating
        # that owned mapping preserves the exact insertion/update semantics of
        # ``{**task.metadata, **metadata}`` without allocating a replacement
        # TaskSpec and a second full metadata dict for every lowered task.
        # Keep a defensive fallback for callers that inject another Mapping
        # implementation into a private builder during tests.
        if isinstance(task.metadata, dict):
            task.metadata.update(metadata)
        else:  # pragma: no cover - _TaskBuilder.add always stores a dict
            builder.tasks[index] = replace(
                task,
                metadata={**dict(task.metadata), **metadata},
            )


def _serving_invocation_group_task_metadata(
    group: _ServingInvocationGroup,
) -> Dict[str, object]:
    """Return the exact top-level audit fields attached to every group task."""

    request_ids = group.request_ids
    metadata: Dict[str, object] = {
        "operator_invocation_group_id": group.group_id,
        "operator_invocation_group_index": group.group_index,
        "operator_invocation_group_kind": group.kind,
        "operator_invocation_group_token_batch": group.token_batch,
        "physical_ubatch_index": group.group_index,
        "physical_ubatch_rows": group.token_batch,
        "operator_invocation_group_request_ids": request_ids,
        "operator_invocation_group_lane_ids": tuple(
            lane.identity_metadata(group.kind)["lane_id"]
            for lane in group.lanes
        ),
        "operator_invocation_group_verifier_positions": tuple(
            lane.position for lane in group.lanes if lane.phase == "mtp"
        ),
        "operator_invocation_group_batching_semantics": (
            group.batching_semantics
        ),
    }
    runtime = group.linear_state_runtime()
    if group.kv_scan_tokens:
        metadata.update({
            "context_tokens_observed": group.context_tokens,
            "llama_cpp_kv_scan_tokens": group.kv_scan_tokens,
            "scan_bound": "occupied_rows_lower_bound",
            "kv_scan_partial_timing": True,
        })
    if group.q4_mma_view_tokens_lower_bound:
        metadata.update({
            "llama_cpp_q4_kv_materialization_view_tokens_lower_bound": group.q4_mma_view_tokens_lower_bound,
            "materialization_extent_completeness": "lower_bound",
        })
    if runtime is not None:
        metadata.update(runtime.audit_metadata())
    if len(request_ids) == 1:
        metadata["serving_request_id"] = request_ids[0]
    return metadata


def _serving_invocation_segment_binding(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    group: _ServingInvocationGroup,
    dependencies: Sequence[str],
) -> Optional[
    Tuple[
        "OrderedDict[Hashable, _TaskSegmentTemplate]",
        Hashable,
        int,
    ]
]:
    """Admit the narrow exact invocation shape used by local decode replay."""

    context = _active_compilation_context(scenario)
    state_runtime = group.linear_state_runtime()
    runtime_structure = (
        _linear_state_runtime_structure_identity(state_runtime)
        if state_runtime is not None
        else None
    )
    if (
        context is None
        or context.scenario is not scenario
        or not context.compiled_serving_invocation_segments
        or group.kind not in {"decode", "mtp"}
        or group.batching_semantics != "serial_stateful_position"
        or len(group.lanes) != 1
        or group.token_batch != 1
        or group.kv_append_tokens != 1
        or group.kv_materialized_tokens != 1
        or group.logit_token_batch != 1
        or group.context_tokens <= 0
        or group.kv_read_tokens <= 0
        or len(tuple(dependencies)) != 1
        or (group.kind == "decode" and state_runtime is not None)
        or (group.kind == "mtp" and runtime_structure is None)
    ):
        return None
    full_attention_layers = tuple(
        layer for layer in _execution_layers(scenario) if not layer.is_linear_attention
    )
    if not full_attention_layers:
        return None
    required_dynamic_attention_invocations = sum(
        len(plan.tp_group(plan.stage_for_layer(layer), 0))
        for layer in full_attention_layers
    )
    if required_dynamic_attention_invocations <= 0:
        return None
    rank_ids = tuple(
        dict.fromkeys(
            rank.rank
            for layer in _execution_layers(scenario)
            for rank in plan.tp_group(plan.stage_for_layer(layer), 0)
        )
    )
    input_components = tuple(
        builder.rank_value_component(dependencies, rank_id)
        for rank_id in rank_ids
    )
    cache = _task_segment_cache(
        context,
        ("serving_invocation_task_segment_cache_v1",),
    )
    if cache is None:
        return None
    key: Hashable = (
        "single_decode_position_v1",
        group.kind,
        group.batching_semantics,
        group.token_batch,
        group.kv_append_tokens,
        group.kv_materialized_tokens,
        group.logit_token_batch,
        group.kv_scan_tokens,
        group.q4_mma_view_tokens_lower_bound,
        runtime_structure,
        plan.tp_degree,
        plan.pp_degree,
        plan.ep_degree,
        input_components,
        len(tuple(dependencies)),
        builder.previous is not None,
        builder._last_coherent_dma_task is not None,
    )
    return cache, key, required_dynamic_attention_invocations


def _compile_or_replay_serving_invocation(
    builder: _TaskBuilder,
    scenario: ScenarioConfig,
    plan: ParallelPlan,
    router: TopologyRouter,
    group: _ServingInvocationGroup,
    *,
    phase: str,
    dependencies: Sequence[str],
) -> str:
    """Compile one group once, then emit its exact dynamic slots on replay."""

    attention_context_tokens = group.kv_scan_tokens or group.context_tokens
    attention_kv_read_tokens = max(group.kv_read_tokens, group.kv_scan_tokens)
    binding = _serving_invocation_segment_binding(
        builder,
        scenario,
        plan,
        group,
        dependencies,
    )
    if binding is not None:
        cache, cache_key, _required_dynamic = binding
        template = _task_segment_cache_get(cache, cache_key)
        if template is not None:
            task_metadata = _serving_invocation_group_task_metadata(group)
            replayed = template.replay(
                builder,
                prefix=phase,
                dependencies=dependencies,
                metadata_overrides={
                    "phase": phase,
                    "context_tokens_observed": group.context_tokens,
                    **task_metadata,
                },
                dynamic_context=_TaskSegmentDynamicReplayContext(
                    scenario=scenario,
                    context_tokens=attention_context_tokens,
                    kv_read_tokens=attention_kv_read_tokens,
                ),
            )
            if replayed is not None:
                return replayed

    first_task_index = len(builder.tasks)
    initial_counter = builder.counter
    initial_previous = builder.previous
    initial_dma = builder._last_coherent_dma_task
    selection = _final_output_selection(scenario, plan, group.token_batch,
        tuple(index for index, lane in enumerate(group.lanes) if lane.requires_logits))
    prepared, output_indices, upload_ids = _prepare_output_selection_inputs(
        builder, scenario, router, plan, selection, phase, dependencies)
    group_end = _compile_parallel_iteration(
        builder,
        scenario,
        plan,
        router,
        token_batch=group.token_batch,
        context_tokens=attention_context_tokens,
        kv_read_tokens=attention_kv_read_tokens,
        kv_append_tokens=group.kv_append_tokens,
        kv_materialized_tokens=group.kv_materialized_tokens,
        q4_mma_view_tokens_lower_bound=group.q4_mma_view_tokens_lower_bound,
        linear_state_runtime=group.linear_state_runtime(),
        output_selection=selection,
        output_indices_dependency=output_indices,
        phase=phase,
        dependencies=prepared,
    )
    if group.logit_token_batch > 0 or (selection is not None and selection.final_norm_rows > 0):
        group_end = _compile_parallel_lm_head(
            builder,
            scenario,
            plan,
            router,
            phase,
            (group_end,),
            token_batch=group.logit_token_batch,
            output_selection=selection,
            output_indices_dependency=output_indices,
        )
        if group.logit_token_batch > 0 and group.batching_semantics != "explicit_equal_length_stateful_ubatch":
            group_end = _add_host_visible_logits_sampling_commit(
                builder,
                scenario,
                plan,
                router,
                (group_end,),
                phase=phase,
                logit_rows=group.logit_token_batch,
                committed_rows=group.committed_logit_token_batch,
            )
    _bind_output_index_upload(builder, first_task_index, upload_ids)
    _tag_serving_invocation_group_tasks(
        builder,
        first_task_index,
        group,
    )
    if binding is not None:
        cache, cache_key, required_dynamic = binding
        context = _active_compilation_context(scenario)
        assert context is not None
        captured = _TaskSegmentTemplate.capture(
            builder,
            first_task_index=first_task_index,
            source_prefix=phase,
            source_counter_before=initial_counter,
            source_dependencies=dependencies,
            source_initial_previous=initial_previous,
            source_initial_dma=initial_dma,
            terminal_task_id=group_end,
            source_phase=phase,
            required_dynamic_attention_invocations=required_dynamic,
        )
        if captured is not None:
            _task_segment_cache_put(context, cache, cache_key, captured)
    return group_end


def _ordered_residency_accesses(
    tasks: Sequence[TaskSpec],
    transient_envelopes: Sequence[Mapping[str, object]] = (),
) -> Tuple[Mapping[str, object], ...]:
    """Return physical weight reads plus cohort-scoped transient accesses.

    Route tasks still carry the request/cohort-derived
    ``weight_read_invocation_id`` emitted while the graph is being built.  A
    serving invocation group, however, is the physical backend launch.  Merge
    accesses inside that boundary by physical owner/view and interval so
    packing more logical request lanes into the same launch does not multiply
    residency touches.  Staged-weight lifecycle events remain ordered barriers
    and are never folded into this read-only union.
    """

    ordered_entries: List[Tuple[int, Mapping[str, object]]] = []
    physical_groups: OrderedDict[
        Tuple[object, ...],
        List[
            Tuple[
                int,
                Dict[str, object],
                Optional[Tuple[int, int]],
                str,
            ]
        ],
    ] = OrderedDict()
    seen_invocations: Set[str] = set()
    lifecycle_epoch = 0
    for task_index, task in enumerate(tasks):
        metadata = task.metadata
        staged_operation = metadata.get("staged_weight_operation")
        if staged_operation is not None:
            operation = str(staged_operation).strip().lower()
            if operation not in {"register", "read", "release"}:
                raise ValueError(
                    "unsupported staged weight lifecycle operation: {}"
                    .format(operation)
                )
            allocation_id = str(
                metadata.get("staged_weight_allocation_id")
                or metadata.get("allocation_id")
                or ""
            )
            byte_count = max(
                0,
                int(
                    metadata.get(
                        "staged_weight_bytes",
                        metadata.get("temporary_weight_bytes", 0),
                    )
                    or 0
                ),
            )
            if not allocation_id or byte_count <= 0:
                raise ValueError(
                    "staged weight lifecycle requires allocation id and bytes"
                )
            staged_access: Dict[str, object] = {
                "kind": "staged_weight",
                "operation": operation,
                "lifecycle": "temporary",
                "allocation_id": allocation_id,
                "tensor_id": allocation_id,
                "requested_tensor_id": allocation_id,
                "requested_tensor": allocation_id,
                "requested_view": allocation_id,
                "canonical_owner_id": allocation_id,
                "canonical_owner": allocation_id,
                "target_id": allocation_id,
                "physical_tensor": allocation_id,
                "byte_count": byte_count,
                "size_bytes": byte_count,
                "offset_bytes": 0,
                "read_only": True,
                "fully_resident": True,
                "h2d_already_charged": True,
                "release_semantics": "clean_discard",
                "clean_eviction_service": "free_discard",
                "dirty_writeback": False,
                "weight_owner_component_id": metadata.get(
                    "weight_owner_component_id"
                ),
                "backing_component_id": metadata.get(
                    "weight_source_component"
                ),
                "target_component_id": metadata.get(
                    "weight_target_component"
                ),
                "residency_component_id": metadata.get(
                    "residency_component_id"
                ),
                "weight_read_invocation_id": metadata.get(
                    "weight_read_invocation_id"
                ),
                "consumer_task_ids": (task.task_id,),
            }
            request_id = metadata.get("serving_request_id")
            if request_id is not None:
                staged_access["request_id"] = str(request_id)
            request_ids = metadata.get(
                "operator_invocation_group_request_ids"
            )
            if isinstance(request_ids, (list, tuple)):
                staged_access["request_ids"] = tuple(
                    str(item) for item in request_ids
                )
            group_id = metadata.get("operator_invocation_group_id")
            if group_id is not None:
                staged_access["operator_invocation_group_id"] = str(
                    group_id
                )
            ordered_entries.append((task_index, staged_access))
            lifecycle_epoch += 1
            continue
        if metadata.get("event_kind") not in {
            "model_weight_read",
            "model_weight_access",
        }:
            continue
        invocation_id = str(
            metadata.get("weight_read_invocation_id") or task.task_id
        )
        if invocation_id in seen_invocations:
            continue
        requested = str(
            metadata.get("weight_tensor_id")
            or metadata.get("tensor_id")
            or metadata.get("logical_weight_tensor")
            or metadata.get("tensor")
            or ""
        )
        canonical_owner = str(
            metadata.get("logical_weight_tensor")
            or metadata.get("tensor")
            or requested
        )
        physical_tensor = str(metadata.get("tensor") or canonical_owner)
        byte_count = max(
            0,
            int(
                metadata.get(
                    "weight_read_bytes", metadata.get("bytes", 0)
                )
                or 0
            ),
        )
        access: Dict[str, object] = {
            "kind": "model_weight",
            "operation": "read",
            "tensor_id": requested,
            "requested_tensor_id": requested,
            "requested_tensor": requested,
            "requested_view": requested,
            "canonical_owner_id": canonical_owner,
            "canonical_owner": canonical_owner,
            "target_id": canonical_owner,
            "physical_tensor": physical_tensor,
            "invocation_id": invocation_id,
            "weight_read_invocation_id": invocation_id,
            "byte_count": byte_count,
            "consumer_task_ids": (task.task_id,),
        }
        request_id = metadata.get("serving_request_id")
        if request_id is not None:
            access["request_id"] = str(request_id)
        request_ids = metadata.get("operator_invocation_group_request_ids")
        if request_ids is None:
            request_ids = metadata.get(
                "mtp_proposer_invocation_group_request_ids"
            )
        if isinstance(request_ids, (list, tuple)):
            access["request_ids"] = tuple(str(item) for item in request_ids)
        group_id = metadata.get("operator_invocation_group_id")
        if group_id is None:
            group_id = metadata.get("mtp_proposer_invocation_group_id")
        if group_id is not None:
            access["operator_invocation_group_id"] = str(group_id)
        proposer_group_id = metadata.get("mtp_proposer_invocation_group_id")
        if proposer_group_id is not None:
            access["mtp_proposer_invocation_group_id"] = str(
                proposer_group_id
            )
        lifecycle = metadata.get("weight_lifecycle_mode")
        if lifecycle is not None:
            access["lifecycle"] = str(lifecycle)
        backing_component = metadata.get("weight_source_component")
        if backing_component is not None:
            access["backing_component_id"] = str(backing_component)
        target_component = metadata.get("weight_target_component")
        if target_component is not None:
            access["target_component_id"] = str(target_component)
        physical_group_id = metadata.get("operator_invocation_group_id")
        if physical_group_id is None:
            physical_group_id = metadata.get("mtp_proposer_invocation_group_id")
        if physical_group_id is None:
            # Offline compilation and older callers have no physical launch
            # boundary.  Preserve their exact invocation-level behavior.
            seen_invocations.add(invocation_id)
            ordered_entries.append((task_index, access))
            continue

        explicit_offset = None
        if "weight_read_offset_bytes" in metadata:
            explicit_offset = metadata.get("weight_read_offset_bytes")
        elif "offset_bytes" in metadata:
            explicit_offset = metadata.get("offset_bytes")
        explicit_size = None
        if "weight_read_size_bytes" in metadata:
            explicit_size = metadata.get("weight_read_size_bytes")
        elif "size_bytes" in metadata:
            explicit_size = metadata.get("size_bytes")
        interval: Optional[Tuple[int, int]] = None
        if explicit_offset is not None or explicit_size is not None:
            try:
                offset_bytes = max(0, int(explicit_offset or 0))
                size_bytes = max(
                    0,
                    int(
                        byte_count
                        if explicit_size is None
                        else explicit_size or 0
                    ),
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "physical weight range offset/size must be integers"
                ) from exc
            interval = (offset_bytes, offset_bytes + size_bytes)

        source_component = str(
            metadata.get("weight_source_component")
            or metadata.get("weight_owner_component_id")
            or ""
        )
        target_component_id = str(
            metadata.get("weight_target_component")
            or metadata.get("residency_component_id")
            or ""
        )
        group_key = (
            lifecycle_epoch,
            str(physical_group_id),
            canonical_owner,
            physical_tensor,
            source_component,
            target_component_id,
        )
        physical_groups.setdefault(group_key, []).append(
            (task_index, access, interval, invocation_id)
        )

    def merged_physical_access(
        group_key: Tuple[object, ...],
        members: Sequence[
            Tuple[int, Dict[str, object], Optional[Tuple[int, int]], str]
        ],
        *,
        interval: Optional[Tuple[int, int]],
    ) -> Tuple[int, Mapping[str, object]]:
        first_index, first_access, _, _ = min(members, key=lambda item: item[0])
        merged = dict(first_access)
        request_ids: List[str] = []
        requested_tensor_ids: List[str] = []
        logical_invocation_ids: List[str] = []
        consumer_task_ids: List[str] = []
        for _, member, _, logical_invocation_id in members:
            raw_request_ids = member.get("request_ids", ())
            if isinstance(raw_request_ids, (list, tuple)):
                request_ids.extend(str(item) for item in raw_request_ids)
            request_id = member.get("request_id")
            if request_id is not None:
                request_ids.append(str(request_id))
            requested_tensor_ids.append(str(member["requested_tensor_id"]))
            logical_invocation_ids.append(logical_invocation_id)
            raw_consumer_ids = member.get("consumer_task_ids", ())
            if isinstance(raw_consumer_ids, (list, tuple)):
                consumer_task_ids.extend(
                    str(item) for item in raw_consumer_ids if item
                )
        unique_request_ids = tuple(dict.fromkeys(request_ids))
        if unique_request_ids:
            merged["request_ids"] = unique_request_ids
        if len(unique_request_ids) > 1:
            merged.pop("request_id", None)
        unique_requested_ids = tuple(dict.fromkeys(requested_tensor_ids))
        if len(unique_requested_ids) > 1:
            merged["requested_tensor_ids"] = unique_requested_ids
        unique_logical_ids = tuple(dict.fromkeys(logical_invocation_ids))
        if len(unique_logical_ids) > 1:
            merged["logical_weight_read_invocation_ids"] = unique_logical_ids
        merged["consumer_task_ids"] = tuple(
            dict.fromkeys(consumer_task_ids)
        )

        physical_group_id = str(group_key[1])
        owner_id = str(group_key[2])
        source_component = str(group_key[4]) or "unknown-source"
        target_component_id = str(group_key[5]) or "unknown-target"
        range_identity = "whole"
        if interval is None:
            merged["byte_count"] = max(
                int(member[1].get("byte_count", 0) or 0)
                for member in members
            )
            merged.pop("offset_bytes", None)
            merged.pop("size_bytes", None)
        else:
            offset_bytes, end_bytes = interval
            size_bytes = max(0, end_bytes - offset_bytes)
            merged["offset_bytes"] = offset_bytes
            merged["size_bytes"] = size_bytes
            merged["byte_count"] = size_bytes
            range_identity = "range-{}-{}".format(offset_bytes, end_bytes)
        physical_invocation_id = (
            "{}:weight:{}:source-{}:target-{}:{}"
        ).format(
            physical_group_id,
            owner_id,
            source_component,
            target_component_id,
            range_identity,
        )
        merged["invocation_id"] = physical_invocation_id
        merged["weight_read_invocation_id"] = physical_invocation_id
        merged["physical_weight_invocation_id"] = physical_invocation_id
        merged["physical_invocation_group_id"] = physical_group_id
        return first_index, merged

    for group_key, candidates in physical_groups.items():
        # A whole-owner access dominates explicit subranges because the
        # physical allocation contract provides no narrower view for it.
        if any(interval is None for _, _, interval, _ in candidates):
            ordered_entries.append(
                merged_physical_access(group_key, candidates, interval=None)
            )
            continue
        ranged = sorted(
            candidates,
            key=lambda item: (
                item[2][0] if item[2] is not None else 0,
                item[2][1] if item[2] is not None else 0,
                item[0],
            ),
        )
        union_members: List[
            Tuple[int, Dict[str, object], Optional[Tuple[int, int]], str]
        ] = []
        union_start = 0
        union_end = 0
        for candidate in ranged:
            assert candidate[2] is not None
            start_bytes, end_bytes = candidate[2]
            if not union_members:
                union_members = [candidate]
                union_start, union_end = start_bytes, end_bytes
                continue
            if start_bytes <= union_end:
                union_members.append(candidate)
                union_end = max(union_end, end_bytes)
                continue
            ordered_entries.append(
                merged_physical_access(
                    group_key,
                    union_members,
                    interval=(union_start, union_end),
                )
            )
            union_members = [candidate]
            union_start, union_end = start_bytes, end_bytes
        if union_members:
            ordered_entries.append(
                merged_physical_access(
                    group_key,
                    union_members,
                    interval=(union_start, union_end),
                )
            )

    accesses = [
        access
        for _, access in sorted(ordered_entries, key=lambda item: item[0])
    ]
    effective_transient_envelopes = tuple(transient_envelopes)
    if not effective_transient_envelopes:
        discovered: List[Mapping[str, object]] = []
        seen_allocations: Set[str] = set()
        for task in tasks:
            envelope = task.metadata.get("transient_residency_envelope")
            if not isinstance(envelope, Mapping):
                continue
            allocation_id = str(envelope.get("allocation_id") or "")
            if not allocation_id or allocation_id in seen_allocations:
                continue
            seen_allocations.add(allocation_id)
            discovered.append(envelope)
        effective_transient_envelopes = tuple(discovered)
    transient_accesses: List[Mapping[str, object]] = []
    for envelope in effective_transient_envelopes:
        selected_bytes = max(
            0, int(envelope.get("selected_bytes", 0) or 0)
        )
        if selected_bytes <= 0:
            continue
        access = dict(envelope)
        allocation_id = str(access.get("allocation_id") or "")
        access.update(
            {
                "operation": "write",
                "tensor_id": allocation_id,
                "requested_tensor_id": allocation_id,
                "requested_tensor": allocation_id,
                "requested_view": allocation_id,
                "canonical_owner_id": allocation_id,
                "canonical_owner": allocation_id,
                "target_id": allocation_id,
                "physical_tensor": allocation_id,
                "byte_count": selected_bytes,
                "size_bytes": selected_bytes,
                "offset_bytes": 0,
            }
        )
        transient_accesses.append(access)
    # The consolidated buffer is live for the backend graph, so register it
    # before the first tensor read.  Serving releases it only in the cohort's
    # final lifecycle boundary.
    return tuple(transient_accesses + accesses)


def _declared_transient_residency_granule(
    scenario: ScenarioConfig,
) -> Tuple[int, str]:
    """Return an explicitly declared residency granule, or exact bytes.

    The placement-owned runtime allocation contract is the closest planner
    input to the compiled physical-pool policy.  Serving-runtime metadata is
    the public fallback, with workload values overriding placement values.
    No KV page size or platform constant is inferred when neither declaration
    is present.
    """

    placement_metadata = getattr(scenario.placement, "metadata", {})
    workload_metadata = getattr(scenario.workload, "metadata", {})
    if isinstance(placement_metadata, Mapping):
        ledger = placement_metadata.get("capacity_ledger")
        if isinstance(ledger, Mapping):
            contract = ledger.get("runtime_allocation_contract")
            if isinstance(contract, Mapping):
                raw = contract.get("residency_granule_bytes")
                if (
                    not isinstance(raw, bool)
                    and isinstance(raw, int)
                    and raw > 0
                ):
                    return raw, (
                        "placement.metadata.capacity_ledger."
                        "runtime_allocation_contract."
                        "residency_granule_bytes"
                    )

    selected: Optional[int] = None
    selected_source = ""
    for owner, metadata in (
        ("placement", placement_metadata),
        ("workload", workload_metadata),
    ):
        if not isinstance(metadata, Mapping):
            continue
        runtime = metadata.get("serving_runtime")
        if not isinstance(runtime, Mapping):
            continue
        raw = runtime.get("residency_granule_bytes")
        if (
            not isinstance(raw, bool)
            and isinstance(raw, int)
            and raw > 0
        ):
            selected = raw
            selected_source = (
                "{}.metadata.serving_runtime."
                "residency_granule_bytes"
            ).format(owner)
    if selected is not None:
        return selected, selected_source
    return 1, "exact_unrounded_no_declared_granule"


def _transient_nonnegative_int(
    values: Mapping[str, object], key: str
) -> Optional[int]:
    raw = values.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


def _transient_tensor_id(
    metadata: Mapping[str, object],
    *,
    component_id: str,
    operator_id: str,
    side: str,
) -> str:
    """Resolve an explicitly declared tensor owner before using a unique id."""

    raw_tensor = metadata.get("{}_tensor_id".format(side))
    aliases = metadata.get("tensor_aliases")
    alias_owner = None
    if isinstance(aliases, Mapping):
        try:
            alias_owner = aliases.get(raw_tensor)
        except TypeError:
            alias_owner = None
    candidates = (
        metadata.get("{}_canonical_tensor_id".format(side)),
        metadata.get("canonical_{}_tensor_id".format(side)),
        metadata.get("{}_tensor_owner_id".format(side)),
        metadata.get("{}_tensor_alias_of".format(side)),
        alias_owner,
        raw_tensor,
    )
    declared = next(
        (
            str(value)
            for value in candidates
            if value is not None and str(value).strip()
        ),
        None,
    )
    suffix = declared or "{}:{}".format(operator_id, side)
    return "{}:{}".format(component_id, suffix)


def _transient_operator_template_id(
    metadata: Mapping[str, object], operator_id: str
) -> str:
    """Return one cohort-independent logical operator-template identity."""

    declared = metadata.get("model_operator_id") or metadata.get(
        "operator_id"
    )
    if declared is not None and str(declared).strip():
        local_id = str(declared)
    else:
        rank_suffix = re.search(r"\.rank\d+\.(.+)$", operator_id)
        local_id = (
            rank_suffix.group(1)
            if rank_suffix is not None
            else operator_id
        )
    layer_id = str(metadata.get("layer_id") or "global")
    rank_id = str(metadata.get("rank") or "0")
    return "{}:rank{}:{}".format(layer_id, rank_id, local_id)


def _transient_source_row_count(
    metadata: Mapping[str, object], realized_rows: int
) -> int:
    for key in (
        "operator_invocation_group_token_batch",
        "draft_step_lanes",
    ):
        value = _transient_nonnegative_int(metadata, key)
        if value is not None and value > 0:
            return min(realized_rows, value)
    return 0


def _transient_task_geometry(
    scenario: ScenarioConfig,
    task: TaskSpec,
    *,
    row_count: int,
) -> Optional[Mapping[str, object]]:
    """Extract one source-grounded GPU activation fact from one task."""

    metadata = task.metadata
    cost_model = metadata.get("cost_model")
    if not isinstance(cost_model, Mapping):
        return None
    component_id = str(metadata.get("target_component") or "").strip()
    if not component_id:
        return None
    try:
        if _kind(_component(scenario, component_id)) != "gpu":
            return None
    except (KeyError, TypeError, ValueError):
        return None

    event_kind = str(metadata.get("event_kind") or "").strip().lower()
    if (
        event_kind.startswith("kv_")
        or event_kind.startswith("linear_state_")
        or event_kind.startswith("model_weight_")
        or metadata.get("state_persistence") is not None
    ):
        return None

    model = str(cost_model.get("model") or "")
    geometry_source = ""
    persistent_input_bytes = 0
    persistent_output_bytes = 0
    if model == "gpu_fused_attention_v3":
        read_bytes = _transient_nonnegative_int(cost_model, "read_bytes")
        write_bytes = _transient_nonnegative_int(cost_model, "write_bytes")
        kv_read_bytes = _transient_nonnegative_int(
            cost_model, "kv_read_bytes"
        )
        if read_bytes is None or write_bytes is None or kv_read_bytes is None:
            return None
        if kv_read_bytes > read_bytes:
            return None
        input_bytes = read_bytes - kv_read_bytes
        output_bytes = write_bytes
        persistent_input_bytes = kv_read_bytes
        geometry_source = "fused_attention_backing_io_excluding_persistent_kv"
    elif (
        "activation_bytes" in cost_model
        or "output_bytes" in cost_model
    ):
        activation_bytes = _transient_nonnegative_int(
            cost_model, "activation_bytes"
        )
        raw_output_bytes = _transient_nonnegative_int(
            cost_model, "output_bytes"
        )
        if activation_bytes is None or raw_output_bytes is None:
            return None
        declared_persistent_outputs = tuple(
            value
            for value in (
                _transient_nonnegative_int(
                    metadata, "modeled_kv_write_bytes"
                ),
                _transient_nonnegative_int(
                    metadata, "kv_materialized_bytes"
                ),
                _transient_nonnegative_int(
                    metadata, "persistent_output_bytes"
                ),
            )
            if value is not None
        )
        persistent_output_bytes = max(
            declared_persistent_outputs, default=0
        )
        if persistent_output_bytes > raw_output_bytes:
            return None
        input_bytes = activation_bytes
        output_bytes = raw_output_bytes - persistent_output_bytes
        geometry_source = "gemm_activation_and_output_bytes"
    elif isinstance(metadata.get("output_selection_tensor_geometry"), Mapping):
        geometry = metadata["output_selection_tensor_geometry"]
        input_bytes = _transient_nonnegative_int(geometry, "input_bytes")
        output_bytes = _transient_nonnegative_int(geometry, "output_bytes")
        if input_bytes is None or output_bytes is None:
            raise ValueError("output selection tensor capacity is invalid")
        geometry_source = "get_rows_explicit_objects_conservative_cohort_envelope"
    elif isinstance(metadata.get("qwen35_tensor_geometry"), Mapping):
        declared_geometry = metadata["qwen35_tensor_geometry"]
        input_bytes = _transient_nonnegative_int(declared_geometry, "input_bytes")
        output_bytes = _transient_nonnegative_int(declared_geometry, "output_bytes")
        if input_bytes is None or output_bytes is None:
            raise ValueError("Qwen3.5 source tensor allocation geometry is invalid")
        # These are object capacities, not cumulative source read requests.
        # The existing cohort envelope remains conservative; alias and distinct
        # source objects remain separately recorded for an exact allocator model.
        geometry_source = "qwen35_explicit_objects_conservative_cohort_envelope"
    else:
        # Generic typed-kernel read/write counters are cumulative off-chip
        # traffic over the invocation, not a simultaneous allocation
        # footprint.  Treating them as input/output tensors multiplies a
        # streaming kernel's resident working set by its token/row count (for
        # example recurrent scan state traffic).  Only kernels with an
        # explicit activation/output geometry, or the dedicated fused-
        # attention contract above, may contribute a transient allocation.
        return None

    if input_bytes <= 0 and output_bytes <= 0:
        return None
    operator_id = str(
        metadata.get("op_name")
        or metadata.get("operator_id")
        or ""
    ).strip()
    if not operator_id:
        return None
    input_tensor_id = _transient_tensor_id(
        metadata,
        component_id=component_id,
        operator_id=operator_id,
        side="input",
    )
    output_tensor_id = _transient_tensor_id(
        metadata,
        component_id=component_id,
        operator_id=operator_id,
        side="output",
    )
    invocation_id = str(
        metadata.get("operator_invocation_group_id")
        or metadata.get("mtp_proposer_invocation_group_id")
        or "ungrouped"
    )
    operator_template_id = _transient_operator_template_id(
        metadata, operator_id
    )
    source_row_count = _transient_source_row_count(metadata, row_count)
    return {
        "component_id": component_id,
        "operator_id": operator_id,
        "operator_template_id": operator_template_id,
        "operator_invocation_group_id": invocation_id,
        "kind": "activation",
        "lifecycle": "temporary",
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "persistent_input_bytes_excluded": persistent_input_bytes,
        "persistent_output_bytes_excluded": persistent_output_bytes,
        "lower_bound_bytes": max(input_bytes, output_bytes),
        "upper_bound_bytes": input_bytes + output_bytes,
        "bound_kind": "source_grounded_non_aliasing_unproven",
        "geometry_source": geometry_source,
        "row_count": row_count,
        "row_cap": row_count,
        "source_row_count": source_row_count,
        "input_tensor_id": input_tensor_id,
        "output_tensor_id": output_tensor_id,
        "source_tensor_ids": tuple(
            dict.fromkeys((input_tensor_id, output_tensor_id))
        ),
        "source_task_ids": (task.task_id,),
    }


def _transient_operator_facts(
    scenario: ScenarioConfig,
    tasks: Sequence[TaskSpec],
    *,
    row_count: int,
) -> Tuple[Mapping[str, object], ...]:
    """Deduplicate CostEstimate phase copies into unique operator facts."""

    grouped: "OrderedDict[Tuple[str, str, str], Dict[str, object]]" = (
        OrderedDict()
    )
    for task in tasks:
        fact = _transient_task_geometry(
            scenario, task, row_count=row_count
        )
        if fact is None:
            continue
        key = (
            str(fact["component_id"]),
            str(fact["operator_invocation_group_id"]),
            str(fact["operator_id"]),
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = dict(fact)
            continue
        source_task_ids = tuple(existing["source_task_ids"])
        task_id = str(tuple(fact["source_task_ids"])[0])
        if task_id not in source_task_ids:
            existing["source_task_ids"] = source_task_ids + (task_id,)
        existing["input_bytes"] = max(
            int(existing["input_bytes"]), int(fact["input_bytes"])
        )
        existing["output_bytes"] = max(
            int(existing["output_bytes"]), int(fact["output_bytes"])
        )
        existing["lower_bound_bytes"] = max(
            int(existing["input_bytes"]), int(existing["output_bytes"])
        )
        existing["upper_bound_bytes"] = (
            int(existing["input_bytes"]) + int(existing["output_bytes"])
        )
    return tuple(grouped.values())


def _align_transient_bytes(byte_count: int, granule: int) -> int:
    if byte_count <= 0:
        return 0
    return ((byte_count + granule - 1) // granule) * granule


def _transient_residency_envelopes(
    scenario: ScenarioConfig,
    tasks: Sequence[TaskSpec],
    *,
    cohort_id: str,
    row_count: int,
) -> Tuple[
    Tuple[Mapping[str, object], ...],
    Tuple[Mapping[str, object], ...],
]:
    """Build one auditable transient peak envelope per GPU component."""

    operator_facts = _transient_operator_facts(
        scenario, tasks, row_count=row_count
    )
    granule, granule_source = _declared_transient_residency_granule(
        scenario
    )
    by_component: "OrderedDict[str, List[Mapping[str, object]]]" = (
        OrderedDict()
    )
    for fact in operator_facts:
        by_component.setdefault(str(fact["component_id"]), []).append(fact)

    envelopes: List[Mapping[str, object]] = []
    for component_id, facts in by_component.items():
        template_facts: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        for fact in facts:
            template_id = str(fact["operator_template_id"])
            consolidated = template_facts.setdefault(
                template_id,
                {
                    "operator_template_id": template_id,
                    "row_count": 0,
                    "row_cap": row_count,
                    "input_bytes": 0,
                    "output_bytes": 0,
                    "source_operator_ids": (),
                    "source_task_ids": (),
                },
            )
            source_rows = int(fact["source_row_count"])
            current_rows = int(consolidated["row_count"])
            if source_rows > 0 and current_rows >= row_count:
                continue
            if source_rows > 0 and current_rows + source_rows > row_count:
                # A partially represented invocation has no byte geometry
                # contract, so fail closed rather than prorating its bytes.
                continue
            consolidated["row_count"] = current_rows + source_rows
            consolidated["input_bytes"] = int(
                consolidated["input_bytes"]
            ) + int(fact["input_bytes"])
            consolidated["output_bytes"] = int(
                consolidated["output_bytes"]
            ) + int(fact["output_bytes"])
            consolidated["source_operator_ids"] = tuple(
                dict.fromkeys(
                    tuple(consolidated["source_operator_ids"])
                    + (str(fact["operator_id"]),)
                )
            )
            consolidated["source_task_ids"] = tuple(
                dict.fromkeys(
                    tuple(consolidated["source_task_ids"])
                    + tuple(fact["source_task_ids"])
                )
            )
        for consolidated in template_facts.values():
            consolidated["lower_bound_bytes"] = max(
                int(consolidated["input_bytes"]),
                int(consolidated["output_bytes"]),
            )
            consolidated["bound_kind"] = (
                "source_grounded_row_consolidated_non_aliasing_unproven"
            )
        exact_lower = max(
            (
                int(consolidated["lower_bound_bytes"])
                for consolidated in template_facts.values()
            ),
            default=0,
        )
        canonical_tensor_bytes: "OrderedDict[str, int]" = OrderedDict()
        for fact in facts:
            for tensor_key, size_key in (
                ("input_tensor_id", "input_bytes"),
                ("output_tensor_id", "output_bytes"),
            ):
                tensor_id = str(fact[tensor_key])
                canonical_tensor_bytes[tensor_id] = max(
                    canonical_tensor_bytes.get(tensor_id, 0),
                    int(fact[size_key]),
                )
        exact_upper = sum(canonical_tensor_bytes.values())
        if exact_lower <= 0 or exact_upper < exact_lower:
            continue
        lower = _align_transient_bytes(exact_lower, granule)
        upper = _align_transient_bytes(exact_upper, granule)
        source_task_ids = tuple(
            dict.fromkeys(
                str(task_id)
                for fact in facts
                for task_id in tuple(fact["source_task_ids"])
            )
        )
        source_tensor_ids = tuple(canonical_tensor_bytes)
        source_operator_ids = tuple(
            dict.fromkeys(str(fact["operator_id"]) for fact in facts)
        )
        envelopes.append(
            {
                "allocation_id": (
                    "runtime.transient.cohort.{}.{}".format(
                        cohort_id, component_id
                    )
                ),
                "component_id": component_id,
                "target_component_id": component_id,
                "kind": "activation",
                "lifecycle": "temporary",
                "lifecycle_scope": "cohort",
                "cohort_id": cohort_id,
                "lower_bound_bytes": lower,
                "upper_bound_bytes": upper,
                "selected_bytes": lower,
                "exact_lower_bound_bytes": exact_lower,
                "exact_upper_bound_bytes": exact_upper,
                "bound_kind": "source_grounded_lower_bound",
                "selection_policy": (
                    "max_logical_operator_template_summed_rows_"
                    "max_input_output"
                ),
                "upper_bound_policy": (
                    "sum_unique_canonical_tensor_facts"
                ),
                "alias_policy": (
                    "explicit_canonical_owner_only_otherwise_distinct"
                ),
                "row_count": row_count,
                "row_cap": row_count,
                "residency_granule_bytes": granule,
                "residency_granule_source": granule_source,
                "alignment_slack_bytes": lower - exact_lower,
                "source_tensor_ids": source_tensor_ids,
                "source_task_ids": source_task_ids,
                "source_operator_ids": source_operator_ids,
                "source_operator_template_ids": tuple(template_facts),
                "operator_template_peak_facts": tuple(
                    dict(fact) for fact in template_facts.values()
                ),
                "source_operator_count": len(facts),
                "resource_accounting": (
                    "cohort_transient_peak_envelope"
                ),
                "excluded_persistent_kinds": (
                    "model_weight",
                    "kv_cache",
                    "recurrent_state",
                    "fused_attention_onchip_sram",
                ),
            }
        )
    return tuple(operator_facts), tuple(envelopes)


def _annotate_transient_residency_facts(
    tasks: Sequence[TaskSpec],
    operator_facts: Sequence[Mapping[str, object]],
    envelopes: Sequence[Mapping[str, object]],
) -> Tuple[TaskSpec, ...]:
    fact_by_task = {
        str(tuple(fact["source_task_ids"])[0]): fact
        for fact in operator_facts
        if tuple(fact["source_task_ids"])
    }
    envelope_by_task = {
        str(tuple(envelope["source_task_ids"])[0]): envelope
        for envelope in envelopes
        if tuple(envelope["source_task_ids"])
    }
    annotated: List[TaskSpec] = []
    for task in tasks:
        fact = fact_by_task.get(task.task_id)
        envelope = envelope_by_task.get(task.task_id)
        if fact is None and envelope is None:
            annotated.append(task)
            continue
        retained_metadata: Dict[str, object] = {}
        if fact is not None:
            retained_metadata["transient_residency_fact"] = dict(fact)
        if envelope is not None:
            retained_metadata["transient_residency_envelope"] = dict(envelope)
        annotated.append(
            replace(
                task,
                metadata={
                    **dict(task.metadata),
                    **retained_metadata,
                },
            )
        )
    return tuple(annotated)


def _serving_cohort_cache_key(cohort: object) -> Tuple[object, ...]:
    """Return every cohort field that can affect topology-aware cost.

    Request identities are retained because invocation-group metadata exposes
    them and stateful request slots may no longer share one backend invocation.
    Cohort numbering remains a task-id namespace: residency accesses are
    consumed anew per returned cohort and never globally deduplicated by their
    audit id.  Per-item shapes are exact and the cache performs no interpolation.
    """

    kind = str(getattr(cohort, "kind", "decode"))

    def item_logit_tokens_key(item: object) -> Optional[int]:
        explicit = getattr(item, "logit_tokens", None)
        if explicit is None:
            return None
        return int(explicit)

    if kind in {
        "kv_swap_out",
        "kv_swap_in",
        "linear_state_swap_out",
        "linear_state_swap_in",
    }:
        raw_metadata = getattr(cohort, "metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        return (
            "transfer",
            kind,
            str(metadata.get("source_component", "")),
            str(metadata.get("target_component", "")),
            int(metadata.get("byte_count", 0)),
            int(metadata.get("page_count", 0)),
        )
    items = tuple(getattr(cohort, "items", ()) or ())
    item_shapes = tuple(
        (
            str(getattr(item, "request_id", "")),
            str(getattr(item, "phase", kind)),
            max(0, int(getattr(item, "token_count", 0))),
            max(0, int(getattr(item, "context_tokens", 0))),
            max(
                0,
                int(
                    getattr(item, "token_count", 0)
                    if getattr(item, "kv_append_tokens", None) is None
                    else getattr(item, "kv_append_tokens")
                ),
            ),
            max(
                0,
                int(
                    (
                        getattr(item, "token_count", 0)
                        if getattr(item, "kv_append_tokens", None) is None
                        else getattr(item, "kv_append_tokens")
                    )
                    if getattr(item, "kv_materialized_tokens", None) is None
                    else getattr(item, "kv_materialized_tokens")
                ),
            ),
            _serving_item_verifier_tokens(item),
            _serving_item_main_tokens(item),
            _serving_item_draft_tokens(item),
            item_logit_tokens_key(item),
            (
                _serving_item_committed_tokens(item)
                if _serving_item_is_mtp(item)
                else 0
            ),
        )
        for item in items
    )
    token_batch = sum(
        max(0, int(getattr(item, "token_count", 0))) for item in items
    )
    proposed_tokens = sum(_serving_item_verifier_tokens(item) for item in items)
    main_tokens = sum(_serving_item_main_tokens(item) for item in items)
    draft_tokens = sum(_serving_item_draft_tokens(item) for item in items)
    accepted_tokens = sum(
        _serving_item_committed_tokens(item)
        for item in items
        if _serving_item_is_mtp(item)
    )
    prior_kv_token_reads, internal_causal_pairs, kv_append_tokens = (
        _serving_cohort_kv_shape(cohort)
    )
    kv_materialized_tokens = _serving_cohort_kv_materialized_tokens(cohort)
    prior_attention_pairs = _serving_cohort_prior_attention_pairs(cohort)
    kv_scan_tokens = _serving_cohort_kv_scan_tokens(cohort)
    q4_view_tokens = _serving_cohort_kv_scan_tokens(
        cohort, metadata_key="llama_cpp_q4_kv_materialization_view_tokens_lower_bound",
    )
    raw_metadata = getattr(cohort, "metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    scan_prefix_identity = (
        (metadata["llama_cpp_kv_scan_phase"], metadata.get("llama_cpp_kv_occupied_rows"))
        if metadata.get("llama_cpp_kv_scan_phase") == "prefill_rectangular" else ()
    )
    return (
        "compute",
        kind,
        float(getattr(cohort, "proposal_cost_scale", 1.0)),
        item_shapes,
        len(items),
        token_batch,
        _serving_cohort_context_tokens(cohort),
        prior_kv_token_reads,
        prior_attention_pairs,
        internal_causal_pairs,
        kv_append_tokens,
        kv_materialized_tokens,
        proposed_tokens,
        main_tokens,
        draft_tokens,
        accepted_tokens,
        *((kv_scan_tokens,) if kv_scan_tokens else ()),
        *(("q4_mma_view_lower_bound", q4_view_tokens) if q4_view_tokens else ()),
        *scan_prefix_identity,
    )


class TopologyAwareBatchCostProvider:
    """Per-run topology-aware lowerer with immutable scenario invariants."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        *,
        execution_control: Optional[ExecutionControl] = None,
        template_cache_entries: int = 512,
        leaf_cache_entries: int = 4096,
    ) -> None:
        self._template_cache_entries = template_cache_entries
        self._leaf_cache_entries = leaf_cache_entries
        self._scenario = scenario
        self._compilation_context = CompilationContext(
            scenario,
            leaf_cache_entries=leaf_cache_entries,
            eager_full_attention_segments=False,
            compiled_serving_invocation_segments=True,
            _serving_invocation_lean_replay_capability=(
                _SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY
            ),
        )
        self._plan = self._compilation_context.parallel_plan()
        self._router = self._compilation_context.router()
        self._scenario_hash = stable_hash(scenario)
        self._control_plane_source: Optional[ScenarioConfig] = None
        self._execution_control = execution_control or ExecutionControl()
        self._templates = ExactTemplateCache(template_cache_entries)
        self._compiled_executor = CompiledGraphExecutor(max_layouts=8)

    def _rebind_control_plane_successor(
        self,
        source: ScenarioConfig,
        scenario: ScenarioConfig,
    ) -> None:
        """Bind to the validated V4 placement derived from ``source``.

        A replay revalidates the same immutable source on every run.  When
        that validation creates a fresh but structurally identical successor,
        move the identity binding without discarding exact lowering caches.
        """

        if self._scenario is source:
            self._control_plane_source = source
            self._scenario = scenario
            self._compilation_context = CompilationContext(
                scenario,
                leaf_cache_entries=self._leaf_cache_entries,
                eager_full_attention_segments=False,
                compiled_serving_invocation_segments=True,
                _serving_invocation_lean_replay_capability=(
                    _SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY
                ),
            )
            self._plan = self._compilation_context.parallel_plan()
            self._router = self._compilation_context.router()
            self._scenario_hash = stable_hash(scenario)
            self._templates = ExactTemplateCache(self._template_cache_entries)
            self._compiled_executor = CompiledGraphExecutor(max_layouts=8)
            return
        if (
            self._control_plane_source is source
            and stable_hash(scenario) == self._scenario_hash
        ):
            self._scenario = scenario
            self._compilation_context.scenario = scenario
            return
        raise ValueError(
            "拓扑感知的批成本提供器绑定到了另一个场景"
        )

    def estimate(
        self, scenario: ScenarioConfig, cohort: object
    ) -> Mapping[str, object]:
        if scenario is not self._scenario:
            raise ValueError(
                "拓扑感知的批成本提供器绑定到了另一个场景"
            )
        with _compilation_scope(scenario, self._compilation_context):
            return self._estimate_exact_in_context(scenario, cohort)

    def _estimate_exact_in_context(
        self, scenario: ScenarioConfig, cohort: object
    ) -> Mapping[str, object]:
        self._execution_control.raise_if_cancelled()
        cache_key = _serving_cohort_cache_key(cohort)
        return self._templates.get_or_create(
            cache_key,
            lambda: _estimate_serving_cohort_cost(
                scenario,
                cohort,
                cached_plan=self._plan,
                cached_router=self._router,
                scenario_hash=self._scenario_hash,
                execution_control=self._execution_control,
                compiled_executor=self._compiled_executor,
            ),
        )

    @property
    def template_cache_stats(self) -> Mapping[str, int]:
        return {
            "hits": self._templates.hits,
            "misses": self._templates.misses,
            "size": self._templates.size,
        }

    @property
    def leaf_cache_stats(self) -> Mapping[str, int]:
        return self._compilation_context.leaf_cache_stats

    @property
    def metadata_cache_stats(self) -> Mapping[str, int]:
        return self._compilation_context.metadata_cache_stats

    @property
    def artifact_workload_cache_stats(self) -> Mapping[str, int]:
        return self._compilation_context.artifact_workload_cache_stats

    @property
    def compiled_graph_cache_stats(self) -> Mapping[str, int]:
        return self._compiled_executor.stats


def _serving_cohort_context_tokens(cohort: object) -> int:
    items = tuple(getattr(cohort, "items", ()) or ())
    token_batch = sum(
        max(0, int(getattr(item, "token_count", 0))) for item in items
    )
    if token_batch <= 0:
        return 1
    weighted_context = sum(
        max(0, int(getattr(item, "context_tokens", 0)))
        * max(0, int(getattr(item, "token_count", 0)))
        for item in items
    )
    return max(1, int(math.ceil(weighted_context / float(token_batch))))


def _serving_cohort_kv_shape(cohort: object) -> Tuple[int, int, int]:
    """Return exact external-KV-read, intra-chunk, and append totals.

    Prefill and recompute chunks read historical persisted KV once per fused
    query chunk, allowing FlashAttention-style reuse across the chunk's query
    rows.  Decode/MTP retain their per-query historical access semantics.
    """

    kind = str(getattr(cohort, "kind", "decode"))
    prior_kv_token_reads = 0
    internal_causal_pairs = 0
    kv_append_tokens = 0
    for item in tuple(getattr(cohort, "items", ()) or ()):
        token_count = max(0, int(getattr(item, "token_count", 0)))
        context_tokens = max(0, int(getattr(item, "context_tokens", 0)))
        phase = str(getattr(item, "phase", kind))
        if kind == "prefill" or phase in {"prefill", "recompute"}:
            prior_kv_token_reads += context_tokens
        else:
            prior_kv_token_reads += token_count * context_tokens
        internal_causal_pairs += token_count * (token_count + 1) // 2
        raw_append = getattr(item, "kv_append_tokens", None)
        kv_append_tokens += max(
            0,
            int(token_count if raw_append is None else raw_append),
        )
    return prior_kv_token_reads, internal_causal_pairs, kv_append_tokens


def _serving_cohort_kv_materialized_tokens(cohort: object) -> int:
    """Return actual per-round KV writes, including rejected MTP proposals."""

    total = 0
    for item in tuple(getattr(cohort, "items", ()) or ()):
        token_count = max(0, int(getattr(item, "token_count", 0)))
        raw_append = getattr(item, "kv_append_tokens", None)
        append_tokens = max(
            0,
            int(token_count if raw_append is None else raw_append),
        )
        raw_materialized = getattr(item, "kv_materialized_tokens", None)
        total += max(
            0,
            int(
                append_tokens
                if raw_materialized is None
                else raw_materialized
            ),
        )
    return total


def _serving_cohort_prior_attention_pairs(cohort: object) -> int:
    """Historical-prompt attention work, independent of external KV reads."""

    return sum(
        max(0, int(getattr(item, "context_tokens", 0)))
        * max(0, int(getattr(item, "token_count", 0)))
        for item in tuple(getattr(cohort, "items", ()) or ())
    )


def estimate_serving_cohort_cost(
    scenario: ScenarioConfig, cohort: object
) -> Mapping[str, object]:
    """Lower one cohort without retaining state between independent calls."""

    with _compilation_scope(scenario):
        return _estimate_serving_cohort_cost(scenario, cohort)


@dataclass(frozen=True)
class _ServingCohortLowering:
    schedule: ScheduleIR
    cohort_id: str
    kind: str
    model: str
    extra_metadata: Mapping[str, object]


class _PreparedExecutionTaskRows(tuple):
    """Serialized task rows with an invisible in-process typed handoff."""

    def __new__(
        cls,
        rows: Sequence[Mapping[str, object]],
        prepared_tasks: Sequence[_PreparedExecutionTask] = (),
    ) -> "_PreparedExecutionTaskRows":
        instance = super().__new__(cls, rows)
        instance.prepared_tasks = tuple(prepared_tasks)
        return instance


_PREPARED_EXECUTION_STAGE_TOKEN = object()


class _PreparedExecutionStageRows(tuple):
    """Report-compatible stage rows carrying planner-validated stages."""

    def __new__(
        cls,
        rows: Sequence[Mapping[str, object]],
        prepared_stages: Sequence[_PreparedExecutionStage] = (),
        *,
        _token: object = None,
    ) -> "_PreparedExecutionStageRows":
        instance = super().__new__(cls, rows)
        instance._prepared_capsule = (
            (
                _PREPARED_EXECUTION_STAGE_TOKEN,
                instance,
                tuple(prepared_stages),
            )
            if _token is _PREPARED_EXECUTION_STAGE_TOKEN
            else None
        )
        return instance


def _trusted_execution_stages(
    value: object,
) -> Optional[Tuple[_PreparedExecutionStage, ...]]:
    """Return planner-owned typed stages only for the original wrapper."""

    if type(value) is not _PreparedExecutionStageRows:
        return None
    capsule = getattr(value, "_prepared_capsule", None)
    if (
        not isinstance(capsule, tuple)
        or len(capsule) != 3
        or capsule[0] is not _PREPARED_EXECUTION_STAGE_TOKEN
        or capsule[1] is not value
        or not isinstance(capsule[2], tuple)
    ):
        return None
    return capsule[2]


def _record_compute_components(
    record: TaskExecutionRecord,
    compute_component_ids: Set[str],
) -> Set[str]:
    dma_engine_resource_id = (
        str(record.metadata.get("dma_engine_resource_id", ""))
        if record.metadata.get("transfer_execution") == "coherent_dma"
        else ""
    )
    return {
        component_id
        for demand in record.demands
        for component_id in compute_component_ids
        if demand.resource_id != dma_engine_resource_id
        if demand.resource_id == component_id
        or demand.resource_id.startswith(component_id + ".")
    }


def _replay_local_record_timeline(
    records: Sequence[TaskExecutionRecord],
) -> Tuple[
    Tuple[TaskExecutionRecord, ...],
    Mapping[str, float],
    Mapping[str, Tuple[float, float]],
]:
    """Replay one stage region without external contention or dependencies."""

    ordered_records = tuple(records)
    previous_order_key: Optional[Tuple[float, float, str]] = None
    for record in ordered_records:
        order_key = (record.start_ns, record.end_ns, record.task_id)
        if previous_order_key is not None and order_key < previous_order_key:
            ordered_records = tuple(
                sorted(
                    ordered_records,
                    key=lambda item: (
                        item.start_ns,
                        item.end_ns,
                        item.task_id,
                    ),
                )
            )
            break
        previous_order_key = order_key
    local_end_by_task: Dict[str, float] = {}
    local_resource_available: Dict[str, float] = {}
    local_interval_by_task: Dict[str, Tuple[float, float]] = {}
    region_task_ids = {record.task_id for record in ordered_records}
    for record in ordered_records:
        local_start = max(
            [0.0]
            + [
                local_end_by_task.get(dependency, 0.0)
                for dependency in record.dependencies
                if dependency in region_task_ids
            ]
            + [
                local_resource_available.get(demand.resource_id, 0.0)
                for demand in record.demands
            ]
        )
        local_end = local_start + max(
            (float(demand.service_ns) for demand in record.demands),
            default=0.0,
        )
        local_end_by_task[record.task_id] = local_end
        local_interval_by_task[record.task_id] = (local_start, local_end)
        for demand in record.demands:
            local_resource_available[demand.resource_id] = (
                local_start + float(demand.service_ns)
            )
    return ordered_records, local_end_by_task, local_interval_by_task


def _execution_task_facts(
    records: Sequence[TaskExecutionRecord],
    request_ids: Sequence[str],
) -> Tuple[Mapping[str, object], ...]:
    """Serialize one stage-local task DAG without losing resource slices.

    Dependencies that cross the stage boundary are represented by the parent
    stage dependency.  Keeping only local task edges makes the payload closed
    and lets online serving validate/replay it through the same resource
    semantics as the planner event kernel.
    """

    record_by_id = {record.task_id: record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("execution stage contains duplicate task ids")
    task_ids = set(record_by_id)
    seen_ids: Set[str] = set()
    ordered_ids = []
    records_are_topological = True
    for record in records:
        if any(
            dependency in task_ids and dependency not in seen_ids
            for dependency in record.dependencies
        ):
            records_are_topological = False
            break
        seen_ids.add(record.task_id)
        ordered_ids.append(record.task_id)
    if not records_are_topological:
        indegree = {
            task_id: sum(
                dependency in task_ids
                for dependency in record.dependencies
            )
            for task_id, record in record_by_id.items()
        }
        dependents: Dict[str, List[str]] = {
            task_id: [] for task_id in task_ids
        }
        for task_id, record in record_by_id.items():
            for dependency in record.dependencies:
                if dependency in task_ids:
                    dependents[dependency].append(task_id)
        ready = [
            (
                record_by_id[task_id].start_ns,
                record_by_id[task_id].end_ns,
                task_id,
            )
            for task_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        ordered_ids = []
        while ready:
            _start_ns, _end_ns, task_id = heapq.heappop(ready)
            ordered_ids.append(task_id)
            for dependent_id in dependents[task_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    dependent = record_by_id[dependent_id]
                    heapq.heappush(
                        ready,
                        (dependent.start_ns, dependent.end_ns, dependent_id),
                    )
        if len(ordered_ids) != len(record_by_id):
            raise ValueError("execution stage task graph contains a cycle")

    normalized_request_ids = tuple(
        dict.fromkeys(str(request_id) for request_id in request_ids)
    )
    facts: List[Mapping[str, object]] = []
    prepared_tasks: List[_PreparedExecutionTask] = []
    for task_id in ordered_ids:
        record = record_by_id[task_id]
        task_dependencies = tuple(
            dependency
            for dependency in record.dependencies
            if dependency in task_ids
        )
        fact: Dict[str, object] = {
            "task_id": task_id,
            "dependencies": task_dependencies,
            "request_ids": normalized_request_ids,
            "opaque_device_fence": (
                record.metadata.get("opaque_device_fence") is True
            ),
            "resource_demands": tuple(
                {
                    "resource_id": demand.resource_id,
                    "service_ns": float(demand.service_ns),
                    "bytes_moved": int(demand.bytes_moved),
                    "energy_pj": float(demand.energy_pj),
                    "work_units": float(demand.work_units),
                }
                for demand in record.demands
            ),
        }
        retained_metadata = {
            key: dict(value)
            for key in (
                "transient_residency_fact",
                "mmq_source_work",
                "native_kv_work",
                "qwen35_attention_work",
                "qwen35_tensor_geometry",
                "qwen35_shared_input_work",
                "qwen35_attention_geometry",
                "final_layer_output_selection",
                "output_selection_tensor_geometry",
            )
            for value in (record.metadata.get(key),)
            if isinstance(value, Mapping)
        }
        if retained_metadata:
            fact["metadata"] = retained_metadata
        facts.append(fact)
        prepared_tasks.append(
            _PreparedExecutionTask(
                task_id,
                task_dependencies,
                normalized_request_ids,
                record.demands,
                opaque_device_fence=(
                    record.metadata.get("opaque_device_fence") is True
                ),
                category=record.category,
                metadata={
                    **retained_metadata,
                    **{
                        key: record.metadata[key]
                        for key in (
                            "event_kind",
                            "runtime_phase",
                            "instruction_class",
                            "instruction_count",
                            "transaction_kind",
                            "transaction_batches",
                            "aggregation",
                        )
                        if key in record.metadata
                    },
                },
            )
        )
    return _PreparedExecutionTaskRows(facts, prepared_tasks)


def _prepare_execution_stage_rows(
    rows: Sequence[Mapping[str, object]],
) -> Optional[Tuple[_PreparedExecutionStage, ...]]:
    """Build typed stages from values already validated during compaction.

    This loop is proportional to stage count.  Task ordering, dependency
    closure, request identity, and ResourceDemand validation were established
    in ``_execution_task_facts`` and are carried by its private tuple wrapper,
    so no task or demand subtree is reparsed here.
    """

    prepared: List[_PreparedExecutionStage] = []
    seen_stage_ids: Set[str] = set()
    for index, row in enumerate(rows):
        stage_id = row.get("stage_id")
        component_id = row.get("component_id")
        dependencies = row.get("dependencies", ())
        request_ids = row.get("request_ids", ())
        raw_tasks = row.get("execution_tasks")
        group_id = row.get("group_id")
        covered_group_ids = row.get(
            "covered_invocation_group_ids",
            (),
        )
        if (
            not isinstance(stage_id, str)
            or not stage_id
            or stage_id in seen_stage_ids
            or not isinstance(component_id, str)
            or not component_id
            or not isinstance(dependencies, tuple)
            or any(
                not isinstance(item, str)
                or not item
                or item not in seen_stage_ids
                for item in dependencies
            )
            or len(dependencies) != len(set(dependencies))
            or not isinstance(request_ids, tuple)
            or not request_ids
            or any(not isinstance(item, str) or not item for item in request_ids)
            or len(request_ids) != len(set(request_ids))
            or type(raw_tasks) is not _PreparedExecutionTaskRows
            or len(raw_tasks.prepared_tasks) != len(raw_tasks)
            or not raw_tasks.prepared_tasks
            or (
                group_id is not None
                and (not isinstance(group_id, str) or not group_id)
            )
            or not isinstance(covered_group_ids, tuple)
            or any(
                not isinstance(item, str) or not item
                for item in covered_group_ids
            )
            or len(covered_group_ids) != len(set(covered_group_ids))
        ):
            return None
        try:
            raw_stage_index = row.get("stage_index", index)
            raw_service_ns = row.get("service_ns", 0.0)
            if isinstance(raw_stage_index, bool) or isinstance(
                raw_service_ns, bool
            ):
                return None
            stage_index = int(raw_stage_index)
            service_ns = float(raw_service_ns)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(float(raw_stage_index))
            or float(raw_stage_index) != stage_index
            or not math.isfinite(service_ns)
            or service_ns <= 0.0
        ):
            return None
        prepared.append(
            _PreparedExecutionStage(
                stage_id,
                stage_index,
                tuple(dependencies),
                tuple(request_ids),
                component_id,
                service_ns,
                raw_tasks.prepared_tasks,
                group_id,
                tuple(covered_group_ids),
            )
        )
        seen_stage_ids.add(stage_id)
    return tuple(prepared) if prepared else None


def _compact_host_control_stage(
    scenario: ScenarioConfig,
    records: Sequence[TaskExecutionRecord],
    request_ids: Sequence[str],
    *,
    orchestration_stage: str,
    stage_role: str,
    stage_id: str,
    stage_index: int,
    dependencies: Sequence[str] = (),
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    host_records = tuple(
        record
        for record in records
        if record.metadata.get("orchestration_stage") == orchestration_stage
    )
    positive_records = tuple(
        record
        for record in host_records
        if max(
            (float(demand.service_ns) for demand in record.demands),
            default=0.0,
        )
        > 0.0
    )
    if not positive_records:
        return None, None

    cpu_component_ids = {
        component.component_id
        for component in scenario.hardware.components
        if normalize_component_kind(component.kind) == "cpu"
    }
    configured_cpu_id = scenario.host_orchestration_profile.cpu_component_id
    if configured_cpu_id not in cpu_component_ids:
        return None, "host orchestration CPU is not a declared CPU component"
    used_cpu_ids: Set[str] = set()
    for record in positive_records:
        used_cpu_ids.update(
            _record_compute_components(record, cpu_component_ids)
        )
    component_id = (
        configured_cpu_id
        if configured_cpu_id in used_cpu_ids or not used_cpu_ids
        else sorted(used_cpu_ids)[0]
    )

    ordered, local_end_by_task, _local_intervals = (
        _replay_local_record_timeline(host_records)
    )
    service_ns = max(local_end_by_task.values(), default=0.0)
    if service_ns <= 0.0:
        return None, "{} has no positive service".format(stage_role)
    normalized_request_ids = tuple(dict.fromkeys(str(item) for item in request_ids))
    if not normalized_request_ids:
        return None, "{} has no request identity".format(stage_role)
    normalized_dependencies = tuple(
        dict.fromkeys(str(dependency) for dependency in dependencies)
    )
    covered_invocation_group_ids = tuple(
        dict.fromkeys(
            str(group_id)
            for record in positive_records
            for raw_group_ids in (
                record.metadata.get("physical_invocation_group_ids", ()),
            )
            for group_id in (
                raw_group_ids
                if isinstance(raw_group_ids, Sequence)
                and not isinstance(raw_group_ids, (str, bytes, Mapping))
                else ()
            )
            if str(group_id)
        )
    )

    return {
        "stage_id": stage_id,
        "stage_index": stage_index,
        "group_id": None,
        "covered_invocation_group_ids": covered_invocation_group_ids,
        "stage_role": stage_role,
        "dependencies": normalized_dependencies,
        "request_ids": normalized_request_ids,
        "component_id": component_id,
        "resource_ids": tuple(
            sorted(
                {
                    demand.resource_id
                    for record in ordered
                    for demand in record.demands
                }
            )
        ),
        "execution_tasks": _execution_task_facts(
            ordered, normalized_request_ids
        ),
        "layer_ids": (),
        "service_ns": service_ns,
        "observed_span_ns": (
            max(record.end_ns for record in positive_records)
            - min(record.start_ns for record in positive_records)
        ),
        "source_task_count": len(positive_records),
        "replay_task_count": len(ordered),
        "source_event_kinds": tuple(
            sorted(
                {
                    str(record.metadata.get("event_kind"))
                    for record in positive_records
                    if record.metadata.get("event_kind") is not None
                }
            )
        ),
    }, None


def _compact_host_prefix_stage(
    scenario: ScenarioConfig,
    records: Sequence[TaskExecutionRecord],
    request_ids: Sequence[str],
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    """Keep aggregate CPU and controller work inside the live task DAG.

    The stage is only a causal envelope.  Its tasks retain the exact CPU
    cache/DRAM, IOMMU, DMA, PCIe, and GPU command-processor resource demands.
    """

    return _compact_host_control_stage(
        scenario,
        records,
        request_ids,
        orchestration_stage="host_prefix",
        stage_role="host_prefix",
        stage_id="serving.host_prefix.stage0000",
        stage_index=-2,
    )


def _compact_device_prefix_stage(
    records: Sequence[TaskExecutionRecord],
    grouped_task_ids: Set[str],
    compute_component_ids: Set[str],
    request_ids: Sequence[str],
    dependencies: Sequence[str] = (),
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    """Preserve executed device ancestors that precede invocation groups.

    Host orchestration is accounted separately.  Any other non-zero task on
    the dependency path into an invocation group is device work and must not
    disappear merely because it has no invocation-group identity.  MTP draft
    proposal kernels are the primary example.
    """

    record_by_id = {record.task_id: record for record in records}
    ancestor_ids: Set[str] = set()
    pending = [
        dependency
        for record in records
        if record.task_id in grouped_task_ids
        for dependency in record.dependencies
    ]
    while pending:
        task_id = pending.pop()
        if task_id in ancestor_ids:
            continue
        ancestor_ids.add(task_id)
        ancestor = record_by_id.get(task_id)
        if ancestor is not None:
            pending.extend(ancestor.dependencies)

    prefix_records = tuple(
        record
        for record in records
        if record.task_id in ancestor_ids
        and record.metadata.get("operator_invocation_group_id") is None
        and record.metadata.get("orchestration_stage") is None
    )
    positive_prefix_records = tuple(
        record
        for record in prefix_records
        if max(
            (float(demand.service_ns) for demand in record.demands),
            default=0.0,
        )
        > 0.0
    )
    if not positive_prefix_records:
        return None, None

    owners: Set[str] = set()
    for record in positive_prefix_records:
        owners.update(
            _record_compute_components(record, compute_component_ids)
        )
    if len(owners) != 1:
        return None, (
            "ungrouped device prefix does not have exactly one compute component"
        )

    ordered, local_end_by_task, _local_intervals = (
        _replay_local_record_timeline(prefix_records)
    )
    service_ns = max(local_end_by_task.values(), default=0.0)
    if service_ns <= 0.0:
        return None, "ungrouped device prefix has no positive service"
    normalized_request_ids = tuple(dict.fromkeys(str(item) for item in request_ids))
    if not normalized_request_ids:
        return None, "ungrouped device prefix has no request identity"
    covered_invocation_group_ids = tuple(
        dict.fromkeys(
            str(group_id)
            for record in positive_prefix_records
            for group_id in (
                record.metadata.get("mtp_proposer_invocation_group_id"),
            )
            if group_id is not None and str(group_id)
        )
    )

    return {
        "stage_id": "serving.device_prefix.stage0000",
        "stage_index": -1,
        "group_id": None,
        "covered_invocation_group_ids": covered_invocation_group_ids,
        "stage_role": "device_prefix",
        "dependencies": tuple(dependencies),
        "request_ids": normalized_request_ids,
        "component_id": next(iter(owners)),
        "resource_ids": tuple(
            sorted(
                {
                    demand.resource_id
                    for record in ordered
                    for demand in record.demands
                }
            )
        ),
        "execution_tasks": _execution_task_facts(
            ordered, normalized_request_ids
        ),
        "layer_ids": (),
        "service_ns": service_ns,
        "observed_span_ns": (
            max(record.end_ns for record in positive_prefix_records)
            - min(record.start_ns for record in positive_prefix_records)
        ),
        "source_task_count": len(positive_prefix_records),
        "replay_task_count": len(ordered),
        "source_event_kinds": tuple(
            sorted(
                {
                    str(record.metadata.get("event_kind"))
                    for record in positive_prefix_records
                    if record.metadata.get("event_kind") is not None
                }
            )
        ),
    }, None


def _compact_device_suffix_stage(
    records: Sequence[TaskExecutionRecord],
    grouped_task_ids: Set[str],
    compute_component_ids: Set[str],
    request_ids: Sequence[str],
    terminal_stage_by_group: Mapping[str, str],
    stage_index: int,
    dependencies: Sequence[str] = (),
) -> Tuple[Optional[Mapping[str, object]], Optional[str]]:
    """Preserve executed device descendants that follow invocation groups.

    MTP context catch-up runs after the target invocation groups and therefore
    cannot be folded into the device prefix or into a target stage.  Reduce the
    ungrouped descendant region to one explicit tail stage whose dependencies
    are the terminal stages of the invocation groups at its boundary.
    """

    record_by_id = {record.task_id: record for record in records}
    dependent_ids_by_task: Dict[str, List[str]] = {}
    for record in records:
        for dependency in record.dependencies:
            dependent_ids_by_task.setdefault(dependency, []).append(
                record.task_id
            )

    descendant_ids: Set[str] = set()
    pending = list(grouped_task_ids)
    while pending:
        task_id = pending.pop()
        for dependent_id in dependent_ids_by_task.get(task_id, ()):
            if (
                dependent_id in grouped_task_ids
                or dependent_id in descendant_ids
            ):
                continue
            descendant_ids.add(dependent_id)
            pending.append(dependent_id)

    suffix_records = tuple(
        record
        for record in records
        if record.task_id in descendant_ids
        and record.metadata.get("operator_invocation_group_id") is None
        and record.metadata.get("orchestration_stage") is None
    )
    positive_suffix_records = tuple(
        record
        for record in suffix_records
        if max(
            (float(demand.service_ns) for demand in record.demands),
            default=0.0,
        )
        > 0.0
    )
    if not positive_suffix_records:
        return None, None

    owners: Set[str] = set()
    for record in positive_suffix_records:
        owners.update(
            _record_compute_components(record, compute_component_ids)
        )
    if len(owners) != 1:
        return None, (
            "ungrouped device suffix does not have exactly one compute component"
        )

    predecessor_group_ids: Set[str] = set()
    visited_ancestors: Set[str] = set()
    pending = [
        dependency
        for record in suffix_records
        for dependency in record.dependencies
    ]
    while pending:
        task_id = pending.pop()
        if task_id in visited_ancestors:
            continue
        visited_ancestors.add(task_id)
        ancestor = record_by_id.get(task_id)
        if ancestor is None:
            continue
        group_id = ancestor.metadata.get("operator_invocation_group_id")
        if group_id is not None:
            predecessor_group_ids.add(str(group_id))
            continue
        pending.extend(ancestor.dependencies)
    if not predecessor_group_ids.issubset(terminal_stage_by_group):
        return None, "ungrouped device suffix predecessor is not ordered"
    predecessor_stage_ids = tuple(
        terminal_stage_id
        for group_id, terminal_stage_id in terminal_stage_by_group.items()
        if group_id in predecessor_group_ids
    )
    explicit_dependencies = tuple(
        dict.fromkeys(str(dependency) for dependency in dependencies)
    )
    if explicit_dependencies:
        predecessor_stage_ids = explicit_dependencies
    if not predecessor_stage_ids:
        return None, "ungrouped device suffix has no invocation-group predecessor"

    ordered, local_end_by_task, _local_intervals = (
        _replay_local_record_timeline(suffix_records)
    )
    service_ns = max(local_end_by_task.values(), default=0.0)
    if service_ns <= 0.0:
        return None, "ungrouped device suffix has no positive service"
    normalized_request_ids = tuple(dict.fromkeys(str(item) for item in request_ids))
    if not normalized_request_ids:
        return None, "ungrouped device suffix has no request identity"
    covered_invocation_group_ids = tuple(
        dict.fromkeys(
            str(group_id)
            for record in positive_suffix_records
            for group_id in (
                record.metadata.get("mtp_proposer_invocation_group_id"),
            )
            if group_id is not None and str(group_id)
        )
    )

    return {
        "stage_id": "serving.device_suffix.stage0000",
        "stage_index": stage_index,
        "group_id": None,
        "covered_invocation_group_ids": covered_invocation_group_ids,
        "stage_role": "device_suffix",
        "dependencies": predecessor_stage_ids,
        "request_ids": normalized_request_ids,
        "component_id": next(iter(owners)),
        "resource_ids": tuple(
            sorted(
                {
                    demand.resource_id
                    for record in ordered
                    for demand in record.demands
                }
            )
        ),
        "execution_tasks": _execution_task_facts(
            ordered, normalized_request_ids
        ),
        "layer_ids": (),
        "service_ns": service_ns,
        "observed_span_ns": (
            max(record.end_ns for record in positive_suffix_records)
            - min(record.start_ns for record in positive_suffix_records)
        ),
        "source_task_count": len(positive_suffix_records),
        "replay_task_count": len(ordered),
        "source_event_kinds": tuple(
            sorted(
                {
                    str(record.metadata.get("event_kind"))
                    for record in suffix_records
                    if record.metadata.get("event_kind") is not None
                }
            )
        ),
    }, None


def _compact_execution_stages(
    scenario: ScenarioConfig,
    records: Sequence[TaskExecutionRecord],
    invocation_groups: Sequence[Mapping[str, object]],
) -> Tuple[Tuple[Mapping[str, object], ...], Optional[str]]:
    """Reduce executed request-local branches to component-stage facts.

    Stage service is replayed from the executed task order using only the
    branch's own dependencies and resource predecessors.  Contention from a
    sibling request therefore remains a cross-stage resource edge instead of
    being baked into both branches' service time.
    """

    compute_component_ids = {
        component.component_id
        for component in scenario.hardware.components
        if normalize_component_kind(component.kind) in {"cpu", "gpu", "cim"}
    }
    if not compute_component_ids:
        return (), "no compute component is declared"
    records_by_group: Dict[str, List[TaskExecutionRecord]] = {}
    for record in records:
        group_id = record.metadata.get("operator_invocation_group_id")
        if group_id is not None:
            records_by_group.setdefault(str(group_id), []).append(record)
    if not records_by_group:
        return (), "executed tasks have no invocation-group identity"

    group_facts = {
        str(group.get("group_id")): group
        for group in invocation_groups
        if isinstance(group, Mapping) and group.get("group_id") is not None
    }
    grouped_task_ids = {
        record.task_id
        for group_records in records_by_group.values()
        for record in group_records
    }
    all_request_ids = tuple(
        str(request_id)
        for group in group_facts.values()
        for request_id in group.get("request_ids", ())
    )
    host_prefix_stage, host_fallback_reason = _compact_host_prefix_stage(
        scenario,
        records,
        all_request_ids,
    )
    if host_fallback_reason is not None:
        return (), host_fallback_reason
    host_prefix_stage_id = (
        str(host_prefix_stage["stage_id"])
        if host_prefix_stage is not None
        else None
    )
    device_prefix_stage, prefix_fallback_reason = _compact_device_prefix_stage(
        records,
        grouped_task_ids,
        compute_component_ids,
        all_request_ids,
        (host_prefix_stage_id,) if host_prefix_stage_id is not None else (),
    )
    if prefix_fallback_reason is not None:
        return (), prefix_fallback_reason
    stages: List[Mapping[str, object]] = []
    stage_records_by_stage_id: Dict[
        str, Tuple[TaskExecutionRecord, ...]
    ] = {}
    if host_prefix_stage is not None:
        stages.append(host_prefix_stage)
    if device_prefix_stage is not None:
        stages.append(device_prefix_stage)
    device_prefix_stage_id = (
        str(device_prefix_stage["stage_id"])
        if device_prefix_stage is not None
        else None
    )
    target_frontend_dependencies: Tuple[str, ...] = ()
    if device_prefix_stage_id is not None:
        target_frontend_dependencies = (device_prefix_stage_id,)
    elif host_prefix_stage_id is not None:
        target_frontend_dependencies = (host_prefix_stage_id,)
    target_frontend_stage, target_frontend_fallback_reason = (
        _compact_host_control_stage(
            scenario,
            records,
            all_request_ids,
            orchestration_stage="host_target_frontend",
            stage_role="host_target_frontend",
            stage_id="serving.host_target_frontend.stage0000",
            stage_index=0,
            dependencies=target_frontend_dependencies,
        )
    )
    if target_frontend_fallback_reason is not None:
        return (), target_frontend_fallback_reason
    if target_frontend_stage is not None:
        stages.append(target_frontend_stage)
    target_frontend_stage_id = (
        str(target_frontend_stage["stage_id"])
        if target_frontend_stage is not None
        else None
    )
    terminal_stage_by_group: Dict[str, str] = {}
    for group_id, group_records in sorted(
        records_by_group.items(),
        key=lambda item: min(record.start_ns for record in item[1]),
    ):
        group = group_facts.get(group_id)
        if group is None:
            return (), "invocation-group metadata is incomplete"
        (
            ordered_records,
            local_end_by_task,
            local_interval_by_task,
        ) = _replay_local_record_timeline(group_records)

        output_device_records = tuple(
            record
            for record in ordered_records
            if record.metadata.get("serving_output_stage")
            == "device_output_completion"
        )
        output_cpu_records = tuple(
            record
            for record in ordered_records
            if record.metadata.get("serving_output_stage")
            == "cpu_output_commit"
        )
        output_task_ids = {
            record.task_id
            for record in (*output_device_records, *output_cpu_records)
        }
        model_records = tuple(
            record
            for record in ordered_records
            if record.task_id not in output_task_ids
        )

        records_by_layer: Dict[str, List[TaskExecutionRecord]] = {}
        for record in model_records:
            layer_id = record.metadata.get("layer_id")
            if layer_id is not None:
                records_by_layer.setdefault(str(layer_id), []).append(record)
        if not records_by_layer:
            return (), "invocation-group tasks expose no layer identity"
        layer_owner: Dict[str, str] = {}
        layer_owner_sets: Dict[str, Set[str]] = {}
        for layer_id, layer_records in records_by_layer.items():
            owners: Set[str] = set()
            for record in layer_records:
                owners.update(
                    _record_compute_components(record, compute_component_ids)
                )
            layer_owner_sets[layer_id] = owners
            if len(owners) == 1:
                layer_owner[layer_id] = next(iter(owners))
        layer_order = sorted(
            records_by_layer,
            key=lambda layer_id: min(
                local_interval_by_task[record.task_id][0]
                for record in records_by_layer[layer_id]
            ),
        )
        mixed_layer_ids = tuple(
            layer_id
            for layer_id in layer_order
            if len(layer_owner_sets[layer_id]) != 1
        )
        segments: List[Tuple[str, List[str]]] = []
        segment_starts: List[float] = []
        if not mixed_layer_ids:
            for layer_id in layer_order:
                owner = layer_owner[layer_id]
                if not segments or segments[-1][0] != owner:
                    segments.append((owner, [layer_id]))
                    segment_starts.append(
                        min(
                            local_interval_by_task[record.task_id][0]
                            for record in records_by_layer[layer_id]
                        )
                    )
                else:
                    segments[-1][1].append(layer_id)
        else:
            mixed_records = tuple(
                record
                for layer_id in mixed_layer_ids
                for record in records_by_layer[layer_id]
            )
            if not any(
                record.metadata.get("host_gemm_offload_applied") is True
                for record in mixed_records
            ):
                return (), (
                    "layer {} does not have exactly one compute component"
                ).format(mixed_layer_ids[0])

            owner_events: List[
                Tuple[float, float, str, str, str]
            ] = []
            for record in model_records:
                owners = _record_compute_components(
                    record, compute_component_ids
                )
                if len(owners) > 1:
                    return (), (
                        "host-offloaded task spans multiple compute components"
                    )
                if not owners:
                    continue
                start_ns, end_ns = local_interval_by_task[record.task_id]
                if end_ns <= start_ns:
                    continue
                layer_id = str(record.metadata.get("layer_id") or "")
                owner_events.append(
                    (
                        start_ns,
                        end_ns,
                        record.task_id,
                        next(iter(owners)),
                        layer_id,
                    )
                )
            owner_events.sort(key=lambda item: item[:3])
            if not owner_events:
                return (), "invocation group has no compute-owner timeline"

            run_end_ns = 0.0
            for start_ns, end_ns, _task_id, owner, layer_id in owner_events:
                if not segments or segments[-1][0] != owner:
                    if segments and start_ns < run_end_ns - 1.0e-9:
                        return (), (
                            "host-offloaded compute components overlap within "
                            "one invocation group"
                        )
                    segments.append((owner, [layer_id] if layer_id else []))
                    segment_starts.append(start_ns)
                    run_end_ns = end_ns
                else:
                    if layer_id and layer_id not in segments[-1][1]:
                        segments[-1][1].append(layer_id)
                    run_end_ns = max(run_end_ns, end_ns)
        if not segments:
            return (), "invocation group has no reducible compute stage"

        boundaries = [0.0]
        boundaries.extend(segment_starts[1:])
        boundaries.append(
            max(
                (
                    local_end_by_task[record.task_id]
                    for record in model_records
                ),
                default=0.0,
            )
        )
        if any(
            later <= earlier
            for earlier, later in zip(boundaries, boundaries[1:])
        ):
            return (), "component stages do not form a positive linear branch"

        request_ids_raw = group.get("request_ids", ())
        if isinstance(request_ids_raw, (str, bytes)) or not isinstance(
            request_ids_raw, Sequence
        ):
            return (), "invocation-group request_ids are invalid"
        request_ids = tuple(str(item) for item in request_ids_raw)
        if not request_ids:
            return (), "invocation group has no request identity"
        previous_stage_id: Optional[str] = None
        predecessor_group_id = group.get("predecessor_group_id")
        for stage_index, ((component_id, layer_ids), start_ns, end_ns) in enumerate(
            zip(segments, boundaries, boundaries[1:])
        ):
            stage_id = "{}.stage{:04d}".format(group_id, stage_index)
            dependencies: List[str] = []
            if previous_stage_id is not None:
                dependencies.append(previous_stage_id)
            elif predecessor_group_id is not None:
                predecessor_stage = terminal_stage_by_group.get(
                    str(predecessor_group_id)
                )
                if predecessor_stage is None:
                    return (), "invocation-group predecessor is not ordered"
                dependencies.append(predecessor_stage)
            elif target_frontend_stage_id is not None:
                dependencies.append(target_frontend_stage_id)
            elif device_prefix_stage_id is not None:
                dependencies.append(device_prefix_stage_id)
            elif host_prefix_stage_id is not None:
                dependencies.append(host_prefix_stage_id)
            stage_records = [
                record
                for record in model_records
                if (
                    start_ns
                    <= local_interval_by_task[record.task_id][0]
                    < end_ns
                )
                or (
                    stage_index == len(segments) - 1
                    and local_interval_by_task[record.task_id][0] == end_ns
                )
            ]
            stage_task_ids = {record.task_id for record in stage_records}
            ordered_stage_records = tuple(
                record
                for record in model_records
                if record.task_id in stage_task_ids
            )
            resource_ids = tuple(
                sorted(
                    {
                        demand.resource_id
                        for record in stage_records
                        for demand in record.demands
                    }
                )
            )
            observed_start = min(record.start_ns for record in stage_records)
            observed_end = max(record.end_ns for record in stage_records)
            stage_row = {
                "stage_id": stage_id,
                "stage_index": stage_index,
                "group_id": group_id,
                "dependencies": tuple(dependencies),
                "request_ids": request_ids,
                "component_id": component_id,
                "resource_ids": resource_ids,
                "execution_tasks": _execution_task_facts(
                    ordered_stage_records, request_ids
                ),
                "layer_ids": tuple(layer_ids),
                "service_ns": end_ns - start_ns,
                "observed_span_ns": observed_end - observed_start,
            }
            stages.append(stage_row)
            stage_records_by_stage_id[stage_id] = ordered_stage_records
            previous_stage_id = stage_id

        device_stage_records = output_device_records
        cpu_stage_records = output_cpu_records
        zero_output_records: Tuple[TaskExecutionRecord, ...] = ()
        if device_stage_records:
            _device_ordered, device_end_by_task, _device_intervals = (
                _replay_local_record_timeline(device_stage_records)
            )
            device_service_ns = max(device_end_by_task.values(), default=0.0)
        else:
            _device_ordered = ()
            device_service_ns = 0.0
        if cpu_stage_records:
            _cpu_ordered, cpu_end_by_task, _cpu_intervals = (
                _replay_local_record_timeline(cpu_stage_records)
            )
            cpu_service_ns = max(cpu_end_by_task.values(), default=0.0)
        else:
            _cpu_ordered = ()
            cpu_service_ns = 0.0

        # A missing optional runtime declaration can leave one side as a
        # zero-service audit chain.  Retain its facts in the adjacent positive
        # stage rather than inventing service solely to satisfy stage shape.
        if device_stage_records and device_service_ns <= 0.0:
            cpu_stage_records = (*device_stage_records, *cpu_stage_records)
            device_stage_records = ()
        if cpu_stage_records:
            _cpu_ordered, cpu_end_by_task, _cpu_intervals = (
                _replay_local_record_timeline(cpu_stage_records)
            )
            cpu_service_ns = max(cpu_end_by_task.values(), default=0.0)
        if cpu_stage_records and cpu_service_ns <= 0.0:
            if device_stage_records and device_service_ns > 0.0:
                device_stage_records = (
                    *device_stage_records,
                    *cpu_stage_records,
                )
                cpu_stage_records = ()
                _device_ordered, device_end_by_task, _device_intervals = (
                    _replay_local_record_timeline(device_stage_records)
                )
                device_service_ns = max(
                    device_end_by_task.values(), default=0.0
                )
            else:
                zero_output_records = (
                    *device_stage_records,
                    *cpu_stage_records,
                )
                device_stage_records = ()
                cpu_stage_records = ()
                device_service_ns = 0.0
                cpu_service_ns = 0.0

        if zero_output_records:
            if (
                previous_stage_id is None
                or previous_stage_id not in stage_records_by_stage_id
            ):
                return (), (
                    "zero-service output chain has no positive model stage"
                )
            stage_index_to_update = next(
                (
                    index
                    for index, stage in enumerate(stages)
                    if stage.get("stage_id") == previous_stage_id
                ),
                -1,
            )
            if stage_index_to_update < 0:
                return (), (
                    "zero-service output chain predecessor is unavailable"
                )
            combined_records = (
                *stage_records_by_stage_id[previous_stage_id],
                *zero_output_records,
            )
            combined_ordered, _combined_end_by_task, _combined_intervals = (
                _replay_local_record_timeline(combined_records)
            )
            updated_stage = dict(stages[stage_index_to_update])
            updated_stage["resource_ids"] = tuple(
                sorted(
                    {
                        demand.resource_id
                        for record in combined_ordered
                        for demand in record.demands
                    }
                )
            )
            updated_stage["execution_tasks"] = _execution_task_facts(
                combined_ordered, request_ids
            )
            updated_stage["zero_service_output_attached"] = True
            updated_stage["zero_service_output_task_count"] = len(
                zero_output_records
            )
            stages[stage_index_to_update] = updated_stage
            stage_records_by_stage_id[previous_stage_id] = combined_ordered

        if device_stage_records and device_service_ns > 0.0:
            source_components = tuple(
                dict.fromkeys(
                    str(record.metadata.get("output_source_component_id", ""))
                    for record in device_stage_records
                    if record.metadata.get("output_source_component_id")
                )
            )
            if len(source_components) != 1:
                return (), "output D2H stage has no unique source component"
            stage_id = group_id + ".output_d2h_completion"
            stages.append(
                {
                    "stage_id": stage_id,
                    "stage_index": len(segments),
                    "group_id": group_id,
                    "stage_role": "device_output_completion",
                    "dependencies": (
                        (previous_stage_id,) if previous_stage_id is not None else ()
                    ),
                    "request_ids": request_ids,
                    "component_id": source_components[0],
                    "resource_ids": tuple(
                        sorted(
                            {
                                demand.resource_id
                                for record in device_stage_records
                                for demand in record.demands
                            }
                        )
                    ),
                    "execution_tasks": _execution_task_facts(
                        _device_ordered, request_ids
                    ),
                    "layer_ids": (),
                    "service_ns": device_service_ns,
                    "observed_span_ns": (
                        max(record.end_ns for record in device_stage_records)
                        - min(record.start_ns for record in device_stage_records)
                    ),
                }
            )
            previous_stage_id = stage_id

        if cpu_stage_records and cpu_service_ns > 0.0:
            _cpu_ordered, cpu_end_by_task, _cpu_intervals = (
                _replay_local_record_timeline(cpu_stage_records)
            )
            cpu_service_ns = max(cpu_end_by_task.values(), default=0.0)
            stage_id = group_id + ".output_cpu_commit"
            stages.append(
                {
                    "stage_id": stage_id,
                    "stage_index": len(segments) + 1,
                    "group_id": group_id,
                    "stage_role": "cpu_output_commit",
                    "dependencies": (
                        (previous_stage_id,) if previous_stage_id is not None else ()
                    ),
                    "request_ids": request_ids,
                    "component_id": (
                        scenario.host_orchestration_profile.cpu_component_id
                    ),
                    "resource_ids": tuple(
                        sorted(
                            {
                                demand.resource_id
                                for record in cpu_stage_records
                                for demand in record.demands
                            }
                        )
                    ),
                    "execution_tasks": _execution_task_facts(
                        _cpu_ordered, request_ids
                    ),
                    "layer_ids": (),
                    "service_ns": cpu_service_ns,
                    "observed_span_ns": (
                        max(record.end_ns for record in cpu_stage_records)
                        - min(record.start_ns for record in cpu_stage_records)
                    ),
                }
            )
            previous_stage_id = stage_id
        if previous_stage_id is not None:
            terminal_stage_by_group[group_id] = previous_stage_id
    host_suffix_stage, host_suffix_fallback_reason = (
        _compact_host_control_stage(
            scenario,
            records,
            all_request_ids,
            orchestration_stage="host_suffix",
            stage_role="host_suffix",
            stage_id="serving.host_suffix.stage0000",
            stage_index=max(
                (int(stage["stage_index"]) for stage in stages),
                default=-1,
            )
            + 1,
            dependencies=tuple(terminal_stage_by_group.values()),
        )
    )
    if host_suffix_fallback_reason is not None:
        return (), host_suffix_fallback_reason
    if host_suffix_stage is not None:
        stages.append(host_suffix_stage)
    host_suffix_stage_id = (
        str(host_suffix_stage["stage_id"])
        if host_suffix_stage is not None
        else None
    )
    device_suffix_stage, suffix_fallback_reason = _compact_device_suffix_stage(
        records,
        grouped_task_ids,
        compute_component_ids,
        all_request_ids,
        terminal_stage_by_group,
        max(
            (int(stage["stage_index"]) for stage in stages),
            default=-1,
        )
        + 1,
        dependencies=(
            (host_suffix_stage_id,) if host_suffix_stage_id is not None else ()
        ),
    )
    if suffix_fallback_reason is not None:
        return (), suffix_fallback_reason
    if device_suffix_stage is not None:
        stages.append(device_suffix_stage)
    prepared_stages = _prepare_execution_stage_rows(stages)
    if prepared_stages is None:
        # Preserve the raw public payload and let serving's fail-closed parser
        # diagnose any invariant not established by planner compaction.
        return tuple(stages), None
    return _PreparedExecutionStageRows(
        stages,
        prepared_stages,
        _token=_PREPARED_EXECUTION_STAGE_TOKEN,
    ), None


def _stage_dag_makespan(stages: Sequence[Mapping[str, object]]) -> float:
    end_by_stage: Dict[str, float] = {}
    available_by_component: Dict[str, float] = {}
    for stage in stages:
        stage_id = str(stage["stage_id"])
        component_id = str(stage["component_id"])
        dependencies = tuple(str(item) for item in stage.get("dependencies", ()))
        start_ns = max(
            [available_by_component.get(component_id, 0.0)]
            + [end_by_stage[dependency] for dependency in dependencies]
        )
        end_ns = start_ns + float(stage["service_ns"])
        end_by_stage[stage_id] = end_ns
        available_by_component[component_id] = end_ns
    return max(end_by_stage.values(), default=0.0)


def compile_serving_cohort_schedule(
    scenario: ScenarioConfig, cohort: object
) -> ScheduleIR:
    """Compile one realized online cohort into its exact planner task graph.

    This is the task-preserving counterpart of
    :func:`estimate_serving_cohort_cost`.  Both APIs use the same lowering
    routine; the estimator reduces the schedule immediately, while trace
    replay can execute the returned graph through the detailed DES engine.
    """

    with _compilation_scope(scenario):
        return _lower_serving_cohort(scenario, cohort).schedule


def _estimate_serving_cohort_cost(
    scenario: ScenarioConfig,
    cohort: object,
    *,
    cached_plan: Optional[ParallelPlan] = None,
    cached_router: Optional[TopologyRouter] = None,
    scenario_hash: Optional[str] = None,
    execution_control: Optional[ExecutionControl] = None,
    compiled_executor: Optional[CompiledGraphExecutor] = None,
) -> Mapping[str, object]:
    """Lower one realized online cohort through the topology-aware planner.

    The serving scheduler owns admission, batching, and KV capacity state.  The
    planner owns local tensor shapes, CIM/GPU costs, collectives, and physical
    transfers.  Keeping this as a callback avoids a dependency from planner to
    the scheduler module while ensuring online results use the same analytical
    hardware model as static runs.
    """

    lowering = _lower_serving_cohort(
        scenario,
        cohort,
        cached_plan=cached_plan,
        cached_router=cached_router,
        scenario_hash=scenario_hash,
    )
    summary = execute_cost_schedule(
        lowering.schedule,
        control=execution_control,
        compiled_executor=compiled_executor,
    )
    kv_traffic = _summarize_kv_task_traffic(lowering.schedule.tasks)
    resource_busy_by_direction = _resource_busy_by_direction(
        lowering.schedule.tasks
    )
    host_orchestration_tasks = tuple(
        task
        for task in lowering.schedule.tasks
        if task.metadata.get("orchestration_stage")
        in {"host_prefix", "host_target_frontend", "host_suffix"}
    )
    host_orchestration_ns = sum(
        max((demand.service_ns for demand in task.demands), default=0.0)
        for task in host_orchestration_tasks
    )
    raw_invocation_groups = lowering.extra_metadata.get(
        "operator_invocation_groups", ()
    )
    invocation_groups = (
        tuple(raw_invocation_groups)
        if isinstance(raw_invocation_groups, Sequence)
        and not isinstance(raw_invocation_groups, (str, bytes))
        else ()
    )
    execution_stages, stage_fallback_reason = _compact_execution_stages(
        scenario,
        summary.execution_records,
        invocation_groups,
    )
    explicit_host_stages = tuple(
        stage
        for stage in execution_stages
        if stage.get("stage_role")
        in {"host_prefix", "host_target_frontend", "host_suffix"}
    )
    execution_stages_include_host_orchestration = bool(explicit_host_stages)
    if execution_stages_include_host_orchestration:
        host_orchestration_ns = sum(
            float(stage["service_ns"]) for stage in explicit_host_stages
        )
    return {
        "duration_ns": summary.makespan_ns,
        "energy_pj": summary.energy_pj,
        "metadata": {
            "model": lowering.model,
            "task_count": summary.task_count,
            "resource_accounted_bytes": summary.bytes_moved,
            "resource_busy_ns": dict(summary.resource_busy_ns),
            "resource_busy_by_direction_ns": resource_busy_by_direction,
            "category_time_ns": {
                category.value: value
                for category, value in summary.category_time_ns.items()
            },
            "critical_path_category_ns": {
                category.value: value
                for category, value in summary.critical_path_category_ns.items()
            },
            "coverage": summary.coverage,
            "coverage_references": _analytical_coverage_references(
                lowering.schedule.tasks
            ),
            "linear_state_bytes": summary.linear_state_bytes,
            "host_orchestration_ns": host_orchestration_ns,
            "device_execution_ns": max(
                0.0, summary.makespan_ns - host_orchestration_ns
            ),
            "host_orchestration_task_count": len(
                host_orchestration_tasks
            ),
            "host_submission_count": sum(
                int(task.metadata.get("submission_count", 0))
                for task in host_orchestration_tasks
            ),
            "execution_stages": execution_stages,
            "execution_stages_include_host_orchestration": (
                execution_stages_include_host_orchestration
            ),
            "execution_stage_makespan_ns": _stage_dag_makespan(
                execution_stages
            ),
            "execution_stage_source": (
                "executed_task_dag_kernel_timeline"
                if execution_stages
                else "serial_fallback"
            ),
            "execution_stage_fallback_reason": stage_fallback_reason,
            **kv_traffic,
            **dict(lowering.extra_metadata),
        },
    }


def _logical_transfer_name(name: str) -> str:
    return re.sub(r"\.(?:link\d+|local)$", "", str(name))


def _summarize_kv_task_traffic(
    tasks: Sequence[TaskSpec],
) -> Mapping[str, int]:
    """Reduce routed phases to one logical/physical byte count per KV access."""

    grouped: Dict[Tuple[str, str, str, object, object], Tuple[int, int]] = {}
    for task in tasks:
        event_kind = str(task.metadata.get("event_kind", ""))
        if event_kind not in {"kv_read", "kv_append", "kv_prefetch", "kv_offload"}:
            continue
        key = (
            event_kind,
            str(task.request_id),
            str(
                task.metadata.get(
                    "kv_access_id", _logical_transfer_name(task.name)
                )
            ),
            task.metadata.get("rank"),
            task.metadata.get("layer_id"),
        )
        physical = max(
            0,
            int(task.metadata.get("physical_bytes", task.metadata.get("bytes", 0))),
        )
        logical = max(0, int(task.metadata.get("logical_bytes", physical)))
        previous = grouped.get(key, (0, 0))
        grouped[key] = (max(previous[0], logical), max(previous[1], physical))
    rows: Dict[str, int] = {}
    for (event_kind, _request, _name, _rank, _layer), (logical, physical) in grouped.items():
        rows["{}_logical_bytes".format(event_kind)] = (
            rows.get("{}_logical_bytes".format(event_kind), 0) + logical
        )
        rows["{}_physical_bytes".format(event_kind)] = (
            rows.get("{}_physical_bytes".format(event_kind), 0) + physical
        )
    return rows


def _task_memory_direction(task: TaskSpec) -> str:
    explicit = str(task.metadata.get("memory_direction", "")).lower()
    if explicit in {"read", "write"}:
        return explicit
    text = " ".join(
        (
            str(task.metadata.get("event_kind", "")),
            str(task.metadata.get("phase", "")),
            str(task.name),
        )
    ).lower()
    return "write" if any(
        token in text for token in ("append", "write", "store", "output")
    ) else "read"


def _resource_busy_by_direction(
    tasks: Sequence[TaskSpec],
) -> Mapping[str, Mapping[str, float]]:
    rows: Dict[str, Dict[str, float]] = {"read": {}, "write": {}}
    for task in tasks:
        direction = _task_memory_direction(task)
        raw_resource_directions = task.metadata.get("resource_directions", {})
        resource_directions = (
            raw_resource_directions
            if isinstance(raw_resource_directions, Mapping)
            else {}
        )
        modeled_write = max(
            0, int(task.metadata.get("modeled_memory_write_bytes", 0))
        )
        for demand in task.demands:
            resource_id = str(demand.resource_id)
            explicit_resource_direction = str(
                resource_directions.get(resource_id, "")
            ).lower()
            resource_direction = explicit_resource_direction or direction
            if resource_direction not in {"read", "write"}:
                resource_direction = direction
            service_ns = max(0.0, float(demand.service_ns))
            if service_ns <= 0.0:
                continue
            if explicit_resource_direction in {"read", "write"}:
                rows[resource_direction][resource_id] = (
                    rows[resource_direction].get(resource_id, 0.0)
                    + service_ns
                )
                continue
            write_share = 0.0
            if modeled_write > 0 and demand.bytes_moved > 0:
                write_share = min(
                    1.0, modeled_write / float(demand.bytes_moved)
                )
            if write_share > 0.0:
                rows["write"][resource_id] = (
                    rows["write"].get(resource_id, 0.0)
                    + service_ns * write_share
                )
                rows["read"][resource_id] = (
                    rows["read"].get(resource_id, 0.0)
                    + service_ns * (1.0 - write_share)
                )
            else:
                rows[resource_direction][resource_id] = (
                    rows[resource_direction].get(resource_id, 0.0) + service_ns
                )
    return {
        direction: dict(sorted(values.items()))
        for direction, values in rows.items()
        if values
    }


def _lower_serving_cohort(
    scenario: ScenarioConfig,
    cohort: object,
    *,
    cached_plan: Optional[ParallelPlan] = None,
    cached_router: Optional[TopologyRouter] = None,
    scenario_hash: Optional[str] = None,
) -> _ServingCohortLowering:
    cohort_id = str(getattr(cohort, "cohort_id", "online-cohort"))
    kind = str(getattr(cohort, "kind", "decode"))
    if kind in {
        "kv_swap_out",
        "kv_swap_in",
        "linear_state_swap_out",
        "linear_state_swap_in",
    }:
        return _lower_serving_transfer(
            scenario,
            cohort,
            cohort_id=cohort_id,
            kind=kind,
            cached_plan=cached_plan,
            cached_router=cached_router,
            scenario_hash=scenario_hash,
        )

    items = tuple(getattr(cohort, "items", ()) or ())
    if not items:
        raise ValueError("服务批次至少需要包含一个条目")
    mtp_items = _serving_mtp_items(cohort)
    token_batch = sum(max(0, int(getattr(item, "token_count", 0))) for item in items)
    if token_batch <= 0:
        raise ValueError("服务批次的 token_count 必须为正数")
    invocation_groups = _serving_invocation_groups(scenario, cohort)
    if not invocation_groups:
        raise ValueError("服务批次没有可执行的 backend invocation group")
    draft_counts = tuple(
        _serving_item_draft_tokens(item) for item in mtp_items
    )
    draft_step_lanes = tuple(
        sum(drafts > step for drafts in draft_counts)
        for step in range(max(draft_counts, default=0))
    )
    draft_step_request_ids = tuple(
        tuple(
            str(getattr(item, "request_id", ""))
            for item, drafts in zip(mtp_items, draft_counts)
            if drafts > step
        )
        for step in range(len(draft_step_lanes))
    )
    scheduler = scenario.workload.scheduler
    physical_ubatch_rows = (
        max(1, int(scheduler.max_num_batched_tokens))
        if scheduler is not None and scheduler.max_num_ubatch_tokens is None
        else (
            max(1, int(scheduler.max_num_ubatch_tokens))
            if scheduler is not None
            else token_batch
        )
    )
    proposer_groups: List[Mapping[str, object]] = []
    proposer_predecessor_group_id: Optional[str] = None
    for draft_step, request_ids in enumerate(draft_step_request_ids):
        for chunk_index, row_start in enumerate(
            range(0, len(request_ids), physical_ubatch_rows)
        ):
            chunk_request_ids = tuple(
                request_ids[row_start : row_start + physical_ubatch_rows]
            )
            group_id = (
                "mtp-proposer-invocation-group-token0000-draft{:04d}"
            ).format(draft_step)
            if chunk_index > 0:
                group_id += "-ubatch{:04d}".format(chunk_index)
            proposer_groups.append(
                {
                    "group_id": group_id,
                    "group_index": len(proposer_groups),
                    "kind": "mtp_proposer",
                    "request_ids": chunk_request_ids,
                    "lane_count": len(chunk_request_ids),
                    "physical_ubatch_rows": len(chunk_request_ids),
                    "physical_chunk_index": chunk_index,
                    "physical_row_start": row_start,
                    "draft_step": draft_step,
                    "predecessor_group_id": proposer_predecessor_group_id,
                    "batching_semantics": "mtp_draft_step_physical_ubatch",
                    "residency_role": "mtp_draft_context",
                }
            )
            proposer_predecessor_group_id = group_id
    mtp_proposer_invocation_groups = tuple(proposer_groups)
    mtp_policy = scenario.workload.mtp
    mtp_draft_catchup_invocation_groups: Tuple[
        Mapping[str, object], ...
    ] = ()
    if mtp_policy is not None and mtp_policy.enabled:
        catchup_rows = tuple(
            str(getattr(item, "request_id", ""))
            for item in items
            for _ in range(max(0, int(getattr(item, "token_count", 0))))
        )
        catchup_groups: List[Mapping[str, object]] = []
        catchup_predecessor_group_id: Optional[str] = None
        catchup_token_offsets: Dict[str, int] = {}
        for chunk_index, row_start in enumerate(
            range(0, len(catchup_rows), physical_ubatch_rows)
        ):
            chunk_rows = catchup_rows[
                row_start : row_start + physical_ubatch_rows
            ]
            token_counts: Dict[str, int] = {}
            for request_id in chunk_rows:
                token_counts[request_id] = token_counts.get(request_id, 0) + 1
            token_offsets = tuple(
                (request_id, catchup_token_offsets.get(request_id, 0))
                for request_id in token_counts
            )
            group_id = (
                "mtp-draft-catchup-invocation-group-token0000-batch{:04d}"
            ).format(chunk_index)
            catchup_groups.append(
                {
                    "group_id": group_id,
                    "group_index": chunk_index,
                    "kind": "mtp_draft_context_catchup",
                    "request_ids": tuple(token_counts),
                    "lane_count": len(chunk_rows),
                    "physical_ubatch_rows": len(chunk_rows),
                    "physical_chunk_index": chunk_index,
                    "physical_row_start": row_start,
                    "token_count_by_request": tuple(token_counts.items()),
                    "token_offset_by_request": token_offsets,
                    "draft_step": 0,
                    "predecessor_group_id": catchup_predecessor_group_id,
                    "batching_semantics": (
                        "mtp_draft_context_catchup_physical_ubatch"
                    ),
                    "residency_role": "mtp_draft_context_catchup",
                }
            )
            for request_id, count in token_counts.items():
                catchup_token_offsets[request_id] = (
                    catchup_token_offsets.get(request_id, 0) + count
                )
            catchup_predecessor_group_id = group_id
        mtp_draft_catchup_invocation_groups = tuple(catchup_groups)
    physical_invocation_group_counts = {
        "target_operator": len(invocation_groups),
        "mtp_proposer": len(mtp_proposer_invocation_groups),
        "mtp_draft_catchup": len(
            mtp_draft_catchup_invocation_groups
        ),
    }
    physical_invocation_group_count = sum(
        physical_invocation_group_counts.values()
    )
    prior_attention_pairs = _serving_cohort_prior_attention_pairs(cohort)
    (
        prior_kv_token_reads,
        internal_causal_pairs,
        kv_append_tokens,
    ) = _serving_cohort_kv_shape(cohort)
    kv_materialized_tokens = _serving_cohort_kv_materialized_tokens(cohort)
    # A chunk of N tokens has triangular causal work inside the chunk in
    # addition to historical-prompt attention.  Both remain compute work.
    # ``prior_kv_token_reads`` is narrower: it is external persisted-cache
    # traffic.  Prefill/recompute count each historical token once per fused
    # query chunk; decode/MTP retain their per-query access semantics.
    causal_attention_pairs = prior_attention_pairs + internal_causal_pairs
    context_tokens = max(
        1, int(math.ceil(causal_attention_pairs / float(token_batch)))
    )
    request = RequestSpec(
        request_id=cohort_id,
        arrival_ns=0.0,
        prompt_tokens=max(1, token_batch),
        output_tokens=0,
    )
    builder = _TaskBuilder(request)
    start = builder.add(
        cohort_id + ".start",
        TaskCategory.POLICY,
        dependency="",
        metadata={
            "event_kind": "serving_cohort_start",
            "participant_request_ids": tuple(
                str(getattr(item, "request_id", "")) for item in items
            ),
            "cohort_kind": kind,
        },
    )
    plan = cached_plan if cached_plan is not None else _parallel_plan(scenario)
    router = (
        cached_router
        if cached_router is not None
        else _topology_router(scenario)
    )
    prepared = _add_host_orchestration(
        builder,
        scenario,
        router,
        plan,
        (start,),
        name=cohort_id + ".host_orchestration",
        request_count=len(items),
        token_count=token_batch,
    )
    dependencies: Tuple[str, ...] = (prepared,)
    proposer_group_ends: Dict[str, str] = {}
    if mtp_items:
        accepted_tokens = sum(
            _serving_item_committed_tokens(item) for item in mtp_items
        )
        verifier_tokens = sum(
            _serving_item_verifier_tokens(item) for item in mtp_items
        )
        if mtp_proposer_invocation_groups:
            dependencies = (
                _add_physical_invocation_frontend(
                    builder,
                    scenario,
                    plan,
                    dependencies,
                    name=cohort_id + ".mtp_proposer_frontend",
                    request_count=len(items),
                    token_count=token_batch,
                    invocation_count=len(mtp_proposer_invocation_groups),
                    invocation_family="mtp_proposer",
                    orchestration_stage="host_prefix",
                    execution_phase=kind,
                    invocation_group_ids=tuple(
                        str(group["group_id"])
                        for group in mtp_proposer_invocation_groups
                    ),
                ),
            )
        proposal = dependencies[0]
        for group in mtp_proposer_invocation_groups:
            group_id = str(group["group_id"])
            predecessor_group_id = group.get("predecessor_group_id")
            group_dependencies = dependencies
            if predecessor_group_id is not None:
                group_dependencies = (
                    proposer_group_ends[str(predecessor_group_id)],
                )
            proposal = _compile_parallel_mtp_proposer(
                builder,
                scenario,
                plan,
                router,
                next_token=max(0, int(group["group_index"])),
                draft_step_lanes=(max(1, int(group["lane_count"])),),
                verifier_tokens=verifier_tokens,
                accepted_tokens=accepted_tokens,
                policy=scenario.workload.mtp or cohort,
                dependencies=group_dependencies,
                draft_step_request_ids=(tuple(group["request_ids"]),),
                draft_step_group_ids=(group_id,),
                draft_step_indices=(max(0, int(group["draft_step"])),),
                draft_step_group_indices=(
                    max(0, int(group["group_index"])),
                ),
                draft_step_chunk_indices=(
                    max(0, int(group["physical_chunk_index"])),
                ),
            )
            proposer_group_ends[group_id] = proposal
        if not mtp_proposer_invocation_groups:
            proposal = _compile_parallel_mtp_proposer(
                builder,
                scenario,
                plan,
                router,
                next_token=0,
                draft_step_lanes=(),
                verifier_tokens=verifier_tokens,
                accepted_tokens=accepted_tokens,
                policy=scenario.workload.mtp or cohort,
                dependencies=dependencies,
            )
        dependencies = (proposal,)
    target_frontend = _add_physical_invocation_frontend(
        builder,
        scenario,
        plan,
        dependencies,
        name=cohort_id + ".target_operator_frontend",
        request_count=len(items),
        token_count=token_batch,
        invocation_count=len(invocation_groups),
        invocation_family="target_operator",
        orchestration_stage=(
            "host_target_frontend" if mtp_items else "host_prefix"
        ),
        execution_phase=kind,
        first_decode_invocation=(
            kind == "decode"
            and all(
                int(getattr(item, "context_tokens", 0) or 0)
                == int(next((r.prompt_tokens for r in scenario.workload.requests
                             if r.request_id == str(getattr(item, "request_id", ""))), -1))
                for item in items
            )
        ),
        invocation_group_ids=tuple(group.group_id for group in invocation_groups),
    )
    dependencies = (target_frontend,)
    group_ends: Dict[str, str] = {}
    for group in invocation_groups:
        group_dependencies = dependencies
        if group.predecessor_group_id is not None:
            group_dependencies = (
                group_ends[group.predecessor_group_id],
            )
        group_phase = "{}.{}".format(cohort_id, kind)
        if len(invocation_groups) > 1:
            group_phase += ".group{:04d}".format(group.group_index)
        group_end = _compile_or_replay_serving_invocation(
            builder,
            scenario,
            plan,
            router,
            group,
            phase=group_phase,
            dependencies=group_dependencies,
        )
        group_ends[group.group_id] = group_end
    completion_dependencies: Tuple[str, ...] = tuple(group_ends.values())
    equal_length_batch = bool(invocation_groups) and all(
        group.batching_semantics == "explicit_equal_length_stateful_ubatch"
        for group in invocation_groups
    )
    if equal_length_batch and any(group.logit_token_batch for group in invocation_groups):
        first_output_task = len(builder.tasks)
        completion_dependencies = (_add_host_visible_logits_sampling_commit(
            builder, scenario, plan, router, completion_dependencies,
            phase="{}.{}".format(cohort_id, kind),
            logit_rows=sum(group.logit_token_batch for group in invocation_groups),
            committed_rows=sum(group.committed_logit_token_batch for group in invocation_groups),
        ),)
        output_sources = tuple(
            lane.identity_metadata(kind) for group in invocation_groups
            for lane in group.lanes if lane.requires_logits
        )
        for index in range(first_output_task, len(builder.tasks)):
            task = builder.tasks[index]
            builder.tasks[index] = replace(task, metadata={
                **task.metadata,
                "operator_invocation_group_ids": tuple(group.group_id for group in invocation_groups),
                "request_ids": tuple(dict.fromkeys(source["request_id"] for source in output_sources)),
                "logit_row_sources": output_sources,
                "output_visibility_boundary": "all_physical_ubatches_complete",
            })
    catchup_group_ends: Dict[str, str] = {}
    if mtp_policy is not None and mtp_policy.enabled:
        catchup_dependencies = completion_dependencies
        if mtp_draft_catchup_invocation_groups:
            catchup_dependencies = (
                _add_physical_invocation_frontend(
                    builder,
                    scenario,
                    plan,
                    completion_dependencies,
                    name=cohort_id + ".mtp_draft_catchup_frontend",
                    request_count=len(items),
                    token_count=token_batch,
                    invocation_count=len(mtp_draft_catchup_invocation_groups),
                    invocation_family="mtp_draft_catchup",
                    orchestration_stage="host_suffix",
                    execution_phase=kind,
                    invocation_group_ids=tuple(
                        str(group["group_id"])
                        for group in mtp_draft_catchup_invocation_groups
                    ),
                ),
            )
        catchup_end = catchup_dependencies[0]
        for group in mtp_draft_catchup_invocation_groups:
            group_id = str(group["group_id"])
            predecessor_group_id = group.get("predecessor_group_id")
            group_dependencies = catchup_dependencies
            if predecessor_group_id is not None:
                group_dependencies = (
                    catchup_group_ends[str(predecessor_group_id)],
                )
            catchup_end = _compile_parallel_mtp_draft_catchup(
                builder,
                scenario,
                plan,
                router,
                token_batch=max(1, int(group["lane_count"])),
                request_ids=tuple(group["request_ids"]),
                policy=mtp_policy,
                dependencies=group_dependencies,
                invocation_group_id=group_id,
                invocation_group_index=max(0, int(group["group_index"])),
                physical_chunk_index=max(
                    0, int(group["physical_chunk_index"])
                ),
            )
            catchup_group_ends[group_id] = catchup_end
        completion_dependencies = (catchup_end,)
    item_terminal_rows: List[Mapping[str, object]] = []
    item_terminal_ids: List[str] = []
    for item_index, item in enumerate(items):
        request_id = str(getattr(item, "request_id", ""))
        item_phase = str(getattr(item, "phase", kind))
        raw_cursor = getattr(item, "completion_cursor", None)
        completion_cursor = (
            None if raw_cursor is None else max(0, int(raw_cursor))
        )
        completion_key = "item{:04d}:{}:{}:{}".format(
            item_index,
            request_id,
            item_phase,
            (
                "cursor-unknown"
                if completion_cursor is None
                else "cursor-{:08d}".format(completion_cursor)
            ),
        )
        covered_group_ids = tuple(
            group.group_id
            for group in invocation_groups
            if any(lane.item_index == item_index for lane in group.lanes)
        )
        terminal_dependencies = (
            completion_dependencies
            if equal_length_batch or (mtp_policy is not None and mtp_policy.enabled)
            else tuple(group_ends[group_id] for group_id in covered_group_ids)
        )
        if not terminal_dependencies:
            terminal_dependencies = completion_dependencies
        terminal_task_id = builder.add(
            "{}.item{:04d}.complete".format(cohort_id, item_index),
            TaskCategory.SYNCHRONIZATION,
            dependencies=terminal_dependencies,
            advance=False,
            metadata={
                "event_kind": "serving_item_complete",
                "cohort_id": cohort_id,
                "item_index": item_index,
                "serving_request_id": request_id,
                "serving_phase": item_phase,
                "completion_cursor": completion_cursor,
                "item_completion_key": completion_key,
                "covered_operator_invocation_group_ids": (
                    covered_group_ids
                ),
            },
        )
        item_terminal_ids.append(terminal_task_id)
        item_terminal_rows.append(
            {
                "item_index": item_index,
                "request_id": request_id,
                "phase": item_phase,
                "completion_cursor": completion_cursor,
                "item_completion_key": completion_key,
                "terminal_task_id": terminal_task_id,
                "operator_invocation_group_ids": covered_group_ids,
            }
        )
    cohort_terminal_id = builder.add(
        cohort_id + ".complete",
        TaskCategory.SYNCHRONIZATION,
        dependencies=tuple(item_terminal_ids) or completion_dependencies,
        metadata={"event_kind": "serving_cohort_complete", "cohort_id": cohort_id},
    )
    sampling_policy = scenario.sampling_policy
    legacy_greedy_sampling_complete = (
        sampling_policy is not None
        and sampling_policy.mode.strip().lower() == "greedy"
        and sampling_policy.implementation is None
    )
    sampling_modeling_partial_reason = (
        None
        if legacy_greedy_sampling_complete
        else (
            "candidate_materialization_compiler_cache_and_sampler_filter_"
            "chain_cost_unknown"
            if sampling_policy is not None
            and sampling_policy.implementation is not None
            and sampling_policy.implementation.strip().lower()
            == _LLAMA_CPP_CPU_SAMPLER_IMPLEMENTATION
            else "sampling_algorithm_not_declared_or_modeled"
        )
    )
    return _serving_lowering_from_builder(
        scenario,
        builder,
        cohort_id=cohort_id,
        kind=kind,
        model="topology_aware_parallel_lowering",
        assumptions=(
            "online cohort lowered through auditable backend invocation groups",
        ),
        scenario_hash=scenario_hash,
        extra_metadata={
            "token_batch": token_batch,
            "physical_batch_rows": token_batch,
            "max_num_ubatch_tokens": (
                scenario.workload.scheduler.max_num_ubatch_tokens
                if scenario.workload.scheduler is not None
                and scenario.workload.scheduler.max_num_ubatch_tokens
                is not None
                else (
                    scenario.workload.scheduler.max_num_batched_tokens
                    if scenario.workload.scheduler is not None
                    else token_batch
                )
            ),
            "phase_item_counts": {
                phase: sum(
                    str(getattr(item, "phase", "")) == phase
                    for item in items
                )
                for phase in tuple(
                    dict.fromkeys(str(getattr(item, "phase", "")) for item in items)
                )
            },
            "phase_token_counts": {
                phase: sum(
                    max(0, int(getattr(item, "token_count", 0)))
                    for item in items
                    if str(getattr(item, "phase", "")) == phase
                )
                for phase in tuple(
                    dict.fromkeys(str(getattr(item, "phase", "")) for item in items)
                )
            },
            "ragged_context_mean": context_tokens,
            "ragged_prior_kv_token_reads": prior_kv_token_reads,
            "causal_attention_pairs": causal_attention_pairs,
            "kv_append_tokens": kv_append_tokens,
            "kv_materialized_tokens": kv_materialized_tokens,
            "scheduler_cohort_item_count": len(items),
            "backend_subbatch_count": len(invocation_groups),
            "operator_invocation_group_count": len(invocation_groups),
            "physical_invocation_group_count": (
                physical_invocation_group_count
            ),
            "physical_invocation_group_counts": {
                **physical_invocation_group_counts,
                "total": physical_invocation_group_count,
            },
            "target_backbone_invocation_count": len(invocation_groups),
            "operator_invocation_groups": tuple(
                {
                    **dict(group.audit_metadata()),
                    "terminal_task_id": group_ends[group.group_id],
                    "consumer_task_ids": (group_ends[group.group_id],),
                }
                for group in invocation_groups
            ),
            "item_terminal_tasks": tuple(item_terminal_rows),
            "cohort_terminal_task_id": cohort_terminal_id,
            "mtp_proposer_invocation_groups": (
                tuple(
                    {
                        **dict(group),
                        "terminal_task_id": proposer_group_ends[
                            str(group["group_id"])
                        ],
                        "consumer_task_ids": (
                            proposer_group_ends[str(group["group_id"])],
                        ),
                    }
                    for group in mtp_proposer_invocation_groups
                )
            ),
            "mtp_draft_catchup_invocation_groups": (
                tuple(
                    {
                        **dict(group),
                        "terminal_task_id": catchup_group_ends[
                            str(group["group_id"])
                        ],
                        "consumer_task_ids": (
                            catchup_group_ends[str(group["group_id"])],
                        ),
                    }
                    for group in mtp_draft_catchup_invocation_groups
                )
            ),
            "mtp_draft_catchup_rows": (
                token_batch if mtp_draft_catchup_invocation_groups else 0
            ),
            "mtp_draft_catchup_aux_logit_rows": 0,
            "execution_group_kv_read_tokens": sum(
                group.kv_read_tokens for group in invocation_groups
            ),
            "execution_group_logit_tokens": sum(
                group.logit_token_batch for group in invocation_groups
            ),
            "target_logit_rows": sum(
                group.logit_token_batch for group in invocation_groups
            ),
            **_host_output_contract_projection(scenario),
            "sampling_modeling_status": (
                "complete" if legacy_greedy_sampling_complete else "partial"
            ),
            "sampling_modeling_partial_reason": sampling_modeling_partial_reason,
            "completion_interrupt_modeling_status": "partial",
            "completion_interrupt_modeling_partial_reason": (
                "completion_interrupt_latency_not_declared"
            ),
            "logits_d2h_bytes": int(
                math.ceil(
                    sum(
                        group.logit_token_batch
                        for group in invocation_groups
                    )
                    * _host_visible_logits_vocabulary_size(scenario)
                    * _host_visible_logits_output_bits(scenario)[0]
                    / 8.0
                )
            ),
            "committed_logit_rows": sum(
                group.committed_logit_token_batch
                for group in invocation_groups
            ),
            "mtp_proposer_request_count": len(
                {
                    str(getattr(item, "request_id", ""))
                    for item in mtp_items
                }
            ),
            "mtp_proposer_verifier_rows": sum(
                _serving_item_verifier_tokens(item) for item in mtp_items
            ),
            "main_tokens": (
                sum(_serving_item_main_tokens(item) for item in mtp_items)
            ),
            "draft_tokens": (
                sum(_serving_item_draft_tokens(item) for item in mtp_items)
            ),
            "verifier_tokens": sum(
                _serving_item_verifier_tokens(item) for item in mtp_items
            ),
            "committed_tokens": (
                sum(_serving_item_committed_tokens(item) for item in mtp_items)
            ),
            "kv_read_semantics": "external_persisted_kv_cache_only",
            "tp_degree": plan.tp_degree,
            "pp_degree": plan.pp_degree,
            "ep_degree": plan.ep_degree,
        },
    )


def _lower_serving_transfer(
    scenario: ScenarioConfig,
    cohort: object,
    *,
    cohort_id: str,
    kind: str,
    cached_plan: Optional[ParallelPlan],
    cached_router: Optional[TopologyRouter],
    scenario_hash: Optional[str],
) -> _ServingCohortLowering:
    raw_metadata = getattr(cohort, "metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    source = str(metadata.get("source_component", ""))
    target = str(metadata.get("target_component", ""))
    byte_count = int(metadata.get("byte_count", 0))
    if not source or not target or source == target:
        raise ValueError("交换传输的源组件与目标组件不能相同")
    if byte_count <= 0:
        raise ValueError("交换传输的 byte_count 必须为正数")
    request = RequestSpec(
        request_id=cohort_id,
        arrival_ns=0.0,
        prompt_tokens=1,
        output_tokens=0,
    )
    builder = _TaskBuilder(request)
    start = builder.add(
        cohort_id + ".start",
        TaskCategory.POLICY,
        dependency="",
        metadata={"event_kind": "serving_transfer_start", "kind": kind},
    )
    plan = cached_plan if cached_plan is not None else _parallel_plan(scenario)
    router = (
        cached_router
        if cached_router is not None
        else _topology_router(scenario)
    )
    control = _add_swap_control(
        builder,
        scenario,
        plan,
        (start,),
        name="{}.{}.control".format(cohort_id, kind),
        page_count=int(metadata.get("page_count", 0)),
        byte_count=byte_count,
    )
    end = _add_transfer_tasks(
        builder,
        router,
        source,
        target,
        byte_count,
        (control,),
        name="{}.{}".format(cohort_id, kind),
        routing_policy=plan.routing_policy,
        metadata={"event_kind": kind, "bytes": byte_count},
    )
    builder.add(
        cohort_id + ".complete",
        TaskCategory.SYNCHRONIZATION,
        dependencies=(end,),
        metadata={"event_kind": "serving_transfer_complete", "kind": kind},
    )
    return _serving_lowering_from_builder(
        scenario,
        builder,
        cohort_id=cohort_id,
        kind=kind,
        model="topology_aware_swap_transfer",
        assumptions=("swap I/O follows the declared physical topology",),
        scenario_hash=scenario_hash,
        extra_metadata={
            "transfer_bytes": byte_count,
            "source_component": source,
            "target_component": target,
        },
    )


def _serving_lowering_from_builder(
    scenario: ScenarioConfig,
    builder: _TaskBuilder,
    *,
    cohort_id: str,
    kind: str,
    model: str,
    assumptions: Tuple[str, ...],
    scenario_hash: Optional[str],
    extra_metadata: Mapping[str, object],
) -> _ServingCohortLowering:
    effective_scenario_hash = scenario_hash or stable_hash(scenario)
    enriched_extra_metadata = dict(extra_metadata)
    row_count = max(
        0,
        int(
            enriched_extra_metadata.get(
                "physical_batch_rows",
                enriched_extra_metadata.get("token_batch", 0),
            )
            or 0
        ),
    )
    raw_tasks = tuple(builder.tasks)
    operator_facts, transient_envelopes = (
        _transient_residency_envelopes(
            scenario,
            raw_tasks,
            cohort_id=cohort_id,
            row_count=row_count,
        )
    )
    tasks = _annotate_transient_residency_facts(
        raw_tasks, operator_facts, transient_envelopes
    )
    enriched_extra_metadata["transient_residency_operator_facts"] = (
        operator_facts
    )
    enriched_extra_metadata["transient_residency_envelopes"] = (
        transient_envelopes
    )
    enriched_extra_metadata["residency_accesses"] = (
        _ordered_residency_accesses(tasks, transient_envelopes)
    )
    manifest_assumptions_zh, manifest_assumptions_en = localized_manifest_assumptions(
        assumptions
    )
    manifest = RunManifest(
        schema_version=scenario.schema_version,
        run_id="{}-{}".format(effective_scenario_hash[:12], cohort_id),
        random_seed=scenario.workload.random_seed,
        simulator_version=__version__,
        model_name=scenario.model.name,
        hardware_name=scenario.hardware.name,
        workload_name=scenario.workload.name,
        calibration_version=ANALYTICAL_MODEL_VERSION,
        evidence=EvidenceStatus.ANALYTICAL,
        assumptions=assumptions,
        assumptions_zh=manifest_assumptions_zh,
        assumptions_en=manifest_assumptions_en,
        metadata={"cohort_id": cohort_id, "cohort_kind": kind},
    )
    iq_panel_contract = scenario.workload.metadata.get("llama_cpp_cpu_iq_panel_reuse")
    if isinstance(iq_panel_contract, Mapping) and iq_panel_contract.get("enabled") is True:
        iq_panel_coverage = summarize_cpu_iq_panel_reuse(tasks)
        enriched_extra_metadata["cpu_iq_panel_reuse"] = iq_panel_coverage
        manifest = replace(manifest, metadata={
            **manifest.metadata, "cpu_iq_panel_reuse": iq_panel_coverage,
        })
    return _ServingCohortLowering(
        schedule=ScheduleIR(manifest=manifest, tasks=tasks),
        cohort_id=cohort_id,
        kind=kind,
        model=model,
        extra_metadata=enriched_extra_metadata,
    )


def summarize_analytical_coverage(
    tasks: Sequence[object],
) -> Dict[str, Dict[str, object]]:
    totals: Dict[str, Dict[str, object]] = {}
    for task in tasks:
        metadata = getattr(task, "metadata", {})
        component = str(metadata.get("coverage_component", ""))
        name = str(getattr(task, "name", ""))
        event_kind = str(metadata.get("event_kind", ""))
        if not component:
            if event_kind.startswith("linear_state_"):
                component = "linear_state"
            elif event_kind.startswith("kv_"):
                component = "kv_cache"
            elif event_kind.startswith("mtp_") or name.startswith("mtp"):
                component = "mtp_policy"
            elif "shared_expert" in name:
                component = "shared_expert"
            elif "expert" in name or "moe_router" in name:
                component = "routed_expert"
            elif "linear_" in name:
                component = "linear_attention"
            elif any(
                marker in name
                for marker in (".qkv", ".attention_", ".kv.")
            ):
                component = "full_attention"
            elif ".mlp_" in name:
                component = "dense_ffn"
        if not component:
            continue
        row = totals.setdefault(
            component,
            {
                "task_count": 0.0,
                "operations": 0.0,
                "bytes": 0.0,
                "latency_ns": 0.0,
                "energy_pj": 0.0,
                "operator_ids": set(),
                "tensor_ids": set(),
            },
        )
        row["task_count"] = float(row["task_count"]) + 1.0
        row["operations"] = float(row["operations"]) + float(
            metadata.get("analytical_ops", 0.0)
        )
        intervals = tuple(getattr(task, "resource_intervals", ()) or ())
        row["bytes"] = float(row["bytes"]) + sum(
            float(getattr(interval, "bytes_moved", 0.0))
            for interval in intervals
        )
        row["energy_pj"] = float(row["energy_pj"]) + sum(
            float(getattr(interval, "energy_pj", 0.0))
            for interval in intervals
        )
        row["latency_ns"] = float(row["latency_ns"]) + float(
            getattr(task, "duration_ns", 0.0)
        )
        operator_id = metadata.get(
            "model_operator_id", metadata.get("operator_id")
        )
        tensor_id = metadata.get(
            "weight_tensor_id", metadata.get("tensor_id")
        )
        if operator_id:
            cast_ids = row["operator_ids"]
            if isinstance(cast_ids, set):
                cast_ids.add(str(operator_id))
        if tensor_id:
            cast_ids = row["tensor_ids"]
            if isinstance(cast_ids, set):
                cast_ids.add(str(tensor_id))
    result: Dict[str, Dict[str, object]] = {}
    for name, values in sorted(totals.items()):
        result[name] = {
            key: (
                sorted(str(item) for item in value)
                if isinstance(value, set)
                else value
            )
            for key, value in values.items()
        }
    return result


def _analytical_coverage_references(
    tasks: Sequence[object],
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Compact replay-safe operator/tensor references for reduced cohorts."""

    rows: Dict[str, Dict[str, set]] = {}
    for task in tasks:
        metadata = getattr(task, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        component = str(metadata.get("coverage_component", ""))
        if not component:
            continue
        row = rows.setdefault(
            component, {"operator_ids": set(), "tensor_ids": set()}
        )
        operator_id = metadata.get(
            "model_operator_id", metadata.get("operator_id")
        )
        tensor_id = metadata.get(
            "weight_tensor_id", metadata.get("tensor_id")
        )
        if operator_id:
            row["operator_ids"].add(str(operator_id))
        if tensor_id:
            row["tensor_ids"].add(str(tensor_id))
    return {
        component: {
            key: tuple(sorted(str(item) for item in values))
            for key, values in row.items()
        }
        for component, row in sorted(rows.items())
    }


def _shared_transfer_demands(
    scenario: ScenarioConfig,
    rank: LogicalRank,
    byte_count: int,
) -> Tuple[ResourceDemand, ...]:
    link = scenario.cim_interconnect
    if link is None:
        raise ValueError("场景未定义 CIM interconnect")
    _gpu_profile, hbm_profile = _gpu_profiles(
        scenario,
        rank.component_id,
        rank.memory_component_id,
    )
    return (
        ResourceDemand(
            _rank_memory_resource(scenario, rank),
            byte_count / hbm_profile.effective_bandwidth_gb_s,
            bytes_moved=byte_count,
            energy_pj=byte_count * hbm_profile.energy_pj_per_byte,
        ),
        ResourceDemand(
            link.resource_id,
            link.transfer_ns(byte_count),
            bytes_moved=byte_count,
            energy_pj=byte_count * link.energy_pj_per_byte,
        ),
    )


def _resolve_target(scenario: ScenarioConfig, layer: LayerSpec, group: str) -> str:
    mapping = scenario.placement.op_to_component
    keys = (
        "{}.{}".format(layer.layer_id, group),
        "*.{}".format(group),
        layer.layer_id,
        group,
    )
    for key in keys:
        if key in mapping:
            return str(mapping[key])
    return _gpu_component(scenario).component_id


_MIXED_ARTIFACT_QUANTIZATION = "IQ3_S-FFN-IQ4_XS"


def _materialize_weight_projection(
    layer: LayerSpec,
    projection_id: str,
    *,
    tp_degree: int = 1,
    tp_rank: int = 0,
    allow_padding: bool = True,
) -> Optional[_MaterializedWeightProjection]:
    """Memoize the pure descriptor materializer per compilation."""

    context = _COMPILATION_CONTEXT.get()
    if context is None:
        return materialize_weight_projection(
            layer.metadata,
            projection_id,
            tp_degree=tp_degree,
            tp_rank=tp_rank,
            allow_padding=allow_padding,
        )
    key = (
        "weight_projection_descriptor",
        context.metadata_source_identity(layer.metadata),
        projection_id,
        tp_degree,
        tp_rank,
        bool(allow_padding),
    )
    return context.invariant(
        key,
        lambda: materialize_weight_projection(
            layer.metadata,
            projection_id,
            tp_degree=tp_degree,
            tp_rank=tp_rank,
            allow_padding=allow_padding,
        ),
    )  # type: ignore[return-value]


def _qwen35_attention_source_work(layer: LayerSpec) -> Optional[_Qwen35AttentionSourceWork]:
    if _QWEN35_ATTENTION_SOURCE_KEY not in layer.metadata:
        return None
    if layer.is_linear_attention:
        raise ValueError("Qwen3.5 ordinary-attention source cannot describe a linear layer")
    resolve = lambda: _resolve_qwen35_source_work(
        layer.metadata, hidden_size=layer.hidden_size, attention_heads=layer.attention_heads,
        kv_heads=layer.effective_kv_heads, head_dim=layer.effective_attention_head_dim,
    )
    context = _COMPILATION_CONTEXT.get()
    if context is None:
        return resolve()
    return context.invariant(("qwen35_attention_source_work", context.metadata_source_identity(layer.metadata)), resolve)


def _resolved_attention_execution_descriptor(layer: LayerSpec) -> Optional[_AttentionExecutionDescriptor]:
    source = _qwen35_attention_source_work(layer)
    if source is not None:
        return source.attention
    return resolve_attention_execution_descriptor(
        layer.metadata, attention_heads=layer.attention_heads, kv_heads=layer.effective_kv_heads,
        head_dim=layer.effective_attention_head_dim, hidden_size=layer.hidden_size,
    )


def _attention_execution_descriptor(
    layer: LayerSpec,
) -> Optional[_AttentionExecutionDescriptor]:
    context = _COMPILATION_CONTEXT.get()
    if context is None:
        return _resolved_attention_execution_descriptor(layer)
    key = (
        "attention_execution_descriptor",
        context.metadata_source_identity(layer.metadata),
    )
    return context.invariant(
        key,
        lambda: _resolved_attention_execution_descriptor(layer),
    )  # type: ignore[return-value]


def _artifact_text(value: object) -> str:
    """Normalize an artifact label without accepting unknown formats."""

    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())


def _canonical_artifact_quantization(value: object) -> Optional[str]:
    """Return a supported artifact label, or ``None`` for a non-artifact.

    The compact aliases keep imported metadata such as ``iq3_s`` and
    ``IQ3-S`` equivalent while leaving unrelated ``wN`` contracts to the
    normal precision parser.
    """

    label = _canonical_primitive_artifact_quantization(value)
    if label is not None or value is None:
        return label
    if _artifact_text(value) == "IQ3SFFNIQ4XS":
        return _MIXED_ARTIFACT_QUANTIZATION
    return None


def _looks_like_artifact_quantization(value: object) -> bool:
    if not isinstance(value, str):
        return False
    compact = _artifact_text(value)
    return (
        compact.startswith("IQ")
        or _canonical_primitive_artifact_quantization(value) is not None
        or compact == "Q4KM"
        or "FFN" in compact
    )


def _metadata_value(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> object:
    """Find the first metadata key, traversing small nested contracts.

    The source ordering is significant: an explicit workload override comes
    first, then active workload/model metadata, then layer metadata.  Within a
    source, direct keys win over nested contract dictionaries.
    """

    context = _COMPILATION_CONTEXT.get()
    if context is not None:
        return context.metadata_value(sources, keys)

    wanted = {str(key).casefold() for key in keys}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        pending = [source]
        visited: Set[int] = set()
        while pending:
            current = pending.pop(0)
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            for key, value in current.items():
                if str(key).casefold() in wanted:
                    return value
            for key, value in current.items():
                if (
                    str(key).casefold()
                    not in _NON_INHERITED_METADATA_SUBTREES
                    and isinstance(value, Mapping)
                ):
                    pending.append(value)
    return None


def _artifact_metadata_sources(
    layer: Optional[LayerSpec],
    explicit_metadata: Optional[Mapping[str, object]],
    scenario: Optional[ScenarioConfig],
) -> Tuple[Mapping[str, object], ...]:
    context = _COMPILATION_CONTEXT.get()
    active_scenario = scenario
    if active_scenario is None and context is not None:
        active_scenario = context.scenario
    sources: List[Mapping[str, object]] = []
    if explicit_metadata is not None:
        sources.append(explicit_metadata)
    if active_scenario is not None:
        sources.append(active_scenario.workload.metadata)
        sources.append(active_scenario.model.metadata)
    if layer is not None:
        # GGUF audit bindings contain nested ``block_size``/``type`` fields
        # for every physical tensor.  They are consumed by the projection
        # materializer, not by the single-format artifact dispatcher below;
        # exposing them here makes a mixed Q4_K_M file look like an incomplete
        # one-format workload and triggers a false validation failure.
        layer_metadata = layer.metadata
        if "gguf_tensor_bindings" in layer_metadata:
            layer_metadata = dict(layer_metadata)
            layer_metadata.pop("gguf_tensor_bindings", None)
        sources.append(layer_metadata)
    return tuple(sources)


def _artifact_label_from_metadata(
    layer: Optional[LayerSpec],
    sources: Sequence[Mapping[str, object]],
) -> Optional[str]:
    """Resolve the artifact identity while keeping runtime ``wN`` separate."""

    explicit_keys = (
        "artifact_quantization",
        "artifact_format",
        "gguf_quantization",
        "weight_format",
    )
    for source in sources:
        for key in explicit_keys:
            value = _metadata_value((source,), (key,))
            if value is None:
                continue
            label = _canonical_artifact_quantization(value)
            if label is None:
                if key == "weight_format" and not _looks_like_artifact_quantization(
                    value
                ):
                    continue
                raise ValueError(
                    "unsupported artifact quantization {}".format(value)
                )
            return label
        value = source.get("quantization") if isinstance(source, Mapping) else None
        if _looks_like_artifact_quantization(value):
            label = _canonical_artifact_quantization(value)
            if label is None:
                raise ValueError("unsupported artifact quantization {}".format(value))
            return label

    if layer is not None and _looks_like_artifact_quantization(layer.quantization):
        label = _canonical_artifact_quantization(layer.quantization)
        if label is None:
            raise ValueError(
                "unsupported artifact quantization {}".format(
                    layer.quantization
                )
            )
        return label
    return None


def _artifact_operator_kind(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).casefold())
    if any(
        marker in normalized
        for marker in ("ffn", "mlp", "expert", "shared", "moe", "router")
    ):
        return "ffn"
    if any(
        marker in normalized
        for marker in (
            "attention",
            "attn",
            "linear",
            "lm_head",
            "mtp",
            "prediction",
            "aux_head",
            "qkv",
            "query",
            "key",
            "value",
        )
    ):
        return "attention"
    return "unknown"


def _artifact_spec_for_operator(
    label: Optional[str],
    name: str,
) -> Optional[_ArtifactQuantizationSpec]:
    if label is None:
        return None
    if label == _MIXED_ARTIFACT_QUANTIZATION:
        kind = _artifact_operator_kind(name)
        if kind == "ffn":
            label = "IQ4_XS"
        elif kind == "attention":
            label = "IQ3_S"
        else:
            raise ValueError(
                "mixed artifact quantization requires an attention or FFN "
                "operator name: {}".format(name)
            )
    spec = _ARTIFACT_QUANTIZATION_REGISTRY.get(label)
    if spec is None:  # pragma: no cover - guarded by canonical parser
        raise ValueError("unsupported artifact quantization {}".format(label))
    return spec


def _parse_runtime_quantization(value: object) -> Optional[Tuple[int, int]]:
    """Parse a simulator ``wNaM`` contract from text or a nested mapping."""

    if isinstance(value, Mapping):
        for key in (
            "simulator_weight_cost_contract",
            "runtime_cost_contract",
            "weight_cost_contract",
            "compute_quantization",
            "quantization",
            "contract",
            "value",
        ):
            if key in value:
                parsed = _parse_runtime_quantization(value[key])
                if parsed is not None:
                    return parsed
        return None
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[-_\s]+", "", value.strip().casefold())
    match = re.fullmatch(r"w(\d+)(?:a(\d+))?", normalized)
    if match is None:
        return None
    weight_bits = int(match.group(1))
    activation_bits = (
        int(match.group(2)) if match.group(2) is not None else None
    )
    if weight_bits <= 0 or (
        activation_bits is not None and activation_bits <= 0
    ):
        raise ValueError("runtime quantization must use positive bit widths")
    return (
        activation_bits if activation_bits is not None else weight_bits,
        weight_bits,
    )


def _layer_runtime_precision_bits(
    layer: LayerSpec,
    sources: Sequence[Mapping[str, object]],
    artifact_label: Optional[str],
) -> Tuple[int, int]:
    for source in sources:
        for key in (
            "runtime_cost_contract",
            "simulator_weight_cost_contract",
            "weight_cost_contract",
            "runtime_quantization",
            "compute_quantization",
        ):
            if key in source:
                parsed = _parse_runtime_quantization(source[key])
                if parsed is not None:
                    return parsed
        value = source.get("quantization")
        parsed = _parse_runtime_quantization(value)
        if parsed is not None:
            return parsed
    parsed_layer = _parse_runtime_quantization(layer.quantization)
    if parsed_layer is not None:
        return parsed_layer
    if artifact_label is not None:
        activation_bits = _dtype_bits(layer.dtype)
        spec = _ARTIFACT_QUANTIZATION_REGISTRY.get(
            "IQ3_S"
            if artifact_label == _MIXED_ARTIFACT_QUANTIZATION
            else artifact_label
        )
        return activation_bits, spec.compute_weight_bits if spec else 4
    return layer_precision_bits(
        layer.dtype,
        layer.quantization,
        unsupported_dtype_message=(
            "unsupported layer dtype {}; add a model adapter or explicit quantization"
            .format(layer.dtype)
        ),
    )


def _metadata_int(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    *,
    non_negative: bool = True,
) -> Optional[int]:
    value = _metadata_value(sources, keys)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(keys[0]))
    if (non_negative and value < 0) or (not non_negative and value <= 0):
        bound = "non-negative" if non_negative else "positive"
        raise ValueError("{} must be a {} integer".format(keys[0], bound))
    return value


def _direct_metadata_int(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    *,
    non_negative: bool = True,
) -> Optional[int]:
    wanted = {str(key).casefold() for key in keys}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if str(key).casefold() not in wanted:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("{} must be an integer".format(keys[0]))
            if (non_negative and value < 0) or (
                not non_negative and value <= 0
            ):
                bound = "non-negative" if non_negative else "positive"
                raise ValueError(
                    "{} must be a {} integer".format(keys[0], bound)
                )
            return value
    return None


_ARTIFACT_WORKLOAD_PROJECTED_KEYS = frozenset(
    key.casefold()
    for key in (
        "artifact_quantization",
        "artifact_format",
        "gguf_quantization",
        "weight_format",
        "block_size",
        "artifact_block_size",
        "quant_block_size",
        "payload_bytes",
        "packed_bytes",
        "block_payload_bytes",
        "payload_bytes_per_block",
        "weight_payload_bytes_per_block",
        "artifact_payload_bytes_per_block",
        "metadata_bytes",
        "block_metadata_bytes",
        "metadata_bytes_per_block",
        "weight_metadata_bytes_per_block",
        "artifact_metadata_bytes_per_block",
        "physical_weight_metadata_bytes_per_block",
        "block_bytes",
        "bytes_per_block",
        "weight_storage_bytes",
        "physical_weight_storage_bytes",
        "physical_packed_bytes",
        "packed_weight_bytes",
        "artifact_packed_bytes",
        "physical_weight_payload_bytes",
        "physical_weight_bytes",
        "total_weight_bytes",
        "artifact_weight_bytes",
        "weight_metadata_bytes",
        "physical_weight_metadata_bytes",
        "total_weight_metadata_bytes",
        "block_count",
        "artifact_block_count",
        "dequant_operations",
        "dequant_ops",
        "weight_dequant_operations",
        "dequant_operations_per_weight",
        "dequant_ops_per_weight",
        "dequant_operations_per_block",
        "dequant_ops_per_block",
        "dequant_m_tile",
        "dequant_tile_m",
        "dequant_transcendental_operations",
        "dequant_transcendental_ops",
        "dequant_output_elements",
        "dequant_dispatch_ns",
        "quantization_dispatch_ns",
        "artifact_dispatch_ns",
    )
)
_ARTIFACT_WORKLOAD_SAFE_VALUE_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
)


def _artifact_operation_metadata_projection(
    metadata: Optional[Mapping[str, object]],
) -> Optional[Tuple[Tuple[str, object], ...]]:
    """Project a fresh flat operation mapping without changing lookup rules."""

    if metadata is None:
        return ()
    if type(metadata) is not dict:
        return None
    if any(
        type(key) is not str or isinstance(value, Mapping)
        for key, value in metadata.items()
    ):
        return None

    projected: List[Tuple[str, object]] = []
    for key, value in metadata.items():
        folded = key.casefold()
        token: Optional[str] = None
        if folded in _ARTIFACT_WORKLOAD_PROJECTED_KEYS:
            token = folded
        elif key == "quantization" and _looks_like_artifact_quantization(value):
            # Artifact label lookup uses exact ``source.get("quantization")``;
            # runtime wNaM values deliberately remain outside this cache key.
            token = "=quantization"
        if token is None:
            continue
        if type(value) not in _ARTIFACT_WORKLOAD_SAFE_VALUE_TYPES:
            return None
        projected.append((token, value))
    return tuple(projected)


def _artifact_workload_cache_key(
    context: CompilationContext,
    layer: Optional[LayerSpec],
    m: int,
    k: int,
    n: int,
    name: str,
    sources: Sequence[Mapping[str, object]],
    explicit_metadata: Optional[Mapping[str, object]],
    *,
    dynamic_rhs: bool,
) -> Optional[Hashable]:
    """Key artifact derivation without first repeating that derivation.

    The exact metadata projection and context-owned source identities determine
    the artifact label.  Only mixed artifact contracts inspect the operator
    name, and they do so exclusively through ``_artifact_operator_kind``.
    Keeping that normalized kind in the key lets a cache hit skip label/spec
    discovery while preserving every input that can affect the result.
    """

    projection = _artifact_operation_metadata_projection(explicit_metadata)
    if projection is None:
        return None

    stable_sources = sources[1:] if explicit_metadata is not None else sources
    source_tokens = tuple(
        context.metadata_source_identity(source) for source in stable_sources
    )
    layer_quantization = (
        layer.quantization
        if layer is not None
        and _looks_like_artifact_quantization(layer.quantization)
        else None
    )
    if type(layer_quantization) not in _ARTIFACT_WORKLOAD_SAFE_VALUE_TYPES:
        return None
    return (
        "artifact_workload_metadata",
        m,
        k,
        n,
        bool(dynamic_rhs),
        _artifact_operator_kind(name),
        projection,
        source_tokens,
        layer_quantization,
    )


def _artifact_workload_metadata(
    layer: Optional[LayerSpec],
    m: int,
    k: int,
    n: int,
    name: str,
    *,
    explicit_metadata: Optional[Mapping[str, object]] = None,
    scenario: Optional[ScenarioConfig] = None,
    dynamic_rhs: bool = False,
) -> Tuple[Optional[_ArtifactQuantizationSpec], Dict[str, object]]:
    sources = _artifact_metadata_sources(layer, explicit_metadata, scenario)

    def derive() -> Tuple[
        Optional[_ArtifactQuantizationSpec], Dict[str, object]
    ]:
        label = _artifact_label_from_metadata(layer, sources)
        spec = _artifact_spec_for_operator(label, name)
        return _derive_artifact_workload_metadata(
            m,
            k,
            n,
            sources,
            spec,
            dynamic_rhs=dynamic_rhs,
        )

    context = _COMPILATION_CONTEXT.get()
    if context is None or (
        scenario is not None and context.scenario is not scenario
    ):
        return derive()

    key = _artifact_workload_cache_key(
        context,
        layer,
        m,
        k,
        n,
        name,
        sources,
        explicit_metadata,
        dynamic_rhs=dynamic_rhs,
    )
    cached_spec, derived = context.artifact_workload_metadata(
        key,
        derive,
    )
    return cached_spec, derived  # type: ignore[return-value]


def _derive_artifact_workload_metadata(
    m: int,
    k: int,
    n: int,
    sources: Sequence[Mapping[str, object]],
    spec: Optional[_ArtifactQuantizationSpec],
    *,
    dynamic_rhs: bool,
) -> Tuple[Optional[_ArtifactQuantizationSpec], Dict[str, object]]:
    if dynamic_rhs:
        # QK/PV RHS operands are runtime K/V activations.  They must never
        # inherit the model's on-disk artifact format or dequant work.
        return None, {
            "dynamic_rhs": True,
            "artifact_quantization_applied": False,
        }

    block_size = _metadata_int(
        sources,
        ("block_size", "artifact_block_size", "quant_block_size"),
        non_negative=False,
    )
    payload_per_block = _metadata_int(
        sources,
        (
            "payload_bytes",
            "packed_bytes",
            "block_payload_bytes",
            "payload_bytes_per_block",
            "weight_payload_bytes_per_block",
            "artifact_payload_bytes_per_block",
        ),
    )
    metadata_per_block = _metadata_int(
        sources,
        (
            "metadata_bytes",
            "block_metadata_bytes",
            "metadata_bytes_per_block",
            "weight_metadata_bytes_per_block",
            "artifact_metadata_bytes_per_block",
            "physical_weight_metadata_bytes_per_block",
        ),
    )
    block_bytes = _metadata_int(
        sources,
        ("block_bytes", "bytes_per_block"),
    )
    storage_override = _metadata_int(
        sources,
        (
            "weight_storage_bytes",
            "physical_weight_storage_bytes",
            "physical_packed_bytes",
            "packed_weight_bytes",
            "artifact_packed_bytes",
            "physical_weight_payload_bytes",
        ),
    )
    total_weight_override = _metadata_int(
        sources,
        (
            "physical_weight_bytes",
            "total_weight_bytes",
            "artifact_weight_bytes",
        ),
    )
    metadata_override = _metadata_int(
        sources,
        (
            "weight_metadata_bytes",
            "physical_weight_metadata_bytes",
            "total_weight_metadata_bytes",
        ),
    )
    block_count_override = _direct_metadata_int(
        sources,
        ("block_count", "artifact_block_count"),
    )

    if spec is None:
        if any(
            value is not None
            for value in (
                block_size,
                payload_per_block,
                metadata_per_block,
                block_bytes,
                block_count_override,
            )
        ) and storage_override is None and total_weight_override is None:
            raise ValueError(
                "physical block metadata requires an artifact quantization"
            )
        packed_bytes = storage_override
        metadata_bytes = metadata_override or 0
        if total_weight_override is not None:
            if packed_bytes is None:
                packed_bytes = total_weight_override - metadata_bytes
                if packed_bytes < 0:
                    raise ValueError(
                        "physical_weight_bytes is smaller than metadata bytes"
                    )
            elif total_weight_override != packed_bytes + metadata_bytes:
                raise ValueError(
                    "physical_weight_bytes must equal storage plus metadata bytes"
                )
        result = {
            "artifact_quantization_applied": False,
            "physical_weight_storage_bytes": packed_bytes,
            "physical_weight_metadata_bytes": metadata_bytes,
        }
        dispatch_ns = _metadata_value(
            sources,
            (
                "dequant_dispatch_ns",
                "quantization_dispatch_ns",
                "artifact_dispatch_ns",
            ),
        )
        if dispatch_ns is not None:
            if isinstance(dispatch_ns, bool) or not isinstance(
                dispatch_ns, (int, float)
            ) or float(dispatch_ns) < 0.0:
                raise ValueError("quantization dispatch must be non-negative")
            result["dequant_dispatch_ns"] = float(dispatch_ns)
        return None, result

    effective_block_size = block_size or spec.block_size
    effective_metadata_per_block = (
        metadata_per_block
        if metadata_per_block is not None
        else spec.metadata_bytes
    )
    effective_payload_per_block = (
        payload_per_block
        if payload_per_block is not None
        else spec.payload_bytes
    )
    if block_bytes is not None and payload_per_block is None:
        effective_payload_per_block = block_bytes - effective_metadata_per_block
        if effective_payload_per_block < 0:
            raise ValueError("block_bytes must include payload and metadata")
    block_count = block_count_override or (
        n * int(math.ceil(k / float(effective_block_size)))
    )
    packed_bytes = storage_override
    if packed_bytes is None:
        packed_bytes = block_count * effective_payload_per_block
    block_metadata_bytes = metadata_override
    if block_metadata_bytes is None:
        block_metadata_bytes = block_count * effective_metadata_per_block
    if total_weight_override is not None:
        if storage_override is None:
            packed_bytes = total_weight_override - block_metadata_bytes
            if packed_bytes < 0:
                raise ValueError(
                    "physical_weight_bytes is smaller than metadata bytes"
                )
        elif total_weight_override != packed_bytes + block_metadata_bytes:
            raise ValueError(
                "physical_weight_bytes must equal storage plus metadata bytes"
            )

    dequant_operations = _metadata_int(
        sources,
        (
            "dequant_operations",
            "dequant_ops",
            "weight_dequant_operations",
        ),
    )
    dequant_per_weight = _metadata_int(
        sources,
        (
            "dequant_operations_per_weight",
            "dequant_ops_per_weight",
        ),
    )
    dequant_per_block = _metadata_int(
        sources,
        (
            "dequant_operations_per_block",
            "dequant_ops_per_block",
        ),
    )
    dequant_m_tile = _metadata_int(
        sources,
        ("dequant_m_tile", "dequant_tile_m"),
        non_negative=False,
    )
    # Artifact weights are unpacked by the quantized dot/GEMM kernel while a
    # packed KxN block is reused across an M tile.  In the absence of a
    # measured/kernel-declared tile, the realized workload itself is the only
    # defensible tile fact: one M tile for this invocation.  This deliberately
    # avoids multiplying format work by every output row.
    effective_dequant_m_tile = dequant_m_tile or m
    dequant_m_tile_count = int(
        math.ceil(m / float(effective_dequant_m_tile))
    )
    explicit_dequant_operations = dequant_operations is not None
    if dequant_operations is None:
        if dequant_per_block is not None:
            dequant_operations = (
                dequant_m_tile_count * block_count * dequant_per_block
            )
        else:
            dequant_operations = (
                dequant_m_tile_count
                * block_count
                * effective_block_size
                * (
                    dequant_per_weight
                    if dequant_per_weight is not None
                    else spec.dequant_operations_per_weight
                )
            )
    dequant_transcendental_operations = _metadata_int(
        sources,
        (
            "dequant_transcendental_operations",
            "dequant_transcendental_ops",
        ),
    ) or 0
    dequant_output_elements = _metadata_int(
        sources,
        ("dequant_output_elements",),
    ) or 0
    result = {
        "artifact_quantization_applied": True,
        "artifact_quantization": spec.name,
        "physical_weight_storage_bytes": packed_bytes,
        "physical_weight_metadata_bytes": block_metadata_bytes,
        "artifact_block_size": effective_block_size,
        "artifact_block_count": block_count,
        "artifact_payload_bytes_per_block": effective_payload_per_block,
        "artifact_metadata_bytes_per_block": effective_metadata_per_block,
        "artifact_packed_bytes": packed_bytes,
        "artifact_metadata_bytes": block_metadata_bytes,
        "dequant_execution_model": "fused_quantized_dot",
        "dequant_m_tile": effective_dequant_m_tile,
        "dequant_m_tile_count": dequant_m_tile_count,
        "fused_dequant_operations": dequant_operations,
        "dequant_operations": dequant_operations,
        "dequant_transcendental_operations": dequant_transcendental_operations,
        "dequant_output_elements": dequant_output_elements,
        "dequant_operations_basis": (
            "explicit_total_fused_work"
            if explicit_dequant_operations
            else (
                "block_layout_per_m_tile_explicit_per_block"
                if dequant_per_block is not None
                else (
                    "block_layout_per_m_tile_explicit_per_weight"
                    if dequant_per_weight is not None
                    else "block_layout_per_m_tile_format_default"
                )
            )
        ),
        "dequant_operations_evidence": (
            "explicit_quantization_metadata"
            if explicit_dequant_operations
            or dequant_per_block is not None
            or dequant_per_weight is not None
            or dequant_m_tile is not None
            else "artifact_block_layout_and_realized_m_tile"
        ),
    }
    dispatch_ns = _metadata_value(
        sources,
        (
            "dequant_dispatch_ns",
            "quantization_dispatch_ns",
            "artifact_dispatch_ns",
        ),
    )
    if dispatch_ns is not None:
        if isinstance(dispatch_ns, bool) or not isinstance(
            dispatch_ns, (int, float)
        ) or float(dispatch_ns) < 0.0:
            raise ValueError("quantization dispatch must be non-negative")
        result["dequant_dispatch_ns"] = float(dispatch_ns)
    return spec, result


def _append_gemm_epilogue(
    workload: GemmWorkload,
    *,
    operations: int = 0,
    transcendental_operations: int = 0,
    output_elements: int = 0,
    name: str,
) -> GemmWorkload:
    """Compose a genuine activation epilogue with an existing GEMM epilogue."""

    names = [item for item in (workload.epilogue_name, name) if item]
    return replace(
        workload,
        epilogue_operations=workload.epilogue_operations + operations,
        epilogue_transcendental_operations=(
            workload.epilogue_transcendental_operations
            + transcendental_operations
        ),
        epilogue_output_elements=max(
            workload.epilogue_output_elements, output_elements
        ),
        epilogue_name="+".join(names),
    )


def _layer_for_gemm_operation(
    scenario: ScenarioConfig,
    name: str,
    metadata: Mapping[str, object],
) -> Optional[LayerSpec]:
    layer_id = metadata.get("layer_id")
    layers = _execution_layers(scenario)
    if layer_id is not None:
        for layer in layers:
            if layer.layer_id == str(layer_id):
                return layer
    for layer in layers:
        if layer.layer_id and layer.layer_id in str(name):
            return layer
    return layers[-1] if layers else None


def _workload_quantization_metadata(
    scenario: ScenarioConfig,
    workload: GemmWorkload,
    name: str,
    operation_metadata: Mapping[str, object],
    *,
    dynamic_rhs: bool = False,
) -> Dict[str, object]:
    """Expose physical/dequant facts alongside each lowered GEMM estimate."""

    layer = _layer_for_gemm_operation(scenario, name, operation_metadata)
    projection_id = operation_metadata.get("projection_id")
    if projection_id is not None:
        if not isinstance(projection_id, str) or not projection_id.strip():
            raise ValueError("projection_id must be non-empty text")
        tp_degree = operation_metadata.get("projection_tp_degree", 1)
        tp_rank = operation_metadata.get("projection_tp_rank", 0)
        allow_padding = operation_metadata.get(
            "projection_allow_padding", True
        )
        if not isinstance(allow_padding, bool):
            raise ValueError("projection_allow_padding must be boolean")
        materialized = (
            _materialize_weight_projection(
                layer,
                projection_id,
                tp_degree=tp_degree,  # type: ignore[arg-type]
                tp_rank=tp_rank,  # type: ignore[arg-type]
                allow_padding=allow_padding,
            )
            if layer is not None
            else None
        )
        if materialized is not None:
            result = {
                "weight_compute_bits": workload.weight_bits,
                "weight_storage_bytes": workload.weight_storage_bytes,
                "weight_metadata_bytes": workload.weight_metadata_bytes,
                "weight_bytes": workload.weight_bytes,
                "packed_weight_formats": workload.packed_weight_formats,
                "packed_weight_transform_operations": (
                    workload.packed_weight_transform_operations
                ),
                "gemm_m": workload.m,
                "gemm_k": workload.k,
                "gemm_n": workload.n,
                **materialized.audit_metadata(),
            }
            # The immutable workload remains authoritative after any typed
            # output-byte override or caller-side replacement.
            result.update(
                {
                    "physical_weight_storage_bytes": (
                        workload.weight_storage_bytes
                    ),
                    "physical_weight_metadata_bytes": (
                        workload.weight_metadata_bytes
                    ),
                    "artifact_packed_bytes": workload.weight_storage_bytes,
                    "artifact_metadata_bytes": workload.weight_metadata_bytes,
                }
            )
            return result
    _spec, derived = _artifact_workload_metadata(
        layer,
        workload.m,
        workload.k,
        workload.n,
        name,
        explicit_metadata=operation_metadata,
        scenario=scenario,
        dynamic_rhs=dynamic_rhs,
    )
    result: Dict[str, object] = {
        "weight_compute_bits": workload.weight_bits,
        "weight_storage_bytes": workload.weight_storage_bytes,
        "weight_metadata_bytes": workload.weight_metadata_bytes,
        "weight_bytes": workload.weight_bytes,
        "packed_weight_formats": workload.packed_weight_formats,
        "packed_weight_transform_operations": (
            workload.packed_weight_transform_operations
        ),
    }
    if _spec is not None:
        result.update(derived)
        # Reconcile the recomputed contract with the immutable workload.  The
        # latter may have been sharded or had typed logical bytes appended
        # after _layer_gemm() returned.
        result.update(
            {
                "physical_weight_storage_bytes": workload.weight_storage_bytes,
                "physical_weight_metadata_bytes": workload.weight_metadata_bytes,
                "artifact_packed_bytes": workload.weight_storage_bytes,
                "artifact_metadata_bytes": workload.weight_metadata_bytes,
            }
        )
    elif workload.epilogue_name.lower().startswith("dequant_"):
        dequant_name = workload.epilogue_name.split("+", 1)[0]
        inferred_label = None
        for candidate in _ARTIFACT_QUANTIZATION_REGISTRY.values():
            if dequant_name.casefold() == (
                "dequant_{}".format(candidate.name).casefold()
            ):
                inferred_label = candidate.name
                break
        result.update(
            {
                "artifact_quantization_applied": True,
                "artifact_quantization": inferred_label,
                "dequant_epilogue_name": workload.epilogue_name,
                "dequant_operations": workload.epilogue_operations,
                "dequant_transcendental_operations": (
                    workload.epilogue_transcendental_operations
                ),
                "dequant_output_elements": workload.epilogue_output_elements,
            }
        )
    else:
        result.update(derived)
    return result


def _layer_gemm(
    layer: LayerSpec,
    m: int,
    k: int,
    n: int,
    *,
    name: str,
    dynamic_rhs: bool = False,
    metadata: Optional[Mapping[str, object]] = None,
    workload_metadata: Optional[Mapping[str, object]] = None,
    scenario: Optional[ScenarioConfig] = None,
    projection_id: Optional[str] = None,
    projection_tp_degree: int = 1,
    projection_tp_rank: int = 0,
    projection_allow_padding: bool = True,
    f32_storage: bool = False,
) -> GemmWorkload:
    if metadata is not None and workload_metadata is not None:
        raise ValueError("provide only one GEMM workload metadata mapping")
    effective_metadata = metadata or workload_metadata
    sources = _artifact_metadata_sources(layer, effective_metadata, scenario)
    artifact_label = _artifact_label_from_metadata(layer, sources)
    activation_bits, weight_bits = _layer_runtime_precision_bits(
        layer,
        sources,
        artifact_label,
    )
    if not isinstance(f32_storage, bool):
        raise ValueError("f32_storage must be boolean")
    declared_f32_storage = f32_storage
    output_storage_bits = 32 if declared_f32_storage else max(16, activation_bits)
    materialized_projection = None
    if projection_id is not None:
        if not isinstance(projection_id, str) or not projection_id.strip():
            raise ValueError("projection_id must be non-empty text")
        materialized_projection = _materialize_weight_projection(
            layer,
            projection_id,
            tp_degree=projection_tp_degree,
            tp_rank=projection_tp_rank,
            allow_padding=projection_allow_padding,
        )
    if materialized_projection is not None:
        packed_weight_formats = tuple(
            dict.fromkeys(
                segment.segment.artifact_spec.name
                for segment in materialized_projection.segments
            )
        )
        return GemmWorkload(
            m=m,
            k=materialized_projection.k,
            n=materialized_projection.n,
            activation_bits=activation_bits,
            weight_bits=materialized_projection.weight_bits,
            output_bits=output_storage_bits,
            activation_storage_bytes=(4 * m * materialized_projection.k if declared_f32_storage else None),
            accumulator_bits=32,
            packed_weight_formats=packed_weight_formats,
            packed_weight_transform_operations=(
                materialized_projection.fused_dequant_operations
            ),
            packed_weight_format_segments=tuple(
                (
                    segment.segment.artifact_spec.name,
                    segment.local_n,
                    segment.local_block_count
                    * segment.segment.artifact_spec.block_size
                    * segment.segment.artifact_spec.dequant_operations_per_weight,
                )
                for segment in materialized_projection.segments
            ),
            weight_metadata_bytes=(
                materialized_projection.weight_metadata_bytes
            ),
            name=name,
            weight_storage_bytes=(
                materialized_projection.weight_storage_bytes
            ),
        )
    artifact_spec, artifact_metadata = _artifact_workload_metadata(
        layer,
        m,
        k,
        n,
        name,
        explicit_metadata=effective_metadata,
        scenario=scenario,
        dynamic_rhs=dynamic_rhs,
    )
    if dynamic_rhs:
        weight_bits = activation_bits
    weight_storage_bytes = None
    weight_metadata_bytes = 0
    if artifact_spec is not None:
        weight_storage_bytes = int(
            artifact_metadata["physical_weight_storage_bytes"]
        )
        weight_metadata_bytes = int(
            artifact_metadata["physical_weight_metadata_bytes"]
        )
    else:
        # Explicit physical overrides are useful for a typed logical tensor
        # whose exact byte count is known but whose format is not modeled.
        if artifact_metadata.get("physical_weight_storage_bytes") is not None:
            weight_storage_bytes = int(
                artifact_metadata["physical_weight_storage_bytes"]
            )
        weight_metadata_bytes = int(
            artifact_metadata.get("physical_weight_metadata_bytes", 0)
        )
    if declared_f32_storage and dynamic_rhs and weight_storage_bytes is None:
        weight_storage_bytes = 4 * k * n
    return GemmWorkload(
        m=m,
        k=k,
        n=n,
        activation_bits=activation_bits,
        weight_bits=weight_bits,
        output_bits=output_storage_bits,
        activation_storage_bytes=4 * m * k if declared_f32_storage else None,
        accumulator_bits=32,
        weight_metadata_bytes=weight_metadata_bytes,
        name=name,
        weight_storage_bytes=weight_storage_bytes,
    )


def _layer_precision_bits(
    layer: LayerSpec,
    metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[int, int]:
    sources = _artifact_metadata_sources(layer, metadata, None)
    artifact_label = _artifact_label_from_metadata(layer, sources)
    return _layer_runtime_precision_bits(layer, sources, artifact_label)


def _dtype_bits(dtype_name: str) -> int:
    return dtype_bits(
        dtype_name,
        unsupported_message=(
            "unsupported layer dtype {}; add a model adapter or explicit quantization"
            .format(dtype_name)
        ),
    )


def _gpu_component(scenario: ScenarioConfig) -> ComponentSpec:
    for component in scenario.hardware.components:
        if _kind(component) == "gpu":
            return component
    raise ValueError("硬件中没有 GPU 组件")


def _kind(component: ComponentSpec) -> str:
    return _normalized_component_kind(str(component.kind))


@lru_cache(maxsize=64)
def _normalized_component_kind(kind: str) -> str:
    """Normalize one immutable kind string with a small exact cache."""

    return normalize_component_kind(kind)


def _is_writable_storage(component: ComponentSpec) -> bool:
    if _kind(component) not in STORAGE_COMPONENT_KINDS:
        return False
    return component.is_writable


def _is_active_resident_weight_storage(component: ComponentSpec) -> bool:
    return component.is_writable and (
        _is_cim(component) or _kind(component) in ACTIVE_MEMORY_COMPONENT_KINDS
    )


def _is_cim(component: ComponentSpec) -> bool:
    return "cim" in _kind(component)


__all__ = [
    "summarize_cpu_iq_panel_reuse",
    "CompilationContext",
    "ScenarioValidationReport",
    "TopologyAwareBatchCostProvider",
    "compile_serving_cohort_schedule",
    "compile_scenario",
    "estimate_serving_cohort_cost",
    "materialize_requests",
    "summarize_analytical_coverage",
    "validate_scenario",
]


