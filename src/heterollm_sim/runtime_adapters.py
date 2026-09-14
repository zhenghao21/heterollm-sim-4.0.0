"""Runtime-specific lowering into the shared simulator task contract.

Adapters annotate and validate an already lowered task stream.  They do not
implement scheduling; :func:`heterollm_sim.engine.simulate_schedule` remains
the single event-simulation path for every runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .contracts import RunManifest, TaskSpec
from .engine import ScheduleIR
from .serde import stable_hash


@dataclass(frozen=True)
class LlamaCppRuntimeConfig:
    """Typed llama.cpp execution knobs used by both simulation and native runs.

    The field names intentionally mirror llama.cpp's command-line/API names.
    Keeping this object immutable makes it safe to include in manifests and
    canonical fingerprints.  ``None`` for KV types means llama.cpp's default
    (model-declared) type.
    """

    threads: int = 16
    threads_batch: int = 16
    batch: int = 512
    ubatch: int = 512
    context: int = 4096
    parallel: int = 1
    gpu_layers: int = -1
    flash_attn: bool = False
    kv_type_k: str | None = None
    kv_type_v: str | None = None
    kv_unified: bool = True
    cont_batching: bool = True
    warmup: bool = True
    seed: int = 0
    mmap: bool = True
    mlock: bool = False
    offload_kqv: bool = True
    op_offload: bool = True
    split_mode: str = "layer"
    main_gpu: int = 0
    tensor_split: str | None = None
    device: str | None = None
    cpu_range: str | None = None
    cpu_range_batch: str | None = None
    numa: str | None = None

    def __post_init__(self) -> None:
        for name in ("threads", "threads_batch"):
            value = getattr(self, name)
            # llama.cpp uses -1 to select the runtime's automatic thread count.
            if isinstance(value, bool) or not isinstance(value, int) or value < -1 or value == 0:
                raise ValueError(f"llama.cpp {name} must be -1 or a positive integer")
        for name in ("batch", "ubatch", "context", "parallel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"llama.cpp {name} must be a positive integer")
        if isinstance(self.gpu_layers, bool) or not isinstance(self.gpu_layers, int) or self.gpu_layers < -1:
            raise ValueError("llama.cpp gpu_layers must be >= -1 (-1 means all layers)")
        for name in ("flash_attn", "kv_unified", "cont_batching", "warmup"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"llama.cpp {name} must be boolean")
        for name in ("mmap", "mlock", "offload_kqv", "op_offload"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"llama.cpp {name} must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("llama.cpp seed must be an integer")
        for name in ("kv_type_k", "kv_type_v"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"llama.cpp {name} must be non-empty text or None")
        if self.split_mode not in {"none", "layer", "row"}:
            raise ValueError("llama.cpp split_mode must be none, layer, or row")
        if isinstance(self.main_gpu, bool) or not isinstance(self.main_gpu, int) or self.main_gpu < 0:
            raise ValueError("llama.cpp main_gpu must be a non-negative integer")
        for name in ("tensor_split", "device", "cpu_range", "cpu_range_batch"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"llama.cpp {name} must be non-empty text or None")
        if self.numa is not None and self.numa not in {"distribute", "isolate", "numactl"}:
            raise ValueError("llama.cpp numa must be distribute, isolate, numactl, or None")
        if self.ubatch > self.batch:
            raise ValueError("llama.cpp ubatch must not exceed batch")
        if self.parallel > self.context:
            raise ValueError("llama.cpp parallel must not exceed context")

    def to_dict(self) -> dict[str, Any]:
        return {
            "threads": self.threads,
            "threads_batch": self.threads_batch,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "context": self.context,
            "parallel": self.parallel,
            "gpu_layers": self.gpu_layers,
            "flash_attn": self.flash_attn,
            "kv_type_k": self.kv_type_k,
            "kv_type_v": self.kv_type_v,
            "kv_unified": self.kv_unified,
            "cont_batching": self.cont_batching,
            "warmup": self.warmup,
            "seed": self.seed,
            "mmap": self.mmap,
            "mlock": self.mlock,
            "offload_kqv": self.offload_kqv,
            "op_offload": self.op_offload,
            "split_mode": self.split_mode,
            "main_gpu": self.main_gpu,
            "tensor_split": self.tensor_split,
            "device": self.device,
            "cpu_range": self.cpu_range,
            "cpu_range_batch": self.cpu_range_batch,
            "numa": self.numa,
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash({"schema": "llama.cpp-runtime-v1", **self.to_dict()})

    # Read-only aliases retain the terminology used by the original adapter.
    @property
    def batch_size(self) -> int:
        return self.batch

    @property
    def ubatch_size(self) -> int:
        return self.ubatch

    @property
    def context_length(self) -> int:
        return self.context

    @property
    def cache_type_k(self) -> str | None:
        return self.kv_type_k

    @property
    def cache_type_v(self) -> str | None:
        return self.kv_type_v


@dataclass(frozen=True)
class RuntimeExecutionPlan:
    runtime: str
    tasks: tuple[TaskSpec, ...]
    semantics: Mapping[str, Any]
    resource_capacities: Mapping[str, int] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Canonical digest of runtime semantics included in every schedule."""

        return stable_hash({"runtime": self.runtime, "semantics": dict(self.semantics)})

    def to_schedule(self, manifest: RunManifest, *, resource_capacities: Mapping[str, int] | None = None) -> ScheduleIR:
        """Use the existing engine contract without a runtime-specific kernel."""

        metadata = dict(manifest.metadata)
        metadata.update({"runtime": self.runtime,
                         "runtime_semantics": dict(self.semantics),
                         "runtime_fingerprint": self.fingerprint})
        capacities = self.resource_capacities if resource_capacities is None else resource_capacities
        return ScheduleIR(replace(manifest, metadata=metadata), self.tasks, capacities or {})


def _plan(runtime: str, tasks: Sequence[TaskSpec], semantics: Mapping[str, Any]) -> RuntimeExecutionPlan:
    def stamp(task: TaskSpec) -> TaskSpec:
        existing_runtime = task.metadata.get("runtime")
        if existing_runtime is not None and existing_runtime != runtime:
            raise ValueError("task runtime {} cannot be lowered by {} adapter".format(existing_runtime, runtime))
        return replace(task, metadata={**task.metadata, "runtime": runtime,
                                       "runtime_semantics": dict(semantics)})
    return RuntimeExecutionPlan(runtime, tuple(stamp(task) for task in tasks), semantics)


class LlamaCppAdapter:
    runtime = "llama.cpp"

    def lower(self, tasks: Sequence[TaskSpec], *, config: LlamaCppRuntimeConfig | None = None,
              batch_size: int | None = None, ubatch_size: int | None = None, parallel: int | None = None,
              gpu_layers: int | None = None, context_length: int | None = None,
              threads: int | None = None, threads_batch: int | None = None,
              flash_attn: bool | None = None, cont_batching: bool | None = None,
              warmup: bool | None = None, seed: int | None = None,
              cache_type_k: str | None = None, cache_type_v: str | None = None,
              kv_unified: bool | None = None,
              op_offload: bool | None = None) -> RuntimeExecutionPlan:
        if config is not None and any(value is not None for value in (
            batch_size, ubatch_size, parallel, gpu_layers, context_length,
            threads, threads_batch, flash_attn, cont_batching, warmup, seed,
            cache_type_k, cache_type_v, kv_unified, op_offload)):
            raise ValueError("provide either config or individual llama.cpp options, not both")
        if config is None:
            config = LlamaCppRuntimeConfig(
                threads=16 if threads is None else threads,
                threads_batch=16 if threads_batch is None else threads_batch,
                batch=512 if batch_size is None else batch_size,
                ubatch=512 if ubatch_size is None else ubatch_size,
                context=4096 if context_length is None else context_length,
                parallel=1 if parallel is None else parallel,
                gpu_layers=-1 if gpu_layers is None else gpu_layers,
                flash_attn=False if flash_attn is None else flash_attn,
                kv_type_k=cache_type_k,
                kv_type_v=cache_type_v,
                kv_unified=True if kv_unified is None else kv_unified,
                cont_batching=True if cont_batching is None else cont_batching,
                warmup=True if warmup is None else warmup,
                seed=0 if seed is None else seed,
                op_offload=True if op_offload is None else bool(op_offload),
            )
        effective_batch = min(config.batch, config.context)
        effective_ubatch = min(config.ubatch, effective_batch)
        semantics = {
            "lowering_status": "typed_runtime_config",
            "config_fingerprint": config.fingerprint,
            "threads": config.threads,
            "threads_batch": config.threads_batch,
            "requested_logical_batch_size": config.batch,
            "requested_physical_ubatch_size": config.ubatch,
            "logical_batch_size": effective_batch,
            "physical_ubatch_size": effective_ubatch,
            "parallel_slots": config.parallel,
            "gpu_layer_count": config.gpu_layers,
            "gpu_layer_count_mode": "all" if config.gpu_layers < 0 else "explicit",
            "gpu_layer_count_is_fraction": False,
            "context_length": config.context,
            "flash_attn": config.flash_attn,
            "cache_type_k": config.kv_type_k,
            "cache_type_v": config.kv_type_v,
            "kv_unified": config.kv_unified,
            "cont_batching": config.cont_batching,
            "warmup": config.warmup,
            "seed": config.seed,
            "mmap": config.mmap,
            "mlock": config.mlock,
            "offload_kqv": config.offload_kqv,
            "op_offload": config.op_offload,
            "split_mode": config.split_mode,
            "main_gpu": config.main_gpu,
            "tensor_split": config.tensor_split,
            "device": config.device,
            "cpu_range": config.cpu_range,
            "cpu_range_batch": config.cpu_range_batch,
            "numa": config.numa,
        }
        return _plan(self.runtime, tasks, semantics)


class VLLMAdapter:
    runtime = "vLLM"

    def lower(self, tasks: Sequence[TaskSpec], *, max_num_batched_tokens: int, max_num_seqs: int,
              kv_cache_dtype: str = "auto", gpu_memory_utilization: float = 0.9,
              enable_chunked_prefill: bool = True, block_size: int | None = None,
              num_gpu_blocks: int | None = None) -> RuntimeExecutionPlan:
        if max_num_batched_tokens < 1 or max_num_seqs < 1:
            raise ValueError("vLLM batch-token and sequence limits must be positive")
        if not 0.0 < float(gpu_memory_utilization) <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if block_size is not None and block_size < 1:
            raise ValueError("vLLM block_size must be positive")
        if num_gpu_blocks is not None and num_gpu_blocks < 1:
            raise ValueError("vLLM num_gpu_blocks must be positive")
        semantics = {
            "lowering_status": "metadata_only",
            "max_num_batched_tokens": int(max_num_batched_tokens),
            "max_num_seqs": int(max_num_seqs),
            "kv_cache_dtype": str(kv_cache_dtype),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "paged_kv_cache": True,
            "enable_chunked_prefill": bool(enable_chunked_prefill),
            "block_size": block_size,
            "num_gpu_blocks": num_gpu_blocks,
            "preemption_semantics": "unspecified_without_native_scheduler_trace",
        }
        return _plan(self.runtime, tasks, semantics)


__all__ = ["LlamaCppAdapter", "LlamaCppRuntimeConfig", "RuntimeExecutionPlan", "VLLMAdapter"]
