"""Runtime-specific lowering into the shared simulator task contract.

Adapters annotate and validate an already lowered task stream.  They do not
implement scheduling; :func:`heterollm_sim.engine.simulate_schedule` remains
the single event-simulation path for every runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import re
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


LLAMA_CUDA_OP_OFFLOAD_SCHEMA = "llama.cpp.cuda.host-weight-op-offload/v1"


def _source_function(text: str, signature: str) -> str:
    """Read one known source function; unknown layouts are not extrapolated."""
    begin = text.index(signature)
    brace = text.index("{", begin)
    depth = 1
    end = brace + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("unterminated llama.cpp source function: " + signature)
    return text[begin:end]



LLAMA_HYBRID_BATCH_SCHEMA = "llama.cpp.hybrid.equal-length-batching/v1"




def _cpp_source_function(text: str, signature: str) -> str:
    """Bound a C++ body without counting braces in comments or literals.

    The slot loop contains explanatory pseudo-code in comments. A raw brace
    counter can stop before its real pending-prompt branch, so the slot proof
    uses this lexical reader rather than guessing a later function boundary.
    """
    start = text.index(signature)
    i = text.index("{", start)
    depth = 0
    while i < len(text):
        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated C++ block comment")
            i = end + 2
            continue
        if text.startswith('R"', i):
            opening = text.find("(", i + 2, i + 19)
            if opening >= 0:
                delimiter = text[i + 2:opening]
                closing = ")" + delimiter + '\"'
                end = text.find(closing, opening + 1)
                if end < 0:
                    raise ValueError("unterminated C++ raw string")
                i = end + len(closing)
                continue
        if text[i] in ('\"', "'"):
            quote = text[i]
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise ValueError("unterminated C++ function: " + signature)


LLAMA_SLOT_ORDER_SCHEMA = "llama.cpp.fresh-cohort.slot-order/v1"


def derive_llama_slot_order_contract(
    server_source: str | Path,
    *,
    source_chain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive stable slot traversal from saved source, without timing inputs.

    Stable admission can represent stable native slot order only for an
    isolated fresh cohort: explicit same-arrival requests, at most one request
    per slot, equal priority/deadline and no preemption or slot reuse. Runtime
    qualification is performed again by apply_llama_runtime_config. This is
    not a general FIFO claim for a server that recycles slots.
    """
    path = Path(server_source).resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    iterate = _cpp_source_function(text, "void iterate(std::vector<server_slot> & slots,")
    update = _cpp_source_function(text, "void update_slots()")
    compatible = _cpp_source_function(text, "bool can_batch_with(server_slot & other_slot) const")
    if ("for (auto & slot : slots)" not in iterate or "callback(slot)" not in iterate
            or any(fragment in iterate for fragment in ("std::sort", "std::rotate", "std::shuffle", "rbegin()"))):
        raise ValueError("unrecognized native fixed slot iteration")
    if not all(fragment in compatible for fragment in (
            "task->type == other_slot.task->type", "inp_embd.size() == other_slot.inp_embd.size()",
            "are_lora_equal(lora, other_slot.lora)")):
        raise ValueError("unrecognized native slot task compatibility")
    fill = update
    fill_symbol = "server_context::update_slots"
    if "if (params_base.cont_batching || batch.size() == 0)" not in fill:
        if "pre_decode();" not in update:
            raise ValueError("unrecognized update_slots to pre_decode call chain")
        fill = _cpp_source_function(text, "void pre_decode()")
        fill_symbol = "server_context::pre_decode"
    pending = fill[fill.index("if (params_base.cont_batching || batch.size() == 0)"):]
    ordered = (
        "iterate(slots,", "if (!add_ok || batch.size() >= n_batch)",
        "if (slot.state == SLOT_STATE_STARTED)", "slot.stats.update_prompt_start()",
        "slot.state = SLOT_STATE_PROCESSING_PROMPT",
        "while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch)",
        "batch.add(slot.id",
    )
    previous = -1
    for fragment in ordered:
        position = pending.find(fragment, previous + 1)
        if position < 0:
            raise ValueError("unrecognized native pending prompt fill/start order: " + fragment)
        previous = position
    guard = pending[pending.index("if (!add_ok || batch.size() >= n_batch)"):pending.index("if (slot.state == SLOT_STATE_STARTED)")]
    if "return;" not in guard:
        raise ValueError("native full-batch guard must skip remaining slots")
    digest = hashlib.sha256(raw).hexdigest()
    chain_binding = None
    if source_chain is not None:
        hashes = source_chain.get("source_sha256", {})
        if not isinstance(hashes, Mapping) or hashes.get(str(path)) != digest:
            raise ValueError("slot-order source differs from supplied source chain")
        chain_binding = {"schema": source_chain.get("schema"), "server_sha256": digest,
                         "runtime_log_ref": source_chain.get("runtime_log_ref"),
                         "binding": "same_server_source_as_existing_recurrent_contract"}
    return {"schema": LLAMA_SLOT_ORDER_SCHEMA, "status": "source_derived",
        "phase_candidate_order": "stable_admission", "native_slot_iteration": "vector_order",
        "prompt_fill_rule": "current_compatible_slot_until_batch_budget_or_existing_prompt_stop",
        "engine_start_boundary": "STARTED_to_PROCESSING_PROMPT_before_prompt_preparation",
        "preserves_engine_start_definition": True,
        "requirements": {"explicit_fresh_cohort": True, "same_arrival": True,
                         "request_count_at_most_slots": True, "no_slot_reuse": True,
                         "equal_priority_deadline": True, "no_preemption": True,
                         "no_priority_aging": True, "compatible_causal_text": True},
        "source_sha256": {str(path): digest}, "source_symbols": ["server_context::iterate(vector<server_slot>)",
            "server_context::update_slots", fill_symbol, "server_slot::can_batch_with", "slot.stats.update_prompt_start"],
        "source_chain_binding": chain_binding,
        "scope": "fresh isolated explicit same-arrival causal-text cohort; no dynamic arrivals, recycled slots, resume, priority aging or preemption",
        "accuracy_validated": False, "native_latency_used": False}


def derive_llama_hybrid_batch_contract(
    source_root: str | Path,
    *,
    server_source: str | Path,
    runtime_log: str | Path,
) -> dict[str, Any]:
    """Bind hybrid batching geometry to locked source and saved runtime facts.

    Only source control flow and static log fields are consumed: this never
    reads native token timing or fits latency. The caller is responsible for
    retaining the build/source relationship to the native runtime. The proof
    is restricted to unified KV, causal text, independent sequence owners and
    no recurrent rollback/MTP. It does not claim calibrated kernel costs.
    """
    root = Path(source_root)
    paths = {"server": Path(server_source), "hybrid_memory": root / "src/llama-memory-hybrid.cpp",
             "batch_allocator": root / "src/llama-batch.cpp", "qwen35_graph": root / "src/models/qwen35.cpp"}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    texts = {name: value.decode("utf-8") for name, value in raw.items()}
    compatibility = _source_function(texts["server"], "bool can_batch_with(server_slot & other_slot) const")
    splitting = _source_function(texts["server"], "bool can_split() const")
    required = {
        "server": ("if (params_base.cont_batching || batch.size() == 0)",
                   "while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch)"),
        "hybrid_memory": ("const bool unified = (mem_attn->get_n_stream() == 1)",
                          "balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0)",
                          "mem_recr->prepare(ubatches)", "mem_attn->prepare(ubatches)"),
        "batch_allocator": ("llama_ubatch llama_batch_allocr::split_equal(",
                            "cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size()",
                            "(idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch",
                            "idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end())"),
        "qwen35_graph": ("GGML_ASSERT(ubatch.equal_seqs())",
                         "GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs)",
                         "head_v_dim, head_v_dim, num_v_heads, n_seqs"),
    }
    if ("task->type == other_slot.task->type" not in compatibility
            or "are_lora_equal(lora, other_slot.lora)" not in compatibility
            or "!task->need_embd()" not in splitting):
        raise ValueError("unrecognized native slot batching compatibility")
    for name, fragments in required.items():
        if not all(fragment in texts[name] for fragment in fragments):
            raise ValueError("unrecognized hybrid equal-length source: " + name)
    log_path = Path(runtime_log)
    log_bytes = log_path.read_bytes()
    log = log_bytes.decode("utf-8", errors="replace")
    facts = {}
    for field in ("n_seq_max", "n_ubatch", "n_rs_seq"):
        values = {int(value) for value in re.findall(r"\b" + field + r"\s*=\s*(\d+)", log)}
        if len(values) != 1:
            raise ValueError("one unambiguous captured hybrid runtime value required: " + field)
        facts[field] = values.pop()
    unified = set(re.findall(r"\bkv_unified\s*=\s*['\"]?(true|false)", log))
    if unified != {"true"} or facts["n_rs_seq"] != 0 or min(facts["n_seq_max"], facts["n_ubatch"]) <= 0:
        raise ValueError("hybrid proof requires unified KV and zero recurrent rollback snapshots")
    return {"schema": LLAMA_HYBRID_BATCH_SCHEMA, "status": "source_derived",
        "architectures": ["qwen3_5_hybrid_transformer"],
        "physical_lowering": "equal_length_stateful_ubatches", "kv_unified": True,
        "recurrent_rollback_snapshots": 0, "captured_sequence_capacity": facts["n_seq_max"],
        "captured_ubatch_capacity": facts["n_ubatch"],
        "source_sha256": {str(paths[name].resolve()): hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        "runtime_log_ref": {"path": str(log_path.resolve()), "sha256": hashlib.sha256(log_bytes).hexdigest()},
        "source_symbols": ["server_slot::can_batch_with", "server_slot::can_split", "server_context::update_slots",
                           "llama_memory_hybrid::init_batch", "llama_batch_allocr::split_equal", "Qwen35 equal_seqs graph"],
        "source_rule": "server emits decode rows then compatible prompt rows; unified hybrid memory splits each microbatch into equal tokens per independent sequence",
        "capabilities": {"supports_batched_stateful_execution": True, "supports_equal_length_stateful_ubatches": True},
        "scope": "source/runtime-supported causal Qwen hybrid text with unified KV, no MTP/rollback/experts/adapters",
        "accuracy_validated": False, "native_latency_used": False}


def derive_llama_cuda_op_offload_contract(
    source_root: str | Path,
    *,
    runtime_environment: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Extract the supported host MUL_MAT rule from local llama.cpp source.

    This describes execution semantics, not measured latency.  The caller
    must establish that this source belongs to its CUDA backend.  A missing
    environment entry means a *conditional default assumption*, never proof
    that a historical run used the default.  Explicit ``None`` records an
    observed absence of GGML_OP_OFFLOAD_MIN_BATCH.
    """
    root = Path(source_root)
    paths = {
        "scheduler": root / "ggml/src/ggml-backend.cpp",
        "cuda": root / "ggml/src/ggml-cuda/ggml-cuda.cu",
        "operators": root / "ggml/src/ggml.c",
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    source = {name: value.decode("utf-8") for name, value in raw.items()}
    scheduler = source["scheduler"]
    required_scheduler = (
        "sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)",
        "ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)",
    )
    if not all(fragment in scheduler for fragment in required_scheduler):
        raise ValueError("unrecognized llama.cpp host-weight offload selection")
    copies = _source_function(scheduler, "static enum ggml_status ggml_backend_sched_compute_splits(")
    if ("ggml_backend_tensor_copy(input, input_cpy)" not in copies
            or "cpy_tensor_async(input_backend, split_backend, input, input_cpy)" not in copies):
        raise ValueError("unrecognized per-invocation host-weight staging")
    cuda = source["cuda"]
    batches = _source_function(cuda, "static int64_t get_op_batch_size(")
    if not re.search(r"case GGML_OP_MUL_MAT:\s*return op->ne\[1\];", batches):
        raise ValueError("unrecognized CUDA MUL_MAT physical batch dimension")
    offload = _source_function(cuda, "static bool ggml_backend_cuda_device_offload_op(")
    if "get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size" not in offload:
        raise ValueError("unrecognized CUDA offload threshold comparison")
    default = re.search(
        r'const int min_batch_size = getenv\("GGML_OP_OFFLOAD_MIN_BATCH"\) \? '
        r'atoi\(getenv\("GGML_OP_OFFLOAD_MIN_BATCH"\)\) : (\d+);', cuda)
    if default is None:
        raise ValueError("unrecognized CUDA offload default/environment rule")
    supports = _source_function(cuda, "static bool ggml_backend_cuda_device_supports_op(")
    matmul = supports.split("case GGML_OP_MUL_MAT:", 1)[1].split("case GGML_OP_OUT_PROD:", 1)[0]
    formats_block = matmul.split("switch (a->type)", 1)[1].split("return true;", 1)[0]
    formats = tuple(dict.fromkeys(re.findall(r"case GGML_TYPE_([A-Z0-9_]+):", formats_block)))
    if not formats or "a->nb[0] != ggml_element_size(a)" not in matmul:
        raise ValueError("unrecognized CUDA matmul format/layout support")
    operators = source["operators"]
    mul_mat = _source_function(operators, "struct ggml_tensor * ggml_mul_mat(")
    get_rows = _source_function(operators, "struct ggml_tensor * ggml_get_rows(")
    if "ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)" not in mul_mat or "enum ggml_type type = GGML_TYPE_F32;" not in get_rows:
        raise ValueError("unrecognized ordinary GGUF hidden storage semantics")
    env_name = "GGML_OP_OFFLOAD_MIN_BATCH"
    captured = runtime_environment is not None and env_name in runtime_environment
    env_value = runtime_environment.get(env_name) if captured else None
    minimum = int(default.group(1))
    env_status = "captured_absent" if captured else "uncaptured_default_assumption"
    if captured and env_value is not None:
        if not isinstance(env_value, str) or not re.fullmatch(r"[+-]?\d+", env_value.strip()):
            raise ValueError("unsupported GGML_OP_OFFLOAD_MIN_BATCH value; do not guess atoi semantics")
        minimum = max(1, int(env_value.strip()))
        env_status = "captured_override"
    return {
        "schema": LLAMA_CUDA_OP_OFFLOAD_SCHEMA,
        "backend": "CUDA",
        "source_sha256": {str(paths[name].resolve()): hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        "source_symbols": ["ggml_backend_sched_backend_id_from_cur", "get_op_batch_size",
                           "ggml_backend_cuda_device_offload_op", "ggml_backend_cuda_device_supports_op",
                           "ggml_backend_sched_compute_splits", "ggml_mul_mat", "ggml_get_rows"],
        "minimum_m": minimum,
        "default_minimum_m": int(default.group(1)),
        "physical_batch_dimension": "MUL_MAT.output.ne[1]",
        "supported_weight_formats": formats,
        "environment": {"name": env_name, "value": env_value, "status": env_status},
        "prediction_provenance": "source_mechanism" if captured else "conditional_development_assumption",
        "accuracy_validated": False,
        "scope": "ordinary contiguous 2D dense text GGUF host-weight MUL_MAT with F32 hidden storage; CUDA backend",
        "staging": "per_invocation_copy_then_temporary_read_clean_discard",
        "unsupported": ["MUL_MAT_ID/expert dispatch", "arbitrary tensor views/layouts", "GPU performance calibration"],
    }


def apply_llama_cuda_op_offload(
    scenario: Any,
    config: LlamaCppRuntimeConfig,
    *,
    source_contract: Mapping[str, Any] | None,
    cuda_backend_available: bool,
) -> Any:
    """Bind source-qualified GEMM execution to existing temporary staging.

    Static weight/KV placement is deliberately preserved.  No empirical
    timing or model name participates in this capability decision.
    """
    from .cost_models import HostGemmOffloadCapability
    from .ir import model_graph_execution_view

    if not isinstance(cuda_backend_available, bool):
        raise ValueError("CUDA backend availability must be explicit boolean")
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    gpu_profiles = profiles.get("gpu", {})
    contract = dict(source_contract or {})
    status = "disabled"
    reason = "runtime_op_offload_disabled"
    capability = None
    if config.op_offload:
        if not cuda_backend_available or not any(c.normalized_kind in {"gpu", "cuda"} for c in scenario.hardware.components):
            reason = "cuda_backend_unavailable"
        elif contract.get("schema") != LLAMA_CUDA_OP_OFFLOAD_SCHEMA or contract.get("backend") != "CUDA":
            reason = "source_contract_missing_or_unsupported"
        elif not isinstance(contract.get("source_sha256"), Mapping) or not contract["source_sha256"] or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
            for value in contract["source_sha256"].values()
        ):
            reason = "source_identity_missing"
        elif not isinstance(contract.get("supported_weight_formats"), (list, tuple)) or not contract["supported_weight_formats"]:
            reason = "source_format_support_missing"
        else:
            view = model_graph_execution_view(scenario.model.graph, schema_version=scenario.model.schema_version)
            if scenario.workload.mtp is not None or any(
                x.layer.kind != "dense" or x.layer.shared_expert_intermediate_size for x in view.layer_instances
            ):
                reason = "model_graph_outside_source_scope"
            else:
                capability = HostGemmOffloadCapability(
                    minimum_m=contract["minimum_m"],
                    evidence="llama.cpp CUDA source-derived host-weight MUL_MAT offload; " + str(contract["prediction_provenance"]),
                    supported_weight_formats=tuple(contract["supported_weight_formats"]),
                    provenance=contract,
                )
                status, reason = "enabled", "source_qualified_physical_m_dispatch"
    for name, profile in gpu_profiles.items():
        gpu_profiles[name] = replace(profile, host_gemm_offload=capability)
    audit = {**contract, "status": status, "reason": reason, "op_offload": config.op_offload,
             "cuda_backend_available": cuda_backend_available, "accuracy_validated": False}
    workload_metadata = {**scenario.workload.metadata, "llama_cpp_cuda_op_offload": audit}
    # Host-offloaded GGUF tensors are copied as F32, while compute precision
    # remains selected by the existing quantized GEMM path.  Reuse the typed
    # storage-byte lowering; do not charge a second activation conversion.
    if capability is not None and any(
        key.rsplit(".", 1)[-1] in {"attention", "linear_attention", "mlp", "lm_head"}
        and scenario.hardware.get_component(target).normalized_kind == "cpu"
        for key, target in scenario.placement.op_to_component.items()
    ):
        workload_metadata["llama_cpp_f32_hidden_storage"] = True
    return replace(scenario, component_profiles=profiles, llama_cpp_config=config,
                   workload=replace(scenario.workload, metadata=workload_metadata))


@dataclass(frozen=True)
class RuntimeExecutionPlan:
    runtime: str
    tasks: tuple[TaskSpec, ...]
    semantics: Mapping[str, Any]
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    resource_owners: Mapping[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Canonical digest of runtime semantics included in every schedule."""

        return stable_hash({"runtime": self.runtime, "semantics": dict(self.semantics)})

    def to_schedule(self, manifest: RunManifest, *, resource_capacities: Mapping[str, int] | None = None, resource_owners: Mapping[str, str] | None = None) -> ScheduleIR:
        """Use the existing engine contract without a runtime-specific kernel."""

        metadata = dict(manifest.metadata)
        metadata.update({"runtime": self.runtime,
                         "runtime_semantics": dict(self.semantics),
                         "runtime_fingerprint": self.fingerprint})
        capacities = self.resource_capacities if resource_capacities is None else resource_capacities
        owners = self.resource_owners if resource_owners is None else resource_owners
        return ScheduleIR(replace(manifest, metadata=metadata), self.tasks, capacities or {}, owners or {})


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


__all__ = ["LlamaCppAdapter", "LlamaCppRuntimeConfig", "RuntimeExecutionPlan", "VLLMAdapter",
           "derive_llama_cuda_op_offload_contract", "apply_llama_cuda_op_offload"]
