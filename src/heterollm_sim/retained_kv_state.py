"""Opt-in singleton-owner retained KV state; counts are bounds, never addresses.

The caller supplies a qualified final-warmup snapshot and explicit request-slot
assignments. This module validates that declaration, not native evidence files.
Two warmup rounds are provenance, not two additive allocations.

Workload metadata must explicitly set ``llama_cpp_retained_kv_state_enabled``
to True. ``llama_cpp_retained_kv_state`` supplies the v1 declaration: identity,
configuration and source hashes bound to the independent runtime identity and
nonflash-view source contract; evidence digest; final-warmup boundary and count
semantics; fully specified ordinary-cache scope; all retained slots with P/O;
and an exact request_slots mapping. The test fixture documents concrete keys.
Missing/None/unknown values reject the enabled path. No proof is inferred from
parallel, model names, timing data, or merely having two warmup rounds.
"""
from collections.abc import Mapping
from typing import NoReturn
import re

KEY = "llama_cpp_retained_kv_state"
ENABLED = KEY + "_enabled"
IDENTITY = "llama_cpp_retained_kv_identity"
BOUND = "llama_cpp_retained_kv_bound"
SCHEMA = "heterollm.retained-kv-state/v1"


def _require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError("retained KV: " + reason)


def _int(value: object, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def _identity(value):
    _require(isinstance(value, Mapping), "missing identity")
    for key in ("process_id", "process_block", "runtime_build_id"):
        _require(isinstance(value.get(key), str) and bool(value[key].strip()), "missing identity." + key)
    return dict(value)


class RetainedKVState:
    """A slot contains either its old retained rows or current materialized rows.

    Admission only reserves an owner. begin_prompt models seq_rm at actual
    prompt preparation, commit models successful KV insertion, and finish
    releases ownership while retaining the final P+O-1 rows. Unsupported state
    operations invalidate the ledger so it cannot emit a stale positive bound.
    """

    @classmethod
    def from_plan(cls, plan, layers):
        scenario = plan.scenario
        metadata = scenario.workload.metadata
        if metadata.get(ENABLED) is not True:
            return None
        raw = metadata.get(KEY)
        _require(isinstance(raw, Mapping) and raw.get("schema") == SCHEMA, "missing or invalid contract")
        identity = _identity(raw.get("identity"))
        _require(identity == _identity(metadata.get(IDENTITY)), "runtime/process identity mismatch")
        _require(raw.get("boundary") == "after_final_qualified_warmup", "unknown initialization boundary")
        _require(raw.get("token_count_semantics") == "native_prompt_including_bos_and_predicted_output", "unknown prompt/BOS/output count semantics")
        _require(raw.get("warmup_batches") == 2 and type(raw.get("warmup_batches")) is int, "two completed warmups required")
        _require(raw.get("complete_distinct_slots") is True, "distinct-slot completion unproven")
        _require(isinstance(raw.get("evidence_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", raw["evidence_sha256"]), "missing evidence digest")
        scope = raw.get("scope")
        required = {"singleton_owner": True, "ordinary_full_attention": True, "completion_count": 1,
            "cache_prompt": False, "cache_ram_mib": 0, "cache_idle_slots": False,
            "shared_prefix": False, "swa": False, "recurrent": False,
            "restore": False, "speculative": False, "external_state_operations": False,
            "context_shift": False, "purge": False, "cancellation": False, "recompute": False}
        _require(isinstance(scope, Mapping) and all(type(scope.get(k)) is type(v) and scope.get(k) == v for k,v in required.items()), "unsupported or unknown cache scope")
        config = scenario.llama_cpp_config
        _require(config is not None and not config.flash_attn and not scenario.fusion_policy.flash_attention and config.kv_unified, "requires non-Flash unified KV")
        _require((config.kv_type_k, config.kv_type_v) == ("f16", "f16"), "only f16 cache supported")
        _require(config.ubatch < 1024, "mask-trimming query groups unsupported")
        _require(layers and not any(layer.is_linear_attention for layer in layers), "hybrid/recurrent state unsupported")
        _require(scenario.model.architecture in {"llama", "qwen2", "llama_decoder", "qwen2_decoder"}, "ordinary architecture unproven")
        _require(not plan.mtp.enabled and scenario.workload.mtp is None and plan.kv_policy.offload_ratio == 0.0, "speculation/offload unsupported")
        binding = {"batch": config.batch, "ubatch": config.ubatch, "parallel": config.parallel,
            "simulator_slot_context_tokens": config.context, "native_context_tokens": config.context * config.parallel,
            "flash_attn": False, "kv_unified": True, "kv_type_k": "f16", "kv_type_v": "f16"}
        actual_binding = raw.get("configuration")
        _require(isinstance(actual_binding, Mapping) and actual_binding == binding
            and all(type(actual_binding[k]) is type(v) for k,v in binding.items()), "configuration mismatch")
        source = metadata.get("llama_cpp_nonflash_kv_view")
        _require(isinstance(source, Mapping) and source.get("schema") == "heterollm.llama-nonflash-kv-view/v1"
            and source.get("runtime_binding_status") == "verified" and source.get("configuration") == binding
            and source.get("n_pad") == 1 and source.get("n_kv_padding") == 256, "nonflash source/configuration binding missing")
        hashes = raw.get("source_sha256")
        _require(isinstance(hashes, Mapping) and bool(hashes) and hashes == source.get("source_sha256")
            and all(isinstance(v,str) and re.fullmatch(r"[0-9a-f]{64}",v) for v in hashes.values()), "source identity mismatch")
        _require(plan.scheduler.max_num_seqs == config.parallel and plan.scheduler.max_num_batched_tokens == config.batch
            and plan.scheduler.max_num_ubatch_tokens == config.ubatch, "scheduler configuration mismatch")
        _require(binding["native_context_tokens"] % 256 == 0, "unaligned native capacity")
        slots = raw.get("slots")
        _require(isinstance(slots, list) and len(slots) == config.parallel, "explicit complete slot set required")
        rows = {}
        for slot in slots:
            _require(isinstance(slot, Mapping) and slot.get("state") == "retained", "unknown initial slot state")
            slot_id = _int(slot.get("slot_id"), "slot_id")
            _require(slot_id not in rows, "duplicate slot identity")
            _require(slot.get("identity") == identity, "slot identity mismatch")
            p = _int(slot.get("prompt_tokens"), "prompt_tokens", 1)
            o = _int(slot.get("output_tokens"), "output_tokens", 1)
            _require(p + o <= config.context, "warmup slot context overflow")
            rows[slot_id] = p + o - 1
        request_slots = raw.get("request_slots")
        shapes = {r.request_id: (r.prompt_tokens, r.output_tokens) for r in plan.requests}
        _require(isinstance(request_slots, Mapping) and set(request_slots) == set(shapes), "explicit request-slot coverage required")
        for request, slot in request_slots.items():
            _require(type(slot) is int and slot in rows, "unknown request slot")
            p,o = shapes[request]
            _require(p > 0 and o > 0 and p + o <= config.context, "request context shift/empty prompt unsupported")
        return cls(rows, request_slots, shapes, config.context, binding["native_context_tokens"])

    def __init__(self, rows, request_slots, shapes, slot_capacity, capacity):
        self.rows = dict(rows)
        self.request_slots = dict(request_slots)
        self.shapes = dict(shapes)
        self.slot_capacity = slot_capacity
        self.capacity = capacity
        self.owners = {}
        self.prepared = set()
        self.completed = set()
        self.invalid_reason = None
        self._check(self.rows)

    def _check(self, rows):
        _require(self.invalid_reason is None, "invalidated lifecycle: " + str(self.invalid_reason))
        _require(all(type(n) is int and 0 <= n <= self.slot_capacity for n in rows.values()), "slot capacity exceeded")
        _require(sum(rows.values()) <= self.capacity, "occupied rows exceed native capacity")

    def mark_invalid(self, reason: str) -> None:
        """Poison the ledger without masking an execution/commit exception."""
        if self.invalid_reason is None:
            self.invalid_reason = str(reason)

    def invalidate(self, reason: str) -> NoReturn:
        self.mark_invalid(reason)
        raise ValueError("retained KV: unsupported lifecycle " + str(reason))

    def admit(self, request: str) -> None:
        self._check(self.rows)
        _require(request in self.request_slots and request not in self.completed, "unknown/completed request")
        slot = self.request_slots[request]
        _require(slot not in self.owners or self.owners[slot] == request, "assigned slot still active")
        self.owners[slot] = request

    def begin_prompt(self, request: str) -> None:
        self._check(self.rows)
        slot = self.request_slots[request]
        _require(self.owners.get(slot) == request, "prompt without slot admission")
        if request not in self.prepared:
            self.rows[slot] = 0
            self.prepared.add(request)

    def projected_rows(self, items):
        self._check(self.rows)
        projected = dict(self.rows)
        seen = set()
        for request, phase, context, count in items:
            _require(request not in seen, "duplicate request in physical cohort")
            seen.add(request)
            _require(request in self.request_slots, "unknown request")
            slot = self.request_slots[request]
            _require(self.owners.get(slot) == request and request in self.prepared, "rows before prompt clear")
            _require(phase in {"prefill", "decode"} and type(count) is int and count > 0, "nonordinary append")
            _require(phase != "decode" or count == 1, "nonordinary decode")
            _require(type(context) is int and projected[slot] == context, "materialized context mismatch")
            projected[slot] += count
        self._check(projected)
        return projected

    def occupied_after(self, items):
        return sum(self.projected_rows(items).values())

    def commit(self, request, phase, context, count):
        self.rows = self.projected_rows(((request, phase, context, count),))

    def finish(self, request: str) -> None:
        self._check(self.rows)
        slot = self.request_slots[request]
        p,o = self.shapes[request]
        _require(self.owners.get(slot) == request and request in self.prepared and self.rows[slot] == p + o - 1, "finish without successful P+O-1 materialization")
        del self.owners[slot]
        self.completed.add(request)
