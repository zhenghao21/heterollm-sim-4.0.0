"""Read-only capacity guards for Decode forecasting on HBM.

Nothing here advances a device calendar or a transaction counter. The page
totals describe monotonic logical K/V appends, not a claim that the
unexecuted physical tail has been simulated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, TYPE_CHECKING

from .data_motion import resolve_service
from .planner import (_execution_layers, _kv_components, _kv_tensor_bytes, _parallel_plan,
                      kv_block_geometry, kv_pool_buffer_id)

if TYPE_CHECKING:
    from .serving import _OnlineRuntime

PAGE_BYTES = 4096


@dataclass(frozen=True)
class FastTailProjection:
    remaining_decode_steps: int
    current_cached_tokens: int
    final_cached_tokens: int
    logical_kv_growth_bytes: int
    required_capacity_pages: int
    capacity_pages: int
    kv_buffers: tuple[Mapping[str, Any], ...]
    safety_reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.safety_reasons

    def to_metadata(self) -> dict[str, Any]:
        return {"schema": "heterollm.fast-tail-projection/v1", **asdict(self),
                "safe": self.safe,
                "semantics": "logical_append_projection_and_observed_prefix_headroom_guard",
                "physical_state_advanced": False,
                "physical_tail_transactions_recorded": False}


def _ceil_pages(byte_count: int) -> int:
    return (byte_count + PAGE_BYTES - 1) // PAGE_BYTES


def project_fast_tail(
    runtime: _OnlineRuntime, *, detailed_decode_steps: int | None = None,
) -> FastTailProjection:
    """Describe the unexecuted single-request tail without mutating runtime.

    The caller still owns model-family eligibility and calibration accuracy.
    A clean guard says only that the declared monotonic-append assumptions
    have headroom; it never certifies the requested timing error.
    """
    if detailed_decode_steps is not None and (
        isinstance(detailed_decode_steps, bool)
        or not isinstance(detailed_decode_steps, int)
        or detailed_decode_steps < 0
    ):
        raise ValueError("detailed_decode_steps must be a non-negative integer")
    policy = runtime.plan.kv_policy
    reasons: list[str] = []
    states = runtime._state_values
    if len(states) != 1:
        return FastTailProjection(0, 0, 0, 0, 0, policy.capacity_pages,
                                  (), ("fast_tail_requires_single_request",))
    state = states[0]
    current = state.cached_tokens
    final = state.spec.prompt_tokens + max(0, state.spec.output_tokens - 1)
    remaining = max(0, state.spec.output_tokens - state.committed)
    if state.prefill_cursor < state.spec.prompt_tokens or state.first_token_ns is None:
        reasons.append("exact_prefill_and_first_token_required")
    if state.status.value not in {"running", "finished"}:
        reasons.append("request_not_running_or_finished")
    if policy.layout_mode != "legacy_single":
        reasons.append("fast_tail_requires_contiguous_generic_kv")
    if (policy.offload_component and policy.offload_ratio) or policy.prefetch_distance:
        reasons.append("kv_offload_or_prefetch_requires_detailed_execution")
    if runtime.plan.mtp.enabled:
        reasons.append("mtp_requires_detailed_execution")
    required_pages = runtime.ledger.pages_for_tokens(final)
    if required_pages > policy.capacity_pages:
        reasons.append("final_kv_exceeds_logical_capacity")
    scenario = runtime.plan.scenario
    parallel = _parallel_plan(scenario)
    if parallel.world_size != 1:
        reasons.append("fast_tail_requires_single_rank")
    buffers: list[Mapping[str, Any]] = []
    # The projection appends rows at contiguous pool offsets: the request's
    # blocks must be the pool's first ones and its growth the next free ones.
    layout = runtime._kv_block_layout
    if layout is None:
        reasons.append("fast_tail_requires_physical_block_kv")
    else:
        blocks = runtime.ledger.blocks
        needed = -(-final // layout["tokens_per_block"])
        for kind in layout["pool_blocks"]:
            table = runtime.ledger.block_table(state.spec.request_id).get(kind, ())
            want = blocks.blocks_for_pages(kind, needed)
            head = blocks.head[0, kind]
            growth = tuple(blocks.free[0, kind][head:head + want - len(table)])
            if table + growth != tuple(range(want)):
                reasons.append("fast_tail_requires_contiguous_kv_blocks")
    for layer in _execution_layers(scenario):
        if layer.is_linear_attention:
            reasons.append("linear_attention_requires_detailed_execution")
            continue
        ranks = tuple(parallel.ranks_for_layer(layer))
        if len(ranks) != 1:
            continue
        rank = ranks[0]
        row_bytes = _kv_tensor_bytes(scenario, layer, parallel.tp_degree, 1)
        component_id, offload_id, ratio = _kv_components(scenario, rank, rank.component_id, layer)
        if not component_id or (offload_id and ratio):
            reasons.append("unsupported_kv_owner_or_offload")
            continue
        component = scenario.hardware.get_component(component_id)
        owner = resolve_service(component).physical_owner
        if component.normalized_kind == "hbf":
            reasons.append("fast_tail_requires_hbm_kv:" + owner)
        if layout is not None:
            _row, stride, _per_page = kv_block_geometry(scenario, parallel, rank, layer,
                                                        layout["tokens_per_block"])
            if stride != layout["tokens_per_block"] * row_bytes:
                reasons.append("fast_tail_requires_unpadded_kv_blocks")
        for part in ("k", "v"):
            buffer_id = kv_pool_buffer_id(layer.layer_id, rank, part)
            start, stop = current * row_bytes, final * row_bytes
            row = {
                "buffer_id": buffer_id, "layer_id": layer.layer_id, "part": part,
                "component_id": component_id, "physical_owner": owner,
                "bytes_per_token": row_bytes,
                "current_bytes": start, "final_bytes": stop,
                "growth_bytes": max(0, stop - start),
                "current_tail_bytes": start % PAGE_BYTES,
                "final_tail_bytes": stop % PAGE_BYTES,
                "new_logical_pages": max(0, _ceil_pages(stop) - _ceil_pages(start)),
                "completed_page_appends": max(0, stop // PAGE_BYTES - start // PAGE_BYTES),
                "tail_page_slots_upper_bound": int(row_bytes % PAGE_BYTES != 0 and remaining > 0),
            }
            buffers.append(row)
    decode_batches = tuple(batch for batch in runtime.batches if batch.kind == "decode")
    if detailed_decode_steps is not None and detailed_decode_steps != len(decode_batches):
        reasons.append("detailed_decode_step_count_does_not_match_runtime")
    return FastTailProjection(remaining, current, final,
        max(0, final - current) * int(policy.logical_bytes_per_token),
        required_pages, policy.capacity_pages, tuple(buffers),
        tuple(dict.fromkeys(reasons)))
