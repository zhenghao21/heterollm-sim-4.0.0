"""Cheap, deterministic estimates for choosing a V4 retention mode.

The estimator intentionally reports logical work units instead of a wall-clock
duration.  Runtime depends on the host, Python build, and concurrent jobs, so a
seconds estimate here would imply precision that the simulator does not have.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import ScenarioConfig
from .contracts import SIMULATION_SCHEMA_VERSION
from .ir import model_graph_execution_view
from .mtp import MTPRequestCursor


# One layer/rank/cohort lowers to several compute, memory, communication, and
# synchronization phases rather than one TaskSpec.  This structural factor is
# deliberately conservative and keeps the cheap estimate independent of full
# graph compilation while avoiding a severe undercount for long static runs.
_EVENT_TASKS_PER_LAYER_RANK = 32


def _request_shapes(scenario: ScenarioConfig) -> Tuple[Tuple[int, int], ...]:
    workload = scenario.workload
    if workload.requests:
        return tuple(
            (max(0, request.prompt_tokens), max(0, request.output_tokens))
            for request in workload.requests
        )
    # Synthetic request_count may be intentionally very large.  Keep it in
    # aggregate form; estimate_scenario handles the identical-request groups
    # arithmetically instead of allocating one tuple per request.
    return ()


def _chunks(
    values: Sequence[Tuple[int, int]], size: int
) -> Iterable[Sequence[Tuple[int, int]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _mtp_configuration(
    scenario: ScenarioConfig,
) -> Any:
    """Return the typed V4 MTP knobs used by the online serving plan."""

    policy = scenario.workload.mtp
    if policy is None or not policy.enabled:
        return None
    return policy


def _estimate_mtp_rounds(
    output_tokens: int,
    *,
    mtp_policy: Any,
) -> int:
    """Estimate exact shared-cursor MTP rounds for one request."""

    remaining = max(0, int(output_tokens))
    cursor = MTPRequestCursor(mtp_policy)
    rounds = 0
    while remaining > 0:
        mtp_round = cursor.next_round(remaining)
        remaining -= mtp_round.committed_tokens
        rounds += 1
    return rounds


def _estimate_decode_rounds(
    output_tokens: int,
    *,
    mtp_policy: Any,
    prefill_emits_first_token: bool = True,
) -> int:
    # Continuous serving can sample token 1 from the final prompt logits only
    # when a real prompt decode exists.  Promptless requests are already
    # decode-ready and must schedule their first token normally.  Static keeps
    # its established one-token-prefill contract by passing the default True.
    free_tokens = 1 if prefill_emits_first_token else 0
    remaining = max(0, int(output_tokens) - free_tokens)
    if mtp_policy is None:
        return remaining
    return _estimate_mtp_rounds(
        remaining,
        mtp_policy=mtp_policy,
    )


def _estimate_cohorts(
    request_shapes: Sequence[Tuple[int, int]],
    *,
    scheduler_mode: str,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prefill_chunk_tokens: int,
    mtp_policy: Any,
) -> Tuple[int, int]:
    if not request_shapes:
        return 0, 0
    if scheduler_mode != "continuous":
        decode_rounds = sum(
            _estimate_decode_rounds(
                output_tokens,
                mtp_policy=mtp_policy,
            )
            for _, output_tokens in request_shapes
        )
        return len(request_shapes) + decode_rounds, (
            decode_rounds if mtp_policy is not None else 0
        )

    # Requests in one cohort can have different shapes.  Capacity-sized groups
    # provide a transparent batching baseline; separated arrivals and
    # preemption can increase the actual cohort count.
    cohorts = 0
    decode_rounds_total = 0
    for group in _chunks(request_shapes, max(1, max_num_seqs)):
        per_request_prefill = max(
            (
                int(math.ceil(prompt_tokens / float(max(1, prefill_chunk_tokens))))
                for prompt_tokens, _ in group
            ),
            default=0,
        )
        shared_token_prefill = int(
            math.ceil(
                sum(prompt_tokens for prompt_tokens, _ in group)
                / float(max(1, max_num_batched_tokens))
            )
        )
        prefill = max(per_request_prefill, shared_token_prefill)
        decode_rounds = max(
            (
                _estimate_decode_rounds(
                    output_tokens,
                    mtp_policy=mtp_policy,
                    prefill_emits_first_token=prompt_tokens > 0,
                )
                for prompt_tokens, output_tokens in group
            ),
            default=0,
        )
        decode_rounds_total += decode_rounds if mtp_policy is not None else 0
        cohorts += prefill + decode_rounds
    return cohorts, decode_rounds_total


def _estimate_uniform_cohorts(
    request_count: int,
    prompt_tokens: int,
    output_tokens: int,
    *,
    scheduler_mode: str,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prefill_chunk_tokens: int,
    mtp_policy: Any,
) -> Tuple[int, int]:
    """Estimate identical synthetic requests in O(1) memory and time."""

    count = max(0, request_count)
    if count == 0:
        return 0, 0
    if scheduler_mode != "continuous":
        decode_rounds = _estimate_decode_rounds(
            output_tokens,
            mtp_policy=mtp_policy,
        )
        total_decode = count * decode_rounds
        return count + total_decode, total_decode if mtp_policy is not None else 0

    group_size = max(1, max_num_seqs)
    full_groups, remainder = divmod(count, group_size)
    per_request_prefill = int(
        math.ceil(max(0, prompt_tokens) / float(max(1, prefill_chunk_tokens)))
    )

    def group_cohorts(size: int) -> int:
        if size <= 0:
            return 0
        shared_prefill = int(
            math.ceil(
                size * max(0, prompt_tokens)
                / float(max(1, max_num_batched_tokens))
            )
        )
        decode_rounds = _estimate_decode_rounds(
            output_tokens,
            mtp_policy=mtp_policy,
            prefill_emits_first_token=prompt_tokens > 0,
        )
        return max(per_request_prefill, shared_prefill) + decode_rounds

    cohorts = full_groups * group_cohorts(group_size) + group_cohorts(remainder)
    decode_rounds = _estimate_decode_rounds(
        output_tokens,
        mtp_policy=mtp_policy,
        prefill_emits_first_token=prompt_tokens > 0,
    )
    mtp_rounds = (
        full_groups * decode_rounds + (decode_rounds if remainder else 0)
        if mtp_policy is not None
        else 0
    )
    return cohorts, mtp_rounds


def _risk_level(event_tasks: int, cohorts: int) -> Tuple[str, str]:
    if event_tasks >= 10_000_000 or cohorts >= 1_000_000:
        return "critical", "极高"
    if event_tasks >= 1_000_000 or cohorts >= 100_000:
        return "high", "高"
    if event_tasks >= 100_000 or cohorts >= 10_000:
        return "medium", "中"
    return "low", "低"


def estimate_scenario(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Return a JSON-compatible logical-size estimate for ``scenario``.

    ``estimated_event_task_count`` is a unified-kernel logical scale,
    not an exact compiled task count.  It is useful for warning about task-graph
    growth without compiling the graph as part of the estimate itself.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario 必须是 ScenarioConfig")

    shapes = _request_shapes(scenario)
    if shapes:
        request_count = len(shapes)
        prompt_tokens = sum(prompt for prompt, _ in shapes)
        output_tokens = sum(output for _, output in shapes)
    else:
        request_count = max(0, scenario.workload.request_count)
        prompt_tokens = max(0, scenario.workload.prompt_tokens) * request_count
        output_tokens = max(0, scenario.workload.output_tokens) * request_count
    total_tokens = prompt_tokens + output_tokens
    layer_count = len(
        model_graph_execution_view(
            scenario.model.graph,
            schema_version=scenario.model.schema_version,
        ).layer_instances
    )
    world_size = scenario.placement.parallel.world_size
    scheduler = scenario.workload.scheduler
    scheduler_mode = scheduler.mode
    mtp_policy = _mtp_configuration(scenario)
    mtp_enabled = mtp_policy is not None
    candidate_tokens = (
        int(mtp_policy.candidate_tokens) if mtp_policy is not None else 1
    )
    if shapes:
        estimated_cohorts, estimated_mtp_rounds = _estimate_cohorts(
            shapes,
            scheduler_mode=scheduler_mode,
            max_num_seqs=scheduler.max_num_seqs,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            prefill_chunk_tokens=scheduler.prefill_chunk_tokens,
            mtp_policy=mtp_policy,
        )
    else:
        estimated_cohorts, estimated_mtp_rounds = _estimate_uniform_cohorts(
            request_count,
            scenario.workload.prompt_tokens,
            scenario.workload.output_tokens,
            scheduler_mode=scheduler_mode,
            max_num_seqs=scheduler.max_num_seqs,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            prefill_chunk_tokens=scheduler.prefill_chunk_tokens,
            mtp_policy=mtp_policy,
        )
    estimated_event_tasks = (
        estimated_cohorts
        * max(1, layer_count)
        * max(1, world_size)
        * _EVENT_TASKS_PER_LAYER_RANK
    )
    risk_level, risk_level_zh = _risk_level(
        estimated_event_tasks, estimated_cohorts
    )
    recommended_retention_policy = (
        "aggregate"
        if scheduler_mode == "continuous"
        else (
            "streaming"
            if risk_level in {"high", "critical"}
            else "exact"
        )
    )

    explanation: List[str] = [
        "规模由请求 token、模型层数和并行 world size 的逻辑乘积估算。",
        "cohort 数按调度容量和 prefill 分块近似；到达间隔与抢占可能改变实际数量。",
        "事件任务数是统一事件内核的等价逻辑规模，不是已编译任务的精确计数。",
        "每个 layer/rank/cohort 使用 {} 个典型 lowering phase 的结构系数，"
        "覆盖计算、内存、通信和同步任务。".format(
            _EVENT_TASKS_PER_LAYER_RANK
        ),
        "estimated_cohort_count 是无抢占基线；不预测动态抢占或重算后的实际 batch 数。",
    ]
    warnings: List[str] = [
        "本结果不包含墙钟秒数；实际运行时间取决于主机、并发和场景细节。"
    ]
    if scheduler_mode == "continuous":
        explanation.append(
            "连续批处理推荐 aggregate，只保留完整精确汇总。"
        )
    elif recommended_retention_policy == "streaming":
        explanation.append(
            "高风险静态工作负载推荐 streaming：保留精确汇总指标，"
            "仅保存有界的代表性任务轨迹。"
        )
    else:
        explanation.append(
            "低/中风险静态工作负载推荐 exact，以保留完整事件轨迹。"
        )
    if risk_level in {"high", "critical"}:
        warnings.append("逻辑规模较大；建议限制并发 Job 数并保留取消能力。")
    if scheduler_mode == "continuous" and risk_level != "low":
        warnings.append("该连续场景不建议保留完整事件历史。")
    if (
        scheduler_mode == "static"
        and recommended_retention_policy == "streaming"
    ):
        warnings.append(
            "如需逐任务完整回放，可显式选择 exact，但内存将随逻辑任务数增长。"
        )
    if (
        scheduler_mode == "continuous"
        and scheduler.preemption_enabled
        and request_count > scheduler.max_num_seqs
    ):
        warnings.append(
            "高风险：这是无抢占基线；长 batch 与 starvation 阈值的相对关系未知，"
            "实际batch可因抢占/重算显著放大。"
        )

    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "scheduler_mode": scheduler_mode,
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "layer_count": layer_count,
        "world_size": world_size,
        "estimated_cohort_count": estimated_cohorts,
        "estimated_event_task_count": estimated_event_tasks,
        "event_tasks_per_layer_rank": _EVENT_TASKS_PER_LAYER_RANK,
        "estimated_mtp_round_count": estimated_mtp_rounds,
        "mtp_enabled": mtp_enabled,
        "mtp_candidate_tokens": candidate_tokens if mtp_enabled else 1,
        "mtp_candidate_tokens_semantics": "max_draft_tokens",
        "mtp_verifier_width_at_max": (
            1 + candidate_tokens if mtp_enabled else 1
        ),
        "mtp_proposal_length_model": (
            str(mtp_policy.proposal_length_model)
            if mtp_policy is not None
            else "disabled"
        ),
        "estimate_basis": "no_preemption_baseline",
        "estimate_basis_zh": "无抢占基线",
        "dynamic_preemption_modeled": False,
        "risk_level": risk_level,
        "risk_level_zh": risk_level_zh,
        "recommended_retention_policy": recommended_retention_policy,
        "explanation": explanation,
        "warnings": warnings,
    }


__all__ = ["estimate_scenario"]
