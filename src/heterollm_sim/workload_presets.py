"""Offline workload preset catalog used by the Web UI.

The catalog describes request shapes and scheduler inputs only.  It does not
claim a measured throughput or latency result; those values depend on the
selected model, hardware topology, and runtime policy.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional


CATALOG_VERSION = "1.0.0"
CATALOG_SOURCE = "35_llm_deployment_audit/48场景典型负载与速度要求_v6.2.json"


def _preset(
    preset_id: str,
    label_zh: str,
    label_en: str,
    description_zh: str,
    description_en: str,
    source: str,
    source_id: str,
    prompt_tokens: int,
    output_tokens: int,
    batch: int,
    batch_tokens: int,
    prefill_chunk_tokens: int = 512,
) -> Dict[str, Any]:
    return {
        "id": preset_id,
        "label": [label_zh, label_en],
        "description": [description_zh, description_en],
        "source": source,
        "sourceId": source_id,
        "promptTokens": prompt_tokens,
        "outputTokens": output_tokens,
        "batch": batch,
        "batchTokens": batch_tokens,
        "prefillChunkTokens": prefill_chunk_tokens,
    }


WORKLOAD_PRESETS: tuple[Mapping[str, Any], ...] = (
    _preset(
        "llama_cpp_default",
        "llama.cpp 对齐基线",
        "llama.cpp aligned baseline",
        "单请求 512 输入 / 128 输出；continuous、B=1、无抢占、无 MTP。",
        "One request with 512 input / 128 output; continuous, B=1, no preemption, no MTP.",
        "llama.cpp runtime config defaults + explicit smoke shape",
        "llama-default",
        512,
        128,
        1,
        512,
    ),
    _preset(
        "personal_assistant",
        "个人 AI 助手（C01）",
        "Personal AI assistant (C01)",
        "长会话、检索与工具调用：4,096 输入 / 1,024 输出 / B=12。",
        "Long conversation with retrieval and tools: 4,096 input / 1,024 output / B=12.",
        CATALOG_SOURCE,
        "C01",
        4096,
        1024,
        12,
        2048,
    ),
    _preset(
        "enterprise_office",
        "企业 AI 办公（C02）",
        "Enterprise AI office (C02)",
        "长文档与多租户办公：16,384 输入 / 2,048 输出 / B=8。",
        "Long documents and multi-tenant office work: 16,384 input / 2,048 output / B=8.",
        CATALOG_SOURCE,
        "C02",
        16384,
        2048,
        8,
        2048,
    ),
    _preset(
        "customer_service",
        "智能客服（C03）",
        "Customer service (C03)",
        "知识库问答与工单转接：2,048 输入 / 256 输出 / B=16。",
        "Knowledge-base Q&A and ticket handoff: 2,048 input / 256 output / B=16.",
        CATALOG_SOURCE,
        "C03",
        2048,
        256,
        16,
        2048,
    ),
    _preset(
        "education",
        "AI 教育（C04）",
        "AI education (C04)",
        "课程材料、习题与讲解：8,192 输入 / 1,536 输出 / B=8。",
        "Course material, exercises, and explanations: 8,192 input / 1,536 output / B=8.",
        CATALOG_SOURCE,
        "C04",
        8192,
        1536,
        8,
        2048,
    ),
    _preset(
        "content_creation",
        "AI 内容创作（C05）",
        "AI content creation (C05)",
        "长篇创作与多媒体脚本：3,072 输入 / 8,192 输出 / B=16。",
        "Long-form creation and multimedia scripts: 3,072 input / 8,192 output / B=16.",
        CATALOG_SOURCE,
        "C05",
        3072,
        8192,
        16,
        2048,
    ),
    _preset(
        "software_development",
        "AI 软件开发（C06）",
        "AI software development (C06)",
        "代码库理解、补全与验证：默认模型安全截面 24,576 输入 / 4,096 输出 / B=16。",
        "Codebase understanding, completion, and verification: a safe 24,576 input / 4,096 output / B=16 slice for the default model.",
        CATALOG_SOURCE,
        "C06",
        24576,
        4096,
        16,
        2048,
    ),
    _preset(
        "search_platform",
        "AI 搜索平台（C07）",
        "AI search platform (C07)",
        "网页与企业资料聚合检索：12,288 输入 / 768 输出 / B=12。",
        "Web and enterprise-document retrieval: 12,288 input / 768 output / B=12.",
        CATALOG_SOURCE,
        "C07",
        12288,
        768,
        12,
        2048,
    ),
    _preset(
        "conversational_commerce",
        "对话式电商（C09）",
        "Conversational commerce (C09)",
        "商品比较、推荐与订单确认：3,584 输入 / 512 输出 / B=16。",
        "Product comparison, recommendations, and order confirmation: 3,584 input / 512 output / B=16.",
        CATALOG_SOURCE,
        "C09",
        3584,
        512,
        16,
        2048,
    ),
    _preset(
        "edge_personal_assistant",
        "端侧个人助手（T01）",
        "Edge personal assistant (T01)",
        "手机语音、日程与本地资料：1,024 输入 / 512 输出 / B=1。",
        "Mobile voice, calendar, and local documents: 1,024 input / 512 output / B=1.",
        CATALOG_SOURCE,
        "T01",
        1024,
        512,
        1,
        512,
    ),
)

_PRESET_BY_ID = {str(item["id"]): item for item in WORKLOAD_PRESETS}


def _catalog_metadata() -> Dict[str, Any]:
    return {
        "version": CATALOG_VERSION,
        "source": CATALOG_SOURCE,
        "kind": "request_shape",
        "claims": "planning_shape_only",
    }


def workload_preset_page(*, query: str = "") -> Dict[str, Any]:
    """Return the list payload consumed by the workload selector."""

    normalized_query = str(query or "").strip().lower()
    items: List[Dict[str, Any]] = []
    for item in WORKLOAD_PRESETS:
        if normalized_query:
            searchable = " ".join(
                str(item.get(key, ""))
                for key in ("id", "sourceId", "source")
            )
            searchable += " " + " ".join(str(value) for value in item.get("label", ()))
            if normalized_query not in searchable.lower():
                continue
        items.append(deepcopy(dict(item)))
    return {
        "items": items,
        "total": len(items),
        "filters": {"query": query or ""},
        "catalog": _catalog_metadata(),
    }


def workload_preset_detail(preset_id: str) -> Dict[str, Any]:
    """Return one preset with its materialized workload shape."""

    try:
        preset = deepcopy(dict(_PRESET_BY_ID[str(preset_id)]))
    except KeyError as exc:
        raise KeyError(str(preset_id)) from exc
    batch = int(preset["batch"])
    requests = [
        {
            "schema_version": "4.0.0",
            "request_id": f"request-{index:04d}",
            "arrival_ns": 0,
            "prompt_tokens": int(preset["promptTokens"]),
            "output_tokens": int(preset["outputTokens"]),
            "priority": 0,
            "deadline_ns": None,
        }
        for index in range(batch)
    ]
    return {
        "preset": preset,
        "workload": {
            "schema_version": "4.0.0",
            "name": preset["label"][0],
            "request_count": batch,
            "prompt_tokens": int(preset["promptTokens"]),
            "output_tokens": int(preset["outputTokens"]),
            "random_seed": 0,
            "arrival_rate_rps": 0,
            "requests": requests,
            "scheduler": {
                "mode": "continuous",
                "max_num_seqs": batch,
                "max_num_batched_tokens": int(preset["batchTokens"]),
                "max_num_ubatch_tokens": int(preset["batchTokens"]),
                "prefill_chunk_tokens": int(preset["prefillChunkTokens"]),
                "policy": "decode_first",
                "mixed_phase_batching": False,
                "phase_candidate_order": "least_recently_served",
                "starvation_ns": 5_000_000,
                "preemption_enabled": False,
                "preemption_granularity": "boundary",
                "preemption_policy": "auto",
                "slo_ttft_ns": None,
                "slo_tbt_ns": None,
                "prefill_stop_offsets": [],
            },
            "mtp": None,
            "metadata": {
                "workload_preset_id": preset["id"],
                "workload_preset_source": preset["source"],
                "workload_preset_source_scenario": preset["sourceId"],
            },
        },
        "catalog": _catalog_metadata(),
    }
