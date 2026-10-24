"""Internal, full-operator multi-turn KV lifecycle study (synthetic forecasts).

Turn prompts include all earlier prompt/output tokens plus new input. The
serving engine hands over physical KV pages; it computes only missing tokens
unless idle TTL expires. No trained predictor or application quality is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from heterollm_sim.ir import RequestSpec
from hbf_hbm_serving_study import build_study_scenario


def build_agent_scenario(*, policy="heuristic", ttl_ns=None, write_budget_bytes=None,
                         hbf_page_read_ns=75000., cold_after_ns=50000.,
                         prefetch_lead_ns=200000., think_times=(500000., 10000.),
                         predictions=(500000., 10000.), background=True):
    scenario = build_study_scenario("C_hot_cold", prompt=128, output=8, batch=1,
        request_count=1, prefetch_distance=128, hbf_page_read_ns=hbf_page_read_ns)
    requests = []
    prompt = 64
    for index in range(len(think_times) + 1):
        metadata = {"session_id": "session-0", "turn_index": index}
        if index:
            metadata.update(parent_request_id=f"agent-{index-1}",
                think_time_ns=think_times[index-1], predicted_think_time_ns=predictions[index-1])
            prompt += 4 + 16  # Previous output and genuinely new user/tool input.
        requests.append(RequestSpec(f"agent-{index}", 0., prompt, 4,
                                    metadata={"agent_turn": metadata}))
    if background:
        requests.append(RequestSpec("independent", 200000., 128, 16))
    config = {"policy": policy, "cold_after_ns": cold_after_ns,
        "prefetch_lead_ns": prefetch_lead_ns, "ttl_ns": ttl_ns,
        "write_budget_bytes": write_budget_bytes}
    return replace(scenario, name="agent-lifecycle-" + policy,
        placement=replace(scenario.placement, kv_policy=replace(
            scenario.placement.kv_policy, allocation_policy="lazy",
            prefetch_distance=0 if policy == "static_cold" else 128)),
        workload=replace(scenario.workload, requests=tuple(requests), request_count=len(requests),
            metadata={**scenario.workload.metadata, "agent_lifecycle": config}))


def run_agent_case(row_id, **parameters):
    from hbf_hbm_experiments import serving_row
    captured = []
    scenario = build_agent_scenario(**parameters)
    row = serving_row(scenario, row_id,
                      placement="C_hot_cold", result_observer=captured.append)
    result = captured[0].serving
    row["parameters"] = parameters
    row["agent_lifecycle"] = dict(result.agent_lifecycle_metrics)
    row["agent_policy"] = dict(scenario.workload.metadata["agent_lifecycle"])
    row["agent_policy"]["write_budget_scope"] = "cumulative logical KV bytes submitted for cold migration; physical amplification is separately billed"
    row["agent_events"] = [asdict(event) for event in result.events
        if event.event_type.startswith("agent_") or event.event_type.startswith("kv_prefetch")]
    row["computed_prefill_tokens"] = sum(item.token_count for batch in result.batches
                                         for item in batch.items if item.phase == "prefill")
    row["independent_prompt_tokens_sum"] = sum(state.prompt_tokens for state in result.request_states.values())
    row["gpu_operator_stages"] = sum(1 for batch in result.batches
        for stage in batch.cost.metadata.get("execution_stage_schedule", ())
        if stage.get("component_id") == "gpu0" and stage.get("invocation_group_id") is not None)
    row["final_pool"] = result.runtime_kernel_metrics["paged_kv_pool"]
    phases = [phase for event in result.events if event.event_type == "kv_prefetch_cancelled"
              for phase in event.details["submitted_phases"]]
    row["cancelled_work"] = {
        "physical_read_bytes": sum(phase.get("physical_read_bytes", 0) for phase in phases),
        "physical_write_bytes": sum(phase.get("physical_write_bytes", 0) for phase in phases),
        "link_bytes": sum(phase.get("bytes", 0) for phase in phases if phase["kind"] == "link"),
        "phase_service_sum_ns": sum(phase.get("service_ns", phase.get("duration_ns", 0.)) for phase in phases),
        "time_semantics": "sum of submitted phase service, including concurrent phases; not elapsed wall time"}
    return row


def run_agent_study():
    cases = (
        ("agent-static-cold", {"policy": "static_cold"}),
        ("agent-heuristic", {}),
        ("agent-write-budget", {"write_budget_bytes": 0}),
        ("agent-prediction-miss", {"think_times": (500000., 10000.), "predictions": (100000., 500000.)}),
        ("agent-ttl-unused", {"ttl_ns": 400000., "think_times": (1000000.,),
             "predictions": (100000.,), "prefetch_lead_ns": 100000., "hbf_page_read_ns": 1000.}),
        ("agent-ttl-cancel", {"ttl_ns": 400000., "think_times": (2000000.,),
             "predictions": (100000.,), "prefetch_lead_ns": 100000., "hbf_page_read_ns": 500000.}),
    )
    rows = []
    for row_id, parameters in cases:
        print("[agent lifecycle] " + row_id, flush=True)
        rows.append(run_agent_case(row_id, **parameters))
    by_id = {row["id"]: row for row in rows}
    checks = [
        {"name": "all_turns_complete", "passed": all(row["completed_requests"] == row["request_count"] for row in rows)},
        {"name": "real_page_handoff_and_avoided_prefill", "passed": all(
            row["agent_lifecycle"]["handoffs"] > 0 and row["computed_prefill_tokens"] < row["independent_prompt_tokens_sum"]
            for row in rows[:4])},
        {"name": "final_pages_released", "passed": all(row["final_pool"]["page_count"] == 0 for row in rows)},
        {"name": "real_gpu_operators_and_safe_copy_dependencies", "passed": all(row["gpu_operator_stages"] > 0
            and row["prefetch_overlap"]["premature_consumer_overlap_ns"] == 0 for row in rows)},
        {"name": "heuristic_overlaps_independent_decode", "passed": by_id["agent-heuristic"]["prefetch_overlap"]["decode_operator_overlap_ns"] > 0},
        {"name": "heuristic_mixes_hot_and_cold", "passed": all(by_id["agent-heuristic"]["agent_lifecycle"][key] > 0
                                                                    for key in ("hot_decisions", "cold_decisions"))},
        {"name": "write_budget_enforced", "passed": by_id["agent-write-budget"]["agent_lifecycle"]["cold_write_bytes"] == 0
            and by_id["agent-write-budget"]["agent_lifecycle"]["write_budget_denials"] > 0},
        {"name": "explicit_forecast_errors_measured", "passed": by_id["agent-prediction-miss"]["agent_lifecycle"]["prediction_misses"] > 0},
        {"name": "completed_unused_prefetch", "passed": by_id["agent-ttl-unused"]["kv"]["prefetch_unused_bytes"] > 0},
        {"name": "inflight_cancel_drained_and_recomputed", "passed": by_id["agent-ttl-cancel"]["agent_lifecycle"]["cancellations"] > 0
            and by_id["agent-ttl-cancel"]["agent_lifecycle"]["expired_tokens"] > 0
            and by_id["agent-ttl-cancel"]["cancelled_work"]["physical_read_bytes"] > 0},
    ]
    passed = all(check["passed"] for check in checks)
    return {"id": "13_agent_lifecycle", "status": "passed" if passed else "failed", "rows": rows,
        "checks": checks, "capability": {"supported": ["single_chain_agent_session", "physical_KV_handoff",
            "pause_cold_migration", "heuristic_TTL_write_budget", "cost_preserving_prefetch_cancellation"] if passed else []},
        "scope": {"synthetic_trace": True, "trained_predictor": False, "external_calibration": False,
            "unsupported": ["branching/shared session inheritance", "speculative or recurrent-state sessions"],
            "hot_and_read_mostly_mix": "active append turns plus idle immutable history; no assumed hit-rate speedup",
            "prediction_inputs": "explicit synthetic hints, fallback last observed think time",
            "ttl_semantics": "idle cache expiration; next turn recomputes history",
            "cold_migration": "synchronous physical offload; asynchronous staged prefetch restore"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hbf_hbm/13_agent_lifecycle.json")
    args = parser.parse_args()
    group = run_agent_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(group, indent=2, ensure_ascii=False), encoding="utf-8")
    print(group["status"], args.output, flush=True)
    raise SystemExit(0 if group["status"] == "passed" else 1)
