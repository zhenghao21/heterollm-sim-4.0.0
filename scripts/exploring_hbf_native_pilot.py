"""Small functional pilots through the existing native serving/physical runtime.

These are synthetic BF16 models and existing research memory profiles, not
paper throughput results, a steady-state claim, or a replacement backend.
HBF controls separately place only KV or both weights and KV in HBF;
activations/intermediates stay in HBM.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hbf_hbm_core_audit import CoreExecutionAudit
from hbf_hbm_experiments import scenario as existing_scenario
from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.ir import (
    LayerSpec, LinkSpec, ModelSpec, ParallelSpec, RankMappingSpec,
    build_model_graph_from_layer_specs,
)
from heterollm_sim.planner import compile_serving_cohort_schedule
from heterollm_sim.reporting import (
    _sum_batch_dram_traffic, _sum_batch_storage_traffic, report_dict, run_scenario,
)
from heterollm_sim.serde import write_json
from heterollm_sim.serving import BatchCohort, RequestStatus


CASES = ("hbm_tp1", "hbf_kv_tp1", "hbm_tp2", "hbf_weights_kv_tp1")


def build_pilot_scenario(case: str):
    if case not in CASES:
        raise ValueError(f"unknown native pilot {case}")
    base = existing_scenario(hbf=case in {"hbf_kv_tp1", "hbf_weights_kv_tp1"}, prompt=32,
                             count=8, output=4, slots=2)
    # Q/K/V/O = 256*(256+128+128+256), gated FFN = 3*256*512;
    # match the native LayerSpec matrix-weight contract in BF16 bytes.
    layer_matrix_bytes = (256 * (256 + 128 + 128 + 256) + 3 * 256 * 512) * 2
    layers = tuple(LayerSpec(f"dense{i}", "dense", 256, 512, 4,
                             kv_heads=2, dtype="bf16", weight_bytes=layer_matrix_bytes)
                   for i in range(2))
    model = ModelSpec(name="native-pilot-2layer-bf16", graph=build_model_graph_from_layer_specs(
        "native-pilot-2layer-bf16", layers, vocabulary_size=1024,
        max_sequence_length=256, embedding_weight_bytes=256 * 1024 * 2,
        output_weight_bytes=256 * 1024 * 2, output_head_dtype="bf16"))
    hardware = base.hardware
    if case == "hbm_tp2":
        gpu = hardware.get_component("gpu0")
        hardware = replace(hardware,
            components=hardware.components + (replace(gpu, component_id="gpu1", die_id="pilot_gpu1_die"),),
            links=tuple(replace(link, source_component="gpu1")
                        if link.link_id == "gpu-hbm1" else link
                        for link in hardware.links) + (
                LinkSpec("pilot-peer", "gpu0", "ucie0", "gpu1", "ucie0", "UCIe",
                         "1.1", 64, 2048., 10., payload="streaming"),))
    tp = 2 if case == "hbm_tp2" else 1
    parallel = ParallelSpec(tp_degree=tp, rank_mapping=tuple(
        RankMappingSpec(rank, f"gpu{rank}", rank, 0, 0, f"hbm{rank}")
        for rank in range(tp)), layer_to_stage={layer.layer_id: 0 for layer in layers})
    configured = replace(base, name="exploring-hbf-native-pilot-" + case, hardware=hardware, model=model,
        runtime_profile=replace(base.runtime_profile, gpu_controllers={
            f"gpu{rank}": base.runtime_profile.gpu_controllers["gpu0"] for rank in range(tp)}),
        placement=replace(base.placement, model_name=model.name, parallel=parallel,
            kv_policy=replace(base.placement.kv_policy, dtype="bf16")),
        assumptions=base.assumptions + (
            "Functional native-runtime pilot: synthetic two-layer BF16 model, not paper reproduction.",
            "HBF case changes KV placement only; weights and intermediates remain in HBM.",
            "Finite eight-request trace, no steady-state throughput or external-PD claim.",))
    if case == "hbf_weights_kv_tp1":
        requirements = _derive_requirements(configured)
        options = {"design_decode_batch_size": 2, "design_prefill_tokens": 32,
            "kv_cache_target": "hbf0",
            "operator_targets": {item.mapping_key: "gpu0" for item in requirements
                                 if item.mapping_key and not item.state_tensor},
            "weight_tensor_targets": {item.tensor_id: "hbf0" for item in requirements
                if item.tensor_id and item.tensor_bytes > 0 and not item.state_tensor}}
        configured = replace(configured, placement=replace(configured.placement,
            metadata={**configured.placement.metadata, "control_plane": {"policy": {"options": options}}}),
            assumptions=tuple(value for value in configured.assumptions
                if value != "HBF case changes KV placement only; weights and intermediates remain in HBM.") + (
                    "HBF mixed placement: all model weights and KV use HBF; activations/intermediates use HBM.",))
    return configured


def _compiled_collectives(result):
    """Inspect the original graphs for executed cohorts; do not execute replay."""
    counts, names = Counter(), set()
    for batch in result.serving.batches:
        cohort = BatchCohort(batch.cohort_id, batch.kind, batch.start_ns, batch.items,
                             batch.proposal_cost_scale, batch.metadata)
        schedule = compile_serving_cohort_schedule(result.scenario, cohort)
        for task in schedule.tasks:
            if task.metadata.get("collective_kind") != "all_reduce":
                continue
            if not any(demand.service_ns > 0 for demand in task.demands):
                continue
            counts["all_reduce_positive_service_tasks"] += 1
            if task.metadata.get("link_id"):
                counts["all_reduce_link_tasks"] += 1
            if task.metadata.get("link_id") == "pilot-peer":
                counts["all_reduce_peer_link_tasks"] += 1
            if task.metadata.get("collective_name"):
                names.add((batch.cohort_id, task.metadata["collective_name"]))
    return {**counts, "distinct_all_reduce_operations": len(names),
            "scope": "compiled native graphs of realized cohorts, not an additional simulation or retained transaction trace"}


def run_pilot(case: str, output_dir: Path):
    scenario = build_pilot_scenario(case)
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "scenario.json", scenario)
    started = time.perf_counter()
    with CoreExecutionAudit() as audit:
        result = run_scenario(scenario)
    wall = time.perf_counter() - started
    core = audit.summary()
    report = report_dict(result, include_visualization=False)
    write_json(case_dir / "report.json", report)
    metrics = list(result.serving.request_metrics.values())
    admissions = [event for event in result.serving.events if event.event_type == "request_admitted"]
    finished = [metric for metric in metrics if metric.status == RequestStatus.FINISHED]
    first_finish = min((metric.finish_ns for metric in finished), default=None)
    batches = [{"id": batch.cohort_id, "kind": batch.kind,
                "start_ns": batch.start_ns, "end_ns": batch.end_ns,
                "request_ids": list(batch.request_ids), "token_count": batch.token_count,
                "batch_size": len(batch.request_ids)} for batch in result.serving.batches]
    means = lambda field: statistics.mean(values) if (values := [getattr(metric, field)
        for metric in finished if getattr(metric, field) is not None]) else None
    peer_busy = Counter()
    for batch in result.serving.batches:
        for resource, busy in batch.cost.metadata.get("resource_busy_ns", {}).items():
            if resource.startswith("link.pilot-peer.") and not resource.endswith(".inflight"):
                peer_busy[resource] += busy
    weights = {item.tensor_id for item in _derive_requirements(result.scenario)
               if item.tensor_id and item.tensor_bytes > 0 and not item.state_tensor}
    resolved_weights = {identity: result.scenario.placement.tensor_to_component.get(identity)
                        for identity in sorted(weights)}
    row = {
        "case": case, "status": "completed" if len(finished) == 8 else "not_completed",
        "scope": "native functional pilot, not paper results or steady-state TPS",
        "execution_path": "reporting.run_scenario -> serving.simulate_online -> TopologyAwareBatchCostProvider -> UnifiedEventKernel",
        "parameters": {"model": scenario.model.name, "layers": 2, "hidden": 256,
            "intermediate": 512, "q_heads": 4, "kv_heads": 2, "vocabulary": 1024,
            "weights_and_kv": "BF16", "prompt_tokens": 32, "output_tokens": 4,
            "request_count": 8, "max_concurrent_sequences": 2,
            "tp": scenario.placement.parallel.tp_degree,
            "kv_memory": scenario.placement.kv_policy.cache_component,
            "weight_memory": "HBF" if case == "hbf_weights_kv_tp1" else "HBM",
            "activation_and_intermediate_memory": "HBM",
            "hbf_protocol": "generic analytical NAND page service, not OCP"},
        "completed_requests": len(finished),
        "completed_visible_output_tokens": sum(metric.visible_output_tokens for metric in finished),
        "admitted_requests": len(admissions),
        "admitted_after_first_completion": sum(event.timestamp_ns >= first_finish
            for event in admissions) if first_finish is not None else 0,
        "admission_events": [{"timestamp_ns": event.timestamp_ns,
            "request_id": event.request_id, **dict(event.details)} for event in admissions],
        "batch_size_distribution": dict(Counter(batch["batch_size"] for batch in batches)),
        "batches": batches, "mean_ttft_ns": means("ttft_ns"), "mean_tpot_ns": means("tpot_ns"),
        "wall_seconds": wall, "dram_traffic": _sum_batch_dram_traffic(result),
        "nand_traffic": _sum_batch_storage_traffic(result),
        "native_collective_graph": _compiled_collectives(result),
        "executed_peer_link_service_ns": dict(peer_busy),
        "executed_collective_category_service_ns": sum(
            batch.cost.metadata.get("category_time_ns", {}).get("collective", 0)
            for batch in result.serving.batches),
        "resolved_placement": {"weight_tensor_owners": resolved_weights,
            "kv_owner": result.scenario.placement.kv_policy.cache_component,
            "rank_active_memory": [rank.memory_component_id
                                   for rank in result.scenario.placement.parallel.rank_mapping]},
        "live_physical_cores": {key: core[key] for key in (
            "scope", "valid_for_committed_byte_totals", "run_count", "finished_run_count",
            "live_runtime_restores", "failed_entrypoint_calls", "owners")},
        "limitations": ["Eight finite requests include initial Prefill and final drain.",
            "No external PD import, constant full-batch steady state, or paper model/hardware accuracy is claimed.",
            "Original synthetic hardware assumptions are retained; this is not a device performance prediction."]}
    if case == "hbm_tp2":
        owners = {owner["physical_owner"] for owner in core["owners"] if owner["transactions"]}
        if not {"hbm0.hbm_fabric", "hbm1.hbm_fabric"}.issubset(owners) or not peer_busy:
            raise RuntimeError("native TP2 did not execute both private HBM owners and its peer link")
    if case == "hbf_weights_kv_tp1" and set(resolved_weights.values()) != {"hbf0"}:
        raise RuntimeError("native mixed placement did not bind all matrix weights to HBF")
    write_json(case_dir / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "artifacts/exploring_hbf_native_20261019/pilot")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        row = run_pilot(case, args.output_dir)
        rows.append(row)
        stored = [json.loads(path.read_text(encoding="utf-8")) for name in CASES
                  if (path := args.output_dir / name / "result.json").exists()]
        write_json(args.output_dir / "results.json", {"scope": "native functional pilots", "results": stored})
        print(json.dumps({key: row[key] for key in ("case", "status", "completed_requests",
            "completed_visible_output_tokens", "wall_seconds")}), flush=True)
    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
