"""Additional real native layer samples for unsampled seed batch shapes."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re
import time


def _task_keys(stages, cohort):
    def key(identity):
        value = str(identity).replace(cohort.cohort_id, "@cohort")
        return re.sub(r"@cohort\.\d+\.", "", value)
    return {key(task.task_id): (task, frozenset(key(item) for item in task.dependencies))
            for stage in stages for task in stage.execution_tasks}


def _monotone_native_work(full_cohort, full_stages, anchor_cohort, anchor_stages):
    """Check the native endpoint Task graph, IO, and nonnegative work envelope.

    This qualifies a conditional monotone service model, not a measured
    physical prefix or a proof of arbitrary queue scheduling monotonicity.
    Initial controller/address/calendar variation remains separately paid.
    """
    full, anchor = _task_keys(full_stages, full_cohort), _task_keys(anchor_stages, anchor_cohort)
    result = {"qualified": False, "basis": "native_nonnegative_work_graph_endpoint_envelope",
        "conditional_service_assumption": "fixed native backend graph; adding batch items and nonnegative operator work cannot reduce boundary-normalized service beyond separately charged controller phase bounds",
        "anchor_task_count": len(anchor), "full_batch_task_count": len(full),
        "physical_state_advanced": False}
    if not anchor.keys() <= full.keys():
        result["reason"] = "native_seed_anchor_task_graph_is_not_contained_in_full_batch"
        return result
    for identity, (task, dependencies) in anchor.items():
        other, full_dependencies = full[identity]
        if not dependencies <= full_dependencies or task.category != other.category:
            result["reason"] = "native_seed_anchor_dependency_graph_is_not_contained_in_full_batch"
            return result
        def demands(row):
            values = {}
            for demand in row.demands:
                prior = values.setdefault(demand.resource_id, [0., 0., 0.])
                for index, value in enumerate((demand.service_ns, demand.bytes_moved, demand.work_units)):
                    prior[index] += value
            return values
        left, right = demands(task), demands(other)
        if any(resource not in right or any(value > upper for value, upper in zip(values, right[resource]))
               for resource, values in left.items()):
            result["reason"] = "native_seed_anchor_resource_service_or_work_is_not_monotone"
            return result
        def io(row):
            values = Counter()
            for access in row.metadata.get("memory_accesses", ()):
                values[(access.get("physical_owner", row.metadata.get("physical_owner")),
                    access.get("operation"), access.get("allocation_generation", 0) == 0,
                    str(access.get("buffer_id", "")).startswith("kv:"))] += access["byte_count"]
            return values
        left, right = io(task), io(other)
        if any(value > right[key] for key, value in left.items()):
            result["reason"] = "native_seed_anchor_physical_IO_is_not_contained_in_full_batch"
            return result
    result["qualified"] = True
    return result


def add_native_seed_shape_anchor(runtime, cohort, cost, stages, request_ready_ns,
                                 payload, options, control):
    """Append independently executed B=1 windows, never a completed B=1 batch."""
    from .native_layer_budget import _query_decode_contract, evaluate_native_layer_windows
    from .native_layer_sampling import sample_native_layer_windows, NativeLayerSamplingGuardError
    from .serving import _lower_cost

    if runtime.plan.scheduler.max_num_seqs <= 1:
        return payload
    control.raise_if_cancelled()
    if not evaluate_native_layer_windows(payload, target_relative_error=options.target_relative_error,
                                         independent_sufficient=False)["accepted"]:
        return {**payload, "seed_shape_anchor_status": "not_run_full_batch_layer_budget_rejected"}
    first = min(cohort.items, key=lambda item: item.request_id)
    anchor = replace(cohort, cohort_id=cohort.cohort_id+".seed-shape-B1", items=(first,))
    anchor_cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, anchor)
    anchor_stages, fallback = runtime._execution_stage_metadata_cache.resolve(anchor_cost.metadata)
    if fallback or not anchor_stages:
        raise NativeLayerSamplingGuardError("native_seed_anchor_IR_unavailable")
    anchor_stages = runtime._with_gpu_controller_stages(anchor_stages, anchor)
    proof = _monotone_native_work(cohort, stages, anchor, anchor_stages)
    if not proof["qualified"]:
        raise NativeLayerSamplingGuardError(proof["reason"])
    prefix = list(payload["records"])
    from .execution_control import ExecutionControl
    def progress(value):
        if value.emitted_monotonic is None:
            value = replace(value, emitted_monotonic=time.perf_counter())
        metadata = dict(value.metadata)
        layer_progress = metadata.get("physical_layer_progress")
        if layer_progress is not None:
            layer_progress = dict(layer_progress)
            layer_progress.update(completed_windows=len(prefix)+layer_progress["completed_windows"],
                total_windows=len(prefix)+layer_progress["total_windows"],
                executed_tasks=sum(row["executed_task_count"] for row in prefix)+layer_progress["executed_tasks"],
                batch_shape_anchor=1)
            metadata["physical_layer_progress"] = layer_progress
            value = replace(value, completed=len(prefix)+value.completed,
                total=None if value.total is None else len(prefix)+value.total, metadata=metadata)
        if control.progress_queue is not None:
            control.progress_queue.put(value)
        if control.progress_callback is not None:
            control.progress_callback(value)
    anchor_control = ExecutionControl(progress_callback=progress,
        cancellation_callback=control.cancellation_callback,
        stop_request_path=control.stop_request_path,
        deadline_monotonic=control.deadline_monotonic)
    def preserve(records):
        for record in records:
            for key in ("sample_id", "window_id"):
                if not record[key].startswith("seed-shape-B1."):
                    record[key] = "seed-shape-B1."+record[key]
            record["physical_observations"]["batch_shape_anchor"] = 1
        runtime._native_layer_sample_records = [*prefix, *records]
        return records
    # Cancellation may arrive while the helper is preparing its graph,
    # before it creates the new sample list. Do not mistake the old full-B
    # records for newly executed B=1 windows in that interval.
    runtime._native_layer_sample_records = []
    try:
        sampled = sample_native_layer_windows(runtime, anchor, anchor_cost, anchor_stages,
            request_ready_ns, replace(options,
                max_sample_layer_windows=options.max_seed_anchor_layer_windows), anchor_control)
    except BaseException:
        preserve(getattr(runtime, "_native_layer_sample_records", ()))
        raise
    records = preserve(sampled["records"])
    control.raise_if_cancelled()
    try:
        decode = _query_decode_contract(runtime, anchor, anchor_stages, query_cost=anchor_cost)
    except ValueError as error:
        if not str(error).startswith("native_layer_"):
            raise
        raise NativeLayerSamplingGuardError(str(error)) from error
    control.raise_if_cancelled()
    return {**payload, "records": [*prefix, *records], "batch_shape_anchors": [{
        "batch_size": 1, "request_id": first.request_id,
        "context_tokens": first.context_tokens, "sample_payload": sampled,
        "query_decode_native_IR": decode, "native_monotone_work_contract": proof,
        "physical_complete_cohort_count": 0, "physical_visible_output_token_count": 0,
        "sample_window_cap": options.max_seed_anchor_layer_windows}]}


__all__ = ["add_native_seed_shape_anchor"]
