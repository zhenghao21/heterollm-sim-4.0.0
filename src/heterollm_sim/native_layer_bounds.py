"""Read-only native graph proofs for layer-window endpoint and work bounds."""
from __future__ import annotations

import math
from typing import Mapping


def _native_backing_payload_upper(count, quanta):
    """Different valid transaction quanta need not divide the largest."""
    return max((((count+quantum-1)//quantum)*quantum for quantum in quanta), default=count)


def native_layer_boundary_bounds(scenario, cost, stages):
    """Inspect existing prepared Tasks; never execute or invent a cohort.

    A nonempty dense batch keeps its authored operator/launch graph. The
    floor retains only native zero-work launch service and preloaded tensor
    read byte rates. Variable arithmetic, activation IO, expert launches and
    opaque callback costs supply no invariant lower bound.
    """
    from .native_window_producer import _bandwidth
    from .physical_contract import parse_execution_physical_config

    result = {"batch_invariant_native_path_floor_ns": 0.,
        "batch_invariant_native_path_floor_basis": None,
        "visible_commit_lag_upper_ns": None,
        "visible_commit_boundary_proof": {"qualified": False}}
    if not stages:
        return result
    stage_ids = {stage.stage_id for stage in stages}
    if len(stage_ids) != len(stages):
        return result
    visited, depended = set(), set()
    for stage in stages:
        if not set(stage.dependencies) <= visited:
            return result
        visited.add(stage.stage_id)
        depended.update(stage.dependencies)
    sinks = stage_ids-depended
    task_stage = {task.task_id: stage.stage_id for stage in stages for task in stage.execution_tasks}
    terminals = cost.metadata.get("item_terminal_tasks", ())
    item_count = scenario.workload.scheduler.max_num_seqs
    if (len(terminals) == item_count
            and {row.get("item_index") for row in terminals} == set(range(item_count))):
        covered, rows = True, []
        for row in terminals:
            terminal_stage = task_stage.get(row.get("terminal_task_id"))
            groups = set(row.get("operator_invocation_group_ids", ()))
            # Exactly the live serving rule: an existing terminal Task uses
            # its stage; an omitted logical marker falls back to max(group).
            selected = ({terminal_stage} if terminal_stage is not None else
                {stage.stage_id for stage in stages
                 if groups.intersection(stage.causal_invocation_group_ids)})
            qualified = sinks <= selected
            covered &= qualified
            rows.append({"item_index": row["item_index"], "qualified": qualified,
                "basis": "explicit_terminal_stage" if terminal_stage is not None else
                         "native_logical_marker_fallback_to_invocation_group_end"})
        result["visible_commit_boundary_proof"] = {"qualified": covered,
            "basis": "every_native_item_terminal_covers_all_prepared_stage_DAG_sinks",
            "sink_stage_ids": sorted(sinks), "items": rows,
            "physical_Token_observed": False}
        if covered:
            result["visible_commit_lag_upper_ns"] = 0.
    if any(instance.layer.is_moe for instance in scenario.model._execution_view.layer_instances):
        return result
    task_finish, stage_finish, parsed = {}, {}, {}
    launches = weights = 0
    for stage in stages:
        stage_start = max((stage_finish[key] for key in stage.dependencies), default=0.)
        for task in stage.execution_tasks:
            md, invariant = task.metadata, 0.
            if md.get("phase") == "kernel_launch" and md.get("cost_model") is True:
                constant = [d.service_ns for d in task.demands
                    if d.work_units == 0 and d.bytes_moved == 0
                    and math.isfinite(d.service_ns) and d.service_ns >= 0]
                if constant:
                    invariant = max(constant)
                    launches += 1
            if md.get("weight_tensor_id") and md.get("weight_lifecycle_mode") == "preloaded_resident":
                configs, owner = md.get("physical_memory_configs", {}), md.get("physical_owner")
                for access in md.get("memory_accesses", ()):
                    count = access.get("byte_count")
                    if (access.get("operation") != "read" or access.get("allocation_generation", 0) != 0
                            or not str(access.get("buffer_id", "")).startswith("tensor:")
                            or type(count) is not int or count < 0):
                        continue
                    raw = configs.get(access.get("physical_owner", owner), md.get("physical_memory_config"))
                    if isinstance(raw, Mapping):
                        key = id(raw)
                        if key not in parsed:
                            parsed[key] = parse_execution_physical_config(raw)
                        raw = parsed[key]
                    if raw is None:
                        continue
                    invariant = max(invariant, count/_bandwidth(raw, "read"))
                    weights += 1
            task_finish[task.task_id] = max(stage_start,
                max((task_finish.get(key, 0.) for key in task.dependencies), default=0.))+invariant
        stage_finish[stage.stage_id] = max(stage_start,
            max((task_finish[task.task_id] for task in stage.execution_tasks), default=stage_start))
    result.update(batch_invariant_native_path_floor_ns=max(stage_finish.values(), default=0.),
        batch_invariant_native_path_floor_basis="native_dense_fixed_operator_graph_zero_work_launch_and_preloaded_weight_read_DAG",
        batch_invariant_launch_task_count=launches, batch_invariant_weight_read_count=weights,
        batch_invariant_scope="any_nonempty_dense_batch_under_the_fixed_native_operator_graph_contract")
    return result


def native_mmu_context_bounds(runtime, cohort, stages, *, payload_maxima=None):
    """Bound the native MMU batching formula without assuming cache hits.

    The live controller translates L2 payload, falling back to physical
    descriptors only when L2 contributes zero bytes. Native closed-interval
    caches never create payload beyond the declared logical read/write
    stream. Retaining its full payload also covers nonmonotone hit curves.
    """
    from .cost_models import GPUProfile, HBMProfile, HostMemoryProfile
    from .memory_types import DramConfig, NandConfig, SramConfig
    from .physical_contract import parse_execution_physical_config
    from .planner import (TopologyAwareBatchCostProvider, _activation_storage_bits,
        _component_resource_id, _dtype_bits, _lm_head_output_dtype,
        _parallel_plan, _rank_memory_resource)
    from .serving import _OnlineRuntime

    if type(runtime) is not _OnlineRuntime or type(runtime.lowerer) is not TopologyAwareBatchCostProvider:
        raise ValueError("native_layer_future_MMU_requires_native_runtime_and_lowerer")
    scenario = runtime.plan.scenario
    # Native estimator payloads can already be rounded to their backing
    # profile's transaction quantum before reaching physical descriptors.
    # That quantum need not equal the physical DRAM burst or NAND page.
    backing_quanta = tuple(sorted({profile.transaction_bytes
        for profiles in scenario.component_profiles.values() for profile in profiles.values()
        if isinstance(profile, (HBMProfile, HostMemoryProfile))}))
    backing_quantum = max(backing_quanta, default=1)
    backing_padded = {}
    controller_stages = {}
    original = []
    for stage in stages:
        controller = [task for task in stage.execution_tasks
                      if task.metadata.get("event_kind") == "gpu_mmu_tlb_batch"]
        if controller:
            if (len(controller) != 1 or len(stage.execution_tasks) != 1
                    or stage.stage_id != controller[0].task_id):
                raise ValueError("native_layer_future_MMU_controller_stage_not_native")
            controller_stages[stage.stage_id] = stage
        else:
            if any(task.metadata.get("event_kind") == "gpu_memory_controller_pipeline"
                   for task in stage.execution_tasks):
                raise ValueError("native_layer_future_MMU_custom_VRAM_path_not_bounded")
            original.append(stage)
    resolved, gpu_rows, parsed, physical_uppers = {}, [], {}, {}
    layers = {instance.layer.layer_id: instance.layer
              for instance in scenario.model._execution_view.layer_instances}
    if payload_maxima is None:
        payload_maxima = {}
    stage_positions = {stage.stage_id: index for index, stage in enumerate(original)}
    graph = tuple((stage.component_id, len(stage.execution_tasks),
        tuple(stage_positions.get(parent, -1) for dependency in stage.dependencies
              for parent in (controller_stages[dependency].dependencies
                  if dependency in controller_stages else (dependency,))))
        for stage in original)
    if payload_maxima.setdefault("native_graph", graph) != graph:
        raise ValueError("native_layer_future_MMU_root_graph_changed")
    maxima = payload_maxima.setdefault("input_upper_bytes", {})
    # Translation payload lower bounds. A context-insensitive native Task
    # has no context input: its payload must be identical at every corner.
    # A no-reuse closed-interval Task (reuse=1, streaming=1 fixed by its
    # native lowering) sends its whole logical stream through L2, which is
    # monotone in context, so corner extrema bound every interior context.
    minima = payload_maxima.setdefault("input_lower_bytes", {})
    tight = payload_maxima.setdefault("input_exact_upper_bytes", {})
    observed = payload_maxima.setdefault("fixed_input_bytes", {})
    from .native_window_producer import _context_sensitive
    known_models = {"gpu_hbm_roofline", "gpu_fused_attention_v3", "gpu_memory_roofline",
                    "gpu_reduction_roofline", "gpu_elementwise_roofline"}
    for gpu_id, controllers in sorted(scenario.runtime_profile.gpu_controllers.items()):
        gpu = scenario.resolve_component_profile(gpu_id, GPUProfile)
        if gpu.kernel_model is not None:
            raise ValueError("native_layer_future_MMU_dynamic_GPU_profile_not_bounded")
        level = next((item for item in reversed(gpu.cache_hierarchy.levels)
                      if str(item.name).strip().lower() == "l2"), None)
        reference = scenario.host_orchestration_profile.gpu_component_id
        l2 = (_component_resource_id(level.resource_id, reference_component_id=reference,
                                    target_component_id=gpu_id) if level else None)
        memory_ids = {_rank_memory_resource(scenario, rank)
                      for rank in _parallel_plan(scenario).ranks if rank.component_id == gpu_id}

        def task_bytes(task):
            actual, upper, positive, exact = _task_bytes(task)
            return actual, upper, positive, exact

        def _task_bytes(task):
            md = task.metadata
            if len({d.resource_id for d in task.demands}) != len(task.demands):
                raise ValueError("native_layer_future_duplicate_resource_demand")
            l2_demands = [d for d in task.demands if d.resource_id == l2]
            observed_l2 = sum(d.bytes_moved for d in l2_demands)
            accesses = tuple(md.get("memory_accesses", ()))
            physical = (md.get("physical_memory_config") is not None
                        or bool(md.get("physical_memory_configs")) or md.get("stateful_l2") is not None)
            if md.get("stateful_l2") is not None:
                raise ValueError("native_layer_future_MMU_stateful_cache_not_bounded")
            actual_physical = 0
            for access in accesses:
                count = access.get("byte_count")
                if type(count) is not int or count < 0:
                    raise ValueError("native_layer_future_MMU_physical_byte_count_invalid")
                actual_physical += count
            if not l2_demands and not physical:
                if any(d.bytes_moved for d in task.demands if d.resource_id in memory_ids):
                    raise ValueError("native_layer_future_MMU_custom_memory_input_not_bounded")
                return 0, 0, False, "fixed"
            if not l2_demands and not accesses:
                return 0, 0, False, "fixed"
            if md.get("event_kind") == "host_cohort_h2d":
                payload = scenario.host_orchestration_profile.payload_bytes(
                    len(cohort.items), sum(item.token_count for item in cohort.items))
                expected_buffer = "transfer:{}.target_operator_frontend.{}.payload.h2d".format(
                    cohort.cohort_id, gpu_id)
                if (l2_demands or len(accesses) != 1
                        or md.get("orchestration_stage") != "gpu_consumer_frontend"
                        or md.get("controller_target_component") != gpu_id
                        or md.get("cohort_request_count") != len(cohort.items)
                        or md.get("cohort_token_count") != sum(item.token_count for item in cohort.items)
                        or md.get("payload_bytes") != payload
                        or accesses[0].get("buffer_id") != expected_buffer
                        or accesses[0].get("byte_count") != payload
                        or accesses[0].get("allocation_size_bytes") != payload
                        or accesses[0].get("offset_bytes") != 0
                        or accesses[0].get("operation") not in {"read", "write"}):
                    raise ValueError("native_layer_future_MMU_frontend_payload_not_native")
                physical_uppers[task.task_id] = (payload,)
                return payload, payload, "native_frontend_payload" if payload > 0 else False, "fixed"
            if md.get("event_kind") == "collective_transfer":
                layer = layers.get(md.get("layer_id"))
                name = md.get("collective_name", "")
                if (layer is None or layer.is_moe or l2_demands
                        or md.get("collective_kind") != "all_reduce"
                        or md.get("algorithm") not in {"tree", "ring"}
                        or not any(name == "{}.{}.{}.{}".format(
                            cohort.cohort_id, cohort.kind, layer.layer_id, suffix)
                            for suffix in ("attention_all_reduce", "mlp_all_reduce"))
                        or len(accesses) != 1
                        or accesses[0].get("operation") not in {"read", "write"}
                        or accesses[0].get("offset_bytes") != 0
                        or not str(accesses[0].get("buffer_id", "")).startswith("transfer:"+name+".")
                        or accesses[0].get("allocation_size_bytes") != actual_physical):
                    raise ValueError("native_layer_future_MMU_collective_payload_not_native")
                bits = _activation_storage_bits(layer, scenario)
                if bits not in {16, 32}:
                    raise ValueError("native_layer_future_MMU_collective_dtype_not_bounded")
                elements = sum(item.token_count for item in cohort.items)*layer.hidden_size
                tp = scenario.placement.parallel.tp_degree
                padded = ((elements+tp-1)//tp)*tp*bits//8
                if not 0 < actual_physical <= padded:
                    raise ValueError("native_layer_future_MMU_collective_bytes_exceed_native_tensor")
                physical_uppers[task.task_id] = (actual_physical,)
                return actual_physical, padded, "native_dense_collective_tensor", "fixed"
            if md.get("event_kind") == "lm_head_all_gather":
                name = "{}.{}.lm_head_all_gather".format(cohort.cohort_id, cohort.kind)
                vocabulary = scenario.model._execution_view.vocabulary_size
                bits = _dtype_bits(_lm_head_output_dtype(scenario)[0])
                if (l2_demands or len(accesses) != 1 or bits not in {16, 32}
                        or md.get("collective_kind") != "all_gather"
                        or md.get("algorithm") not in {"tree", "ring"}
                        or md.get("collective_name") != name
                        or md.get("logits_vocabulary_size") != vocabulary
                        or md.get("logits_output_bits") != bits
                        or accesses[0].get("operation") not in {"read", "write"}
                        or accesses[0].get("offset_bytes") != 0
                        or not str(accesses[0].get("buffer_id", "")).startswith("transfer:"+name+".")
                        or accesses[0].get("allocation_size_bytes") != actual_physical):
                    raise ValueError("native_layer_future_MMU_logits_collective_not_native")
                tp = scenario.placement.parallel.tp_degree
                padded = (max(1, sum(item.logit_tokens for item in cohort.items))
                          *((vocabulary+tp-1)//tp)*tp*bits//8)
                if not 0 < actual_physical <= padded:
                    raise ValueError("native_layer_future_MMU_logits_bytes_exceed_native_tensor")
                physical_uppers[task.task_id] = (actual_physical,)
                return actual_physical, padded, "native_logits_collective_tensor", "fixed"
            model = md.get("cost_model")
            if not isinstance(model, Mapping):
                model = md.get("native_cost_model")
            if (not isinstance(model, Mapping) or model.get("model") not in known_models
                    or model.get("profile") != gpu.name
                    or model.get("source_partial_service_ns", 0.)
                    or model.get("quantized_weight_path") or model.get("mmvq_work") or model.get("mmq_work")):
                raise ValueError("native_layer_future_MMU_payload_source_not_bounded:"
                    +repr((task.task_id, md.get("event_kind"),
                           model.get("model") if isinstance(model, Mapping) else None)))
            cache = model.get("cache", {})
            if (cache.get("cache_model") != "v4_directional_closed_interval"
                    or cache.get("cross_invocation_cache_state") is not False):
                raise ValueError("native_layer_future_MMU_cache_payload_not_closed")
            reads, writes = cache.get("logical_read_bytes"), cache.get("logical_write_bytes")
            if any(type(value) is not int or value < 0 for value in (reads, writes)):
                raise ValueError("native_layer_future_MMU_logical_payload_invalid")
            logical = reads+writes
            if type(observed_l2) is not int or observed_l2 < 0 or observed_l2 > logical:
                raise ValueError("native_layer_future_MMU_L2_exceeds_native_payload")
            # Cover the L2==0 physical fallback with full directional bytes
            # at the actual physical burst/page granularity. Each descriptor
            # keeps its own rounding; no assumed common address alignment.
            physical_upper, descriptor_uppers = 0, []
            for access in accesses:
                owner = access.get("physical_owner", md.get("physical_owner"))
                raw = md.get("physical_memory_configs", {}).get(owner, md.get("physical_memory_config"))
                if id(raw) not in parsed:
                    parsed[id(raw)] = parse_execution_physical_config(raw)
                config = parsed[id(raw)]
                if isinstance(config, (DramConfig, SramConfig)):
                    quantum = config.burst_bytes
                elif isinstance(config, NandConfig):
                    quantum = config.page_bytes
                else:
                    raise ValueError("native_layer_future_MMU_physical_geometry_unknown")
                operation = access.get("operation")
                if operation not in {"read", "write"}:
                    raise ValueError("native_layer_future_MMU_physical_direction_unknown")
                count = reads if operation == "read" else writes
                if count not in backing_padded:
                    backing_padded[count] = _native_backing_payload_upper(count, backing_quanta)
                backing_bytes = backing_padded[count]
                rounded = ((backing_bytes+quantum-1)//quantum+1)*quantum if count else 0
                descriptor_upper = max(access["byte_count"], rounded)
                physical_upper += descriptor_upper
                descriptor_uppers.append(descriptor_upper)
            if accesses:
                physical_uppers[task.task_id] = tuple(descriptor_uppers)
            actual = observed_l2 if observed_l2 else actual_physical
            # Closed-interval cache writes are propagated even on hits.
            # Thus a positive native write descriptor keeps this stage in
            # the controller candidate population at every cache phase.
            positive = ("native_closed_interval_positive_write"
                if writes > 0 and any(access["operation"] == "write"
                    and access["byte_count"] > 0 for access in accesses) else False)
            if not (_context_sensitive(md) or ".attention." in str(md.get("operator_id", ""))):
                exact = "fixed"
            elif (l2_demands and model.get("model") in {"gpu_fused_attention_v3", "gpu_memory_roofline"}
                    and cache.get("reuse_factor") == 1.0 and cache.get("streaming_fraction") == 1.0
                    and observed_l2 == reads+(writes if gpu.cache_hierarchy.write_allocate else 0)):
                exact = "monotone"
            else:
                exact = None
            return actual, max(logical if l2_demands else 0, physical_upper), positive, exact

        candidates, actual_total, upper_total, positive_sources = [], 0, 0, {}
        lower_total = 0
        for stage_index, stage in enumerate(original):
            if stage.component_id != gpu_id:
                continue
            values = [task_bytes(task) for task in stage.execution_tasks]
            actual = sum(row[0] for row in values)
            for task_index, row in enumerate(values):
                key = (gpu_id, stage_index, task_index)
                maxima[key] = max(maxima.get(key, 0), row[1])
                if row[3] == "fixed":
                    observed.setdefault(key, set()).add(row[0])
                if row[3] is None or len(observed.get(key, ())) > 1:
                    # Unknown hit curve or a changed "fixed" payload: keep the
                    # full-payload upper and no lower bound for this Task.
                    minima[key], tight[key] = 0, None
                elif tight.get(key, 0) is not None:
                    minima[key] = min(minima.get(key, row[0]), row[0])
                    tight[key] = max(tight.get(key, 0), row[0])
            upper = sum(maxima[(gpu_id, stage_index, index)] if tight.get((gpu_id, stage_index, index)) is None
                        else tight[(gpu_id, stage_index, index)] for index in range(len(values)))
            lower_total += sum(minima[(gpu_id, stage_index, index)] for index in range(len(values)))
            if bool(actual) != bool(upper) or actual and not any(row[2] for row in values):
                raise ValueError("native_layer_future_MMU_candidate_population_not_proven")
            if actual:
                candidates.append(stage)
                actual_total += actual
                upper_total += upper
                for source in {row[2] for row in values if row[2]}:
                    positive_sources[source] = positive_sources.get(source, 0)+1
        ids = {stage.stage_id for stage in candidates}
        def original_dependencies(stage):
            return tuple(parent for dependency in stage.dependencies for parent in (
                controller_stages[dependency].dependencies if dependency in controller_stages else (dependency,)))
        roots = [stage for stage in candidates if not any(key in ids for key in original_dependencies(stage))]
        if not roots:
            continue
        maximum = (upper_total+len(roots)-1)//len(roots)
        minimum = lower_total//len(roots)
        mmu = controllers.mmu_tlb
        def service(count):
            pages = (count+mmu.page_size_bytes-1)//mmu.page_size_bytes
            batches = (pages+mmu.translation_batch_size-1)//mmu.translation_batch_size
            waves = (batches+mmu.max_outstanding_page_walks-1)//mmu.max_outstanding_page_walks
            return pages, waves*mmu.page_walk_latency_ns
        _, upper_ns = service(maximum)
        _, lower_ns = service(minimum)
        for index, root in enumerate(roots):
            task_id = "runtime.controller.{}.{}.root{:04d}.mmu_tlb".format(cohort.cohort_id, gpu_id, index)
            stage = controller_stages.get(task_id)
            count = actual_total//len(roots)+int(index < actual_total % len(roots))
            pages, expected_ns = service(count)
            if stage is None or len(stage.execution_tasks[0].demands) != 1:
                raise ValueError("native_layer_future_MMU_root_graph_not_native")
            demand = stage.execution_tasks[0].demands[0]
            if (demand.resource_id != gpu_id+".mmu_tlb" or demand.bytes_moved != count
                    or demand.work_units != float(pages) or demand.service_ns != expected_ns):
                raise ValueError("native_layer_future_MMU_root_service_not_native")
            if not lower_ns <= demand.service_ns <= upper_ns:
                raise ValueError("native_layer_future_MMU_payload_bounds_exclude_native_service")
            resolved[task_id] = {"resource_id": demand.resource_id, "lower_ns": lower_ns,
                "upper_ns": upper_ns, "translation_bytes_upper": maximum,
                "translation_bytes_lower": minimum}
        gpu_rows.append({"gpu_id": gpu_id, "native_root_count": len(roots),
            "observed_translation_bytes": actual_total, "full_payload_translation_bytes_upper": upper_total,
            "per_root_translation_bytes_upper": maximum, "per_root_service_upper_ns": upper_ns,
            "full_payload_translation_bytes_lower": lower_total,
            "per_root_translation_bytes_lower": minimum, "per_root_service_lower_ns": lower_ns,
            "native_backing_transaction_quantum_upper_bytes": backing_quantum,
            "native_backing_transaction_quanta_bytes": backing_quanta,
            "positive_candidate_stage_count": len(candidates),
            "positive_candidate_sources": positive_sources,
            "payload_extrema_composition": "sum_of_per_native_task_maxima_across_all_context_corners"})
    if set(resolved) != set(controller_stages):
        raise ValueError("native_layer_future_MMU_controller_population_not_native")
    return {"qualified": True, "task_bounds": resolved, "gpu_bounds": gpu_rows,
        "physical_access_byte_uppers": physical_uppers,
        "basis": "native_L2_closed_interval_full_payload_or_physical_descriptor_geometry; native_root_partition_and_page_batch_wave_formula",
        "physical_state_advanced": False}


__all__ = ["native_layer_boundary_bounds", "native_mmu_context_bounds"]
