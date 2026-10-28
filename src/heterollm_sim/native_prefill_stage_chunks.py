"""Multi-chunk PP Prefill pass from per-stage native layer blocks.

A prompt longer than one scheduler chunk is lowered as serial native
cohorts. Each chunk splits into its PP stage blocks (``_pipeline_boundary``);
one request is one microbatch, so a chunk's blocks run one after another.
The native runtime places a whole chunk before it schedules the next, whose
back-dated frontends (host and per-GPU prefix stages) wait on the lanes the
previous chunk still holds on their GPUs. Compute stage s of a chunk depends
on the frontends of the GPUs of several PP stages (s and s+1 in the planner
graph), so with PP >= 3 consecutive chunks overlap by stage. The pass is a
max-plus chain over (chunk, stage) (``_chained_chunks``):

    start(k, s) = max(end(k, s-1) + exit(k, s-1), end(k-1, s) + exit(k-1, s),
                      max over waited frontend stages d of end(k-1, d) + G_d)
    end(k, s)   = start(k, s) + body(k, s)

* chunk 0: every block's stationary runs on the real boundary; stage 0's
  body starts at the cohort boundary and includes the cohort prefix;
* G_d: from a chunk's last layer on PP stage d to the end of the next
  chunk's frontends on that stage's GPUs, and H: from a later chunk's first
  compute stage start to the end of its first layer (stage 0's entry).
  Both come from native two-phase transition windows
  (``_execute_transition_window``): every stage's final layers, then the
  graph's own prefix and first layer on the lanes they left. G is sampled
  on the graphs of chunk 0 and of the last full chunk; H on a full chunk's
  and, for a shorter last chunk, on that chunk's own graph;
* body(k, s), k >= 1: a layer whose native work is unchanged from chunk 0
  takes chunk 0's class step. A context-dependent layer (full attention
  over the whole earlier prompt) takes its step at chunk k's context from
  the line through its steps sampled at chunk 0, at a middle full chunk
  and at the last full chunk (declared inputs), widened by how far the
  middle step lies from the end-point line. A last chunk of another size is
  sampled on its own.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from statistics import fmean
import time

from .native_fast_window import NativeLayerWindowSample
from .native_layer_sampling import NativeLayerSamplingGuardError, _execute_transition_window
from .native_prefill_chunk_inputs import (DECLARED_CHUNK_ADDRESS_ASSUMPTION, DECLARED_CHUNK_STATE_ASSUMPTION,
                                          _description_runtime, _lower_description)
from .native_stationary import (HOLDOUT_SHIFT_NS, SAME_CLASS_ADDRESS_ASSUMPTION, _completions, _interval,
                                _pipeline_boundary, _pipeline_step, _pipeline_window, _widened_hull, _within,
                                sample_pipeline_blocks)

LINEAR_CONTEXT_ASSUMPTION = (
    "A layer whose native work changes with its chunk's context (attention over the earlier prompt) "
    "steps linearly in that context between the sampled contexts, never less than at the shorter one nor "
    "more than at the longer one; the middle sampled chunk's distance from the end-point line widens "
    "every interpolated chunk. A layer whose native work is unchanged takes chunk 0's class step.")
TRANSITION_ASSUMPTION = (
    "Each PP stage's gap from a chunk's last layer there to the end of the next chunk's frontends on its "
    "GPUs lies within the hull of the native transitions sampled on chunk 0's and the last full chunk's "
    "graphs (every stage's final layers followed by the graph's own prefix), widened by their "
    "difference; the next chunk's first layer after its compute stage start is sampled on a graph of "
    "its own size. Consecutive chunks chain by stage (max-plus) over these gaps.")
UNIFORM_CHUNK_ASSUMPTION = (
    "Every chunk but the last has the first chunk's token count, as the scheduler's fixed chunk size "
    "gives and every sampled chunk confirms.")


def _declared_chunk(runtime, first_cohort, index, context, control):
    """The ordinary native lowering of chunk ``index`` on a description copy (declared inputs)."""
    described = _description_runtime(runtime, control)
    state = described.states[first_cohort.items[0].request_id]
    state.prefill_cursor = context
    cohort = described._prefill_cohort([state], recompute=False)
    if cohort is None:
        raise NativeLayerSamplingGuardError("native_prefill_declared_chunk_cannot_be_admitted")
    cohort = replace(cohort, cohort_id=f"{first_cohort.cohort_id}.declared-chunk-{index:04d}")
    cohort, stages, ready = _lower_description(described, cohort)
    item = cohort.items[0]
    if item.context_tokens != context or item.token_count <= 0:
        raise NativeLayerSamplingGuardError("native_prefill_declared_chunk_not_contiguous")
    return described, cohort, stages, ready


def _layer_steps(prepared, parts):
    """Every block's later layers -> their (lower, upper) class step in ``_pipeline_step`` parts."""
    steps = {}
    for row in parts["pipeline_blocks"]:
        key = (row["microbatch"], row["stage"])
        classes = {item["class_key"]: (item["step_ns"]["lower"], item["step_ns"]["upper"])
                   for item in row["classes"]}
        steps[key] = {layer["layer_id"]: classes[layer["class_key"]]
                      for layer in prepared["pipeline"]["blocks"][key]["inventory"][1:]}
    return steps


def _context_layers(first, other):
    """Per block, the layers whose native class differs between two chunks of one size."""
    changed = {}
    for key, block in first["pipeline"]["blocks"].items():
        mine = {row["layer_id"]: row["class_key"] for row in block["inventory"]}
        theirs = {row["layer_id"]: row["class_key"] for row in other["pipeline"]["blocks"][key]["inventory"]}
        if list(mine) != list(theirs):
            raise NativeLayerSamplingGuardError("native_prefill_chunk_block_layers_differ")
        layers = [layer for layer in mine if mine[layer] != theirs[layer]]
        if block["inventory"][0]["layer_id"] in layers:
            # A block's first layer is inside its entry, sampled on chunk 0 only.
            raise NativeLayerSamplingGuardError("native_prefill_chunk_context_dependent_block_entry")
        changed[key] = layers
    return changed


def _growth_runs(block, dependent):
    """Two contiguous runs that step every context-dependent class of ``block``:
    forward from the first dependent layer's predecessor, back from the last one."""
    order = [row["layer_id"] for row in block["inventory"]]
    classes = {row["layer_id"]: row["class_key"] for row in block["inventory"]}
    wanted = {classes[layer] for layer in dependent}
    start, end = order.index(dependent[0]) - 1, order.index(dependent[0])
    while not wanted <= {classes[layer] for layer in order[start + 1:end + 1]}:
        end += 1
    forward = order[start:end + 1]
    start, end = order.index(dependent[-1]) - 1, order.index(dependent[-1])
    while not wanted <= {classes[layer] for layer in order[start + 1:end + 1]}:
        start -= 1
    if start < 0:
        raise NativeLayerSamplingGuardError("native_prefill_chunk_context_layers_not_stepped")
    return forward, order[start:end + 1]


def _class_hull(block, views):
    """Each stepped class's (lower, upper) step over ``views``, widened as ``compose_layers``."""
    classes = {row["layer_id"]: row["class_key"] for row in block["inventory"]}
    steps, widths = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for name, view in views.items():
        times = _completions(view, view["layer_ids"])
        for layer, before, after in zip(view["layer_ids"][1:], times, times[1:]):
            steps[classes[layer]][name].append(after - before)
            widths[classes[layer]].append(view["first_access_ns"].get(layer, 0.))
    hull = {}
    for key, runs in steps.items():
        values = [value for run in runs.values() for value in run]
        means = [fmean(run) for run in runs.values()]
        lower, upper = _widened_hull(values, widths[key], max(means) - min(means))
        hull[key] = (max(0., lower), upper)
    return hull


def _linear_steps(points, contexts):
    """(lower, upper) of one layer's step at each of ``contexts`` from its sampled
    ``points`` [(context, (lower, upper))], and the middle point's line residual.

    Between two sampled contexts the step follows their bounds' line widened
    by the residual, and stays within those two contexts' bounds: attention
    over a longer context never does less native work.
    """
    points = sorted(points)
    residual = 0.
    if len(points) >= 3:
        (c0, a), (cm, m), (cl, b) = points[0], points[len(points) // 2], points[-1]
        mid = lambda row: (row[0] + row[1]) / 2
        residual = abs(mid(a) + (mid(b) - mid(a)) * (cm - c0) / (cl - c0) - mid(m))
    sampled = dict(points)
    rows = []
    for context in contexts:
        if context in sampled:
            rows.append(sampled[context])
            continue
        (ca, a), (cb, b) = next((left, right) for left, right in zip(points, points[1:])
                                if left[0] < context < right[0])
        share = (context - ca) / (cb - ca)
        rows.append((max(a[0], a[0] + (b[0] - a[0]) * share - residual),
                     min(b[1], a[1] + (b[1] - a[1]) * share + residual)))
    return rows, residual


def _chained_chunks(chunks, waits, gaps, bound):
    """End of the last chunk's last layer after chunk 0's boundary (max-plus).

    Chunk k's stage s starts behind its own stage s-1, behind chunk k-1's
    stage s, and once the frontends it waits for (``waits[s]``: their PP
    stages) are done; a stage's frontends finish ``gaps[stage]`` after the
    previous chunk's last layer on that stage. ``bound`` 0 or 1 picks every
    interval's lower or upper end; the chain is monotone in each.
    """
    layers, exits = None, None
    for stages in chunks:
        ends, done, ready = [], [], 0.
        for stage, row in enumerate(stages):
            start = ready
            if layers is not None:
                start = max(start, exits[stage], *(layers[other] + gaps[other][bound] for other in waits[stage]))
            end = start + row["body"][bound]
            ends.append(end)
            ready = end + row["exit"][bound]
            done.append(ready)
        layers, exits = ends, done
    return layers[-1]


def sample_pipeline_prefill_chunks(runtime, cohort, stages, request_ready_ns, control, options, records,
                                   *, holdout_shift_ns=HOLDOUT_SHIFT_NS):
    """Sample and compose a multi-chunk PP Prefill pass; the source kernel never advances."""
    started = time.perf_counter()
    target = options.target_relative_error
    limit = target / (2 + target)
    item = cohort.items[0]
    state = runtime.states[item.request_id]
    if (cohort.kind != "prefill" or len(cohort.items) != 1 or item.context_tokens or state.prefill_cursor
            or runtime.batches):
        raise NativeLayerSamplingGuardError("native_prefill_chunk_inputs_require_first_real_boundary")
    if runtime._external_executor is not None or runtime.residency_manager is not None:
        raise NativeLayerSamplingGuardError("native_prefill_chunk_inputs_require_unshared_native_boundary")
    size, total = item.token_count, state.spec.prompt_tokens
    count = math.ceil(total / size)
    full = total // size
    layers = (options.calibration_layers, options.holdout_layers)
    symmetry = (runtime.plan.scenario, options.tp_rank_symmetry)
    lowered = {}

    def boundary(index):
        if index == 0:
            active, chunk, chunk_stages, ready = runtime, cohort, stages, request_ready_ns
        else:
            active, chunk, chunk_stages, ready = _declared_chunk(runtime, cohort, index, index * size, control)
        expected = min(size, total - index * size)
        if chunk.items[0].token_count != expected:
            raise NativeLayerSamplingGuardError("native_prefill_chunk_sizes_not_uniform")
        prepared = _pipeline_boundary(active, chunk, chunk_stages, ready, f"native-prefill-chunk{index:04d}.",
                                      layers, final_exit=False)
        prepared["tp_symmetry"] = symmetry
        lowered[index] = prepared
        return prepared

    def full_pass(prepared, index):
        runs, windows = sample_pipeline_blocks(prepared, control, records, prepared["origin"],
                                               prepared["origin"] + holdout_shift_ns, f"prefill-pp-c{index}-")
        lower, upper, parts = _pipeline_step(prepared, {"runs": runs})
        stages = []
        for row in sorted(parts["pipeline_blocks"], key=lambda row: row["stage"]):
            exit = tuple(row.get("exit_interval_ns") or (0., 0.))
            service = row["service_ns"]
            stages.append({"body": (service["lower"] - exit[0], service["upper"] - exit[1]), "exit": exit,
                           "entry": tuple(row["entry_interval_ns"])})
        return {"pass_ns": (lower, upper), "stages": stages, "parts": parts, "windows": windows}

    def transition(prepared, index):
        pipeline = prepared["pipeline"]
        blocks = [pipeline["blocks"][(0, stage)] for stage in range(pipeline["stages"])]
        last = [block["inventory"][-1]["layer_id"] for block in blocks]
        tail = [identity for block in blocks for row in block["inventory"][-2:]
                for identity in block["by_layer"][row["layer_id"]]]
        tail += [identity for block in blocks for identity in block["after_last_layer"]]
        head_block = blocks[0]
        first = head_block["inventory"][0]["layer_id"]
        head = [*pipeline["cohort_prefix"], *head_block["prefix"], *head_block["by_layer"][first]]
        terms = {**{identity: "tail" for identity in tail}, **{identity: "entry" for identity in head}}
        record = _execute_transition_window(prepared["source"], prepared["specs"], tail, head,
            origin_ns=prepared["origin"], control=control, sample_id=f"prefill-pp-c{index}-transition",
            row_terms=terms)
        chunk = prepared["cohort"]
        records.append(NativeLayerWindowSample(window_id=record["sample_id"], role="chunk_transition",
            cohort_id=chunk.cohort_id, cohort_kind=chunk.kind, decode_step=0,
            layer_ids=(*(row["layer_id"] for block in blocks for row in block["inventory"][-2:]), first),
            rank_ids=tuple(sorted({prepared["ranks"][identity] for identity in (*tail, *head)
                                   if identity in prepared["ranks"]})),
            context_tokens_by_request=prepared["context"], executed_task_count=record["task_count"],
            start_ns=record["tail"]["start_ns"], end_ns=record["head"]["end_ns"],
            wall_seconds=record["wall_seconds"], physical_observations=record["physical_observations"]))
        control.report("native_stationary_sample", len(records), unit="work_units",
            message="原生分段过渡样本已保存，不计完整批次或输出词元",
            metadata={"sample_id": record["sample_id"], "role": "chunk_transition",
                      "executed_tasks": record["task_count"]})
        widths = record["first_access_ns"] or {}
        ends, done = record["tail"]["layer_completion_ns"], record["head"]["stage_completion_ns"]
        # Each stage's frontends finish this long after its tail's last layer.
        gaps = defaultdict(lambda: -math.inf)
        for stage_id, stage in pipeline["prefix_stage_pp"].items():
            if stage is not None:
                gaps[stage] = max(gaps[stage], done[stage_id] - ends[last[stage]])
        start = record["head"]["stage_start_ns"][head_block["stage_id"]]
        return {"frontend_gap_ns": dict(gaps), "head_ns": record["head"]["layer_completion_ns"][first] - start,
                "tail_width_ns": widths.get("tail", 0.), "head_width_ns": widths.get("entry", 0.),
                "wall_seconds": record["wall_seconds"]}

    first = boundary(0)
    if any(block["transfer_stage"] is not None for block in first["pipeline"]["blocks"].values()):
        raise NativeLayerSamplingGuardError("native_prefill_chunk_transfer_stage_unsupported")
    zero = full_pass(first, 0)
    depth = first["pipeline"]["stages"]
    lower, upper = zero["pass_ns"]
    budget = {"target_relative_error": target, "effective_independent_target": limit, "prompt_tokens": total,
              "chunk_tokens": size, "chunk_count": count, "chunk0": {"pass_ns": zero["pass_ns"],
                                                                     "composition": zero["parts"]}}
    if count > 1:
        last_full = full - 1
        sampled = sorted({last_full // 2, last_full} - {0})
        steps0 = _layer_steps(first, zero["parts"])
        growth, dependent = {}, None
        for index in sampled:
            prepared = boundary(index)
            changed = _context_layers(first, prepared)
            if dependent is None:
                dependent = changed
            elif changed != dependent:
                raise NativeLayerSamplingGuardError("native_prefill_chunk_context_layers_inconsistent")
            growth[index] = {}
            for key, block in sorted(prepared["pipeline"]["blocks"].items()):
                if not dependent[key]:
                    continue
                runs = _growth_runs(block, dependent[key])
                views = {}
                for name, layer_ids, origin in zip(("first", "last"), runs,
                                                   (prepared["origin"], prepared["origin"] + holdout_shift_ns)):
                    window = _pipeline_window(prepared, control, records, "context_growth", [key], origin,
                        f"prefill-pp-c{index}-growth-{name}-{key[1]}", layers={key: layer_ids})
                    views[name] = window["views"][key]
                hull = _class_hull(block, views)
                classes = {row["layer_id"]: row["class_key"] for row in block["inventory"]}
                growth[index][key] = {layer: hull[classes[layer]] for layer in dependent[key]}
        dependent = dependent or {key: [] for key in first["pipeline"]["blocks"]}
        # Each stage's step shift from chunk 0 at every full chunk k >= 1.
        contexts = [index * size for index in range(1, full)]
        shifts = [[[0., 0.] for _ in range(depth)] for _ in contexts]
        linear = []
        for key, layer_ids in sorted(dependent.items()):
            for layer in layer_ids:
                points = [(0, steps0[key][layer])] + [(index * size, growth[index][key][layer]) for index in sampled]
                rows, residual = _linear_steps(points, contexts)
                for position, (low, high) in enumerate(rows):
                    shifts[position][key[1]][0] += low - steps0[key][layer][0]
                    shifts[position][key[1]][1] += high - steps0[key][layer][1]
                linear.append({"block": list(key), "layer_id": layer, "points_ns": points,
                               "line_residual_ns": residual})
        # Transitions on chunk 0's and the last full chunk's graphs.
        moves = {0: transition(first, 0)}
        if last_full:
            moves[last_full] = transition(lowered[last_full], last_full)
        gaps = {}
        for stage in moves[0]["frontend_gap_ns"]:
            values = [row["frontend_gap_ns"][stage] for row in moves.values()]
            gaps[stage] = _widened_hull(values, [row["tail_width_ns"] for row in moves.values()],
                                        max(values) - min(values))
        heads = [row["head_ns"] for row in moves.values()]
        head = _widened_hull(heads, [row["head_width_ns"] for row in moves.values()], max(heads) - min(heads))

        def chunk_stages(stages, head_ns, shift):
            rows = []
            for stage, row in enumerate(stages):
                body = [row["body"][0] + shift[stage][0], row["body"][1] + shift[stage][1]]
                if stage == 0:
                    # A later chunk enters its first layer from its compute stage start.
                    body = [body[0] - row["entry"][0] + head_ns[0], body[1] - row["entry"][1] + head_ns[1]]
                rows.append({"body": tuple(body), "exit": row["exit"]})
            return rows

        chunks = [[{"body": row["body"], "exit": row["exit"]} for row in zero["stages"]]]
        chunks += [chunk_stages(zero["stages"], head, shift) for shift in shifts]
        if total % size:
            tail_chunk = full_pass(boundary(count - 1), count - 1)
            short = transition(lowered[count - 1], count - 1)
            width = short["head_width_ns"]
            chunks.append(chunk_stages(tail_chunk["stages"], (short["head_ns"] - width, short["head_ns"] + width),
                                       [[0., 0.]] * depth))
            budget["last_chunk"] = {"pass_ns": tail_chunk["pass_ns"], "composition": tail_chunk["parts"],
                                    "transition": short}
        waits = [sorted({first["pipeline"]["prefix_stage_pp"][stage]
                         for stage in first["pipeline"]["blocks"][(0, index)]["prefix_stage_waits"]} - {None})
                 for index in range(depth)]
        if not waits[0] or any(stage not in gaps for row in waits for stage in row):
            raise NativeLayerSamplingGuardError("native_prefill_chunk_frontend_waits_not_sampled")
        lower, upper = (_chained_chunks(chunks, waits, gaps, bound) for bound in (0, 1))
        budget.update(sampled_full_chunks=sampled, context_dependent_layers=linear, transitions=moves,
            frontend_gap_ns={stage: list(row) for stage, row in gaps.items()}, transition_head_ns=list(head),
            stage_frontend_waits=waits, chunk_stages_ns=chunks)
    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {}, "error_budget": budget}
    budget.update(basis="per_stage_block_samples_per_chunk; context_linear_layer_steps; "
                        "native_two_phase_chunk_transitions; per_stage_chunk_chain",
        assumptions=[SAME_CLASS_ADDRESS_ASSUMPTION, LINEAR_CONTEXT_ASSUMPTION, TRANSITION_ASSUMPTION,
                     UNIFORM_CHUNK_ASSUMPTION, DECLARED_CHUNK_STATE_ASSUMPTION, DECLARED_CHUNK_ADDRESS_ASSUMPTION],
        sampled_layer_window_count=len(records), sample_wall_seconds=time.perf_counter() - started,
        physical_full_cohorts_committed=0, physical_full_prompt_executed=False)
    if lower <= 0:
        result["reasons"].append("native_prefill_pass_lower_bound_not_positive")
        return result
    row = _interval(lower, upper)
    result["metric_bounds"] = {"prefill_pass_ns": row}
    if not _within(row, limit):
        result["reasons"].append("native_prefill_independent_budget_exceeded")
        return result
    result.update(accepted=True, metrics={"prefill_pass_ns": row["estimate"]})
    return result


__all__ = ["sample_pipeline_prefill_chunks", "LINEAR_CONTEXT_ASSUMPTION", "TRANSITION_ASSUMPTION"]
