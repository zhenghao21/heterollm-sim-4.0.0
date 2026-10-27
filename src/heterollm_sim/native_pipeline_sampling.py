"""Stage-level PP sampling helpers.

The live runtime is never advanced by these helpers.  A representative layer
window is executed on an independent physical-kernel copy; the returned stage
services are then composed by an explicit FIFO recurrence.  Missing layer
work is therefore an announced conditional assumption rather than a silently
truncated live task graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

from .native_pipeline_forecast import PipelineStageSample


_COMPUTE = re.compile(r"^operator-invocation-group-(\d+)\.stage(\d+)$")
_TRANSFER = re.compile(r"^operator-invocation-group-(\d+)\.pp(\d+)\.activation_transfer$")


class PipelineStageSamplingGuardError(ValueError):
    """The authored PP graph cannot provide a closed representative sample."""


@dataclass(frozen=True)
class PipelineFifoPoint:
    microbatch: int
    stage: int
    kind: str
    start_ns: float
    end_ns: float
    bubble_ns: float = 0.0


def _authored_stage_id(stage_id: str):
    match = _COMPUTE.match(stage_id)
    if match:
        return int(match.group(1)), int(match.group(2)), "compute"
    match = _TRANSFER.match(stage_id)
    if match:
        return int(match.group(1)), int(match.group(2)), "pipeline_activation_transfer"
    return None


def select_representative_pipeline_tasks(stages, specs):
    """Select first-layer work plus explicit activation edges per PP stage.

    The selected task graph is intentionally independent: omitted layer
    predecessors become declared boundary inputs in ``_execute_window``.  The
    stage FIFO and activation edge are restored by the pure composition step,
    which is the contract recorded in the forecast metadata.
    """
    authored = {}
    transfers = {}
    for stage in stages:
        parsed = _authored_stage_id(str(stage.stage_id))
        if parsed is None:
            continue
        group, number, kind = parsed
        (transfers if kind != "compute" else authored).setdefault((group, number), stage)
    if not authored:
        raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_no_authored_stages")
    groups = sorted({group for group, _number in authored})
    if groups != list(range(groups[-1] + 1)):
        raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_noncontiguous_microbatches")
    selected = []
    descriptors = []
    # With one pipeline microbatch the planner keeps the activation task in
    # the consumer stage instead of materializing a separate ``.ppNNNN``
    # stage.  Keep that authored task in the independent sample and expose a
    # synthetic transfer descriptor so the FIFO composer still sees the
    # producer-to-consumer edge.
    embedded_transfers = {}
    by_stage = {str(stage.stage_id): stage for stage in stages}
    for (group, number), stage in sorted(authored.items()):
        layers = sorted({task.metadata.get("layer_id") for task in stage.execution_tasks
                         if task.metadata.get("layer_id") is not None})
        if not layers:
            raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_missing_layer")
        # One real layer is enough to bound a repeated stage service.  The
        # independent budget checker compares this interval across holdouts.
        representative_layers = (layers[0], layers[-1]) if len(layers) > 1 else (layers[0],)
        ids = [task.task_id for task in specs
               if task.metadata.get("execution_stage_id") == stage.stage_id
               and task.metadata.get("layer_id") in representative_layers]
        if not ids:
            raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_layer_tasks_missing")
        selected.extend(ids)
        descriptors.append({"microbatch": group, "stage": number, "kind": "compute",
                            "layer_ids": tuple(str(layer) for layer in representative_layers),
                            "layer_count": len(layers),
                            "stage_id": stage.stage_id})
    # A single-microbatch plan keeps the transfer task in the consumer
    # stage's authored task list in some layouts, while other plans expose it
    # only in the flattened task specs.  Scan specs directly so both forms
    # produce the same synthetic producer-to-consumer edge.
    for spec in specs:
        metadata = spec.metadata
        if (metadata.get("event_kind") != "pipeline_activation"
                or type(metadata.get("source_stage")) is not int
                or type(metadata.get("target_stage")) is not int):
            continue
        stage_id = str(metadata.get("execution_stage_id", ""))
        match = re.match(r"^operator-invocation-group-(\d+)\.stage(\d+)$", stage_id)
        if match is None:
            continue
        group, target = int(match.group(1)), int(match.group(2))
        if target != metadata["target_stage"]:
            continue
        source = metadata["source_stage"]
        transfer_id = "operator-invocation-group-{:04d}.pp{:04d}.activation_transfer".format(
            group, source)
        transfer_key = (group, source)
        if transfer_key in embedded_transfers:
            continue
        selected.append(spec.task_id)
        embedded_transfers[transfer_key] = transfer_id
        descriptors.append({"microbatch": group, "stage": source,
                            "kind": "pipeline_activation_transfer", "stage_id": transfer_id,
                            "embedded_task_id": spec.task_id})
    for (group, number), stage in sorted(transfers.items()):
        ids = [task.task_id for task in specs if task.metadata.get("execution_stage_id") == stage.stage_id]
        if not ids:
            raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_transfer_tasks_missing")
        selected.extend(ids)
        descriptors.append({"microbatch": group, "stage": number,
                            "kind": "pipeline_activation_transfer", "stage_id": stage.stage_id})
    return tuple(dict.fromkeys(selected)), tuple(descriptors)


def stage_samples_from_window(row: Mapping[str, object], descriptors, *, cohort: int):
    """Turn one independent window into stage samples with explicit edges."""
    layer_start = row.get("layer_start_ns") or {}
    layer_end = row.get("layer_completion_ns") or {}
    stage_start = row.get("stage_start_ns") or {}
    stage_end = row.get("stage_completion_ns") or {}
    output = []
    for item in descriptors:
        stage_id = item["stage_id"]
        if item["kind"] == "compute":
            intervals = [(layer_start.get(layer), layer_end.get(layer))
                         for layer in item["layer_ids"]]
            if any(start is None or end is None or end <= start for start, end in intervals):
                raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_layer_interval_missing")
            # First/last representative layers bound the repeated stage
            # service. Middle layers remain a conditional model assumption;
            # the independent sample budget guards this interval.
            duration = sum(float(end - start) for start, end in intervals) / len(intervals)
            start = min(float(start) for start, _end in intervals)
            end = start + duration
        else:
            transfer_stages = row.get("pipeline_transfer_start_ns") or {}
            transfer_ends = row.get("pipeline_transfer_completion_ns") or {}
            start = transfer_stages.get(stage_id, stage_start.get(stage_id))
            end = transfer_ends.get(stage_id, stage_end.get(stage_id))
            if start is None or end is None or end <= start:
                raise PipelineStageSamplingGuardError("native_pipeline_stage_sampling_transfer_interval_missing")
            start, end = float(start), float(end)
        output.append(PipelineStageSample(int(cohort), int(item["microbatch"]), int(item["stage"]),
            start, end, kind=item["kind"], target_stage=(int(item["stage"]) + 1
                if item["kind"] == "pipeline_activation_transfer" else None),
            dependencies=(f"stage_fifo:{item['stage']}",)))
    return tuple(output)


def compose_fifo_points(stage_services: Mapping[int, float], transfer_services: Mapping[int, float],
                        microbatches: int, arrivals: Sequence[float] = ()):
    """Compose stage FIFO, activation transfers, and refill bubbles."""
    if microbatches < 1 or not stage_services:
        raise PipelineStageSamplingGuardError("native_pipeline_fifo_empty")
    stages = sorted(stage_services)
    available = {stage: 0.0 for stage in stages}
    points = []
    for mb in range(microbatches):
        previous = float(arrivals[mb]) if mb < len(arrivals) else 0.0
        for position, stage in enumerate(stages):
            start = max(previous, available[stage])
            end = start + float(stage_services[stage])
            points.append(PipelineFifoPoint(mb, stage, "compute", start, end,
                                            max(0.0, start - previous)))
            available[stage] = end
            previous = end
            if position + 1 < len(stages):
                transfer = float(transfer_services.get(stage, 0.0))
                if transfer <= 0 or not math.isfinite(transfer):
                    raise PipelineStageSamplingGuardError("native_pipeline_fifo_transfer_missing")
                end = previous + transfer
                points.append(PipelineFifoPoint(mb, stage, "pipeline_activation_transfer", previous,
                                                end, 0.0))
                previous = end
    return tuple(points)


__all__ = ["PipelineFifoPoint", "PipelineStageSamplingGuardError",
           "select_representative_pipeline_tasks", "stage_samples_from_window",
           "compose_fifo_points"]
