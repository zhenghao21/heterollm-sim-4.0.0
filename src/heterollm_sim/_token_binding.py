"""Resident dependency wiring for exact, live physical Token execution.

Only authored graph structure lives here.  Each bind uses the current cohort's
demands, metadata, allocation generations, and stage release times; controller
state and physical completion times are never memoized.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, Sequence, Tuple

from .contracts import TaskSpec
from .event_kernel import CompiledGraphLayout


@dataclass(frozen=True)
class AuthoredTokenProgram:
    """Validated position graph, independent of a cohort's task payloads."""

    compiled: CompiledGraphLayout
    suffixes: Tuple[str, ...]
    source_positions: Tuple[Tuple[int, int], ...]

    @classmethod
    def compile(cls, stages: Sequence[Any], build_specs: Callable) -> "AuthoredTokenProgram":
        namespace = "serving.program"
        ready = {stage.stage_id: 0.0 for stage in stages}
        specs, _sources = build_specs(stages, ready, namespace)
        prefix_length = len(namespace) + 1
        return cls(
            CompiledGraphLayout.compile(specs),
            tuple(task.task_id[prefix_length:] for task in specs),
            tuple((stage_position, task_position)
                  for stage_position, stage in enumerate(stages)
                  for task_position, _task in enumerate(stage.execution_tasks)),
        )

    def bind(
        self,
        stages: Sequence[Any],
        ready_by_stage: Mapping[str, float],
        namespace: str,
        bind_metadata: Callable,
    ) -> Tuple[TaskSpec, ...]:
        """Bind already validated prepared stages without rebuilding wiring."""

        ready_values = tuple(ready_by_stage[stage.stage_id] for stage in stages)
        for value in ready_values:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(float(value))):
                raise ValueError("earliest_start_ns must be a finite number")
            if value < 0.0:
                raise ValueError("earliest_start_ns must be non-negative")
        prefix = namespace + "."
        task_ids = tuple(prefix + suffix for suffix in self.suffixes)
        specs = []
        for position, (stage_position, task_position) in enumerate(self.source_positions):
            stage = stages[stage_position]
            task = stage.execution_tasks[task_position]
            # Prepared stages validated these fields at planner handoff.  Live
            # ResourceDemand constructors still validate all dynamic quantities.
            spec = object.__new__(TaskSpec)
            spec.__dict__.update(task_id=task_ids[position], request_id=task.request_ids[0],
                name=task.task_id, category=task.category, dependencies=tuple(
                    task_ids[parent] for parent in self.compiled.dependency_positions[position]),
                demands=task.demands, earliest_start_ns=ready_values[stage_position], marker=None,
                token_index=None, metadata={**bind_metadata(task.metadata, namespace),
                                           "execution_stage_id": stage.stage_id})
            specs.append(spec)
        return tuple(specs)
