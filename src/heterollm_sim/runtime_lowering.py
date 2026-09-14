"""Closed action dispatcher and aggregate control-plane task lowering."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .runtime_ir import (
    ControllerKind,
    ControllerTransactionBatch,
    ExpansionResult,
    HardwareTask,
    InstructionBatch,
    InstructionClass,
    KernelCompletion,
    RuntimeAction,
    RuntimeDelta,
    RuntimePhase,
    RuntimeTask,
)
from .runtime_state import RuntimeState


class RuntimeDispatchError(ValueError):
    """A closed action could not be resolved from the supplied data context."""


_ACTION_PHASE = {
    RuntimeAction.CAPACITY_CHECK: RuntimePhase.CAPACITY_CHECK,
    RuntimeAction.PLACEMENT_DECISION: RuntimePhase.PLACEMENT_DECISION,
    RuntimeAction.ALLOCATE: RuntimePhase.ALLOCATION,
    RuntimeAction.WEIGHT_CACHE_LOOKUP: RuntimePhase.CACHE,
    RuntimeAction.PAGE_CACHE_LOOKUP: RuntimePhase.CACHE,
    RuntimeAction.NVME_READ: RuntimePhase.STORAGE,
    RuntimeAction.IOMMU_TRANSLATE: RuntimePhase.ADDRESS_TRANSLATION,
    RuntimeAction.DMA_MAP: RuntimePhase.ADDRESS_TRANSLATION,
    RuntimeAction.PCIE_TRANSFER: RuntimePhase.TRANSFER,
    RuntimeAction.BATCH_SCHEDULE: RuntimePhase.SCHEDULING,
    RuntimeAction.OPERATOR_SCHEDULE: RuntimePhase.SCHEDULING,
    RuntimeAction.COMMAND_BUILD: RuntimePhase.COMMAND,
    RuntimeAction.COMMAND_SUBMIT: RuntimePhase.COMMAND,
    RuntimeAction.GPU_COMMAND_PROCESS: RuntimePhase.GPU_FRONTEND,
    RuntimeAction.MMU_TLB_LOOKUP: RuntimePhase.MEMORY,
    RuntimeAction.L2_LOOKUP: RuntimePhase.MEMORY,
    RuntimeAction.VRAM_CONTROLLER: RuntimePhase.MEMORY,
    RuntimeAction.COMPUTE: RuntimePhase.COMPUTE,
    RuntimeAction.INTERRUPT: RuntimePhase.INTERRUPT,
    RuntimeAction.CPU_COMPLETE: RuntimePhase.CPU_COMPLETION,
    RuntimeAction.NOOP: RuntimePhase.COMPUTE,
}

_ACTION_CONTROLLER = {
    RuntimeAction.CAPACITY_CHECK: ControllerKind.CPU,
    RuntimeAction.PLACEMENT_DECISION: ControllerKind.CPU,
    RuntimeAction.ALLOCATE: ControllerKind.ALLOCATOR,
    RuntimeAction.WEIGHT_CACHE_LOOKUP: ControllerKind.WEIGHT_CACHE,
    RuntimeAction.PAGE_CACHE_LOOKUP: ControllerKind.PAGE_CACHE,
    RuntimeAction.NVME_READ: ControllerKind.NVME,
    RuntimeAction.IOMMU_TRANSLATE: ControllerKind.IOMMU,
    RuntimeAction.DMA_MAP: ControllerKind.DMA,
    RuntimeAction.PCIE_TRANSFER: ControllerKind.PCIE,
    RuntimeAction.BATCH_SCHEDULE: ControllerKind.SCHEDULER,
    RuntimeAction.OPERATOR_SCHEDULE: ControllerKind.SCHEDULER,
    RuntimeAction.COMMAND_BUILD: ControllerKind.COMMAND_BUILDER,
    RuntimeAction.COMMAND_SUBMIT: ControllerKind.CPU,
    RuntimeAction.GPU_COMMAND_PROCESS: ControllerKind.GPU_COMMAND_PROCESSOR,
    RuntimeAction.MMU_TLB_LOOKUP: ControllerKind.MMU_TLB,
    RuntimeAction.L2_LOOKUP: ControllerKind.L2,
    RuntimeAction.VRAM_CONTROLLER: ControllerKind.VRAM,
    RuntimeAction.COMPUTE: ControllerKind.GPU_COMMAND_PROCESSOR,
    RuntimeAction.INTERRUPT: ControllerKind.INTERRUPT,
    RuntimeAction.CPU_COMPLETE: ControllerKind.CPU,
    RuntimeAction.NOOP: ControllerKind.CPU,
}

_ACTION_INSTRUCTION_CLASS = {
    RuntimeAction.NVME_READ: InstructionClass.MEMORY,
    RuntimeAction.IOMMU_TRANSLATE: InstructionClass.DMA,
    RuntimeAction.DMA_MAP: InstructionClass.DMA,
    RuntimeAction.PCIE_TRANSFER: InstructionClass.DMA,
    RuntimeAction.COMMAND_BUILD: InstructionClass.GPU_COMMAND,
    RuntimeAction.COMMAND_SUBMIT: InstructionClass.GPU_COMMAND,
    RuntimeAction.GPU_COMMAND_PROCESS: InstructionClass.GPU_COMMAND,
    RuntimeAction.MMU_TLB_LOOKUP: InstructionClass.MEMORY,
    RuntimeAction.L2_LOOKUP: InstructionClass.MEMORY,
    RuntimeAction.VRAM_CONTROLLER: InstructionClass.MEMORY,
    RuntimeAction.COMPUTE: InstructionClass.COMPUTE,
    RuntimeAction.INTERRUPT: InstructionClass.INTERRUPT,
}

_DEFAULT_RESOURCES = {
    RuntimeAction.CAPACITY_CHECK: ("cpu.capacity",),
    RuntimeAction.PLACEMENT_DECISION: ("cpu.placement",),
    RuntimeAction.ALLOCATE: ("allocator",),
    RuntimeAction.WEIGHT_CACHE_LOOKUP: ("weight_cache",),
    RuntimeAction.PAGE_CACHE_LOOKUP: ("page_cache",),
    RuntimeAction.NVME_READ: ("nvme",),
    RuntimeAction.IOMMU_TRANSLATE: ("iommu",),
    RuntimeAction.DMA_MAP: ("dma",),
    RuntimeAction.PCIE_TRANSFER: ("pcie",),
    RuntimeAction.BATCH_SCHEDULE: ("cpu.batch_scheduler",),
    RuntimeAction.OPERATOR_SCHEDULE: ("cpu.operator_scheduler",),
    RuntimeAction.COMMAND_BUILD: ("cpu.command_builder",),
    RuntimeAction.COMMAND_SUBMIT: ("cpu.command_submit",),
    RuntimeAction.GPU_COMMAND_PROCESS: ("gpu.command_processor",),
    RuntimeAction.MMU_TLB_LOOKUP: ("gpu.mmu_tlb",),
    RuntimeAction.L2_LOOKUP: ("gpu.l2",),
    RuntimeAction.VRAM_CONTROLLER: ("gpu.vram_controller",),
    RuntimeAction.COMPUTE: ("gpu.compute",),
    RuntimeAction.INTERRUPT: ("interrupt",),
    RuntimeAction.CPU_COMPLETE: ("cpu.completion",),
    RuntimeAction.NOOP: (),
}


_WEIGHT_CACHE_FILL_ON_PCIE = "weight_cache_fill_on_pcie"


def _reject_callbacks(value: object, path: str = "context") -> None:
    if callable(value):
        raise TypeError("{} must contain data, not callbacks".format(path))
    if value is None or isinstance(value, (str, bytes, bool, int, TaskSpec)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} must contain finite numbers".format(path))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_callbacks(item, "{}.{}".format(path, key))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_callbacks(item, "{}[{}]".format(path, index))
        return
    raise TypeError("{} contains unsupported type {}".format(path, type(value).__name__))


def _non_negative_int(value: object, field_name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeDispatchError("{} must be a non-negative integer".format(field_name))
    return value


def _merge_metric(delta: RuntimeDelta, action: RuntimeAction) -> RuntimeDelta:
    metrics = dict(delta.metric_increments)
    metric_id = "action.{}.count".format(action.value)
    metrics[metric_id] = metrics.get(metric_id, 0.0) + 1.0
    return replace(delta, metric_increments=metrics)


class RuntimeDispatcher:
    """Dispatch ``RuntimeAction`` with an explicit closed ``if`` chain."""

    def __init__(self, profile: Optional[Mapping[str, object]] = None) -> None:
        self.profile = dict(profile or {})
        _reject_callbacks(self.profile, "profile")

    def _service_ns(
        self,
        action: RuntimeAction,
        payload: Optional[Mapping[str, object]] = None,
    ) -> float:
        overrides = (payload or {}).get("service_ns_by_action", {})
        if overrides and not isinstance(overrides, Mapping):
            raise RuntimeDispatchError("service_ns_by_action must be a mapping")
        configured = self.profile.get("service_ns", {})
        if not isinstance(configured, Mapping):
            raise RuntimeDispatchError("profile.service_ns must be a mapping")
        raw = (
            overrides.get(action.value, overrides.get(action))
            if isinstance(overrides, Mapping)
            else None
        )
        if raw is None:
            raw = configured.get(action.value, configured.get(action, 1.0))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeDispatchError("service time must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise RuntimeDispatchError("service time must be finite and non-negative")
        return value

    def _resources(
        self,
        action: RuntimeAction,
        payload: Optional[Mapping[str, object]] = None,
    ) -> Tuple[str, ...]:
        overrides = (payload or {}).get("resources_by_action", {})
        if overrides and not isinstance(overrides, Mapping):
            raise RuntimeDispatchError("resources_by_action must be a mapping")
        configured = self.profile.get("resources", {})
        if not isinstance(configured, Mapping):
            raise RuntimeDispatchError("profile.resources must be a mapping")
        raw = (
            overrides.get(action.value, overrides.get(action))
            if isinstance(overrides, Mapping)
            else None
        )
        if raw is None:
            raw = configured.get(action.value, configured.get(action))
        if raw is None:
            return _DEFAULT_RESOURCES[action]
        values = (raw,) if isinstance(raw, str) else tuple(raw)
        if any(not isinstance(value, str) or not value for value in values):
            raise RuntimeDispatchError("resource ids must be non-empty strings")
        if len(values) != len(set(values)):
            raise RuntimeDispatchError("an action may demand each resource once")
        return values

    def _transaction_count(
        self,
        action: RuntimeAction,
        payload: Mapping[str, object],
    ) -> int:
        configured = self.profile.get("transaction_count", {})
        if not isinstance(configured, Mapping):
            raise RuntimeDispatchError(
                "profile.transaction_count must be a mapping"
            )
        overrides = payload.get("transaction_count_by_action", {})
        if overrides and not isinstance(overrides, Mapping):
            raise RuntimeDispatchError(
                "transaction_count_by_action must be a mapping"
            )
        raw = (
            overrides.get(action.value, overrides.get(action))
            if isinstance(overrides, Mapping)
            else None
        )
        if raw is None:
            raw = configured.get(action.value, configured.get(action))
        if raw is None:
            raw = payload.get("transaction_count")
        return max(
            1,
            _non_negative_int(raw, "transaction_count", 1),
        )

    def _make_task(
        self,
        completion: KernelCompletion,
        action: RuntimeAction,
        payload: Mapping[str, object],
        *,
        dependencies: Optional[Sequence[str]] = None,
        sequence: Optional[int] = None,
        task_suffix: Optional[str] = None,
    ) -> HardwareTask:
        sequence_value = completion.sequence + 1 if sequence is None else sequence
        task_id = "{}>{:04d}:{}".format(
            completion.task.task_id,
            sequence_value,
            action.value,
        )
        if task_suffix:
            task_id = "{}:{}".format(task_id, task_suffix)
        byte_count = _non_negative_int(payload.get("byte_count"), "byte_count")
        instruction_count = max(
            1,
            _non_negative_int(
                payload.get("instruction_count"),
                "instruction_count",
                1,
            ),
        )
        operation_count = _non_negative_int(
            payload.get("operation_count"),
            "operation_count",
        )
        instruction_batch = InstructionBatch(
            batch_id="{}.instructions".format(task_id),
            instruction_class=_ACTION_INSTRUCTION_CLASS.get(
                action,
                InstructionClass.CONTROL,
            ),
            instruction_count=instruction_count,
            operation_count=operation_count,
            byte_count=byte_count,
        )
        controller_batch = ControllerTransactionBatch(
            batch_id="{}.transactions".format(task_id),
            controller=_ACTION_CONTROLLER[action],
            transaction_count=self._transaction_count(action, payload),
            byte_count=byte_count,
        )
        service_ns = self._service_ns(action, payload)
        demands = tuple(
            ResourceDemand(
                resource_id=resource_id,
                service_ns=service_ns,
                bytes_moved=byte_count,
                work_units=float(operation_count),
            )
            for resource_id in self._resources(action, payload)
        )
        category = TaskCategory.COMPUTE if action is RuntimeAction.COMPUTE else (
            TaskCategory.COMMUNICATION
            if action in {
                RuntimeAction.DMA_MAP,
                RuntimeAction.PCIE_TRANSFER,
                RuntimeAction.INTERRUPT,
            }
            else TaskCategory.MEMORY
            if action in {
                RuntimeAction.WEIGHT_CACHE_LOOKUP,
                RuntimeAction.PAGE_CACHE_LOOKUP,
                RuntimeAction.NVME_READ,
                RuntimeAction.MMU_TLB_LOOKUP,
                RuntimeAction.L2_LOOKUP,
                RuntimeAction.VRAM_CONTROLLER,
            }
            else TaskCategory.POLICY
        )
        return HardwareTask(
            task_id=task_id,
            request_id=completion.task.request_id,
            name=action.value,
            action=action,
            phase=int(_ACTION_PHASE[action]),
            sequence=sequence_value,
            category=category,
            dependencies=tuple(
                dependencies
                if dependencies is not None
                else (completion.task.task_id,)
            ),
            demands=demands,
            payload=dict(payload),
            instruction_batch=instruction_batch,
            controller_batch=controller_batch,
        )

    def _result(
        self,
        completion: KernelCompletion,
        tasks: Iterable[RuntimeTask] = (),
        delta: Optional[RuntimeDelta] = None,
    ) -> ExpansionResult:
        return ExpansionResult(
            tasks=tuple(tasks),
            delta=_merge_metric(delta or RuntimeDelta(), completion.action),
        )

    @staticmethod
    def _placement(
        completion: KernelCompletion,
        state: RuntimeState,
        context: Mapping[str, object],
    ) -> str:
        payload = completion.payload
        flow_id = str(payload.get("flow_id", completion.task.request_id))
        for key in (flow_id, completion.task.request_id):
            value = state.placement_decisions.get(key)
            if value:
                return value
        decisions = context.get("placement_decisions", {})
        if decisions and not isinstance(decisions, Mapping):
            raise RuntimeDispatchError("context.placement_decisions must be a mapping")
        if isinstance(decisions, Mapping):
            for key in (flow_id, completion.task.request_id):
                value = decisions.get(key)
                if isinstance(value, str) and value:
                    return value
        for value in (payload.get("placement"), context.get("placement")):
            if isinstance(value, str) and value:
                return value
        raise RuntimeDispatchError(
            "placement decision is required for {}".format(flow_id)
        )

    @staticmethod
    def _continuations(
        completion: KernelCompletion,
        context: Mapping[str, object],
    ) -> Tuple[TaskSpec, ...]:
        configured = context.get("continuations", ())
        if isinstance(configured, Mapping):
            flow_id = str(
                completion.payload.get("flow_id", completion.task.request_id)
            )
            configured = configured.get(
                flow_id,
                configured.get(completion.task.request_id, ()),
            )
        values = tuple(configured or ())
        if any(not isinstance(task, TaskSpec) for task in values):
            raise RuntimeDispatchError("continuations must contain TaskSpec values")
        return values

    def dispatch(
        self,
        completion: KernelCompletion,
        state: RuntimeState,
        context: Mapping[str, object],
    ) -> ExpansionResult:
        """Expand one completion using only the enumerated action set."""

        if not isinstance(completion, KernelCompletion):
            raise TypeError("completion must be a KernelCompletion")
        if not isinstance(state, RuntimeState):
            raise TypeError("state must be a RuntimeState")
        if not isinstance(context, Mapping):
            raise TypeError("runtime context must be a mapping")
        _reject_callbacks(context)

        action = completion.action
        payload = dict(completion.payload)
        flow_id = str(payload.get("flow_id", completion.task.request_id))
        required_bytes = _non_negative_int(
            payload.get("required_bytes", payload.get("byte_count", 0)),
            "required_bytes",
        )

        if action is RuntimeAction.NOOP:
            return self._result(completion)

        if action is RuntimeAction.CAPACITY_CHECK:
            resource_id = str(payload.get("capacity_resource", "vram"))
            available = state.available_bytes(resource_id)
            if available is not None and required_bytes > available:
                raise RuntimeDispatchError(
                    "insufficient {} capacity: need {}, available {}".format(
                        resource_id,
                        required_bytes,
                        available,
                    )
                )
            next_task = self._make_task(
                completion,
                RuntimeAction.PLACEMENT_DECISION,
                payload,
            )
            return self._result(
                completion,
                (next_task,),
                RuntimeDelta(capacity_reservations={resource_id: required_bytes}),
            )

        if action is RuntimeAction.PLACEMENT_DECISION:
            placement = self._placement(completion, state, context)
            payload["placement"] = placement
            next_task = self._make_task(
                completion,
                RuntimeAction.ALLOCATE,
                payload,
            )
            return self._result(
                completion,
                (next_task,),
                RuntimeDelta(placement_decisions={flow_id: placement}),
            )

        if action is RuntimeAction.ALLOCATE:
            placement = str(payload.get("placement") or self._placement(completion, state, context))
            next_task = self._make_task(
                completion,
                RuntimeAction.WEIGHT_CACHE_LOOKUP,
                payload,
            )
            return self._result(
                completion,
                (next_task,),
                RuntimeDelta(
                    allocations={"{}@{}".format(flow_id, placement): required_bytes}
                ),
            )

        if action is RuntimeAction.WEIGHT_CACHE_LOOKUP:
            weight_key = str(payload.get("weight_key", "{}.weights".format(flow_id)))
            payload["weight_key"] = weight_key
            payload["weight_cache_hit"] = weight_key in state.weight_cache
            next_action = (
                RuntimeAction.BATCH_SCHEDULE
                if payload["weight_cache_hit"]
                else RuntimeAction.PAGE_CACHE_LOOKUP
            )
            return self._result(
                completion,
                (self._make_task(completion, next_action, payload),),
            )

        if action is RuntimeAction.PAGE_CACHE_LOOKUP:
            page_key = str(payload.get("page_key", "{}.pages".format(flow_id)))
            payload["page_key"] = page_key
            payload["page_cache_hit"] = page_key in state.page_cache
            payload[_WEIGHT_CACHE_FILL_ON_PCIE] = True
            transfer_bytes = _non_negative_int(
                payload.get("transfer_byte_count", required_bytes),
                "transfer_byte_count",
            )
            if payload["page_cache_hit"]:
                payload["byte_count"] = transfer_bytes
                next_action = (
                    RuntimeAction.IOMMU_TRANSLATE
                    if transfer_bytes > 0
                    else RuntimeAction.BATCH_SCHEDULE
                )
                delta = (
                    RuntimeDelta(
                        weight_cache_additions=(str(payload["weight_key"]),)
                    )
                    if transfer_bytes == 0
                    else RuntimeDelta()
                )
            else:
                payload["byte_count"] = _non_negative_int(
                    payload.get("storage_byte_count", required_bytes),
                    "storage_byte_count",
                )
                next_action = RuntimeAction.NVME_READ
                delta = RuntimeDelta()
            return self._result(
                completion,
                (self._make_task(completion, next_action, payload),),
                delta,
            )

        if action is RuntimeAction.NVME_READ:
            transfer_bytes = _non_negative_int(
                payload.get("transfer_byte_count", required_bytes),
                "transfer_byte_count",
            )
            payload["byte_count"] = transfer_bytes
            next_action = (
                RuntimeAction.IOMMU_TRANSLATE
                if transfer_bytes > 0
                else RuntimeAction.BATCH_SCHEDULE
            )
            next_task = self._make_task(
                completion,
                next_action,
                payload,
            )
            weight_cache_additions = (
                (str(payload["weight_key"]),)
                if transfer_bytes == 0
                else ()
            )
            return self._result(
                completion,
                (next_task,),
                RuntimeDelta(
                    page_cache_additions=(str(payload["page_key"]),),
                    weight_cache_additions=weight_cache_additions,
                ),
            )

        if action is RuntimeAction.PCIE_TRANSFER:
            payload["byte_count"] = _non_negative_int(
                payload.get("control_byte_count", payload.get("byte_count")),
                "control_byte_count",
            )
            delta = RuntimeDelta()
            if payload.get(_WEIGHT_CACHE_FILL_ON_PCIE) is True:
                delta = RuntimeDelta(
                    weight_cache_additions=(str(payload["weight_key"]),),
                )
            return self._result(
                completion,
                (
                    self._make_task(
                        completion,
                        RuntimeAction.BATCH_SCHEDULE,
                        payload,
                    ),
                ),
                delta,
            )

        if action is RuntimeAction.OPERATOR_SCHEDULE:
            raw_targets = self.profile.get("gpu_command_targets")
            if raw_targets is not None:
                targets = tuple(raw_targets)
                if any(not isinstance(target, Mapping) for target in targets):
                    raise RuntimeDispatchError(
                        "profile.gpu_command_targets must contain mappings"
                    )
                if not targets:
                    return self._result(
                        completion,
                        (
                            self._make_task(
                                completion,
                                RuntimeAction.INTERRUPT,
                                payload,
                            ),
                        ),
                    )

                branch_tasks = []
                compute_task_ids = []
                branch_actions = (
                    RuntimeAction.COMMAND_BUILD,
                    RuntimeAction.COMMAND_SUBMIT,
                    RuntimeAction.GPU_COMMAND_PROCESS,
                    RuntimeAction.MMU_TLB_LOOKUP,
                    RuntimeAction.L2_LOOKUP,
                    RuntimeAction.VRAM_CONTROLLER,
                    RuntimeAction.COMPUTE,
                )
                for target in targets:
                    component_id = str(target.get("component_id", "")).strip()
                    if not component_id:
                        raise RuntimeDispatchError(
                            "GPU command target component_id is required"
                        )
                    branch_payload = dict(payload)
                    branch_payload.update(
                        {
                            "gpu_component_id": component_id,
                            "service_ns_by_action": dict(
                                target.get("service_ns", {})
                            ),
                            "resources_by_action": dict(
                                target.get("resources", {})
                            ),
                            "transaction_count_by_action": dict(
                                target.get("transaction_count", {})
                            ),
                            "_prelowered_gpu_branch": True,
                        }
                    )
                    dependency = completion.task.task_id
                    for offset, branch_action in enumerate(branch_actions, 1):
                        branch_task = self._make_task(
                            completion,
                            branch_action,
                            branch_payload,
                            dependencies=(dependency,),
                            sequence=completion.sequence + offset,
                            task_suffix=component_id,
                        )
                        branch_tasks.append(branch_task)
                        dependency = branch_task.task_id
                    compute_task_ids.append(dependency)
                interrupt = self._make_task(
                    completion,
                    RuntimeAction.INTERRUPT,
                    payload,
                    dependencies=tuple(compute_task_ids),
                    sequence=completion.sequence + len(branch_actions) + 1,
                    task_suffix="gpu-join",
                )
                return self._result(completion, (*branch_tasks, interrupt))

        if (
            payload.get("_prelowered_gpu_branch") is True
            and action
            in {
                RuntimeAction.COMMAND_BUILD,
                RuntimeAction.COMMAND_SUBMIT,
                RuntimeAction.GPU_COMMAND_PROCESS,
                RuntimeAction.MMU_TLB_LOOKUP,
                RuntimeAction.L2_LOOKUP,
                RuntimeAction.VRAM_CONTROLLER,
                RuntimeAction.COMPUTE,
            }
        ):
            return self._result(completion)

        transition = {
            RuntimeAction.IOMMU_TRANSLATE: RuntimeAction.DMA_MAP,
            RuntimeAction.DMA_MAP: RuntimeAction.PCIE_TRANSFER,
            RuntimeAction.BATCH_SCHEDULE: RuntimeAction.OPERATOR_SCHEDULE,
            RuntimeAction.OPERATOR_SCHEDULE: RuntimeAction.COMMAND_BUILD,
            RuntimeAction.COMMAND_BUILD: RuntimeAction.COMMAND_SUBMIT,
            RuntimeAction.COMMAND_SUBMIT: RuntimeAction.GPU_COMMAND_PROCESS,
            RuntimeAction.GPU_COMMAND_PROCESS: RuntimeAction.MMU_TLB_LOOKUP,
            RuntimeAction.MMU_TLB_LOOKUP: RuntimeAction.L2_LOOKUP,
            RuntimeAction.L2_LOOKUP: RuntimeAction.VRAM_CONTROLLER,
        }.get(action)
        if transition is not None:
            return self._result(
                completion,
                (self._make_task(completion, transition, payload),),
            )

        if action is RuntimeAction.VRAM_CONTROLLER:
            continuations = self._continuations(completion, context)
            if not continuations:
                return self._result(
                    completion,
                    (self._make_task(completion, RuntimeAction.COMPUTE, payload),),
                )
            attached = tuple(
                replace(
                    task,
                    dependencies=tuple(
                        dict.fromkeys(
                            (completion.task.task_id, *task.dependencies)
                        )
                    ),
                )
                for task in continuations
            )
            interrupt = self._make_task(
                completion,
                RuntimeAction.INTERRUPT,
                payload,
                dependencies=tuple(task.task_id for task in attached),
                sequence=completion.sequence + 2,
            )
            return self._result(completion, (*attached, interrupt))

        if action is RuntimeAction.COMPUTE:
            return self._result(
                completion,
                (self._make_task(completion, RuntimeAction.INTERRUPT, payload),),
            )

        if action is RuntimeAction.INTERRUPT:
            return self._result(
                completion,
                (self._make_task(completion, RuntimeAction.CPU_COMPLETE, payload),),
            )

        if action is RuntimeAction.CPU_COMPLETE:
            return self._result(
                completion,
                delta=RuntimeDelta(completed_requests=(completion.task.request_id,)),
            )

        # Every enum member must be handled explicitly; this guard makes a
        # future enum extension fail closed instead of becoming a callback hook.
        raise RuntimeDispatchError("unhandled runtime action {}".format(action.value))


def build_initial_control_tasks(
    profile: Optional[Mapping[str, object]] = None,
    state: Optional[RuntimeState] = None,
    context: Optional[Mapping[str, object]] = None,
) -> Tuple[HardwareTask, ...]:
    """Build aggregate capacity-check roots from a closed request context."""

    profile_data = dict(profile or {})
    context_data = dict(context or {})
    _reject_callbacks(profile_data, "profile")
    _reject_callbacks(context_data, "context")
    if state is not None and not isinstance(state, RuntimeState):
        raise TypeError("state must be a RuntimeState")

    raw_requests = context_data.get("requests")
    if raw_requests is None:
        requests = (context_data,)
    else:
        requests = tuple(raw_requests)
    if any(not isinstance(request, Mapping) for request in requests):
        raise RuntimeDispatchError("context.requests must contain mappings")

    dispatcher = RuntimeDispatcher(profile_data)
    roots = []
    for index, raw_request in enumerate(requests):
        request = dict(raw_request)
        request_id = str(request.get("request_id", "request-{}".format(index)))
        flow_id = str(request.get("flow_id", request_id))
        required_bytes = _non_negative_int(
            request.get(
                "required_bytes",
                profile_data.get("default_required_bytes", 4096),
            ),
            "required_bytes",
        )
        byte_count = _non_negative_int(
            request.get("byte_count", required_bytes),
            "byte_count",
        )
        payload = {
            key: value
            for key, value in request.items()
            if key
            not in {
                "request_id",
                "requests",
                "continuations",
                "placement_decisions",
                "capacity_bytes",
            }
        }
        payload.update(
            {
                "flow_id": flow_id,
                "required_bytes": required_bytes,
                "byte_count": byte_count,
                "capacity_resource": str(
                    request.get("capacity_resource", "vram")
                ),
            }
        )
        action = RuntimeAction.CAPACITY_CHECK
        task_id = "{}:0000:{}".format(flow_id, action.value)
        instruction_count = max(
            1,
            _non_negative_int(
                payload.get("instruction_count"),
                "instruction_count",
                1,
            ),
        )
        controller_batch = ControllerTransactionBatch(
            batch_id="{}.transactions".format(task_id),
            controller=ControllerKind.CPU,
            transaction_count=dispatcher._transaction_count(action, payload),
            byte_count=byte_count,
        )
        roots.append(
            HardwareTask(
                task_id=task_id,
                request_id=request_id,
                name=action.value,
                action=action,
                phase=int(RuntimePhase.CAPACITY_CHECK),
                sequence=0,
                category=TaskCategory.POLICY,
                demands=tuple(
                    ResourceDemand(
                        resource_id=resource_id,
                        service_ns=dispatcher._service_ns(action),
                        bytes_moved=byte_count,
                    )
                    for resource_id in dispatcher._resources(action)
                ),
                payload=payload,
                instruction_batch=InstructionBatch(
                    batch_id="{}.instructions".format(task_id),
                    instruction_class=InstructionClass.CONTROL,
                    instruction_count=instruction_count,
                    byte_count=byte_count,
                ),
                controller_batch=controller_batch,
            )
        )
    return tuple(roots)


__all__ = [
    "RuntimeDispatchError",
    "RuntimeDispatcher",
    "build_initial_control_tasks",
]
